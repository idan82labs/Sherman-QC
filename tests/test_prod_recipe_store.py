"""C05 RecipeStore: immutable versions, SKU registry, drafts, compile records, flats.

Pure store tests: recipes come from the C01 builder on synthetic dicts (no CAD,
no compile, no catalog DB, no server).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from apps.production_line.recipes.store import (
    _SCHEMA_V1,
    KEEP,
    SCHEMA_VERSION,
    SKU_RESERVATION_PREFIX,
    SKU_RESERVATION_TTL_S,
    CompiledRecord,
    RecipeStore,
    RecipeStoreError,
    parse_recipe_id,
)
from contracts.production.part_recipe import CadRef, DrawingRef, ReleasedBy
from domains.production.recipe.builder import build_part_recipe

CAD = "a" * 64
DRAWING = "b" * 64
GEN = "c" * 64
WHO = ReleasedBy(user_id="1", username="admin", role="admin")


def make_recipe(sku: str = "SYN-HAT-1", version: int = 1, part_id: str = "p1", flange_plus: float = 0.5,
                revision: str = "A", supersedes: int | None = None):
    lines = [([0.0, 0.0, 0.0], [0.0, 100.0, 0.0]), ([20.0, 0.0, 30.0], [20.0, 100.0, 30.0])]
    specs, marks = [], []
    for i, (start, end) in enumerate(lines, start=1):
        fid = f"spec-{i}"
        specs.append({"id": fid, "bend_id": f"B{i}", "target_angle": 90.0, "target_radius": 2.0,
                      "bend_line_start": json.dumps(start), "bend_line_end": json.dumps(end)})
        marks.append({"feature_id": fid, "label": f"B{i}", "source": "cad_candidate", "status": "active",
                      "review_status": "confirmed", "line_start": start, "line_end": end,
                      "target_angle": 90.0, "target_radius": 2.0})
    draft = {"schema_version": 1, "cad_file_hash": CAD, "drawing_file_hash": DRAWING, "candidate_generation": GEN,
             "marks": marks, "sequence": ["spec-1", "spec-2"], "operator_sequence_confirmed": True,
             "draft_revision": 1, "is_stale": False}
    release = {"expected_draft_revision": 1,
               "default_tolerances": {"angle_deg": {"plus": 1, "minus": 1},
                                      "flange_mm": {"plus": flange_plus, "minus": 0.5}, "source": "SHOP_STANDARD"}}
    return build_part_recipe(
        part={"id": part_id, "part_number": sku, "revision": revision}, spec_rows=specs, draft=draft, release=release,
        cad=CadRef(sha256=CAD, stored_rel_path=f"recipes/cad/{CAD}.step", candidate_generation=GEN,
                   candidate_angle_convention="NATIVE_SWEEP"),
        drawing=DrawingRef(sha256=DRAWING, stored_rel_path=f"recipes/drawings/{DRAWING}.pdf"),
        version=version, supersedes_version=supersedes, released_by=WHO,
        released_at=f"2026-10-17T05:0{version}:00Z",
    )


def compiled_for(recipe, status: str = "COMPILED") -> CompiledRecord:
    return CompiledRecord(
        recipe_sha256=recipe.recipe_sha256, status=status, compiled_part_sha256="d" * 64, compiler_version="e1-test",
        blocker_codes=("UNMARKED_BEND",) if status == "BLOCKED" else (),
        stored_rel_path=f"recipes/compiled/{recipe.recipe_sha256}/compiled_part.json",
    )


@pytest.fixture()
def store(tmp_path: Path) -> RecipeStore:
    return RecipeStore(tmp_path / "db" / "recipes.sqlite3")


def test_schema_wal_and_reopen(tmp_path: Path) -> None:
    db = tmp_path / "db" / "recipes.sqlite3"
    s = RecipeStore(db)
    assert s.schema_version() == SCHEMA_VERSION == 2
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.execute("UPDATE schema_meta SET version = 99 WHERE component = 'recipe_store'")
    conn.commit()
    conn.close()
    with pytest.raises(RecipeStoreError) as err:
        RecipeStore(db)
    assert err.value.code == "SCHEMA_TOO_NEW"


def test_insert_get_active_and_versions(store: RecipeStore) -> None:
    v1 = make_recipe()
    stored = store.insert_recipe(v1, compiled=compiled_for(v1), expectations={"k": 1}, benchmark_nominals={"bends": []})
    assert stored.recipe_id == "SYN-HAT-1@v1" and stored.status == "released"
    assert stored.recipe() == v1  # bytes round-trip and re-validate (hashes recomputed)
    assert store.get_by_sha(v1.recipe_sha256).recipe_id == v1.recipe_id
    assert store.sku_owner("SYN-HAT-1") == "p1" and store.sku_for_part("p1") == "SYN-HAT-1"
    v2 = make_recipe(version=2, flange_plus=0.7, supersedes=1)
    store.insert_recipe(v2, compiled=compiled_for(v2), expectations={"k": 2}, benchmark_nominals={"bends": []})
    assert store.get_active("SYN-HAT-1").recipe_id == "SYN-HAT-1@v2"
    assert [r.version for r in store.list_versions("SYN-HAT-1")] == [1, 2]
    assert [r.recipe_id for r in store.list_active()] == ["SYN-HAT-1@v2"]
    assert store.get_expectations(v2.recipe_sha256)["status"] == "READY"
    assert store.get_benchmark_reference("SYN-HAT-1@v2")["status"] == "REGISTERED"
    assert store.get_compiled(v2.recipe_sha256).compiled_part_sha256 == "d" * 64


def test_version_must_be_max_plus_one(store: RecipeStore) -> None:
    store.insert_recipe(make_recipe())
    for bad in (1, 3):
        with pytest.raises(RecipeStoreError) as err:
            store.insert_recipe(make_recipe(version=bad, flange_plus=0.9, supersedes=None if bad == 1 else 1))
        assert err.value.code == "CONCURRENT_RELEASE"
    assert store.max_version("SYN-HAT-1") == 1


def test_concurrent_inserts_exactly_one_wins(store: RecipeStore) -> None:
    candidates = [make_recipe(flange_plus=0.5 + i / 10) for i in range(6)]
    results: list[str] = []

    def go(r) -> None:
        try:
            store.insert_recipe(r)
            results.append("ok")
        except RecipeStoreError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=go, args=(r,)) for r in candidates]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("CONCURRENT_RELEASE") == 5


def test_sku_registry_collisions(store: RecipeStore) -> None:
    assert store.register_sku("ABC-1", "p1", "abc 1") is True
    assert store.register_sku("ABC-1", "p1") is False  # idempotent
    with pytest.raises(RecipeStoreError) as err:
        store.register_sku("ABC-1", "p2", "ABC-1")
    assert err.value.code == "SKU_COLLISION"
    with pytest.raises(RecipeStoreError) as err:
        store.register_sku("ABC-2", "p1")  # one part, one SKU
    assert err.value.code == "SKU_COLLISION"
    with pytest.raises(RecipeStoreError) as err:
        store.register_sku("abc 1", "p3")  # not canonical: refused, never normalised silently
    assert err.value.code == "INVALID_ARGUMENT"
    # A recipe for another part under a registered SKU is refused in the same transaction.
    with pytest.raises(RecipeStoreError) as err:
        store.insert_recipe(make_recipe(sku="ABC-1", part_id="p9"))
    assert err.value.code == "SKU_COLLISION"
    assert store.get("ABC-1@v1") is None


def test_retire_falls_back_to_previous_version(store: RecipeStore) -> None:
    store.insert_recipe(make_recipe(), benchmark_nominals={"bends": []})
    store.insert_recipe(make_recipe(version=2, flange_plus=0.8, supersedes=1), benchmark_nominals={"bends": []})
    retired = store.retire("SYN-HAT-1@v2", {"kind": "user", "id": "1", "role": "admin"})
    assert retired.status == "retired" and retired.retired_by == {"id": "1", "kind": "user", "role": "admin"}
    assert store.get_active("SYN-HAT-1").recipe_id == "SYN-HAT-1@v1"
    assert store.get_benchmark_reference("SYN-HAT-1@v2")["status"] == "RETIRED"
    store.retire("SYN-HAT-1@v1")
    assert store.get_active("SYN-HAT-1") is None and store.list_active() == []
    assert store.retire("SYN-HAT-1@v1").status == "retired"  # idempotent
    with pytest.raises(RecipeStoreError) as err:
        store.retire("NOPE@v1")
    assert err.value.code == "NOT_FOUND"
    # A retired version still counts for numbering: the next release is v3.
    with pytest.raises(RecipeStoreError):
        store.insert_recipe(make_recipe(version=2, flange_plus=0.9, supersedes=1))
    assert store.insert_recipe(make_recipe(version=3, flange_plus=0.9)).recipe_id == "SYN-HAT-1@v3"


def test_compiled_record_rules(store: RecipeStore) -> None:
    r = make_recipe()
    with pytest.raises(RecipeStoreError):
        CompiledRecord(recipe_sha256=r.recipe_sha256, status="BLOCKED", compiled_part_sha256="d" * 64)
    with pytest.raises(RecipeStoreError):
        CompiledRecord(recipe_sha256=r.recipe_sha256, status="COMPILED", compiled_part_sha256=None)
    with pytest.raises(RecipeStoreError):
        store.insert_recipe(r, compiled=compiled_for(r, "BLOCKED"), expectations={"x": 1})  # not from a BLOCKED compile
    with pytest.raises(RecipeStoreError) as err:
        store.set_compiled(compiled_for(r))
    assert err.value.code == "NOT_FOUND"
    store.insert_recipe(r)
    store.set_compiled(compiled_for(r, "BLOCKED"))
    assert store.get_compiled(r.recipe_sha256).blocker_codes == ("UNMARKED_BEND",)


def test_release_draft_optimistic_revision(store: RecipeStore) -> None:
    assert store.get_release_draft("p1") is None
    first = store.put_release_draft("p1", {"x": 1}, 0, {"kind": "user", "id": "1", "role": "admin"})
    assert first.revision == 1 and first.request == {"x": 1}
    for stale in (0, 2):
        with pytest.raises(RecipeStoreError) as err:
            store.put_release_draft("p1", {"x": 2}, stale)
        assert err.value.code == "RELEASE_DRAFT_REVISION_CHANGED"
    assert store.put_release_draft("p1", {"x": 3}, 1).revision == 2
    assert store.get_release_draft("p1").request == {"x": 3}


def test_part_flat_lifecycle(store: RecipeStore) -> None:
    rec = store.put_part_flat("p1", flat_sha256="e" * 64, format="DXF", original_filename="f.dxf",
                              stored_rel_path=f"recipes/flats/{'e' * 64}.dxf", size_bytes=10, state="UPLOADED",
                              parse={"status": "PARSED"})
    assert rec.revision == 1 and rec.proposal is None
    rec = store.update_part_flat("p1", 1, state="PROPOSED", proposal={"status": "PROPOSED"})
    assert rec.revision == 2 and rec.proposal == {"status": "PROPOSED"}
    with pytest.raises(RecipeStoreError) as err:
        store.update_part_flat("p1", 1, state="CONFIRMED", binding={"a": 1})
    assert err.value.code == "FLAT_REVISION_CHANGED"
    rec = store.update_part_flat("p1", 2, state="CONFIRMED", proposal=KEEP, binding={"a": 1})
    assert rec.proposal == {"status": "PROPOSED"} and rec.binding == {"a": 1}
    # Replacing the flat resets proposal and binding.
    rec = store.put_part_flat("p1", flat_sha256="f" * 64, format="STEP", original_filename=None,
                              stored_rel_path=f"recipes/flats/{'f' * 64}.step", size_bytes=11, state="UPLOADED",
                              parse=None)
    assert rec.revision == 4 and rec.binding is None and rec.proposal is None and rec.format == "STEP"
    with pytest.raises(RecipeStoreError):
        store.put_part_flat("p1", flat_sha256="f" * 64, format="IGES", original_filename=None,
                            stored_rel_path="x", size_bytes=1, state="UPLOADED", parse=None)
    with pytest.raises(RecipeStoreError):
        store.update_part_flat("p1", 4, state="WEIRD")
    with pytest.raises(RecipeStoreError):
        store.delete_part_flat("p1", expected_revision=3)
    assert store.delete_part_flat("p1", expected_revision=4) is True
    assert store.delete_part_flat("p1") is False
    # Revisions are monotonic across a delete: a stale tab holding revision 1..4 never matches the new flat.
    rec = store.put_part_flat("p1", flat_sha256="e" * 64, format="DXF", original_filename="f.dxf",
                              stored_rel_path=f"recipes/flats/{'e' * 64}.dxf", size_bytes=10, state="UPLOADED",
                              parse=None)
    assert rec.revision == 5
    for stale in (1, 2, 3, 4):
        with pytest.raises(RecipeStoreError) as err:
            store.update_part_flat("p1", stale, state="DECLINED", binding={"state": "UNBOUND"})
        assert err.value.code == "FLAT_REVISION_CHANGED"
        with pytest.raises(RecipeStoreError):
            store.delete_part_flat("p1", expected_revision=stale)
    # Another part's counter is independent.
    other = store.put_part_flat("p2", flat_sha256="e" * 64, format="DXF", original_filename=None,
                                stored_rel_path="x", size_bytes=1, state="UPLOADED", parse=None)
    assert other.revision == 1


def test_v1_store_migrates_to_v2(tmp_path: Path) -> None:
    """A schema-v1 database (no content hash, no history, no flat counters) upgrades in place."""
    db = tmp_path / "db" / "recipes.sqlite3"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE schema_meta (component TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at TEXT NOT NULL)")
    for statement in _SCHEMA_V1:
        conn.execute(statement)
    conn.execute("INSERT INTO schema_meta VALUES ('recipe_store', 1, '2026-10-17T00:00:00.000Z')")
    conn.execute("INSERT INTO compile_attempts(part_id, status, codes_json, report_json, created_at)"
                 " VALUES ('p1', 'BLOCKED', '[\"COMPILE_BLOCKED\"]', '{}', '2026-10-17T00:00:00.000Z')")
    conn.execute("INSERT INTO part_flats(part_id, revision, flat_sha256, format, stored_rel_path, size_bytes,"
                 " uploaded_at, state, updated_at) VALUES ('p1', 7, ?, 'DXF', 'x', 1, 'now', 'UPLOADED', 'now')",
                 ("e" * 64,))
    conn.commit()
    conn.close()
    s = RecipeStore(db)
    assert s.schema_version() == 2
    old = s.last_compile_attempt("p1")
    assert old.status == "BLOCKED" and old.content_sha256 is None
    assert s.delete_part_flat("p1", expected_revision=7) is True
    rec = s.put_part_flat("p1", flat_sha256="e" * 64, format="DXF", original_filename=None, stored_rel_path="x",
                          size_bytes=1, state="UPLOADED", parse=None)
    assert rec.revision == 8
    s.record_compile_attempt("p1", status="FAILED", codes=["COMPILE_TIMEOUT"], report={}, content_sha256="9" * 64)
    assert s.last_compile_attempt("p1").content_sha256 == "9" * 64
    RecipeStore(db)  # re-opening a v2 store is a no-op


class _Clock:
    def __init__(self) -> None:
        from datetime import datetime, timezone

        self.now = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_sku_reservation_lifecycle(tmp_path: Path) -> None:
    from datetime import timedelta

    clock = _Clock()
    store = RecipeStore(tmp_path / "r.sqlite3", clock=clock)
    token = store.reserve_sku("ABC-1", "abc 1")
    assert store.sku_owner("ABC-1") == SKU_RESERVATION_PREFIX + token
    # A live reservation blocks a second creator and a register for another part.
    with pytest.raises(RecipeStoreError) as err:
        store.reserve_sku("ABC-1")
    assert err.value.code == "SKU_COLLISION"
    with pytest.raises(RecipeStoreError):
        store.register_sku("ABC-1", "p9")
    # Bind it to the created catalog part; a wrong token cannot bind.
    with pytest.raises(RecipeStoreError):
        store.bind_sku_reservation("ABC-1", "0" * 32, "p1")
    store.bind_sku_reservation("ABC-1", token, "p1")
    assert store.sku_owner("ABC-1") == "p1" and store.sku_for_part("p1") == "ABC-1"
    assert store.release_sku_reservation("ABC-1", token) is False  # bound: not a reservation any more
    # Released reservations free the SKU.
    t2 = store.reserve_sku("XYZ-2")
    assert store.release_sku_reservation("XYZ-2", t2) is True and store.sku_owner("XYZ-2") is None
    # A crashed creator's reservation expires and is reclaimed; the stale token can no longer bind.
    t3 = store.reserve_sku("OLD-3")
    clock.now += timedelta(seconds=SKU_RESERVATION_TTL_S + 1)
    assert store.sku_owner("OLD-3") is None
    t4 = store.reserve_sku("OLD-3")
    with pytest.raises(RecipeStoreError):
        store.bind_sku_reservation("OLD-3", t3, "p3")
    store.bind_sku_reservation("OLD-3", t4, "p4")
    assert store.sku_owner("OLD-3") == "p4"


def test_compiled_history_supersede_and_find(store: RecipeStore) -> None:
    r = make_recipe()
    store.insert_recipe(r, compiled=compiled_for(r), expectations={"k": 1}, benchmark_nominals={"bends": []})
    old = store.get_compiled(r.recipe_sha256)
    new = CompiledRecord(recipe_sha256=r.recipe_sha256, status="COMPILED", compiled_part_sha256="7" * 64,
                         compiler_version="e1-next", stored_rel_path=old.stored_rel_path)
    hist_rel = f"recipes/compiled_history/{r.recipe_sha256}/{'d' * 64}/compiled_part.json"
    store.set_compiled(new, expectations={"k": 2}, benchmark_nominals={"bends": [1]}, superseded=old,
                       superseded_rel_path=hist_rel)
    assert store.get_compiled(r.recipe_sha256).compiled_part_sha256 == "7" * 64
    assert store.get_benchmark_reference(r.recipe_id)["compiled_part_sha256"] == "7" * 64
    assert store.find_compiled(r.recipe_sha256, "7" * 64) == ("COMPILED", old.stored_rel_path)
    assert store.find_compiled(r.recipe_sha256, "d" * 64) == ("COMPILED", hist_rel)
    assert store.find_compiled(r.recipe_sha256, "1" * 64) is None
    hist = store.list_compiled_history(r.recipe_sha256)
    assert [(h.reason, h.compiled_part_sha256) for h in hist] == [("SUPERSEDED", "d" * 64)]
    # Only this recipe's COMPILED install can be superseded; the refused call changes nothing.
    with pytest.raises(RecipeStoreError):
        store.set_compiled(new, superseded=compiled_for(r, "BLOCKED"), superseded_rel_path="x")
    blocked = CompiledRecord(recipe_sha256=r.recipe_sha256, status="BLOCKED", compiled_part_sha256="8" * 64,
                             compiler_version="e1-next", blocker_codes=("UNMARKED_BEND",))
    store.record_compiled_history(blocked, reason="RECOMPUTE_BLOCKED", stored_rel_path="y/compiled_part.json",
                                  report={"status": "BLOCKED"})
    assert [h.reason for h in store.list_compiled_history(r.recipe_sha256)] == ["SUPERSEDED", "RECOMPUTE_BLOCKED"]
    assert store.get_compiled(r.recipe_sha256).status == "COMPILED"
    with pytest.raises(RecipeStoreError):
        store.record_compiled_history(blocked, reason="WHATEVER", stored_rel_path=None)


def test_compile_attempts_and_parse_recipe_id(store: RecipeStore) -> None:
    assert store.last_compile_attempt("p1") is None
    store.record_compile_attempt("p1", status="BLOCKED", codes=["UNMARKED_BEND", "COMPILE_BLOCKED", "UNMARKED_BEND"],
                                 report={"issues": []}, cad_sha256=CAD)
    store.record_compile_attempt("p1", status="COMPILED", codes=[], report={}, cad_sha256=CAD)
    last = store.last_compile_attempt("p1")
    assert last.status == "COMPILED" and last.codes == ()
    assert parse_recipe_id("SYN-HAT-1@v12") == ("SYN-HAT-1", 12)
    assert parse_recipe_id("syn@v1") is None and parse_recipe_id("X@v0") is None
