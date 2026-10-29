"""C05 release gate: E2 synthetic STEP -> confirmed draft -> supervised E1 compile -> stored recipe.

Fixtures: one E2 DEV smoke hat (``E2-DEV-SMHT-000``: real STEP BRep, flat DXF,
drawing PDF) generated into a tmp dir, a tmp ``PartCatalogManager``, a tmp data
root. The real E1 compile runs as a supervised subprocess from the repo root.
Failure mapping uses fake engine modules in a tmp engine root. No repo data,
no server import.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from part_catalog import PartCatalogManager

from apps.production_line.recipes.catalog_port import CatalogManagerPort
from apps.production_line.recipes.export_benchmark import (
    export,
    iter_rows,
    out_path_refusal,
)
from apps.production_line.recipes.export_benchmark import main as export_main
from apps.production_line.recipes.release_service import (
    CompileLauncher,
    ReleaseGateError,
    add_release_listener,
    release,
    remove_release_listener,
)
from apps.production_line.recipes.resolver import LineRecipeResolver
from apps.production_line.recipes.store import RecipeStore
from apps.production_line.recipes.synthetic_onboarding import (
    generate_e2_case,
    onboard_case,
    release_request,
    vendor_setup_loader,
)
from apps.production_line.store.line_store import LineStore
from contracts.production.compiled_part import (
    CompiledPartV1,
    check_compiled_against_recipe,
)

REPO = Path(__file__).resolve().parents[1]
ADMIN = {"kind": "user", "user_id": "1", "username": "admin", "role": "admin"}
LAUNCHER = CompileLauncher(engine_root=REPO, python_exe=Path(sys.executable))


class NeverCompile:
    """The gate must refuse before compiling."""

    def compile(self, *args, **kwargs):  # pragma: no cover - reaching it is the failure
        raise AssertionError("the gate compiled a refused release")


@pytest.fixture(scope="module")
def hat_case(tmp_path_factory) -> Path:
    return generate_e2_case(tmp_path_factory.mktemp("e2"), "E2-DEV-SMHT-000")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_env(tmp_path: Path, case_dir: Path, *, part_number: str = "SYN-HAT-1", **onboard) -> SimpleNamespace:
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    ob = onboard_case(catalog, case_dir, part_number=part_number, upload_dir=tmp_path / "uploads", **onboard)
    root = tmp_path / "root"
    return SimpleNamespace(
        catalog=catalog,
        ob=ob,
        port=CatalogManagerPort(catalog, vendor_setup_loader=vendor_setup_loader(catalog)),
        root=root,
        store=RecipeStore(root / "db" / "recipes.sqlite3"),
        line=LineStore(root / "db" / "line.sqlite3"),
    )


def do_release(env: SimpleNamespace, request: dict | None = None, *, compiler=LAUNCHER, principal=ADMIN,
               part_id: str | None = None, store=None, line_store="default"):
    return release(
        part_id or env.ob.part_id,
        request if request is not None else release_request(env.ob),
        principal,
        catalog=env.port,
        recipe_store=store or env.store,
        data_root=env.root,
        compiler=compiler,
        line_store=env.line if line_store == "default" else line_store,
    )


def refused(env: SimpleNamespace, request: dict | None = None, **kw) -> ReleaseGateError:
    with pytest.raises(ReleaseGateError) as err:
        do_release(env, request, compiler=kw.pop("compiler", NeverCompile()), **kw)
    return err.value


def resave_draft(env: SimpleNamespace, mutate) -> dict:
    draft = env.catalog.get_vendor_setup_draft(env.ob.part_id)
    payload = {k: v for k, v in draft.items() if k not in ("updated_at", "draft_revision", "is_stale")}
    payload["expected_draft_revision"] = draft["draft_revision"]
    mutate(payload)
    return env.catalog.save_vendor_setup_draft(env.ob.part_id, payload)


def compiled_dirs(root: Path) -> list[str]:
    base = root / "recipes" / "compiled"
    return sorted(p.name for p in base.iterdir()) if base.is_dir() else []


# ---------------------------------------------------------------------------- happy path


def test_release_compiles_freezes_registers_and_versions(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    seen: list[str] = []

    def on_release(r) -> None:
        seen.append(r.recipe_id)

    add_release_listener(on_release)
    try:
        out = do_release(env)
    finally:
        remove_release_listener(on_release)
    recipe = out.recipe
    assert out.outcome == "released" and recipe.recipe_id == "SYN-HAT-1@v1" and recipe.supersedes_version is None
    # Bends in the operator's draft sequence order, labelled B1..BN.
    assert [b.source_feature_id for b in recipe.bends] == env.ob.sequence
    assert [b.label for b in recipe.bends] == [f"B{i}" for i in range(1, len(recipe.bends) + 1)]
    # CAD and drawing frozen under the data root by content address.
    part = env.catalog.get_part(env.ob.part_id)
    cad = env.root / recipe.cad.stored_rel_path
    drawing = env.root / recipe.drawing.stored_rel_path
    assert recipe.cad.stored_rel_path == f"recipes/cad/{part.cad_file_hash}.step" and _sha(cad) == part.cad_file_hash
    assert recipe.drawing.stored_rel_path.startswith("recipes/drawings/") and _sha(drawing) == part.drawing_file_hash
    # The compile is installed, hash-bound to the recipe and consistent with it.
    rec = env.store.get_compiled(recipe.recipe_sha256)
    assert rec.status == "COMPILED" and rec.stored_rel_path == f"recipes/compiled/{recipe.recipe_sha256}/compiled_part.json"
    compiled = CompiledPartV1.model_validate_json((env.root / rec.stored_rel_path).read_bytes())
    assert compiled.compiled_part_sha256 == rec.compiled_part_sha256 and compiled.status == "COMPILED"
    assert compiled.recipe.recipe_sha256 == recipe.recipe_sha256
    assert check_compiled_against_recipe(compiled, recipe) == []
    for stage in compiled.stages:
        assert _sha(env.root / Path(rec.stored_rel_path).parent / stage.sample_cloud.rel_path) == stage.sample_cloud.sha256
    assert compiled_dirs(env.root) == [recipe.recipe_sha256]  # staging removed
    # Expectations READY from the compile; benchmark reference with per-bend nominals.
    exp = env.store.get_expectations(recipe.recipe_sha256)
    assert exp["status"] == "READY" and exp["algo_version"] == compiled.compiler.version
    assert {c["characteristic_id"] for c in exp["expectations"]["characteristics"]} == {
        c.characteristic_id for c in compiled.characteristics
    }
    bench = env.store.get_benchmark_reference(recipe.recipe_id)
    assert bench["status"] == "REGISTERED" and bench["compiled_part_sha256"] == compiled.compiled_part_sha256
    rows = bench["nominals"]["bends"]
    assert [r["bend_uid"] for r in rows] == [b.bend_uid for b in recipe.bends]
    by_uid = {b.bend_uid: b for b in compiled.bends}
    for row in rows:
        assert row["nominal_included_deg"] == by_uid[row["bend_uid"]].nominal_included_deg
        assert row["angle"]["nominal"] == pytest.approx(by_uid[row["bend_uid"]].nominal_included_deg)
        assert row["flange_a"]["characteristic_id"].endswith("/FLANGE_LENGTH/A")
        assert row["flange_b"]["characteristic_id"].endswith("/FLANGE_LENGTH/B")
    # Exactly one RECIPE_RELEASED (with RECIPE_COMPILED), principal recorded; no job enqueued.
    events = env.line.events_after(0)
    assert [e["kind"] for e in events] == ["RECIPE_COMPILED", "RECIPE_RELEASED"]
    assert events[1]["principal"] == {"kind": "user", "id": "1", "role": "admin"}
    assert events[1]["payload"]["recipe_sha256"] == recipe.recipe_sha256
    assert env.line.list_jobs() == []
    assert recipe.recipe_id in seen

    # Identical release: unchanged, nothing compiled, stored or announced.
    attempts_before = env.store.last_compile_attempt(env.ob.part_id).id
    again = do_release(env, compiler=NeverCompile())
    assert again.outcome == "unchanged" and again.recipe.recipe_id == "SYN-HAT-1@v1"
    assert len(env.line.events_after(0)) == 2 and env.store.last_compile_attempt(env.ob.part_id).id == attempts_before

    # A changed tolerance is a new version that supersedes v1.
    req = release_request(env.ob)
    req["default_tolerances"] = {**req["default_tolerances"], "angle_deg": {"plus": 0.75, "minus": 0.75}}
    v2 = do_release(env, req)
    assert v2.outcome == "released" and v2.recipe.recipe_id == "SYN-HAT-1@v2" and v2.recipe.supersedes_version == 1
    assert env.store.get_active("SYN-HAT-1").recipe_id == "SYN-HAT-1@v2"
    assert sorted(compiled_dirs(env.root)) == sorted([recipe.recipe_sha256, v2.recipe.recipe_sha256])
    angle = [c for c in json.loads((env.root / env.store.get_compiled(v2.recipe.recipe_sha256).stored_rel_path)
                                   .read_text())["characteristics"] if c["kind"] == "INCLUDED_ANGLE"]
    defaults = [c for c in angle if c["tolerance"]["origin"] == "RECIPE_DEFAULT"]
    assert defaults and all(c["tolerance"]["plus"] == 0.75 for c in defaults)
    for c in angle:  # per-bend overrides (E2 sets some) still win over the new default
        resolved = v2.recipe.resolved_tolerance(c["bend_uid"], "angle_deg")
        assert (c["tolerance"]["plus"], c["tolerance"]["minus"]) == (resolved.plus, resolved.minus)
    assert [e["kind"] for e in env.line.events_after(0)].count("RECIPE_RELEASED") == 2

    # The resolver's CompiledPartSink re-installs a recompile and refreshes expectations.
    resolver = LineRecipeResolver(env.store, data_root=env.root)
    calls: list[str] = []
    listener = lambda r: calls.append(r.recipe_id)  # noqa: E731
    add_release_listener(listener)
    try:
        src = env.root / Path(env.store.get_compiled(v2.recipe.recipe_sha256).stored_rel_path).parent
        job_out = tmp_path / "job_compile_out"
        shutil.copytree(src, job_out)
        compiled_v2 = CompiledPartV1.model_validate_json((job_out / "compiled_part.json").read_bytes())
        with pytest.raises(ValueError):
            resolver.store_compiled(v2.recipe.recipe_sha256, job_out, status="BLOCKED",
                                    compiled_part_sha256=compiled_v2.compiled_part_sha256, blocker_codes=("X",))
        resolver.store_compiled(v2.recipe.recipe_sha256, job_out, status="COMPILED",
                                compiled_part_sha256=compiled_v2.compiled_part_sha256, blocker_codes=())
        assert job_out.is_dir()  # the runner's workspace copy is left intact
        assert env.store.get_compiled(v2.recipe.recipe_sha256).status == "COMPILED"
        assert calls == [v2.recipe.recipe_id]
        loaded = resolver.load(v2.recipe.recipe_id)
        assert loaded.compiled_dir == (env.root / f"recipes/compiled/{v2.recipe.recipe_sha256}").resolve()
        assert loaded.cad_path.is_absolute() and loaded.cad_path.is_file()
        assert loaded.compiled_part_sha256 == compiled_v2.compiled_part_sha256
    finally:
        remove_release_listener(listener)

    # Benchmark export: one row per bend, per released recipe.
    out_file = tmp_path / "bench.jsonl"
    assert export(env.root, out_file) == 2 * len(recipe.bends)
    first = json.loads(out_file.read_text().splitlines()[0])
    assert first["recipe_id"] == "SYN-HAT-1@v1" and first["expectations"]["ready"] is True
    assert first["nominal_angle"]["compiled_included_deg"] is not None and first["flange_a"]["nominal"] is not None
    assert len(list(iter_rows(env.store, sku="SYN-HAT-1"))) == 2 * len(recipe.bends)


# ---------------------------------------------------------------------------- refusals before the compile


def test_gate_codes_before_compile(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    req = release_request(env.ob)

    err = refused(env, {**req, "default_tolerances": None})
    assert err.codes == ("TOLERANCES_MISSING",) and err.http_status == 422
    err = refused(env, {**req, "expected_draft_revision": req["expected_draft_revision"] + 1})
    assert err.codes == ("DRAFT_REVISION_CHANGED",) and err.http_status == 409
    err = refused(env, {**req, "expected_release_draft_revision": 4})
    assert err.codes == ("RELEASE_DRAFT_REVISION_CHANGED",) and err.http_status == 409
    err = refused(env, {**req, "default_tolerances": {**req["default_tolerances"], "angle_deg": {"plus": -1, "minus": 1}}})
    assert "TOLERANCE_INVALID" in err.codes
    err = refused(env, part_id="no-such-part")
    assert err.codes == ("PART_NOT_FOUND",) and err.http_status == 404
    err = refused(env, {**req, "bend_overrides": {"not-a-mark": {}}})
    assert err.codes == ("OVERRIDE_UNKNOWN_BEND",)

    # Unconfirmed bending order (the Vendor Setup validator and the builder agree).
    saved = resave_draft(env, lambda d: d.update(operator_sequence_confirmed=False))
    err = refused(env, {**req, "expected_draft_revision": saved["draft_revision"]})
    assert "SEQUENCE_NOT_CONFIRMED" in err.codes and err.http_status == 422
    # One active mark back to pending.
    def pending(d):
        d["operator_sequence_confirmed"] = True
        d["marks"][0]["review_status"] = "pending"
    saved = resave_draft(env, pending)
    err = refused(env, {**req, "expected_draft_revision": saved["draft_revision"]})
    assert err.codes == ("UNCONFIRMED_ACTIVE_MARKS",)
    assert err.details["UNCONFIRMED_ACTIVE_MARKS"] == (saved["marks"][0]["feature_id"],)
    saved = resave_draft(env, lambda d: d["marks"][0].update(review_status="confirmed"))
    req = {**req, "expected_draft_revision": saved["draft_revision"]}

    # Tampered drawing and CAD bytes are refused by hash.
    part = env.catalog.get_part(env.ob.part_id)
    drawing = Path(part.drawing_file_path)
    original_drawing = drawing.read_bytes()
    drawing.write_bytes(original_drawing + b"%tampered")
    assert refused(env, req).codes == ("DRAWING_HASH_MISMATCH",)
    drawing.write_bytes(original_drawing)
    cad = Path(part.cad_file_path)
    original_cad = cad.read_bytes()
    cad.write_bytes(original_cad + b"\n")
    assert refused(env, req).codes == ("CAD_HASH_MISMATCH",)
    cad.write_bytes(original_cad)

    # CAD republished after the draft was saved -> the draft is stale.
    new_cad = tmp_path / "uploads" / "republished.step"
    new_cad.write_bytes(original_cad.replace(b"ISO-10303-21;", b"ISO-10303-21;\n", 1))
    env.catalog.update_part(env.ob.part_id, cad_file_path=str(new_cad), cad_file_hash=_sha(new_cad),
                            cad_uploaded_at="2026-10-17T07:00:00")
    err = refused(env, req)
    assert "DRAFT_STALE" in err.codes and err.http_status == 422
    # A mesh CAD is refused outright: production recipes require STEP.
    stl = tmp_path / "uploads" / "part.stl"
    stl.write_bytes(b"solid x\nendsolid x\n")
    env.catalog.update_part(env.ob.part_id, cad_file_path=str(stl), cad_file_hash=_sha(stl))
    assert "CAD_NOT_STEP" in refused(env, req).codes
    env.catalog.update_part(env.ob.part_id, cad_file_path=None, cad_file_hash=None)
    assert "CAD_MISSING" in refused(env, req).codes
    # Nothing was stored, announced or compiled by any refusal.
    assert env.store.list_all() == [] and env.line.events_after(0) == []
    assert env.store.last_compile_attempt(env.ob.part_id) is None and compiled_dirs(env.root) == []


def test_drawing_missing_and_sku_rules(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case, with_drawing=False)
    assert refused(env).codes == ("DRAWING_MISSING",)
    # Two catalog parts whose numbers canonicalise to the same SKU collide (409).
    other = env.catalog.create_part("syn hat 1", revision="A")
    err = refused(env)
    assert "SKU_COLLISION" in err.codes and err.http_status == 409
    env.catalog.delete_part(other.id)
    # A part number that is not a valid SKU.
    bad = make_env(tmp_path / "bad", hat_case, part_number="BAD/SKU")
    assert "SKU_INVALID" in refused(bad).codes
    # The registry is authoritative too: the SKU registered to another part id.
    env.store.register_sku("SYN-HAT-1", "someone-else")
    assert "SKU_COLLISION" in refused(env).codes


# ---------------------------------------------------------------------------- compile refusals


def test_compile_blockers_are_returned_and_nothing_is_released(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    # The operator removes a real bend ("not a bend") and adds a mark where the CAD has none.
    def mutate(d):
        victim = d["marks"][1]
        victim.update(status="removed", issue="not_a_bend", review_status="pending")
        d["sequence"] = [f for f in d["sequence"] if f != victim["feature_id"]]
        d["machine_moves"] = [[f for f in m if f != victim["feature_id"]] for m in d["machine_moves"]]
        d["machine_moves"] = [m for m in d["machine_moves"] if m]
        d["machine_moves_explicit"] = bool(d["machine_moves"])
        d["marks"].append({
            "feature_id": "manual:ghost", "label": "ghost", "source": "manual", "status": "active",
            "review_status": "confirmed", "issue": "missing", "corrects_feature_id": None, "note": "",
            "line_start": [5000.0, 5000.0, 5000.0], "line_end": [5000.0, 5100.0, 5000.0],
            "target_angle": None, "target_radius": None,
        })  # fmt: skip
        d["sequence"].append("manual:ghost")
    saved = resave_draft(env, mutate)
    req = release_request(env.ob)
    req["expected_draft_revision"] = saved["draft_revision"]
    req["bend_overrides"] = {**req["bend_overrides"], "manual:ghost": {"manual_included_angle_deg": 90.0}}
    req["bend_overrides"].pop(env.ob.draft["marks"][1]["feature_id"], None)
    with pytest.raises(ReleaseGateError) as err:
        do_release(env, req)
    exc = err.value
    assert {"COMPILE_BLOCKED", "UNMARKED_BEND", "MARK_NOT_IN_FLANGE_GRAPH"} <= set(exc.codes)
    assert exc.http_status == 422
    report = exc.compile_report
    assert report["status"] == "BLOCKED" and report["unmarked_graph_bends"]
    ghost = [i for i in report["issues"] if i["code"] == "MARK_NOT_IN_FLANGE_GRAPH"]
    assert ghost and ghost[0]["bend_label"] is not None
    assert exc.to_detail()["compile"]["blocker_codes"] == report["blocker_codes"]
    # Nothing released, staging cleaned, the attempt logged for the checklist, no line event.
    assert env.store.list_all() == [] and compiled_dirs(env.root) == []
    attempt = env.store.last_compile_attempt(env.ob.part_id)
    assert attempt.status == "BLOCKED" and "UNMARKED_BEND" in attempt.codes
    assert env.line.events_after(0) == []
    assert (env.root / "logs" / "recipe_compile").is_dir()


_FAKE_E1 = textwrap.dedent(
    '''
    """Fake E1 compile for C05 failure-mapping tests (mode from mode.txt next to this file)."""
    import argparse, json, os, shutil, sys, time
    from pathlib import Path
    here = Path(__file__).resolve().parent
    mode = (here / "mode.txt").read_text().strip()
    p = argparse.ArgumentParser()
    for a in ("--recipe", "--cad", "--out", "--flat", "--shop-k", "--config"):
        p.add_argument(a)
    args = p.parse_args()
    out = Path(args.out)
    if mode == "sleep":
        time.sleep(30)
    elif mode == "crash":
        print("ERROR\\t" + json.dumps({"code": "COMPILE_TIMEOUT", "message": "boom"}), flush=True)
        sys.exit(3)
    elif mode == "input_error":
        out.mkdir(parents=True)
        (out / "compile_error.json").write_text(json.dumps({"code": "CAD_HASH_MISMATCH", "message": "bytes differ"}))
        sys.exit(2)
    elif mode == "garbage":
        out.mkdir(parents=True)
        (out / "compiled_part.json").write_text("{not json")
        sys.exit(0)
    elif mode == "foreign":
        shutil.copytree(here / "foreign", out)
        sys.exit(0)
    elif mode == "silent":
        sys.exit(0)
    '''
)


@pytest.fixture(scope="module")
def foreign_compile(tmp_path_factory, hat_case: Path) -> Path:
    """A real, valid compile of *another* recipe (E2's own recipe.json for the same STEP)."""
    import subprocess

    out = tmp_path_factory.mktemp("foreign") / "out"
    subprocess.run(
        [sys.executable, "-m", "domains.production.engine.compile", "--recipe", str(hat_case / "recipe.json"),
         "--cad", str(hat_case / "part.step"), "--flat", str(hat_case / "flat.dxf"), "--out", str(out)],
        cwd=REPO, check=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )  # fmt: skip
    return out


@pytest.mark.parametrize(
    "mode,codes",
    [
        ("sleep", {"COMPILE_TIMEOUT"}),
        ("crash", {"COMPILE_FAILED"}),
        ("input_error", {"COMPILE_FAILED", "CAD_HASH_MISMATCH"}),
        ("garbage", {"COMPILE_FAILED"}),
        ("silent", {"COMPILE_FAILED"}),
        ("foreign", {"COMPILE_INCONSISTENT"}),
    ],
)
def test_compile_failures_map_to_explicit_codes(tmp_path: Path, hat_case: Path, foreign_compile: Path,
                                                mode: str, codes: set[str]) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "fake_e1.py").write_text(_FAKE_E1)
    (engine / "mode.txt").write_text(mode)
    shutil.copytree(foreign_compile, engine / "foreign")
    launcher = CompileLauncher(engine_root=engine, python_exe=Path(sys.executable), module="fake_e1",
                               timeout_s=1.5, kill_grace_s=0.5)
    env = make_env(tmp_path, hat_case)
    with pytest.raises(ReleaseGateError) as err:
        do_release(env, compiler=launcher)
    assert set(err.value.codes) == codes
    if mode == "foreign":
        assert "RECIPE_REF_MISMATCH" in err.value.details["COMPILE_INCONSISTENT"]
    assert env.store.list_all() == [] and compiled_dirs(env.root) == []
    assert env.store.last_compile_attempt(env.ob.part_id).status == "FAILED"


# ---------------------------------------------------------------------------- races and side effects


class _StaleVersionStore:
    """A store whose version read is stale (another releaser inserted in between)."""

    def __init__(self, inner: RecipeStore) -> None:
        self._inner = inner

    def max_version(self, sku: str) -> int:
        return 0

    def get_active(self, sku: str):
        return None

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_concurrent_release_loser_is_refused_and_cleaned(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    first = do_release(env)
    req = release_request(env.ob)
    req["default_tolerances"] = {**req["default_tolerances"], "flange_mm": {"plus": 0.4, "minus": 0.4}}
    with pytest.raises(ReleaseGateError) as err:
        do_release(env, req, store=_StaleVersionStore(env.store))
    assert err.value.codes == ("CONCURRENT_RELEASE",) and err.value.http_status == 409
    assert compiled_dirs(env.root) == [first.recipe.recipe_sha256]
    assert [r.recipe_id for r in env.store.list_all()] == ["SYN-HAT-1@v1"]
    assert [e["kind"] for e in env.line.events_after(0)].count("RECIPE_RELEASED") == 1


class _BrokenLineStore:
    def append_event(self, *a, **k):
        raise RuntimeError("line store down")


def test_event_failure_never_rolls_back_a_release(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    out = do_release(env, line_store=_BrokenLineStore())
    assert out.outcome == "released" and env.store.get_active("SYN-HAT-1").recipe_id == "SYN-HAT-1@v1"


def test_export_refuses_research_dirs(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    (fake_repo / "data").mkdir(parents=True)
    assert out_path_refusal(fake_repo / "data" / "x.jsonl", fake_repo) is not None
    assert out_path_refusal(fake_repo / "data" / "deep" / "x.jsonl", fake_repo) is not None
    assert out_path_refusal(tmp_path / "elsewhere" / "x.jsonl", fake_repo) is None
    # The real CLI refuses the real repo's data/ before touching anything.
    target = REPO / "data" / "c05_export_refusal_probe.jsonl"
    assert export_main(["--data-root", str(tmp_path), "--out", str(target)]) == 2
    assert not target.exists()
    # A data root without a recipe store is refused, not created.
    assert export_main(["--data-root", str(tmp_path / "nowhere"), "--out", str(tmp_path / "o.jsonl")]) == 2
    assert not (tmp_path / "nowhere").exists()


# ---------------------------------------------------------------------------- recompute sink (CompiledPartSink)


def _variant(src: Path, dest: Path, *, version: str, blocked: bool = False) -> CompiledPartV1:
    """A recompute output of the same recipe: another compiler version, optionally BLOCKED."""
    shutil.copytree(src, dest)
    data = json.loads((dest / "compiled_part.json").read_text())
    data.pop("compiled_part_sha256")
    data["compiler"]["version"] = version
    if blocked:
        data["status"] = "BLOCKED"
        data["issues"] = [*data["issues"], {"code": "STAGE_MODEL_FAILED", "severity": "BLOCKER",
                                            "detail": "a newer E1 refuses this part"}]
    part = CompiledPartV1.finalize(**data)
    (dest / "compiled_part.json").write_text(part.model_dump_json(by_alias=True))
    return part


def test_recompute_blocked_keeps_the_working_install(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    recipe = do_release(env).recipe
    sha = recipe.recipe_sha256
    original = env.store.get_compiled(sha)
    install = env.root / Path(original.stored_rel_path).parent
    bench_before = env.store.get_benchmark_reference(recipe.recipe_id)
    resolver = LineRecipeResolver(env.store, data_root=env.root)
    assert resolver.resolve("SYN-HAT-1_FULL_SCAN 1.ply").status == "MATCHED"

    # A recompute (newer E1) that BLOCKS: the line keeps scanning against the working compile.
    blocked = _variant(install, tmp_path / "job_blocked", version="e1-next-blocks", blocked=True)
    resolver.store_compiled(sha, tmp_path / "job_blocked", status="BLOCKED",
                            compiled_part_sha256=blocked.compiled_part_sha256, blocker_codes=blocked.blocker_codes)
    kept = env.store.get_compiled(sha)
    assert kept == original and (install / "compiled_part.json").is_file()
    assert CompiledPartV1.model_validate_json((install / "compiled_part.json").read_bytes()).compiled_part_sha256 == (
        original.compiled_part_sha256
    )
    out = resolver.resolve("SYN-HAT-1_FULL_SCAN 1.ply")
    assert out.status == "MATCHED" and out.compiled_part_sha256 == original.compiled_part_sha256
    assert env.store.get_benchmark_reference(recipe.recipe_id) == bench_before
    assert bench_before["compiled_part_sha256"] == original.compiled_part_sha256
    assert (env.root / original.stored_rel_path).is_file()
    assert env.store.get_expectations(sha)["status"] == "READY"
    # The blocked recompute is kept for the wizard (history + a compile attempt of the part).
    hist = env.store.list_compiled_history(sha)
    assert [(h.reason, h.status, h.compiled_part_sha256) for h in hist] == [
        ("RECOMPUTE_BLOCKED", "BLOCKED", blocked.compiled_part_sha256)
    ]
    assert (env.root / hist[0].stored_rel_path).is_file()
    attempt = env.store.last_compile_attempt(env.ob.part_id)
    assert attempt.status == "BLOCKED" and "STAGE_MODEL_FAILED" in attempt.codes
    assert attempt.report["source"] == "RECOMPUTE" and attempt.content_sha256 == recipe.content_sha256
    # A recompute output made for another (stored) recipe is refused outright, and changes nothing.
    req = release_request(env.ob)
    req["default_tolerances"] = {**req["default_tolerances"], "angle_deg": {"plus": 0.8, "minus": 0.8}}
    v2 = do_release(env, req).recipe
    v2_before = env.store.get_compiled(v2.recipe_sha256)
    with pytest.raises(ValueError):
        resolver.store_compiled(v2.recipe_sha256, tmp_path / "job_blocked", status="BLOCKED",
                                compiled_part_sha256=blocked.compiled_part_sha256,
                                blocker_codes=blocked.blocker_codes)
    assert env.store.get_compiled(v2.recipe_sha256) == v2_before


def test_recompute_compiled_supersedes_and_keeps_the_pinned_artifact(tmp_path: Path, hat_case: Path) -> None:
    env = make_env(tmp_path, hat_case)
    recipe = do_release(env).recipe
    sha = recipe.recipe_sha256
    original = env.store.get_compiled(sha)
    install = env.root / Path(original.stored_rel_path).parent
    resolver = LineRecipeResolver(env.store, data_root=env.root)

    newer = _variant(install, tmp_path / "job_new", version="e1-next")
    assert newer.compiled_part_sha256 != original.compiled_part_sha256
    resolver.store_compiled(sha, tmp_path / "job_new", status="COMPILED",
                            compiled_part_sha256=newer.compiled_part_sha256, blocker_codes=())
    now = env.store.get_compiled(sha)
    assert now.status == "COMPILED" and now.compiled_part_sha256 == newer.compiled_part_sha256
    assert now.stored_rel_path == original.stored_rel_path  # the install path is stable
    # Benchmark row and expectations move to the new artifact, which exists.
    bench = env.store.get_benchmark_reference(recipe.recipe_id)
    assert bench["compiled_part_sha256"] == newer.compiled_part_sha256 and bench["algo_version"] == "e1-next"
    assert env.store.get_expectations(sha)["algo_version"] == "e1-next"
    # The superseded artifact is kept, and a job pinned to it still loads it.
    hist = env.store.list_compiled_history(sha)
    assert [(h.reason, h.compiled_part_sha256) for h in hist] == [("SUPERSEDED", original.compiled_part_sha256)]
    pinned = resolver.load(recipe.recipe_id, compiled_part_sha256=original.compiled_part_sha256)
    assert pinned.compiled_part_sha256 == original.compiled_part_sha256
    old_json = CompiledPartV1.model_validate_json((pinned.compiled_dir / "compiled_part.json").read_bytes())
    assert old_json.compiled_part_sha256 == original.compiled_part_sha256
    for stage in old_json.stages:
        assert _sha(pinned.compiled_dir / stage.sample_cloud.rel_path) == stage.sample_cloud.sha256
    current = resolver.load(recipe.recipe_id)
    assert current.compiled_part_sha256 == newer.compiled_part_sha256 and current.compiled_dir == install.resolve()
    # An unknown pin falls back to the installed compile (the runner's hash check then refuses it).
    assert resolver.load(recipe.recipe_id, compiled_part_sha256="0" * 64).compiled_part_sha256 == (
        newer.compiled_part_sha256
    )
