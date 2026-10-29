"""C05 onboarding checklist: every state and the codes that block it (E2 synthetic hat, tmp catalog).

The checklist is a read-only dry run of the release gate (no file is frozen,
nothing compiled); RELEASED / RELEASED_STALE use one real release.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from apps.production_line.recipes.onboarding import ONBOARDING_STATES, checklist
from apps.production_line.recipes.release_service import (
    CompileLauncher,
    ReleaseGateError,
    prepare_release,
)
from apps.production_line.recipes.synthetic_onboarding import release_request
from tests.test_prod_release_service import (  # noqa: F401 (fixture)
    _FAKE_E1,
    ADMIN,
    do_release,
    hat_case,
    make_env,
    resave_draft,
)

assert hat_case  # the module-scoped E2 fixture is reused here


def check(env, part_id: str | None = None) -> dict:
    doc = checklist(part_id or env.ob.part_id, catalog=env.port, recipe_store=env.store, data_root=env.root)
    assert doc is not None and doc["onboarding_state"] in ONBOARDING_STATES
    return doc


def step(doc: dict, key: str) -> dict:
    return next(item for item in doc["checklist"] if item["key"] == key)


def save_tolerances(env) -> None:
    req = release_request(env.ob)
    req.pop("expected_draft_revision")
    current = env.store.get_release_draft(env.ob.part_id)
    env.store.put_release_draft(env.ob.part_id, {**req, "expected_draft_revision": 0},
                                current.revision if current else 0)


def test_unknown_part_is_none(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    assert checklist("nope", catalog=env.port, recipe_store=env.store, data_root=env.root) is None


def test_needs_cad_and_step_only(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    bare = env.catalog.create_part("BARE-1", revision="A")
    doc = check(env, bare.id)
    assert doc["onboarding_state"] == "NEEDS_CAD" and "CAD_MISSING" in doc["blockers"]
    assert not step(doc, "cad")["ok"] and doc["ready_to_release"] is False and doc["sku"] == "BARE-1"
    stl = tmp_path / "x.stl"
    stl.write_bytes(b"solid x\nendsolid x\n")
    env.catalog.update_part(bare.id, cad_file_path=str(stl), cad_file_hash=hashlib.sha256(stl.read_bytes()).hexdigest())
    doc = check(env, bare.id)
    assert doc["onboarding_state"] == "NEEDS_CAD" and "CAD_NOT_STEP" in step(doc, "cad")["codes"]


def test_needs_drawing(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case, with_drawing=False)
    save_tolerances(env)
    doc = check(env)
    assert doc["onboarding_state"] == "NEEDS_DRAWING" and step(doc, "drawing")["codes"] == ["DRAWING_MISSING"]
    assert step(doc, "cad")["ok"] and step(doc, "bend_order")["ok"]


def test_needs_bend_order(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case, confirm_marks=False, confirm_sequence=False)
    save_tolerances(env)
    doc = check(env)
    assert doc["onboarding_state"] == "NEEDS_BEND_ORDER"
    assert {"SEQUENCE_NOT_CONFIRMED", "UNCONFIRMED_ACTIVE_MARKS"} <= set(step(doc, "bend_order")["codes"])
    # No draft at all.
    other = env.catalog.create_part("NODRAFT-1", revision="A")
    part = env.catalog.get_part(env.ob.part_id)
    env.catalog.update_part(other.id, cad_file_path=part.cad_file_path, cad_file_hash=part.cad_file_hash,
                            drawing_file_path=part.drawing_file_path, drawing_file_hash=part.drawing_file_hash)
    doc = check(env, other.id)
    assert doc["onboarding_state"] == "NEEDS_BEND_ORDER" and "DRAFT_MISSING" in doc["blockers"]


def test_tolerances_then_ready_then_flat_blocks(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    doc = check(env)
    assert doc["onboarding_state"] == "NEEDS_TOLERANCES" and step(doc, "tolerances")["codes"] == ["TOLERANCES_MISSING"]
    save_tolerances(env)
    doc = check(env)
    assert doc["onboarding_state"] == "READY_TO_RELEASE" and doc["ready_to_release"] is True and doc["blockers"] == []
    assert doc["links"]["vendor_setup"].endswith(env.ob.part_id) and doc["active_recipe"] is None
    # The dry run freezes and compiles nothing.
    assert not (env.root / "recipes" / "cad").exists() and env.store.last_compile_attempt(env.ob.part_id) is None
    # An attached, undecided flat blocks the release until the operator decides.
    flat = hat_case / "flat.dxf"
    sha = hashlib.sha256(flat.read_bytes()).hexdigest()
    dest = env.root / "recipes" / "flats" / f"{sha}.dxf"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(flat.read_bytes())
    env.store.put_part_flat(env.ob.part_id, flat_sha256=sha, format="DXF", original_filename="flat.dxf",
                            stored_rel_path=f"recipes/flats/{sha}.dxf", size_bytes=dest.stat().st_size,
                            state="UPLOADED", parse={"status": "PARSED"})
    doc = check(env)
    assert doc["onboarding_state"] == "NEEDS_BEND_ORDER" and step(doc, "flat")["codes"] == ["FLAT_BINDING_UNCONFIRMED"]
    assert doc["flat"]["state"] == "UPLOADED"
    rec = env.store.get_part_flat(env.ob.part_id)
    env.store.update_part_flat(env.ob.part_id, rec.revision, state="DECLINED", binding={"state": "UNBOUND"})
    assert check(env)["onboarding_state"] == "READY_TO_RELEASE"
    dest.write_bytes(b"0\nSECTION\n0\nEOF\n")  # the stored bytes changed behind our back
    assert "FLAT_HASH_MISMATCH" in step(check(env), "flat")["codes"]


def _dry_run_content(env) -> str:
    stored = env.store.get_release_draft(env.ob.part_id).request
    draft = env.catalog.get_vendor_setup_draft(env.ob.part_id)
    req = {**stored, "expected_draft_revision": draft["draft_revision"]}
    return prepare_release(env.ob.part_id, req, ADMIN, catalog=env.port, store=env.store,
                           data_root=env.root).recipe.content_sha256


def test_last_compile_blocks_only_the_same_recipe_content(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    save_tolerances(env)
    content = _dry_run_content(env)
    env.store.record_compile_attempt(env.ob.part_id, status="BLOCKED", codes=["COMPILE_BLOCKED", "UNMARKED_BEND"],
                                     report={"issues": [{"code": "UNMARKED_BEND"}]}, content_sha256=content)
    doc = check(env)
    assert doc["onboarding_state"] == "NEEDS_BEND_ORDER" and step(doc, "compile")["codes"] == ["COMPILE_BLOCKED",
                                                                                               "UNMARKED_BEND"]
    assert doc["last_compile"]["applies_to_current_draft"] is True and doc["last_compile"]["blocking"] is True
    assert "UNMARKED_BEND" in doc["blockers"] and doc["ready_to_release"] is False
    # A change that reaches the recipe but not the Vendor Setup draft (the release thickness) clears it.
    current = env.store.get_release_draft(env.ob.part_id)
    corrected = round(float(current.request.get("thickness_mm") or 2.0) + 0.1, 3)
    env.store.put_release_draft(env.ob.part_id, {**current.request, "thickness_mm": corrected}, current.revision)
    doc = check(env)
    assert doc["last_compile"]["applies_to_current_draft"] is False and step(doc, "compile")["ok"]
    assert doc["ready_to_release"] is True and doc["onboarding_state"] == "READY_TO_RELEASE"


def test_schema_v1_attempts_fall_back_to_the_draft_basis(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    from domains.production.recipe.builder import draft_sha256

    env = make_env(tmp_path, hat_case)
    save_tolerances(env)
    part = env.catalog.get_part(env.ob.part_id)
    draft = env.catalog.get_vendor_setup_draft(env.ob.part_id)
    env.store.record_compile_attempt(env.ob.part_id, status="BLOCKED", codes=["COMPILE_BLOCKED", "UNMARKED_BEND"],
                                     report={}, cad_sha256=part.cad_file_hash, draft_sha256=draft_sha256(draft))
    doc = check(env)
    assert doc["last_compile"]["content_sha256"] is None and step(doc, "compile")["codes"] == ["COMPILE_BLOCKED",
                                                                                               "UNMARKED_BEND"]
    # The operator edits the draft: the old compile no longer describes it.
    resave_draft(env, lambda d: d.update(completed_steps=1))
    doc = check(env)
    assert doc["last_compile"]["applies_to_current_draft"] is False and step(doc, "compile")["ok"]


@pytest.mark.parametrize("status,code", [("FAILED", "COMPILE_FAILED"), ("FAILED", "COMPILE_INCONSISTENT")])
def test_failed_attempts_are_advisory(tmp_path: Path, hat_case: Path, status: str, code: str) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    save_tolerances(env)
    env.store.record_compile_attempt(env.ob.part_id, status=status, codes=[code], report={},
                                     content_sha256=_dry_run_content(env))
    doc = check(env)
    assert step(doc, "compile")["ok"] and step(doc, "compile")["advisory"] == [code]
    assert doc["ready_to_release"] is True and doc["blockers"] == [] and doc["last_compile"]["blocking"] is False


def test_a_timed_out_release_leaves_the_part_releasable(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    """Reviewer repro: one COMPILE_TIMEOUT must not disable Release in the wizard."""
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "fake_e1.py").write_text(_FAKE_E1)
    (engine / "mode.txt").write_text("sleep")
    slow = CompileLauncher(engine_root=engine, python_exe=Path(sys.executable), module="fake_e1", timeout_s=1.0,
                           kill_grace_s=0.5)
    env = make_env(tmp_path, hat_case)
    save_tolerances(env)
    with pytest.raises(ReleaseGateError) as err:
        do_release(env, compiler=slow)
    assert err.value.codes == ("COMPILE_TIMEOUT",)
    attempt = env.store.last_compile_attempt(env.ob.part_id)
    assert attempt.status == "FAILED" and attempt.content_sha256 == _dry_run_content(env)
    doc = check(env)
    assert doc["ready_to_release"] is True and doc["onboarding_state"] == "READY_TO_RELEASE" and doc["blockers"] == []
    assert step(doc, "compile")["advisory"] == ["COMPILE_TIMEOUT"] and doc["last_compile"]["status"] == "FAILED"
    # And the retry with the real E1 releases.
    assert do_release(env).outcome == "released"


def test_released_and_released_stale(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    save_tolerances(env)
    do_release(env)
    doc = check(env)
    assert doc["onboarding_state"] == "RELEASED" and doc["active_recipe"]["recipe_id"] == "SYN-HAT-1@v1"
    assert doc["active_recipe"]["compile"]["status"] == "COMPILED"
    assert doc["release_would_be_unchanged"] is True and doc["ready_to_release"] is False
    # A tolerance change in the release draft would make a new version: stale, without any CAD/draft change.
    current = env.store.get_release_draft(env.ob.part_id)
    tighter = {**current.request, "default_tolerances": {**current.request["default_tolerances"],
                                                         "angle_deg": {"plus": 0.4, "minus": 0.4}}}
    env.store.put_release_draft(env.ob.part_id, tighter, current.revision)
    doc = check(env)
    assert doc["onboarding_state"] == "RELEASED_STALE" and doc["ready_to_release"] is True
    assert doc["release_would_be_unchanged"] is False
    env.store.put_release_draft(env.ob.part_id, current.request, current.revision + 1)
    assert check(env)["onboarding_state"] == "RELEASED"
    # A re-saved draft (new revision) is no longer what the line inspects against.
    resave_draft(env, lambda d: d.update(completed_steps=1))
    doc = check(env)
    assert doc["onboarding_state"] == "RELEASED_STALE" and doc["ready_to_release"] is True
    # So is a republished drawing.
    env2 = make_env(tmp_path / "b", hat_case)
    save_tolerances(env2)
    do_release(env2)
    pdf = tmp_path / "b" / "new.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%new drawing\n")
    env2.catalog.update_part(env2.ob.part_id, drawing_file_path=str(pdf),
                             drawing_file_hash=hashlib.sha256(pdf.read_bytes()).hexdigest())
    assert check(env2)["onboarding_state"] == "RELEASED_STALE"


@pytest.mark.parametrize("state", ONBOARDING_STATES)
def test_states_vocabulary_is_closed(state: str) -> None:
    assert state.isupper() and " " not in state
