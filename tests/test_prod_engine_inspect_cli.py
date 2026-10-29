"""End-to-end tests of the per-scan engine CLI (E3) on E2 synthetic cases compiled by E1.

The cases are generated in a temporary directory (DEV seeds only, smoke density)
and compiled with the real E1 compile, so the engine sees exactly what the line
gives it. Truth (``truth.json``) is read by the tests only, never by the engine.

Run: PYT tests/test_prod_engine_inspect_cli.py -q   (about three minutes)
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from contracts.production.consistency import check_result_against_compiled
from contracts.production.compiled_part import CompiledPartV1
from contracts.production.inspection_result import BendGeometryFileV1, ShopfloorInspectionResultV1
from contracts.production.part_recipe import PartRecipeV1

REPO = Path(__file__).resolve().parents[1]
RUNTIME_FIELDS = ("created_at",)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: The smoke HAT's STAGE_K case is decoded at S2; its S2 backgauge runs across the flat strip of B4.
KINKED_SEQUENCE_INDEX = 4
KINK_FRACTION = 0.06


@contextlib.contextmanager
def _kinked_unformed_bend(sequence_index: int, fraction: float):
    """E2's PARTIAL_BEND kinematics on a not-yet-formed bend of a staged case (and no BEND_LINE_SHIFT).

    A fraction below E3's NOT_YET_FORMED threshold (0.12) leaves a few degrees of
    kink on a bend the decode calls unformed: the island a backgauge crosses is
    then no longer one plane.
    """
    from domains.production.engine.synthetic import cases as e2_cases, injection

    orig_build, orig_range = e2_cases._build_scenario, injection.PARTIAL_FRACTION_RANGE

    def build(spec, design, nominal, rng):
        sc, d = orig_build(spec, design, nominal, rng)
        inj = tuple(i for i in sc.injections if i.kind != "BEND_LINE_SHIFT")
        inj += (injection.Injection("PARTIAL_BEND", {"sequence_index": sequence_index, "fraction": fraction}),)
        return dataclasses.replace(sc, injections=inj), d

    e2_cases._build_scenario, injection.PARTIAL_FRACTION_RANGE = build, (0.0, 1.0)
    try:
        yield
    finally:
        e2_cases._build_scenario, injection.PARTIAL_FRACTION_RANGE = orig_build, orig_range


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    root = tmp_path_factory.mktemp("e3_cases")
    specs = plan_cases("smoke", "DEV")
    out = {}
    for name, family, scenario, plan, feats in (
            ("hat", "HAT", "COMPLETE_NOMINAL", "FULL", {}), ("hat_stage", "HAT", "STAGE_K", "FULL", {}),
            ("lb_errors", "L_BRACKET", "COMPLETE_ERRORS", "FULL", {}),
            ("ucs_stage1", "U_CHANNEL", "STAGE_1", "FULL", {"symmetric": True}),
            ("td_extra", "TWO_DIRECTION_TAB", "EXTRA_BEND", "FULL", {}),
            ("hat_top", "HAT", "COMPLETE_ERRORS", "TOP_ONLY", {}),
            ("td_partial", "TWO_DIRECTION_TAB", "PARTIAL_BEND", "FULL", {}),
            ("ucs", "U_CHANNEL", "COMPLETE_NOMINAL", "FULL", {"symmetric": True}),
            ("hat_kink", "HAT", "STAGE_K", "FULL", {})):
        spec = next(s for s in specs if s.family == family)
        spec = dataclasses.replace(spec, case_id=f"E3T-{name}", scenario=scenario, exact=scenario == "COMPLETE_NOMINAL",
                                   plan=plan, features=dataclasses.replace(spec.features, **feats))
        if name == "hat_kink":
            with _kinked_unformed_bend(KINKED_SEQUENCE_INDEX, KINK_FRACTION):
                generate_case(spec, root)
        else:
            generate_case(spec, root)
        d = root / spec.case_id
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        res = compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf")
        assert res.status == "COMPILED"
        out[name] = d
    return out


def run_cli(case: Path, out: Path, *extra: str, scan: Path | None = None, scan_sha: str | None = None,
            compiled: Path | None = None, recipe: Path | None = None) -> tuple[int, list[tuple[str, dict]]]:
    scan = scan or case / "scan.ply"
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(compiled or case / "compiled"),
           "--recipe", str(recipe or case / "recipe.json"), "--scan", str(scan), "--out", str(out),
           "--scan-sha256", scan_sha or (_sha(scan) if scan.exists() else "0" * 64), *extra]
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    records = []
    for line in proc.stdout.splitlines():
        tag, _, payload = line.partition("\t")
        assert tag in ("PROGRESS", "RESULT", "ERROR"), f"stdout carries a non-protocol line: {line!r}"
        records.append((tag, json.loads(payload)))
    return proc.returncode, records


def load(case: Path, out: Path):
    result = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    recipe = PartRecipeV1.model_validate_json((case / "recipe.json").read_bytes())
    truth = json.loads((case / "truth.json").read_text())
    return result, compiled, recipe, truth


def chars(result):
    return {c.characteristic_id: c for b in result.bends for c in b.characteristics}


def stripped(path: Path) -> dict:
    doc = json.loads(path.read_text())
    for key in RUNTIME_FIELDS:
        doc.pop(key, None)
    doc.pop("context", None)
    doc["engine"].pop("duration_s", None)
    doc["engine"].pop("peak_rss_bytes", None)
    return doc


# --------------------------------------------------------------------------- complete part
def test_complete_part_is_valid_consistent_and_close_to_truth(cases, tmp_path):
    case = cases["hat"]
    out = tmp_path / "run"
    rc, rec = run_cli(case, out)
    assert rc == 0
    result_rec = [p for t, p in rec if t == "RESULT"]
    assert len(result_rec) == 1 and result_rec[0]["result"] == "result.json"
    result, compiled, recipe, truth = load(case, out)
    assert result_rec[0]["summary_state"] == result.summary.state
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert result.inspection_validity.state == "VALID"
    assert all(b.state == "FORMED" for b in result.bends)
    assert result.stage_decision.stage_model_id == f"S{len(result.bends)}"
    measured = 0
    for cid, c in chars(result).items():
        tc = truth["characteristics"].get(cid)
        if not tc or tc.get("truth") is None or c.state != "MEASURED":
            continue
        measured += 1
        err = c.actual - tc["truth"]
        limit = 0.3 if c.unit == "deg" else 0.35
        assert abs(err) < limit, (cid, err)
        assert c.uncertainty > 0 and c.expected is not None
    assert measured >= 12
    # nothing NOT_MEASURED carries a value; the nominal is never copied into actual
    for c in chars(result).values():
        if c.state != "MEASURED":
            assert c.actual is None and c.uncertainty is None and c.deviation is None


def test_output_files_are_index_aligned_and_in_the_declared_frame(cases, tmp_path):
    case = cases["hat"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, *_ = load(case, out)
    va = result.viewer_assets
    raw = (out / va.registered_scan).read_bytes()
    header, _, body = raw.partition(b"end_header\n")
    assert b"binary_little_endian 1.0" in header and b"element face" not in header
    n = int([ln for ln in header.split(b"\n") if ln.startswith(b"element vertex")][0].split()[-1])
    assert n == va.point_count == len(body) // 12
    dev = np.fromfile(out / va.deviations, dtype="<f4")
    lab = np.fromfile(out / va.segment_labels, dtype="<u2")
    assert len(dev) == n and len(lab) == n
    assert va.deviation_reference_stage == result.stage_decision.stage_model_id
    assert np.isfinite(dev).mean() > 0.8 and np.nanmedian(np.abs(dev)) < 0.2
    # the registered scan lands on the compiled final stage: the deviations are small
    geo = BendGeometryFileV1.model_validate_json((out / "bend_geometry.json").read_bytes())
    assert geo == BendGeometryFileV1.from_result(result)
    for f in ("display/stage_mesh.ply", "display/final_mesh.ply", "engine.log"):
        assert (out / f).is_file()


def test_result_is_deterministic_and_blind_to_the_scan_name(cases, tmp_path):
    case = cases["hat"]
    renamed = tmp_path / "SYN-OPAQUE_02_FULL_SCAN 1.ply"
    shutil.copyfile(case / "scan.ply", renamed)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_cli(case, out_a)[0] == 0
    assert run_cli(case, out_b, scan=renamed)[0] == 0
    assert stripped(out_a / "result.json") == stripped(out_b / "result.json")
    assert (out_a / "deviations.f32").read_bytes() == (out_b / "deviations.f32").read_bytes()


def test_engine_never_receives_job_identity(cases, tmp_path):
    """Plan §10 amendment 4 / ENGINE_IO §4: there is no --context option (argparse usage
    error, exit 2, nothing written) and result.context is always empty."""
    case = cases["hat"]
    ctx = tmp_path / "context.json"
    ctx.write_text(json.dumps({"job_id": "line_" + "a" * 32, "station_id": "st1", "scan_file_name": "x_SCAN_1 1.ply",
                               "serial": "77"}))
    refused = tmp_path / "refused"
    assert run_cli(case, refused, "--context", str(ctx))[0] == 2
    assert not (refused / "result.json").exists()
    out = tmp_path / "a"
    assert run_cli(case, out)[0] == 0
    res = json.loads((out / "result.json").read_text())
    assert all(v is None for v in res["context"].values()), res["context"]


# --------------------------------------------------------------------------- staged part
def test_mid_sequence_scan_decodes_bend_states_and_backgauge(cases, tmp_path):
    case = cases["hat_stage"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    want = truth["expected"]["bend_states_contract"]
    got = {b.bend_uid: b.state for b in result.bends}
    assert got == want
    assert result.stage_decision.stage_model_id == truth["stage"]["expected_stage_model_id"]
    for b in result.bends:
        if b.state == "NOT_YET_FORMED":
            assert all(c.state == "NOT_APPLICABLE" and c.decision != "FAIL" for c in b.characteristics)
    k = result.stage_decision.inferred_stage
    if 1 <= k <= len(result.bends) - 1:
        bg = [c for c in result.bends[k - 1].characteristics if c.kind == "BACKGAUGE_LENGTH"]
        assert len(bg) == 1 and bg[0].characteristic_id.endswith(f"/S{k}")
    # a FINAL recipe makes an unfinished part a review, never green
    assert result.summary.state != "ALL_MEASURED_IN_TOLERANCE"


def test_declared_stage_is_checked_never_trusted(cases, tmp_path):
    case = cases["hat_stage"]
    out_ok, out_bad = tmp_path / "ok", tmp_path / "bad"
    assert run_cli(case, out_ok)[0] == 0
    k = json.loads((out_ok / "result.json").read_text())["stage_decision"]["inferred_stage"]
    assert run_cli(case, out_bad, "--declared-stage", str(k + 1), "--declared-stage-source", "OPERATOR")[0] == 0
    bad = json.loads((out_bad / "result.json").read_text())["stage_decision"]
    assert "STAGE_MISMATCH" in bad["flags"] and bad["decision"] != "PASS" and bad["inferred_stage"] == k
    out_same = tmp_path / "same"
    assert run_cli(case, out_same, "--declared-stage", str(k), "--declared-stage-source", "SIDECAR")[0] == 0
    assert "STAGE_MISMATCH" not in json.loads((out_same / "result.json").read_text())["stage_decision"]["flags"]


# --------------------------------------------------------------------------- defects
def test_injected_errors_are_never_passed(cases, tmp_path):
    case = cases["lb_errors"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    cs = chars(result)
    for cid in truth["expected"]["characteristics_never_pass"]:
        if cid in cs:
            assert cs[cid].decision != "PASS", cid
    for cid, tc in truth["characteristics"].items():
        c = cs.get(cid)
        if c is None or c.state != "MEASURED" or tc.get("truth") is None or c.tolerance is None:
            continue
        if tc["truth_in_tolerance"] is False:
            assert c.decision in ("FAIL", "REVIEW"), cid
    assert result.summary.state != "ALL_MEASURED_IN_TOLERANCE" or not truth["expected"]["characteristics_never_pass"]


def test_mirrored_scan_is_never_passed(cases, tmp_path):
    case = cases["hat"]
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    pts, faces, _ = read_ply(case / "scan.ply")
    mirrored = tmp_path / "mirrored.ply"
    write_ply(mirrored, pts * np.array([-1.0, 1.0, 1.0]))
    out = tmp_path / "run"
    assert run_cli(case, out, scan=mirrored)[0] == 0
    res = json.loads((out / "result.json").read_text())
    assert res["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
    assert res["inspection_validity"]["state"] in ("MIRRORED_PART_SUSPECTED", "WRONG_PART_SUSPECTED",
                                                   "UNTRUSTED_REGISTRATION")
    assert res["inspection_validity"]["registration"]["raw_to_frame"] is None or \
        np.linalg.det(np.array(res["inspection_validity"]["registration"]["raw_to_frame"])[:3, :3]) > 0


def _truth_rows(result, truth):
    """(characteristic, truth value, truth deviation, truth in tolerance) for every MEASURED row with defined truth."""
    rows = []
    for cid, c in chars(result).items():
        tc = truth["characteristics"].get(cid)
        if c.state != "MEASURED" or not tc or tc.get("truth") is None or tc.get("well_defined") is False:
            continue
        rows.append((c, tc["truth"], tc.get("true_deviation"), tc.get("truth_in_tolerance")))
    return rows


def test_symmetric_part_mid_sequence_takes_the_order_consistent_twin(cases, tmp_path):
    # a symmetric U-channel with one flange formed is congruent to "the other flange formed";
    # its stage S1 is not symmetric (E1), so the pose is not ambiguous and the order decides
    case = cases["ucs_stage1"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert not result.inspection_validity.registration.pose_ambiguous
    assert {b.bend_uid: b.state for b in result.bends} == truth["expected"]["bend_states_contract"]
    assert result.stage_decision.stage_model_id == "S1"
    assert "OUT_OF_ORDER_SUSPECTED" not in result.stage_decision.flags
    for c, tv, _, _ in _truth_rows(result, truth):
        assert abs(c.actual - tv) <= c.uncertainty, c.characteristic_id


def test_unplanned_crease_is_reported_and_ends_the_flange_it_splits(cases, tmp_path):
    case = cases["td_extra"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert result.unplanned_creases and all(c.reason == "EXTRA_BEND_SUSPECTED" for c in result.unplanned_creases)
    assert result.summary.state not in truth["expected"]["summary_never"]
    labels = np.fromfile(out / result.viewer_assets.segment_labels, dtype="<u2")
    assert (labels == 65535).any()  # the creased segment's points carry the crease label
    cs = chars(result)
    split = [cid for cid, tc in truth["characteristics"].items() if tc.get("boundary_changed")]
    assert split
    for cid in split:
        c = cs[cid]
        assert c.decision != "PASS"
        if c.state == "MEASURED":  # measured to the crease's OML sharp, as the contract measures to a bend
            assert "UNPLANNED_CREASE" in c.evidence.method
            assert abs(c.actual - truth["characteristics"][cid]["truth"]) <= c.uncertainty
        else:
            assert c.reason == "AMBIGUOUS_SEGMENTATION"


def test_top_only_scan_keeps_u_honest(cases, tmp_path):
    # one-sided scans erode bare skin edges: the registration and the far edges carry that bias in U
    case = cases["hat_top"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    rows = _truth_rows(result, truth)
    assert len(rows) >= 10
    covered = sum(abs(c.actual - tv) <= c.uncertainty for c, tv, _, _ in rows)
    assert covered >= 0.9 * len(rows)
    for c, _, _, tin in rows:
        if c.tolerance is None or tin is None:
            continue
        assert not (tin is False and c.decision == "PASS"), c.characteristic_id
        assert not (tin is True and c.decision == "FAIL"), c.characteristic_id


def test_partially_formed_bend_and_the_bends_beyond_it(cases, tmp_path):
    # the fold kinematics of a partial bend (arc centred on the bend line) carry its children
    case = cases["td_partial"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert {b.bend_uid: b.state for b in result.bends} == truth["expected"]["bend_states_contract"]
    assert result.stage_decision.status == "PARTIAL_BEND_FOUND"
    partial = [b for b in result.bends if b.state == "PARTIALLY_FORMED"]
    assert partial and all(c.decision != "PASS" for b in partial for c in b.characteristics if c.kind == "INCLUDED_ANGLE")
    for c, tv, _, _ in _truth_rows(result, truth):
        assert abs(c.actual - tv) <= max(c.uncertainty, 0.05), c.characteristic_id


# --------------------------------------------------------------------------- honesty under bad input
def test_unit_mismatch_is_invalid_and_an_explicit_factor_is_accepted(cases, tmp_path):
    case = cases["lb_errors"]
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    pts, _, _ = read_ply(case / "scan.ply")
    inch = tmp_path / "inch.ply"
    write_ply(inch, pts / 25.4)
    out = tmp_path / "inch"
    assert run_cli(case, out, scan=inch)[0] == 0
    res = json.loads((out / "result.json").read_text())
    assert res["inspection_validity"]["state"] == "UNIT_MISMATCH_SUSPECTED"
    assert res["summary"]["state"] == "INVALID_INSPECTION"
    assert all(c["state"] != "MEASURED" for b in res["bends"] for c in b["characteristics"])
    assert not (out / "deviations.f32").exists() and not (out / "registered_scan.ply").exists()
    out2 = tmp_path / "inch_explicit"
    assert run_cli(case, out2, "--unit-factor", "25.4", scan=inch)[0] == 0
    res2 = json.loads((out2 / "result.json").read_text())
    assert res2["inspection_validity"]["state"] == "VALID"
    assert res2["inspection_validity"]["registration"]["unit_factor"] == 25.4


def test_heavy_occlusion_gives_unknown_not_an_error(cases, tmp_path):
    case = cases["hat"]
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    pts, _, _ = read_ply(case / "scan.ply")
    c = pts.mean(axis=0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    keep = (pts - c) @ axis < np.quantile((pts - c) @ axis, 0.55)
    cropped = tmp_path / "cropped.ply"
    write_ply(cropped, pts[keep])
    out = tmp_path / "run"
    assert run_cli(case, out, scan=cropped)[0] == 0
    res = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    states = [ch.state for b in res.bends for ch in b.characteristics]
    assert "NOT_MEASURED" in states or res.inspection_validity.state != "VALID"
    assert res.summary.state != "ALL_MEASURED_IN_TOLERANCE"


def test_time_budget_expiry_is_a_timeout_result(cases, tmp_path):
    case = cases["hat"]
    out = tmp_path / "run"
    rc, rec = run_cli(case, out, "--time-budget-s", "0.01")
    assert rc == 0
    res = json.loads((out / "result.json").read_text())
    assert res["inspection_validity"]["state"] == "TIMEOUT"
    assert all(c["state"] != "MEASURED" for b in res["bends"] for c in b["characteristics"])


def test_unreadable_scan_is_unsupported_input_result(cases, tmp_path):
    case = cases["hat"]
    junk = tmp_path / "junk.ply"
    junk.write_bytes(b"not a scan at all")
    out = tmp_path / "run"
    assert run_cli(case, out, scan=junk)[0] == 0
    res = json.loads((out / "result.json").read_text())
    assert res["inspection_validity"]["state"] == "UNSUPPORTED_INPUT"
    assert res["summary"]["state"] == "INVALID_INSPECTION"


# --------------------------------------------------------------------------- integrity (exit 2)
def _error_code(out: Path) -> str:
    return json.loads((out / "engine_error.json").read_text())["code"]


def test_integrity_failures_exit_2_with_the_contract_codes(cases, tmp_path):
    case = cases["hat"]
    rc, rec = run_cli(case, tmp_path / "a", scan_sha="0" * 64)
    assert rc == 2 and _error_code(tmp_path / "a") == "SCAN_HASH_MISMATCH" and rec[-1][0] == "ERROR"
    rc, _ = run_cli(case, tmp_path / "b", scan=tmp_path / "missing.ply")
    assert rc == 2 and _error_code(tmp_path / "b") == "SCAN_MISSING"
    rc, _ = run_cli(case, tmp_path / "c", compiled=tmp_path / "nowhere")
    assert rc == 2 and _error_code(tmp_path / "c") == "COMPILED_PART_MISSING"
    rc, _ = run_cli(case, tmp_path / "d", "--compiled-sha256", "f" * 64)
    assert rc == 2 and _error_code(tmp_path / "d") == "COMPILED_PART_HASH_MISMATCH"
    rc, _ = run_cli(case, tmp_path / "e", "--recipe-sha256", "e" * 64)
    assert rc == 2 and _error_code(tmp_path / "e") == "RECIPE_SHA_MISMATCH"
    # a stage file altered after the compile
    bad = tmp_path / "compiled_copy"
    shutil.copytree(case / "compiled", bad)
    npz = sorted((bad / "stages").glob("*.npz"))[0]
    blob = bytearray(npz.read_bytes())
    blob[-10] ^= 0xFF
    npz.write_bytes(bytes(blob))
    rc, _ = run_cli(case, tmp_path / "f", compiled=bad)
    assert rc == 2 and _error_code(tmp_path / "f") == "COMPILED_PART_HASH_MISMATCH"
    # a recipe that is not the compiled part's
    other = json.loads((case / "recipe.json").read_text())
    other["release_note"] = "edited"
    bad_recipe = tmp_path / "recipe.json"
    bad_recipe.write_text(json.dumps(other))
    rc, _ = run_cli(case, tmp_path / "g", recipe=bad_recipe)
    assert rc == 2 and _error_code(tmp_path / "g") == "RECIPE_SHA_MISMATCH"


def test_out_dir_must_be_empty_and_usage_errors_exit_2(cases, tmp_path):
    case = cases["hat"]
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    rc, rec = run_cli(case, out)
    assert rc == 2 and rec[-1][1]["code"] == "OUT_DIR_NOT_EMPTY"
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
    rc, rec = run_cli(case, tmp_path / "u", "--declared-stage", "1")
    assert rc == 2 and rec[-1][1]["code"] == "DECLARED_STAGE_INVALID"
    rc, rec = run_cli(case, tmp_path / "v", "--unit-factor", "2.5")
    assert rc == 2 and rec[-1][0] == "ERROR"


# --------------------------------------------------------------------------- library API
def test_library_api_agrees_with_the_cli(cases, tmp_path):
    from domains.production.engine.inspect import InspectOptions, inspect_scan, load_config, load_inputs

    case = cases["lb_errors"]
    out = tmp_path / "cli"
    assert run_cli(case, out)[0] == 0
    inputs = load_inputs(case / "compiled", case / "recipe.json", case / "scan.ply", _sha(case / "scan.ply"))
    outcome = inspect_scan(inputs, tmp_path / "lib", load_config(None), InspectOptions(write_outputs=False))
    cli = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    assert outcome.result.summary == cli.summary
    assert [b.state for b in outcome.result.bends] == [b.state for b in cli.bends]


# --------------------------------------------------------------------------- general measurements
def test_bend_pair_general_measurement_is_measured_between_the_same_loci(cases, tmp_path):
    from domains.production.engine.compile import compile_part

    case = cases["hat"]
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    recipe = json.loads((case / "recipe.json").read_text())
    b1, b4 = compiled.bends[0], compiled.bends[1]  # two parallel bends of the hat (brim bends)
    p = np.mean([b1.loci.oml_virtual_sharp.start_mm, b1.loci.oml_virtual_sharp.end_mm], axis=0)
    a4 = np.asarray(b4.loci.oml_virtual_sharp.start_mm)
    d4 = np.asarray(b4.loci.oml_virtual_sharp.end_mm) - a4
    d4 = d4 / np.linalg.norm(d4)
    q = a4 + ((p - a4) @ d4) * d4  # the point of b4's OML sharp facing p
    dist = float(np.linalg.norm(q - p))
    recipe["general_measurements"] = [{
        "measurement_id": "w1", "label": "brim to brim", "kind": "bend_pair", "bend_uids": [b1.bend_uid, b4.bend_uid],
        "point_start_mm": [float(x) for x in p], "point_end_mm": [float(x) for x in q], "cad_distance_mm": dist,
        "nominal_mm": dist, "nominal_source": "CAD_DISTANCE", "tolerance": {"plus": 0.5, "minus": 0.5},
        "required": True}]
    for key in ("content_sha256", "recipe_sha256"):
        recipe.pop(key)
    gm_recipe = PartRecipeV1.finalize(**recipe)
    gdir = tmp_path / "gm_case"
    gdir.mkdir()
    (gdir / "recipe.json").write_bytes(gm_recipe.model_dump_json(by_alias=True).encode())
    res = compile_part(gm_recipe, case / "part.step", gdir / "compiled", flat_path=case / "flat.dxf")
    assert res.status == "COMPILED"
    out = tmp_path / "run"
    rc, _ = run_cli(case, out, compiled=gdir / "compiled", recipe=gdir / "recipe.json")
    assert rc == 0
    result = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    assert check_result_against_compiled(result, res.compiled, gm_recipe) == []
    (row,) = result.general_measurements
    c = row.characteristic
    assert c.state == "MEASURED" and row.measured_start_mm is not None
    assert abs(c.actual - dist) < 0.5 and c.uncertainty > 0  # the nominal part: distance close to the CAD


# --------------------------------------------------------------------------- fix-stage regressions
def _raw_to_cad(truth) -> np.ndarray:
    return np.asarray(truth["scan"]["raw_to_cad"], dtype=np.float64)


def _occluded_far_edge(case: Path, out: Path, depth: float) -> tuple[Path, str]:
    """The scan without its points within ``depth`` mm of the far edge of the single bend's non-datum flange.

    A clamp or an out-of-view edge: the visible material ends short of the cut edge,
    and no cut wall is seen there. Returns the scan and that side's characteristic id.
    """
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    truth = json.loads((case / "truth.json").read_text())
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    (bend,) = compiled.bends
    flange = bend.child_panel_id if bend.parent_panel_id == compiled.datum_panel_id else bend.parent_panel_id
    panel = next(p for p in compiled.panels if p.panel_id == flange)
    axis = np.asarray(bend.axis.direction, dtype=np.float64)
    pts, _, _ = read_ply(case / "scan.ply")
    m = _raw_to_cad(truth)
    q = pts @ m[:3, :3].T + m[:3, 3]
    drop = np.zeros(len(q), dtype=bool)
    for fe in panel.free_edges:
        line = np.asarray(fe.polyline_mm, dtype=np.float64)
        for a, b in zip(line[:-1], line[1:]):
            d = b - a
            if abs(float(d @ axis)) < 0.98 * float(np.linalg.norm(d)):
                continue  # only the far edge, parallel to the bend
            s = np.clip((q - a) @ d / float(d @ d), 0.0, 1.0)
            drop |= np.linalg.norm(q - (a + s[:, None] * d), axis=1) < depth
    assert drop.any()
    write_ply(out, pts[~drop])
    side = next(sd.side for sd in bend.sides if sd.panel_id == flange)
    return out, f"{bend.bend_uid}/FLANGE_LENGTH/{side}"


def test_a_kinked_unformed_bend_is_unfolded_never_a_false_backgauge_fail(cases, tmp_path):
    # review finding: a not-yet-formed bend with a few degrees of kink cut the backgauge's far-edge run short
    # (a confident 12.8 mm FAIL on a part with no backgauge error)
    case = cases["hat_kink"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    kinked = next(b for b in result.bends if b.sequence_index == KINKED_SEQUENCE_INDEX)
    assert kinked.state == "NOT_YET_FORMED" and 0.02 < kinked.state_evidence.formed_fraction <= 0.12
    k = result.stage_decision.inferred_stage
    crossed = compiled.diagnostics["backgauge_nominals"][f"S{k}"]["crossed_bend_uids"]
    assert kinked.bend_uid in crossed  # the backgauge really runs across the kinked strip
    (bg,) = [c for b in result.bends for c in b.characteristics if c.kind == "BACKGAUGE_LENGTH"]
    assert bg.decision != "FAIL"
    assert bg.state == "MEASURED" and any(n.startswith(f"unfolded_{kinked.label}_deg=") for n in bg.evidence.notes)
    # no backgauge error was injected: the developed length is the stage model's nominal
    assert abs(bg.actual - bg.expected) <= bg.uncertainty


def test_unhashed_diagnostics_never_change_a_result(cases, tmp_path):
    # review finding: the backgauge U read E1 diagnostics, which compiled_part_sha256 does not cover
    case = cases["hat_stage"]
    comp = tmp_path / "compiled"
    shutil.copytree(case / "compiled", comp)
    doc = json.loads((comp / "compiled_part.json").read_text())
    assert doc["diagnostics"]["backgauge_nominals"]
    for rec in doc["diagnostics"]["backgauge_nominals"].values():
        rec["k_sigma_mm"], rec["d_nominal_d_k_mm"] = 0.0, 0.0
    (comp / "compiled_part.json").write_text(json.dumps(doc))
    original = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    tampered = CompiledPartV1.model_validate_json((comp / "compiled_part.json").read_bytes())
    assert tampered.compiled_part_sha256 == original.compiled_part_sha256  # outside the hash: the integrity checks pass
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_cli(case, out_a, "--compiled-sha256", original.compiled_part_sha256)[0] == 0
    assert run_cli(case, out_b, "--compiled-sha256", original.compiled_part_sha256, compiled=comp)[0] == 0
    assert stripped(out_a / "result.json") == stripped(out_b / "result.json")
    res = ShopfloorInspectionResultV1.model_validate_json((out_b / "result.json").read_bytes())
    (bg,) = [c for b in res.bends for c in b.characteristics if c.kind == "BACKGAUGE_LENGTH"]
    k_sigma = [float(n.split("=")[1]) for n in bg.evidence.notes if n.startswith("k_sigma_mm=")]
    assert bg.state == "MEASURED" and k_sigma and k_sigma[0] > 0.05


def test_budget_expiring_inside_the_articulated_walk_is_a_timeout_never_a_wrong_part(cases, monkeypatch):
    # review finding: a budget cut inside the articulated fit left flanges unfitted, which read as an unexplained
    # scan: WRONG_PART_SUSPECTED on a correct part
    import domains.production.engine.inspect.engine as engine
    from domains.production.engine.inspect import InspectOptions, inspect_scan, load_config, load_inputs

    case = cases["hat"]
    inputs = load_inputs(case / "compiled", case / "recipe.json", case / "scan.ply", _sha(case / "scan.ply"))
    cfg = load_config(None)
    state = {"n": 0, "limit": None, "art": None, "after": None}

    def left(self):
        state["n"] += 1
        return 100.0 if state["limit"] is None or state["n"] < state["limit"] else -1.0

    orig_art, orig_amb = engine._articulate, engine._ambiguity

    def articulate(*a, **k):
        if state["art"] is None:
            state["art"] = state["n"]
        return orig_art(*a, **k)

    def ambiguity(*a, **k):
        if state["after"] is None:
            state["after"] = state["n"]
        return orig_amb(*a, **k)

    monkeypatch.setattr(engine._Budget, "left", left)
    monkeypatch.setattr(engine, "_articulate", articulate)
    monkeypatch.setattr(engine, "_ambiguity", ambiguity)

    def run(limit):
        state.update(n=0, limit=limit, art=None, after=None)
        return inspect_scan(inputs, Path("unused"), cfg, InspectOptions(write_outputs=False)).result

    full = run(None)
    start, end = state["art"], state["after"]
    assert full.inspection_validity.state == "VALID" and start is not None and end is not None and end > start + 1
    for limit in sorted({start, start + 1, (start + end) // 2, end}):  # inside the walks of the pose hypotheses
        res = run(limit)
        assert res.inspection_validity.state == "TIMEOUT", (limit, res.inspection_validity)
        assert all(c.state != "MEASURED" for b in res.bends for c in b.characteristics)
    late = run(end + 1)  # the metrology: a valid result whose unmeasured characteristics say TIMEOUT
    assert late.inspection_validity.state == "VALID"
    assert any(c.reason == "TIMEOUT" for b in late.bends for c in b.characteristics)


def test_a_degenerate_in_plane_pose_withholds_only_the_position_and_the_heatmap(cases, tmp_path):
    # the datum's free edges marked unusable for registration (FreeEdgeV1.registration_usable): only its two
    # holes remain to fix the pose in its plane, which is not enough. Since A1 iteration 2 the cut ends of the
    # flanges also fix a direction the datum leaves free (registration.child_edge_rows), so theirs are marked
    # unusable too: this test keeps exercising the degenerate path (tests/test_prod_engine_inspect_a1_iter2.py
    # covers the resolution)
    from contracts.production.canonical import canonical_json_bytes

    case = cases["hat"]
    comp = tmp_path / "compiled"
    shutil.copytree(case / "compiled", comp)
    doc = json.loads((comp / "compiled_part.json").read_text())
    for panel in doc["panels"]:
        for fe in panel["free_edges"]:
            fe["registration_usable"] = False
    doc.pop("compiled_part_sha256")
    (comp / "compiled_part.json").write_bytes(canonical_json_bytes(CompiledPartV1.finalize(**doc)))
    out = tmp_path / "run"
    assert run_cli(case, out, compiled=comp)[0] == 0
    result = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    assert result.inspection_validity.state == "VALID"
    assert result.inspection_validity.registration.in_plane_degenerate
    va = result.viewer_assets
    assert va.deviations is None and va.heatmap_unavailable_reason == "REGISTRATION_UNTRUSTED"
    assert not (out / "deviations.f32").exists()
    assert any(w.startswith("IN_PLANE_POSE_DEGENERATE") for w in result.warnings)
    for b in result.bends:
        assert b.state == "FORMED"
        by_kind = {c.kind: c for c in b.characteristics}
        assert by_kind["BEND_LINE_POSITION"].state == "NOT_MEASURED"
        assert by_kind["BEND_LINE_POSITION"].reason == "REGISTRATION_UNTRUSTED"
        assert b.display.measured_polyline_mm is None
        assert by_kind["INCLUDED_ANGLE"].state == "MEASURED"  # angles and lengths do not depend on the pose
        assert any(c.state == "MEASURED" for c in b.characteristics if c.kind == "FLANGE_LENGTH")


def test_an_occluded_far_edge_is_unknown_never_a_short_fail(cases, tmp_path):
    # review finding: points removed within 4 mm of a far edge (a clamp) gave a confident 3.6 mm FAIL
    case = cases["lb_errors"]
    scan, cid = _occluded_far_edge(case, tmp_path / "occluded.ply", 4.0)
    out = tmp_path / "run"
    assert run_cli(case, out, scan=scan)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert result.inspection_validity.state == "VALID"
    c = chars(result)[cid]
    assert c.state == "NOT_MEASURED" and c.reason == "EDGE_NOT_SEEN" and c.decision == "UNKNOWN"
    assert any("no cut wall" in n for n in c.evidence.notes)
    assert result.summary.state != "ALL_MEASURED_IN_TOLERANCE"
    # the same side on the full scan: its cut wall is seen and it is measured
    out_full = tmp_path / "full"
    assert run_cli(case, out_full)[0] == 0
    full = ShopfloorInspectionResultV1.model_validate_json((out_full / "result.json").read_bytes())
    assert chars(full)[cid].state == "MEASURED"


def test_stl_scan_gives_the_decisions_of_its_ply(cases, tmp_path):
    import struct

    from domains.production.engine.synthetic.ply import read_ply

    case = cases["hat"]
    pts, _, _ = read_ply(case / "scan.ply")
    # a triangle soup over the scan's vertices (STL has no shared vertices; the reader merges exact duplicates)
    tri = np.vstack([pts, pts[: (-len(pts)) % 3]]).reshape(-1, 3, 3)
    rec = np.zeros(len(tri), dtype=[("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["v"] = tri
    stl = tmp_path / "scan.stl"
    # a binary STL whose 80-byte header starts with "solid" (common in cell exports) is still binary
    stl.write_bytes(b"solid exported by the scan cell".ljust(80, b" ") + struct.pack("<I", len(rec)) + rec.tobytes())
    out_p, out_s = tmp_path / "ply", tmp_path / "stl"
    assert run_cli(case, out_p)[0] == 0 and run_cli(case, out_s, scan=stl)[0] == 0
    a, b = (ShopfloorInspectionResultV1.model_validate_json((o / "result.json").read_bytes()) for o in (out_p, out_s))
    assert b.scan.format == "STL_BINARY" and b.inspection_validity.state == "VALID"
    assert [x.state for x in a.bends] == [x.state for x in b.bends] and a.summary.state == b.summary.state
    cb = chars(b)
    for cid, c in chars(a).items():
        assert (cb[cid].state, cb[cid].decision) == (c.state, c.decision), cid
        if c.state == "MEASURED":
            assert abs(cb[cid].actual - c.actual) < 0.05, cid


def test_points_measurements_and_pairs_with_an_unformed_bend(cases, tmp_path):
    from domains.production.engine.compile import compile_part

    case = cases["hat_stage"]
    truth = json.loads((case / "truth.json").read_text())
    states = truth["expected"]["bend_states_contract"]
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    formed = next(b for b in compiled.bends if states[b.bend_uid] == "FORMED")
    unformed = next(b for b in compiled.bends if states[b.bend_uid] == "NOT_YET_FORMED")
    datum = next(p for p in compiled.panels if p.is_datum)
    p0, p1 = (np.asarray(v, dtype=float) for v in datum.outer_boundary_mm[:2])
    a = np.mean([formed.loci.oml_virtual_sharp.start_mm, formed.loci.oml_virtual_sharp.end_mm], axis=0)
    b = np.mean([unformed.loci.oml_virtual_sharp.start_mm, unformed.loci.oml_virtual_sharp.end_mm], axis=0)
    recipe = json.loads((case / "recipe.json").read_text())

    def gm(mid, kind, s, e, uids=()):
        dist = float(np.linalg.norm(e - s))
        return {"measurement_id": mid, "label": mid, "kind": kind, "bend_uids": list(uids),
                "point_start_mm": [float(x) for x in s], "point_end_mm": [float(x) for x in e], "cad_distance_mm": dist,
                "nominal_mm": dist, "nominal_source": "CAD_DISTANCE", "tolerance": {"plus": 0.5, "minus": 0.5},
                "required": True}

    recipe["general_measurements"] = [gm("pts", "points", p0, p1),
                                      gm("pair", "bend_pair", a, b, (formed.bend_uid, unformed.bend_uid))]
    for key in ("content_sha256", "recipe_sha256"):
        recipe.pop(key)
    gm_recipe = PartRecipeV1.finalize(**recipe)
    gdir = tmp_path / "gm_case"
    gdir.mkdir()
    (gdir / "recipe.json").write_bytes(gm_recipe.model_dump_json(by_alias=True).encode())
    res = compile_part(gm_recipe, case / "part.step", gdir / "compiled", flat_path=case / "flat.dxf")
    assert res.status == "COMPILED"
    out = tmp_path / "run"
    assert run_cli(case, out, compiled=gdir / "compiled", recipe=gdir / "recipe.json")[0] == 0
    result = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    assert check_result_against_compiled(result, res.compiled, gm_recipe) == []
    rows = {r.measurement_id: r.characteristic for r in result.general_measurements}
    assert rows["pts"].state == "NOT_MEASURED" and rows["pts"].reason == "UNSUPPORTED_FEATURE"
    assert rows["pair"].state == "NOT_APPLICABLE" and rows["pair"].reason == "NOT_APPLICABLE_AT_STAGE"
    assert result.summary.state != "ALL_MEASURED_IN_TOLERANCE"


def test_a_sentinel_vertex_is_dropped_at_ingest_not_fatal(cases, tmp_path):
    # review finding: one finite 3e38 vertex overflowed the integer voxel keys and invalidated a good scan
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    case = cases["hat"]
    pts, _, _ = read_ply(case / "scan.ply")
    clean, bad = tmp_path / "clean.ply", tmp_path / "sentinel.ply"
    write_ply(clean, pts)
    write_ply(bad, np.vstack([pts, [[3.0e38, 0.0, 0.0]]]))
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_cli(case, out_a, scan=clean)[0] == 0 and run_cli(case, out_b, scan=bad)[0] == 0
    ra, rb = (ShopfloorInspectionResultV1.model_validate_json((o / "result.json").read_bytes()) for o in (out_a, out_b))
    assert rb.inspection_validity.state == "VALID" and "SCAN_GROSS_OUTLIERS_DROPPED:1" in rb.warnings
    assert rb.viewer_assets.point_count == len(pts)  # index-aligned files describe the kept points
    assert np.fromfile(out_b / "deviations.f32", dtype="<f4").size == len(pts)
    doc_a, doc_b = stripped(out_a / "result.json"), stripped(out_b / "result.json")
    for doc in (doc_a, doc_b):
        doc.pop("warnings")
        doc.pop("scan")
    assert doc_a == doc_b


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads unreadable files")
def test_an_unreadable_input_outside_the_engine_is_exit_3_not_a_traceback(cases, tmp_path):
    case = cases["hat"]
    scan = tmp_path / "locked.ply"
    shutil.copyfile(case / "scan.ply", scan)
    sha = _sha(scan)
    scan.chmod(0)
    try:
        rc, rec = run_cli(case, tmp_path / "run", scan=scan, scan_sha=sha)
    finally:
        scan.chmod(0o644)
    assert rc == 3 and rec[-1][0] == "ERROR" and rec[-1][1]["code"] == "INTERNAL_ERROR"


def test_a_sheet_thicker_than_its_cad_is_measured_and_kept_inside_u(cases, tmp_path, monkeypatch):
    # review finding: the compiled t was used for every single-skin offset, and a 10 % stock gap pushed one-skin
    # flange lengths outside U while they were decided PASS (a TOP_ONLY scan, as in the review)
    from contracts.production.canonical import canonical_json_bytes

    import domains.production.engine.inspect.registration as registration
    from domains.production.engine.inspect import InspectOptions, inspect_scan, load_config, load_inputs

    case = cases["hat_top"]
    comp = tmp_path / "compiled"
    shutil.copytree(case / "compiled", comp)
    doc = json.loads((comp / "compiled_part.json").read_text())
    t_real = doc["thickness"]["t_mm"]
    doc["thickness"]["t_mm"] = 0.9 * t_real  # the CAD says 10 % thinner than the sheet the scan sees
    for b in doc["bends"]:
        b["thickness_mm"] *= 0.9
    for p in doc["panels"]:
        if p.get("thickness_mm") is not None:
            p["thickness_mm"] *= 0.9
    doc.pop("compiled_part_sha256")
    (comp / "compiled_part.json").write_bytes(canonical_json_bytes(CompiledPartV1.finalize(**doc)))
    out = tmp_path / "run"
    assert run_cli(case, out, compiled=comp)[0] == 0
    result = ShopfloorInspectionResultV1.model_validate_json((out / "result.json").read_bytes())
    truth = json.loads((case / "truth.json").read_text())
    # the datum's two skins give the sheet's thickness, and every single-skin offset uses it
    used = [ln for ln in (out / "engine.log").read_text().splitlines() if "thickness used" in ln]
    assert used and "(measured on" in used[0]
    t_used = float(used[0].split("thickness used ")[1].split()[0])
    assert abs(t_used - t_real) < 0.1 and abs(t_used - 0.9 * t_real) > 0.1
    rows = _truth_rows(result, truth)
    assert len([r for r in rows if r[0].kind == "FLANGE_LENGTH"]) >= 6
    for c, tv, _, tin in rows:
        assert abs(c.actual - tv) <= c.uncertainty, (c.characteristic_id, c.actual - tv, c.uncertainty)
        assert not (tin is False and c.decision == "PASS")
    # mutate-and-check: with the compiled t (nothing measured) the same scan leaves rows outside U
    monkeypatch.setattr(registration, "tile_thickness", lambda *a, **k: None)
    inputs = load_inputs(comp, case / "recipe.json", case / "scan.ply", _sha(case / "scan.ply"))
    old = inspect_scan(inputs, tmp_path / "unused", load_config(None), InspectOptions(write_outputs=False)).result
    assert sum(abs(c.actual - tv) > c.uncertainty for c, tv, _, _ in _truth_rows(old, truth)) >= 1


def test_free_edges_marked_unusable_never_drive_the_registration(cases):
    from domains.production.engine.inspect.part import PartView
    from domains.production.engine.inspect.registration import DatumModel

    case = cases["hat"]
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    base = DatumModel.from_part(PartView(compiled, case / "compiled"))
    doc = compiled.model_dump(mode="json")
    datum = next(p for p in doc["panels"] if p["is_datum"])
    assert datum["free_edges"]
    for fe in datum["free_edges"]:
        fe["registration_usable"] = False
    doc.pop("compiled_part_sha256")
    unusable = CompiledPartV1.finalize(**doc)
    model = DatumModel.from_part(PartView(unusable, case / "compiled"))
    assert base.free.any() and not model.free.any()


def test_ambiguous_pose_keeps_the_nominal_lines_on_the_part_and_labels_only_the_datum(cases, tmp_path):
    case = cases["ucs"]
    out = tmp_path / "run"
    assert run_cli(case, out)[0] == 0
    result, compiled, recipe, truth = load(case, out)
    assert check_result_against_compiled(result, compiled, recipe) == []
    assert result.inspection_validity.registration.pose_ambiguous
    # amendment ENGINE_IO-A2 (2026-10-19): tied twins publish only the readings identical under both
    # (tests/test_prod_engine_fixes3_2026_10_19.py); bend identities, and so the measured lines, stay withheld
    assert result.inspection_validity.registration.identity_ambiguous
    assert all(b.state in ("UNKNOWN", "FORMED") and b.display.measured_polyline_mm is None for b in result.bends)
    final = next(s for s in compiled.stages if s.stage_model_id == f"S{len(compiled.bends)}")
    for b in result.bends:  # the decoded (complete) stage's lines, not the flat blank's
        ref = np.array([final.bend_lines[b.bend_uid].start_mm, final.bend_lines[b.bend_uid].end_mm])
        assert np.abs(np.asarray(b.display.nominal_polyline_mm) - ref).max() < 1e-3
    labels = np.fromfile(out / result.viewer_assets.segment_labels, dtype="<u2")
    datum = next(i for i, p in enumerate(compiled.panels) if p.is_datum)
    assert set(np.unique(labels).tolist()) <= {0, datum + 1} and (labels == datum + 1).any()
