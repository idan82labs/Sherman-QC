"""E4b legacy control arm: run plumbing, report parsing, bend mapping and the LEGACY render scene.

The real worker is never run here (it is heavy and reads the legacy runtime);
a stub worker script checks the command line, the opaque read-only inputs and
the environment, and writes canned reports. Report parsing and the render scene
use a real E2 part compiled by the real E1.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from domains.production.engine.compile import compile_part
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.freeze import common, legacy
from domains.production.engine.eval.g4 import closeup_class
from domains.production.engine.eval.plyio import write_points_ply
from domains.production.engine.eval.render import plan_closeups, render_closeup, render_global_views
from domains.production.engine.synthetic.cases import generate_tier

CASE = "E2-DEV-SMTD-000"


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4b_legacy")
    generate_tier("smoke", "DEV", root / "tier", case_ids=[CASE])
    cdir = root / "compiled"
    recipe = load_recipe(root / "tier" / CASE / "recipe.json")
    assert compile_part(recipe, root / "tier" / CASE / "part.step", cdir,
                        flat_path=root / "tier" / CASE / "flat.dxf").status == "COMPILED"
    truth = json.loads((root / "tier" / CASE / "truth.json").read_text())
    return root, load_compiled(cdir / "compiled_part.json"), cdir, truth


def _rigid() -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = [12.0, -40.0, 7.0]
    return t


def _report(compiled, t_ref: np.ndarray) -> dict:
    """A legacy-shaped report: one scan-fit match per bend except a placeholder and a miss, plus an extra."""
    final = max(compiled.stages, key=lambda s: s.stage_index)
    matches = []
    for i, b in enumerate(compiled.bends):
        cyl = b.loci.cylinder_axis
        cad = [list(cyl.start_mm), list(cyl.end_mm)]
        oml = final.bend_lines[b.bend_uid]
        det = [list(np.asarray(oml.start_mm) + [0.1, 0, 0]), list(np.asarray(oml.end_mm) + [0.1, 0, 0])]
        folded = max(float(b.nominal_included_deg), 180.0 - float(b.nominal_included_deg))
        m = {"bend_id": f"CAD_B{i + 1}", "status": "PASS", "target_angle": folded,
             "measured_angle": folded + 0.7, "cad_line_start": cad[0], "cad_line_end": cad[1],
             "detected_line_start": det[0], "detected_line_end": det[1], "assignment_source": "GLOBAL_DETECTION",
             "detected_line_geometry_source": "SCAN_FIT", "observability_state": "FORMED",
             "measured_edge_distance_mm": 50.0 + i, "expected_edge_distance_mm": 50.0}
        if i == 1:
            m.update(assignment_source="CAD_LOCAL_NEIGHBORHOOD", detected_line_geometry_source="CAD_REFERENCE_PLACEHOLDER",
                     detected_line_start=cad[0], detected_line_end=cad[1], measured_edge_distance_mm=None)
        if i == 2:
            m.update(status="NOT_DETECTED", measured_angle=None, detected_line_start=None, detected_line_end=None)
        matches.append(m)
    matches.append({"bend_id": "CAD_B99", "status": "PASS", "target_angle": 90.0, "measured_angle": 91.0,
                    "cad_line_start": [5000.0, 0, 0], "cad_line_end": [5100.0, 0, 0],
                    "detected_line_start": [5000.0, 1, 0], "detected_line_end": [5100.0, 1, 0],
                    "assignment_source": "GLOBAL_DETECTION", "detected_line_geometry_source": "SCAN_FIT"})
    return {"alignment": {"raw_scan_to_reference_transform": t_ref.tolist(), "fitness": 0.9, "rmse_mm": 0.8,
                          "scale": 1.0, "geometry_frame": "reference_mesh_raw"}, "matches": matches}


def _run_record(tmp: Path, report: dict, truth: dict, part_root: Path, t_ref: np.ndarray) -> dict:
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bend_inspection_report.json").write_text(json.dumps(report), encoding="utf-8")
    from domains.production.engine.eval.plyio import read_ply

    raw = read_ply(part_root / "tier" / CASE / "scan.ply", with_faces=False).vertices
    cad_to_raw = np.asarray(truth["scan"]["cad_to_raw"], dtype=float)
    aligned = raw @ np.linalg.inv(cad_to_raw)[:3, :3].T + np.linalg.inv(cad_to_raw)[:3, 3]
    write_points_ply(out / "aligned_scan.ply", aligned)
    np.save(out / "deviations.npy", np.zeros(len(aligned), dtype=np.float32))
    return {"case_ref": "pc_test00000000", "status": "OK",
            "outputs": {"bend_inspection_report.json": str(out / "bend_inspection_report.json"),
                        "aligned_scan.ply": str(out / "aligned_scan.ply"),
                        "deviations.npy": str(out / "deviations.npy"), "reference_mesh.ply": None}}


def test_report_parsing_maps_bends_and_drops_placeholder_lines(part, tmp_path):
    root, compiled, cdir, truth = part
    assert len(compiled.bends) >= 3
    t_ref = _rigid()
    run = _run_record(tmp_path, _report(compiled, t_ref), truth, root, t_ref)
    parsed = legacy.parse_legacy_report(run, compiled, cdir)
    rows = {r["sequence_index"]: r for r in parsed["bends"]}
    assert all(r["legacy_bend_id"] == f"CAD_B{s}" for s, r in rows.items())  # mapped through the cylinder-axis locus
    assert rows[1]["line"]["measured"] and rows[1]["angle"]["measured"]
    assert rows[1]["angle"]["legacy_delta_deg"] == pytest.approx(0.7)
    assert rows[1]["angle"]["legacy_delta_vs_e1_deg"] == pytest.approx(0.7)
    assert rows[1]["angle"]["convention"] == "FOLDED_INCLUDED"
    assert rows[2]["line"]["placeholder"] and not rows[2]["line"]["measured"] and rows[2]["angle"]["measured"]
    assert not rows[3]["angle"]["measured"] and not rows[3]["line"]["measured"]
    assert parsed["counts"]["unmapped_legacy_bends"] == 1
    raw_lines = legacy.legacy_line_predictions(parsed)
    assert {r["bend_id"] for r in raw_lines} == {"legacy_CAD_B99"} | {f"legacy_CAD_B{s}" for s in rows if s not in (2, 3)}
    first = next(r for r in raw_lines if r["bend_id"] == "legacy_CAD_B1")
    det = np.asarray(_report(compiled, t_ref)["matches"][0]["detected_line_start"])
    inv = np.linalg.inv(t_ref)
    assert np.allclose(first["line_start"], inv[:3, :3] @ det + inv[:3, 3])


def test_failed_runs_measure_nothing(part):
    _, compiled, cdir, _ = part
    parsed = legacy.parse_legacy_report({"case_ref": "pc_x", "status": "WORKER_ERROR", "outputs": {}}, compiled, cdir)
    assert parsed["bends"] == [] and parsed["line_predictions_raw"] == [] and parsed["counts"]["angle_measured"] == 0


def test_legacy_scene_renders_like_the_e3_arm(part, tmp_path):
    root, compiled, cdir, truth = part
    t_ref = _rigid()
    run = _run_record(tmp_path, _report(compiled, t_ref), truth, root, t_ref)
    parsed = legacy.parse_legacy_report(run, compiled, cdir)
    scene = legacy.legacy_render_scene(parsed, run, compiled, cdir, title="legacy test")
    assert scene.validity_state == "VALID" and scene.meta["arm"] == "LEGACY"
    classes = {b.sequence_index: closeup_class(scene, b) for b in scene.bends}
    assert classes[1] == "MEASURED" and classes[2] == "UNKNOWN" and classes[3] == "UNKNOWN"
    assert scene.bend(compiled.bends[1].bend_uid).position_reason == "LEGACY_PLACEHOLDER_LINE"
    recs = render_global_views(scene, tmp_path / "r", show_values=False)
    assert {r["kind"] for r in recs} >= {"GLOBAL_ISO", "GLOBAL_TOP", "GLOBAL_HEATMAP"}
    plan = plan_closeups(scene)[0]
    rec = render_closeup(scene, plan, tmp_path / "r" / "c.png")
    assert rec["measured_line_shown"] and rec["apex_source"] == "LEGACY_DETECTED_LINE"
    failed = legacy.legacy_render_scene({"case_ref": "pc_x", "bends": [], "alignment": None},
                                        {"status": "WORKER_ERROR", "outputs": {}}, compiled, cdir)
    assert failed.validity_state == "INVALID_INSPECTION" and failed.points is None


STUB = textwrap.dedent('''
    import json, os, sys, time
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    mode = os.environ.get("E4B_STUB_MODE", "ok")
    out = args["--output-dir"]
    with open(os.path.join(out, "seen.json"), "w") as fh:
        json.dump({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in (
            "BEND_INSPECTION_CONFIG", "SHERMAN_QC_F1_PROGRAM_STATUS_PATH", "OMP_NUM_THREADS",
            "PYTHONDONTWRITEBYTECODE", "BEND_FEATURE_POLICIES")},
            "scan_writable": os.access(args["--scan-path"], os.W_OK)}, fh)
    if mode == "error":
        print("ERROR\\t" + json.dumps({"error": "Alignment unavailable: insufficient shared registration fitness"}))
        sys.exit(1)
    if mode == "sleep":
        time.sleep(30)
    json.dump({"matches": [], "alignment": {}}, open(os.path.join(out, "bend_inspection_report.json"), "w"))
    print("RESULT\\t" + json.dumps({"processing_time_ms": 12.0}))
''')


def test_run_plumbing_opaque_inputs_env_and_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "custody_check", lambda p, purpose: Path(p).resolve())
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB, encoding="utf-8")
    scan = tmp_path / "src" / "12345678_A_FULL_SCAN 1.ply"
    cad = tmp_path / "src" / "12345678_A.stp"
    scan.parent.mkdir()
    scan.write_bytes(b"ply-bytes")
    cad.write_bytes(b"step-bytes")
    case = {"case_ref": "pc_abcdef012345", "scan_sha256": common.sha256_file(scan), "cad_sha256": common.sha256_file(cad)}
    rec = legacy.run_legacy_case(case, scan, cad, tmp_path / "work", worker=stub)
    assert rec["status"] == "OK" and rec["inputs_intact"] and rec["processing_time_ms"] == 12.0
    seen = json.loads((tmp_path / "work/legacy/pc_abcdef012345/out/seen.json").read_text())
    argv = " ".join(seen["argv"])
    assert "12345678" not in argv and "FULL_SCAN" not in argv  # opaque names and ids only
    assert "--tolerance-angle 1" in argv and "--tolerance-radius 0.5" in argv
    assert not seen["scan_writable"]
    for key in ("BEND_INSPECTION_CONFIG", "SHERMAN_QC_F1_PROGRAM_STATUS_PATH", "BEND_FEATURE_POLICIES"):
        assert "/ABSENT/" in seen["env"][key] and not Path(seen["env"][key]).exists()
    assert seen["env"]["OMP_NUM_THREADS"] == "1" and seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    monkeypatch.setenv("E4B_STUB_MODE", "error")
    err = legacy.run_legacy_case(case, scan, cad, tmp_path / "work", worker=stub)
    assert err["status"] == "WORKER_ERROR" and "Alignment unavailable" in err["worker_error"]
    monkeypatch.setenv("E4B_STUB_MODE", "sleep")
    slow = legacy.run_legacy_case(case, scan, cad, tmp_path / "work", worker=stub, timeout_s=1.0)
    assert slow["status"] == "TIMEOUT" and slow["wall_s"] < 15
    assert os.stat(tmp_path / "work/legacy/pc_abcdef012345/inputs").st_mode & stat.S_IFDIR
    assert sys.executable  # the worker runs on the same interpreter as the harness


def test_render_job_rebuilds_the_legacy_scene_from_the_run_log(part, tmp_path):
    root, compiled, cdir, truth = part
    t_ref = _rigid()
    run = _run_record(tmp_path, _report(compiled, t_ref), truth, root, t_ref)
    case_dir = tmp_path / "pc_test00000000"
    case_dir.mkdir()
    (case_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (case_dir / "legacy_case.json").write_text(json.dumps(legacy.parse_legacy_report(run, compiled, cdir)),
                                               encoding="utf-8")
    job = legacy.legacy_render_job({"case_ref": "pc_test00000000", "stage_class": "STAGED_PARTIAL"}, case_dir, cdir)
    assert job.source == "LEGACY" and job.population == "PHYSICAL_DEV" and job.incomplete is True
    assert job.scene is not None and job.scene.validity_state == "VALID" and job.engine_dir is None


def test_an_earlier_run_is_reused_only_on_identical_inputs_and_conditions(tmp_path):
    case = {"case_ref": "pc_" + "a" * 12, "scan_sha256": "a" * 64, "cad_sha256": "c" * 64}
    out = tmp_path / "out"
    out.mkdir()
    (out / "bend_inspection_report.json").write_text("{}", encoding="utf-8")
    prev = {"case_ref": case["case_ref"], "status": "OK", "inputs_intact": True, "legacy_code_fingerprint": "f" * 64,
            "report_sha256": common.sha256_file(out / "bend_inspection_report.json"), "peak_rss_bytes": 2 * 1024**3,
            "rss_exceeded": False,
            "conditions": {"worker": legacy.LEGACY_WORKER_REL, "tolerance_angle_deg": 1.0, "tolerance_radius_mm": 0.5,
                           "timeout_s": 600.0, "rss_limit_bytes": 4 * 1024**3,
                           "runtime_config": "built-in defaults (config absent)", "blind": True, "opaque_inputs": True}}
    prev_case = {"scan_sha256": "a" * 64, "cad_sha256": "c" * 64}
    ok = legacy.reusable_run(prev, case, "f" * 64, prev_case=prev_case, work_case=tmp_path)
    assert ok["reusable"] and ok["why"] == "identical inputs and conditions"
    assert ok["checks"]["rss_cap_never_bound"]  # a 4 GiB run that peaked at 2 GiB is the run a 12 GiB cap gives
    for change, key in ((dict(case, cad_sha256="d" * 64), "cad_sha256"),):
        assert key in legacy.reusable_run(prev, change, "f" * 64, prev_case=prev_case, work_case=tmp_path)["why"]
    assert "legacy_code_fingerprint" in legacy.reusable_run(prev, case, "0" * 64, prev_case=prev_case,
                                                            work_case=tmp_path)["why"]
    slow = {**prev, "conditions": {**prev["conditions"], "timeout_s": 300.0}}
    assert "conditions" in legacy.reusable_run(slow, case, "f" * 64, prev_case=prev_case, work_case=tmp_path)["why"]
    for killed in ({**prev, "status": "RSS_LIMIT", "rss_exceeded": True, "peak_rss_bytes": 4 * 1024**3},
                   {**prev, "peak_rss_bytes": legacy.LEGACY_RSS_LIMIT_BYTES}, {**prev, "peak_rss_bytes": None}):
        again = legacy.reusable_run(killed, case, "f" * 64, prev_case=prev_case, work_case=tmp_path)
        assert not again["reusable"] and "rss_cap_never_bound" in again["why"]  # re-run at today's cap
    assert legacy.LEGACY_RSS_LIMIT_BYTES == 12 * 1024**3
    gone = legacy.reusable_run(prev, case, "f" * 64, prev_case=prev_case, work_case=tmp_path / "elsewhere")
    assert "report_present" in gone["why"]
    failed = {**prev, "status": "WORKER_ERROR"}  # a failed run is a valid record of a failure
    assert legacy.reusable_run(failed, case, "f" * 64, prev_case=prev_case, work_case=tmp_path / "x")["reusable"]
    assert legacy.reusable_run(None, case, "f" * 64)["why"] == "NO_EARLIER_RUN"


def test_the_unary_count_model_condition_is_read_back_from_every_report(part, tmp_path):
    root, compiled, cdir, truth = part
    assert not legacy.unary_count_model_applied({"matches": [{"bend_id": "B1"}]})
    assert legacy.unary_count_model_applied({"structured_context": {"unary": 1}, "matches": []})
    assert legacy.unary_count_model_applied({"matches": [{"bend_id": "B1", "unary_predictions": {"p": 0.9}}]})
    t_ref = _rigid()
    report = _report(compiled, t_ref)
    run = _run_record(tmp_path, report, truth, root, t_ref)
    assert legacy.parse_legacy_report(run, compiled, cdir)["unary_count_model_applied"] is False
    report["structured_context"] = {"unary_model": "latest"}
    run = _run_record(tmp_path / "u", report, truth, root, t_ref)
    assert legacy.parse_legacy_report(run, compiled, cdir)["unary_count_model_applied"] is True
    failed = legacy.parse_legacy_report({"case_ref": "pc_x", "status": "RSS_LIMIT", "outputs": {}}, compiled, cdir)
    assert failed["unary_count_model_applied"] is None  # no report: not determinable
