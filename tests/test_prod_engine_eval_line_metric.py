"""E4a G2 line-metric adapter: parity with scripts/evaluate_f1_against_manual_gold.py,
raw-frame lines with the declared OUTER_APEX locus, exclusion before scoring.

Uses the C01 contract examples (synthetic SYN-HAT-1); no real scan or Gold label is read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from contracts.production.compiled_part import CompiledPartV1
from contracts.production.inspection_result import ShopfloorInspectionResultV1
from domains.production.engine.eval.engine_output import (
    decoded_stage,
    measured_outer_apex,
    raw_to_frame_points,
    row_characteristic,
    stage_transform,
)
from domains.production.engine.eval.line_metric import (
    DEFAULT_THRESHOLDS,
    GoldCase,
    engine_lines,
    match_lines,
    pair_metrics,
    prepare_gold,
    score_line_metric,
)

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "contracts" / "production" / "examples"


def _script():
    path = REPO / "scripts" / "evaluate_f1_against_manual_gold.py"
    spec = importlib.util.spec_from_file_location("e4a_gold_script_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = module  # dataclasses resolve the defining module through sys.modules
    try:
        spec.loader.exec_module(module)  # defines functions and constants only; reads no data
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _lines(rng: np.random.Generator, n: int, jitter: float) -> list[dict]:
    out = []
    for i in range(n):
        a = rng.uniform(-100, 100, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        b = a + d * rng.uniform(20, 120)
        out.append({"bend_id": f"g{i}", "line_start": (a + rng.normal(0, jitter, 3)).tolist(),
                    "line_end": (b + rng.normal(0, jitter, 3)).tolist()})
    return out


def test_matcher_is_identical_to_the_gold_script():
    script = _script()
    assert (script.Thresholds().angle_deg, script.Thresholds().transverse_offset_mm,
            script.Thresholds().gold_span_overlap, script.Thresholds().prediction_span_overlap) == (
        DEFAULT_THRESHOLDS.angle_deg, DEFAULT_THRESHOLDS.transverse_offset_mm, DEFAULT_THRESHOLDS.gold_span_overlap,
        DEFAULT_THRESHOLDS.prediction_span_overlap)
    rng = np.random.default_rng(3)
    for trial in range(25):
        gold = _lines(rng, int(rng.integers(0, 7)), 0.0)
        preds = [{**g, "bend_id": f"p{k}", "line_start": (np.asarray(g["line_start"]) + rng.normal(0, 2.0, 3)).tolist(),
                  "line_end": (np.asarray(g["line_end"]) + rng.normal(0, 2.0, 3)).tolist()}
                 for k, g in enumerate(gold) if rng.random() < 0.8]
        preds += _lines(rng, int(rng.integers(0, 3)), 0.0)
        ours = match_lines(gold, preds)
        theirs = script.match_lines(gold, preds, script.Thresholds())
        assert ours == theirs, trial
    a = np.asarray([[0, 0, 0], [100, 0, 0]], dtype=float)
    b = np.asarray([[10, 3, 0], [90, 3.5, 0]], dtype=float)
    assert pair_metrics(a, b) == script.pair_metrics(a, b)


def _examples():
    compiled = CompiledPartV1.model_validate(json.loads((EX / "compiled_part_v1.example.json").read_text()))
    final = ShopfloorInspectionResultV1.model_validate(json.loads((EX / "shopfloor_result_v1.example.json").read_text()))
    staged = ShopfloorInspectionResultV1.model_validate(
        json.loads((EX / "shopfloor_result_v1.in_process.example.json").read_text()))
    invalid = ShopfloorInspectionResultV1.model_validate(
        json.loads((EX / "shopfloor_result_v1.invalid.example.json").read_text()))
    return compiled, final, staged, invalid


def _apex_geometry(compiled, stage, row, apex_line):
    """(|offset|, angle(offset, parent flange) in deg, axial component) of an apex line vs the OML."""
    oml = np.asarray(row.display.measured_polyline_mm, dtype=float)[[0, -1]]
    axis = (oml[1] - oml[0]) / np.linalg.norm(oml[1] - oml[0])
    off = np.asarray(apex_line).mean(axis=0) - oml.mean(axis=0)
    bend = next(b for b in compiled.bends if b.bend_uid == row.bend_uid)
    side = next(s for s in bend.sides if s.panel_id == bend.parent_panel_id)
    d_p = stage_transform(stage, bend.parent_panel_id)[:3, :3] @ np.asarray(side.outward_dir)
    ang = np.degrees(np.arccos(np.clip(off @ d_p / (np.linalg.norm(off) * np.linalg.norm(d_p)), -1, 1)))
    return float(np.linalg.norm(off)), float(ang), float(off @ axis), bend


def test_engine_lines_use_the_measured_apex_locus_in_the_raw_frame():
    compiled, final, staged, invalid = _examples()
    lines = engine_lines(final, compiled)
    measured = [r for r in final.bends if row_characteristic(r, "BEND_LINE_POSITION").state == "MEASURED"]
    assert [x["bend_id"] for x in lines] == [r.bend_uid for r in measured]
    stage = decoded_stage(final, compiled)
    for rec, row in zip(lines, measured):
        raw = np.asarray([rec["line_start"], rec["line_end"]])
        frame = raw_to_frame_points(final, raw)  # back into cad_model_mm
        theta = row_characteristic(row, "INCLUDED_ANGLE").actual
        mag, ang, axial, bend = _apex_geometry(compiled, stage, row, frame)
        ro = bend.inner_radius_mm + bend.thickness_mm
        # E1's loci rule at the MEASURED angle: |apex - OML| = (Ri + t)(1/sin(theta/2) - 1), bisector at theta/2
        assert mag == pytest.approx(ro * (1 / np.sin(np.radians(theta) / 2) - 1), abs=1e-6)
        assert ang == pytest.approx(theta / 2, abs=1e-4) and abs(axial) < 1e-6
        assert rec["locus"] == "OUTER_APEX" and rec["frame"] == "raw" and rec["apex_source"] == "MEASURED_ANGLE"
    oml = engine_lines(final, compiled, locus="OML_VIRTUAL_SHARP", frame="cad")
    assert np.allclose(oml[0]["line_start"], measured[0].display.measured_polyline_mm[0])
    # NOT_YET_FORMED bends emit no line; an invalid inspection emits nothing
    staged_ids = {x["bend_id"] for x in engine_lines(staged, compiled)}
    assert staged_ids == {r.bend_uid for r in staged.bends if r.state != "NOT_YET_FORMED"
                          and row_characteristic(r, "BEND_LINE_POSITION").state == "MEASURED"}
    assert engine_lines(invalid, compiled) == []
    with pytest.raises(ValueError):
        engine_lines(final, compiled, locus="CYLINDER_AXIS")


def _with_angle(row, theta):
    """A copy of a bend row whose INCLUDED_ANGLE reads ``theta`` (no re-validation: geometry probe only)."""
    chars = tuple(c.model_copy(update={"actual": theta}) if c.kind == "INCLUDED_ANGLE" else c for c in row.characteristics)
    return row.model_copy(update={"characteristics": chars})


def test_measured_apex_follows_a_partially_formed_angle():
    """On a partially formed bend (150 deg instead of 90) the nominal offset would put the
    'apex' 0.38 (Ri + t) off the material; the measured-angle apex does not."""
    compiled, final, _, _ = _examples()
    stage = decoded_stage(final, compiled)
    row = final.bends[0]
    nominal_like, _ = measured_outer_apex(compiled, stage, _with_angle(row, 90.0))
    partial_row = _with_angle(row, 150.0)
    partial, source = measured_outer_apex(compiled, stage, partial_row)
    mag90, ang90, _, bend = _apex_geometry(compiled, stage, row, nominal_like)
    mag150, ang150, _, _ = _apex_geometry(compiled, stage, partial_row, partial)
    ro = bend.inner_radius_mm + bend.thickness_mm
    assert source == "MEASURED_ANGLE"
    assert mag90 == pytest.approx(ro * (np.sqrt(2) - 1), abs=1e-6) and ang90 == pytest.approx(45.0, abs=1e-4)
    assert mag150 == pytest.approx(ro * (1 / np.sin(np.radians(75)) - 1), abs=1e-6) and ang150 == pytest.approx(75.0)
    assert np.linalg.norm(partial.mean(axis=0) - nominal_like.mean(axis=0)) > 0.3 * ro
    # no measured angle: renders fall back to the OML itself, the metric to the nominal offset (both flagged)
    no_angle = row.model_copy(update={
        "characteristics": tuple(c.model_copy(update={"state": "NOT_MEASURED", "actual": None})
                                 if c.kind == "INCLUDED_ANGLE" else c for c in row.characteristics),
        "state_evidence": row.state_evidence.model_copy(update={"measured_included_deg": None})})
    line, src = measured_outer_apex(compiled, stage, no_angle, fallback="OML")
    assert src == "OML_FALLBACK_NO_ANGLE" and np.allclose(line, np.asarray(row.display.measured_polyline_mm)[[0, -1]])
    assert measured_outer_apex(compiled, stage, no_angle, fallback="NOMINAL")[1] == "NOMINAL_ANGLE_FALLBACK"
    assert measured_outer_apex(compiled, stage, no_angle, fallback="NONE") == (None, "NO_MEASURED_ANGLE")


def test_gold_frames_are_recovered_or_excluded_before_scoring():
    rot = np.eye(4)
    c, s = np.cos(0.3), np.sin(0.3)
    rot[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    rot[:3, 3] = [5.0, -2.0, 1.0]
    raw_line = {"bend_id": "b1", "line_start": [0, 0, 0], "line_end": [50, 0, 0]}
    ref = np.asarray([raw_line["line_start"], raw_line["line_end"]], dtype=float) @ rot[:3, :3].T + rot[:3, 3]
    cases = [
        GoldCase("raw", (raw_line,), "raw_scan"),
        GoldCase("ref_ok", ({"bend_id": "b1", "line_start": ref[0].tolist(), "line_end": ref[1].tolist()},),
                 "reference", raw_to_reference=rot, residual_mm=0.004),
        GoldCase("ref_none", (raw_line,), "reference"),
        GoldCase("ref_bad", (raw_line,), "reference", raw_to_reference=rot, residual_mm=0.05),
    ]
    usable, excluded = prepare_gold(cases)
    assert set(usable) == {"raw", "ref_ok"}
    assert np.allclose(usable["ref_ok"][0]["line_start"], [0, 0, 0], atol=1e-9)
    assert {e["case_id"]: e["reason"] for e in excluded} == {
        "ref_none": "NO_RECOVERABLE_TRANSFORM", "ref_bad": "TRANSFORM_RESIDUAL_TOO_LARGE"}


def test_scoring_counts_missing_predictions_and_compares_with_legacy():
    gold = [GoldCase("a", ({"bend_id": "1", "line_start": [0, 0, 0], "line_end": [60, 0, 0]},
                           {"bend_id": "2", "line_start": [0, 50, 0], "line_end": [60, 50, 0]}), "raw_scan"),
            GoldCase("b", ({"bend_id": "1", "line_start": [0, 0, 0], "line_end": [60, 0, 0]},
                           {"bend_id": "2", "line_start": [0, 30, 0], "line_end": [60, 30, 0]}), "raw_scan")]
    preds = {"a": [{"bend_id": "x", "line_start": [0, 1, 0], "line_end": [60, 1, 0]},
                   {"bend_id": "y", "line_start": [0, 51, 0], "line_end": [60, 51, 0]}]}  # case b: engine silent
    report = score_line_metric(gold, preds)
    eng = report["engine"]
    assert (eng["true_positive"], eng["false_positive"], eng["false_negative"]) == (2, 0, 2)
    assert eng["precision"] == 1.0 and eng["recall"] == pytest.approx(0.5) and eng["exact_part_count"] == 1
    assert eng["f1"] == pytest.approx(2 / 3)
    assert report["gates"]["f1"] == "FAIL" and report["label"] == "recipe-guided vs blind union, not like-for-like"
    assert report["gates"]["exact_part_rate_beats_legacy"] == "NOT_EVALUABLE"
    perfect = dict(preds, b=[{"bend_id": "z", "line_start": [1, 0.5, 0], "line_end": [59, 0.5, 0]},
                             {"bend_id": "w", "line_start": [1, 30.5, 0], "line_end": [59, 30.5, 0]}])
    report = score_line_metric(gold, perfect, legacy_predictions={"a": preds["a"]})
    assert report["engine"]["f1"] == 1.0 and report["gates"]["exact_part_rate_beats_legacy"] == "PASS"
    assert report["verdict"] == "PASS"
    report = score_line_metric(gold, perfect, legacy_predictions=perfect)
    assert report["gates"]["exact_part_rate_beats_legacy"] == "FAIL"  # equal is not better


def _write_gold_case(root: Path, job: str, gold: list[dict], preds: list[dict]) -> Path:
    (root / "output" / job).mkdir(parents=True, exist_ok=True)
    (root / "output" / job / "bend_instance_labels.json").write_text(json.dumps({"manual_bends": gold}))
    pred = root / "preds" / f"{job}.json"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.write_text(json.dumps({"owned_bend_regions": preds}))
    return pred


def test_aggregate_matches_the_script_evaluate_including_a_silent_engine(tmp_path):
    """Totals, precision, recall, F1 and the exact-part rate equal the script's evaluate()
    on synthetic Gold files (the script's ROOT and case table are redirected to tmp_path;
    no real Gold label is read). A silent engine scores F1 = 0 and FAILs."""
    script = _script()
    rng = np.random.default_rng(11)
    cases, preds_by_case, jobs = [], {}, {}
    for i in range(5):
        gold = _lines(rng, int(rng.integers(1, 6)), 0.0)
        preds = [{**g, "bend_id": f"p{k}", "line_start": (np.asarray(g["line_start"]) + rng.normal(0, 1.5, 3)).tolist(),
                  "line_end": (np.asarray(g["line_end"]) + rng.normal(0, 1.5, 3)).tolist()}
                 for k, g in enumerate(gold) if rng.random() < 0.7]
        preds += _lines(rng, int(rng.integers(0, 2)), 0.0)
        if i == 2:
            preds = []  # the engine emitted nothing on this scan
        cid = f"case_{i}"
        jobs[cid] = _write_gold_case(tmp_path, f"job_{i}", gold, preds)
        cases.append(GoldCase(cid, tuple(gold), "raw_scan"))
        preds_by_case[cid] = preds
    script.ROOT = tmp_path
    script.CASE_JOBS = {cid: f"job_{i}" for i, cid in enumerate(jobs)}
    theirs = script.evaluate(jobs, script.Thresholds())["metrics"]
    ours = score_line_metric(cases, preds_by_case)["engine"]
    for key in ("true_positive", "false_positive", "false_negative", "exact_part_count", "case_count",
                "gold_count", "prediction_count"):
        assert ours[key] == theirs[{"gold_count": "gold_bend_count"}.get(key, key)], key
    for key in ("precision", "recall", "f1", "exact_part_rate"):
        assert round(ours[key], 6) == theirs[key], key
    # a completely silent engine on usable Gold: the script says 0.0, so do we, and the gate FAILs
    silent_jobs = {cid: _write_gold_case(tmp_path / "silent", f"job_{i}", list(c.gold_lines), [])
                   for i, (cid, c) in enumerate(zip(jobs, cases))}
    script.ROOT = tmp_path / "silent"
    theirs = script.evaluate(silent_jobs, script.Thresholds())["metrics"]
    report = score_line_metric(cases, {})
    assert (theirs["precision"], theirs["recall"], theirs["f1"]) == (0.0, 0.0, 0.0)
    assert (report["engine"]["precision"], report["engine"]["recall"], report["engine"]["f1"]) == (0.0, 0.0, 0.0)
    assert report["gates"]["f1"] == "FAIL" and report["gates"]["precision"] == "FAIL" and report["verdict"] == "FAIL"
    # no usable Gold at all is the only NOT_EVALUABLE case
    empty = score_line_metric([GoldCase("x", (), "reference")], {})
    assert empty["engine"]["f1"] is None and empty["verdict"] == "NOT_EVALUABLE"
