"""E4a scorer and truth loader: every G1 gate must PASS on exact truth and FAIL on a
deliberately wrong "engine" (mutate-and-check).

The "engine" here is the E4 truth oracle (domains.production.engine.eval.oracle),
which reads E2 truth and writes contract-valid results; it is never an engine.
Cases are real E2 cases (regression/full DEV specs) compiled by the real E1.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from domains.production.engine.compile import compile_part
from domains.production.engine.eval import stats
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.oracle import OracleSpec, oracle_result
from domains.production.engine.eval.scorer import (
    CaseRow,
    CaseScore,
    EngineRun,
    build_scorecard,
    g1_verdict,
    run_condition_reasons,
    scanner_model_gaps,
    score_case,
)
from domains.production.engine.eval.thresholds import G1_THRESHOLDS, thresholds_sha256
from domains.production.engine.eval.truth import TruthError, load_manifest, load_truth_case, scorable
from domains.production.engine.synthetic.cases import generate_tier

CASES = (
    "E2-DEV-UC-000",  # complete, exact nominal
    "E2-DEV-UC-001",  # complete with injected errors (angle +2.7 deg, flange +1.49 mm)
    "E2-DEV-LB-001",  # complete with errors (flange -0.92 mm, angle -3 deg)
    "E2-DEV-HT-005",  # stage k: backgauge (+1.42 mm), must-be-unknown flanges, NOT_YET_FORMED bends
    "E2-DEV-HT-006",  # mirrored part
    "E2-DEV-LB-006",  # missing bend in a FINAL scan
    "E2-DEV-UCS-000",  # symmetric U channel
    "E2-DEV-LB-009",  # TOP_ONLY plan (more occlusion)
)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4a_scorer")
    tier = root / "tier"
    generate_tier("full", "DEV", tier, case_ids=CASES)
    out = {}
    for cid in CASES:
        tc = load_truth_case(tier / cid)
        recipe = load_recipe(tier / cid / "recipe.json")
        cdir = root / "compiled" / cid
        res = compile_part(recipe, tier / cid / "part.step", cdir, flat_path=tier / cid / "flat.dxf")
        assert res.status == "COMPILED", (cid, res.compiled.blocker_codes)
        out[cid] = (tc, recipe, load_compiled(cdir / "compiled_part.json"))
    return {"tier": tier, "cases": out, "manifest": load_manifest(tier)}


def _score(suite, specs=None, runs=None, ids=CASES):
    scores = []
    for cid in ids:
        tc, recipe, compiled = suite["cases"][cid]
        run = (runs or {}).get(cid, EngineRun("OK"))
        spec = (specs or {}).get(cid, OracleSpec())
        result = oracle_result(recipe=recipe, compiled=compiled, truth=tc, scan_sha256=tc.files["scan.ply"],
                               spec=spec)
        scores.append(score_case(tc, run, result, compiled))
    return scores


def _card(suite, scores, thresholds=G1_THRESHOLDS, tier="regression"):
    return build_scorecard(scores, tier=tier, manifest=suite["manifest"], thresholds=thresholds, bootstrap=False)


def _gate(card, gid):
    return next(g for g in card["gates"] if g["id"] == gid)


# ---------------------------------------------------------------------------
# truth loader
# ---------------------------------------------------------------------------


def test_truth_loader_types_scoring_rule_and_tamper_check(suite, tmp_path):
    tc, _, _ = suite["cases"]["E2-DEV-HT-005"]
    assert tc.scenario == "STAGE_K" and tc.inspection_stage in ("FINAL", "IN_PROCESS")
    assert tc.formed_count == 2 and tc.expected_stage_model_id == "S2" and not tc.complete
    nyf = [b for b in tc.bends if b.state == "NOT_YET_FORMED"]
    assert len(nyf) == 2 and all(b.actual_oml is None for b in nyf)
    assert all(b.actual_oml is not None for b in tc.bends if b.state == "FORMED")
    # must-be-unknown characteristics have no truth and are never scorable
    for cid in tc.must_be_unknown:
        c = tc.characteristics[cid]
        assert c.applicable and c.truth is None and not scorable(c)
    # the backgauge resolves per side; an unknown side is unscorable
    bg = [c for c in tc.characteristics.values() if c.kind == "BACKGAUGE_LENGTH"]
    assert len(bg) == 1 and bg[0].per_side
    side = next(iter(bg[0].per_side))
    assert scorable(bg[0].for_side(side)) and not scorable(bg[0].for_side("Z"))
    # tampering with any listed file is refused
    copy = tmp_path / "case"
    shutil.copytree(tc.case_dir, copy)
    doc = json.loads((copy / "truth.json").read_text())
    doc["characteristics"][next(iter(doc["characteristics"]))]["truth"] = 1.0
    (copy / "truth.json").write_text(json.dumps(doc))
    with pytest.raises(TruthError):
        load_truth_case(copy)
    load_truth_case(copy, verify_hashes=False)  # explicit opt-out still loads


def test_mirrored_and_wrong_parts_have_no_scorable_truth(suite):
    tc, _, _ = suite["cases"]["E2-DEV-HT-006"]
    assert tc.mirrored and not tc.defined_part and not tc.complete
    assert not any(scorable(c) for c in tc.characteristics.values())
    assert tc.never_pass  # every characteristic is never-PASS


# ---------------------------------------------------------------------------
# exact truth passes every hard rule
# ---------------------------------------------------------------------------


def test_exact_oracle_passes_the_hard_gates(suite):
    card = _card(suite, _score(suite))
    for gid in ("G1.FALSE_PASS", "G1.FALSE_FAIL", "G1.U_CALIBRATION", "G1.STAGE_STATE", "G1.BACKGAUGE_DETECTION",
                "G1.NOT_YET_FORMED_NEVER_FAIL", "G1.WRONG_OR_MIRRORED_NEVER_PASS", "G1.SUMMARY_NEVER",
                "G1.MUST_BE_UNKNOWN", "G1.ENGINE_FAILURES", "G1.SYMMETRIC_NO_WRONG_ID", "G1.COVERAGE",
                "G1.ACC.FLANGE_LENGTH.BIAS", "G1.ACC.FLANGE_LENGTH.P95", "G1.ACC.INCLUDED_ANGLE.P95",
                "G1.ACC.BEND_LINE_POSITION.P95", "G1.DECISIVENESS", "G1.OCCLUSION_RAISES_UNKNOWN"):
        assert _gate(card, gid)["verdict"] == "PASS", (gid, _gate(card, gid))
    # a DEV oracle run is never a qualifying PASS
    assert not card["qualifying"]
    assert {"SEEDSET_DEV", "ORACLE_RESULTS", "NO_FREEZE_LEDGER"} <= set(card["not_qualifying_reasons"])
    assert card["verdict"] != "PASS"
    assert card["thresholds"]["sha256"] == thresholds_sha256(G1_THRESHOLDS)


def test_rows_join_truth_and_engine_per_characteristic(suite):
    scores = _score(suite, ids=("E2-DEV-UC-001",))
    rows = scores[0].chars
    tc, _, _ = suite["cases"]["E2-DEV-UC-001"]
    assert {r.characteristic_id for r in rows} == set(tc.characteristics)
    out = [r for r in rows if r.truth_in_tolerance is False]
    assert out and all(r.decision == "FAIL" for r in out)  # exact values, small U: clear FAILs
    assert all(abs(r.error) < 1e-5 for r in rows if r.error is not None)
    assert all(r.within_u for r in rows if r.within_u is not None)


# ---------------------------------------------------------------------------
# mutate-and-check: each gate fails on the defect it guards against
# ---------------------------------------------------------------------------


def _nominal(suite, cid, char_id):
    _, _, compiled = suite["cases"][cid]
    return compiled.nominal(char_id).nominal


def test_false_pass_is_judged_on_truth_not_on_u(suite):
    tc, _, _ = suite["cases"]["E2-DEV-UC-001"]
    target = next(c for c in tc.characteristics.values()
                  if c.kind == "FLANGE_LENGTH" and c.truth_in_tolerance is False)
    nominal = _nominal(suite, "E2-DEV-UC-001", target.characteristic_id)
    spec = OracleSpec(overrides={target.characteristic_id: {"actual": nominal, "u": 0.05}})
    card = _card(suite, _score(suite, {"E2-DEV-UC-001": spec}))
    g = _gate(card, "G1.FALSE_PASS")
    assert g["verdict"] == "FAIL" and g["value"] == 1
    assert f"E2-DEV-UC-001:{target.characteristic_id}:OUT_OF_TOLERANCE" in g["details"]["examples"]
    # the same value is also a U-calibration miss (|actual - truth| > U)
    assert _gate(card, "G1.U_CALIBRATION")["value"] < 1.0


def test_false_fail_rate(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-UC-000"]
    target = next(c for c in tc.characteristics.values() if c.kind == "FLANGE_LENGTH" and scorable(c))
    nom = compiled.nominal(target.characteristic_id)
    spec = OracleSpec(overrides={target.characteristic_id: {"actual": nom.nominal + 3 * nom.tolerance.plus}})
    thr = json.loads(json.dumps(G1_THRESHOLDS))
    card = _card(suite, _score(suite, {"E2-DEV-UC-000": spec}), thr)
    g = _gate(card, "G1.FALSE_FAIL")
    assert g["details"]["count"] == 1 and g["verdict"] == "FAIL"  # 1 / small population > 0.5 %


def test_overconfident_u_fails_calibration_and_noise_is_seeded(suite):
    noisy = OracleSpec(noise_mm=0.05, noise_deg=0.1, u_mm=0.01, u_deg=0.02, seed=7)
    specs = {cid: noisy for cid in CASES}
    card = _card(suite, _score(suite, specs))
    assert _gate(card, "G1.U_CALIBRATION")["verdict"] == "FAIL"
    again = _card(suite, _score(suite, specs))
    assert _gate(again, "G1.U_CALIBRATION")["value"] == _gate(card, "G1.U_CALIBRATION")["value"]
    honest = {cid: replace(noisy, u_mm=0.25, u_deg=0.5) for cid in CASES}
    assert _gate(_card(suite, _score(suite, honest)), "G1.U_CALIBRATION")["verdict"] == "PASS"


def test_bias_and_p95_gates_detect_a_biased_engine(suite):
    specs = {cid: OracleSpec(bias={"FLANGE_LENGTH": 0.12}) for cid in CASES}
    card = _card(suite, _score(suite, specs))
    bias = _gate(card, "G1.ACC.FLANGE_LENGTH.BIAS")
    assert bias["verdict"] == "FAIL" and bias["value"] == pytest.approx(0.12, abs=1e-6)
    assert _gate(card, "G1.ACC.FLANGE_LENGTH.P95")["verdict"] == "PASS"  # 0.12 < 0.20
    specs = {cid: OracleSpec(bias={"INCLUDED_ANGLE": 0.35}) for cid in CASES}
    card = _card(suite, _score(suite, specs))
    assert _gate(card, "G1.ACC.INCLUDED_ANGLE.P95")["verdict"] == "FAIL"
    assert _gate(card, "G1.ACC.INCLUDED_ANGLE.BIAS")["verdict"] == "FAIL"


def test_slope_detects_shrinkage_toward_nominal(suite):
    thr = json.loads(json.dumps(G1_THRESHOLDS))
    thr["slope"]["min_points"], thr["slope"]["min_large_points"] = 3, 2
    honest = _card(suite, _score(suite), thr)
    assert _gate(honest, "G1.SLOPE.FLANGE_LENGTH")["verdict"] == "PASS"
    shrunk = {cid: OracleSpec(gain={"FLANGE_LENGTH": 0.8, "INCLUDED_ANGLE": 0.8}) for cid in CASES}
    card = _card(suite, _score(suite, shrunk), thr)
    g = _gate(card, "G1.SLOPE.FLANGE_LENGTH")
    assert g["verdict"] == "FAIL" and g["value"] == pytest.approx(0.8, abs=1e-3)
    assert _gate(card, "G1.SLOPE.INCLUDED_ANGLE")["verdict"] == "FAIL"
    # too few large deviations is NOT_EVALUABLE, never PASS
    assert _gate(_card(suite, _score(suite)), "G1.SLOPE.BACKGAUGE_LENGTH")["verdict"] == "NOT_EVALUABLE"


def test_decisiveness_and_median_u(suite):
    wide = {cid: OracleSpec(u_mm=0.45, u_deg=0.9) for cid in CASES}
    card = _card(suite, _score(suite, wide))
    assert _gate(card, "G1.DECISIVENESS")["verdict"] == "FAIL"
    assert _gate(card, "G1.MEDIAN_U.mm")["verdict"] == "FAIL"
    assert _gate(card, "G1.MEDIAN_U.deg")["verdict"] == "FAIL"
    assert _gate(_card(suite, _score(suite)), "G1.MEDIAN_U.mm")["verdict"] == "PASS"


def test_unknown_bend_state_counts_as_an_error_on_unambiguous_cases(suite):
    tc, _, _ = suite["cases"]["E2-DEV-HT-005"]
    formed = next(b.bend_uid for b in tc.bends if b.state == "FORMED")
    spec = OracleSpec(bend_states={formed: "UNKNOWN"})
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": spec}))
    g = _gate(card, "G1.STAGE_STATE")
    assert g["verdict"] == "FAIL" and g["details"]["unknown_or_ambiguous_errors"] == 1
    # a wrong formed/unformed call is an error as well
    nyf = next(b.bend_uid for b in tc.bends if b.state == "NOT_YET_FORMED")
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": OracleSpec(bend_states={nyf: "FORMED"})}))
    assert _gate(card, "G1.STAGE_STATE")["verdict"] == "FAIL"


def test_symmetric_case_confident_wrong_id_fails(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-UCS-000"]
    scores = _score(suite, ids=("E2-DEV-UCS-000",))
    assert scores[0].case.symmetric and scores[0].case.ambiguity_source.startswith("E1:")
    a, b = (x.bend_uid for x in tc.bends)
    card = _card(suite, _score(suite, {"E2-DEV-UCS-000": OracleSpec(swap_lines=(a, b))}))
    g = _gate(card, "G1.SYMMETRIC_NO_WRONG_ID")
    assert g["verdict"] == "FAIL" and g["value"] == 2
    # declaring the pose ambiguous is the honest answer and passes (and is excused from coverage)
    card = _card(suite, _score(suite, {"E2-DEV-UCS-000": OracleSpec(pose_ambiguous=True)}))
    assert _gate(card, "G1.SYMMETRIC_NO_WRONG_ID")["verdict"] == "PASS"
    cov = _gate(card, "G1.COVERAGE")
    assert cov["details"]["excused_ambiguous"] > 0 and cov["verdict"] == "PASS"


def test_ambiguous_pose_on_an_unambiguous_case_is_not_excused(suite):
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": OracleSpec(pose_ambiguous=True)}))
    assert _gate(card, "G1.AMBIGUOUS_ON_UNAMBIGUOUS")["value"] == 1
    assert _gate(card, "G1.STAGE_STATE")["verdict"] == "FAIL"
    assert _gate(card, "G1.COVERAGE")["details"]["excused_ambiguous"] == 0


def test_wrong_line_on_an_asymmetric_part_is_caught_by_the_truth_frame_check(suite):
    tc, _, _ = suite["cases"]["E2-DEV-UC-001"]
    a, b = (x.bend_uid for x in tc.bends)
    scores = _score(suite, {"E2-DEV-UC-001": OracleSpec(swap_lines=(a, b))}, ids=("E2-DEV-UC-001",))
    checks = {r.bend_uid: r.line_check for r in scores[0].bends}
    assert set(checks.values()) == {"WRONG_ID"}
    honest = _score(suite, ids=("E2-DEV-UC-001",))
    assert {r.line_check for r in honest[0].bends} == {"ON_OWN_BEND"}


def test_mirrored_part_reported_valid_fails_never_pass_and_summary_rules(suite):
    card = _card(suite, _score(suite, {"E2-DEV-HT-006": OracleSpec(ignore_part_identity=True)}))
    assert _gate(card, "G1.WRONG_OR_MIRRORED_NEVER_PASS")["verdict"] == "FAIL"
    assert _gate(card, "G1.SUMMARY_NEVER")["verdict"] == "FAIL"


def test_missing_bend_claimed_formed_and_in_tolerance_fails_summary_never(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-LB-006"]
    assert tc.scenario == "MISSING_BEND" and tc.inspection_stage == "FINAL"
    uid = tc.bends[0].bend_uid
    overrides = {c.characteristic_id: {"actual": c.nominal} for c in compiled.characteristics
                 if c.bend_uid == uid and c.nominal is not None and c.recipe_requirement != "NOT_INSPECTED"}
    spec = OracleSpec(bend_states={uid: "FORMED"}, overrides=overrides)
    scores = _score(suite, {"E2-DEV-LB-006": spec})
    case = next(s.case for s in scores if s.case.case_id == "E2-DEV-LB-006")
    assert case.summary_state == "ALL_MEASURED_IN_TOLERANCE" and case.summary_never_violation
    card = _card(suite, scores)
    assert _gate(card, "G1.SUMMARY_NEVER")["verdict"] == "FAIL"
    assert _gate(card, "G1.NOT_YET_FORMED_NEVER_FAIL")["verdict"] == "PASS"  # PASS is not FAIL, but...
    assert _gate(card, "G1.STAGE_STATE")["verdict"] == "FAIL"  # ...the state is wrong


def test_not_yet_formed_bend_decided_fail_is_caught(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-HT-005"]
    nyf = next(b.bend_uid for b in tc.bends if b.state == "NOT_YET_FORMED")
    angle = f"{nyf}/INCLUDED_ANGLE"
    spec = OracleSpec(bend_states={nyf: "FORMED"}, overrides={angle: {"actual": compiled.nominal(angle).nominal + 20}})
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": spec}))
    g = _gate(card, "G1.NOT_YET_FORMED_NEVER_FAIL")
    assert g["verdict"] == "FAIL" and g["value"] >= 1


def test_must_be_unknown_measured_is_caught(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-HT-005"]
    cid = sorted(tc.must_be_unknown)[0]
    spec = OracleSpec(overrides={cid: {"actual": compiled.nominal(cid).nominal}})
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": spec}))
    g = _gate(card, "G1.MUST_BE_UNKNOWN")
    assert g["verdict"] == "FAIL" and g["details"]["decided"] == 1


def test_backgauge_errors_must_be_flagged(suite):
    tc, _, compiled = suite["cases"]["E2-DEV-HT-005"]
    bg = next(c for c in compiled.characteristics if c.kind == "BACKGAUGE_LENGTH" and c.stage_index == 2)
    honest = _card(suite, _score(suite))
    g = _gate(honest, "G1.BACKGAUGE_DETECTION")
    assert g["verdict"] == "PASS" and g["n"] >= 1
    spec = OracleSpec(overrides={bg.characteristic_id: {"actual": bg.nominal}})
    card = _card(suite, _score(suite, {"E2-DEV-HT-005": spec}))
    assert _gate(card, "G1.BACKGAUGE_DETECTION")["verdict"] == "FAIL"
    assert _gate(card, "G1.FALSE_PASS")["verdict"] == "FAIL"


def test_engine_failures_stay_in_every_denominator(suite):
    honest = _card(suite, _score(suite))
    runs = {"E2-DEV-UC-001": EngineRun("ENGINE_TIMEOUT", "hard stop", wall_s=180.0)}
    card = _card(suite, _score(suite, runs=runs))
    g = _gate(card, "G1.ENGINE_FAILURES")
    assert g["verdict"] == "FAIL" and g["details"]["by_status"] == {"ENGINE_TIMEOUT": 1}
    assert _gate(card, "G1.COVERAGE")["n"] == _gate(honest, "G1.COVERAGE")["n"]
    assert _gate(card, "G1.COVERAGE")["value"] < _gate(honest, "G1.COVERAGE")["value"]
    rows = next(s for s in _score(suite, runs=runs) if s.case.case_id == "E2-DEV-UC-001").chars
    assert {r.engine_state for r in rows} == {"ABSENT"} and not any(r.false_pass for r in rows)
    # a compile failure is not an engine failure but still costs coverage
    card = _card(suite, _score(suite, runs={"E2-DEV-UC-001": EngineRun("COMPILE_BLOCKED", "X")}))
    assert _gate(card, "G1.ENGINE_FAILURES")["verdict"] == "PASS"
    assert _gate(card, "G1.ENGINE_FAILURES")["details"]["compile_failures"] == ["E2-DEV-UC-001:COMPILE_BLOCKED"]
    assert _gate(card, "G1.COVERAGE")["value"] < _gate(honest, "G1.COVERAGE")["value"]


def test_occlusion_gate_requires_valid_results_on_degraded_plans(suite):
    honest = _card(suite, _score(suite))
    g = _gate(honest, "G1.OCCLUSION_RAISES_UNKNOWN")
    assert g["verdict"] == "PASS" and g["n"] == 1
    card = _card(suite, _score(suite, runs={"E2-DEV-LB-009": EngineRun("ENGINE_EXIT_3", "boom")}))
    assert _gate(card, "G1.OCCLUSION_RAISES_UNKNOWN")["verdict"] == "FAIL"


def test_case_score_round_trip_and_bootstrap_determinism(suite):
    scores = _score(suite)
    back = [CaseScore.from_dict(json.loads(json.dumps(s.to_dict()))) for s in scores]
    assert [s.to_dict() for s in back] == [s.to_dict() for s in scores]
    alien = json.loads(json.dumps(scores[0].to_dict()))
    alien["chars"][0]["a_field_from_another_eval_version"] = 1
    with pytest.raises(ValueError):
        CaseScore.from_dict(alien)  # rows scored by other eval code are never silently mixed in
    a = build_scorecard(scores, tier="regression", manifest=suite["manifest"])
    b = build_scorecard(back, tier="regression", manifest=suite["manifest"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    cov = _gate(a, "G1.COVERAGE")["ci"]
    assert cov["clusters"] >= 2 and cov["resamples_used"] > 0


def _fake_case(i: int, *, wall: float, rss: int, points: int = 1_000_000, family: str = "HAT") -> CaseScore:
    row = CaseRow(
        case_id=f"RT-{i}", family=family, scenario="COMPLETE_NOMINAL", scan_plan="FULL", scan_output="points",
        inspection_stage="FINAL", density_policy="TARGET_POINTS_OVERRIDE", point_count=points, complete=True,
        defined_part=True, mirrored=False, wrong_sku=False, mirror_distinguishable_by_rms=None, symmetric=False,
        ambiguity_source="E1:S4", symmetric_e2=False, run_status="OK", run_detail="", wall_s=wall,
        peak_rss_bytes=rss, engine_version="e3-test", oracle=False, validity_state="VALID",
        summary_state="ALL_MEASURED_IN_TOLERANCE", summary_never=(), summary_never_violation=False,
        pose_ambiguous=False, stage_model_id="S4", expected_stage_model_id="S4", stage_model_correct=True,
        inferred_stage=4, truth_formed_count=4, stage_status="CONSISTENT", stage_flags=("CONSISTENT",),
        unplanned_creases=0, truth_extra_bends=0, unmatched_engine_characteristics=(), unmatched_engine_decided=0)
    return CaseScore(row)


def test_runtime_gates_on_the_runtime_tier():
    manifest = {"tier": "runtime", "seedset": "DEV", "scanner_model": {"provenance": "ASSUMED"}}
    ok = [_fake_case(i, wall=10.0 + i, rss=2 * 1024**3) for i in range(10)]
    card = build_scorecard(ok, tier="runtime", manifest=manifest, bootstrap=False)
    p95, rss = _gate(card, "G1.RUNTIME.P95"), _gate(card, "G1.RUNTIME.RSS")
    assert p95["verdict"] == "PASS" and not p95["informational"]
    assert p95["value"] == pytest.approx(stats.percentile([10.0 + i for i in range(10)], 95))
    assert rss["verdict"] == "PASS"
    # accuracy gates never bind on a point-count override tier
    assert _gate(card, "G1.COVERAGE")["informational"]
    slow = ok[:9] + [_fake_case(9, wall=400.0, rss=5 * 1024**3)]
    card = build_scorecard(slow, tier="runtime", manifest=manifest, bootstrap=False)
    assert _gate(card, "G1.RUNTIME.RSS")["verdict"] == "FAIL"
    assert _gate(card, "G1.RUNTIME.P95")["verdict"] == "FAIL"
    assert card["verdict"] == "FAIL"
    # too few points: the runtime gate cannot be evaluated
    small = [_fake_case(i, wall=5.0, rss=10**9, points=200_000) for i in range(5)]
    assert _gate(build_scorecard(small, tier="runtime", manifest=manifest, bootstrap=False),
                 "G1.RUNTIME.P95")["verdict"] == "NOT_EVALUABLE"


def test_overall_g1_needs_qualifying_full_and_runtime_scorecards():
    manifest = {"tier": "runtime", "seedset": "DEV", "scanner_model": {"provenance": "ASSUMED"}}
    rt = build_scorecard([_fake_case(i, wall=5.0, rss=10**9) for i in range(10)], tier="runtime", manifest=manifest,
                         bootstrap=False)
    full = dict(rt, tier="full")
    v = g1_verdict(full, rt)
    assert v["verdict"] != "PASS" and v["not_qualifying_reasons"]


def test_stats_helpers_never_turn_empty_into_numbers():
    assert stats.mean([]) is None and stats.p95_abs([None]) is None and stats.rate(0, 0) is None
    assert stats.ols_slope([1.0, 1.0], [2.0, 3.0]) is None
    s, b = stats.ols_slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert s == pytest.approx(2.0) and b == pytest.approx(1.0)
    ci = stats.cluster_bootstrap({"a": [1.0], "b": [2.0], "c": [3.0]},
                                 lambda parts: sum(x for p in parts for x in p) / len(parts), resamples=200, seed=1)
    assert 1.0 <= ci["lo"] <= ci["hi"] <= 3.0
    assert stats.cluster_bootstrap({"a": [1.0]}, lambda p: 1.0, resamples=10, seed=1)["lo"] is None


# ---------------------------------------------------------------------------
# fix-stage mutate-and-check tests (review of 2026-10-17)
# ---------------------------------------------------------------------------


def test_pass_on_a_not_yet_formed_bend_is_a_false_pass(suite):
    """A contract-valid result can claim a truly flat bend FORMED and in tolerance; every
    such PASS is false (its true angle deviation is the whole sweep)."""
    tc, _, compiled = suite["cases"]["E2-DEV-HT-005"]
    nyf = next(b.bend_uid for b in tc.bends if b.state == "NOT_YET_FORMED")
    overrides = {c.characteristic_id: {"actual": c.nominal} for c in compiled.characteristics
                 if c.bend_uid == nyf and c.nominal is not None and c.recipe_requirement != "NOT_INSPECTED"}
    spec = OracleSpec(bend_states={nyf: "FORMED"}, overrides=overrides)
    scores = _score(suite, {"E2-DEV-HT-005": spec})
    rows = [r for s in scores for r in s.chars if r.bend_uid == nyf and r.decision == "PASS"]
    assert rows and all(r.false_pass and r.false_pass_kind == "NOT_YET_FORMED" for r in rows)
    card = _card(suite, scores)
    g = _gate(card, "G1.FALSE_PASS")
    assert g["verdict"] == "FAIL" and g["value"] == len(rows)
    assert g["details"]["by_kind"] == {"NOT_YET_FORMED": len(rows)}
    assert _gate(card, "G1.NOT_YET_FORMED_NEVER_FAIL")["details"]["decided_pass_counted_as_false_pass"] == len(rows)
    assert _gate(_card(suite, _score(suite)), "G1.FALSE_PASS")["verdict"] == "PASS"


def test_pass_on_a_mirrored_part_counts_as_a_false_pass(suite):
    card = _card(suite, _score(suite, {"E2-DEV-HT-006": OracleSpec(ignore_part_identity=True)}))
    g = _gate(card, "G1.FALSE_PASS")
    assert g["verdict"] == "FAIL" and set(g["details"]["by_kind"]) == {"NEVER_PASS_LIST"}


def _withheld_lines(tc) -> dict:
    return {f"{b.bend_uid}/BEND_LINE_POSITION": {"state": "NOT_MEASURED", "reason": "REGISTRATION_UNTRUSTED"}
            for b in tc.bends}


def test_symmetric_flip_is_caught_without_published_lines(suite):
    """The confident-wrong-id gate must not pass vacuously when E3 withholds every line:
    the registration itself is checked against the truth (pose check)."""
    tc, _, _ = suite["cases"]["E2-DEV-UCS-000"]
    honest = _score(suite, {"E2-DEV-UCS-000": OracleSpec(overrides=_withheld_lines(tc))}, ids=("E2-DEV-UCS-000",))
    assert {b.line_check for b in honest[0].bends} == {"NO_LINE"}
    assert honest[0].case.pose_check == "POSE_OK" and {b.identity for b in honest[0].bends} == {"RIGHT_ID"}
    assert _gate(_card(suite, honest), "G1.SYMMETRIC_NO_WRONG_ID")["verdict"] == "PASS"
    for spec in (OracleSpec(pose_flip=True, overrides=_withheld_lines(tc)), OracleSpec(pose_flip=True)):
        flipped = _score(suite, {"E2-DEV-UCS-000": spec}, ids=("E2-DEV-UCS-000",))
        case = flipped[0].case
        assert case.pose_check == "POSE_SYMMETRIC_FLIP" and case.pose_rotation_deg > 90 and case.pose_candidate
        assert {b.identity for b in flipped[0].bends} == {"WRONG_ID"}
        a, b = (x.bend_uid for x in tc.bends)
        assert {r.bend_uid: r.identity_target for r in flipped[0].bends} == {a: b, b: a}
        g = _gate(_card(suite, flipped), "G1.SYMMETRIC_NO_WRONG_ID")
        assert g["verdict"] == "FAIL" and g["value"] == 2
    lines = _score(suite, {"E2-DEV-UCS-000": OracleSpec(pose_flip=True)}, ids=("E2-DEV-UCS-000",))
    assert {b.line_check for b in lines[0].bends} == {"WRONG_ID"}  # the line check agrees when lines exist


def test_measured_bends_without_identity_evidence_are_never_a_pass(suite):
    tc, recipe, compiled = suite["cases"]["E2-DEV-UCS-000"]
    blind = replace(tc, raw_to_cad=None)  # no registration truth and no lines: identity cannot be judged
    result = oracle_result(recipe=recipe, compiled=compiled, truth=tc, scan_sha256=tc.files["scan.ply"],
                           spec=OracleSpec(overrides=_withheld_lines(tc)))
    score = score_case(blind, EngineRun("OK"), result, compiled)
    assert {b.identity for b in score.bends} == {"NOT_EVALUATED"} and all(b.measured_any for b in score.bends)
    g = _gate(_card(suite, [score]), "G1.SYMMETRIC_NO_WRONG_ID")
    assert g["verdict"] == "NOT_EVALUABLE" and g["details"]["measured_without_identity_evidence"] == 2


def test_gross_misregistration_is_a_wrong_pose(suite):
    import numpy as np

    tc, recipe, compiled = suite["cases"]["E2-DEV-UC-001"]
    shifted = np.array(tc.raw_to_cad, dtype=float)
    shifted[:3, 3] += [25.0, 0.0, 0.0]  # the engine believes the part sits 25 mm away
    result = oracle_result(recipe=recipe, compiled=compiled, truth=replace(tc, raw_to_cad=shifted),
                           scan_sha256=tc.files["scan.ply"], spec=OracleSpec(overrides=_withheld_lines(tc)))
    score = score_case(tc, EngineRun("OK"), result, compiled)
    assert score.case.pose_check == "POSE_WRONG" and score.case.pose_rms_mm == pytest.approx(25.0, abs=0.01)
    assert {b.identity for b in score.bends} == {"WRONG_POSE"}
    assert _gate(_card(suite, [score]), "G1.POSE_CHECK")["details"]["verdicts"] == {"POSE_WRONG": 1}


def test_regression_verdict_ignores_only_gates_absent_from_the_tier(suite):
    ids = tuple(c for c in CASES if c != "E2-DEV-LB-009")  # no degraded scan plan left
    card = _card(suite, _score(suite, ids=ids))
    occ = _gate(card, "G1.OCCLUSION_RAISES_UNKNOWN")
    assert occ["verdict"] == "NOT_EVALUABLE" and occ["not_applicable_to_tier"] and occ["truth_population_n"] == 0
    assert "G1.OCCLUSION_RAISES_UNKNOWN" in card["not_applicable_to_tier_gates"]
    assert card["verdict"] == "INCOMPLETE" and card["regression_verdict"] == "PASS"
    # an engine that measures nothing leaves truth populations non-empty: never "not applicable"
    silent = {cid: OracleSpec(overrides={c: {"state": "NOT_MEASURED", "reason": "OCCLUDED"}
                                         for c in suite["cases"][cid][0].characteristics})
              for cid in ids}
    card = _card(suite, _score(suite, silent, ids=ids))
    acc = _gate(card, "G1.ACC.FLANGE_LENGTH.P95")
    assert acc["verdict"] == "NOT_EVALUABLE" and not acc["not_applicable_to_tier"] and acc["truth_population_n"] > 0
    assert card["regression_verdict"] == "FAIL"  # coverage collapses
    timeouts = {cid: EngineRun("ENGINE_TIMEOUT", "x", wall_s=180.0) for cid in ids}
    assert _card(suite, _score(suite, runs=timeouts, ids=ids))["regression_verdict"] == "FAIL"


_REPO = Path(__file__).resolve().parents[1]
_CARD = _REPO / "docs" / "production" / "runlog" / "scanner_model_card.json"
_STANDARD = {"engine_argv_tail": ["-m", "domains.production.engine.inspect"], "engine_timeout_s": 180.0,
             "engine_rss_limit_bytes": 4 * 1024**3, "engine_time_budget_s": 170.0, "engine_extra_env_keys": []}


def test_partial_scanner_card_and_run_conditions_block_qualification(suite):
    card_doc = json.loads(_CARD.read_text(encoding="utf-8"))
    gaps = scanner_model_gaps({"scanner_model": card_doc})
    assert gaps["provenance"] == "MODEL_CARD"
    assert set(gaps["unmeasured_required_fields"]) == {"grazing_start_deg", "grazing_cutoff_deg", "specular_cone_deg",
                                                       "specular_dropout_probability"}
    assert gaps["field_status"]["edge_flare_mm"] == "LOWER_BOUND"
    assert gaps["field_status"]["duplicate_skin_offset_mm"] == "UPPER_BOUND"
    manifest = {"tier": "full", "seedset": "QUAL", "scanner_model": card_doc, "scanner_model_sha256": "a" * 64}
    ledger = {"ok": True, "anchor_checked": True, "first_entry_sha256": "b" * 64, "waivers": []}
    sc = build_scorecard([], tier="full", manifest=manifest, ledger_check=ledger, bootstrap=False,
                         provenance={"run_conditions": _STANDARD})
    assert "SCANNER_MODEL_PARTIAL" in sc["not_qualifying_reasons"]
    assert set(sc["provenance"]["scanner_model_unmeasured_required_fields"]) == set(gaps["unmeasured_required_fields"])
    assert not any(r.startswith("NON_STANDARD") or r == "RUN_CONDITIONS_NOT_RECORDED" for r in sc["not_qualifying_reasons"])
    waiver = {"scope": "SCANNER_MODEL_PARTIAL", "scanner_model_sha256": "a" * 64, "approved_by": "owner",
              "fields": gaps["unmeasured_required_fields"], "entry_id": "frz_x"}
    sc = build_scorecard([], tier="full", manifest=manifest, ledger_check=dict(ledger, waivers=[waiver]),
                         bootstrap=False, provenance={"run_conditions": _STANDARD})
    assert "SCANNER_MODEL_PARTIAL" not in sc["not_qualifying_reasons"] and "owner waiver frz_x" in sc["label"]
    partial_waiver = dict(waiver, fields=gaps["unmeasured_required_fields"][:1])
    sc = build_scorecard([], tier="full", manifest=manifest, ledger_check=dict(ledger, waivers=[partial_waiver]),
                         bootstrap=False, provenance={"run_conditions": _STANDARD})
    assert "SCANNER_MODEL_PARTIAL" in sc["not_qualifying_reasons"]
    # run conditions
    assert run_condition_reasons(_STANDARD) == []
    assert run_condition_reasons(None) == ["RUN_CONDITIONS_NOT_RECORDED"]
    assert run_condition_reasons(dict(_STANDARD, engine_timeout_s=2.0)) == ["NON_STANDARD_LIMITS"]
    assert run_condition_reasons(dict(_STANDARD, engine_time_budget_s=60.0)) == ["NON_STANDARD_LIMITS"]
    assert run_condition_reasons(dict(_STANDARD, engine_argv_tail=["-m", "domains.production.engine.eval.oracle"])) == [
        "NON_STANDARD_ENGINE_CMD"]
    assert run_condition_reasons(dict(_STANDARD, engine_extra_env_keys=["SHERMAN_E4_ORACLE_CASES"])) == [
        "NON_STANDARD_ENGINE_ENV"]
    unanchored = build_scorecard([], tier="full", manifest=manifest, ledger_check=dict(ledger, anchor_checked=False),
                                 bootstrap=False, provenance={"run_conditions": _STANDARD})
    assert "LEDGER_NOT_ANCHORED" in unanchored["not_qualifying_reasons"]
    reused = build_scorecard([], tier="full", manifest=manifest, ledger_check=dict(ledger, qual_seeds_reused=True),
                             bootstrap=False, provenance={"run_conditions": _STANDARD})
    assert "QUAL_SEEDS_REUSED" in reused["not_qualifying_reasons"]


def _qualifying(tier: str, *, engine="e" * 64, ledger="l" * 64, cfg=None, failing: str | None = None) -> dict:
    gates = [{"id": f"G1.X{i}", "verdict": "PASS", "informational": False} for i in range(3)]
    if failing:
        gates.append({"id": failing, "verdict": "FAIL", "informational": False})
    return {"tier": tier, "qualifying": True, "not_qualifying_reasons": [], "gates": gates,
            "provenance": {"fingerprints": {"engine": engine}, "engine_config_sha256": cfg,
                           "ledger_check": {"first_entry_sha256": ledger}}}


def test_g1_verdict_positive_path_and_identity_checks():
    assert g1_verdict(_qualifying("full"), _qualifying("runtime"))["verdict"] == "PASS"
    v = g1_verdict(_qualifying("full"), _qualifying("runtime", engine="f" * 64))
    assert v["verdict"] == "INCOMPLETE" and "ENGINE_IDENTITY_DIFFERS_BETWEEN_TIERS" in v["not_qualifying_reasons"]
    v = g1_verdict(_qualifying("full", cfg="c" * 64), _qualifying("runtime"))
    assert "ENGINE_IDENTITY_DIFFERS_BETWEEN_TIERS" in v["not_qualifying_reasons"]
    v = g1_verdict(_qualifying("full"), _qualifying("runtime", ledger="m" * 64))
    assert v["verdict"] == "INCOMPLETE" and "LEDGER_DIFFERS_BETWEEN_TIERS" in v["not_qualifying_reasons"]
    v = g1_verdict(_qualifying("full", failing="G1.FALSE_PASS"), _qualifying("runtime"))
    assert v["verdict"] == "FAIL" and v["failed_gates"] == ["G1.FALSE_PASS"]
    v = g1_verdict(_qualifying("regression"), _qualifying("runtime"))
    assert v["verdict"] == "INCOMPLETE" and "FULL_SCORECARD_HAS_TIER_regression" in v["not_qualifying_reasons"]
