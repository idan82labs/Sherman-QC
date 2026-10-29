from __future__ import annotations

import copy

import pytest

from research.f1_unified_gate_v1 import UnifiedGateError, evaluate


HASHES = {key: value * 64 for key, value in zip(("detector", "baseline", "scorer", "ontology", "dataset"), "abcde")}


def evidence() -> dict[str, object]:
    return {
        "schema": "F1_GENERALIZATION_EVIDENCE_V1",
        "version": "candidate",
        "lane": "CAD_ASSISTED",
        "evidence_class": "UNTOUCHED_PHYSICAL",
        "protocol_valid": True,
        "leakage_free": True,
        "hashes": dict(HASHES),
        "counts": {"physical_scans": 30, "morphology_families": 15, "independent_instances": 30, "jobs_attempted": 60, "jobs_succeeded": 60},
        "truth": {"independent_line_truth": True, "dual_review_verified": True, "morphology_held_out": True},
        "comparison": {"matched_frozen_baseline": True, "baseline_lane": "CAD_ASSISTED", "candidate_lane": "CAD_ASSISTED"},
        "baseline_metrics": {"precision": .85, "recall": .84, "f1": 2 * .85 * .84 / (.85 + .84), "exact_count_accuracy": .8, "exact_localized_target_set_accuracy": .7, "minimum_family_recall": .8, "marker_admissibility": .98},
        "candidate_metrics": {"precision": .92, "recall": .91, "f1": 2 * .92 * .91 / (.92 + .91), "exact_count_accuracy": .9, "exact_localized_target_set_accuracy": .85, "minimum_family_recall": .82, "marker_admissibility": .995},
        "visual": {"complete": True, "passed": True},
        "operations": {"deterministic_replay_rate": 1.0, "p95_latency_sec": 200, "peak_process_tree_rss_bytes": 2 * 1024**3, "job_success_rate": 1.0},
    }


def test_all_research_gates_yield_a_and_autonomous_research_badge() -> None:
    report = evaluate(evidence())
    assert report["grade"] == "A_VERIFIED_PHYSICAL_PROGRESS"
    assert report["operational_badge"] == "AUTONOMOUS_RESEARCH_GATE"
    assert report["gates"]["operations"] == {"replay": True, "latency": True, "memory": True, "job_success": True, "job_denominator": True}


def test_clean_underpowered_gain_is_b() -> None:
    row = evidence(); row["counts"]["physical_scans"] = 10
    report = evaluate(row)
    assert report["grade"] == "B_POSITIVE_PHYSICAL_INDICATOR"


def test_consumed_positive_and_negative_evidence_remain_c_with_direction() -> None:
    positive = evidence(); positive["evidence_class"] = "CONSUMED_PHYSICAL"
    assert evaluate(positive)["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    assert evaluate(positive)["direction"] == "POSITIVE_ENGINEERING"
    negative = copy.deepcopy(positive); negative["candidate_metrics"].update(precision=.7, recall=.7, f1=.7, minimum_family_recall=.6)
    assert evaluate(negative)["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    assert evaluate(negative)["direction"] == "NEGATIVE_DEVELOPMENT"


def test_clean_regression_or_visual_failure_is_d() -> None:
    row = evidence(); row["candidate_metrics"].update(precision=.8, recall=.8, f1=.8, minimum_family_recall=.7)
    assert evaluate(row)["grade"] == "D_NEGATIVE_PHYSICAL_INDICATOR"
    row = evidence(); row["visual"]["passed"] = False
    assert evaluate(row)["grade"] == "D_NEGATIVE_PHYSICAL_INDICATOR"


def test_missing_visual_evidence_is_non_evaluable() -> None:
    row = evidence(); row["visual"]["complete"] = False
    assert evaluate(row)["grade"] == "F_NON_EVALUABLE"


def test_cross_lane_or_leakage_is_f() -> None:
    row = evidence(); row["comparison"]["baseline_lane"] = "SCAN_ONLY"
    assert evaluate(row)["grade"] == "F_NON_EVALUABLE"
    row = evidence(); row["leakage_free"] = False
    assert evaluate(row)["grade"] == "F_NON_EVALUABLE"
    row = evidence(); row["comparison"]["matched_frozen_baseline"] = False
    report = evaluate(row)
    assert report["grade"] == "F_NON_EVALUABLE"
    assert "unmatched_frozen_baseline" in report["reasons"]


def test_operational_gates_are_independent_and_fail_closed() -> None:
    row = evidence(); row["operations"]["p95_latency_sec"] = 301
    report = evaluate(row)
    assert report["grade"] == "B_POSITIVE_PHYSICAL_INDICATOR"
    assert report["gates"]["operations"]["latency"] is False
    assert report["gates"]["operations"]["memory"] is True
    assert report["operational_badge"] == "RED"
    row = evidence(); row["counts"]["jobs_attempted"] = 59; row["counts"]["jobs_succeeded"] = 59
    report = evaluate(row)
    assert report["gates"]["operations"]["job_denominator"] is False
    assert report["grade"] == "B_POSITIVE_PHYSICAL_INDICATOR"


def test_threshold_overrides_may_tighten_but_never_weaken_or_add_keys() -> None:
    strict = evaluate(evidence(), {"minimum_precision": 0.95, "maximum_p95_latency_sec": 250.0})
    assert strict["gates"]["accuracy"]["precision"] is False
    with pytest.raises(UnifiedGateError, match="may not weaken"):
        evaluate(evidence(), {"minimum_precision": 0.1})
    with pytest.raises(UnifiedGateError, match="may not weaken"):
        evaluate(evidence(), {"maximum_p95_latency_sec": 999.0})
    with pytest.raises(UnifiedGateError, match="unknown threshold"):
        evaluate(evidence(), {"minimum_magic": 0.0})


@pytest.mark.parametrize("mutation", [
    lambda row: row["hashes"].__setitem__("dataset", "bad"),
    lambda row: row["counts"].__setitem__("jobs_succeeded", 61),
    lambda row: row["candidate_metrics"].__setitem__("precision", 1.1),
    lambda row: row["operations"].__setitem__("peak_process_tree_rss_bytes", float("nan")),
    lambda row: row["operations"].__setitem__("p95_latency_sec", -1),
    lambda row: row["operations"].__setitem__("peak_process_tree_rss_bytes", -1),
    lambda row: row["counts"].__setitem__("jobs_succeeded", 59),
    lambda row: row["candidate_metrics"].__setitem__("f1", 0.1),
    lambda row: row["candidate_metrics"].__setitem__("exact_localized_target_set_accuracy", 0.95),
    lambda row: row["candidate_metrics"].__setitem__("minimum_family_recall", 0.99),
])
def test_malformed_evidence_is_rejected(mutation) -> None:
    row = evidence(); mutation(row)
    with pytest.raises(UnifiedGateError):
        evaluate(row)
