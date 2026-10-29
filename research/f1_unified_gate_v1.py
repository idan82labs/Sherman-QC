"""Fail-closed, lane-aware F1 research generalization and operations gate."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA = "F1_GENERALIZATION_EVIDENCE_V1"
LANES = frozenset({"SCAN_ONLY", "CAD_ASSISTED"})
EVIDENCE_CLASSES = frozenset({"SOURCE_FREE", "SYNTHETIC", "CONSUMED_PHYSICAL", "UNTOUCHED_PHYSICAL"})
HEX = frozenset("0123456789abcdef")
GIB = 1024 ** 3
DEFAULT_THRESHOLDS = {
    "minimum_scans": 30,
    "minimum_morphology_families": 15,
    "minimum_independent_instances": 30,
    "minimum_job_attempts": 60,
    "minimum_precision": 0.90,
    "minimum_recall": 0.90,
    "minimum_exact_count_accuracy": 0.90,
    "minimum_marker_admissibility": 0.99,
    "minimum_family_recall": 0.80,
    "minimum_replay_rate": 1.0,
    "maximum_p95_latency_sec": 300.0,
    "maximum_peak_process_tree_rss_bytes": 16 * GIB,
    "minimum_job_success_rate": 0.995,
}
METRICS = (
    "precision", "recall", "f1", "exact_count_accuracy",
    "exact_localized_target_set_accuracy", "minimum_family_recall",
    "marker_admissibility",
)


class UnifiedGateError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnifiedGateError(f"{path} must be an object")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise UnifiedGateError(f"{path} must be boolean")
    return value


def _number(value: Any, path: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise UnifiedGateError(f"{path} must be a finite number")
    return float(value)


def _ratio(value: Any, path: str, *, optional: bool = False) -> float | None:
    result = _number(value, path, optional=optional)
    if result is not None and not 0.0 <= result <= 1.0:
        raise UnifiedGateError(f"{path} must be within [0,1]")
    return result


def _count(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnifiedGateError(f"{path} must be a non-negative integer")
    return value


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in HEX for item in value):
        raise UnifiedGateError(f"{path} must be lowercase hex64")
    return value


def _thresholds(overrides: Mapping[str, Any] | None) -> dict[str, int | float]:
    raw = dict(overrides or {})
    unknown = set(raw) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise UnifiedGateError(f"unknown threshold keys: {sorted(unknown)}")
    result: dict[str, int | float] = dict(DEFAULT_THRESHOLDS)
    count_keys = {
        "minimum_scans", "minimum_morphology_families", "minimum_independent_instances",
        "minimum_job_attempts", "maximum_peak_process_tree_rss_bytes",
    }
    ratio_keys = {
        "minimum_precision", "minimum_recall", "minimum_exact_count_accuracy",
        "minimum_marker_admissibility", "minimum_family_recall", "minimum_replay_rate",
        "minimum_job_success_rate",
    }
    for key, value in raw.items():
        if key in count_keys:
            normalized: int | float = _count(value, f"thresholds.{key}")
            if key == "maximum_peak_process_tree_rss_bytes" and normalized <= 0:
                raise UnifiedGateError(f"thresholds.{key} must be positive")
        elif key in ratio_keys:
            checked = _ratio(value, f"thresholds.{key}")
            assert checked is not None
            normalized = checked
        elif key == "maximum_p95_latency_sec":
            checked = _number(value, f"thresholds.{key}")
            assert checked is not None
            if checked <= 0:
                raise UnifiedGateError(f"thresholds.{key} must be positive")
            normalized = checked
        else:  # guarded by the unknown-key check
            raise UnifiedGateError(f"unsupported threshold {key}")
        default = DEFAULT_THRESHOLDS[key]
        if key.startswith("minimum_") and normalized < default:
            raise UnifiedGateError(f"thresholds.{key} may not weaken the frozen gate")
        if key.startswith("maximum_") and normalized > default:
            raise UnifiedGateError(f"thresholds.{key} may not weaken the frozen gate")
        result[key] = normalized
    return result


def _metrics(value: Any, path: str) -> dict[str, float | None]:
    row = _mapping(value, path)
    result = {key: _ratio(row.get(key), f"{path}.{key}", optional=True) for key in METRICS}
    precision, recall, f1 = result["precision"], result["recall"], result["f1"]
    if precision is not None and recall is not None and f1 is not None:
        expected_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if abs(f1 - expected_f1) > 1e-9:
            raise UnifiedGateError(f"{path}.f1 contradicts precision and recall")
    exact_count = result["exact_count_accuracy"]
    exact_localized = result["exact_localized_target_set_accuracy"]
    if exact_count is not None and exact_localized is not None and exact_localized > exact_count + 1e-12:
        raise UnifiedGateError(f"{path}.exact_localized_target_set_accuracy exceeds exact count accuracy")
    minimum_family = result["minimum_family_recall"]
    if recall is not None and minimum_family is not None and minimum_family > recall + 1e-12:
        raise UnifiedGateError(f"{path}.minimum_family_recall exceeds aggregate recall")
    return result


def evaluate(payload: Any, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = _mapping(payload, "payload")
    if row.get("schema") != SCHEMA:
        raise UnifiedGateError(f"schema must be {SCHEMA}")
    version = str(row.get("version") or "").strip()
    if not version:
        raise UnifiedGateError("version is required")
    lane = str(row.get("lane") or "")
    evidence_class = str(row.get("evidence_class") or "")
    if lane not in LANES or evidence_class not in EVIDENCE_CLASSES:
        raise UnifiedGateError("invalid lane or evidence_class")
    cfg = _thresholds(thresholds)
    hashes = _mapping(row.get("hashes"), "hashes")
    bound_hashes = {key: _hex64(hashes.get(key), f"hashes.{key}") for key in ("detector", "baseline", "scorer", "ontology", "dataset")}
    counts = _mapping(row.get("counts"), "counts")
    normalized_counts = {key: _count(counts.get(key), f"counts.{key}") for key in ("physical_scans", "morphology_families", "independent_instances", "jobs_attempted", "jobs_succeeded")}
    if normalized_counts["jobs_succeeded"] > normalized_counts["jobs_attempted"]:
        raise UnifiedGateError("jobs_succeeded cannot exceed jobs_attempted")
    truth = _mapping(row.get("truth"), "truth")
    normalized_truth = {key: _bool(truth.get(key), f"truth.{key}") for key in ("independent_line_truth", "dual_review_verified", "morphology_held_out")}
    comparison = _mapping(row.get("comparison"), "comparison")
    matched = _bool(comparison.get("matched_frozen_baseline"), "comparison.matched_frozen_baseline")
    baseline_lane = str(comparison.get("baseline_lane") or "")
    candidate_lane = str(comparison.get("candidate_lane") or "")
    baseline_metrics = _metrics(row.get("baseline_metrics"), "baseline_metrics")
    candidate_metrics = _metrics(row.get("candidate_metrics"), "candidate_metrics")
    visual = _mapping(row.get("visual"), "visual")
    visual_complete = _bool(visual.get("complete"), "visual.complete")
    visual_passed = _bool(visual.get("passed"), "visual.passed")
    operations = _mapping(row.get("operations"), "operations")
    replay = _ratio(operations.get("deterministic_replay_rate"), "operations.deterministic_replay_rate", optional=True)
    latency = _number(operations.get("p95_latency_sec"), "operations.p95_latency_sec", optional=True)
    rss = _number(operations.get("peak_process_tree_rss_bytes"), "operations.peak_process_tree_rss_bytes", optional=True)
    job_success = _ratio(operations.get("job_success_rate"), "operations.job_success_rate", optional=True)
    if latency is not None and latency < 0:
        raise UnifiedGateError("operations.p95_latency_sec must be non-negative")
    if rss is not None and rss < 0:
        raise UnifiedGateError("operations.peak_process_tree_rss_bytes must be non-negative")
    expected_job_success = (
        normalized_counts["jobs_succeeded"] / normalized_counts["jobs_attempted"]
        if normalized_counts["jobs_attempted"] else None
    )
    if job_success is not None and (expected_job_success is None or abs(job_success - expected_job_success) > 1e-12):
        raise UnifiedGateError("operations.job_success_rate contradicts job counts")
    protocol_valid = _bool(row.get("protocol_valid"), "protocol_valid")
    leakage_free = _bool(row.get("leakage_free"), "leakage_free")
    integrity_reasons = []
    if not protocol_valid:
        integrity_reasons.append("protocol_invalid")
    if not leakage_free:
        integrity_reasons.append("leakage_detected")
    if baseline_lane != lane or candidate_lane != lane:
        integrity_reasons.append("cross_lane_comparison")
    if not matched:
        integrity_reasons.append("unmatched_frozen_baseline")
    if evidence_class != "UNTOUCHED_PHYSICAL" and normalized_counts["physical_scans"] == 0:
        pass
    clean = (
        evidence_class == "UNTOUCHED_PHYSICAL"
        and normalized_counts["physical_scans"] > 0
        and normalized_counts["morphology_families"] > 0
        and normalized_truth["independent_line_truth"]
        and normalized_truth["dual_review_verified"]
        and normalized_truth["morphology_held_out"]
        and matched
    )
    deltas = {
        key: (
            None if baseline_metrics[key] is None or candidate_metrics[key] is None
            else candidate_metrics[key] - baseline_metrics[key]
        ) for key in METRICS
    }
    ops_gates = {
        "replay": replay is not None and replay >= float(cfg["minimum_replay_rate"]),
        "latency": latency is not None and latency <= float(cfg["maximum_p95_latency_sec"]),
        "memory": rss is not None and rss < float(cfg["maximum_peak_process_tree_rss_bytes"]),
        "job_success": job_success is not None and job_success >= float(cfg["minimum_job_success_rate"]),
        "job_denominator": normalized_counts["jobs_attempted"] >= int(cfg["minimum_job_attempts"]),
    }
    accuracy_gates = {
        "precision": candidate_metrics["precision"] is not None and candidate_metrics["precision"] >= float(cfg["minimum_precision"]),
        "recall": candidate_metrics["recall"] is not None and candidate_metrics["recall"] >= float(cfg["minimum_recall"]),
        "exact_count": candidate_metrics["exact_count_accuracy"] is not None and candidate_metrics["exact_count_accuracy"] >= float(cfg["minimum_exact_count_accuracy"]),
        "marker_admissibility": candidate_metrics["marker_admissibility"] is not None and candidate_metrics["marker_admissibility"] >= float(cfg["minimum_marker_admissibility"]),
        "minimum_family_recall": candidate_metrics["minimum_family_recall"] is not None and candidate_metrics["minimum_family_recall"] >= float(cfg["minimum_family_recall"]),
    }
    scale_gates = {
        "scan_count": normalized_counts["physical_scans"] >= int(cfg["minimum_scans"]),
        "family_count": normalized_counts["morphology_families"] >= int(cfg["minimum_morphology_families"]),
        "independent_instances": normalized_counts["independent_instances"] >= int(cfg["minimum_independent_instances"]),
    }
    if integrity_reasons:
        grade = "F_NON_EVALUABLE"; direction = "NON_EVALUABLE"; reasons = integrity_reasons
    elif not clean:
        grade = "C_NO_GENERALIZATION_EVIDENCE"
        direction = "NEGATIVE_DEVELOPMENT" if deltas["f1"] is not None and deltas["f1"] < 0 else "POSITIVE_ENGINEERING" if deltas["f1"] is not None and deltas["f1"] > 0 else "NO_CLEAN_PHYSICAL_COMPARISON"
        reasons = ["not_untouched_independent_morphology_held_out_comparison"]
    else:
        required_deltas = (deltas["f1"], deltas["exact_count_accuracy"], deltas["minimum_family_recall"])
        if any(value is None for value in required_deltas):
            grade = "F_NON_EVALUABLE"; direction = "NON_EVALUABLE"; reasons = ["required_comparison_metric_missing"]
        elif visual_complete is False:
            grade = "F_NON_EVALUABLE"; direction = "NON_EVALUABLE"; reasons = ["native_resolution_visual_evidence_incomplete"]
        elif visual_passed is False or deltas["f1"] < 0 or deltas["exact_count_accuracy"] < 0 or deltas["minimum_family_recall"] < 0:
            grade = "D_NEGATIVE_PHYSICAL_INDICATOR"; direction = "NEGATIVE"; reasons = ["clean_physical_regression_or_visual_failure"]
        elif deltas["f1"] <= 0:
            grade = "C_NO_GENERALIZATION_EVIDENCE"; direction = "INCONCLUSIVE_PHYSICAL_COMPARISON"; reasons = ["no_strict_f1_gain"]
        elif all(scale_gates.values()) and all(accuracy_gates.values()) and all(ops_gates.values()) and visual_complete and visual_passed:
            grade = "A_VERIFIED_PHYSICAL_PROGRESS"; direction = "VERIFIED_POSITIVE"; reasons = ["all_research_30x15_gates_passed"]
        else:
            grade = "B_POSITIVE_PHYSICAL_INDICATOR"; direction = "POSITIVE_BUT_INCOMPLETE"; reasons = ["clean_physical_gain_but_scale_accuracy_or_operations_incomplete"]
    operational_badge = "RED"
    if all(ops_gates.values()) and visual_complete and visual_passed:
        operational_badge = "SHADOW"
    if operational_badge == "SHADOW" and visual_passed and candidate_metrics["precision"] is not None and candidate_metrics["recall"] is not None and candidate_metrics["precision"] >= 0.95 and candidate_metrics["recall"] >= 0.95:
        operational_badge = "SUPERVISED"
    if grade == "A_VERIFIED_PHYSICAL_PROGRESS":
        operational_badge = "AUTONOMOUS_RESEARCH_GATE"
    return {
        "schema_version": 1,
        "kind": "F1_UNIFIED_GENERALIZATION_GATE",
        "version": version,
        "lane": lane,
        "evidence_class": evidence_class,
        "grade": grade,
        "direction": direction,
        "operational_badge": operational_badge,
        "reasons": reasons,
        "hashes": bound_hashes,
        "counts": normalized_counts,
        "truth": normalized_truth,
        "comparison": {"matched_frozen_baseline": matched, "baseline_lane": baseline_lane, "candidate_lane": candidate_lane},
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": deltas,
        "visual": {"complete": visual_complete, "passed": visual_passed},
        "operations": {"deterministic_replay_rate": replay, "p95_latency_sec": latency, "peak_process_tree_rss_bytes": rss, "job_success_rate": job_success},
        "gates": {"scale": scale_gates, "accuracy": accuracy_gates, "operations": ops_gates},
        "thresholds": cfg,
    }
