"""Numerical and visual qualification gates for independent CIBEC annotations."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping

from research.cibec_v1.agreement import is_agreement, match_lines, pair_metrics
from research.cibec_v1.contract import canonicalize_scan_truth


SCHEMA = "CIBEC_V1_PHYSICAL_ANNOTATION_QUALIFICATION_INPUT"
HEX = frozenset("0123456789abcdef")
MAX_TOOL_RSS_BYTES = 2 * 1024**3


class QualificationInputError(ValueError):
    pass


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _finite_nonnegative(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise QualificationInputError(f"{path} must be finite and non-negative")
    return float(value)


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise QualificationInputError(f"{path} must be lowercase hex64")
    return value


def score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise QualificationInputError(f"schema must be {SCHEMA}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 13:
        raise QualificationInputError("exactly 13 consumed qualification cases are required")
    total_first = total_second = total_matches = class_matches = exact_counts = 0
    offsets: list[float] = []
    angles: list[float] = []
    unresolved = nonline = duplicates = 0
    visual_passes = 0
    export_latencies: list[float] = []
    job_successes = 0
    replay_failures = 0
    peak_rss_bytes = 0
    written_justifications = 0
    rows = []
    seen_cases = set()
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            raise QualificationInputError(f"cases[{index}] must be an object")
        first = canonicalize_scan_truth(raw.get("annotation_a"))
        second = canonicalize_scan_truth(raw.get("annotation_b"))
        final = canonicalize_scan_truth(raw.get("adjudicated"))
        bindings = {(item["case_id"], item["scan_sha256"], item["ontology_sha256"], item["renderer_sha256"]) for item in (first, second, final)}
        if len(bindings) != 1:
            raise QualificationInputError(f"cases[{index}] annotation bindings differ")
        case_id = first["case_id"]
        if case_id in seen_cases:
            raise QualificationInputError("duplicate case_id")
        seen_cases.add(case_id)
        if first["case_status"] != "RESOLVED" or second["case_status"] != "RESOLVED":
            raise QualificationInputError("primary annotations must be resolved before agreement scoring")
        agreement = match_lines(first["events"], second["events"])
        first_by_id = {row["event_id"]: row for row in first["events"]}
        second_by_id = {row["event_id"]: row for row in second["events"]}
        matched_classes = 0
        for match in agreement["matches"]:
            offsets.append(float(match["transverse_offset_mm"]))
            angles.append(float(match["axis_angle_deg"]))
            if first_by_id[match["first_event_id"]]["class_code"] == second_by_id[match["second_event_id"]]["class_code"]:
                matched_classes += 1
        total_first += len(first["events"])
        total_second += len(second["events"])
        total_matches += agreement["true_positive"]
        class_matches += matched_classes
        exact_counts += int(len(first["events"]) == len(second["events"]))
        unresolved += int(final["case_status"] == "ONTOLOGY_UNRESOLVED")
        nonline += int(final["case_status"] == "NONLINE_EVENT_CONFLICT")
        for left in range(len(final["events"])):
            for right in range(left + 1, len(final["events"])):
                metrics = pair_metrics(
                    [final["events"][left]["line_start_mm"], final["events"][left]["line_end_mm"]],
                    [final["events"][right]["line_start_mm"], final["events"][right]["line_end_mm"]],
                )
                duplicates += int(is_agreement(metrics))
        visual_passed = raw.get("native_visual_passed") is True
        _hex64(raw.get("native_visual_audit_sha256"), "native_visual_audit_sha256")
        justification = raw.get("charter_rule_justification_sha256")
        if justification is not None:
            _hex64(justification, "charter_rule_justification_sha256")
            written_justifications += 1
        visual_passes += int(visual_passed)
        latencies = raw.get("export_latency_seconds")
        if not isinstance(latencies, list) or len(latencies) != 6:
            raise QualificationInputError("each case requires six primary/replay export latency measurements")
        export_latencies.extend(_finite_nonnegative(value, "export_latency_seconds") for value in latencies)
        replay_ok = all(raw.get(key) is True for key in (
            "annotation_a_replay_identical", "annotation_b_replay_identical", "adjudicated_replay_identical",
        ))
        replay_failures += int(not replay_ok)
        case_peak = raw.get("peak_rss_bytes")
        if isinstance(case_peak, bool) or not isinstance(case_peak, int) or case_peak < 0:
            raise QualificationInputError("peak_rss_bytes must be a non-negative integer")
        peak_rss_bytes = max(peak_rss_bytes, case_peak)
        succeeded = raw.get("jobs_succeeded")
        attempted = raw.get("jobs_attempted")
        if isinstance(succeeded, bool) or isinstance(attempted, bool) or not isinstance(succeeded, int) or not isinstance(attempted, int) or attempted != 6 or not 0 <= succeeded <= attempted:
            raise QualificationInputError("each case requires exactly six annotation export job attempts")
        job_successes += succeeded
        rows.append({
            "case_id": case_id,
            "annotation_a_count": len(first["events"]),
            "annotation_b_count": len(second["events"]),
            "matched_lines": agreement["true_positive"],
            "exact_count": len(first["events"]) == len(second["events"]),
            "class_matches": matched_classes,
            "native_visual_passed": visual_passed,
            "deterministic_replay": replay_ok,
            "final_status": final["case_status"],
        })
    hash_manifest_verified = payload.get("hash_manifest_verified") is True
    prohibited = payload.get("leakage_audit_prohibited_field_or_asset_count")
    if isinstance(prohibited, bool) or not isinstance(prohibited, int) or prohibited < 0:
        raise QualificationInputError("leakage audit count must be a non-negative integer")
    precision = total_matches / total_second if total_second else (1.0 if total_first == 0 else 0.0)
    recall = total_matches / total_first if total_first else (1.0 if total_second == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    class_agreement = class_matches / total_matches if total_matches else 0.0
    metrics = {
        "exact_event_count_cases": exact_counts,
        "case_count": 13,
        "inter_annotator_precision": precision,
        "inter_annotator_recall": recall,
        "inter_annotator_f1": f1,
        "median_transverse_disagreement_mm": statistics.median(offsets) if offsets else None,
        "p95_transverse_disagreement_mm": _nearest_rank(offsets, 0.95),
        "p95_angle_disagreement_deg": _nearest_rank(angles, 0.95),
        "exact_class_code_agreement": class_agreement,
        "post_adjudication_unresolved_cases": unresolved,
        "post_adjudication_nonline_conflicts": nonline,
        "final_duplicate_event_identities": duplicates,
        "native_visual_passed_cases": visual_passes,
        "written_charter_justification_cases": written_justifications,
        "replay_failures": replay_failures,
        "annotation_export_p95_seconds": _nearest_rank(export_latencies, 0.95),
        "annotation_tool_job_success_rate": job_successes / 78,
        "peak_rss_bytes": peak_rss_bytes,
        "hash_manifest_verified": hash_manifest_verified,
        "leakage_audit_prohibited_field_or_asset_count": prohibited,
    }
    gates = {
        "exact_event_count": exact_counts >= 12,
        "line_precision": precision >= 0.90,
        "line_recall": recall >= 0.90,
        "line_f1": f1 >= 0.90,
        "median_transverse": metrics["median_transverse_disagreement_mm"] is not None and metrics["median_transverse_disagreement_mm"] <= 1.5,
        "p95_transverse": metrics["p95_transverse_disagreement_mm"] is not None and metrics["p95_transverse_disagreement_mm"] <= 3.0,
        "p95_angle": metrics["p95_angle_disagreement_deg"] is not None and metrics["p95_angle_disagreement_deg"] <= 2.0,
        "class_code": class_agreement >= 0.90,
        "no_unresolved": unresolved == 0,
        "no_nonline_conflicts": nonline == 0,
        "no_duplicate_event_identities": duplicates == 0,
        "native_visual": visual_passes == 13,
        "three_required_written_justifications": written_justifications == 3,
        "deterministic_replay": replay_failures == 0,
        "export_latency": metrics["annotation_export_p95_seconds"] is not None and metrics["annotation_export_p95_seconds"] <= 5.0,
        "memory": peak_rss_bytes < MAX_TOOL_RSS_BYTES,
        "job_success": job_successes == 78,
        "hash_manifest": hash_manifest_verified,
        "leakage_audit": prohibited == 0,
    }
    return {
        "schema_version": 1,
        "kind": "CIBEC_V1_PHYSICAL_ANNOTATION_QUALIFICATION",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "cases": rows,
    }
