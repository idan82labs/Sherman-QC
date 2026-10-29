"""Independent CAD primitive-grouping agreement for CIBEC_V1."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Mapping

from research.cibec_v1.contract import validate_cad_correspondence


SCHEMA = "CIBEC_V1_CAD_GROUPING_QUALIFICATION_INPUT"
MAX_TOOL_RSS_BYTES = 2 * 1024**3


class CadGroupingInputError(ValueError):
    pass


def _positive_pairs(mapping: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    primitives: set[str] = set()
    positives: set[tuple[str, str]] = set()
    for event in mapping["events"]:
        group = sorted(set(event["v9_primitive_ids"]))
        if primitives.intersection(group):
            raise CadGroupingInputError("duplicate V9 primitive ownership")
        primitives.update(group)
        positives.update(combinations(group, 2))
    return primitives, positives


def _finite_nonnegative(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise CadGroupingInputError(f"{path} must be finite and non-negative")
    return float(value)


def score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise CadGroupingInputError(f"schema must be {SCHEMA}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CadGroupingInputError("cases must be non-empty")
    reference_positive: set[tuple[str, str, str]] = set()
    candidate_positive: set[tuple[str, str, str]] = set()
    all_pairs: set[tuple[str, str, str]] = set()
    unresolved = 0
    replay_failures = 0
    latencies: list[float] = []
    jobs_attempted = jobs_succeeded = 0
    trusted = abstained = 0
    peak_rss_bytes = 0
    seen = set()
    rows = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("case_id"), str):
            raise CadGroupingInputError(f"cases[{index}] is invalid")
        case_id = raw["case_id"]
        if case_id in seen:
            raise CadGroupingInputError("duplicate case_id")
        seen.add(case_id)
        first = validate_cad_correspondence(raw.get("mapper_a"))
        second = validate_cad_correspondence(raw.get("mapper_b"))
        if any(first[key] != second[key] for key in ("primary_truth_sha256", "step_sha256", "overlay_transform_sha256", "overlay_status")):
            raise CadGroupingInputError(f"{case_id}: mapper bindings differ")
        replay_ok = raw.get("mapper_a_replay_identical") is True and raw.get("mapper_b_replay_identical") is True
        replay_failures += int(not replay_ok)
        case_latencies = raw.get("load_export_latency_seconds")
        if not isinstance(case_latencies, list) or len(case_latencies) != 4:
            raise CadGroupingInputError("all cases require four primary/replay latency measurements")
        latencies.extend(_finite_nonnegative(value, "load_export_latency_seconds") for value in case_latencies)
        attempted = raw.get("jobs_attempted")
        succeeded = raw.get("jobs_succeeded")
        if isinstance(attempted, bool) or isinstance(succeeded, bool) or attempted != 4 or not isinstance(succeeded, int) or not 0 <= succeeded <= 4:
            raise CadGroupingInputError("all cases require exactly four job attempts")
        jobs_attempted += attempted
        jobs_succeeded += succeeded
        case_peak = raw.get("peak_rss_bytes")
        if isinstance(case_peak, bool) or not isinstance(case_peak, int) or case_peak < 0:
            raise CadGroupingInputError("peak_rss_bytes must be a non-negative integer")
        peak_rss_bytes = max(peak_rss_bytes, case_peak)
        if first["overlay_status"] == "ABSTAIN":
            abstained += 1
            if raw.get("registration_abstention_excluded") is not True:
                raise CadGroupingInputError(f"{case_id}: abstention must be explicitly excluded")
            rows.append({"case_id": case_id, "overlay_status": "ABSTAIN", "excluded": True, "replay_identical": replay_ok})
            continue
        trusted += 1
        native = raw.get("native_v9_primitive_ids")
        if (
            not isinstance(native, list)
            or not native
            or any(not isinstance(value, str) or not value for value in native)
            or len(set(native)) != len(native)
        ):
            raise CadGroupingInputError(f"{case_id}: native_v9_primitive_ids must be a non-empty unique string array")
        native_primitives = set(native)
        first_primitives, first_positive = _positive_pairs(first)
        second_primitives, second_positive = _positive_pairs(second)
        unknown = (first_primitives | second_primitives) - native_primitives
        if unknown:
            raise CadGroupingInputError(f"{case_id}: correspondence contains primitive outside the native inventory")
        universe = sorted(native_primitives)
        all_pairs.update((case_id, *pair) for pair in combinations(universe, 2))
        reference_positive.update((case_id, *pair) for pair in first_positive)
        candidate_positive.update((case_id, *pair) for pair in second_positive)
        unresolved += sum(event["mapping_status"] != "RESOLVED" for event in first["events"] + second["events"])
        rows.append({
            "case_id": case_id,
            "overlay_status": "TRUSTED",
            "primitive_count": len(universe),
            "mapper_a_positive_pairs": len(first_positive),
            "mapper_b_positive_pairs": len(second_positive),
            "replay_identical": replay_ok,
        })
    hash_manifest_verified = payload.get("hash_manifest_verified") is True
    prohibited = payload.get("leakage_audit_prohibited_field_or_asset_count")
    if isinstance(prohibited, bool) or not isinstance(prohibited, int) or prohibited < 0:
        raise CadGroupingInputError("leakage audit count must be a non-negative integer")
    true_positive = len(reference_positive & candidate_positive)
    false_positive = len(candidate_positive - reference_positive)
    false_negative = len(reference_positive - candidate_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else (1.0 if not reference_positive else 0.0)
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else (1.0 if not candidate_positive else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p95_latency = sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None
    metrics = {
        "case_count": len(cases),
        "trusted_case_count": trusted,
        "registration_abstention_count": abstained,
        "pair_universe_count": len(all_pairs),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "grouping_precision": precision,
        "grouping_recall": recall,
        "grouping_f1": f1,
        "unresolved_mapping_records": unresolved,
        "replay_failures": replay_failures,
        "p95_load_export_seconds": p95_latency,
        "job_success_rate": jobs_succeeded / jobs_attempted if jobs_attempted else None,
        "peak_rss_bytes": peak_rss_bytes,
        "hash_manifest_verified": hash_manifest_verified,
        "leakage_audit_prohibited_field_or_asset_count": prohibited,
    }
    gates = {
        "complete_13_case_denominator": len(cases) == 13,
        "exact_registration_abstention": trusted == 12 and abstained == 1,
        "grouping_precision": precision >= 0.90,
        "grouping_recall": recall >= 0.90,
        "grouping_f1": f1 >= 0.90,
        "no_unresolved_mapping": unresolved == 0,
        "deterministic_replay": replay_failures == 0,
        "latency": p95_latency is not None and p95_latency <= 30.0,
        "memory": peak_rss_bytes < MAX_TOOL_RSS_BYTES,
        "job_success": jobs_attempted > 0 and jobs_succeeded == jobs_attempted,
        "hash_manifest": hash_manifest_verified,
        "leakage_audit": prohibited == 0,
    }
    return {
        "schema_version": 1,
        "kind": "CIBEC_V1_CAD_GROUPING_QUALIFICATION",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "cases": rows,
    }
