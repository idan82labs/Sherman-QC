"""Pure reconstruction of four legacy cached-line variants, never inspection.

No I/O, Gold, scoring, inference, pose/unit estimation or historical execution
attestation. Callers freeze/authenticate the complete eleven case IDs and cached
payload/source bytes. Cohort cardinality is not byte authentication. The original
fusion duplicate rule uses SHORTER-span overlap, not the scorer's BOTH-span gate.

Source-only authoring scope, 2026-09-10: field NAMES/TYPES only were projected
from the exact eleven A0 files bound in gold11_a0_provenance_trace_v7 and eleven
gated-CAD files bound in gold11_retained_control_recovery. CAD hashes were checked
again; A0 hashes had matched V7. A0 uses span_endpoints and auxiliary direction,
span/ownership/semantic fields; CAD uses line_start/end and legacy assignment,
measurement/observability metadata. No endpoint values or geometry were examined.
This separate schema read did not reconstruct or score the physical cache.

_line, _duplicate and _normalized_a0 preserve the original function AST from
diagnose_f1_cad_scan_fusion_gold11.py. _original_variants preserves its six
reconstruction statements, including the rows_by_variant assignment.
Only the callback binding adds counting around the unchanged duplicate function.
Admissibility/snapshot/resource failures are new explicit whole-case refusals;
on admitted input there is no geometry repair, sorting, ID dedup or extra filter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .evaluate_f1_against_manual_gold import pair_metrics


CASE_COUNT = 11
MAX_INPUT_ROWS = 4096
MAX_PAIR_CHECKS_PER_CASE = 2_000_000
MAX_JSON_NODES = 131_072
MAX_JSON_STRING_CHARS = 4 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_STRING_LENGTH = 8192
MAX_COORDINATE_MAGNITUDE = 1e6
VARIANTS = ("scan_only_a0", "cad_only_gated", "intersection", "union")
RAW_FRAME = "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"


# Exact original pure function ASTs; keep constants/precedence/order unchanged.
def _line(row: dict[str, Any]) -> np.ndarray:
    endpoints = row.get("span_endpoints") or [row["line_start"], row["line_end"]]
    return np.asarray(endpoints, dtype=np.float64)


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = pair_metrics(_line(left), _line(right))
    return (
        metrics["axis_angle_deg"] <= 5.0
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.60
    )


def _normalized_a0(row: dict[str, Any]) -> dict[str, Any]:
    endpoints = _line(row)
    return {
        "bend_id": str(row["bend_id"]),
        "line_start": endpoints[0].tolist(),
        "line_end": endpoints[1].tolist(),
        "fusion_source": "scan_only_a0",
    }


def _original_variants(a0_payload, cad_payload, _duplicate):
    scan_rows = [_normalized_a0(row) for row in a0_payload["owned_bend_regions"]]
    cad_rows = [dict(row) for row in cad_payload["predictions"]]
    intersection = [row for row in scan_rows if any(_duplicate(row, cad) for cad in cad_rows)]
    union = list(scan_rows)
    for cad in cad_rows:
        if not any(_duplicate(cad, existing) for existing in union):
            added = dict(cad)
            added["bend_id"] = "CADF_" + str(added["bend_id"])
            added["fusion_source"] = "cad_guided_gated_rescue"
            union.append(added)
    rows_by_variant = {
        "scan_only_a0": scan_rows,
        "cad_only_gated": cad_rows,
        "intersection": intersection,
        "union": union,
    }
    return rows_by_variant


class _Malformed(ValueError):
    pass


class _Budget(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Malformed(reason)


def _identifier(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 512


def _copy_json(value, budget, depth=0):
    """Detach bounded exact builtin JSON containers; no user copy hooks."""
    budget["nodes"] -= 1
    if budget["nodes"] < 0 or depth > MAX_JSON_DEPTH:
        raise _Budget("JSON_NODE_OR_DEPTH_BUDGET")
    kind = type(value)
    if value is None or kind is bool:
        return value
    if kind is str:
        budget["chars"] -= len(value)
        if len(value) > MAX_STRING_LENGTH or budget["chars"] < 0:
            raise _Budget("JSON_STRING_BUDGET")
        return value
    if kind is int:
        _require(value.bit_length() <= 4096, "UNSUPPORTED_METADATA_INTEGER")
        return value
    if kind is float:
        _require(math.isfinite(value), "NONFINITE_COPIED_JSON_NUMBER")
        return value
    if kind is list:
        if len(value) > budget["nodes"]:
            raise _Budget("JSON_NODE_OR_DEPTH_BUDGET")
        return [_copy_json(item, budget, depth + 1) for item in value]
    if kind is dict:
        if 2 * len(value) > budget["nodes"]:
            raise _Budget("JSON_NODE_OR_DEPTH_BUDGET")
        result = {}
        for key, item in value.items():
            _require(type(key) is str, "NONSTRING_JSON_KEY")
            copied_key = _copy_json(key, budget, depth + 1)
            result[copied_key] = _copy_json(item, budget, depth + 1)
        return result
    raise _Malformed("UNSUPPORTED_JSON_VALUE")


def _snapshot_budget(multiplier=1):
    return {"nodes": multiplier * MAX_JSON_NODES,
            "chars": multiplier * MAX_JSON_STRING_CHARS}


def _line_projection(row):
    """Validate only legacy-selected geometry; never reconcile alternate axes.

    Truthy span_endpoints takes precedence even when direct endpoints coexist.
    None/absent/empty-list span uses direct endpoints. Other auxiliary fields
    (including counted, semantic_label, axis_direction and span_mm) do not filter
    A0 rows. Direct fields ignored by span precedence are not repaired or used.
    Duplicate IDs, including CADF_ collisions, are preserved as legacy IDs, not
    asserted unique physical events. Nonstring IDs are outside this input schema.
    """
    _require(type(row) is dict and _identifier(row.get("bend_id")), "INVALID_LINE_ID")
    span = row.get("span_endpoints")
    _require(span is None or type(span) is list, "UNSUPPORTED_SPAN_CONTAINER")
    if span:
        _require(len(span) == 2, "INVALID_ENDPOINT_PAIR")
        selected = {"span_endpoints": span}
        endpoints = span
    else:
        endpoints = [row.get("line_start"), row.get("line_end")]
        selected = {"line_start": endpoints[0], "line_end": endpoints[1]}
    for point in endpoints:
        _require(type(point) is list and len(point) == 3, "INVALID_LINE_POINT")
        for value in point:
            _require(type(value) in (int, float), "INVALID_LINE_COORDINATE")
            try:
                number = float(value)
            except (ValueError, TypeError, OverflowError) as error:
                raise _Malformed("INVALID_LINE_COORDINATE") from error
            _require(math.isfinite(number) and abs(number) <= MAX_COORDINATE_MAGNITUDE,
                     "INVALID_LINE_COORDINATE")
    core = {"bend_id": row["bend_id"], **selected}
    line = _line(core)
    _require(float(np.linalg.norm(line[1] - line[0])) > 1e-9, "DEGENERATE_LINE")
    _require("unit" not in row or row["unit"] == "mm", "EXPLICIT_LINE_UNIT_CONFLICT")
    _require("reference_frame" not in row or row["reference_frame"] == RAW_FRAME,
             "EXPLICIT_LINE_FRAME_CONFLICT")
    return core


def _prepare(a0, cad, case_id, work):
    _require(type(a0) is dict and type(cad) is dict, "CACHED_PAYLOAD_OBJECTS_REQUIRED")
    for payload in (a0, cad):
        _require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
                 "UNSUPPORTED_SCHEMA_VERSION")
        _require("case_id" not in payload or payload["case_id"] == case_id,
                 "EMBEDDED_CASE_IDENTITY_CONFLICT")
    _require(a0.get("kind") == "sherman_qc_f1_typed_union_counted_components"
             and a0.get("algorithm") == "f1_typed_union_exact_birth_v1"
             and a0.get("coordinate_frame") == "scan_raw", "UNSUPPORTED_A0_SCHEMA")
    _require(cad.get("kind") == "sherman_qc_f1_cad_guided_scan_line_predictions"
             and cad.get("route") == "blind_cad_ply_detected_lines",
             "UNSUPPORTED_CAD_SCHEMA")
    _require(cad.get("research_only") is True and cad.get("cad_guided_zero_shot") is True
             and cad.get("source_only") is False and "status" not in cad,
             "UNSUPPORTED_CAD_RESEARCH_SCHEMA")
    a_rows, c_rows = a0.get("owned_bend_regions"), cad.get("predictions")
    _require(type(a_rows) is list and type(c_rows) is list, "INVALID_ROW_CONTAINERS")
    a, c = len(a_rows), len(c_rows)
    work.update(a0_input_rows=a, cad_input_rows=c)
    if max(a, c) > MAX_INPUT_ROWS:
        raise _Budget("INPUT_ROW_BUDGET")
    # Intersection <= A*C. Sequential union <= A*C + C*(C-1)/2.
    bound = 2 * a * c + c * (c - 1) // 2
    work["pair_checks_preflight_upper_bound"] = bound
    if bound > MAX_PAIR_CHECKS_PER_CASE:
        raise _Budget("PAIR_CHECK_PREFLIGHT_BUDGET")
    budget = _snapshot_budget()
    # Normalized A0 ignores auxiliary metadata in the original source. Snapshot
    # only its selected representation; CAD copies all row metadata originally.
    owned_a = [_copy_json(_line_projection(row), budget) for row in a_rows]
    owned_c = []
    for row in c_rows:
        _line_projection(row)
        owned_c.append(_copy_json(row, budget))
    _require(len(owned_a) == a and len(owned_c) == c,
             "INPUT_ROW_COUNT_CHANGED_DURING_SNAPSHOT")
    # Revalidate the OWNED snapshots before arithmetic; caller mutation cannot
    # turn an early validation into admission of a changed endpoint value.
    for row in owned_a + owned_c:
        _line_projection(row)
    return {"owned_bend_regions": owned_a}, {"predictions": owned_c}


def _case(a0, cad, case_id):
    work = {"a0_input_rows": None, "cad_input_rows": None,
            "pair_checks_preflight_upper_bound": None, "pair_checks_attempted": 0}
    status, reason = "COMPLETE", None
    variants = {name: [] for name in VARIANTS}
    try:
        owned_a, owned_c = _prepare(a0, cad, case_id, work)

        def counted_duplicate(left, right):
            work["pair_checks_attempted"] += 1
            if work["pair_checks_attempted"] > work["pair_checks_preflight_upper_bound"]:
                raise _Budget("PAIR_CHECK_ACCOUNTING_BOUND_EXCEEDED")
            return _duplicate(left, right)

        reconstructed = _original_variants(owned_a, owned_c, counted_duplicate)
        # Detach variants from one another as well as from caller-owned inputs.
        variants = _copy_json(reconstructed, _snapshot_budget(4))
    except _Budget as error:
        status, reason = "RESOURCE_REFUSAL", str(error)
    except _Malformed as error:
        status, reason = "MALFORMED_RESULT", str(error)
    except Exception:
        status, reason = "RECONSTRUCTION_FAILURE", "UNEXPECTED_RECONSTRUCTION_ERROR"
    if status != "COMPLETE":
        variants = {name: [] for name in VARIANTS}
    return {
        "case_id": case_id, "status": status, "reason": reason,
        "usable_variants": status == "COMPLETE", "variants": variants,
        "cached_line_counts": {name: len(rows) if status == "COMPLETE" else None
                               for name, rows in variants.items()},
        "work": work, "count_semantics": "CACHED_LINE_CARDINALITY_NOT_OBSERVED_BENDS",
        "pair_counter_semantics": "ATTEMPTED_CALLBACKS_INCLUDING_FAILED_COMPARISON",
        "observed_bend_count": None, "production_use_allowed": False,
    }


def reconstruct_gold11_retained_fusion_v1(*, a0_results_by_case, cad_results_by_case, case_ids):
    """Reconstruct every caller-ordered case or retain its whole-case refusal.

    Bad cohort declarations raise ValueError. Malformed input, numeric failure
    or resource refusal clears ALL four variants for that case, retaining the
    case/reason/work evidence. Counts are None on refused cases, not absence.
    No condition asserts 33 A0 / 44 CAD / 60 union rows. Original duplicate-ID
    behavior is preserved; geometry/ID validation by a later scorer is separate.

    Limits are fixed new reconstruction safety bounds, not old-runtime gates.
    Preflight reserves 2*A*C+C*(C-1)/2 comparisons before snapshots/fusion; short
    circuiting can use fewer. Each case has <=4096 rows per input, <=2M checks,
    bounded JSON snapshots and output. Ordinary failures never publish a prefix.
    """
    _require(type(case_ids) in (list, tuple) and len(case_ids) == CASE_COUNT,
             "REQUIRE_ELEVEN_CALLER_CASE_IDS")
    ordered = tuple(case_ids)
    _require(len(ordered) == CASE_COUNT and all(_identifier(case) for case in ordered),
             "INVALID_CASE_ID")
    _require(len(set(ordered)) == CASE_COUNT, "DUPLICATE_CASE_ID")
    owned_maps = []
    for mapping in (a0_results_by_case, cad_results_by_case):
        _require(type(mapping) is dict, "CASE_SET_MISMATCH")
        owned_map = dict(mapping)
        _require(set(owned_map) == set(ordered), "CASE_SET_MISMATCH")
        owned_maps.append(owned_map)
    cases = {case: _case(owned_maps[0][case], owned_maps[1][case], case)
             for case in ordered}
    failed = sum(row["status"] != "COMPLETE" for row in cases.values())
    return {
        "schema": "GOLD11_RETAINED_FUSION_RECONSTRUCTION_V1",
        "status": "COMPLETE" if not failed else "COMPLETE_WITH_CASE_REFUSALS",
        "case_order": list(ordered), "case_count": CASE_COUNT, "cases": cases,
        "failed_case_count": failed, "all_cases_usable": not failed,
        "context": "CACHED_A0_AND_GATED_CAD_RECONSTRUCTION_NOT_FRESH_INFERENCE",
        "variant_contexts": {"scan_only_a0": "CACHED_SCAN_A0_LINES",
                             "cad_only_gated": "CACHED_SCAN_FRAME_GATED_CAD_LINES",
                             "intersection": "CACHED_A0_LINES_CORROBORATED_BY_GATED_CAD",
                             "union": "CACHED_A0_THEN_NONDUPLICATE_GATED_CAD_LINES"},
        "duplicate_rule": {"axis_angle_deg_max": 5., "transverse_offset_mm_max": 4.,
                           "shorter_span_overlap_min": .60,
                           "order": "ALL_A0_FIRST_THEN_INPUT_ORDER_CAD_GREEDY_APPEND",
                           "cad_added_id_prefix": "CADF_"},
        "pair_check_cap_per_case": MAX_PAIR_CHECKS_PER_CASE,
        "pair_bound_formula": "2*A*C+C*(C-1)/2",
        "input_row_cap_per_case_per_source": MAX_INPUT_ROWS,
        "failure_policy": "ALL_VARIANTS_CLEARED_WHOLE_CASE_RETAINED_NO_PREFIX",
        "cohort_authentication": "CALLER_FROZEN_CASE_AND_PAYLOAD_BYTES_REQUIRED",
        "historical_execution_attestation": "NOT_ESTABLISHED_BY_RECONSTRUCTION",
        "historical_score_comparability": "UNESTABLISHED",
        "units_status": "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        "carrier_semantics": "LEGACY_PREDICTED_LINES_NOT_ASSERTED_MEASURED_SURFACE_CARRIERS",
        "inference_execution": "NOT_PERFORMED", "scoring": "NOT_PERFORMED",
        "observed_bend_count": None, "conformance": "INDETERMINATE",
        "production_use_allowed": False, "floor_release_allowed": False,
    }
