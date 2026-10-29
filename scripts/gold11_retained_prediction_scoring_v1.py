"""Pure cached-CAD versus fresh-engine geometry scoring, never QC authority.

No I/O, inference, transform, filtering, calibration, or historical replay claim.
The caller authenticates/freezes cohort, payload, reference and source bytes and
keeps Gold outside inference. Cardinality checks do not authenticate those bytes.
Cached CAD lines are not relabelled as measured surface carriers or fresh output.
"""
import math

from .evaluate_f1_against_manual_gold import Thresholds, match_lines
from .f1_physical_evaluation_contract_v2 import score_localized_cases
from .gold11_mesh_scoring_v1 import (
    CASE_COUNT, FAMILY_COUNT, REFERENCE_LINE_COUNT, MAX_PREDICTIONS_PER_CASE,
    RAW_FRAME, PROVISIONAL_UNITS, _identifier, _require, _point, _lines,
    _lane, _delta,
)


CACHED_CONTEXT = "CACHED_SCAN_FRAME_GATED_CAD_NOT_FRESH_INFERENCE"
CACHED_KIND = "sherman_qc_f1_cad_guided_scan_line_predictions"
CACHED_ROUTE = "blind_cad_ply_detected_lines"
ALTERNATE_LINE_KEYS = frozenset({
    "span_endpoints", "anchor", "axis_direction", "span_mm", "endpoints",
    "start", "end", "p0", "p1", "lineStart", "lineEnd",
})
METRIC_KEYS = ("true_positive", "false_positive", "false_negative",
               "precision", "recall", "f1",
               "exact_localized_target_set_cases", "exact_localized_target_set_accuracy")


def _cached_predictions(payload, case_id, family_id):
    """Validate the whole retained schema; one bad row invalidates the case.

    Direct line_start/end only: aliases are rejected even if they agree. Absent
    units/frame are the legacy schema, not verified units/frame. Optional explicit
    declarations may not conflict. No line_convention is admitted: cached rows
    have no authenticated measured-carrier contract. Other legacy row metadata
    (including measurement_method/observability_state) never filters predictions
    or grants measurement, presence, absence, or conformance authority.
    """
    try:
        _require(type(payload) is dict, "CACHED_PAYLOAD_REQUIRED")
        _require(type(payload.get("schema_version")) is int
                 and payload["schema_version"] == 1, "CACHED_SCHEMA_VERSION_REQUIRED")
        _require(payload.get("kind") == CACHED_KIND and payload.get("route") == CACHED_ROUTE,
                 "CACHED_KIND_OR_ROUTE_CONFLICT")
        _require(payload.get("research_only") is True
                 and payload.get("cad_guided_zero_shot") is True
                 and payload.get("source_only") is False, "CACHED_RESEARCH_SCHEMA_REQUIRED")
        _require("status" not in payload, "WORKFLOW_STATUS_NOT_CACHED_SCHEMA")
        for key, expected in (("case_id", case_id), ("family_id", family_id)):
            _require(key not in payload or payload[key] == expected, "EMBEDDED_CASE_IDENTITY_CONFLICT")
        records = payload.get("predictions")
        _require(type(records) is list and len(records) <= MAX_PREDICTIONS_PER_CASE,
                 "INVALID_LINE_CONTAINER_OR_BUDGET")
        predictions, identifiers = [], set()
        for row in records:
            _require(type(row) is dict and _identifier(row.get("bend_id")), "INVALID_LINE_ID")
            _require(row["bend_id"] not in identifiers, "DUPLICATE_LINE_ID")
            identifiers.add(row["bend_id"])
            _require(not ALTERNATE_LINE_KEYS.intersection(row),
                     "ALTERNATE_LINE_REPRESENTATION_NOT_ADMITTED")
            start, end = _point(row.get("line_start")), _point(row.get("line_end"))
            _require(math.hypot(*(b - a for a, b in zip(start, end))) > 1e-9, "DEGENERATE_LINE")
            _require("unit" not in row or row["unit"] == "mm", "EXPLICIT_LINE_UNIT_CONFLICT")
            _require("reference_frame" not in row or row["reference_frame"] == RAW_FRAME,
                     "EXPLICIT_LINE_FRAME_CONFLICT")
            _require("line_convention" not in row, "CACHED_CARRIER_DECLARATION_NOT_ADMITTED")
            predictions.append({"bend_id": row["bend_id"], "line_start": start, "line_end": end})
        return predictions, "SCORED_CACHED_GEOMETRY", None
    except ValueError as error:
        return [], "MALFORMED_RESULT", str(error)


def _cached_summary(rows):
    raw = score_localized_cases(rows)
    return {
        "case_count": raw["case_count"],
        "reference_line_count": sum(row["gold_count"] for row in rows),
        "cached_prediction_line_count": sum(row["prediction_count"] for row in rows),
        **{key: raw[key] for key in METRIC_KEYS},
        "exact_cached_line_count_cases": raw["exact_count_cases"],
        "exact_cached_line_count_accuracy": raw["exact_count_accuracy"],
        "count_metric_semantics": "CACHED_LINE_CARDINALITY_NOT_PHYSICAL_OBSERVED_BENDS",
    }


def _cached_lane(results, references, families):
    rows, dispositions = [], {}
    thresholds = Thresholds(angle_deg=5., transverse_offset_mm=4.,
                            gold_span_overlap=.60, prediction_span_overlap=.60)
    for case_id in sorted(families):
        predictions, disposition, reason = _cached_predictions(results[case_id], case_id, families[case_id])
        # Unexpected matcher errors remain scoring failures, not invented empty
        # predictions. Only declared malformed-case policy supplies zero rows.
        matched = match_lines(references[case_id], predictions, thresholds)
        row = {**matched, "case_id": case_id, "family_id": families[case_id]}
        rows.append(row)
        singleton = score_localized_cases([row])
        dispositions[case_id] = {
            "disposition": disposition, "reason": reason, "family_id": families[case_id],
            **matched, **{key: singleton[key] for key in ("precision", "recall", "f1")},
            "exact_cached_line_count": singleton["exact_count_cases"] == 1,
            "count_metric_semantics": "CACHED_LINE_CARDINALITY_NOT_PHYSICAL_OBSERVED_BENDS",
            "observed_bend_count": None,
        }
    family_scores = {family: _cached_summary([row for row in rows if row["family_id"] == family])
                     for family in sorted(set(families.values()))}
    return {
        "lane_context": CACHED_CONTEXT,
        "carrier_semantics": "LEGACY_CAD_PREDICTED_LINE_NOT_ASSERTED_MEASURED_SURFACE_CARRIER",
        "inference_execution": "NOT_EVALUATED_RETAINED_PAYLOAD_ONLY",
        "geometry": _cached_summary(rows), "families": family_scores,
        "minimum_family_recall": min(score["recall"] for score in family_scores.values()),
        "case_dispositions": dispositions,
        "malformed_case_count": sum(row["disposition"] == "MALFORMED_RESULT" for row in dispositions.values()),
        "observed_bend_count": None,
    }


def score_gold11_retained_predictions_v1(*, reference_records_by_case, cached_results_by_case,
                                        engine_results_by_case, case_family_by_id):
    """Compare complete maps on 11 cases / six families / 71 reference lines.

    Bad cohort/reference declarations raise ValueError. A malformed cached case
    retains all its reference FNs with zero predictions, never a favorable prefix.
    Empty valid cached lists are allowed but do not establish observed absence.
    Modern engine eligibility, refusals and malformed-case behavior are delegated
    unchanged to the frozen V1 scorer. Undefined precision/F1 remain None.

    Endpoint values, IDs and within-case order are preserved after finite numeric
    validation; there is no pose/unit estimation, line fusion, crop or selection.
    Shared frame/mm are provisional caller declarations, not verified compatibility
    between legacy CAD lines, measured engine carriers and reference annotations.
    """
    _require(type(case_family_by_id) is dict and len(case_family_by_id) == CASE_COUNT,
             "REQUIRE_COMPLETE_ELEVEN_CASE_FAMILY_MAP")
    _require(all(_identifier(case) and _identifier(family) for case, family in case_family_by_id.items()),
             "INVALID_CASE_OR_FAMILY_ID")
    _require(len(set(case_family_by_id.values())) == FAMILY_COUNT, "REQUIRE_SIX_DECLARED_FAMILIES")
    expected = set(case_family_by_id)
    for mapping in (reference_records_by_case, cached_results_by_case, engine_results_by_case):
        _require(type(mapping) is dict and set(mapping) == expected, "CASE_SET_MISMATCH")
    references = {case: _lines(reference_records_by_case[case], prediction=False) for case in sorted(expected)}
    _require(sum(map(len, references.values())) == REFERENCE_LINE_COUNT, "REQUIRE_SEVENTY_ONE_REFERENCE_LINES")
    cached = _cached_lane(cached_results_by_case, references, case_family_by_id)
    engine = {"lane_context": "CAD_ENGINE", **_lane(engine_results_by_case, references, case_family_by_id)}
    return {
        "schema": "GOLD11_RETAINED_PREDICTION_GEOMETRY_COMPARISON_V1",
        "status": "COMPUTED_DEVELOPMENT_DIAGNOSTIC", "case_count": CASE_COUNT,
        "family_count": FAMILY_COUNT, "reference_line_count": REFERENCE_LINE_COUNT,
        "cohort_authentication": "CALLER_FROZEN_CASE_IDS_PAYLOAD_SOURCE_AND_REFERENCE_BYTES_REQUIRED",
        "control_identity": "NEW_CACHED_CAD_CONTROL_NOT_HISTORICAL_FUSION_REPLAY",
        "historical_score_comparability": "UNESTABLISHED",
        "comparison_scope": "PROVISIONAL_COMMON_MM_RAW_ENDPOINT_GEOMETRY_ONLY",
        "declared_common_frame": RAW_FRAME, "units_status": PROVISIONAL_UNITS,
        "reference_carrier_compatibility": "UNVERIFIED_NOT_ASSERTED_BY_SCORING",
        "reference_frame_compatibility": "PROVISIONAL_ASSUMPTION_NOT_VERIFIED",
        "scoring_operations": "DIRECT_ENDPOINT_MATCHING_ONLY_NO_TRANSFORM_OR_UNIT_ESTIMATION",
        "caller_inference_gold_isolation": "REQUIRES_INDEPENDENT_PROTOCOL_AND_SOURCE_AUDIT",
        "matching": {"axis_angle_deg_max": 5., "transverse_offset_mm_max": 4.,
                     "gold_span_overlap_min": .60, "prediction_span_overlap_min": .60,
                     "method": "UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST",
                     "per_case_exact_flag": "EXACT_LOCALIZED_REFERENCE_SET_NOT_PHYSICAL_COUNT"},
        "failure_policy": "WHOLE_CASE_ZERO_PREDICTIONS_ALL_REFERENCE_FNS_NO_PREFIX_OR_DROPPED_CASE",
        "prediction_safety_cap_per_case": MAX_PREDICTIONS_PER_CASE,
        "cached_control": cached, "engine": engine,
        "paired_deltas": {**{key: _delta(cached["geometry"][key], engine["geometry"][key])
                               for key in METRIC_KEYS},
                          "minimum_family_recall": _delta(cached["minimum_family_recall"],
                                                           engine["minimum_family_recall"])},
        "family_paired_deltas": {
            family: {key: _delta(cached["families"][family][key], engine["families"][family][key])
                     for key in METRIC_KEYS} for family in cached["families"]},
        "observed_bend_count": None, "strict_physical_engine_targets_met": False,
        "per_bend_measurement_accuracy": "NOT_EVALUATED_NO_MEASUREMENT_TRUTH",
        "production_use_allowed": False, "floor_release_allowed": False,
        "conformance": "INDETERMINATE", "uncertainty_status": "UNCALIBRATED",
    }
