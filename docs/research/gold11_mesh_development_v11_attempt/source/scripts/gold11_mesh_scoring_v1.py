"""Pure post-inference Gold11 geometry diagnostics, never physical authority.

No file, raw-data, native, registration, inference or authorization access. The
caller freezes/authenticates the exact selected cohort/reference bytes before
calling. This helper validates its complete declared 11-case/six-family/71-line
denominator, not the historical origin of those declarations.
"""
import math

from .evaluate_f1_against_manual_gold import Thresholds, match_lines
from .f1_physical_evaluation_contract_v2 import score_localized_cases


CASE_COUNT = 11
FAMILY_COUNT = 6
REFERENCE_LINE_COUNT = 71
# Post-inference safety cap, not permission for native extraction/measurement to
# exceed their separate, stricter window/local-row budgets.
MAX_PREDICTIONS_PER_CASE = 4096
RAW_FRAME = "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
PROVISIONAL_UNITS = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
CARRIER = "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"


def _identifier(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 512


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _point(value):
    _require(type(value) in (list, tuple) and len(value) == 3, "INVALID_LINE_POINT")
    point = []
    for coordinate in value:
        _require(type(coordinate) in (int, float), "INVALID_LINE_COORDINATE")
        try:
            number = float(coordinate)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("INVALID_LINE_COORDINATE") from error
        _require(math.isfinite(number) and abs(number) <= 1e6, "INVALID_LINE_COORDINATE")
        point.append(number)
    return point


def _lines(records, *, prediction):
    limit = MAX_PREDICTIONS_PER_CASE if prediction else REFERENCE_LINE_COUNT
    _require(type(records) is list and len(records) <= limit, "INVALID_LINE_CONTAINER_OR_BUDGET")
    if not prediction:
        _require(bool(records), "REFERENCE_CASE_MUST_HAVE_LINES")
    normalized, identifiers = [], set()
    for row in records:
        _require(type(row) is dict and _identifier(row.get("bend_id")), "INVALID_LINE_ID")
        _require(row["bend_id"] not in identifiers, "DUPLICATE_LINE_ID")
        identifiers.add(row["bend_id"])
        # The frozen workflow and reviewed Gold use direct endpoints. Reject
        # ambiguous alternate carriers rather than trigger legacy precedence.
        _require(not any(key in row for key in ("span_endpoints", "anchor", "axis_direction", "span_mm")),
                 "ALTERNATE_LINE_REPRESENTATION_NOT_ADMITTED")
        start, end = _point(row.get("line_start")), _point(row.get("line_end"))
        _require(math.hypot(*(b - a for a, b in zip(start, end))) > 1e-9, "DEGENERATE_LINE")
        _require("unit" not in row or row["unit"] == "mm", "EXPLICIT_LINE_UNIT_CONFLICT")
        _require("reference_frame" not in row or row["reference_frame"] == RAW_FRAME,
                 "EXPLICIT_LINE_FRAME_CONFLICT")
        if prediction:
            _require(row.get("unit") == "mm" and row.get("reference_frame") == RAW_FRAME,
                     "PREDICTION_FRAME_OR_UNIT_CONTRACT")
            _require(row.get("line_convention") == CARRIER, "PREDICTION_CARRIER_CONTRACT")
        normalized.append({"bend_id": row["bend_id"], "line_start": start, "line_end": end})
    return normalized


def _case_predictions(payload, case_id, family_id):
    """Return the whole admissible case or zero rows, never a valid prefix."""
    if type(payload) is not dict:
        return [], "MALFORMED_RESULT", "WORKFLOW_RESULT_REQUIRED"
    if payload.get("status") == "UNAVAILABLE":
        if type(payload.get("predictions")) is list and not payload["predictions"]:
            return [], "REFUSED", "EXPLICIT_WORKFLOW_UNAVAILABLE"
        return [], "MALFORMED_RESULT", "UNAVAILABLE_RESULT_HAS_MISSING_OR_NONEMPTY_PREDICTIONS"
    try:
        _require(payload.get("status") == "PARTIAL" and payload.get("phase") == "COMPLETE",
                 "COMPLETED_CONDITIONAL_WORKFLOW_REQUIRED")
        _require("observed_bend_count" in payload and payload["observed_bend_count"] is None,
                 "PHYSICAL_BEND_COUNT_CLAIM_NOT_SUPPORTED")
        _require(payload.get("production_use_allowed") is False
                 and payload.get("conformance") == "INDETERMINATE"
                 and payload.get("uncertainty_status") == "UNCALIBRATED",
                 "RESEARCH_ONLY_WORKFLOW_CONTRACT")
        _require(payload.get("prediction_frame") == RAW_FRAME
                 and payload.get("units_status") == PROVISIONAL_UNITS
                 and payload.get("reference_frame_compatibility") == "NOT_ASSERTED_BY_INFERENCE",
                 "PROVISIONAL_FRAME_UNIT_CONTRACT")
        for key, expected in (("case_id", case_id), ("family_id", family_id)):
            _require(key not in payload or payload[key] == expected, "EMBEDDED_CASE_IDENTITY_CONFLICT")
        predictions = _lines(payload.get("predictions"), prediction=True)
        _require(bool(predictions), "PARTIAL_RESULT_REQUIRES_EMITTED_PREDICTIONS")
        return predictions, "SCORED_CONDITIONAL", None
    except ValueError as error:
        return [], "MALFORMED_RESULT", str(error)


def _summary(rows):
    # Do not reimplement the old scorer's undefined precision/F1 conventions.
    raw = score_localized_cases(rows)
    return {
        "case_count": raw["case_count"],
        "reference_line_count": sum(row["gold_count"] for row in rows),
        "emitted_support_line_count": sum(row["prediction_count"] for row in rows),
        **{key: raw[key] for key in ("true_positive", "false_positive", "false_negative",
                                    "precision", "recall", "f1",
                                    "exact_localized_target_set_cases",
                                    "exact_localized_target_set_accuracy")},
        "exact_emitted_support_count_cases": raw["exact_count_cases"],
        "exact_emitted_support_count_accuracy": raw["exact_count_accuracy"],
        "count_metric_semantics": "EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS",
    }


def _workflow_diagnostics(payload):
    if type(payload) is not dict:
        return {"reported_workflow_phase": None, "reported_workflow_reasons": []}
    phase = payload.get("phase")
    if type(phase) is not str or len(phase) > 128:
        phase = None
    reasons = payload.get("reasons", [])
    if (type(reasons) is not list or len(reasons) > 32
            or not all(type(reason) is str and len(reason) <= 256 for reason in reasons)):
        reasons = ["MALFORMED_OR_OVERSIZED_REPORTED_REASONS_OMITTED"]
    return {"reported_workflow_phase": phase, "reported_workflow_reasons": list(reasons)}


def _lane(results, references, families):
    rows, dispositions = [], {}
    thresholds = Thresholds(angle_deg=5., transverse_offset_mm=4.,
                            gold_span_overlap=.60, prediction_span_overlap=.60)
    for case_id in sorted(families):
        predictions, disposition, reason = _case_predictions(results[case_id], case_id, families[case_id])
        # Unexpected matcher errors propagate as scoring failures, not invented
        # engine failures. All malformed/refused workflow results already have
        # an explicit, frozen zero-prediction policy before matching.
        matched = match_lines(references[case_id], predictions, thresholds)
        row = {**matched, "case_id": case_id, "family_id": families[case_id]}
        rows.append(row)
        singleton = score_localized_cases([row])
        dispositions[case_id] = {"disposition": disposition, "reason": reason,
                                 "family_id": families[case_id], **matched,
                                 **{key: singleton[key] for key in ("precision", "recall", "f1")},
                                 "exact_emitted_support_count": singleton["exact_count_cases"] == 1,
                                 "count_metric_semantics": "EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS",
                                 "observed_bend_count": None,
                                 **_workflow_diagnostics(results[case_id])}
    family_scores = {family: _summary([row for row in rows if row["family_id"] == family])
                     for family in sorted(set(families.values()))}
    return {
        "geometry": _summary(rows), "families": family_scores,
        "minimum_family_recall": min(score["recall"] for score in family_scores.values()),
        "case_dispositions": dispositions,
        "refused_case_count": sum(row["disposition"] == "REFUSED" for row in dispositions.values()),
        "malformed_case_count": sum(row["disposition"] == "MALFORMED_RESULT" for row in dispositions.values()),
        "physical_observed_bend_count": {
            "known_case_count": 0, "unknown_case_count": CASE_COUNT, "unknown_rate": 1.,
            "strict_count_target_met": False, "status": "UNMET_NO_PHYSICAL_OBSERVED_COUNTS",
        },
    }


def _delta(control, challenger):
    return None if control is None or challenger is None else challenger - control


def score_gold11_mesh_workflows_v1(*, reference_records_by_case, control_results_by_case,
                                  challenger_results_by_case, case_family_by_id):
    """Score two fully declared development arms, with no inference or fitting.

    References are per-case lists of unchanged direct-endpoint bend records.
    Complete-map/reference failures raise ValueError. Malformed or unavailable
    workflow cases retain all reference false negatives and zero predictions;
    one bad prediction invalidates that entire case, not only the bad row.

    Coordinates are compared as supplied under a *provisional* common-mm/raw-
    frame assumption. No pose, unit, datum or carrier choice is optimized using
    Gold. The fixed measured surface carrier is not claimed compatible with
    historical Gold, and the new control is not a historical incumbent replay.
    """
    _require(type(case_family_by_id) is dict and len(case_family_by_id) == CASE_COUNT,
             "REQUIRE_COMPLETE_ELEVEN_CASE_FAMILY_MAP")
    _require(all(_identifier(case) and _identifier(family) for case, family in case_family_by_id.items()),
             "INVALID_CASE_OR_FAMILY_ID")
    _require(len(set(case_family_by_id.values())) == FAMILY_COUNT, "REQUIRE_SIX_DECLARED_FAMILIES")
    expected = set(case_family_by_id)
    for mapping in (reference_records_by_case, control_results_by_case, challenger_results_by_case):
        _require(type(mapping) is dict and set(mapping) == expected, "CASE_SET_MISMATCH")
    references = {case: _lines(reference_records_by_case[case], prediction=False) for case in sorted(expected)}
    _require(sum(map(len, references.values())) == REFERENCE_LINE_COUNT, "REQUIRE_SEVENTY_ONE_REFERENCE_LINES")
    control = _lane(control_results_by_case, references, case_family_by_id)
    challenger = _lane(challenger_results_by_case, references, case_family_by_id)
    delta_keys = ("true_positive", "false_positive", "false_negative", "precision", "recall", "f1",
                  "exact_emitted_support_count_accuracy", "exact_localized_target_set_accuracy")
    return {
        "schema": "GOLD11_MESH_GEOMETRIC_DEVELOPMENT_COMPARISON_V1",
        "status": "COMPUTED_DEVELOPMENT_DIAGNOSTIC", "case_count": CASE_COUNT,
        "family_count": FAMILY_COUNT, "reference_line_count": REFERENCE_LINE_COUNT,
        "cohort_authentication": "CALLER_FROZEN_CASE_IDS_AND_REFERENCE_BYTES_REQUIRED",
        "control_identity": "NEW_COMMON_WRAPPER_CONTROL_NOT_HISTORICAL_INCUMBENT",
        "historical_score_comparability": "UNESTABLISHED",
        "comparison_scope": "PROVISIONAL_COMMON_MM_RAW_ENDPOINT_GEOMETRY_ONLY",
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
        "control": control, "challenger": challenger,
        "paired_deltas": {**{key: _delta(control["geometry"][key], challenger["geometry"][key])
                               for key in delta_keys},
                          "minimum_family_recall": _delta(control["minimum_family_recall"],
                                                           challenger["minimum_family_recall"])},
        "family_paired_deltas": {
            family: {key: _delta(control["families"][family][key], challenger["families"][family][key])
                     for key in delta_keys} for family in control["families"]},
        "observed_bend_count": None, "strict_physical_engine_targets_met": False,
        "per_bend_measurement_accuracy": "NOT_EVALUATED_NO_MEASUREMENT_TRUTH",
        "production_use_allowed": False, "floor_release_allowed": False,
        "conformance": "INDETERMINATE", "uncertainty_status": "UNCALIBRATED",
    }
