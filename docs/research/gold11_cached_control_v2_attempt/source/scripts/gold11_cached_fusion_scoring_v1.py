"""Pure post-inference scoring of four reconstructed cached Gold11 variants.

No IO, inference, pose/unit estimation, reference changes or measured-carrier
claims. The caller authenticates/freezes exact cached payload and reference bytes.
Reconstruction preserves legacy ordering and shorter-span duplicate semantics;
scoring preserves the original BOTH-span one-to-one matcher. They are not the
same geometric decision. This is a NEW cached control, not historical replay.
"""

from .evaluate_f1_against_manual_gold import Thresholds, match_lines
from .gold11_retained_fusion_v1 import reconstruct_gold11_retained_fusion_v1, VARIANTS
from .gold11_retained_prediction_scoring_v1 import _cached_summary
from .gold11_mesh_scoring_v1 import (
    CASE_COUNT, FAMILY_COUNT, REFERENCE_LINE_COUNT, MAX_PREDICTIONS_PER_CASE,
    RAW_FRAME, PROVISIONAL_UNITS, _identifier, _require, _lines,
)


def _variant_lane(reconstruction, variant, references, families, identity_conflicts):
    thresholds = Thresholds(angle_deg=5., transverse_offset_mm=4.,
                            gold_span_overlap=.60, prediction_span_overlap=.60)
    rows, dispositions = [], {}
    for case_id in sorted(families):
        case = reconstruction["cases"][case_id]
        predictions = case["variants"][variant]
        disposition, reason = "SCORED_CACHED_GEOMETRY", None
        if case_id in identity_conflicts:
            predictions = []
            disposition, reason = "SCORING_INPUT_REFUSED", "EMBEDDED_FAMILY_IDENTITY_CONFLICT"
        elif case["status"] != "COMPLETE":
            predictions = []
            disposition, reason = "RECONSTRUCTION_REFUSED", case["reason"]
        elif len(predictions) > MAX_PREDICTIONS_PER_CASE:
            # Separate fixed scorer safety cap, not a favorable line prefix.
            predictions = []
            disposition, reason = "SCORING_INPUT_REFUSED", "PREDICTION_LINE_BUDGET"
        # Reconstructed rows have already had selected endpoint validation and
        # owned snapshots. Preserve original span precedence and duplicate IDs;
        # match_lines performs one-to-one assignment by ROW INDEX, not ID. Do
        # not fabricate cached-CAD payloads or modern measured-carrier contracts.
        matched = match_lines(references[case_id], predictions, thresholds)
        row = {**matched, "case_id": case_id, "family_id": families[case_id]}
        rows.append(row)
        summary = _cached_summary([row])
        dispositions[case_id] = {
            "disposition": disposition, "reason": reason,
            "reconstruction_status": case["status"],
            "family_id": families[case_id], **matched,
            **{key: summary[key] for key in ("precision", "recall", "f1")},
            "exact_cached_line_count": summary["exact_cached_line_count_cases"] == 1,
            "count_metric_semantics": "CACHED_LINE_CARDINALITY_NOT_PHYSICAL_OBSERVED_BENDS",
            "observed_bend_count": None,
        }
    family_scores = {family: _cached_summary([row for row in rows if row["family_id"] == family])
                     for family in sorted(set(families.values()))}
    return {
        "lane_context": reconstruction["variant_contexts"][variant],
        "geometry": _cached_summary(rows), "families": family_scores,
        "minimum_family_recall": min(score["recall"] for score in family_scores.values()),
        "case_dispositions": dispositions,
        "refused_case_count": sum(row["disposition"] != "SCORED_CACHED_GEOMETRY"
                                  for row in dispositions.values()),
        "observed_bend_count": None,
    }


def score_gold11_cached_fusion_v1(*, reference_records_by_case, a0_results_by_case,
                                  cad_results_by_case, case_family_by_id):
    """Reconstruct and score all four variants on exactly11/6/71 declarations.

    Gold has no input into the reconstruction function. This helper operates on
    already retained predictions, not scan/CAD geometry or fresh inference.
    Cohort/reference errors abort; unexpected matcher failures propagate to the
    retaining caller. Whole-case reconstruction or line-cap refusals retain all
    reference FNs, without scoring a favorable prefix or dropping a case.

    All four named variants are always reported; this API selects no winner.
    Cached line counts and exact localized sets are not physical observed counts.
    Original endpoint precedence, IDs/order and matcher geometry are unchanged.
    Payload snapshots are per-case, not atomic authentication against concurrent
    mutation; the caller must supply frozen authenticated inputs.
    """
    _require(type(case_family_by_id) is dict and len(case_family_by_id) == CASE_COUNT,
             "REQUIRE_COMPLETE_ELEVEN_CASE_FAMILY_MAP")
    families = dict(case_family_by_id)
    _require(all(_identifier(case) and _identifier(family) for case, family in families.items()),
             "INVALID_CASE_OR_FAMILY_ID")
    _require(len(set(families.values())) == FAMILY_COUNT, "REQUIRE_SIX_DECLARED_FAMILIES")
    expected = set(families)
    for mapping in (reference_records_by_case, a0_results_by_case, cad_results_by_case):
        _require(type(mapping) is dict and set(mapping) == expected, "CASE_SET_MISMATCH")
    a0_results_by_case, cad_results_by_case = dict(a0_results_by_case), dict(cad_results_by_case)
    identity_conflicts = {case for case in expected
        if any(type(payload) is dict and "family_id" in payload
               and (type(payload["family_id"]) is not str or payload["family_id"] != families[case])
               for payload in (a0_results_by_case[case], cad_results_by_case[case]))}
    references = {case: _lines(reference_records_by_case[case], prediction=False)
                  for case in sorted(expected)}
    _require(sum(map(len, references.values())) == REFERENCE_LINE_COUNT,
             "REQUIRE_SEVENTY_ONE_REFERENCE_LINES")
    reconstruction = reconstruct_gold11_retained_fusion_v1(
        a0_results_by_case=a0_results_by_case, cad_results_by_case=cad_results_by_case,
        case_ids=sorted(expected))
    lanes = {variant: _variant_lane(reconstruction, variant, references, families, identity_conflicts)
             for variant in VARIANTS}
    return {
        "schema": "GOLD11_CACHED_FUSION_GEOMETRY_DIAGNOSTIC_V1",
        "status": "COMPUTED_DEVELOPMENT_DIAGNOSTIC",
        "case_count": CASE_COUNT, "family_count": FAMILY_COUNT,
        "reference_line_count": REFERENCE_LINE_COUNT,
        "control_identity": "NEW_FROZEN_CACHED_FUSION_CONTROL_NOT_HISTORICAL_ENGINE_REPLAY",
        "historical_score_comparability": "UNESTABLISHED",
        "cohort_authentication": "CALLER_FROZEN_CASE_PAYLOAD_SOURCE_AND_REFERENCE_BYTES_REQUIRED",
        "comparison_scope": "PROVISIONAL_COMMON_MM_RAW_ENDPOINT_GEOMETRY_ONLY",
        "declared_common_frame": RAW_FRAME, "units_status": PROVISIONAL_UNITS,
        "reference_carrier_compatibility": "UNVERIFIED_NOT_ASSERTED_BY_SCORING",
        "reference_frame_compatibility": "PROVISIONAL_ASSUMPTION_NOT_VERIFIED",
        "carrier_semantics": "LEGACY_PREDICTED_LINES_NOT_ASSERTED_MEASURED_SURFACE_CARRIERS",
        "scoring_operations": "DIRECT_ENDPOINT_MATCHING_ONLY_NO_TRANSFORM_OR_UNIT_ESTIMATION",
        "inference_execution": "NOT_PERFORMED",
        "variants": lanes, "reconstruction": reconstruction,
        "variant_selection": "NONE_ALL_FOUR_REPORTED",
        "matching": {"axis_angle_deg_max": 5., "transverse_offset_mm_max": 4.,
                     "gold_span_overlap_min": .60, "prediction_span_overlap_min": .60,
                     "method": "UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST",
                     "assignment_identity": "ROW_INDEX_ORIGINAL_DISPLAY_IDS_PRESERVED"},
        "failure_policy": "WHOLE_CASE_ZERO_PREDICTIONS_ALL_REFERENCE_FNS_NO_PREFIX_OR_DROPPED_CASE",
        "prediction_safety_cap_per_case": MAX_PREDICTIONS_PER_CASE,
        "observed_bend_count": None, "strict_physical_engine_targets_met": False,
        "per_bend_measurement_accuracy": "NOT_EVALUATED_NO_MEASUREMENT_TRUTH",
        "conformance": "INDETERMINATE", "uncertainty_status": "UNCALIBRATED",
        "production_use_allowed": False, "floor_release_allowed": False,
    }
