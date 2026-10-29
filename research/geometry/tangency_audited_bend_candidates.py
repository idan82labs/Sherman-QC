"""Add a bounded same-face relation audit without replacing radius candidates.

Discovery, supports, graphs, all raw radius alternatives and their statuses are
preserved under base_analysis. A secant-plane exclusion is not a missing bend,
invalid radius, rejected multiface model or production decision.
"""
from __future__ import annotations

import numpy as np

from research.geometry.cap_supported_bend_candidates import analyze_cap_supported_bend_candidates
from research.geometry.position_bend_candidates import MAX_BRIDGE_EVALUATIONS, _finite_payload, _integer
from research.geometry.same_face_tangency import _assess_same_face_tangency


def analyze_tangency_audited_bend_candidates(points, *, max_center_evaluations, **discovery_policy):
    """Reuse each internal radius result; at most one center solve per single.

    The additional budget is explicit, at most the retained 256 association cap.
    Search completeness concerns ONLY this declared necessary diagnostic family,
    never all geometric explanations. All prefixes clear on unavailable analysis.
    """
    cap = _integer(max_center_evaluations, "max_center_evaluations", MAX_BRIDGE_EVALUATIONS)
    base = analyze_cap_supported_bend_candidates(points, **discovery_policy)
    result = {
        "schema_version": 1, "kind": "TANGENCY_AUDITED_BEND_CANDIDATES_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "base_analysis": base, "assessments": [], "reasons": [],
        "search_complete_within_declared_policy": False,
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "policy": {
            "max_center_evaluations": cap, "max_flange_centroid_checks_per_center": 2,
            "center_evaluation_count_scope": "ASSESSMENT_ATTEMPTS_UPPER_BOUND_ON_CENTER_SOLVES",
            "radius_analysis_recomputed": False, "base_candidates_modified": False,
            "two_radius_candidates_checked_as_single_face": False,
            "support_selection_or_trimming": False,
            "scope": "NECESSARY_SINGLE_SAME_FACE_TANGENCY_DIAGNOSTIC_NOT_MODEL_SELECTION",
        },
        "diagnostics": {
            "base_association_count": len(base["candidates"]),
            "conditional_single_count": 0, "conditional_two_count": 0,
            "center_enclosure_evaluation_count": 0,
            "same_face_excluded_count": 0, "tangency_not_excluded_count": 0,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "NO_NEW_PHYSICAL_EVIDENCE",
            "NOT_EXCLUDED_IS_NOT_ORIGINAL_JOINT_FEASIBILITY_OR_TANGENCY",
            "EXCLUDED_SAME_FACE_RELATION_DOES_NOT_INVALIDATE_RADIUS_OR_MULTIFACE_MODELS",
            "BASE_CANDIDATES_AND_ALL_UNMATCHED_ALTERNATIVES_RETAINED",
            "BOUND_NOT_NUMERICALLY_CERTIFIED", "POSITION_ERROR_BOUND_UNCALIBRATED",
        ],
    }
    if not base["search_complete_within_declared_policy"]:
        result["reasons"] = ["BASE_CANDIDATE_SEARCH_INCOMPLETE"]
        return result
    singles = [c for c in base["candidates"] if c["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"]
    diagnostics = result["diagnostics"]
    diagnostics["conditional_single_count"] = len(singles)
    diagnostics["conditional_two_count"] = sum(c["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"
                                              for c in base["candidates"])
    if len(singles) > cap:
        result["reasons"] = ["CENTER_ENCLOSURE_BUDGET_EXCEEDED"]
        return result
    p = np.asarray(base["positions"], dtype=float)
    regions = {r["region_id"]: r for r in base["extended_regions"]}
    for candidate in singles:
        one, two = [regions[i]["point_indices"] for i in candidate["plane_region_ids"]]
        assessment = _assess_same_face_tangency(
            p[one], p[two], p[candidate["transition_point_indices"]],
            candidate["single_radius_analysis"],
        )
        diagnostics["center_enclosure_evaluation_count"] += 1
        if not assessment["analysis_available"]:
            result.update(assessments=[], reasons=["SAME_FACE_TANGENCY_ANALYSIS_UNAVAILABLE"])
            return result
        result["assessments"].append(dict(candidate_id=candidate["candidate_id"], analysis=assessment))
    if not _finite_payload(result):
        result.update(assessments=[], reasons=["TANGENCY_AUDIT_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED"])
        return result
    diagnostics["same_face_excluded_count"] = sum(
        item["analysis"]["single_same_face_tangent_model_excluded"] for item in result["assessments"])
    diagnostics["tangency_not_excluded_count"] = len(singles)-diagnostics["same_face_excluded_count"]
    result["search_complete_within_declared_policy"] = True
    if singles:
        result["status"] = "MODEL_RELATION_DIAGNOSTICS"
    else:
        result["reasons"] = ["NO_CONDITIONAL_SINGLE_RADIUS_CANDIDATES"]
    return result
