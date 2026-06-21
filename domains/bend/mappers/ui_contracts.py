from __future__ import annotations

from typing import Any, Dict, List

from contracts.ui.bend_results import (
    BendHeatmapConsistencyResult,
    BendHeatmapConsistencySummaryResult,
    BendInspectionResult,
    BendMatchResult,
    ReportSummaryResult,
    StructuredCountResult,
)


def map_bend_report_to_contract(part_id: str, report_dict: Dict[str, Any]) -> BendInspectionResult:
    summary_payload = dict(report_dict.get("summary") or {})
    structured_count = None
    if summary_payload.get("structured_completed_bends") is not None:
        structured_count = StructuredCountResult(
            completed_bends=int(summary_payload.get("structured_completed_bends") or 0),
            remaining_bends=int(summary_payload.get("structured_remaining_bends") or 0),
            source=str(summary_payload.get("structured_count_source") or "posterior_median"),
            mean_completed_bends=summary_payload.get("structured_count_mean"),
            map_completed_bends=summary_payload.get("structured_count_map"),
            delta_vs_match=summary_payload.get("structured_count_delta_vs_match"),
        )

    summary = ReportSummaryResult(
        total_bends=int(summary_payload.get("total_bends") or 0),
        expected_bends=summary_payload.get("expected_bends"),
        completed_bends=int(summary_payload.get("completed_bends", summary_payload.get("detected", 0)) or 0),
        completed_in_spec=int(summary_payload.get("completed_in_spec", summary_payload.get("passed", 0)) or 0),
        completed_out_of_spec=int(summary_payload.get("completed_out_of_spec", 0) or 0),
        remaining_bends=int(summary_payload.get("remaining_bends", 0) or 0),
        structured_count=structured_count,
        heatmap_consistency_summary=(
            BendHeatmapConsistencySummaryResult(**summary_payload.get("heatmap_consistency_summary"))
            if isinstance(summary_payload.get("heatmap_consistency_summary"), dict)
            else None
        ),
        payload=summary_payload,
    )

    matches: List[BendMatchResult] = []
    for raw in report_dict.get("matches") or []:
        payload = dict(raw)
        matches.append(BendMatchResult(
            bend_id=str(payload.get("bend_id") or payload.get("name") or "UNKNOWN"),
            status=str(payload.get("status") or "NOT_DETECTED"),
            bend_form=payload.get("bend_form"),
            expected_angle=payload.get("expected_angle"),
            measured_angle=payload.get("measured_angle"),
            angle_deviation=payload.get("angle_deviation"),
            radius_deviation=payload.get("radius_deviation"),
            line_center_deviation_mm=payload.get("line_center_deviation_mm"),
            tolerance_angle=payload.get("tolerance_angle"),
            tolerance_radius=payload.get("tolerance_radius"),
            physical_completion_state=payload.get("physical_completion_state"),
            observability_state=payload.get("observability_state"),
            observability_detail_state=payload.get("observability_detail_state"),
            feature_family=payload.get("feature_family"),
            measurement_primitive=payload.get("measurement_primitive"),
            assignment_source=payload.get("assignment_source"),
            assignment_confidence=payload.get("assignment_confidence"),
            visibility_score=payload.get("visibility_score"),
            heatmap_consistency=(
                BendHeatmapConsistencyResult(**payload.get("heatmap_consistency"))
                if isinstance(payload.get("heatmap_consistency"), dict)
                else None
            ),
            payload=payload,
        ))

    return BendInspectionResult(
        part_id=part_id,
        summary=summary,
        matches=matches,
        alignment=(dict(report_dict.get("alignment")) if isinstance(report_dict.get("alignment"), dict) else None),
        report_payload=dict(report_dict),
    )
