from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class StructuredCountResult(BaseModel):
    completed_bends: int = 0
    remaining_bends: int = 0
    source: str = "direct_evidence"
    mean_completed_bends: Optional[float] = None
    map_completed_bends: Optional[int] = None
    delta_vs_match: Optional[int] = None


class BendHeatmapConsistencyResult(BaseModel):
    status: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NOT_APPLICABLE"]
    consistency_score: Optional[float] = None
    independence_class: Optional[Literal["INDEPENDENT", "PARTIAL", "NON_INDEPENDENT"]] = None
    roi_mode: Optional[str] = None
    axial_half_window_mm: Optional[float] = None
    radial_radius_mm: Optional[float] = None
    roi_point_count: int = 0
    roi_coverage_ratio: Optional[float] = None
    local_abs_mean_mm: Optional[float] = None
    local_abs_p95_mm: Optional[float] = None
    local_out_of_tol_ratio: Optional[float] = None
    side_asymmetry_score: Optional[float] = None
    signal_extent_along_flange: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class BendHeatmapConsistencySummaryResult(BaseModel):
    status: Literal["SUPPORTED", "WEAK_SUPPORT", "CONTRADICTED", "NOT_APPLICABLE"]
    inconsistent_bend_count: int = 0
    insufficient_evidence_bend_count: int = 0
    evaluable_bend_count: Optional[int] = None
    bend_roi_energy_share: Optional[float] = None
    global_in_tolerance_rate: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class BendMatchResult(BaseModel):
    bend_id: str
    status: Literal["PASS", "FAIL", "WARNING", "NOT_DETECTED"]
    bend_form: Optional[str] = None
    expected_angle: Optional[float] = None
    measured_angle: Optional[float] = None
    angle_deviation: Optional[float] = None
    radius_deviation: Optional[float] = None
    line_center_deviation_mm: Optional[float] = None
    tolerance_angle: Optional[float] = None
    tolerance_radius: Optional[float] = None
    physical_completion_state: Optional[str] = None
    observability_state: Optional[str] = None
    observability_detail_state: Optional[str] = None
    feature_family: Optional[str] = None
    measurement_primitive: Optional[str] = None
    assignment_source: Optional[str] = None
    assignment_confidence: Optional[float] = None
    visibility_score: Optional[float] = None
    heatmap_consistency: Optional[BendHeatmapConsistencyResult] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReportSummaryResult(BaseModel):
    total_bends: int = 0
    expected_bends: Optional[int] = None
    completed_bends: int = 0
    completed_in_spec: int = 0
    completed_out_of_spec: int = 0
    remaining_bends: int = 0
    structured_count: Optional[StructuredCountResult] = None
    heatmap_consistency_summary: Optional[BendHeatmapConsistencySummaryResult] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class BendInspectionResult(BaseModel):
    part_id: str
    summary: ReportSummaryResult
    matches: List[BendMatchResult] = Field(default_factory=list)
    alignment: Optional[Dict[str, Any]] = None
    report_payload: Dict[str, Any] = Field(default_factory=dict)
