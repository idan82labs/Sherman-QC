from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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


class BendObservedCounts(BaseModel):
    formed: int = Field(ge=0, strict=True)
    unformed: int = Field(ge=0, strict=True)
    unknown: int = Field(ge=0, strict=True)
    total: int = Field(ge=0, strict=True)
    partially_observed: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_partition(self):
        if self.formed + self.unformed + self.unknown != self.total:
            raise ValueError("observed counts must partition the countable targets")
        if self.partially_observed > self.unknown:
            raise ValueError("partial observation is a subset of unknown physical state")
        return self


class BendDecisionRow(BaseModel):
    bend_id: str
    state: Literal["FORMED", "UNFORMED", "UNKNOWN"]
    observation_detail: str
    reasons: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_observation_detail(self):
        details = {"FORMED": {"FORMED", "OBSERVED_FORMED"}, "UNFORMED": {"UNFORMED", "OBSERVED_NOT_FORMED"}}
        if self.state in details and self.observation_detail not in details[self.state]:
            raise ValueError("a positive observation requires consistent explicit detail")
        return self


class BendDecisionContract(BaseModel):
    schema_version: Literal[1]
    kind: Literal["BEND_DECISION_CONTRACT_V1"]
    observed_counts: BendObservedCounts
    registration: Literal["ABSTAINED", "UNVERIFIED"]
    render_evidence: Literal["UNAVAILABLE", "UNVERIFIED"]
    production_disposition: Literal["NOT_AUTHORIZED"]
    reasons: List[str] = Field(default_factory=list)
    bends: List[BendDecisionRow] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the explicit integer 1")
        return value

    @model_validator(mode="after")
    def validate_evidence_rows(self):
        identities = [row.bend_id for row in self.bends]
        if any(not identity.strip() for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("decision rows require nonempty unique identities")
        if self.registration == "ABSTAINED" and any(row.state != "UNKNOWN" for row in self.bends):
            raise ValueError("abstained registration cannot support a physical observation")
        if self.observed_counts.total > len(self.bends):
            raise ValueError("countable targets cannot exceed decision rows")
        for field, state in (("formed", "FORMED"), ("unformed", "UNFORMED"), ("unknown", "UNKNOWN")):
            if getattr(self.observed_counts, field) > sum(row.state == state for row in self.bends):
                raise ValueError("counts cannot exceed corresponding evidence rows")
        return self


class BendInspectionResult(BaseModel):
    part_id: str
    summary: ReportSummaryResult
    matches: List[BendMatchResult] = Field(default_factory=list)
    alignment: Optional[Dict[str, Any]] = None
    decision_contract: Optional[BendDecisionContract] = None
    report_payload: Dict[str, Any] = Field(default_factory=dict)
