"""
Data Exchange Schemas for Multi-Model AI Analysis Pipeline

These dataclasses define the structured data exchanged between:
- Drawing analysis (Stage 1)
- Feature detection (Stage 2)
- 2D-3D correlation (Stage 3)
- Root cause analysis (Stage 4)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime


# ============================================================================
# Stage 1: Drawing Analysis Schemas
# ============================================================================

@dataclass
class DrawingDimension:
    """A dimension extracted from a technical drawing."""
    id: str
    dimension_type: str  # "linear", "angular", "radial", "diameter"
    nominal: float
    tolerance_plus: float
    tolerance_minus: float
    unit: str = "mm"
    feature_ref: Optional[str] = None  # Reference to what this dimension measures
    location_hint: Optional[str] = None  # "top_flange", "bend_1_angle", etc.


@dataclass
class GDTCallout:
    """A Geometric Dimensioning and Tolerancing callout."""
    id: str
    gdt_type: str  # "flatness", "parallelism", "perpendicularity", "position", etc.
    tolerance: float
    unit: str = "mm"
    feature_ref: Optional[str] = None
    datums: List[str] = field(default_factory=list)  # ["A", "B", "C"]
    modifier: Optional[str] = None  # "MMC", "LMC", "RFS"


@dataclass
class DrawingBend:
    """A bend specification from a technical drawing."""
    id: str
    sequence: int  # Order in bending sequence
    angle_nominal: float  # degrees
    angle_tolerance: float = 1.0  # degrees
    radius: float = 0.0  # inside bend radius, mm
    direction: str = "up"  # "up" or "down"
    bend_line_location: Optional[str] = None  # Description or coords


@dataclass
class DrawingExtraction:
    """Complete extraction from a technical drawing (Stage 1 output)."""
    drawing_id: str
    extraction_timestamp: str
    dimensions: List[DrawingDimension] = field(default_factory=list)
    gdt_callouts: List[GDTCallout] = field(default_factory=list)
    bends: List[DrawingBend] = field(default_factory=list)
    material: Optional[str] = None
    thickness: Optional[float] = None
    part_number: Optional[str] = None
    revision: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    extraction_confidence: float = 0.0
    model_used: str = ""
    raw_extraction: Optional[Dict] = None  # Raw model output for debugging

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "drawing_id": self.drawing_id,
            "extraction_timestamp": self.extraction_timestamp,
            "dimensions": [
                {
                    "id": d.id,
                    "type": d.dimension_type,
                    "nominal": d.nominal,
                    "tolerance_plus": d.tolerance_plus,
                    "tolerance_minus": d.tolerance_minus,
                    "unit": d.unit,
                    "feature_ref": d.feature_ref,
                    "location_hint": d.location_hint,
                }
                for d in self.dimensions
            ],
            "gdt_callouts": [
                {
                    "id": g.id,
                    "type": g.gdt_type,
                    "tolerance": g.tolerance,
                    "unit": g.unit,
                    "feature_ref": g.feature_ref,
                    "datums": g.datums,
                    "modifier": g.modifier,
                }
                for g in self.gdt_callouts
            ],
            "bends": [
                {
                    "id": b.id,
                    "sequence": b.sequence,
                    "angle_nominal": b.angle_nominal,
                    "angle_tolerance": b.angle_tolerance,
                    "radius": b.radius,
                    "direction": b.direction,
                    "bend_line_location": b.bend_line_location,
                }
                for b in self.bends
            ],
            "material": self.material,
            "thickness": self.thickness,
            "part_number": self.part_number,
            "revision": self.revision,
            "notes": self.notes,
            "extraction_confidence": self.extraction_confidence,
            "model_used": self.model_used,
        }


# ============================================================================
# Stage 2: Feature Detection Schemas
# ============================================================================

class FeatureType(Enum):
    """Types of geometric features detected in point clouds."""
    BEND = "bend"
    FLAT_SURFACE = "flat_surface"
    EDGE = "edge"
    HOLE = "hole"
    SLOT = "slot"
    CORNER = "corner"
    CURVED_SURFACE = "curved_surface"
    UNKNOWN = "unknown"


@dataclass
class BendFeature:
    """A bend detected in the point cloud."""
    bend_id: int  # Sequential ID (1, 2, 3...)
    bend_name: str  # "Bend 1", "Bend 2", etc.
    angle_degrees: float  # Measured bend angle
    radius_mm: float  # Inside bend radius
    bend_line_start: Tuple[float, float, float]  # 3D coordinates
    bend_line_end: Tuple[float, float, float]
    bend_apex: Tuple[float, float, float]  # Center point of bend
    region_point_indices: List[int] = field(default_factory=list)  # Points in bend zone
    adjacent_surface_ids: List[int] = field(default_factory=list)  # Connected flat surfaces
    detection_confidence: float = 0.0
    bend_direction: str = "unknown"  # "up", "down", "left", "right"

    def to_dict(self) -> Dict:
        return {
            "bend_id": self.bend_id,
            "bend_name": self.bend_name,
            "angle_degrees": self.angle_degrees,
            "radius_mm": self.radius_mm,
            "bend_line_start": list(self.bend_line_start),
            "bend_line_end": list(self.bend_line_end),
            "bend_apex": list(self.bend_apex),
            "region_point_count": len(self.region_point_indices),
            "adjacent_surfaces": self.adjacent_surface_ids,
            "detection_confidence": self.detection_confidence,
            "bend_direction": self.bend_direction,
        }


@dataclass
class Feature3D:
    """A generic 3D feature detected in the point cloud."""
    feature_id: str
    feature_type: FeatureType
    location: Tuple[float, float, float]  # Center/reference point
    normal: Optional[Tuple[float, float, float]] = None  # Surface normal if applicable
    parameters: Dict[str, Any] = field(default_factory=dict)  # Type-specific params
    point_indices: List[int] = field(default_factory=list)
    deviation_stats: Dict[str, float] = field(default_factory=dict)
    detection_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type.value,
            "location": list(self.location),
            "normal": list(self.normal) if self.normal else None,
            "parameters": self.parameters,
            "point_count": len(self.point_indices),
            "deviation_stats": self.deviation_stats,
            "detection_confidence": self.detection_confidence,
        }


@dataclass
class BendDetectionResult:
    """Complete bend detection output."""
    bends: List[BendFeature] = field(default_factory=list)
    total_bends_detected: int = 0
    detection_method: str = "curvature_segmentation"
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "bends": [b.to_dict() for b in self.bends],
            "total_bends_detected": self.total_bends_detected,
            "detection_method": self.detection_method,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
        }


@dataclass
class PointCloudFeatures:
    """All features detected in a point cloud (Stage 2 output)."""
    scan_id: str
    detection_timestamp: str
    bends: List[BendFeature] = field(default_factory=list)
    surfaces: List[Feature3D] = field(default_factory=list)
    edges: List[Feature3D] = field(default_factory=list)
    other_features: List[Feature3D] = field(default_factory=list)
    total_points: int = 0
    processed_points: int = 0
    detection_confidence: float = 0.0
    model_used: str = ""

    def to_dict(self) -> Dict:
        return {
            "scan_id": self.scan_id,
            "detection_timestamp": self.detection_timestamp,
            "bends": [b.to_dict() for b in self.bends],
            "surfaces": [s.to_dict() for s in self.surfaces],
            "edges": [e.to_dict() for e in self.edges],
            "other_features": [f.to_dict() for f in self.other_features],
            "total_points": self.total_points,
            "processed_points": self.processed_points,
            "detection_confidence": self.detection_confidence,
            "model_used": self.model_used,
        }


# ============================================================================
# Stage 3: 2D-3D Correlation Schemas
# ============================================================================

@dataclass
class CorrelationMapping:
    """A mapping between a drawing element and a detected 3D feature."""
    mapping_id: str
    drawing_element_id: str  # ID from DrawingExtraction
    drawing_element_type: str  # "dimension", "gdt", "bend"
    feature_3d_id: str  # ID from PointCloudFeatures
    feature_3d_type: str
    confidence: float  # 0.0 to 1.0
    nominal_value: Optional[float] = None
    measured_value: Optional[float] = None
    deviation: Optional[float] = None
    is_in_tolerance: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "mapping_id": self.mapping_id,
            "drawing_element_id": self.drawing_element_id,
            "drawing_element_type": self.drawing_element_type,
            "feature_3d_id": self.feature_3d_id,
            "feature_3d_type": self.feature_3d_type,
            "confidence": self.confidence,
            "nominal_value": self.nominal_value,
            "measured_value": self.measured_value,
            "deviation": self.deviation,
            "is_in_tolerance": self.is_in_tolerance,
            "notes": self.notes,
        }


@dataclass
class CriticalDeviation:
    """A significant deviation that requires attention."""
    deviation_id: str
    feature_id: str
    feature_type: str
    location: Tuple[float, float, float]
    deviation_mm: float
    severity: str  # "critical", "major", "minor"
    affected_dimension_ids: List[str] = field(default_factory=list)
    affected_gdt_ids: List[str] = field(default_factory=list)
    affected_bend_ids: List[int] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "deviation_id": self.deviation_id,
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "location": list(self.location),
            "deviation_mm": self.deviation_mm,
            "severity": self.severity,
            "affected_dimensions": self.affected_dimension_ids,
            "affected_gdt": self.affected_gdt_ids,
            "affected_bends": self.affected_bend_ids,
            "description": self.description,
        }


@dataclass
class Correlation2D3D:
    """Complete 2D-3D correlation output (Stage 3 output)."""
    correlation_id: str
    correlation_timestamp: str
    mappings: List[CorrelationMapping] = field(default_factory=list)
    critical_deviations: List[CriticalDeviation] = field(default_factory=list)
    unmatched_drawing_elements: List[str] = field(default_factory=list)
    unmatched_3d_features: List[str] = field(default_factory=list)
    overall_correlation_score: float = 0.0
    model_used: str = ""

    def to_dict(self) -> Dict:
        return {
            "correlation_id": self.correlation_id,
            "correlation_timestamp": self.correlation_timestamp,
            "mappings": [m.to_dict() for m in self.mappings],
            "critical_deviations": [d.to_dict() for d in self.critical_deviations],
            "unmatched_drawing_elements": self.unmatched_drawing_elements,
            "unmatched_3d_features": self.unmatched_3d_features,
            "overall_correlation_score": self.overall_correlation_score,
            "model_used": self.model_used,
        }


# ============================================================================
# Stage 4: Root Cause & Bend Analysis Schemas
# ============================================================================

@dataclass
class BendRegionAnalysis:
    """Detailed analysis of a specific bend region."""
    bend: BendFeature
    nominal_angle: Optional[float] = None  # From drawing if available
    angle_deviation_deg: float = 0.0
    springback_indicator: float = 0.0  # 0-1 likelihood of springback
    over_bend_indicator: float = 0.0  # 0-1 likelihood of overbend
    apex_deviation_mm: float = 0.0
    radius_deviation_mm: float = 0.0
    twist_angle_deg: float = 0.0
    status: str = "unknown"  # "pass", "fail", "warning"
    recommendations: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "bend": self.bend.to_dict(),
            "nominal_angle": self.nominal_angle,
            "angle_deviation_deg": self.angle_deviation_deg,
            "springback_indicator": self.springback_indicator,
            "over_bend_indicator": self.over_bend_indicator,
            "apex_deviation_mm": self.apex_deviation_mm,
            "radius_deviation_mm": self.radius_deviation_mm,
            "twist_angle_deg": self.twist_angle_deg,
            "status": self.status,
            "recommendations": self.recommendations,
            "root_causes": self.root_causes,
        }


@dataclass
class EnhancedRootCause:
    """An enhanced root cause with feature references."""
    cause_id: str
    category: str  # "tooling", "material", "process", "design", "measurement"
    description: str
    severity: str  # "critical", "major", "minor"
    confidence: float  # 0-1
    affected_features: List[str] = field(default_factory=list)
    affected_bends: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_impact: Optional[str] = None  # e.g., "Reduce deviation by 40%"
    priority: int = 0  # 1 = highest

    def to_dict(self) -> Dict:
        return {
            "cause_id": self.cause_id,
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence,
            "affected_features": self.affected_features,
            "affected_bends": self.affected_bends,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "estimated_impact": self.estimated_impact,
            "priority": self.priority,
        }


# ============================================================================
# Pipeline Status Tracking
# ============================================================================

class PipelineStage(Enum):
    """Stages in the enhanced analysis pipeline."""
    DRAWING_ANALYSIS = "drawing_analysis"
    FEATURE_DETECTION = "feature_detection"
    CORRELATION_2D_3D = "correlation_2d_3d"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    COMPLETE = "complete"


@dataclass
class StageStatus:
    """Status of a single pipeline stage."""
    stage: PipelineStage
    status: str  # "pending", "running", "completed", "failed", "skipped"
    model_used: Optional[str] = None
    fallback_used: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


@dataclass
class PipelineStatus:
    """Overall pipeline execution status."""
    job_id: str
    current_stage: PipelineStage
    stages: Dict[str, StageStatus] = field(default_factory=dict)
    overall_progress: float = 0.0  # 0-100
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "current_stage": self.current_stage.value,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "overall_progress": self.overall_progress,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
        }


# ============================================================================
# Final Enhanced Analysis Result
# ============================================================================

@dataclass
class EnhancedAnalysisResult:
    """Complete enhanced analysis output combining all stages."""
    job_id: str
    analysis_timestamp: str

    # Stage outputs
    drawing_extraction: Optional[DrawingExtraction] = None
    point_cloud_features: Optional[PointCloudFeatures] = None
    correlation_2d_3d: Optional[Correlation2D3D] = None

    # Bend-specific analysis
    bend_results: List[BendRegionAnalysis] = field(default_factory=list)

    # Root causes with feature references
    enhanced_root_causes: List[EnhancedRootCause] = field(default_factory=list)

    # Pipeline status
    pipeline_status: Optional[PipelineStatus] = None

    # Summary
    total_bends_detected: int = 0
    bends_in_tolerance: int = 0
    bends_out_of_tolerance: int = 0
    critical_issues_count: int = 0

    # Metadata
    models_used: List[str] = field(default_factory=list)
    fallbacks_triggered: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "analysis_timestamp": self.analysis_timestamp,
            "drawing_extraction": self.drawing_extraction.to_dict() if self.drawing_extraction else None,
            "point_cloud_features": self.point_cloud_features.to_dict() if self.point_cloud_features else None,
            "correlation_2d_3d": self.correlation_2d_3d.to_dict() if self.correlation_2d_3d else None,
            "bend_results": [b.to_dict() for b in self.bend_results],
            "enhanced_root_causes": [r.to_dict() for r in self.enhanced_root_causes],
            "pipeline_status": self.pipeline_status.to_dict() if self.pipeline_status else None,
            "summary": {
                "total_bends_detected": self.total_bends_detected,
                "bends_in_tolerance": self.bends_in_tolerance,
                "bends_out_of_tolerance": self.bends_out_of_tolerance,
                "critical_issues_count": self.critical_issues_count,
            },
            "metadata": {
                "models_used": self.models_used,
                "fallbacks_triggered": self.fallbacks_triggered,
                "processing_time_ms": self.processing_time_ms,
            },
        }
