"""
Feature Types Module

Unified data structures for the feature-by-feature dimension measurement system.
Enables accurate three-way comparison: XLSX specification <-> CAD design <-> Scan actual.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import math
import logging

if TYPE_CHECKING:
    from dimension_parser import DimensionSpec

logger = logging.getLogger(__name__)


def safe_float(v: Any, decimals: int = 3) -> Optional[float]:
    """Convert to JSON-safe float (handle NaN/Inf and numpy types)."""
    if v is None:
        return None
    try:
        # Convert to Python float (handles numpy scalars, integers, etc.)
        float_val = float(v)
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return round(float_val, decimals)
    except (TypeError, ValueError):
        return None


def safe_array_to_list(arr: Optional[np.ndarray], decimals: int = 3) -> Optional[List[float]]:
    """Convert numpy array to JSON-safe list."""
    if arr is None:
        return None
    try:
        return [safe_float(x, decimals) for x in arr]
    except (TypeError, ValueError):
        return None


class FeatureType(Enum):
    """Types of measurable geometric features."""
    LINEAR = "linear"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ANGLE = "angle"


@dataclass
class MeasurableFeature:
    """Base class for all measurable geometric features."""
    feature_id: str
    feature_type: FeatureType
    nominal_value: float          # Expected value from CAD
    unit: str                     # "mm" or "deg"
    position: np.ndarray          # 3D reference point [x,y,z]
    direction: np.ndarray         # Measurement direction
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type.value,
            "nominal_value": safe_float(self.nominal_value, 3),
            "unit": self.unit,
            "position": safe_array_to_list(self.position, 3),
            "direction": safe_array_to_list(self.direction, 4),
            "confidence": safe_float(self.confidence, 3),
        }


@dataclass
class HoleFeature(MeasurableFeature):
    """Cylindrical hole or circular feature."""
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))  # Cylinder axis direction
    depth: float = 0.0            # Hole depth (0 for through)
    entry_point: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # Top surface intersection
    rim_points: Optional[np.ndarray] = None  # Points forming the hole rim (for measurement)

    def __post_init__(self):
        # Ensure arrays are numpy arrays
        if not isinstance(self.axis, np.ndarray):
            self.axis = np.array(self.axis)
        if not isinstance(self.entry_point, np.ndarray):
            self.entry_point = np.array(self.entry_point)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        base = super().to_dict()
        base.update({
            "axis": safe_array_to_list(self.axis, 4),
            "depth": safe_float(self.depth, 3),
            "entry_point": safe_array_to_list(self.entry_point, 3),
            "has_rim_points": self.rim_points is not None,
        })
        return base


@dataclass
class LinearFeature(MeasurableFeature):
    """Linear dimension between two references."""
    point1: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # Start point
    point2: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # End point
    surface1_normal: Optional[np.ndarray] = None
    surface2_normal: Optional[np.ndarray] = None
    measurement_type: str = "surface_to_surface"

    def __post_init__(self):
        # Ensure arrays are numpy arrays
        if not isinstance(self.point1, np.ndarray):
            self.point1 = np.array(self.point1)
        if not isinstance(self.point2, np.ndarray):
            self.point2 = np.array(self.point2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        base = super().to_dict()
        base.update({
            "point1": safe_array_to_list(self.point1, 3),
            "point2": safe_array_to_list(self.point2, 3),
            "surface1_normal": safe_array_to_list(self.surface1_normal, 4),
            "surface2_normal": safe_array_to_list(self.surface2_normal, 4),
            "measurement_type": self.measurement_type,
        })
        return base


@dataclass
class AngleFeature(MeasurableFeature):
    """Angle between two surfaces."""
    surface1_normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    surface2_normal: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    apex_point: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    bend_line: Optional[np.ndarray] = None
    surface1_id: Optional[int] = None
    surface2_id: Optional[int] = None

    def __post_init__(self):
        # Ensure arrays are numpy arrays
        if not isinstance(self.surface1_normal, np.ndarray):
            self.surface1_normal = np.array(self.surface1_normal)
        if not isinstance(self.surface2_normal, np.ndarray):
            self.surface2_normal = np.array(self.surface2_normal)
        if not isinstance(self.apex_point, np.ndarray):
            self.apex_point = np.array(self.apex_point)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        base = super().to_dict()
        base.update({
            "surface1_normal": safe_array_to_list(self.surface1_normal, 4),
            "surface2_normal": safe_array_to_list(self.surface2_normal, 4),
            "apex_point": safe_array_to_list(self.apex_point, 3),
            "bend_line": safe_array_to_list(self.bend_line, 4),
            "surface1_id": self.surface1_id,
            "surface2_id": self.surface2_id,
        })
        return base


@dataclass
class FeatureRegistry:
    """Collection of all detected features from a part."""
    holes: List[HoleFeature] = field(default_factory=list)
    linear_dims: List[LinearFeature] = field(default_factory=list)
    angles: List[AngleFeature] = field(default_factory=list)
    radii: List[HoleFeature] = field(default_factory=list)  # Fillets/rounds

    def find_by_type_and_value(
        self,
        ftype: FeatureType,
        value: float,
        tolerance: float = 2.0
    ) -> List[MeasurableFeature]:
        """Find features matching type and approximate value."""
        candidates = []

        if ftype == FeatureType.DIAMETER:
            for hole in self.holes:
                if abs(hole.nominal_value - value) <= tolerance:
                    candidates.append(hole)
        elif ftype == FeatureType.RADIUS:
            for radius in self.radii:
                if abs(radius.nominal_value - value) <= tolerance:
                    candidates.append(radius)
        elif ftype == FeatureType.LINEAR:
            for linear in self.linear_dims:
                if abs(linear.nominal_value - value) <= tolerance:
                    candidates.append(linear)
        elif ftype == FeatureType.ANGLE:
            for angle in self.angles:
                if abs(angle.nominal_value - value) <= tolerance:
                    candidates.append(angle)

        return candidates

    def get_all_features(self) -> List[MeasurableFeature]:
        """Get all features as a flat list."""
        return list(self.holes) + list(self.linear_dims) + list(self.angles) + list(self.radii)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "holes": [h.to_dict() for h in self.holes],
            "linear_dims": [l.to_dict() for l in self.linear_dims],
            "angles": [a.to_dict() for a in self.angles],
            "radii": [r.to_dict() for r in self.radii],
            "total_features": len(self.holes) + len(self.linear_dims) + len(self.angles) + len(self.radii),
        }


@dataclass
class FeatureMatch:
    """Matched feature across XLSX, CAD, and Scan."""
    dim_id: int
    xlsx_spec: Optional['DimensionSpec']
    cad_feature: MeasurableFeature

    # Measurements
    xlsx_value: float
    cad_value: float
    scan_value: Optional[float] = None

    # Tolerances
    tolerance_plus: float = 0.5
    tolerance_minus: float = 0.5

    # Computed deviations
    scan_vs_xlsx: Optional[float] = None
    scan_vs_cad: Optional[float] = None

    # Status
    status: str = "pending"       # pass, fail, warning, not_measured
    match_confidence: float = 0.0
    measurement_confidence: float = 0.0

    # Feature type info
    feature_type: str = ""
    unit: str = "mm"
    description: str = ""
    axis: Optional[str] = None  # Measurement axis for linear dims: X, Y, Z

    def compute_status(self):
        """Compute deviation and pass/fail status."""
        if self.scan_value is None:
            self.status = "not_measured"
            return

        # Compute deviations
        self.scan_vs_xlsx = self.scan_value - self.xlsx_value
        self.scan_vs_cad = self.scan_value - self.cad_value

        # Use XLSX as source of truth for status
        deviation = self.scan_vs_xlsx

        # Check against tolerance (handle asymmetric tolerances)
        if deviation >= 0:
            within_tolerance = deviation <= self.tolerance_plus
            within_warning = deviation <= self.tolerance_plus * 2
        else:
            within_tolerance = abs(deviation) <= self.tolerance_minus
            within_warning = abs(deviation) <= self.tolerance_minus * 2

        if within_tolerance:
            self.status = "pass"
        elif within_warning:
            self.status = "warning"
        else:
            self.status = "fail"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "dim_id": self.dim_id,
            "feature_type": self.feature_type,
            "description": self.description,
            "xlsx_value": safe_float(self.xlsx_value, 3),
            "cad_value": safe_float(self.cad_value, 3),
            "scan_value": safe_float(self.scan_value, 3),
            "unit": self.unit,
            "tolerance_plus": safe_float(self.tolerance_plus, 3),
            "tolerance_minus": safe_float(self.tolerance_minus, 3),
            "scan_vs_xlsx": safe_float(self.scan_vs_xlsx, 3),
            "scan_vs_cad": safe_float(self.scan_vs_cad, 3),
            "status": self.status,
            "match_confidence": safe_float(self.match_confidence, 3),
            "measurement_confidence": safe_float(self.measurement_confidence, 3),
            "cad_feature": self.cad_feature.to_dict() if self.cad_feature else None,
        }


@dataclass
class ThreeWayReport:
    """Complete feature-by-feature comparison report."""
    part_id: str
    matches: List[FeatureMatch]
    unmatched_xlsx: List[Any] = field(default_factory=list)  # List[DimensionSpec]
    unmatched_cad: List[MeasurableFeature] = field(default_factory=list)

    # Summary
    total_dimensions: int = 0
    measured_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

    # Timing
    processing_time_ms: float = 0.0

    def compute_summary(self):
        """Compute summary statistics."""
        self.total_dimensions = len(self.matches)
        self.measured_count = sum(1 for m in self.matches if m.status != "not_measured")
        self.pass_count = sum(1 for m in self.matches if m.status == "pass")
        self.fail_count = sum(1 for m in self.matches if m.status == "fail")
        self.warning_count = sum(1 for m in self.matches if m.status == "warning")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        self.compute_summary()

        return {
            "part_id": self.part_id,
            "summary": {
                "total_dimensions": self.total_dimensions,
                "measured": self.measured_count,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "warnings": self.warning_count,
                "pass_rate": round((self.pass_count / self.measured_count * 100) if self.measured_count > 0 else 0, 1),
            },
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_xlsx": [
                d.to_dict() if hasattr(d, 'to_dict') else str(d)
                for d in self.unmatched_xlsx
            ],
            "unmatched_cad": [f.to_dict() for f in self.unmatched_cad],
            "processing_time_ms": safe_float(self.processing_time_ms, 1),
        }

    def to_table_string(self) -> str:
        """Generate ASCII table representation for display."""
        self.compute_summary()

        lines = []
        lines.append("=" * 90)
        lines.append(f"THREE-WAY DIMENSION COMPARISON REPORT - {self.part_id}")
        lines.append("=" * 90)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Dimensions: {self.total_dimensions}")
        lines.append(f"Measured: {self.measured_count}")
        lines.append(f"Passed: {self.pass_count}")
        lines.append(f"Failed: {self.fail_count}")
        lines.append(f"Warnings: {self.warning_count}")
        if self.measured_count > 0:
            lines.append(f"Pass Rate: {self.pass_count / self.measured_count * 100:.1f}%")
        lines.append("")

        # Dimension table
        lines.append("DIMENSION DETAILS")
        lines.append("-" * 90)
        header = f"{'ID':<4} {'Type':<10} {'XLSX':<10} {'Tol':<8} {'CAD':<10} {'Scan':<10} {'Dev':<10} {'Status':<8}"
        lines.append(header)
        lines.append("-" * 90)

        for m in self.matches:
            xlsx_str = f"{m.xlsx_value:.2f}" if m.xlsx_value is not None else "N/A"
            cad_str = f"{m.cad_value:.2f}" if m.cad_value is not None else "N/A"
            scan_str = f"{m.scan_value:.2f}" if m.scan_value is not None else "N/A"
            dev_str = f"{m.scan_vs_xlsx:+.2f}" if m.scan_vs_xlsx is not None else "N/A"

            if m.tolerance_plus == m.tolerance_minus:
                tol_str = f"±{m.tolerance_plus:.2f}"
            else:
                tol_str = f"+{m.tolerance_plus}/−{m.tolerance_minus}"

            status_symbol = {
                "pass": "PASS",
                "fail": "FAIL",
                "warning": "WARN",
                "not_measured": "N/A",
                "pending": "PEND",
            }.get(m.status, m.status.upper())

            row = f"{m.dim_id:<4} {m.feature_type:<10} {xlsx_str:<10} {tol_str:<8} {cad_str:<10} {scan_str:<10} {dev_str:<10} {status_symbol:<8}"
            lines.append(row)

        lines.append("-" * 90)

        if self.unmatched_xlsx:
            lines.append("")
            lines.append(f"UNMATCHED XLSX DIMENSIONS: {len(self.unmatched_xlsx)}")
            for d in self.unmatched_xlsx:
                if hasattr(d, 'dim_id') and hasattr(d, 'value'):
                    lines.append(f"  Dim {d.dim_id}: {d.value}")

        if self.unmatched_cad:
            lines.append("")
            lines.append(f"UNMATCHED CAD FEATURES: {len(self.unmatched_cad)}")
            for f in self.unmatched_cad:
                lines.append(f"  {f.feature_id}: {f.nominal_value} {f.unit}")

        return "\n".join(lines)
