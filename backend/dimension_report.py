"""
Dimension Report Generator

Generates comprehensive reports comparing:
- XLSX specifications (expected values)
- CAD measurements (design values)
- Scan measurements (actual values)

Includes bend-by-bend analysis with pass/fail status.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from dimension_parser import DimensionParseResult, DimensionSpec, DimensionType
from bend_matcher import BendMatchResult, BendMatch

logger = logging.getLogger(__name__)


@dataclass
class DimensionComparison:
    """Comparison of a single dimension across XLSX spec, CAD geometry, and scan measurement.

    Reference priority:
    1. XLSX spec is source of truth when available (it's the engineering spec)
    2. CAD geometry is fallback when XLSX doesn't cover a dimension
    3. Always cross-reference XLSX vs CAD to detect discrepancies
    4. Works with or without XLSX (CAD-only mode)

    Deviation is: scan_value - expected (XLSX preferred, CAD fallback)
    """
    dim_id: int
    dim_type: str
    description: str

    # Values from each source
    xlsx_value: Optional[float] = None  # From XLSX spec (source of truth when present)
    cad_value: Optional[float] = None   # From CAD geometry (always available, cross-reference)
    scan_value: Optional[float] = None  # From scan measurement (actual)
    measured_output: Optional[float] = None  # Raw output from measurement engine (before any adjustment)
    unit: str = "mm"

    # Tolerances (from XLSX/drawing, with defaults)
    tolerance_plus: float = 0.5
    tolerance_minus: float = 0.5

    # Computed fields
    scan_deviation: Optional[float] = None          # scan - expected
    scan_deviation_percent: Optional[float] = None
    xlsx_vs_cad_deviation: Optional[float] = None   # Cross-reference discrepancy
    scan_vs_cad_deviation: Optional[float] = None   # scan - CAD (always computed)

    # Status
    status: str = "pending"  # pass, fail, warning, not_measured

    @property
    def expected(self) -> float:
        """The reference value: XLSX is source of truth, CAD is fallback."""
        if self.xlsx_value is not None:
            return self.xlsx_value
        if self.cad_value is not None:
            return self.cad_value
        return 0.0

    @property
    def reference_source(self) -> str:
        """Which source provides the 'expected' value."""
        if self.xlsx_value is not None:
            return "XLSX"
        if self.cad_value is not None:
            return "CAD"
        return "N/A"

    @property
    def has_cross_reference(self) -> bool:
        """Whether both XLSX and CAD values are available for cross-referencing."""
        return self.xlsx_value is not None and self.cad_value is not None

    def compute_status(self):
        """Compute deviation and pass/fail status.

        Logic:
        - Deviation = scan - expected (XLSX when available, else CAD)
        - Also compute scan-vs-CAD for cross-reference
        - Also compute XLSX-vs-CAD discrepancy
        """
        if self.scan_value is None:
            self.status = "not_measured"
            return

        # Primary deviation: scan vs expected (XLSX preferred, CAD fallback)
        reference = self.expected
        self.scan_deviation = self.scan_value - reference

        if reference != 0:
            self.scan_deviation_percent = (self.scan_deviation / reference) * 100
        else:
            self.scan_deviation_percent = 0

        # Cross-reference: scan vs CAD (always useful even when XLSX is primary)
        if self.cad_value is not None:
            self.scan_vs_cad_deviation = self.scan_value - self.cad_value

        # Cross-reference: XLSX vs CAD discrepancy (flag if specs differ from design)
        if self.xlsx_value is not None and self.cad_value is not None:
            self.xlsx_vs_cad_deviation = self.xlsx_value - self.cad_value

        # Check against tolerance (handle asymmetric tolerances)
        if self.scan_deviation >= 0:
            within_tolerance = self.scan_deviation <= self.tolerance_plus
            within_warning = self.scan_deviation <= self.tolerance_plus * 2
        else:
            within_tolerance = abs(self.scan_deviation) <= self.tolerance_minus
            within_warning = abs(self.scan_deviation) <= self.tolerance_minus * 2

        if within_tolerance:
            self.status = "pass"
        elif within_warning:
            self.status = "warning"
        else:
            self.status = "fail"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dim_id": self.dim_id,
            "type": self.dim_type,
            "description": self.description,
            "expected": round(self.expected, 3),
            "reference_source": self.reference_source,
            "xlsx_value": round(self.xlsx_value, 3) if self.xlsx_value is not None else None,
            "cad_value": round(self.cad_value, 3) if self.cad_value is not None else None,
            "scan_value": round(self.scan_value, 3) if self.scan_value is not None else None,
            "measured_output": round(self.measured_output, 3) if self.measured_output is not None else None,
            "unit": self.unit,
            "tolerance": f"+{self.tolerance_plus}/-{self.tolerance_minus}",
            "tolerance_plus": self.tolerance_plus,
            "tolerance_minus": self.tolerance_minus,
            "xlsx_vs_cad_deviation": round(self.xlsx_vs_cad_deviation, 3) if self.xlsx_vs_cad_deviation is not None else None,
            "scan_vs_cad_deviation": round(self.scan_vs_cad_deviation, 3) if self.scan_vs_cad_deviation is not None else None,
            "scan_deviation": round(self.scan_deviation, 3) if self.scan_deviation is not None else None,
            "scan_deviation_percent": round(self.scan_deviation_percent, 2) if self.scan_deviation_percent is not None else None,
            "status": self.status,
        }


@dataclass
class DimensionReport:
    """Complete dimension comparison report."""
    # Metadata
    part_id: str = ""
    part_name: str = ""
    report_timestamp: str = ""

    # Dimension comparisons
    comparisons: List[DimensionComparison] = field(default_factory=list)

    # Bend-specific results
    bend_matches: Optional[BendMatchResult] = None

    # Summary
    total_dimensions: int = 0
    measured_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0
    pass_rate: float = 0.0

    # Bend summary
    total_bends: int = 0
    bend_pass_count: int = 0
    bend_fail_count: int = 0

    def compute_summary(self):
        """Compute summary statistics."""
        self.total_dimensions = len(self.comparisons)
        self.measured_count = sum(1 for c in self.comparisons if c.status != "not_measured")
        self.pass_count = sum(1 for c in self.comparisons if c.status == "pass")
        self.fail_count = sum(1 for c in self.comparisons if c.status == "fail")
        self.warning_count = sum(1 for c in self.comparisons if c.status == "warning")

        if self.measured_count > 0:
            self.pass_rate = (self.pass_count / self.measured_count) * 100

        if self.bend_matches:
            self.total_bends = len(self.bend_matches.matches)
            self.bend_pass_count = sum(1 for m in self.bend_matches.matches if m.status == "pass")
            self.bend_fail_count = sum(1 for m in self.bend_matches.matches if m.status == "fail")

    def get_failed_dimensions(self) -> List[DimensionComparison]:
        """Get list of failed dimensions."""
        return [c for c in self.comparisons if c.status == "fail"]

    def get_worst_deviations(self, n: int = 5) -> List[DimensionComparison]:
        """Get dimensions with worst deviations."""
        measured = [c for c in self.comparisons if c.scan_deviation is not None]
        sorted_by_deviation = sorted(measured, key=lambda c: abs(c.scan_deviation), reverse=True)
        return sorted_by_deviation[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.compute_summary()

        return {
            "metadata": {
                "part_id": self.part_id,
                "part_name": self.part_name,
                "timestamp": self.report_timestamp,
            },
            "summary": {
                "total_dimensions": self.total_dimensions,
                "measured": self.measured_count,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "warnings": self.warning_count,
                "pass_rate": round(self.pass_rate, 1),
            },
            "bend_summary": {
                "total_bends": self.total_bends,
                "passed": self.bend_pass_count,
                "failed": self.bend_fail_count,
            },
            "dimensions": [c.to_dict() for c in self.comparisons],
            "bend_analysis": self.bend_matches.to_dict() if self.bend_matches else None,
            "failed_dimensions": [c.to_dict() for c in self.get_failed_dimensions()],
            "worst_deviations": [c.to_dict() for c in self.get_worst_deviations()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_table_string(self) -> str:
        """Generate ASCII table representation."""
        self.compute_summary()

        lines = []
        lines.append("=" * 100)
        lines.append(f"DIMENSION ANALYSIS REPORT - {self.part_name or self.part_id}")
        lines.append(f"Generated: {self.report_timestamp}")
        lines.append("=" * 100)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Dimensions: {self.total_dimensions}")
        lines.append(f"Measured: {self.measured_count}")
        lines.append(f"Passed: {self.pass_count}")
        lines.append(f"Failed: {self.fail_count}")
        lines.append(f"Warnings: {self.warning_count}")
        lines.append(f"Pass Rate: {self.pass_rate:.1f}%")
        lines.append("")

        if self.total_bends > 0:
            lines.append("BEND SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Total Bends: {self.total_bends}")
            lines.append(f"Passed: {self.bend_pass_count}")
            lines.append(f"Failed: {self.bend_fail_count}")
            lines.append("")

        # Dimension table (detailed format with tolerance)
        lines.append("DIMENSION DETAILS")
        lines.append("-" * 120)
        header = f"{'ID':<4} {'Type':<8} {'XLSX Value':<14} {'Tolerance':<12} {'Measured':<12} {'Deviation':<12} {'Status':<8}"
        lines.append(header)
        lines.append("-" * 120)

        for comp in self.comparisons:
            # XLSX value with unit
            if comp.xlsx_value is not None:
                xlsx_str = f"{comp.xlsx_value:.2f} {comp.unit}"
            elif comp.cad_value is not None:
                xlsx_str = f"{comp.cad_value:.2f} {comp.unit} (CAD)"
            else:
                xlsx_str = "N/A"

            # Tolerance string
            if comp.tolerance_plus == comp.tolerance_minus:
                tol_str = f"±{comp.tolerance_plus}"
            else:
                tol_str = f"+{comp.tolerance_plus}/-{comp.tolerance_minus}"

            # Measured value (prefer measured_output if available)
            measured = comp.measured_output if comp.measured_output is not None else comp.scan_value
            if measured is not None:
                measured_str = f"{measured:.2f}"
            else:
                measured_str = "N/A"

            # Deviation
            if comp.scan_deviation is not None:
                dev_str = f"{comp.scan_deviation:+.3f}"
            else:
                dev_str = "N/A"

            status_symbol = {
                "pass": "PASS",
                "fail": "FAIL",
                "warning": "WARN",
                "not_measured": "N/A",
                "pending": "PEND",
            }.get(comp.status, comp.status)

            row = f"{comp.dim_id:<4} {comp.dim_type:<8} {xlsx_str:<14} {tol_str:<12} {measured_str:<12} {dev_str:<12} {status_symbol:<8}"
            lines.append(row)

        lines.append("-" * 120)

        # Failed dimensions detail
        failed = self.get_failed_dimensions()
        if failed:
            lines.append("")
            lines.append("FAILED DIMENSIONS DETAIL")
            lines.append("-" * 60)
            for comp in failed:
                lines.append(f"  Dim {comp.dim_id} ({comp.dim_type}):")
                lines.append(f"    Reference ({comp.reference_source}): {comp.expected:.2f}{comp.unit}")
                if comp.xlsx_value and comp.cad_value is not None:
                    lines.append(f"    XLSX Spec: {comp.xlsx_value:.2f}{comp.unit}")
                lines.append(f"    Actual:   {comp.scan_value:.2f}{comp.unit}")
                lines.append(f"    Deviation: {comp.scan_deviation:+.2f}{comp.unit} ({comp.scan_deviation_percent:+.1f}%)")
                lines.append(f"    Tolerance: +{comp.tolerance_plus}/-{comp.tolerance_minus}{comp.unit}")
                lines.append("")

        return "\n".join(lines)


class DimensionReportGenerator:
    """
    Generates dimension comparison reports from XLSX specs and measurements.
    """

    def __init__(self):
        pass

    def generate_report(
        self,
        xlsx_result: DimensionParseResult,
        bend_match_result: Optional[BendMatchResult] = None,
        part_id: str = "",
        part_name: str = "",
    ) -> DimensionReport:
        """
        Generate a dimension comparison report.

        Args:
            xlsx_result: Parsed XLSX dimension specifications
            bend_match_result: Optional bend matching results with CAD/scan measurements
            part_id: Part identifier
            part_name: Part name

        Returns:
            DimensionReport
        """
        report = DimensionReport(
            part_id=part_id,
            part_name=part_name,
            report_timestamp=datetime.now().isoformat(),
            bend_matches=bend_match_result,
        )

        # Add bend dimensions from bend_match_result
        if bend_match_result and bend_match_result.matches:
            for match in bend_match_result.matches:
                comp = DimensionComparison(
                    dim_id=match.dim_id,
                    dim_type="angle",
                    description=f"Bend {match.dim_id}",
                    xlsx_value=match.dim_spec.value,  # XLSX as guide
                    cad_value=match.cad_angle,  # CAD as reference
                    scan_value=match.scan_angle,  # Scan as actual
                    unit="°",
                    tolerance_plus=match.dim_spec.tolerance_plus,
                    tolerance_minus=match.dim_spec.tolerance_minus,
                )
                comp.compute_status()
                report.comparisons.append(comp)

        # Add other dimensions (linear, radius, diameter) from XLSX
        for dim in xlsx_result.dimensions:
            # Skip angles - already handled above
            if dim.dim_type == DimensionType.ANGLE:
                continue

            comp = DimensionComparison(
                dim_id=dim.dim_id,
                dim_type=dim.dim_type.value,
                description=dim.description or f"Dimension {dim.dim_id}",
                xlsx_value=dim.value,  # XLSX as guide
                # CAD value would come from CAD geometry measurement
                # (not yet implemented for linear/radius dims)
                unit="mm",
                tolerance_plus=dim.tolerance_plus,
                tolerance_minus=dim.tolerance_minus,
            )
            # For now, these are not measured from scan
            comp.status = "not_measured"
            report.comparisons.append(comp)

        # Sort by dim_id
        report.comparisons.sort(key=lambda c: c.dim_id)

        report.compute_summary()
        return report

    def generate_from_files(
        self,
        xlsx_path: str,
        cad_bends: List[Dict],
        aligned_scan_points: Optional[Any] = None,
        part_id: str = "",
        part_name: str = "",
    ) -> DimensionReport:
        """
        Generate report directly from file paths.

        Args:
            xlsx_path: Path to XLSX dimension file
            cad_bends: List of detected CAD bends
            aligned_scan_points: Optional aligned scan points
            part_id: Part identifier
            part_name: Part name

        Returns:
            DimensionReport
        """
        from dimension_parser import parse_dimension_file
        from bend_matcher import BendMatcher

        # Parse XLSX
        xlsx_result = parse_dimension_file(xlsx_path)
        if not xlsx_result.success:
            logger.error(f"Failed to parse XLSX: {xlsx_result.error}")
            return DimensionReport(
                part_id=part_id,
                part_name=part_name,
                report_timestamp=datetime.now().isoformat(),
            )

        # Match bends
        matcher = BendMatcher()
        bend_result = matcher.match_xlsx_to_cad(xlsx_result, cad_bends)

        # Measure scan bends if available
        if aligned_scan_points is not None and bend_result.success:
            import numpy as np
            bend_result = matcher.measure_scan_bends(
                bend_result,
                np.asarray(aligned_scan_points)
            )

        # Generate report
        return self.generate_report(
            xlsx_result=xlsx_result,
            bend_match_result=bend_result,
            part_id=part_id,
            part_name=part_name,
        )
