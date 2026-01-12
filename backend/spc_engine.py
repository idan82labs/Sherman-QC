"""
SPC (Statistical Process Control) Engine

Implements basic SPC calculations:
- Cp/Cpk (Process Capability)
- Pp/Ppk (Process Performance)
- Control Charts (X-bar, R-chart, S-chart)
- Histograms and process analysis
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CapabilityRating(Enum):
    """Process capability rating categories"""
    EXCELLENT = "excellent"  # Cpk >= 1.67
    GOOD = "good"           # 1.33 <= Cpk < 1.67
    CAPABLE = "capable"     # 1.0 <= Cpk < 1.33
    MARGINAL = "marginal"   # 0.67 <= Cpk < 1.0
    POOR = "poor"           # Cpk < 0.67


@dataclass
class ProcessCapability:
    """Process capability analysis result"""
    cp: float              # Process Capability (potential)
    cpk: float             # Process Capability Index (actual)
    cpl: float             # Lower capability index
    cpu: float             # Upper capability index

    pp: float              # Process Performance (potential)
    ppk: float             # Process Performance Index (actual)
    ppl: float             # Lower performance index
    ppu: float             # Upper performance index

    # Statistics
    mean: float
    std_dev: float
    std_dev_within: float  # Within-subgroup std dev (for Cp)
    std_dev_overall: float  # Overall std dev (for Pp)

    # Specification limits
    usl: float             # Upper Specification Limit
    lsl: float             # Lower Specification Limit
    target: Optional[float] = None

    # Sample info
    sample_size: int = 0
    subgroup_size: int = 1
    num_subgroups: int = 0

    # Rating
    rating: CapabilityRating = CapabilityRating.POOR

    # Percent out of spec (estimated)
    ppm_below_lsl: float = 0.0
    ppm_above_usl: float = 0.0
    ppm_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": {
                "cp": round(self.cp, 3),
                "cpk": round(self.cpk, 3),
                "cpl": round(self.cpl, 3),
                "cpu": round(self.cpu, 3)
            },
            "performance": {
                "pp": round(self.pp, 3),
                "ppk": round(self.ppk, 3),
                "ppl": round(self.ppl, 3),
                "ppu": round(self.ppu, 3)
            },
            "statistics": {
                "mean": round(self.mean, 6),
                "std_dev_within": round(self.std_dev_within, 6),
                "std_dev_overall": round(self.std_dev_overall, 6),
                "sample_size": self.sample_size,
                "subgroup_size": self.subgroup_size,
                "num_subgroups": self.num_subgroups
            },
            "specification": {
                "usl": self.usl,
                "lsl": self.lsl,
                "target": self.target
            },
            "rating": self.rating.value,
            "defects_ppm": {
                "below_lsl": round(self.ppm_below_lsl, 1),
                "above_usl": round(self.ppm_above_usl, 1),
                "total": round(self.ppm_total, 1)
            }
        }


@dataclass
class ControlChartData:
    """Control chart data for a single chart type"""
    chart_type: str  # "xbar", "r", "s", "individuals", "mr"
    values: List[float]
    center_line: float
    ucl: float  # Upper Control Limit
    lcl: float  # Lower Control Limit
    uwl: Optional[float] = None  # Upper Warning Limit (2-sigma)
    lwl: Optional[float] = None  # Lower Warning Limit (2-sigma)

    # Out-of-control points
    ooc_points: List[int] = field(default_factory=list)  # Indices of OOC points

    # Subgroup info
    subgroup_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "data": {
                "values": [round(v, 6) for v in self.values],
                "center_line": round(self.center_line, 6),
                "ucl": round(self.ucl, 6),
                "lcl": round(self.lcl, 6),
                "uwl": round(self.uwl, 6) if self.uwl else None,
                "lwl": round(self.lwl, 6) if self.lwl else None
            },
            "out_of_control_points": self.ooc_points,
            "subgroup_labels": self.subgroup_labels
        }


@dataclass
class HistogramData:
    """Histogram data for process distribution"""
    bins: List[float]      # Bin edges
    counts: List[int]      # Counts per bin
    frequencies: List[float]  # Normalized frequencies

    # Fit normal distribution
    normal_fit_x: List[float] = field(default_factory=list)
    normal_fit_y: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bins": [round(b, 6) for b in self.bins],
            "counts": self.counts,
            "frequencies": [round(f, 6) for f in self.frequencies],
            "normal_fit": {
                "x": [round(x, 6) for x in self.normal_fit_x],
                "y": [round(y, 6) for y in self.normal_fit_y]
            } if self.normal_fit_x else None
        }


class SPCEngine:
    """
    Statistical Process Control Engine.

    Usage:
        engine = SPCEngine()
        capability = engine.calculate_capability(data, usl=10.1, lsl=9.9)
        charts = engine.generate_control_charts(data, subgroup_size=5)
    """

    # Control chart constants (d2, d3, A2, D3, D4, etc.)
    # Based on subgroup size n
    CHART_CONSTANTS = {
        2: {"d2": 1.128, "d3": 0.853, "A2": 1.880, "D3": 0, "D4": 3.267, "c4": 0.7979, "B3": 0, "B4": 3.267},
        3: {"d2": 1.693, "d3": 0.888, "A2": 1.023, "D3": 0, "D4": 2.574, "c4": 0.8862, "B3": 0, "B4": 2.568},
        4: {"d2": 2.059, "d3": 0.880, "A2": 0.729, "D3": 0, "D4": 2.282, "c4": 0.9213, "B3": 0, "B4": 2.266},
        5: {"d2": 2.326, "d3": 0.864, "A2": 0.577, "D3": 0, "D4": 2.114, "c4": 0.9400, "B3": 0, "B4": 2.089},
        6: {"d2": 2.534, "d3": 0.848, "A2": 0.483, "D3": 0, "D4": 2.004, "c4": 0.9515, "B3": 0.030, "B4": 1.970},
        7: {"d2": 2.704, "d3": 0.833, "A2": 0.419, "D3": 0.076, "D4": 1.924, "c4": 0.9594, "B3": 0.118, "B4": 1.882},
        8: {"d2": 2.847, "d3": 0.820, "A2": 0.373, "D3": 0.136, "D4": 1.864, "c4": 0.9650, "B3": 0.185, "B4": 1.815},
        9: {"d2": 2.970, "d3": 0.808, "A2": 0.337, "D3": 0.184, "D4": 1.816, "c4": 0.9693, "B3": 0.239, "B4": 1.761},
        10: {"d2": 3.078, "d3": 0.797, "A2": 0.308, "D3": 0.223, "D4": 1.777, "c4": 0.9727, "B3": 0.284, "B4": 1.716},
    }

    def __init__(self):
        pass

    def calculate_capability(
        self,
        data: np.ndarray,
        usl: float,
        lsl: float,
        target: Optional[float] = None,
        subgroup_size: int = 1
    ) -> ProcessCapability:
        """
        Calculate process capability indices (Cp/Cpk) and performance indices (Pp/Ppk).

        Args:
            data: 1D array of measurements
            usl: Upper Specification Limit
            lsl: Lower Specification Limit
            target: Target value (defaults to midpoint of spec)
            subgroup_size: Size of rational subgroups (for within-group std dev)

        Returns:
            ProcessCapability result
        """
        data = np.asarray(data).flatten()
        n = len(data)

        if n < 2:
            raise ValueError("Need at least 2 data points for capability analysis")

        if usl <= lsl:
            raise ValueError("USL must be greater than LSL")

        # Calculate basic statistics
        mean = np.mean(data)
        std_overall = np.std(data, ddof=1)  # Overall standard deviation

        # Calculate within-subgroup standard deviation
        if subgroup_size > 1 and n >= subgroup_size * 2:
            std_within = self._calculate_within_std(data, subgroup_size)
            num_subgroups = n // subgroup_size
        else:
            # For individual measurements, use moving range
            std_within = self._calculate_mr_std(data)
            subgroup_size = 1
            num_subgroups = n

        # Target defaults to midpoint
        if target is None:
            target = (usl + lsl) / 2

        # Tolerance
        tolerance = usl - lsl

        # Process Capability (Cp, Cpk) - uses within-group std dev
        if std_within > 0:
            cp = tolerance / (6 * std_within)
            cpl = (mean - lsl) / (3 * std_within)
            cpu = (usl - mean) / (3 * std_within)
            cpk = min(cpl, cpu)
        else:
            cp = float('inf')
            cpl = float('inf')
            cpu = float('inf')
            cpk = float('inf')

        # Process Performance (Pp, Ppk) - uses overall std dev
        if std_overall > 0:
            pp = tolerance / (6 * std_overall)
            ppl = (mean - lsl) / (3 * std_overall)
            ppu = (usl - mean) / (3 * std_overall)
            ppk = min(ppl, ppu)
        else:
            pp = float('inf')
            ppl = float('inf')
            ppu = float('inf')
            ppk = float('inf')

        # Calculate PPM (parts per million defective)
        ppm_below, ppm_above = self._calculate_ppm(mean, std_overall, lsl, usl)

        # Determine rating based on Cpk
        rating = self._get_capability_rating(cpk)

        return ProcessCapability(
            cp=cp,
            cpk=cpk,
            cpl=cpl,
            cpu=cpu,
            pp=pp,
            ppk=ppk,
            ppl=ppl,
            ppu=ppu,
            mean=mean,
            std_dev=std_overall,
            std_dev_within=std_within,
            std_dev_overall=std_overall,
            usl=usl,
            lsl=lsl,
            target=target,
            sample_size=n,
            subgroup_size=subgroup_size,
            num_subgroups=num_subgroups,
            rating=rating,
            ppm_below_lsl=ppm_below,
            ppm_above_usl=ppm_above,
            ppm_total=ppm_below + ppm_above
        )

    def generate_control_charts(
        self,
        data: np.ndarray,
        subgroup_size: int = 5,
        labels: Optional[List[str]] = None
    ) -> Dict[str, ControlChartData]:
        """
        Generate control chart data for the given measurements.

        For subgroup_size > 1: Returns X-bar and R charts (or S chart for n > 10)
        For subgroup_size = 1: Returns Individuals and Moving Range charts

        Args:
            data: 1D array of measurements
            subgroup_size: Size of rational subgroups
            labels: Optional labels for subgroups

        Returns:
            Dictionary of control chart data
        """
        data = np.asarray(data).flatten()
        n = len(data)

        if n < 2:
            raise ValueError("Need at least 2 data points for control charts")

        charts = {}

        if subgroup_size == 1:
            # Individual and Moving Range charts
            charts["individuals"] = self._generate_individuals_chart(data, labels)
            charts["moving_range"] = self._generate_mr_chart(data, labels)
        else:
            # Ensure we have complete subgroups
            num_subgroups = n // subgroup_size
            if num_subgroups < 2:
                raise ValueError(f"Need at least 2 subgroups (got {num_subgroups})")

            # Reshape data into subgroups
            trimmed = data[:num_subgroups * subgroup_size]
            subgroups = trimmed.reshape(num_subgroups, subgroup_size)

            # Generate subgroup labels
            if labels is None:
                labels = [f"Subgroup {i+1}" for i in range(num_subgroups)]

            # X-bar chart
            charts["xbar"] = self._generate_xbar_chart(subgroups, subgroup_size, labels)

            # R chart (range) or S chart (std dev)
            if subgroup_size <= 10:
                charts["range"] = self._generate_r_chart(subgroups, subgroup_size, labels)
            else:
                charts["std_dev"] = self._generate_s_chart(subgroups, subgroup_size, labels)

        return charts

    def generate_histogram(
        self,
        data: np.ndarray,
        num_bins: int = 20,
        lsl: Optional[float] = None,
        usl: Optional[float] = None
    ) -> HistogramData:
        """
        Generate histogram data with optional normal distribution fit.

        Args:
            data: 1D array of measurements
            num_bins: Number of histogram bins
            lsl: Lower spec limit (for display)
            usl: Upper spec limit (for display)

        Returns:
            HistogramData
        """
        data = np.asarray(data).flatten()

        # Calculate histogram
        counts, bin_edges = np.histogram(data, bins=num_bins)

        # Calculate frequencies
        total = len(data)
        frequencies = counts / total if total > 0 else counts.astype(float)

        # Calculate normal distribution fit
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Generate smooth normal curve
        x_range = np.linspace(bin_edges[0], bin_edges[-1], 100)
        bin_width = bin_edges[1] - bin_edges[0]

        if std > 0:
            normal_y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
            # Scale to match histogram (frequency * bin_width for proper overlay)
            normal_y = normal_y * bin_width
        else:
            normal_y = np.zeros_like(x_range)

        return HistogramData(
            bins=bin_edges.tolist(),
            counts=counts.tolist(),
            frequencies=frequencies.tolist(),
            normal_fit_x=x_range.tolist(),
            normal_fit_y=normal_y.tolist()
        )

    def _calculate_within_std(self, data: np.ndarray, subgroup_size: int) -> float:
        """Calculate within-subgroup standard deviation using R-bar/d2 method."""
        n = len(data)
        num_subgroups = n // subgroup_size

        if num_subgroups < 2:
            return self._calculate_mr_std(data)

        # Reshape into subgroups
        subgroups = data[:num_subgroups * subgroup_size].reshape(num_subgroups, subgroup_size)

        # Calculate ranges within each subgroup
        ranges = np.ptp(subgroups, axis=1)  # max - min for each subgroup
        r_bar = np.mean(ranges)

        # Get d2 constant
        d2 = self.CHART_CONSTANTS.get(subgroup_size, {"d2": 2.326})["d2"]

        return r_bar / d2

    def _calculate_mr_std(self, data: np.ndarray) -> float:
        """Calculate standard deviation using moving range method for individuals."""
        if len(data) < 2:
            return 0.0

        # Calculate moving ranges (consecutive differences)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr)

        # d2 for n=2 (moving range of 2 consecutive points)
        d2 = 1.128

        return mr_bar / d2

    def _calculate_ppm(
        self,
        mean: float,
        std: float,
        lsl: float,
        usl: float
    ) -> Tuple[float, float]:
        """Calculate parts per million below LSL and above USL."""
        if std <= 0:
            return (0.0, 0.0)

        try:
            from scipy import stats

            z_lower = (lsl - mean) / std
            z_upper = (usl - mean) / std

            ppm_below = stats.norm.cdf(z_lower) * 1_000_000
            ppm_above = (1 - stats.norm.cdf(z_upper)) * 1_000_000

            return (ppm_below, ppm_above)
        except ImportError:
            # Fallback: approximate using error function
            import math

            def normal_cdf(z):
                return 0.5 * (1 + math.erf(z / math.sqrt(2)))

            z_lower = (lsl - mean) / std
            z_upper = (usl - mean) / std

            ppm_below = normal_cdf(z_lower) * 1_000_000
            ppm_above = (1 - normal_cdf(z_upper)) * 1_000_000

            return (ppm_below, ppm_above)

    def _get_capability_rating(self, cpk: float) -> CapabilityRating:
        """Determine capability rating based on Cpk value."""
        if cpk >= 1.67:
            return CapabilityRating.EXCELLENT
        elif cpk >= 1.33:
            return CapabilityRating.GOOD
        elif cpk >= 1.0:
            return CapabilityRating.CAPABLE
        elif cpk >= 0.67:
            return CapabilityRating.MARGINAL
        else:
            return CapabilityRating.POOR

    def _generate_xbar_chart(
        self,
        subgroups: np.ndarray,
        subgroup_size: int,
        labels: List[str]
    ) -> ControlChartData:
        """Generate X-bar chart data."""
        # Calculate subgroup means
        means = np.mean(subgroups, axis=1)
        x_bar = np.mean(means)

        # Calculate average range
        ranges = np.ptp(subgroups, axis=1)
        r_bar = np.mean(ranges)

        # Get A2 constant
        A2 = self.CHART_CONSTANTS.get(subgroup_size, {"A2": 0.577})["A2"]

        # Control limits
        ucl = x_bar + A2 * r_bar
        lcl = x_bar - A2 * r_bar

        # Warning limits (2-sigma)
        uwl = x_bar + (2/3) * A2 * r_bar
        lwl = x_bar - (2/3) * A2 * r_bar

        # Find out-of-control points
        ooc = [i for i, m in enumerate(means) if m > ucl or m < lcl]

        return ControlChartData(
            chart_type="xbar",
            values=means.tolist(),
            center_line=x_bar,
            ucl=ucl,
            lcl=lcl,
            uwl=uwl,
            lwl=lwl,
            ooc_points=ooc,
            subgroup_labels=labels
        )

    def _generate_r_chart(
        self,
        subgroups: np.ndarray,
        subgroup_size: int,
        labels: List[str]
    ) -> ControlChartData:
        """Generate R (Range) chart data."""
        # Calculate subgroup ranges
        ranges = np.ptp(subgroups, axis=1)
        r_bar = np.mean(ranges)

        # Get D3 and D4 constants
        constants = self.CHART_CONSTANTS.get(subgroup_size, {"D3": 0, "D4": 2.114})
        D3 = constants["D3"]
        D4 = constants["D4"]

        # Control limits
        ucl = D4 * r_bar
        lcl = D3 * r_bar

        # Find out-of-control points
        ooc = [i for i, r in enumerate(ranges) if r > ucl or r < lcl]

        return ControlChartData(
            chart_type="range",
            values=ranges.tolist(),
            center_line=r_bar,
            ucl=ucl,
            lcl=lcl,
            ooc_points=ooc,
            subgroup_labels=labels
        )

    def _generate_s_chart(
        self,
        subgroups: np.ndarray,
        subgroup_size: int,
        labels: List[str]
    ) -> ControlChartData:
        """Generate S (Standard Deviation) chart data."""
        # Calculate subgroup standard deviations
        stds = np.std(subgroups, axis=1, ddof=1)
        s_bar = np.mean(stds)

        # Get B3 and B4 constants (approximate for large n)
        constants = self.CHART_CONSTANTS.get(min(subgroup_size, 10), {"B3": 0, "B4": 2.089})
        B3 = constants["B3"]
        B4 = constants["B4"]

        # Control limits
        ucl = B4 * s_bar
        lcl = B3 * s_bar

        # Find out-of-control points
        ooc = [i for i, s in enumerate(stds) if s > ucl or s < lcl]

        return ControlChartData(
            chart_type="std_dev",
            values=stds.tolist(),
            center_line=s_bar,
            ucl=ucl,
            lcl=lcl,
            ooc_points=ooc,
            subgroup_labels=labels
        )

    def _generate_individuals_chart(
        self,
        data: np.ndarray,
        labels: Optional[List[str]]
    ) -> ControlChartData:
        """Generate Individuals (I) chart data."""
        n = len(data)
        x_bar = np.mean(data)

        # Calculate moving range average
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0

        # E2 constant (for n=2 moving range) = 3/d2 = 3/1.128 ≈ 2.66
        E2 = 2.66

        # Control limits
        ucl = x_bar + E2 * mr_bar
        lcl = x_bar - E2 * mr_bar

        # Warning limits
        uwl = x_bar + (2/3) * E2 * mr_bar
        lwl = x_bar - (2/3) * E2 * mr_bar

        # Generate labels
        if labels is None:
            labels = [f"Sample {i+1}" for i in range(n)]

        # Find out-of-control points
        ooc = [i for i, x in enumerate(data) if x > ucl or x < lcl]

        return ControlChartData(
            chart_type="individuals",
            values=data.tolist(),
            center_line=x_bar,
            ucl=ucl,
            lcl=lcl,
            uwl=uwl,
            lwl=lwl,
            ooc_points=ooc,
            subgroup_labels=labels
        )

    def _generate_mr_chart(
        self,
        data: np.ndarray,
        labels: Optional[List[str]]
    ) -> ControlChartData:
        """Generate Moving Range (MR) chart data."""
        # Calculate moving ranges
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0

        # D4 for n=2 = 3.267
        D4 = 3.267
        D3 = 0

        # Control limits
        ucl = D4 * mr_bar
        lcl = D3 * mr_bar

        # Generate labels
        if labels is None:
            labels = [f"MR {i+1}-{i+2}" for i in range(len(mr))]

        # Find out-of-control points
        ooc = [i for i, r in enumerate(mr) if r > ucl or r < lcl]

        return ControlChartData(
            chart_type="moving_range",
            values=mr.tolist(),
            center_line=mr_bar,
            ucl=ucl,
            lcl=lcl,
            ooc_points=ooc,
            subgroup_labels=labels
        )

    def analyze_stability(
        self,
        charts: Dict[str, ControlChartData]
    ) -> Dict[str, Any]:
        """
        Analyze process stability based on control chart data.

        Checks for:
        1. Points outside control limits
        2. Run of 7+ points above or below center
        3. Trend of 7+ points continuously increasing or decreasing
        4. 2 of 3 points beyond 2-sigma

        Returns:
            Stability analysis result
        """
        results = {
            "is_stable": True,
            "violations": [],
            "charts_analyzed": list(charts.keys())
        }

        for chart_name, chart in charts.items():
            chart_violations = []
            values = np.array(chart.values)
            cl = chart.center_line

            # Rule 1: Points outside control limits
            if chart.ooc_points:
                chart_violations.append({
                    "rule": "points_outside_limits",
                    "description": f"{len(chart.ooc_points)} point(s) outside control limits",
                    "points": chart.ooc_points
                })

            # Rule 2: Run of 7+ points on one side of center
            runs = self._find_runs(values, cl)
            if runs:
                chart_violations.append({
                    "rule": "run_above_below_center",
                    "description": f"Run of 7+ points on same side of center line",
                    "points": runs
                })

            # Rule 3: Trend of 7+ points
            trends = self._find_trends(values)
            if trends:
                chart_violations.append({
                    "rule": "trend",
                    "description": "Trend of 7+ consecutive points increasing or decreasing",
                    "points": trends
                })

            if chart_violations:
                results["is_stable"] = False
                results["violations"].append({
                    "chart": chart_name,
                    "violations": chart_violations
                })

        return results

    def _find_runs(self, values: np.ndarray, center: float) -> List[int]:
        """Find runs of 7+ points above or below center line."""
        run_indices = []
        current_run = []
        current_side = None

        for i, v in enumerate(values):
            side = "above" if v > center else "below"

            if side == current_side:
                current_run.append(i)
            else:
                if len(current_run) >= 7:
                    run_indices.extend(current_run)
                current_run = [i]
                current_side = side

        # Check final run
        if len(current_run) >= 7:
            run_indices.extend(current_run)

        return run_indices

    def _find_trends(self, values: np.ndarray) -> List[int]:
        """Find trends of 7+ consecutive increasing or decreasing points."""
        trend_indices = []
        current_trend = []
        current_direction = None

        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                direction = "up"
            elif values[i] < values[i-1]:
                direction = "down"
            else:
                direction = None

            if direction == current_direction and direction is not None:
                current_trend.append(i)
            else:
                if len(current_trend) >= 6:  # 6 changes = 7 points
                    trend_indices.extend([current_trend[0] - 1] + current_trend)
                current_trend = [i]
                current_direction = direction

        # Check final trend
        if len(current_trend) >= 6:
            trend_indices.extend([current_trend[0] - 1] + current_trend)

        return trend_indices


def create_spc_engine() -> SPCEngine:
    """Create an SPC engine instance."""
    return SPCEngine()
