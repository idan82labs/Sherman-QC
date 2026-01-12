"""
Tests for SPC (Statistical Process Control) Engine
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from spc_engine import (
    SPCEngine, ProcessCapability, ControlChartData,
    HistogramData, CapabilityRating, create_spc_engine
)


class TestSPCEngineCreation:
    """Test SPC engine initialization"""

    def test_create_engine(self):
        """Test engine creation"""
        engine = create_spc_engine()
        assert engine is not None
        assert isinstance(engine, SPCEngine)


class TestProcessCapability:
    """Test process capability calculations"""

    @pytest.fixture
    def engine(self):
        return SPCEngine()

    def test_excellent_capability(self, engine):
        """Test process with excellent capability (Cpk >= 1.67)"""
        # Generate data centered at 10 with very small std dev
        np.random.seed(42)
        data = np.random.normal(10.0, 0.03, 100)  # Mean=10, std=0.03

        result = engine.calculate_capability(
            data,
            usl=10.5,  # +0.5 from nominal
            lsl=9.5,   # -0.5 from nominal
            target=10.0
        )

        assert result.cpk >= 1.67
        assert result.rating == CapabilityRating.EXCELLENT
        assert result.ppm_total < 1  # Should have almost no defects

    def test_good_capability(self, engine):
        """Test process with good capability (1.33 <= Cpk < 1.67)"""
        np.random.seed(42)
        # Adjust std dev to achieve Cpk in target range
        data = np.random.normal(10.0, 0.075, 100)

        result = engine.calculate_capability(
            data,
            usl=10.3,
            lsl=9.7,
            target=10.0
        )

        assert 1.33 <= result.cpk < 1.67
        assert result.rating == CapabilityRating.GOOD

    def test_capable_process(self, engine):
        """Test process with acceptable capability (1.0 <= Cpk < 1.33)"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.08, 100)

        result = engine.calculate_capability(
            data,
            usl=10.3,
            lsl=9.7,
            target=10.0
        )

        assert 1.0 <= result.cpk < 1.33
        assert result.rating == CapabilityRating.CAPABLE

    def test_poor_capability(self, engine):
        """Test process with poor capability (Cpk < 0.67)"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.3, 100)  # Large variation

        result = engine.calculate_capability(
            data,
            usl=10.3,
            lsl=9.7,
            target=10.0
        )

        assert result.cpk < 0.67
        assert result.rating == CapabilityRating.POOR
        assert result.ppm_total > 10000  # Should have many defects

    def test_off_center_process(self, engine):
        """Test process that is off-center (Cp > Cpk)"""
        np.random.seed(42)
        data = np.random.normal(10.15, 0.05, 100)  # Mean shifted up

        result = engine.calculate_capability(
            data,
            usl=10.3,
            lsl=9.7,
            target=10.0
        )

        # Cp should be good (small variation)
        assert result.cp >= 1.33
        # But Cpk should be lower due to off-center
        assert result.cpk < result.cp
        # Should have more defects above USL than below LSL
        assert result.ppm_above_usl > result.ppm_below_lsl

    def test_capability_with_subgroups(self, engine):
        """Test capability with rational subgroups"""
        np.random.seed(42)
        # Generate 100 samples in 20 subgroups of 5
        data = np.random.normal(10.0, 0.05, 100)

        result = engine.calculate_capability(
            data,
            usl=10.3,
            lsl=9.7,
            subgroup_size=5
        )

        assert result.sample_size == 100
        assert result.subgroup_size == 5
        assert result.num_subgroups == 20
        # Within-group std should be similar to or less than overall
        assert result.std_dev_within <= result.std_dev_overall * 1.2

    def test_capability_to_dict(self, engine):
        """Test capability result serialization"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.05, 50)

        result = engine.calculate_capability(data, usl=10.3, lsl=9.7)
        result_dict = result.to_dict()

        assert "capability" in result_dict
        assert "cpk" in result_dict["capability"]
        assert "performance" in result_dict
        assert "statistics" in result_dict
        assert "rating" in result_dict
        assert "defects_ppm" in result_dict

    def test_capability_invalid_inputs(self, engine):
        """Test capability with invalid inputs"""
        # Too few data points
        with pytest.raises(ValueError):
            engine.calculate_capability([1.0], usl=1.5, lsl=0.5)

        # USL <= LSL
        with pytest.raises(ValueError):
            engine.calculate_capability([1.0, 1.1, 1.2], usl=0.5, lsl=1.5)


class TestControlCharts:
    """Test control chart generation"""

    @pytest.fixture
    def engine(self):
        return SPCEngine()

    def test_xbar_r_chart(self, engine):
        """Test X-bar and R chart generation"""
        np.random.seed(42)
        # 50 samples in 10 subgroups of 5
        data = np.random.normal(10.0, 0.1, 50)

        charts = engine.generate_control_charts(data, subgroup_size=5)

        assert "xbar" in charts
        assert "range" in charts

        xbar = charts["xbar"]
        assert xbar.chart_type == "xbar"
        assert len(xbar.values) == 10  # 10 subgroups
        assert xbar.ucl > xbar.center_line > xbar.lcl
        assert len(xbar.subgroup_labels) == 10

        r_chart = charts["range"]
        assert r_chart.chart_type == "range"
        assert len(r_chart.values) == 10
        assert r_chart.ucl > r_chart.center_line >= r_chart.lcl

    def test_individuals_mr_chart(self, engine):
        """Test Individuals and Moving Range chart generation"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.1, 30)

        charts = engine.generate_control_charts(data, subgroup_size=1)

        assert "individuals" in charts
        assert "moving_range" in charts

        i_chart = charts["individuals"]
        assert i_chart.chart_type == "individuals"
        assert len(i_chart.values) == 30
        assert i_chart.ucl > i_chart.center_line > i_chart.lcl

        mr_chart = charts["moving_range"]
        assert mr_chart.chart_type == "moving_range"
        assert len(mr_chart.values) == 29  # n-1 moving ranges
        assert mr_chart.ucl > mr_chart.center_line >= mr_chart.lcl

    def test_ooc_detection(self, engine):
        """Test out-of-control point detection"""
        # Create data with an obvious outlier
        data = [10.0] * 20 + [15.0] + [10.0] * 9  # Outlier at index 20

        charts = engine.generate_control_charts(data, subgroup_size=1)

        i_chart = charts["individuals"]
        # The outlier should be detected
        assert 20 in i_chart.ooc_points

    def test_chart_to_dict(self, engine):
        """Test control chart serialization"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.1, 50)

        charts = engine.generate_control_charts(data, subgroup_size=5)
        xbar_dict = charts["xbar"].to_dict()

        assert "chart_type" in xbar_dict
        assert "data" in xbar_dict
        assert "values" in xbar_dict["data"]
        assert "center_line" in xbar_dict["data"]
        assert "ucl" in xbar_dict["data"]
        assert "lcl" in xbar_dict["data"]

    def test_insufficient_subgroups(self, engine):
        """Test with insufficient data for subgroups"""
        data = [1.0, 2.0, 3.0, 4.0]  # Only 4 points

        with pytest.raises(ValueError):
            engine.generate_control_charts(data, subgroup_size=5)


class TestHistogram:
    """Test histogram generation"""

    @pytest.fixture
    def engine(self):
        return SPCEngine()

    def test_histogram_generation(self, engine):
        """Test basic histogram generation"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.1, 100)

        hist = engine.generate_histogram(data, num_bins=15)

        assert len(hist.bins) == 16  # n+1 bin edges for n bins
        assert len(hist.counts) == 15
        assert len(hist.frequencies) == 15
        assert sum(hist.counts) == 100
        assert abs(sum(hist.frequencies) - 1.0) < 0.001

    def test_histogram_normal_fit(self, engine):
        """Test histogram with normal distribution fit"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.5, 200)

        hist = engine.generate_histogram(data, num_bins=20)

        assert len(hist.normal_fit_x) > 0
        assert len(hist.normal_fit_y) > 0
        assert len(hist.normal_fit_x) == len(hist.normal_fit_y)

    def test_histogram_to_dict(self, engine):
        """Test histogram serialization"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.1, 50)

        hist = engine.generate_histogram(data)
        hist_dict = hist.to_dict()

        assert "bins" in hist_dict
        assert "counts" in hist_dict
        assert "frequencies" in hist_dict
        assert "normal_fit" in hist_dict


class TestStabilityAnalysis:
    """Test process stability analysis"""

    @pytest.fixture
    def engine(self):
        return SPCEngine()

    def test_stable_process(self, engine):
        """Test analysis of a stable process"""
        np.random.seed(42)
        data = np.random.normal(10.0, 0.1, 50)

        charts = engine.generate_control_charts(data, subgroup_size=5)
        stability = engine.analyze_stability(charts)

        # Stable process should have no violations
        assert stability["is_stable"] == True
        assert len(stability["violations"]) == 0

    def test_unstable_process_ooc(self, engine):
        """Test detection of out-of-control points"""
        # Create data with obvious out-of-control points
        np.random.seed(42)
        data = list(np.random.normal(10.0, 0.05, 45))
        data.extend([12.0, 12.0, 12.0, 12.0, 12.0])  # Add OOC points

        charts = engine.generate_control_charts(data, subgroup_size=5)
        stability = engine.analyze_stability(charts)

        assert stability["is_stable"] == False
        assert any(
            "points_outside_limits" in str(v)
            for v in stability["violations"]
        )

    def test_run_detection(self, engine):
        """Test detection of runs above/below center line"""
        # Create data with a run of 8 points above center
        data = [10.0, 10.0, 10.0, 10.0, 10.0,  # Subgroup 1
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 2 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 3 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 4 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 5 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 6 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 7 - above
                10.1, 10.1, 10.1, 10.1, 10.1,  # Subgroup 8 - above
                10.0, 10.0, 10.0, 10.0, 10.0,  # Subgroup 9
                10.0, 10.0, 10.0, 10.0, 10.0]  # Subgroup 10

        charts = engine.generate_control_charts(data, subgroup_size=5)
        stability = engine.analyze_stability(charts)

        # Should detect the run
        assert stability["is_stable"] == False


class TestCapabilityRating:
    """Test capability rating thresholds"""

    def test_rating_thresholds(self):
        """Verify rating thresholds are correct"""
        engine = SPCEngine()

        # Test each threshold
        assert engine._get_capability_rating(2.0) == CapabilityRating.EXCELLENT
        assert engine._get_capability_rating(1.67) == CapabilityRating.EXCELLENT
        assert engine._get_capability_rating(1.5) == CapabilityRating.GOOD
        assert engine._get_capability_rating(1.33) == CapabilityRating.GOOD
        assert engine._get_capability_rating(1.2) == CapabilityRating.CAPABLE
        assert engine._get_capability_rating(1.0) == CapabilityRating.CAPABLE
        assert engine._get_capability_rating(0.8) == CapabilityRating.MARGINAL
        assert engine._get_capability_rating(0.67) == CapabilityRating.MARGINAL
        assert engine._get_capability_rating(0.5) == CapabilityRating.POOR


class TestChartConstants:
    """Test that chart constants are properly defined"""

    def test_constants_available(self):
        """Test that common subgroup sizes have constants"""
        engine = SPCEngine()

        for n in range(2, 11):
            assert n in engine.CHART_CONSTANTS
            constants = engine.CHART_CONSTANTS[n]
            assert "d2" in constants
            assert "A2" in constants
            assert "D3" in constants
            assert "D4" in constants


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
