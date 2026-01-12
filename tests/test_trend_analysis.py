"""
Tests for Trend Analysis and Tool Wear Prediction
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from trend_analysis import (
    TrendType,
    AlertLevel,
    WearStage,
    DataPoint,
    TrendResult,
    ChangePoint,
    WearPrediction,
    TrendAlert,
    TrendAnalyzer,
    analyze_measurement_trend,
    predict_maintenance,
)


class TestTrendType:
    """Test TrendType enum"""

    def test_trend_types(self):
        """Test all trend types defined"""
        assert TrendType.STABLE.value == "stable"
        assert TrendType.INCREASING.value == "increasing"
        assert TrendType.DECREASING.value == "decreasing"
        assert TrendType.CYCLIC.value == "cyclic"
        assert TrendType.STEP_CHANGE.value == "step_change"
        assert TrendType.DRIFT.value == "drift"


class TestAlertLevel:
    """Test AlertLevel enum"""

    def test_alert_levels(self):
        """Test all alert levels defined"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"


class TestWearStage:
    """Test WearStage enum"""

    def test_wear_stages(self):
        """Test all wear stages defined"""
        assert WearStage.NEW.value == "new"
        assert WearStage.NORMAL.value == "normal"
        assert WearStage.MODERATE_WEAR.value == "moderate_wear"
        assert WearStage.SIGNIFICANT_WEAR.value == "significant_wear"
        assert WearStage.CRITICAL_WEAR.value == "critical_wear"
        assert WearStage.END_OF_LIFE.value == "end_of_life"


class TestDataPoint:
    """Test DataPoint dataclass"""

    def test_data_point_creation(self):
        """Test creating data point"""
        now = datetime.now()
        dp = DataPoint(
            timestamp=now,
            value=0.5,
            job_id="job_1",
            tool_id="T001"
        )
        assert dp.value == 0.5
        assert dp.tool_id == "T001"

    def test_data_point_to_dict(self):
        """Test data point serialization"""
        dp = DataPoint(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            value=0.25
        )
        data = dp.to_dict()
        assert "timestamp" in data
        assert data["value"] == 0.25


class TestTrendResult:
    """Test TrendResult dataclass"""

    def test_trend_result_creation(self):
        """Test creating trend result"""
        result = TrendResult(
            trend_type=TrendType.INCREASING,
            slope=0.01,
            r_squared=0.85,
            confidence=0.9
        )
        assert result.trend_type == TrendType.INCREASING
        assert result.slope == 0.01
        assert result.r_squared == 0.85

    def test_trend_result_to_dict(self):
        """Test trend result serialization"""
        result = TrendResult(
            trend_type=TrendType.STABLE,
            slope=0.001,
            r_squared=0.5,
            confidence=0.6,
            mean=10.0,
            std=0.5
        )
        data = result.to_dict()
        assert data["trend_type"] == "stable"
        assert data["statistics"]["mean"] == 10.0


class TestChangePoint:
    """Test ChangePoint dataclass"""

    def test_change_point_creation(self):
        """Test creating change point"""
        cp = ChangePoint(
            timestamp=datetime.now(),
            index=50,
            before_mean=0.1,
            after_mean=0.3,
            magnitude=0.2,
            confidence=0.95
        )
        assert cp.magnitude == 0.2
        assert cp.confidence == 0.95

    def test_change_point_to_dict(self):
        """Test change point serialization"""
        cp = ChangePoint(
            timestamp=datetime(2024, 1, 1),
            index=25,
            before_mean=0.0,
            after_mean=0.5,
            magnitude=0.5,
            confidence=0.8
        )
        data = cp.to_dict()
        assert data["index"] == 25
        assert data["magnitude"] == 0.5


class TestWearPrediction:
    """Test WearPrediction dataclass"""

    def test_wear_prediction_creation(self):
        """Test creating wear prediction"""
        pred = WearPrediction(
            tool_id="T001",
            current_stage=WearStage.MODERATE_WEAR,
            wear_rate=0.01,
            estimated_remaining_life=48.0,
            confidence=0.85,
            recommended_action="Monitor closely",
            urgency=AlertLevel.INFO
        )
        assert pred.tool_id == "T001"
        assert pred.estimated_remaining_life == 48.0

    def test_wear_prediction_to_dict(self):
        """Test wear prediction serialization"""
        pred = WearPrediction(
            tool_id="T002",
            current_stage=WearStage.CRITICAL_WEAR,
            wear_rate=0.05,
            estimated_remaining_life=8.0,
            confidence=0.9,
            recommended_action="Replace soon",
            urgency=AlertLevel.CRITICAL
        )
        data = pred.to_dict()
        assert data["tool_id"] == "T002"
        assert data["current_stage"] == "critical_wear"
        assert data["urgency"] == "critical"


class TestTrendAlert:
    """Test TrendAlert dataclass"""

    def test_alert_creation(self):
        """Test creating trend alert"""
        alert = TrendAlert(
            alert_id="alert_001",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            metric="deviation",
            tool_id="T001"
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.metric == "deviation"


class TestTrendAnalyzer:
    """Test TrendAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer(min_data_points=5)

    @pytest.fixture
    def stable_data(self):
        """Generate stable data with minimal variation"""
        now = datetime.now()
        np.random.seed(42)
        return [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=10.0 + np.random.normal(0, 0.1)
            )
            for i in range(50)
        ]

    @pytest.fixture
    def increasing_data(self):
        """Generate data with increasing trend"""
        now = datetime.now()
        np.random.seed(42)
        return [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=0.1 + 0.02 * i + np.random.normal(0, 0.05)
            )
            for i in range(50)
        ]

    @pytest.fixture
    def decreasing_data(self):
        """Generate data with decreasing trend"""
        now = datetime.now()
        np.random.seed(42)
        return [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=10.0 - 0.05 * i + np.random.normal(0, 0.1)
            )
            for i in range(50)
        ]

    @pytest.fixture
    def step_change_data(self):
        """Generate data with step change"""
        now = datetime.now()
        np.random.seed(42)
        data = []
        for i in range(50):
            base = 0.1 if i < 25 else 0.5  # Step change at midpoint
            data.append(DataPoint(
                timestamp=now + timedelta(hours=i),
                value=base + np.random.normal(0, 0.05)
            ))
        return data

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creation"""
        assert analyzer.min_data_points == 5
        assert analyzer.significance_level == 0.05

    def test_insufficient_data(self, analyzer):
        """Test handling of insufficient data"""
        data = [
            DataPoint(datetime.now(), 1.0),
            DataPoint(datetime.now(), 1.1)
        ]
        result = analyzer.analyze_trend(data)
        assert result.trend_type == TrendType.STABLE
        assert result.confidence == 0.0

    def test_detect_stable_trend(self, analyzer, stable_data):
        """Test detection of stable trend"""
        result = analyzer.analyze_trend(stable_data)
        assert result.trend_type == TrendType.STABLE
        assert abs(result.slope) < 0.01

    def test_detect_increasing_trend(self, analyzer, increasing_data):
        """Test detection of increasing trend"""
        result = analyzer.analyze_trend(increasing_data)
        assert result.trend_type == TrendType.INCREASING
        assert result.slope > 0

    def test_detect_decreasing_trend(self, analyzer, decreasing_data):
        """Test detection of decreasing trend"""
        result = analyzer.analyze_trend(decreasing_data)
        assert result.trend_type == TrendType.DECREASING
        assert result.slope < 0

    def test_prediction_with_threshold(self, analyzer, increasing_data):
        """Test prediction with threshold"""
        result = analyzer.analyze_trend(increasing_data, threshold=1.5)
        assert result.predicted_value is not None
        assert result.prediction_interval is not None
        # Time to threshold should be calculable for increasing trend
        if result.time_to_threshold is not None:
            assert result.time_to_threshold > 0

    def test_statistics_calculation(self, analyzer, stable_data):
        """Test statistics calculation"""
        result = analyzer.analyze_trend(stable_data)
        assert result.mean > 0
        assert result.std > 0
        assert result.min_value <= result.mean <= result.max_value


class TestChangePointDetection:
    """Test change point detection"""

    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer(min_data_points=5)

    def test_detect_step_change(self, analyzer):
        """Test detection of step change"""
        now = datetime.now()
        np.random.seed(42)
        data = []
        for i in range(60):
            base = 0.1 if i < 30 else 0.6
            data.append(DataPoint(
                timestamp=now + timedelta(hours=i),
                value=base + np.random.normal(0, 0.02)
            ))

        change_points = analyzer.detect_change_points(data)
        # Should detect at least one change point near index 30
        assert len(change_points) >= 1

    def test_no_change_in_stable(self, analyzer):
        """Test no change points in stable data"""
        now = datetime.now()
        np.random.seed(42)
        data = [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=1.0 + np.random.normal(0, 0.01)
            )
            for i in range(50)
        ]

        change_points = analyzer.detect_change_points(data)
        # Should detect few or no change points
        assert len(change_points) <= 1


class TestToolWearPrediction:
    """Test tool wear prediction"""

    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer(min_data_points=5)

    def test_predict_new_tool(self, analyzer):
        """Test prediction for new tool"""
        now = datetime.now()
        data = [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=0.02 + 0.001 * i
            )
            for i in range(20)
        ]

        pred = analyzer.predict_tool_wear(data, "T001")
        assert pred.current_stage in [WearStage.NEW, WearStage.NORMAL]
        assert pred.urgency == AlertLevel.INFO

    def test_predict_critical_wear(self, analyzer):
        """Test prediction for critical wear"""
        now = datetime.now()
        data = [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=0.7 + 0.01 * i
            )
            for i in range(20)
        ]

        pred = analyzer.predict_tool_wear(data, "T001", wear_threshold=0.5, critical_threshold=0.8)
        assert pred.current_stage in [WearStage.SIGNIFICANT_WEAR, WearStage.CRITICAL_WEAR]
        assert pred.urgency in [AlertLevel.WARNING, AlertLevel.CRITICAL]

    def test_predict_end_of_life(self, analyzer):
        """Test prediction for end of life"""
        now = datetime.now()
        data = [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=1.0 + 0.01 * i
            )
            for i in range(20)
        ]

        pred = analyzer.predict_tool_wear(data, "T001")
        assert pred.current_stage == WearStage.END_OF_LIFE
        assert pred.urgency == AlertLevel.CRITICAL


class TestAlertGeneration:
    """Test alert generation"""

    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer(min_data_points=5)

    def test_generate_critical_alert(self, analyzer):
        """Test critical alert generation"""
        now = datetime.now()
        data = [
            DataPoint(timestamp=now + timedelta(hours=i), value=0.9)
            for i in range(20)
        ]

        alerts = analyzer.generate_alerts(
            data,
            metric_name="deviation",
            warning_threshold=0.5,
            critical_threshold=0.8
        )

        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) >= 1

    def test_generate_warning_alert(self, analyzer):
        """Test warning alert generation"""
        now = datetime.now()
        data = [
            DataPoint(timestamp=now + timedelta(hours=i), value=0.6)
            for i in range(20)
        ]

        alerts = analyzer.generate_alerts(
            data,
            metric_name="deviation",
            warning_threshold=0.5,
            critical_threshold=0.8
        )

        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warnings) >= 1

    def test_no_alerts_for_normal(self, analyzer):
        """Test no alerts for normal values"""
        now = datetime.now()
        data = [
            DataPoint(timestamp=now + timedelta(hours=i), value=0.1)
            for i in range(20)
        ]

        alerts = analyzer.generate_alerts(
            data,
            metric_name="deviation",
            warning_threshold=0.5,
            critical_threshold=0.8
        )

        # Should not have critical or warning alerts
        critical_warnings = [a for a in alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]]
        assert len(critical_warnings) == 0


class TestSmoothingFunctions:
    """Test smoothing functions"""

    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer()

    @pytest.fixture
    def noisy_data(self):
        """Generate noisy data"""
        now = datetime.now()
        np.random.seed(42)
        return [
            DataPoint(
                timestamp=now + timedelta(hours=i),
                value=10 + np.random.normal(0, 2)
            )
            for i in range(30)
        ]

    def test_moving_average(self, analyzer, noisy_data):
        """Test moving average calculation"""
        result = analyzer.moving_average(noisy_data, window=5)

        assert len(result) == len(noisy_data) - 4
        # Moving average should reduce variance
        original_std = np.std([d.value for d in noisy_data])
        smoothed_std = np.std([v for _, v in result])
        assert smoothed_std < original_std

    def test_exponential_smoothing(self, analyzer, noisy_data):
        """Test exponential smoothing"""
        result = analyzer.exponential_smoothing(noisy_data, alpha=0.3)

        assert len(result) == len(noisy_data)
        # Smoothing should reduce variance
        original_std = np.std([d.value for d in noisy_data])
        smoothed_std = np.std([v for _, v in result])
        assert smoothed_std < original_std


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_analyze_measurement_trend(self):
        """Test analyze_measurement_trend function"""
        now = datetime.now()
        measurements = [
            {"value": 0.1 + 0.02 * i, "timestamp": (now + timedelta(hours=i)).isoformat()}
            for i in range(20)
        ]

        result = analyze_measurement_trend(measurements)
        assert result.trend_type in [TrendType.INCREASING, TrendType.STABLE]

    def test_predict_maintenance(self):
        """Test predict_maintenance function"""
        now = datetime.now()
        history = [
            {"deviation": 0.1 + 0.01 * i, "timestamp": (now + timedelta(hours=i)).isoformat()}
            for i in range(20)
        ]

        pred = predict_maintenance(history, "T001")
        assert pred.tool_id == "T001"
        assert pred.current_stage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
