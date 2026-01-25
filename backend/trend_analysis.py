"""
Trend Analysis for Tool Wear Prediction

Analyzes historical QC data to detect:
- Progressive tool wear patterns
- Process drift indicators
- Predictive maintenance triggers
- Quality degradation trends

Uses statistical and ML methods:
- Linear/polynomial regression for trend detection
- CUSUM for change point detection
- Moving window analysis
- Exponential smoothing forecasting
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Types of trends detected"""
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    CYCLIC = "cyclic"
    STEP_CHANGE = "step_change"
    DRIFT = "drift"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class WearStage(Enum):
    """Tool wear stages"""
    NEW = "new"
    NORMAL = "normal"
    MODERATE_WEAR = "moderate_wear"
    SIGNIFICANT_WEAR = "significant_wear"
    CRITICAL_WEAR = "critical_wear"
    END_OF_LIFE = "end_of_life"


@dataclass
class WearThresholdConfig:
    """Configurable thresholds for wear stage classification.

    Allows customization per tool type, material, or process.
    All values are normalized wear indices (0.0-1.0+).
    """
    # Threshold below which tool is considered NEW
    new_threshold: float = 0.1
    # Threshold below which tool is in NORMAL wear
    normal_threshold: float = 0.3
    # Threshold for MODERATE_WEAR (default: uses wear_threshold parameter)
    moderate_wear_threshold: Optional[float] = None
    # Threshold for SIGNIFICANT_WEAR (default: uses critical_threshold parameter)
    significant_wear_threshold: Optional[float] = None
    # Value at or above which tool is END_OF_LIFE
    end_of_life_threshold: float = 1.0

    @classmethod
    def default(cls) -> 'WearThresholdConfig':
        """Return default configuration"""
        return cls()

    @classmethod
    def for_material(cls, material: str) -> 'WearThresholdConfig':
        """Get optimized thresholds for specific materials"""
        material_lower = material.lower()
        if 'titanium' in material_lower or 'inconel' in material_lower:
            # Hard materials cause faster tool wear
            return cls(new_threshold=0.08, normal_threshold=0.25, end_of_life_threshold=0.9)
        elif 'aluminum' in material_lower:
            # Soft materials are gentler on tools
            return cls(new_threshold=0.12, normal_threshold=0.35, end_of_life_threshold=1.1)
        elif 'steel' in material_lower:
            return cls(new_threshold=0.1, normal_threshold=0.3, end_of_life_threshold=1.0)
        return cls.default()


@dataclass
class DataPoint:
    """Single measurement data point"""
    timestamp: datetime
    value: float
    job_id: Optional[str] = None
    part_id: Optional[str] = None
    tool_id: Optional[str] = None
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "job_id": self.job_id,
            "part_id": self.part_id,
            "tool_id": self.tool_id,
            "batch_id": self.batch_id
        }


@dataclass
class TrendResult:
    """Result of trend analysis"""
    trend_type: TrendType
    slope: float  # Rate of change per unit time
    r_squared: float  # Goodness of fit
    confidence: float  # Confidence level (0-1)

    # Prediction
    predicted_value: Optional[float] = None
    prediction_interval: Optional[Tuple[float, float]] = None
    time_to_threshold: Optional[float] = None  # Hours until threshold exceeded

    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "trend_type": self.trend_type.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "confidence": self.confidence,
            "predicted_value": self.predicted_value,
            "prediction_interval": self.prediction_interval,
            "time_to_threshold": self.time_to_threshold,
            "statistics": {
                "mean": self.mean,
                "std": self.std,
                "min": self.min_value,
                "max": self.max_value
            }
        }


@dataclass
class ChangePoint:
    """Detected change point in data"""
    timestamp: datetime
    index: int
    before_mean: float
    after_mean: float
    magnitude: float
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "index": self.index,
            "before_mean": self.before_mean,
            "after_mean": self.after_mean,
            "magnitude": self.magnitude,
            "confidence": self.confidence
        }


@dataclass
class WearPrediction:
    """Tool wear prediction result"""
    tool_id: str
    current_stage: WearStage
    wear_rate: float  # Units per hour
    estimated_remaining_life: float  # Hours
    confidence: float

    # Recommendation
    recommended_action: str
    urgency: AlertLevel

    # Supporting data
    data_points_analyzed: int = 0
    trend_result: Optional[TrendResult] = None

    def to_dict(self) -> Dict:
        return {
            "tool_id": self.tool_id,
            "current_stage": self.current_stage.value,
            "wear_rate": self.wear_rate,
            "estimated_remaining_life_hours": self.estimated_remaining_life,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "urgency": self.urgency.value,
            "data_points_analyzed": self.data_points_analyzed,
            "trend": self.trend_result.to_dict() if self.trend_result else None
        }


@dataclass
class TrendAlert:
    """Alert generated from trend analysis"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metric: str
    tool_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    # Supporting data
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    trend_data: Optional[TrendResult] = None

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metric": self.metric,
            "tool_id": self.tool_id,
            "timestamp": self.timestamp.isoformat(),
            "current_value": self.current_value,
            "threshold_value": self.threshold_value
        }


class TrendAnalyzer:
    """
    Analyzer for detecting trends in QC measurements.

    Usage:
        analyzer = TrendAnalyzer()
        data = [DataPoint(...), ...]
        trend = analyzer.analyze_trend(data)
        prediction = analyzer.predict_tool_wear(data, tool_id="T001")
    """

    def __init__(
        self,
        min_data_points: int = 10,
        significance_level: float = 0.05,
        trend_threshold: float = 0.01,
        wear_config: Optional[WearThresholdConfig] = None
    ):
        """
        Initialize trend analyzer.

        Args:
            min_data_points: Minimum points required for analysis
            significance_level: Statistical significance threshold
            trend_threshold: Minimum slope to consider a trend
            wear_config: Configurable thresholds for wear classification
        """
        self.min_data_points = min_data_points
        self.significance_level = significance_level
        self.trend_threshold = trend_threshold
        self.wear_config = wear_config or WearThresholdConfig.default()

    def analyze_trend(
        self,
        data: List[DataPoint],
        threshold: Optional[float] = None
    ) -> TrendResult:
        """
        Analyze trend in measurement data.

        Args:
            data: List of data points
            threshold: Optional threshold for prediction

        Returns:
            TrendResult with trend analysis
        """
        if len(data) < self.min_data_points:
            return TrendResult(
                trend_type=TrendType.STABLE,
                slope=0.0,
                r_squared=0.0,
                confidence=0.0
            )

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda d: d.timestamp)
        values = np.array([d.value for d in sorted_data])

        # Convert timestamps to hours from start
        start_time = sorted_data[0].timestamp
        times = np.array([
            (d.timestamp - start_time).total_seconds() / 3600
            for d in sorted_data
        ])

        # Basic statistics
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, values)
        r_squared = r_value ** 2

        # Determine trend type
        trend_type = self._classify_trend(slope, r_squared, p_value, values)

        # Calculate confidence
        confidence = 1 - p_value if p_value < 0.5 else 0.5

        result = TrendResult(
            trend_type=trend_type,
            slope=slope,
            r_squared=r_squared,
            confidence=confidence,
            mean=mean,
            std=std,
            min_value=min_val,
            max_value=max_val
        )

        # Prediction
        if len(times) > 0:
            next_time = times[-1] + 1  # Predict 1 hour ahead
            predicted = slope * next_time + intercept

            # Prediction interval (95%)
            n = len(values)
            se = std_err * np.sqrt(1 + 1/n + (next_time - np.mean(times))**2 / np.sum((times - np.mean(times))**2))
            t_val = stats.t.ppf(0.975, n - 2)
            interval = (predicted - t_val * se, predicted + t_val * se)

            result.predicted_value = predicted
            result.prediction_interval = interval

            # Time to threshold
            if threshold is not None and abs(slope) > 1e-10:
                current = values[-1]
                if slope > 0 and current < threshold:
                    result.time_to_threshold = (threshold - current) / slope
                elif slope < 0 and current > threshold:
                    result.time_to_threshold = (threshold - current) / slope

        return result

    def _classify_trend(
        self,
        slope: float,
        r_squared: float,
        p_value: float,
        values: np.ndarray
    ) -> TrendType:
        """Classify the trend type"""
        # Check statistical significance
        if p_value > self.significance_level:
            return TrendType.STABLE

        # Check if slope is meaningful
        if abs(slope) < self.trend_threshold:
            return TrendType.STABLE

        # Check fit quality
        if r_squared < 0.3:
            # Poor linear fit - check for cyclic or step change
            return self._check_complex_pattern(values)

        # Good linear fit
        if slope > self.trend_threshold:
            return TrendType.INCREASING
        elif slope < -self.trend_threshold:
            return TrendType.DECREASING
        else:
            return TrendType.STABLE

    def _check_complex_pattern(self, values: np.ndarray) -> TrendType:
        """Check for complex patterns like cycles or step changes"""
        # Simple autocorrelation check for cyclic patterns
        if len(values) > 20:
            autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            # Check for peaks in autocorrelation (cyclic pattern)
            peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0]
            if len(peaks) > 0 and autocorr[peaks[0] + 1] > 0.5:
                return TrendType.CYCLIC

        # Check for step change
        mid = len(values) // 2
        first_half_mean = np.mean(values[:mid])
        second_half_mean = np.mean(values[mid:])
        overall_std = np.std(values)

        if overall_std > 0 and abs(second_half_mean - first_half_mean) > 2 * overall_std:
            return TrendType.STEP_CHANGE

        return TrendType.DRIFT

    def detect_change_points(
        self,
        data: List[DataPoint],
        threshold: float = 4.0
    ) -> List[ChangePoint]:
        """
        Detect change points using CUSUM algorithm.

        Args:
            data: List of data points
            threshold: CUSUM threshold for detection

        Returns:
            List of detected change points
        """
        if len(data) < self.min_data_points:
            return []

        sorted_data = sorted(data, key=lambda d: d.timestamp)
        values = np.array([d.value for d in sorted_data])

        # Calculate CUSUM
        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-10:
            return []

        normalized = (values - mean) / std

        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))

        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + normalized[i] + 0.5)

        # Find change points
        change_points = []

        for i in range(1, len(values) - 1):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                before_mean = np.mean(values[:i])
                after_mean = np.mean(values[i:])
                magnitude = after_mean - before_mean

                # Check if this is a significant change
                if abs(magnitude) > std:
                    change_points.append(ChangePoint(
                        timestamp=sorted_data[i].timestamp,
                        index=i,
                        before_mean=before_mean,
                        after_mean=after_mean,
                        magnitude=magnitude,
                        confidence=min(abs(cusum_pos[i] + cusum_neg[i]) / threshold, 1.0)
                    ))

        # Remove redundant points (keep most significant in each cluster)
        filtered = self._filter_change_points(change_points)

        return filtered

    def _filter_change_points(
        self,
        change_points: List[ChangePoint],
        min_gap: int = 5
    ) -> List[ChangePoint]:
        """Filter redundant change points"""
        if not change_points:
            return []

        filtered = [change_points[0]]
        for cp in change_points[1:]:
            if cp.index - filtered[-1].index >= min_gap:
                filtered.append(cp)
            elif abs(cp.magnitude) > abs(filtered[-1].magnitude):
                filtered[-1] = cp

        return filtered

    def predict_tool_wear(
        self,
        data: List[DataPoint],
        tool_id: str,
        wear_threshold: float = 0.5,
        critical_threshold: float = 0.8
    ) -> WearPrediction:
        """
        Predict tool wear and remaining life.

        Args:
            data: Historical measurement data
            tool_id: Tool identifier
            wear_threshold: Moderate wear threshold
            critical_threshold: Critical wear threshold

        Returns:
            WearPrediction with wear stage and remaining life
        """
        if len(data) < self.min_data_points:
            return WearPrediction(
                tool_id=tool_id,
                current_stage=WearStage.NEW,
                wear_rate=0.0,
                estimated_remaining_life=float('inf'),
                confidence=0.0,
                recommended_action="Insufficient data for analysis",
                urgency=AlertLevel.INFO,
                data_points_analyzed=len(data)
            )

        # Analyze trend
        trend = self.analyze_trend(data, threshold=critical_threshold)

        # Determine wear stage
        current_value = data[-1].value if data else 0
        wear_stage = self._classify_wear_stage(current_value, wear_threshold, critical_threshold)

        # Calculate wear rate (positive slope means increasing deviation = wear)
        wear_rate = max(0, trend.slope)  # Only positive slopes indicate wear

        # Estimate remaining life
        if wear_rate > 1e-10 and current_value < critical_threshold:
            remaining_life = (critical_threshold - current_value) / wear_rate
        else:
            remaining_life = float('inf')

        # Generate recommendation
        action, urgency = self._generate_recommendation(wear_stage, remaining_life, trend)

        return WearPrediction(
            tool_id=tool_id,
            current_stage=wear_stage,
            wear_rate=wear_rate,
            estimated_remaining_life=remaining_life,
            confidence=trend.confidence,
            recommended_action=action,
            urgency=urgency,
            data_points_analyzed=len(data),
            trend_result=trend
        )

    def _classify_wear_stage(
        self,
        current_value: float,
        wear_threshold: float,
        critical_threshold: float
    ) -> WearStage:
        """Classify current wear stage using configurable thresholds.

        Uses self.wear_config for NEW/NORMAL/END_OF_LIFE thresholds,
        and method parameters for MODERATE/SIGNIFICANT wear (allows per-call customization).
        """
        cfg = self.wear_config

        # Use config thresholds with parameter overrides where specified
        moderate_threshold = cfg.moderate_wear_threshold or wear_threshold
        significant_threshold = cfg.significant_wear_threshold or critical_threshold

        if current_value < cfg.new_threshold:
            return WearStage.NEW
        elif current_value < cfg.normal_threshold:
            return WearStage.NORMAL
        elif current_value < moderate_threshold:
            return WearStage.MODERATE_WEAR
        elif current_value < significant_threshold:
            return WearStage.SIGNIFICANT_WEAR
        elif current_value < cfg.end_of_life_threshold:
            return WearStage.CRITICAL_WEAR
        else:
            return WearStage.END_OF_LIFE

    def _generate_recommendation(
        self,
        stage: WearStage,
        remaining_life: float,
        trend: TrendResult
    ) -> Tuple[str, AlertLevel]:
        """Generate action recommendation based on wear analysis"""
        if stage == WearStage.END_OF_LIFE:
            return "Replace tool immediately - end of life reached", AlertLevel.CRITICAL

        if stage == WearStage.CRITICAL_WEAR:
            if remaining_life < 8:
                return f"Replace tool within {remaining_life:.1f} hours - critical wear detected", AlertLevel.CRITICAL
            return "Schedule tool replacement soon - significant wear", AlertLevel.WARNING

        if stage == WearStage.SIGNIFICANT_WEAR:
            if remaining_life < 24:
                return f"Plan tool replacement - approximately {remaining_life:.1f} hours remaining", AlertLevel.WARNING
            return "Monitor closely - approaching wear limit", AlertLevel.WARNING

        if stage == WearStage.MODERATE_WEAR:
            if trend.trend_type == TrendType.INCREASING:
                return f"Wear progressing - estimated {remaining_life:.0f} hours to critical", AlertLevel.INFO
            return "Normal wear progression - continue monitoring", AlertLevel.INFO

        if stage == WearStage.NORMAL:
            return "Tool performing normally", AlertLevel.INFO

        return "New tool - establish baseline", AlertLevel.INFO

    def generate_alerts(
        self,
        data: List[DataPoint],
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        tool_id: Optional[str] = None
    ) -> List[TrendAlert]:
        """
        Generate alerts based on trend analysis.

        Args:
            data: Measurement data
            metric_name: Name of the metric
            warning_threshold: Warning level threshold
            critical_threshold: Critical level threshold
            tool_id: Optional tool identifier

        Returns:
            List of generated alerts
        """
        alerts = []

        if len(data) < self.min_data_points:
            return alerts

        trend = self.analyze_trend(data, threshold=critical_threshold)
        current_value = data[-1].value

        # Current value alerts
        if current_value >= critical_threshold:
            alerts.append(TrendAlert(
                alert_id=f"critical_{metric_name}_{datetime.now().timestamp()}",
                level=AlertLevel.CRITICAL,
                title=f"Critical {metric_name} Threshold Exceeded",
                message=f"Current {metric_name} ({current_value:.4f}) exceeds critical threshold ({critical_threshold})",
                metric=metric_name,
                tool_id=tool_id,
                current_value=current_value,
                threshold_value=critical_threshold,
                trend_data=trend
            ))
        elif current_value >= warning_threshold:
            alerts.append(TrendAlert(
                alert_id=f"warning_{metric_name}_{datetime.now().timestamp()}",
                level=AlertLevel.WARNING,
                title=f"{metric_name} Warning Threshold",
                message=f"Current {metric_name} ({current_value:.4f}) exceeds warning threshold ({warning_threshold})",
                metric=metric_name,
                tool_id=tool_id,
                current_value=current_value,
                threshold_value=warning_threshold,
                trend_data=trend
            ))

        # Trend-based alerts
        if trend.trend_type == TrendType.INCREASING and trend.confidence > 0.8:
            if trend.time_to_threshold is not None and trend.time_to_threshold < 24:
                alerts.append(TrendAlert(
                    alert_id=f"trend_{metric_name}_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    title=f"Increasing {metric_name} Trend Detected",
                    message=f"{metric_name} is increasing. Estimated time to critical: {trend.time_to_threshold:.1f} hours",
                    metric=metric_name,
                    tool_id=tool_id,
                    current_value=current_value,
                    threshold_value=critical_threshold,
                    trend_data=trend
                ))

        # Change point alerts
        change_points = self.detect_change_points(data)
        if change_points:
            latest_cp = change_points[-1]
            if latest_cp.confidence > 0.8:
                alerts.append(TrendAlert(
                    alert_id=f"change_{metric_name}_{datetime.now().timestamp()}",
                    level=AlertLevel.INFO,
                    title=f"Process Change Detected in {metric_name}",
                    message=f"Significant change detected: {latest_cp.magnitude:.4f} shift at {latest_cp.timestamp}",
                    metric=metric_name,
                    tool_id=tool_id,
                    current_value=latest_cp.after_mean,
                    trend_data=trend
                ))

        return alerts

    def moving_average(
        self,
        data: List[DataPoint],
        window: int = 5
    ) -> List[Tuple[datetime, float]]:
        """
        Calculate moving average.

        Args:
            data: List of data points
            window: Window size

        Returns:
            List of (timestamp, moving_average) tuples
        """
        if len(data) < window:
            return []

        sorted_data = sorted(data, key=lambda d: d.timestamp)
        values = [d.value for d in sorted_data]

        result = []
        for i in range(window - 1, len(values)):
            avg = np.mean(values[i - window + 1:i + 1])
            result.append((sorted_data[i].timestamp, avg))

        return result

    def exponential_smoothing(
        self,
        data: List[DataPoint],
        alpha: float = 0.3
    ) -> List[Tuple[datetime, float]]:
        """
        Apply exponential smoothing.

        Args:
            data: List of data points
            alpha: Smoothing factor (0-1)

        Returns:
            List of (timestamp, smoothed_value) tuples
        """
        if not data:
            return []

        sorted_data = sorted(data, key=lambda d: d.timestamp)
        result = [(sorted_data[0].timestamp, sorted_data[0].value)]

        for i in range(1, len(sorted_data)):
            smoothed = alpha * sorted_data[i].value + (1 - alpha) * result[-1][1]
            result.append((sorted_data[i].timestamp, smoothed))

        return result


# Convenience functions
def analyze_measurement_trend(
    measurements: List[Dict],
    value_key: str = "value",
    timestamp_key: str = "timestamp"
) -> TrendResult:
    """
    Analyze trend from measurement dictionaries.

    Args:
        measurements: List of measurement dicts
        value_key: Key for value in dict
        timestamp_key: Key for timestamp in dict

    Returns:
        TrendResult
    """
    data = []
    for m in measurements:
        ts = m.get(timestamp_key)
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            ts = datetime.now()

        data.append(DataPoint(
            timestamp=ts,
            value=m.get(value_key, 0)
        ))

    analyzer = TrendAnalyzer()
    return analyzer.analyze_trend(data)


def predict_maintenance(
    tool_history: List[Dict],
    tool_id: str
) -> WearPrediction:
    """
    Predict tool maintenance from history.

    Args:
        tool_history: List of measurement dicts
        tool_id: Tool identifier

    Returns:
        WearPrediction
    """
    data = [
        DataPoint(
            timestamp=datetime.fromisoformat(h["timestamp"]) if isinstance(h["timestamp"], str) else h["timestamp"],
            value=h.get("deviation", h.get("value", 0)),
            tool_id=tool_id
        )
        for h in tool_history
    ]

    analyzer = TrendAnalyzer()
    return analyzer.predict_tool_wear(data, tool_id)
