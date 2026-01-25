"""
Integration Tests for Multi-Model AI Pipeline

Tests the orchestrator, fallback mechanisms, and data flow between stages.
"""

import pytest
import numpy as np
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from multi_model.exchange_schema import (
    DrawingExtraction,
    DrawingDimension,
    DrawingBend,
    PointCloudFeatures,
    BendFeature,
    Correlation2D3D,
    CorrelationMapping,
    BendRegionAnalysis,
    EnhancedRootCause,
    EnhancedAnalysisResult,
    PipelineStage,
    PipelineStatus,
)
from multi_model.orchestrator import MultiModelOrchestrator
from multi_model.fallback_manager import FallbackManager


class TestExchangeSchemas:
    """Test data exchange schema classes."""

    def test_drawing_dimension_creation(self):
        """Test DrawingDimension dataclass."""
        dim = DrawingDimension(
            id="D1",
            dimension_type="linear",
            nominal=100.0,
            tolerance_plus=0.5,
            tolerance_minus=0.5,
        )

        assert dim.id == "D1"
        assert dim.dimension_type == "linear"
        assert dim.nominal == 100.0
        assert dim.unit == "mm"  # Default

    def test_drawing_extraction_to_dict(self):
        """Test DrawingExtraction serialization."""
        extraction = DrawingExtraction(
            drawing_id="test_drawing",
            extraction_timestamp=datetime.now().isoformat(),
            dimensions=[
                DrawingDimension(
                    id="D1",
                    dimension_type="linear",
                    nominal=100.0,
                    tolerance_plus=0.5,
                    tolerance_minus=0.5,
                )
            ],
            bends=[
                DrawingBend(
                    id="B1",
                    sequence=1,
                    angle_nominal=90.0,
                    angle_tolerance=1.0,
                    radius=3.0,
                )
            ],
            material="Al-5053-H32",
            thickness=2.0,
            extraction_confidence=0.85,
        )

        data = extraction.to_dict()

        assert data["drawing_id"] == "test_drawing"
        assert len(data["dimensions"]) == 1
        assert len(data["bends"]) == 1
        assert data["material"] == "Al-5053-H32"

    def test_bend_feature_to_dict(self):
        """Test BendFeature serialization."""
        bend = BendFeature(
            bend_id=1,
            bend_name="Bend 1",
            angle_degrees=90.5,
            radius_mm=3.0,
            bend_line_start=(0, 0, 0),
            bend_line_end=(100, 0, 0),
            bend_apex=(50, 0, 5),
            region_point_indices=[1, 2, 3],
            detection_confidence=0.9,
        )

        data = bend.to_dict()

        assert data["bend_id"] == 1
        assert data["angle_degrees"] == 90.5
        assert data["region_point_count"] == 3

    def test_correlation_mapping(self):
        """Test CorrelationMapping serialization."""
        mapping = CorrelationMapping(
            mapping_id="M1",
            drawing_element_id="B1",
            drawing_element_type="bend",
            feature_3d_id="Bend_1",
            feature_3d_type="bend",
            confidence=0.95,
            nominal_value=90.0,
            measured_value=88.5,
            deviation=-1.5,
            is_in_tolerance=False,
        )

        data = mapping.to_dict()

        assert data["mapping_id"] == "M1"
        assert data["deviation"] == -1.5
        assert data["is_in_tolerance"] is False

    def test_enhanced_root_cause(self):
        """Test EnhancedRootCause serialization."""
        cause = EnhancedRootCause(
            cause_id="RC1",
            category="process",
            description="Springback in bend region",
            severity="major",
            confidence=0.8,
            affected_bends=[1, 2],
            evidence=["Measured angle < nominal"],
            recommendations=["Increase overbend"],
            priority=1,
        )

        data = cause.to_dict()

        assert data["cause_id"] == "RC1"
        assert data["category"] == "process"
        assert data["affected_bends"] == [1, 2]

    def test_pipeline_status(self):
        """Test PipelineStatus tracking."""
        status = PipelineStatus(
            job_id="test_job",
            current_stage=PipelineStage.FEATURE_DETECTION,
            overall_progress=50.0,
        )

        data = status.to_dict()

        assert data["job_id"] == "test_job"
        assert data["current_stage"] == "feature_detection"
        assert data["overall_progress"] == 50.0


class TestFallbackManager:
    """Test fallback mechanisms."""

    @pytest.fixture
    def fallback_manager(self):
        """Create a FallbackManager instance."""
        return FallbackManager()

    def test_generate_basic_root_causes_springback(self, fallback_manager):
        """Test rule-based root cause generation for springback."""
        bend_result = Mock()
        bend_result.status = "fail"
        bend_result.springback_indicator = 0.6
        bend_result.over_bend_indicator = 0.0
        bend_result.apex_deviation_mm = 0.2
        bend_result.angle_deviation_deg = -2.0
        bend_result.bend = Mock()
        bend_result.bend.bend_id = 1

        deviations = np.random.randn(1000) * 0.1

        causes = fallback_manager.generate_basic_root_causes(
            bend_results=[bend_result],
            deviations=deviations,
            tolerance=0.5,
        )

        assert len(causes) >= 1
        # Should identify springback
        springback_causes = [c for c in causes if "springback" in c.description.lower()]
        assert len(springback_causes) >= 1

    def test_generate_basic_root_causes_overbend(self, fallback_manager):
        """Test rule-based root cause generation for overbend."""
        bend_result = Mock()
        bend_result.status = "fail"
        bend_result.springback_indicator = 0.0
        bend_result.over_bend_indicator = 0.7
        bend_result.apex_deviation_mm = 0.2
        bend_result.angle_deviation_deg = 3.0
        bend_result.bend = Mock()
        bend_result.bend.bend_id = 1

        deviations = np.random.randn(1000) * 0.1

        causes = fallback_manager.generate_basic_root_causes(
            bend_results=[bend_result],
            deviations=deviations,
            tolerance=0.5,
        )

        assert len(causes) >= 1
        # Should identify overbend
        overbend_causes = [c for c in causes if "overbend" in c.description.lower()]
        assert len(overbend_causes) >= 1

    def test_generate_bend_recommendations(self, fallback_manager):
        """Test recommendation generation for bends."""
        bend_result = Mock()
        bend_result.springback_indicator = 0.5
        bend_result.over_bend_indicator = 0.0
        bend_result.apex_deviation_mm = 0.1
        bend_result.angle_deviation_deg = -2.0
        bend_result.twist_angle_deg = 0.5

        recommendations = fallback_manager.generate_bend_recommendations(
            bend_result=bend_result,
            nominal_angle=90.0,
        )

        assert len(recommendations) >= 1
        # Should recommend increasing bend angle
        assert any("increase" in r.lower() or "angle" in r.lower() for r in recommendations)

    def test_estimate_springback(self, fallback_manager):
        """Test springback estimation."""
        springback = fallback_manager.estimate_springback(
            material="aluminum",
            thickness=2.0,
            bend_angle=90.0,
            bend_radius=3.0,
        )

        assert springback > 0
        assert springback < 10  # Less than 10 degrees for 90-degree bend

    def test_estimate_springback_different_materials(self, fallback_manager):
        """Test springback varies by material."""
        aluminum_sb = fallback_manager.estimate_springback(
            material="aluminum",
            thickness=2.0,
            bend_angle=90.0,
            bend_radius=3.0,
        )

        steel_sb = fallback_manager.estimate_springback(
            material="steel",
            thickness=2.0,
            bend_angle=90.0,
            bend_radius=3.0,
        )

        # Aluminum typically has more springback than steel
        assert aluminum_sb > steel_sb


class TestMultiModelOrchestrator:
    """Test the multi-model orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a MultiModelOrchestrator instance."""
        return MultiModelOrchestrator(
            enable_fallbacks=True,
            max_retries=1,
            timeout_seconds=5.0,
        )

    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud."""
        return np.random.randn(1000, 3) * 50

    @pytest.fixture
    def sample_deviations(self):
        """Create sample deviations."""
        return np.random.randn(1000) * 0.3

    @pytest.mark.asyncio
    async def test_analyze_without_api_keys(self, orchestrator, sample_points, sample_deviations):
        """Test analysis falls back gracefully without API keys."""
        with patch.dict('os.environ', {}, clear=True):
            result = await orchestrator.analyze(
                job_id="test_job",
                points=sample_points,
                deviations=sample_deviations,
                tolerance=0.5,
            )

        assert result is not None
        assert result.job_id == "test_job"
        assert result.pipeline_status is not None

    @pytest.mark.asyncio
    async def test_analyze_with_bend_detection(self, orchestrator, sample_points, sample_deviations):
        """Test analysis with pre-computed bend detection."""
        from bend_detector import BendDetectionResult

        mock_bend_result = BendDetectionResult(
            bends=[
                BendFeature(
                    bend_id=1,
                    bend_name="Bend 1",
                    angle_degrees=90.0,
                    radius_mm=3.0,
                    bend_line_start=(0, 0, 0),
                    bend_line_end=(100, 0, 0),
                    bend_apex=(50, 0, 5),
                    detection_confidence=0.85,
                )
            ],
            total_bends_detected=1,
        )

        with patch.dict('os.environ', {}, clear=True):
            result = await orchestrator.analyze(
                job_id="test_job",
                points=sample_points,
                deviations=sample_deviations,
                bend_detection_result=mock_bend_result,
                tolerance=0.5,
            )

        assert result is not None
        # Should have processed the bend
        assert result.point_cloud_features is not None

    @pytest.mark.asyncio
    async def test_pipeline_stages_tracked(self, orchestrator, sample_points, sample_deviations):
        """Test that pipeline stages are properly tracked."""
        with patch.dict('os.environ', {}, clear=True):
            result = await orchestrator.analyze(
                job_id="test_job",
                points=sample_points,
                deviations=sample_deviations,
                tolerance=0.5,
            )

        status = result.pipeline_status
        assert status is not None
        assert status.current_stage == PipelineStage.COMPLETE

    @pytest.mark.asyncio
    async def test_analyze_with_part_info(self, orchestrator, sample_points, sample_deviations):
        """Test analysis with part information."""
        part_info = {
            "part_id": "TEST-001",
            "part_name": "Test Part",
            "material": "Al-5053-H32",
        }

        with patch.dict('os.environ', {}, clear=True):
            result = await orchestrator.analyze(
                job_id="test_job",
                points=sample_points,
                deviations=sample_deviations,
                part_info=part_info,
                tolerance=0.5,
            )

        assert result is not None

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes with correct settings."""
        assert orchestrator.enable_fallbacks is True
        assert orchestrator.max_retries == 1
        assert orchestrator.timeout_seconds == 5.0

    def test_model_configs_defined(self, orchestrator):
        """Test that model configs are properly defined."""
        assert "drawing_analysis" in orchestrator.MODELS
        assert "feature_detection" in orchestrator.MODELS
        assert "correlation" in orchestrator.MODELS
        assert "root_cause" in orchestrator.MODELS

    def test_fallback_order_defined(self, orchestrator):
        """Test that fallback orders are defined."""
        assert "drawing_analysis" in orchestrator.FALLBACK_ORDER
        assert "feature_detection" in orchestrator.FALLBACK_ORDER
        assert "correlation" in orchestrator.FALLBACK_ORDER
        assert "root_cause" in orchestrator.FALLBACK_ORDER


class TestEnhancedAnalysisResult:
    """Test the complete enhanced analysis result."""

    def test_result_to_dict(self):
        """Test complete result serialization."""
        result = EnhancedAnalysisResult(
            job_id="test_job",
            analysis_timestamp=datetime.now().isoformat(),
            bend_results=[
                BendRegionAnalysis(
                    bend=BendFeature(
                        bend_id=1,
                        bend_name="Bend 1",
                        angle_degrees=88.5,
                        radius_mm=3.0,
                        bend_line_start=(0, 0, 0),
                        bend_line_end=(100, 0, 0),
                        bend_apex=(50, 0, 5),
                        detection_confidence=0.85,
                    ),
                    nominal_angle=90.0,
                    angle_deviation_deg=-1.5,
                    springback_indicator=0.3,
                    over_bend_indicator=0.0,
                    apex_deviation_mm=0.2,
                    status="warning",
                )
            ],
            enhanced_root_causes=[
                EnhancedRootCause(
                    cause_id="RC1",
                    category="process",
                    description="Springback detected",
                    severity="minor",
                    confidence=0.7,
                    affected_bends=[1],
                    evidence=["Angle deviation -1.5 deg"],
                    recommendations=["Increase overbend"],
                    priority=2,
                )
            ],
            total_bends_detected=1,
            bends_in_tolerance=0,
            bends_out_of_tolerance=1,
            critical_issues_count=0,
            models_used=["local_fallback"],
            processing_time_ms=150.0,
        )

        data = result.to_dict()

        assert data["job_id"] == "test_job"
        assert len(data["bend_results"]) == 1
        assert len(data["enhanced_root_causes"]) == 1
        assert data["summary"]["total_bends_detected"] == 1
        assert data["metadata"]["processing_time_ms"] == 150.0


class TestIntegrationFlow:
    """Test complete integration flow."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete analysis from point cloud to results."""
        # Create synthetic point cloud with a bend
        points = []

        # Flat surface
        for x in np.linspace(0, 50, 25):
            for y in np.linspace(0, 50, 25):
                points.append([x, y, 0])

        # Bend transition
        for angle in np.linspace(0, np.pi/2, 10):
            for y in np.linspace(0, 50, 25):
                x = 50 + 3 * np.sin(angle)
                z = 3 * (1 - np.cos(angle))
                points.append([x, y, z])

        # Second surface
        for z in np.linspace(3, 50, 25):
            for y in np.linspace(0, 50, 25):
                points.append([53, y, z])

        points = np.array(points)
        deviations = np.random.randn(len(points)) * 0.2

        # Run bend detection
        from bend_detector import BendDetector

        detector = BendDetector(min_bend_points=20, min_surface_points=30)
        bend_result = detector.detect_bends(points, deviations)

        # Run orchestration
        orchestrator = MultiModelOrchestrator()

        with patch.dict('os.environ', {}, clear=True):
            result = await orchestrator.analyze(
                job_id="integration_test",
                points=points,
                deviations=deviations,
                bend_detection_result=bend_result,
                part_info={"part_id": "TEST", "material": "Aluminum"},
                tolerance=0.5,
            )

        # Verify result structure
        assert result is not None
        assert result.job_id == "integration_test"
        assert result.pipeline_status is not None
        assert result.processing_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
