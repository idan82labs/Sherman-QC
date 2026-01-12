"""
Tests for QC Engine - Core Analysis Module
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from qc_engine import (
    ScanQCEngine, QCReport, QCResult, RegionAnalysis, RootCause,
    ProgressUpdate, AI_ENABLED
)


class TestQCReportDataclass:
    """Test QCReport dataclass and serialization"""

    def test_qc_report_defaults(self):
        """Test QCReport initializes with correct defaults"""
        report = QCReport(
            part_id="TEST-001",
            part_name="Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            timestamp="2026-01-12T00:00:00"
        )

        assert report.part_id == "TEST-001"
        assert report.overall_result == QCResult.PASS
        assert report.quality_score == 0.0
        assert report.total_points == 0
        assert report.heatmap_multi_view == ""
        assert report.ai_model_used == ""

    def test_qc_report_to_dict(self):
        """Test QCReport serialization to dict"""
        report = QCReport(
            part_id="TEST-001",
            part_name="Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            timestamp="2026-01-12T00:00:00"
        )
        report.total_points = 10000
        report.pass_rate = 0.95
        report.mean_deviation = 0.05
        report.max_deviation = 0.15
        report.heatmap_multi_view = "/path/to/heatmap.png"

        report_dict = report.to_dict()

        assert report_dict["part_id"] == "TEST-001"
        assert report_dict["statistics"]["total_points"] == 10000
        assert report_dict["statistics"]["pass_rate"] == 95.0
        assert report_dict["heatmaps"]["multi_view"] == "/path/to/heatmap.png"

    def test_qc_report_regions_serialization(self):
        """Test regional analysis serialization"""
        report = QCReport(
            part_id="TEST-001",
            part_name="Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            timestamp="2026-01-12T00:00:00"
        )
        report.regions = [
            RegionAnalysis(
                name="Top Surface",
                point_count=1000,
                mean_deviation=0.03,
                max_deviation=0.08,
                min_deviation=-0.02,
                std_deviation=0.02,
                pass_rate=0.98,
                status="OK",
                deviation_direction="outward",
                ai_interpretation="Within specification"
            )
        ]

        report_dict = report.to_dict()

        assert len(report_dict["regions"]) == 1
        assert report_dict["regions"][0]["name"] == "Top Surface"
        assert report_dict["regions"][0]["pass_rate"] == 98.0


class TestAIIntegration:
    """Test AI analyzer integration"""

    def test_ai_enabled_flag(self):
        """Test AI_ENABLED flag detection"""
        # This tests the flag detection logic
        # In real tests, we'd mock environment variables
        assert AI_ENABLED is None or isinstance(AI_ENABLED, str)

    def test_rule_based_fallback(self):
        """Test that rule-based analysis works when AI unavailable"""
        # This would require mocking the AI API calls
        pass


class TestProgressUpdates:
    """Test progress update functionality"""

    def test_progress_update_creation(self):
        """Test ProgressUpdate dataclass"""
        update = ProgressUpdate(
            stage="load",
            progress=25.0,
            message="Loading reference model"
        )

        assert update.stage == "load"
        assert update.progress == 25.0
        assert update.message == "Loading reference model"


class TestRootCauseAnalysis:
    """Test root cause analysis"""

    def test_root_cause_creation(self):
        """Test RootCause dataclass"""
        cause = RootCause(
            issue="Springback on curved section",
            likely_cause="Elastic recovery after bending",
            technical_explanation="Material springs back due to elastic stress",
            confidence=0.85,
            recommendation="Increase overbend by 2-4 degrees",
            priority="HIGH"
        )

        assert cause.issue == "Springback on curved section"
        assert cause.confidence == 0.85
        assert cause.priority == "HIGH"


class TestHeatmapGeneration:
    """Test heatmap generation functionality"""

    def test_engine_has_heatmap_renderer(self):
        """Test that engine can access heatmap renderer"""
        engine = ScanQCEngine()
        # Renderer is lazy-loaded, so accessing it should work
        # even if matplotlib isn't available
        renderer = engine.heatmap_renderer
        # May be None if dependencies not installed
        assert renderer is None or hasattr(renderer, 'render_heatmap')


# Integration tests (require actual files)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual test files"""

    @pytest.fixture
    def test_files_dir(self):
        return Path(__file__).parent.parent / "test_files"

    def test_load_stl_file(self, test_files_dir):
        """Test loading STL reference file"""
        if not test_files_dir.exists():
            pytest.skip("Test files not available")

        engine = ScanQCEngine()
        ref_path = test_files_dir / "reference.stl"

        if ref_path.exists():
            result = engine.load_reference(str(ref_path))
            assert result is True
            assert engine.reference_mesh is not None

    def test_load_ply_scan(self, test_files_dir):
        """Test loading PLY scan file"""
        if not test_files_dir.exists():
            pytest.skip("Test files not available")

        engine = ScanQCEngine()
        scan_path = test_files_dir / "scan.ply"

        if scan_path.exists():
            result = engine.load_scan(str(scan_path))
            assert result is True
            assert engine.scan_pcd is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
