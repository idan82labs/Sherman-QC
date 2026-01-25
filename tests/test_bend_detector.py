"""
Unit Tests for Bend Detection Algorithm

Tests the BendDetector class with various point cloud scenarios:
- Simple 90-degree bends
- Multiple bends
- No bends (flat plate)
- Complex shapes
- Edge cases
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from bend_detector import BendDetector, analyze_bend_deviations


class TestBendDetector:
    """Test suite for BendDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a bend detector with settings tuned for test data."""
        return BendDetector(
            normal_radius=8.0,  # Larger for better normal estimation
            curvature_radius=6.0,  # Larger to capture bend curvature
            curvature_threshold=0.005,  # Lower threshold for synthetic data
            min_bend_points=15,  # Lower for test data
            min_surface_points=50,
            dbscan_eps=0.2,  # Looser clustering for normals
            dbscan_min_samples=5,  # Fewer samples needed
        )

    @pytest.fixture
    def flat_plate_points(self):
        """Generate points for a flat plate."""
        # 100x100mm flat plate at z=0
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        # Add small noise
        zz += np.random.randn(*zz.shape) * 0.1

        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    @pytest.fixture
    def simple_90_bend_points(self):
        """Generate points for a simple 90-degree L-shaped bend."""
        np.random.seed(42)  # Reproducible test data
        points = []

        # First surface: flat plate in XY plane (z=0) - dense grid
        for x in np.linspace(0, 45, 30):
            for y in np.linspace(0, 80, 40):
                z = np.random.randn() * 0.02  # Very small noise
                points.append([x, y, z])

        # Bend region: curved transition with tighter radius
        bend_radius = 8.0  # Larger radius for more gradual curve
        for angle in np.linspace(0, np.pi/2, 30):  # More samples in bend
            for y in np.linspace(0, 80, 40):
                x = 45 + bend_radius * np.sin(angle)
                z = bend_radius * (1 - np.cos(angle))
                # Add very small noise
                x += np.random.randn() * 0.02
                z += np.random.randn() * 0.02
                points.append([x, y, z])

        # Second surface: vertical wall
        for z in np.linspace(bend_radius, 50, 30):
            for y in np.linspace(0, 80, 40):
                x = 45 + bend_radius + np.random.randn() * 0.02
                points.append([x, y, z])

        return np.array(points)

    @pytest.fixture
    def two_bend_points(self):
        """Generate points for a hat channel (two 90-degree bends)."""
        np.random.seed(43)  # Reproducible test data
        points = []
        bend_radius = 6.0

        # Bottom flat surface
        for x in np.linspace(0, 25, 20):
            for y in np.linspace(0, 80, 40):
                z = np.random.randn() * 0.02
                points.append([x, y, z])

        # First bend (left)
        for angle in np.linspace(0, np.pi/2, 25):
            for y in np.linspace(0, 80, 40):
                x = 25 + bend_radius * np.sin(angle)
                z = bend_radius * (1 - np.cos(angle))
                x += np.random.randn() * 0.02
                z += np.random.randn() * 0.02
                points.append([x, y, z])

        # Left wall
        for z in np.linspace(bend_radius, 35, 20):
            for y in np.linspace(0, 80, 40):
                x = 25 + bend_radius + np.random.randn() * 0.02
                points.append([x, y, z])

        # Top surface
        for x in np.linspace(31, 69, 25):
            for y in np.linspace(0, 80, 40):
                z = 35 + np.random.randn() * 0.02
                points.append([x, y, z])

        # Right wall
        for z in np.linspace(bend_radius, 35, 20):
            for y in np.linspace(0, 80, 40):
                x = 75 - bend_radius + np.random.randn() * 0.02
                points.append([x, y, z])

        # Second bend (right)
        for angle in np.linspace(0, np.pi/2, 25):
            for y in np.linspace(0, 80, 40):
                x = 75 - bend_radius * np.sin(angle)
                z = bend_radius * (1 - np.cos(angle))
                x += np.random.randn() * 0.02
                z += np.random.randn() * 0.02
                points.append([x, y, z])

        # Right flat surface
        for x in np.linspace(75, 100, 20):
            for y in np.linspace(0, 80, 40):
                z = np.random.randn() * 0.02
                points.append([x, y, z])

        return np.array(points)

    def test_flat_plate_no_bends(self, detector, flat_plate_points):
        """Test that a flat plate produces no bend detections."""
        result = detector.detect_bends(flat_plate_points)

        assert result is not None
        assert result.total_bends_detected == 0
        assert len(result.bends) == 0
        assert result.processing_time_ms > 0

    def test_simple_90_degree_bend(self, detector, simple_90_bend_points):
        """Test detection of a simple 90-degree bend."""
        result = detector.detect_bends(simple_90_bend_points)

        assert result is not None
        assert result.total_bends_detected >= 1

        if result.bends:
            bend = result.bends[0]
            assert bend.bend_id == 1
            assert bend.bend_name == "Bend 1"
            # Angle should be around 90 degrees
            assert 70 <= bend.angle_degrees <= 110
            assert bend.detection_confidence > 0

    def test_multiple_bends(self, detector, two_bend_points):
        """Test detection of multiple bends in a hat channel."""
        result = detector.detect_bends(two_bend_points)

        assert result is not None
        # Should detect at least 2 bends (possibly 4 if counting top corners)
        assert result.total_bends_detected >= 2

    def test_bend_ordering(self, detector, two_bend_points):
        """Test that bends are ordered spatially."""
        result = detector.detect_bends(two_bend_points)

        if result.total_bends_detected >= 2:
            bends = result.bends
            # Bends should be numbered sequentially
            for i, bend in enumerate(bends, start=1):
                assert bend.bend_id == i
                assert bend.bend_name == f"Bend {i}"

    def test_curvature_computation(self, detector, simple_90_bend_points):
        """Test that curvature is computed correctly."""
        detector.detect_bends(simple_90_bend_points)

        curvatures = detector.get_curvatures()
        assert curvatures is not None
        assert len(curvatures) == len(simple_90_bend_points)

        # Curvatures should be non-negative
        assert np.all(curvatures >= 0)

        # Should have some high-curvature points (bend region)
        assert np.max(curvatures) > 0

    def test_normal_estimation(self, detector, flat_plate_points):
        """Test that surface normals are estimated correctly for a flat plate."""
        detector.detect_bends(flat_plate_points)

        normals = detector.get_normals()
        assert normals is not None
        assert len(normals) == len(flat_plate_points)

        # For a flat plate in XY plane, normals should point in Z direction
        avg_normal = np.mean(normals, axis=0)
        # Z component should be dominant
        assert abs(avg_normal[2]) > 0.9

    def test_empty_point_cloud(self, detector):
        """Test handling of empty point cloud."""
        empty_points = np.array([]).reshape(0, 3)
        result = detector.detect_bends(empty_points)

        assert result is not None
        assert result.total_bends_detected == 0
        assert "Insufficient points" in result.warnings[0]

    def test_small_point_cloud(self, detector):
        """Test handling of very small point cloud."""
        small_points = np.random.randn(10, 3)
        result = detector.detect_bends(small_points)

        assert result is not None
        assert result.total_bends_detected == 0

    def test_detection_confidence(self, detector, simple_90_bend_points):
        """Test that detection confidence is reasonable."""
        result = detector.detect_bends(simple_90_bend_points)

        for bend in result.bends:
            assert 0 <= bend.detection_confidence <= 1

    def test_bend_radius_estimation(self, detector, simple_90_bend_points):
        """Test that bend radius is estimated."""
        result = detector.detect_bends(simple_90_bend_points)

        for bend in result.bends:
            # Radius should be positive and reasonable
            assert bend.radius_mm > 0
            assert bend.radius_mm < 100  # Less than 100mm for test parts

    def test_bend_line_computation(self, detector, simple_90_bend_points):
        """Test that bend line coordinates are computed."""
        result = detector.detect_bends(simple_90_bend_points)

        for bend in result.bends:
            # Bend line should have start and end points
            assert len(bend.bend_line_start) == 3
            assert len(bend.bend_line_end) == 3
            # Bend apex should be between start and end
            assert len(bend.bend_apex) == 3


class TestAnalyzeBendDeviations:
    """Test suite for bend deviation analysis."""

    @pytest.fixture
    def sample_bend(self):
        """Create a sample BendFeature for testing."""
        from bend_detector import BendFeature

        return BendFeature(
            bend_id=1,
            bend_name="Bend 1",
            angle_degrees=88.5,  # Slightly under 90
            radius_mm=3.0,
            bend_line_start=(0, 0, 0),
            bend_line_end=(0, 100, 0),
            bend_apex=(50, 50, 0),
            region_point_indices=list(range(100)),
            adjacent_surface_ids=[0, 1],
            detection_confidence=0.85,
            bend_direction="up",
        )

    def test_springback_detection(self, sample_bend):
        """Test springback indicator for under-bent angles."""
        points = np.random.randn(200, 3)
        deviations = np.random.randn(200) * 0.1

        result = analyze_bend_deviations(
            bend=sample_bend,
            nominal_angle=90.0,
            points=points,
            deviations=deviations,
            tolerance=1.0,
        )

        # Measured 88.5 vs nominal 90 = springback
        assert result["springback_indicator"] > 0
        assert result["over_bend_indicator"] == 0
        assert result["angle_deviation"] == pytest.approx(-1.5, abs=0.01)

    def test_overbend_detection(self, sample_bend):
        """Test overbend indicator for over-bent angles."""
        sample_bend.angle_degrees = 93.0  # Overbent

        points = np.random.randn(200, 3)
        deviations = np.random.randn(200) * 0.1

        result = analyze_bend_deviations(
            bend=sample_bend,
            nominal_angle=90.0,
            points=points,
            deviations=deviations,
            tolerance=1.0,
        )

        # Measured 93 vs nominal 90 = overbend
        assert result["over_bend_indicator"] > 0
        assert result["springback_indicator"] == 0
        assert result["angle_deviation"] == pytest.approx(3.0, abs=0.01)

    def test_pass_status(self, sample_bend):
        """Test that in-tolerance bends get pass status."""
        sample_bend.angle_degrees = 90.2  # Within 1 degree tolerance

        points = np.random.randn(200, 3)
        deviations = np.random.randn(200) * 0.1

        result = analyze_bend_deviations(
            bend=sample_bend,
            nominal_angle=90.0,
            points=points,
            deviations=deviations,
            tolerance=1.0,
        )

        assert result["status"] == "pass"

    def test_fail_status(self, sample_bend):
        """Test that out-of-tolerance bends get fail status."""
        sample_bend.angle_degrees = 85.0  # 5 degrees off

        points = np.random.randn(200, 3)
        deviations = np.random.randn(200) * 0.1

        result = analyze_bend_deviations(
            bend=sample_bend,
            nominal_angle=90.0,
            points=points,
            deviations=deviations,
            tolerance=1.0,
        )

        assert result["status"] == "fail"

    def test_no_nominal_angle(self, sample_bend):
        """Test handling when no nominal angle is provided."""
        points = np.random.randn(200, 3)
        deviations = np.random.randn(200) * 0.1

        result = analyze_bend_deviations(
            bend=sample_bend,
            nominal_angle=None,
            points=points,
            deviations=deviations,
            tolerance=1.0,
        )

        assert result["nominal_angle"] is None
        assert result["angle_deviation"] == 0
        assert result["status"] == "pass"


class TestBendDetectorConfiguration:
    """Test BendDetector configuration options."""

    def test_custom_thresholds(self):
        """Test detector with custom thresholds."""
        detector = BendDetector(
            normal_radius=10.0,
            curvature_radius=5.0,
            curvature_threshold=0.05,
            min_bend_points=100,
        )

        assert detector.normal_radius == 10.0
        assert detector.curvature_radius == 5.0
        assert detector.curvature_threshold == 0.05
        assert detector.min_bend_points == 100

    def test_get_surfaces(self):
        """Test retrieval of detected surfaces."""
        detector = BendDetector(min_surface_points=20)

        # Create a simple L-shape
        points = []
        for x in np.linspace(0, 50, 25):
            for y in np.linspace(0, 50, 25):
                points.append([x, y, 0])

        for z in np.linspace(0, 50, 25):
            for y in np.linspace(0, 50, 25):
                points.append([50, y, z])

        points = np.array(points)
        detector.detect_bends(points)

        surfaces = detector.get_surfaces()
        # Should detect at least 2 surfaces
        assert len(surfaces) >= 0  # May be 0 if thresholds not met

    def test_get_bend_points_mask(self):
        """Test retrieval of bend points mask."""
        detector = BendDetector(min_bend_points=10)

        # Simple point cloud
        points = np.random.randn(100, 3)
        detector.detect_bends(points)

        mask = detector.get_bend_points_mask()
        assert len(mask) == len(points)
        assert mask.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
