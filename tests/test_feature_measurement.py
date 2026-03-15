"""
Unit Tests for Feature-by-Feature Dimension Measurement System

Tests the complete pipeline for three-way comparison:
- XLSX specification <-> CAD design <-> Scan actual

Test coverage:
- Feature type data structures
- Hole detection in CAD mesh
- CAD feature extraction
- XLSX to CAD feature matching
- Scan feature measurement
- Full pipeline integration
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from feature_detection import (
    FeatureType,
    MeasurableFeature,
    HoleFeature,
    LinearFeature,
    AngleFeature,
    FeatureRegistry,
    FeatureMatch,
    ThreeWayReport,
    HoleDetector,
    CADFeatureExtractor,
    FeatureMatcher,
    ScanFeatureMeasurer,
)


class TestFeatureTypes:
    """Test feature type data structures."""

    def test_feature_type_enum(self):
        """Test FeatureType enum values."""
        assert FeatureType.LINEAR.value == "linear"
        assert FeatureType.DIAMETER.value == "diameter"
        assert FeatureType.RADIUS.value == "radius"
        assert FeatureType.ANGLE.value == "angle"

    def test_hole_feature_creation(self):
        """Test HoleFeature dataclass."""
        hole = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([10, 20, 0]),
            direction=np.array([0, 0, 1]),
            confidence=0.95,
            axis=np.array([0, 0, 1]),
            depth=10.0,
            entry_point=np.array([10, 20, 0]),
        )

        assert hole.feature_id == "hole_1"
        assert hole.nominal_value == 5.8
        assert hole.depth == 10.0
        np.testing.assert_array_equal(hole.axis, [0, 0, 1])

    def test_hole_feature_to_dict(self):
        """Test HoleFeature JSON serialization."""
        hole = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([10.5, 20.3, 0.1]),
            direction=np.array([0, 0, 1]),
            confidence=0.95,
            axis=np.array([0, 0, 1]),
            depth=10.0,
            entry_point=np.array([10.5, 20.3, 0.1]),
        )

        d = hole.to_dict()
        assert d["feature_id"] == "hole_1"
        assert d["nominal_value"] == 5.8
        assert d["position"] == [10.5, 20.3, 0.1]
        assert d["depth"] == 10.0

    def test_linear_feature_creation(self):
        """Test LinearFeature dataclass."""
        linear = LinearFeature(
            feature_id="linear_1",
            feature_type=FeatureType.LINEAR,
            nominal_value=50.0,
            unit="mm",
            position=np.array([25, 0, 0]),
            direction=np.array([1, 0, 0]),
            confidence=0.9,
            point1=np.array([0, 0, 0]),
            point2=np.array([50, 0, 0]),
            measurement_type="overall_x",
        )

        assert linear.feature_id == "linear_1"
        assert linear.nominal_value == 50.0
        np.testing.assert_array_equal(linear.point1, [0, 0, 0])
        np.testing.assert_array_equal(linear.point2, [50, 0, 0])

    def test_angle_feature_creation(self):
        """Test AngleFeature dataclass."""
        angle = AngleFeature(
            feature_id="angle_1",
            feature_type=FeatureType.ANGLE,
            nominal_value=90.0,
            unit="deg",
            position=np.array([25, 0, 0]),
            direction=np.array([0, 1, 0]),
            confidence=0.85,
            surface1_normal=np.array([0, 0, 1]),
            surface2_normal=np.array([1, 0, 0]),
            apex_point=np.array([25, 0, 0]),
        )

        assert angle.feature_id == "angle_1"
        assert angle.nominal_value == 90.0
        np.testing.assert_array_equal(angle.surface1_normal, [0, 0, 1])

    def test_feature_registry_find_by_type_and_value(self):
        """Test FeatureRegistry.find_by_type_and_value."""
        hole1 = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
        )
        hole2 = HoleFeature(
            feature_id="hole_2",
            feature_type=FeatureType.DIAMETER,
            nominal_value=10.0,
            unit="mm",
            position=np.array([50, 0, 0]),
            direction=np.array([0, 0, 1]),
        )

        registry = FeatureRegistry(holes=[hole1, hole2])

        # Find hole with diameter ~5.8
        matches = registry.find_by_type_and_value(FeatureType.DIAMETER, 6.0, tolerance=1.0)
        assert len(matches) == 1
        assert matches[0].nominal_value == 5.8

        # Find hole with diameter ~10
        matches = registry.find_by_type_and_value(FeatureType.DIAMETER, 10.0, tolerance=0.5)
        assert len(matches) == 1
        assert matches[0].nominal_value == 10.0

        # No match
        matches = registry.find_by_type_and_value(FeatureType.DIAMETER, 20.0, tolerance=1.0)
        assert len(matches) == 0


class TestFeatureMatch:
    """Test FeatureMatch and status computation."""

    def test_feature_match_pass(self):
        """Test FeatureMatch with passing measurement."""
        cad_feature = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
        )

        match = FeatureMatch(
            dim_id=1,
            xlsx_spec=None,
            cad_feature=cad_feature,
            xlsx_value=5.8,
            cad_value=5.8,
            scan_value=5.78,
            tolerance_plus=0.1,
            tolerance_minus=0.1,
            feature_type="diameter",
            unit="mm",
        )

        match.compute_status()

        assert match.status == "pass"
        assert match.scan_vs_xlsx == pytest.approx(-0.02, abs=0.001)
        assert match.scan_vs_cad == pytest.approx(-0.02, abs=0.001)

    def test_feature_match_fail(self):
        """Test FeatureMatch with failing measurement."""
        cad_feature = LinearFeature(
            feature_id="linear_1",
            feature_type=FeatureType.LINEAR,
            nominal_value=50.0,
            unit="mm",
            position=np.array([25, 0, 0]),
            direction=np.array([1, 0, 0]),
        )

        match = FeatureMatch(
            dim_id=1,
            xlsx_spec=None,
            cad_feature=cad_feature,
            xlsx_value=50.0,
            cad_value=50.0,
            scan_value=51.5,  # 1.5mm over tolerance
            tolerance_plus=0.25,
            tolerance_minus=0.25,
            feature_type="linear",
            unit="mm",
        )

        match.compute_status()

        assert match.status == "fail"
        assert match.scan_vs_xlsx == pytest.approx(1.5, abs=0.001)

    def test_feature_match_warning(self):
        """Test FeatureMatch with warning status."""
        cad_feature = LinearFeature(
            feature_id="linear_1",
            feature_type=FeatureType.LINEAR,
            nominal_value=50.0,
            unit="mm",
            position=np.array([25, 0, 0]),
            direction=np.array([1, 0, 0]),
        )

        match = FeatureMatch(
            dim_id=1,
            xlsx_spec=None,
            cad_feature=cad_feature,
            xlsx_value=50.0,
            cad_value=50.0,
            scan_value=50.4,  # Over tolerance but within 2x
            tolerance_plus=0.25,
            tolerance_minus=0.25,
            feature_type="linear",
            unit="mm",
        )

        match.compute_status()

        assert match.status == "warning"

    def test_feature_match_not_measured(self):
        """Test FeatureMatch with no scan value."""
        cad_feature = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
        )

        match = FeatureMatch(
            dim_id=1,
            xlsx_spec=None,
            cad_feature=cad_feature,
            xlsx_value=5.8,
            cad_value=5.8,
            scan_value=None,  # Not measured
            feature_type="diameter",
            unit="mm",
        )

        match.compute_status()

        assert match.status == "not_measured"


class TestThreeWayReport:
    """Test ThreeWayReport generation."""

    def test_report_summary(self):
        """Test ThreeWayReport summary computation."""
        matches = []
        for i in range(5):
            cad_feature = LinearFeature(
                feature_id=f"linear_{i}",
                feature_type=FeatureType.LINEAR,
                nominal_value=10.0 * (i + 1),
                unit="mm",
                position=np.array([0, 0, 0]),
                direction=np.array([1, 0, 0]),
            )
            match = FeatureMatch(
                dim_id=i + 1,
                xlsx_spec=None,
                cad_feature=cad_feature,
                xlsx_value=10.0 * (i + 1),
                cad_value=10.0 * (i + 1),
                scan_value=10.0 * (i + 1) + (0.05 if i < 3 else 1.0),  # 3 pass, 2 fail
                tolerance_plus=0.25,
                tolerance_minus=0.25,
                feature_type="linear",
                unit="mm",
            )
            match.compute_status()
            matches.append(match)

        report = ThreeWayReport(
            part_id="test_part",
            matches=matches,
        )
        report.compute_summary()

        assert report.total_dimensions == 5
        assert report.measured_count == 5
        assert report.pass_count == 3
        assert report.fail_count == 2

    def test_report_to_dict(self):
        """Test ThreeWayReport JSON serialization."""
        cad_feature = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
        )

        match = FeatureMatch(
            dim_id=1,
            xlsx_spec=None,
            cad_feature=cad_feature,
            xlsx_value=5.8,
            cad_value=5.8,
            scan_value=5.78,
            tolerance_plus=0.1,
            tolerance_minus=0.1,
            feature_type="diameter",
            unit="mm",
        )
        match.compute_status()

        report = ThreeWayReport(
            part_id="test_part",
            matches=[match],
        )

        d = report.to_dict()
        assert d["part_id"] == "test_part"
        assert d["summary"]["total_dimensions"] == 1
        assert d["summary"]["passed"] == 1
        assert len(d["matches"]) == 1


class TestHoleDetector:
    """Test hole detection in CAD mesh."""

    @pytest.fixture
    def detector(self):
        """Create hole detector."""
        return HoleDetector(
            min_diameter=0.5,
            max_diameter=100.0,
            circularity_threshold=0.85,
        )

    def test_taubin_circle_fit(self, detector):
        """Test Taubin algebraic circle fitting."""
        # Generate perfect circle points
        n_points = 50
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = 5.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points_2d = np.column_stack([x, y])

        result = detector._taubin_circle_fit(points_2d)

        assert result is not None
        center, fitted_radius, circularity = result
        np.testing.assert_array_almost_equal(center, [0, 0], decimal=2)
        assert fitted_radius == pytest.approx(5.0, abs=0.1)
        assert circularity > 0.95

    def test_taubin_circle_fit_with_noise(self, detector):
        """Test Taubin circle fitting with noisy points."""
        np.random.seed(42)
        n_points = 100
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = 10.0
        noise = 0.2

        x = radius * np.cos(angles) + np.random.randn(n_points) * noise
        y = radius * np.sin(angles) + np.random.randn(n_points) * noise
        points_2d = np.column_stack([x, y])

        result = detector._taubin_circle_fit(points_2d)

        assert result is not None
        center, fitted_radius, circularity = result
        np.testing.assert_array_almost_equal(center, [0, 0], decimal=1)
        assert fitted_radius == pytest.approx(10.0, abs=0.5)
        assert circularity > 0.9


class TestFeatureMatcher:
    """Test XLSX to CAD feature matching."""

    @pytest.fixture
    def matcher(self):
        """Create feature matcher."""
        return FeatureMatcher(value_tolerance=2.0, angle_tolerance=10.0)

    @pytest.fixture
    def mock_dimension_spec(self):
        """Create mock DimensionSpec."""
        mock = Mock()
        mock.dim_id = 1
        mock.value = 5.8
        mock.tolerance_plus = 0.1
        mock.tolerance_minus = 0.1
        mock.description = "Test dimension"
        mock.unit = "mm"

        # Mock DimensionType
        mock_dim_type = Mock()
        mock_dim_type.value = "diameter"
        mock.dim_type = mock_dim_type

        return mock

    def test_match_single_diameter(self, matcher, mock_dimension_spec):
        """Test matching single diameter dimension."""
        hole = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([10, 20, 0]),
            direction=np.array([0, 0, 1]),
        )

        registry = FeatureRegistry(holes=[hole])

        # Mock dimension_type_to_feature_type
        import feature_detection.feature_matcher as fm
        original_func = fm.dimension_type_to_feature_type
        fm.dimension_type_to_feature_type = lambda dt: FeatureType.DIAMETER

        try:
            matches, unmatched_xlsx, unmatched_cad = matcher.match(
                [mock_dimension_spec],
                registry
            )

            assert len(matches) == 1
            assert len(unmatched_xlsx) == 0
            assert len(unmatched_cad) == 0
            assert matches[0].xlsx_value == 5.8
            assert matches[0].cad_value == 5.8
        finally:
            fm.dimension_type_to_feature_type = original_func

    def test_match_multiple_similar_values(self, matcher):
        """Test matching when multiple CAD features have similar values."""
        holes = [
            HoleFeature(
                feature_id=f"hole_{i}",
                feature_type=FeatureType.DIAMETER,
                nominal_value=5.8,
                unit="mm",
                position=np.array([10 * i, 0, 0]),
                direction=np.array([0, 0, 1]),
            )
            for i in range(3)
        ]

        registry = FeatureRegistry(holes=holes)

        # Create 3 mock specs with same value
        mock_specs = []
        for i in range(3):
            mock = Mock()
            mock.dim_id = i + 1
            mock.value = 5.8
            mock.tolerance_plus = 0.1
            mock.tolerance_minus = 0.1
            mock.description = f"Hole {i+1}"
            mock.unit = "mm"
            mock_dim_type = Mock()
            mock_dim_type.value = "diameter"
            mock.dim_type = mock_dim_type
            mock_specs.append(mock)

        import feature_detection.feature_matcher as fm
        original_func = fm.dimension_type_to_feature_type
        fm.dimension_type_to_feature_type = lambda dt: FeatureType.DIAMETER

        try:
            matches, unmatched_xlsx, unmatched_cad = matcher.match(
                mock_specs,
                registry
            )

            # All 3 should match (one-to-one)
            assert len(matches) == 3
            assert len(unmatched_xlsx) == 0
            assert len(unmatched_cad) == 0

            # Each match should have unique CAD feature
            cad_ids = {id(m.cad_feature) for m in matches}
            assert len(cad_ids) == 3
        finally:
            fm.dimension_type_to_feature_type = original_func


class TestScanFeatureMeasurer:
    """Test scan feature measurement."""

    @pytest.fixture
    def measurer(self):
        """Create scan feature measurer."""
        return ScanFeatureMeasurer(
            search_radius=10.0,
            min_points_hole=20,
            ransac_iterations=100,
        )

    @pytest.fixture
    def hole_point_cloud(self):
        """Generate synthetic hole point cloud."""
        np.random.seed(42)

        # Create points around a cylindrical hole
        diameter = 5.8
        radius = diameter / 2
        n_rim_points = 200
        noise = 0.05

        # Rim points
        angles = np.linspace(0, 2 * np.pi, n_rim_points, endpoint=False)
        x = radius * np.cos(angles) + np.random.randn(n_rim_points) * noise
        y = radius * np.sin(angles) + np.random.randn(n_rim_points) * noise
        z = np.random.randn(n_rim_points) * 0.5  # Spread along z

        # Add some surrounding surface points
        n_surface = 500
        surf_x = np.random.uniform(-20, 20, n_surface)
        surf_y = np.random.uniform(-20, 20, n_surface)
        # Exclude points near hole
        dist = np.sqrt(surf_x**2 + surf_y**2)
        mask = dist > radius + 1
        surf_x = surf_x[mask]
        surf_y = surf_y[mask]
        surf_z = np.random.randn(len(surf_x)) * 0.05

        all_x = np.concatenate([x, surf_x])
        all_y = np.concatenate([y, surf_y])
        all_z = np.concatenate([z, surf_z])

        return np.column_stack([all_x, all_y, all_z])

    def test_measure_hole_diameter(self, measurer, hole_point_cloud):
        """Test diameter measurement in point cloud."""
        hole = HoleFeature(
            feature_id="hole_1",
            feature_type=FeatureType.DIAMETER,
            nominal_value=5.8,
            unit="mm",
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
            axis=np.array([0, 0, 1]),
            depth=10.0,
            entry_point=np.array([0, 0, 0]),
        )

        measured = measurer.measure_hole(hole_point_cloud, hole)

        assert measured is not None
        # Should be close to 5.8mm (within measurement tolerance)
        assert abs(measured - 5.8) < 0.5

    def test_measure_linear_dimension(self, measurer):
        """Test linear dimension measurement."""
        np.random.seed(42)

        # Create two flat surfaces 50mm apart
        n_points_per_surface = 200
        noise = 0.05

        # Surface 1 at x=0
        s1_x = np.random.randn(n_points_per_surface) * noise
        s1_y = np.random.uniform(-10, 10, n_points_per_surface)
        s1_z = np.random.uniform(-10, 10, n_points_per_surface)

        # Surface 2 at x=50
        s2_x = 50 + np.random.randn(n_points_per_surface) * noise
        s2_y = np.random.uniform(-10, 10, n_points_per_surface)
        s2_z = np.random.uniform(-10, 10, n_points_per_surface)

        points = np.column_stack([
            np.concatenate([s1_x, s2_x]),
            np.concatenate([s1_y, s2_y]),
            np.concatenate([s1_z, s2_z]),
        ])

        linear = LinearFeature(
            feature_id="linear_1",
            feature_type=FeatureType.LINEAR,
            nominal_value=50.0,
            unit="mm",
            position=np.array([25, 0, 0]),
            direction=np.array([1, 0, 0]),
            point1=np.array([0, 0, 0]),
            point2=np.array([50, 0, 0]),
            measurement_type="surface_to_surface_x",
        )

        measured = measurer.measure_linear(points, linear)

        assert measured is not None
        # Should be close to 50mm
        assert abs(measured - 50.0) < 1.0


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_mock(self):
        """Test full pipeline with mock data."""
        # Create mock CAD features
        holes = [
            HoleFeature(
                feature_id="hole_1",
                feature_type=FeatureType.DIAMETER,
                nominal_value=5.8,
                unit="mm",
                position=np.array([10, 20, 0]),
                direction=np.array([0, 0, 1]),
                axis=np.array([0, 0, 1]),
                depth=10.0,
                entry_point=np.array([10, 20, 0]),
            ),
        ]

        linear_dims = [
            LinearFeature(
                feature_id="linear_1",
                feature_type=FeatureType.LINEAR,
                nominal_value=50.0,
                unit="mm",
                position=np.array([25, 0, 0]),
                direction=np.array([1, 0, 0]),
                point1=np.array([0, 0, 0]),
                point2=np.array([50, 0, 0]),
            ),
        ]

        angles = [
            AngleFeature(
                feature_id="angle_1",
                feature_type=FeatureType.ANGLE,
                nominal_value=90.0,
                unit="deg",
                position=np.array([25, 0, 0]),
                direction=np.array([0, 1, 0]),
                surface1_normal=np.array([0, 0, 1]),
                surface2_normal=np.array([1, 0, 0]),
                apex_point=np.array([25, 0, 0]),
            ),
        ]

        registry = FeatureRegistry(
            holes=holes,
            linear_dims=linear_dims,
            angles=angles,
        )

        # Create mock XLSX specs
        mock_specs = []

        # Diameter spec
        mock_dia = Mock()
        mock_dia.dim_id = 1
        mock_dia.value = 5.8
        mock_dia.tolerance_plus = 0.1
        mock_dia.tolerance_minus = 0.1
        mock_dia.description = "Hole diameter"
        mock_dia.unit = "mm"
        mock_dia_type = Mock()
        mock_dia_type.value = "diameter"
        mock_dia.dim_type = mock_dia_type
        mock_specs.append(mock_dia)

        # Linear spec
        mock_lin = Mock()
        mock_lin.dim_id = 2
        mock_lin.value = 50.0
        mock_lin.tolerance_plus = 0.25
        mock_lin.tolerance_minus = 0.25
        mock_lin.description = "Overall length"
        mock_lin.unit = "mm"
        mock_lin_type = Mock()
        mock_lin_type.value = "linear"
        mock_lin.dim_type = mock_lin_type
        mock_specs.append(mock_lin)

        # Angle spec
        mock_ang = Mock()
        mock_ang.dim_id = 3
        mock_ang.value = 90.0
        mock_ang.tolerance_plus = 1.0
        mock_ang.tolerance_minus = 1.0
        mock_ang.description = "Bend angle"
        mock_ang.unit = "deg"
        mock_ang_type = Mock()
        mock_ang_type.value = "angle"
        mock_ang.dim_type = mock_ang_type
        mock_specs.append(mock_ang)

        # Patch dimension_type_to_feature_type
        import feature_detection.feature_matcher as fm

        def mock_type_to_feature(dt):
            val = dt.value
            return {
                "diameter": FeatureType.DIAMETER,
                "linear": FeatureType.LINEAR,
                "angle": FeatureType.ANGLE,
            }.get(val, FeatureType.LINEAR)

        original_func = fm.dimension_type_to_feature_type
        fm.dimension_type_to_feature_type = mock_type_to_feature

        try:
            # Run matcher
            matcher = FeatureMatcher()
            matches, unmatched_xlsx, unmatched_cad = matcher.match(mock_specs, registry)

            # All should match
            assert len(matches) == 3
            assert len(unmatched_xlsx) == 0
            assert len(unmatched_cad) == 0

            # Simulate scan measurements
            for match in matches:
                if match.feature_type == "diameter":
                    match.scan_value = 5.78  # Good measurement
                elif match.feature_type == "linear":
                    match.scan_value = 50.05  # Good measurement
                elif match.feature_type == "angle":
                    match.scan_value = 89.85  # Good measurement
                match.compute_status()

            # Generate report
            report = ThreeWayReport(
                part_id="test_part",
                matches=matches,
            )
            report.compute_summary()

            # All should pass
            assert report.pass_count == 3
            assert report.fail_count == 0

            # Check table output
            table_str = report.to_table_string()
            assert "test_part" in table_str
            assert "PASS" in table_str

        finally:
            fm.dimension_type_to_feature_type = original_func


def create_test_mesh_with_holes(diameters):
    """
    Create a test mesh with holes of specified diameters.

    This is a helper function for testing hole detection on actual geometry.
    """
    try:
        import open3d as o3d
    except ImportError:
        return None

    # Create a simple box
    box = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=10)
    box.compute_vertex_normals()

    # Note: Actually adding cylindrical holes to the mesh would require
    # more complex geometry operations. For unit tests, we use simpler
    # approaches or mock the mesh.

    return box


def create_hole_point_cloud(diameter, noise=0.05, n_points=200):
    """
    Create a synthetic point cloud representing a hole.

    Args:
        diameter: Hole diameter in mm
        noise: Noise level in mm
        n_points: Number of points on the rim

    Returns:
        Nx3 numpy array of points
    """
    np.random.seed(42)
    radius = diameter / 2

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles) + np.random.randn(n_points) * noise
    y = radius * np.sin(angles) + np.random.randn(n_points) * noise
    z = np.random.randn(n_points) * 0.5

    return np.column_stack([x, y, z])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
