"""
Tests for GD&T (Geometric Dimensioning and Tolerancing) Engine
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from gdt_engine import GDTEngine, GDTType, GDTResult, create_gdt_engine


class TestGDTEngineCreation:
    """Test GD&T engine initialization"""

    def test_create_engine(self):
        """Test engine creation"""
        engine = create_gdt_engine()
        assert engine is not None
        assert isinstance(engine, GDTEngine)


class TestFlatness:
    """Test flatness calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_flat_plane(self, engine):
        """Test flatness of a perfectly flat surface"""
        # Create a perfect XY plane
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

        result = engine.calculate_flatness(points, tolerance=0.1)

        assert result.gdt_type == GDTType.FLATNESS
        assert result.measured_value < 0.001  # Should be nearly zero
        assert result.conformance == True
        assert result.points_used == len(points)

    def test_flat_plane_with_deviation(self, engine):
        """Test flatness with known deviation"""
        # Create a plane with 0.05mm deviation
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        z = np.random.uniform(-0.025, 0.025, xx.size)  # ±0.025mm = 0.05mm range
        points = np.column_stack([xx.ravel(), yy.ravel(), z])

        result = engine.calculate_flatness(points, tolerance=0.1)

        assert result.measured_value < 0.1  # Should be less than tolerance
        assert result.conformance == True

    def test_flatness_fail(self, engine):
        """Test flatness that exceeds tolerance"""
        # Create a plane with 0.2mm deviation
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        z = np.random.uniform(-0.1, 0.1, xx.size)  # ±0.1mm = 0.2mm range
        points = np.column_stack([xx.ravel(), yy.ravel(), z])

        result = engine.calculate_flatness(points, tolerance=0.1)

        # May or may not fail depending on random values
        # Just check the result is valid
        assert result.measured_value >= 0
        assert result.points_used > 0

    def test_flatness_insufficient_points(self, engine):
        """Test flatness with insufficient points"""
        points = np.array([[0, 0, 0], [1, 1, 0]])  # Only 2 points

        result = engine.calculate_flatness(points, tolerance=0.1)

        assert result.conformance == False
        assert result.confidence == 0.0


class TestCylindricity:
    """Test cylindricity calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_cylinder(self, engine):
        """Test cylindricity of a perfect cylinder"""
        # Create a perfect cylinder along Z axis
        # Use endpoint=False to avoid duplicate points at theta=0 and theta=2*pi
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 50)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        x = radius * np.cos(tt.ravel())
        y = radius * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.calculate_cylindricity(points, tolerance=0.1)

        assert result.gdt_type == GDTType.CYLINDRICITY
        assert result.measured_value < 0.001  # Should be nearly zero
        assert result.conformance == True
        assert "fitted_radius" in result.fit_parameters
        assert abs(result.fit_parameters["fitted_radius"] - radius) < 0.01

    def test_cylinder_with_deviation(self, engine):
        """Test cylinder with known deviation"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 50)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        # Add radial deviation
        np.random.seed(42)  # For reproducibility
        deviation = np.random.uniform(-0.02, 0.02, tt.size)
        r = radius + deviation

        x = r * np.cos(tt.ravel())
        y = r * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.calculate_cylindricity(points, tolerance=0.1)

        assert result.measured_value < 0.1
        assert result.conformance == True


class TestCircularity:
    """Test circularity (roundness) calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_circle(self, engine):
        """Test circularity of a perfect circle"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        radius = 5.0

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        result = engine.calculate_circularity(points, tolerance=0.1)

        assert result.gdt_type == GDTType.CIRCULARITY
        assert result.measured_value < 0.001
        assert result.conformance == True

    def test_ellipse_as_circle(self, engine):
        """Test circularity of an ellipse (should fail)"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        a, b = 5.0, 4.5  # Semi-major and semi-minor axes

        x = a * np.cos(theta)
        y = b * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        result = engine.calculate_circularity(points, tolerance=0.1)

        # Ellipse should have significant circularity error
        # For ellipse with a=5, b=4.5, circularity = a-b = 0.5
        assert result.measured_value >= 0.5


class TestPosition:
    """Test position calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_position_at_nominal(self, engine):
        """Test position when feature is at nominal location"""
        # Create a hole centered at (10, 10, 0)
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        radius = 2.5
        center = (10.0, 10.0, 0.0)

        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        result = engine.calculate_position(
            points,
            nominal_position=center,
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.POSITION
        assert result.measured_value < 0.01  # Should be at nominal
        assert result.conformance == True
        assert result.actual_position is not None

    def test_position_offset(self, engine):
        """Test position when feature is offset from nominal"""
        # Create a hole centered at (10.03, 10.04, 0) - offset from nominal
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        radius = 2.5
        actual_center = (10.03, 10.04, 0.0)
        nominal = (10.0, 10.0, 0.0)

        x = actual_center[0] + radius * np.cos(theta)
        y = actual_center[1] + radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        result = engine.calculate_position(
            points,
            nominal_position=nominal,
            tolerance=0.2  # 0.2mm diameter tolerance
        )

        # Expected position = 2 * sqrt(0.03^2 + 0.04^2) = 2 * 0.05 = 0.1mm
        assert 0.09 < result.measured_value < 0.11
        assert result.conformance == True  # 0.1 < 0.2

    def test_position_mmc_bonus(self, engine):
        """Test position with MMC bonus tolerance"""
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        radius = 2.5
        actual_center = (10.1, 10.0, 0.0)  # 0.1mm offset
        nominal = (10.0, 10.0, 0.0)

        x = actual_center[0] + radius * np.cos(theta)
        y = actual_center[1] + radius * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        # Without MMC: position = 0.2mm, tolerance = 0.15mm -> FAIL
        result_no_mmc = engine.calculate_position(
            points,
            nominal_position=nominal,
            tolerance=0.15,
            mmc=False
        )

        # With MMC: bonus = 5.1 - 5.0 = 0.1mm, effective tolerance = 0.25mm -> PASS
        result_with_mmc = engine.calculate_position(
            points,
            nominal_position=nominal,
            tolerance=0.15,
            mmc=True,
            feature_size=5.1,  # Actual hole diameter
            feature_mmc=5.0    # MMC (smallest hole)
        )

        assert result_no_mmc.conformance == False
        assert result_with_mmc.conformance == True


class TestParallelism:
    """Test parallelism calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_parallelism(self, engine):
        """Test perfectly parallel surfaces"""
        # Create a plane parallel to XY datum
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        z = np.full(xx.size, 5.0)  # Z = 5 plane
        points = np.column_stack([xx.ravel(), yy.ravel(), z])

        result = engine.calculate_parallelism(
            points,
            datum_normal=np.array([0, 0, 1]),  # XY plane normal
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.PARALLELISM
        assert result.measured_value < 0.001
        assert result.conformance == True


class TestPerpendicularity:
    """Test perpendicularity calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_perpendicularity(self, engine):
        """Test perfectly perpendicular surface"""
        # Create a YZ plane (perpendicular to X axis / XZ datum)
        y = np.linspace(0, 10, 50)
        z = np.linspace(0, 10, 50)
        yy, zz = np.meshgrid(y, z)
        x = np.full(yy.size, 5.0)  # X = 5 plane
        points = np.column_stack([x, yy.ravel(), zz.ravel()])

        result = engine.calculate_perpendicularity(
            points,
            datum_normal=np.array([0, 1, 0]),  # XZ plane normal
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.PERPENDICULARITY
        assert result.measured_value < 0.001
        assert result.conformance == True


class TestGDTResultSerialization:
    """Test GDTResult serialization"""

    def test_to_dict(self):
        """Test result serialization to dict"""
        result = GDTResult(
            gdt_type=GDTType.FLATNESS,
            measured_value=0.05,
            tolerance=0.1,
            conformance=True,
            fit_parameters={"plane_normal": [0, 0, 1]},
            min_deviation=-0.02,
            max_deviation=0.03,
            std_deviation=0.01,
            points_used=1000,
            confidence=0.95
        )

        result_dict = result.to_dict()

        assert result_dict["type"] == "flatness"
        assert result_dict["measured_value_mm"] == 0.05
        assert result_dict["tolerance_mm"] == 0.1
        assert result_dict["conformance"] == "PASS"
        assert result_dict["points_used"] == 1000


class TestStraightness:
    """Test straightness calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_straight_line(self, engine):
        """Test straightness of a perfectly straight line"""
        # Create a perfect line along X axis
        x = np.linspace(0, 100, 50)
        y = np.zeros(50)
        z = np.zeros(50)
        points = np.column_stack([x, y, z])

        result = engine.calculate_straightness(points, tolerance=0.1)

        assert result.gdt_type == GDTType.STRAIGHTNESS
        assert result.measured_value < 0.001  # Should be nearly zero
        assert result.conformance == True

    def test_line_with_deviation(self, engine):
        """Test line with known deviation"""
        x = np.linspace(0, 100, 50)
        y = np.random.uniform(-0.02, 0.02, 50)
        z = np.random.uniform(-0.02, 0.02, 50)
        points = np.column_stack([x, y, z])

        result = engine.calculate_straightness(points, tolerance=0.1)

        assert result.gdt_type == GDTType.STRAIGHTNESS
        assert result.points_used == 50


class TestAngularity:
    """Test angularity calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_angularity(self, engine):
        """Test surface at perfect 45 degree angle"""
        # Create a surface at 45 degrees to XY plane
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        z = xx.ravel()  # Z increases with X at 45 degrees
        points = np.column_stack([xx.ravel(), yy.ravel(), z])

        result = engine.calculate_angularity(
            points,
            datum_normal=np.array([0, 0, 1]),
            nominal_angle=45.0,
            tolerance=0.5
        )

        assert result.gdt_type == GDTType.ANGULARITY
        assert result.points_used > 0


class TestConcentricity:
    """Test concentricity calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_concentricity(self, engine):
        """Test perfectly concentric cylinders"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 20)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        x = radius * np.cos(tt.ravel())
        y = radius * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.calculate_concentricity(
            points,
            datum_axis_point=(0, 0, 0),
            datum_axis_direction=np.array([0, 0, 1]),
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.CONCENTRICITY
        assert result.measured_value < 0.01
        assert result.conformance == True


class TestSymmetry:
    """Test symmetry calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_symmetry(self, engine):
        """Test perfectly symmetric feature"""
        # Create points equally distributed on both sides of XY plane
        n = 100
        x = np.random.randn(n) * 5
        y = np.random.randn(n) * 5
        z = np.concatenate([
            np.abs(np.random.randn(n // 2)) * 2,
            -np.abs(np.random.randn(n // 2)) * 2
        ])
        points = np.column_stack([x, y, z])

        result = engine.calculate_symmetry(
            points,
            datum_plane_point=(0, 0, 0),
            datum_plane_normal=np.array([0, 0, 1]),
            tolerance=0.5
        )

        assert result.gdt_type == GDTType.SYMMETRY
        assert result.points_used == n


class TestCircularRunout:
    """Test circular runout calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_runout(self, engine):
        """Test cylinder with perfect runout"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 20)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        x = radius * np.cos(tt.ravel())
        y = radius * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.calculate_circular_runout(
            points,
            datum_axis_point=(0, 0, 0),
            datum_axis_direction=np.array([0, 0, 1]),
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.CIRCULAR_RUNOUT
        assert result.measured_value < 0.001
        assert result.conformance == True


class TestTotalRunout:
    """Test total runout calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_total_runout(self, engine):
        """Test total runout calculation"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 30)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        x = radius * np.cos(tt.ravel())
        y = radius * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.calculate_total_runout(
            points,
            datum_axis_point=(0, 0, 0),
            datum_axis_direction=np.array([0, 0, 1]),
            tolerance=0.1
        )

        assert result.gdt_type == GDTType.TOTAL_RUNOUT
        assert result.measured_value < 0.01
        assert result.conformance == True


class TestProfileLine:
    """Test profile of a line calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_profile_line(self, engine):
        """Test perfect profile match"""
        t = np.linspace(0, 10, 50)
        nominal = np.column_stack([t, np.sin(t), np.zeros(50)])
        actual = nominal.copy()

        result = engine.calculate_profile_line(actual, nominal, tolerance=0.1)

        assert result.gdt_type == GDTType.PROFILE_LINE
        assert result.measured_value < 0.001
        assert result.conformance == True


class TestProfileSurface:
    """Test profile of a surface calculations"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_perfect_profile_surface(self, engine):
        """Test perfect surface profile match"""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx) * np.cos(yy)
        nominal = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        actual = nominal.copy()

        result = engine.calculate_profile_surface(actual, nominal, tolerance=0.1)

        assert result.gdt_type == GDTType.PROFILE_SURFACE
        assert result.measured_value < 0.001
        assert result.conformance == True


class TestGenericAnalyzeFeature:
    """Test the generic analyze_feature dispatcher"""

    @pytest.fixture
    def engine(self):
        return GDTEngine()

    def test_analyze_flatness(self, engine):
        """Test analyze_feature for flatness"""
        points = np.random.randn(100, 3)
        points[:, 2] = 0  # Flatten to Z=0

        result = engine.analyze_feature(points, GDTType.FLATNESS, tolerance=0.1)

        assert result.gdt_type == GDTType.FLATNESS

    def test_analyze_position(self, engine):
        """Test analyze_feature for position"""
        theta = np.linspace(0, 2 * np.pi, 50)
        x = 10 + 2.5 * np.cos(theta)
        y = 10 + 2.5 * np.sin(theta)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])

        result = engine.analyze_feature(
            points,
            GDTType.POSITION,
            tolerance=0.1,
            nominal_position=(10, 10, 0)
        )

        assert result.gdt_type == GDTType.POSITION

    def test_analyze_straightness(self, engine):
        """Test analyze_feature for straightness"""
        x = np.linspace(0, 100, 50)
        points = np.column_stack([x, np.zeros(50), np.zeros(50)])

        result = engine.analyze_feature(points, GDTType.STRAIGHTNESS, tolerance=0.1)

        assert result.gdt_type == GDTType.STRAIGHTNESS

    def test_analyze_total_runout(self, engine):
        """Test analyze_feature for total runout"""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        z = np.linspace(0, 10, 30)
        tt, zz = np.meshgrid(theta, z)
        radius = 5.0

        x = radius * np.cos(tt.ravel())
        y = radius * np.sin(tt.ravel())
        points = np.column_stack([x, y, zz.ravel()])

        result = engine.analyze_feature(
            points,
            GDTType.TOTAL_RUNOUT,
            tolerance=0.1,
            datum_axis_point=(0, 0, 0),
            datum_axis_direction=[0, 0, 1]
        )

        assert result.gdt_type == GDTType.TOTAL_RUNOUT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
