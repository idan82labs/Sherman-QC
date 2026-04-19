"""
GD&T (Geometric Dimensioning and Tolerancing) Engine

Implements basic GD&T calculations per ASME Y14.5M:
- Flatness (Form tolerance)
- Cylindricity (Form tolerance)
- Position (Location tolerance)
- Parallelism (Orientation tolerance)
- Perpendicularity (Orientation tolerance)
- Circularity/Roundness (Form tolerance)

Uses 3D point cloud data and least-squares fitting algorithms.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GDTType(Enum):
    """GD&T characteristic types per ASME Y14.5"""
    # Form tolerances
    FLATNESS = "flatness"
    STRAIGHTNESS = "straightness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"

    # Orientation tolerances
    PARALLELISM = "parallelism"
    PERPENDICULARITY = "perpendicularity"
    ANGULARITY = "angularity"

    # Location tolerances
    POSITION = "position"
    CONCENTRICITY = "concentricity"
    SYMMETRY = "symmetry"

    # Profile tolerances
    PROFILE_LINE = "profile_of_a_line"
    PROFILE_SURFACE = "profile_of_a_surface"

    # Runout tolerances
    CIRCULAR_RUNOUT = "circular_runout"
    TOTAL_RUNOUT = "total_runout"


@dataclass
class GDTResult:
    """Result of a GD&T calculation"""
    gdt_type: GDTType
    measured_value: float  # Actual measured tolerance zone
    tolerance: float  # Specified tolerance
    conformance: bool  # Pass/Fail

    # Fit parameters
    fit_parameters: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    min_deviation: float = 0.0
    max_deviation: float = 0.0
    std_deviation: float = 0.0

    # For position: actual position vs nominal
    actual_position: Optional[Tuple[float, float, float]] = None
    nominal_position: Optional[Tuple[float, float, float]] = None

    # Confidence
    points_used: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "type": self.gdt_type.value,
            "measured_value_mm": round(self.measured_value, 4),
            "tolerance_mm": round(self.tolerance, 4),
            "conformance": "PASS" if self.conformance else "FAIL",
            "fit_parameters": self.fit_parameters,
            "statistics": {
                "min_deviation_mm": round(self.min_deviation, 4),
                "max_deviation_mm": round(self.max_deviation, 4),
                "std_deviation_mm": round(self.std_deviation, 4)
            },
            "actual_position": self.actual_position,
            "nominal_position": self.nominal_position,
            "points_used": self.points_used,
            "confidence": round(self.confidence, 2)
        }


class GDTEngine:
    """
    Engine for computing GD&T values from 3D point cloud data.

    Usage:
        engine = GDTEngine()
        result = engine.calculate_flatness(points, tolerance=0.1)
    """

    def __init__(self):
        self._scipy = None

    @property
    def scipy(self):
        """Lazy-load scipy"""
        if self._scipy is None:
            import scipy
            self._scipy = scipy
        return self._scipy

    def calculate_flatness(
        self,
        points: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate flatness - the zone between two parallel planes containing all points.

        ASME Y14.5: Flatness is a condition of a surface or derived median plane
        having all elements in one plane.

        Args:
            points: Nx3 array of surface points
            tolerance: Flatness tolerance in mm

        Returns:
            GDTResult with measured flatness value
        """
        if len(points) < 3:
            return GDTResult(
                gdt_type=GDTType.FLATNESS,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Fit a plane using SVD (least squares)
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]  # Normal vector of best-fit plane

        # Calculate signed distances to the plane
        distances = np.dot(centered, normal)

        # Flatness is the range (max - min) of distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        flatness = max_dist - min_dist

        return GDTResult(
            gdt_type=GDTType.FLATNESS,
            measured_value=flatness,
            tolerance=tolerance,
            conformance=flatness <= tolerance,
            fit_parameters={
                "plane_normal": normal.tolist(),
                "plane_centroid": centroid.tolist()
            },
            min_deviation=min_dist,
            max_deviation=max_dist,
            std_deviation=float(np.std(distances)),
            points_used=len(points),
            confidence=self._calculate_confidence(len(points), 100)
        )

    def calculate_cylindricity(
        self,
        points: np.ndarray,
        tolerance: float,
        axis_hint: Optional[np.ndarray] = None
    ) -> GDTResult:
        """
        Calculate cylindricity - the zone between two coaxial cylinders.

        ASME Y14.5: Cylindricity is a condition of a surface of revolution
        where all points are equidistant from a common axis.

        Args:
            points: Nx3 array of cylinder surface points
            tolerance: Cylindricity tolerance in mm
            axis_hint: Optional hint for cylinder axis direction [x, y, z]

        Returns:
            GDTResult with measured cylindricity value
        """
        if len(points) < 6:
            return GDTResult(
                gdt_type=GDTType.CYLINDRICITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        centroid = np.mean(points, axis=0)
        centered = points - centroid

        if axis_hint is not None:
            # Use provided axis hint
            axis = axis_hint / np.linalg.norm(axis_hint)
            cylindricity, mean_radius, min_dev, max_dev, std_dev = self._fit_cylinder_to_axis(
                centered, axis
            )
        else:
            # Try all 3 principal components and pick best fit
            _, _, vh = np.linalg.svd(centered, full_matrices=False)

            best_cylindricity = float('inf')
            best_axis = vh[0]
            best_radius = 0.0
            best_min_dev = 0.0
            best_max_dev = 0.0
            best_std_dev = 0.0

            for i in range(3):
                axis_candidate = vh[i]
                cyl, rad, min_d, max_d, std_d = self._fit_cylinder_to_axis(
                    centered, axis_candidate
                )
                if cyl < best_cylindricity:
                    best_cylindricity = cyl
                    best_axis = axis_candidate
                    best_radius = rad
                    best_min_dev = min_d
                    best_max_dev = max_d
                    best_std_dev = std_d

            axis = best_axis
            cylindricity = best_cylindricity
            mean_radius = best_radius
            min_dev = best_min_dev
            max_dev = best_max_dev
            std_dev = best_std_dev

        return GDTResult(
            gdt_type=GDTType.CYLINDRICITY,
            measured_value=cylindricity,
            tolerance=tolerance,
            conformance=cylindricity <= tolerance,
            fit_parameters={
                "axis_direction": axis.tolist(),
                "axis_point": centroid.tolist(),
                "fitted_radius": float(mean_radius)
            },
            min_deviation=min_dev,
            max_deviation=max_dev,
            std_deviation=float(std_dev),
            points_used=len(points),
            confidence=self._calculate_confidence(len(points), 500)
        )

    def _fit_cylinder_to_axis(
        self,
        centered_points: np.ndarray,
        axis: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Fit a cylinder with given axis and return cylindricity metrics.

        Returns:
            (cylindricity, mean_radius, min_deviation, max_deviation, std_deviation)
        """
        # Project points onto the axis and find perpendicular distances
        projections = np.dot(centered_points, axis)[:, np.newaxis] * axis
        perpendicular = centered_points - projections

        # Calculate radial distances from axis
        radii = np.linalg.norm(perpendicular, axis=1)

        # Fit best cylinder radius
        mean_radius = np.mean(radii)
        deviations = radii - mean_radius

        # Cylindricity is the range of radial deviations
        min_dev = float(np.min(deviations))
        max_dev = float(np.max(deviations))
        cylindricity = max_dev - min_dev
        std_dev = float(np.std(deviations))

        return cylindricity, mean_radius, min_dev, max_dev, std_dev

    def calculate_circularity(
        self,
        points: np.ndarray,
        tolerance: float,
        plane_normal: Optional[np.ndarray] = None
    ) -> GDTResult:
        """
        Calculate circularity (roundness) - the zone between two concentric circles.

        ASME Y14.5: Circularity is a condition of a surface of revolution
        where all points are equidistant from the axis (in a plane perpendicular to axis).

        Args:
            points: Nx3 array of circular feature points
            tolerance: Circularity tolerance in mm
            plane_normal: Normal of the plane containing the circle

        Returns:
            GDTResult with measured circularity value
        """
        if len(points) < 4:
            return GDTResult(
                gdt_type=GDTType.CIRCULARITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # If no plane normal provided, fit one
        if plane_normal is None:
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            plane_normal = vh[-1]

        # Project points onto the plane
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        distances_to_plane = np.dot(centered, plane_normal)
        projected = centered - distances_to_plane[:, np.newaxis] * plane_normal

        # Calculate radial distances from center
        radii = np.linalg.norm(projected, axis=1)

        # Mean radius
        mean_radius = np.mean(radii)
        deviations = radii - mean_radius

        # Circularity is the radial range
        min_dev = np.min(deviations)
        max_dev = np.max(deviations)
        circularity = max_dev - min_dev

        return GDTResult(
            gdt_type=GDTType.CIRCULARITY,
            measured_value=circularity,
            tolerance=tolerance,
            conformance=circularity <= tolerance,
            fit_parameters={
                "center": centroid.tolist(),
                "plane_normal": plane_normal.tolist(),
                "fitted_radius": float(mean_radius)
            },
            min_deviation=min_dev,
            max_deviation=max_dev,
            std_deviation=float(np.std(deviations)),
            points_used=len(points),
            confidence=self._calculate_confidence(len(points), 50)
        )

    def calculate_position(
        self,
        points: np.ndarray,
        nominal_position: Tuple[float, float, float],
        tolerance: float,
        mmc: bool = False,
        feature_size: Optional[float] = None,
        feature_mmc: Optional[float] = None
    ) -> GDTResult:
        """
        Calculate true position - the location of a feature relative to datum.

        ASME Y14.5: Position is the location of a feature relative to its
        true position established by basic dimensions.

        Position tolerance zone is typically cylindrical (diameter).

        Args:
            points: Nx3 array of feature points (e.g., hole edge points)
            nominal_position: (x, y, z) nominal/theoretical position
            tolerance: Position tolerance diameter in mm
            mmc: If True, apply MMC bonus tolerance
            feature_size: Actual feature size (for MMC)
            feature_mmc: Maximum Material Condition size (for MMC)

        Returns:
            GDTResult with measured position deviation
        """
        if len(points) < 3:
            return GDTResult(
                gdt_type=GDTType.POSITION,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Calculate actual center of feature
        actual_center = np.mean(points, axis=0)
        nominal = np.array(nominal_position)

        # Calculate 3D distance from nominal
        deviation_vector = actual_center - nominal
        deviation_distance = np.linalg.norm(deviation_vector)

        # Position value per ASME Y14.5-2018:
        # Tolerance is specified as diameter of cylindrical zone
        # measured_value is the diameter of the zone needed to contain the feature
        # (which is 2x the deviation distance from nominal)
        position_value = 2 * deviation_distance

        # Apply MMC bonus if applicable
        effective_tolerance = tolerance
        if mmc and feature_size is not None and feature_mmc is not None:
            bonus = abs(feature_size - feature_mmc)
            effective_tolerance = tolerance + bonus

        # Conformance check: measured diameter zone <= tolerance diameter
        return GDTResult(
            gdt_type=GDTType.POSITION,
            measured_value=position_value,
            tolerance=effective_tolerance,
            conformance=position_value <= effective_tolerance,
            fit_parameters={
                "deviation_x": float(deviation_vector[0]),
                "deviation_y": float(deviation_vector[1]),
                "deviation_z": float(deviation_vector[2]),
                "mmc_bonus": effective_tolerance - tolerance if mmc else 0
            },
            actual_position=tuple(actual_center.tolist()),
            nominal_position=nominal_position,
            min_deviation=0,
            max_deviation=deviation_distance,
            std_deviation=float(np.std(np.linalg.norm(points - actual_center, axis=1))),
            points_used=len(points),
            confidence=self._calculate_confidence(len(points), 20)
        )

    def calculate_parallelism(
        self,
        surface_points: np.ndarray,
        datum_normal: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate parallelism - distance between two parallel planes containing all surface points.

        ASME Y14.5: Parallelism is the condition of a surface, center plane,
        or axis that is equidistant at all points from a datum plane or axis.
        Measured as the range of signed perpendicular distances from datum plane.

        Args:
            surface_points: Nx3 array of surface points
            datum_normal: Normal vector of datum plane
            tolerance: Parallelism tolerance in mm

        Returns:
            GDTResult with measured parallelism value
        """
        if len(surface_points) < 3:
            return GDTResult(
                gdt_type=GDTType.PARALLELISM,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Normalize datum normal
        datum_normal = np.array(datum_normal, dtype=np.float64)
        datum_norm = np.linalg.norm(datum_normal)
        if datum_norm < 1e-10:
            return GDTResult(
                gdt_type=GDTType.PARALLELISM,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )
        datum_normal = datum_normal / datum_norm

        # Fit plane to surface to get surface normal (for angle reporting)
        centroid = np.mean(surface_points, axis=0)
        centered = surface_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        surface_normal = vh[-1]

        # Ensure consistent normal direction
        if np.dot(surface_normal, datum_normal) < 0:
            surface_normal = -surface_normal

        # Calculate angle between normals (for diagnostic purposes)
        dot_product = np.clip(np.dot(surface_normal, datum_normal), -1.0, 1.0)
        angle_rad = np.arccos(abs(dot_product))

        # ASME Y14.5 Parallelism: Range of signed perpendicular distances from datum plane
        # Project all points onto datum normal to get signed distances
        # Using centroid as reference point on datum plane
        signed_distances = np.dot(surface_points - centroid, datum_normal)
        min_dist = np.min(signed_distances)
        max_dist = np.max(signed_distances)

        # Parallelism = range of distances (zone width)
        parallelism = max_dist - min_dist

        return GDTResult(
            gdt_type=GDTType.PARALLELISM,
            measured_value=parallelism,
            tolerance=tolerance,
            conformance=parallelism <= tolerance,
            fit_parameters={
                "surface_normal": surface_normal.tolist(),
                "datum_normal": datum_normal.tolist(),
                "angle_degrees": float(np.degrees(angle_rad))
            },
            min_deviation=float(min_dist),
            max_deviation=float(max_dist),
            std_deviation=float(np.std(signed_distances)),
            points_used=len(surface_points),
            confidence=self._calculate_confidence(len(surface_points), 100)
        )

    def calculate_perpendicularity(
        self,
        surface_points: np.ndarray,
        datum_normal: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate perpendicularity - deviation from 90 degrees to datum.

        ASME Y14.5: Perpendicularity is the condition of a surface, center plane,
        or axis at 90 degrees to a datum plane or axis.

        Args:
            surface_points: Nx3 array of surface points
            datum_normal: Normal vector of datum plane
            tolerance: Perpendicularity tolerance in mm

        Returns:
            GDTResult with measured perpendicularity value
        """
        if len(surface_points) < 3:
            return GDTResult(
                gdt_type=GDTType.PERPENDICULARITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Fit plane to surface
        centroid = np.mean(surface_points, axis=0)
        centered = surface_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        surface_normal = vh[-1]

        # Normalize datum normal
        datum_normal = np.array(datum_normal, dtype=np.float64)
        datum_norm = np.linalg.norm(datum_normal)
        if datum_norm < 1e-10:
            return GDTResult(
                gdt_type=GDTType.PERPENDICULARITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )
        datum_normal = datum_normal / datum_norm

        # Angle diagnostics (surface normal vs datum normal)
        dot_product = np.clip(np.dot(surface_normal, datum_normal), -1.0, 1.0)
        actual_angle = np.arccos(abs(dot_product))
        angle_from_perp = abs(actual_angle - np.pi / 2)

        # Ideal zone normal for perpendicularity:
        # nearest direction to the measured surface normal that is perpendicular to datum.
        ideal_normal = surface_normal - np.dot(surface_normal, datum_normal) * datum_normal
        ideal_norm = np.linalg.norm(ideal_normal)
        if ideal_norm < 1e-10:
            # Surface normal is nearly parallel to datum; choose any stable orthogonal direction.
            ref = np.array([1.0, 0.0, 0.0]) if abs(datum_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ideal_normal = np.cross(datum_normal, ref)
            ideal_norm = np.linalg.norm(ideal_normal)
        ideal_normal = ideal_normal / ideal_norm

        # Perpendicularity is zone width along the ideal zone normal.
        signed_distances = np.dot(surface_points - centroid, ideal_normal)
        min_dist = np.min(signed_distances)
        max_dist = np.max(signed_distances)
        perpendicularity = max_dist - min_dist

        return GDTResult(
            gdt_type=GDTType.PERPENDICULARITY,
            measured_value=perpendicularity,
            tolerance=tolerance,
            conformance=perpendicularity <= tolerance,
            fit_parameters={
                "surface_normal": surface_normal.tolist(),
                "datum_normal": datum_normal.tolist(),
                "ideal_zone_normal": ideal_normal.tolist(),
                "angle_from_90_degrees": float(np.degrees(angle_from_perp)),
                "min_distance": float(min_dist),
                "max_distance": float(max_dist)
            },
            min_deviation=float(min_dist),
            max_deviation=float(max_dist),
            std_deviation=float(np.std(signed_distances)),
            points_used=len(surface_points),
            confidence=self._calculate_confidence(len(surface_points), 100)
        )

    def calculate_straightness(
        self,
        points: np.ndarray,
        tolerance: float,
        direction_hint: Optional[np.ndarray] = None
    ) -> GDTResult:
        """
        Calculate straightness - deviation of line elements from a straight line.

        ASME Y14.5: Straightness is a condition where an element of a surface
        or derived median line is a straight line.

        Args:
            points: Nx3 array of points along a line
            tolerance: Straightness tolerance in mm
            direction_hint: Optional hint for line direction

        Returns:
            GDTResult with measured straightness value
        """
        if len(points) < 2:
            return GDTResult(
                gdt_type=GDTType.STRAIGHTNESS,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Fit a line using SVD
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        line_direction = vh[0]  # Direction of best-fit line

        if direction_hint is not None:
            # Use provided direction if specified
            line_direction = direction_hint / np.linalg.norm(direction_hint)

        # Calculate perpendicular distances to the line
        projections = np.dot(centered, line_direction)[:, np.newaxis] * line_direction
        perpendicular = centered - projections
        distances = np.linalg.norm(perpendicular, axis=1)

        # Straightness is twice the max distance (diameter of tolerance zone)
        max_dist = np.max(distances)
        straightness = 2 * max_dist

        return GDTResult(
            gdt_type=GDTType.STRAIGHTNESS,
            measured_value=straightness,
            tolerance=tolerance,
            conformance=straightness <= tolerance,
            fit_parameters={
                "line_direction": line_direction.tolist(),
                "line_point": centroid.tolist()
            },
            min_deviation=0,
            max_deviation=max_dist,
            std_deviation=float(np.std(distances)),
            points_used=len(points),
            confidence=self._calculate_confidence(len(points), 20)
        )

    def calculate_angularity(
        self,
        surface_points: np.ndarray,
        datum_normal: np.ndarray,
        nominal_angle: float,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate angularity - deviation from specified angle to datum.

        ASME Y14.5: Angularity is the condition of a surface, center plane,
        or axis at a specified angle (other than 90°) to a datum plane or axis.

        Args:
            surface_points: Nx3 array of surface points
            datum_normal: Normal vector of datum plane
            nominal_angle: Specified angle in degrees
            tolerance: Angularity tolerance in mm

        Returns:
            GDTResult with measured angularity value
        """
        if len(surface_points) < 3:
            return GDTResult(
                gdt_type=GDTType.ANGULARITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Fit plane to surface
        centroid = np.mean(surface_points, axis=0)
        centered = surface_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        surface_normal = vh[-1]

        # Normalize datum normal
        datum_normal = np.array(datum_normal, dtype=np.float64)
        datum_norm = np.linalg.norm(datum_normal)
        if datum_norm < 1e-10:
            return GDTResult(
                gdt_type=GDTType.ANGULARITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )
        datum_normal = datum_normal / datum_norm

        # Angle diagnostics
        dot_product = np.clip(np.dot(surface_normal, datum_normal), -1.0, 1.0)
        actual_angle = np.degrees(np.arccos(abs(dot_product)))
        angle_error = abs(actual_angle - nominal_angle)

        # Build an ideal zone normal at nominal angle, preserving measured azimuth.
        nominal_angle_rad = np.radians(nominal_angle)
        lateral = surface_normal - np.dot(surface_normal, datum_normal) * datum_normal
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-10:
            ref = np.array([1.0, 0.0, 0.0]) if abs(datum_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            lateral = np.cross(datum_normal, ref)
            lateral_norm = np.linalg.norm(lateral)
        lateral = lateral / lateral_norm

        sign = 1.0 if np.dot(surface_normal, datum_normal) >= 0 else -1.0
        ideal_normal = (
            sign * np.cos(nominal_angle_rad) * datum_normal +
            np.sin(nominal_angle_rad) * lateral
        )
        ideal_normal = ideal_normal / np.linalg.norm(ideal_normal)

        # Angularity is zone width along the nominal orientation normal.
        signed_distances = np.dot(surface_points - centroid, ideal_normal)
        min_dist = np.min(signed_distances)
        max_dist = np.max(signed_distances)
        angularity = max_dist - min_dist

        return GDTResult(
            gdt_type=GDTType.ANGULARITY,
            measured_value=angularity,
            tolerance=tolerance,
            conformance=angularity <= tolerance,
            fit_parameters={
                "surface_normal": surface_normal.tolist(),
                "datum_normal": datum_normal.tolist(),
                "ideal_zone_normal": ideal_normal.tolist(),
                "nominal_angle_degrees": nominal_angle,
                "actual_angle_degrees": float(actual_angle),
                "angle_error_degrees": float(angle_error),
                "min_distance": float(min_dist),
                "max_distance": float(max_dist)
            },
            min_deviation=float(min_dist),
            max_deviation=float(max_dist),
            std_deviation=float(np.std(signed_distances)),
            points_used=len(surface_points),
            confidence=self._calculate_confidence(len(surface_points), 100)
        )

    def calculate_concentricity(
        self,
        feature_points: np.ndarray,
        datum_axis_point: Tuple[float, float, float],
        datum_axis_direction: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate concentricity - deviation of axis from datum axis.

        ASME Y14.5: Concentricity is the condition where the median points
        of all diametrically opposed elements of a figure of revolution
        are congruent with a datum axis.

        Args:
            feature_points: Nx3 array of feature surface points
            datum_axis_point: Point on datum axis
            datum_axis_direction: Direction of datum axis
            tolerance: Concentricity tolerance (diameter)

        Returns:
            GDTResult with measured concentricity value
        """
        if len(feature_points) < 6:
            return GDTResult(
                gdt_type=GDTType.CONCENTRICITY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        datum_point = np.array(datum_axis_point)
        datum_dir = np.array(datum_axis_direction) / np.linalg.norm(datum_axis_direction)

        # Calculate the actual axis of the feature by fitting a cylinder
        centroid = np.mean(feature_points, axis=0)
        centered = feature_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)

        # Find axis most parallel to datum
        best_axis = None
        best_alignment = 0
        for i in range(3):
            alignment = abs(np.dot(vh[i], datum_dir))
            if alignment > best_alignment:
                best_alignment = alignment
                best_axis = vh[i]

        if best_axis is None:
            best_axis = vh[0]

        # Ensure consistent direction
        if np.dot(best_axis, datum_dir) < 0:
            best_axis = -best_axis

        # Calculate perpendicular distance between axes
        # Project centroid onto datum axis
        to_centroid = centroid - datum_point
        proj_length = np.dot(to_centroid, datum_dir)
        closest_point_on_datum = datum_point + proj_length * datum_dir
        axis_offset = np.linalg.norm(centroid - closest_point_on_datum)

        # Concentricity is diameter of zone containing axis offset
        concentricity = 2 * axis_offset

        return GDTResult(
            gdt_type=GDTType.CONCENTRICITY,
            measured_value=concentricity,
            tolerance=tolerance,
            conformance=concentricity <= tolerance,
            fit_parameters={
                "feature_axis_direction": best_axis.tolist(),
                "feature_center": centroid.tolist(),
                "datum_axis_direction": datum_dir.tolist(),
                "axis_offset": float(axis_offset)
            },
            actual_position=tuple(centroid.tolist()),
            nominal_position=tuple(closest_point_on_datum.tolist()),
            min_deviation=0,
            max_deviation=axis_offset,
            std_deviation=0,
            points_used=len(feature_points),
            confidence=self._calculate_confidence(len(feature_points), 200)
        )

    def calculate_symmetry(
        self,
        feature_points: np.ndarray,
        datum_plane_point: Tuple[float, float, float],
        datum_plane_normal: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate symmetry - equal distribution about datum center plane.

        ASME Y14.5: Symmetry is a condition in which a feature
        is symmetrically disposed about the center plane of a datum feature.

        Args:
            feature_points: Nx3 array of feature points
            datum_plane_point: Point on datum plane
            datum_plane_normal: Normal of datum plane
            tolerance: Symmetry tolerance in mm

        Returns:
            GDTResult with measured symmetry value
        """
        if len(feature_points) < 4:
            return GDTResult(
                gdt_type=GDTType.SYMMETRY,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        datum_point = np.array(datum_plane_point)
        datum_normal = np.array(datum_plane_normal) / np.linalg.norm(datum_plane_normal)

        # Calculate signed distances to datum plane
        to_points = feature_points - datum_point
        signed_distances = np.dot(to_points, datum_normal)

        # Find center plane of feature
        feature_center_distance = np.mean(signed_distances)

        # Symmetry is twice the offset of feature center from datum
        symmetry = 2 * abs(feature_center_distance)

        # Calculate distribution statistics
        positive_points = signed_distances[signed_distances > 0]
        negative_points = signed_distances[signed_distances < 0]

        return GDTResult(
            gdt_type=GDTType.SYMMETRY,
            measured_value=symmetry,
            tolerance=tolerance,
            conformance=symmetry <= tolerance,
            fit_parameters={
                "datum_plane_normal": datum_normal.tolist(),
                "feature_center_offset": float(feature_center_distance),
                "points_positive_side": len(positive_points),
                "points_negative_side": len(negative_points)
            },
            min_deviation=0,
            max_deviation=abs(feature_center_distance),
            std_deviation=float(np.std(signed_distances)),
            points_used=len(feature_points),
            confidence=self._calculate_confidence(len(feature_points), 50)
        )

    def calculate_circular_runout(
        self,
        feature_points: np.ndarray,
        datum_axis_point: Tuple[float, float, float],
        datum_axis_direction: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate circular runout - FIM at single cross-section.

        ASME Y14.5: Circular runout is the composite deviation from the
        desired form of a part surface of revolution during full rotation
        about the datum axis.

        Args:
            feature_points: Nx3 array of surface points
            datum_axis_point: Point on datum axis
            datum_axis_direction: Direction of datum axis
            tolerance: Runout tolerance (FIM)

        Returns:
            GDTResult with measured runout value
        """
        if len(feature_points) < 10:
            return GDTResult(
                gdt_type=GDTType.CIRCULAR_RUNOUT,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        datum_point = np.array(datum_axis_point)
        datum_dir = np.array(datum_axis_direction) / np.linalg.norm(datum_axis_direction)

        # Calculate radial distances from datum axis
        to_points = feature_points - datum_point
        axial_projections = np.dot(to_points, datum_dir)[:, np.newaxis] * datum_dir
        radial_vectors = to_points - axial_projections
        radii = np.linalg.norm(radial_vectors, axis=1)

        # Group by axial position and find max runout at any cross-section
        # For simplicity, calculate overall runout (FIM = max - min)
        runout = np.max(radii) - np.min(radii)

        return GDTResult(
            gdt_type=GDTType.CIRCULAR_RUNOUT,
            measured_value=runout,
            tolerance=tolerance,
            conformance=runout <= tolerance,
            fit_parameters={
                "datum_axis_direction": datum_dir.tolist(),
                "datum_axis_point": datum_point.tolist(),
                "max_radius": float(np.max(radii)),
                "min_radius": float(np.min(radii)),
                "mean_radius": float(np.mean(radii))
            },
            min_deviation=float(np.min(radii) - np.mean(radii)),
            max_deviation=float(np.max(radii) - np.mean(radii)),
            std_deviation=float(np.std(radii)),
            points_used=len(feature_points),
            confidence=self._calculate_confidence(len(feature_points), 100)
        )

    def calculate_total_runout(
        self,
        feature_points: np.ndarray,
        datum_axis_point: Tuple[float, float, float],
        datum_axis_direction: np.ndarray,
        tolerance: float
    ) -> GDTResult:
        """
        Calculate total runout - FIM across entire surface.

        ASME Y14.5: Total runout is a composite control of all surface
        elements across the entire surface. It controls cumulative
        variations of circularity, straightness, coaxiality, angularity,
        taper, and profile.

        Args:
            feature_points: Nx3 array of surface points
            datum_axis_point: Point on datum axis
            datum_axis_direction: Direction of datum axis
            tolerance: Total runout tolerance (FIM)

        Returns:
            GDTResult with measured total runout value
        """
        if len(feature_points) < 20:
            return GDTResult(
                gdt_type=GDTType.TOTAL_RUNOUT,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        datum_point = np.array(datum_axis_point)
        datum_dir = np.array(datum_axis_direction) / np.linalg.norm(datum_axis_direction)

        # Calculate radial distances from datum axis
        to_points = feature_points - datum_point
        axial_projections = np.dot(to_points, datum_dir)[:, np.newaxis] * datum_dir
        radial_vectors = to_points - axial_projections
        radii = np.linalg.norm(radial_vectors, axis=1)
        axial_positions = np.dot(to_points, datum_dir)

        # For cylindrical surfaces, total runout considers variation across length
        # Fit a line to radius vs axial position (accounts for taper)
        if len(set(axial_positions)) > 1:
            coeffs = np.polyfit(axial_positions, radii, 1)
            fitted_radii = np.polyval(coeffs, axial_positions)
            deviations = radii - fitted_radii
            total_runout = np.max(deviations) - np.min(deviations)
            taper = float(coeffs[0])
        else:
            total_runout = np.max(radii) - np.min(radii)
            taper = 0.0

        return GDTResult(
            gdt_type=GDTType.TOTAL_RUNOUT,
            measured_value=total_runout,
            tolerance=tolerance,
            conformance=total_runout <= tolerance,
            fit_parameters={
                "datum_axis_direction": datum_dir.tolist(),
                "datum_axis_point": datum_point.tolist(),
                "taper_per_unit": taper,
                "axial_range": float(np.max(axial_positions) - np.min(axial_positions)),
                "mean_radius": float(np.mean(radii))
            },
            min_deviation=float(np.min(radii) - np.mean(radii)),
            max_deviation=float(np.max(radii) - np.mean(radii)),
            std_deviation=float(np.std(radii)),
            points_used=len(feature_points),
            confidence=self._calculate_confidence(len(feature_points), 200)
        )

    def calculate_profile_line(
        self,
        actual_points: np.ndarray,
        nominal_points: np.ndarray,
        tolerance: float,
        bilateral: bool = True
    ) -> GDTResult:
        """
        Calculate profile of a line - 2D profile deviation.

        ASME Y14.5: Profile of a line establishes a tolerance zone
        bounded by two uniform lines along an element of a surface.

        Args:
            actual_points: Nx3 array of actual measured points
            nominal_points: Nx3 array of corresponding nominal points
            tolerance: Profile tolerance (total zone width)
            bilateral: If True, tolerance is split equally (+/-)

        Returns:
            GDTResult with measured profile deviation
        """
        if len(actual_points) < 3 or len(nominal_points) < 3:
            return GDTResult(
                gdt_type=GDTType.PROFILE_LINE,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # Calculate point-to-point deviations
        if len(actual_points) == len(nominal_points):
            deviations = np.linalg.norm(actual_points - nominal_points, axis=1)
        else:
            # Find closest nominal point for each actual point
            deviations = []
            for actual in actual_points:
                distances = np.linalg.norm(nominal_points - actual, axis=1)
                deviations.append(np.min(distances))
            deviations = np.array(deviations)

        max_deviation = np.max(deviations)

        # Profile tolerance per ASME Y14.5:
        # - tolerance specifies total zone width (e.g., 0.5mm = ±0.25mm for bilateral)
        # - For bilateral: max deviation must be <= tolerance/2
        # - For unilateral: max deviation must be <= tolerance
        # We report max_deviation as measured_value (actual deviation from nominal)
        profile_value = max_deviation
        if bilateral:
            conformance = max_deviation <= (tolerance / 2)
        else:
            conformance = max_deviation <= tolerance

        return GDTResult(
            gdt_type=GDTType.PROFILE_LINE,
            measured_value=profile_value,
            tolerance=tolerance if not bilateral else tolerance / 2,
            conformance=conformance,
            fit_parameters={
                "bilateral": bilateral,
                "max_point_deviation": float(max_deviation),
                "mean_deviation": float(np.mean(deviations))
            },
            min_deviation=float(np.min(deviations)),
            max_deviation=float(max_deviation),
            std_deviation=float(np.std(deviations)),
            points_used=len(actual_points),
            confidence=self._calculate_confidence(len(actual_points), 30)
        )

    def calculate_profile_surface(
        self,
        actual_points: np.ndarray,
        nominal_points: np.ndarray,
        tolerance: float,
        bilateral: bool = True
    ) -> GDTResult:
        """
        Calculate profile of a surface - 3D profile deviation.

        ASME Y14.5: Profile of a surface establishes a tolerance zone
        bounded by two surfaces offset from the nominal surface.

        Args:
            actual_points: Nx3 array of actual measured points
            nominal_points: Mx3 array of nominal surface points
            tolerance: Profile tolerance (total zone width)
            bilateral: If True, tolerance is split equally (+/-)

        Returns:
            GDTResult with measured profile deviation
        """
        if len(actual_points) < 4 or len(nominal_points) < 4:
            return GDTResult(
                gdt_type=GDTType.PROFILE_SURFACE,
                measured_value=float('inf'),
                tolerance=tolerance,
                conformance=False,
                confidence=0.0
            )

        # For each actual point, find distance to nearest nominal point
        # More sophisticated approaches would use signed distance to nominal surface
        deviations = []
        for actual in actual_points:
            distances = np.linalg.norm(nominal_points - actual, axis=1)
            deviations.append(np.min(distances))
        deviations = np.array(deviations)

        max_deviation = np.max(deviations)

        # Profile tolerance per ASME Y14.5:
        # - tolerance specifies total zone width (e.g., 0.5mm = ±0.25mm for bilateral)
        # - For bilateral: max deviation must be <= tolerance/2
        # - For unilateral: max deviation must be <= tolerance
        # We report max_deviation as measured_value (actual deviation from nominal)
        profile_value = max_deviation
        effective_tolerance = tolerance / 2 if bilateral else tolerance
        if bilateral:
            conformance = max_deviation <= (tolerance / 2)
        else:
            conformance = max_deviation <= tolerance

        return GDTResult(
            gdt_type=GDTType.PROFILE_SURFACE,
            measured_value=profile_value,
            tolerance=effective_tolerance,
            conformance=conformance,
            fit_parameters={
                "bilateral": bilateral,
                "total_zone_width": float(tolerance),
                "max_point_deviation": float(max_deviation),
                "mean_deviation": float(np.mean(deviations)),
                "coverage_percentage": float(100 * np.sum(deviations <= effective_tolerance) / len(deviations))
            },
            min_deviation=float(np.min(deviations)),
            max_deviation=float(max_deviation),
            std_deviation=float(np.std(deviations)),
            points_used=len(actual_points),
            confidence=self._calculate_confidence(len(actual_points), 100)
        )

    def _calculate_confidence(self, points_used: int, min_points: int) -> float:
        """Calculate confidence based on number of points used"""
        if points_used < min_points / 2:
            return 0.5
        elif points_used < min_points:
            return 0.5 + 0.5 * (points_used / min_points)
        else:
            return min(1.0, 0.9 + 0.1 * (points_used / (min_points * 5)))

    def analyze_feature(
        self,
        points: np.ndarray,
        gdt_type: GDTType,
        tolerance: float,
        **kwargs
    ) -> GDTResult:
        """
        Generic feature analysis dispatcher.

        Args:
            points: Nx3 array of feature points
            gdt_type: Type of GD&T to calculate
            tolerance: Tolerance value in mm
            **kwargs: Additional parameters for specific calculations

        Returns:
            GDTResult
        """
        if gdt_type == GDTType.FLATNESS:
            return self.calculate_flatness(points, tolerance)
        elif gdt_type == GDTType.CYLINDRICITY:
            return self.calculate_cylindricity(points, tolerance, kwargs.get("axis_hint"))
        elif gdt_type == GDTType.CIRCULARITY:
            return self.calculate_circularity(points, tolerance, kwargs.get("plane_normal"))
        elif gdt_type == GDTType.POSITION:
            return self.calculate_position(
                points,
                kwargs.get("nominal_position", (0, 0, 0)),
                tolerance,
                kwargs.get("mmc", False),
                kwargs.get("feature_size"),
                kwargs.get("feature_mmc")
            )
        elif gdt_type == GDTType.PARALLELISM:
            return self.calculate_parallelism(
                points,
                kwargs.get("datum_normal", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.PERPENDICULARITY:
            return self.calculate_perpendicularity(
                points,
                kwargs.get("datum_normal", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.STRAIGHTNESS:
            return self.calculate_straightness(
                points,
                tolerance,
                kwargs.get("direction_hint")
            )
        elif gdt_type == GDTType.ANGULARITY:
            return self.calculate_angularity(
                points,
                kwargs.get("datum_normal", [0, 0, 1]),
                kwargs.get("nominal_angle", 45.0),
                tolerance
            )
        elif gdt_type == GDTType.CONCENTRICITY:
            return self.calculate_concentricity(
                points,
                kwargs.get("datum_axis_point", (0, 0, 0)),
                kwargs.get("datum_axis_direction", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.SYMMETRY:
            return self.calculate_symmetry(
                points,
                kwargs.get("datum_plane_point", (0, 0, 0)),
                kwargs.get("datum_plane_normal", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.CIRCULAR_RUNOUT:
            return self.calculate_circular_runout(
                points,
                kwargs.get("datum_axis_point", (0, 0, 0)),
                kwargs.get("datum_axis_direction", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.TOTAL_RUNOUT:
            return self.calculate_total_runout(
                points,
                kwargs.get("datum_axis_point", (0, 0, 0)),
                kwargs.get("datum_axis_direction", [0, 0, 1]),
                tolerance
            )
        elif gdt_type == GDTType.PROFILE_LINE:
            return self.calculate_profile_line(
                points,
                kwargs.get("nominal_points", points),
                tolerance,
                kwargs.get("bilateral", True)
            )
        elif gdt_type == GDTType.PROFILE_SURFACE:
            return self.calculate_profile_surface(
                points,
                kwargs.get("nominal_points", points),
                tolerance,
                kwargs.get("bilateral", True)
            )
        else:
            raise ValueError(f"Unsupported GD&T type: {gdt_type}")


# Convenience function
def create_gdt_engine() -> GDTEngine:
    """Create a GD&T engine instance"""
    return GDTEngine()
