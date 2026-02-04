"""
Scan Feature Measurer Module

Measures matched features in scan point cloud.
Supports measurement of:
- Holes/diameters via RANSAC circle fitting
- Linear dimensions via surface projection
- Angles via signed distance gradient or surface classification
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

from .feature_types import (
    FeatureType,
    MeasurableFeature,
    HoleFeature,
    LinearFeature,
    AngleFeature,
    FeatureMatch,
)

try:
    from scipy.spatial import KDTree
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class ScanFeatureMeasurer:
    """
    Measure matched features in scan point cloud.

    Methods:
    - Holes: RANSAC circle fit on extracted edge points
    - Linear: Surface projection and distance measurement
    - Angles: Signed distance gradient method (preferred) or surface classification
    """

    def __init__(
        self,
        search_radius: float = 10.0,
        min_points_hole: int = 20,
        min_points_surface: int = 30,
        ransac_iterations: int = 100,
        ransac_threshold: float = 0.1,
    ):
        """
        Initialize measurer.

        Args:
            search_radius: Default search radius for finding nearby points (mm)
            min_points_hole: Minimum points needed for hole measurement
            min_points_surface: Minimum points needed for surface measurement
            ransac_iterations: Number of RANSAC iterations for circle fitting
            ransac_threshold: RANSAC inlier threshold (mm)
        """
        self.search_radius = search_radius
        self.min_points_hole = min_points_hole
        self.min_points_surface = min_points_surface
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold

    def measure_all(
        self,
        scan_points: np.ndarray,
        matches: List[FeatureMatch],
        deviations: Optional[np.ndarray] = None,
    ) -> List[FeatureMatch]:
        """
        Measure all matched features in the scan.

        Args:
            scan_points: Nx3 array of scan point coordinates
            matches: List of FeatureMatch objects to measure
            deviations: Optional signed deviations for gradient-based angle measurement

        Returns:
            Updated list of FeatureMatch with scan_value populated
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available for feature measurement")
            return matches

        if len(scan_points) < 100:
            logger.warning("Insufficient scan points for measurement")
            return matches

        # Build KDTree once for efficiency
        tree = KDTree(scan_points)

        for match in matches:
            try:
                cad_feature = match.cad_feature

                if match.feature_type == FeatureType.DIAMETER.value:
                    # Measure hole diameter
                    measured = self.measure_hole(scan_points, cad_feature, tree)
                elif match.feature_type == FeatureType.LINEAR.value:
                    # Measure linear dimension - pass axis if available
                    axis = getattr(match, 'axis', None)
                    measured = self.measure_linear(scan_points, cad_feature, tree, axis=axis)
                elif match.feature_type == FeatureType.ANGLE.value:
                    # Measure angle
                    measured = self.measure_angle(
                        scan_points, cad_feature, tree, deviations
                    )
                elif match.feature_type == FeatureType.RADIUS.value:
                    # Measure radius (similar to diameter)
                    measured = self.measure_radius(scan_points, cad_feature, tree)
                else:
                    measured = None

                if measured is not None:
                    match.scan_value = measured
                    match.measurement_confidence = 0.8  # Default confidence

                # Compute status
                match.compute_status()

            except Exception as e:
                logger.warning(f"Failed to measure {match.feature_type} #{match.dim_id}: {e}")
                match.status = "not_measured"

        return matches

    def _compute_radius_tolerance(self, expected_radius: float) -> float:
        """
        Compute adaptive radius tolerance based on hole size.

        Tighter tolerance for small holes to avoid capturing surrounding surface points.
        """
        if expected_radius < 1.0:
            return expected_radius * 0.3  # 30% for tiny holes (<2mm diameter)
        elif expected_radius < 5.0:
            return expected_radius * 0.4  # 40% for small holes (<10mm diameter)
        else:
            return expected_radius * 0.5  # 50% for large holes

    def _compute_ransac_threshold(self, expected_radius: float) -> float:
        """
        Compute adaptive RANSAC inlier threshold based on hole size.

        Smaller threshold for small holes to get tighter circle fits.
        """
        return min(0.1, expected_radius * 0.05)

    def measure_hole(
        self,
        scan_points: np.ndarray,
        hole: HoleFeature,
        tree: Optional[KDTree] = None,
    ) -> Optional[float]:
        """
        Measure hole diameter in point cloud with center fitting.

        Algorithm:
        1. Extract points near approximate hole position using KDTree
        2. Project to plane perpendicular to axis (thin slice)
        3. Filter to points near expected radius (hole edge region)
        4. RANSAC circle fit to find BOTH center and radius from scan data
        5. Return diameter = 2 * radius

        Key improvements:
        - Fits both center and radius from scan data (compensates for CAD center offset)
        - Use tight plane tolerance (2.0mm) to isolate hole edge
        - Filter to expected radius range BEFORE RANSAC to avoid surface contamination
        - Adaptive radius tolerance and RANSAC threshold based on hole size
        """
        if tree is None:
            tree = KDTree(scan_points)

        # CAD center is approximate - we'll fit the actual center from scan data
        approx_center = np.asarray(hole.position)
        axis = np.asarray(hole.axis)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        expected_radius = hole.nominal_value / 2

        # Search radius - use larger area to capture entire hole even if center is offset
        search_r = max(self.search_radius, expected_radius * 2.5)

        # Find points near approximate hole center
        nearby_idx = tree.query_ball_point(approx_center, search_r)
        if len(nearby_idx) < self.min_points_hole:
            search_r *= 2
            nearby_idx = tree.query_ball_point(approx_center, search_r)
            if len(nearby_idx) < self.min_points_hole:
                return None

        nearby_points = scan_points[nearby_idx]
        relative = nearby_points - approx_center

        # Compute distance along hole axis and perpendicular to it
        along_axis = np.dot(relative, axis)
        perp = relative - np.outer(along_axis, axis)
        radial_dist = np.linalg.norm(perp, axis=1)

        # Plane tolerance to isolate the hole edge plane
        # Use 2.0mm or adaptive based on hole size
        plane_threshold = min(2.0, max(1.0, expected_radius * 0.2))
        plane_mask = np.abs(along_axis) < plane_threshold

        if plane_mask.sum() < self.min_points_hole:
            # Try larger plane tolerance
            plane_threshold = min(4.0, expected_radius * 0.4)
            plane_mask = np.abs(along_axis) < plane_threshold

        if plane_mask.sum() < self.min_points_hole:
            return None

        # Get perpendicular vectors and radial distances at the plane
        perp_at_plane = perp[plane_mask]
        radial_at_plane = radial_dist[plane_mask]

        # Filter to points near expected radius (hole edge)
        # Use slightly wider tolerance since CAD center may be offset
        radius_tolerance = self._compute_radius_tolerance(expected_radius)
        # Increase tolerance to account for potential center offset
        radius_tolerance = max(radius_tolerance, expected_radius * 0.2)

        edge_mask = (
            (radial_at_plane > expected_radius - radius_tolerance) &
            (radial_at_plane < expected_radius + radius_tolerance)
        )

        min_edge_points = 30  # Need more points for reliable center+radius fitting
        if edge_mask.sum() < min_edge_points:
            # Try wider tolerance
            for scale in [1.5, 2.0, 2.5]:
                wider_tol = radius_tolerance * scale
                edge_mask = (
                    (radial_at_plane > expected_radius - wider_tol) &
                    (radial_at_plane < expected_radius + wider_tol)
                )
                if edge_mask.sum() >= min_edge_points:
                    break

        if edge_mask.sum() < min_edge_points:
            logger.warning(
                f"Insufficient edge points ({edge_mask.sum()}) for hole measurement, "
                f"expected radius={expected_radius:.2f}mm"
            )
            return None

        edge_points_3d = perp_at_plane[edge_mask]

        # Project to 2D for circle fitting
        # Create local coordinate system perpendicular to axis
        arbitrary = np.array([0, 1, 0]) if abs(axis[1]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(axis, arbitrary)
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(axis, u)

        edge_points_2d = np.column_stack([
            np.dot(edge_points_3d, u),
            np.dot(edge_points_3d, v)
        ])

        # RANSAC circle fit - fits BOTH center and radius from scan data
        # This compensates for any offset between CAD center and actual hole center
        adaptive_threshold = self._compute_ransac_threshold(expected_radius)
        original_threshold = self.ransac_threshold
        # Use slightly larger threshold for center fitting robustness
        self.ransac_threshold = max(adaptive_threshold, 0.15)
        # Increase iterations for better center estimation
        original_iterations = self.ransac_iterations
        self.ransac_iterations = max(self.ransac_iterations, 500)

        circle_result = self._ransac_circle_fit(edge_points_2d)

        self.ransac_threshold = original_threshold
        self.ransac_iterations = original_iterations

        if circle_result is None:
            return None

        fitted_center, radius, inlier_ratio = circle_result

        # Sanity check: fitted radius should be reasonably close to expected
        if abs(radius - expected_radius) > expected_radius * 0.5:
            logger.warning(
                f"Fitted radius {radius:.2f}mm differs significantly from "
                f"expected {expected_radius:.2f}mm"
            )
            # Still return it - let the caller decide if it's acceptable

        diameter = 2 * radius
        return diameter

    def measure_linear(
        self,
        scan_points: np.ndarray,
        linear: LinearFeature,
        tree: Optional[KDTree] = None,
        axis: Optional[str] = None,
    ) -> Optional[float]:
        """
        Measure linear dimension in point cloud.

        If axis (X, Y, or Z) is specified, uses axis-aligned measurement
        which is more reliable. Otherwise falls back to surface-based method.

        Args:
            scan_points: Point cloud data
            linear: LinearFeature with measurement info
            tree: Optional KDTree for efficient neighbor search
            axis: Optional measurement axis ('X', 'Y', or 'Z')

        Returns:
            Measured dimension in mm, or None if measurement failed
        """
        if tree is None:
            tree = KDTree(scan_points)

        # If axis is specified, use axis-aligned measurement
        if axis and axis.upper() in ['X', 'Y', 'Z']:
            return self._measure_linear_axis_aligned(
                scan_points, linear, tree, axis.upper()
            )

        # Otherwise use surface-based measurement
        return self._measure_linear_surface_based(scan_points, linear, tree)

    def _measure_linear_axis_aligned(
        self,
        scan_points: np.ndarray,
        linear: LinearFeature,
        tree: KDTree,
        axis: str,
    ) -> Optional[float]:
        """
        Measure linear dimension using axis-aligned approach.

        Uses CAD reference points to identify regions, then measures
        extent along the specified axis.
        """
        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        expected_value = linear.nominal_value

        point1 = np.asarray(linear.point1)
        point2 = np.asarray(linear.point2)
        center = (point1 + point2) / 2

        # Search for points near the measurement region
        search_r = max(expected_value * 0.5, 10.0)
        nearby_idx = tree.query_ball_point(center, search_r)

        if len(nearby_idx) < 50:
            search_r *= 2
            nearby_idx = tree.query_ball_point(center, search_r)

        if len(nearby_idx) < 50:
            return None

        nearby_pts = scan_points[nearby_idx]

        # For this measurement, we find the extent along the axis
        # in the region around the CAD reference points

        # Get the expected min/max along axis from CAD
        cad_min = min(point1[axis_idx], point2[axis_idx])
        cad_max = max(point1[axis_idx], point2[axis_idx])

        # Find points near the expected boundaries
        boundary_tol = expected_value * 0.15  # 15% tolerance for finding boundaries

        # Points near the min boundary
        near_min_mask = np.abs(nearby_pts[:, axis_idx] - cad_min) < boundary_tol
        # Points near the max boundary
        near_max_mask = np.abs(nearby_pts[:, axis_idx] - cad_max) < boundary_tol

        if near_min_mask.sum() < 10 or near_max_mask.sum() < 10:
            # Fall back to overall extent measurement
            axis_vals = nearby_pts[:, axis_idx]
            p5 = np.percentile(axis_vals, 5)
            p95 = np.percentile(axis_vals, 95)
            return p95 - p5

        # Use robust statistics on boundary regions
        min_pts = nearby_pts[near_min_mask, axis_idx]
        max_pts = nearby_pts[near_max_mask, axis_idx]

        measured_min = np.median(min_pts)
        measured_max = np.median(max_pts)

        return abs(measured_max - measured_min)

    def _measure_linear_surface_based(
        self,
        scan_points: np.ndarray,
        linear: LinearFeature,
        tree: KDTree,
    ) -> Optional[float]:
        """
        Measure linear dimension using surface-based approach.

        For surface_to_surface measurements:
        1. Find points near surface1 position
        2. Find points near surface2 position
        3. Project onto measurement direction
        4. Return distance between median positions
        """
        point1 = linear.point1
        point2 = linear.point2
        direction = linear.direction

        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Adaptive search radius based on dimension size
        dim_length = np.linalg.norm(point2 - point1)
        search_r = max(self.search_radius, dim_length * 0.15)

        # Find points near each endpoint
        idx1 = tree.query_ball_point(point1, search_r)
        idx2 = tree.query_ball_point(point2, search_r)

        if len(idx1) < self.min_points_surface or len(idx2) < self.min_points_surface:
            # Try larger radius
            search_r *= 2
            idx1 = tree.query_ball_point(point1, search_r)
            idx2 = tree.query_ball_point(point2, search_r)

        if len(idx1) < 5 or len(idx2) < 5:
            return None

        pts1 = scan_points[idx1]
        pts2 = scan_points[idx2]

        # If surface normals available, filter by normal direction
        if linear.surface1_normal is not None:
            pts1 = self._filter_by_normal(
                pts1, scan_points, idx1, linear.surface1_normal, tree
            )
        if linear.surface2_normal is not None:
            pts2 = self._filter_by_normal(
                pts2, scan_points, idx2, linear.surface2_normal, tree
            )

        if len(pts1) < 5 or len(pts2) < 5:
            return None

        # Project onto measurement direction
        proj1 = np.dot(pts1, direction)
        proj2 = np.dot(pts2, direction)

        # Use median for robustness
        pos1 = np.median(proj1)
        pos2 = np.median(proj2)

        measured_distance = abs(pos2 - pos1)
        return measured_distance

    def measure_angle(
        self,
        scan_points: np.ndarray,
        angle: AngleFeature,
        tree: Optional[KDTree] = None,
        signed_deviations: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Measure angle between two surfaces in point cloud.

        Primary method: Signed distance gradient
        Fallback: Surface classification
        """
        if tree is None:
            tree = KDTree(scan_points)

        apex = angle.apex_point
        search_r = max(self.search_radius * 3, 30.0)

        # Try gradient method first if deviations available
        if signed_deviations is not None:
            gradient_angle = self._measure_angle_gradient(
                scan_points, angle, tree, signed_deviations, search_r
            )
            if gradient_angle is not None:
                return gradient_angle

        # Fallback: surface classification method
        return self._measure_angle_surface(scan_points, angle, tree, search_r)

    def _measure_angle_gradient(
        self,
        scan_points: np.ndarray,
        angle: AngleFeature,
        tree: KDTree,
        deviations: np.ndarray,
        search_r: float,
    ) -> Optional[float]:
        """
        Measure angle using signed distance gradient method.

        This method is more accurate as it uses the deviation field
        to detect the bend transition.
        """
        apex = angle.apex_point
        bend_line = angle.bend_line if angle.bend_line is not None else angle.direction

        # Normalize bend line direction
        bend_line = bend_line / (np.linalg.norm(bend_line) + 1e-10)

        # Find points near bend apex
        nearby_idx = tree.query_ball_point(apex, search_r)
        if len(nearby_idx) < 100:
            return None

        nearby_points = scan_points[nearby_idx]
        nearby_deviations = deviations[nearby_idx]

        # Compute perpendicular direction to bend line
        # (direction along which surfaces meet)
        perp_dir = np.cross(bend_line, [0, 0, 1])
        if np.linalg.norm(perp_dir) < 0.1:
            perp_dir = np.cross(bend_line, [1, 0, 0])
        perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-10)

        # Project points onto perpendicular direction
        relative_pos = nearby_points - apex
        proj_along_perp = np.dot(relative_pos, perp_dir)

        # Split into two sides
        side1_mask = proj_along_perp < -1.0
        side2_mask = proj_along_perp > 1.0

        if side1_mask.sum() < 20 or side2_mask.sum() < 20:
            return None

        # Analyze deviation gradient on each side
        side1_proj = proj_along_perp[side1_mask]
        side1_dev = nearby_deviations[side1_mask]
        side2_proj = proj_along_perp[side2_mask]
        side2_dev = nearby_deviations[side2_mask]

        # Fit linear model to each side: deviation = slope * distance + intercept
        try:
            slope1, _ = np.polyfit(side1_proj, side1_dev, 1)
            slope2, _ = np.polyfit(side2_proj, side2_dev, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Convert slopes to angles (slope = tan(angle deviation from flat))
        angle1 = np.degrees(np.arctan(slope1))
        angle2 = np.degrees(np.arctan(slope2))

        # The bend angle is related to the slope difference
        # This is an approximation; exact relationship depends on geometry
        measured_angle = 180 - abs(angle2 - angle1)

        # Sanity check
        if measured_angle < 0 or measured_angle > 180:
            return None

        return measured_angle

    def _measure_angle_surface(
        self,
        scan_points: np.ndarray,
        angle: AngleFeature,
        tree: KDTree,
        search_r: float,
    ) -> Optional[float]:
        """
        Measure angle using surface classification method.

        Classifies points into two surfaces based on expected normals,
        then computes angle between fitted planes.
        """
        apex = angle.apex_point
        n1 = angle.surface1_normal
        n2 = angle.surface2_normal

        # Find nearby points
        nearby_idx = tree.query_ball_point(apex, search_r)
        if len(nearby_idx) < 100:
            return None

        nearby_points = scan_points[nearby_idx]

        # Estimate normals for nearby points
        estimated_normals = self._estimate_local_normals(
            scan_points, nearby_idx, tree
        )

        if estimated_normals is None:
            return None

        # Classify points by which surface normal they're closer to
        dot1 = np.abs(np.dot(estimated_normals, n1))
        dot2 = np.abs(np.dot(estimated_normals, n2))

        surface1_mask = dot1 > dot2
        surface2_mask = ~surface1_mask

        if surface1_mask.sum() < 20 or surface2_mask.sum() < 20:
            return None

        # Fit planes to each surface
        plane1 = self._fit_plane(nearby_points[surface1_mask])
        plane2 = self._fit_plane(nearby_points[surface2_mask])

        if plane1 is None or plane2 is None:
            return None

        # Compute angle between planes
        fitted_n1, _ = plane1
        fitted_n2, _ = plane2

        cos_angle = np.clip(np.dot(fitted_n1, fitted_n2), -1, 1)
        angle_between = np.degrees(np.arccos(abs(cos_angle)))

        # Bend angle is supplement
        measured_angle = 180 - angle_between

        return measured_angle

    def measure_radius(
        self,
        scan_points: np.ndarray,
        feature: MeasurableFeature,
        tree: Optional[KDTree] = None,
    ) -> Optional[float]:
        """
        Measure radius feature (fillet/round).

        Similar to hole measurement but for convex curves.
        """
        # Use similar approach to hole measurement
        if isinstance(feature, HoleFeature):
            diameter = self.measure_hole(scan_points, feature, tree)
            if diameter is not None:
                return diameter / 2
        return None

    def _project_to_plane(
        self,
        points: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project points onto plane defined by center and normal.

        Returns:
            Tuple of (projected_2d_points, signed_distances_from_plane)
        """
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # Compute signed distance from plane
        relative = points - center
        plane_distances = np.dot(relative, normal)

        # Project to plane (remove normal component)
        projected_3d = relative - np.outer(plane_distances, normal)

        # Create local 2D coordinate system
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(normal, arbitrary)
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(normal, u)

        # Project to 2D
        proj_u = np.dot(projected_3d, u)
        proj_v = np.dot(projected_3d, v)
        projected_2d = np.column_stack([proj_u, proj_v])

        return projected_2d, plane_distances

    def _ransac_circle_fit(
        self,
        points_2d: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        RANSAC circle fitting for 2D points.

        Returns:
            Tuple of (center, radius, inlier_ratio) or None
        """
        if len(points_2d) < 3:
            return None

        best_center = None
        best_radius = None
        best_inliers = 0

        n_points = len(points_2d)

        for _ in range(self.ransac_iterations):
            # Random sample of 3 points
            indices = np.random.choice(n_points, 3, replace=False)
            sample = points_2d[indices]

            # Fit circle through 3 points
            circle = self._fit_circle_3points(sample)
            if circle is None:
                continue

            center, radius = circle

            # Count inliers
            distances = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
            inliers = np.sum(distances < self.ransac_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_center = center
                best_radius = radius

        if best_center is None:
            return None

        inlier_ratio = best_inliers / n_points

        # Refine with all inliers
        distances = np.abs(np.linalg.norm(points_2d - best_center, axis=1) - best_radius)
        inlier_mask = distances < self.ransac_threshold * 2
        inlier_points = points_2d[inlier_mask]

        if len(inlier_points) >= 3:
            refined = self._taubin_circle_fit(inlier_points)
            if refined is not None:
                best_center, best_radius, _ = refined

        return best_center, best_radius, inlier_ratio

    def _fit_circle_3points(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Fit circle through exactly 3 points."""
        if len(points) != 3:
            return None

        ax, ay = points[0]
        bx, by = points[1]
        cx, cy = points[2]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

        center = np.array([ux, uy])
        radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)

        return center, radius

    def _taubin_circle_fit(
        self,
        points_2d: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """Taubin algebraic circle fit (same as hole_detector)."""
        if len(points_2d) < 3:
            return None

        x = points_2d[:, 0]
        y = points_2d[:, 1]

        x_mean = x.mean()
        y_mean = y.mean()

        u = x - x_mean
        v = y - y_mean

        Suv = np.sum(u * v)
        Suu = np.sum(u * u)
        Svv = np.sum(v * v)
        Suuv = np.sum(u * u * v)
        Suvv = np.sum(u * v * v)
        Suuu = np.sum(u * u * u)
        Svvv = np.sum(v * v * v)

        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = np.array([(Suuu + Suvv) / 2, (Svvv + Suuv) / 2])

        try:
            center_local = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        uc, vc = center_local
        center = np.array([uc + x_mean, vc + y_mean])

        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        radius = distances.mean()

        if radius < 1e-10:
            return None

        std_dist = distances.std()
        circularity = 1 - min(1, std_dist / radius)

        return center, radius, circularity

    def _filter_by_normal(
        self,
        points: np.ndarray,
        all_points: np.ndarray,
        indices: List[int],
        expected_normal: np.ndarray,
        tree: KDTree,
        angle_threshold_deg: float = 30.0,
    ) -> np.ndarray:
        """
        Filter points by checking if local normal matches expected surface normal.

        Args:
            points: Points to filter
            all_points: All scan points (for neighbor lookup)
            indices: Indices of points in all_points
            expected_normal: Expected surface normal direction
            tree: KDTree for neighbor queries
            angle_threshold_deg: Maximum angle deviation from expected normal (degrees)

        Returns:
            Filtered points that lie on surfaces aligned with expected_normal
        """
        if len(points) < 10:
            return points

        normals = self._estimate_local_normals(all_points, indices, tree)
        if normals is None:
            return points

        # Keep points with normals aligned to expected (within angle_threshold)
        # cos(30°) ≈ 0.866, cos(45°) ≈ 0.707
        cos_threshold = np.cos(np.radians(angle_threshold_deg))
        dots = np.abs(np.dot(normals, expected_normal))
        mask = dots > cos_threshold

        filtered_count = mask.sum()
        if filtered_count >= 5:
            return points[mask]
        else:
            # If too few points pass strict filter, try relaxed threshold
            relaxed_threshold = np.cos(np.radians(45.0))
            relaxed_mask = dots > relaxed_threshold
            if relaxed_mask.sum() >= 5:
                logger.debug(
                    f"Relaxed normal filter from {angle_threshold_deg}° to 45° "
                    f"({filtered_count} -> {relaxed_mask.sum()} points)"
                )
                return points[relaxed_mask]
            return points

    def _estimate_local_normals(
        self,
        scan_points: np.ndarray,
        indices: List[int],
        tree: KDTree,
        k: int = 10,
    ) -> Optional[np.ndarray]:
        """Estimate normals for selected points using local PCA."""
        normals = []
        points = scan_points[indices]

        for i, idx in enumerate(indices):
            point = scan_points[idx]

            # Find k nearest neighbors
            _, neighbor_idx = tree.query(point, k=k + 1)

            if len(neighbor_idx) < 4:
                normals.append([0, 0, 1])
                continue

            neighbors = scan_points[neighbor_idx[1:]]  # Exclude point itself

            # PCA for normal estimation
            centered = neighbors - neighbors.mean(axis=0)
            try:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # Smallest eigenvalue
            except np.linalg.LinAlgError:
                normal = np.array([0, 0, 1])

            normals.append(normal)

        return np.array(normals)

    def _fit_plane(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Fit plane to points using SVD."""
        if len(points) < 3:
            return None

        centroid = points.mean(axis=0)
        centered = points - centroid

        try:
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Last row is normal
        except np.linalg.LinAlgError:
            return None

        # Plane equation: normal . (p - centroid) = 0
        # => normal . p = d, where d = normal . centroid
        d = np.dot(normal, centroid)

        return normal, d
