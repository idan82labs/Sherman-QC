"""
Bend Detector Module

Detects bend features in point clouds without requiring global alignment.
Uses RANSAC plane segmentation and dihedral angle computation.

Mathematical foundations:
- RANSAC: Fischler & Bolles (1981)
- Dihedral angles: θ = arccos(|n₁ · n₂|)
- Bend line: d = n₁ × n₂

Industry standards referenced:
- ISO 2768-1: General tolerances
- VDI/VDE 2634: Optical 3D measuring systems
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import KDTree
from scipy.optimize import least_squares
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlaneSegment:
    """A detected planar region in the point cloud."""
    plane_id: int
    normal: np.ndarray          # Unit normal vector [a, b, c]
    d: float                    # Plane equation: ax + by + cz + d = 0
    centroid: np.ndarray        # Center of the plane segment
    points: np.ndarray          # Points belonging to this plane
    boundary_points: np.ndarray # Points on the boundary
    area: float                 # Estimated surface area (mm²)
    inlier_count: int

    def point_to_plane_distance(self, point: np.ndarray) -> float:
        """Signed distance from point to plane."""
        return np.dot(self.normal, point) + self.d

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "normal": self.normal.tolist(),
            "centroid": self.centroid.tolist(),
            "area": round(self.area, 2),
            "point_count": len(self.points),
        }


@dataclass
class DetectedBend:
    """A bend detected in the scan point cloud."""
    bend_id: str
    measured_angle: float       # Dihedral angle in degrees
    measured_radius: float      # Estimated bend radius in mm
    bend_line_start: np.ndarray
    bend_line_end: np.ndarray
    bend_line_direction: np.ndarray
    confidence: float           # 0-1, based on fit quality

    # Adjacent flanges
    flange1: PlaneSegment
    flange2: PlaneSegment

    # Quality metrics
    inlier_count: int = 0
    radius_fit_error: float = 0.0

    @property
    def bend_line_length(self) -> float:
        return np.linalg.norm(self.bend_line_end - self.bend_line_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bend_id": self.bend_id,
            "measured_angle": round(self.measured_angle, 2),
            "measured_radius": round(self.measured_radius, 3),
            "bend_line_length": round(self.bend_line_length, 2),
            "bend_line_start": self.bend_line_start.tolist(),
            "bend_line_end": self.bend_line_end.tolist(),
            "confidence": round(self.confidence, 3),
            "flange1_area": round(self.flange1.area, 2),
            "flange2_area": round(self.flange2.area, 2),
        }


@dataclass
class BendSpecification:
    """Bend specification extracted from CAD."""
    bend_id: str
    target_angle: float         # Target bend angle in degrees
    target_radius: float        # Target inner bend radius in mm
    bend_line_start: np.ndarray
    bend_line_end: np.ndarray
    tolerance_angle: float = 1.0    # ± degrees
    tolerance_radius: float = 0.5   # ± mm

    # Surface normals for the two flanges
    flange1_normal: Optional[np.ndarray] = None
    flange2_normal: Optional[np.ndarray] = None

    # Flange dimensions for matching
    flange1_area: float = 0.0
    flange2_area: float = 0.0

    @property
    def bend_line_length(self) -> float:
        return np.linalg.norm(self.bend_line_end - self.bend_line_start)

    @property
    def bend_line_direction(self) -> np.ndarray:
        direction = self.bend_line_end - self.bend_line_start
        return direction / (np.linalg.norm(direction) + 1e-10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bend_id": self.bend_id,
            "target_angle": self.target_angle,
            "target_radius": self.target_radius,
            "bend_line_length": round(self.bend_line_length, 2),
            "tolerance_angle": self.tolerance_angle,
            "tolerance_radius": self.tolerance_radius,
        }


@dataclass
class BendMatch:
    """Result of matching a detected bend to a CAD specification."""
    cad_bend: BendSpecification
    detected_bend: Optional[DetectedBend]

    # Deviations
    angle_deviation: float = 0.0    # Measured - Target (degrees)
    radius_deviation: float = 0.0   # Measured - Target (mm)

    # Status
    status: str = "NOT_DETECTED"    # PASS, FAIL, WARNING, NOT_DETECTED
    match_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bend_id": self.cad_bend.bend_id,
            "target_angle": self.cad_bend.target_angle,
            "measured_angle": self.detected_bend.measured_angle if self.detected_bend else None,
            "angle_deviation": round(self.angle_deviation, 2) if self.detected_bend else None,
            "target_radius": self.cad_bend.target_radius,
            "measured_radius": self.detected_bend.measured_radius if self.detected_bend else None,
            "radius_deviation": round(self.radius_deviation, 3) if self.detected_bend else None,
            "status": self.status,
            "confidence": round(self.match_confidence, 3),
            "tolerance_angle": self.cad_bend.tolerance_angle,
            "tolerance_radius": self.cad_bend.tolerance_radius,
        }


@dataclass
class BendInspectionReport:
    """Complete report from progressive bend inspection."""
    part_id: str
    matches: List[BendMatch]
    unmatched_detections: List[DetectedBend] = field(default_factory=list)

    # Summary
    total_cad_bends: int = 0
    detected_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

    # Timing
    processing_time_ms: float = 0.0

    def compute_summary(self):
        """Compute summary statistics."""
        self.total_cad_bends = len(self.matches)
        self.detected_count = sum(1 for m in self.matches if m.status != "NOT_DETECTED")
        self.pass_count = sum(1 for m in self.matches if m.status == "PASS")
        self.fail_count = sum(1 for m in self.matches if m.status == "FAIL")
        self.warning_count = sum(1 for m in self.matches if m.status == "WARNING")

    @property
    def progress_percentage(self) -> float:
        if self.total_cad_bends == 0:
            return 0.0
        return (self.detected_count / self.total_cad_bends) * 100

    def to_dict(self) -> Dict[str, Any]:
        self.compute_summary()
        return {
            "part_id": self.part_id,
            "summary": {
                "total_bends": self.total_cad_bends,
                "detected": self.detected_count,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "warnings": self.warning_count,
                "progress_pct": round(self.progress_percentage, 1),
            },
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_detections": [d.to_dict() for d in self.unmatched_detections],
            "processing_time_ms": round(self.processing_time_ms, 1),
        }

    def to_table_string(self) -> str:
        """Generate ASCII table for operator display."""
        self.compute_summary()

        lines = []
        lines.append("=" * 75)
        lines.append(f"  PROGRESSIVE BEND INSPECTION - {self.part_id}")
        lines.append("=" * 75)
        lines.append("")
        lines.append(f"  Progress: {self.detected_count} of {self.total_cad_bends} bends complete ({self.progress_percentage:.0f}%)")
        lines.append("")
        lines.append("  " + "-" * 71)
        lines.append(f"  {'Bend':<6} {'Target':>8} {'Measured':>10} {'Deviation':>10} {'Status':>10}")
        lines.append("  " + "-" * 71)

        for m in self.matches:
            target = f"{m.cad_bend.target_angle:.1f}°"
            if m.detected_bend:
                measured = f"{m.detected_bend.measured_angle:.1f}°"
                deviation = f"{m.angle_deviation:+.1f}°"
            else:
                measured = "-"
                deviation = "-"

            status_symbol = {
                "PASS": "✓ OK",
                "FAIL": "✗ FAIL",
                "WARNING": "⚠ WARN",
                "NOT_DETECTED": "Not yet",
            }.get(m.status, m.status)

            lines.append(f"  {m.cad_bend.bend_id:<6} {target:>8} {measured:>10} {deviation:>10} {status_symbol:>10}")

        lines.append("  " + "-" * 71)

        # Add warnings for failures
        failures = [m for m in self.matches if m.status == "FAIL"]
        if failures:
            lines.append("")
            for m in failures:
                lines.append(f"  ✗ {m.cad_bend.bend_id}: {abs(m.angle_deviation):.1f}° deviation exceeds ±{m.cad_bend.tolerance_angle}° tolerance")

        warnings = [m for m in self.matches if m.status == "WARNING"]
        if warnings:
            lines.append("")
            for m in warnings:
                lines.append(f"  ⚠ {m.cad_bend.bend_id}: {abs(m.angle_deviation):.1f}° deviation - approaching tolerance limit")

        lines.append("")
        lines.append("=" * 75)

        return "\n".join(lines)


class BendDetector:
    """
    Detect bend features in point clouds using RANSAC plane segmentation.

    Algorithm:
    1. Segment point cloud into planar regions (RANSAC)
    2. Find adjacent plane pairs (boundary analysis)
    3. Compute dihedral angle for each pair
    4. Estimate bend radius from transition zone
    """

    def __init__(
        self,
        ransac_iterations: int = 100,
        plane_threshold: float = 0.5,       # mm - inlier distance threshold
        min_plane_points: int = 100,
        min_plane_area: float = 50.0,       # mm²
        adjacency_threshold: float = 5.0,   # mm - max gap between adjacent planes
        min_bend_angle: float = 10.0,       # degrees - ignore angles < this
        max_bend_angle: float = 170.0,      # degrees - ignore angles > this
        adaptive_threshold: bool = True,    # Adapt RANSAC threshold to point density
        ransac_confidence: float = 0.99,    # Target probability for RANSAC
        ransac_seed: Optional[int] = 42,    # Seed for reproducibility
    ):
        self.ransac_iterations = ransac_iterations
        self.plane_threshold = plane_threshold
        self.min_plane_points = min_plane_points
        self.min_plane_area = min_plane_area
        self.adjacency_threshold = adjacency_threshold
        self.min_bend_angle = min_bend_angle
        self.max_bend_angle = max_bend_angle
        self.adaptive_threshold = adaptive_threshold
        self.ransac_confidence = ransac_confidence
        self._rng = np.random.default_rng(ransac_seed)

    def detect_bends(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        preprocess: bool = True,
        voxel_size: float = 0.3,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        multiscale: bool = False,
    ) -> List[DetectedBend]:
        """
        Detect all bends in the point cloud.

        Args:
            points: Nx3 array of point coordinates
            normals: Optional Nx3 array of point normals (improves segmentation)
            preprocess: Whether to apply voxel downsampling and outlier removal
            voxel_size: Voxel size for downsampling (mm)
            outlier_nb_neighbors: Number of neighbors for outlier detection
            outlier_std_ratio: Standard deviation ratio for outlier removal
            multiscale: Enable multi-scale detection for complex parts (38+ bends)

        Returns:
            List of detected bends
        """
        if len(points) < self.min_plane_points * 2:
            logger.warning(f"Insufficient points for bend detection: {len(points)}")
            return []

        # Optional preprocessing
        if preprocess:
            logger.info("Preprocessing point cloud...")
            points, normals = self._preprocess_point_cloud(
                points, normals, voxel_size, outlier_nb_neighbors, outlier_std_ratio
            )
            logger.info(f"After preprocessing: {len(points)} points")

        # Multi-scale detection for complex parts
        if multiscale:
            logger.info("Using multi-scale detection...")
            return self._detect_bends_multiscale(points, normals, scales=[0.5, 1.0, 2.0])

        # Single-scale detection (standard path)
        return self._detect_bends_single_scale(points, normals)

    def _detect_bends_single_scale(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> List[DetectedBend]:
        """
        Standard single-scale bend detection.

        Args:
            points: Preprocessed Nx3 array of point coordinates
            normals: Optional Nx3 array of point normals

        Returns:
            List of detected bends
        """
        # Step 1: Segment into planes
        logger.info("Segmenting point cloud into planes...")
        planes = self._segment_planes(points, normals)
        logger.info(f"Found {len(planes)} planar segments")

        if len(planes) < 2:
            logger.warning("Need at least 2 planes to detect bends")
            return []

        # Step 2: Find adjacent plane pairs
        logger.info("Finding adjacent plane pairs...")
        adjacent_pairs = self._find_adjacent_planes(planes)
        logger.info(f"Found {len(adjacent_pairs)} adjacent plane pairs")

        # Step 3: Compute bend features for each pair
        bends = []
        for i, (plane1, plane2) in enumerate(adjacent_pairs):
            bend = self._analyze_bend(plane1, plane2, points, bend_id=f"B{i+1}")
            if bend is not None:
                bends.append(bend)

        logger.info(f"Detected {len(bends)} bends")
        return bends

    def _detect_bends_multiscale(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        scales: List[float] = [0.5, 1.0, 2.0],
    ) -> List[DetectedBend]:
        """
        Multi-scale bend detection for complex parts.

        Runs detection at multiple scales and merges results, keeping
        the highest-confidence detection for each unique bend.

        Args:
            points: Preprocessed Nx3 array of point coordinates
            normals: Optional Nx3 array of point normals
            scales: List of scale factors for adjacency_threshold and min_plane_area

        Returns:
            Merged list of detected bends
        """
        all_bends: List[DetectedBend] = []
        original_adjacency = self.adjacency_threshold
        original_min_area = self.min_plane_area

        for scale in scales:
            logger.info(f"Multi-scale detection at scale {scale}...")

            # Adjust parameters for this scale
            self.adjacency_threshold = original_adjacency * scale
            self.min_plane_area = original_min_area * scale

            # Run single-scale detection
            scale_bends = self._detect_bends_single_scale(points, normals)

            # Tag bends with scale info
            for bend in scale_bends:
                bend.bend_id = f"{bend.bend_id}_s{scale}"
            all_bends.extend(scale_bends)

        # Restore original parameters
        self.adjacency_threshold = original_adjacency
        self.min_plane_area = original_min_area

        # Merge overlapping bends, keeping highest confidence
        merged_bends = self._merge_multiscale_bends(all_bends)
        logger.info(f"Multi-scale detection: {len(all_bends)} raw -> {len(merged_bends)} merged")

        # Renumber bend IDs
        for i, bend in enumerate(merged_bends):
            bend.bend_id = f"B{i+1}"

        return merged_bends

    def _merge_multiscale_bends(
        self,
        bends: List[DetectedBend],
        merge_threshold: float = 10.0,
    ) -> List[DetectedBend]:
        """
        Merge overlapping bends from multi-scale detection.

        Two bends are considered duplicates if their bend line midpoints
        are within merge_threshold and their angles are within 10 degrees.

        Args:
            bends: List of bends from all scales
            merge_threshold: Distance threshold for considering bends as duplicates (mm)

        Returns:
            Deduplicated list keeping highest-confidence bends
        """
        if not bends:
            return []

        # Sort by confidence (highest first)
        sorted_bends = sorted(bends, key=lambda b: b.confidence, reverse=True)
        merged: List[DetectedBend] = []

        for bend in sorted_bends:
            # Check if this bend overlaps with any already-merged bend
            midpoint = (bend.bend_line_start + bend.bend_line_end) / 2
            is_duplicate = False

            for existing in merged:
                existing_midpoint = (existing.bend_line_start + existing.bend_line_end) / 2
                distance = np.linalg.norm(midpoint - existing_midpoint)
                angle_diff = abs(bend.measured_angle - existing.measured_angle)

                if distance < merge_threshold and angle_diff < 10.0:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(bend)

        return merged

    def _preprocess_point_cloud(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        voxel_size: float,
        nb_neighbors: int,
        std_ratio: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply voxel downsampling and statistical outlier removal.

        Args:
            points: Nx3 array of points
            normals: Optional Nx3 array of normals
            voxel_size: Voxel size for downsampling (mm)
            nb_neighbors: Number of neighbors for outlier detection
            std_ratio: Standard deviation ratio for outlier removal

        Returns:
            Tuple of (processed_points, processed_normals)
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        # Step 1: Voxel downsampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Step 2: Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        # Step 3: Ensure normals exist
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )

        processed_points = np.asarray(pcd.points)
        processed_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

        return processed_points, processed_normals

    def _compute_adaptive_threshold(self, points: np.ndarray) -> float:
        """
        Compute RANSAC threshold based on local point density.

        Uses median nearest-neighbor distance to adapt to scan quality.

        Returns:
            Adaptive threshold in mm, clamped to [0.1, 2.0]
        """
        # Sample up to 1000 points for efficiency
        n_points = len(points)
        sample_size = min(1000, n_points)
        sample_idx = self._rng.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_idx]

        # Compute median nearest-neighbor distance
        tree = KDTree(sample_points)
        distances, _ = tree.query(sample_points, k=2)
        median_nn = np.median(distances[:, 1])  # [:, 0] is distance to self (0)

        # Adaptive: 3× median NN, clamped to [0.1, 2.0]mm
        adaptive = 3.0 * median_nn
        return float(np.clip(adaptive, 0.1, 2.0))

    def _compute_ransac_iterations(self, n_points: int, inlier_ratio: float = 0.3) -> int:
        """
        Compute RANSAC iterations for target confidence.

        Formula: k = log(1-p) / log(1-w³)
        Where p = confidence, w = inlier ratio, 3 = sample size

        Args:
            n_points: Number of points in dataset
            inlier_ratio: Expected inlier ratio (default 0.3 = 30%)

        Returns:
            Number of iterations, clamped to [100, 3000]
        """
        p = self.ransac_confidence
        w = max(inlier_ratio, 0.1)  # Avoid log(0)

        # k = log(1-p) / log(1-w³)
        k = int(np.log(1 - p) / np.log(1 - w**3))

        return int(np.clip(k, 100, 3000))

    def _segment_planes(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> List[PlaneSegment]:
        """
        Segment point cloud into planar regions using iterative RANSAC.

        Uses the RANSAC algorithm (Fischler & Bolles, 1981) for robust plane fitting.
        """
        planes = []
        remaining_mask = np.ones(len(points), dtype=bool)
        plane_id = 0

        # Compute adaptive parameters once for the whole point cloud
        if self.adaptive_threshold:
            effective_threshold = self._compute_adaptive_threshold(points)
            logger.debug(f"Adaptive RANSAC threshold: {effective_threshold:.3f}mm")
        else:
            effective_threshold = self.plane_threshold

        iterations = self._compute_ransac_iterations(len(points))
        logger.debug(f"RANSAC iterations: {iterations}")

        while np.sum(remaining_mask) >= self.min_plane_points:
            remaining_points = points[remaining_mask]
            remaining_indices = np.where(remaining_mask)[0]

            # RANSAC plane fitting with adaptive parameters
            best_plane = self._ransac_plane_fit(
                remaining_points,
                threshold=effective_threshold,
                iterations=iterations
            )

            if best_plane is None:
                break

            normal, d, inlier_mask = best_plane

            if np.sum(inlier_mask) < self.min_plane_points:
                break

            # Extract inlier points
            inlier_indices = remaining_indices[inlier_mask]
            plane_points = points[inlier_indices]

            # Compute plane properties
            centroid = np.mean(plane_points, axis=0)
            area = self._estimate_area(plane_points)

            if area < self.min_plane_area:
                # Mark as used but don't add to planes
                remaining_mask[inlier_indices] = False
                continue

            # Extract boundary points
            boundary = self._extract_boundary(plane_points)

            plane = PlaneSegment(
                plane_id=plane_id,
                normal=normal,
                d=d,
                centroid=centroid,
                points=plane_points,
                boundary_points=boundary,
                area=area,
                inlier_count=len(plane_points),
            )
            planes.append(plane)

            # Remove inliers from remaining points
            remaining_mask[inlier_indices] = False
            plane_id += 1

            # Limit number of planes to avoid infinite loops
            if len(planes) >= 50:
                break

        return planes

    def _ransac_plane_fit(
        self,
        points: np.ndarray,
        threshold: Optional[float] = None,
        iterations: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        RANSAC plane fitting with adaptive parameters.

        Mathematical model: ax + by + cz + d = 0

        Args:
            points: Nx3 array of points
            threshold: Inlier distance threshold (uses self.plane_threshold if None)
            iterations: Number of RANSAC iterations (uses self.ransac_iterations if None)

        Returns:
            Tuple of (normal, d, inlier_mask) or None if failed
        """
        n_points = len(points)
        if n_points < 3:
            return None

        # Use provided parameters or defaults
        effective_threshold = threshold if threshold is not None else self.plane_threshold
        effective_iterations = iterations if iterations is not None else self.ransac_iterations

        best_inliers = 0
        best_normal = None
        best_d = None
        best_mask = None

        for _ in range(effective_iterations):
            # Sample 3 random points using seeded RNG for reproducibility
            indices = self._rng.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[indices]

            # Compute plane normal: n = (p2 - p1) × (p3 - p1)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            norm_length = np.linalg.norm(normal)
            if norm_length < 1e-10:
                continue  # Collinear points

            normal = normal / norm_length

            # Compute d: d = -n · p1
            d = -np.dot(normal, p1)

            # Count inliers: |ax + by + cz + d| < threshold
            distances = np.abs(np.dot(points, normal) + d)
            inlier_mask = distances < effective_threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_normal = normal
                best_d = d
                best_mask = inlier_mask

        if best_normal is None:
            return None

        # Refine with least squares on inliers
        inlier_points = points[best_mask]
        if len(inlier_points) >= 3:
            refined_normal, refined_d = self._least_squares_plane(inlier_points)

            # Recompute inliers with refined plane
            distances = np.abs(np.dot(points, refined_normal) + refined_d)
            best_mask = distances < effective_threshold
            best_normal = refined_normal
            best_d = refined_d

        return best_normal, best_d, best_mask

    def _least_squares_plane(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Least squares plane fitting using SVD.

        Minimizes sum of squared distances from points to plane.
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD: the normal is the eigenvector with smallest eigenvalue
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # Last row = smallest singular value direction

        # Ensure consistent normal direction (pointing "up" on average)
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, centroid)

        return normal, d

    def _estimate_area(self, points: np.ndarray) -> float:
        """
        Estimate planar area using convex hull projection.
        """
        if len(points) < 3:
            return 0.0

        try:
            # Project to 2D using PCA
            centroid = np.mean(points, axis=0)
            centered = points - centroid

            _, _, Vt = np.linalg.svd(centered)

            # Project onto the two principal axes
            proj_2d = centered @ Vt[:2].T

            # Compute convex hull area
            from scipy.spatial import ConvexHull
            if len(proj_2d) >= 3:
                hull = ConvexHull(proj_2d)
                return hull.volume  # In 2D, "volume" is area
        except Exception:
            pass

        # Fallback: bounding box area
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        sorted_sizes = np.sort(bbox_size)
        return sorted_sizes[1] * sorted_sizes[2]  # Two largest dimensions

    def _extract_boundary(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Extract boundary points of a planar region using alpha shapes concept.
        """
        if len(points) < k:
            return points

        try:
            # Project to 2D
            centroid = np.mean(points, axis=0)
            centered = points - centroid

            _, _, Vt = np.linalg.svd(centered)
            proj_2d = centered @ Vt[:2].T

            # Find boundary using convex hull
            from scipy.spatial import ConvexHull
            if len(proj_2d) >= 3:
                hull = ConvexHull(proj_2d)
                boundary_indices = hull.vertices
                return points[boundary_indices]
        except Exception:
            pass

        return points[:min(len(points), 50)]  # Fallback: return subset

    def _find_adjacent_planes(self, planes: List[PlaneSegment]) -> List[Tuple[PlaneSegment, PlaneSegment]]:
        """
        Find pairs of planes that share an edge (adjacent planes forming a bend).

        Criteria:
        1. Boundary points are within adjacency_threshold
        2. Boundaries are roughly parallel (shared edge)
        3. Planes are not coplanar (angle between normals > min_bend_angle)
        """
        adjacent_pairs = []

        for i, plane1 in enumerate(planes):
            for j, plane2 in enumerate(planes):
                if j <= i:
                    continue

                # Check if planes are not coplanar
                cos_angle = abs(np.dot(plane1.normal, plane2.normal))
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                # The dihedral angle (bend angle) is 180 - angle between normals
                dihedral_angle = 180 - angle_deg

                if dihedral_angle < self.min_bend_angle or dihedral_angle > self.max_bend_angle:
                    continue

                # Check boundary proximity
                if self._boundaries_adjacent(plane1, plane2):
                    adjacent_pairs.append((plane1, plane2))

        return adjacent_pairs

    def _boundaries_adjacent(self, plane1: PlaneSegment, plane2: PlaneSegment) -> bool:
        """
        Check if two planes share an adjacent boundary.
        """
        if len(plane1.boundary_points) == 0 or len(plane2.boundary_points) == 0:
            return False

        # Build KDTree for efficient nearest neighbor queries
        tree1 = KDTree(plane1.boundary_points)

        # Find closest points from plane2 boundary to plane1 boundary
        distances, _ = tree1.query(plane2.boundary_points)

        # Count how many boundary points are close
        close_count = np.sum(distances < self.adjacency_threshold)

        # Need at least 20% of boundary points to be close
        min_close = max(3, int(0.2 * min(len(plane1.boundary_points), len(plane2.boundary_points))))

        return close_count >= min_close

    def _analyze_bend(
        self,
        plane1: PlaneSegment,
        plane2: PlaneSegment,
        all_points: np.ndarray,
        bend_id: str
    ) -> Optional[DetectedBend]:
        """
        Analyze a potential bend between two planes.

        Computes:
        1. Dihedral angle (bend angle)
        2. Bend line (intersection of planes)
        3. Bend radius (from transition zone)
        """
        # Compute dihedral angle
        # θ = arccos(|n₁ · n₂|) gives angle between normals
        # Bend angle = 180° - θ
        cos_angle = np.dot(plane1.normal, plane2.normal)
        angle_between_normals = np.degrees(np.arccos(np.clip(abs(cos_angle), -1, 1)))
        dihedral_angle = 180 - angle_between_normals

        # Compute bend line direction: d = n₁ × n₂
        bend_direction = np.cross(plane1.normal, plane2.normal)
        bend_dir_norm = np.linalg.norm(bend_direction)

        if bend_dir_norm < 1e-10:
            return None  # Planes are parallel

        bend_direction = bend_direction / bend_dir_norm

        # Find a point on the bend line (intersection of planes)
        bend_point = self._find_line_point(plane1, plane2, bend_direction)

        if bend_point is None:
            return None

        # Determine bend line extent from boundary proximity
        bend_start, bend_end = self._find_bend_extent(
            plane1, plane2, bend_point, bend_direction
        )

        # Estimate bend radius from transition zone
        radius, radius_confidence = self._estimate_bend_radius(
            all_points, bend_point, bend_direction,
            plane1.normal, plane2.normal, dihedral_angle
        )

        # Compute overall confidence
        confidence = self._compute_confidence(
            plane1, plane2, radius_confidence, dihedral_angle
        )

        return DetectedBend(
            bend_id=bend_id,
            measured_angle=dihedral_angle,
            measured_radius=radius,
            bend_line_start=bend_start,
            bend_line_end=bend_end,
            bend_line_direction=bend_direction,
            confidence=confidence,
            flange1=plane1,
            flange2=plane2,
            inlier_count=plane1.inlier_count + plane2.inlier_count,
            radius_fit_error=0.0,  # Could be computed from radius fitting
        )

    def _find_line_point(
        self,
        plane1: PlaneSegment,
        plane2: PlaneSegment,
        direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find a point on the intersection line of two planes.

        Solves:
        n₁ · p = -d₁
        n₂ · p = -d₂
        """
        # Use the formula: p₀ = ((n₁·p₁)(n₂×d) + (n₂·p₂)(d×n₁)) / |d|²
        # Where d is the line direction and p₁, p₂ are points on the planes

        n1, d1 = plane1.normal, plane1.d
        n2, d2 = plane2.normal, plane2.d

        # Simplified approach: find point closest to both plane centroids
        # that lies on both planes

        # Set up system: find point p such that n1·p = -d1 and n2·p = -d2
        # Using least squares with constraint that p is close to centroids

        A = np.array([n1, n2])
        b = np.array([-d1, -d2])

        # Add direction constraint (p should be on the line)
        # Solve using pseudoinverse
        try:
            # Find point on line closest to origin
            # p = (A^T A)^(-1) A^T b + t * direction
            AtA = A @ A.T
            if np.linalg.det(AtA) < 1e-10:
                return (plane1.centroid + plane2.centroid) / 2

            # Use centroid of centroids projected onto intersection line
            mid_centroid = (plane1.centroid + plane2.centroid) / 2

            # Project onto intersection line
            # First find any point on the line
            p0 = np.linalg.lstsq(A, b, rcond=None)[0]

            # Project mid_centroid onto the line through p0 with direction
            t = np.dot(mid_centroid - p0, direction)
            return p0 + t * direction

        except Exception:
            return (plane1.centroid + plane2.centroid) / 2

    def _find_bend_extent(
        self,
        plane1: PlaneSegment,
        plane2: PlaneSegment,
        bend_point: np.ndarray,
        bend_direction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the start and end points of the bend line.
        """
        # Project all boundary points onto the bend line
        all_boundary = np.vstack([plane1.boundary_points, plane2.boundary_points])

        # Project onto bend line: t = (p - bend_point) · direction
        projections = np.dot(all_boundary - bend_point, bend_direction)

        t_min, t_max = np.min(projections), np.max(projections)

        # Add small margin
        margin = 0.05 * (t_max - t_min)

        bend_start = bend_point + (t_min - margin) * bend_direction
        bend_end = bend_point + (t_max + margin) * bend_direction

        return bend_start, bend_end

    def _estimate_bend_radius(
        self,
        points: np.ndarray,
        bend_point: np.ndarray,
        bend_direction: np.ndarray,
        normal1: np.ndarray,
        normal2: np.ndarray,
        dihedral_angle: float
    ) -> Tuple[float, float]:
        """
        Improved radius estimation using Taubin algebraic circle fit.

        More robust than RANSAC for partial arcs typical in bend transitions.

        Industry constraints:
        - MIN_RADIUS: 0.5mm (sheet metal minimum)
        - MAX_RADIUS: 25mm (practical upper bound)
        """
        MIN_RADIUS, MAX_RADIUS = 0.5, 25.0

        # Step 1: Adaptive search radius based on expected radius range
        search_radius = min(30.0, MAX_RADIUS * 1.5)
        tree = KDTree(points)
        indices = tree.query_ball_point(bend_point, search_radius)

        if len(indices) < 20:
            # Insufficient points - use heuristic fallback
            return self._estimate_radius_from_angle(dihedral_angle), 0.3

        nearby_points = points[indices]

        # Step 2: Project to 2D perpendicular to bend line
        bend_dir = bend_direction / (np.linalg.norm(bend_direction) + 1e-10)
        perp1 = normal1 - np.dot(normal1, bend_dir) * bend_dir
        perp1_norm = np.linalg.norm(perp1)
        if perp1_norm < 1e-10:
            # Fallback if normal1 is parallel to bend direction
            perp1 = normal2 - np.dot(normal2, bend_dir) * bend_dir
            perp1_norm = np.linalg.norm(perp1)
        perp1 = perp1 / (perp1_norm + 1e-10)
        perp2 = np.cross(bend_dir, perp1)

        relative = nearby_points - bend_point
        x_coords = np.dot(relative, perp1)
        y_coords = np.dot(relative, perp2)
        points_2d = np.column_stack([x_coords, y_coords])

        # Step 3: Taubin circle fit (robust for partial arcs)
        radius, center_2d, fit_quality = self._taubin_circle_fit(points_2d)

        # Step 4: Validate and clamp to industry constraints
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            radius = float(np.clip(radius, MIN_RADIUS, MAX_RADIUS))
            fit_quality *= 0.5  # Reduce confidence when clamping

        return radius, fit_quality

    def _taubin_circle_fit(self, points_2d: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Taubin algebraic circle fit - more robust than RANSAC for partial arcs.

        Based on: Taubin, G. "Estimation of Planar Curves, Surfaces, and
        Nonplanar Space Curves Defined by Implicit Equations with Applications
        to Edge and Range Image Segmentation" (1991).

        Args:
            points_2d: Nx2 array of 2D points

        Returns:
            Tuple of (radius, center_2d, fit_quality)
        """
        if len(points_2d) < 3:
            return 3.0, np.zeros(2), 0.0

        x, y = points_2d[:, 0], points_2d[:, 1]
        x_mean, y_mean = x.mean(), y.mean()
        u, v = x - x_mean, y - y_mean

        # Compute sums for the Taubin method
        Suu = np.sum(u * u)
        Svv = np.sum(v * v)
        Suv = np.sum(u * v)
        Suuu = np.sum(u * u * u)
        Svvv = np.sum(v * v * v)
        Suuv = np.sum(u * u * v)
        Suvv = np.sum(u * v * v)

        # Solve the linear system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = np.array([(Suuu + Suvv) / 2, (Svvv + Suuv) / 2])

        try:
            center_local = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular matrix - fall back to centroid
            return 3.0, np.array([x_mean, y_mean]), 0.0

        center_2d = np.array([center_local[0] + x_mean, center_local[1] + y_mean])

        # Compute radius and fit quality
        distances = np.sqrt((x - center_2d[0])**2 + (y - center_2d[1])**2)
        radius = float(distances.mean())

        # Fit quality: 1 - normalized standard deviation of distances
        if radius > 1e-10:
            fit_quality = 1.0 - min(1.0, distances.std() / radius)
        else:
            fit_quality = 0.0

        return radius, center_2d, fit_quality

    def _estimate_radius_from_angle(self, angle: float) -> float:
        """
        Fallback: heuristic radius estimation from bend angle.

        Based on typical sheet metal forming practices.

        Args:
            angle: Dihedral angle in degrees

        Returns:
            Estimated radius in mm
        """
        if angle < 60:
            return 4.0
        elif angle < 100:
            return 2.5
        elif angle < 150:
            return 2.0
        else:
            return 1.5

    def _compute_confidence(
        self,
        plane1: PlaneSegment,
        plane2: PlaneSegment,
        radius_confidence: float,
        dihedral_angle: float,
        inlier_ratio: Optional[float] = None,
    ) -> float:
        """
        Calibrated confidence scoring based on empirical factors.

        Weights:
        - Point count: 30% (logistic curve, saturates at 500 points)
        - Area: 20% (logistic curve, saturates at 1000mm²)
        - Radius fit: 15%
        - Angle reasonableness: 15% (peak at 90°)
        - Inlier ratio: 20% (from RANSAC)

        Args:
            plane1: First plane segment
            plane2: Second plane segment
            radius_confidence: Confidence from radius fitting (0-1)
            dihedral_angle: Measured dihedral angle in degrees
            inlier_ratio: Optional RANSAC inlier ratio (0-1)

        Returns:
            Confidence score (0-1)
        """
        # Factor 1: Point count (30%) - logistic curve, saturates at 500
        total_points = plane1.inlier_count + plane2.inlier_count
        point_score = 1.0 / (1.0 + np.exp(-0.01 * (total_points - 200)))

        # Factor 2: Area (20%) - logistic, saturates at 1000mm²
        total_area = plane1.area + plane2.area
        area_score = 1.0 / (1.0 + np.exp(-0.005 * (total_area - 400)))

        # Factor 3: Radius fit (15%)
        radius_score = max(0.0, min(1.0, radius_confidence))

        # Factor 4: Angle reasonableness (15%) - peak at 90°
        if dihedral_angle < 30:
            angle_score = 0.3
        elif dihedral_angle < 60:
            angle_score = 0.6
        elif dihedral_angle < 120:
            angle_score = 1.0
        elif dihedral_angle < 150:
            angle_score = 0.7
        else:
            angle_score = 0.4

        # Factor 5: Inlier ratio (20%)
        inlier_score = inlier_ratio if inlier_ratio is not None else 0.5

        confidence = (
            0.30 * point_score +
            0.20 * area_score +
            0.15 * radius_score +
            0.15 * angle_score +
            0.20 * inlier_score
        )

        return min(1.0, max(0.0, confidence))


class ProgressiveBendMatcher:
    """
    Match detected bends to CAD bend specifications.

    Uses a cost-function based matching approach that considers:
    - Angle similarity (primary)
    - Radius similarity
    - Bend line length similarity
    - Flange area similarity
    """

    def __init__(
        self,
        angle_weight: float = 0.5,
        radius_weight: float = 0.2,
        length_weight: float = 0.2,
        area_weight: float = 0.1,
        max_angle_diff: float = 15.0,  # Maximum angle difference for matching
    ):
        self.angle_weight = angle_weight
        self.radius_weight = radius_weight
        self.length_weight = length_weight
        self.area_weight = area_weight
        self.max_angle_diff = max_angle_diff

    def match(
        self,
        detected_bends: List[DetectedBend],
        cad_bends: List[BendSpecification],
        part_id: str = "unknown"
    ) -> BendInspectionReport:
        """
        Match detected bends to CAD specifications.

        Uses greedy matching based on cost function.
        For more complex cases, could use Hungarian algorithm.
        """
        matches = []
        used_detections = set()

        # For each CAD bend, find best matching detection
        for cad_bend in cad_bends:
            best_match = None
            best_cost = float('inf')
            best_detection = None

            for i, detected in enumerate(detected_bends):
                if i in used_detections:
                    continue

                cost = self._compute_match_cost(detected, cad_bend)

                if cost < best_cost and cost < 1.0:  # 1.0 = maximum acceptable cost
                    best_cost = cost
                    best_detection = detected
                    best_match = i

            if best_match is not None:
                used_detections.add(best_match)

                # Compute deviations
                angle_dev = best_detection.measured_angle - cad_bend.target_angle
                radius_dev = best_detection.measured_radius - cad_bend.target_radius

                # Determine status
                status = self._determine_status(cad_bend, best_detection)

                matches.append(BendMatch(
                    cad_bend=cad_bend,
                    detected_bend=best_detection,
                    angle_deviation=angle_dev,
                    radius_deviation=radius_dev,
                    status=status,
                    match_confidence=1.0 - best_cost,
                ))
            else:
                # No match found - bend not yet made
                matches.append(BendMatch(
                    cad_bend=cad_bend,
                    detected_bend=None,
                    status="NOT_DETECTED",
                    match_confidence=0.0,
                ))

        # Collect unmatched detections
        unmatched = [d for i, d in enumerate(detected_bends) if i not in used_detections]

        report = BendInspectionReport(
            part_id=part_id,
            matches=matches,
            unmatched_detections=unmatched,
        )
        report.compute_summary()

        return report

    def _compute_match_cost(self, detected: DetectedBend, cad: BendSpecification) -> float:
        """
        Compute cost of matching a detection to a CAD specification.

        Lower cost = better match.
        """
        # Angle difference (normalized by max acceptable diff)
        angle_diff = abs(detected.measured_angle - cad.target_angle)
        if angle_diff > self.max_angle_diff:
            return float('inf')  # Reject if angle too different

        angle_cost = angle_diff / self.max_angle_diff

        # Radius difference (normalized)
        radius_diff = abs(detected.measured_radius - cad.target_radius)
        radius_cost = min(1.0, radius_diff / (cad.target_radius + 1e-10))

        # Length difference (normalized)
        length_diff = abs(detected.bend_line_length - cad.bend_line_length)
        length_cost = min(1.0, length_diff / (cad.bend_line_length + 1e-10))

        # Area similarity (if available)
        area_cost = 0.0
        if cad.flange1_area > 0 and cad.flange2_area > 0:
            cad_total = cad.flange1_area + cad.flange2_area
            det_total = detected.flange1.area + detected.flange2.area
            area_cost = min(1.0, abs(cad_total - det_total) / cad_total)

        total_cost = (
            self.angle_weight * angle_cost +
            self.radius_weight * radius_cost +
            self.length_weight * length_cost +
            self.area_weight * area_cost
        )

        return total_cost

    def _determine_status(self, cad: BendSpecification, detected: DetectedBend) -> str:
        """
        Determine pass/fail/warning status.
        """
        angle_dev = abs(detected.measured_angle - cad.target_angle)

        if angle_dev <= cad.tolerance_angle:
            return "PASS"
        elif angle_dev <= cad.tolerance_angle * 2:
            return "WARNING"
        else:
            return "FAIL"


class CADBendExtractor:
    """
    Extract bend specifications from CAD models.

    For mesh files (PLY/STL): Uses geometric analysis similar to BendDetector
    For STEP files: Can extract exact bend features from B-rep geometry
    """

    def __init__(
        self,
        min_bend_angle: float = 10.0,
        max_bend_angle: float = 170.0,
        default_tolerance_angle: float = 1.0,
        default_tolerance_radius: float = 0.5,
        default_bend_radius: float = 3.0,
    ):
        self.min_bend_angle = min_bend_angle
        self.max_bend_angle = max_bend_angle
        self.default_tolerance_angle = default_tolerance_angle
        self.default_tolerance_radius = default_tolerance_radius
        self.default_bend_radius = default_bend_radius

    def extract_from_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> List[BendSpecification]:
        """
        Extract bend specifications from mesh geometry.

        Uses similar approach to BendDetector but on dense mesh vertices.
        """
        # Use BendDetector on mesh vertices
        detector = BendDetector(
            min_plane_points=50,
            min_plane_area=20.0,
            min_bend_angle=self.min_bend_angle,
            max_bend_angle=self.max_bend_angle,
        )

        detected = detector.detect_bends(vertices)

        # Convert detected bends to specifications
        specs = []
        for i, d in enumerate(detected):
            spec = BendSpecification(
                bend_id=f"CAD_B{i+1}",
                target_angle=d.measured_angle,
                target_radius=d.measured_radius if d.measured_radius > 0.5 else self.default_bend_radius,
                bend_line_start=d.bend_line_start,
                bend_line_end=d.bend_line_end,
                tolerance_angle=self.default_tolerance_angle,
                tolerance_radius=self.default_tolerance_radius,
                flange1_normal=d.flange1.normal,
                flange2_normal=d.flange2.normal,
                flange1_area=d.flange1.area,
                flange2_area=d.flange2.area,
            )
            specs.append(spec)

        return specs

    def extract_from_file(self, file_path: str) -> List[BendSpecification]:
        """
        Extract bend specifications from CAD file.
        """
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            logger.error(f"Failed to load mesh from {file_path}")
            return []

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        return self.extract_from_mesh(vertices, faces)
