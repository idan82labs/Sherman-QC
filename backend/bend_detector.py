"""
Bend Detection Algorithm for Sheet Metal Parts

This module implements a sophisticated bend detection system using:
1. Surface normal estimation via PCA on local neighborhoods
2. Curvature computation using eigenvalue ratio method
3. Surface segmentation by normal consistency using DBSCAN
4. Bend identification from high-curvature regions
5. Angle computation from adjacent plane normals
6. Bend radius estimation from curvature profile
7. Automatic bend ordering and numbering

The algorithm is specifically designed for sheet metal parts scanned
with LiDAR or structured light scanners.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import time

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from multi_model.exchange_schema import (
    BendFeature,
    BendDetectionResult,
    Feature3D,
    FeatureType,
)

logger = logging.getLogger(__name__)


# Material-specific springback thresholds (in degrees)
# Based on industry data from The Fabricator, CustomPartNet, and manufacturing handbooks
# Format: (typical_min, typical_max, expected_avg)
MATERIAL_SPRINGBACK = {
    "mild_steel": (0.5, 2.0, 1.25),
    "aluminum": (1.0, 4.0, 2.5),
    "stainless_steel": (2.0, 5.0, 3.5),
    "high_strength_steel": (3.0, 8.0, 5.5),
    "copper": (0.5, 2.0, 1.25),
    "brass": (1.0, 3.0, 2.0),
    "titanium": (4.0, 10.0, 7.0),
    "galvanized_steel": (1.0, 3.0, 2.0),
    # Default/unknown material
    "unknown": (1.0, 5.0, 3.0),
}


def get_springback_threshold(material: str = "unknown") -> float:
    """
    Get the expected springback threshold for a given material.

    Args:
        material: Material type (e.g., "mild_steel", "aluminum", "stainless_steel")

    Returns:
        Expected average springback in degrees
    """
    material_lower = material.lower().replace(" ", "_").replace("-", "_")
    if material_lower in MATERIAL_SPRINGBACK:
        return MATERIAL_SPRINGBACK[material_lower][2]  # Return expected average
    return MATERIAL_SPRINGBACK["unknown"][2]


def get_springback_range(material: str = "unknown") -> Tuple[float, float]:
    """
    Get the typical springback range for a given material.

    Args:
        material: Material type

    Returns:
        Tuple of (min_springback, max_springback) in degrees
    """
    material_lower = material.lower().replace(" ", "_").replace("-", "_")
    if material_lower in MATERIAL_SPRINGBACK:
        data = MATERIAL_SPRINGBACK[material_lower]
        return (data[0], data[1])
    return (MATERIAL_SPRINGBACK["unknown"][0], MATERIAL_SPRINGBACK["unknown"][1])


@dataclass
class SurfaceSegment:
    """A planar surface segment detected in the point cloud."""
    segment_id: int
    point_indices: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    plane_d: float  # ax + by + cz + d = 0
    area_estimate: float
    flatness_score: float  # How planar is this segment (0-1)


@dataclass
class BendCandidate:
    """A potential bend before validation."""
    point_indices: np.ndarray
    curvature_values: np.ndarray
    mean_curvature: float
    adjacent_segments: List[int]
    centroid: np.ndarray


class BendDetector:
    """
    Detects and analyzes bends in sheet metal point clouds.

    Algorithm Overview:
    1. Compute local surface normals using PCA
    2. Compute curvature at each point
    3. Segment flat surfaces using DBSCAN on normals
    4. Identify high-curvature transition regions
    5. Pair adjacent surfaces to identify bends
    6. Compute bend angles and radii
    7. Order bends spatially
    """

    def __init__(
        self,
        normal_radius: float = 5.0,  # mm, radius for normal estimation
        curvature_radius: float = 3.0,  # mm, radius for curvature
        curvature_threshold: float = 0.02,  # Threshold for bend detection
        min_bend_points: int = 50,  # Minimum points in a bend region
        surface_normal_tolerance: float = 0.15,  # Tolerance for surface grouping
        min_surface_points: int = 100,  # Minimum points in a surface
        dbscan_eps: float = 0.1,  # DBSCAN epsilon for normal clustering
        dbscan_min_samples: int = 10,  # DBSCAN minimum samples
    ):
        self.normal_radius = normal_radius
        self.curvature_radius = curvature_radius
        self.curvature_threshold = curvature_threshold
        self.min_bend_points = min_bend_points
        self.surface_normal_tolerance = surface_normal_tolerance
        self.min_surface_points = min_surface_points
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Cached computations
        self._points: Optional[np.ndarray] = None
        self._kdtree: Optional[KDTree] = None
        self._normals: Optional[np.ndarray] = None
        self._curvatures: Optional[np.ndarray] = None
        self._surfaces: List[SurfaceSegment] = []
        self._bends: List[BendFeature] = []

    def detect_bends(
        self,
        points: np.ndarray,
        deviations: Optional[np.ndarray] = None,
    ) -> BendDetectionResult:
        """
        Main entry point for bend detection.

        Args:
            points: Nx3 array of point cloud coordinates
            deviations: Optional Nx1 array of deviation values for each point

        Returns:
            BendDetectionResult with detected bends
        """
        start_time = time.time()
        warnings = []

        if not HAS_SKLEARN:
            warnings.append("sklearn not available, using fallback clustering")

        if len(points) < 100:
            return BendDetectionResult(
                bends=[],
                total_bends_detected=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                warnings=["Insufficient points for bend detection"],
            )

        logger.info(f"Starting bend detection on {len(points)} points")

        # Store points and build KDTree
        self._points = np.asarray(points, dtype=np.float64)
        self._kdtree = KDTree(self._points)

        # Step 1: Estimate surface normals
        logger.info("Step 1: Estimating surface normals...")
        self._normals = self._estimate_normals()

        # Step 2: Compute curvature
        logger.info("Step 2: Computing curvature...")
        self._curvatures = self._compute_curvature()

        # Step 3: Segment surfaces
        logger.info("Step 3: Segmenting surfaces...")
        self._surfaces = self._segment_surfaces()
        logger.info(f"  Found {len(self._surfaces)} surface segments")

        # Step 4: Identify bend regions
        logger.info("Step 4: Identifying bend regions...")
        bend_candidates = self._identify_bend_regions()
        logger.info(f"  Found {len(bend_candidates)} bend candidates")

        # Step 5: Analyze and validate bends
        logger.info("Step 5: Analyzing bends...")
        self._bends = self._analyze_bends(bend_candidates, deviations)
        logger.info(f"  Validated {len(self._bends)} bends")

        # Step 6: Order and number bends
        logger.info("Step 6: Ordering bends...")
        self._bends = self._order_bends(self._bends)

        processing_time = (time.time() - start_time) * 1000

        return BendDetectionResult(
            bends=self._bends,
            total_bends_detected=len(self._bends),
            detection_method="curvature_segmentation",
            processing_time_ms=processing_time,
            warnings=warnings,
        )

    def _estimate_normals(self) -> np.ndarray:
        """Estimate surface normals using PCA on local neighborhoods."""
        n_points = len(self._points)
        normals = np.zeros((n_points, 3))

        for i in range(n_points):
            # Find neighbors within radius
            neighbors_idx = self._kdtree.query_ball_point(
                self._points[i], self.normal_radius
            )

            if len(neighbors_idx) < 3:
                # Not enough neighbors, use a default normal
                normals[i] = [0, 0, 1]
                continue

            # Get neighbor points
            neighbors = self._points[neighbors_idx]

            # Center the points
            centered = neighbors - neighbors.mean(axis=0)

            # PCA to find normal (smallest eigenvector)
            try:
                if HAS_SKLEARN:
                    pca = PCA(n_components=3)
                    pca.fit(centered)
                    normal = pca.components_[-1]  # Smallest variance direction
                else:
                    # Fallback: use covariance eigenvectors
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    normal = eigenvectors[:, 0]  # Smallest eigenvalue

                # Ensure consistent orientation (pointing "up" on average)
                if normal[2] < 0:
                    normal = -normal

                normals[i] = normal / np.linalg.norm(normal)
            except Exception:
                normals[i] = [0, 0, 1]

        return normals

    def _compute_curvature(self) -> np.ndarray:
        """Compute curvature at each point using eigenvalue ratio method."""
        n_points = len(self._points)
        curvatures = np.zeros(n_points)

        for i in range(n_points):
            neighbors_idx = self._kdtree.query_ball_point(
                self._points[i], self.curvature_radius
            )

            if len(neighbors_idx) < 4:
                continue

            neighbors = self._points[neighbors_idx]
            centered = neighbors - neighbors.mean(axis=0)

            try:
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)

                # Curvature estimate: ratio of smallest to sum of eigenvalues
                # High curvature = points are not on a plane
                total = eigenvalues.sum()
                if total > 1e-10:
                    curvatures[i] = eigenvalues[0] / total
            except Exception:
                pass

        return curvatures

    def _segment_surfaces(self) -> List[SurfaceSegment]:
        """Segment point cloud into planar surfaces based on normal similarity."""
        surfaces = []

        if not HAS_SKLEARN:
            # Fallback: simple binning by normal direction
            return self._segment_surfaces_fallback()

        # Use DBSCAN on normals to cluster points with similar orientations
        # Also consider spatial proximity
        n_points = len(self._points)

        # Create feature vector: [normal_x, normal_y, normal_z, x/scale, y/scale, z/scale]
        scale = np.max(np.abs(self._points)) / 10  # Normalize positions
        features = np.hstack([
            self._normals,
            self._points / scale * 0.3,  # Weight position less than normal
        ])

        # Only cluster low-curvature points (likely on flat surfaces)
        flat_mask = self._curvatures < self.curvature_threshold * 0.5
        flat_indices = np.where(flat_mask)[0]

        if len(flat_indices) < self.min_surface_points:
            logger.warning("Not enough flat points for surface segmentation")
            return surfaces

        flat_features = features[flat_indices]

        try:
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            ).fit(flat_features)

            labels = clustering.labels_
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label

            for label in unique_labels:
                cluster_mask = labels == label
                cluster_indices = flat_indices[cluster_mask]

                if len(cluster_indices) < self.min_surface_points:
                    continue

                cluster_points = self._points[cluster_indices]
                cluster_normals = self._normals[cluster_indices]

                # Compute average normal
                avg_normal = cluster_normals.mean(axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)

                # Compute centroid
                centroid = cluster_points.mean(axis=0)

                # Compute plane equation (ax + by + cz + d = 0)
                plane_d = -np.dot(avg_normal, centroid)

                # Estimate area (convex hull approximation)
                area_estimate = len(cluster_indices) * (self.normal_radius ** 2) / 100

                # Compute flatness score
                distances_to_plane = np.abs(
                    np.dot(cluster_points - centroid, avg_normal)
                )
                flatness_score = 1.0 - min(1.0, distances_to_plane.std() / 0.5)

                surfaces.append(SurfaceSegment(
                    segment_id=len(surfaces),
                    point_indices=cluster_indices,
                    centroid=centroid,
                    normal=avg_normal,
                    plane_d=plane_d,
                    area_estimate=area_estimate,
                    flatness_score=flatness_score,
                ))

        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")

        return surfaces

    def _segment_surfaces_fallback(self) -> List[SurfaceSegment]:
        """Fallback surface segmentation without sklearn."""
        surfaces = []

        # Bin normals into octants
        normal_bins: Dict[Tuple[int, int, int], List[int]] = {}

        for i, normal in enumerate(self._normals):
            if self._curvatures[i] >= self.curvature_threshold * 0.5:
                continue  # Skip high-curvature points

            # Quantize normal to 8 directions
            key = (
                int(np.sign(normal[0])),
                int(np.sign(normal[1])),
                int(np.sign(normal[2])),
            )
            if key not in normal_bins:
                normal_bins[key] = []
            normal_bins[key].append(i)

        for bin_key, indices in normal_bins.items():
            if len(indices) < self.min_surface_points:
                continue

            cluster_indices = np.array(indices)
            cluster_points = self._points[cluster_indices]
            cluster_normals = self._normals[cluster_indices]

            avg_normal = cluster_normals.mean(axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            centroid = cluster_points.mean(axis=0)
            plane_d = -np.dot(avg_normal, centroid)

            surfaces.append(SurfaceSegment(
                segment_id=len(surfaces),
                point_indices=cluster_indices,
                centroid=centroid,
                normal=avg_normal,
                plane_d=plane_d,
                area_estimate=len(cluster_indices) * (self.normal_radius ** 2) / 100,
                flatness_score=0.8,
            ))

        return surfaces

    def _identify_bend_regions(self) -> List[BendCandidate]:
        """Identify regions with high curvature as bend candidates."""
        candidates = []

        # Find high-curvature points
        high_curv_mask = self._curvatures >= self.curvature_threshold
        high_curv_indices = np.where(high_curv_mask)[0]

        if len(high_curv_indices) < self.min_bend_points:
            return candidates

        # Cluster high-curvature points spatially
        high_curv_points = self._points[high_curv_indices]

        if HAS_SKLEARN:
            try:
                clustering = DBSCAN(
                    eps=self.normal_radius * 1.5,
                    min_samples=self.min_bend_points // 2,
                ).fit(high_curv_points)

                labels = clustering.labels_
                unique_labels = set(labels)
                unique_labels.discard(-1)

                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_indices = high_curv_indices[cluster_mask]

                    if len(cluster_indices) < self.min_bend_points:
                        continue

                    cluster_points = self._points[cluster_indices]
                    cluster_curvatures = self._curvatures[cluster_indices]

                    # Find adjacent surfaces
                    adjacent = self._find_adjacent_surfaces(cluster_indices)

                    candidates.append(BendCandidate(
                        point_indices=cluster_indices,
                        curvature_values=cluster_curvatures,
                        mean_curvature=cluster_curvatures.mean(),
                        adjacent_segments=adjacent,
                        centroid=cluster_points.mean(axis=0),
                    ))

            except Exception as e:
                logger.warning(f"Bend region clustering failed: {e}")
        else:
            # Fallback: treat all high-curvature points as one bend region
            candidates.append(BendCandidate(
                point_indices=high_curv_indices,
                curvature_values=self._curvatures[high_curv_indices],
                mean_curvature=self._curvatures[high_curv_indices].mean(),
                adjacent_segments=list(range(len(self._surfaces))),
                centroid=high_curv_points.mean(axis=0),
            ))

        return candidates

    def _find_adjacent_surfaces(self, bend_indices: np.ndarray) -> List[int]:
        """Find surface segments adjacent to a bend region."""
        adjacent = []
        bend_points = self._points[bend_indices]
        bend_centroid = bend_points.mean(axis=0)

        for surface in self._surfaces:
            # Check if any surface points are near the bend region
            surface_points = self._points[surface.point_indices]

            # Compute minimum distance from surface to bend
            min_dist = np.inf
            for bp in bend_points[::max(1, len(bend_points) // 20)]:  # Sample
                dists = np.linalg.norm(surface_points - bp, axis=1)
                min_dist = min(min_dist, dists.min())

            if min_dist < self.normal_radius * 3:
                adjacent.append(surface.segment_id)

        return adjacent

    def _analyze_bends(
        self,
        candidates: List[BendCandidate],
        deviations: Optional[np.ndarray],
    ) -> List[BendFeature]:
        """Analyze bend candidates to compute angles, radii, and validate."""
        bends = []

        for i, candidate in enumerate(candidates):
            # Need at least 2 adjacent surfaces to determine a bend
            if len(candidate.adjacent_segments) < 2:
                logger.debug(f"Candidate {i} rejected: only {len(candidate.adjacent_segments)} adjacent surfaces")
                continue

            # Get the two most prominent adjacent surfaces
            surface_pairs = []
            for j, seg_id_1 in enumerate(candidate.adjacent_segments):
                for seg_id_2 in candidate.adjacent_segments[j + 1:]:
                    surf1 = self._surfaces[seg_id_1]
                    surf2 = self._surfaces[seg_id_2]

                    # Compute angle between surface normals
                    # For sheet metal with consistent outward-pointing normals:
                    # - Parallel surfaces (flat): dot ≈ 1 → angle ≈ 0° → bend = 180° (not a bend)
                    # - 90° bend: dot ≈ 0 → angle = 90° → bend = 90°
                    # - Opposite normals: dot ≈ -1 → angle = 180° → bend = 0°
                    dot = np.clip(np.dot(surf1.normal, surf2.normal), -1, 1)

                    # Skip near-parallel surfaces - these are flat regions, not bends
                    if abs(dot) > 0.95:  # Within ~18° of parallel
                        continue

                    angle_between_normals = np.arccos(dot) * 180 / np.pi

                    # Bend angle is supplement of angle between normals
                    # (perpendicular normals = 90° between them = 90° bend)
                    bend_angle = 180 - angle_between_normals

                    if 10 < bend_angle < 170:  # Reasonable bend angle
                        surface_pairs.append({
                            "surf1": surf1,
                            "surf2": surf2,
                            "bend_angle": bend_angle,
                            "normal_angle": angle_between_normals,
                        })

            if not surface_pairs:
                logger.debug(f"Candidate {i} rejected: no valid surface pairs")
                continue

            # Use the pair with largest area surfaces
            best_pair = max(
                surface_pairs,
                key=lambda p: p["surf1"].area_estimate + p["surf2"].area_estimate,
            )

            # Compute bend line (intersection of two planes)
            bend_line_start, bend_line_end = self._compute_bend_line(
                best_pair["surf1"],
                best_pair["surf2"],
                candidate.centroid,
            )

            # Estimate bend radius from curvature
            # Curvature k ≈ 1/R for a cylindrical bend
            mean_curv = candidate.mean_curvature
            if mean_curv > 1e-6:
                # Empirical scaling factor (curvature metric to actual radius)
                radius_mm = 1.0 / (mean_curv * 10)  # Adjust based on calibration
                radius_mm = max(0.5, min(50, radius_mm))  # Reasonable bounds
            else:
                radius_mm = 5.0  # Default

            # Determine bend direction
            direction = self._determine_bend_direction(
                candidate.centroid,
                best_pair["surf1"].normal,
                best_pair["surf2"].normal,
            )

            # Compute confidence based on surface flatness and curvature consistency
            confidence = min(1.0, (
                best_pair["surf1"].flatness_score * 0.25 +
                best_pair["surf2"].flatness_score * 0.25 +
                (1 - candidate.curvature_values.std() / mean_curv) * 0.5
            ))

            bends.append(BendFeature(
                bend_id=len(bends) + 1,
                bend_name=f"Bend {len(bends) + 1}",
                angle_degrees=best_pair["bend_angle"],
                radius_mm=radius_mm,
                bend_line_start=tuple(bend_line_start),
                bend_line_end=tuple(bend_line_end),
                bend_apex=tuple(candidate.centroid),
                region_point_indices=candidate.point_indices.tolist(),
                adjacent_surface_ids=[
                    best_pair["surf1"].segment_id,
                    best_pair["surf2"].segment_id,
                ],
                detection_confidence=confidence,
                bend_direction=direction,
            ))

        return bends

    def _compute_bend_line(
        self,
        surf1: SurfaceSegment,
        surf2: SurfaceSegment,
        bend_centroid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the bend line as intersection of two planes."""
        # Direction of bend line is cross product of normals
        direction = np.cross(surf1.normal, surf2.normal)

        if np.linalg.norm(direction) < 1e-6:
            # Parallel surfaces, use centroid connection
            direction = surf2.centroid - surf1.centroid

        direction = direction / np.linalg.norm(direction)

        # Bend line passes through bend centroid
        # Create line segment of reasonable length
        line_length = 50.0  # mm

        start = bend_centroid - direction * line_length / 2
        end = bend_centroid + direction * line_length / 2

        return start, end

    def _determine_bend_direction(
        self,
        bend_apex: np.ndarray,
        normal1: np.ndarray,
        normal2: np.ndarray,
    ) -> str:
        """Determine the bend direction (up/down/left/right)."""
        # Average of the two surface normals points "into" the bend
        avg_normal = (normal1 + normal2) / 2
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        # Determine primary direction
        abs_normal = np.abs(avg_normal)
        max_idx = np.argmax(abs_normal)

        if max_idx == 2:  # Z is primary
            return "up" if avg_normal[2] > 0 else "down"
        elif max_idx == 0:  # X is primary
            return "right" if avg_normal[0] > 0 else "left"
        else:  # Y is primary
            return "forward" if avg_normal[1] > 0 else "backward"

    def _order_bends(self, bends: List[BendFeature]) -> List[BendFeature]:
        """Order bends spatially and renumber them."""
        if len(bends) <= 1:
            return bends

        # Find the primary axis of the part (PCA on bend centroids)
        centroids = np.array([b.bend_apex for b in bends])

        if HAS_SKLEARN:
            pca = PCA(n_components=1)
            pca.fit(centroids)
            primary_axis = pca.components_[0]
        else:
            # Use axis with most variance
            variances = centroids.var(axis=0)
            primary_axis = np.zeros(3)
            primary_axis[np.argmax(variances)] = 1

        # Project centroids onto primary axis
        projections = np.dot(centroids, primary_axis)

        # Sort by projection
        sorted_indices = np.argsort(projections)

        # Renumber bends
        ordered_bends = []
        for new_id, old_idx in enumerate(sorted_indices, start=1):
            bend = bends[old_idx]
            ordered_bends.append(BendFeature(
                bend_id=new_id,
                bend_name=f"Bend {new_id}",
                angle_degrees=bend.angle_degrees,
                radius_mm=bend.radius_mm,
                bend_line_start=bend.bend_line_start,
                bend_line_end=bend.bend_line_end,
                bend_apex=bend.bend_apex,
                region_point_indices=bend.region_point_indices,
                adjacent_surface_ids=bend.adjacent_surface_ids,
                detection_confidence=bend.detection_confidence,
                bend_direction=bend.bend_direction,
            ))

        return ordered_bends

    def get_surfaces(self) -> List[SurfaceSegment]:
        """Return the detected surface segments."""
        return self._surfaces

    def get_bend_points_mask(self) -> np.ndarray:
        """Return a boolean mask of points that are part of bends."""
        if self._points is None:
            return np.array([])

        mask = np.zeros(len(self._points), dtype=bool)
        for bend in self._bends:
            mask[bend.region_point_indices] = True
        return mask

    def get_curvatures(self) -> Optional[np.ndarray]:
        """Return computed curvature values."""
        return self._curvatures

    def get_normals(self) -> Optional[np.ndarray]:
        """Return computed surface normals."""
        return self._normals


def analyze_bend_deviations(
    bend: BendFeature,
    nominal_angle: Optional[float],
    points: np.ndarray,
    deviations: np.ndarray,
    tolerance: float = 1.0,
    material: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze deviations in a specific bend region.

    Args:
        bend: The detected bend feature
        nominal_angle: Expected angle from drawing (degrees)
        points: All point cloud points
        deviations: Deviation values for all points
        tolerance: Tolerance threshold in degrees/mm
        material: Material type for springback calibration (e.g., "mild_steel", "aluminum")

    Returns:
        Dictionary with bend analysis results
    """
    bend_points = points[bend.region_point_indices]
    bend_deviations = deviations[bend.region_point_indices]

    # Compute deviation statistics
    mean_dev = float(np.mean(bend_deviations))
    max_dev = float(np.max(np.abs(bend_deviations)))
    std_dev = float(np.std(bend_deviations))

    # Determine pass/fail
    if nominal_angle is not None:
        angle_deviation = bend.angle_degrees - nominal_angle
    else:
        angle_deviation = 0.0

    # Get material-specific springback threshold
    # Springback factor Ks = measured_angle / nominal_angle
    # Typical values by material (from The Fabricator, CustomPartNet):
    # - Mild Steel: 0.5° - 2°
    # - Aluminum: 1° - 4°
    # - Stainless Steel: 2° - 5°
    # - High-Strength Steel: 3° - 8°
    springback_threshold = get_springback_threshold(material)
    springback_min, springback_max = get_springback_range(material)

    # Springback: measured angle is less than nominal (under-bent)
    # Overbend: measured angle is greater than nominal
    springback_indicator = 0.0
    over_bend_indicator = 0.0
    springback_diagnosis = None

    if nominal_angle is not None:
        if angle_deviation < 0:  # Under-bent (springback)
            # Normalize by material-specific expected springback
            springback_indicator = min(1.0, abs(angle_deviation) / springback_max)
            # Diagnose if springback is within typical range for material
            if abs(angle_deviation) <= springback_min:
                springback_diagnosis = "minimal"
            elif abs(angle_deviation) <= springback_threshold:
                springback_diagnosis = "typical"
            elif abs(angle_deviation) <= springback_max:
                springback_diagnosis = "moderate"
            else:
                springback_diagnosis = "excessive"
        else:  # Over-bent
            # Overbend is usually intentional compensation, normalize similarly
            over_bend_indicator = min(1.0, angle_deviation / springback_max)

    # Status determination
    status = "pass"
    if abs(angle_deviation) > tolerance or max_dev > tolerance:
        if abs(angle_deviation) > tolerance * 2 or max_dev > tolerance * 2:
            status = "fail"
        else:
            status = "warning"

    return {
        "bend_id": bend.bend_id,
        "bend_name": bend.bend_name,
        "measured_angle": bend.angle_degrees,
        "nominal_angle": nominal_angle,
        "angle_deviation": angle_deviation,
        "mean_deviation_mm": mean_dev,
        "max_deviation_mm": max_dev,
        "std_deviation_mm": std_dev,
        "springback_indicator": springback_indicator,
        "over_bend_indicator": over_bend_indicator,
        "springback_diagnosis": springback_diagnosis,
        "material": material,
        "expected_springback_range": f"{springback_min:.1f}° - {springback_max:.1f}°",
        "status": status,
        "point_count": len(bend_points),
    }


def measure_scan_bend_via_signed_distance_gradient(
    scan_points: np.ndarray,
    deviations: np.ndarray,
    cad_bend_center: np.ndarray,
    cad_surf1_normal: np.ndarray,
    cad_surf2_normal: np.ndarray,
    cad_bend_direction: np.ndarray,
    cad_bend_angle: float = 90.0,
    search_radius: float = 30.0,
    min_offset_mm: float = 3.0,
) -> Dict[str, Any]:
    """
    Measure bend angle deviation using signed distance gradient analysis.

    Since the scan is ICP-aligned to the CAD, the signed distance from each
    scan point to the CAD surface encodes how the scan deviates. Near a bend:
    - If the bend angle matches the CAD, signed distance ≈ 0 everywhere
    - If the part has springback (under-bent), the signed distance shows a
      systematic gradient across the bend: positive on one side, negative
      on the other
    - The slope of this gradient directly encodes the angular deviation

    For each surface adjacent to the bend, we:
    1. Select scan points on that surface (away from bend transition)
    2. Project each point onto the surface plane's perpendicular-to-bend axis
    3. Fit: signed_distance = slope * d_perpendicular + intercept
    4. The angular deviation for that surface = arctan(slope)
    5. Total bend deviation = surf1_deviation + surf2_deviation

    This approach:
    - Does NOT require surface classification (uses CAD normals directly)
    - Does NOT require plane fitting
    - Directly measures angular deviation, not absolute angle
    - Is robust to scan noise (linear regression on many points)
    - Works equally for 90° and 135° bends

    Args:
        scan_points: Nx3 scan point cloud (ICP-aligned to CAD)
        deviations: N-length signed distance array (scan-to-CAD, from compute_deviations)
        cad_bend_center: 3D center of the CAD bend
        cad_surf1_normal: Normal of CAD surface 1
        cad_surf2_normal: Normal of CAD surface 2
        cad_bend_direction: Direction vector of the bend line
        cad_bend_angle: The CAD bend angle in degrees
        search_radius: Radius around bend center to search for scan points (mm)
        min_offset_mm: Minimum distance from bend line to use points (excludes transition)

    Returns:
        Dictionary with angular deviation and quality metrics
    """
    # Normalize directions
    bend_dir = cad_bend_direction / (np.linalg.norm(cad_bend_direction) + 1e-12)
    n1 = cad_surf1_normal / (np.linalg.norm(cad_surf1_normal) + 1e-12)
    n2 = cad_surf2_normal / (np.linalg.norm(cad_surf2_normal) + 1e-12)

    # Step 1: Find scan points near bend center
    scan_tree = KDTree(scan_points)
    nearby_idx = scan_tree.query_ball_point(cad_bend_center, search_radius)
    if len(nearby_idx) < 50:
        return _fail(f"Only {len(nearby_idx)} scan points within {search_radius}mm of bend")

    nearby_points = scan_points[nearby_idx]
    nearby_devs = deviations[nearby_idx]

    # Step 2: Classify points to surfaces using tangent distance (along surface,
    # perpendicular to bend line) and plane distance (closeness to surface plane).
    vecs = nearby_points - cad_bend_center

    # Tangent directions: along each surface, perpendicular to bend line
    # tang = cross(normal, bend_dir) gives the in-surface direction away from bend
    tang1 = np.cross(n1, bend_dir)
    tang1_norm = np.linalg.norm(tang1)
    tang2 = np.cross(n2, bend_dir)
    tang2_norm = np.linalg.norm(tang2)

    if tang1_norm < 0.01 or tang2_norm < 0.01:
        return _fail("Bend direction parallel to surface normal")
    tang1 = tang1 / tang1_norm
    tang2 = tang2 / tang2_norm

    # Tangent distances: how far from bend along each surface (absolute: either side)
    d_tang1 = np.abs(vecs @ tang1)
    d_tang2 = np.abs(vecs @ tang2)

    # Plane distances: how close each point is to each surface's plane
    d_plane1 = np.abs(vecs @ n1)
    d_plane2 = np.abs(vecs @ n2)

    # Distance along bend line
    d_along_bend = vecs @ bend_dir

    # Remove points far along the bend line
    bend_line_limit = search_radius * 0.8
    along_bend_ok = np.abs(d_along_bend) < bend_line_limit

    # Try multiple transition zone exclusion radii
    best_gradient_result = None
    best_gradient_quality = -1
    _grad_fail_reasons = []

    for offset in [min_offset_mm, min_offset_mm * 0.5, min_offset_mm * 1.5, min_offset_mm * 2.0]:
        # Surface classification: a point belongs to surface 1 if:
        # 1. It's close to surface 1's plane (d_plane1 small — "on" the surface)
        # 2. It's far from the bend along surface 1 (d_tang1 large)
        # 3. It's NOT near the bend transition zone
        plane_thresh = 3.0  # mm — max distance to surface plane
        surf1_mask = (d_plane1 < plane_thresh) & (d_tang1 > offset) & along_bend_ok
        surf2_mask = (d_plane2 < plane_thresh) & (d_tang2 > offset) & along_bend_ok
        # Remove points that are on BOTH planes (near bend line — ambiguous)
        both_close = (d_plane1 < plane_thresh) & (d_plane2 < plane_thresh)
        surf1_mask &= ~both_close
        surf2_mask &= ~both_close

        n1_pts = int(surf1_mask.sum())
        n2_pts = int(surf2_mask.sum())

        if n1_pts < 15 or n2_pts < 15:
            _grad_fail_reasons.append(f"offset={offset:.1f}: pts={n1_pts}/{n2_pts}")
            continue

        # Step 4: For each surface, fit the signed distance gradient
        # signed_distance = slope * d_perpendicular + intercept
        # angular_deviation_rad ≈ arctan(slope)

        def _fit_gradient(mask, tang_dists):
            """Fit linear gradient of signed distance vs perpendicular distance."""
            devs = nearby_devs[mask]
            dists = tang_dists[mask]

            # Remove outlier deviations (> 5mm from CAD)
            inlier = np.abs(devs) < 5.0
            if inlier.sum() < 15:
                return None, None, 0, 0.0, None

            devs = devs[inlier]
            dists = dists[inlier]

            # RANSAC-like: do iterative fit, removing outliers from fit residuals
            for iteration in range(3):
                A = np.column_stack([dists, np.ones(len(dists))])
                try:
                    result, residuals, rank, sv = np.linalg.lstsq(A, devs, rcond=None)
                except np.linalg.LinAlgError:
                    return None, None, 0, 0.0, None

                slope, intercept = result

                predicted = slope * dists + intercept
                res = np.abs(devs - predicted)
                if iteration < 2:
                    threshold = max(np.percentile(res, 85), 0.3)
                    keep = res < threshold
                    if keep.sum() < 15:
                        break
                    devs = devs[keep]
                    dists = dists[keep]

            # Compute slope standard error (more meaningful than R² for this use case:
            # a correctly-bent part has R²≈0 since there's no gradient, but the slope
            # estimate is still precise if there are many points spread over distance)
            n_pts = len(devs)
            ss_res = np.sum((devs - (slope * dists + intercept))**2)
            ss_tot = np.sum((devs - devs.mean())**2)
            r_squared = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else 0.0

            # Standard error of slope
            mse = ss_res / max(n_pts - 2, 1)
            dists_var = np.sum((dists - dists.mean())**2)
            se_slope = np.sqrt(mse / max(dists_var, 1e-12))

            return slope, intercept, n_pts, r_squared, se_slope

        slope1, intercept1, count1, r2_1, se1 = _fit_gradient(surf1_mask, d_tang1)
        slope2, intercept2, count2, r2_2, se2 = _fit_gradient(surf2_mask, d_tang2)

        if slope1 is None and slope2 is None:
            _grad_fail_reasons.append(f"offset={offset:.1f}: fit_failed")
            continue

        # Step 5: Compute angular deviations from slopes
        ang_dev1 = np.degrees(np.arctan(slope1)) if slope1 is not None else 0.0
        ang_dev2 = np.degrees(np.arctan(slope2)) if slope2 is not None else 0.0

        # Total bend deviation is the sum of both surface deviations
        total_angular_deviation = ang_dev1 + ang_dev2
        measured_angle = round(cad_bend_angle + total_angular_deviation, 2)

        # Confidence based on point counts and slope standard error
        # SE of slope is more meaningful than R² — a correctly-bent part has R²≈0
        # but the slope estimate can still be precise (small SE) with many points
        min_count = min(count1 if count1 else 0, count2 if count2 else 0)
        max_se = max(
            se1 if se1 is not None else 1.0,
            se2 if se2 is not None else 1.0,
        )
        # Convert SE to angular uncertainty: se_angle = arctan(se_slope) in degrees
        se_angle_deg = np.degrees(np.arctan(max_se)) if max_se < 10 else 90.0

        if min_count >= 50 and se_angle_deg < 1.0:
            confidence = "high"
        elif min_count >= 20 and se_angle_deg < 3.0:
            confidence = "medium"
        else:
            confidence = "low"

        # Skip low confidence — try next offset
        if confidence == "low":
            _grad_fail_reasons.append(f"offset={offset:.1f}: low_conf(SE={se_angle_deg:.1f}°,pts={min_count})")
            continue

        # Skip implausible deviations
        if abs(total_angular_deviation) > 15.0:
            _grad_fail_reasons.append(f"offset={offset:.1f}: implausible({total_angular_deviation:.1f}°)")
            continue

        # Quality score for ranking across offsets
        # Higher is better: more points and lower slope standard error
        inv_se = 1.0 / (se_angle_deg + 0.01)
        quality = inv_se * min_count

        if quality > best_gradient_quality:
            best_gradient_quality = quality
            best_gradient_result = {
                "measured_angle": measured_angle,
                "deviation": round(total_angular_deviation, 2),
                "surface1_angular_dev": round(ang_dev1, 3),
                "surface2_angular_dev": round(ang_dev2, 3),
                "surface1_slope": round(float(slope1), 6) if slope1 is not None else None,
                "surface2_slope": round(float(slope2), 6) if slope2 is not None else None,
                "surface1_r_squared": round(float(r2_1), 4) if r2_1 else 0.0,
                "surface2_r_squared": round(float(r2_2), 4) if r2_2 else 0.0,
                "side1_points": count1 if count1 else 0,
                "side2_points": count2 if count2 else 0,
                "measurement_method": "signed_distance_gradient",
                "measurement_confidence": confidence,
                "success": True,
                "point_count": (count1 or 0) + (count2 or 0),
            }

    if best_gradient_result is not None:
        return best_gradient_result

    detail = "; ".join(_grad_fail_reasons[:4]) if _grad_fail_reasons else "no attempts"
    return _fail(f"Gradient failed: {detail}")


def measure_scan_bend_via_surface_classification(
    scan_points: np.ndarray,
    cad_vertices: np.ndarray,
    cad_triangles: np.ndarray,
    surf1_face_indices: np.ndarray,
    surf2_face_indices: np.ndarray,
    cad_bend_center: np.ndarray,
    cad_surf1_normal: np.ndarray,
    cad_surf2_normal: np.ndarray,
    cad_bend_angle: float = 90.0,
    search_radius: float = 30.0,
    min_offset_mm: float = 5.0,
    cad_bend_direction: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Measure bend angle deviation using the aligned scan overlaid on the CAD.

    Since the scan is already ICP-aligned to the CAD, we leverage the alignment
    rather than independently measuring angles. The approach:

    1. Find scan points near the CAD bend center.
    2. Classify scan points to CAD surfaces via nearest-face lookup.
    3. For each surface's scan points, compute signed distance to the CAD plane.
    4. Fit the scan's actual surface normal from the classified points.
    5. Compute the angle between the two scan-fitted normals.
    6. Angular deviation = measured_angle - cad_angle.

    The key insight: because the scan is aligned to the CAD, we can use CAD
    geometry (face centers, surface planes) to classify which scan points belong
    to which surface. This is much more robust than trying to segment the scan
    independently.

    Args:
        scan_points: Nx3 scan point cloud (already ICP-aligned to CAD)
        cad_vertices: Mx3 CAD mesh vertices
        cad_triangles: Kx3 CAD mesh triangle indices
        surf1_face_indices: Face indices belonging to surface 1
        surf2_face_indices: Face indices belonging to surface 2
        cad_bend_center: 3D center of the CAD bend
        cad_surf1_normal: Expected normal of surface 1 from CAD
        cad_surf2_normal: Expected normal of surface 2 from CAD
        cad_bend_angle: The CAD bend angle in degrees
        search_radius: Radius around bend center to search for scan points (mm)
        min_offset_mm: Minimum distance from bend center to use points (excludes transition zone)

    Returns:
        Dictionary with measured angle, deviation, and quality metrics
    """
    # Step 1: Find scan points near bend center
    scan_tree = KDTree(scan_points)
    nearby_idx = scan_tree.query_ball_point(cad_bend_center, search_radius)
    if len(nearby_idx) < 30:
        return _fail(f"Only {len(nearby_idx)} scan points within {search_radius}mm of bend")

    nearby_points = scan_points[nearby_idx]

    # Step 2: Build LOCAL CAD face center arrays for both surfaces
    # Only use face centers near the bend center — large surfaces like S0 can
    # span the entire part, and distant faces would misclassify scan points.
    face_centers_1_all = cad_vertices[cad_triangles[surf1_face_indices]].mean(axis=1)
    face_centers_2_all = cad_vertices[cad_triangles[surf2_face_indices]].mean(axis=1)

    local_radius = search_radius * 2.0
    dist1 = np.linalg.norm(face_centers_1_all - cad_bend_center, axis=1)
    dist2 = np.linalg.norm(face_centers_2_all - cad_bend_center, axis=1)
    local1_mask = dist1 < local_radius
    local2_mask = dist2 < local_radius

    face_centers_1 = face_centers_1_all[local1_mask]
    face_centers_2 = face_centers_2_all[local2_mask]

    if len(face_centers_1) < 3 or len(face_centers_2) < 3:
        return _fail(f"Not enough local CAD faces near bend (S1: {len(face_centers_1)}, S2: {len(face_centers_2)})")

    # Step 3: Classify scan points via nearest CAD face center
    all_face_centers = np.vstack([face_centers_1, face_centers_2])
    labels = np.array([1] * len(face_centers_1) + [2] * len(face_centers_2))
    face_tree = KDTree(all_face_centers)
    dists_to_face, nearest_face_idx = face_tree.query(nearby_points)
    point_labels = labels[nearest_face_idx]

    # Filter: only use scan points close to a CAD face (reject noise/transition)
    max_face_dist = 5.0
    close_to_face = dists_to_face < max_face_dist

    # Step 4: Try multiple transition zone exclusion radii, pick best
    best_result = None
    best_quality = -1

    # Compute perpendicular distance from bend line (not from bend center)
    # This correctly excludes the transition zone for long bends
    bend_dir = cad_bend_direction
    if bend_dir is None:
        # Fallback: compute from surface normals
        bend_dir = np.cross(cad_surf1_normal, cad_surf2_normal)
        norm = np.linalg.norm(bend_dir)
        if norm > 1e-6:
            bend_dir = bend_dir / norm
        else:
            bend_dir = None

    vecs_from_center = nearby_points - cad_bend_center
    euclidean_dists = np.linalg.norm(vecs_from_center, axis=1)
    if bend_dir is not None:
        # Project out the bend-line component to get perpendicular distance
        along_bend = (vecs_from_center @ bend_dir)[:, None] * bend_dir
        perp_vecs = vecs_from_center - along_bend
        perp_dists = np.linalg.norm(perp_vecs, axis=1)
    else:
        perp_dists = euclidean_dists

    # Try perpendicular distance first, then fall back to Euclidean if needed
    dist_strategies = [("perp", perp_dists)]
    if bend_dir is not None:
        dist_strategies.append(("euclidean", euclidean_dists))

    for _strategy_name, dists_for_offset in dist_strategies:
      for offset in [min_offset_mm, min_offset_mm * 0.5, min_offset_mm * 1.5, min_offset_mm * 2.0]:
        far_enough = dists_for_offset > offset

        surf1_mask = (point_labels == 1) & far_enough & close_to_face
        surf2_mask = (point_labels == 2) & far_enough & close_to_face

        n1_pts = int(surf1_mask.sum())
        n2_pts = int(surf2_mask.sum())

        if n1_pts < 15 or n2_pts < 15:
            continue

        surf1_points = nearby_points[surf1_mask]
        surf2_points = nearby_points[surf2_mask]

        # Step 5: Compute signed distance from each surface's CAD plane
        # Use the CAD surface centroid + normal to define each plane
        cad_centroid1 = face_centers_1.mean(axis=0)
        cad_centroid2 = face_centers_2.mean(axis=0)

        # Signed distances of scan points from their expected CAD planes
        signed_dists1 = np.dot(surf1_points - cad_centroid1, cad_surf1_normal)
        signed_dists2 = np.dot(surf2_points - cad_centroid2, cad_surf2_normal)

        # Remove outlier points (> 2mm from CAD plane — likely misclassified)
        inlier1 = np.abs(signed_dists1) < 2.0
        inlier2 = np.abs(signed_dists2) < 2.0

        if inlier1.sum() < 12 or inlier2.sum() < 12:
            # Fall back to percentile-based outlier removal
            thresh1 = np.percentile(np.abs(signed_dists1), 85)
            thresh2 = np.percentile(np.abs(signed_dists2), 85)
            inlier1 = np.abs(signed_dists1) < max(0.5, thresh1)
            inlier2 = np.abs(signed_dists2) < max(0.5, thresh2)
            if inlier1.sum() < 12 or inlier2.sum() < 12:
                continue

        surf1_clean = surf1_points[inlier1]
        surf2_clean = surf2_points[inlier2]
        n1_pts = len(surf1_clean)
        n2_pts = len(surf2_clean)

        # Step 6: Fit planes to the clean scan points
        normal1, planarity1 = _fit_plane(surf1_clean)
        normal2, planarity2 = _fit_plane(surf2_clean)

        if normal1 is None or normal2 is None:
            continue

        # Orient to match CAD normals
        if np.dot(normal1, cad_surf1_normal) < 0:
            normal1 = -normal1
        if np.dot(normal2, cad_surf2_normal) < 0:
            normal2 = -normal2

        # Check alignment with CAD normals
        align1 = abs(np.dot(normal1, cad_surf1_normal))
        align2 = abs(np.dot(normal2, cad_surf2_normal))

        # Reject if normals deviate too much from CAD (> ~25°)
        if align1 < 0.9 or align2 < 0.9:
            continue

        # Step 7: Compute measured bend angle from scan normals
        # Use abs(dot) to get the acute angle between normals, then disambiguate
        # using the CAD angle to choose the correct supplement.
        dot = np.clip(np.dot(normal1, normal2), -1, 1)
        angle_between_abs = np.degrees(np.arccos(abs(dot)))  # always in [0, 90]
        # Two candidate bend angles: supplement and direct
        candidate1 = 180 - angle_between_abs  # in [90, 180]
        candidate2 = angle_between_abs         # in [0, 90]
        # Pick the one closest to the CAD angle
        if abs(candidate1 - cad_bend_angle) <= abs(candidate2 - cad_bend_angle):
            measured_angle = candidate1
        else:
            measured_angle = candidate2

        # Quality score
        quality = (align1 + align2) / 2 * min(n1_pts, n2_pts) / 50.0

        # Also compute deviation statistics
        mean_dev1 = float(np.mean(signed_dists1[inlier1]))
        mean_dev2 = float(np.mean(signed_dists2[inlier2]))

        if quality > best_quality:
            best_quality = quality
            # Confidence based on point count and CAD alignment
            avg_align = (align1 + align2) / 2
            min_pts = min(n1_pts, n2_pts)
            if min_pts >= 100 and avg_align >= 0.98:
                sc_confidence = "high"
            elif min_pts >= 30 and avg_align >= 0.95:
                sc_confidence = "medium"
            else:
                sc_confidence = "low"
            best_result = {
                "measured_angle": round(float(measured_angle), 2),
                "angle_between_normals": round(float(180 - measured_angle), 2),
                "scan_normal1": normal1.tolist(),
                "scan_normal2": normal2.tolist(),
                "side1_points": n1_pts,
                "side2_points": n2_pts,
                "planarity1": round(float(planarity1), 4),
                "planarity2": round(float(planarity2), 4),
                "cad_alignment1": round(float(align1), 3),
                "cad_alignment2": round(float(align2), 3),
                "mean_dev_surf1_mm": round(mean_dev1, 3),
                "mean_dev_surf2_mm": round(mean_dev2, 3),
                "min_offset_used": round(offset, 1),
                "measurement_method": "surface_classification",
                "measurement_confidence": sc_confidence,
                "success": True,
                "point_count": n1_pts + n2_pts,
            }

    if best_result is None:
        return _fail("Could not classify enough scan points to both surfaces")

    return best_result


def measure_scan_bend_at_cad_location(
    scan_points: np.ndarray,
    cad_bend_center: np.ndarray,
    cad_surf1_normal: np.ndarray,
    cad_surf2_normal: np.ndarray,
    cad_bend_direction: np.ndarray,
    search_radius: float = 30.0,
    probe_offset: float = 15.0,
    probe_radius: float = 10.0,
    cad_bend_angle: float = 90.0,
) -> Dict[str, Any]:
    """
    Fallback: Measure bend angle using probe-region approach (CMM-like).

    Used when CAD mesh surface data is not available for surface classification.

    Strategy:
    1. From the CAD bend center, compute two "probe locations" - one on each
       surface, offset along each surface's tangent direction away from the bend.
    2. Find scan points within a sphere around each probe location.
    3. Fit a plane (via PCA) to each set of probe points.
    4. Compute the angle between the two fitted planes.

    Args:
        scan_points: Nx3 scan point cloud
        cad_bend_center: 3D center of the CAD bend
        cad_surf1_normal: Normal of first CAD surface
        cad_surf2_normal: Normal of second CAD surface
        cad_bend_direction: Direction of the CAD bend line
        search_radius: Overall search radius (mm)
        probe_offset: Distance from bend center to probe location (mm)
        probe_radius: Radius of probe sphere to collect points (mm)

    Returns:
        Dictionary with measured angle, deviation, and quality metrics
    """
    tree = KDTree(scan_points)

    # Compute tangent directions for each surface (direction away from bend line
    # along each surface's plane)
    tang1 = np.cross(cad_bend_direction, cad_surf1_normal)
    tang1_norm = np.linalg.norm(tang1)
    if tang1_norm < 0.01:
        return _fail("Surface 1 tangent computation failed")
    tang1 = tang1 / tang1_norm

    tang2 = np.cross(cad_bend_direction, cad_surf2_normal)
    tang2_norm = np.linalg.norm(tang2)
    if tang2_norm < 0.01:
        return _fail("Surface 2 tangent computation failed")
    tang2 = tang2 / tang2_norm

    # Ensure tangents point away from each other (outward from the bend)
    if np.dot(tang1, tang2) > 0:
        if np.dot(tang1, cad_surf2_normal) < 0:
            tang1 = -tang1
        if np.dot(tang2, cad_surf1_normal) < 0:
            tang2 = -tang2

    if np.dot(tang1, tang2) > 0.5:
        tang2 = -tang2

    # Try multiple offsets to find good probe locations
    best_result = None
    best_quality = -1

    for offset_mult in [1.0, 1.5, 2.0, 0.7]:
        offset = probe_offset * offset_mult
        probe1_center = cad_bend_center + tang1 * offset
        probe2_center = cad_bend_center + tang2 * offset

        probe1_idx = tree.query_ball_point(probe1_center, probe_radius)
        probe2_idx = tree.query_ball_point(probe2_center, probe_radius)

        if len(probe1_idx) < 15 or len(probe2_idx) < 15:
            continue

        probe1_points = scan_points[probe1_idx]
        probe2_points = scan_points[probe2_idx]

        normal1, planarity1 = _fit_plane(probe1_points)
        normal2, planarity2 = _fit_plane(probe2_points)

        if normal1 is None or normal2 is None:
            continue

        if np.dot(normal1, cad_surf1_normal) < 0:
            normal1 = -normal1
        if np.dot(normal2, cad_surf2_normal) < 0:
            normal2 = -normal2

        align1 = abs(np.dot(normal1, cad_surf1_normal))
        align2 = abs(np.dot(normal2, cad_surf2_normal))

        alt_align1 = abs(np.dot(normal1, cad_surf2_normal))
        alt_align2 = abs(np.dot(normal2, cad_surf1_normal))
        if (alt_align1 + alt_align2) > (align1 + align2):
            normal1, normal2 = normal2, normal1
            planarity1, planarity2 = planarity2, planarity1
            align1 = alt_align1
            align2 = alt_align2
            if np.dot(normal1, cad_surf1_normal) < 0:
                normal1 = -normal1
            if np.dot(normal2, cad_surf2_normal) < 0:
                normal2 = -normal2

        quality = (align1 + align2) / 2 * (1 - planarity1) * (1 - planarity2)

        if quality > best_quality:
            best_quality = quality

            dot = np.clip(np.dot(normal1, normal2), -1, 1)
            angle_between_abs = np.degrees(np.arccos(abs(dot)))
            candidate1 = 180 - angle_between_abs
            candidate2 = angle_between_abs
            if abs(candidate1 - cad_bend_angle) <= abs(candidate2 - cad_bend_angle):
                measured_angle = candidate1
            else:
                measured_angle = candidate2
            angle_between = angle_between_abs

            best_result = {
                "measured_angle": round(float(measured_angle), 2),
                "angle_between_normals": round(float(angle_between), 2),
                "scan_normal1": normal1.tolist(),
                "scan_normal2": normal2.tolist(),
                "side1_points": len(probe1_idx),
                "side2_points": len(probe2_idx),
                "planarity1": round(float(planarity1), 4),
                "planarity2": round(float(planarity2), 4),
                "cad_alignment1": round(float(align1), 3),
                "cad_alignment2": round(float(align2), 3),
                "probe_offset_used": round(offset, 1),
                "measurement_method": "probe_region",
                "measurement_confidence": "low",
                "success": True,
                "point_count": len(probe1_idx) + len(probe2_idx),
            }

    if best_result is None:
        return _fail("Could not find sufficient probe points at any offset")

    return best_result


def _fit_plane(points: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Fit a plane to points using PCA. Returns (normal, planarity_ratio)."""
    try:
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue direction
        total = eigenvalues.sum()
        planarity = eigenvalues[0] / total if total > 0 else 1.0
        return normal, planarity
    except Exception:
        return None, 1.0


def _fail(error: str) -> Dict[str, Any]:
    """Return a failure result."""
    return {
        "measured_angle": None,
        "success": False,
        "error": error,
        "point_count": 0,
    }


def measure_all_scan_bends(
    scan_points: np.ndarray,
    cad_bends: List[Dict[str, Any]],
    search_radius: float = 30.0,
    cad_vertices: np.ndarray = None,
    cad_triangles: np.ndarray = None,
    surface_face_map: Dict[int, np.ndarray] = None,
    deviations: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Measure all bend angles in the scan using CAD bend locations as guide.

    Strategy: Run both primary methods, pick the best result.
    1. Signed distance gradient (requires deviations array from compute_deviations).
       Measures angular deviation directly from the signed distance field gradient.
    2. Surface classification (requires cad_vertices, cad_triangles, surface_face_map).
       Classifies scan points to CAD surfaces via nearest-face lookup, fits planes.
    Best candidate is selected by confidence level, preferring gradient at same confidence.
    3. Probe-region method (CMM-like tangent probes). Fallback when both methods fail.

    Args:
        scan_points: Nx3 scan point cloud (ICP-aligned to CAD)
        cad_bends: List of CAD bend dictionaries (from CADBendExtractor)
        search_radius: Search radius around each bend center (mm)
        cad_vertices: Optional Mx3 CAD mesh vertices (enables surface classification)
        cad_triangles: Optional Kx3 CAD mesh triangle indices
        surface_face_map: Optional dict mapping surface_id -> face indices array
        deviations: Optional N-length signed distance array (enables gradient analysis)

    Returns:
        List of measurement results for each bend
    """
    has_mesh_data = (
        cad_vertices is not None
        and cad_triangles is not None
        and surface_face_map is not None
    )
    has_deviations = deviations is not None and len(deviations) == len(scan_points)

    results = []

    for cad_bend in cad_bends:
        bend_center = np.array(cad_bend["bend_center"])
        surf1_normal = np.array(cad_bend["surface1_normal"])
        surf2_normal = np.array(cad_bend["surface2_normal"])
        bend_direction = np.array(cad_bend["bend_direction"])
        cad_angle = cad_bend["bend_angle"]
        surf1_id = cad_bend["surface1_id"]
        surf2_id = cad_bend["surface2_id"]

        measurement = None
        candidates = []

        # Method 1: Signed distance gradient (measures deviation from distance field)
        gradient_result = None
        if has_deviations:
            gradient_result = measure_scan_bend_via_signed_distance_gradient(
                scan_points=scan_points,
                deviations=deviations,
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                cad_bend_angle=cad_angle,
                search_radius=search_radius,
            )
            if gradient_result and gradient_result.get("success"):
                candidates.append(gradient_result)
            elif gradient_result and not gradient_result.get("success"):
                logger.info(f"  {cad_bend['bend_name']}: gradient failed: {gradient_result.get('error', '?')}")

        # Method 2: Surface classification (requires mesh data)
        sc_result = None
        if has_mesh_data and surf1_id in surface_face_map and surf2_id in surface_face_map:
            sc_result = measure_scan_bend_via_surface_classification(
                scan_points=scan_points,
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                surf1_face_indices=surface_face_map[surf1_id],
                surf2_face_indices=surface_face_map[surf2_id],
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_angle=cad_angle,
                search_radius=search_radius,
                cad_bend_direction=bend_direction,
            )
            if sc_result and sc_result.get("success"):
                candidates.append(sc_result)
            elif sc_result and not sc_result.get("success"):
                logger.info(f"  {cad_bend['bend_name']}: surface_classification failed: {sc_result.get('error', '?')}")

        # Choose best candidate: prefer high-confidence gradient, then high-confidence SC
        if candidates:
            _conf_rank = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
            # Sort: prefer gradient method at same confidence, prefer higher confidence
            candidates.sort(key=lambda c: (
                _conf_rank.get(c.get("measurement_confidence", "low"), 0),
                1 if c.get("measurement_method") == "signed_distance_gradient" else 0,
            ), reverse=True)
            measurement = candidates[0]

        # Fallback: probe-region method (only if both methods failed)
        if measurement is None:
            measurement = measure_scan_bend_at_cad_location(
                scan_points=scan_points,
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                search_radius=search_radius,
                cad_bend_angle=cad_angle,
            )

        result = {
            "bend_id": cad_bend["bend_id"],
            "bend_name": cad_bend["bend_name"],
            "cad_angle": round(cad_angle, 2),
            "bend_center": [round(x, 2) for x in bend_center.tolist()],
        }

        if measurement["success"]:
            measured = measurement["measured_angle"]
            # For gradient method, deviation is already computed directly
            if "deviation" in measurement and measurement["deviation"] is not None:
                deviation = measurement["deviation"]
            else:
                deviation = round(measured - cad_angle, 2)
            # Sanity check: flag implausible deviations (>20° from CAD)
            if abs(deviation) > 20:
                logger.warning(f"  {cad_bend['bend_name']}: implausible deviation {deviation:.1f}° "
                             f"(CAD={cad_angle:.1f}°, measured={measured:.1f}°, "
                             f"method={measurement.get('measurement_method')})")
                measurement["measurement_confidence"] = "very_low"
            result.update({
                "measured_angle": measured,
                "deviation": deviation,
                "abs_deviation": round(abs(deviation), 2),
                "side1_points": measurement.get("side1_points", 0),
                "side2_points": measurement.get("side2_points", 0),
                "measurement_method": measurement.get("measurement_method", "unknown"),
                "measurement_confidence": measurement.get("measurement_confidence", "unknown"),
                "success": True,
            })
            # Include gradient-specific fields if available
            if measurement.get("measurement_method") == "signed_distance_gradient":
                result["surface1_angular_dev"] = measurement.get("surface1_angular_dev", 0)
                result["surface2_angular_dev"] = measurement.get("surface2_angular_dev", 0)
                result["surface1_r_squared"] = measurement.get("surface1_r_squared", 0)
                result["surface2_r_squared"] = measurement.get("surface2_r_squared", 0)
            # Include classification-specific fields if available
            if measurement.get("scan_normal1"):
                result["scan_normal1"] = measurement["scan_normal1"]
                result["scan_normal2"] = measurement["scan_normal2"]
                result["cad_alignment1"] = measurement.get("cad_alignment1", 0)
                result["cad_alignment2"] = measurement.get("cad_alignment2", 0)
        else:
            result.update({
                "measured_angle": None,
                "deviation": None,
                "abs_deviation": None,
                "success": False,
                "error": measurement.get("error", "Unknown error"),
            })

        results.append(result)

    return results
