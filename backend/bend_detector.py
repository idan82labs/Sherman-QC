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


def _point_to_segment_distances(
    points: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return point-to-segment distances and normalized segment parameters."""
    pts = np.asarray(points, dtype=np.float64)
    start = np.asarray(seg_start, dtype=np.float64)
    end = np.asarray(seg_end, dtype=np.float64)
    seg = end - start
    seg_len_sq = float(np.dot(seg, seg))
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if seg_len_sq <= 1e-12:
        return np.linalg.norm(pts - start[None, :], axis=1), np.zeros((len(pts),), dtype=np.float64)
    rel = pts - start[None, :]
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    closest = start[None, :] + t[:, None] * seg[None, :]
    return np.linalg.norm(pts - closest, axis=1), t


def _locality_mask_along_bend(
    points: np.ndarray,
    cad_bend_center: np.ndarray,
    radius: float,
    *,
    cad_bend_line_start: Optional[np.ndarray] = None,
    cad_bend_line_end: Optional[np.ndarray] = None,
    longitudinal_margin_mm: float = 0.0,
) -> np.ndarray:
    """Select points near a bend line segment, falling back to bend-center sphere."""
    pts = np.asarray(points, dtype=np.float64)
    center = np.asarray(cad_bend_center, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    if cad_bend_line_start is None or cad_bend_line_end is None:
        return np.linalg.norm(pts - center[None, :], axis=1) <= radius

    start = np.asarray(cad_bend_line_start, dtype=np.float64)
    end = np.asarray(cad_bend_line_end, dtype=np.float64)
    seg = end - start
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1e-6:
        return np.linalg.norm(pts - center[None, :], axis=1) <= radius

    expanded_radius = float(np.hypot(radius, (0.5 * seg_len) + max(0.0, longitudinal_margin_mm)))
    pre_mask = np.linalg.norm(pts - center[None, :], axis=1) <= expanded_radius
    if not np.any(pre_mask):
        return pre_mask

    pre_pts = pts[pre_mask]
    dists, t = _point_to_segment_distances(pre_pts, start, end)
    along_mm = t * seg_len
    local_mask = (
        (dists <= radius)
        & (along_mm >= -max(0.0, longitudinal_margin_mm))
        & (along_mm <= seg_len + max(0.0, longitudinal_margin_mm))
    )
    mask = np.zeros((len(pts),), dtype=bool)
    mask[np.flatnonzero(pre_mask)] = local_mask
    return mask


def _station_centers_along_bend(
    cad_bend_center: np.ndarray,
    *,
    cad_bend_line_start: Optional[np.ndarray] = None,
    cad_bend_line_end: Optional[np.ndarray] = None,
    minimum_span_mm: float = 18.0,
) -> List[np.ndarray]:
    """Generate stable section/probe stations along a bend line."""
    center = np.asarray(cad_bend_center, dtype=np.float64)
    stations = [center]
    if cad_bend_line_start is None or cad_bend_line_end is None:
        return stations

    start = np.asarray(cad_bend_line_start, dtype=np.float64)
    end = np.asarray(cad_bend_line_end, dtype=np.float64)
    seg = end - start
    seg_len = float(np.linalg.norm(seg))
    if seg_len < minimum_span_mm:
        return stations

    seen = {tuple(np.round(center, 4))}
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        station = start + t * seg
        key = tuple(np.round(station, 4))
        if key in seen:
            continue
        seen.add(key)
        stations.append(station)
    return stations


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
    cad_bend_line_start: np.ndarray = None,
    cad_bend_line_end: np.ndarray = None,
    allow_line_locality: bool = True,
    allow_face_recovery: bool = False,
    all_face_centers: Optional[np.ndarray] = None,
    all_face_normals: Optional[np.ndarray] = None,
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
    adaptive = _adaptive_shallow_bend_settings(
        cad_bend_angle,
        search_radius=search_radius,
        min_offset_mm=min_offset_mm,
        bend_line_length=(
            float(
                np.linalg.norm(
                    np.asarray(cad_bend_line_end, dtype=np.float64)
                    - np.asarray(cad_bend_line_start, dtype=np.float64)
                )
            )
            if cad_bend_line_start is not None and cad_bend_line_end is not None
            else None
        ),
    )
    search_radius = adaptive["search_radius"]
    min_side_points = int(adaptive["min_side_points"])
    min_inlier_points = int(adaptive["min_inlier_points"])
    max_face_dist = float(adaptive["max_face_dist"])
    alignment_floor = float(adaptive["alignment_floor"])
    offset_candidates = adaptive["surface_offset_candidates"]

    # Step 1: Find scan points near bend center
    if allow_line_locality:
        nearby_mask = _locality_mask_along_bend(
            scan_points,
            cad_bend_center,
            search_radius,
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            longitudinal_margin_mm=max(search_radius * 0.25, 4.0),
        )
    else:
        center = np.asarray(cad_bend_center, dtype=np.float64)
        nearby_mask = np.linalg.norm(np.asarray(scan_points, dtype=np.float64) - center[None, :], axis=1) <= search_radius
    nearby_idx = np.flatnonzero(nearby_mask).tolist()
    if len(nearby_idx) < 30:
        return _fail(f"Only {len(nearby_idx)} scan points within {search_radius}mm of bend")

    nearby_points = scan_points[nearby_idx]

    # Step 2: Build LOCAL CAD face center arrays for both surfaces
    # Only use face centers near the bend center — large surfaces like S0 can
    # span the entire part, and distant faces would misclassify scan points.
    face_centers_1_all = cad_vertices[cad_triangles[surf1_face_indices]].mean(axis=1)
    face_centers_2_all = cad_vertices[cad_triangles[surf2_face_indices]].mean(axis=1)

    local_radius = search_radius * 2.0
    if allow_line_locality:
        local1_mask = _locality_mask_along_bend(
            face_centers_1_all,
            cad_bend_center,
            local_radius,
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            longitudinal_margin_mm=max(local_radius * 0.5, 6.0),
        )
        local2_mask = _locality_mask_along_bend(
            face_centers_2_all,
            cad_bend_center,
            local_radius,
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            longitudinal_margin_mm=max(local_radius * 0.5, 6.0),
        )
    else:
        center = np.asarray(cad_bend_center, dtype=np.float64)
        local1_mask = np.linalg.norm(face_centers_1_all - center[None, :], axis=1) <= local_radius
        local2_mask = np.linalg.norm(face_centers_2_all - center[None, :], axis=1) <= local_radius

    face_centers_1 = face_centers_1_all[local1_mask]
    face_centers_2 = face_centers_2_all[local2_mask]

    if allow_face_recovery and (len(face_centers_1) < 3 or len(face_centers_2) < 3):
        if all_face_centers is None or all_face_normals is None:
            all_face_centers, all_face_normals = _compute_triangle_centers_and_normals(cad_vertices, cad_triangles)
        if len(face_centers_1) < 3:
            recovered_1 = _recover_local_surface_face_centers(
                all_face_centers=all_face_centers,
                all_face_normals=all_face_normals,
                cad_bend_center=np.asarray(cad_bend_center, dtype=np.float64),
                cad_surface_normal=np.asarray(cad_surf1_normal, dtype=np.float64),
                search_radius=float(search_radius),
                cad_bend_line_start=np.asarray(cad_bend_line_start, dtype=np.float64) if cad_bend_line_start is not None else None,
                cad_bend_line_end=np.asarray(cad_bend_line_end, dtype=np.float64) if cad_bend_line_end is not None else None,
            )
            if len(recovered_1) >= 3:
                face_centers_1 = recovered_1
        if len(face_centers_2) < 3:
            recovered_2 = _recover_local_surface_face_centers(
                all_face_centers=all_face_centers,
                all_face_normals=all_face_normals,
                cad_bend_center=np.asarray(cad_bend_center, dtype=np.float64),
                cad_surface_normal=np.asarray(cad_surf2_normal, dtype=np.float64),
                search_radius=float(search_radius),
                cad_bend_line_start=np.asarray(cad_bend_line_start, dtype=np.float64) if cad_bend_line_start is not None else None,
                cad_bend_line_end=np.asarray(cad_bend_line_end, dtype=np.float64) if cad_bend_line_end is not None else None,
            )
            if len(recovered_2) >= 3:
                face_centers_2 = recovered_2

    if len(face_centers_1) < 3 or len(face_centers_2) < 3:
        return _fail(f"Not enough local CAD faces near bend (S1: {len(face_centers_1)}, S2: {len(face_centers_2)})")

    # Step 3: Classify scan points via nearest CAD face center
    all_face_centers = np.vstack([face_centers_1, face_centers_2])
    labels = np.array([1] * len(face_centers_1) + [2] * len(face_centers_2))
    face_tree = KDTree(all_face_centers)
    dists_to_face, nearest_face_idx = face_tree.query(nearby_points)
    point_labels = labels[nearest_face_idx]

    # Filter: only use scan points close to a CAD face (reject noise/transition)
    close_to_face = dists_to_face < max_face_dist

    # Step 4: Try multiple transition zone exclusion radii, pick best
    best_result = None
    best_quality = -1
    candidate_results: List[Dict[str, Any]] = []
    best_partial = {
        "side1_points": 0,
        "side2_points": 0,
        "point_count": 0,
        "probe_offset_used": None,
        "measurement_method": "probe_region",
    }
    best_partial = {
        "nearby_points": len(nearby_idx),
        "side1_points": 0,
        "side2_points": 0,
        "inlier_side1_points": 0,
        "inlier_side2_points": 0,
        "min_offset_used": None,
        "max_face_dist": max_face_dist,
        "measurement_method": "surface_classification",
    }

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
      for offset in offset_candidates:
        far_enough = dists_for_offset > offset

        surf1_mask = (point_labels == 1) & far_enough & close_to_face
        surf2_mask = (point_labels == 2) & far_enough & close_to_face

        n1_pts = int(surf1_mask.sum())
        n2_pts = int(surf2_mask.sum())

        best_partial["side1_points"] = max(int(best_partial["side1_points"]), n1_pts)
        best_partial["side2_points"] = max(int(best_partial["side2_points"]), n2_pts)
        if n1_pts < min_side_points or n2_pts < min_side_points:
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

        if inlier1.sum() < min_inlier_points or inlier2.sum() < min_inlier_points:
            # Fall back to percentile-based outlier removal
            thresh1 = np.percentile(np.abs(signed_dists1), 85)
            thresh2 = np.percentile(np.abs(signed_dists2), 85)
            inlier1 = np.abs(signed_dists1) < max(0.5, thresh1)
            inlier2 = np.abs(signed_dists2) < max(0.5, thresh2)
            best_partial["inlier_side1_points"] = max(int(best_partial["inlier_side1_points"]), int(inlier1.sum()))
            best_partial["inlier_side2_points"] = max(int(best_partial["inlier_side2_points"]), int(inlier2.sum()))
            if inlier1.sum() < min_inlier_points or inlier2.sum() < min_inlier_points:
                continue

        surf1_clean = surf1_points[inlier1]
        surf2_clean = surf2_points[inlier2]
        n1_pts = len(surf1_clean)
        n2_pts = len(surf2_clean)
        best_partial["inlier_side1_points"] = max(int(best_partial["inlier_side1_points"]), n1_pts)
        best_partial["inlier_side2_points"] = max(int(best_partial["inlier_side2_points"]), n2_pts)

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
        if align1 < alignment_floor or align2 < alignment_floor:
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

        best_quality = max(best_quality, float(quality))
        best_partial["min_offset_used"] = round(offset, 1)
        # Confidence based on point count and CAD alignment
        avg_align = (align1 + align2) / 2
        min_pts = min(n1_pts, n2_pts)
        if min_pts >= 100 and avg_align >= 0.98:
            sc_confidence = "high"
        elif min_pts >= 30 and avg_align >= 0.95:
            sc_confidence = "medium"
        else:
            sc_confidence = "low"
        candidate_results.append(
            {
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
                "_quality_score": float(quality),
                "_angle_gap": abs(float(measured_angle) - float(cad_bend_angle)),
                "_min_side_points": int(min_pts),
            }
        )

    if not candidate_results:
        return _fail(
            "Could not classify enough scan points to both surfaces",
            measurement_method="surface_classification",
            point_count=int(best_partial["nearby_points"]),
            nearby_points=int(best_partial["nearby_points"]),
            side1_points=int(best_partial["side1_points"]),
            side2_points=int(best_partial["side2_points"]),
            inlier_side1_points=int(best_partial["inlier_side1_points"]),
            inlier_side2_points=int(best_partial["inlier_side2_points"]),
            min_offset_used=best_partial["min_offset_used"],
            max_face_dist=float(best_partial["max_face_dist"]),
        )

    bend_line_length = None
    if cad_bend_line_start is not None and cad_bend_line_end is not None:
        bend_line_length = float(
            np.linalg.norm(
                np.asarray(cad_bend_line_end, dtype=np.float64)
                - np.asarray(cad_bend_line_start, dtype=np.float64)
            )
        )
    best_result = _choose_surface_classification_candidate(
        candidate_results,
        best_quality,
        cad_bend_angle=float(cad_bend_angle),
        bend_line_length=bend_line_length,
    )
    best_result.pop("_quality_score", None)
    best_result.pop("_angle_gap", None)
    best_result.pop("_min_side_points", None)

    return best_result


def _choose_surface_classification_candidate(
    candidate_results: List[Dict[str, Any]],
    best_quality: float,
    *,
    cad_bend_angle: Optional[float] = None,
    bend_line_length: Optional[float] = None,
) -> Dict[str, Any]:
    quality_floor = max(float(best_quality) * 0.94, float(best_quality) - 0.12)
    is_short_obtuse = (
        cad_bend_angle is not None
        and 110.0 <= float(cad_bend_angle) <= 130.0
        and bend_line_length is not None
        and float(bend_line_length) <= 60.0
    )
    if is_short_obtuse:
        has_close_supported_candidate = any(
            float(result.get("_angle_gap") or 0.0) <= 1.0
            and int(result.get("_min_side_points") or 0) >= 80
            and int(result.get("point_count") or 0) >= 300
            for result in candidate_results
        )
        if has_close_supported_candidate:
            quality_floor = min(
                quality_floor,
                max(float(best_quality) * 0.25, float(best_quality) - 5.5),
            )
    eligible_results = [
        result
        for result in candidate_results
        if float(result.get("_quality_score") or 0.0) >= quality_floor
    ]
    if not eligible_results:
        eligible_results = list(candidate_results)
    eligible_results.sort(
        key=lambda result: (
            float(result.get("_angle_gap") or 0.0),
            -float(result.get("_quality_score") or 0.0),
            -int(result.get("_min_side_points") or 0),
        )
    )
    return dict(eligible_results[0])


def measure_scan_bend_via_profile_section(
    scan_points: np.ndarray,
    cad_bend_center: np.ndarray,
    cad_surf1_normal: np.ndarray,
    cad_surf2_normal: np.ndarray,
    cad_bend_direction: np.ndarray,
    cad_bend_angle: float = 90.0,
    search_radius: float = 30.0,
    section_half_width: float = 8.0,
    cad_bend_line_start: np.ndarray = None,
    cad_bend_line_end: np.ndarray = None,
    allow_station_retries: bool = True,
) -> Dict[str, Any]:
    """
    Measure a bend from a thin cross-section slice around the bend line.

    This is more stable for short flange bends than large probe spheres because
    it uses the entire local profile instead of two discrete probe locations.
    """
    bend_line_length = None
    if cad_bend_line_start is not None and cad_bend_line_end is not None:
        bend_line_length = float(
            np.linalg.norm(
                np.asarray(cad_bend_line_end, dtype=np.float64)
                - np.asarray(cad_bend_line_start, dtype=np.float64)
            )
        )
    adaptive = _adaptive_shallow_bend_settings(
        cad_bend_angle,
        search_radius=search_radius,
        probe_offset=6.0,
        probe_radius=10.0,
        bend_line_length=bend_line_length,
    )
    tangent = np.asarray(cad_bend_direction, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-9:
        return _fail("Bend direction unavailable", measurement_method="profile_section")
    tangent = tangent / tangent_norm

    n1 = np.asarray(cad_surf1_normal, dtype=np.float64)
    n2 = np.asarray(cad_surf2_normal, dtype=np.float64)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)

    line1 = np.cross(tangent, n1)
    line2 = np.cross(tangent, n2)
    if np.linalg.norm(line1) < 1e-9 or np.linalg.norm(line2) < 1e-9:
        return _fail("Could not derive section tangents", measurement_method="profile_section")
    line1 = line1 / (np.linalg.norm(line1) + 1e-12)
    line2 = line2 / (np.linalg.norm(line2) + 1e-12)
    if float(np.dot(line1, line2)) > 0.0:
        line2 = -line2

    axis_u = line1
    axis_v = np.cross(tangent, axis_u)
    axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)

    line1_2d = np.array([1.0, 0.0], dtype=np.float64)
    line2_3d_proj = np.array([float(np.dot(line2, axis_u)), float(np.dot(line2, axis_v))], dtype=np.float64)
    if np.linalg.norm(line2_3d_proj) < 1e-9:
        return _fail("Could not project second line into section", measurement_method="profile_section")
    line2_2d = line2_3d_proj / (np.linalg.norm(line2_3d_proj) + 1e-12)
    if float(np.dot(line1_2d, line2_2d)) > 0.0:
        line2_2d = -line2_2d

    offset_min = 2.5 if adaptive["shallow"] else 4.0
    offset_max = max(16.0, adaptive["search_radius"] * 0.75)
    band_tol = 2.0 if adaptive["shallow"] else 1.8
    min_side_points = max(6, adaptive["min_side_points"])

    def _measure_at_station(station_center: np.ndarray) -> Dict[str, Any]:
        diffs = np.asarray(scan_points, dtype=np.float64) - np.asarray(station_center, dtype=np.float64)
        longitudinal = diffs @ tangent
        perp_vecs = diffs - longitudinal[:, None] * tangent[None, :]
        radial = np.linalg.norm(perp_vecs, axis=1)
        in_slab = (np.abs(longitudinal) <= max(section_half_width, 4.0)) & (radial <= adaptive["search_radius"])
        section_points = scan_points[in_slab]
        if len(section_points) < 30:
            return _fail(
                f"Only {len(section_points)} points in section slice",
                measurement_method="profile_section",
                point_count=int(len(section_points)),
            )

        section_diffs = np.asarray(section_points, dtype=np.float64) - np.asarray(station_center, dtype=np.float64)
        qx = section_diffs @ axis_u
        qy = section_diffs @ axis_v
        q = np.column_stack([qx, qy])

        pts1 = _select_profile_strip_points(
            q,
            line1_2d,
            offset_min=offset_min,
            offset_max=offset_max,
            band_tol=band_tol,
        )
        pts2 = _select_profile_strip_points(
            q,
            line2_2d,
            offset_min=offset_min,
            offset_max=offset_max,
            band_tol=band_tol,
        )

        fit_kind = "line_line"
        fallback1 = None
        fallback2 = None
        if len(pts1) < min_side_points and len(pts2) >= min_side_points:
            fallback1 = _attempt_line_arc_fallback(
                q,
                line1_2d,
                offset_min=offset_min,
                offset_max=offset_max,
                band_tol=band_tol,
                min_side_points=min_side_points,
            )
            if fallback1 is not None:
                pts1 = fallback1["points"]
                fit_kind = "line_arc_fallback"
        if len(pts2) < min_side_points and len(pts1) >= min_side_points:
            fallback2 = _attempt_line_arc_fallback(
                q,
                line2_2d,
                offset_min=offset_min,
                offset_max=offset_max,
                band_tol=band_tol,
                min_side_points=min_side_points,
            )
            if fallback2 is not None:
                pts2 = fallback2["points"]
                fit_kind = "line_arc_fallback"
        if len(pts1) < min_side_points or len(pts2) < min_side_points:
            return _fail(
                "Could not isolate enough profile points on both bend sides",
                measurement_method="profile_section",
                point_count=int(len(section_points)),
                side1_points=int(len(pts1)),
                side2_points=int(len(pts2)),
            )

        if fallback1 is not None:
            fit1, residual1 = fallback1["fit"], float(fallback1["residual"])
        else:
            fit1, residual1 = _fit_line_2d(pts1)
        if fallback2 is not None:
            fit2, residual2 = fallback2["fit"], float(fallback2["residual"])
        else:
            fit2, residual2 = _fit_line_2d(pts2)
        if fit1 is None or fit2 is None:
            return _fail("Profile line fit failed", measurement_method="profile_section")

        if float(np.dot(fit1, line1_2d)) < 0.0:
            fit1 = -fit1
        if float(np.dot(fit2, line2_2d)) < 0.0:
            fit2 = -fit2

        dot = np.clip(abs(float(np.dot(fit1, fit2))), -1.0, 1.0)
        angle_between_abs = float(np.degrees(np.arccos(dot)))
        candidate1 = 180.0 - angle_between_abs
        candidate2 = angle_between_abs
        measured_angle = candidate1 if abs(candidate1 - cad_bend_angle) <= abs(candidate2 - cad_bend_angle) else candidate2

        fit1_3d = fit1[0] * axis_u + fit1[1] * axis_v
        fit2_3d = fit2[0] * axis_u + fit2[1] * axis_v
        scan_n1 = np.cross(tangent, fit1_3d)
        scan_n2 = np.cross(tangent, fit2_3d)
        scan_n1 = scan_n1 / (np.linalg.norm(scan_n1) + 1e-12)
        scan_n2 = scan_n2 / (np.linalg.norm(scan_n2) + 1e-12)
        if float(np.dot(scan_n1, n1)) < 0.0:
            scan_n1 = -scan_n1
        if float(np.dot(scan_n2, n2)) < 0.0:
            scan_n2 = -scan_n2
        align1 = abs(float(np.dot(scan_n1, n1)))
        align2 = abs(float(np.dot(scan_n2, n2)))

        residual = max(residual1, residual2)
        min_pts = min(len(pts1), len(pts2))
        side_balance_score = float(min(len(pts1), len(pts2)) / max(len(pts1), len(pts2)))
        if fit_kind == "line_arc_fallback":
            if min_pts >= 10 and residual <= 0.9 and min(align1, align2) >= 0.82:
                confidence = "medium"
            else:
                confidence = "low"
        elif min_pts >= 20 and residual <= 0.6 and min(align1, align2) >= 0.9:
            confidence = "high"
        elif min_pts >= 10 and residual <= 1.0 and min(align1, align2) >= 0.82:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "measured_angle": round(float(measured_angle), 2),
            "angle_between_normals": round(float(180.0 - measured_angle), 2),
            "scan_normal1": scan_n1.tolist(),
            "scan_normal2": scan_n2.tolist(),
            "side1_points": int(len(pts1)),
            "side2_points": int(len(pts2)),
            "planarity1": round(float(residual1), 4),
            "planarity2": round(float(residual2), 4),
            "cad_alignment1": round(float(align1), 3),
            "cad_alignment2": round(float(align2), 3),
            "measurement_method": "profile_section",
            "measurement_confidence": confidence,
            "success": True,
            "point_count": int(len(section_points)),
            "section_half_width": float(max(section_half_width, 4.0)),
            "slice_fit_kind": fit_kind,
            "side_balance_score": round(float(side_balance_score), 4),
            "line_residual_max": round(float(residual), 4),
            "arc_support_points": int((fallback1 or fallback2 or {}).get("arc_support_points") or 0),
            "station_center": np.asarray(station_center, dtype=np.float64).tolist(),
        }

    best_success = None
    best_failure = None
    station_centers = (
        _station_centers_along_bend(
            np.asarray(cad_bend_center, dtype=np.float64),
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            minimum_span_mm=max(18.0, 2.0 * max(section_half_width, 4.0)),
        )
        if allow_station_retries
        else [np.asarray(cad_bend_center, dtype=np.float64)]
    )

    for station_center in station_centers:
        result = _measure_at_station(station_center)
        if result.get("success"):
            score = (
                {"high": 3, "medium": 2, "low": 1}.get(result.get("measurement_confidence"), 0),
                float(result.get("side_balance_score") or 0.0),
                int(min(result.get("side1_points") or 0, result.get("side2_points") or 0)),
                -float(result.get("line_residual_max") or 0.0),
            )
            if best_success is None or score > best_success[0]:
                best_success = (score, result)
        else:
            score = (
                int(min(result.get("side1_points") or 0, result.get("side2_points") or 0)),
                int(result.get("point_count") or 0),
            )
            if best_failure is None or score > best_failure[0]:
                best_failure = (score, result)

    if best_success is not None:
        return best_success[1]
    if best_failure is not None:
        return best_failure[1]
    return _fail("Profile section yielded no usable stations", measurement_method="profile_section")


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
    cad_bend_line_start: np.ndarray = None,
    cad_bend_line_end: np.ndarray = None,
    allow_station_retries: bool = True,
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
    adaptive = _adaptive_shallow_bend_settings(
        cad_bend_angle,
        search_radius=search_radius,
        probe_offset=probe_offset,
        probe_radius=probe_radius,
        bend_line_length=(
            float(
                np.linalg.norm(
                    np.asarray(cad_bend_line_end, dtype=np.float64)
                    - np.asarray(cad_bend_line_start, dtype=np.float64)
                )
            )
            if cad_bend_line_start is not None and cad_bend_line_end is not None
            else None
        ),
    )
    search_radius = adaptive["search_radius"]
    probe_radius = adaptive["probe_radius"]
    min_side_points = int(adaptive["min_side_points"])
    offset_candidates = adaptive["offset_candidates"]

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
    best_partial = {
        "side1_points": 0,
        "side2_points": 0,
        "point_count": 0,
        "probe_offset_used": None,
        "measurement_method": "probe_region",
    }

    station_centers = (
        _station_centers_along_bend(
            np.asarray(cad_bend_center, dtype=np.float64),
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            minimum_span_mm=max(18.0, probe_radius * 2.0),
        )
        if allow_station_retries
        else [np.asarray(cad_bend_center, dtype=np.float64)]
    )

    for station_center in station_centers:
      for offset in offset_candidates:
        probe1_center = station_center + tang1 * offset
        probe2_center = station_center + tang2 * offset

        probe1_idx = tree.query_ball_point(probe1_center, probe_radius)
        probe2_idx = tree.query_ball_point(probe2_center, probe_radius)

        best_partial["side1_points"] = max(int(best_partial["side1_points"]), len(probe1_idx))
        best_partial["side2_points"] = max(int(best_partial["side2_points"]), len(probe2_idx))
        best_partial["point_count"] = max(int(best_partial["point_count"]), len(probe1_idx) + len(probe2_idx))
        best_partial["probe_offset_used"] = round(offset, 1)
        if len(probe1_idx) < min_side_points or len(probe2_idx) < min_side_points:
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
                "station_center": np.asarray(station_center, dtype=np.float64).tolist(),
            }

    if best_result is None:
        return _fail(
            "Could not find sufficient probe points at any offset",
            measurement_method="probe_region",
            point_count=int(best_partial["point_count"]),
            side1_points=int(best_partial["side1_points"]),
            side2_points=int(best_partial["side2_points"]),
            probe_offset_used=best_partial["probe_offset_used"],
        )

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


def _fail(error: str, **extra: Any) -> Dict[str, Any]:
    """Return a failure result."""
    result = {
        "measured_angle": None,
        "success": False,
        "error": error,
        "point_count": 0,
        "measurement_confidence": "very_low",
    }
    result.update(extra)
    return result


def _adaptive_shallow_bend_settings(
    cad_bend_angle: float,
    *,
    search_radius: float,
    probe_offset: float = 15.0,
    probe_radius: float = 10.0,
    min_offset_mm: float = 5.0,
    bend_line_length: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Adapt local measurement settings for shallow bends.

    Small flange returns on rolled-shell parts do not behave like 90° folds:
    large probe offsets overshoot the short flange leg, and fixed point-count
    thresholds make the local measurement path classify them as unobserved.
    """
    bend_angle = abs(float(cad_bend_angle))
    shallow = bend_angle <= 35.0
    if not shallow:
        return {
            "shallow": False,
            "search_radius": float(search_radius),
            "probe_radius": float(probe_radius),
            "offset_candidates": [
                float(probe_offset),
                float(probe_offset * 1.5),
                float(probe_offset * 2.0),
                float(probe_offset * 0.7),
            ],
            "surface_offset_candidates": [
                float(min_offset_mm),
                float(min_offset_mm * 0.5),
                float(min_offset_mm * 1.5),
                float(min_offset_mm * 2.0),
            ],
            "min_side_points": 15,
            "min_inlier_points": 12,
            "max_face_dist": 5.0,
            "alignment_floor": 0.9,
        }

    tuned_probe_radius = max(float(probe_radius), 14.0)
    tuned_search_radius = max(float(search_radius), 40.0)
    return {
        "shallow": True,
        "search_radius": tuned_search_radius,
        "probe_radius": tuned_probe_radius,
        "offset_candidates": [3.0, 5.0, 7.5, 10.0, 12.0, float(probe_offset)],
        "surface_offset_candidates": [2.5, 4.0, 6.0, 8.0, 10.0],
        "min_side_points": 6,
        "min_inlier_points": 5,
        "max_face_dist": 8.0,
        "alignment_floor": 0.82,
    }


def _trusted_parent_frame_alignment(
    alignment_metadata: Optional[Dict[str, Any]],
    *,
    min_confidence: float = 0.55,
) -> bool:
    if not isinstance(alignment_metadata, dict):
        return False
    if bool(alignment_metadata.get("alignment_abstained")):
        return False
    if str(alignment_metadata.get("alignment_source") or "").upper() != "PLANE_DATUM_REFINEMENT":
        return False
    try:
        confidence = float(alignment_metadata.get("alignment_confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return confidence >= float(min_confidence)


def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = sorted(
        (float(v), max(0.0, float(w)))
        for v, w in zip(values, weights)
        if np.isfinite(v) and np.isfinite(w) and float(w) > 0.0
    )
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    threshold = total * 0.5
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _measurement_confidence_value(label: Optional[str]) -> float:
    mapping = {
        "high": 0.9,
        "medium": 0.72,
        "low": 0.5,
        "very_low": 0.25,
    }
    return float(mapping.get(str(label or "").strip().lower(), 0.4))


def _weighted_effective_count(weights: List[float]) -> float:
    finite = [max(0.0, float(w)) for w in weights if np.isfinite(w) and float(w) > 0.0]
    if not finite:
        return 0.0
    numerator = float(sum(finite)) ** 2
    denominator = float(sum(w * w for w in finite))
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _is_selected_short_obtuse_bend(cad_bend: Dict[str, Any]) -> bool:
    try:
        bend_angle = abs(float(cad_bend.get("bend_angle") or 0.0))
    except Exception:
        return False
    if not (105.0 <= bend_angle <= 140.0):
        return False
    if not bool(cad_bend.get("selected_baseline_direct")):
        return False
    try:
        start = np.asarray(cad_bend.get("bend_line_start"), dtype=np.float64)
        end = np.asarray(cad_bend.get("bend_line_end"), dtype=np.float64)
    except Exception:
        return False
    if start.shape != (3,) or end.shape != (3,):
        return False
    return float(np.linalg.norm(end - start)) <= 70.0


def _prefer_surface_measurement_result(
    primary: Optional[Dict[str, Any]],
    alternate: Optional[Dict[str, Any]],
    *,
    cad_angle: float,
) -> Optional[Dict[str, Any]]:
    if not alternate:
        return primary
    if not primary:
        return alternate
    if bool(alternate.get("success")) and not bool(primary.get("success")):
        return alternate
    if bool(primary.get("success")) and not bool(alternate.get("success")):
        return primary
    if not bool(primary.get("success")) and not bool(alternate.get("success")):
        return primary
    primary_gap = abs(float(primary.get("measured_angle") or 0.0) - float(cad_angle))
    alternate_gap = abs(float(alternate.get("measured_angle") or 0.0) - float(cad_angle))
    if alternate_gap + 0.5 < primary_gap:
        return alternate
    return primary


def _median_abs_deviation(values: List[float], center: float) -> float:
    finite = [abs(float(v) - float(center)) for v in values if np.isfinite(v)]
    if not finite:
        return 0.0
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _fit_line_2d(points_2d: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
    if points_2d is None or len(points_2d) < 2:
        return None, float("inf")
    weights_arr: Optional[np.ndarray] = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights_arr) != len(points_2d):
            weights_arr = None
        else:
            finite = np.isfinite(weights_arr) & (weights_arr > 0.0)
            if not np.any(finite):
                weights_arr = None
            else:
                points_2d = points_2d[finite]
                weights_arr = weights_arr[finite]
    if len(points_2d) < 2:
        return None, float("inf")
    if weights_arr is None:
        center = points_2d.mean(axis=0)
        centered = points_2d - center
    else:
        norm = float(np.sum(weights_arr))
        if norm <= 1e-12:
            center = points_2d.mean(axis=0)
            centered = points_2d - center
            weights_arr = None
        else:
            weights_arr = weights_arr / norm
            center = np.sum(points_2d * weights_arr[:, None], axis=0)
            centered = points_2d - center
    try:
        fit_input = centered if weights_arr is None else centered * np.sqrt(weights_arr)[:, None]
        _, _, vh = np.linalg.svd(fit_input, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, float("inf")
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    residuals = np.abs(centered @ normal)
    residual = float(np.mean(residuals)) if weights_arr is None else float(np.sum(residuals * weights_arr))
    return direction / (np.linalg.norm(direction) + 1e-12), residual


def _compute_triangle_centers_and_normals(
    cad_vertices: np.ndarray,
    cad_triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tri_points = np.asarray(cad_vertices, dtype=np.float64)[np.asarray(cad_triangles, dtype=np.int32)]
    centers = tri_points.mean(axis=1)
    normals = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    normals[valid] = normals[valid] / norms[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return centers, normals


def _recover_local_surface_face_centers(
    *,
    all_face_centers: np.ndarray,
    all_face_normals: np.ndarray,
    cad_bend_center: np.ndarray,
    cad_surface_normal: np.ndarray,
    search_radius: float,
    cad_bend_line_start: Optional[np.ndarray] = None,
    cad_bend_line_end: Optional[np.ndarray] = None,
    min_faces: int = 3,
) -> np.ndarray:
    target_normal = np.asarray(cad_surface_normal, dtype=np.float64)
    target_norm = float(np.linalg.norm(target_normal))
    if target_norm < 1e-9:
        return np.empty((0, 3), dtype=np.float64)
    target_normal = target_normal / target_norm
    centers = np.asarray(all_face_centers, dtype=np.float64)
    normals = np.asarray(all_face_normals, dtype=np.float64)
    local_radius = max(float(search_radius) * 2.0, 12.0)

    def _candidate_mask(radius: float, normal_floor: float) -> np.ndarray:
        locality = _locality_mask_along_bend(
            centers,
            cad_bend_center,
            radius,
            cad_bend_line_start=cad_bend_line_start,
            cad_bend_line_end=cad_bend_line_end,
            longitudinal_margin_mm=max(radius * 0.5, 6.0),
        )
        alignment = np.abs(normals @ target_normal)
        return locality & (alignment >= normal_floor)

    for radius, normal_floor in (
        (local_radius, 0.94),
        (local_radius, 0.88),
        (local_radius * 1.5, 0.88),
        (local_radius * 2.0, 0.82),
    ):
        mask = _candidate_mask(radius, normal_floor)
        if int(mask.sum()) >= int(min_faces):
            return centers[mask]

    return np.empty((0, 3), dtype=np.float64)


def _select_profile_strip_points(
    q: np.ndarray,
    line_dir: np.ndarray,
    *,
    offset_min: float,
    offset_max: float,
    band_tol: float,
) -> np.ndarray:
    line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    line_norm = np.array([-line_dir[1], line_dir[0]], dtype=np.float64)
    along = q @ line_dir
    perp = np.abs(q @ line_norm)
    pos_mask = (along >= offset_min) & (along <= offset_max) & (perp <= band_tol)
    neg_mask = (along <= -offset_min) & (along >= -offset_max) & (perp <= band_tol)
    if int(pos_mask.sum()) >= int(neg_mask.sum()):
        return q[pos_mask]
    return q[neg_mask]


def _attempt_line_arc_fallback(
    q: np.ndarray,
    expected_dir: np.ndarray,
    *,
    offset_min: float,
    offset_max: float,
    band_tol: float,
    min_side_points: int,
) -> Optional[Dict[str, Any]]:
    relaxed_strip = _select_profile_strip_points(
        q,
        expected_dir,
        offset_min=max(1.0, offset_min * 0.45),
        offset_max=offset_max,
        band_tol=band_tol * 1.7,
    )
    min_relaxed = max(4, min_side_points - 2)
    if len(relaxed_strip) < min_relaxed:
        return None

    expected_dir = expected_dir / (np.linalg.norm(expected_dir) + 1e-12)
    expected_norm = np.array([-expected_dir[1], expected_dir[0]], dtype=np.float64)
    along = q @ expected_dir
    perp = np.abs(q @ expected_norm)
    arc_mask = (
        (np.abs(along) <= max(offset_min * 1.5, 4.0))
        & (perp <= band_tol * 2.2)
        & (np.linalg.norm(q, axis=1) <= max(offset_min * 2.5, 6.0))
    )
    arc_points = q[arc_mask]
    fit_input = relaxed_strip if len(arc_points) < 4 else np.vstack([relaxed_strip, arc_points])
    fit, residual = _fit_line_2d(fit_input)
    if fit is None:
        return None
    if float(np.dot(fit, expected_dir)) < 0.0:
        fit = -fit
    expected_alignment = abs(float(np.dot(fit, expected_dir)))
    if expected_alignment < 0.72:
        return None
    return {
        "points": fit_input,
        "fit": fit,
        "residual": residual,
        "arc_support_points": int(len(arc_points)),
        "strip_points": int(len(relaxed_strip)),
        "expected_alignment": expected_alignment,
    }


def _longest_true_run(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    start: Optional[int] = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    if start is not None:
        end = len(mask) - 1
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _estimate_visible_support_interval(
    scan_points: np.ndarray,
    *,
    line_start: np.ndarray,
    tangent: np.ndarray,
    line_length: float,
    search_radius: float,
    slice_spacing: float,
    min_valid_slices: int,
) -> Dict[str, Any]:
    diffs = np.asarray(scan_points, dtype=np.float64) - np.asarray(line_start, dtype=np.float64)
    longitudinal = diffs @ tangent
    perp_vecs = diffs - longitudinal[:, None] * tangent[None, :]
    radial = np.linalg.norm(perp_vecs, axis=1)
    in_tube = (longitudinal >= 0.0) & (longitudinal <= line_length) & (radial <= search_radius)
    support_positions = longitudinal[in_tube]
    if len(support_positions) < max(24, min_valid_slices * 6):
        return {
            "success": False,
            "error": f"Only {len(support_positions)} points support the parent-frame bend tube",
            "support_point_count": int(len(support_positions)),
            "visible_span_mm": 0.0,
            "visible_span_ratio": 0.0,
        }

    bin_size = max(3.0, slice_spacing * 0.85)
    bin_count = max(min_valid_slices * 2, min(24, int(np.ceil(line_length / bin_size))))
    hist, edges = np.histogram(support_positions, bins=bin_count, range=(0.0, line_length))
    positive = hist[hist > 0]
    threshold = max(6, int(np.ceil(float(np.median(positive)) * 0.5))) if len(positive) else 6
    support_mask = hist >= threshold
    relaxed_threshold = max(4, int(np.ceil(threshold * 0.6)))
    for idx in range(1, len(support_mask) - 1):
        if not support_mask[idx] and hist[idx] >= relaxed_threshold and support_mask[idx - 1] and support_mask[idx + 1]:
            support_mask[idx] = True

    run = _longest_true_run(support_mask)
    if run is None:
        return {
            "success": False,
            "error": "No contiguous visible support span along the bend line",
            "support_point_count": int(len(support_positions)),
            "visible_span_mm": 0.0,
            "visible_span_ratio": 0.0,
        }

    start_idx, end_idx = run
    visible_start = float(edges[start_idx])
    visible_end = float(edges[end_idx + 1])
    visible_span = max(0.0, visible_end - visible_start)
    visible_ratio = visible_span / max(line_length, 1e-9)
    min_required_span = max(slice_spacing * max(min_valid_slices - 1, 3) * 0.5, 10.0)
    if visible_span < min_required_span:
        return {
            "success": False,
            "error": (
                f"Visible bend support span {visible_span:.1f}mm is too short for "
                f"{min_valid_slices} section slices"
            ),
            "support_point_count": int(len(support_positions)),
            "visible_span_mm": round(float(visible_span), 4),
            "visible_span_ratio": round(float(visible_ratio), 4),
            "supported_span_start_mm": round(float(visible_start), 4),
            "supported_span_end_mm": round(float(visible_end), 4),
            "slice_support_bins": hist.tolist(),
        }

    planned_slice_count = max(
        min_valid_slices,
        min(11, int(np.floor(visible_span / max(slice_spacing, 3.0))) + 1),
    )
    margin = min(slice_spacing * 0.4, visible_span * 0.15)
    if visible_span <= margin * 2.0:
        margin = 0.0
    slice_positions = np.linspace(
        visible_start + margin,
        visible_end - margin,
        planned_slice_count,
    )
    return {
        "success": True,
        "support_point_count": int(len(support_positions)),
        "visible_span_mm": round(float(visible_span), 4),
        "visible_span_ratio": round(float(visible_ratio), 4),
        "supported_span_start_mm": round(float(visible_start), 4),
        "supported_span_end_mm": round(float(visible_end), 4),
        "slice_support_bins": hist.tolist(),
        "planned_slice_count": int(planned_slice_count),
        "slice_positions_mm": [round(float(v), 4) for v in slice_positions.tolist()],
    }


def _parent_frame_slice_validity(
    slice_result: Dict[str, Any],
    *,
    provisional_angle: Optional[float] = None,
    provisional_mad: Optional[float] = None,
) -> Tuple[bool, str, float]:
    if not slice_result.get("success") or slice_result.get("measured_angle") is None:
        return False, "measurement_failed", 0.0
    side1 = int(slice_result.get("side1_points") or 0)
    side2 = int(slice_result.get("side2_points") or 0)
    if min(side1, side2) < 4:
        return False, "too_few_side_points", 0.0
    balance = float(slice_result.get("side_balance_score") or 0.0)
    max_residual = float(
        slice_result.get("line_residual_max")
        or max(float(slice_result.get("planarity1") or 0.0), float(slice_result.get("planarity2") or 0.0))
    )
    alignments = [
        float(value)
        for value in (slice_result.get("cad_alignment1"), slice_result.get("cad_alignment2"))
        if isinstance(value, (int, float))
    ]
    min_alignment = min(alignments) if alignments else 0.0
    fit_kind = str(slice_result.get("slice_fit_kind") or "line_line").strip().lower()
    arc_support = int(slice_result.get("arc_support_points") or 0)

    if fit_kind == "line_arc_fallback":
        if arc_support < 4:
            return False, "insufficient_arc_support", 0.0
        if balance < 0.10:
            return False, "unbalanced_line_arc_slice", 0.0
        if max_residual > 1.5 or min_alignment < 0.72:
            return False, "weak_line_arc_fit", 0.0
    else:
        if balance < 0.18:
            return False, "unbalanced_slice", 0.0
        if max_residual > 1.25 or min_alignment < 0.78:
            return False, "weak_line_fit", 0.0

    measured_angle = float(slice_result["measured_angle"])
    if provisional_angle is not None:
        tol = max(8.0, 2.0 * float(provisional_mad or 0.0) + 4.0)
        if abs(measured_angle - float(provisional_angle)) > tol:
            return False, "angle_outlier", 0.0

    confidence = _measurement_confidence_value(slice_result.get("measurement_confidence"))
    weight = max(1.0, float(min(side1, side2))) * max(confidence, 0.25) / (1.0 + max_residual)
    if fit_kind == "line_arc_fallback":
        weight *= 0.85
    return True, "valid", weight


def measure_scan_bend_via_parent_frame_profile_section(
    scan_points: np.ndarray,
    cad_bend_line_start: np.ndarray,
    cad_bend_line_end: np.ndarray,
    cad_surf1_normal: np.ndarray,
    cad_surf2_normal: np.ndarray,
    cad_bend_direction: np.ndarray,
    *,
    cad_bend_angle: float = 90.0,
    search_radius: float = 30.0,
    section_half_width: float = 8.0,
    alignment_metadata: Optional[Dict[str, Any]] = None,
    min_valid_slices: int = 5,
) -> Dict[str, Any]:
    measurement_method = "parent_frame_profile_section"
    if not _trusted_parent_frame_alignment(alignment_metadata):
        return _fail(
            "Parent-frame alignment is not trusted for local section metrology",
            measurement_method=measurement_method,
            alignment_source=(alignment_metadata or {}).get("alignment_source"),
            alignment_confidence=(alignment_metadata or {}).get("alignment_confidence"),
        )

    line_start = np.asarray(cad_bend_line_start, dtype=np.float64)
    line_end = np.asarray(cad_bend_line_end, dtype=np.float64)
    line_vec = line_end - line_start
    line_length = float(np.linalg.norm(line_vec))
    if line_length < 1e-9:
        return _fail("Bend line unavailable for parent-frame section metrology", measurement_method=measurement_method)

    tangent = np.asarray(cad_bend_direction, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-9:
        tangent = line_vec / line_length
    else:
        tangent = tangent / tangent_norm

    slice_half_width = max(4.0, min(float(section_half_width), 6.0))
    slice_spacing = max(slice_half_width * 1.1, 3.5)
    support = _estimate_visible_support_interval(
        scan_points,
        line_start=line_start,
        tangent=tangent,
        line_length=line_length,
        search_radius=max(float(search_radius), 10.0),
        slice_spacing=slice_spacing,
        min_valid_slices=min_valid_slices,
    )
    if not support.get("success"):
        return _fail(
            str(support.get("error") or "Visible bend support span unavailable"),
            measurement_method=measurement_method,
            parent_frame_trusted=True,
            support_point_count=int(support.get("support_point_count") or 0),
            visible_span_mm=float(support.get("visible_span_mm") or 0.0),
            visible_span_ratio=float(support.get("visible_span_ratio") or 0.0),
            supported_span_start_mm=support.get("supported_span_start_mm"),
            supported_span_end_mm=support.get("supported_span_end_mm"),
            slice_support_bins=support.get("slice_support_bins"),
            section_method="visible_span_multi_slice_weighted_median_v2",
        )
    planned_slice_count = int(support["planned_slice_count"])
    slice_positions_mm = [float(v) for v in support["slice_positions_mm"]]
    fractions = np.asarray([pos / max(line_length, 1e-9) for pos in slice_positions_mm], dtype=np.float64)
    required_valid_slices = int(min_valid_slices)
    if planned_slice_count <= min_valid_slices and float(support["visible_span_ratio"]) >= 0.5:
        required_valid_slices = max(4, int(min_valid_slices) - 1)

    slice_results: List[Dict[str, Any]] = []
    total_points = 0

    for index, fraction in enumerate(fractions):
        center = line_start + float(fraction) * line_vec
        slice_result = measure_scan_bend_via_profile_section(
            scan_points=scan_points,
            cad_bend_center=center,
            cad_surf1_normal=cad_surf1_normal,
            cad_surf2_normal=cad_surf2_normal,
            cad_bend_direction=tangent,
            cad_bend_angle=cad_bend_angle,
            search_radius=search_radius,
            section_half_width=slice_half_width,
        )
        slice_result = dict(slice_result)
        slice_result["slice_index"] = int(index)
        slice_result["slice_fraction"] = round(float(fraction), 4)
        slice_result["slice_position_mm"] = round(float(slice_positions_mm[index]), 4)
        slice_result["slice_center"] = [round(float(x), 4) for x in center.tolist()]
        slice_results.append(slice_result)
        total_points += int(slice_result.get("point_count") or 0)

    base_angles = [
        float(result["measured_angle"])
        for result in slice_results
        if result.get("success") and result.get("measured_angle") is not None
    ]
    provisional_angle = float(np.median(np.asarray(base_angles, dtype=np.float64))) if base_angles else None
    provisional_mad = _median_abs_deviation(base_angles, provisional_angle) if base_angles and provisional_angle is not None else None

    slice_angles: List[float] = []
    weights: List[float] = []
    total_side1 = 0
    total_side2 = 0
    for slice_result in slice_results:
        valid, reason, weight = _parent_frame_slice_validity(
            slice_result,
            provisional_angle=provisional_angle,
            provisional_mad=provisional_mad,
        )
        slice_result["slice_valid"] = bool(valid)
        slice_result["slice_invalid_reason"] = reason
        slice_result["slice_weight"] = round(float(weight), 4)
        if not valid:
            continue
        angle = float(slice_result["measured_angle"])
        slice_angles.append(angle)
        weights.append(weight)
        total_side1 += int(slice_result.get("side1_points") or 0)
        total_side2 += int(slice_result.get("side2_points") or 0)

    if len(slice_angles) < required_valid_slices:
        return _fail(
            f"Only {len(slice_angles)} valid section slices",
            measurement_method=measurement_method,
            slice_count=int(len(slice_angles)),
            planned_slice_count=int(planned_slice_count),
            slice_angle_samples_deg=[round(float(v), 4) for v in slice_angles],
            visible_span_mm=float(support["visible_span_mm"]),
            visible_span_ratio=float(support["visible_span_ratio"]),
            supported_span_start_mm=support["supported_span_start_mm"],
            supported_span_end_mm=support["supported_span_end_mm"],
            slice_support_bins=support["slice_support_bins"],
            section_method="visible_span_multi_slice_weighted_median_v2",
            parent_frame_trusted=True,
            expected_zone_visible=bool(
                float(support["visible_span_ratio"]) >= 0.4
                and len(slice_angles) >= max(3, int(np.ceil(0.5 * planned_slice_count)))
            ),
            slice_results=slice_results,
        )

    aggregate_angle = _weighted_median(slice_angles, weights)
    if aggregate_angle is None:
        return _fail(
            "Could not aggregate valid section slices",
            measurement_method=measurement_method,
            slice_count=int(len(slice_angles)),
            planned_slice_count=int(planned_slice_count),
            visible_span_mm=float(support["visible_span_mm"]),
            visible_span_ratio=float(support["visible_span_ratio"]),
            supported_span_start_mm=support["supported_span_start_mm"],
            supported_span_end_mm=support["supported_span_end_mm"],
            slice_support_bins=support["slice_support_bins"],
            section_method="visible_span_multi_slice_weighted_median_v2",
            parent_frame_trusted=True,
            slice_results=slice_results,
        )

    angle_dispersion = _median_abs_deviation(slice_angles, aggregate_angle)
    effective_slice_count = _weighted_effective_count(weights)
    confidence_label = "low"
    if effective_slice_count >= 6.0 and angle_dispersion <= 2.0:
        confidence_label = "high"
    elif effective_slice_count >= 4.5 and angle_dispersion <= 4.0:
        confidence_label = "medium"

    alignment_values = [
        float(value)
        for result in slice_results
        for value in (result.get("cad_alignment1"), result.get("cad_alignment2"))
        if result.get("success") and isinstance(value, (int, float))
    ]
    min_alignment = min(alignment_values) if alignment_values else None

    return {
        "measured_angle": round(float(aggregate_angle), 2),
        "aggregate_angle_deg": round(float(aggregate_angle), 4),
        "aggregate_radius_mm": None,
        "angle_dispersion_deg": round(float(angle_dispersion), 4),
        "slice_count": int(len(slice_angles)),
        "planned_slice_count": int(planned_slice_count),
        "effective_slice_count": round(float(effective_slice_count), 4),
        "slice_angle_samples_deg": [round(float(v), 4) for v in slice_angles],
        "slice_results": slice_results,
        "section_method": "visible_span_multi_slice_weighted_median_v2",
        "measurement_method": measurement_method,
        "measurement_confidence": confidence_label,
        "success": bool(
            effective_slice_count >= max(3.5, float(required_valid_slices) - 0.5)
            and angle_dispersion <= 6.0
            and float(support["visible_span_ratio"]) >= 0.35
        ),
        "point_count": int(total_points),
        "side1_points": int(total_side1),
        "side2_points": int(total_side2),
        "cad_alignment1": round(float(np.median(alignment_values[0::2])), 3) if len(alignment_values) >= 2 else min_alignment,
        "cad_alignment2": round(float(np.median(alignment_values[1::2])), 3) if len(alignment_values) >= 2 else min_alignment,
        "parent_frame_trusted": True,
        "expected_zone_visible": bool(
            float(support["visible_span_ratio"]) >= 0.4
            and len(slice_angles) >= max(3, int(np.ceil(0.5 * planned_slice_count)))
        ),
        "section_half_width": float(slice_half_width),
        "visible_span_mm": float(support["visible_span_mm"]),
        "visible_span_ratio": float(support["visible_span_ratio"]),
        "supported_span_start_mm": support["supported_span_start_mm"],
        "supported_span_end_mm": support["supported_span_end_mm"],
        "slice_support_bins": support["slice_support_bins"],
    }


def measure_all_scan_bends(
    scan_points: np.ndarray,
    cad_bends: List[Dict[str, Any]],
    search_radius: float = 30.0,
    cad_vertices: np.ndarray = None,
    cad_triangles: np.ndarray = None,
    surface_face_map: Dict[int, np.ndarray] = None,
    deviations: np.ndarray = None,
    alignment_metadata: Optional[Dict[str, Any]] = None,
    scan_state: Optional[str] = None,
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
    all_face_centers = None
    all_face_normals = None
    if has_mesh_data:
        all_face_centers, all_face_normals = _compute_triangle_centers_and_normals(cad_vertices, cad_triangles)
    has_deviations = deviations is not None and len(deviations) == len(scan_points)
    use_parent_frame_sections = (
        str(scan_state or "").strip().lower() == "partial"
        and _trusted_parent_frame_alignment(alignment_metadata)
    )
    enable_generic_station_retries = str(scan_state or "").strip().lower() != "partial"
    enable_generic_line_locality = str(scan_state or "").strip().lower() != "partial"

    results = []

    for cad_bend in cad_bends:
        bend_center = np.array(cad_bend["bend_center"])
        surf1_normal = np.array(cad_bend["surface1_normal"])
        surf2_normal = np.array(cad_bend["surface2_normal"])
        bend_direction = np.array(cad_bend["bend_direction"])
        cad_angle = cad_bend["bend_angle"]
        surf1_id = cad_bend["surface1_id"]
        surf2_id = cad_bend["surface2_id"]
        feature_type = str(cad_bend.get("feature_type") or "DISCRETE_BEND").upper()
        countable_in_regression = bool(cad_bend.get("countable_in_regression", True))

        measurement = None
        candidates = []

        # Method 1: Signed distance gradient (measures deviation from distance field)
        gradient_result = None
        adaptive = _adaptive_shallow_bend_settings(
            cad_angle,
            search_radius=search_radius,
            probe_offset=15.0,
            probe_radius=10.0,
            min_offset_mm=5.0,
            bend_line_length=(
                float(
                    np.linalg.norm(
                        np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64)
                        - np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64)
                    )
                )
                if cad_bend.get("bend_line_start") is not None and cad_bend.get("bend_line_end") is not None
                else None
            ),
        )
        if has_deviations:
            gradient_result = measure_scan_bend_via_signed_distance_gradient(
                scan_points=scan_points,
                deviations=deviations,
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                cad_bend_angle=cad_angle,
                search_radius=adaptive["search_radius"],
            )
            if gradient_result and gradient_result.get("success"):
                candidates.append(gradient_result)
            elif gradient_result and not gradient_result.get("success"):
                logger.info(f"  {cad_bend['bend_name']}: gradient failed: {gradient_result.get('error', '?')}")

        parent_frame_result = None
        if use_parent_frame_sections and countable_in_regression and feature_type == "DISCRETE_BEND":
            line_start = np.asarray(
                cad_bend.get("bend_line_start", cad_bend.get("bend_line_point1", bend_center)),
                dtype=np.float64,
            )
            line_end = np.asarray(
                cad_bend.get("bend_line_end", cad_bend.get("bend_line_point2", bend_center)),
                dtype=np.float64,
            )
            section_half_width = max(5.0, min(10.0, 0.12 * float(np.linalg.norm(line_end - line_start))))
            parent_frame_result = measure_scan_bend_via_parent_frame_profile_section(
                scan_points=scan_points,
                cad_bend_line_start=line_start,
                cad_bend_line_end=line_end,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                cad_bend_angle=cad_angle,
                search_radius=adaptive["search_radius"],
                section_half_width=section_half_width,
                alignment_metadata=alignment_metadata,
            )
            if parent_frame_result and parent_frame_result.get("success"):
                candidates.append(parent_frame_result)
            elif parent_frame_result and not parent_frame_result.get("success"):
                logger.info(
                    f"  {cad_bend['bend_name']}: parent_frame_profile_section failed: "
                    f"{parent_frame_result.get('error', '?')}"
                )

        # Method 2: Surface classification (requires mesh data)
        sc_result = None
        has_explicit_surface_ids = has_mesh_data and surf1_id in surface_face_map and surf2_id in surface_face_map
        allow_recovered_surface_classification = bool(cad_bend.get("selected_baseline_direct")) and has_mesh_data
        if has_mesh_data and (has_explicit_surface_ids or allow_recovered_surface_classification):
            sc_result = measure_scan_bend_via_surface_classification(
                scan_points=scan_points,
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                surf1_face_indices=(
                    surface_face_map[surf1_id]
                    if has_explicit_surface_ids
                    else np.empty((0,), dtype=np.int32)
                ),
                surf2_face_indices=(
                    surface_face_map[surf2_id]
                    if has_explicit_surface_ids
                    else np.empty((0,), dtype=np.int32)
                ),
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_angle=cad_angle,
                search_radius=adaptive["search_radius"],
                min_offset_mm=adaptive["surface_offset_candidates"][0],
                cad_bend_direction=bend_direction,
                cad_bend_line_start=np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64),
                cad_bend_line_end=np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64),
                allow_line_locality=enable_generic_line_locality,
                allow_face_recovery=bool(cad_bend.get("selected_baseline_direct")),
                all_face_centers=all_face_centers,
                all_face_normals=all_face_normals,
            )
            if has_explicit_surface_ids and _is_selected_short_obtuse_bend(cad_bend):
                recovered_sc_result = measure_scan_bend_via_surface_classification(
                    scan_points=scan_points,
                    cad_vertices=cad_vertices,
                    cad_triangles=cad_triangles,
                    surf1_face_indices=np.empty((0,), dtype=np.int32),
                    surf2_face_indices=np.empty((0,), dtype=np.int32),
                    cad_bend_center=bend_center,
                    cad_surf1_normal=surf1_normal,
                    cad_surf2_normal=surf2_normal,
                    cad_bend_angle=cad_angle,
                    search_radius=adaptive["search_radius"],
                    min_offset_mm=adaptive["surface_offset_candidates"][0],
                    cad_bend_direction=bend_direction,
                    cad_bend_line_start=np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64),
                    cad_bend_line_end=np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64),
                    allow_line_locality=enable_generic_line_locality,
                    allow_face_recovery=True,
                    all_face_centers=all_face_centers,
                    all_face_normals=all_face_normals,
                )
                sc_result = _prefer_surface_measurement_result(
                    sc_result,
                    recovered_sc_result,
                    cad_angle=float(cad_angle),
                )
            if (
                _is_selected_short_obtuse_bend(cad_bend)
                and sc_result
                and sc_result.get("success")
                and sc_result.get("measured_angle") is not None
            ):
                sc_gap = abs(float(sc_result["measured_angle"]) - float(cad_angle))
                sc_side_balance = sc_result.get("side_balance_score")
                if sc_side_balance is None:
                    side1_count = int(sc_result.get("side1_points") or 0)
                    side2_count = int(sc_result.get("side2_points") or 0)
                    if side1_count > 0 and side2_count > 0:
                        sc_side_balance = min(side1_count, side2_count) / max(side1_count, side2_count)
                sc_side_balance = float(sc_side_balance if sc_side_balance is not None else 1.0)
                should_retry_short_obtuse = (
                    sc_gap > 4.0
                    or (
                        sc_gap > 1.5
                        and sc_side_balance < 0.35
                    )
                    or (
                        sc_gap > 2.0
                        and sc_side_balance < 0.5
                    )
                )
            else:
                should_retry_short_obtuse = False
            if should_retry_short_obtuse:
                narrow_search_candidates = []
                for radius in (
                    min(float(adaptive["search_radius"]), 30.0),
                    min(float(adaptive["search_radius"]), 26.0),
                    22.0 if sc_gap > 2.0 and sc_side_balance < 0.5 else None,
                    18.0 if sc_gap > 2.0 and sc_side_balance < 0.5 else None,
                    14.0 if sc_gap > 2.0 and sc_side_balance < 0.5 else None,
                ):
                    if radius is None:
                        continue
                    if radius >= 14.0 and not any(abs(radius - existing) < 1e-6 for existing in narrow_search_candidates):
                        narrow_search_candidates.append(radius)
                narrow_min_offsets = []
                for offset in (
                    float(adaptive["surface_offset_candidates"][0]),
                    2.5,
                    5.0,
                    7.5,
                ):
                    if offset >= 2.5 and not any(abs(offset - existing) < 1e-6 for existing in narrow_min_offsets):
                        narrow_min_offsets.append(offset)
                for narrow_radius in narrow_search_candidates:
                    for narrow_min_offset in narrow_min_offsets:
                        for narrow_line_locality in (True, False):
                            narrow_sc_result = measure_scan_bend_via_surface_classification(
                                scan_points=scan_points,
                                cad_vertices=cad_vertices,
                                cad_triangles=cad_triangles,
                                surf1_face_indices=np.empty((0,), dtype=np.int32),
                                surf2_face_indices=np.empty((0,), dtype=np.int32),
                                cad_bend_center=bend_center,
                                cad_surf1_normal=surf1_normal,
                                cad_surf2_normal=surf2_normal,
                                cad_bend_angle=cad_angle,
                                search_radius=float(narrow_radius),
                                min_offset_mm=float(narrow_min_offset),
                                cad_bend_direction=bend_direction,
                                cad_bend_line_start=np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64),
                                cad_bend_line_end=np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64),
                                allow_line_locality=narrow_line_locality,
                                allow_face_recovery=True,
                                all_face_centers=all_face_centers,
                                all_face_normals=all_face_normals,
                            )
                            sc_result = _prefer_surface_measurement_result(
                                sc_result,
                                narrow_sc_result,
                                cad_angle=float(cad_angle),
                            )
            if sc_result and sc_result.get("success"):
                candidates.append(sc_result)
            elif sc_result and not sc_result.get("success"):
                logger.info(f"  {cad_bend['bend_name']}: surface_classification failed: {sc_result.get('error', '?')}")

        # Method 3: profile cross-section fit
        profile_result = None
        if bend_direction is not None:
            section_half_width = max(5.0, min(10.0, 0.12 * float(np.linalg.norm(np.asarray(cad_bend.get("bend_line_end", bend_center)) - np.asarray(cad_bend.get("bend_line_start", bend_center))))))
            profile_result = measure_scan_bend_via_profile_section(
                scan_points=scan_points,
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                cad_bend_angle=cad_angle,
                search_radius=adaptive["search_radius"],
                section_half_width=section_half_width,
                cad_bend_line_start=np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64),
                cad_bend_line_end=np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64),
                allow_station_retries=enable_generic_station_retries,
            )
            if profile_result and profile_result.get("success"):
                candidates.append(profile_result)
            elif profile_result and not profile_result.get("success"):
                logger.info(f"  {cad_bend['bend_name']}: profile_section failed: {profile_result.get('error', '?')}")

        # Choose best candidate: prefer high-confidence gradient, then surface classification, then profile
        if candidates:
            _conf_rank = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
            _method_rank = {
                "parent_frame_profile_section": 4,
                "signed_distance_gradient": 3,
                "surface_classification": 2,
                "profile_section": 1,
                "probe_region": 0,
            }
            if use_parent_frame_sections:
                candidates.sort(key=lambda c: (
                    _method_rank.get(c.get("measurement_method"), -1),
                    _conf_rank.get(c.get("measurement_confidence", "low"), 0),
                ), reverse=True)
            else:
                candidates.sort(key=lambda c: (
                    _conf_rank.get(c.get("measurement_confidence", "low"), 0),
                    _method_rank.get(c.get("measurement_method"), -1),
                ), reverse=True)
            measurement = candidates[0]

        # Fallback: probe-region method
        if measurement is None:
            measurement = measure_scan_bend_at_cad_location(
                scan_points=scan_points,
                cad_bend_center=bend_center,
                cad_surf1_normal=surf1_normal,
                cad_surf2_normal=surf2_normal,
                cad_bend_direction=bend_direction,
                search_radius=adaptive["search_radius"],
                probe_offset=adaptive["offset_candidates"][2] if adaptive["shallow"] else 15.0,
                probe_radius=adaptive["probe_radius"],
                cad_bend_angle=cad_angle,
                cad_bend_line_start=np.asarray(cad_bend.get("bend_line_start", bend_center), dtype=np.float64),
                cad_bend_line_end=np.asarray(cad_bend.get("bend_line_end", bend_center), dtype=np.float64),
                allow_station_retries=enable_generic_station_retries,
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
                "side1_points": measurement.get("side1_points", 0),
                "side2_points": measurement.get("side2_points", 0),
                "point_count": measurement.get("point_count", 0),
                "measurement_method": measurement.get("measurement_method", "unknown"),
                "measurement_confidence": measurement.get("measurement_confidence", "very_low"),
                "probe_offset_used": measurement.get("probe_offset_used"),
                "min_offset_used": measurement.get("min_offset_used"),
                "inlier_side1_points": measurement.get("inlier_side1_points", 0),
                "inlier_side2_points": measurement.get("inlier_side2_points", 0),
                "nearby_points": measurement.get("nearby_points", measurement.get("point_count", 0)),
            })

        results.append(result)

    return results
