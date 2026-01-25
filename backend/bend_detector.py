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
                    dot = np.clip(np.dot(surf1.normal, surf2.normal), -1, 1)
                    angle_between_normals = np.arccos(dot) * 180 / np.pi

                    # Bend angle is supplement of angle between normals
                    # (parallel surfaces = 180° between normals = 0° bend)
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
) -> Dict[str, Any]:
    """
    Analyze deviations in a specific bend region.

    Args:
        bend: The detected bend feature
        nominal_angle: Expected angle from drawing (degrees)
        points: All point cloud points
        deviations: Deviation values for all points
        tolerance: Tolerance threshold in degrees/mm

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

    # Springback: measured angle is less than nominal (under-bent)
    # Overbend: measured angle is greater than nominal
    springback_indicator = 0.0
    over_bend_indicator = 0.0

    if nominal_angle is not None:
        if angle_deviation < 0:  # Under-bent
            springback_indicator = min(1.0, abs(angle_deviation) / 5.0)
        else:  # Over-bent
            over_bend_indicator = min(1.0, angle_deviation / 5.0)

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
        "status": status,
        "point_count": len(bend_points),
    }
