"""
CAD Feature Extractor Module

Extracts all measurable features from CAD mesh geometry:
- Holes/cylinders (via HoleDetector)
- Linear dimensions with surface context
- Angles (from existing CADBendExtractor)
- Radii/fillets

Produces a FeatureRegistry containing all detected features.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import logging
import time

from .feature_types import (
    FeatureType,
    MeasurableFeature,
    HoleFeature,
    LinearFeature,
    AngleFeature,
    FeatureRegistry,
)
from .hole_detector import HoleDetector

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class CADFeatureExtractor:
    """
    Extract all measurable features from CAD mesh.

    This class extends existing dimension extraction with feature-aware
    extraction that provides context needed for three-way comparison.
    """

    def __init__(
        self,
        min_surface_area_pct: float = 0.1,  # Lowered from 0.5% to capture smaller internal surfaces
        min_dimension: float = 1.0,
        hole_min_diameter: float = 0.5,
        hole_max_diameter: float = 100.0,
        parallelism_cos_threshold: float = 0.95,  # Relaxed from 0.98 for better detection
    ):
        """
        Initialize extractor.

        Args:
            min_surface_area_pct: Minimum surface area as % of total (default 0.1%)
            min_dimension: Minimum dimension to report (mm)
            hole_min_diameter: Minimum hole diameter to detect (mm)
            hole_max_diameter: Maximum hole diameter to detect (mm)
            parallelism_cos_threshold: Cosine threshold for parallel surface detection
        """
        self.min_surface_area_pct = min_surface_area_pct
        self.min_dimension = min_dimension
        self.parallelism_cos_threshold = parallelism_cos_threshold

        self.hole_detector = HoleDetector(
            min_diameter=hole_min_diameter,
            max_diameter=hole_max_diameter,
        )

    def extract(self, mesh: Any) -> FeatureRegistry:
        """
        Extract all features from a triangulated mesh.

        Args:
            mesh: Open3D TriangleMesh

        Returns:
            FeatureRegistry containing all detected features
        """
        if not HAS_OPEN3D:
            logger.warning("Open3D not available for feature extraction")
            return FeatureRegistry()

        start_time = time.time()

        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        face_normals = np.asarray(mesh.triangle_normals)

        if len(triangles) < 10:
            logger.warning("Mesh has too few faces for feature extraction")
            return FeatureRegistry()

        # Extract holes
        holes = self._extract_holes(mesh)

        # Extract linear dimensions with surface context
        linear_dims = self._extract_linear_with_context(
            vertices, triangles, face_normals
        )

        # Extract angles (bends) using existing logic
        angles = self._extract_angles(mesh, vertices, triangles, face_normals)

        # Extract radii/fillets (future enhancement)
        radii = self._extract_radii(vertices, triangles, face_normals)

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"CAD feature extraction: {len(holes)} holes, {len(linear_dims)} linear, "
            f"{len(angles)} angles, {len(radii)} radii ({processing_time:.0f}ms)"
        )

        return FeatureRegistry(
            holes=holes,
            linear_dims=linear_dims,
            angles=angles,
            radii=radii,
        )

    def _extract_holes(self, mesh: Any) -> List[HoleFeature]:
        """Extract holes/cylinders using HoleDetector."""
        try:
            return self.hole_detector.detect_holes(mesh)
        except Exception as e:
            logger.warning(f"Hole detection failed: {e}")
            return []

    def _extract_linear_with_context(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
    ) -> List[LinearFeature]:
        """
        Extract linear dimensions with surface context.

        Adds surface normals for surface-to-surface dimensions
        and reference points for edge-to-edge dimensions.
        """
        if not HAS_SCIPY:
            return []

        linear_dims = []
        feature_id = 1

        # Compute face data
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        face_centers = (v0 + v1 + v2) / 3
        total_area = face_areas.sum()

        # Bounding box
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)
        bb_size = bb_max - bb_min

        # 1. Overall dimensions
        for axis in range(3):
            axis_name = ["X", "Y", "Z"][axis]
            if bb_size[axis] >= self.min_dimension:
                p1 = bb_min.copy()
                p2 = bb_min.copy()
                p2[axis] = bb_max[axis]

                direction = np.zeros(3)
                direction[axis] = 1.0

                linear_dims.append(LinearFeature(
                    feature_id=f"linear_{feature_id}",
                    feature_type=FeatureType.LINEAR,
                    nominal_value=float(bb_size[axis]),
                    unit="mm",
                    position=(p1 + p2) / 2,
                    direction=direction,
                    confidence=1.0,
                    point1=p1,
                    point2=p2,
                    measurement_type=f"overall_{axis_name.lower()}",
                ))
                feature_id += 1

        # 2. Surface-to-surface dimensions
        surfaces = self._cluster_major_surfaces(
            face_normals, face_areas, face_centers, total_area
        )

        # Find parallel surface pairs
        for i, s1 in enumerate(surfaces):
            for s2 in surfaces[i + 1:]:
                # Check if parallel (normals aligned or opposite)
                cos_angle = abs(np.dot(s1["normal"], s2["normal"]))
                if cos_angle < self.parallelism_cos_threshold:  # Not parallel
                    continue

                # Compute distance between surface centroids along normal
                centroid_diff = s2["centroid"] - s1["centroid"]
                distance = abs(np.dot(centroid_diff, s1["normal"]))

                if distance < self.min_dimension:
                    continue

                # Determine measurement axis
                dominant_axis = np.argmax(np.abs(s1["normal"]))
                axis_name = ["X", "Y", "Z"][dominant_axis]

                linear_dims.append(LinearFeature(
                    feature_id=f"linear_{feature_id}",
                    feature_type=FeatureType.LINEAR,
                    nominal_value=float(distance),
                    unit="mm",
                    position=(s1["centroid"] + s2["centroid"]) / 2,
                    direction=s1["normal"].copy(),
                    confidence=min(s1["area_pct"], s2["area_pct"]) / 5,
                    point1=s1["centroid"].copy(),
                    point2=s2["centroid"].copy(),
                    surface1_normal=s1["normal"].copy(),
                    surface2_normal=s2["normal"].copy(),
                    measurement_type=f"surface_to_surface_{axis_name.lower()}",
                ))
                feature_id += 1

        # 3. Edge-based dimensions
        edge_dims = self._extract_edge_dimensions(
            vertices, triangles, face_normals, bb_min, bb_max, feature_id
        )
        linear_dims.extend(edge_dims)
        feature_id += len(edge_dims)

        # 4. Internal dimensions via cross-section analysis
        internal_dims = self._extract_internal_dimensions(
            vertices, triangles, face_normals, bb_min, bb_max, feature_id
        )
        linear_dims.extend(internal_dims)

        # Deduplicate similar dimensions
        linear_dims = self._deduplicate_linear(linear_dims)

        return linear_dims

    def _extract_angles(
        self,
        mesh: Any,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
    ) -> List[AngleFeature]:
        """
        Extract angle features (bends) from mesh.

        Uses logic similar to CADBendExtractor but returns AngleFeature objects.
        """
        angles = []
        feature_id = 1

        # Import CADBendExtractor if available
        try:
            from cad_bend_extractor import CADBendExtractor

            extractor = CADBendExtractor()
            result = extractor.extract_from_mesh(mesh)

            for bend in result.bends:
                angle = AngleFeature(
                    feature_id=f"angle_{feature_id}",
                    feature_type=FeatureType.ANGLE,
                    nominal_value=float(bend.bend_angle),
                    unit="deg",
                    position=bend.bend_center.copy(),
                    direction=bend.bend_direction.copy(),
                    confidence=bend.confidence,
                    surface1_normal=bend.surface1_normal.copy(),
                    surface2_normal=bend.surface2_normal.copy(),
                    apex_point=bend.bend_center.copy(),
                    bend_line=bend.bend_direction.copy(),
                    surface1_id=bend.surface1_id,
                    surface2_id=bend.surface2_id,
                )
                angles.append(angle)
                feature_id += 1

        except ImportError:
            # Fallback: compute angles directly
            angles = self._compute_angles_fallback(
                vertices, triangles, face_normals
            )

        return angles

    def _compute_angles_fallback(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
    ) -> List[AngleFeature]:
        """Fallback angle computation when CADBendExtractor not available."""
        angles = []
        feature_id = 1

        # Face data
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        face_centers = (v0 + v1 + v2) / 3
        total_area = face_areas.sum()

        # Get major surfaces
        surfaces = self._cluster_major_surfaces(
            face_normals, face_areas, face_centers, total_area
        )

        # Find surface pairs with significant angle between them
        min_bend_angle = 15.0
        max_bend_gap = 30.0

        for i, s1 in enumerate(surfaces):
            for s2 in surfaces[i + 1:]:
                # Angle between normals
                dot = np.clip(np.dot(s1["normal"], s2["normal"]), -1, 1)
                angle_between = np.degrees(np.arccos(abs(dot)))

                if angle_between < min_bend_angle:
                    continue

                # Check proximity
                centroid_dist = np.linalg.norm(s1["centroid"] - s2["centroid"])
                if centroid_dist > max_bend_gap * 3:
                    continue

                # Bend angle is supplement of angle between normals
                bend_angle = 180 - angle_between

                # Bend center
                bend_center = (s1["centroid"] + s2["centroid"]) / 2

                # Bend direction (perpendicular to both normals)
                bend_dir = np.cross(s1["normal"], s2["normal"])
                norm = np.linalg.norm(bend_dir)
                if norm > 1e-10:
                    bend_dir = bend_dir / norm
                else:
                    bend_dir = np.array([1, 0, 0])

                angle = AngleFeature(
                    feature_id=f"angle_{feature_id}",
                    feature_type=FeatureType.ANGLE,
                    nominal_value=float(bend_angle),
                    unit="deg",
                    position=bend_center.copy(),
                    direction=bend_dir,
                    confidence=0.7,
                    surface1_normal=s1["normal"].copy(),
                    surface2_normal=s2["normal"].copy(),
                    apex_point=bend_center.copy(),
                    bend_line=bend_dir,
                    surface1_id=s1["id"],
                    surface2_id=s2["id"],
                )
                angles.append(angle)
                feature_id += 1

        return angles

    def _extract_radii(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
    ) -> List[HoleFeature]:
        """
        Extract radius features (fillets, rounds).

        Currently returns empty list - future enhancement.
        """
        # Future: detect fillet/round features by analyzing
        # curved surface regions
        return []

    def _cluster_major_surfaces(
        self,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        face_centers: np.ndarray,
        total_area: float,
    ) -> List[Dict]:
        """Cluster faces into major planar surfaces by normal direction."""
        surfaces = []
        min_area = total_area * self.min_surface_area_pct / 100

        # Quantize normals
        quantized = np.round(face_normals * 10) / 10
        unique_normals = {}

        for i, (qn, area, center) in enumerate(zip(quantized, face_areas, face_centers)):
            key = tuple(qn)
            if key not in unique_normals:
                unique_normals[key] = {
                    "normal": face_normals[i].copy(),
                    "area": 0,
                    "centers": [],
                    "count": 0,
                }
            unique_normals[key]["area"] += area
            unique_normals[key]["centers"].append(center)
            unique_normals[key]["count"] += 1
            unique_normals[key]["normal"] = (
                unique_normals[key]["normal"] * (unique_normals[key]["count"] - 1) +
                face_normals[i]
            ) / unique_normals[key]["count"]

        surface_id = 0
        for key, data in unique_normals.items():
            if data["area"] >= min_area:
                normal = data["normal"]
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                centroid = np.mean(data["centers"], axis=0)

                surfaces.append({
                    "id": surface_id,
                    "normal": normal,
                    "centroid": centroid,
                    "area": data["area"],
                    "area_pct": 100 * data["area"] / total_area,
                    "centers": np.array(data["centers"]),
                })
                surface_id += 1

        return surfaces

    def _extract_edge_dimensions(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        bb_min: np.ndarray,
        bb_max: np.ndarray,
        start_id: int,
    ) -> List[LinearFeature]:
        """Extract dimensions between sharp edges."""
        if not HAS_SCIPY:
            return []

        linear_dims = []
        feature_id = start_id

        # Build edge-to-face map
        edge_faces = {}
        for fi, tri in enumerate(triangles):
            for i in range(3):
                v1, v2 = sorted([tri[i], tri[(i + 1) % 3]])
                edge = (v1, v2)
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)

        # Find sharp edges
        sharp_edges = []
        edge_angle_threshold = np.radians(30.0)

        for edge, faces in edge_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                cos_angle = np.clip(np.dot(n1, n2), -1, 1)
                angle = np.arccos(cos_angle)
                if angle > edge_angle_threshold:
                    mid = (vertices[edge[0]] + vertices[edge[1]]) / 2
                    sharp_edges.append({
                        "edge": edge,
                        "midpoint": mid,
                        "angle_deg": np.degrees(angle),
                    })

        if len(sharp_edges) < 2:
            return linear_dims

        # Measure distances between edges along each axis
        edge_midpoints = np.array([e["midpoint"] for e in sharp_edges])

        for axis in range(3):
            axis_name = ["X", "Y", "Z"][axis]

            sorted_indices = np.argsort(edge_midpoints[:, axis])
            sorted_positions = edge_midpoints[sorted_indices, axis]

            if len(sorted_positions) < 2:
                continue

            gaps = np.diff(sorted_positions)
            significant_gap_threshold = (bb_max[axis] - bb_min[axis]) * 0.05

            for i, gap in enumerate(gaps):
                if gap >= max(self.min_dimension, significant_gap_threshold):
                    p1 = edge_midpoints[sorted_indices[i]].copy()
                    p2 = edge_midpoints[sorted_indices[i + 1]].copy()

                    direction = np.zeros(3)
                    direction[axis] = 1.0

                    linear_dims.append(LinearFeature(
                        feature_id=f"linear_{feature_id}",
                        feature_type=FeatureType.LINEAR,
                        nominal_value=float(gap),
                        unit="mm",
                        position=(p1 + p2) / 2,
                        direction=direction,
                        confidence=0.7,
                        point1=p1,
                        point2=p2,
                        measurement_type=f"edge_to_edge_{axis_name.lower()}",
                    ))
                    feature_id += 1

        return linear_dims[:20]  # Limit edge dimensions

    def _extract_internal_dimensions(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        bb_min: np.ndarray,
        bb_max: np.ndarray,
        start_id: int,
    ) -> List[LinearFeature]:
        """
        Extract internal dimensions via cross-section slicing.

        For sheet metal parts, internal dimensions are typically:
        - Slot widths
        - Hole-to-edge distances
        - Feature-to-feature distances

        Algorithm:
        1. Slice mesh along each axis at multiple positions
        2. Find parallel line segments in each slice contour
        3. Measure distances between parallel segments
        """
        if not HAS_SCIPY:
            return []

        linear_dims = []
        feature_id = start_id
        bb_size = bb_max - bb_min

        # Precompute face data
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        # For each axis, take cross-sections and find internal gaps
        for axis in range(3):
            axis_name = ["X", "Y", "Z"][axis]
            other_axes = [i for i in range(3) if i != axis]

            # Generate slice positions (10 slices along each axis)
            num_slices = 10
            slice_positions = np.linspace(
                bb_min[axis] + bb_size[axis] * 0.1,
                bb_max[axis] - bb_size[axis] * 0.1,
                num_slices
            )

            for slice_pos in slice_positions:
                # Find triangles that intersect this slice plane
                intersections = self._slice_mesh_at_plane(
                    v0, v1, v2, axis, slice_pos
                )

                if len(intersections) < 2:
                    continue

                # Analyze intersection segments to find internal gaps
                gaps = self._find_internal_gaps(
                    intersections, other_axes, bb_min, bb_max
                )

                for gap_axis, gap_start, gap_end, gap_dist in gaps:
                    if gap_dist < self.min_dimension:
                        continue
                    if gap_dist > bb_size[gap_axis] * 0.8:
                        # Too close to overall dimension, skip
                        continue

                    # Create measurement points
                    p1 = np.zeros(3)
                    p2 = np.zeros(3)
                    p1[axis] = slice_pos
                    p2[axis] = slice_pos
                    p1[gap_axis] = gap_start
                    p2[gap_axis] = gap_end

                    # Set the third coordinate to middle of range
                    third_axis = [a for a in other_axes if a != gap_axis][0]
                    mid_third = (bb_min[third_axis] + bb_max[third_axis]) / 2
                    p1[third_axis] = mid_third
                    p2[third_axis] = mid_third

                    direction = np.zeros(3)
                    direction[gap_axis] = 1.0

                    gap_axis_name = ["X", "Y", "Z"][gap_axis]

                    linear_dims.append(LinearFeature(
                        feature_id=f"linear_{feature_id}",
                        feature_type=FeatureType.LINEAR,
                        nominal_value=float(gap_dist),
                        unit="mm",
                        position=(p1 + p2) / 2,
                        direction=direction,
                        confidence=0.6,  # Lower confidence for cross-section derived
                        point1=p1,
                        point2=p2,
                        measurement_type=f"internal_{gap_axis_name.lower()}",
                    ))
                    feature_id += 1

        return linear_dims[:30]  # Limit internal dimensions

    def _slice_mesh_at_plane(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        axis: int,
        position: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Slice mesh triangles with a plane perpendicular to given axis.

        Returns list of line segments (start, end) where mesh intersects plane.
        """
        intersections = []

        # Signed distances from plane for each vertex
        d0 = v0[:, axis] - position
        d1 = v1[:, axis] - position
        d2 = v2[:, axis] - position

        # For each triangle, check if it crosses the plane
        for i in range(len(d0)):
            # Get signed distances for this triangle
            dists = np.array([d0[i], d1[i], d2[i]])
            verts = np.array([v0[i], v1[i], v2[i]])

            # Check if triangle crosses plane
            signs = np.sign(dists)
            if np.all(signs >= 0) or np.all(signs <= 0):
                # All vertices on same side of plane
                continue

            # Find intersection points
            intersection_points = []
            for j in range(3):
                k = (j + 1) % 3
                if dists[j] * dists[k] < 0:  # Edge crosses plane
                    t = dists[j] / (dists[j] - dists[k])
                    point = verts[j] + t * (verts[k] - verts[j])
                    intersection_points.append(point)
                elif abs(dists[j]) < 1e-10:  # Vertex on plane
                    intersection_points.append(verts[j].copy())

            if len(intersection_points) >= 2:
                intersections.append((intersection_points[0], intersection_points[1]))

        return intersections

    def _find_internal_gaps(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        axes: List[int],
        bb_min: np.ndarray,
        bb_max: np.ndarray,
    ) -> List[Tuple[int, float, float, float]]:
        """
        Find internal gaps from cross-section segments.

        Returns list of (axis, start, end, distance) tuples for detected gaps.
        """
        gaps = []

        for axis in axes:
            # Project segment endpoints onto this axis
            positions = []
            for seg_start, seg_end in segments:
                positions.append(seg_start[axis])
                positions.append(seg_end[axis])

            if len(positions) < 4:
                continue

            positions = np.sort(np.unique(np.array(positions)))

            # Look for gaps between positions
            for i in range(len(positions) - 1):
                gap = positions[i + 1] - positions[i]

                # Filter: gap should be significant but not the full extent
                min_gap = self.min_dimension
                max_gap = (bb_max[axis] - bb_min[axis]) * 0.7

                if min_gap < gap < max_gap:
                    # Check if this is truly an internal gap (not at boundary)
                    if (positions[i] > bb_min[axis] + 1.0 and
                        positions[i + 1] < bb_max[axis] - 1.0):
                        gaps.append((axis, positions[i], positions[i + 1], gap))

        return gaps

    def _deduplicate_linear(
        self,
        dims: List[LinearFeature],
        value_tolerance: float = 0.5,
        position_tolerance: float = 2.0,
    ) -> List[LinearFeature]:
        """Remove duplicate linear dimensions."""
        if not dims:
            return dims

        # Sort by confidence (highest first)
        dims_sorted = sorted(dims, key=lambda d: -d.confidence)

        unique = []
        for dim in dims_sorted:
            is_duplicate = False
            for existing in unique:
                # Check value similarity
                if abs(dim.nominal_value - existing.nominal_value) > value_tolerance:
                    continue

                # Check position similarity
                pos_diff = np.linalg.norm(dim.position - existing.position)
                if pos_diff > position_tolerance:
                    continue

                is_duplicate = True
                break

            if not is_duplicate:
                unique.append(dim)

        return unique

    def extract_from_file(self, mesh_path: str) -> FeatureRegistry:
        """Extract features from a mesh file."""
        if not HAS_OPEN3D:
            return FeatureRegistry()

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) == 0:
            logger.warning(f"Failed to load mesh from {mesh_path}")
            return FeatureRegistry()

        return self.extract(mesh)
