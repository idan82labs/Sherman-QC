"""
CAD Linear Dimension Extractor - Comprehensive Version

Extracts linear dimensions from CAD mesh geometry when XLSX specs are not available.
This provides a fallback for dimension comparison using CAD as ground truth.

Extracts:
1. Bounding box dimensions (overall size)
2. Cross-section dimensions at multiple slices
3. Edge-to-edge distances
4. Surface-to-surface distances (parallel and perpendicular)
5. Step heights and flange widths
6. Profile dimensions along principal axes

These dimensions are then compared against scan measurements.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
import logging
import time
import math

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    from scipy.spatial import KDTree, ConvexHull
    from scipy.ndimage import label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def safe_float(v, decimals=3):
    """Convert to JSON-safe float (handle NaN/Inf)."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), decimals)


def safe_list(lst, decimals=3):
    """Convert list of floats to JSON-safe list."""
    if lst is None:
        return None
    return [safe_float(x, decimals) for x in lst]


@dataclass
class CADLinearDimension:
    """A linear dimension extracted from CAD geometry."""
    dim_id: int
    dimension_type: str  # "length", "width", "height", "distance", "step", "flange", "section"
    value: float  # Nominal value in mm
    unit: str = "mm"

    # Geometry references
    description: str = ""
    point1: Optional[Tuple[float, float, float]] = None
    point2: Optional[Tuple[float, float, float]] = None
    direction: Optional[Tuple[float, float, float]] = None  # Measurement direction

    # Surface references (for surface-to-surface distances)
    surface1_id: Optional[int] = None
    surface2_id: Optional[int] = None

    # Confidence and metadata
    confidence: float = 1.0
    extraction_method: str = ""

    # Measurement axis for easier scan comparison
    measurement_axis: str = ""  # "X", "Y", "Z", or "custom"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim_id": self.dim_id,
            "type": self.dimension_type,
            "value": safe_float(self.value, 3),
            "unit": self.unit,
            "description": self.description,
            "point1": safe_list(self.point1, 3),
            "point2": safe_list(self.point2, 3),
            "direction": safe_list(self.direction, 4),
            "surface1_id": self.surface1_id,
            "surface2_id": self.surface2_id,
            "confidence": safe_float(self.confidence, 3),
            "extraction_method": self.extraction_method,
            "measurement_axis": self.measurement_axis,
        }


@dataclass
class CADDimensionExtractionResult:
    """Result of CAD dimension extraction."""
    dimensions: List[CADLinearDimension]
    bounding_box: Dict[str, float]
    total_dimensions: int
    processing_time_ms: float
    extraction_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "bounding_box": self.bounding_box,
            "total_dimensions": self.total_dimensions,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "extraction_summary": self.extraction_summary,
            "warnings": self.warnings,
        }


class CADDimensionExtractor:
    """
    Comprehensive linear dimension extractor from triangulated CAD mesh.

    Extracts dimensions using multiple methods:
    - Bounding box analysis
    - Cross-section slicing
    - Edge detection
    - Surface pair analysis
    - Profile sampling
    """

    def __init__(
        self,
        min_surface_area_pct: float = 0.5,  # Min surface area as % of total
        num_slices: int = 10,  # Number of cross-section slices per axis
        edge_angle_threshold: float = 30.0,  # Degrees to consider an edge
        min_dimension: float = 1.0,  # Minimum dimension to report (mm)
        max_dimensions: int = 100,  # Maximum number of dimensions to extract
        # Backward-compatible args kept for older tests/callers.
        coplanar_threshold: Optional[float] = None,
        parallel_threshold: Optional[float] = None,
    ):
        self.min_surface_area_pct = min_surface_area_pct
        self.num_slices = num_slices
        self.edge_angle_threshold = edge_angle_threshold
        self.min_dimension = min_dimension
        self.max_dimensions = max_dimensions
        self.coplanar_threshold = coplanar_threshold
        self.parallel_threshold = parallel_threshold

    def extract_from_mesh(self, mesh: Any) -> CADDimensionExtractionResult:
        """
        Extract comprehensive linear dimensions from an Open3D TriangleMesh.
        """
        start_time = time.time()
        warnings = []
        dimensions = []
        dim_id = 1
        extraction_summary = {}

        if not HAS_OPEN3D:
            return CADDimensionExtractionResult(
                dimensions=[],
                bounding_box={},
                total_dimensions=0,
                processing_time_ms=0,
                warnings=["Open3D not available"],
            )

        if not HAS_SCIPY:
            warnings.append("SciPy not available - some extraction methods disabled")

        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        face_normals = np.asarray(mesh.triangle_normals)

        if len(triangles) < 10:
            return CADDimensionExtractionResult(
                dimensions=[],
                bounding_box={},
                total_dimensions=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                warnings=["Mesh has too few faces"],
            )

        # Compute basic geometry
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)
        bb_size = bb_max - bb_min
        bb_center = (bb_min + bb_max) / 2

        bounding_box = {
            "min": [float(x) for x in bb_min],
            "max": [float(x) for x in bb_max],
            "center": [float(x) for x in bb_center],
            "length_x": float(bb_size[0]),
            "width_y": float(bb_size[1]),
            "height_z": float(bb_size[2]),
        }

        # Compute face data
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        face_centers = (v0 + v1 + v2) / 3
        total_area = face_areas.sum()

        # 1. BOUNDING BOX DIMENSIONS
        bb_dims = self._extract_bounding_box_dims(bb_min, bb_max, bb_size, dim_id)
        dimensions.extend(bb_dims)
        dim_id += len(bb_dims)
        extraction_summary["bounding_box"] = len(bb_dims)

        # 2. CROSS-SECTION DIMENSIONS (slices along each axis)
        try:
            section_dims = self._extract_cross_section_dims(
                vertices, bb_min, bb_max, bb_size, dim_id
            )
            dimensions.extend(section_dims)
            dim_id += len(section_dims)
            extraction_summary["cross_sections"] = len(section_dims)
        except Exception as e:
            warnings.append(f"Cross-section extraction failed: {e}")
            extraction_summary["cross_sections"] = 0

        # 3. EDGE-BASED DIMENSIONS
        try:
            edge_dims = self._extract_edge_dimensions(
                vertices, triangles, face_normals, bb_min, bb_max, dim_id
            )
            dimensions.extend(edge_dims)
            dim_id += len(edge_dims)
            extraction_summary["edge_dimensions"] = len(edge_dims)
        except Exception as e:
            warnings.append(f"Edge dimension extraction failed: {e}")
            extraction_summary["edge_dimensions"] = 0

        # 4. SURFACE-TO-SURFACE DIMENSIONS
        try:
            surface_dims = self._extract_surface_dimensions(
                vertices, triangles, face_normals, face_areas, face_centers,
                total_area, dim_id
            )
            dimensions.extend(surface_dims)
            dim_id += len(surface_dims)
            extraction_summary["surface_distances"] = len(surface_dims)
        except Exception as e:
            warnings.append(f"Surface distance extraction failed: {e}")
            extraction_summary["surface_distances"] = 0

        # 5. PROFILE DIMENSIONS (sample along edges of bounding box)
        try:
            profile_dims = self._extract_profile_dimensions(
                vertices, bb_min, bb_max, bb_size, dim_id
            )
            dimensions.extend(profile_dims)
            dim_id += len(profile_dims)
            extraction_summary["profile_dimensions"] = len(profile_dims)
        except Exception as e:
            warnings.append(f"Profile dimension extraction failed: {e}")
            extraction_summary["profile_dimensions"] = 0

        # 6. STEP/FLANGE DIMENSIONS
        try:
            step_dims = self._extract_step_dimensions(
                vertices, triangles, face_normals, face_areas, face_centers,
                bb_min, bb_max, dim_id
            )
            dimensions.extend(step_dims)
            dim_id += len(step_dims)
            extraction_summary["step_dimensions"] = len(step_dims)
        except Exception as e:
            warnings.append(f"Step dimension extraction failed: {e}")
            extraction_summary["step_dimensions"] = 0

        # Remove duplicates and sort by significance
        dimensions = self._deduplicate_dimensions(dimensions)
        dimensions = sorted(dimensions, key=lambda d: (-d.confidence, d.dim_id))

        # Limit total number
        if len(dimensions) > self.max_dimensions:
            dimensions = dimensions[:self.max_dimensions]
            warnings.append(f"Limited to {self.max_dimensions} dimensions")

        # Renumber dim_ids
        for i, dim in enumerate(dimensions):
            dim.dim_id = i + 1

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"CAD dimension extraction: {len(dimensions)} dimensions in {processing_time:.0f}ms")
        logger.info(f"  Breakdown: {extraction_summary}")

        return CADDimensionExtractionResult(
            dimensions=dimensions,
            bounding_box=bounding_box,
            total_dimensions=len(dimensions),
            processing_time_ms=processing_time,
            extraction_summary=extraction_summary,
            warnings=warnings,
        )

    def _extract_bounding_box_dims(
        self, bb_min: np.ndarray, bb_max: np.ndarray, bb_size: np.ndarray, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract overall bounding box dimensions."""
        dimensions = []
        dim_id = start_id

        axis_info = [
            ("X", "Overall Length (X)", (1, 0, 0), 0),
            ("Y", "Overall Width (Y)", (0, 1, 0), 1),
            ("Z", "Overall Height (Z)", (0, 0, 1), 2),
        ]

        for axis_name, desc, direction, axis_idx in axis_info:
            if bb_size[axis_idx] >= self.min_dimension:
                p1 = bb_min.copy()
                p2 = bb_min.copy()
                p2[axis_idx] = bb_max[axis_idx]

                dimensions.append(CADLinearDimension(
                    dim_id=dim_id,
                    dimension_type="bounding_box",
                    value=float(bb_size[axis_idx]),
                    description=desc,
                    point1=tuple(p1),
                    point2=tuple(p2),
                    direction=direction,
                    confidence=1.0,
                    extraction_method="bounding_box",
                    measurement_axis=axis_name,
                ))
                dim_id += 1

        return dimensions

    def _extract_cross_section_dims(
        self, vertices: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray,
        bb_size: np.ndarray, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract dimensions from cross-sections at regular intervals."""
        dimensions = []
        dim_id = start_id

        # For each primary axis, take slices and measure extent in other axes
        slice_configs = [
            # (slice_axis, measure_axis1, measure_axis2, slice_name, measure_names)
            (0, 1, 2, "X", ("Width (Y)", "Height (Z)")),  # Slice along X, measure Y and Z
            (1, 0, 2, "Y", ("Length (X)", "Height (Z)")),  # Slice along Y, measure X and Z
            (2, 0, 1, "Z", ("Length (X)", "Width (Y)")),  # Slice along Z, measure X and Y
        ]

        for slice_axis, meas_axis1, meas_axis2, slice_name, meas_names in slice_configs:
            # Create slice positions (skip first and last 10%)
            slice_min = bb_min[slice_axis] + 0.1 * bb_size[slice_axis]
            slice_max = bb_max[slice_axis] - 0.1 * bb_size[slice_axis]

            if slice_max <= slice_min:
                continue

            slice_positions = np.linspace(slice_min, slice_max, min(self.num_slices, 5))

            for i, slice_pos in enumerate(slice_positions):
                # Find vertices near this slice
                tolerance = bb_size[slice_axis] * 0.05  # 5% of size
                mask = np.abs(vertices[:, slice_axis] - slice_pos) < tolerance
                slice_verts = vertices[mask]

                if len(slice_verts) < 10:
                    continue

                # Measure extent in each measurement axis
                for meas_axis, meas_name in [(meas_axis1, meas_names[0]), (meas_axis2, meas_names[1])]:
                    extent_min = slice_verts[:, meas_axis].min()
                    extent_max = slice_verts[:, meas_axis].max()
                    extent = extent_max - extent_min

                    if extent >= self.min_dimension:
                        # Create measurement points
                        p1 = np.zeros(3)
                        p1[slice_axis] = slice_pos
                        p1[meas_axis] = extent_min

                        p2 = p1.copy()
                        p2[meas_axis] = extent_max

                        direction = [0, 0, 0]
                        direction[meas_axis] = 1

                        pct = (slice_pos - bb_min[slice_axis]) / bb_size[slice_axis] * 100

                        dimensions.append(CADLinearDimension(
                            dim_id=dim_id,
                            dimension_type="section",
                            value=float(extent),
                            description=f"{meas_name} at {slice_name}={pct:.0f}%",
                            point1=tuple(p1),
                            point2=tuple(p2),
                            direction=tuple(direction),
                            confidence=0.8,
                            extraction_method="cross_section",
                            measurement_axis=["X", "Y", "Z"][meas_axis],
                        ))
                        dim_id += 1

        return dimensions

    def _extract_edge_dimensions(
        self, vertices: np.ndarray, triangles: np.ndarray, face_normals: np.ndarray,
        bb_min: np.ndarray, bb_max: np.ndarray, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract dimensions between sharp edges (feature edges)."""
        dimensions = []
        dim_id = start_id

        if not HAS_SCIPY:
            return dimensions

        # Build edge-to-face map
        edge_faces = {}
        for fi, tri in enumerate(triangles):
            for i in range(3):
                v1, v2 = sorted([tri[i], tri[(i + 1) % 3]])
                edge = (v1, v2)
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)

        # Find sharp edges (edges where adjacent face normals differ significantly)
        sharp_edges = []
        angle_threshold = np.radians(self.edge_angle_threshold)

        for edge, faces in edge_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                cos_angle = np.clip(np.dot(n1, n2), -1, 1)
                angle = np.arccos(cos_angle)
                if angle > angle_threshold:
                    mid = (vertices[edge[0]] + vertices[edge[1]]) / 2
                    sharp_edges.append({
                        "edge": edge,
                        "midpoint": mid,
                        "angle_deg": np.degrees(angle),
                    })

        if len(sharp_edges) < 2:
            return dimensions

        # Cluster edges by position and measure distances between clusters
        edge_midpoints = np.array([e["midpoint"] for e in sharp_edges])

        # Measure distances between edges along each axis
        for axis in range(3):
            axis_name = ["X", "Y", "Z"][axis]

            # Sort edges by position along this axis
            sorted_indices = np.argsort(edge_midpoints[:, axis])
            sorted_positions = edge_midpoints[sorted_indices, axis]

            # Find significant gaps (clusters)
            if len(sorted_positions) < 2:
                continue

            gaps = np.diff(sorted_positions)
            significant_gap_threshold = (bb_max[axis] - bb_min[axis]) * 0.05

            for i, gap in enumerate(gaps):
                if gap >= max(self.min_dimension, significant_gap_threshold):
                    p1 = edge_midpoints[sorted_indices[i]].copy()
                    p2 = edge_midpoints[sorted_indices[i + 1]].copy()

                    direction = [0, 0, 0]
                    direction[axis] = 1

                    dimensions.append(CADLinearDimension(
                        dim_id=dim_id,
                        dimension_type="edge_distance",
                        value=float(gap),
                        description=f"Edge-to-edge ({axis_name})",
                        point1=tuple(p1),
                        point2=tuple(p2),
                        direction=tuple(direction),
                        confidence=0.7,
                        extraction_method="edge_analysis",
                        measurement_axis=axis_name,
                    ))
                    dim_id += 1

        return dimensions[:20]  # Limit edge dimensions

    def _extract_surface_dimensions(
        self, vertices: np.ndarray, triangles: np.ndarray, face_normals: np.ndarray,
        face_areas: np.ndarray, face_centers: np.ndarray, total_area: float, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract distances between major surfaces."""
        dimensions = []
        dim_id = start_id

        # Cluster surfaces by normal
        surfaces = self._cluster_surfaces(face_normals, face_areas, face_centers, total_area)

        if len(surfaces) < 2:
            return dimensions

        # For each pair of surfaces, compute distances
        for i, s1 in enumerate(surfaces):
            for s2 in surfaces[i + 1:]:
                # Check relationship between surface normals
                cos_angle = np.dot(s1["normal"], s2["normal"])
                angle = np.degrees(np.arccos(np.clip(abs(cos_angle), 0, 1)))

                if angle < 15:  # Parallel surfaces
                    dist = self._compute_surface_distance(s1, s2)
                    if dist >= self.min_dimension:
                        # Determine measurement axis
                        avg_normal = (s1["normal"] + np.sign(cos_angle) * s2["normal"]) / 2
                        if np.linalg.norm(avg_normal) > 0.1:
                            avg_normal = avg_normal / np.linalg.norm(avg_normal)
                            dominant_axis = np.argmax(np.abs(avg_normal))
                            axis_name = ["X", "Y", "Z"][dominant_axis]
                        else:
                            axis_name = "custom"
                            avg_normal = s1["normal"]

                        dimensions.append(CADLinearDimension(
                            dim_id=dim_id,
                            dimension_type="parallel_surfaces",
                            value=float(dist),
                            description=f"Surface distance ({axis_name})",
                            point1=tuple(s1["centroid"]),
                            point2=tuple(s2["centroid"]),
                            direction=tuple(avg_normal),
                            surface1_id=s1["id"],
                            surface2_id=s2["id"],
                            confidence=min(s1["area_pct"], s2["area_pct"]) / 5,
                            extraction_method="parallel_surface_distance",
                            measurement_axis=axis_name,
                        ))
                        dim_id += 1

                elif 75 < angle < 105:  # Perpendicular surfaces (flanges)
                    # Compute flange dimension
                    flange_dim = self._compute_flange_dimension(s1, s2, vertices, triangles)
                    if flange_dim and flange_dim >= self.min_dimension:
                        dimensions.append(CADLinearDimension(
                            dim_id=dim_id,
                            dimension_type="flange",
                            value=float(flange_dim),
                            description=f"Flange width",
                            point1=tuple(s1["centroid"]),
                            point2=tuple(s2["centroid"]),
                            direction=tuple(s2["normal"]),
                            surface1_id=s1["id"],
                            surface2_id=s2["id"],
                            confidence=0.6,
                            extraction_method="perpendicular_surface",
                            measurement_axis="custom",
                        ))
                        dim_id += 1

        return dimensions[:30]

    def _extract_profile_dimensions(
        self, vertices: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray,
        bb_size: np.ndarray, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract dimensions by sampling the profile along axes."""
        dimensions = []
        dim_id = start_id

        if not HAS_SCIPY:
            return dimensions

        # Sample profiles along each axis at different heights
        for primary_axis in range(3):
            for secondary_axis in range(3):
                if primary_axis == secondary_axis:
                    continue

                tertiary_axis = 3 - primary_axis - secondary_axis

                # Sample at 25%, 50%, 75% along tertiary axis
                for pct in [0.25, 0.5, 0.75]:
                    tertiary_pos = bb_min[tertiary_axis] + pct * bb_size[tertiary_axis]
                    tolerance = bb_size[tertiary_axis] * 0.1

                    # Get vertices in this slice
                    mask = np.abs(vertices[:, tertiary_axis] - tertiary_pos) < tolerance
                    slice_verts = vertices[mask]

                    if len(slice_verts) < 20:
                        continue

                    # Sample along primary axis
                    primary_min = slice_verts[:, primary_axis].min()
                    primary_max = slice_verts[:, primary_axis].max()

                    sample_positions = np.linspace(primary_min, primary_max, 10)

                    prev_extent = None
                    prev_pos = None

                    for sample_pos in sample_positions:
                        sample_tolerance = (primary_max - primary_min) * 0.1
                        mask2 = np.abs(slice_verts[:, primary_axis] - sample_pos) < sample_tolerance
                        local_verts = slice_verts[mask2]

                        if len(local_verts) < 5:
                            continue

                        extent = local_verts[:, secondary_axis].max() - local_verts[:, secondary_axis].min()

                        # Detect step changes
                        if prev_extent is not None and abs(extent - prev_extent) > self.min_dimension * 2:
                            # Step detected
                            step_height = abs(extent - prev_extent)
                            axis_names = ["X", "Y", "Z"]

                            p1 = np.zeros(3)
                            p1[primary_axis] = (sample_pos + prev_pos) / 2
                            p1[secondary_axis] = min(extent, prev_extent)
                            p1[tertiary_axis] = tertiary_pos

                            p2 = p1.copy()
                            p2[secondary_axis] = max(extent, prev_extent)

                            direction = [0, 0, 0]
                            direction[secondary_axis] = 1

                            dimensions.append(CADLinearDimension(
                                dim_id=dim_id,
                                dimension_type="step",
                                value=float(step_height),
                                description=f"Step height ({axis_names[secondary_axis]})",
                                point1=tuple(p1),
                                point2=tuple(p2),
                                direction=tuple(direction),
                                confidence=0.65,
                                extraction_method="profile_analysis",
                                measurement_axis=axis_names[secondary_axis],
                            ))
                            dim_id += 1

                        prev_extent = extent
                        prev_pos = sample_pos

        return dimensions[:20]

    def _extract_step_dimensions(
        self, vertices: np.ndarray, triangles: np.ndarray, face_normals: np.ndarray,
        face_areas: np.ndarray, face_centers: np.ndarray, bb_min: np.ndarray,
        bb_max: np.ndarray, start_id: int
    ) -> List[CADLinearDimension]:
        """Extract step heights and local feature dimensions."""
        dimensions = []
        dim_id = start_id

        # Group faces by height (Z) to find steps
        bb_size = bb_max - bb_min

        for axis in range(3):
            axis_name = ["X", "Y", "Z"][axis]

            # Find faces that are perpendicular to this axis (potential step surfaces)
            normal_component = np.abs(face_normals[:, axis])
            step_faces = normal_component > 0.9  # Nearly perpendicular to axis

            if step_faces.sum() < 10:
                continue

            # Get heights of step faces
            step_heights = face_centers[step_faces, axis]
            step_areas = face_areas[step_faces]

            # Cluster by height
            height_tolerance = bb_size[axis] * 0.02
            unique_heights = []

            sorted_indices = np.argsort(step_heights)
            current_cluster = [sorted_indices[0]]
            current_height = step_heights[sorted_indices[0]]

            for idx in sorted_indices[1:]:
                if abs(step_heights[idx] - current_height) < height_tolerance:
                    current_cluster.append(idx)
                else:
                    if step_areas[current_cluster].sum() > face_areas.sum() * 0.005:
                        avg_height = np.average(step_heights[current_cluster],
                                               weights=step_areas[current_cluster])
                        unique_heights.append(avg_height)
                    current_cluster = [idx]
                    current_height = step_heights[idx]

            # Don't forget last cluster
            if step_areas[current_cluster].sum() > face_areas.sum() * 0.005:
                avg_height = np.average(step_heights[current_cluster],
                                       weights=step_areas[current_cluster])
                unique_heights.append(avg_height)

            # Compute step heights
            unique_heights = sorted(unique_heights)
            for i in range(len(unique_heights) - 1):
                step = unique_heights[i + 1] - unique_heights[i]
                if step >= self.min_dimension:
                    p1 = bb_min.copy()
                    p1[axis] = unique_heights[i]
                    p2 = bb_min.copy()
                    p2[axis] = unique_heights[i + 1]

                    direction = [0, 0, 0]
                    direction[axis] = 1

                    dimensions.append(CADLinearDimension(
                        dim_id=dim_id,
                        dimension_type="step_height",
                        value=float(step),
                        description=f"Step height ({axis_name}): {unique_heights[i]:.1f} to {unique_heights[i+1]:.1f}",
                        point1=tuple(p1),
                        point2=tuple(p2),
                        direction=tuple(direction),
                        confidence=0.75,
                        extraction_method="step_analysis",
                        measurement_axis=axis_name,
                    ))
                    dim_id += 1

        return dimensions

    def _cluster_surfaces(
        self, face_normals: np.ndarray, face_areas: np.ndarray,
        face_centers: np.ndarray, total_area: float
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

    def _compute_surface_distance(self, s1: Dict, s2: Dict) -> float:
        """Compute distance between two parallel surfaces."""
        centroid_diff = s2["centroid"] - s1["centroid"]
        distance = abs(np.dot(centroid_diff, s1["normal"]))
        return distance

    def _compute_flange_dimension(
        self, s1: Dict, s2: Dict, vertices: np.ndarray, triangles: np.ndarray
    ) -> Optional[float]:
        """Compute flange width for perpendicular surfaces."""
        # Project s2 centers onto s1 normal to get flange extent
        if "centers" not in s2:
            return None

        centers = s2["centers"]
        projections = np.dot(centers - s1["centroid"], s1["normal"])

        # Flange width is the extent along s1's normal
        flange_width = projections.max() - projections.min()
        return flange_width if flange_width > 0 else None

    def _deduplicate_dimensions(
        self, dimensions: List[CADLinearDimension], tolerance: float = 0.5
    ) -> List[CADLinearDimension]:
        """Remove duplicate dimensions (similar value and location)."""
        if not dimensions:
            return dimensions

        unique = []
        for dim in dimensions:
            is_duplicate = False
            for existing in unique:
                # Check if values are similar
                if abs(dim.value - existing.value) < tolerance:
                    # Check if points are similar
                    if dim.point1 and existing.point1:
                        dist = np.linalg.norm(
                            np.array(dim.point1) - np.array(existing.point1)
                        )
                        if dist < tolerance * 2:
                            is_duplicate = True
                            # Keep the one with higher confidence
                            if dim.confidence > existing.confidence:
                                unique.remove(existing)
                                unique.append(dim)
                            break
            if not is_duplicate:
                unique.append(dim)

        return unique

    def extract_from_file(self, mesh_path: str) -> CADDimensionExtractionResult:
        """Extract dimensions from a mesh file."""
        if not HAS_OPEN3D:
            return CADDimensionExtractionResult(
                dimensions=[],
                bounding_box={},
                total_dimensions=0,
                processing_time_ms=0,
                warnings=["Open3D not available"],
            )

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return self.extract_from_mesh(mesh)


def measure_scan_dimension(
    scan_points: np.ndarray,
    cad_dimension: CADLinearDimension,
    search_radius: float = 5.0,
) -> Optional[float]:
    """
    Measure a corresponding dimension in the aligned scan.

    Uses the CAD dimension's measurement points and direction to find
    and measure the same feature in the scan point cloud.
    """
    if cad_dimension.point1 is None or cad_dimension.point2 is None:
        return None

    if not HAS_SCIPY:
        return None

    p1 = np.array(cad_dimension.point1)
    p2 = np.array(cad_dimension.point2)

    # Handle NaN in points
    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return None

    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 0.01:
        return None
    direction = direction / length

    # Build KDTree for scan points
    tree = KDTree(scan_points)

    # Adaptive search radius based on dimension size
    adaptive_radius = max(search_radius, length * 0.1)

    # Find scan points near p1 and p2
    idx1 = tree.query_ball_point(p1, adaptive_radius)
    idx2 = tree.query_ball_point(p2, adaptive_radius)

    if len(idx1) < 5 or len(idx2) < 5:
        # Try larger radius
        adaptive_radius *= 2
        idx1 = tree.query_ball_point(p1, adaptive_radius)
        idx2 = tree.query_ball_point(p2, adaptive_radius)

        if len(idx1) < 5 or len(idx2) < 5:
            return None

    pts1 = scan_points[idx1]
    pts2 = scan_points[idx2]

    # For bounding box and cross-section dimensions, use extent-based measurement
    if cad_dimension.dimension_type in ["bounding_box", "section", "step_height"]:
        # Measure extent along the measurement axis
        if cad_dimension.measurement_axis in ["X", "Y", "Z"]:
            axis = {"X": 0, "Y": 1, "Z": 2}[cad_dimension.measurement_axis]

            # Find all points in a cylinder along the measurement direction
            all_pts = np.vstack([pts1, pts2])
            extent_min = all_pts[:, axis].min()
            extent_max = all_pts[:, axis].max()

            return extent_max - extent_min

    # For surface distances and other types, use projection method
    proj1 = np.dot(pts1 - p1, direction)
    proj2 = np.dot(pts2 - p1, direction)

    # Use median positions as robust estimators
    pos1 = np.median(proj1)
    pos2 = np.median(proj2)

    measured_distance = abs(pos2 - pos1)
    return measured_distance


def measure_scan_dimensions_batch(
    scan_points: np.ndarray,
    dimensions: List[CADLinearDimension],
    search_radius: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Measure multiple dimensions in the scan efficiently.

    Returns list of measurement results with CAD value, scan value, and deviation.
    """
    if not HAS_SCIPY:
        return []

    # Build KDTree once for all measurements
    tree = KDTree(scan_points)

    results = []
    for dim in dimensions:
        scan_value = measure_scan_dimension(scan_points, dim, search_radius)

        if scan_value is not None:
            deviation = scan_value - dim.value

            # Determine tolerance based on dimension size (ISO 2768-m)
            if dim.value < 6:
                tolerance = 0.1
            elif dim.value < 30:
                tolerance = 0.2
            elif dim.value < 120:
                tolerance = 0.3
            elif dim.value < 400:
                tolerance = 0.5
            else:
                tolerance = 0.8

            status = "PASS" if abs(deviation) <= tolerance else "FAIL"

            results.append({
                "dim_id": dim.dim_id,
                "type": dim.dimension_type,
                "description": dim.description,
                "measurement_axis": dim.measurement_axis,
                "cad_value": safe_float(dim.value, 3),
                "scan_value": safe_float(scan_value, 3),
                "deviation": safe_float(deviation, 3),
                "tolerance": tolerance,
                "status": status,
                "confidence": safe_float(dim.confidence, 3),
            })

    return results
