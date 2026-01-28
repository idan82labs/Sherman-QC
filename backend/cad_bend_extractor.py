"""
CAD Mesh Bend Extractor

Extracts bend features from a triangulated CAD mesh by:
1. Building face adjacency and computing dihedral angles at shared edges
2. Region growing to identify major planar surfaces (using coplanar threshold)
3. Finding bends between major surfaces based on proximity and angle
4. Deduplicating and ordering bends spatially

This approach is much more reliable than point cloud curvature analysis because
the CAD mesh has clean, structured geometry with well-defined face normals.

The extracted CAD bends serve as ground truth for comparing against scan measurements.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

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


@dataclass
class MeshSurface:
    """A planar surface region identified in the CAD mesh."""
    surface_id: int
    face_indices: np.ndarray
    normal: np.ndarray  # Average surface normal
    centroid: np.ndarray  # Center of the surface
    area: float  # Total area in mm²
    area_pct: float  # Percentage of total mesh area
    face_count: int
    planarity: float  # How planar the surface is (0-1)


@dataclass
class CADBend:
    """A bend detected in the CAD mesh between two planar surfaces."""
    bend_id: int
    bend_name: str

    # Surfaces involved
    surface1_id: int
    surface2_id: int
    surface1_normal: np.ndarray
    surface2_normal: np.ndarray
    surface1_area: float
    surface2_area: float

    # Bend geometry
    bend_angle: float  # Degrees (e.g., 90° for a right-angle bend)
    angle_between_normals: float  # Complement of bend_angle
    bend_center: np.ndarray  # 3D center point of the bend line
    bend_direction: np.ndarray  # Direction vector of the bend line
    bend_line_start: np.ndarray
    bend_line_end: np.ndarray

    # Quality metrics
    min_surface_gap: float  # Minimum distance between surfaces at the bend
    confidence: float  # Detection confidence (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "bend_id": self.bend_id,
            "bend_name": self.bend_name,
            "bend_angle": round(self.bend_angle, 2),
            "angle_between_normals": round(self.angle_between_normals, 2),
            "bend_center": [round(float(x), 3) for x in self.bend_center],
            "bend_direction": [round(float(x), 3) for x in self.bend_direction],
            "bend_line_start": [round(float(x), 3) for x in self.bend_line_start],
            "bend_line_end": [round(float(x), 3) for x in self.bend_line_end],
            "surface1_id": self.surface1_id,
            "surface2_id": self.surface2_id,
            "surface1_normal": [round(float(x), 4) for x in self.surface1_normal],
            "surface2_normal": [round(float(x), 4) for x in self.surface2_normal],
            "surface1_area": round(self.surface1_area, 1),
            "surface2_area": round(self.surface2_area, 1),
            "min_surface_gap": round(self.min_surface_gap, 2),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CADBendExtractionResult:
    """Complete result of CAD bend extraction."""
    bends: List[CADBend]
    surfaces: List[MeshSurface]
    total_bends: int
    total_surfaces: int
    total_major_surfaces: int
    mesh_area: float
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bends": [b.to_dict() for b in self.bends],
            "total_bends": self.total_bends,
            "total_surfaces": self.total_surfaces,
            "total_major_surfaces": self.total_major_surfaces,
            "mesh_area": round(self.mesh_area, 1),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings": self.warnings,
        }


class CADBendExtractor:
    """
    Extracts bends from a triangulated CAD mesh.

    Algorithm:
    1. Compute face normals and areas
    2. Build face adjacency graph via shared edges
    3. Compute dihedral angle at each shared edge
    4. Region growing: group faces into planar surfaces (connected by flat edges)
    5. Identify major surfaces (above minimum area threshold)
    6. Find bends: pairs of major surfaces that are spatially close with significant
       angle difference
    7. Deduplicate bends at similar locations
    8. Order bends spatially along the part's primary axis
    """

    def __init__(
        self,
        coplanar_threshold: float = 25.0,  # Max angle (degrees) for faces to be coplanar
        min_area_pct: float = 0.5,  # Min surface area as % of total to be "major"
        max_bend_gap: float = 30.0,  # Max distance (mm) between surfaces for a bend
        min_bend_angle: float = 15.0,  # Min angle between normals to count as a bend
        dedup_distance: float = 15.0,  # Distance threshold for deduplicating bends
    ):
        self.coplanar_threshold = coplanar_threshold
        self.min_area_pct = min_area_pct
        self.max_bend_gap = max_bend_gap
        self.min_bend_angle = min_bend_angle
        self.dedup_distance = dedup_distance

    def extract_from_mesh(
        self,
        mesh: Any,
    ) -> CADBendExtractionResult:
        """
        Extract bends from an Open3D TriangleMesh.

        Args:
            mesh: open3d.geometry.TriangleMesh

        Returns:
            CADBendExtractionResult
        """
        start_time = time.time()
        warnings = []

        if not HAS_OPEN3D:
            return CADBendExtractionResult(
                bends=[], surfaces=[], total_bends=0,
                total_surfaces=0, total_major_surfaces=0,
                mesh_area=0, processing_time_ms=0,
                warnings=["Open3D not available"],
            )

        # Ensure mesh has triangle normals
        mesh.compute_triangle_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        face_normals = np.asarray(mesh.triangle_normals)
        n_faces = len(triangles)

        if n_faces < 10:
            return CADBendExtractionResult(
                bends=[], surfaces=[], total_bends=0,
                total_surfaces=0, total_major_surfaces=0,
                mesh_area=0, processing_time_ms=(time.time() - start_time) * 1000,
                warnings=["Mesh has too few faces for bend detection"],
            )

        logger.info(f"CAD bend extraction: {len(vertices)} vertices, {n_faces} faces")

        # Step 1: Compute face areas and centers
        face_areas = self._compute_face_areas(vertices, triangles)
        face_centers = vertices[triangles].mean(axis=1)
        total_area = face_areas.sum()

        # Step 2: Build face adjacency graph
        edge_to_faces, face_adj = self._build_adjacency(triangles)

        # Step 3: Compute dihedral angles at shared edges
        flat_edges = set()
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                dot = np.clip(np.dot(n1, n2), -1, 1)
                angle = np.degrees(np.arccos(abs(dot)))
                if angle < self.coplanar_threshold:
                    flat_edges.add(edge)

        # Step 4: Region growing to find planar surfaces
        surfaces = self._region_growing(
            n_faces, face_normals, face_adj, flat_edges,
            face_areas, face_centers, total_area,
        )
        logger.info(f"Found {len(surfaces)} surface regions")

        # Step 5: Filter to major surfaces
        min_area = total_area * self.min_area_pct / 100
        major_surfaces = {
            s.surface_id: s for s in surfaces if s.area >= min_area
        }
        logger.info(f"Major surfaces (>{self.min_area_pct}% area): {len(major_surfaces)}")

        for sid, s in sorted(major_surfaces.items(), key=lambda x: -x[1].area):
            logger.debug(
                f"  Surface {sid}: {s.face_count} faces, "
                f"area={s.area:.0f}mm² ({s.area_pct:.1f}%), "
                f"normal=[{s.normal[0]:.3f}, {s.normal[1]:.3f}, {s.normal[2]:.3f}]"
            )

        # Step 5b: Merge parallel sheet surfaces (top/bottom of same physical surface)
        major_surfaces = self._merge_parallel_surfaces(
            major_surfaces, face_centers, face_areas, face_normals
        )
        logger.info(f"After merging parallel surfaces: {len(major_surfaces)}")

        # Step 6: Find bends between major surfaces
        raw_bends = self._find_bends(
            major_surfaces, face_centers, vertices, triangles, face_normals,
        )
        logger.info(f"Raw bends found: {len(raw_bends)}")

        # Step 7: Deduplicate
        unique_bends = self._deduplicate_bends(raw_bends)
        logger.info(f"After deduplication: {len(unique_bends)}")

        # Step 8: Order and number
        ordered_bends = self._order_bends(unique_bends)

        for b in ordered_bends:
            logger.info(
                f"  {b.bend_name}: {b.bend_angle:.1f}° "
                f"(S{b.surface1_id} ↔ S{b.surface2_id})"
            )

        processing_time = (time.time() - start_time) * 1000

        return CADBendExtractionResult(
            bends=ordered_bends,
            surfaces=list(major_surfaces.values()),
            total_bends=len(ordered_bends),
            total_surfaces=len(surfaces),
            total_major_surfaces=len(major_surfaces),
            mesh_area=total_area,
            processing_time_ms=processing_time,
            warnings=warnings,
        )

    def extract_from_file(self, mesh_path: str) -> CADBendExtractionResult:
        """
        Extract bends from a mesh file (PLY, STL, OBJ).

        Args:
            mesh_path: Path to the mesh file

        Returns:
            CADBendExtractionResult
        """
        if not HAS_OPEN3D:
            return CADBendExtractionResult(
                bends=[], surfaces=[], total_bends=0,
                total_surfaces=0, total_major_surfaces=0,
                mesh_area=0, processing_time_ms=0,
                warnings=["Open3D not available"],
            )

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) == 0:
            return CADBendExtractionResult(
                bends=[], surfaces=[], total_bends=0,
                total_surfaces=0, total_major_surfaces=0,
                mesh_area=0, processing_time_ms=0,
                warnings=[f"Failed to load mesh from {mesh_path}"],
            )

        return self.extract_from_mesh(mesh)

    def _compute_face_areas(
        self, vertices: np.ndarray, triangles: np.ndarray,
    ) -> np.ndarray:
        """Compute area of each triangle face."""
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return areas

    def _build_adjacency(
        self, triangles: np.ndarray,
    ) -> Tuple[Dict, Dict]:
        """Build edge-to-face and face-to-face adjacency mappings."""
        edge_to_faces = defaultdict(list)
        for fi, tri in enumerate(triangles):
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edge_to_faces[edge].append(fi)

        face_adj = defaultdict(set)
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                face_adj[faces[0]].add(faces[1])
                face_adj[faces[1]].add(faces[0])

        return dict(edge_to_faces), dict(face_adj)

    def _region_growing(
        self,
        n_faces: int,
        face_normals: np.ndarray,
        face_adj: Dict,
        flat_edges: set,
        face_areas: np.ndarray,
        face_centers: np.ndarray,
        total_area: float,
    ) -> List[MeshSurface]:
        """Group faces into planar surface regions using region growing."""
        face_labels = -np.ones(n_faces, dtype=int)
        current_label = 0
        surfaces = []

        # Process faces in order of area (largest first for more stable regions)
        face_order = np.argsort(-face_areas)

        for start_face in face_order:
            if face_labels[start_face] >= 0:
                continue

            queue = [start_face]
            region = []
            ref_normal = face_normals[start_face].copy()

            while queue:
                fi = queue.pop(0)
                if face_labels[fi] >= 0:
                    continue

                # Check coplanarity with region
                dot = np.clip(np.dot(face_normals[fi], ref_normal), -1, 1)
                angle = np.degrees(np.arccos(abs(dot)))
                if angle > self.coplanar_threshold and len(region) > 0:
                    continue

                face_labels[fi] = current_label
                region.append(fi)

                # Update reference normal (area-weighted average)
                if len(region) > 1:
                    region_arr = np.array(region)
                    weights = face_areas[region_arr]
                    weighted_normals = face_normals[region_arr] * weights[:, None]
                    avg = weighted_normals.sum(axis=0)
                    norm = np.linalg.norm(avg)
                    if norm > 0:
                        ref_normal = avg / norm

                # Add neighbors connected by flat edges
                for neighbor in face_adj.get(fi, set()):
                    if face_labels[neighbor] < 0:
                        queue.append(neighbor)

            if not region:
                continue

            region_arr = np.array(region)
            area = face_areas[region_arr].sum()
            avg_normal = face_normals[region_arr].mean(axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal /= norm
            centroid = face_centers[region_arr].mean(axis=0)

            # Planarity score
            dots = np.abs(np.dot(face_normals[region_arr], avg_normal))
            planarity = float(dots.mean()) if len(dots) > 0 else 1.0

            surfaces.append(MeshSurface(
                surface_id=current_label,
                face_indices=region_arr,
                normal=avg_normal,
                centroid=centroid,
                area=area,
                area_pct=area / total_area * 100,
                face_count=len(region),
                planarity=planarity,
            ))
            current_label += 1

        return surfaces

    def _merge_parallel_surfaces(
        self,
        major_surfaces: Dict[int, MeshSurface],
        face_centers: np.ndarray,
        face_areas: np.ndarray,
        face_normals: np.ndarray,
        parallel_threshold: float = 10.0,
        overlap_threshold: float = 0.5,
    ) -> Dict[int, MeshSurface]:
        """Merge parallel, overlapping surfaces (top/bottom of same sheet).

        For sheet metal parts, each physical surface has two mesh surfaces
        (one for each side of the sheet). This method identifies and merges
        them so each physical surface is represented once.

        Two surfaces are merged if:
        1. Their normals are nearly parallel (< parallel_threshold degrees)
        2. They significantly overlap when projected onto their shared plane
        """
        surface_list = sorted(major_surfaces.values(), key=lambda s: -s.area)
        merged_ids: set = set()
        result: Dict[int, MeshSurface] = {}

        for i, s1 in enumerate(surface_list):
            if s1.surface_id in merged_ids:
                continue

            for j in range(i + 1, len(surface_list)):
                s2 = surface_list[j]
                if s2.surface_id in merged_ids:
                    continue

                # Check if normals are nearly parallel
                cos_angle = abs(np.dot(s1.normal, s2.normal))
                angle = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                if angle > parallel_threshold:
                    continue

                # Check spatial overlap: project both onto a shared plane
                plane_normal = s1.normal
                arbitrary = np.array([1, 0, 0]) if abs(plane_normal[0]) < 0.9 else np.array([0, 1, 0])
                u = np.cross(plane_normal, arbitrary)
                u /= np.linalg.norm(u)
                v = np.cross(plane_normal, u)

                pts1 = face_centers[s1.face_indices]
                pts2 = face_centers[s2.face_indices]

                proj1_u = pts1 @ u
                proj1_v = pts1 @ v
                proj2_u = pts2 @ u
                proj2_v = pts2 @ v

                # Compute bounding box overlap
                overlap_u = max(0, min(proj1_u.max(), proj2_u.max()) - max(proj1_u.min(), proj2_u.min()))
                overlap_v = max(0, min(proj1_v.max(), proj2_v.max()) - max(proj1_v.min(), proj2_v.min()))
                extent1_u = proj1_u.ptp()
                extent1_v = proj1_v.ptp()

                if extent1_u > 0 and extent1_v > 0:
                    overlap_ratio = (overlap_u * overlap_v) / (extent1_u * extent1_v)
                else:
                    overlap_ratio = 0

                if overlap_ratio > overlap_threshold:
                    merged_ids.add(s2.surface_id)
                    logger.info(
                        f"  Merging S{s2.surface_id} into S{s1.surface_id} "
                        f"(parallel={angle:.1f} deg, overlap={overlap_ratio:.0%})"
                    )

            result[s1.surface_id] = s1

        return result

    def _find_bends(
        self,
        major_surfaces: Dict[int, MeshSurface],
        face_centers: np.ndarray,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
    ) -> List[CADBend]:
        """Find bends between pairs of major surfaces."""
        bends = []
        major_ids = list(major_surfaces.keys())

        for i, r1_id in enumerate(major_ids):
            for r2_id in major_ids[i + 1:]:
                r1 = major_surfaces[r1_id]
                r2 = major_surfaces[r2_id]

                # Angle between normals (use abs(dot) to handle flipped normals)
                dot = np.clip(np.dot(r1.normal, r2.normal), -1, 1)
                angle_between = np.degrees(np.arccos(abs(dot)))

                # Skip nearly-parallel or nearly-anti-parallel surfaces
                if angle_between < self.min_bend_angle:
                    continue

                # Check spatial proximity
                pts1 = face_centers[r1.face_indices]
                pts2 = face_centers[r2.face_indices]

                # Sample for speed (deterministic stride-based)
                n_sample = min(300, len(pts1))
                stride = max(1, len(pts1) // n_sample)
                sample1 = pts1[::stride][:n_sample]

                tree2 = KDTree(pts2)
                dists, nearest_idx = tree2.query(sample1)
                min_dist = dists.min()

                if min_dist > self.max_bend_gap:
                    continue

                # Find the bend boundary region
                close_mask = dists < self.max_bend_gap
                if close_mask.sum() < 3:
                    continue

                # Bend center: midpoint between closest points
                closest_i = np.argmin(dists)
                pt1 = sample1[closest_i]
                pt2 = pts2[nearest_idx[closest_i]]
                bend_center = (pt1 + pt2) / 2

                # Compute bend line direction from boundary points
                boundary_pts = sample1[close_mask]
                if len(boundary_pts) >= 3:
                    centered = boundary_pts - boundary_pts.mean(axis=0)
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    bend_dir = eigenvectors[:, -1]  # Largest variance direction
                else:
                    bend_dir = np.array([1, 0, 0])

                # Normalize direction
                bend_dir = bend_dir / np.linalg.norm(bend_dir)

                # Bend line extent
                if len(boundary_pts) >= 3:
                    projections = np.dot(boundary_pts - bend_center, bend_dir)
                    line_half = max(abs(projections.min()), abs(projections.max()))
                else:
                    line_half = 25.0

                bend_line_start = bend_center - bend_dir * line_half
                bend_line_end = bend_center + bend_dir * line_half

                # Bend angle is supplement of angle between normals
                bend_angle = 180 - angle_between

                # Confidence based on number of close points and surface planarity
                close_ratio = close_mask.sum() / len(close_mask)
                confidence = min(1.0, (
                    r1.planarity * 0.3 +
                    r2.planarity * 0.3 +
                    close_ratio * 0.4
                ))

                bends.append(CADBend(
                    bend_id=0,  # Will be set during ordering
                    bend_name="",
                    surface1_id=r1_id,
                    surface2_id=r2_id,
                    surface1_normal=r1.normal.copy(),
                    surface2_normal=r2.normal.copy(),
                    surface1_area=r1.area,
                    surface2_area=r2.area,
                    bend_angle=bend_angle,
                    angle_between_normals=angle_between,
                    bend_center=bend_center,
                    bend_direction=bend_dir,
                    bend_line_start=bend_line_start,
                    bend_line_end=bend_line_end,
                    min_surface_gap=float(min_dist),
                    confidence=confidence,
                ))

        return bends

    def _deduplicate_bends(self, bends: List[CADBend]) -> List[CADBend]:
        """Remove duplicate bends at the same spatial location.

        When multiple surface pairs meet at the same bend line, keep the
        one with the largest combined surface area (most reliable).
        """
        if not bends:
            return []

        # Sort by total area (largest first = most reliable)
        bends_sorted = sorted(
            bends,
            key=lambda b: -(b.surface1_area + b.surface2_area),
        )

        kept = []
        used_centers = []

        for b in bends_sorted:
            center = b.bend_center
            is_duplicate = False
            for uc in used_centers:
                if np.linalg.norm(center - uc) < self.dedup_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(b)
                used_centers.append(center)

        return kept

    def _order_bends(self, bends: List[CADBend]) -> List[CADBend]:
        """Order bends spatially along the part's primary axis and assign IDs."""
        if not bends:
            return []

        if len(bends) == 1:
            bends[0].bend_id = 1
            bends[0].bend_name = "Bend 1"
            return bends

        # PCA on bend centers to find primary axis
        centroids = np.array([b.bend_center for b in bends])
        centered = centroids - centroids.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        primary_axis = eigenvectors[:, -1]

        # Project onto primary axis and sort
        projections = np.dot(centroids, primary_axis)
        order = np.argsort(projections)

        ordered = []
        for new_id, old_idx in enumerate(order, start=1):
            b = bends[old_idx]
            b.bend_id = new_id
            b.bend_name = f"Bend {new_id}"
            ordered.append(b)

        return ordered
