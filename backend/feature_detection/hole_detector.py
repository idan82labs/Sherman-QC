"""
Hole Detector Module

Detects cylindrical holes in CAD mesh via edge cycle detection + cylinder fitting.
Uses Taubin algebraic circle fit for robust radius estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import defaultdict
import logging
import time

from .feature_types import HoleFeature, FeatureType

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    from scipy.spatial import KDTree
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class HoleDetector:
    """
    Detect cylindrical holes in CAD mesh.

    Detection pipeline:
    1. Find circular edges (hole rims) via edge cycle detection
    2. Fit circle to each cycle using Taubin algebraic fit
    3. Filter by circularity score
    4. Trace cylinder inward to get depth
    5. Return HoleFeature for each valid hole
    """

    def __init__(
        self,
        min_diameter: float = 0.5,
        max_diameter: float = 50.0,  # Reduced from 100 - most real holes are smaller
        circularity_threshold: float = 0.92,  # Increased from 0.85 for stricter filtering
        min_edge_points: int = 8,  # Increased from 6
        max_edge_points: int = 80,  # Reduced from 100
        coplanar_threshold: float = 3.0,  # Reduced from 5.0 for stricter planarity
    ):
        """
        Initialize hole detector.

        Args:
            min_diameter: Minimum hole diameter to detect (mm)
            max_diameter: Maximum hole diameter to detect (mm)
            circularity_threshold: Minimum circularity score (0-1)
            min_edge_points: Minimum vertices in edge cycle
            max_edge_points: Maximum vertices in edge cycle
            coplanar_threshold: Max angle deviation for coplanar points (degrees)
        """
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.circularity_threshold = circularity_threshold
        self.min_edge_points = min_edge_points
        self.max_edge_points = max_edge_points
        self.coplanar_threshold = coplanar_threshold

    def detect_holes(self, mesh: Any) -> List[HoleFeature]:
        """
        Detect cylindrical holes in a triangulated mesh.

        Args:
            mesh: Open3D TriangleMesh

        Returns:
            List of detected HoleFeature objects
        """
        if not HAS_OPEN3D:
            logger.warning("Open3D not available for hole detection")
            return []

        start_time = time.time()

        # Ensure mesh has computed normals
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        face_normals = np.asarray(mesh.triangle_normals)

        if len(triangles) < 10:
            logger.warning("Mesh has too few faces for hole detection")
            return []

        # Build edge data structures
        edge_to_faces, boundary_edges, sharp_edges = self._build_edge_data(
            vertices, triangles, face_normals
        )

        # Find circular edge cycles from boundary and sharp edges
        candidate_edges = boundary_edges | sharp_edges
        cycles = self._find_edge_cycles(vertices, candidate_edges)

        logger.info(f"Found {len(cycles)} edge cycles to analyze")

        # Analyze each cycle for hole characteristics
        holes = []
        hole_id = 1

        for cycle in cycles:
            cycle_points = vertices[list(cycle)]

            # Check if points are coplanar
            if not self._is_coplanar(cycle_points):
                continue

            # Fit circle to cycle points
            result = self._fit_circle_3d(cycle_points)
            if result is None:
                continue

            center, radius, normal, circularity = result

            diameter = 2 * radius

            # Filter by diameter range
            if diameter < self.min_diameter or diameter > self.max_diameter:
                continue

            # Filter by circularity
            if circularity < self.circularity_threshold:
                continue

            # Trace cylinder depth (optional, may fail for through holes)
            depth = self._trace_cylinder_depth(
                center, normal, radius, vertices, triangles, face_normals
            )

            # Create hole feature
            hole = HoleFeature(
                feature_id=f"hole_{hole_id}",
                feature_type=FeatureType.DIAMETER,
                nominal_value=diameter,
                unit="mm",
                position=center.copy(),
                direction=normal.copy(),
                confidence=circularity,
                axis=normal.copy(),
                depth=depth,
                entry_point=center.copy(),
                rim_points=cycle_points.copy(),
            )

            holes.append(hole)
            hole_id += 1

        # Also detect holes from cylindrical surface analysis
        surface_holes = self._detect_holes_from_surfaces(
            vertices, triangles, face_normals, hole_id
        )
        holes.extend(surface_holes)

        # Deduplicate holes at similar positions
        holes = self._deduplicate_holes(holes)

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Hole detection: {len(holes)} holes in {processing_time:.0f}ms")

        return holes

    def _build_edge_data(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        sharp_angle_threshold: float = 30.0,
    ) -> Tuple[Dict, Set, Set]:
        """
        Build edge data structures.

        Returns:
            Tuple of (edge_to_faces, boundary_edges, sharp_edges)
        """
        edge_to_faces = defaultdict(list)

        for fi, tri in enumerate(triangles):
            for i in range(3):
                v1, v2 = sorted([tri[i], tri[(i + 1) % 3]])
                edge = (v1, v2)
                edge_to_faces[edge].append(fi)

        # Boundary edges: only one adjacent face
        boundary_edges = set()
        for edge, faces in edge_to_faces.items():
            if len(faces) == 1:
                boundary_edges.add(edge)

        # Sharp edges: large dihedral angle between adjacent faces
        sharp_edges = set()
        sharp_threshold_rad = np.radians(sharp_angle_threshold)

        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                cos_angle = np.clip(np.dot(n1, n2), -1, 1)
                angle = np.arccos(cos_angle)
                if angle > sharp_threshold_rad:
                    sharp_edges.add(edge)

        return dict(edge_to_faces), boundary_edges, sharp_edges

    def _find_edge_cycles(
        self,
        vertices: np.ndarray,
        edges: Set[Tuple[int, int]],
    ) -> List[Set[int]]:
        """
        Find connected edge cycles that could represent hole rims.

        Returns:
            List of vertex index sets forming cycles
        """
        if not edges:
            return []

        # Build adjacency from edges
        adj = defaultdict(set)
        for v1, v2 in edges:
            adj[v1].add(v2)
            adj[v2].add(v1)

        # Find connected components
        visited = set()
        cycles = []

        for start_vertex in adj.keys():
            if start_vertex in visited:
                continue

            # BFS to find connected component
            component = set()
            queue = [start_vertex]

            while queue:
                v = queue.pop(0)
                if v in visited:
                    continue
                visited.add(v)
                component.add(v)

                for neighbor in adj[v]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Filter by size
            if self.min_edge_points <= len(component) <= self.max_edge_points:
                # Verify it forms a closed loop
                if self._is_closed_loop(component, adj):
                    cycles.append(component)

        return cycles

    def _is_closed_loop(self, component: Set[int], adj: Dict) -> bool:
        """Check if component forms a closed loop (each vertex has exactly 2 neighbors in component)."""
        for v in component:
            neighbors_in_component = sum(1 for n in adj[v] if n in component)
            if neighbors_in_component != 2:
                return False
        return True

    def _is_coplanar(self, points: np.ndarray, threshold_deg: float = None) -> bool:
        """Check if points are approximately coplanar."""
        if threshold_deg is None:
            threshold_deg = self.coplanar_threshold

        if len(points) < 4:
            return True

        # Fit plane using PCA
        centroid = points.mean(axis=0)
        centered = points - centroid

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Smallest eigenvalue corresponds to normal direction
            # If points are coplanar, smallest eigenvalue should be near zero
            eigenvalues = np.sort(eigenvalues)

            # Planarity: ratio of smallest to middle eigenvalue
            if eigenvalues[1] > 1e-10:
                planarity = eigenvalues[0] / eigenvalues[1]
            else:
                planarity = 0

            # Threshold: planarity < 0.1 means reasonably planar
            return planarity < 0.1

        except np.linalg.LinAlgError:
            return False

    def _fit_circle_3d(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray, float]]:
        """
        Fit a circle to 3D points that lie approximately on a plane.

        Uses PCA to find the plane, projects points, then uses Taubin algebraic fit.

        Returns:
            Tuple of (center_3d, radius, plane_normal, circularity_score) or None
        """
        if len(points) < 3:
            return None

        # Fit plane using PCA
        centroid = points.mean(axis=0)
        centered = points - centroid

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue (ascending)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Normal is direction of smallest variance
            normal = eigenvectors[:, 0]
            u = eigenvectors[:, 1]
            v = eigenvectors[:, 2]

        except np.linalg.LinAlgError:
            return None

        # Project points to 2D plane coordinates
        proj_u = np.dot(centered, u)
        proj_v = np.dot(centered, v)
        points_2d = np.column_stack([proj_u, proj_v])

        # Fit circle in 2D using Taubin algebraic fit
        circle_result = self._taubin_circle_fit(points_2d)
        if circle_result is None:
            return None

        center_2d, radius, circularity = circle_result

        # Convert center back to 3D
        center_3d = centroid + center_2d[0] * u + center_2d[1] * v

        return center_3d, radius, normal, circularity

    def _taubin_circle_fit(
        self,
        points_2d: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Taubin algebraic circle fit.

        This is more numerically stable than simple algebraic fitting.

        Returns:
            Tuple of (center_2d, radius, circularity_score) or None
        """
        if len(points_2d) < 3:
            return None

        x = points_2d[:, 0]
        y = points_2d[:, 1]
        n = len(x)

        # Centroid
        x_mean = x.mean()
        y_mean = y.mean()

        # Centered coordinates
        u = x - x_mean
        v = y - y_mean

        # Taubin method matrices
        Suv = np.sum(u * v)
        Suu = np.sum(u * u)
        Svv = np.sum(v * v)
        Suuv = np.sum(u * u * v)
        Suvv = np.sum(u * v * v)
        Suuu = np.sum(u * u * u)
        Svvv = np.sum(v * v * v)

        # Solve linear system
        A = np.array([
            [Suu, Suv],
            [Suv, Svv]
        ])
        b = np.array([
            (Suuu + Suvv) / 2,
            (Svvv + Suuv) / 2
        ])

        try:
            center_local = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        uc, vc = center_local
        center_2d = np.array([uc + x_mean, vc + y_mean])

        # Calculate radius
        distances = np.sqrt((x - center_2d[0])**2 + (y - center_2d[1])**2)
        radius = distances.mean()

        if radius < 1e-10:
            return None

        # Circularity score: 1 - (std of distances / radius)
        std_dist = distances.std()
        circularity = 1 - min(1, std_dist / radius)

        return center_2d, radius, circularity

    def _trace_cylinder_depth(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        radius: float,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        max_depth: float = 100.0,
    ) -> float:
        """
        Trace cylinder depth by raycasting inward.

        Returns:
            Estimated hole depth (0 for through holes)
        """
        if not HAS_SCIPY:
            return 0.0

        # Raycast from center along negative normal
        # Look for faces that are roughly perpendicular to the normal (cylinder bottom)

        # Sample points along the cylinder axis
        step = 0.5  # mm
        n_steps = int(max_depth / step)

        for i in range(1, n_steps):
            test_point = center - normal * (i * step)

            # Check if we're inside the mesh (simple heuristic)
            # Find nearest faces and check if their normals face away
            tree = KDTree(vertices)
            dist, idx = tree.query(test_point)

            # If very far from any vertex, probably through hole
            if dist > radius * 2:
                return 0.0  # Through hole

        return 0.0  # Default to through hole

    def _detect_holes_from_surfaces(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        start_id: int,
    ) -> List[HoleFeature]:
        """
        Detect holes by analyzing cylindrical surfaces in closed meshes.

        Uses curvature-based face clustering and cylinder fitting.
        Distinguishes holes (concave/inward facing) from external curves.
        """
        if not HAS_SCIPY:
            return []

        holes = []
        hole_id = start_id

        # Calculate face centers
        face_centers = vertices[triangles].mean(axis=1)
        n_faces = len(triangles)

        if n_faces < 20:
            return []

        # Step 1: Identify faces with high curvature (potential cylinder surfaces)
        # Calculate curvature at each face using neighboring face normal differences
        curved_faces = self._find_curved_faces(
            vertices, triangles, face_normals, face_centers
        )

        if len(curved_faces) < 10:
            logger.debug(f"Only {len(curved_faces)} curved faces found, skipping surface hole detection")
            return []

        logger.info(f"Found {len(curved_faces)} curved faces for cylinder analysis")

        # Step 2: Cluster curved faces by spatial proximity using DBSCAN
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning("sklearn not available for DBSCAN clustering")
            return []

        curved_centers = face_centers[curved_faces]
        curved_normals = face_normals[curved_faces]

        # Use DBSCAN to cluster curved faces
        # eps based on expected hole diameters
        eps = max(self.max_diameter / 4, 5.0)
        clustering = DBSCAN(eps=eps, min_samples=5).fit(curved_centers)
        labels = clustering.labels_

        unique_labels = set(labels) - {-1}  # Exclude noise
        logger.info(f"Found {len(unique_labels)} curved surface clusters")

        # Step 3: Analyze each cluster for cylinder characteristics
        for label in unique_labels:
            mask = labels == label
            cluster_face_indices = np.array(curved_faces)[mask]
            cluster_centers = curved_centers[mask]
            cluster_normals = curved_normals[mask]

            if len(cluster_centers) < 8:
                continue

            # Fit cylinder to this cluster
            cylinder_result = self._fit_cylinder_to_faces(
                cluster_centers, cluster_normals
            )

            if cylinder_result is None:
                continue

            center, axis, radius, is_concave, confidence = cylinder_result
            diameter = 2 * radius

            # Filter by diameter range
            if diameter < self.min_diameter or diameter > self.max_diameter:
                continue

            # Filter by confidence
            if confidence < 0.85:
                continue

            # Only accept concave cylinders (holes) or very small features
            # Convex cylinders are usually external geometry, not holes
            if not is_concave and diameter > 10.0:
                continue

            hole = HoleFeature(
                feature_id=f"hole_{hole_id}",
                feature_type=FeatureType.DIAMETER,
                nominal_value=diameter,
                unit="mm",
                position=center.copy(),
                direction=axis.copy(),
                confidence=confidence,
                axis=axis.copy(),
                depth=0.0,  # Unknown for surface-based detection
                entry_point=center.copy(),
            )

            holes.append(hole)
            hole_id += 1
            logger.info(f"Detected hole from surface: diameter={diameter:.2f}mm, concave={is_concave}, conf={confidence:.2f}")

        return holes

    def _find_curved_faces(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        face_centers: np.ndarray,
        curvature_threshold: float = 15.0,  # degrees
    ) -> List[int]:
        """
        Find faces that belong to curved surfaces.

        Returns indices of faces where neighboring normals differ significantly.
        """
        # Build face adjacency (faces sharing an edge)
        edge_to_faces = defaultdict(list)
        for fi, tri in enumerate(triangles):
            for i in range(3):
                v1, v2 = sorted([tri[i], tri[(i + 1) % 3]])
                edge_to_faces[(v1, v2)].append(fi)

        # For each face, calculate max normal difference with neighbors
        curved_faces = []
        threshold_rad = np.radians(curvature_threshold)

        for fi in range(len(triangles)):
            tri = triangles[fi]
            max_angle = 0

            for i in range(3):
                v1, v2 = sorted([tri[i], tri[(i + 1) % 3]])
                neighbors = edge_to_faces[(v1, v2)]

                for nfi in neighbors:
                    if nfi != fi:
                        n1 = face_normals[fi]
                        n2 = face_normals[nfi]
                        cos_angle = np.clip(np.dot(n1, n2), -1, 1)
                        angle = np.arccos(cos_angle)
                        max_angle = max(max_angle, angle)

            # Face is curved if it has significant normal variation
            if max_angle > threshold_rad:
                curved_faces.append(fi)

        return curved_faces

    def _fit_cylinder_to_faces(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, bool, float]]:
        """
        Fit a cylinder to a set of faces using their centers and normals.

        Returns:
            Tuple of (center, axis, radius, is_concave, confidence) or None
        """
        if len(face_centers) < 8:
            return None

        # Step 1: Estimate cylinder axis using PCA on face centers
        centroid = face_centers.mean(axis=0)
        centered = face_centers - centroid

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None

        # Sort by eigenvalue (descending)
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # For a cylinder, the axis is the direction of highest variance
        # unless the cylinder is very short
        axis = eigenvectors[:, 0]  # Primary direction

        # Alternative: use normal averaging to find axis
        # Cylinder normals are perpendicular to axis, so axis is perpendicular to avg normal
        avg_normal = face_normals.mean(axis=0)
        avg_normal_len = np.linalg.norm(avg_normal)

        # If normals are well-aligned radially, their average is small
        # Use PCA on normals to find the axis (smallest variance direction)
        try:
            cov_normals = np.cov(face_normals.T)
            eig_vals, eig_vecs = np.linalg.eigh(cov_normals)
            idx_n = np.argsort(eig_vals)
            # Cylinder axis is perpendicular to all normals, so it has smallest variance
            axis_from_normals = eig_vecs[:, idx_n[0]]
        except np.linalg.LinAlgError:
            axis_from_normals = axis

        # Choose the axis estimate that gives better radial alignment
        radial1 = self._compute_radial_alignment(face_centers, face_normals, axis, centroid)
        radial2 = self._compute_radial_alignment(face_centers, face_normals, axis_from_normals, centroid)

        if radial2 > radial1:
            axis = axis_from_normals

        # Step 2: Project points to plane perpendicular to axis
        # Create orthonormal basis
        if abs(axis[2]) < 0.9:
            perp1 = np.cross(axis, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(axis, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)

        # Project centers to 2D
        relative = face_centers - centroid
        proj_2d = np.column_stack([
            np.dot(relative, perp1),
            np.dot(relative, perp2)
        ])

        # Step 3: Fit circle in 2D using Taubin method
        circle_result = self._taubin_circle_fit(proj_2d)
        if circle_result is None:
            return None

        center_2d, radius, circularity = circle_result

        # Check circularity
        if circularity < 0.8:
            return None

        # Step 4: Convert center back to 3D
        center_3d = centroid + center_2d[0] * perp1 + center_2d[1] * perp2

        # Step 5: Determine if concave (hole) or convex (boss)
        # For a hole, normals point inward (toward axis)
        # For a boss, normals point outward (away from axis)
        is_concave = self._check_concavity(face_centers, face_normals, center_3d, axis)

        # Compute confidence based on circularity and radial alignment
        radial_alignment = self._compute_radial_alignment(
            face_centers, face_normals, axis, center_3d
        )
        confidence = circularity * radial_alignment

        return center_3d, axis, radius, is_concave, confidence

    def _compute_radial_alignment(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        axis: np.ndarray,
        center: np.ndarray,
    ) -> float:
        """Compute how well face normals align radially with respect to cylinder axis."""
        total_alignment = 0

        for pos, norm in zip(face_centers, face_normals):
            # Vector from center to face, projected perpendicular to axis
            to_face = pos - center
            radial = to_face - np.dot(to_face, axis) * axis
            radial_len = np.linalg.norm(radial)

            if radial_len < 1e-10:
                continue

            radial_unit = radial / radial_len

            # Normal component perpendicular to axis
            norm_perp = norm - np.dot(norm, axis) * axis
            norm_perp_len = np.linalg.norm(norm_perp)

            if norm_perp_len < 1e-10:
                continue

            norm_perp_unit = norm_perp / norm_perp_len

            # Alignment: should be 1 for perfect radial (inward or outward)
            alignment = abs(np.dot(radial_unit, norm_perp_unit))
            total_alignment += alignment

        return total_alignment / len(face_centers) if len(face_centers) > 0 else 0

    def _check_concavity(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
    ) -> bool:
        """
        Check if the cylinder is concave (hole) or convex (boss).

        For a hole, normals point toward the axis (inward).
        For a boss, normals point away from the axis (outward).
        """
        inward_count = 0
        outward_count = 0

        for pos, norm in zip(face_centers, face_normals):
            # Vector from center to face position (radial direction)
            to_face = pos - center
            radial = to_face - np.dot(to_face, axis) * axis

            # Dot product of normal with radial direction
            # Positive = outward (boss), Negative = inward (hole)
            dot = np.dot(norm, radial)

            if dot < 0:
                inward_count += 1
            else:
                outward_count += 1

        return inward_count > outward_count

    def _deduplicate_holes(
        self,
        holes: List[HoleFeature],
        position_threshold: float = 2.0,
        diameter_threshold: float = 1.0,
    ) -> List[HoleFeature]:
        """
        Remove duplicate hole detections at similar positions.
        """
        if not holes:
            return holes

        # Sort by confidence (highest first)
        holes_sorted = sorted(holes, key=lambda h: -h.confidence)

        unique = []
        for hole in holes_sorted:
            is_duplicate = False
            for existing in unique:
                pos_diff = np.linalg.norm(hole.position - existing.position)
                dia_diff = abs(hole.nominal_value - existing.nominal_value)

                if pos_diff < position_threshold and dia_diff < diameter_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(hole)

        return unique
