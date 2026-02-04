"""
Coverage Calculator for Live Scanning

Computes scan coverage by comparing scan point cloud against CAD reference.
Uses voxel grid representation for efficient coverage calculation.

For sheet metal parts:
- Surface voxelization (not volumetric) since parts are hollow
- 2mm default resolution (balances accuracy and performance)
- Handles partial scans and incremental updates

Usage:
    from coverage_calculator import CoverageCalculator

    calc = CoverageCalculator()
    result = calc.compute_coverage(cad_points, scan_points)
    print(f"Coverage: {result.coverage_percent:.1f}%")
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Set
from dataclasses import dataclass, field
from scipy.spatial import KDTree
import time

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_VOXEL_SIZE = 2.0  # mm
DEFAULT_COVERAGE_TOLERANCE = 3.0  # mm - distance to count as "covered"
MIN_POINTS_FOR_COVERAGE = 100


@dataclass
class CoverageResult:
    """Result of coverage calculation."""
    coverage_percent: float
    total_voxels: int
    covered_voxels: int
    uncovered_voxels: int

    # Spatial analysis
    gap_clusters: List[Dict]  # Clusters of uncovered regions
    largest_gap_size_mm: float

    # Performance
    processing_time_ms: float

    # Metadata
    voxel_size_mm: float
    tolerance_mm: float
    cad_points_count: int
    scan_points_count: int

    def to_dict(self) -> dict:
        return {
            "coverage_percent": round(self.coverage_percent, 1),
            "total_voxels": self.total_voxels,
            "covered_voxels": self.covered_voxels,
            "uncovered_voxels": self.uncovered_voxels,
            "gap_clusters": self.gap_clusters,
            "largest_gap_size_mm": round(self.largest_gap_size_mm, 1),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "voxel_size_mm": self.voxel_size_mm,
            "tolerance_mm": self.tolerance_mm,
            "cad_points_count": self.cad_points_count,
            "scan_points_count": self.scan_points_count,
        }


class VoxelGrid:
    """
    Efficient voxel grid for coverage tracking.

    Uses sparse representation - only stores occupied voxels.
    """

    def __init__(self, voxel_size: float = DEFAULT_VOXEL_SIZE):
        self.voxel_size = voxel_size
        self._voxels: Set[Tuple[int, int, int]] = set()
        self._covered: Set[Tuple[int, int, int]] = set()
        self._origin: np.ndarray = np.zeros(3)

    def _point_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a 3D point to voxel coordinates."""
        coords = np.floor((point - self._origin) / self.voxel_size).astype(int)
        return tuple(coords)

    def _voxel_to_point(self, voxel: Tuple[int, int, int]) -> np.ndarray:
        """Convert voxel coordinates to 3D point (center)."""
        return self._origin + (np.array(voxel) + 0.5) * self.voxel_size

    def build_from_points(self, points: np.ndarray):
        """Build voxel grid from point cloud."""
        if len(points) == 0:
            return

        # Set origin to minimum point
        self._origin = np.min(points, axis=0)

        # Add all points to voxel set
        for point in points:
            voxel = self._point_to_voxel(point)
            self._voxels.add(voxel)

        logger.debug(f"Built voxel grid: {len(self._voxels)} voxels from {len(points)} points")

    def mark_covered(self, points: np.ndarray, tolerance: float = 0.0):
        """
        Mark voxels as covered based on scan points.

        Args:
            points: Scan point cloud
            tolerance: Additional tolerance in mm
        """
        if len(points) == 0:
            return

        # Compute tolerance in voxel units
        voxel_tolerance = int(np.ceil(tolerance / self.voxel_size))

        for point in points:
            base_voxel = self._point_to_voxel(point)

            # Check base voxel and neighbors within tolerance
            for dx in range(-voxel_tolerance, voxel_tolerance + 1):
                for dy in range(-voxel_tolerance, voxel_tolerance + 1):
                    for dz in range(-voxel_tolerance, voxel_tolerance + 1):
                        check_voxel = (
                            base_voxel[0] + dx,
                            base_voxel[1] + dy,
                            base_voxel[2] + dz
                        )
                        if check_voxel in self._voxels:
                            self._covered.add(check_voxel)

    def get_coverage(self) -> Tuple[int, int]:
        """Get coverage statistics."""
        return len(self._covered), len(self._voxels)

    def get_uncovered_voxels(self) -> List[np.ndarray]:
        """Get centers of uncovered voxels."""
        uncovered = self._voxels - self._covered
        return [self._voxel_to_point(v) for v in uncovered]

    def reset_coverage(self):
        """Reset coverage tracking (keep voxel grid)."""
        self._covered.clear()


class CoverageCalculator:
    """
    Calculates scan coverage against CAD reference.

    Uses voxel grid approach:
    1. Voxelize CAD surface at specified resolution
    2. Mark voxels as covered when scan points fall within tolerance
    3. Compute coverage percentage

    For incremental updates during live scanning:
    - Build voxel grid once from CAD
    - Update coverage as new scan files arrive
    - Reset for new sessions
    """

    def __init__(
        self,
        voxel_size: float = DEFAULT_VOXEL_SIZE,
        tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    ):
        """
        Initialize coverage calculator.

        Args:
            voxel_size: Voxel size in mm (default 2mm)
            tolerance: Distance tolerance for coverage in mm (default 3mm)
        """
        self.voxel_size = voxel_size
        self.tolerance = tolerance

        self._voxel_grid: Optional[VoxelGrid] = None
        self._cad_tree: Optional[KDTree] = None
        self._cad_points: Optional[np.ndarray] = None

    def set_cad_reference(self, cad_points: np.ndarray):
        """
        Set CAD reference for coverage calculation.

        Call this once when part is identified, then call
        add_scan_points() for each scan file.

        Args:
            cad_points: Nx3 numpy array of CAD surface points
        """
        self._cad_points = cad_points

        # Build voxel grid
        self._voxel_grid = VoxelGrid(voxel_size=self.voxel_size)
        self._voxel_grid.build_from_points(cad_points)

        # Build KD-tree for fast nearest neighbor queries
        self._cad_tree = KDTree(cad_points)

        logger.info(
            f"Set CAD reference: {len(cad_points)} points, "
            f"{len(self._voxel_grid._voxels)} voxels"
        )

    def add_scan_points(self, scan_points: np.ndarray) -> float:
        """
        Add scan points and update coverage.

        Args:
            scan_points: Nx3 numpy array of scan points

        Returns:
            Updated coverage percentage
        """
        if self._voxel_grid is None:
            raise ValueError("CAD reference not set. Call set_cad_reference() first.")

        self._voxel_grid.mark_covered(scan_points, tolerance=self.tolerance)

        covered, total = self._voxel_grid.get_coverage()
        coverage = 100.0 * covered / total if total > 0 else 0.0

        return coverage

    def get_current_coverage(self) -> float:
        """Get current coverage percentage."""
        if self._voxel_grid is None:
            return 0.0

        covered, total = self._voxel_grid.get_coverage()
        return 100.0 * covered / total if total > 0 else 0.0

    def reset(self):
        """Reset coverage tracking (keep CAD reference)."""
        if self._voxel_grid:
            self._voxel_grid.reset_coverage()

    def compute_coverage(
        self,
        cad_points: np.ndarray,
        scan_points: np.ndarray,
    ) -> CoverageResult:
        """
        Compute coverage in one call (for simple use cases).

        Args:
            cad_points: Nx3 CAD surface points
            scan_points: Mx3 scan points

        Returns:
            CoverageResult with detailed statistics
        """
        start_time = time.time()

        # Validate inputs
        if len(cad_points) < MIN_POINTS_FOR_COVERAGE:
            return CoverageResult(
                coverage_percent=0.0,
                total_voxels=0,
                covered_voxels=0,
                uncovered_voxels=0,
                gap_clusters=[],
                largest_gap_size_mm=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                voxel_size_mm=self.voxel_size,
                tolerance_mm=self.tolerance,
                cad_points_count=len(cad_points),
                scan_points_count=len(scan_points),
            )

        # Save CAD points for gap location hints
        self._cad_points = cad_points

        # Build voxel grid
        voxel_grid = VoxelGrid(voxel_size=self.voxel_size)
        voxel_grid.build_from_points(cad_points)

        # Mark covered voxels
        if len(scan_points) > 0:
            voxel_grid.mark_covered(scan_points, tolerance=self.tolerance)

        # Compute coverage
        covered, total = voxel_grid.get_coverage()
        coverage_percent = 100.0 * covered / total if total > 0 else 0.0

        # Find gap clusters
        gap_clusters = []
        largest_gap_size = 0.0

        uncovered_points = voxel_grid.get_uncovered_voxels()
        if len(uncovered_points) > 0:
            gap_clusters, largest_gap_size = self._cluster_gaps(
                np.array(uncovered_points)
            )

        processing_time = (time.time() - start_time) * 1000

        return CoverageResult(
            coverage_percent=coverage_percent,
            total_voxels=total,
            covered_voxels=covered,
            uncovered_voxels=total - covered,
            gap_clusters=gap_clusters,
            largest_gap_size_mm=largest_gap_size,
            processing_time_ms=processing_time,
            voxel_size_mm=self.voxel_size,
            tolerance_mm=self.tolerance,
            cad_points_count=len(cad_points),
            scan_points_count=len(scan_points),
        )

    def _cluster_gaps(
        self,
        uncovered_points: np.ndarray,
        min_cluster_size: int = 5,
    ) -> Tuple[List[Dict], float]:
        """
        Cluster uncovered voxels into gap regions.

        Uses DBSCAN-like clustering based on voxel adjacency.
        """
        if len(uncovered_points) == 0:
            return [], 0.0

        # Simple clustering: use connected components based on distance
        # Two voxels are connected if distance <= sqrt(3) * voxel_size
        max_distance = np.sqrt(3) * self.voxel_size * 1.5

        tree = KDTree(uncovered_points)
        n_points = len(uncovered_points)

        # Find connected components
        visited = np.zeros(n_points, dtype=bool)
        clusters = []

        for i in range(n_points):
            if visited[i]:
                continue

            # BFS from this point
            cluster_indices = []
            queue = [i]

            while queue:
                idx = queue.pop(0)
                if visited[idx]:
                    continue

                visited[idx] = True
                cluster_indices.append(idx)

                # Find neighbors
                neighbors = tree.query_ball_point(uncovered_points[idx], max_distance)
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)

            if len(cluster_indices) >= min_cluster_size:
                cluster_points = uncovered_points[cluster_indices]

                # Compute cluster properties
                center = np.mean(cluster_points, axis=0)
                size = len(cluster_indices) * (self.voxel_size ** 3)  # Volume estimate

                # Approximate diameter
                if len(cluster_points) > 1:
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    diameter = 2 * np.max(distances)
                else:
                    diameter = self.voxel_size

                clusters.append({
                    "center": center.tolist(),
                    "size_voxels": len(cluster_indices),
                    "size_mm3": round(size, 1),
                    "diameter_mm": round(diameter, 1),
                    "location_hint": self._get_location_hint(center, uncovered_points),
                })

        # Find largest gap
        largest_gap = max([c["diameter_mm"] for c in clusters]) if clusters else 0.0

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c["size_voxels"], reverse=True)

        return clusters[:10], largest_gap  # Return top 10 gaps

    def _get_location_hint(
        self,
        center: np.ndarray,
        all_uncovered: np.ndarray,
    ) -> str:
        """
        Generate human-readable location hint for a gap.

        Returns descriptions like "top-left corner", "center", "right edge".
        """
        if self._cad_points is None or len(self._cad_points) == 0:
            return "center"

        # Use CAD bounding box for normalization (not uncovered points)
        min_pt = np.min(self._cad_points, axis=0)
        max_pt = np.max(self._cad_points, axis=0)
        extents = max_pt - min_pt

        # Normalize center position to [0, 1] range
        # Handle zero extents (e.g., flat plates where z=0)
        normalized = np.zeros(3)
        for i in range(3):
            if extents[i] > 0:
                normalized[i] = (center[i] - min_pt[i]) / extents[i]
            else:
                normalized[i] = 0.5  # Default to center for zero-extent dimensions

        # Generate description
        x_pos = normalized[0]
        y_pos = normalized[1]
        z_pos = normalized[2]

        # Horizontal position (left/center/right)
        if x_pos < 0.33:
            h_desc = "left"
        elif x_pos > 0.67:
            h_desc = "right"
        else:
            h_desc = "center"

        # Vertical position (front/center/back or top/bottom)
        if y_pos < 0.33:
            v_desc = "front"
        elif y_pos > 0.67:
            v_desc = "back"
        else:
            v_desc = ""

        # Combine
        if h_desc == "center" and v_desc == "":
            return "center"
        elif v_desc:
            return f"{v_desc}-{h_desc}" if h_desc != "center" else v_desc
        else:
            return h_desc


def generate_scan_guidance(coverage_result: CoverageResult) -> List[str]:
    """
    Generate human-readable scanning guidance based on coverage gaps.

    Args:
        coverage_result: Result from coverage calculation

    Returns:
        List of guidance messages
    """
    messages = []

    if coverage_result.coverage_percent >= 95:
        messages.append("Coverage is excellent (95%+). Scan is complete.")
        return messages

    if coverage_result.coverage_percent < 50:
        messages.append(
            f"Coverage is low ({coverage_result.coverage_percent:.0f}%). "
            "Continue scanning more of the part."
        )
    elif coverage_result.coverage_percent < 80:
        messages.append(
            f"Coverage is {coverage_result.coverage_percent:.0f}%. "
            "Scan the following areas to complete:"
        )
    else:
        messages.append(
            f"Coverage is {coverage_result.coverage_percent:.0f}%. "
            "Almost done - scan these remaining areas:"
        )

    # Add specific guidance for each gap
    for i, gap in enumerate(coverage_result.gap_clusters[:5]):
        location = gap.get("location_hint", "unknown area")
        size = gap.get("diameter_mm", 0)

        if size > 50:
            messages.append(f"  - Large gap at {location} (~{size:.0f}mm)")
        elif size > 20:
            messages.append(f"  - Medium gap at {location}")
        else:
            messages.append(f"  - Small gap at {location}")

    return messages


def compute_coverage(
    cad_points: np.ndarray,
    scan_points: np.ndarray,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
) -> CoverageResult:
    """
    Convenience function to compute coverage.

    Args:
        cad_points: Nx3 CAD surface points
        scan_points: Mx3 scan points
        voxel_size: Voxel size in mm
        tolerance: Coverage tolerance in mm

    Returns:
        CoverageResult
    """
    calc = CoverageCalculator(voxel_size=voxel_size, tolerance=tolerance)
    return calc.compute_coverage(cad_points, scan_points)


# Test code
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("="*60)
    print("COVERAGE CALCULATOR TEST")
    print("="*60)

    np.random.seed(42)

    # Test 1: Synthetic full coverage
    print("\n[1] SYNTHETIC FULL COVERAGE")

    # Create CAD points (flat plate)
    n_cad = 5000
    cad_x = np.random.uniform(0, 100, n_cad)
    cad_y = np.random.uniform(0, 50, n_cad)
    cad_z = np.zeros(n_cad)  # Flat plate at z=0
    cad_points = np.column_stack([cad_x, cad_y, cad_z])

    # Create scan points (full coverage with noise)
    scan_points = cad_points + np.random.normal(0, 0.5, cad_points.shape)

    result = compute_coverage(cad_points, scan_points)
    print(f"    Coverage: {result.coverage_percent:.1f}%")
    print(f"    Voxels: {result.covered_voxels}/{result.total_voxels}")
    print(f"    Time: {result.processing_time_ms:.0f}ms")

    if result.coverage_percent >= 90:
        print("    ✓ PASS: Full coverage detected")
    else:
        print("    ✗ FAIL: Expected >90% coverage")

    # Test 2: Partial coverage (50%)
    print("\n[2] PARTIAL COVERAGE (50%)")

    # Only scan left half of plate
    partial_mask = cad_x < 50
    partial_scan = cad_points[partial_mask] + np.random.normal(0, 0.5, (sum(partial_mask), 3))

    result = compute_coverage(cad_points, partial_scan)
    print(f"    Coverage: {result.coverage_percent:.1f}%")
    print(f"    Gap clusters: {len(result.gap_clusters)}")
    print(f"    Largest gap: {result.largest_gap_size_mm:.1f}mm")

    if 40 <= result.coverage_percent <= 60:
        print("    ✓ PASS: ~50% coverage detected")
    else:
        print("    ✗ FAIL: Expected ~50% coverage")

    # Test 3: No coverage
    print("\n[3] NO COVERAGE")

    result = compute_coverage(cad_points, np.array([]).reshape(0, 3))
    print(f"    Coverage: {result.coverage_percent:.1f}%")

    if result.coverage_percent == 0:
        print("    ✓ PASS: 0% coverage detected")
    else:
        print("    ✗ FAIL: Expected 0% coverage")

    # Test 4: Real files if available
    print("\n[4] REAL FILE TEST")

    cad_file = "/Users/idant/Downloads/44211000_A.stl"
    scan_file = "/Users/idant/Downloads/44211000_A.ply"

    import os
    if HAS_OPEN3D and os.path.exists(cad_file) and os.path.exists(scan_file):

        mesh = o3d.io.read_triangle_mesh(cad_file)
        pcd_cad = mesh.sample_points_uniformly(number_of_points=20000)
        cad_pts = np.asarray(pcd_cad.points)

        pcd_scan = o3d.io.read_point_cloud(scan_file)
        scan_pts = np.asarray(pcd_scan.points)

        result = compute_coverage(cad_pts, scan_pts)
        print(f"    Part: 44211000_A")
        print(f"    CAD points: {result.cad_points_count}")
        print(f"    Scan points: {result.scan_points_count}")
        print(f"    Coverage: {result.coverage_percent:.1f}%")
        print(f"    Time: {result.processing_time_ms:.0f}ms")
        print(f"    Gap clusters: {len(result.gap_clusters)}")

        if result.coverage_percent > 0:
            print("    ✓ PASS: Coverage computed")
        else:
            print("    ⚠ Coverage is 0% - scan may need alignment")
    else:
        print("    ⚠ Skipped (Open3D or files not available)")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
