"""
Bend Matcher Module

Matches XLSX dimension specifications to detected bends in CAD and scan.

Pipeline:
1. Parse XLSX to get expected bend angles
2. Detect bends in CAD
3. Match XLSX bends to CAD bends by angle proximity
4. After scan alignment, measure bends at matched positions
5. Generate comparison report
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

from dimension_parser import DimensionSpec, DimensionType, DimensionParseResult

logger = logging.getLogger(__name__)


@dataclass
class BendMatch:
    """A matched bend between XLSX spec, CAD, and scan."""
    dim_id: int
    dim_spec: DimensionSpec  # From XLSX

    # CAD bend info
    cad_angle: Optional[float] = None
    cad_radius: Optional[float] = None
    cad_position: Optional[Tuple[float, float, float]] = None
    cad_bend_id: Optional[int] = None
    cad_confidence: float = 0.0

    # Scan bend info (measured at CAD position after alignment)
    scan_angle: Optional[float] = None
    scan_radius: Optional[float] = None
    scan_confidence: float = 0.0

    # Deviations
    cad_deviation: Optional[float] = None  # CAD angle - XLSX angle
    scan_deviation: Optional[float] = None  # Scan angle - XLSX angle

    # Status
    cad_matched: bool = False
    scan_measured: bool = False
    status: str = "pending"  # pending, pass, fail, warning

    def compute_deviations(self):
        """Compute deviations from XLSX spec."""
        if self.cad_angle is not None:
            self.cad_deviation = self.cad_angle - self.dim_spec.value

        if self.scan_angle is not None:
            self.scan_deviation = self.scan_angle - self.dim_spec.value

            # Determine status based on tolerance
            if abs(self.scan_deviation) <= self.dim_spec.tolerance_plus:
                self.status = "pass"
            elif abs(self.scan_deviation) <= self.dim_spec.tolerance_plus * 2:
                self.status = "warning"
            else:
                self.status = "fail"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dim_id": self.dim_id,
            "type": "bend_angle",
            "expected": self.dim_spec.value,
            "tolerance": f"±{self.dim_spec.tolerance_plus}°",
            "cad": {
                "angle": round(self.cad_angle, 2) if self.cad_angle else None,
                "radius": round(self.cad_radius, 2) if self.cad_radius else None,
                "position": [round(p, 2) for p in self.cad_position] if self.cad_position else None,
                "deviation": round(self.cad_deviation, 2) if self.cad_deviation else None,
                "matched": self.cad_matched,
            },
            "scan": {
                "angle": round(self.scan_angle, 2) if self.scan_angle else None,
                "radius": round(self.scan_radius, 2) if self.scan_radius else None,
                "deviation": round(self.scan_deviation, 2) if self.scan_deviation else None,
                "measured": self.scan_measured,
            },
            "status": self.status,
        }


@dataclass
class BendMatchResult:
    """Result of bend matching process."""
    success: bool
    matches: List[BendMatch] = field(default_factory=list)
    unmatched_xlsx: List[DimensionSpec] = field(default_factory=list)
    unmatched_cad: List[Dict] = field(default_factory=list)
    bend_radius_spec: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # Summary stats
    total_bends: int = 0
    matched_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

    def compute_summary(self):
        """Compute summary statistics."""
        self.total_bends = len(self.matches)
        self.matched_count = sum(1 for m in self.matches if m.cad_matched)
        self.pass_count = sum(1 for m in self.matches if m.status == "pass")
        self.fail_count = sum(1 for m in self.matches if m.status == "fail")
        self.warning_count = sum(1 for m in self.matches if m.status == "warning")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.compute_summary()
        return {
            "success": self.success,
            "summary": {
                "total_bends": self.total_bends,
                "matched": self.matched_count,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "warnings": self.warning_count,
            },
            "bend_radius_spec": self.bend_radius_spec,
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_xlsx": [d.to_dict() for d in self.unmatched_xlsx],
            "unmatched_cad_count": len(self.unmatched_cad),
            "warnings": self.warnings,
            "error": self.error,
        }


class BendMatcher:
    """
    Matches XLSX bend specifications to detected bends in CAD and scan.

    Usage:
        matcher = BendMatcher()
        result = matcher.match_xlsx_to_cad(xlsx_result, cad_bends)
        result = matcher.measure_scan_bends(result, aligned_scan_points, cad_points)
    """

    def __init__(
        self,
        angle_match_threshold: float = 10.0,  # Max angle difference for matching
        position_search_radius: float = 20.0,  # mm, radius to search for bend in scan
    ):
        """
        Initialize bend matcher.

        Args:
            angle_match_threshold: Maximum angle difference (degrees) to consider a match
            position_search_radius: Radius (mm) around CAD bend position to search in scan
        """
        self.angle_match_threshold = angle_match_threshold
        self.position_search_radius = position_search_radius

    def match_xlsx_to_cad(
        self,
        xlsx_result: DimensionParseResult,
        cad_bends: List[Dict],
    ) -> BendMatchResult:
        """
        Match XLSX bend specifications to detected CAD bends.

        Args:
            xlsx_result: Parsed XLSX result with bend angles
            cad_bends: List of detected bends from CAD, each with:
                - bend_id: int
                - angle_degrees: float
                - radius_mm: float
                - bend_apex: [x, y, z]
                - detection_confidence: float

        Returns:
            BendMatchResult with matches
        """
        if not xlsx_result.success:
            return BendMatchResult(
                success=False,
                error="XLSX parsing failed"
            )

        xlsx_bends = xlsx_result.bend_angles

        if not xlsx_bends:
            return BendMatchResult(
                success=False,
                error="No bend angles found in XLSX"
            )

        logger.info(f"Matching {len(xlsx_bends)} XLSX bends to {len(cad_bends)} CAD bends")

        # Create matches for each XLSX bend
        matches = []
        used_cad_indices = set()

        for xlsx_bend in xlsx_bends:
            match = BendMatch(
                dim_id=xlsx_bend.dim_id,
                dim_spec=xlsx_bend,
            )

            # Find best matching CAD bend by angle proximity
            best_cad_idx = None
            best_angle_diff = float('inf')

            for i, cad_bend in enumerate(cad_bends):
                if i in used_cad_indices:
                    continue

                cad_angle = cad_bend.get('angle_degrees', 0)
                angle_diff = abs(cad_angle - xlsx_bend.value)

                # Also check supplementary angle (180 - angle)
                supp_diff = abs((180 - cad_angle) - xlsx_bend.value)
                angle_diff = min(angle_diff, supp_diff)

                if angle_diff < best_angle_diff and angle_diff <= self.angle_match_threshold:
                    best_angle_diff = angle_diff
                    best_cad_idx = i

            if best_cad_idx is not None:
                cad_bend = cad_bends[best_cad_idx]
                used_cad_indices.add(best_cad_idx)

                # Check if we need to use supplementary angle
                cad_angle = cad_bend.get('angle_degrees', 0)
                if abs((180 - cad_angle) - xlsx_bend.value) < abs(cad_angle - xlsx_bend.value):
                    cad_angle = 180 - cad_angle

                match.cad_angle = cad_angle
                match.cad_radius = cad_bend.get('radius_mm')
                match.cad_position = tuple(cad_bend.get('bend_apex', [0, 0, 0]))
                match.cad_bend_id = cad_bend.get('bend_id')
                match.cad_confidence = cad_bend.get('detection_confidence', 0)
                match.cad_matched = True

                match.compute_deviations()

                logger.info(f"  Matched XLSX Dim {xlsx_bend.dim_id} ({xlsx_bend.value}°) "
                           f"to CAD Bend {match.cad_bend_id} ({match.cad_angle:.1f}°)")
            else:
                logger.warning(f"  No CAD match for XLSX Dim {xlsx_bend.dim_id} ({xlsx_bend.value}°)")

            matches.append(match)

        # Track unmatched CAD bends
        unmatched_cad = [
            cad_bends[i] for i in range(len(cad_bends))
            if i not in used_cad_indices
        ]

        # Track unmatched XLSX bends
        unmatched_xlsx = [m.dim_spec for m in matches if not m.cad_matched]

        warnings = []
        if unmatched_xlsx:
            warnings.append(f"{len(unmatched_xlsx)} XLSX bends could not be matched to CAD")
        if unmatched_cad:
            warnings.append(f"{len(unmatched_cad)} CAD bends were not matched to XLSX specs")

        result = BendMatchResult(
            success=True,
            matches=matches,
            unmatched_xlsx=unmatched_xlsx,
            unmatched_cad=unmatched_cad,
            bend_radius_spec=xlsx_result.bend_radius,
            warnings=warnings,
        )

        result.compute_summary()
        return result

    def measure_scan_bends(
        self,
        match_result: BendMatchResult,
        aligned_scan_points: np.ndarray,
        tolerance: float = 0.5,
    ) -> BendMatchResult:
        """
        Measure bend angles in aligned scan at CAD bend positions.

        Args:
            match_result: Result from match_xlsx_to_cad
            aligned_scan_points: Nx3 array of aligned scan points
            tolerance: Angle tolerance for pass/fail determination

        Returns:
            Updated BendMatchResult with scan measurements
        """
        if not match_result.success:
            return match_result

        logger.info(f"Measuring {len(match_result.matches)} bends in scan")

        for match in match_result.matches:
            if not match.cad_matched or match.cad_position is None:
                continue

            # Extract local region around CAD bend position
            position = np.array(match.cad_position)
            distances = np.linalg.norm(aligned_scan_points - position, axis=1)
            local_mask = distances < self.position_search_radius
            local_points = aligned_scan_points[local_mask]

            if len(local_points) < 50:
                logger.warning(f"  Dim {match.dim_id}: Not enough points at bend position")
                continue

            # Compute bend angle from local points
            scan_angle, scan_radius, confidence = self._compute_local_bend_angle(
                local_points, position
            )

            if scan_angle is not None:
                match.scan_angle = scan_angle
                match.scan_radius = scan_radius
                match.scan_confidence = confidence
                match.scan_measured = True
                match.compute_deviations()

                logger.info(f"  Dim {match.dim_id}: Scan angle = {scan_angle:.1f}° "
                           f"(deviation: {match.scan_deviation:+.2f}°, status: {match.status})")

        match_result.compute_summary()
        return match_result

    def _compute_local_bend_angle(
        self,
        points: np.ndarray,
        bend_center: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Compute bend angle from local point cloud around bend center.

        Uses PCA to find the two dominant planes and computes angle between them.

        Returns:
            Tuple of (angle_degrees, radius_mm, confidence)
        """
        try:
            from scipy.spatial import KDTree
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("sklearn/scipy required for bend measurement")
            return None, None, 0.0

        if len(points) < 50:
            return None, None, 0.0

        # Center points
        centered = points - bend_center

        # Try to separate into two surface groups using normals
        # First, estimate local normals
        tree = KDTree(points)
        normals = []

        sample_indices = np.random.choice(len(points), min(200, len(points)), replace=False)

        for idx in sample_indices:
            neighbors = tree.query_ball_point(points[idx], 5.0)
            if len(neighbors) < 5:
                continue

            neighbor_pts = points[neighbors]
            if len(neighbor_pts) < 3:
                continue

            # PCA to get normal
            pca = PCA(n_components=3)
            pca.fit(neighbor_pts - neighbor_pts.mean(axis=0))
            normal = pca.components_[2]  # Smallest variance = normal
            normals.append(normal)

        if len(normals) < 20:
            return None, None, 0.0

        normals = np.array(normals)

        # Cluster normals into two groups (two surfaces of the bend)
        # Handle sign ambiguity by using absolute dot product
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

            # Use the normals directly, but allow for flipping
            labels = kmeans.fit_predict(normals)

            # Get average normal for each cluster
            normal1 = normals[labels == 0].mean(axis=0)
            normal2 = normals[labels == 1].mean(axis=0)

            # Normalize
            normal1 = normal1 / np.linalg.norm(normal1)
            normal2 = normal2 / np.linalg.norm(normal2)

            # Compute angle between normals
            dot = np.clip(np.dot(normal1, normal2), -1, 1)
            angle_between_normals = np.degrees(np.arccos(abs(dot)))

            # The bend angle is 180° - angle between normals (for a typical bend)
            bend_angle = 180 - angle_between_normals

            # Estimate confidence from cluster separation
            cluster_sizes = [np.sum(labels == i) for i in range(2)]
            size_balance = min(cluster_sizes) / max(cluster_sizes) if max(cluster_sizes) > 0 else 0
            confidence = size_balance

            # Estimate radius from curvature (simplified)
            radius = self.position_search_radius / 2  # Rough estimate

            return bend_angle, radius, confidence

        except Exception as e:
            logger.warning(f"Bend angle computation failed: {e}")
            return None, None, 0.0


def match_bends(
    xlsx_path: str,
    cad_bends: List[Dict],
    aligned_scan_points: Optional[np.ndarray] = None,
) -> BendMatchResult:
    """
    Convenience function to match bends from XLSX to CAD and optionally measure scan.

    Args:
        xlsx_path: Path to XLSX dimension file
        cad_bends: List of detected CAD bends
        aligned_scan_points: Optional aligned scan points for measurement

    Returns:
        BendMatchResult
    """
    from dimension_parser import parse_dimension_file

    # Parse XLSX
    xlsx_result = parse_dimension_file(xlsx_path)
    if not xlsx_result.success:
        return BendMatchResult(success=False, error=xlsx_result.error)

    # Match to CAD
    matcher = BendMatcher()
    result = matcher.match_xlsx_to_cad(xlsx_result, cad_bends)

    # Measure scan if provided
    if aligned_scan_points is not None and result.success:
        result = matcher.measure_scan_bends(result, aligned_scan_points)

    return result
