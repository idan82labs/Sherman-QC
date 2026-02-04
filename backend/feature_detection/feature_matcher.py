"""
Feature Matcher Module

Matches XLSX dimension specifications to CAD features.
Enables accurate three-way comparison by linking:
- XLSX spec (engineering specification) -> CAD feature (design) -> Scan measurement (actual)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from .feature_types import (
    FeatureType,
    MeasurableFeature,
    HoleFeature,
    LinearFeature,
    AngleFeature,
    FeatureRegistry,
    FeatureMatch,
)

try:
    from dimension_parser import DimensionSpec, DimensionType
    HAS_DIMENSION_PARSER = True
except ImportError:
    HAS_DIMENSION_PARSER = False

logger = logging.getLogger(__name__)


def dimension_type_to_feature_type(dim_type: 'DimensionType') -> FeatureType:
    """Convert DimensionType enum to FeatureType enum."""
    if not HAS_DIMENSION_PARSER:
        return FeatureType.LINEAR

    mapping = {
        DimensionType.LINEAR: FeatureType.LINEAR,
        DimensionType.ANGLE: FeatureType.ANGLE,
        DimensionType.RADIUS: FeatureType.RADIUS,
        DimensionType.DIAMETER: FeatureType.DIAMETER,
    }
    return mapping.get(dim_type, FeatureType.LINEAR)


class FeatureMatcher:
    """
    Match XLSX dimensions to CAD features.

    Matching strategy:
    1. Filter by type (linear->linear, diameter->hole, etc.)
    2. Filter by value (within tolerance)
    3. Rank by value closeness
    4. Resolve ambiguity by position ordering
    """

    def __init__(
        self,
        value_tolerance: float = 5.0,  # Increased for better linear matching
        angle_tolerance: float = 10.0,
        position_weight: float = 0.3,
    ):
        """
        Initialize matcher.

        Args:
            value_tolerance: Maximum difference for value matching (mm or degrees)
            angle_tolerance: Specific tolerance for angle matching (degrees)
            position_weight: Weight for position-based disambiguation (0-1)
        """
        self.value_tolerance = value_tolerance
        self.angle_tolerance = angle_tolerance
        self.position_weight = position_weight

    def match(
        self,
        xlsx_dims: List['DimensionSpec'],
        cad_features: FeatureRegistry,
    ) -> Tuple[List[FeatureMatch], List['DimensionSpec'], List[MeasurableFeature]]:
        """
        Match XLSX dimensions to CAD features.

        Args:
            xlsx_dims: List of DimensionSpec from XLSX parsing
            cad_features: FeatureRegistry from CAD extraction

        Returns:
            Tuple of (matches, unmatched_xlsx, unmatched_cad)
        """
        if not xlsx_dims:
            return [], [], cad_features.get_all_features()

        matches = []
        unmatched_xlsx = []
        used_cad_features = set()

        # Sort XLSX dimensions by ID for consistent ordering
        sorted_xlsx = sorted(xlsx_dims, key=lambda d: d.dim_id)

        # Group features by type for efficient matching
        type_groups = self._group_features_by_type(cad_features)

        # Match each XLSX dimension
        for xlsx_dim in sorted_xlsx:
            feature_type = dimension_type_to_feature_type(xlsx_dim.dim_type)

            # Get candidate CAD features of matching type
            candidates = type_groups.get(feature_type, [])

            # Filter candidates not already used
            available = [c for c in candidates if id(c) not in used_cad_features]

            if not available:
                # No features of this type available
                unmatched_xlsx.append(xlsx_dim)
                continue

            # Find best match
            match = self._match_single(xlsx_dim, available, feature_type)

            if match:
                matches.append(match)
                used_cad_features.add(id(match.cad_feature))
            else:
                unmatched_xlsx.append(xlsx_dim)

        # Collect unmatched CAD features
        all_cad = cad_features.get_all_features()
        unmatched_cad = [f for f in all_cad if id(f) not in used_cad_features]

        logger.info(
            f"Feature matching: {len(matches)} matched, "
            f"{len(unmatched_xlsx)} unmatched XLSX, "
            f"{len(unmatched_cad)} unmatched CAD"
        )

        return matches, unmatched_xlsx, unmatched_cad

    def _group_features_by_type(
        self,
        cad_features: FeatureRegistry,
    ) -> Dict[FeatureType, List[MeasurableFeature]]:
        """Group CAD features by their type."""
        groups = {
            FeatureType.LINEAR: list(cad_features.linear_dims),
            FeatureType.DIAMETER: list(cad_features.holes),
            FeatureType.RADIUS: list(cad_features.radii),
            FeatureType.ANGLE: list(cad_features.angles),
        }
        return groups

    def _match_single(
        self,
        xlsx_dim: 'DimensionSpec',
        candidates: List[MeasurableFeature],
        feature_type: FeatureType,
    ) -> Optional[FeatureMatch]:
        """
        Find the best matching CAD feature for an XLSX dimension.

        Scoring:
        - value_score = 1 - |cad - xlsx| / xlsx (closer values score higher)
        - axis_bonus: +0.2 if axis matches (for linear dimensions)
        - For multiple close matches, use position ordering
        """
        if not candidates:
            return None

        xlsx_value = xlsx_dim.value
        xlsx_axis = getattr(xlsx_dim, 'axis', None)  # X, Y, Z, or None

        # Get appropriate tolerance
        if feature_type == FeatureType.ANGLE:
            tolerance = self.angle_tolerance
        else:
            tolerance = self.value_tolerance

        # Score each candidate
        scored_candidates = []
        for cad_feature in candidates:
            value_diff = abs(cad_feature.nominal_value - xlsx_value)

            # Also check for supplementary angle (180 - angle) for angles
            if feature_type == FeatureType.ANGLE:
                supplementary_diff = abs((180 - cad_feature.nominal_value) - xlsx_value)
                value_diff = min(value_diff, supplementary_diff)

            if value_diff > tolerance:
                continue

            # Compute value score (0-1, higher is better)
            if xlsx_value > 0:
                value_score = 1.0 - (value_diff / xlsx_value)
            else:
                value_score = 1.0 - (value_diff / (tolerance + 1e-10))

            value_score = max(0.0, min(1.0, value_score))

            # Axis matching bonus for linear dimensions
            axis_match = False
            if xlsx_axis and feature_type == FeatureType.LINEAR and hasattr(cad_feature, 'direction'):
                # Check if CAD feature direction aligns with specified axis
                direction = np.asarray(cad_feature.direction)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                axis_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(xlsx_axis.upper())
                if axis_idx is not None:
                    # Feature direction should be mostly along this axis
                    axis_alignment = abs(direction[axis_idx])
                    if axis_alignment > 0.7:  # More than ~45° alignment
                        axis_match = True
                        value_score = min(1.0, value_score + 0.2)  # Bonus for axis match

            scored_candidates.append({
                "feature": cad_feature,
                "value_diff": value_diff,
                "value_score": value_score,
                "axis_match": axis_match,
            })

        if not scored_candidates:
            return None

        # Sort by value score (descending)
        scored_candidates.sort(key=lambda x: -x["value_score"])

        # If only one candidate or clear winner, use it
        if len(scored_candidates) == 1:
            best = scored_candidates[0]
        elif scored_candidates[0]["value_score"] - scored_candidates[1]["value_score"] > 0.1:
            best = scored_candidates[0]
        else:
            # Multiple close candidates - use position ordering
            best = self._resolve_ambiguity(xlsx_dim, scored_candidates)

        cad_feature = best["feature"]

        # Create FeatureMatch
        match = FeatureMatch(
            dim_id=xlsx_dim.dim_id,
            xlsx_spec=xlsx_dim,
            cad_feature=cad_feature,
            xlsx_value=xlsx_value,
            cad_value=cad_feature.nominal_value,
            tolerance_plus=xlsx_dim.tolerance_plus,
            tolerance_minus=xlsx_dim.tolerance_minus,
            match_confidence=best["value_score"],
            feature_type=feature_type.value,
            unit=xlsx_dim.unit,
            description=xlsx_dim.description or f"Dimension {xlsx_dim.dim_id}",
            axis=xlsx_axis,  # Pass axis information for linear measurement
        )

        return match

    def _resolve_ambiguity(
        self,
        xlsx_dim: 'DimensionSpec',
        candidates: List[Dict],
    ) -> Dict:
        """
        Resolve ambiguity when multiple CAD features have similar scores.

        Strategy:
        1. First filter to only top-scoring candidates (within 0.1 of best)
        2. Use spatial position ordering for tie-breaking
        """
        if len(candidates) <= 1:
            return candidates[0] if candidates else None

        # IMPORTANT: Only consider candidates with scores close to the best
        best_score = candidates[0]["value_score"]
        top_candidates = [c for c in candidates if best_score - c["value_score"] < 0.1]

        # If only one top candidate after filtering, use it
        if len(top_candidates) == 1:
            return top_candidates[0]

        # Get positions of top candidates only
        positions = []
        for c in top_candidates:
            feature = c["feature"]
            if hasattr(feature, "position"):
                positions.append(feature.position)
            elif hasattr(feature, "apex_point"):
                positions.append(feature.apex_point)
            else:
                positions.append(np.zeros(3))

        positions = np.array(positions)

        # Find primary axis (largest extent)
        if len(positions) > 1:
            extent = positions.max(axis=0) - positions.min(axis=0)
            primary_axis = np.argmax(extent)

            # Sort by position along primary axis
            sorted_indices = np.argsort(positions[:, primary_axis])

            # Map dim_id to expected position order
            # Lower dim_id -> earlier in spatial order
            expected_order_idx = min(xlsx_dim.dim_id - 1, len(sorted_indices) - 1)
            expected_order_idx = max(0, expected_order_idx)

            return top_candidates[sorted_indices[expected_order_idx]]

        return top_candidates[0]

    def match_by_value_only(
        self,
        xlsx_value: float,
        xlsx_type: 'DimensionType',
        cad_features: FeatureRegistry,
        tolerance: float = None,
    ) -> Optional[MeasurableFeature]:
        """
        Simple value-based matching without considering position.

        Useful for quick lookups or when position info is not needed.
        """
        if tolerance is None:
            tolerance = self.value_tolerance

        feature_type = dimension_type_to_feature_type(xlsx_type)

        candidates = cad_features.find_by_type_and_value(
            feature_type, xlsx_value, tolerance
        )

        if not candidates:
            return None

        # Return closest match
        candidates.sort(key=lambda c: abs(c.nominal_value - xlsx_value))
        return candidates[0]

    def suggest_matches(
        self,
        xlsx_dims: List['DimensionSpec'],
        cad_features: FeatureRegistry,
        max_suggestions: int = 3,
    ) -> Dict[int, List[Dict]]:
        """
        Suggest possible matches for each XLSX dimension.

        Returns a dict mapping dim_id to list of possible matches with scores.
        Useful for interactive disambiguation.
        """
        suggestions = {}

        type_groups = self._group_features_by_type(cad_features)

        for xlsx_dim in xlsx_dims:
            feature_type = dimension_type_to_feature_type(xlsx_dim.dim_type)
            candidates = type_groups.get(feature_type, [])

            dim_suggestions = []
            for cad_feature in candidates:
                value_diff = abs(cad_feature.nominal_value - xlsx_dim.value)

                # Allow wider tolerance for suggestions
                tolerance = self.value_tolerance * 2
                if feature_type == FeatureType.ANGLE:
                    tolerance = self.angle_tolerance * 2

                if value_diff > tolerance:
                    continue

                if xlsx_dim.value > 0:
                    score = 1.0 - (value_diff / xlsx_dim.value)
                else:
                    score = 1.0 - (value_diff / (tolerance + 1e-10))

                score = max(0.0, min(1.0, score))

                dim_suggestions.append({
                    "feature_id": cad_feature.feature_id,
                    "cad_value": cad_feature.nominal_value,
                    "value_diff": value_diff,
                    "score": score,
                    "position": cad_feature.position.tolist() if hasattr(cad_feature, "position") else None,
                })

            # Sort by score and limit
            dim_suggestions.sort(key=lambda x: -x["score"])
            suggestions[xlsx_dim.dim_id] = dim_suggestions[:max_suggestions]

        return suggestions


def match_xlsx_to_cad_features(
    xlsx_path: str,
    mesh_path: str,
) -> Tuple[List[FeatureMatch], List['DimensionSpec'], List[MeasurableFeature]]:
    """
    Convenience function to match XLSX to CAD features from file paths.

    Args:
        xlsx_path: Path to XLSX dimension file
        mesh_path: Path to CAD mesh file (PLY, STL, OBJ)

    Returns:
        Tuple of (matches, unmatched_xlsx, unmatched_cad)
    """
    from dimension_parser import parse_dimension_file
    from .cad_feature_extractor import CADFeatureExtractor

    # Parse XLSX
    xlsx_result = parse_dimension_file(xlsx_path)
    if not xlsx_result.success:
        logger.error(f"Failed to parse XLSX: {xlsx_result.error}")
        return [], [], []

    # Extract CAD features
    extractor = CADFeatureExtractor()
    cad_features = extractor.extract_from_file(mesh_path)

    # Match
    matcher = FeatureMatcher()
    return matcher.match(xlsx_result.dimensions, cad_features)
