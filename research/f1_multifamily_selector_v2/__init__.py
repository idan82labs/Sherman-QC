"""Deterministic source-only multi-family selector V2."""

from .model import (
    FEATURE_NAMES,
    NMS_ANGLE_DEG,
    NMS_DISTANCE_MM,
    NMS_OVERLAP,
    SCORE_THRESHOLD,
    build_feature_matrix,
    endpoints,
    select_indices,
)

__all__ = [
    "FEATURE_NAMES",
    "NMS_ANGLE_DEG",
    "NMS_DISTANCE_MM",
    "NMS_OVERLAP",
    "SCORE_THRESHOLD",
    "build_feature_matrix",
    "endpoints",
    "select_indices",
]
