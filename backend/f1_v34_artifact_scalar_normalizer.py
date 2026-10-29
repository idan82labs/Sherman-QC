"""Narrow V34 public-artifact NumPy scalar normalization.

This module is a post-detector serialization adapter. It must never be called by
or from the frozen V34 detector engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ArtifactScalarNormalizationError(TypeError):
    """Raised when a value is outside the exact rerun waiver boundary."""


def normalize_numpy_scalars(value: Any) -> Any:
    """Return the same structure with only approved NumPy scalars converted.

    Dictionary keys must already be exact Python strings. Arrays and every
    NumPy scalar family other than bool, integer, and floating fail closed.
    """

    if isinstance(value, (np.datetime64, np.timedelta64)):
        raise ArtifactScalarNormalizationError(
            f"unsupported public-artifact value type: {type(value).__module__}.{type(value).__name__}"
        )
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactScalarNormalizationError(
                    f"dictionary key must be an exact Python str, got {type(key).__name__}"
                )
            normalized[key] = normalize_numpy_scalars(item)
        return normalized
    if isinstance(value, list):
        return [normalize_numpy_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_numpy_scalars(item) for item in value)
    raise ArtifactScalarNormalizationError(
        f"unsupported public-artifact value type: {type(value).__module__}.{type(value).__name__}"
    )
