"""Identity-free geometry features for scan-mesh adjacency candidates.

The features in this module are computed from one candidate's physical
geometry only.  They deliberately do not consume a part identifier, an
expected bend count, labels, or any corpus-level statistic.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def log_minimum_surface_area_span_ratio(candidate: Mapping[str, Any]) -> float:
    """Return ``log1p(min(surface areas) / interface span**2)``.

    A long adjacency attached to a very small segmented patch has a low value,
    while a forming interface backed by two surfaces of meaningful extent has
    a higher value.  The raw ratio is dimensionless and therefore invariant to
    a uniform change of mesh units.  Invalid or incomplete geometry fails
    closed to ``0.0`` so the selector can impute/treat it as no positive
    evidence rather than receiving NaN or infinity.
    """

    try:
        first_area = float(candidate.get("surface1_area") or 0.0)
        second_area = float(candidate.get("surface2_area") or 0.0)
        span_mm = float(candidate.get("span_mm") or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if not all(math.isfinite(value) for value in (first_area, second_area, span_mm)):
        return 0.0
    if first_area <= 0.0 or second_area <= 0.0 or span_mm <= 0.0:
        return 0.0

    return math.log1p(min(first_area, second_area) / (span_mm * span_mm))
