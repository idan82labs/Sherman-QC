"""Identity-free geometric vetoes for scan-mesh bend candidates.

The selector tunes the caps on training folds only.  This module deliberately
does not inspect a part identifier, an expected bend count, or manual truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


MESH_NORMAL_COHERENCE_CAP_GRID = (0.80, 0.85, 0.90, 0.95, 1.0)


def retains_mesh_adjacency_candidate(
    candidate: Mapping[str, Any],
    *,
    maximum_surface_gap_mm: float,
    maximum_two_sided_normal_coherence: float,
) -> bool:
    """Return whether a candidate passes the tuned scan-only mesh vetoes.

    The veto applies only to ``scan_mesh_adjacency`` candidates so topology
    completions and other independently generated scan-only lines remain
    unaffected.  Missing finite evidence fails closed.
    """

    if candidate.get("candidate_source") != "scan_mesh_adjacency":
        return True

    try:
        surface_gap = float(candidate["min_surface_gap"])
        normal_coherence = float(candidate["two_sided_normal_coherence"])
    except (KeyError, TypeError, ValueError):
        return False

    return (
        isfinite(surface_gap)
        and isfinite(normal_coherence)
        and surface_gap <= float(maximum_surface_gap_mm)
        and normal_coherence <= float(maximum_two_sided_normal_coherence)
    )
