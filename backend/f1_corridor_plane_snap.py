"""Snap corridor-owned axes only under Pareto incident-plane improvement."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np


_CORRIDOR_KEY = re.compile(r"scan_corridor_completion::ZS(\d+)")


def _line_to_plane_intersection_distance(
    point: np.ndarray,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float | None:
    first_normal = np.asarray(first.get("normal"), dtype=np.float64)
    second_normal = np.asarray(second.get("normal"), dtype=np.float64)
    if first_normal.shape != (3,) or second_normal.shape != (3,):
        return None
    axis = np.cross(first_normal, second_normal)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis /= norm
    matrix = np.vstack((first_normal, second_normal))
    try:
        anchor = matrix.T @ np.linalg.solve(
            matrix @ matrix.T,
            -np.asarray((float(first["d"]), float(second["d"])), dtype=np.float64),
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    delta = point - anchor
    return float(np.linalg.norm(delta - axis * np.dot(delta, axis)))


def snap_corridor_axes_by_plane_pareto(
    selected: Sequence[Mapping[str, Any]],
    bend_hypotheses: Sequence[Mapping[str, Any]],
    flange_hypotheses: Sequence[Mapping[str, Any]],
    *,
    minimum_pair_improvement_mm: float = 0.75,
    minimum_mean_improvement_mm: float = 1.25,
    minimum_plane_pair_count: int = 2,
) -> dict[str, Any]:
    """Restore source geometry only when every incident-plane test improves."""

    bends = {str(item.get("bend_id")): dict(item) for item in bend_hypotheses}
    flanges = {str(item.get("flange_id")): dict(item) for item in flange_hypotheses}
    output = []
    replacements = []
    for source in selected:
        owned = dict(source)
        match = _CORRIDOR_KEY.fullmatch(str(owned.get("canonical_bend_key") or ""))
        hypothesis = bends.get(f"CORRIDOR_ZS{match.group(1)}") if match else None
        if hypothesis is None:
            output.append(owned)
            continue
        try:
            owned_anchor = np.asarray(owned["anchor"], dtype=np.float64)
            source_anchor = np.asarray(hypothesis["anchor"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            output.append(owned)
            continue
        improvements = []
        for pair in hypothesis.get("candidate_incident_flange_pairs") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            first = flanges.get(str(pair[0]))
            second = flanges.get(str(pair[1]))
            if first is None or second is None:
                continue
            owned_distance = _line_to_plane_intersection_distance(owned_anchor, first, second)
            source_distance = _line_to_plane_intersection_distance(source_anchor, first, second)
            if owned_distance is None or source_distance is None:
                continue
            improvements.append(owned_distance - source_distance)
        if (
            len(improvements) >= minimum_plane_pair_count
            and min(improvements) >= minimum_pair_improvement_mm
            and float(np.mean(improvements)) >= minimum_mean_improvement_mm
        ):
            axis = np.asarray(hypothesis.get("axis_direction"), dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            span = float(hypothesis.get("span_mm") or 0.0)
            if axis.shape == (3,) and norm > 1e-9 and span > 0.0:
                axis /= norm
                half = 0.5 * span * axis
                owned["ownership_refit_anchor"] = owned.get("anchor")
                owned["anchor"] = source_anchor.tolist()
                owned["axis_direction"] = axis.tolist()
                owned["span_endpoints"] = [(source_anchor - half).tolist(), (source_anchor + half).tolist()]
                owned["selection_reason"] = "corridor_source_plane_pareto_snap"
                replacements.append(
                    {
                        "owned_bend_id": owned.get("bend_id"),
                        "source_hypothesis_id": hypothesis.get("bend_id"),
                        "plane_pair_count": len(improvements),
                        "minimum_improvement_mm": min(improvements),
                        "mean_improvement_mm": float(np.mean(improvements)),
                    }
                )
        output.append(owned)
    return {
        "selected": output,
        "replacements": replacements,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
