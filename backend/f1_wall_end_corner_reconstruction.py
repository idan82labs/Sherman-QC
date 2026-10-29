"""Reconstruct wall-end corner folds from a bounded scan sample."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(item: Mapping[str, Any]) -> np.ndarray:
    value = item.get("span_endpoints")
    if value is None:
        value = [item["line_start"], item["line_end"]]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("bend endpoints must be finite with shape (2, 3)")
    return result


def _stable_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample = points - center
    _values, vectors = np.linalg.eigh(sample.T @ sample / max(len(sample), 1))
    basis = vectors[:, ::-1]
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, 2] *= -1.0
    return center, basis, sample @ basis


def _plane_signature(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    planes = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "plane_lattice"
        and len(item.get("incident_plane_ids") or []) == 2
    ]
    if len(planes) != 6:
        return None
    adjacency: dict[str, set[str]] = {}
    by_edge: dict[frozenset[str], dict[str, Any]] = {}
    for item in planes:
        left, right = (str(value) for value in item["incident_plane_ids"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        by_edge[frozenset((left, right))] = item
    if sorted(len(values) for values in adjacency.values()) != [1, 1, 2, 2, 2, 4]:
        return None
    hubs = [node for node, neighbors in adjacency.items() if len(neighbors) == 4]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    base_options = []
    for neighbor in adjacency[hub]:
        edge = by_edge[frozenset((hub, neighbor))]
        if (
            len(adjacency.get(neighbor, set())) == 2
            and float(edge.get("span_mm") or 0.0) >= 150.0
            and float(edge.get("two_sided_transition_score") or 0.0) >= 0.65
        ):
            base_options.append((neighbor, edge))
    if len(base_options) != 2:
        return None
    base_options.sort(
        key=lambda pair: (
            str(pair[0]),
            str(pair[1].get("candidate_id") or ""),
        )
    )
    common_caps = (
        (adjacency[base_options[0][0]] - {hub})
        & (adjacency[base_options[1][0]] - {hub})
    )
    if len(common_caps) != 1:
        return None
    cap = next(iter(common_caps))
    returns = [
        by_edge[frozenset((neighbor, cap))]
        for neighbor, _base in base_options
        if frozenset((neighbor, cap)) in by_edge
    ]
    if len(returns) != 2:
        return None
    returns.sort(
        key=lambda item: (
            -float(item.get("two_sided_transition_score") or 0.0),
            str(item.get("candidate_id") or ""),
        )
    )
    best_score = float(returns[0].get("two_sided_transition_score") or 0.0)
    runner_up_score = float(returns[1].get("two_sided_transition_score") or 0.0)
    if best_score < 0.60 or best_score - runner_up_score < 0.10:
        return None
    return {
        "hub_plane_id": hub,
        "cap_plane_id": cap,
        "base_lines": [item for _neighbor, item in base_options],
        "selected_return": returns[0],
        "rejected_return": returns[1],
    }


def has_wall_end_corner_signature(candidates: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether candidates expose the unambiguous wall/cap plane graph."""

    return _plane_signature(candidates) is not None


def reconstruct_wall_end_corners(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    points: np.ndarray | None,
    *,
    point_budget: int = 200_000,
) -> dict[str, Any]:
    """Add one top return and four finite vertical wall-end folds."""

    source = [dict(item) for item in selected]
    signature = _plane_signature(candidates)
    if signature is None:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "wall_end_plane_graph_signature_failed",
            "transient_input_point_count": 0,
            "transient_sampled_point_count": 0,
            "point_budget": point_budget,
        }
    if points is None:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "scan_points_unavailable",
            "transient_input_point_count": 0,
            "transient_sampled_point_count": 0,
            "point_budget": point_budget,
        }
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    points = points[np.isfinite(points).all(axis=1)]
    input_point_count = len(points)
    if len(points) > point_budget:
        indices = np.linspace(0, len(points) - 1, point_budget, dtype=np.int64)
        points = points[indices]
    if len(points) < 1_000:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "insufficient_scan_points",
            "transient_input_point_count": input_point_count,
            "transient_sampled_point_count": len(points),
            "point_budget": point_budget,
        }
    center, basis, pca_points = _stable_pca(points)
    base_lines = list(signature["base_lines"])
    base_pca = [(_endpoints(item) - center) @ basis for item in base_lines]
    return_pca = (_endpoints(signature["selected_return"]) - center) @ basis
    wall_centers = [float(np.mean(line[:, 0])) for line in base_pca]
    return_wall_center = float(np.mean(return_pca[:, 0]))
    z_low, z_high = np.quantile(pca_points[:, 2], (0.001, 0.999))
    z_range = float(z_high - z_low)
    if z_range < 20.0:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "insufficient_wall_height",
            "transient_input_point_count": input_point_count,
            "transient_sampled_point_count": len(points),
            "point_budget": point_budget,
        }
    corners = []
    for wall_index, wall_center in enumerate(wall_centers, start=1):
        has_selected_return = abs(wall_center - return_wall_center) <= 10.0
        elevated = (
            (np.abs(pca_points[:, 0] - wall_center) <= 5.0)
            & (pca_points[:, 2] >= z_low + 0.33 * z_range)
            & (pca_points[:, 2] <= z_low + 0.88 * z_range)
        )
        wall_y = pca_points[elevated, 1]
        if len(wall_y) < 100:
            return {
                "selected": source,
                "reconstructed": [],
                "activated": False,
                "reason": "insufficient_elevated_wall_support",
                "transient_input_point_count": input_point_count,
                "transient_sampled_point_count": len(points),
                "point_budget": point_budget,
            }
        lower_y = float(np.quantile(wall_y, 0.01))
        upper_y = float(np.quantile(wall_y, 0.99 if has_selected_return else 0.95))
        for end_index, y_coordinate in enumerate((lower_y, upper_y), start=1):
            local = (
                (np.abs(pca_points[:, 0] - wall_center) <= 4.0)
                & (np.abs(pca_points[:, 1] - y_coordinate) <= 4.0)
            )
            local_points = pca_points[local]
            if len(local_points) < 100:
                return {
                    "selected": source,
                    "reconstructed": [],
                    "activated": False,
                    "reason": "insufficient_local_corner_support",
                    "transient_input_point_count": input_point_count,
                    "transient_sampled_point_count": len(points),
                    "point_budget": point_budget,
                }
            x_coordinate = float(np.median(local_points[:, 0]))
            corner_z_low, corner_z_high = np.quantile(local_points[:, 2], (0.005, 0.995))
            endpoints_pca = np.asarray(
                [
                    [x_coordinate, y_coordinate, corner_z_low],
                    [x_coordinate, y_coordinate, corner_z_high],
                ],
                dtype=np.float64,
            )
            endpoints = endpoints_pca @ basis.T + center
            corners.append(
                {
                    "bend_id": f"WEC{wall_index}_{end_index}",
                    "candidate_id": f"wall_end_corner::{wall_index}_{end_index}",
                    "candidate_source": "wall_end_corner_reconstruction",
                    "span_endpoints": endpoints.tolist(),
                    "line_start": endpoints[0].tolist(),
                    "line_end": endpoints[1].tolist(),
                    "anchor": np.mean(endpoints, axis=0).tolist(),
                    "axis_direction": basis[:, 2].tolist(),
                    "span_mm": float(corner_z_high - corner_z_low),
                    "wall_has_selected_top_return": has_selected_return,
                    "wall_end_quantile": 0.01 if end_index == 1 else (0.99 if has_selected_return else 0.95),
                }
            )
    selected_return = dict(signature["selected_return"])
    selected_return["candidate_source"] = "wall_end_top_return_completion"
    reconstructed = source + [selected_return] + corners
    return {
        "selected": reconstructed,
        "reconstructed": [selected_return] + corners,
        "activated": True,
        "reason": "wall_end_plane_graph_and_bounded_corner_support",
        "transient_input_point_count": input_point_count,
        "transient_sampled_point_count": len(points),
        "point_budget": point_budget,
        "selected_return_candidate_id": selected_return.get("candidate_id"),
    }
