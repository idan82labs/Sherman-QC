"""Research-only residual bend birth for the typed F1 support graph.

This is the engineering-fixed BIRTH_V1 slice specified in the 2026-08-04
professor follow-up.  It consumes only compact scan atoms, atom adjacency,
scan-derived planes, and the current exact residual ownership assignment.
It never consumes expected counts, part IDs, CAD, filenames, or truth.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from statistics import median
from typing import Any

import numpy as np


BIRTH_PROFILE = "BIRTH_V1"
INCIDENCE_PROFILE = "INCIDENCE_V1"
ENERGY_SCALE = 1_000_000
TOTAL_BIRTH_COLUMN_CAP = 64
BEND = "BEND_STRAIGHT_COUNTED"
RESIDUAL = "RESIDUAL_NONCOUNTED"


def _unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    vector = vector / norm
    for component in vector:
        if abs(float(component)) > 1e-12:
            vector = -vector if component < 0.0 else vector
            break
    # Negative zero is numerically harmless but changes canonical JSON bytes.
    vector[np.abs(vector) <= 1e-12] = 0.0
    return vector


def _unoriented_angle_deg(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(float(np.dot(_unit(left), _unit(right))))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _line_distance(point: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> float:
    delta = point - anchor
    return float(np.linalg.norm(delta - np.dot(delta, axis) * axis))


def _line_endpoints(path: Mapping[str, Any]) -> np.ndarray:
    endpoints = np.asarray(path.get("span_endpoints"), dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("latent path must provide finite span_endpoints")
    return endpoints


def _shorter_overlap_fraction(left: np.ndarray, right: np.ndarray, axis: np.ndarray) -> float:
    left_projection = np.sort(left @ axis)
    right_projection = np.sort(right @ axis)
    overlap = max(
        0.0,
        min(float(left_projection[1]), float(right_projection[1]))
        - max(float(left_projection[0]), float(right_projection[0])),
    )
    shorter = min(
        float(left_projection[1] - left_projection[0]),
        float(right_projection[1] - right_projection[0]),
    )
    return overlap / max(shorter, 1e-12)


def _plane_intersection(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    left_normal = _unit(left["normal"])
    right_normal = _unit(right["normal"])
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(abs(float(np.dot(left_normal, right_normal))), -1.0, 1.0)
            )
        )
    )
    physical_angle = min(angle, 180.0 - angle)
    if physical_angle < 20.0 or physical_angle > 160.0:
        raise ValueError("plane pair is outside BIRTH_V1 angular eligibility")
    axis = _unit(np.cross(left_normal, right_normal))
    matrix = np.vstack((left_normal, right_normal, axis))
    rhs = np.asarray(
        (-float(left.get("offset") or 0.0), -float(right.get("offset") or 0.0), 0.0),
        dtype=np.float64,
    )
    anchor = np.linalg.solve(matrix, rhs)
    return anchor, axis, physical_angle


def _residual_ids(solution: Mapping[str, Any]) -> set[str]:
    owners = solution.get("atom_owners")
    if isinstance(owners, Mapping):
        result = set()
        for atom_id, owner in owners.items():
            if isinstance(owner, Mapping):
                owner_kind = str(owner.get("owner_kind") or "").upper()
                owner_value = str(owner.get("owner_id") or owner.get("owner") or "").upper()
                latent_path_id = owner.get("latent_path_id")
                residual = owner_kind == "RESIDUAL" or owner_value in {
                    "RESIDUAL",
                    "RESIDUAL_NONCOUNTED",
                }
                if owner_kind == "PATH" and latent_path_id is None:
                    raise ValueError("PATH atom owner must provide latent_path_id")
            else:
                residual = str(owner).upper() in {
                    "RESIDUAL",
                    "RESIDUAL_NONCOUNTED",
                }
            if residual:
                result.add(str(atom_id))
        return result
    if isinstance(owners, Sequence) and not isinstance(owners, (str, bytes)):
        result = set()
        for record in owners:
            if not isinstance(record, Mapping):
                raise ValueError("atom_owners records must be objects")
            owner_kind = str(record.get("owner_kind") or "").upper()
            owner = str(record.get("owner_id") or record.get("owner") or "").upper()
            latent_path_id = record.get("latent_path_id")
            if owner_kind == "PATH" and latent_path_id is None:
                raise ValueError("PATH atom owner must provide latent_path_id")
            if owner_kind == "RESIDUAL" or owner in {
                "RESIDUAL",
                "RESIDUAL_NONCOUNTED",
            }:
                result.add(str(record["atom_id"]))
        return result
    raise ValueError("exact solution must expose per-atom owners for residual birth")


def _components(ids: set[str], edges: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {atom_id: set() for atom_id in ids}
    for record in edges:
        pair = record.get("atom_ids")
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError("support atom edge must contain two atom_ids")
        left, right = str(pair[0]), str(pair[1])
        if left in ids and right in ids and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    result: list[list[str]] = []
    unseen = set(ids)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        unseen -= component
        result.append(sorted(component))
    return result


def _atom_masses(atoms: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    positive = [
        float(atom.get("area_proxy_mm2") or 0.0)
        for atom in atoms
        if float(atom.get("area_proxy_mm2") or 0.0) > 0.0
    ]
    reference = median(positive) if positive else 1.0
    masses: dict[str, int] = {}
    for atom in atoms:
        raw_mass = atom.get("normalized_support_mass")
        if raw_mass is None:
            mass = int(
                np.clip(
                    round(float(atom.get("area_proxy_mm2") or reference) / reference),
                    1,
                    4,
                )
            )
        else:
            mass = int(raw_mass)
            if float(raw_mass) != float(mass) or mass < 1 or mass > 4:
                raise ValueError("normalized_support_mass must be an integer in [1,4]")
        masses[str(atom["atom_id"])] = mass
    return masses


def _duplicates_existing(
    endpoints: np.ndarray,
    axis: np.ndarray,
    paths: Sequence[Mapping[str, Any]],
    *,
    offset_tolerance_mm: float,
) -> bool:
    midpoint = np.mean(endpoints, axis=0)
    for path in paths:
        try:
            other_endpoints = _line_endpoints(path)
            other_axis = _unit(path.get("axis_direction") or (other_endpoints[1] - other_endpoints[0]))
        except (KeyError, TypeError, ValueError):
            continue
        if _unoriented_angle_deg(axis, other_axis) > 4.0:
            continue
        other_midpoint = np.mean(other_endpoints, axis=0)
        symmetric_offset = max(
            _line_distance(midpoint, other_midpoint, other_axis),
            _line_distance(other_midpoint, midpoint, axis),
        )
        if symmetric_offset > offset_tolerance_mm:
            continue
        if _shorter_overlap_fraction(endpoints, other_endpoints, axis) >= 0.75:
            return True
    return False


def _graph_hash(payload: Mapping[str, Any]) -> str:
    sanitized = {key: value for key, value in payload.items() if key != "graph_sha256"}
    encoded = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_residual_birth_columns(
    graph: Mapping[str, Any],
    solution: Mapping[str, Any],
    *,
    total_column_cap: int = TOTAL_BIRTH_COLUMN_CAP,
) -> list[dict[str, Any]]:
    """Generate deterministic negative-lower-bound BIRTH_V1 columns."""

    if int(total_column_cap) < 0 or int(total_column_cap) > TOTAL_BIRTH_COLUMN_CAP:
        raise ValueError(f"birth column cap must be between 0 and {TOTAL_BIRTH_COLUMN_CAP}")
    atoms = [dict(atom) for atom in graph.get("support_atom_table") or ()]
    planes = [dict(plane) for plane in graph.get("plane_table") or ()]
    paths = [dict(path) for path in graph.get("latent_paths") or ()]
    edges = [dict(edge) for edge in graph.get("support_atom_edges") or ()]
    residual_ids = _residual_ids(solution)
    atom_by_id = {str(atom["atom_id"]): atom for atom in atoms}
    if not residual_ids <= set(atom_by_id):
        raise ValueError("solution references residual atoms absent from graph")
    spacing_values = [
        float(atom.get("local_spacing_mm") or 0.0)
        for atom in atoms
        if float(atom.get("local_spacing_mm") or 0.0) > 0.0
    ]
    spacing = median(spacing_values) if spacing_values else 1.5
    bin_width = max(3.0, 2.0 * spacing)
    transverse_tolerance = min(3.0, max(1.5, 1.25 * spacing))
    duplicate_offset = max(3.0, 2.0 * spacing)
    masses = _atom_masses(atoms)
    raw: list[dict[str, Any]] = []

    ordered_planes = sorted(planes, key=lambda item: str(item.get("plane_id") or ""))
    for left_index, left in enumerate(ordered_planes):
        for right in ordered_planes[left_index + 1 :]:
            try:
                anchor, axis, plane_angle = _plane_intersection(left, right)
            except (ValueError, np.linalg.LinAlgError):
                continue
            left_normal = _unit(left["normal"])
            right_normal = _unit(right["normal"])
            eligible: set[str] = set()
            side_by_id: dict[str, str] = {}
            for atom_id in sorted(residual_ids):
                atom = atom_by_id[atom_id]
                centroid = np.asarray(atom["centroid"], dtype=np.float64)
                normal = _unit(atom["normal"])
                if _line_distance(centroid, anchor, axis) > transverse_tolerance:
                    continue
                left_angle = _unoriented_angle_deg(normal, left_normal)
                right_angle = _unoriented_angle_deg(normal, right_normal)
                if min(left_angle, right_angle) > 15.0:
                    continue
                atom_axis = np.asarray(atom.get("axis_direction") or (0.0, 0.0, 0.0), dtype=np.float64)
                if float(np.linalg.norm(atom_axis)) > 1e-9 and _unoriented_angle_deg(atom_axis, axis) > 15.0:
                    continue
                left_distance = abs(float(np.dot(left_normal, centroid)) + float(left.get("offset") or 0.0))
                right_distance = abs(float(np.dot(right_normal, centroid)) + float(right.get("offset") or 0.0))
                side = min(
                    ((left_angle, left_distance, "left"), (right_angle, right_distance, "right"))
                )[2]
                eligible.add(atom_id)
                side_by_id[atom_id] = side

            for component in _components(eligible, edges):
                if len(component) < 6:
                    continue
                centroids = np.asarray([atom_by_id[atom_id]["centroid"] for atom_id in component], dtype=np.float64)
                projections = centroids @ axis
                lower = float(np.min(projections))
                upper = float(np.max(projections))
                span = upper - lower
                if span < max(12.0, 6.0 * spacing):
                    continue
                bin_by_id = {
                    atom_id: int(math.floor((float(projection) - lower) / bin_width + 1e-9))
                    for atom_id, projection in zip(component, projections)
                }
                occupied = set(bin_by_id.values())
                left_bins = {bin_by_id[atom_id] for atom_id in component if side_by_id[atom_id] == "left"}
                right_bins = {bin_by_id[atom_id] for atom_id in component if side_by_id[atom_id] == "right"}
                if len(occupied) < 6 or len(left_bins) < 3 or len(right_bins) < 3:
                    continue
                if len(left_bins & right_bins) < 3:
                    continue
                empty_run = current_empty = 0
                for bin_id in range(min(occupied), max(occupied) + 1):
                    current_empty = 0 if bin_id in occupied else current_empty + 1
                    empty_run = max(empty_run, current_empty)
                if empty_run > 1:
                    continue

                tls_anchor = np.average(
                    centroids,
                    axis=0,
                    weights=np.asarray([masses[atom_id] for atom_id in component], dtype=np.float64),
                )
                centered = centroids - tls_anchor
                _singular, _values, basis = np.linalg.svd(centered, full_matrices=False)
                tls_axis = _unit(basis[0])
                if _unoriented_angle_deg(tls_axis, axis) > 4.0:
                    continue
                residuals = np.asarray(
                    [_line_distance(point, tls_anchor, tls_axis) for point in centroids],
                    dtype=np.float64,
                )
                p90_residual = float(np.quantile(residuals, 0.90))
                if p90_residual > transverse_tolerance:
                    continue

                bin_neighbors: dict[int, set[int]] = defaultdict(set)
                for edge in edges:
                    pair = edge.get("atom_ids") or ()
                    if len(pair) != 2 or str(pair[0]) not in bin_by_id or str(pair[1]) not in bin_by_id:
                        continue
                    left_bin = bin_by_id[str(pair[0])]
                    right_bin = bin_by_id[str(pair[1])]
                    if left_bin != right_bin:
                        bin_neighbors[left_bin].add(right_bin)
                        bin_neighbors[right_bin].add(left_bin)
                minimum_bin, maximum_bin = min(occupied), max(occupied)
                if any(
                    len(neighbors) > 2 and bin_id not in {minimum_bin, maximum_bin}
                    for bin_id, neighbors in bin_neighbors.items()
                ):
                    continue

                endpoints = np.asarray((anchor + lower * axis, anchor + upper * axis), dtype=np.float64)
                if _duplicates_existing(
                    endpoints,
                    axis,
                    paths,
                    offset_tolerance_mm=duplicate_offset,
                ):
                    continue
                pair = sorted(
                    (
                        f"PLANE:{str(left['plane_id'])}",
                        f"PLANE:{str(right['plane_id'])}",
                    )
                )
                signature = {
                    "profile": BIRTH_PROFILE,
                    "panel_pair": pair,
                    "axis": np.rint(axis * 1_000_000.0).astype(np.int64).tolist(),
                    "midpoint": np.rint(np.mean(endpoints, axis=0) * 1_000.0).astype(np.int64).tolist(),
                    "atoms": component,
                }
                digest = hashlib.sha256(
                    json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:16]
                raw.append(
                    {
                        "birth_id": f"BIRTH_{digest}",
                        "latent_path_id": f"LP_BIRTH_{digest}",
                        "is_birth": True,
                        "birth_profile": BIRTH_PROFILE,
                        "geometry_kind": "LINE",
                        "span_endpoints": endpoints.tolist(),
                        "axis_direction": axis.tolist(),
                        "midpoint": np.mean(endpoints, axis=0).tolist(),
                        "span_mm": float(np.linalg.norm(endpoints[1] - endpoints[0])),
                        "primary_panel_pair": pair,
                        "primary_panel_pair_namespace": "PLANE",
                        "incident_plane_pairs": [pair],
                        "support_atom_ids": component,
                        "support_bin_ids": sorted(occupied),
                        "incidence_complete": True,
                        "incidence_origin": {"residual_compact_atom_birth": component},
                        "incidence_certificate": {
                            "tolerance_profile": INCIDENCE_PROFILE,
                            "birth_profile": BIRTH_PROFILE,
                            "plane_angle_deg": plane_angle,
                            "p90_line_residual_mm": p90_residual,
                            "transverse_tolerance_mm": transverse_tolerance,
                            "maximum_empty_bins": empty_run,
                        },
                        "support_atom_ids_truncated": False,
                        "label_eligibility": {
                            BEND: True,
                            "CORNER_NONCOUNTED": False,
                            "PERIMETER_NONCOUNTED": False,
                            "HOLE_NONCOUNTED": False,
                            "JUNCTION_NONCOUNTED": False,
                            "CURVE_NONCOUNTED": False,
                            RESIDUAL: True,
                        },
                        "normalized_support_mass": sum(masses[atom_id] for atom_id in component),
                    }
                )

    raw.sort(
        key=lambda item: (
            tuple(item["primary_panel_pair"]),
            tuple(np.rint(np.asarray(item["axis_direction"]) * 1_000_000.0).astype(np.int64)),
            tuple(np.rint(np.asarray(item["midpoint"]) * 1_000.0).astype(np.int64)),
            float(item["span_mm"]),
            str(item["birth_id"]),
        )
    )
    solver_contract = dict(solution.get("solver") or {})
    if (
        solver_contract.get("object_cost_int") is None
        or solver_contract.get("residual_cost_per_mass_int") is None
    ):
        raise ValueError(
            "residual birth requires the exact solver's frozen MDL cost profile"
        )
    object_cost = int(solver_contract["object_cost_int"])
    residual_cost = int(solver_contract["residual_cost_per_mass_int"])
    if object_cost <= 0 or residual_cost <= 0:
        raise ValueError("exact solver must report positive MDL costs")
    negative: list[dict[str, Any]] = []
    for record in raw:
        saved_residual_cost = residual_cost * int(record["normalized_support_mass"])
        lower_bound = object_cost - saved_residual_cost
        record["object_cost_int"] = object_cost
        record["residual_cost_per_mass_int"] = residual_cost
        record["saved_residual_cost_upper_bound_int"] = saved_residual_cost
        record["lower_bound_int"] = lower_bound
        record["lower_bound_cost_source"] = "current_exact_solver_fixed_mdl"
        if lower_bound < 0:
            negative.append(record)
    negative.sort(
        key=lambda item: (
            int(item["lower_bound_int"]),
            tuple(item["primary_panel_pair"]),
            tuple(np.rint(np.asarray(item["axis_direction"]) * 1_000_000.0).astype(np.int64)),
            tuple(np.rint(np.asarray(item["midpoint"]) * 1_000.0).astype(np.int64)),
            float(item["span_mm"]),
            str(item["birth_id"]),
        )
    )
    return negative[: int(total_column_cap)]


def add_birth_columns(
    graph: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return a canonical graph copy with bounded, unique birth paths added."""

    result = deepcopy(dict(graph))
    existing_birth_ids = {
        str(value) for value in result.get("birth_columns") or () if str(value)
    }
    if len(existing_birth_ids) + len(proposals) > TOTAL_BIRTH_COLUMN_CAP:
        raise ValueError("birth column cap exceeded")
    paths = [dict(path) for path in result.get("latent_paths") or ()]
    existing = {str(path.get("latent_path_id")) for path in paths}
    added = []
    object_costs = {
        int(proposal.get("object_cost_int") or 0) for proposal in proposals
    }
    residual_costs = {
        int(proposal.get("residual_cost_per_mass_int") or 0) for proposal in proposals
    }
    if proposals and (
        len(object_costs) != 1
        or len(residual_costs) != 1
        or min(object_costs) <= 0
        or min(residual_costs) <= 0
    ):
        raise ValueError("birth proposals must share one positive frozen MDL cost profile")
    for proposal in sorted(proposals, key=lambda item: str(item.get("latent_path_id") or "")):
        path_id = str(proposal.get("latent_path_id") or "")
        if not path_id or path_id in existing:
            raise ValueError("birth latent path IDs must be nonempty and unique")
        if int(proposal.get("lower_bound_int") or 0) >= 0:
            raise ValueError("non-improving birth lower bound must not be added")
        birth_id = str(proposal.get("birth_id") or "")
        if not birth_id or birth_id in existing_birth_ids:
            raise ValueError("birth IDs must be nonempty and globally unique")
        existing.add(path_id)
        paths.append(dict(proposal))
        added.append(birth_id)
        existing_birth_ids.add(birth_id)
    result["latent_paths"] = sorted(paths, key=lambda item: str(item["latent_path_id"]))
    result["birth_columns"] = sorted(existing_birth_ids)
    result["birth_profile"] = BIRTH_PROFILE
    result["birth_column_cap"] = TOTAL_BIRTH_COLUMN_CAP
    if proposals:
        parameters = dict(result.get("energy_parameters") or {})
        object_cost = next(iter(object_costs))
        residual_cost = next(iter(residual_costs))
        if parameters.get("object_cost_int") not in (None, object_cost):
            raise ValueError("birth proposal object cost disagrees with frozen graph MDL")
        if parameters.get("residual_cost_per_mass_int") not in (None, residual_cost):
            raise ValueError("birth proposal residual cost disagrees with frozen graph MDL")
        parameters["object_cost_int"] = object_cost
        parameters["residual_cost_per_mass_int"] = residual_cost
        result["energy_parameters"] = parameters
    result["graph_sha256"] = _graph_hash(result)
    return result
