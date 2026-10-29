"""Bounded, deterministic offline diagnostics for BIRTH_V1 eligibility.

The diagnostic deliberately consumes the same compact graph and exact-solver
ownership records as BIRTH_V1.  It does not accept truth, CAD, filenames,
expected counts, or source points, and it is not wired into runtime inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from statistics import median
from typing import Any

import numpy as np

from backend.f1_structured_residual_birth import (
    BIRTH_PROFILE,
    TOTAL_BIRTH_COLUMN_CAP,
    _atom_masses,
    _components,
    _duplicates_existing,
    _line_distance,
    _line_endpoints,
    _plane_intersection,
    _residual_ids,
    _unit,
    _unoriented_angle_deg,
    generate_residual_birth_columns,
)


DIAGNOSTIC_PROFILE = "BIRTH_V1_STAGE_CENSUS_V1"
MAX_DIAGNOSTIC_ATOMS = 20_000
MAX_DIAGNOSTIC_EDGES = 200_000
MAX_DIAGNOSTIC_PLANES = 64
MAX_DIAGNOSTIC_PATHS = 20_000

# TRANSITION_MODE_PROBE_V1 is diagnostic-only.  These caps are part of its
# inference contract: overflowing a table aborts the job rather than selecting
# a ranked subset whose contents could later depend on gold attribution.
TRANSITION_DIAGNOSTIC_PROFILE = "TRANSITION_MODE_PROBE_V1"
TRANSITION_DIAGNOSTIC_KIND = "f1_transition_diagnostic_census_v1"
MAX_DIAGNOSTIC_PLANE_PAIRS = 512
MAX_MULTIMODAL_PARENT_RECORDS = 4_000
MAX_DIAGNOSTIC_COMPONENTS = 4_096
MAX_COMPONENT_OWNER_BLOCKERS = 16
MAX_MATCHING_PATH_IDS = 32
MAX_NORMAL_MODES_PER_PARENT = 4
FROZEN_LINE_AXIS_ANGLE_DEG = 5.0
FROZEN_LINE_TRANSVERSE_OFFSET_MM = 4.0
FROZEN_LINE_GOLD_SPAN_OVERLAP = 0.60
FROZEN_LINE_PREDICTION_SPAN_OVERLAP = 0.60

_TRANSITION_FORBIDDEN_KEYS = frozenset(
    {
        "point_indices",
        "source_indices",
        "source_point_indices",
        "source_points",
        "points",
        "bbox_points",
        "vertex_indices",
        "point_coordinates",
        "coordinates",
    }
)

FUNNEL_STAGES = (
    "plane_pairs_total",
    "plane_pairs_angular_eligible",
    "atom_pair_trials",
    "atoms_after_transverse_distance",
    "atoms_after_normal",
    "atoms_after_atom_axis",
    "connected_components",
    "components_after_connectivity_size",
    "components_after_axial_span",
    "components_after_bin_coverage",
    "components_after_two_side_bin_overlap",
    "components_after_bin_gap",
    "components_after_tls_axis",
    "components_after_line_residual",
    "components_after_bin_branching",
    "components_after_duplicate_suppression",
    "components_after_mdl_lower_bound",
    "emitted_columns_after_cap",
)


def _check_bounds(
    atoms: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    planes: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
) -> None:
    limits = (
        ("support atoms", len(atoms), MAX_DIAGNOSTIC_ATOMS),
        ("support-atom edges", len(edges), MAX_DIAGNOSTIC_EDGES),
        ("planes", len(planes), MAX_DIAGNOSTIC_PLANES),
        ("latent paths", len(paths), MAX_DIAGNOSTIC_PATHS),
    )
    for label, actual, cap in limits:
        if actual > cap:
            raise ValueError(f"diagnostic {label} exceed cap: {actual} > {cap}")


def _first_empty_transition(funnel: Mapping[str, int]) -> str | None:
    previous = None
    for stage in FUNNEL_STAGES:
        current = int(funnel.get(stage) or 0)
        if previous is not None and previous > 0 and current == 0:
            return stage
        previous = current
    return None


def diagnose_residual_birth_eligibility(
    graph: Mapping[str, Any],
    solution: Mapping[str, Any],
    *,
    total_column_cap: int = TOTAL_BIRTH_COLUMN_CAP,
) -> dict[str, Any]:
    """Return a truth-free stage census for the exact BIRTH_V1 predicates."""

    if int(total_column_cap) < 0 or int(total_column_cap) > TOTAL_BIRTH_COLUMN_CAP:
        raise ValueError(
            f"birth column cap must be between 0 and {TOTAL_BIRTH_COLUMN_CAP}"
        )
    atoms = [dict(atom) for atom in graph.get("support_atom_table") or ()]
    planes = [dict(plane) for plane in graph.get("plane_table") or ()]
    paths = [dict(path) for path in graph.get("latent_paths") or ()]
    edges = [dict(edge) for edge in graph.get("support_atom_edges") or ()]
    _check_bounds(atoms, edges, planes, paths)

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
    solver_contract = dict(solution.get("solver") or {})
    if (
        solver_contract.get("object_cost_int") is None
        or solver_contract.get("residual_cost_per_mass_int") is None
    ):
        raise ValueError("residual birth requires the exact solver's frozen MDL cost profile")
    object_cost = int(solver_contract["object_cost_int"])
    residual_cost = int(solver_contract["residual_cost_per_mass_int"])
    if object_cost <= 0 or residual_cost <= 0:
        raise ValueError("exact solver must report positive MDL costs")

    funnel: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    pair_rows: list[dict[str, Any]] = []
    ordered_planes = sorted(planes, key=lambda item: str(item.get("plane_id") or ""))
    funnel["plane_pairs_total"] = len(ordered_planes) * (len(ordered_planes) - 1) // 2

    for left_index, left in enumerate(ordered_planes):
        for right in ordered_planes[left_index + 1 :]:
            pair_id = [str(left.get("plane_id") or ""), str(right.get("plane_id") or "")]
            pair_funnel: Counter[str] = Counter()
            pair_rejections: Counter[str] = Counter()
            try:
                anchor, axis, _plane_angle = _plane_intersection(left, right)
            except np.linalg.LinAlgError:
                rejections["plane_pair_singular"] += 1
                pair_rejections["plane_pair_singular"] += 1
                pair_rows.append({"plane_ids": pair_id, "funnel": {}, "rejections": dict(pair_rejections)})
                continue
            except ValueError:
                rejections["plane_pair_angle"] += 1
                pair_rejections["plane_pair_angle"] += 1
                pair_rows.append({"plane_ids": pair_id, "funnel": {}, "rejections": dict(pair_rejections)})
                continue
            funnel["plane_pairs_angular_eligible"] += 1
            pair_funnel["plane_pairs_angular_eligible"] = 1
            left_normal = _unit(left["normal"])
            right_normal = _unit(right["normal"])
            eligible: set[str] = set()
            side_by_id: dict[str, str] = {}
            for atom_id in sorted(residual_ids):
                funnel["atom_pair_trials"] += 1
                pair_funnel["atom_pair_trials"] += 1
                atom = atom_by_id[atom_id]
                centroid = np.asarray(atom["centroid"], dtype=np.float64)
                normal = _unit(atom["normal"])
                if _line_distance(centroid, anchor, axis) > transverse_tolerance:
                    rejections["transverse_distance"] += 1
                    pair_rejections["transverse_distance"] += 1
                    continue
                funnel["atoms_after_transverse_distance"] += 1
                pair_funnel["atoms_after_transverse_distance"] += 1
                left_angle = _unoriented_angle_deg(normal, left_normal)
                right_angle = _unoriented_angle_deg(normal, right_normal)
                if min(left_angle, right_angle) > 15.0:
                    rejections["normal"] += 1
                    pair_rejections["normal"] += 1
                    continue
                funnel["atoms_after_normal"] += 1
                pair_funnel["atoms_after_normal"] += 1
                atom_axis = np.asarray(
                    atom.get("axis_direction") or (0.0, 0.0, 0.0), dtype=np.float64
                )
                if (
                    float(np.linalg.norm(atom_axis)) > 1e-9
                    and _unoriented_angle_deg(atom_axis, axis) > 15.0
                ):
                    rejections["atom_axis"] += 1
                    pair_rejections["atom_axis"] += 1
                    continue
                funnel["atoms_after_atom_axis"] += 1
                pair_funnel["atoms_after_atom_axis"] += 1
                left_distance = abs(
                    float(np.dot(left_normal, centroid)) + float(left.get("offset") or 0.0)
                )
                right_distance = abs(
                    float(np.dot(right_normal, centroid)) + float(right.get("offset") or 0.0)
                )
                side_by_id[atom_id] = min(
                    ((left_angle, left_distance, "left"), (right_angle, right_distance, "right"))
                )[2]
                eligible.add(atom_id)

            for component in _components(eligible, edges):
                funnel["connected_components"] += 1
                pair_funnel["connected_components"] += 1
                if len(component) < 6:
                    rejections["connectivity_component_size"] += 1
                    pair_rejections["connectivity_component_size"] += 1
                    continue
                funnel["components_after_connectivity_size"] += 1
                pair_funnel["components_after_connectivity_size"] += 1
                centroids = np.asarray(
                    [atom_by_id[atom_id]["centroid"] for atom_id in component],
                    dtype=np.float64,
                )
                projections = centroids @ axis
                lower = float(np.min(projections))
                upper = float(np.max(projections))
                if upper - lower < max(12.0, 6.0 * spacing):
                    rejections["connectivity_axial_span"] += 1
                    pair_rejections["connectivity_axial_span"] += 1
                    continue
                funnel["components_after_axial_span"] += 1
                pair_funnel["components_after_axial_span"] += 1
                bin_by_id = {
                    atom_id: int(math.floor((float(projection) - lower) / bin_width + 1e-9))
                    for atom_id, projection in zip(component, projections)
                }
                occupied = set(bin_by_id.values())
                left_bins = {
                    bin_by_id[atom_id]
                    for atom_id in component
                    if side_by_id[atom_id] == "left"
                }
                right_bins = {
                    bin_by_id[atom_id]
                    for atom_id in component
                    if side_by_id[atom_id] == "right"
                }
                if len(occupied) < 6 or len(left_bins) < 3 or len(right_bins) < 3:
                    rejections["bin_coverage"] += 1
                    pair_rejections["bin_coverage"] += 1
                    continue
                funnel["components_after_bin_coverage"] += 1
                pair_funnel["components_after_bin_coverage"] += 1
                if len(left_bins & right_bins) < 3:
                    rejections["bin_two_side_overlap"] += 1
                    pair_rejections["bin_two_side_overlap"] += 1
                    continue
                funnel["components_after_two_side_bin_overlap"] += 1
                pair_funnel["components_after_two_side_bin_overlap"] += 1
                empty_run = current_empty = 0
                for bin_id in range(min(occupied), max(occupied) + 1):
                    current_empty = 0 if bin_id in occupied else current_empty + 1
                    empty_run = max(empty_run, current_empty)
                if empty_run > 1:
                    rejections["bin_empty_run"] += 1
                    pair_rejections["bin_empty_run"] += 1
                    continue
                funnel["components_after_bin_gap"] += 1
                pair_funnel["components_after_bin_gap"] += 1

                tls_anchor = np.average(
                    centroids,
                    axis=0,
                    weights=np.asarray([masses[atom_id] for atom_id in component], dtype=np.float64),
                )
                centered = centroids - tls_anchor
                _singular, _values, basis = np.linalg.svd(centered, full_matrices=False)
                tls_axis = _unit(basis[0])
                if _unoriented_angle_deg(tls_axis, axis) > 4.0:
                    rejections["tls_axis"] += 1
                    pair_rejections["tls_axis"] += 1
                    continue
                funnel["components_after_tls_axis"] += 1
                pair_funnel["components_after_tls_axis"] += 1
                residuals = np.asarray(
                    [_line_distance(point, tls_anchor, tls_axis) for point in centroids],
                    dtype=np.float64,
                )
                if float(np.quantile(residuals, 0.90)) > transverse_tolerance:
                    rejections["line_residual"] += 1
                    pair_rejections["line_residual"] += 1
                    continue
                funnel["components_after_line_residual"] += 1
                pair_funnel["components_after_line_residual"] += 1
                bin_neighbors: dict[int, set[int]] = defaultdict(set)
                for edge in edges:
                    pair = edge.get("atom_ids") or ()
                    if (
                        len(pair) != 2
                        or str(pair[0]) not in bin_by_id
                        or str(pair[1]) not in bin_by_id
                    ):
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
                    rejections["bin_internal_branching"] += 1
                    pair_rejections["bin_internal_branching"] += 1
                    continue
                funnel["components_after_bin_branching"] += 1
                pair_funnel["components_after_bin_branching"] += 1
                endpoints = np.asarray(
                    (anchor + lower * axis, anchor + upper * axis), dtype=np.float64
                )
                if _duplicates_existing(
                    endpoints, axis, paths, offset_tolerance_mm=duplicate_offset
                ):
                    rejections["duplicate_existing"] += 1
                    pair_rejections["duplicate_existing"] += 1
                    continue
                funnel["components_after_duplicate_suppression"] += 1
                pair_funnel["components_after_duplicate_suppression"] += 1
                saved_residual_cost = residual_cost * sum(masses[atom_id] for atom_id in component)
                if object_cost - saved_residual_cost >= 0:
                    rejections["mdl_nonnegative_lower_bound"] += 1
                    pair_rejections["mdl_nonnegative_lower_bound"] += 1
                    continue
                funnel["components_after_mdl_lower_bound"] += 1
                pair_funnel["components_after_mdl_lower_bound"] += 1
            pair_rows.append(
                {
                    "plane_ids": pair_id,
                    "funnel": {key: int(pair_funnel[key]) for key in FUNNEL_STAGES if pair_funnel[key]},
                    "rejections": dict(sorted(pair_rejections.items())),
                }
            )

    emitted = generate_residual_birth_columns(
        graph, solution, total_column_cap=int(total_column_cap)
    )
    funnel["emitted_columns_after_cap"] = len(emitted)
    negative_before_cap = int(funnel["components_after_mdl_lower_bound"])
    if negative_before_cap > len(emitted):
        rejections["column_cap"] = negative_before_cap - len(emitted)
    result_funnel = {stage: int(funnel[stage]) for stage in FUNNEL_STAGES}
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_birth_v1_stage_census",
        "diagnostic_profile": DIAGNOSTIC_PROFILE,
        "birth_profile": BIRTH_PROFILE,
        "research_only": True,
        "runtime_integration": False,
        "strict_scan_only": True,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "input_counts": {
            "support_atoms": len(atoms),
            "support_atom_edges": len(edges),
            "planes": len(planes),
            "latent_paths": len(paths),
            "residual_atoms": len(residual_ids),
        },
        "thresholds": {
            "median_spacing_mm": float(spacing),
            "bin_width_mm": float(bin_width),
            "transverse_tolerance_mm": float(transverse_tolerance),
            "duplicate_offset_tolerance_mm": float(duplicate_offset),
            "minimum_component_atoms": 6,
            "minimum_occupied_bins": 6,
            "minimum_bins_per_plane_side": 3,
            "minimum_two_side_overlap_bins": 3,
            "maximum_empty_bin_run": 1,
            "maximum_tls_axis_angle_deg": 4.0,
            "object_cost_int": object_cost,
            "residual_cost_per_mass_int": residual_cost,
            "column_cap": int(total_column_cap),
        },
        "funnel": result_funnel,
        "rejections": dict(sorted(rejections.items())),
        "first_empty_transition": _first_empty_transition(result_funnel),
        "proposal_ids": [str(item["birth_id"]) for item in emitted],
        "plane_pairs": pair_rows,
    }


def _record_dict(record: Any) -> dict[str, Any]:
    """Convert a compact mapping or frozen dataclass-like record to a dict."""

    if isinstance(record, Mapping):
        return dict(record)
    fields = getattr(record, "__dataclass_fields__", None)
    if fields:
        return {name: getattr(record, name) for name in fields}
    raise TypeError(f"transition diagnostic expected a mapping/dataclass, got {type(record)!r}")


def _transition_tables(graph: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Normalize TransitionAtomGraphV2 or its compact serializer.

    The builder lives in ``patch_graph_latent_decomposition``.  Keeping this
    adapter structural lets the diagnostic import neither point arrays nor the
    builder's transient point partition.
    """

    if isinstance(graph, Mapping):
        profile = str(graph.get("support_representation_profile") or graph.get("profile") or "")
        raw_parents = graph.get("support_parent_table", graph.get("parents"))
        raw_modes = graph.get("support_mode_table", graph.get("modes"))
        raw_edges = graph.get("support_mode_edges", graph.get("mode_edges"))
        caps = dict(graph.get("transition_atom_caps") or graph.get("caps") or {})
    else:
        profile = str(getattr(graph, "profile", ""))
        raw_parents = getattr(graph, "parents", None)
        raw_modes = getattr(graph, "modes", None)
        raw_edges = getattr(graph, "mode_edges", None)
        caps = dict(getattr(graph, "caps", {}) or {})
    if profile != "TRANSITION_ATOM_V2" or raw_parents is None or raw_modes is None or raw_edges is None:
        raise RuntimeError(
            "TRANSITION_MODE_PROBE_V1 requires build_transition_atom_graph_v2() "
            "output or compact_transition_graph_v2() output with profile TRANSITION_ATOM_V2"
        )
    parents = [_record_dict(item) for item in raw_parents]
    modes = [_record_dict(item) for item in raw_modes]
    edges = [_record_dict(item) for item in raw_edges]
    return profile, parents, modes, edges, caps


def _validate_sha256(label: str, value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a lowercase hexadecimal sha256")
    return normalized


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _table_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rounded_vector(value: Sequence[float]) -> list[float]:
    vector = _unit(value)
    return [round(float(component), 12) for component in vector]


def _recursive_key_check(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in _TRANSITION_FORBIDDEN_KEYS:
                raise ValueError(f"transition diagnostic forbids {path}.{key}")
            _recursive_key_check(value, f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            _recursive_key_check(value, f"{path}[{index}]")


def _parent_owner_lookup(solution: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    raw = solution.get("transition_parent_owners")
    if raw is None:
        raw = solution.get("parent_owners")
    if raw is None:
        raw = solution.get("atom_owners")
    result: dict[str, dict[str, str | None]] = {}
    if isinstance(raw, Mapping):
        records = []
        for parent_id, owner in raw.items():
            if isinstance(owner, Mapping):
                record = dict(owner)
                record.setdefault("parent_atom_id", parent_id)
            else:
                record = {"parent_atom_id": parent_id, "owner_kind": owner}
            records.append(record)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records = list(raw)
    else:
        return result
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("parent ownership records must be objects")
        parent_id = str(raw_record.get("parent_atom_id") or raw_record.get("atom_id") or "")
        if not parent_id:
            raise ValueError("parent ownership record is missing parent_atom_id")
        owner_kind = str(raw_record.get("owner_kind") or raw_record.get("owner") or "UNKNOWN").upper()
        path_id = raw_record.get("latent_path_id") or raw_record.get("path_id")
        semantic_label = raw_record.get("semantic_label")
        if owner_kind in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}:
            semantic_label = "RESIDUAL_NONCOUNTED"
            path_id = None
        result[parent_id] = {
            "owner_kind": owner_kind,
            "semantic_label": None if semantic_label is None else str(semantic_label),
            "path_id": None if path_id is None else str(path_id),
        }
    return result


def _maximum_empty_run(occupied: set[int]) -> int:
    if not occupied:
        return 0
    maximum = current = 0
    for bin_id in range(min(occupied), max(occupied) + 1):
        current = 0 if bin_id in occupied else current + 1
        maximum = max(maximum, current)
    return maximum


def _longest_consecutive_run(occupied: set[int]) -> int:
    maximum = current = 0
    previous: int | None = None
    for bin_id in sorted(occupied):
        current = current + 1 if previous is not None and bin_id == previous + 1 else 1
        maximum = max(maximum, current)
        previous = bin_id
    return maximum


def _mode_mean(modes: Sequence[Mapping[str, Any]]) -> np.ndarray:
    weighted_sum = np.zeros(3, dtype=np.float64)
    moment = np.zeros((3, 3), dtype=np.float64)
    for mode in modes:
        normal = _unit(mode.get("normal_projective") or mode.get("normal"))
        weight = max(1, int(mode.get("point_count") or 1))
        weighted_sum += float(weight) * normal
        moment += float(weight) * np.outer(normal, normal)
    if float(np.linalg.norm(weighted_sum)) > 1e-12:
        return _unit(weighted_sum)
    _values, vectors = np.linalg.eigh(moment)
    return _unit(vectors[:, -1])


def _matching_paths(
    *,
    anchor: np.ndarray,
    axis: np.ndarray,
    lower: float,
    upper: float,
    paths: Sequence[Mapping[str, Any]],
) -> list[str]:
    endpoints = np.asarray((anchor + lower * axis, anchor + upper * axis), dtype=np.float64)
    midpoint = np.mean(endpoints, axis=0)
    matches: list[str] = []
    for path in sorted(paths, key=lambda item: str(item.get("latent_path_id") or "")):
        eligibility = path.get("label_eligibility")
        if (
            not bool(path.get("incidence_complete"))
            or not isinstance(eligibility, Mapping)
            or not bool(eligibility.get("BEND_STRAIGHT_COUNTED"))
        ):
            continue
        try:
            other_endpoints = _line_endpoints(path)
            other_axis = _unit(path.get("axis_direction") or (other_endpoints[1] - other_endpoints[0]))
        except (KeyError, TypeError, ValueError):
            continue
        if _unoriented_angle_deg(axis, other_axis) > FROZEN_LINE_AXIS_ANGLE_DEG:
            continue
        other_midpoint = np.mean(other_endpoints, axis=0)
        offset = max(
            _line_distance(midpoint, other_midpoint, other_axis),
            _line_distance(other_midpoint, midpoint, axis),
        )
        if offset > FROZEN_LINE_TRANSVERSE_OFFSET_MM:
            continue
        interval = np.sort(endpoints @ axis)
        other_interval = np.sort(other_endpoints @ axis)
        overlap = max(
            0.0,
            min(float(interval[1]), float(other_interval[1]))
            - max(float(interval[0]), float(other_interval[0])),
        )
        component_span = max(float(interval[1] - interval[0]), 1e-12)
        path_span = max(float(other_interval[1] - other_interval[0]), 1e-12)
        if (
            overlap / component_span < FROZEN_LINE_GOLD_SPAN_OVERLAP
            or overlap / path_span < FROZEN_LINE_PREDICTION_SPAN_OVERLAP
        ):
            continue
        matches.append(str(path.get("latent_path_id") or ""))
    if len(matches) > MAX_MATCHING_PATH_IDS:
        raise ValueError(
            f"diagnostic matching path IDs exceed cap: {len(matches)} > {MAX_MATCHING_PATH_IDS}"
        )
    return matches


def diagnose_transition_mode_probe(
    transition_graph: Any,
    prediction_graph: Mapping[str, Any],
    a0_solution: Mapping[str, Any],
    *,
    scan_sha256: str,
    candidate_pool_sha256: str,
    graph_sha256: str,
    a0_solution_sha256: str,
    diagnostic_profile: str = TRANSITION_DIAGNOSTIC_PROFILE,
) -> dict[str, Any]:
    """Build the bounded inference-safe TRANSITION_MODE_PROBE census.

    This function emits no candidate column and does not mutate either input.
    Gold, CAD, filenames, expected counts, and point samples are not accepted.
    A0 ownership that cannot be mapped to V2 physical parents is explicitly
    recorded as unavailable and makes birth eligibility fail closed.
    """

    if diagnostic_profile not in {"TRANSITION_MODE_PROBE_V1", "TRANSITION_MODE_PROBE_V2"}:
        raise ValueError("unsupported transition diagnostic profile")
    profile, parents, modes, mode_edges, builder_caps = _transition_tables(transition_graph)
    _recursive_key_check(
        {
            "support_parent_table": parents,
            "support_mode_table": modes,
            "support_mode_edges": mode_edges,
        }
    )
    if isinstance(transition_graph, Mapping):
        claimed_table_hashes = {
            "parent_table_sha256": transition_graph.get("parent_table_sha256")
            or transition_graph.get("transition_parent_table_sha256"),
            "mode_table_sha256": transition_graph.get("mode_table_sha256")
            or transition_graph.get("transition_mode_table_sha256"),
            "mode_edge_table_sha256": transition_graph.get("mode_edge_table_sha256")
            or transition_graph.get("transition_mode_edge_table_sha256"),
        }
    else:
        claimed_table_hashes = {
            "parent_table_sha256": getattr(transition_graph, "parent_table_sha256", None),
            "mode_table_sha256": getattr(transition_graph, "mode_table_sha256", None),
            "mode_edge_table_sha256": getattr(transition_graph, "mode_edge_table_sha256", None),
        }
    transition_table_hashes = {
        "parent_table_sha256": str(claimed_table_hashes["parent_table_sha256"] or _table_sha256(parents)),
        "mode_table_sha256": str(claimed_table_hashes["mode_table_sha256"] or _table_sha256(modes)),
        "mode_edge_table_sha256": str(
            claimed_table_hashes["mode_edge_table_sha256"] or _table_sha256(mode_edges)
        ),
    }
    transition_table_hashes = {
        key: _validate_sha256(key, value) for key, value in transition_table_hashes.items()
    }
    planes = [dict(item) for item in prediction_graph.get("plane_table") or ()]
    paths = [dict(item) for item in prediction_graph.get("latent_paths") or ()]
    pair_count = len(planes) * (len(planes) - 1) // 2
    if pair_count > MAX_DIAGNOSTIC_PLANE_PAIRS:
        raise ValueError(
            f"diagnostic plane pairs exceed cap: {pair_count} > {MAX_DIAGNOSTIC_PLANE_PAIRS}"
        )
    if len(parents) > MAX_MULTIMODAL_PARENT_RECORDS:
        raise ValueError(
            f"diagnostic parents exceed cap: {len(parents)} > {MAX_MULTIMODAL_PARENT_RECORDS}"
        )
    if len(paths) > MAX_DIAGNOSTIC_PATHS:
        raise ValueError(f"diagnostic latent paths exceed cap: {len(paths)} > {MAX_DIAGNOSTIC_PATHS}")

    parent_by_id: dict[str, dict[str, Any]] = {}
    modes_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode_by_id: dict[str, dict[str, Any]] = {}
    for parent in parents:
        parent_id = str(parent.get("parent_atom_id") or "")
        if not parent_id or parent_id in parent_by_id:
            raise ValueError("transition parents require unique parent_atom_id values")
        parent_by_id[parent_id] = parent
    for mode in modes:
        mode_id = str(mode.get("mode_atom_id") or "")
        parent_id = str(mode.get("parent_atom_id") or "")
        if not mode_id or mode_id in mode_by_id or parent_id not in parent_by_id:
            raise ValueError("transition modes require unique IDs and known physical parents")
        mode_by_id[mode_id] = mode
        modes_by_parent[parent_id].append(mode)
    for parent_id, child_modes in modes_by_parent.items():
        if len(child_modes) > MAX_NORMAL_MODES_PER_PARENT:
            raise ValueError(
                f"normal modes for {parent_id} exceed cap: {len(child_modes)} > "
                f"{MAX_NORMAL_MODES_PER_PARENT}"
            )

    edge_by_id: dict[str, dict[str, Any]] = {}
    mode_adjacency: dict[str, set[str]] = {mode_id: set() for mode_id in mode_by_id}
    for edge in mode_edges:
        edge_id = str(edge.get("mode_edge_id") or "")
        pair = edge.get("mode_ids") or ()
        if not edge_id or edge_id in edge_by_id or len(pair) != 2:
            raise ValueError("transition mode edges require unique IDs and exactly two mode_ids")
        left_id, right_id = str(pair[0]), str(pair[1])
        if left_id not in mode_by_id or right_id not in mode_by_id or left_id == right_id:
            raise ValueError("transition mode edge references invalid mode IDs")
        edge_by_id[edge_id] = edge
        mode_adjacency[left_id].add(right_id)
        mode_adjacency[right_id].add(left_id)

    spacing = float(
        (transition_graph.get("local_spacing_mm") if isinstance(transition_graph, Mapping) else getattr(transition_graph, "local_spacing_mm", 0.0))
        or median([float(parent.get("local_spacing_mm") or 0.0) for parent in parents] or [1.5])
    )
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("transition graph must provide positive local_spacing_mm")
    bin_width = max(3.0, 2.0 * spacing)
    tube_tolerance = min(3.0, max(1.5, 1.25 * spacing))
    owner_lookup = _parent_owner_lookup(a0_solution)
    ownership_projection = dict(
        a0_solution.get("transition_parent_ownership_projection") or {}
    )
    solver = dict(a0_solution.get("solver") or {})
    object_cost = int(solver.get("object_cost_int") or 0)
    residual_cost = int(solver.get("residual_cost_per_mass_int") or 0)
    if object_cost <= 0 or residual_cost <= 0:
        raise ValueError("A0 solution must provide positive frozen MDL cost terms")

    ordered_planes = sorted(planes, key=lambda item: str(item.get("plane_id") or ""))
    multimodal_records: list[dict[str, Any]] = []
    raw_multimodal_count = 0
    v1_merged_count = 0
    for parent_id in sorted(parent_by_id):
        parent = parent_by_id[parent_id]
        child_modes = sorted(modes_by_parent.get(parent_id, ()), key=lambda item: str(item["mode_atom_id"]))
        if len(child_modes) < 2:
            continue
        raw_multimodal_count += 1
        separations = [
            _unoriented_angle_deg(left["normal_projective"], right["normal_projective"])
            for index, left in enumerate(child_modes)
            for right in child_modes[index + 1 :]
        ]
        mean_normal = _mode_mean(child_modes)
        mean_residuals = [
            round(_unoriented_angle_deg(mean_normal, mode["normal_projective"]), 6)
            for mode in child_modes
        ]
        merged = len(child_modes) >= 2 and min(mean_residuals) > 15.0
        v1_merged_count += int(merged)
        pairwise_angles = [
            {
                "mode_ids": sorted((str(left["mode_atom_id"]), str(right["mode_atom_id"]))),
                "angle_deg": round(
                    _unoriented_angle_deg(left["normal_projective"], right["normal_projective"]),
                    6,
                ),
            }
            for index, left in enumerate(child_modes)
            for right in child_modes[index + 1 :]
        ]
        parent_centroid = np.asarray(parent["centroid"], dtype=np.float64)
        v1_compatible_planes = [
            str(plane.get("plane_id") or "")
            for plane in ordered_planes
            if _unoriented_angle_deg(mean_normal, plane["normal"]) <= 15.0
            and abs(
                float(np.dot(_unit(plane["normal"]), parent_centroid))
                + float(plane.get("offset") or 0.0)
            )
            <= tube_tolerance
        ]
        multimodal_records.append(
            {
                "parent_atom_id": parent_id,
                "voxel_key": [int(value) for value in parent.get("voxel_key") or ()],
                "raw_valid_mode_count": int(
                    parent.get("raw_valid_mode_count")
                    or parent.get("valid_mode_count")
                    or len(child_modes)
                ),
                "mode_overflow": bool(parent.get("mode_overflow", False)),
                "transition_certificate_allowed": bool(
                    parent.get("transition_certificate_allowed", not parent.get("mode_overflow", False))
                ),
                "mode_summaries": [
                    {
                        "mode_atom_id": str(mode["mode_atom_id"]),
                        "normal": _rounded_vector(mode["normal_projective"]),
                        "centroid": [round(float(value), 6) for value in mode["centroid"]],
                        "point_count": int(mode.get("point_count") or 0),
                        "occupied_fine_cell_count": int(mode.get("occupied_fine_cell_count") or 0),
                        "support_fraction": round(float(mode.get("support_fraction") or 0.0), 6),
                    }
                    for mode in child_modes
                ],
                "maximum_mode_separation_deg": round(max(separations, default=0.0), 6),
                "pairwise_mode_angles_deg": pairwise_angles,
                "v1_mean_normal": _rounded_vector(mean_normal),
                "v1_mean_normal_residuals_deg": mean_residuals,
                "v1_compatible_plane_mode_count": len(v1_compatible_planes),
                "v1_compatible_plane_ids": v1_compatible_planes,
                "v1_modes_merged": merged,
            }
        )

    plane_pair_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    birth_records: list[dict[str, Any]] = []
    rejection_counter: Counter[str] = Counter()
    angle_eligible_pairs = locally_supported_pairs = 0
    component_serial = 0
    for left_index, left in enumerate(ordered_planes):
        for right in ordered_planes[left_index + 1 :]:
            plane_ids = [str(left.get("plane_id") or ""), str(right.get("plane_id") or "")]
            pair_id = _stable_id("PP", plane_ids)
            pair_rejections: Counter[str] = Counter()
            try:
                anchor, axis, plane_angle = _plane_intersection(left, right)
            except np.linalg.LinAlgError:
                rejection_counter["PLANE_PAIR_INTERSECTION_SINGULAR"] += 1
                pair_rejections["PLANE_PAIR_INTERSECTION_SINGULAR"] += 1
                plane_pair_records.append(
                    {
                        "plane_pair_id": pair_id,
                        "plane_ids": plane_ids,
                        "plane_angle_deg": None,
                        "intersection_line_hash": None,
                        "intersection_axis": None,
                        "transverse_parent_count": 0,
                        "left_mode_count": 0,
                        "right_mode_count": 0,
                        "two_mode_rung_count": 0,
                        "v1_merged_parent_count": 0,
                        "component_count": 0,
                        "largest_component": None,
                        "rejection_counts": dict(pair_rejections),
                    }
                )
                continue
            except ValueError:
                rejection_counter["PLANE_PAIR_ANGLE_INELIGIBLE"] += 1
                pair_rejections["PLANE_PAIR_ANGLE_INELIGIBLE"] += 1
                plane_pair_records.append(
                    {
                        "plane_pair_id": pair_id,
                        "plane_ids": plane_ids,
                        "plane_angle_deg": None,
                        "intersection_line_hash": None,
                        "intersection_axis": None,
                        "transverse_parent_count": 0,
                        "left_mode_count": 0,
                        "right_mode_count": 0,
                        "two_mode_rung_count": 0,
                        "v1_merged_parent_count": 0,
                        "component_count": 0,
                        "largest_component": None,
                        "rejection_counts": dict(pair_rejections),
                    }
                )
                continue
            angle_eligible_pairs += 1
            left_normal = _unit(left["normal"])
            right_normal = _unit(right["normal"])
            transverse_parents = {
                parent_id
                for parent_id, parent in parent_by_id.items()
                if _line_distance(np.asarray(parent["centroid"], dtype=np.float64), anchor, axis)
                <= tube_tolerance
            }
            left_modes: set[str] = set()
            right_modes: set[str] = set()
            for parent_id in sorted(transverse_parents):
                for mode in modes_by_parent.get(parent_id, ()):
                    mode_id = str(mode["mode_atom_id"])
                    centroid = np.asarray(mode["centroid"], dtype=np.float64)
                    normal = mode["normal_projective"]
                    if (
                        _unoriented_angle_deg(normal, left_normal) <= 15.0
                        and abs(float(np.dot(left_normal, centroid)) + float(left.get("offset") or 0.0))
                        <= tube_tolerance
                    ):
                        left_modes.add(mode_id)
                    if (
                        _unoriented_angle_deg(normal, right_normal) <= 15.0
                        and abs(float(np.dot(right_normal, centroid)) + float(right.get("offset") or 0.0))
                        <= tube_tolerance
                    ):
                        right_modes.add(mode_id)
            rung_edges = [
                edge
                for edge in mode_edges
                if str(edge.get("edge_kind") or "") == "CROSS_MODE_LOCAL"
                and (
                    (str(edge["mode_ids"][0]) in left_modes and str(edge["mode_ids"][1]) in right_modes)
                    or (str(edge["mode_ids"][1]) in left_modes and str(edge["mode_ids"][0]) in right_modes)
                )
            ]
            supported_modes = left_modes | right_modes
            if transverse_parents and left_modes and right_modes:
                locally_supported_pairs += 1
            if not transverse_parents:
                pair_rejections["NO_LOCAL_PAIR_SUPPORT"] += 1
            if not left_modes:
                pair_rejections["MISSING_LEFT_PLANE_MODE"] += 1
            if not right_modes:
                pair_rejections["MISSING_RIGHT_PLANE_MODE"] += 1
            if left_modes and right_modes and not rung_edges:
                pair_rejections["NO_TWO_MODE_TRANSITION_RUNG"] += 1

            components: list[list[str]] = []
            unseen = set(supported_modes)
            while unseen:
                seed = min(unseen)
                stack = [seed]
                component: set[str] = set()
                while stack:
                    current = stack.pop()
                    if current in component:
                        continue
                    component.add(current)
                    stack.extend(sorted((mode_adjacency[current] & supported_modes) - component, reverse=True))
                unseen -= component
                components.append(sorted(component))
            if len(component_records) + len(components) > MAX_DIAGNOSTIC_COMPONENTS:
                raise ValueError(
                    f"diagnostic components exceed cap: {len(component_records) + len(components)} > "
                    f"{MAX_DIAGNOSTIC_COMPONENTS}"
                )

            pair_component_records: list[dict[str, Any]] = []
            for mode_ids in components:
                component_serial += 1
                component_id = _stable_id("TC", [pair_id, mode_ids])
                parent_ids = sorted({str(mode_by_id[mode_id]["parent_atom_id"]) for mode_id in mode_ids})
                centroids = np.asarray([parent_by_id[parent_id]["centroid"] for parent_id in parent_ids], dtype=np.float64)
                projections = centroids @ axis
                lower = float(np.min(projections))
                upper = float(np.max(projections))
                bin_by_parent = {
                    parent_id: int(math.floor((float(value) - lower) / bin_width + 1e-9))
                    for parent_id, value in zip(parent_ids, projections)
                }
                occupied_bins = set(bin_by_parent.values())
                left_parent_ids = {str(mode_by_id[item]["parent_atom_id"]) for item in mode_ids if item in left_modes}
                right_parent_ids = {str(mode_by_id[item]["parent_atom_id"]) for item in mode_ids if item in right_modes}
                left_bins = {bin_by_parent[item] for item in left_parent_ids}
                right_bins = {bin_by_parent[item] for item in right_parent_ids}
                component_rungs = [
                    edge for edge in rung_edges if all(str(item) in mode_ids for item in edge["mode_ids"])
                ]
                rung_parent_ids = {
                    str(mode_by_id[str(mode_id)]["parent_atom_id"])
                    for edge in component_rungs
                    for mode_id in edge["mode_ids"]
                }
                rung_bins = {bin_by_parent[item] for item in rung_parent_ids}
                bin_neighbors: dict[int, set[int]] = defaultdict(set)
                for edge in mode_edges:
                    edge_modes = [str(item) for item in edge.get("mode_ids") or ()]
                    if len(edge_modes) != 2 or not all(item in mode_ids for item in edge_modes):
                        continue
                    left_parent = str(mode_by_id[edge_modes[0]]["parent_atom_id"])
                    right_parent = str(mode_by_id[edge_modes[1]]["parent_atom_id"])
                    left_bin = bin_by_parent[left_parent]
                    right_bin = bin_by_parent[right_parent]
                    if left_bin != right_bin:
                        bin_neighbors[left_bin].add(right_bin)
                        bin_neighbors[right_bin].add(left_bin)
                minimum_bin, maximum_bin = min(occupied_bins), max(occupied_bins)
                branch_count = sum(
                    len(neighbors) > 2 and bin_id not in {minimum_bin, maximum_bin}
                    for bin_id, neighbors in bin_neighbors.items()
                )
                weights = np.asarray(
                    [max(1, int(parent_by_id[item].get("normalized_support_mass") or 1)) for item in parent_ids],
                    dtype=np.float64,
                )
                tls_anchor = np.average(centroids, axis=0, weights=weights)
                if len(parent_ids) >= 2:
                    _singular, _values, basis = np.linalg.svd(centroids - tls_anchor, full_matrices=False)
                    tls_axis = _unit(basis[0])
                else:
                    tls_axis = axis
                tls_angle = _unoriented_angle_deg(tls_axis, axis)
                residuals = np.asarray(
                    [_line_distance(point, tls_anchor, tls_axis) for point in centroids], dtype=np.float64
                )
                p90_residual = float(np.quantile(residuals, 0.90))
                full_windows: list[set[str]] = []
                if len(occupied_bins) >= 3:
                    for start in range(minimum_bin, maximum_bin - 1):
                        required = set(range(start, start + 3))
                        if (
                            required <= occupied_bins
                            and required <= left_bins
                            and required <= right_bins
                            and len(required & rung_bins) >= 2
                        ):
                            full_windows.append(
                                {parent_id for parent_id, bin_id in bin_by_parent.items() if bin_id in required}
                            )
                residual_windows = [
                    window
                    for window in full_windows
                    if all(
                        parent_id in owner_lookup
                        and owner_lookup[parent_id]["owner_kind"] in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}
                        for parent_id in window
                    )
                ]
                window_parent_ids = set().union(*full_windows) if full_windows else set()
                blocker_parents: dict[tuple[str, str | None], set[str]] = defaultdict(set)
                for parent_id in sorted(window_parent_ids):
                    owner = owner_lookup.get(parent_id)
                    if owner is None or owner["owner_kind"] == "UNKNOWN":
                        blocker_parents[("UNKNOWN_A0_OWNER", None)].add(parent_id)
                    elif owner["owner_kind"] not in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}:
                        blocker_parents[
                            (
                                str(owner["semantic_label"] or owner["owner_kind"]),
                                None if owner["path_id"] is None else str(owner["path_id"]),
                            )
                        ].add(parent_id)
                if len(blocker_parents) > MAX_COMPONENT_OWNER_BLOCKERS:
                    raise ValueError(
                        f"component owner blocker categories exceed cap: {len(blocker_parents)} > "
                        f"{MAX_COMPONENT_OWNER_BLOCKERS}"
                    )
                blockers = [
                    {
                        "semantic_label": semantic_label,
                        "path_id": path_id,
                        "parent_count": len(blocker_parents[(semantic_label, path_id)]),
                    }
                    for semantic_label, path_id in sorted(
                        blocker_parents,
                        key=lambda item: (item[0], "" if item[1] is None else item[1]),
                    )
                ]
                unknown_owner = ("UNKNOWN_A0_OWNER", None) in blocker_parents
                matching_path_ids = _matching_paths(
                    anchor=anchor,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                    paths=paths,
                )
                normalized_mass = sum(
                    max(1, int(parent_by_id[item].get("normalized_support_mass") or 1)) for item in parent_ids
                )
                residual_mass = max(
                    (
                        sum(
                            max(1, int(parent_by_id[item].get("normalized_support_mass") or 1))
                            for item in window
                        )
                        for window in residual_windows
                    ),
                    default=0,
                )
                saved_upper = residual_mass * residual_cost
                reduced_lower = object_cost - saved_upper
                overflow = any(
                    bool(parent_by_id[item].get("mode_overflow", False))
                    or not bool(parent_by_id[item].get("transition_certificate_allowed", True))
                    for item in parent_ids
                )
                gates = {
                    "minimum_parent_count": len(parent_ids) >= 3,
                    "minimum_axial_span": upper - lower >= 2.0 * bin_width,
                    "minimum_occupied_bins": len(occupied_bins) >= 3,
                    "minimum_left_side_bins": len(left_bins) >= 3,
                    "minimum_right_side_bins": len(right_bins) >= 3,
                    "minimum_transition_rung_bins": len(rung_bins) >= 2,
                    "minimum_consecutive_occupied_run": _longest_consecutive_run(occupied_bins) >= 3,
                    "maximum_empty_bin_run": _maximum_empty_run(occupied_bins) <= 1,
                    "no_internal_branching": branch_count == 0,
                    "tls_axis": tls_angle <= 4.0,
                    "line_residual": p90_residual <= tube_tolerance,
                    "transition_certificate_complete": not overflow,
                    "certified_three_bin_window": bool(full_windows),
                    "a0_parent_ownership_available": not unknown_owner or bool(residual_windows),
                    "residual_certified_window": bool(residual_windows),
                    "duplicate_free": not matching_path_ids,
                    "negative_reduced_cost_lower_bound": reduced_lower < 0,
                }
                rejection_order = (
                    ("minimum_parent_count", "COMPONENT_TOO_SMALL"),
                    ("minimum_axial_span", "INSUFFICIENT_AXIAL_SPAN"),
                    ("minimum_occupied_bins", "INSUFFICIENT_OCCUPIED_BINS"),
                    ("minimum_left_side_bins", "INSUFFICIENT_LEFT_SIDE_BINS"),
                    ("minimum_right_side_bins", "INSUFFICIENT_RIGHT_SIDE_BINS"),
                    ("minimum_transition_rung_bins", "INSUFFICIENT_TRANSITION_RUNG_BINS"),
                    ("minimum_consecutive_occupied_run", "INSUFFICIENT_CONSECUTIVE_BIN_RUN"),
                    ("maximum_empty_bin_run", "EXCESSIVE_EMPTY_BIN_RUN"),
                    ("no_internal_branching", "INTERNAL_BRANCHING"),
                    ("tls_axis", "TLS_AXIS_MISMATCH"),
                    ("line_residual", "LINE_RESIDUAL_EXCEEDED"),
                    ("transition_certificate_complete", "MODE_OVERFLOW_CERTIFICATE_DISABLED"),
                    ("certified_three_bin_window", "NO_CERTIFIED_THREE_BIN_WINDOW"),
                    ("a0_parent_ownership_available", "A0_PARENT_OWNER_UNAVAILABLE"),
                    ("residual_certified_window", "NO_RESIDUAL_CERTIFIED_WINDOW"),
                    ("duplicate_free", "DUPLICATE_EXISTING_PATH"),
                    ("negative_reduced_cost_lower_bound", "MDL_NONNEGATIVE_LOWER_BOUND"),
                )
                first_rejection = next((code for gate, code in rejection_order if not gates[gate]), None)
                eligibility_code = first_rejection or "ELIGIBLE_DIAGNOSTIC_ONLY"
                rejection_counter[eligibility_code] += 1
                geometry_gate_names = (
                    "minimum_parent_count",
                    "minimum_axial_span",
                    "minimum_occupied_bins",
                    "minimum_left_side_bins",
                    "minimum_right_side_bins",
                    "minimum_transition_rung_bins",
                    "minimum_consecutive_occupied_run",
                    "maximum_empty_bin_run",
                    "no_internal_branching",
                    "tls_axis",
                    "line_residual",
                    "transition_certificate_complete",
                    "certified_three_bin_window",
                )
                valid_two_mode_connected_run = all(gates[name] for name in geometry_gate_names)
                if not valid_two_mode_connected_run:
                    first_failed_stage: int | None = 4
                elif not residual_windows:
                    first_failed_stage = 5
                elif matching_path_ids or reduced_lower >= 0:
                    first_failed_stage = 6
                else:
                    first_failed_stage = None
                owner_histogram = {
                    f"{semantic_label}|{path_id or ''}": len(parent_set)
                    for (semantic_label, path_id), parent_set in sorted(
                        blocker_parents.items(),
                        key=lambda item: (item[0][0], "" if item[0][1] is None else item[0][1]),
                    )
                }
                span_endpoints = [
                    [round(float(value), 6) for value in anchor + lower * axis],
                    [round(float(value), 6) for value in anchor + upper * axis],
                ]
                component_record = {
                    "component_id": component_id,
                    "matching_component_ids": [component_id],
                    "plane_pair_id": pair_id,
                    "matching_plane_pair_ids": [pair_id],
                    "intersection_span_endpoints": span_endpoints,
                    "support_parent_ids": parent_ids,
                    "parent_count": len(parent_ids),
                    "mode_count": len(mode_ids),
                    "occupied_bin_count": len(occupied_bins),
                    "left_side_bin_count": len(left_bins),
                    "right_side_bin_count": len(right_bins),
                    "transition_rung_bin_count": len(rung_bins),
                    "maximum_empty_bin_run": _maximum_empty_run(occupied_bins),
                    "internal_branch_count": int(branch_count),
                    "normalized_parent_mass": int(normalized_mass),
                    "residual_parent_mass_upper_bound": int(residual_mass),
                    "full_window_count": len(full_windows),
                    "residual_window_count": len(residual_windows),
                    "nonresidual_blocker_histogram": owner_histogram,
                    "owner_blockers": blockers,
                    "valid_two_mode_connected_run": valid_two_mode_connected_run,
                    "geometry_certificate_valid": valid_two_mode_connected_run,
                    "first_failed_stage": first_failed_stage,
                    "first_rejection_code": first_rejection,
                    "bounded_counters": {
                        "parent_count": len(parent_ids),
                        "mode_count": len(mode_ids),
                        "occupied_bin_count": len(occupied_bins),
                        "full_window_count": len(full_windows),
                        "residual_window_count": len(residual_windows),
                        "owner_blocker_category_count": len(blockers),
                    },
                }
                component_records.append(component_record)
                pair_component_records.append(component_record)
                birth_records.append(
                    {
                        "birth_diagnostic_id": _stable_id("BD", [pair_id, component_id]),
                        "birth_eligibility_id": _stable_id("BD", [pair_id, component_id]),
                        "plane_pair_id": pair_id,
                        "matching_plane_pair_ids": [pair_id],
                        "component_id": component_id,
                        "matching_component_ids": [component_id],
                        "intersection_span_endpoints": span_endpoints,
                        "support_parent_ids": parent_ids,
                        "geometry_gates": gates,
                        "valid_two_mode_connected_run": valid_two_mode_connected_run,
                        "geometry_certificate_valid": valid_two_mode_connected_run,
                        "first_failed_stage": first_failed_stage,
                        "first_rejection_code": first_rejection,
                        "eligibility_code": eligibility_code,
                        "bounded_counters": {
                            "parent_count": len(parent_ids),
                            "mode_count": len(mode_ids),
                            "occupied_bin_count": len(occupied_bins),
                            "full_window_count": len(full_windows),
                            "residual_window_count": len(residual_windows),
                            "owner_blocker_category_count": len(blockers),
                        },
                        "dimensionless_residuals": {
                            "parent_count_fraction": round(len(parent_ids) / 3.0, 6),
                            "axial_span_fraction": round(
                                (upper - lower) / (2.0 * bin_width), 6
                            ),
                            "occupied_bin_fraction": round(len(occupied_bins) / 3.0, 6),
                            "left_side_bin_fraction": round(len(left_bins) / 3.0, 6),
                            "right_side_bin_fraction": round(len(right_bins) / 3.0, 6),
                            "transition_rung_bin_fraction": round(len(rung_bins) / 2.0, 6),
                            "consecutive_run_fraction": round(
                                _longest_consecutive_run(occupied_bins) / 3.0, 6
                            ),
                            "tls_axis_fraction": round(tls_angle / 4.0, 6),
                            "line_p90_fraction": round(p90_residual / tube_tolerance, 6),
                            "empty_run_fraction": round(_maximum_empty_run(occupied_bins) / 1.0, 6),
                            "internal_branch_count": int(branch_count),
                            "mode_overflow_parent_fraction": round(
                                sum(
                                    bool(parent_by_id[item].get("mode_overflow", False))
                                    for item in parent_ids
                                )
                                / max(len(parent_ids), 1),
                                6,
                            ),
                            "residual_window_fraction": round(
                                len(residual_windows) / max(len(full_windows), 1), 6
                            ),
                            "duplicate_match_count_fraction": round(
                                len(matching_path_ids) / float(MAX_MATCHING_PATH_IDS), 6
                            ),
                            "reduced_cost_fraction": round(
                                reduced_lower / float(max(object_cost, 1)), 6
                            ),
                        },
                        "matching_path_ids": matching_path_ids,
                        "saved_residual_mass_upper_bound": int(residual_mass),
                        "object_cost_int": object_cost,
                        "birth_cost_int": object_cost,
                        "residual_cost_per_mass_int": residual_cost,
                        "saved_residual_cost_upper_bound_int": int(saved_upper),
                        "reduced_cost_lower_bound_int": int(reduced_lower),
                        "column_emitted": False,
                    }
                )
            if not components and supported_modes:
                pair_rejections["NO_TRANSITION_COMPONENT"] += 1
            rejection_counter.update(pair_rejections)
            largest = max(
                pair_component_records,
                key=lambda item: (item["parent_count"], item["mode_count"], item["component_id"]),
                default=None,
            )
            pair_merged_count = sum(
                1
                for record in multimodal_records
                if record["parent_atom_id"] in transverse_parents and record["v1_modes_merged"]
            )
            plane_pair_records.append(
                {
                    "plane_pair_id": pair_id,
                    "plane_ids": plane_ids,
                    "plane_angle_deg": round(float(plane_angle), 6),
                    "intersection_line_hash": _stable_id(
                        "IL",
                        [
                            [round(float(value), 6) for value in anchor],
                            _rounded_vector(axis),
                        ],
                    ),
                    "intersection_axis": _rounded_vector(axis),
                    "transverse_parent_count": len(transverse_parents),
                    "left_mode_count": len(left_modes),
                    "right_mode_count": len(right_modes),
                    "two_mode_rung_count": len(rung_edges),
                    "v1_merged_parent_count": pair_merged_count,
                    "component_count": len(components),
                    "largest_component": None
                    if largest is None
                    else {
                        key: largest[key]
                        for key in (
                            "parent_count",
                            "mode_count",
                            "occupied_bin_count",
                            "left_side_bin_count",
                            "right_side_bin_count",
                            "transition_rung_bin_count",
                            "maximum_empty_bin_run",
                            "internal_branch_count",
                            "normalized_parent_mass",
                            "residual_parent_mass_upper_bound",
                        )
                    },
                    "rejection_counts": dict(sorted(pair_rejections.items())),
                }
            )

    hard_caps = {
        "plane_pairs": MAX_DIAGNOSTIC_PLANE_PAIRS,
        "multimodal_parents": MAX_MULTIMODAL_PARENT_RECORDS,
        "components": MAX_DIAGNOSTIC_COMPONENTS,
        "owner_blocker_categories_per_component": MAX_COMPONENT_OWNER_BLOCKERS,
        "matching_path_ids_per_birth": MAX_MATCHING_PATH_IDS,
        "normal_modes_per_parent": MAX_NORMAL_MODES_PER_PARENT,
    }
    result = {
        "schema_version": 1,
        "kind": TRANSITION_DIAGNOSTIC_KIND,
        "inference_safe": True,
        "research_only": True,
        "runtime_integration": False,
        "birth_columns_enabled": False,
        "prediction_mutated": False,
        "truth_inputs_consumed": False,
        "source_points_serialized": False,
        "scan_sha256": _validate_sha256("scan_sha256", scan_sha256),
        "candidate_pool_sha256": _validate_sha256("candidate_pool_sha256", candidate_pool_sha256),
        "graph_sha256": _validate_sha256("graph_sha256", graph_sha256),
        "a0_solution_sha256": _validate_sha256("a0_solution_sha256", a0_solution_sha256),
        "transition_table_hashes": transition_table_hashes,
        "atomization_profiles": {
            "prediction": "SURFACE_ATOM_V1",
            "diagnostic_probe": diagnostic_profile,
            "support": profile,
        },
        "a0_parent_ownership_projection": ownership_projection,
        "hard_caps": hard_caps,
        "builder_caps": dict(sorted(builder_caps.items())),
        "thresholds": {
            "local_spacing_mm": round(spacing, 6),
            "axial_bin_mm": round(bin_width, 6),
            "tube_tolerance_mm": round(tube_tolerance, 6),
            "plane_mode_normal_deg": 15.0,
            "minimum_transition_mode_angle_deg": 20.0,
            "minimum_parent_count": 3,
            "minimum_axial_span_bins": 2,
            "minimum_occupied_bins": 3,
            "minimum_side_bins": 3,
            "minimum_transition_rung_bins": 2,
            "minimum_consecutive_occupied_run": 3,
            "maximum_empty_bin_run": 1,
            "maximum_tls_axis_deg": 4.0,
            "matching_path_axis_angle_deg": FROZEN_LINE_AXIS_ANGLE_DEG,
            "matching_path_transverse_offset_mm": FROZEN_LINE_TRANSVERSE_OFFSET_MM,
            "matching_path_component_span_overlap": FROZEN_LINE_GOLD_SPAN_OVERLAP,
            "matching_path_existing_span_overlap": FROZEN_LINE_PREDICTION_SPAN_OVERLAP,
        },
        "case_counters": {
            "plane_count": len(planes),
            "plane_pair_count": pair_count,
            "angle_eligible_pair_count": angle_eligible_pairs,
            "locally_supported_pair_count": locally_supported_pairs,
            "plane_support_component_count": len(component_records),
            "raw_multimodal_parent_count": raw_multimodal_count,
            "v1_merged_parent_count": v1_merged_count,
            "transition_edge_count": len(mode_edges),
            "component_count": len(component_records),
            "birth_eligible_diagnostic_count": sum(
                record["eligibility_code"] == "ELIGIBLE_DIAGNOSTIC_ONLY" for record in birth_records
            ),
            "emitted_birth_column_count": 0,
            "rejection_counts": dict(sorted(rejection_counter.items())),
        },
        "plane_pair_records": plane_pair_records,
        "multimodal_parent_records": multimodal_records,
        "transition_component_records": component_records,
        "birth_eligibility_records": birth_records,
    }
    _recursive_key_check(result)
    return result
