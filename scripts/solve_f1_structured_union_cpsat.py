#!/usr/bin/env python3
"""Exact research-only CP-SAT solver for the F1 typed union graph.

This worker is deliberately isolated from runtime/main.  It consumes only the
bounded scan-derived graph, proves every optimization and feasibility stage,
and emits atom ownership needed by the residual-birth loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from ortools.sat.python import cp_model


BEND = "BEND_STRAIGHT_COUNTED"
CORNER = "CORNER_NONCOUNTED"
PERIMETER = "PERIMETER_NONCOUNTED"
HOLE = "HOLE_NONCOUNTED"
JUNCTION = "JUNCTION_NONCOUNTED"
CURVE = "CURVE_NONCOUNTED"
RESIDUAL = "RESIDUAL_NONCOUNTED"
LABELS = (BEND, CORNER, PERIMETER, HOLE, JUNCTION, CURVE, RESIDUAL)
NON_RESIDUAL_LABELS = LABELS[:-1]

ORDINARY_END = "ORDINARY_END"
CONTINUATION_END = "CONTINUATION_END"
MULTIWAY_JUNCTION = "MULTIWAY_JUNCTION"
NODE_TYPES = (ORDINARY_END, CONTINUATION_END, MULTIWAY_JUNCTION)

ENERGY_SCALE = 1_000_000
C_RESIDUAL = ENERGY_SCALE
DEFAULT_MIN_BINS = {label: 2 for label in NON_RESIDUAL_LABELS}
DEFAULT_MIN_MASS = {label: 2 for label in NON_RESIDUAL_LABELS}
DEFAULT_MIN_SIDE_BINS = {BEND: 1}
CONTINUATION_RESIDUAL_KEYS = {
    "axis",
    "perpendicular",
    "gap",
    "chord",
    "refit",
    "overlap",
}


def _configure(solver: cp_model.CpSolver) -> None:
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 0
    solver.parameters.linearization_level = 2


def _status_name(solver: cp_model.CpSolver, status: int) -> str:
    return str(solver.StatusName(status))


def _solve_optimal(model: cp_model.CpModel, *, stage: str) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    _configure(solver)
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"structured solver {stage} requires OPTIMAL, got "
            f"{_status_name(solver, status)}"
        )
    return solver


def _prove_trial(model: cp_model.CpModel, *, stage: str) -> int:
    """Return OPTIMAL or INFEASIBLE; reject every unproved solver state."""

    model.Minimize(0)
    solver = cp_model.CpSolver()
    _configure(solver)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.INFEASIBLE):
        raise RuntimeError(
            f"structured solver {stage} did not prove feasibility, got "
            f"{_status_name(solver, status)}"
        )
    return status


def _bool_in_clone(model: cp_model.CpModel, variable: cp_model.IntVar) -> cp_model.IntVar:
    return model.get_bool_var_from_proto_index(variable.Index())


def _fix_bool_lexicographically(
    model: cp_model.CpModel,
    ordered_variables: Sequence[cp_model.IntVar],
    *,
    prefer: int,
    stage: str,
) -> None:
    for index, variable in enumerate(ordered_variables):
        trial = model.Clone()
        trial.Add(_bool_in_clone(trial, variable) == prefer)
        status = _prove_trial(trial, stage=f"{stage}[{index}]")
        model.Add(variable == (prefer if status == cp_model.OPTIMAL else 1 - prefer))


def _fix_label_lexicographically(
    model: cp_model.CpModel,
    *,
    path_ids: Sequence[str],
    label_vars: Mapping[tuple[str, str], cp_model.IntVar],
) -> None:
    """Fix one-hot rows in frozen path/ontology order with proved feasibility."""

    for path_index, path_id in enumerate(path_ids):
        selected: str | None = None
        for label_index, label in enumerate(LABELS):
            trial = model.Clone()
            trial.Add(_bool_in_clone(trial, label_vars[(path_id, label)]) == 1)
            status = _prove_trial(
                trial,
                stage=f"canonical-label[{path_index},{label_index}]",
            )
            if status == cp_model.OPTIMAL:
                selected = label
                break
        if selected is None:
            raise RuntimeError(f"no proved canonical label for path {path_id}")
        model.Add(label_vars[(path_id, selected)] == 1)


def _fix_one_hot_lexicographically(
    model: cp_model.CpModel,
    ordered_groups: Sequence[Sequence[cp_model.IntVar]],
    *,
    stage: str,
) -> None:
    """Choose the first feasible value in each frozen one-hot ordinal."""

    for group_index, variables in enumerate(ordered_groups):
        selected: cp_model.IntVar | None = None
        for value_index, variable in enumerate(variables):
            trial = model.Clone()
            trial.Add(_bool_in_clone(trial, variable) == 1)
            status = _prove_trial(
                trial,
                stage=f"{stage}[{group_index},{value_index}]",
            )
            if status == cp_model.OPTIMAL:
                selected = variable
                break
            model.Add(variable == 0)
        if selected is None:
            raise RuntimeError(f"no proved canonical value for {stage}[{group_index}]")
        model.Add(selected == 1)


def _path_pair(path: Mapping[str, Any]) -> tuple[str, str] | None:
    value = path.get("primary_panel_pair")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    pair = tuple(sorted((str(value[0]), str(value[1]))))
    if not all(pair) or pair[0] == pair[1]:
        return None
    return pair


def _as_pair(value: Any, *, field: str) -> tuple[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two IDs")
    left, right = str(value[0]), str(value[1])
    if not left or not right or left == right:
        raise ValueError(f"{field} must contain distinct nonempty IDs")
    return tuple(sorted((left, right)))


def _continuation_records(
    graph: Mapping[str, Any],
    *,
    path_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str, int, int]] = set()
    for index, raw in enumerate(graph.get("continuations") or (), start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("continuation records must be objects")
        path_ids = raw.get("path_ids")
        endpoint_ids = raw.get("endpoint_ids")
        if not isinstance(path_ids, Sequence) or isinstance(path_ids, (str, bytes)) or len(path_ids) != 2:
            raise ValueError("continuation must connect exactly two paths")
        if not isinstance(endpoint_ids, Sequence) or isinstance(endpoint_ids, (str, bytes)) or len(endpoint_ids) != 2:
            raise ValueError("continuation endpoint_ids must contain two endpoints")
        left, right = str(path_ids[0]), str(path_ids[1])
        left_endpoint, right_endpoint = int(endpoint_ids[0]), int(endpoint_ids[1])
        if left == right or left not in path_by_id or right not in path_by_id:
            raise ValueError("continuation references invalid paths")
        if left_endpoint not in (0, 1) or right_endpoint not in (0, 1):
            raise ValueError("continuation endpoints must be 0 or 1")
        pair = _path_pair(path_by_id[left])
        if pair is None or pair != _path_pair(path_by_id[right]):
            raise ValueError("counted continuations require the same incident panel pair")

        residuals = raw.get("dimensionless_residuals")
        if not isinstance(residuals, Mapping) or set(residuals) != CONTINUATION_RESIDUAL_KEYS:
            raise ValueError("incomplete continuation residual certificate")
        numeric_residuals = {key: float(residuals[key]) for key in sorted(residuals)}
        if any(not math.isfinite(value) for value in numeric_residuals.values()):
            raise ValueError("non-finite continuation residual")
        if any(value < 0.0 or value > 1.0 for value in numeric_residuals.values()):
            raise ValueError("continuation outside frozen engineering tolerance")
        if raw.get("engineering_tolerance_profile") != "CONTINUATION_V1":
            raise ValueError("continuation has an unknown engineering tolerance profile")
        if not raw.get("bridge_certificate_sha256"):
            raise ValueError("continuation lacks scan-support certificate")
        bridge_ids = sorted({str(value) for value in raw.get("bridge_atom_ids") or ()})

        continuation_id = str(raw.get("continuation_id") or f"CONT_{index:06d}")
        if continuation_id in seen_ids:
            raise ValueError("continuation IDs must be unique")
        seen_ids.add(continuation_id)
        edge = (left, right, left_endpoint, right_endpoint)
        reverse = (right, left, right_endpoint, left_endpoint)
        if edge in seen_edges or reverse in seen_edges:
            raise ValueError("duplicate continuation edge")
        seen_edges.add(edge)
        records.append(
            {
                "continuation_id": continuation_id,
                "path_ids": [left, right],
                "endpoint_ids": [left_endpoint, right_endpoint],
                "dimensionless_residuals": numeric_residuals,
                "engineering_tolerance_profile": "CONTINUATION_V1",
                "bridge_atom_ids": bridge_ids,
                "bridge_certificate_sha256": str(raw["bridge_certificate_sha256"]),
            }
        )

    # CONTINUATION_V1 proposals are proof-friendly chains.  Reject cyclic input
    # instead of silently relying on the objective to avoid a cycle.
    parent = {path_id: path_id for path_id in path_by_id}

    def find(path_id: str) -> str:
        while parent[path_id] != path_id:
            parent[path_id] = parent[parent[path_id]]
            path_id = parent[path_id]
        return path_id

    endpoint_degree: dict[tuple[str, int], int] = defaultdict(int)
    for record in sorted(records, key=lambda item: str(item["continuation_id"])):
        left, right = record["path_ids"]
        left_endpoint, right_endpoint = record["endpoint_ids"]
        endpoint_degree[(left, left_endpoint)] += 1
        endpoint_degree[(right, right_endpoint)] += 1
        if endpoint_degree[(left, left_endpoint)] > 1 or endpoint_degree[(right, right_endpoint)] > 1:
            raise ValueError("continuation proposal endpoint degree exceeds one")
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            raise ValueError("continuation proposal graph must be acyclic")
        parent[right_root] = left_root
    return sorted(records, key=lambda item: str(item["continuation_id"]))


def _label_eligibility(path: Mapping[str, Any]) -> dict[str, bool]:
    raw = path.get("label_eligibility")
    if raw is None:
        # Legacy research graphs remain readable, but exact support constraints
        # still make unsupported semantic selections impossible.
        return {label: True for label in LABELS}
    if not isinstance(raw, Mapping) or set(raw) != set(LABELS):
        raise ValueError("path label_eligibility must enumerate the seven-label ontology")
    eligibility = {label: bool(raw[label]) for label in LABELS}
    if not eligibility[RESIDUAL]:
        raise ValueError("every latent path must remain residual-eligible")
    return eligibility


def _incidence_is_certified(path: Mapping[str, Any]) -> bool:
    certificate = path.get("incidence_certificate")
    if not isinstance(certificate, Mapping):
        return False
    residuals = certificate.get("max_dimensionless_residuals")
    required = {"perpendicular", "endpoint", "normal", "axis"}
    if (
        certificate.get("tolerance_profile") != "INCIDENCE_V1"
        or not certificate.get("atom_table_sha256")
        or not isinstance(residuals, Mapping)
        or set(residuals) != required
    ):
        return False
    try:
        values = [float(residuals[key]) for key in required]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)


def _normalized_atom_masses(atom_table: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    areas = [
        float(atom.get("area_proxy_mm2") or 0.0)
        for atom in atom_table
        if math.isfinite(float(atom.get("area_proxy_mm2") or 0.0))
        and float(atom.get("area_proxy_mm2") or 0.0) > 0.0
    ]
    area_median = median(areas) if areas else 1.0
    masses: dict[str, int] = {}
    for atom in atom_table:
        atom_id = str(atom["atom_id"])
        raw = atom.get("normalized_support_mass")
        if raw is None:
            area = float(atom.get("area_proxy_mm2") or area_median)
            mass = max(1, min(4, int(round(area / area_median))))
        else:
            mass = int(raw)
            if mass < 1 or mass > 4:
                raise ValueError(
                    f"atom {atom_id} normalized_support_mass must be in [1,4]"
                )
        masses[atom_id] = mass
    return masses


def _atom_bin_map(
    path: Mapping[str, Any],
    atom_ids: Sequence[str],
    atom_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    for field in ("support_atom_bins", "support_atom_bin_ids", "support_bin_by_atom"):
        raw = path.get(field)
        if isinstance(raw, Mapping):
            return {atom_id: str(raw[atom_id]) for atom_id in atom_ids if atom_id in raw}
    raw_bins = path.get("support_bin_ids")
    if isinstance(raw_bins, Mapping):
        return {atom_id: str(raw_bins[atom_id]) for atom_id in atom_ids if atom_id in raw_bins}
    if isinstance(raw_bins, Sequence) and not isinstance(raw_bins, (str, bytes)) and len(raw_bins) == len(atom_ids):
        return {atom_id: str(bin_id) for atom_id, bin_id in zip(atom_ids, raw_bins)}

    # A compact graph may serialize only the occupied-bin set.  Reconstruct the
    # atom-to-bin incidence from already serialized centroids and path geometry;
    # this does not create points or support evidence.
    endpoints = path.get("span_endpoints")
    axis = path.get("axis_direction")
    if (
        isinstance(endpoints, Sequence)
        and not isinstance(endpoints, (str, bytes))
        and len(endpoints) == 2
        and isinstance(axis, Sequence)
        and not isinstance(axis, (str, bytes))
        and len(axis) == 3
    ):
        p0 = tuple(float(value) for value in endpoints[0])
        unit = tuple(float(value) for value in axis)
        norm = math.sqrt(sum(value * value for value in unit))
        if norm > 1e-12:
            unit = tuple(value / norm for value in unit)
            spacings = [
                float(atom_by_id[atom_id].get("local_spacing_mm") or 0.0)
                for atom_id in atom_ids
                if float(atom_by_id[atom_id].get("local_spacing_mm") or 0.0) > 0.0
            ]
            width = max(3.0, 2.0 * (median(spacings) if spacings else 1.5))
            result: dict[str, str] = {}
            for atom_id in atom_ids:
                centroid = tuple(float(value) for value in atom_by_id[atom_id]["centroid"])
                projection = sum((centroid[index] - p0[index]) * unit[index] for index in range(3))
                result[atom_id] = str(math.floor(projection / width))
            return result
    return {}


def _atom_side_map(
    path: Mapping[str, Any], atom_ids: Sequence[str], atom_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[str, ...]]:
    raw = (
        path.get("support_atom_sides")
        or path.get("support_atom_side_ids")
        or path.get("support_sides_by_atom")
    )
    if isinstance(raw, Mapping):
        result: dict[str, tuple[str, ...]] = {}
        for atom_id in atom_ids:
            value = raw.get(atom_id) or ()
            if isinstance(value, str):
                value = [value]
            result[atom_id] = tuple(sorted({str(side) for side in value if str(side)}))
        return result
    pair = _path_pair(path)
    if pair is None:
        return {}
    result = {}
    for atom_id in atom_ids:
        incident = atom_by_id[atom_id].get("incident_plane_ids") or atom_by_id[atom_id].get("side_ids") or ()
        result[atom_id] = tuple(side for side in pair if side in {str(value) for value in incident})
    return result


def _class_minimum(
    graph: Mapping[str, Any], path: Mapping[str, Any], label: str, field: str, default: int
) -> int:
    value: Any = default
    for owner in (graph, path):
        by_class = owner.get("class_ownership_minima")
        if isinstance(by_class, Mapping) and isinstance(by_class.get(label), Mapping):
            value = by_class[label].get(field, value)
        direct = owner.get(f"class_{field}")
        if isinstance(direct, Mapping):
            value = direct.get(label, value)
    result = int(value)
    if result < 0:
        raise ValueError(f"negative {field} for {label}")
    return result


def _factor_implications(
    factor: Mapping[str, Any], *, allow_empty: bool = False
) -> list[tuple[str, str]]:
    raw = factor.get("label_implications")
    implications: list[tuple[str, str]] = []
    if raw is not None:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("factor label_implications must be a sequence")
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("factor label implication must be an object")
            implications.append((str(item["path_id"]), str(item["label"])))
    else:
        implications.extend((str(path_id), BEND) for path_id in factor.get("counted_path_ids") or ())
        implications.extend((str(path_id), CORNER) for path_id in factor.get("noncounted_path_ids") or ())
    if not implications and not allow_empty:
        raise ValueError("factor must imply at least one typed path label")
    return sorted(set(implications))


def _validate_audit_only_factor(
    factor: Mapping[str, Any],
    *,
    path_ids: set[str],
    atom_ids: set[str],
    all_factor_ids: set[str],
) -> None:
    """Validate a disabled proposal without admitting it to the exact model."""

    factor_id = str(factor.get("factor_id") or "")
    if not factor_id:
        raise ValueError("audit-only factor ID must be nonempty")
    if not str(factor.get("weight_status") or "").startswith("DISABLED"):
        raise ValueError("audit-only factor must have disabled weight_status")
    for path_id, label in _factor_implications(factor, allow_empty=True):
        if path_id not in path_ids or label not in LABELS:
            raise ValueError(
                f"disabled audit factor {factor_id} references an invalid typed path"
            )
    for item in factor.get("endpoint_type_implications") or ():
        if not isinstance(item, Mapping):
            raise ValueError("disabled audit factor endpoint implication must be an object")
        path_id = str(item.get("path_id") or "")
        endpoint = int(item.get("endpoint_id", -1))
        node_type = str(item.get("node_type") or "")
        if path_id not in path_ids or endpoint not in (0, 1) or node_type not in NODE_TYPES:
            raise ValueError(
                f"disabled audit factor {factor_id} has invalid endpoint implication"
            )
    required_atoms = {
        str(value) for value in factor.get("required_support_atom_ids") or ()
    }
    if not required_atoms <= atom_ids:
        raise ValueError(
            f"disabled audit factor {factor_id} references absent support atoms"
        )
    for other_id in factor.get("conflicting_factor_ids") or ():
        if str(other_id) not in all_factor_ids:
            raise ValueError("disabled audit factor conflict references an absent factor")
    if factor.get("energy_int") is not None:
        int(factor["energy_int"])
    else:
        score = float(factor.get("score") or 0.0)
        if not math.isfinite(score):
            raise ValueError("disabled audit factor score must be finite")


def _factor_conflict_pairs(
    graph: Mapping[str, Any],
    factors: Sequence[Mapping[str, Any]],
    implications: Mapping[str, Sequence[tuple[str, str]]],
    *,
    all_factor_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    factor_ids = {str(factor["factor_id"]) for factor in factors}
    valid_factor_ids = all_factor_ids if all_factor_ids is not None else factor_ids
    conflicts: set[tuple[str, str]] = set()
    raw_relations = graph.get("factor_conflicts") or graph.get("conflicting_factor_pairs") or ()
    for raw in raw_relations:
        value = raw.get("factor_ids") if isinstance(raw, Mapping) else raw
        pair = _as_pair(value, field="factor conflict")
        if not set(pair) <= valid_factor_ids:
            raise ValueError("factor conflict references an absent factor")
        if set(pair) <= factor_ids:
            conflicts.add(pair)

    for left_index, left in enumerate(factors):
        left_id = str(left["factor_id"])
        for right_id in left.get("conflicting_factor_ids") or ():
            right_id = str(right_id)
            if right_id not in valid_factor_ids:
                raise ValueError("factor conflict references an absent factor")
            if right_id != left_id and right_id in factor_ids:
                conflicts.add(tuple(sorted((left_id, right_id))))
        left_labels = dict(implications[left_id])
        left_atoms = {str(value) for value in left.get("required_support_atom_ids") or ()}
        left_nodes = {
            (str(item["path_id"]), int(item["endpoint_id"])): str(item["node_type"])
            for item in left.get("endpoint_type_implications") or ()
        }
        for right in factors[left_index + 1 :]:
            right_id = str(right["factor_id"])
            right_labels = dict(implications[right_id])
            contradictory = any(
                path_id in right_labels and right_labels[path_id] != label
                for path_id, label in left_labels.items()
            )
            shared_atoms = bool(left_atoms & {str(value) for value in right.get("required_support_atom_ids") or ()})
            right_nodes = {
                (str(item["path_id"]), int(item["endpoint_id"])): str(item["node_type"])
                for item in right.get("endpoint_type_implications") or ()
            }
            incompatible_nodes = any(
                endpoint in right_nodes and right_nodes[endpoint] != node_type
                for endpoint, node_type in left_nodes.items()
            )
            if contradictory or shared_atoms or incompatible_nodes:
                conflicts.add(tuple(sorted((left_id, right_id))))
    return sorted(conflicts)


def _duplicate_pairs(graph: Mapping[str, Any], path_ids: set[str]) -> list[tuple[str, str]]:
    raw_relations = (
        graph.get("duplicate_pairs")
        or graph.get("hard_duplicate_pairs")
        or graph.get("duplicate_relations")
        or ()
    )
    result: set[tuple[str, str]] = set()
    for raw in raw_relations:
        if isinstance(raw, Mapping) and "hard" in raw and not bool(raw.get("hard")):
            continue
        value = raw.get("path_ids") if isinstance(raw, Mapping) else raw
        pair = _as_pair(value, field="hard duplicate relation")
        if not set(pair) <= path_ids:
            raise ValueError("hard duplicate relation references an absent path")
        result.add(pair)
    return sorted(result)


def _unary_cost(path: Mapping[str, Any], label: str) -> int:
    for field in (
        "unary_energy_int_by_label",
        "label_energy_int",
        "semantic_energy_int",
        "class_energy_int",
        "label_costs_int",
        "unary_costs_int",
    ):
        raw = path.get(field)
        if isinstance(raw, Mapping) and label in raw:
            return int(raw[label])
    return 0


def solve(graph: dict[str, Any]) -> dict[str, Any]:
    contract = dict(graph.get("inference_contract") or {})
    if any(bool(contract.get(field)) for field in (
        "uses_part_identifier", "uses_filename_features", "uses_expected_count", "uses_manual_truth", "uses_cad"
    )):
        raise ValueError("typed graph violates the strict-blind inference contract")
    if contract.get("source_point_serialization") is not False:
        raise ValueError("typed graph must explicitly prohibit source-point serialization")
    if tuple(graph.get("semantic_labels") or ()) != LABELS:
        raise ValueError("typed graph semantic ontology mismatch")

    paths = sorted(graph.get("latent_paths") or (), key=lambda item: str(item["latent_path_id"]))
    all_factors = sorted(
        graph.get("topology_factors") or (),
        key=lambda item: str(item["factor_id"]),
    )
    disabled_audit_factors = [
        factor
        for factor in all_factors
        if str(factor.get("weight_status") or "").startswith("DISABLED")
    ]
    factors = [factor for factor in all_factors if factor not in disabled_audit_factors]
    path_ids = [str(path["latent_path_id"]) for path in paths]
    path_id_set = set(path_ids)
    path_by_id = {str(path["latent_path_id"]): path for path in paths}
    if len(path_ids) != len(path_id_set):
        raise ValueError("latent path IDs must be unique")
    factor_ids = [str(factor["factor_id"]) for factor in all_factors]
    if len(factor_ids) != len(set(factor_ids)):
        raise ValueError("topology factor IDs must be unique")
    for path in paths:
        if bool(path.get("support_atom_ids_truncated")):
            raise ValueError("exact solve rejects truncated support incidence")

    atom_table = [dict(record) for record in graph.get("support_atom_table") or ()]
    atom_by_id = {str(record["atom_id"]): record for record in atom_table}
    if len(atom_by_id) != len(atom_table):
        raise ValueError("support atom IDs must be unique")
    atom_masses = _normalized_atom_masses(atom_table)
    for factor in disabled_audit_factors:
        _validate_audit_only_factor(
            factor,
            path_ids=path_id_set,
            atom_ids=set(atom_by_id),
            all_factor_ids=set(factor_ids),
        )

    model = cp_model.CpModel()
    label_vars: dict[tuple[str, str], cp_model.IntVar] = {}
    active_vars: dict[str, cp_model.IntVar] = {}
    eligibility_by_path: dict[str, dict[str, bool]] = {}
    for path in paths:
        path_id = str(path["latent_path_id"])
        eligibility = _label_eligibility(path)
        eligibility_by_path[path_id] = eligibility
        row = []
        for label in LABELS:
            variable = model.NewBoolVar(f"z__{path_id}__{label}")
            label_vars[(path_id, label)] = variable
            row.append(variable)
            if not eligibility[label]:
                model.Add(variable == 0)
        model.AddExactlyOne(row)
        active = model.NewBoolVar(f"active__{path_id}")
        active_vars[path_id] = active
        model.Add(active + label_vars[(path_id, RESIDUAL)] == 1)
        if not bool(path.get("incidence_complete")) or not _incidence_is_certified(path):
            model.Add(active == 0)

    # Endpoint types are inferred variables, not upstream labels.
    endpoint_vars: dict[tuple[str, int, str], cp_model.IntVar] = {}
    for path in paths:
        path_id = str(path["latent_path_id"])
        admissibility = path.get("endpoint_type_eligibility")
        for endpoint in (0, 1):
            row = []
            for node_type in NODE_TYPES:
                variable = model.NewBoolVar(f"j__{path_id}__{endpoint}__{node_type}")
                endpoint_vars[(path_id, endpoint, node_type)] = variable
                row.append(variable)
                endpoint_rule = None
                if isinstance(admissibility, Sequence) and not isinstance(admissibility, (str, bytes)) and len(admissibility) == 2:
                    endpoint_rule = admissibility[endpoint]
                elif isinstance(admissibility, Mapping):
                    endpoint_rule = admissibility.get(str(endpoint), admissibility.get(endpoint))
                if isinstance(endpoint_rule, Mapping) and not bool(endpoint_rule.get(node_type, False)):
                    model.Add(variable == 0)
            model.AddExactlyOne(row)
            model.Add(endpoint_vars[(path_id, endpoint, ORDINARY_END)] == 1).OnlyEnforceIf(label_vars[(path_id, RESIDUAL)])
            model.Add(endpoint_vars[(path_id, endpoint, MULTIWAY_JUNCTION)] == 1).OnlyEnforceIf(label_vars[(path_id, CORNER)])

    # Typed relational factors. Disjoint factors coexist; only explicit or
    # mechanically provable conflicts receive pairwise exclusions.
    factor_vars: dict[str, cp_model.IntVar] = {}
    factor_implications: dict[str, list[tuple[str, str]]] = {}
    factor_energy: dict[str, int] = {}
    for factor in factors:
        factor_id = str(factor["factor_id"])
        variable = model.NewBoolVar(f"h__{factor_id}")
        factor_vars[factor_id] = variable
        implications = _factor_implications(factor)
        factor_implications[factor_id] = implications
        for path_id, label in implications:
            if path_id not in path_id_set or label not in LABELS:
                raise ValueError(f"factor {factor_id} references an invalid typed path")
            model.Add(variable <= label_vars[(path_id, label)])
        for item in factor.get("endpoint_type_implications") or ():
            path_id = str(item["path_id"])
            endpoint = int(item["endpoint_id"])
            node_type = str(item["node_type"])
            if path_id not in path_id_set or endpoint not in (0, 1) or node_type not in NODE_TYPES:
                raise ValueError(f"factor {factor_id} has invalid endpoint implication")
            model.Add(variable <= endpoint_vars[(path_id, endpoint, node_type)])
        factor_energy[factor_id] = int(
            factor.get("energy_int")
            if factor.get("energy_int") is not None
            else -max(1, round(float(factor.get("score") or 0.0) * ENERGY_SCALE))
        )
    conflict_pairs = _factor_conflict_pairs(
        graph,
        factors,
        factor_implications,
        all_factor_ids=set(factor_ids),
    )
    for left, right in conflict_pairs:
        model.Add(factor_vars[left] + factor_vars[right] <= 1)

    duplicate_pairs = _duplicate_pairs(graph, path_id_set)
    for left, right in duplicate_pairs:
        model.Add(label_vars[(left, BEND)] + label_vars[(right, BEND)] <= 1)

    # Exclusive compact-atom ownership with mass, axial distribution, and side
    # minima. Unsupported paths were already hard-fixed to residual above.
    atom_to_paths: dict[str, list[str]] = defaultdict(list)
    atom_bins_by_path: dict[str, dict[str, str]] = {}
    atom_sides_by_path: dict[str, dict[str, tuple[str, ...]]] = {}
    for path in paths:
        path_id = str(path["latent_path_id"])
        atom_ids = sorted({str(value) for value in path.get("support_atom_ids") or ()})
        absent = set(atom_ids) - set(atom_by_id)
        if absent:
            raise ValueError(f"path {path_id} references absent support atoms: {sorted(absent)}")
        atom_bins_by_path[path_id] = _atom_bin_map(path, atom_ids, atom_by_id)
        atom_sides_by_path[path_id] = _atom_side_map(path, atom_ids, atom_by_id)
        complete_bin_incidence = bool(atom_ids) and set(atom_bins_by_path[path_id]) == set(atom_ids)
        if not _incidence_is_certified(path) or not complete_bin_incidence:
            # Hard support admissibility is not weakened when a learned class
            # minimum happens to be zero.
            model.Add(active_vars[path_id] == 0)
        for atom_id in atom_ids:
            atom_to_paths[atom_id].append(path_id)

    ownership_vars: dict[tuple[str, str], cp_model.IntVar] = {}
    residual_owner_vars: dict[str, cp_model.IntVar] = {}
    owners_by_path: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    owners_by_path_bin: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    owners_by_path_side_bin: dict[tuple[str, str, str], list[cp_model.IntVar]] = defaultdict(list)
    for atom_id in sorted(atom_by_id):
        residual_owner = model.NewBoolVar(f"o__{atom_id}__RESIDUAL")
        residual_owner_vars[atom_id] = residual_owner
        row: list[cp_model.IntVar] = [residual_owner]
        for path_id in sorted(set(atom_to_paths.get(atom_id) or ())):
            owner = model.NewBoolVar(f"o__{atom_id}__{path_id}")
            ownership_vars[(atom_id, path_id)] = owner
            owners_by_path[path_id].append(owner)
            model.Add(owner <= active_vars[path_id])
            bin_id = atom_bins_by_path[path_id].get(atom_id)
            if bin_id is not None:
                owners_by_path_bin[(path_id, bin_id)].append(owner)
                for side in atom_sides_by_path[path_id].get(atom_id, ()):
                    owners_by_path_side_bin[(path_id, side, bin_id)].append(owner)
            row.append(owner)
        model.AddExactlyOne(row)

    bin_vars: dict[tuple[str, str], cp_model.IntVar] = {}
    for key, owners in sorted(owners_by_path_bin.items()):
        path_id, bin_id = key
        variable = model.NewBoolVar(f"bin__{path_id}__{bin_id}")
        bin_vars[key] = variable
        for owner in owners:
            model.Add(owner <= variable)
        model.Add(variable <= sum(owners))
    side_bin_vars: dict[tuple[str, str, str], cp_model.IntVar] = {}
    for key, owners in sorted(owners_by_path_side_bin.items()):
        path_id, side, bin_id = key
        variable = model.NewBoolVar(f"side_bin__{path_id}__{side}__{bin_id}")
        side_bin_vars[key] = variable
        for owner in owners:
            model.Add(owner <= variable)
        model.Add(variable <= sum(owners))

    for path in paths:
        path_id = str(path["latent_path_id"])
        path_bins = [variable for (owner_path, _), variable in bin_vars.items() if owner_path == path_id]
        weighted_mass = sum(
            atom_masses[atom_id] * variable
            for (atom_id, owner_path), variable in ownership_vars.items()
            if owner_path == path_id
        )
        for label in NON_RESIDUAL_LABELS:
            min_bins = _class_minimum(graph, path, label, "min_bins", DEFAULT_MIN_BINS[label])
            min_mass = _class_minimum(graph, path, label, "min_mass", DEFAULT_MIN_MASS[label])
            model.Add(sum(path_bins) >= min_bins).OnlyEnforceIf(label_vars[(path_id, label)])
            model.Add(weighted_mass >= min_mass).OnlyEnforceIf(label_vars[(path_id, label)])
        pair = _path_pair(path)
        min_side_bins = _class_minimum(graph, path, BEND, "min_side_bins", DEFAULT_MIN_SIDE_BINS[BEND])
        if pair is None or any(
            not any(owner_path == path_id and side == panel for owner_path, side, _ in side_bin_vars)
            for panel in pair or ()
        ):
            if min_side_bins > 0:
                model.Add(label_vars[(path_id, BEND)] == 0)
        else:
            for panel in pair:
                variables = [
                    variable
                    for (owner_path, side, _), variable in side_bin_vars.items()
                    if owner_path == path_id and side == panel
                ]
                model.Add(sum(variables) >= min_side_bins).OnlyEnforceIf(label_vars[(path_id, BEND)])

    # Exclusive physical boundary ownership.
    boundary_owner_vars: dict[tuple[str, str], cp_model.IntVar] = {}
    boundary_residual_vars: dict[str, cp_model.IntVar] = {}
    for arc in sorted(graph.get("panel_boundary_arcs") or (), key=lambda item: str(item["arc_id"])):
        arc_id = str(arc["arc_id"])
        compatible = sorted({str(value) for value in arc.get("compatible_object_ids") or arc.get("compatible_path_ids") or ()})
        if not compatible or not set(compatible) <= path_id_set:
            raise ValueError(f"boundary arc {arc_id} has invalid compatible objects")
        row: list[cp_model.IntVar] = []
        if arc.get("residual_owner_allowed") is True:
            residual = model.NewBoolVar(f"arc__{arc_id}__RESIDUAL")
            boundary_residual_vars[arc_id] = residual
            row.append(residual)
        for path_id in compatible:
            owner = model.NewBoolVar(f"arc__{arc_id}__{path_id}")
            boundary_owner_vars[(arc_id, path_id)] = owner
            model.Add(owner <= active_vars[path_id])
            row.append(owner)
        model.AddExactlyOne(row)

    continuation_records = _continuation_records(graph, path_by_id=path_by_id)
    for record in continuation_records:
        if set(record["bridge_atom_ids"]) - set(atom_by_id):
            raise ValueError(
                "continuation bridge certificate references absent support atoms"
            )
    continuation_vars: dict[str, cp_model.IntVar] = {}
    endpoint_incidence: dict[tuple[str, int], list[cp_model.IntVar]] = defaultdict(list)
    for record in continuation_records:
        continuation_id = str(record["continuation_id"])
        left, right = record["path_ids"]
        left_endpoint, right_endpoint = record["endpoint_ids"]
        variable = model.NewBoolVar(f"y__{continuation_id}")
        continuation_vars[continuation_id] = variable
        model.Add(variable <= label_vars[(left, BEND)])
        model.Add(variable <= label_vars[(right, BEND)])
        model.Add(variable <= endpoint_vars[(left, left_endpoint, CONTINUATION_END)])
        model.Add(variable <= endpoint_vars[(right, right_endpoint, CONTINUATION_END)])
        endpoint_incidence[(left, left_endpoint)].append(variable)
        endpoint_incidence[(right, right_endpoint)].append(variable)
    for key, variables in endpoint_incidence.items():
        model.AddAtMostOne(variables)
        model.Add(endpoint_vars[(key[0], key[1], CONTINUATION_END)] <= sum(variables))

    birth_vars: dict[str, cp_model.IntVar] = {}
    birth_ids_by_path: dict[str, str] = {}
    for path in paths:
        path_id = str(path["latent_path_id"])
        is_birth = bool(
            path.get("is_birth")
            or path.get("birth_id") is not None
            or path.get("is_birth_column")
            or path.get("birth_column")
            or path.get("origin") == "RESIDUAL_BIRTH"
        )
        if not is_birth:
            continue
        birth_id = str(path.get("birth_id") or path_id)
        if birth_id in birth_ids_by_path.values():
            raise ValueError("birth IDs must be unique")
        if int(path.get("birth_cost_int") or 0) < 0:
            raise ValueError("birth_cost_int must be non-negative")
        birth_ids_by_path[path_id] = birth_id
        variable = model.NewBoolVar(f"birth__{path_id}")
        birth_vars[path_id] = variable
        model.Add(active_vars[path_id] <= variable)
        model.Add(variable <= active_vars[path_id])
        for (atom_id, owner_path), owner in ownership_vars.items():
            if owner_path == path_id:
                model.Add(owner <= variable)
        for record in continuation_records:
            if path_id in record["path_ids"]:
                model.Add(continuation_vars[str(record["continuation_id"])] <= variable)

    # Engineering-fixed MDL primary energy plus optional frozen unaries and
    # structured factor energies. This makes residual-birth lower bounds real.
    energy_parameters = graph.get("energy_parameters") or {}
    default_object_cost = ENERGY_SCALE * (1 + math.ceil(math.log2(len(paths) + 1)))
    object_cost = int(energy_parameters.get("object_cost_int", default_object_cost))
    residual_cost = int(energy_parameters.get("residual_cost_per_mass_int", C_RESIDUAL))
    if object_cost < 0 or residual_cost < 0:
        raise ValueError("fixed MDL object and residual costs must be non-negative")
    primary_terms: list[Any] = []
    for path in paths:
        path_id = str(path["latent_path_id"])
        for label in LABELS:
            cost = _unary_cost(path, label)
            if label != RESIDUAL:
                cost += object_cost
            if cost:
                primary_terms.append(cost * label_vars[(path_id, label)])
    primary_terms.extend(
        residual_cost * atom_masses[atom_id] * variable
        for atom_id, variable in residual_owner_vars.items()
    )
    primary_terms.extend(factor_energy[factor_id] * variable for factor_id, variable in factor_vars.items())
    primary_terms.extend(
        int(path_by_id[path_id].get("birth_cost_int") or 0) * variable
        for path_id, variable in birth_vars.items()
    )
    primary_energy = sum(primary_terms)
    model.Minimize(primary_energy)

    # Clone before tie constraints: second-best is measured under exactly the
    # same primary model, never under an ownership or canonical tie objective.
    second_best_model = model.Clone()
    primary_solver = _solve_optimal(model, stage="primary-energy")
    primary_objective = int(round(primary_solver.ObjectiveValue()))
    model.Add(primary_energy == primary_objective)

    weighted_owned_mass = sum(
        atom_masses[atom_id] * variable
        for (atom_id, _path_id), variable in ownership_vars.items()
    )
    model.Maximize(weighted_owned_mass)
    mass_solver = _solve_optimal(model, stage="owned-support-mass")
    maximum_owned_mass = int(round(mass_solver.ObjectiveValue()))
    model.Add(weighted_owned_mass == maximum_owned_mass)

    selected_continuation_count = sum(continuation_vars.values())
    model.Maximize(selected_continuation_count)
    continuation_solver = _solve_optimal(model, stage="continuations")
    maximum_continuations = int(round(continuation_solver.ObjectiveValue()))
    model.Add(selected_continuation_count == maximum_continuations)

    selected_birth_count = sum(birth_vars.values())
    model.Minimize(selected_birth_count)
    birth_solver = _solve_optimal(model, stage="birth-activations")
    minimum_birth_activations = int(round(birth_solver.ObjectiveValue()))
    model.Add(selected_birth_count == minimum_birth_activations)

    _fix_label_lexicographically(model, path_ids=path_ids, label_vars=label_vars)
    _fix_one_hot_lexicographically(
        model,
        [
            [endpoint_vars[(path_id, endpoint, node_type)] for node_type in NODE_TYPES]
            for path_id in path_ids
            for endpoint in (0, 1)
        ],
        stage="canonical-endpoint-type",
    )
    _fix_bool_lexicographically(model, [factor_vars[key] for key in sorted(factor_vars)], prefer=0, stage="canonical-factor")
    _fix_bool_lexicographically(model, [continuation_vars[key] for key in sorted(continuation_vars)], prefer=0, stage="canonical-continuation")
    _fix_bool_lexicographically(model, [birth_vars[key] for key in sorted(birth_vars)], prefer=0, stage="canonical-birth")
    owner_order = [ownership_vars[key] for key in sorted(ownership_vars)] + [residual_owner_vars[key] for key in sorted(residual_owner_vars)]
    _fix_bool_lexicographically(model, owner_order, prefer=0, stage="canonical-owner")
    boundary_order = [boundary_owner_vars[key] for key in sorted(boundary_owner_vars)] + [
        boundary_residual_vars[key] for key in sorted(boundary_residual_vars)
    ]
    _fix_bool_lexicographically(model, boundary_order, prefer=1, stage="canonical-boundary-owner")
    model.Minimize(0)
    solver = _solve_optimal(model, stage="deterministic-map")

    assignments = []
    for path_id in path_ids:
        selected_label = next(label for label in LABELS if solver.Value(label_vars[(path_id, label)]))
        assignments.append({
            "latent_path_id": path_id,
            "semantic_label": selected_label,
            "counted": selected_label == BEND,
            "endpoint_types": [
                next(node_type for node_type in NODE_TYPES if solver.Value(endpoint_vars[(path_id, endpoint, node_type)]))
                for endpoint in (0, 1)
            ],
        })

    selected_factors = [factor_id for factor_id in sorted(factor_vars) if solver.Value(factor_vars[factor_id])]
    selected_continuations = [
        record for record in continuation_records
        if solver.Value(continuation_vars[str(record["continuation_id"])])
    ]
    atom_owners = []
    owned_mass_by_path = {path_id: 0 for path_id in path_ids}
    owned_bins_by_path: dict[str, set[str]] = {path_id: set() for path_id in path_ids}
    residual_owned_mass = 0
    for atom_id in sorted(atom_by_id):
        owner_path = next(
            (path_id for path_id in sorted(atom_to_paths.get(atom_id) or ()) if solver.Value(ownership_vars[(atom_id, path_id)])),
            None,
        )
        mass = atom_masses[atom_id]
        if owner_path is None:
            residual_owned_mass += mass
            atom_owners.append({
                "atom_id": atom_id,
                "owner_kind": "RESIDUAL",
                "latent_path_id": None,
                "normalized_support_mass": mass,
            })
        else:
            owned_mass_by_path[owner_path] += mass
            bin_id = atom_bins_by_path[owner_path].get(atom_id)
            if bin_id is not None:
                owned_bins_by_path[owner_path].add(bin_id)
            atom_owners.append({
                "atom_id": atom_id,
                "owner_kind": "PATH",
                "latent_path_id": owner_path,
                "normalized_support_mass": mass,
            })

    boundary_owners = []
    boundary_arc_ids = set(boundary_residual_vars) | {
        arc_id for arc_id, _path_id in boundary_owner_vars
    }
    for arc_id in sorted(boundary_arc_ids):
        owner_path = next(
            (path_id for candidate_arc, path_id in sorted(boundary_owner_vars) if candidate_arc == arc_id and solver.Value(boundary_owner_vars[(candidate_arc, path_id)])),
            None,
        )
        boundary_owners.append({
            "arc_id": arc_id,
            "owner": owner_path or RESIDUAL,
            "owner_kind": "PATH" if owner_path else "RESIDUAL",
            "latent_path_id": owner_path,
        })

    # Exact second-best complete semantic assignment: labels, factor choices,
    # continuations and births. Ownership-only alternatives are deliberately not
    # reported as semantic confidence.
    semantic_original = (
        [label_vars[(path_id, label)] for path_id in path_ids for label in LABELS]
        + [factor_vars[key] for key in sorted(factor_vars)]
        + [continuation_vars[key] for key in sorted(continuation_vars)]
        + [birth_vars[key] for key in sorted(birth_vars)]
    )
    no_good_terms = []
    for variable in semantic_original:
        clone_var = _bool_in_clone(second_best_model, variable)
        no_good_terms.append(1 - clone_var if solver.Value(variable) else clone_var)
    alternative_energy: int | None = None
    second_status_name = "INFEASIBLE"
    second_assignments: list[dict[str, Any]] = []
    second_selected_factor_ids: list[str] = []
    second_selected_continuation_ids: list[str] = []
    second_selected_birth_ids: list[str] = []
    if no_good_terms:
        second_best_model.Add(sum(no_good_terms) >= 1)
        second_solver = cp_model.CpSolver()
        _configure(second_solver)
        second_status = second_solver.Solve(second_best_model)
        if second_status not in (cp_model.OPTIMAL, cp_model.INFEASIBLE):
            raise RuntimeError(
                "structured solver second-best-primary requires a proof, got "
                f"{_status_name(second_solver, second_status)}"
            )
        second_status_name = _status_name(second_solver, second_status)
        if second_status == cp_model.INFEASIBLE:
            second_solver = None
        else:
            alternative_energy = int(round(second_solver.ObjectiveValue()))

    if alternative_energy is not None:
        second_label_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in label_vars.items()
        }
        second_endpoint_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in endpoint_vars.items()
        }
        second_factor_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in factor_vars.items()
        }
        second_continuation_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in continuation_vars.items()
        }
        second_birth_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in birth_vars.items()
        }
        second_ownership_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in ownership_vars.items()
        }
        second_residual_owner_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in residual_owner_vars.items()
        }
        second_boundary_owner_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in boundary_owner_vars.items()
        }
        second_boundary_residual_vars = {
            key: _bool_in_clone(second_best_model, variable)
            for key, variable in boundary_residual_vars.items()
        }

        second_primary_terms: list[Any] = []
        for path in paths:
            path_id = str(path["latent_path_id"])
            for label in LABELS:
                cost = _unary_cost(path, label)
                if label != RESIDUAL:
                    cost += object_cost
                if cost:
                    second_primary_terms.append(cost * second_label_vars[(path_id, label)])
        second_primary_terms.extend(
            residual_cost * atom_masses[atom_id] * variable
            for atom_id, variable in second_residual_owner_vars.items()
        )
        second_primary_terms.extend(
            factor_energy[factor_id] * variable
            for factor_id, variable in second_factor_vars.items()
        )
        second_primary_terms.extend(
            int(path_by_id[path_id].get("birth_cost_int") or 0) * variable
            for path_id, variable in second_birth_vars.items()
        )
        second_primary_energy = sum(second_primary_terms)
        second_best_model.Add(second_primary_energy == alternative_energy)

        second_owned_mass = sum(
            atom_masses[atom_id] * variable
            for (atom_id, _path_id), variable in second_ownership_vars.items()
        )
        second_best_model.Maximize(second_owned_mass)
        second_mass_solver = _solve_optimal(second_best_model, stage="second-best-owned-support-mass")
        second_best_model.Add(second_owned_mass == int(round(second_mass_solver.ObjectiveValue())))
        second_continuation_count = sum(second_continuation_vars.values())
        second_best_model.Maximize(second_continuation_count)
        second_cont_solver = _solve_optimal(second_best_model, stage="second-best-continuations")
        second_best_model.Add(second_continuation_count == int(round(second_cont_solver.ObjectiveValue())))
        second_birth_count = sum(second_birth_vars.values())
        second_best_model.Minimize(second_birth_count)
        second_birth_solver = _solve_optimal(
            second_best_model, stage="second-best-birth-activations"
        )
        second_minimum_births = int(round(second_birth_solver.ObjectiveValue()))
        second_best_model.Add(second_birth_count == second_minimum_births)
        _fix_label_lexicographically(
            second_best_model,
            path_ids=path_ids,
            label_vars=second_label_vars,
        )
        _fix_one_hot_lexicographically(
            second_best_model,
            [
                [
                    second_endpoint_vars[(path_id, endpoint, node_type)]
                    for node_type in NODE_TYPES
                ]
                for path_id in path_ids
                for endpoint in (0, 1)
            ],
            stage="second-best-endpoint-type",
        )
        _fix_bool_lexicographically(
            second_best_model,
            [second_factor_vars[key] for key in sorted(second_factor_vars)],
            prefer=0,
            stage="second-best-factor",
        )
        _fix_bool_lexicographically(
            second_best_model,
            [second_continuation_vars[key] for key in sorted(second_continuation_vars)],
            prefer=0,
            stage="second-best-continuation",
        )
        _fix_bool_lexicographically(
            second_best_model,
            [second_birth_vars[key] for key in sorted(second_birth_vars)],
            prefer=0,
            stage="second-best-birth",
        )
        _fix_bool_lexicographically(
            second_best_model,
            [second_ownership_vars[key] for key in sorted(second_ownership_vars)]
            + [second_residual_owner_vars[key] for key in sorted(second_residual_owner_vars)],
            prefer=0,
            stage="second-best-owner",
        )
        _fix_bool_lexicographically(
            second_best_model,
            [second_boundary_owner_vars[key] for key in sorted(second_boundary_owner_vars)]
            + [second_boundary_residual_vars[key] for key in sorted(second_boundary_residual_vars)],
            prefer=1,
            stage="second-best-boundary-owner",
        )
        second_best_model.Minimize(0)
        second_solver = _solve_optimal(second_best_model, stage="second-best-deterministic-map")
        second_assignments = [
            {
                "latent_path_id": path_id,
                "semantic_label": next(
                    label for label in LABELS
                    if second_solver.Value(second_label_vars[(path_id, label)])
                ),
                "endpoint_types": [
                    next(
                        node_type
                        for node_type in NODE_TYPES
                        if second_solver.Value(
                            second_endpoint_vars[(path_id, endpoint, node_type)]
                        )
                    )
                    for endpoint in (0, 1)
                ],
            }
            for path_id in path_ids
        ]
        second_selected_factor_ids = [
            factor_id
            for factor_id in sorted(second_factor_vars)
            if second_solver.Value(second_factor_vars[factor_id])
        ]
        second_selected_continuation_ids = [
            continuation_id
            for continuation_id in sorted(second_continuation_vars)
            if second_solver.Value(second_continuation_vars[continuation_id])
        ]
        second_selected_birth_ids = [
            birth_ids_by_path[path_id]
            for path_id in sorted(second_birth_vars)
            if second_solver.Value(second_birth_vars[path_id])
        ]

    counted_fragment_count = sum(int(item["counted"]) for item in assignments)
    counted_component_count = counted_fragment_count - len(selected_continuations)
    counted_ids = {
        item["latent_path_id"] for item in assignments if item["counted"]
    }
    component_parent = {path_id: path_id for path_id in counted_ids}

    def component_find(path_id: str) -> str:
        while component_parent[path_id] != path_id:
            component_parent[path_id] = component_parent[component_parent[path_id]]
            path_id = component_parent[path_id]
        return path_id

    for continuation in selected_continuations:
        left, right = continuation["path_ids"]
        left_root, right_root = component_find(left), component_find(right)
        component_parent[right_root] = left_root
    component_groups: dict[str, list[str]] = defaultdict(list)
    for path_id in sorted(counted_ids):
        component_groups[component_find(path_id)].append(path_id)
    counted_components = sorted(
        (sorted(group) for group in component_groups.values()),
        key=lambda group: tuple(group),
    )
    unsupported_selected = [
        item["latent_path_id"] for item in assignments
        if item["semantic_label"] != RESIDUAL and not bool(path_by_id[item["latent_path_id"]].get("incidence_complete"))
    ]
    if unsupported_selected:
        raise RuntimeError("internal error: unsupported non-residual selection escaped hard constraint")

    return {
        "schema_version": 2,
        "kind": "sherman_qc_f1_structured_union_cpsat_solution",
        "research_only": True,
        "solver": {
            "engine": "OR-Tools CP-SAT",
            "status": "OPTIMAL",
            "num_search_workers": 1,
            "random_seed": 0,
            "search_branching": "FIXED_SEARCH",
            "objective_scale": ENERGY_SCALE,
            "object_cost_int": object_cost,
            "residual_cost_per_mass_int": residual_cost,
            "primary_objective_int": primary_objective,
            "second_best_status": second_status_name,
            "second_best_objective_int": alternative_energy,
            "energy_margin_int": alternative_energy - primary_objective if alternative_energy is not None else None,
            "second_best_energy_margin_int": alternative_energy - primary_objective if alternative_energy is not None else None,
            "energy_margin_is_calibrated_confidence": False,
            "exact_primary_optimality_proven": True,
            "exact_restricted_model_optimality_proven": True,
            "tie_breaking": [
                "minimum_primary_energy",
                "maximum_nonresidual_owned_support_mass",
                "maximum_valid_continuations",
                "minimum_birth_activations",
                "proved_lexicographic_assignment",
            ],
            "maximum_normalized_nonresidual_owned_mass": maximum_owned_mass,
            "maximum_compatible_continuations": maximum_continuations,
            "minimum_birth_activations": minimum_birth_activations,
        },
        "model_scope": {
            "typed_one_hot_labels": True,
            "active_seven_label_eligibility": True,
            "all_seven_semantic_labels_operational": True,
            "endpoint_type_variables": True,
            "relational_factors": True,
            "disabled_factors_retained_as_audit_only": True,
            "pairwise_factor_conflicts": True,
            "global_topology_factor_at_most_one": False,
            "exclusive_sparse_support_ownership": True,
            "exclusive_sparse_support_ownership_scope": "complete_serialized_compact_atom_incidence",
            "weighted_mass_and_axial_bin_minima": True,
            "bend_side_bin_minima": True,
            "hard_duplicate_constraints": True,
            "exclusive_panel_boundary_ownership": True,
            "panel_boundary_arc_ownership": bool(boundary_arc_ids),
            "continuation_certificates": True,
            "endpoint_aware_acyclic_continuations": True,
            "mdl_object_and_residual_cost": True,
            "residual_birth_columns": bool(birth_vars),
            "learned_potentials": any(any(_unary_cost(path, label) for label in LABELS) for path in paths),
            "complete_structured_model_implemented": False,
            "runtime_promoted": False,
        },
        "primary_objective_terms": [
            "fixed_mdl_nonresidual_object_cost",
            "fixed_normalized_residual_atom_mass_cost",
            "frozen_typed_unary_costs",
            "relational_factor_activation",
        ],
        "graph_sha256": str(graph.get("graph_sha256") or ""),
        "latent_path_count": len(paths),
        "serialized_support_atom_count": len(atom_by_id),
        "support_atom_incidence_count": len(ownership_vars),
        "residual_owned_support_count": sum(1 for item in atom_owners if item["owner_kind"] == "RESIDUAL"),
        "residual_owned_support_mass": residual_owned_mass,
        "owned_support_count_by_path": {
            path_id: sum(1 for item in atom_owners if item["latent_path_id"] == path_id)
            for path_id in path_ids
        },
        "owned_support_mass_by_path": owned_mass_by_path,
        "owned_axial_bin_count_by_path": {path_id: len(bins) for path_id, bins in owned_bins_by_path.items()},
        "atom_owners": atom_owners,
        "boundary_owners": boundary_owners,
        "panel_boundary_arc_owners": [
            {"arc_id": item["arc_id"], "owner": item["owner"]}
            for item in boundary_owners
        ],
        "unsupported_selected_path_count": 0,
        "unsupported_selected_path_ids": [],
        "selected_factor_ids": selected_factors,
        "disabled_audit_factor_ids": [
            str(factor["factor_id"]) for factor in disabled_audit_factors
        ],
        "input_topology_factor_count": len(all_factors),
        "modeled_topology_factor_count": len(factors),
        "factor_conflict_pairs": [list(pair) for pair in conflict_pairs],
        "selected_continuations": selected_continuations,
        "selected_birth_path_ids": [path_id for path_id in sorted(birth_vars) if solver.Value(birth_vars[path_id])],
        "selected_birth_ids": [
            str(path_by_id[path_id].get("birth_id") or path_id)
            for path_id in sorted(birth_vars)
            if solver.Value(birth_vars[path_id])
        ],
        "counted_fragment_count": counted_fragment_count,
        "counted_component_count": counted_component_count,
        "counted_components": counted_components,
        "endpoint_types_by_path": {
            item["latent_path_id"]: [
                "CONTINUATION" if node_type == CONTINUATION_END else node_type
                for node_type in item["endpoint_types"]
            ]
            for item in assignments
        },
        "second_best_assignment": {
            "status": second_status_name,
            "objective_int": alternative_energy,
            "assignments": second_assignments,
            "selected_factor_ids": second_selected_factor_ids,
            "selected_continuation_ids": second_selected_continuation_ids,
            "selected_birth_ids": second_selected_birth_ids,
        },
        "assignments": assignments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    result = solve(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("kind", "solver", "model_scope", "counted_fragment_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
