from __future__ import annotations

from copy import deepcopy

import pytest

from backend.f1_structured_residual_birth import (
    BIRTH_PROFILE,
    TOTAL_BIRTH_COLUMN_CAP,
    add_birth_columns,
    generate_residual_birth_columns,
)


def _graph() -> dict:
    atoms = []
    edges = []
    for index in range(8):
        x = 3.0 * index
        for side, y, normal in (
            ("L", 0.5, [0.0, 1.0, 0.0]),
            ("R", -0.5, [0.0, 0.0, 1.0]),
        ):
            atom_id = f"A{index}_{side}"
            atoms.append(
                {
                    "atom_id": atom_id,
                    "centroid": [x, y if side == "L" else 0.0, 0.0 if side == "L" else y],
                    "normal": normal,
                    "axis_direction": [1.0, 0.0, 0.0],
                    "local_spacing_mm": 1.0,
                    "area_proxy_mm2": 4.0,
                }
            )
            if index:
                edges.append({"atom_ids": [f"A{index - 1}_{side}", atom_id], "weight": 1.0})
        edges.append({"atom_ids": [f"A{index}_L", f"A{index}_R"], "weight": 1.0})
    return {
        "support_atom_table": atoms,
        "support_atom_edges": edges,
        "plane_table": [
            {"plane_id": "P1", "normal": [0.0, 1.0, 0.0], "offset": 0.0},
            {"plane_id": "P2", "normal": [0.0, 0.0, 1.0], "offset": 0.0},
        ],
        "latent_paths": [],
    }


def _solution(graph: dict) -> dict:
    return {
        "solver": {
            "object_cost_int": 2_000_000,
            "residual_cost_per_mass_int": 1_000_000,
        },
        "atom_owners": {
            atom["atom_id"]: "RESIDUAL"
            for atom in graph["support_atom_table"]
        }
    }


def test_birth_v1_adds_only_negative_bound_two_sided_residual_column():
    graph = _graph()
    proposals = generate_residual_birth_columns(graph, _solution(graph))

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["birth_profile"] == BIRTH_PROFILE
    assert proposal["lower_bound_int"] < 0
    assert proposal["incidence_complete"] is True
    assert proposal["label_eligibility"]["BEND_STRAIGHT_COUNTED"] is True
    assert proposal["primary_panel_pair"] == ["PLANE:P1", "PLANE:P2"]

    augmented = add_birth_columns(graph, proposals)
    assert augmented["birth_columns"] == [proposal["birth_id"]]
    assert augmented["latent_paths"][0]["latent_path_id"] == proposal["latent_path_id"]
    assert augmented["energy_parameters"] == {
        "object_cost_int": proposal["object_cost_int"],
        "residual_cost_per_mass_int": proposal["residual_cost_per_mass_int"],
    }


def test_birth_v1_is_permutation_invariant():
    graph = _graph()
    baseline = generate_residual_birth_columns(graph, _solution(graph))
    permuted = deepcopy(graph)
    permuted["support_atom_table"].reverse()
    permuted["support_atom_edges"].reverse()
    permuted["plane_table"].reverse()

    assert generate_residual_birth_columns(permuted, _solution(permuted)) == baseline


def test_birth_v1_rejects_surviving_duplicate_candidate():
    graph = _graph()
    graph["latent_paths"] = [
        {
            "latent_path_id": "EXISTING",
            "span_endpoints": [[0.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
            "axis_direction": [1.0, 0.0, 0.0],
        }
    ]

    assert generate_residual_birth_columns(graph, _solution(graph)) == []


def test_birth_v1_fails_closed_without_per_atom_residual_ownership():
    graph = _graph()
    try:
        generate_residual_birth_columns(graph, {"residual_owned_support_count": 16})
    except ValueError as error:
        assert "per-atom owners" in str(error)
    else:
        raise AssertionError("birth generation must fail without per-atom ownership")


def test_birth_v1_fails_closed_without_exact_solver_mdl_costs():
    graph = _graph()
    solution = _solution(graph)
    solution.pop("solver")
    with pytest.raises(ValueError, match="frozen MDL cost profile"):
        generate_residual_birth_columns(graph, solution)


def test_birth_v1_accepts_exact_solver_owner_records_and_serialized_masses():
    graph = _graph()
    for atom in graph["support_atom_table"]:
        atom["normalized_support_mass"] = 1
    solution = {
        "solver": {
            "object_cost_int": 20_000_000,
            "residual_cost_per_mass_int": 2_000_000,
        },
        "atom_owners": [
            {
                "atom_id": atom["atom_id"],
                "owner_kind": "RESIDUAL",
                "latent_path_id": None,
                "normalized_support_mass": 1,
            }
            for atom in graph["support_atom_table"]
        ],
    }

    proposals = generate_residual_birth_columns(graph, solution)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["normalized_support_mass"] == 16
    assert proposal["object_cost_int"] == 20_000_000
    assert proposal["residual_cost_per_mass_int"] == 2_000_000
    assert proposal["saved_residual_cost_upper_bound_int"] == 32_000_000
    assert proposal["lower_bound_int"] == -12_000_000
    assert proposal["lower_bound_cost_source"] == "current_exact_solver_fixed_mdl"


def test_birth_columns_preserve_cumulative_ids_and_frozen_costs():
    graph = _graph()
    first = generate_residual_birth_columns(graph, _solution(graph))[0]
    once = add_birth_columns(graph, [first])
    second = deepcopy(first)
    second["birth_id"] = "BIRTH_SECOND"
    second["latent_path_id"] = "LP_BIRTH_SECOND"

    twice = add_birth_columns(once, [second])

    assert twice["birth_columns"] == sorted([first["birth_id"], "BIRTH_SECOND"])
    assert len(twice["latent_paths"]) == 2
    assert twice["energy_parameters"] == once["energy_parameters"]


def test_birth_column_cap_is_hard():
    graph = _graph()
    try:
        generate_residual_birth_columns(
            graph,
            _solution(graph),
            total_column_cap=TOTAL_BIRTH_COLUMN_CAP + 1,
        )
    except ValueError as error:
        assert "cap" in str(error)
    else:
        raise AssertionError("birth cap must be enforced")
