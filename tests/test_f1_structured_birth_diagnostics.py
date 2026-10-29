from __future__ import annotations

from copy import deepcopy

from backend.f1_structured_birth_diagnostics import (
    FUNNEL_STAGES,
    diagnose_residual_birth_eligibility,
)
from backend.f1_structured_residual_birth import generate_residual_birth_columns


def _eligible_fixture() -> tuple[dict, dict]:
    atoms = []
    edges = []
    for bin_index in range(6):
        for side, normal in (("L", [1.0, 0.0, 0.0]), ("R", [0.0, 0.0, 1.0])):
            atom_id = f"{side}{bin_index}"
            atoms.append(
                {
                    "atom_id": atom_id,
                    "centroid": [0.0, 3.0 * bin_index, 0.0],
                    "normal": normal,
                    "axis_direction": [0.0, 1.0, 0.0],
                    "local_spacing_mm": 1.5,
                    "area_proxy_mm2": 1.0,
                    "normalized_support_mass": 1,
                }
            )
            if bin_index:
                edges.append({"atom_ids": [f"{side}{bin_index - 1}", atom_id], "weight": 1.0})
        edges.append({"atom_ids": [f"L{bin_index}", f"R{bin_index}"], "weight": 1.0})
    graph = {
        "support_atom_table": atoms,
        "support_atom_edges": edges,
        "plane_table": [
            {"plane_id": "PX", "normal": [1.0, 0.0, 0.0], "offset": 0.0},
            {"plane_id": "PZ", "normal": [0.0, 0.0, 1.0], "offset": 0.0},
        ],
        "latent_paths": [],
    }
    solution = {
        "atom_owners": [
            {"atom_id": atom["atom_id"], "owner_kind": "RESIDUAL", "latent_path_id": None}
            for atom in atoms
        ],
        "solver": {
            "object_cost_int": 11_000_000,
            "residual_cost_per_mass_int": 1_000_000,
        },
    }
    return graph, solution


def test_stage_census_matches_birth_v1_and_is_deterministic() -> None:
    graph, solution = _eligible_fixture()
    first = diagnose_residual_birth_eligibility(graph, solution)
    second = diagnose_residual_birth_eligibility(deepcopy(graph), deepcopy(solution))
    proposals = generate_residual_birth_columns(graph, solution)

    assert first == second
    assert first["proposal_ids"] == [proposal["birth_id"] for proposal in proposals]
    assert first["funnel"]["components_after_mdl_lower_bound"] == 1
    assert first["funnel"]["emitted_columns_after_cap"] == 1
    assert first["first_empty_transition"] is None
    assert tuple(first["funnel"]) == FUNNEL_STAGES
    assert first["truth_inputs_consumed"] is False
    assert first["source_points_consumed"] is False


def test_stage_census_attributes_transverse_normal_axis_and_mdl_failures() -> None:
    graph, solution = _eligible_fixture()
    graph["support_atom_table"][0]["centroid"] = [10.0, 0.0, 0.0]
    graph["support_atom_table"][1]["normal"] = [0.0, 1.0, 0.0]
    graph["support_atom_table"][2]["axis_direction"] = [1.0, 0.0, 0.0]
    diagnostic = diagnose_residual_birth_eligibility(graph, solution)

    assert diagnostic["rejections"]["transverse_distance"] == 1
    assert diagnostic["rejections"]["normal"] == 1
    assert diagnostic["rejections"]["atom_axis"] == 1
    assert diagnostic["funnel"]["atoms_after_atom_axis"] == 9

    graph, solution = _eligible_fixture()
    solution["solver"]["object_cost_int"] = 12_000_000
    diagnostic = diagnose_residual_birth_eligibility(graph, solution)
    assert diagnostic["rejections"]["mdl_nonnegative_lower_bound"] == 1
    assert diagnostic["funnel"]["components_after_duplicate_suppression"] == 1
    assert diagnostic["funnel"]["components_after_mdl_lower_bound"] == 0
    assert diagnostic["first_empty_transition"] == "components_after_mdl_lower_bound"


def test_stage_census_reports_duplicate_suppression() -> None:
    graph, solution = _eligible_fixture()
    graph["latent_paths"] = [
        {
            "latent_path_id": "P_EXISTING",
            "span_endpoints": [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]],
            "axis_direction": [0.0, 1.0, 0.0],
        }
    ]
    diagnostic = diagnose_residual_birth_eligibility(graph, solution)
    assert diagnostic["rejections"]["duplicate_existing"] == 1
    assert diagnostic["funnel"]["components_after_bin_branching"] == 1
    assert diagnostic["funnel"]["components_after_duplicate_suppression"] == 0
