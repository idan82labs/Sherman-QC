from __future__ import annotations

from copy import deepcopy
import json

import pytest

from backend.f1_structured_birth_diagnostics import (
    TRANSITION_DIAGNOSTIC_KIND,
    diagnose_transition_mode_probe,
)
from scripts.diagnose_f1_transition_mode_probe import (
    A0_PARENT_PROJECTION_PROFILE,
    build_transition_probe,
    project_a0_v1_owners_to_transition_parents,
)


HASHES = {
    "scan_sha256": "1" * 64,
    "candidate_pool_sha256": "2" * 64,
    "graph_sha256": "3" * 64,
    "a0_solution_sha256": "4" * 64,
}


def _fixture(bin_count: int = 6) -> tuple[dict, dict, dict]:
    parents = []
    modes = []
    edges = []
    for bin_index in range(bin_count):
        parent_id = f"P{bin_index}"
        left_id = f"ML{bin_index}"
        right_id = f"MR{bin_index}"
        parents.append(
            {
                "parent_atom_id": parent_id,
                "voxel_key": [0, bin_index, 0],
                "centroid": [0.0, 3.0 * bin_index, 0.0],
                "local_spacing_mm": 1.5,
                "voxel_size_mm": 3.0,
                "normalized_support_mass": 1,
                "mode_overflow": False,
                "transition_certificate_allowed": True,
            }
        )
        modes.extend(
            [
                {
                    "mode_atom_id": left_id,
                    "parent_atom_id": parent_id,
                    "centroid": [0.0, 3.0 * bin_index, 0.0],
                    "normal_projective": [1.0, 0.0, 0.0],
                    "point_count": 8,
                    "occupied_fine_cell_count": 2,
                    "support_fraction": 0.5,
                },
                {
                    "mode_atom_id": right_id,
                    "parent_atom_id": parent_id,
                    "centroid": [0.0, 3.0 * bin_index, 0.0],
                    "normal_projective": [0.0, 0.0, 1.0],
                    "point_count": 8,
                    "occupied_fine_cell_count": 2,
                    "support_fraction": 0.5,
                },
            ]
        )
        edges.append(
            {
                "mode_edge_id": f"ER{bin_index}",
                "mode_ids": [left_id, right_id],
                "edge_kind": "CROSS_MODE_LOCAL",
            }
        )
        if bin_index:
            edges.extend(
                [
                    {
                        "mode_edge_id": f"EL{bin_index - 1}_{bin_index}",
                        "mode_ids": [f"ML{bin_index - 1}", left_id],
                        "edge_kind": "SAME_SURFACE",
                    },
                    {
                        "mode_edge_id": f"EZ{bin_index - 1}_{bin_index}",
                        "mode_ids": [f"MR{bin_index - 1}", right_id],
                        "edge_kind": "SAME_SURFACE",
                    },
                ]
            )
    transition = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "support_parent_table": parents,
        "support_mode_table": modes,
        "support_mode_edges": edges,
        "transition_atom_caps": {"mode_children_per_parent_cap": 4},
        "support_atom_table": [
            {
                "atom_id": f"A{bin_index}",
                "centroid": [0.0, 3.0 * bin_index, 0.0],
            }
            for bin_index in range(bin_count)
        ],
    }
    graph = {
        "plane_table": [
            {"plane_id": "PX", "normal": [1.0, 0.0, 0.0], "offset": 0.0},
            {"plane_id": "PZ", "normal": [0.0, 0.0, 1.0], "offset": 0.0},
        ],
        "latent_paths": [],
    }
    solution = {
        "atom_owners": [
            {"atom_id": f"A{bin_index}", "owner_kind": "RESIDUAL"}
            for bin_index in range(bin_count)
        ],
        "transition_parent_owners": [
            {"parent_atom_id": parent["parent_atom_id"], "owner_kind": "RESIDUAL"}
            for parent in parents
        ],
        "solver": {
            "object_cost_int": 11_000_000,
            "residual_cost_per_mass_int": 4_000_000,
        },
    }
    return transition, graph, solution


def test_transition_probe_is_deterministic_bounded_and_prediction_inert() -> None:
    transition, graph, solution = _fixture()
    frozen_transition = deepcopy(transition)
    frozen_graph = deepcopy(graph)
    frozen_solution = deepcopy(solution)

    first = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)
    second = diagnose_transition_mode_probe(
        deepcopy(transition), deepcopy(graph), deepcopy(solution), **HASHES
    )

    assert first == second
    assert first["kind"] == TRANSITION_DIAGNOSTIC_KIND
    assert first["inference_safe"] is True
    assert first["birth_columns_enabled"] is False
    assert first["prediction_mutated"] is False
    assert first["case_counters"]["birth_eligible_diagnostic_count"] == 1
    assert first["case_counters"]["emitted_birth_column_count"] == 0
    assert first["birth_eligibility_records"][0]["eligibility_code"] == "ELIGIBLE_DIAGNOSTIC_ONLY"
    assert first["birth_eligibility_records"][0]["column_emitted"] is False
    birth = first["birth_eligibility_records"][0]
    assert birth["valid_two_mode_connected_run"] is True
    assert birth["geometry_certificate_valid"] is True
    assert birth["first_failed_stage"] is None
    assert birth["intersection_span_endpoints"] == [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]]
    assert birth["matching_plane_pair_ids"] == [birth["plane_pair_id"]]
    assert birth["matching_component_ids"] == [birth["component_id"]]
    assert birth["support_parent_ids"] == [f"P{index}" for index in range(6)]
    assert birth["bounded_counters"]["full_window_count"] == 4
    assert transition == frozen_transition
    assert graph == frozen_graph
    assert solution == frozen_solution
    encoded = json.dumps(first)
    for forbidden in ('"point_indices"', '"source_points"', '"bbox_points"', '"points"'):
        assert forbidden not in encoded


def test_transition_probe_fails_closed_when_a0_parent_ownership_is_unavailable() -> None:
    transition, graph, solution = _fixture()
    solution.pop("transition_parent_owners")
    solution["atom_owners"] = [
        {"atom_id": "A_V1", "owner_kind": "RESIDUAL"}
    ]
    result = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)

    record = result["birth_eligibility_records"][0]
    assert record["first_rejection_code"] == "A0_PARENT_OWNER_UNAVAILABLE"
    assert record["eligibility_code"] == "A0_PARENT_OWNER_UNAVAILABLE"
    assert record["first_failed_stage"] == 5
    assert record["valid_two_mode_connected_run"] is True
    assert record["column_emitted"] is False


def test_transition_probe_matches_existing_latent_path_and_rejects_duplicate() -> None:
    transition, graph, solution = _fixture()
    graph["latent_paths"] = [
        {
            "latent_path_id": "LP_EXISTING",
            "span_endpoints": [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]],
            "axis_direction": [0.0, 1.0, 0.0],
            "incidence_complete": True,
            "label_eligibility": {"BEND_STRAIGHT_COUNTED": True},
            "semantic_label": "BEND_STRAIGHT_COUNTED",
        }
    ]

    result = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)
    birth = result["birth_eligibility_records"][0]

    assert birth["matching_path_ids"] == ["LP_EXISTING"]
    assert birth["geometry_gates"]["duplicate_free"] is False
    assert birth["first_rejection_code"] == "DUPLICATE_EXISTING_PATH"
    assert birth["first_failed_stage"] == 6
    assert birth["column_emitted"] is False


def test_transition_probe_ignores_more_than_cap_ineligible_path_aliases() -> None:
    transition, graph, solution = _fixture()
    graph["latent_paths"] = [
        {
            "latent_path_id": f"LP_ALIAS_{index:03d}",
            "span_endpoints": [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]],
            "axis_direction": [0.0, 1.0, 0.0],
            "incidence_complete": index % 2 == 0,
            "label_eligibility": {"BEND_STRAIGHT_COUNTED": index % 2 == 1},
        }
        for index in range(40)
    ]
    graph["latent_paths"].append(
        {
            "latent_path_id": "LP_ONLY_ELIGIBLE",
            "span_endpoints": [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]],
            "axis_direction": [0.0, 1.0, 0.0],
            "incidence_complete": True,
            "label_eligibility": {"BEND_STRAIGHT_COUNTED": True},
        }
    )

    result = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)

    assert result["birth_eligibility_records"][0]["matching_path_ids"] == [
        "LP_ONLY_ELIGIBLE"
    ]
    assert result["birth_eligibility_records"][0]["first_rejection_code"] == (
        "DUPLICATE_EXISTING_PATH"
    )


def test_three_bin_two_sided_ladder_is_the_minimal_geometry_certificate() -> None:
    transition, graph, solution = _fixture(bin_count=3)
    result = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)
    birth = result["birth_eligibility_records"][0]

    assert birth["valid_two_mode_connected_run"] is True
    assert birth["geometry_gates"]["minimum_occupied_bins"] is True
    assert birth["geometry_gates"]["minimum_transition_rung_bins"] is True
    assert birth["bounded_counters"]["full_window_count"] == 1
    assert result["thresholds"]["minimum_occupied_bins"] == 3
    assert result["thresholds"]["minimum_transition_rung_bins"] == 2


def test_owner_blocker_cap_counts_distinct_semantic_path_categories_not_parents() -> None:
    transition, graph, solution = _fixture(bin_count=20)
    for owner in solution["transition_parent_owners"]:
        owner["owner_kind"] = "PATH"
        owner["latent_path_id"] = "LP_SHARED"
        owner["semantic_label"] = "PERIMETER_NONCOUNTED"

    result = diagnose_transition_mode_probe(transition, graph, solution, **HASHES)
    birth = result["birth_eligibility_records"][0]
    component = result["transition_component_records"][0]

    assert birth["first_failed_stage"] == 5
    assert birth["first_rejection_code"] == "NO_RESIDUAL_CERTIFIED_WINDOW"
    assert component["owner_blockers"] == [
        {
            "semantic_label": "PERIMETER_NONCOUNTED",
            "path_id": "LP_SHARED",
            "parent_count": 20,
        }
    ]
    assert component["bounded_counters"]["owner_blocker_category_count"] == 1


def test_owner_blocker_cap_fails_on_more_than_sixteen_distinct_categories() -> None:
    transition, graph, solution = _fixture(bin_count=20)
    for index, owner in enumerate(solution["transition_parent_owners"]):
        owner["owner_kind"] = "PATH"
        owner["latent_path_id"] = f"LP_{index:03d}"
        owner["semantic_label"] = "PERIMETER_NONCOUNTED"

    with pytest.raises(ValueError, match="owner blocker categories exceed cap: 20 > 16"):
        diagnose_transition_mode_probe(transition, graph, solution, **HASHES)


def test_transition_probe_rejects_missing_v2_and_cap_breaches() -> None:
    _transition, graph, solution = _fixture()
    with pytest.raises(RuntimeError, match="build_transition_atom_graph_v2"):
        diagnose_transition_mode_probe({}, graph, solution, **HASHES)

    transition, graph, solution = _fixture()
    extra = dict(transition["support_mode_table"][0])
    extra["mode_atom_id"] = "M_OVERFLOW_A"
    transition["support_mode_table"].append(extra)
    extra = dict(extra)
    extra["mode_atom_id"] = "M_OVERFLOW_B"
    transition["support_mode_table"].append(extra)
    extra = dict(extra)
    extra["mode_atom_id"] = "M_OVERFLOW_C"
    transition["support_mode_table"].append(extra)
    with pytest.raises(ValueError, match="normal modes.*exceed cap"):
        diagnose_transition_mode_probe(transition, graph, solution, **HASHES)


def test_transition_probe_rejects_forbidden_point_payloads() -> None:
    transition, graph, solution = _fixture()
    transition["support_parent_table"][0]["point_indices"] = [1, 2, 3]
    with pytest.raises(ValueError, match="forbids"):
        diagnose_transition_mode_probe(transition, graph, solution, **HASHES)


def test_transition_probe_script_hashes_inputs_and_emits_same_schema(tmp_path) -> None:
    transition, graph, solution = _fixture()
    scan_path = tmp_path / "scan.ply"
    pool_path = tmp_path / "pool.json"
    graph_path = tmp_path / "graph.json"
    solution_path = tmp_path / "solution.json"
    scan_path.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    pool_path.write_text(json.dumps(transition), encoding="utf-8")
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    solution_path.write_text(json.dumps(solution), encoding="utf-8")

    result = build_transition_probe(
        scan_path=scan_path,
        candidate_pool_path=pool_path,
        graph_path=graph_path,
        a0_solution_path=solution_path,
    )

    assert result["kind"] == TRANSITION_DIAGNOSTIC_KIND
    assert len(result["scan_sha256"]) == 64
    assert result["truth_inputs_consumed"] is False
    assert result["case_counters"]["emitted_birth_column_count"] == 0
    projection = result["a0_parent_ownership_projection"]
    assert projection["profile"] == A0_PARENT_PROJECTION_PROFILE
    assert projection["containment_rule"] == "HALF_OPEN_V2_VOXEL"
    assert projection["counters"]["residual_parent_count"] == 6
    assert projection["counters"]["unknown_parent_count"] == 0
    assert result["birth_eligibility_records"][0]["eligibility_code"] == "ELIGIBLE_DIAGNOSTIC_ONLY"


def test_a0_projection_uses_half_open_voxels_and_preserves_unanimous_owner() -> None:
    transition, _graph, solution = _fixture()
    solution["atom_owners"][0] = {
        "atom_id": "A0",
        "owner_kind": "PATH",
        "latent_path_id": "LP_KEEP",
    }
    transition["support_atom_table"].append(
        {"atom_id": "A0B", "centroid": [0.5, 0.5, 0.5]}
    )
    solution["atom_owners"].append(
        {"atom_id": "A0B", "owner_kind": "PATH", "latent_path_id": "LP_KEEP"}
    )

    records, diagnostics = project_a0_v1_owners_to_transition_parents(
        transition, transition, solution
    )
    by_parent = {record["parent_atom_id"]: record for record in records}

    assert by_parent["P0"]["owner_kind"] == "PATH"
    assert by_parent["P0"]["latent_path_id"] == "LP_KEEP"
    assert by_parent["P0"]["enclosed_legacy_atom_count"] == 2
    # A1 lies exactly on P0's upper Y boundary and therefore belongs to P1.
    assert by_parent["P1"]["enclosed_legacy_atom_count"] == 1
    assert diagnostics["counters"]["unanimous_nonresidual_parent_count"] == 1
    assert diagnostics["nearest_neighbor_used"] is False


def test_a0_projection_marks_mixed_and_missing_owners_fail_closed() -> None:
    transition, _graph, solution = _fixture()
    transition["support_atom_table"].append(
        {"atom_id": "A0B", "centroid": [0.5, 0.5, 0.5]}
    )
    solution["atom_owners"].append(
        {"atom_id": "A0B", "owner_kind": "PATH", "latent_path_id": "LP_OTHER"}
    )
    solution["atom_owners"] = [
        owner for owner in solution["atom_owners"] if owner["atom_id"] != "A5"
    ]

    records, diagnostics = project_a0_v1_owners_to_transition_parents(
        transition, transition, solution
    )
    by_parent = {record["parent_atom_id"]: record for record in records}

    assert by_parent["P0"]["owner_kind"] == "MIXED_A0_OWNERS"
    assert by_parent["P5"]["owner_kind"] == "UNKNOWN"
    assert diagnostics["counters"]["mixed_parent_count"] == 1
    assert diagnostics["counters"]["unknown_parent_count"] == 1
