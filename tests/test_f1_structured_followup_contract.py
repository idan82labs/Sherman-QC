from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from backend import f1_structured_union_graph as union
from backend.f1_structured_residual_birth import (
    BIRTH_PROFILE,
    TOTAL_BIRTH_COLUMN_CAP,
    add_birth_columns,
    generate_residual_birth_columns,
)
from scripts import build_f1_scan_only_candidate_pool as pool_builder
from scripts import run_f1_structured_union_e2e as e2e_runner


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/f1_structured_union"
INCIDENCE_POOL = FIXTURES / "compact_incidence_pool.json"
ABLATION_CONTRACTS = FIXTURES / "ablation_contracts.json"
EXACT_SEVEN_LABEL_GRAPH = FIXTURES / "exact_seven_label_graph.json"
ISOLATED_SOLVER_PYTHON = ROOT / ".venv-f1-structured/bin/python"

EXPECTED_LABELS = (
    "BEND_STRAIGHT_COUNTED",
    "CORNER_NONCOUNTED",
    "PERIMETER_NONCOUNTED",
    "HOLE_NONCOUNTED",
    "JUNCTION_NONCOUNTED",
    "CURVE_NONCOUNTED",
    "RESIDUAL_NONCOUNTED",
)
NON_RESIDUAL_LABELS = set(EXPECTED_LABELS[:-1])

CONTINUATION_CERTIFICATE = {
    "dimensionless_residuals": {
        "axis": 0.1,
        "perpendicular": 0.1,
        "gap": 0.1,
        "chord": 0.1,
        "refit": 0.1,
        "overlap": 0.0,
    },
    "engineering_tolerance_profile": "CONTINUATION_V1",
    "bridge_atom_ids": ["A1", "A2"],
    "bridge_certificate_sha256": "fixture-bridge-sha256",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _graph_api(name: str) -> Callable[..., Any]:
    function = getattr(union, name, None)
    assert callable(function), (
        "[graph-contract] professor follow-up requires public "
        f"backend.f1_structured_union_graph.{name}"
    )
    return function


@pytest.fixture(scope="module")
def exact_solver_solution(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Cross the same subprocess boundary used by the research e2e runner.

    The normal application test environment deliberately need not install
    OR-Tools.  A missing/broken isolated solver therefore fails as a solver
    contract, rather than silently skipping the integration coverage.
    """

    assert ISOLATED_SOLVER_PYTHON.is_file(), (
        "[solver-contract] pinned isolated solver interpreter is missing: "
        f"{ISOLATED_SOLVER_PYTHON}"
    )
    output_dir = tmp_path_factory.mktemp("f1-structured-solver-contract")
    return e2e_runner._run_exact_solver(
        solver_python=ISOLATED_SOLVER_PYTHON,
        graph_path=EXACT_SEVEN_LABEL_GRAPH,
        solution_path=output_dir / "solution.json",
    )


def _path_id(path: dict[str, Any]) -> str:
    return str(path["latent_path_id"])


def _line_path(
    path_id: str,
    start: float,
    end: float,
    *,
    label: str = "BEND_STRAIGHT_COUNTED",
    pair: tuple[str, str] = ("PLANE:RP1", "PLANE:RP2"),
) -> dict[str, Any]:
    return {
        "latent_path_id": path_id,
        "semantic_label": label,
        "primary_panel_pair": list(pair),
        "span_endpoints": [[start, 0.0, 0.0], [end, 0.0, 0.0]],
        "axis_direction": [1.0, 0.0, 0.0],
        "incidence_complete": True,
        "support_atom_ids": [f"{path_id}_A0", f"{path_id}_A1"],
        "support_bin_ids": [0, 1],
        "label_eligibility": {label_name: True for label_name in EXPECTED_LABELS},
    }


def test_strict_geometry_view_whitelists_inference_fields() -> None:
    raw = _read(INCIDENCE_POOL)
    raw.update(
        {
            "expected_count": 99,
            "manual_truth": {"bend_count": 99},
            "cad_path": "/forbidden/reference.step",
            "generator_diagnostics": {"metadata": "not graph input"},
        }
    )

    view = _graph_api("strict_geometry_view")(raw)

    assert set(view) <= {
        "schema_version",
        "algorithm",
        "inference_contract",
        "source_point_serialization",
        "support_atom_table",
        "support_atom_edges",
        "plane_table",
        "candidates",
        "line_objects",
    }
    assert not {
        "case_id",
        "scan_path",
        "coordinate_frame",
        "expected_count",
        "manual_truth",
        "cad_path",
        "generator_diagnostics",
    } & set(view)


def test_graph_is_invariant_to_metadata_order_endpoint_direction_and_source_ids() -> None:
    original = _read(INCIDENCE_POOL)
    mutated = deepcopy(original)
    mutated["case_id"] = "different-metadata-token"
    mutated["scan_path"] = "/different/metadata-only-name.ply"
    mutated["candidates"].reverse()
    mutated["line_objects"].reverse()

    aliases: dict[str, str] = {}
    for index, candidate in enumerate(mutated["candidates"], start=1):
        old_ids = {
            str(candidate.get(field) or "")
            for field in ("candidate_id", "bend_id", "source_candidate_id")
            if candidate.get(field)
        }
        source = str(candidate["candidate_source"])
        bare = f"RENAMED_{index}"
        replacement = f"{source}::{bare}"
        for old in old_ids:
            aliases[old] = replacement
            aliases[old.split("::", 1)[-1]] = bare
        candidate["candidate_id"] = replacement
        candidate["bend_id"] = replacement
        candidate["source_candidate_id"] = bare
        candidate["span_endpoints"].reverse()
    for line in mutated["line_objects"]:
        line["cluster_member_ids"] = [
            aliases.get(str(member_id), str(member_id))
            for member_id in reversed(line.get("cluster_member_ids") or [])
        ]
        line["span_endpoints"].reverse()

    baseline = union.build_structured_union_graph(original)
    replay = union.build_structured_union_graph(mutated)

    assert replay["graph_sha256"] == baseline["graph_sha256"]
    assert replay["latent_paths"] == baseline["latent_paths"]
    assert replay["support_atom_table"] == baseline["support_atom_table"]
    assert replay["support_atom_edges"] == baseline["support_atom_edges"]


def test_active_seven_label_ontology_is_serialized_per_path() -> None:
    graph = union.build_structured_union_graph(_read(INCIDENCE_POOL))

    assert tuple(union.SEMANTIC_LABELS) == EXPECTED_LABELS
    assert tuple(graph["semantic_labels"]) == EXPECTED_LABELS
    assert graph["latent_paths"]
    for path in graph["latent_paths"]:
        eligibility = path.get("label_eligibility")
        assert isinstance(eligibility, dict)
        assert tuple(eligibility) == EXPECTED_LABELS
        assert all(isinstance(value, bool) for value in eligibility.values())
        assert eligibility["RESIDUAL_NONCOUNTED"] is True


def test_equivalent_incidence_inheritance_is_verified_and_far_path_is_rejected() -> None:
    graph = union.build_structured_union_graph(_read(INCIDENCE_POOL))
    paths = graph["latent_paths"]

    equivalent = min(
        (
            path
            for path in paths
            if "plane_lattice" in (path.get("raw_member_sources") or [])
            and abs(float(path["midpoint"][1])) < 1.0
        ),
        key=lambda path: abs(float(path["midpoint"][0]) - 9.5),
    )
    far = min(
        paths,
        key=lambda path: abs(float(path["midpoint"][1]) - 10.0),
    )

    assert equivalent["incidence_complete"] is True
    assert set(equivalent["support_atom_ids"]) == {f"A{index}" for index in range(6)}
    assert len(equivalent["support_bin_ids"]) >= 2
    origin = equivalent["incidence_origin"]
    assert origin["equivalent_path_propagation"] or origin["compact_atom_completion"]
    certificate = equivalent["incidence_certificate"]
    assert certificate["tolerance_profile"] == "INCIDENCE_V1"
    assert max(certificate["max_dimensionless_residuals"].values()) <= 1.0

    assert far["incidence_complete"] is False
    assert far["support_atom_ids"] == []
    assert all(
        far["label_eligibility"][label] is False
        for label in NON_RESIDUAL_LABELS
    )
    assert far["label_eligibility"]["RESIDUAL_NONCOUNTED"] is True


def test_truncated_incidence_is_never_silently_exact() -> None:
    pool = _read(INCIDENCE_POOL)
    for record in (*pool["candidates"], *pool["line_objects"]):
        if "SOURCE_WITH_SUPPORT" in json.dumps(record, sort_keys=True):
            record["support_atom_ids_truncated"] = True

    try:
        graph = union.build_structured_union_graph(pool)
    except ValueError as error:
        assert "truncat" in str(error).lower()
    else:
        for path in graph["latent_paths"]:
            assert path["support_atom_ids_truncated"] is False
            if path["incidence_complete"]:
                assert path["incidence_origin"].get("recomputed_from_complete_atom_table") is True


def test_normalized_atom_mass_and_distributed_bin_contract_are_present() -> None:
    graph = union.build_structured_union_graph(_read(INCIDENCE_POOL))
    masses = {
        str(atom["atom_id"]): int(atom["normalized_support_mass"])
        for atom in graph["support_atom_table"]
    }

    assert set(masses.values()) == {1, 4}
    assert all(1 <= mass <= 4 for mass in masses.values())
    supported = [
        path for path in graph["latent_paths"] if path.get("incidence_complete")
    ]
    assert supported
    assert all(len(set(path["support_bin_ids"])) >= 2 for path in supported)


def test_duplicate_and_shared_boundary_constraints_are_explicit_graph_relations() -> None:
    graph = union.build_structured_union_graph(_read(INCIDENCE_POOL))
    duplicate_pairs = graph.get("hard_duplicate_pairs") or graph.get("duplicate_relations")
    boundary_arcs = graph.get("panel_boundary_arcs")

    assert duplicate_pairs, "equivalent source and plane-lattice paths need a hard duplicate relation"
    assert boundary_arcs, "shared panel-boundary ownership must be explicit"
    path_ids = {_path_id(path) for path in graph["latent_paths"]}
    for relation in duplicate_pairs:
        pair = relation.get("path_ids") if isinstance(relation, dict) else relation
        assert len(pair) == 2
        assert set(map(str, pair)) <= path_ids
    for arc in boundary_arcs:
        assert arc["arc_id"]
        assert arc.get("compatible_object_ids")
        assert arc.get("residual_owner_allowed") is True


def test_continuation_validation_is_endpoint_aware_acyclic_and_count_exact() -> None:
    paths = [
        _line_path("P0", 0.0, 8.0),
        _line_path("P1", 9.0, 17.0),
        _line_path("P2", 18.0, 26.0),
    ]
    continuations = [
        {
            "continuation_id": "CONT_01",
            "path_ids": ["P0", "P1"],
            "endpoint_ids": [1, 0],
            **CONTINUATION_CERTIFICATE,
        },
        {
            "continuation_id": "CONT_12",
            "path_ids": ["P1", "P2"],
            "endpoint_ids": [1, 0],
            **CONTINUATION_CERTIFICATE,
        },
    ]

    validator = _graph_api("validate_continuations")
    validated = validator(paths, continuations)
    assert len(validated) == 2
    assert union.count_bend_components(paths, validated) == 1

    same_endpoint = deepcopy(continuations)
    same_endpoint[1]["endpoint_ids"] = [0, 0]
    with pytest.raises(ValueError, match="endpoint|degree"):
        validator(paths, same_endpoint)

    cycle = [
        *continuations,
        {
            "continuation_id": "CONT_20",
            "path_ids": ["P2", "P0"],
            "endpoint_ids": [1, 0],
            **CONTINUATION_CERTIFICATE,
        },
    ]
    with pytest.raises(ValueError, match="acyclic|cycle"):
        validator(paths, cycle)

    incomplete = deepcopy(continuations[:1])
    incomplete[0]["dimensionless_residuals"].pop("refit")
    with pytest.raises(ValueError, match="certificate|residual"):
        validator(paths, incomplete)


def test_disjoint_factors_coexist_and_only_real_conflicts_are_serialized() -> None:
    conflict_builder = _graph_api("build_factor_conflicts")
    factors = [
        {
            "factor_id": "F1",
            "label_implications": [
                {"path_id": "P0", "label": "BEND_STRAIGHT_COUNTED"}
            ],
            "required_support_atom_ids": ["A0"],
        },
        {
            "factor_id": "F2",
            "label_implications": [
                {"path_id": "P1", "label": "CORNER_NONCOUNTED"}
            ],
            "required_support_atom_ids": ["A1"],
        },
        {
            "factor_id": "F3",
            "label_implications": [
                {"path_id": "P0", "label": "PERIMETER_NONCOUNTED"}
            ],
            "required_support_atom_ids": ["A2"],
        },
    ]

    conflicts = {
        frozenset(pair.get("factor_ids", pair) if isinstance(pair, dict) else pair)
        for pair in conflict_builder(factors)
    }

    assert frozenset(("F1", "F2")) not in conflicts
    assert frozenset(("F1", "F3")) in conflicts
    assert frozenset(("F2", "F3")) not in conflicts


def test_global_working_cloud_is_strictly_bounded_and_permutation_invariant() -> None:
    count = pool_builder.GLOBAL_POINT_BUDGET + 321
    x = np.arange(count, dtype=np.float64)
    points = np.column_stack((x, x % 17, x % 31))
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    permutation = np.random.default_rng(1729).permutation(count)

    first_points, first_normals = pool_builder.deterministic_geometry_sample(
        points, normals
    )
    replay_points, replay_normals = pool_builder.deterministic_geometry_sample(
        points[permutation], normals[permutation]
    )

    assert len(first_points) == pool_builder.GLOBAL_POINT_BUDGET == 200_000
    np.testing.assert_array_equal(first_points, replay_points)
    np.testing.assert_array_equal(first_normals, replay_normals)


def test_every_candidate_generator_receives_the_same_bounded_working_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[int] = []

    def cylindrical(points: np.ndarray, normals: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, Any]:
        observed_sizes.append(len(points))
        assert kwargs["point_budget"] == 200_000
        return {"algorithm": "fixture", "candidates": []}

    def planes(points: np.ndarray) -> list[dict[str, Any]]:
        observed_sizes.append(len(points))
        return []

    def intersections(points: np.ndarray, planes: list[dict[str, Any]]) -> dict[str, Any]:
        observed_sizes.append(len(points))
        return {"semantic": "fixture", "candidates": []}

    monkeypatch.setattr(pool_builder, "complete_cylindrical_bands", cylindrical)
    monkeypatch.setattr(pool_builder, "extract_deterministic_planes", planes)
    monkeypatch.setattr(pool_builder, "build_supported_plane_intersections", intersections)
    monkeypatch.setattr(pool_builder, "merge_collinear_fragments", lambda values: [])
    monkeypatch.setattr(pool_builder, "complete_parallel_endpoint_connectors", lambda values: [])

    count = 200_123
    points = np.column_stack(
        (
            np.arange(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    )
    normals = np.tile([0.0, 1.0, 0.0], (count, 1))
    payload = pool_builder.build_case_pool(
        decomposition={
            "owned_bend_regions": [],
            "bend_hypotheses": [],
            "atom_graph": {"atoms": [], "adjacency": {}, "edge_weights": {}},
            "current_zero_shot_result": {
                "bend_support_strips": [],
                "bend_evidence_atoms": [],
            },
        },
        points=points,
        normals=normals,
        scan_mesh_adjacency=[],
    )

    assert observed_sizes
    assert set(observed_sizes) == {200_000}
    assert payload["transient_sampled_point_count"] == 200_000
    assert payload["source_point_serialization"] is False


def test_ablation_a_contract_is_falsifiable_and_not_a_length_rule() -> None:
    contract = _read(ABLATION_CONTRACTS)["outline_free_edge_rejection"]

    assert set(contract["arms"]) == {"A0", "A1", "A2"}
    assert contract["arms"]["A0"] == {
        "class_conditioned_side_ownership": False,
        "perimeter_semantics": False,
    }
    assert contract["arms"]["A1"]["class_conditioned_side_ownership"] is True
    assert contract["arms"]["A2"]["perimeter_semantics"] is True
    assert contract["factor_policy"] == "457_specific_chord_factor_disabled_in_all_arms"
    assert contract["forbidden_discriminator"] == "span_or_length_threshold"
    assert contract["go"]["minimum_relative_outline_fp_reduction"] == 0.5
    assert contract["go"]["maximum_absolute_bend_recall_loss"] == 0.05
    assert contract["go"]["required_deterministic_replay_rate"] == 1.0
    assert contract["stop"]["outline_fp_reduction_below"] == 0.3


def test_ablation_b_contract_removes_all_candidates_without_truth_leakage() -> None:
    contract = _read(ABLATION_CONTRACTS)["missing_bend_birth"]
    deletion = contract["deletion_protocol"]

    assert set(contract["arms"]) == {"B0", "B1"}
    assert contract["arms"]["B0"]["birth_profile"] is None
    assert contract["arms"]["B1"]["birth_profile"] == "BIRTH_V1"
    assert deletion["remove_matching_candidates_from_every_source"] is True
    assert deletion["require_pre_solve_candidate_recall_zero"] is True
    assert deletion["retain_support_atom_table"] is True
    assert deletion["retain_support_atom_edges"] is True
    assert deletion["retain_plane_table"] is True
    assert deletion["expose_deleted_identity_to_inference"] is False
    assert deletion["expose_expected_count_to_inference"] is False
    assert deletion["require_undeleted_false_birth_control"] is True
    assert contract["go"]["minimum_deleted_object_recovery_rate"] == 0.5
    assert contract["go"]["maximum_false_births_per_undeleted_scan"] == 0.25
    assert contract["stop"]["deleted_object_recovery_below"] == 0.3


def test_solver_contract_exact_seven_labels_mass_continuation_and_factors(
    exact_solver_solution: dict[str, Any],
) -> None:
    first = exact_solver_solution
    expected = {
        "P_B0": "BEND_STRAIGHT_COUNTED",
        "P_B1": "BEND_STRAIGHT_COUNTED",
        "P_CORNER": "CORNER_NONCOUNTED",
        "P_PERIMETER": "PERIMETER_NONCOUNTED",
        "P_HOLE": "HOLE_NONCOUNTED",
        "P_JUNCTION": "JUNCTION_NONCOUNTED",
        "P_CURVE": "CURVE_NONCOUNTED",
        "P_RESIDUAL": "RESIDUAL_NONCOUNTED",
    }

    assert {
        str(item["latent_path_id"]): str(item["semantic_label"])
        for item in first["assignments"]
    } == expected
    assert set(first["selected_factor_ids"]) == {
        "F_TRUE_BENDS",
        "F_TYPED_NONCOUNTED",
    }
    assert [item["continuation_id"] for item in first["selected_continuations"]] == [
        "CONT_B0_B1"
    ]
    assert first["counted_fragment_count"] == 2
    assert first["counted_component_count"] == 1
    assert first["owned_support_mass_by_path"] == {
        "P_B0": 5,
        "P_B1": 4,
        "P_CORNER": 2,
        "P_CURVE": 4,
        "P_HOLE": 5,
        "P_JUNCTION": 4,
        "P_PERIMETER": 3,
        "P_RESIDUAL": 0,
    }
    assert first["residual_owned_support_mass"] == 4
    owners = {str(item["atom_id"]): dict(item) for item in first["atom_owners"]}
    assert owners["A_R"] == {
        "atom_id": "A_R",
        "owner_kind": "RESIDUAL",
        "latent_path_id": None,
        "normalized_support_mass": 4,
    }
    assert owners["A_B0_0"]["owner_kind"] == "PATH"
    assert owners["A_B0_0"]["latent_path_id"] == "P_B0"

    solver = first["solver"]
    assert solver["status"] == "OPTIMAL"
    assert solver["second_best_status"] in {"OPTIMAL", "INFEASIBLE"}
    assert solver["exact_primary_optimality_proven"] is True
    assert solver["maximum_normalized_nonresidual_owned_mass"] == 27
    assert solver["maximum_compatible_continuations"] == 1
    if solver["second_best_objective_int"] is not None:
        assert solver["second_best_objective_int"] >= solver["primary_objective_int"]
        assert solver["energy_margin_int"] == (
            solver["second_best_objective_int"] - solver["primary_objective_int"]
        )


def test_solver_contract_is_byte_stable_under_graph_order_reversal(
    tmp_path: Path,
    exact_solver_solution: dict[str, Any],
) -> None:
    graph = _read(EXACT_SEVEN_LABEL_GRAPH)
    for field in (
        "support_atom_table",
        "latent_paths",
        "topology_factors",
        "factor_conflicts",
        "continuations",
    ):
        graph[field].reverse()
    replay_graph = tmp_path / "reversed.json"
    replay_graph.write_text(json.dumps(graph), encoding="utf-8")
    replay = e2e_runner._run_exact_solver(
        solver_python=ISOLATED_SOLVER_PYTHON,
        graph_path=replay_graph,
        solution_path=tmp_path / "replay-solution.json",
    )

    assert replay == exact_solver_solution


def test_birth_v1_accepts_exact_solver_owner_schema_and_remains_bounded(
    exact_solver_solution: dict[str, Any],
) -> None:
    graph = _read(EXACT_SEVEN_LABEL_GRAPH)

    # This compact fixture has no residual plane-supported run, so BIRTH_V1
    # must consume the solver's explicit owner records and deterministically
    # produce no hallucinated columns.
    assert generate_residual_birth_columns(graph, exact_solver_solution) == []
    assert BIRTH_PROFILE == "BIRTH_V1"
    assert TOTAL_BIRTH_COLUMN_CAP == 64

    proposal = deepcopy(graph["latent_paths"][0])
    proposal.update(
        {
            "birth_id": "BIRTH_FIXTURE",
            "latent_path_id": "LP_BIRTH_FIXTURE",
            "is_birth": True,
            "birth_profile": BIRTH_PROFILE,
            "object_cost_int": 1_000_000,
            "residual_cost_per_mass_int": 1_000_000,
            "lower_bound_int": -1,
        }
    )
    augmented = add_birth_columns(graph, [proposal])
    assert augmented["birth_profile"] == BIRTH_PROFILE
    assert augmented["birth_column_cap"] == TOTAL_BIRTH_COLUMN_CAP
    assert augmented["birth_columns"] == ["BIRTH_FIXTURE"]
    assert augmented["graph_sha256"] != graph["graph_sha256"]


def test_e2e_contract_refits_one_component_from_two_counted_fragments(
    exact_solver_solution: dict[str, Any],
) -> None:
    graph = _read(EXACT_SEVEN_LABEL_GRAPH)
    components = e2e_runner.emit_counted_components(graph, exact_solver_solution)

    assert len(components) == 1
    component = components[0]
    assert component["semantic_label"] == "BEND_STRAIGHT_COUNTED"
    assert component["member_path_ids"] == ["P_B0", "P_B1"]
    assert component["owned_atom_ids"] == ["A_B0_0", "A_B0_1", "A_B1_0", "A_B1_1"]
    assert component["span_endpoints"] == [[0.0, 0.0, 0.0], [17.0, 0.0, 0.0]]
    assert component["span_mm"] == pytest.approx(17.0)
    assert component["coordinate_frame"] == "scan_raw"
