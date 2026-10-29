from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.solve_f1_structured_union_cpsat import (
    BEND,
    CONTINUATION_END,
    CORNER,
    LABELS,
    MULTIWAY_JUNCTION,
    RESIDUAL,
    solve,
)


def _contract() -> dict:
    return {
        "uses_part_identifier": False,
        "uses_filename_features": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
        "uses_cad": False,
        "source_point_serialization": False,
    }


def _path(
    path_id: str,
    label: str,
    atom_ids: list[str],
    *,
    x0: float = 0.0,
    pair: tuple[str, str] | None = None,
    extra_eligible: tuple[str, ...] = (),
) -> dict:
    bins = {atom_id: str(index) for index, atom_id in enumerate(atom_ids)}
    result = {
        "latent_path_id": path_id,
        "geometry_kind": "LINE",
        "span_endpoints": [[x0, 0.0, 0.0], [x0 + 9.0, 0.0, 0.0]],
        "axis_direction": [1.0, 0.0, 0.0],
        "incidence_complete": True,
        "incidence_certificate": {
            "tolerance_profile": "INCIDENCE_V1",
            "atom_table_sha256": "fixture-atoms",
            "max_dimensionless_residuals": {
                "perpendicular": 0.1,
                "endpoint": 0.1,
                "normal": 0.1,
                "axis": 0.1,
            },
        },
        "support_atom_ids_truncated": False,
        "support_atom_ids": atom_ids,
        "support_atom_bin_ids": bins,
        "label_eligibility": {
            candidate: candidate in {label, RESIDUAL, *extra_eligible}
            for candidate in LABELS
        },
    }
    if pair is not None:
        result["primary_panel_pair"] = list(pair)
        result["support_atom_side_ids"] = {
            atom_id: list(pair) for atom_id in atom_ids
        }
    return result


def _atom(atom_id: str, x: float, *, mass: int = 4) -> dict:
    return {
        "atom_id": atom_id,
        "centroid": [x, 0.0, 0.0],
        "normal": [0.0, 1.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "local_spacing_mm": 1.0,
        "area_proxy_mm2": float(mass),
        "normalized_support_mass": mass,
        "incident_plane_ids": ["PLANE:F0", "PLANE:F1"],
    }


def _graph(paths: list[dict], atoms: list[dict], **extra: object) -> dict:
    return {
        "graph_sha256": "hand-solvable-professor-followup",
        "inference_contract": _contract(),
        "semantic_labels": list(LABELS),
        "latent_paths": paths,
        "support_atom_table": atoms,
        "support_atom_edges": [],
        "topology_factors": [],
        "continuations": [],
        **extra,
    }


def _certificate() -> dict:
    return {
        "dimensionless_residuals": {
            "axis": 0.1,
            "perpendicular": 0.2,
            "gap": 0.3,
            "chord": 0.2,
            "refit": 0.1,
            "overlap": 0.0,
        },
        "engineering_tolerance_profile": "CONTINUATION_V1",
        "bridge_atom_ids": ["A1", "B0"],
        "bridge_certificate_sha256": "fixture-scan-bridge",
    }


def test_all_seven_labels_are_active_and_emit_exact_atom_owners() -> None:
    semantic = list(LABELS[:-1])
    paths: list[dict] = []
    atoms: list[dict] = []
    for index, label in enumerate(semantic):
        atom_ids = [f"A{index}_0", f"A{index}_1"]
        pair = ("PLANE:F0", "PLANE:F1") if label == BEND else None
        paths.append(_path(f"P{index}", label, atom_ids, x0=20.0 * index, pair=pair))
        atoms.extend(_atom(atom_id, 20.0 * index + 3.0 * atom_index) for atom_index, atom_id in enumerate(atom_ids))

    result = solve(_graph(paths, atoms))
    assignment = {item["latent_path_id"]: item for item in result["assignments"]}

    assert {item["semantic_label"] for item in assignment.values()} == set(semantic)
    assert assignment["P1"]["endpoint_types"] == [MULTIWAY_JUNCTION, MULTIWAY_JUNCTION]
    assert result["residual_owned_support_count"] == 0
    assert len(result["atom_owners"]) == len(atoms)
    assert {item["owner_kind"] for item in result["atom_owners"]} == {"PATH"}
    assert all(result["owned_axial_bin_count_by_path"][path["latent_path_id"]] == 2 for path in paths)


def test_unsupported_is_hard_residual_and_truncated_exact_input_is_rejected() -> None:
    path = _path("P0", BEND, [], pair=("PLANE:F0", "PLANE:F1"))
    path["incidence_complete"] = False
    result = solve(_graph([path], []))

    assert result["assignments"][0]["semantic_label"] == RESIDUAL
    assert result["unsupported_selected_path_count"] == 0

    truncated = deepcopy(path)
    truncated["support_atom_ids_truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        solve(_graph([truncated], []))


def test_weighted_mdl_ownership_differs_from_atom_count() -> None:
    heavy_ids = ["H0", "H1"]
    light_ids = ["L0", "L1"]
    paths = [
        _path("HEAVY", "PERIMETER_NONCOUNTED", heavy_ids),
        _path("LIGHT", "PERIMETER_NONCOUNTED", light_ids),
    ]
    atoms = [
        _atom("H0", 0.0, mass=4),
        _atom("H1", 3.0, mass=4),
        _atom("L0", 20.0, mass=1),
        _atom("L1", 23.0, mass=1),
    ]
    result = solve(_graph(paths, atoms))
    labels = {item["latent_path_id"]: item["semantic_label"] for item in result["assignments"]}

    assert labels == {"HEAVY": "PERIMETER_NONCOUNTED", "LIGHT": RESIDUAL}
    assert result["owned_support_count_by_path"] == {"HEAVY": 2, "LIGHT": 0}
    assert result["owned_support_mass_by_path"] == {"HEAVY": 8, "LIGHT": 0}
    assert result["residual_owned_support_count"] == 2
    assert result["residual_owned_support_mass"] == 2


def test_disjoint_factors_coexist_but_real_conflict_is_pairwise() -> None:
    paths = [
        _path("P0", BEND, ["A0", "A1"], pair=("PLANE:F0", "PLANE:F1"), extra_eligible=(CORNER,)),
        _path("P1", "PERIMETER_NONCOUNTED", ["B0", "B1"]),
    ]
    atoms = [_atom("A0", 0.0), _atom("A1", 3.0), _atom("B0", 20.0), _atom("B1", 23.0)]
    factors = [
        {"factor_id": "F_BEND", "energy_int": -2_000_000, "label_implications": [{"path_id": "P0", "label": BEND}]},
        {"factor_id": "F_DISJOINT", "energy_int": -2_000_000, "label_implications": [{"path_id": "P1", "label": "PERIMETER_NONCOUNTED"}]},
        {"factor_id": "F_CORNER", "energy_int": -1_000_000, "label_implications": [{"path_id": "P0", "label": CORNER}]},
    ]
    result = solve(
        _graph(
            paths,
            atoms,
            topology_factors=factors,
            factor_conflicts=[{"factor_ids": ["F_BEND", "F_CORNER"], "reason": "contradictory_label"}],
        )
    )

    assert result["selected_factor_ids"] == ["F_BEND", "F_DISJOINT"]
    assert ["F_BEND", "F_CORNER"] in result["factor_conflict_pairs"]


def test_certified_continuation_sets_endpoint_types_and_exact_component_count() -> None:
    paths = [
        _path("P0", BEND, ["A0", "A1"], pair=("PLANE:F0", "PLANE:F1")),
        _path("P1", BEND, ["B0", "B1"], x0=10.0, pair=("PLANE:F0", "PLANE:F1")),
    ]
    atoms = [_atom("A0", 0.0), _atom("A1", 3.0), _atom("B0", 10.0), _atom("B1", 13.0)]
    continuation = {
        "continuation_id": "CONT_01",
        "path_ids": ["P0", "P1"],
        "endpoint_ids": [1, 0],
        **_certificate(),
    }
    result = solve(_graph(paths, atoms, continuations=[continuation]))
    assignment = {item["latent_path_id"]: item for item in result["assignments"]}

    assert result["counted_fragment_count"] == 2
    assert result["counted_component_count"] == 1
    assert [item["continuation_id"] for item in result["selected_continuations"]] == ["CONT_01"]
    assert assignment["P0"]["endpoint_types"][1] == CONTINUATION_END
    assert assignment["P1"]["endpoint_types"][0] == CONTINUATION_END

    invalid = deepcopy(continuation)
    invalid["dimensionless_residuals"]["overlap"] = 1.01
    with pytest.raises(ValueError, match="outside frozen"):
        solve(_graph(paths, atoms, continuations=[invalid]))


def test_duplicate_boundary_and_exact_lex_replay_are_order_invariant() -> None:
    paths = [
        _path("P0", BEND, ["A0", "A1"], pair=("PLANE:F0", "PLANE:F1")),
        _path("P1", BEND, ["B0", "B1"], pair=("PLANE:F0", "PLANE:F1")),
    ]
    atoms = [_atom("A0", 0.0), _atom("A1", 3.0), _atom("B0", 20.0), _atom("B1", 23.0)]
    graph = _graph(
        paths,
        atoms,
        hard_duplicate_pairs=[{"path_ids": ["P0", "P1"]}],
        panel_boundary_arcs=[
            {"arc_id": "ARC0", "compatible_object_ids": ["P0", "P1"], "residual_owner_allowed": True}
        ],
    )
    baseline = solve(graph)
    replay_graph = deepcopy(graph)
    replay_graph["latent_paths"].reverse()
    replay_graph["support_atom_table"].reverse()
    replay_graph["hard_duplicate_pairs"][0]["path_ids"].reverse()
    replay_graph["panel_boundary_arcs"][0]["compatible_object_ids"].reverse()
    replay = solve(replay_graph)

    assert json.dumps(replay, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    assert sum(item["counted"] for item in baseline["assignments"]) == 1
    assert baseline["solver"]["status"] == "OPTIMAL"
    assert baseline["solver"]["second_best_status"] == "OPTIMAL"
    assert len(baseline["boundary_owners"]) == 1
