from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.f1_structured_union_graph import (
    BEND_STRAIGHT_COUNTED,
    CORNER_NONCOUNTED,
    RESIDUAL_NONCOUNTED,
    SEMANTIC_LABELS,
    apply_relational_chord_corner_factor,
    build_structured_union_graph,
    count_bend_components,
    solve_structured_union_baseline,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[2]
PROFESSOR_PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Structured_Inference_Dropoff_20260804"
)
CANDIDATE_POOL = PROFESSOR_PACKAGE / "evidence/candidate_pool.json"
STEP_LABELS = (
    ROOT
    / "output/research/f1-unseen-challenge-20260804/45796000"
    / "manual_adjudication/cad_drawing_backed_bend_labels.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_labels(solution: dict) -> dict[tuple[str, str], str]:
    return {
        tuple(path["primary_panel_pair"]): str(path["semantic_label"])
        for path in solution["latent_paths"]
        if path.get("primary_panel_pair") is not None
        and path["semantic_label"] != RESIDUAL_NONCOUNTED
    }


def test_457_designed_regression_labels_are_eight_bends_and_four_corners() -> None:
    solution = solve_structured_union_baseline(_read(CANDIDATE_POOL))

    expected_bends = {
        tuple(sorted(pair))
        for pair in (
            ("RP1", "RP7"),
            ("RP1", "RP12"),
            ("RP1", "RP9"),
            ("RP1", "RP5"),
            ("RP3", "RP5"),
            ("RP3", "RP7"),
            ("RP3", "RP12"),
            ("RP3", "RP9"),
        )
    }
    expected_corners = {
        tuple(sorted(pair))
        for pair in (
            ("RP5", "RP9"),
            ("RP12", "RP7"),
            ("RP12", "RP5"),
            ("RP7", "RP9"),
        )
    }
    labels = _pair_labels(solution)

    assert solution["bend_count"] == 8
    assert {pair for pair, label in labels.items() if label == BEND_STRAIGHT_COUNTED} == expected_bends
    assert {pair for pair, label in labels.items() if label == CORNER_NONCOUNTED} == expected_corners
    assert len(solution["topology_factors"]) == 1
    assert solution["topology_factors"][0]["host_panels"] == ["RP1", "RP3"]
    for path in solution["latent_paths"]:
        if path.get("semantic_label") == CORNER_NONCOUNTED:
            assert path["endpoint_types"] == ["MULTIWAY_JUNCTION", "MULTIWAY_JUNCTION"]


def test_457_designed_regression_geometry_is_postlock_exact() -> None:
    # STEP truth enters only after the scan-only solution is frozen above this
    # line.  This is a post-design regression, not an inference input.
    solution = solve_structured_union_baseline(_read(CANDIDATE_POOL))
    predictions = [
        {
            "bend_id": path["latent_path_id"],
            "span_endpoints": path["span_endpoints"],
        }
        for path in solution["latent_paths"]
        if path["semantic_label"] == BEND_STRAIGHT_COUNTED
    ]
    gold = _read(STEP_LABELS)["owned_bend_regions"]

    evaluation = match_lines(gold, predictions, Thresholds())

    assert evaluation["true_positive"] == 8
    assert evaluation["false_positive"] == 0
    assert evaluation["false_negative"] == 0
    assert evaluation["exact"] is True


def test_union_solution_ignores_metadata_order_and_endpoint_direction() -> None:
    original = _read(CANDIDATE_POOL)
    mutated = deepcopy(original)
    mutated["case_id"] = "randomized-name-that-must-not-be-a-feature"
    mutated["scan_path"] = "/tmp/different_filename.ply"
    mutated["candidates"].reverse()
    mutated["line_objects"].reverse()
    for record in (*mutated["candidates"], *mutated["line_objects"]):
        if record.get("span_endpoints") is not None:
            record["span_endpoints"].reverse()
        if record.get("cluster_member_ids") is not None:
            record["cluster_member_ids"].reverse()
        if record.get("cluster_sources") is not None:
            record["cluster_sources"].reverse()

    baseline = solve_structured_union_baseline(original)
    replay = solve_structured_union_baseline(mutated)

    assert replay["graph_sha256"] == baseline["graph_sha256"]
    assert replay["latent_paths"] == baseline["latent_paths"]
    assert replay["topology_factors"] == baseline["topology_factors"]
    assert replay["bend_count"] == baseline["bend_count"]


def test_short_line_is_not_a_corner_without_relational_topology() -> None:
    pool = {
        "source_point_serialization": False,
        "inference_contract": {},
        "candidates": [],
        "line_objects": [
            {
                "span_endpoints": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                "incident_plane_ids": ["RP1", "RP2"],
                "representative_source": "plane_lattice",
                "cluster_sources": ["plane_lattice", "scan_mesh_adjacency"],
                "finite_support_longest_run_fraction": 1.0,
                "endpoint_support_fraction": 1.0,
            }
        ],
    }

    graph = build_structured_union_graph(pool)
    solution = apply_relational_chord_corner_factor(graph)

    assert solution["topology_factors"] == []
    assert solution["latent_paths"][0]["semantic_label"] == RESIDUAL_NONCOUNTED
    assert solution["bend_count"] == 0


def test_only_counted_label_contributes_and_continuations_count_components() -> None:
    assert set(SEMANTIC_LABELS) == {
        "BEND_STRAIGHT_COUNTED",
        "CORNER_NONCOUNTED",
        "PERIMETER_NONCOUNTED",
        "HOLE_NONCOUNTED",
        "JUNCTION_NONCOUNTED",
        "CURVE_NONCOUNTED",
        "RESIDUAL_NONCOUNTED",
    }
    paths = [
        {
            "latent_path_id": f"P{index}",
            "semantic_label": BEND_STRAIGHT_COUNTED,
            "primary_panel_pair": ["F1", "F2"],
        }
        for index in range(3)
    ]
    paths.append(
        {
            "latent_path_id": "CORNER",
            "semantic_label": CORNER_NONCOUNTED,
            "primary_panel_pair": ["F2", "F3"],
        }
    )

    assert count_bend_components(paths, [("P0", "P1"), ("P1", "P2")]) == 1
    assert count_bend_components(paths, []) == 3


def test_continuations_reject_mixed_panel_pairs_and_cycles() -> None:
    paths = [
        {
            "latent_path_id": "P0",
            "semantic_label": BEND_STRAIGHT_COUNTED,
            "primary_panel_pair": ["F1", "F2"],
        },
        {
            "latent_path_id": "P1",
            "semantic_label": BEND_STRAIGHT_COUNTED,
            "primary_panel_pair": ["F2", "F3"],
        },
    ]
    with pytest.raises(ValueError, match="same incident panel pair"):
        count_bend_components(paths, [("P0", "P1")])

    same_pair = [
        {**path, "primary_panel_pair": ["F1", "F2"]}
        for path in (
            *paths,
            {
                "latent_path_id": "P2",
                "semantic_label": BEND_STRAIGHT_COUNTED,
            },
        )
    ]
    with pytest.raises(ValueError, match="acyclic"):
        count_bend_components(
            same_pair,
            [("P0", "P1"), ("P1", "P2"), ("P2", "P0")],
        )
