from __future__ import annotations

import json
from pathlib import Path

from backend.f1_surface_graph_outline_pruning import (
    prune_surface_graph_outlines,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[1]


def _edge(
    candidate_id: str,
    first: int,
    second: int,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    **values: float,
) -> dict:
    span = sum((right - left) ** 2 for left, right in zip(start, end)) ** 0.5
    return {
        "bend_id": candidate_id,
        "candidate_id": f"scan_mesh_adjacency::{candidate_id}",
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": first,
        "surface2_id": second,
        "line_start": list(start),
        "line_end": list(end),
        "span_endpoints": [list(start), list(end)],
        "span_mm": span,
        "surface1_area": values.pop("surface1_area", 100.0),
        "surface2_area": values.pop("surface2_area", 80.0),
        "bend_angle": values.pop("bend_angle", 90.0),
        "min_surface_gap": values.pop("min_surface_gap", 1.0),
        "two_sided_normal_angle_deg": values.pop(
            "two_sided_normal_angle_deg", 80.0
        ),
        "two_sided_transition_score": values.pop(
            "two_sided_transition_score", 0.7
        ),
        **values,
    }


def test_prunes_multiply_closed_finite_nonjunction_chord() -> None:
    chord = _edge(
        "chord",
        1,
        2,
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
    )
    selected = [
        chord,
        _edge("a1", 1, 3, (0.0, 10.0, 0.0), (100.0, 5.0, 0.0)),
        _edge("a2", 2, 3, (0.0, 20.0, 0.0), (100.0, 10.0, 0.0)),
        _edge("b1", 1, 4, (0.0, 30.0, 0.0), (100.0, 15.0, 0.0)),
        _edge("b2", 2, 4, (0.0, 40.0, 0.0), (100.0, 20.0, 0.0)),
    ]

    result = prune_surface_graph_outlines(selected)

    assert result["pruned_candidate_ids"] == ["scan_mesh_adjacency::chord"]
    assert (
        result["conflicts"][0]["lane"]
        == "multiply_closed_finite_nonjunction_chord"
    )


def test_prunes_short_broad_face_triangle_chord_but_not_orthogonal_fold() -> None:
    short_chord = _edge(
        "short",
        1,
        2,
        (0.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        surface1_area=1000.0,
        surface2_area=950.0,
        bend_angle=120.0,
    )
    aligned_side = _edge(
        "aligned", 1, 3, (0.0, 5.0, 0.0), (100.0, 5.0, 0.0)
    )
    other_side = _edge(
        "other", 2, 3, (0.0, 10.0, 0.0), (0.0, 110.0, 0.0)
    )
    orthogonal = _edge(
        "orthogonal",
        4,
        5,
        (0.0, 0.0, 0.0),
        (0.0, 40.0, 0.0),
        surface1_area=1000.0,
        surface2_area=950.0,
        bend_angle=120.0,
    )
    orthogonal_sides = [
        _edge("o1", 4, 6, (0.0, 0.0, 5.0), (100.0, 0.0, 5.0)),
        _edge("o2", 5, 6, (0.0, 0.0, 10.0), (100.0, 0.0, 10.0)),
    ]

    result = prune_surface_graph_outlines(
        [short_chord, aligned_side, other_side, orthogonal, *orthogonal_sides]
    )

    assert result["pruned_candidate_ids"] == ["scan_mesh_adjacency::short"]
    assert result["conflicts"][0]["lane"] == "short_broad_face_triangle_chord"


def test_prunes_isolated_equal_span_interface_dominated_by_connected_graph() -> None:
    isolated = _edge(
        "isolated",
        1,
        2,
        (0.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        two_sided_transition_score=0.70,
    )
    witness = _edge(
        "witness",
        3,
        4,
        (0.0, 30.0, 0.0),
        (80.0, 30.0, 0.0),
        two_sided_transition_score=0.85,
    )
    graph_edge = _edge(
        "graph", 4, 5, (0.0, 0.0, 10.0), (0.0, 80.0, 10.0)
    )

    result = prune_surface_graph_outlines([isolated, witness, graph_edge])

    assert result["pruned_candidate_ids"] == ["scan_mesh_adjacency::isolated"]
    assert (
        result["conflicts"][0]["lane"]
        == "isolated_parallel_dominated_interface"
    )


def test_missing_geometry_is_deterministic_and_fails_closed() -> None:
    malformed = {
        "bend_id": "bad",
        "candidate_id": "scan_mesh_adjacency::bad",
        "surface1_id": 1,
        "surface2_id": 2,
    }
    first = prune_surface_graph_outlines([malformed])
    second = prune_surface_graph_outlines([malformed])

    assert first == second
    assert first["selected"] == [malformed]
    assert first["pruned_candidate_ids"] == []


def test_duplicate_surface_pair_makes_triangle_lane_fail_closed() -> None:
    candidate = _edge(
        "candidate",
        1,
        2,
        (0.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        surface1_area=1000.0,
        surface2_area=950.0,
        bend_angle=120.0,
    )
    duplicated_side = _edge(
        "side-a", 1, 3, (0.0, 5.0, 0.0), (100.0, 5.0, 0.0)
    )
    duplicate = {**duplicated_side, "candidate_id": "scan_mesh_adjacency::side-b"}
    other_side = _edge(
        "other", 2, 3, (0.0, 10.0, 0.0), (0.0, 110.0, 0.0)
    )

    result = prune_surface_graph_outlines(
        [candidate, duplicated_side, duplicate, other_side]
    )

    assert result["pruned_candidate_ids"] == []


def test_v17_composition_removes_only_three_false_lines() -> None:
    prediction_root = ROOT / (
        "output/experiments/"
        "shaked-mesh-nested-lopo-v17-weak-parallel-outline-pruning/predictions"
    )
    label_root = ROOT / (
        "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
    )
    if not prediction_root.exists():
        return

    before = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    after = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    removals: dict[str, list[str]] = {}
    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        case_id = path.parent.name
        selected = json.loads(path.read_text())["owned_bend_regions"]
        labels = json.loads(
            (label_root / case_id / "bend_instance_labels.json").read_text()
        )["manual_bends"]
        baseline = match_lines(labels, selected, Thresholds())
        pruning = prune_surface_graph_outlines(selected)
        improved = match_lines(labels, pruning["selected"], Thresholds())
        removals[case_id] = list(pruning["pruned_candidate_ids"])
        for metrics, result in ((before, baseline), (after, improved)):
            metrics["tp"] += result["true_positive"]
            metrics["fp"] += result["false_positive"]
            metrics["fn"] += result["false_negative"]
            metrics["exact"] += int(result["exact"])

    assert {case_id: values for case_id, values in removals.items() if values} == {
        "01_49191023": ["scan_mesh_adjacency::SMA5"],
        "02_24694000": [
            "scan_mesh_adjacency::SMA8",
            "scan_mesh_adjacency::SMA7",
        ],
    }
    assert before == {"tp": 59, "fp": 7, "fn": 0, "exact": 7}
    assert after == {"tp": 59, "fp": 4, "fn": 0, "exact": 9}


def test_all_stored_shaked_checkpoints_preserve_every_matched_line() -> None:
    experiment_root = ROOT / "output/experiments"
    label_root = ROOT / (
        "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
    )
    prediction_roots = sorted(
        experiment_root.glob("shaked-mesh-nested-lopo-v*/predictions")
    )
    if not prediction_roots:
        return

    inspected = 0
    for prediction_root in prediction_roots:
        for path in sorted(prediction_root.glob("*/decomposition_result.json")):
            case_id = path.parent.name
            label_path = label_root / case_id / "bend_instance_labels.json"
            if not label_path.exists():
                continue
            selected = json.loads(path.read_text())["owned_bend_regions"]
            labels = json.loads(label_path.read_text())["manual_bends"]
            baseline = match_lines(labels, selected, Thresholds())
            false_ids = set(baseline["unmatched_prediction_ids"])
            pruning = prune_surface_graph_outlines(selected)
            pruned_bend_ids = {
                item["bend_id"]
                for item in selected
                if item.get("candidate_id") in pruning["pruned_candidate_ids"]
            }
            assert pruned_bend_ids <= false_ids
            inspected += len(pruning["pruned_candidate_ids"])

    assert inspected >= 30


def test_is_inert_on_corrected_original13_v19_predictions() -> None:
    prediction_root = ROOT / (
        "output/experiments/f1-10of13/"
        "grouped-holdout-selector-v19-mesh-v4-tight"
    )
    if not prediction_root.exists():
        return

    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        selected = json.loads(path.read_text())["owned_bend_regions"]
        assert prune_surface_graph_outlines(selected)["pruned_candidate_ids"] == []
