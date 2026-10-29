from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from backend.f1_weak_parallel_outline_pruning import prune_weak_parallel_outlines
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[1]


def _edge(
    candidate_id: str,
    surfaces: tuple[int, int],
    y: float,
    *,
    source: str,
    transition: float,
    transition_angle: float,
    tight_to_wide: float,
):
    return {
        "candidate_id": candidate_id,
        "candidate_source": source,
        "surface1_id": surfaces[0],
        "surface2_id": surfaces[1],
        "line_start": [0.0, y, 0.0],
        "line_end": [40.0, y, 0.0],
        "two_sided_transition_score": transition,
        "two_sided_normal_angle_deg": transition_angle,
        "tight_to_wide_normal_second_ratio": tight_to_wide,
    }


def test_prunes_narrow_one_sided_and_concentrated_parallel_outlines() -> None:
    witness = _edge(
        "witness",
        (1, 2),
        0.0,
        source="scan_mesh_adjacency",
        transition=0.80,
        transition_angle=85.0,
        tight_to_wide=0.20,
    )
    narrow = _edge(
        "narrow",
        (1, 3),
        12.0,
        source="scan_mesh_adjacency",
        transition=0.12,
        transition_angle=20.0,
        tight_to_wide=1.50,
    )
    concentrated = _edge(
        "concentrated",
        (2, 4),
        8.0,
        source="cross_source_line_object",
        transition=0.42,
        transition_angle=70.0,
        tight_to_wide=1.70,
    )

    result = prune_weak_parallel_outlines([narrow, witness, concentrated])

    assert [item["candidate_id"] for item in result["selected"]] == ["witness"]
    assert result["pruned_candidate_ids"] == ["narrow", "concentrated"]
    assert [item["lane"] for item in result["conflicts"]] == [
        "narrow_one_sided_transition",
        "tight_corridor_concentration",
    ]


def test_preserves_strong_distinct_or_unsupported_parallel_interfaces() -> None:
    witness = _edge(
        "witness",
        (1, 2),
        0.0,
        source="scan_mesh_adjacency",
        transition=0.80,
        transition_angle=85.0,
        tight_to_wide=0.20,
    )
    candidates = [
        _edge(
            "strong",
            (1, 3),
            12.0,
            source="scan_mesh_adjacency",
            transition=0.60,
            transition_angle=70.0,
            tight_to_wide=1.50,
        ),
        _edge(
            "unshared",
            (4, 5),
            12.0,
            source="scan_mesh_adjacency",
            transition=0.12,
            transition_angle=20.0,
            tight_to_wide=1.50,
        ),
        _edge(
            "too_near",
            (1, 6),
            5.0,
            source="scan_mesh_adjacency",
            transition=0.12,
            transition_angle=20.0,
            tight_to_wide=1.50,
        ),
    ]

    for candidate in candidates:
        result = prune_weak_parallel_outlines([candidate, witness])
        assert result["pruned_candidate_ids"] == []


def test_is_deterministic_order_preserving_and_does_not_mutate_inputs() -> None:
    witness = _edge(
        "witness",
        (1, 2),
        0.0,
        source="scan_mesh_adjacency",
        transition=0.80,
        transition_angle=85.0,
        tight_to_wide=0.20,
    )
    outline = _edge(
        "outline",
        (1, 3),
        12.0,
        source="scan_mesh_adjacency",
        transition=0.12,
        transition_angle=20.0,
        tight_to_wide=1.50,
    )
    selected = [outline, witness]
    before = deepcopy(selected)

    first = prune_weak_parallel_outlines(selected)
    second = prune_weak_parallel_outlines(selected)

    assert first == second
    assert selected == before
    assert [item["candidate_id"] for item in first["selected"]] == ["witness"]


def test_v16_prunes_only_three_false_outlines_without_losing_true_bends() -> None:
    prediction_root = ROOT / (
        "output/experiments/"
        "shaked-mesh-nested-lopo-v16-scan-supported-short-seed-completion/"
        "predictions"
    )
    label_root = ROOT / (
        "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
    )
    if not prediction_root.exists():
        return

    before = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    after = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    removals = {}
    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        case_id = path.parent.name
        prediction = json.loads(path.read_text())
        labels = json.loads(
            (label_root / case_id / "bend_instance_labels.json").read_text()
        )["manual_bends"]
        selected = prediction["owned_bend_regions"]
        pruning = prune_weak_parallel_outlines(selected)
        removals[case_id] = pruning["pruned_candidate_ids"]
        for aggregate, lines in ((before, selected), (after, pruning["selected"])):
            result = match_lines(labels, lines, Thresholds())
            aggregate["tp"] += result["true_positive"]
            aggregate["fp"] += result["false_positive"]
            aggregate["fn"] += result["false_negative"]
            aggregate["exact"] += int(result["exact"])

    assert {case_id: values for case_id, values in removals.items() if values} == {
        "04_48340008": ["scan_mesh_adjacency::SMA3"],
        "08_48333018": [
            "scan_mesh_adjacency::SMA4",
            "scan_mesh_adjacency::SMA1",
        ],
    }
    assert before == {"tp": 59, "fp": 10, "fn": 0, "exact": 5}
    assert after == {"tp": 59, "fp": 7, "fn": 0, "exact": 7}


def test_stored_shaked_checkpoints_never_prune_a_matched_line() -> None:
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
            pruning = prune_weak_parallel_outlines(selected)
            pruned_bend_ids = {
                item["bend_id"]
                for item in selected
                if item.get("candidate_id") in pruning["pruned_candidate_ids"]
            }
            assert pruned_bend_ids <= false_ids
            inspected += len(pruning["pruned_candidate_ids"])

    assert inspected >= 3


def test_is_inert_on_original13_v19_predictions() -> None:
    prediction_root = ROOT / (
        "output/experiments/f1-10of13/"
        "grouped-holdout-selector-v19-mesh-v4-tight"
    )
    if not prediction_root.exists():
        return

    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        selected = json.loads(path.read_text())["owned_bend_regions"]
        assert prune_weak_parallel_outlines(selected)["pruned_candidate_ids"] == []
