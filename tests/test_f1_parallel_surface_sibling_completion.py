import inspect
import json
from pathlib import Path

from backend.f1_parallel_surface_sibling_completion import (
    complete_parallel_surface_siblings,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[1]


def _edge(candidate_id, first, second, x, *, bend_angle, **extra):
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": first,
        "surface2_id": second,
        "bend_angle": bend_angle,
        "span_endpoints": [[x, 0, 0], [x, 100, 0]],
        "finite_support_fraction": 1.0,
        "finite_support_longest_run_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "median_line_distance_mm": 0.5,
        "p90_line_distance_mm": 1.0,
        "two_sided_normal_angle_deg": 85.0,
        "two_sided_normal_coherence": 0.85,
        "min_surface_gap": 1.0,
        **extra,
    }


def test_completes_disconnected_surface_component_bridge():
    selected = [
        _edge("right-obtuse", 20, 30, 0.0, bend_angle=135.0),
        _edge("left-branch", 10, 40, 20.0, bend_angle=90.0),
    ]
    candidate = _edge("bridge", 10, 20, 1.25, bend_angle=90.0)

    result = complete_parallel_surface_siblings(selected, [candidate])

    assert [item["candidate_id"] for item in result["added"]] == ["bridge"]
    assert result["added"][0]["topology"] == "disconnected_surface_component_bridge"
    assert result["selected"][-1]["selection_reason"] == (
        "parallel_surface_sibling_completion"
    )


def test_completes_leaf_only_on_explicit_high_degree_fold_hub():
    selected = [
        _edge("witness", 1, 2, 0.0, bend_angle=90.0),
        *[
            _edge(f"branch-{surface}", 1, surface, 20.0 + surface, bend_angle=90.0)
            for surface in (3, 4, 5, 6)
        ],
    ]
    leaf = _edge("leaf", 1, 7, 1.25, bend_angle=135.0)
    underconnected = selected[:4]

    accepted = complete_parallel_surface_siblings(selected, [leaf])
    rejected = complete_parallel_surface_siblings(underconnected, [leaf])

    assert accepted["added"][0]["topology"] == "high_degree_fold_hub_leaf"
    assert rejected["added"] == []


def test_rejects_same_axis_weak_finite_line_and_noncomplementary_outline():
    selected = [
        _edge("right-obtuse", 20, 30, 0.0, bend_angle=135.0),
        _edge("left-branch", 10, 40, 20.0, bend_angle=90.0),
    ]
    same_axis = _edge("same-axis", 10, 20, 0.1, bend_angle=90.0)
    weak = _edge(
        "weak", 10, 20, 1.25, bend_angle=90.0, finite_support_fraction=0.8
    )
    outline = _edge("outline", 10, 20, 1.25, bend_angle=125.0)

    result = complete_parallel_surface_siblings(
        selected, [same_axis, weak, outline]
    )

    assert result["added"] == []


def test_is_identity_and_count_free_and_deterministic():
    signature = inspect.signature(complete_parallel_surface_siblings)
    assert all(
        forbidden not in name
        for name in signature.parameters
        for forbidden in ("case", "part", "expected", "target", "bend_count", "truth")
    )
    selected = [
        _edge("right-obtuse", 20, 30, 0.0, bend_angle=135.0),
        _edge("left-branch", 10, 40, 20.0, bend_angle=90.0),
    ]
    candidate = _edge("bridge", 10, 20, 1.25, bend_angle=90.0)

    forward = complete_parallel_surface_siblings(selected, [candidate])
    reverse = complete_parallel_surface_siblings(list(reversed(selected)), [candidate])

    assert forward["added"] == reverse["added"]


def test_v13_full_composition_adds_only_two_true_siblings_without_regression():
    """Exercise the persisted selector + short + snap + wedge + lane result."""

    prediction_root = ROOT / (
        "output/experiments/"
        "shaked-mesh-nested-lopo-v13-surface-wedge-completion/predictions"
    )
    candidate_root = ROOT / (
        "output/manual_labels/shaked-count-backed-12-20260717/"
        "candidate_pool_mesh_v4"
    )
    label_root = ROOT / (
        "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
    )
    if not prediction_root.exists():
        return

    before = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    after = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    additions = {}
    case_results = {}
    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        case_id = path.parent.name
        prediction = json.loads(path.read_text())
        candidate_payload = json.loads((candidate_root / f"{case_id}.json").read_text())
        labels = json.loads(
            (label_root / case_id / "bend_instance_labels.json").read_text()
        )["manual_bends"]
        selected = prediction["owned_bend_regions"]
        completion = complete_parallel_surface_siblings(
            selected, candidate_payload["candidates"]
        )
        additions[case_id] = [item["candidate_id"] for item in completion["added"]]
        case_results[case_id] = []
        for aggregate, lines in ((before, selected), (after, completion["selected"])):
            result = match_lines(labels, lines, Thresholds())
            case_results[case_id].append(result)
            aggregate["tp"] += result["true_positive"]
            aggregate["fp"] += result["false_positive"]
            aggregate["fn"] += result["false_negative"]
            aggregate["exact"] += int(result["exact"])

    assert {case_id: values for case_id, values in additions.items() if values} == {
        "07_49191038": ["scan_mesh_adjacency::SMA1"],
        "09_49191037": ["scan_mesh_adjacency::SMA3"],
    }
    assert before == {"tp": 55, "fp": 11, "fn": 4, "exact": 4}
    assert after == {"tp": 57, "fp": 11, "fn": 2, "exact": 5}
    for baseline, completed in case_results.values():
        assert completed["false_positive"] <= baseline["false_positive"]
        assert not baseline["exact"] or completed["exact"]
