from __future__ import annotations

import inspect
import json
from pathlib import Path

from backend.f1_scan_supported_short_seed_completion import (
    complete_scan_supported_short_seeds,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[1]


def _line(candidate_id="cylindrical_standard::fit", **extra):
    return {
        "candidate_id": candidate_id,
        "bend_id": "line-object",
        "span_endpoints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "span_mm": 10.0,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "cluster_sources": ["bend_hypothesis", "cylindrical_standard"],
        "cluster_member_ids": [
            "bend_hypothesis::seed",
            "cylindrical_standard::fit",
        ],
        **extra,
    }


def _seed(**extra):
    return {
        "candidate_id": "bend_hypothesis::seed",
        "source_bend_ids": ["detector-owned"],
        "finite_support_fraction": 0.92,
        "endpoint_support_fraction": 2.0 / 3.0,
        **extra,
    }


def test_adds_fully_supported_short_refit_of_supported_detector_seed():
    result = complete_scan_supported_short_seeds([], [_line()], [_seed()])

    assert [item["line_object_id"] for item in result["added"]] == ["line-object"]
    assert result["selected"][0]["selection_reason"] == (
        "scan_supported_short_seed_completion"
    )


def test_rejects_interface_birth_fragmented_seed_weak_refit_and_long_line():
    interface_birth = _seed(source_bend_ids=[])
    fragmented_seed = _seed(
        finite_support_fraction=0.89, endpoint_support_fraction=1.0
    )
    weak_refit = _line(finite_support_fraction=0.98)
    long_line = _line(span_mm=15.01)

    assert complete_scan_supported_short_seeds([], [_line()], [interface_birth])["added"] == []
    assert complete_scan_supported_short_seeds([], [_line()], [fragmented_seed])["added"] == []
    assert complete_scan_supported_short_seeds([], [weak_refit], [_seed()])["added"] == []
    assert complete_scan_supported_short_seeds([], [long_line], [_seed()])["added"] == []


def test_rejects_geometric_duplicate_and_is_deterministic_and_count_free():
    selected = [
        {
            "candidate_id": "existing",
            "span_endpoints": [[0.0, 0.5, 0.0], [10.0, 0.5, 0.0]],
        }
    ]
    forward = complete_scan_supported_short_seeds(selected, [_line()], [_seed()])
    reverse = complete_scan_supported_short_seeds(selected, list(reversed([_line()])), [_seed()])

    assert forward == reverse
    assert forward["added"] == []
    signature = inspect.signature(complete_scan_supported_short_seeds)
    assert all(
        forbidden not in name
        for name in signature.parameters
        for forbidden in ("case", "part", "expected", "target", "bend_count", "truth")
    )


def test_v15_composition_recovers_only_case05_misses_without_any_regression():
    prediction_root = ROOT / (
        "output/experiments/"
        "shaked-mesh-nested-lopo-v15-contained-fragment-pruning/predictions"
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
    for path in sorted(prediction_root.glob("*/decomposition_result.json")):
        case_id = path.parent.name
        prediction = json.loads(path.read_text())
        candidate_payload = json.loads((candidate_root / f"{case_id}.json").read_text())
        labels = json.loads(
            (label_root / case_id / "bend_instance_labels.json").read_text()
        )["manual_bends"]
        selected = prediction["owned_bend_regions"]
        completion = complete_scan_supported_short_seeds(
            selected,
            candidate_payload["line_objects"],
            candidate_payload["candidates"],
        )
        additions[case_id] = [item["line_object_id"] for item in completion["added"]]
        baseline = match_lines(labels, selected, Thresholds())
        completed = match_lines(labels, completion["selected"], Thresholds())
        for aggregate, result in ((before, baseline), (after, completed)):
            aggregate["tp"] += result["true_positive"]
            aggregate["fp"] += result["false_positive"]
            aggregate["fn"] += result["false_negative"]
            aggregate["exact"] += int(result["exact"])
        if case_id != "05_49191032":
            assert completed == baseline
        assert completed["false_positive"] == baseline["false_positive"]
        assert not baseline["exact"] or completed["exact"]

    assert {case_id: values for case_id, values in additions.items() if values} == {
        "05_49191032": ["LO100", "LO95"]
    }
    assert before == {"tp": 57, "fp": 10, "fn": 2, "exact": 5}
    assert after == {"tp": 59, "fp": 10, "fn": 0, "exact": 5}


def test_rule_is_inert_on_the_independent_original13_candidate_pool():
    candidate_root = ROOT / (
        "output/experiments/f1-10of13/strict-blind-uniform-candidate-pool-mesh-v4"
    )
    if not candidate_root.exists():
        return
    additions = {}
    for path in sorted(candidate_root.glob("[0-9][0-9]_*.json")):
        payload = json.loads(path.read_text())
        result = complete_scan_supported_short_seeds(
            [], payload["line_objects"], payload["candidates"]
        )
        additions[path.stem] = result["added"]
    assert all(not values for values in additions.values())
