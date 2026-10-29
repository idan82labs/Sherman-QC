from __future__ import annotations

import json

from backend.f1_deterministic_paired_return_completion_v45 import (
    compose_v45,
    incumbent_first_output_nms,
)


def _row(candidate_id, y, z, x0, x1, *, good=True):
    return {
        "candidate_id": candidate_id,
        "representative_source": "cylindrical_standard",
        "span_endpoints": [[x0, y, z], [x1, y, z]],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": x1 - x0,
        "finite_support_fraction": 1.0 if good else 0.2,
        "finite_support_longest_run_fraction": 1.0 if good else 0.2,
        "endpoint_support_fraction": 1.0 if good else 0.2,
        "two_sided_normal_angle_deg": 85.0,
        "two_sided_transition_score": 0.8,
    }


def _prediction(candidate_id, y, z, x0, x1):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": [x0, y, z],
        "line_end": [x1, y, z],
        "selection_route": "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair",
        "source_hypothesis_id": candidate_id,
        "source_kind": "legacy",
    }


def test_all_rows_permutation_is_byte_exact():
    selected = [
        _prediction("left_base", -190, 0, 0, 300),
        _prediction("left_return", -190, 16, 0, 280),
        _prediction("right_base", 190, 0, 0, 500),
    ]
    rows = [
        _row("f1", 190, -16, 0, 170),
        _row("f2", 191, -16, 180, 350),
        _row("f3", 192, -16, 340, 500),
    ]
    source = {
        "route": "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair",
        "predictions": selected,
    }
    first = compose_v45(source, rows, 500.0)
    second = compose_v45(source, list(reversed(rows)), 500.0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first["predictions"]) == 4


def test_incumbent_precedes_colliding_addition():
    incumbent = _row("incumbent", 0, 0, 0, 100)
    addition = _row("addition", 1, 0, 0, 100)
    accepted, suppressed = incumbent_first_output_nms([incumbent], [addition])
    assert [value["candidate_id"] for value in accepted] == ["incumbent"]
    assert suppressed == ["addition"]


def test_requires_three_lane_configuration():
    source = {
        "route": "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair",
        "predictions": [_prediction("only", 0, 0, 0, 100)],
    }
    result = compose_v45(source, [], 500.0)
    assert len(result["predictions"]) == 1
    assert result["paired_return_completion_changes"] == []
