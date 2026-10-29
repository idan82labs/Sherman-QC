from __future__ import annotations

from scripts.run_f1_v41_v16_real_scan_compatibility import (
    _line_metrics,
    adapt_candidates,
    incumbent_first_merge,
)


def _plane(plane_id: str, normal: list[float], offset: float) -> dict:
    return {
        "plane_id": plane_id,
        "normal": normal,
        "offset": offset,
        "tangent_basis": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "extent_uv": [[-2.0, 2.0], [-3.0, 3.0]],
    }


def test_adapter_exposes_only_rejected_plane_lattice_candidates() -> None:
    pool = {
        "transition_local_spacing_mm": 1.0,
        "plane_table": [
            _plane("A", [0.0, 0.0, 1.0], -1.0),
            _plane("B", [0.0, 1.0, 0.0], -2.0),
        ],
        "line_objects": [
            {
                "candidate_id": "plane_lattice::selected",
                "representative_source": "plane_lattice",
                "incident_plane_ids": ["A", "B"],
                "span_endpoints": [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                "angle_deg": 90.0,
            },
            {
                "candidate_id": "cylindrical::rejected",
                "representative_source": "cylindrical_standard",
            },
            {
                "candidate_id": "plane_lattice::rejected",
                "representative_source": "plane_lattice",
                "incident_plane_ids": ["A", "B"],
                "span_endpoints": [[0.0, 2.0, 0.0], [30.0, 2.0, 0.0]],
                "angle_deg": 90.0,
            },
        ],
    }
    retained = {
        "predictions": [{"source_hypothesis_id": "plane_lattice::selected"}]
    }
    candidates, mappings = adapt_candidates(pool, retained)
    assert len(candidates) == 1
    assert mappings[0]["source_hypothesis_id"] == "plane_lattice::rejected"
    assert candidates[0]["frozen_v16_order"] == 2
    assert candidates[0]["selected_legacy"] is False


def test_incumbent_first_merge_suppresses_duplicate_and_accepts_distinct() -> None:
    retained = [{
        "bend_id": "V16-A",
        "source_hypothesis_id": "incumbent",
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [100.0, 0.0, 0.0],
    }]
    additions = [
        {"candidate_id": 1, "line_start": [0.0, 2.0, 0.0], "line_end": [100.0, 2.0, 0.0]},
        {"candidate_id": 2, "line_start": [0.0, 20.0, 0.0], "line_end": [100.0, 20.0, 0.0]},
    ]
    mappings = {
        1: {"source_hypothesis_id": "duplicate"},
        2: {"source_hypothesis_id": "distinct"},
    }
    predictions, audit = incumbent_first_merge(retained, additions, mappings)
    assert [value["accepted"] for value in audit] == [False, True]
    assert audit[0]["duplicate_of"] == "incumbent"
    assert len(predictions) == 2
    assert predictions[-1]["source_hypothesis_id"] == "distinct"


def test_line_metrics_are_direction_invariant() -> None:
    forward = _line_metrics([0, 0, 0], [100, 0, 0], [0, 3, 0], [100, 3, 0])
    reversed_line = _line_metrics([100, 0, 0], [0, 0, 0], [100, 3, 0], [0, 3, 0])
    assert forward == reversed_line
    assert forward == (0.0, 3.0, 1.0)
