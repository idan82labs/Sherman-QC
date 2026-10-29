from __future__ import annotations

import json

from scripts.apply_f1_v51_source_only_recovery_v1 import (
    MODEL_KIND,
    apply_recovery,
    duplicate_lines,
    tree_probability,
)


def _profile() -> dict:
    return {
        "tree_probability_threshold": 0.9,
        "tree_recovery_cap": 3,
        "duplicate_angle_deg": 5.0,
        "duplicate_offset_mm": 4.0,
        "duplicate_span_overlap": 0.6,
        "adjacency_supplement": {
            "candidate_source": "scan_mesh_adjacency",
            "min_confidence": 0.98,
            "min_transition_score": 0.30,
            "max_transition_score": 0.50,
            "min_normal_second_eigenvalue": 0.31,
            "min_corridor_points_per_mm": 14.0,
            "emission_score": 1.001,
        },
        "narrow_replacement": {
            "min_tree_probability": 1.0,
            "recovery_source_prefixes": ["cylindrical_standard", "cylindrical_wide"],
            "incumbent_source_kind": "scan_mesh_adjacency",
            "max_incumbent_selector_score": 0.7,
        },
    }


def _model() -> dict:
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "development_tuned_model": True,
        "feature_names": ["signal"],
        "tree": {
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature_index": [0, -2, -2],
            "threshold": [0.5, -2.0, -2.0],
            "positive_probability": [0.5, 0.0, 1.0],
        },
        "profiles": {"balanced": _profile()},
    }


def _incumbent(predictions: list[dict] | None = None) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "predictions": predictions or [],
    }


def _candidate(
    candidate_id: str,
    y: float,
    *,
    signal: float = 1.0,
    representative_source: str = "cylindrical_standard",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "cross_source_line_object",
        "representative_source": representative_source,
        "span_endpoints": [[0.0, y, 0.0], [20.0, y, 0.0]],
        "signal": signal,
    }


def _pool(line_objects: list[dict], candidates: list[dict] | None = None) -> dict:
    return {
        "working_point_budget": 200000,
        "line_objects": line_objects,
        "candidates": candidates or [],
    }


def test_tree_probability_uses_frozen_arrays() -> None:
    model = _model()
    assert tree_probability({"signal": 0.5}, model) == 0.0
    assert tree_probability({"signal": 0.5001}, model) == 1.0
    assert tree_probability({}, model) == 0.0


def test_source_only_application_is_byte_deterministic_and_truth_free() -> None:
    model = _model()
    pool = _pool([_candidate("A", 0.0), _candidate("B", 12.0)])
    first = apply_recovery(_incumbent(), pool, model)
    second = apply_recovery(_incumbent(), pool, model)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["candidate_count"] == 2
    assert first["labels_used"] is False
    assert first["expected_count_used"] is False
    assert first["cad_used"] is False
    assert first["part_identity_used"] is False
    assert "truth" not in json.dumps(first).lower()
    assert "gold" not in json.dumps(first).lower()


def test_incumbent_owns_duplicate_by_default() -> None:
    incumbent_row = {
        "bend_id": "I1",
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [20.0, 0.0, 0.0],
        "source_kind": "plane_lattice",
        "selector_score": 0.1,
    }
    result = apply_recovery(
        _incumbent([incumbent_row]),
        _pool([_candidate("DUP", 1.0)]),
        _model(),
    )
    assert [row["bend_id"] for row in result["predictions"]] == ["I1"]
    assert result["recovery_audit"]["recovery_emitted_count"] == 0
    assert result["recovery_audit"]["rejection_audit"][0]["reason"] == "INCUMBENT_OWNS_DUPLICATE"


def test_narrow_rerank_replaces_only_low_score_adjacency_owner() -> None:
    incumbent_row = {
        "bend_id": "I1",
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [20.0, 0.0, 0.0],
        "source_kind": "scan_mesh_adjacency",
        "selector_score": 0.6,
    }
    result = apply_recovery(
        _incumbent([incumbent_row]),
        _pool([_candidate("BETTER", 1.0)]),
        _model(),
    )
    assert result["candidate_count"] == 1
    assert result["predictions"][0]["source_hypothesis_id"] == "BETTER"
    assert result["recovery_audit"]["replaced_incumbent_count"] == 1
    assert result["recovery_audit"]["replacement_audit"][0][
        "replaced_incumbent_bend_ids"
    ] == ["I1"]


def test_high_curvature_adjacency_supplement_is_source_feature_only() -> None:
    supplement = {
        "candidate_id": "SMA7",
        "candidate_source": "scan_mesh_adjacency",
        "span_endpoints": [[0.0, 30.0, 0.0], [20.0, 30.0, 0.0]],
        "confidence": 0.983,
        "two_sided_transition_score": 0.38,
        "normal_second_eigenvalue": 0.314,
        "corridor_points_per_mm": 15.2,
    }
    result = apply_recovery(_incumbent(), _pool([], [supplement]), _model())
    assert result["candidate_count"] == 1
    assert result["predictions"][0]["selection_route"].endswith("adjacency_supplement_v1")
    assert result["recovery_audit"]["adjacency_supplement_emitted_count"] == 1


def test_duplicate_geometry_is_symmetric_for_equal_spans() -> None:
    import numpy as np

    first = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    second = np.asarray([[0.0, 3.0, 0.0], [20.0, 3.0, 0.0]])
    assert duplicate_lines(first, second, angle_deg=5.0, offset_mm=4.0, span_overlap=0.6)
    assert duplicate_lines(second, first, angle_deg=5.0, offset_mm=4.0, span_overlap=0.6)
