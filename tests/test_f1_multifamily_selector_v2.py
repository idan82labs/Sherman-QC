from __future__ import annotations

import numpy as np

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, endpoints, select_indices


def row(candidate_id: str, offset: float, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "anchor": [0.0, offset, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 100.0,
        "representative_source": "cylindrical_standard",
        "case_percentile_score": score,
    }


def test_feature_schema_and_finite_values() -> None:
    matrix = build_feature_matrix([row("a", 0.0, 0.9), row("b", 1.0, 0.8)])
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_geometric_nms_keeps_best_alias() -> None:
    rows = [row("a", 0.0, 0.9), row("b", 1.0, 0.8), row("c", 20.0, 0.9)]
    assert select_indices(rows, np.asarray([0.91, 0.90, 0.89])) == [0, 2]


def test_endpoints_are_derived_without_serialized_span_endpoints() -> None:
    line = endpoints(row("a", 0.0, 0.9))
    assert np.allclose(line, [[-50.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
