from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.apply_f1_v51_today_axis_span_recovery_v1 import (
    FEATURE_NAMES,
    MODEL_KIND,
    compose_axis_span_recovery,
    generate_axis_span_variants,
)


def _candidate(token: str = "C1", *, y: float = 0.0) -> dict:
    return {
        "candidate_id": token,
        "anchor": [5.0, y, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 10.0,
        "span_endpoints": [[0.0, y, 0.0], [10.0, y, 0.0]],
        "candidate_source": "cross_source_line_object",
        "representative_source": "plane_lattice",
    }


def _pool(candidates: list[dict]) -> dict:
    return {
        "working_point_budget": 200000,
        "line_objects": candidates,
        "candidates": candidates,
    }


def _model(*, probability: float = 1.0, cap: int = 1) -> dict:
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "feature_names": list(FEATURE_NAMES),
        "tree": {
            "children_left": [-1],
            "children_right": [-1],
            "feature_index": [-2],
            "threshold": [-2.0],
            "positive_probability": [probability],
        },
        "profile": {
            "probability_threshold": 1.0,
            "recovery_cap": cap,
            "duplicate_angle_deg": 5.0,
            "duplicate_offset_mm": 4.0,
            "duplicate_span_overlap": 0.60,
            "span_scales": [],
            "fixed_spans_mm": [20.0],
        },
    }


def _source(kind: str) -> dict:
    return {
        "kind": kind,
        "predictions": [],
        "source_only": True,
        "research_only": True,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
    }


def test_generates_center_and_both_endpoint_anchored_fixed_span_variants() -> None:
    rows = generate_axis_span_variants(
        _pool([_candidate()]),
        span_scales=(1.0,),
        fixed_spans_mm=(20.0, 30.0),
    )
    assert len(rows) == 9
    modes = {row["axis_span_variant"]["anchor_mode"] for row in rows}
    spans = {row["axis_span_variant"]["variant_span_mm"] for row in rows}
    assert modes == {"center", "low_endpoint", "high_endpoint"}
    assert spans == {10.0, 20.0, 30.0}
    low = next(
        row
        for row in rows
        if row["axis_span_variant"]["anchor_mode"] == "low_endpoint"
        and row["axis_span_variant"]["variant_span_mm"] == 20.0
    )
    assert low["line_start"] == [0.0, 0.0, 0.0]
    assert low["line_end"] == [20.0, 0.0, 0.0]


def test_generation_is_independent_of_pool_order_and_axis_sign() -> None:
    first = _candidate("A", y=0.0)
    second = _candidate("B", y=10.0)
    forward = generate_axis_span_variants(
        _pool([first, second]), span_scales=(), fixed_spans_mm=(20.0,)
    )
    reversed_axis = deepcopy(first)
    reversed_axis["axis_direction"] = [-1.0, 0.0, 0.0]
    reversed_axis["span_endpoints"] = list(reversed(reversed_axis["span_endpoints"]))
    reordered = generate_axis_span_variants(
        _pool([second, reversed_axis]), span_scales=(), fixed_spans_mm=(20.0,)
    )
    assert forward == reordered


def test_compose_emits_at_most_one_variant_per_source_hypothesis() -> None:
    v51 = _source("v51-test")
    recall = _source("recall-test")
    result = compose_axis_span_recovery(v51, recall, _pool([_candidate()]), _model())
    assert len(result["predictions"]) == 1
    assert result["axis_span_audit"]["generated_variant_count"] == 3
    assert result["axis_span_audit"]["emitted_variant_count"] == 1
    assert result["labels_used"] is False
    assert result["promotion_eligible"] is False
    assert result["generalization_evidence"] is False


@pytest.mark.parametrize("flag", ["labels_used", "cad_used", "part_identity_used"])
def test_compose_rejects_forbidden_runtime_inputs(flag: str) -> None:
    v51 = _source("v51-test")
    v51[flag] = True
    with pytest.raises(ValueError, match="forbidden inference source"):
        compose_axis_span_recovery(
            v51,
            _source("recall-test"),
            _pool([_candidate()]),
            _model(),
        )
