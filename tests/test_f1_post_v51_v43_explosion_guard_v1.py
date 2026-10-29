from __future__ import annotations

from copy import deepcopy

import pytest

from backend.f1_post_v51_v43_explosion_guard_v1 import (
    MAX_SAFE_V43_POST_MERGE_ADDITIONS,
    canonical_bytes,
    compose_v43_explosion_guard,
)


def _prediction(name: str, y: float) -> dict:
    return {
        "bend_id": name,
        "line_start": [0.0, y, 0.0],
        "line_end": [10.0, y, 0.0],
    }


def _stage(predictions: list[dict], **extra: object) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "part_identity_used": False,
        "cad_used": False,
        "predictions": predictions,
        **extra,
    }


def _inputs(additions: int) -> tuple[dict, dict, dict]:
    v16 = _stage([_prediction("V16-A", 0.0), _prediction("V16-B", 10.0)])
    v43_predictions = deepcopy(v16["predictions"]) + [
        _prediction(f"V43-{index}", 20.0 + index * 10.0)
        for index in range(additions)
    ]
    v43 = _stage(
        v43_predictions,
        retained_v16_prediction_count=2,
        v43_post_merge_addition_count=additions,
        candidate_count=2 + additions,
    )
    v51 = _stage(deepcopy(v43_predictions))
    return v16, v43, v51


def test_multi_addition_explosion_falls_back_to_v16_without_mutation() -> None:
    v16, v43, v51 = _inputs(MAX_SAFE_V43_POST_MERGE_ADDITIONS + 1)
    snapshot = deepcopy((v16, v43, v51))
    result = compose_v43_explosion_guard(v16, v43, v51)
    assert result["predictions"] == v16["predictions"]
    assert result["guard_audit"]["guard_triggered"] is True
    assert result["guard_audit"]["selection"] == "FALL_BACK_TO_V16"
    assert (v16, v43, v51) == snapshot


def test_zero_or_one_addition_retains_v51() -> None:
    for additions in (0, MAX_SAFE_V43_POST_MERGE_ADDITIONS):
        v16, v43, v51 = _inputs(additions)
        result = compose_v43_explosion_guard(v16, v43, v51)
        assert result["predictions"] == v51["predictions"]
        assert result["guard_audit"]["guard_triggered"] is False
        assert result["guard_audit"]["selection"] == "RETAIN_V51"


def test_fail_closed_abstention_is_preserved_without_partial_stages() -> None:
    v51 = {
        "schema_version": 1,
        "kind": "sherman_qc_v51_fail_closed_case_abstention",
        "pipeline_status": "FAILED_ABSTAIN",
        "source_only": True,
        "research_only": True,
        "predictions": [],
    }
    result = compose_v43_explosion_guard(None, None, v51)
    assert result["predictions"] == []
    assert result["guard_audit"]["selection"] == "PRESERVE_FAIL_CLOSED_V51_ABSTENTION"


def test_forbidden_inference_input_is_rejected() -> None:
    v16, v43, v51 = _inputs(2)
    v51["labels_used"] = True
    with pytest.raises(ValueError, match="forbidden inference input"):
        compose_v43_explosion_guard(v16, v43, v51)


def test_inconsistent_v43_accounting_is_rejected() -> None:
    v16, v43, v51 = _inputs(2)
    v43["candidate_count"] = 99
    with pytest.raises(ValueError, match="merge accounting"):
        compose_v43_explosion_guard(v16, v43, v51)


def test_repeated_composition_is_byte_identical() -> None:
    v16, v43, v51 = _inputs(3)
    first = compose_v43_explosion_guard(v16, v43, v51)
    second = compose_v43_explosion_guard(v16, v43, v51)
    assert canonical_bytes(first) == canonical_bytes(second)
    assert first["result_sha256"] == second["result_sha256"]

