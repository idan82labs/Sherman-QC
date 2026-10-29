from __future__ import annotations

import json

import numpy as np
import pytest

import backend.f1_localized_turn_corridor_ownership_v63 as v63


def _profile(*, transition_width: float, turn_degrees: float = 90.0):
    u = np.arange(-30.0, 30.01, 2.0, dtype=np.float64)
    theta = np.radians(turn_degrees) * 0.5 * (1.0 + np.tanh(u / transition_width))
    normals = np.column_stack((np.zeros_like(u), np.cos(theta), np.sin(theta)))
    return u, normals


def _prediction(index: int, route: str) -> dict:
    return {
        "bend_id": f"B{index:03d}",
        "line_start": [0.0, float(index), 0.0],
        "line_end": [20.0, float(index), 0.0],
        "selection_route": route,
        "source_hypothesis_id": f"H{index:03d}",
    }


def _bounded_grid() -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.arange(7.0), np.arange(7.0), indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    return points, normals


def test_localized_turn_passes_and_broad_roll_fails():
    localized_u, localized_normals = _profile(transition_width=2.0)
    localized, artifact = v63.evaluate_station_profile(
        localized_u,
        localized_normals,
        np.asarray([1.0, 0.0, 0.0]),
        candidate_turn=np.radians(90.0),
        profile_half_extent=30.0,
        spacing=1.0,
    )
    broad_u, broad_normals = _profile(transition_width=10.0)
    broad, _ = v63.evaluate_station_profile(
        broad_u,
        broad_normals,
        np.asarray([1.0, 0.0, 0.0]),
        candidate_turn=np.radians(90.0),
        profile_half_extent=30.0,
        spacing=1.0,
    )

    assert localized["passed"] is True
    assert localized["M_f"] == pytest.approx(1.0)
    assert artifact.fine_interval is not None
    assert broad["passed"] is False


def test_nearby_second_turn_does_not_merge_with_centered_lobe():
    u = np.arange(-30.0, 30.01, 2.0, dtype=np.float64)
    theta = np.radians(60.0) * 0.5 * (1.0 + np.tanh(u / 1.5))
    theta += np.radians(35.0) * 0.5 * (1.0 + np.tanh((u - 18.0) / 1.5))
    normals = np.column_stack((np.zeros_like(u), np.cos(theta), np.sin(theta)))
    record, artifact = v63.evaluate_station_profile(
        u,
        normals,
        np.asarray([1.0, 0.0, 0.0]),
        candidate_turn=np.radians(60.0),
        profile_half_extent=30.0,
        spacing=1.0,
    )

    assert record["passed"] is True
    assert artifact.fine_interval is not None
    assert artifact.fine_interval[1] < 12.0


def test_profile_contract_fails_closed_on_nonfinite_or_nonmonotone_input():
    u, normals = _profile(transition_width=2.0)
    bad_normals = normals.copy()
    bad_normals[3, 0] = np.nan
    with pytest.raises(v63._StationUnusable, match="NONFINITE_OR_INVALID_PROFILE"):
        v63.evaluate_station_profile(
            u,
            bad_normals,
            np.asarray([1.0, 0.0, 0.0]),
            candidate_turn=np.radians(90.0),
            profile_half_extent=30.0,
            spacing=1.0,
        )
    with pytest.raises(v63._StationUnusable, match="NONMONOTONE_PROFILE_ARCLENGTH"):
        v63.evaluate_station_profile(
            u[::-1],
            normals,
            np.asarray([1.0, 0.0, 0.0]),
            candidate_turn=np.radians(90.0),
            profile_half_extent=30.0,
            spacing=1.0,
        )


def test_spacing_is_deterministic_under_exact_distance_ties():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert v63.estimate_scan_spacing(points) == v63.estimate_scan_spacing(points)
    assert v63.estimate_scan_spacing(points) > 0.0


def test_apply_is_deletion_only_preserves_non_v52_and_bounds_public_audit(monkeypatch):
    points, normals = _bounded_grid()
    rows = [
        _prediction(1, "v16_route"),
        _prediction(2, v63.V52_ROUTE),
        _prediction(3, v63.V52_ROUTE),
        _prediction(4, "another_route"),
    ]

    def fake_audit(prediction, *args, **kwargs):
        passed = prediction["bend_id"] == "B002"
        scalar = {
            "candidate_id": prediction["source_hypothesis_id"],
            "bend_id": prediction["bend_id"],
            "decision": "PASS" if passed else "REJECT",
            "reason": "PASS" if passed else "TEST_REJECT",
            "valid_mask": 0,
            "pass_mask": 0,
            "passing_station_count": 0,
            "R_j": [None] * 9,
            "w_f": [None] * 9,
            "w_c": [None] * 9,
            "mu_f": [None] * 9,
            "mu_c": [None] * 9,
            "T_f": [None] * 9,
            "T_c": [None] * 9,
            "M_f": [None] * 9,
            "M_c": [None] * 9,
            "reasons": ["TEST"] * 9,
        }
        scalar["serialized_bytes"] = 0
        while True:
            size = len(v63._canonical_json(scalar))
            if scalar["serialized_bytes"] == size:
                break
            scalar["serialized_bytes"] = size
        return scalar, []

    monkeypatch.setattr(v63, "_candidate_audit", fake_audit)
    result, artifacts = v63.apply_v63(
        {"source_only": True, "research_only": True, "predictions": rows},
        {"line_objects": [], "candidates": []},
        object(),
        points,
        normals,
        {},
    )

    assert artifacts == []
    assert [row["bend_id"] for row in result["predictions"]] == ["B001", "B002", "B004"]
    assert result["predictions"][0] == rows[0]
    assert result["predictions"][-1] == rows[-1]
    assert result["audit"]["second_nms_used"] is False
    for scalar in result["audit"]["candidate_audits"]:
        assert scalar["serialized_bytes"] == len(v63._canonical_json(scalar))
        assert scalar["serialized_bytes"] <= v63.MAXIMUM_PUBLIC_AUDIT_BYTES_PER_CANDIDATE
    encoded = json.dumps(result, sort_keys=True)
    assert "support_indices" not in encoded
    assert "source_points" not in encoded


def test_apply_rejects_more_than_frozen_v52_capacity():
    points, normals = _bounded_grid()
    rows = [_prediction(index, v63.V52_ROUTE) for index in range(33)]
    with pytest.raises(v63.V63CapacityFail, match="CAPACITY_FAIL"):
        v63.apply_v63(
            {"source_only": True, "research_only": True, "predictions": rows},
            {"line_objects": [], "candidates": []},
            object(),
            points,
            normals,
            {},
        )
