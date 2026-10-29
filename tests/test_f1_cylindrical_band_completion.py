from __future__ import annotations

import numpy as np

from backend.f1_cylindrical_band_completion import complete_cylindrical_bands


def _partial_cylinder(*, x0: float, x1: float, center_y: float, radius: float = 2.0):
    xs = np.linspace(x0, x1, 121)
    angles = np.radians(np.linspace(-48.0, 48.0, 31))
    x, angle = np.meshgrid(xs, angles, indexing="ij")
    points = np.column_stack(
        (x.ravel(), center_y + radius * np.cos(angle).ravel(), radius * np.sin(angle).ravel())
    )
    normals = np.column_stack(
        (np.zeros(angle.size), np.cos(angle).ravel(), np.sin(angle).ravel())
    )
    return points, normals


def test_completion_splits_two_finite_collinear_bands_deterministically():
    first = _partial_cylinder(x0=-50.0, x1=-10.0, center_y=0.0)
    second = _partial_cylinder(x0=15.0, x1=45.0, center_y=0.0)
    points = np.vstack((first[0], second[0]))
    normals = np.vstack((first[1], second[1]))
    seed = {"bend_id": "B1", "anchor": [0.0, 0.0, 0.0], "axis_direction": [1.0, 0.0, 0.0], "span_mm": 110.0}

    left = complete_cylindrical_bands(points, normals, [seed], point_budget=5000)
    right = complete_cylindrical_bands(points, normals, [seed], point_budget=5000)

    assert left == right
    candidates = left["candidates"]
    assert any(item["span_endpoints"][1][0] < 0.0 and item["span_mm"] >= 35.0 for item in candidates)
    assert any(item["span_endpoints"][0][0] > 0.0 and item["span_mm"] >= 25.0 for item in candidates)


def test_completion_enforces_point_and_candidate_budgets():
    points, normals = _partial_cylinder(x0=-100.0, x1=100.0, center_y=0.0)
    seed = {"bend_id": "B1", "anchor": [0.0, 0.0, 0.0], "axis_direction": [1.0, 0.0, 0.0], "span_mm": 210.0}
    result = complete_cylindrical_bands(
        points,
        normals,
        [seed],
        point_budget=1000,
        candidate_budget=3,
    )
    assert result["sampled_point_count"] == 1000
    assert result["selected_candidate_count"] <= 3


def test_transient_support_sink_does_not_change_public_result_or_serialize_indices():
    points, normals = _partial_cylinder(x0=-50.0, x1=50.0, center_y=0.0)
    seed = {
        "bend_id": "B1",
        "anchor": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 110.0,
    }
    support: dict[str, tuple[int, ...]] = {}
    without_sink = complete_cylindrical_bands(points, normals, [seed], point_budget=5000)
    with_sink = complete_cylindrical_bands(
        points,
        normals,
        [seed],
        point_budget=5000,
        transient_support_sink=support,
    )

    assert with_sink == without_sink
    assert set(support) == {row["candidate_id"] for row in with_sink["candidates"]}
    assert support
    for values in support.values():
        assert values == tuple(sorted(set(values)))
        assert values
        assert min(values) >= 0
        assert max(values) < len(points)
    assert "support_indices" not in repr(with_sink)
