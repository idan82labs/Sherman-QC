from __future__ import annotations

import math

from backend.f1_mesh_interface_geometry import (
    log_minimum_surface_area_span_ratio,
)


def test_interface_area_span_ratio_is_scale_invariant() -> None:
    candidate = {"surface1_area": 480.0, "surface2_area": 120.0, "span_mm": 40.0}
    scaled = {
        "surface1_area": 480.0 * 25.0,
        "surface2_area": 120.0 * 25.0,
        "span_mm": 40.0 * 5.0,
    }

    assert math.isclose(
        log_minimum_surface_area_span_ratio(candidate),
        log_minimum_surface_area_span_ratio(scaled),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_small_segmented_patch_has_less_interface_extent_evidence() -> None:
    forming_interface = {
        "surface1_area": 480.0,
        "surface2_area": 170.0,
        "span_mm": 40.0,
    }
    segmented_boundary = {
        "surface1_area": 480.0,
        "surface2_area": 50.0,
        "span_mm": 40.0,
    }

    assert log_minimum_surface_area_span_ratio(
        forming_interface
    ) > log_minimum_surface_area_span_ratio(segmented_boundary)


def test_invalid_geometry_fails_closed_to_finite_zero() -> None:
    invalid_candidates = (
        {},
        {"surface1_area": 10.0, "surface2_area": 8.0, "span_mm": 0.0},
        {"surface1_area": -10.0, "surface2_area": 8.0, "span_mm": 4.0},
        {"surface1_area": float("nan"), "surface2_area": 8.0, "span_mm": 4.0},
        {"surface1_area": "bad", "surface2_area": 8.0, "span_mm": 4.0},
    )

    assert [log_minimum_surface_area_span_ratio(item) for item in invalid_candidates] == [
        0.0
    ] * len(invalid_candidates)
