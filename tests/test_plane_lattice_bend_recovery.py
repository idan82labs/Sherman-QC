import numpy as np

from backend.plane_lattice_bend_recovery import (
    build_supported_plane_intersections,
    extract_deterministic_planes,
    merge_near_duplicate_planes,
)


def _grid_on_xy(z: float, size: int = 21) -> np.ndarray:
    axis = np.linspace(-20.0, 20.0, size)
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _grid_on_xz(y: float, size: int = 21) -> np.ndarray:
    axis = np.linspace(-20.0, 20.0, size)
    x, z = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), np.full(x.size, y), z.ravel()))


def _grid_on_yz(x: float, size: int = 21) -> np.ndarray:
    axis = np.linspace(-20.0, 20.0, size)
    y, z = np.meshgrid(axis, axis)
    return np.column_stack((np.full(y.size, x), y.ravel(), z.ravel()))


def test_near_duplicate_planes_merge_by_geometry_and_support():
    merged = merge_near_duplicate_planes(
        [
            {"plane_id": "thin", "normal": [0, 0, 1], "offset": -1.2, "support_count": 40},
            {"plane_id": "main", "normal": [0, 0, 1], "offset": 0.0, "support_count": 400},
            {"plane_id": "side", "normal": [0, 1, 0], "offset": 0.0, "support_count": 200},
        ]
    )

    assert len(merged) == 2
    main = next(item for item in merged if item["plane_id"] == "main")
    assert main["merged_plane_ids"] == ["main", "thin"]


def test_supported_plane_lattice_builds_three_distinct_corner_lines():
    points = np.vstack((_grid_on_xy(0.0), _grid_on_xz(0.0), _grid_on_yz(0.0)))
    result = build_supported_plane_intersections(
        points,
        [
            {"plane_id": "XY", "normal": [0, 0, 1], "offset": 0.0, "support_count": 441},
            {"plane_id": "XZ", "normal": [0, 1, 0], "offset": 0.0, "support_count": 441},
            {"plane_id": "YZ", "normal": [1, 0, 0], "offset": 0.0, "support_count": 441},
        ],
        corridor_radius_mm=4.1,
        min_support_points=20,
        min_span_mm=20.0,
    )

    assert result["candidate_count"] == 3
    assert {tuple(item["incident_plane_ids"]) for item in result["candidates"]} == {
        ("XY", "XZ"),
        ("XY", "YZ"),
        ("XZ", "YZ"),
    }
    assert all(item["span_mm"] >= 35.0 for item in result["candidates"])


def test_intersection_requires_overlapping_support_from_both_planes():
    xy = _grid_on_xy(0.0)
    xz = _grid_on_xz(30.0)
    result = build_supported_plane_intersections(
        np.vstack((xy, xz)),
        [
            {"plane_id": "XY", "normal": [0, 0, 1], "offset": 0.0, "support_count": len(xy)},
            {"plane_id": "XZ", "normal": [0, 1, 0], "offset": -30.0, "support_count": len(xz)},
        ],
        corridor_radius_mm=4.0,
        min_support_points=20,
    )

    assert result["candidate_count"] == 0


def test_collinear_multi_plane_intersections_collapse_to_one_marker():
    x = np.linspace(-20.0, 20.0, 31)
    y = np.linspace(-12.0, 12.0, 21)
    xx, yy = np.meshgrid(x, y)
    plane_a = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    plane_b = np.column_stack((xx.ravel(), np.zeros(xx.size), yy.ravel()))
    plane_c = np.column_stack((xx.ravel(), yy.ravel(), -yy.ravel()))
    result = build_supported_plane_intersections(
        np.vstack((plane_a, plane_b, plane_c)),
        [
            {"plane_id": "A", "normal": [0, 0, 1], "offset": 0.0, "support_count": len(plane_a)},
            {"plane_id": "B", "normal": [0, 1, 0], "offset": 0.0, "support_count": len(plane_b)},
            {
                "plane_id": "C",
                "normal": [0, 2 ** -0.5, 2 ** -0.5],
                "offset": 0.0,
                "support_count": len(plane_c),
            },
        ],
        corridor_radius_mm=5.0,
        min_support_points=20,
        min_span_mm=20.0,
        max_abs_normal_dot=0.8,
    )

    assert result["raw_candidate_count"] == 3
    assert result["candidate_count"] == 1
    assert len(result["candidates"][0]["merged_candidate_ids"]) == 3


def test_plane_extraction_is_deterministic_and_recovers_dominant_planes():
    points = np.vstack((_grid_on_xy(0.0, 17), _grid_on_xz(8.0, 13)))
    kwargs = {
        "distance_threshold_mm": 0.05,
        "max_planes": 2,
        "min_plane_points": 100,
        "seed": 19,
        "iterations": 120,
    }

    first = extract_deterministic_planes(points, **kwargs)
    second = extract_deterministic_planes(points[::-1], **kwargs)

    assert first == second
    assert len(first) == 2
    assert sorted(item["support_count"] for item in first) == [156, 302]
