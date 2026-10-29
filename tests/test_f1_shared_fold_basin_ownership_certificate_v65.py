from __future__ import annotations

import math

import numpy as np
import pytest

import backend.f1_shared_fold_basin_ownership_certificate_v65 as v65


def _extrude(profile: np.ndarray, *, length: int = 80) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[float(x), float(point[0]), float(point[1])] for x in range(length + 1) for point in profile],
        dtype=np.float64,
    )
    width = len(profile)
    faces = []
    for x in range(length):
        for j in range(width - 1):
            a = x * width + j
            b = (x + 1) * width + j
            faces.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _polyline(turns: list[tuple[int, float]], *, before: int = 20, after: int = 20) -> tuple[np.ndarray, list[int]]:
    points = [np.asarray([-float(before), 0.0])]
    heading = 0.0
    bend_indices = []
    first = True
    for distance, next_heading in turns:
        steps = distance + (before if first else 0)
        first = False
        for _ in range(steps):
            angle = math.radians(heading)
            points.append(points[-1] + (math.cos(angle), math.sin(angle)))
        bend_indices.append(len(points) - 1)
        heading = next_heading
    for _ in range(after):
        angle = math.radians(heading)
        points.append(points[-1] + (math.cos(angle), math.sin(angle)))
    return np.asarray(points, dtype=np.float64), bend_indices


def _row(index: int, point: np.ndarray, *, start: float = 0.0, end: float = 80.0) -> dict:
    return {
        "bend_id": f"SF-{index}",
        "line_start": [start, float(point[0]), float(point[1])],
        "line_end": [end, float(point[0]), float(point[1])],
        "selection_route": "fixture",
    }


def _baseline(points: list[np.ndarray], *, starts: list[float] | None = None, ends: list[float] | None = None) -> dict:
    starts = starts or [0.0] * len(points)
    ends = ends or [80.0] * len(points)
    return {
        "source_only": True,
        "research_only": True,
        "predictions": [_row(i, point, start=starts[i], end=ends[i]) for i, point in enumerate(points)],
    }


def _point_at_arc(profile: np.ndarray, bend_index: int, offset: int) -> np.ndarray:
    return profile[bend_index + offset]


def test_equal_area_70_degree_dihedral_confidence_is_below_locked_threshold():
    first = np.asarray([0.0, 0.0, 1.0])
    angle = math.radians(70.0)
    second = np.asarray([0.0, -math.sin(angle), math.cos(angle)])
    projector = 0.5 * (np.outer(first, first) + np.outer(second, second))
    values, _vectors = v65._jacobi_eigh3(projector)

    assert float(values[-1]) == pytest.approx((1.0 + math.cos(angle)) / 2.0, abs=1e-12)
    assert float(values[-1]) < 0.85


@pytest.mark.xfail(
    strict=True,
    reason="Locked SF01 is incompatible with the locked >=0.85 averaged-projector confidence at an equal-area 70-degree dihedral",
)
def test_sf01_sharp_one_fold_three_routes():
    profile, bends = _polyline([(0, 70.0)])
    vertices, faces = _extrude(profile)
    points = [_point_at_arc(profile, bends[0], offset) for offset in (-3, 1, 4)]
    result, pairs = v65.apply_sfboc_v1(
        _baseline(points), vertices, faces, ordinary_nms_pair=lambda _a, _b: False
    )

    assert len(pairs) == 3
    assert all(pair.positive for pair in pairs)
    assert all(sum(state == v65.SAME_BASIN for state in pair.states) >= 8 for pair in pairs)
    assert len(result["predictions"]) == 1
    assert len(result["sfboc_v1_audit"]["suppressed_hashes"]) == 2
