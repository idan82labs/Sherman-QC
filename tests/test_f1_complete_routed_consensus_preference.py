from __future__ import annotations

from copy import deepcopy
from math import cos, radians, sin

import pytest

from backend.f1_complete_routed_consensus_preference import (
    prefer_complete_routed_consensus,
)


def _line(
    bend_id: str,
    center_y: float,
    span: float,
    *,
    angle_deg: float = 0.0,
    routed: bool = False,
):
    half = span / 2.0
    angle = radians(angle_deg)
    direction = (cos(angle), sin(angle), 0.0)
    center = (0.0, center_y, 0.0)
    start = [center[i] - half * direction[i] for i in range(3)]
    end = [center[i] + half * direction[i] for i in range(3)]
    return {
        "bend_id": bend_id,
        "span_endpoints": [start, end],
        **({"source_kind": "zero_shot_support_strip"} if routed else {}),
    }


def test_prefers_complete_longer_routed_consensus_with_one_tilted_mesh_axis() -> None:
    routed = [
        _line("R1", 0.0, 80.0, routed=True),
        _line("R2", 30.0, 80.0, routed=True),
    ]
    mesh = [
        _line("M1", 0.0, 60.0),
        _line("M2", 30.0, 60.0, angle_deg=10.0),
    ]

    result = prefer_complete_routed_consensus(mesh, routed)

    assert result["activated"] is True
    assert [item["bend_id"] for item in result["selected"]] == ["R1", "R2"]
    assert max(item["axis_angle_deg"] for item in result["pairings"]) == pytest.approx(10.0)


def test_fails_closed_without_complete_stronger_routed_geometry() -> None:
    routed = [
        _line("R1", 0.0, 80.0, routed=True),
        _line("R2", 30.0, 80.0, routed=True),
    ]
    cases = [
        [_line("M1", 0.0, 60.0)],
        [_line("M1", 0.0, 60.0), _line("M2", 45.0, 60.0, angle_deg=10.0)],
        [_line("M1", 0.0, 79.0), _line("M2", 30.0, 79.0, angle_deg=10.0)],
        [_line("M1", 0.0, 60.0), _line("M2", 30.0, 60.0, angle_deg=2.0)],
    ]

    for mesh in cases:
        result = prefer_complete_routed_consensus(mesh, routed)
        assert result["activated"] is False
        assert result["selected"] == mesh


def test_is_deterministic_and_does_not_mutate_inputs() -> None:
    routed = [
        _line("R1", 0.0, 80.0, routed=True),
        _line("R2", 30.0, 80.0, routed=True),
    ]
    mesh = [
        _line("M1", 0.0, 60.0),
        _line("M2", 30.0, 60.0, angle_deg=10.0),
    ]
    before = (deepcopy(mesh), deepcopy(routed))

    first = prefer_complete_routed_consensus(mesh, routed)
    second = prefer_complete_routed_consensus(mesh, routed)

    assert first == second
    assert (mesh, routed) == before
