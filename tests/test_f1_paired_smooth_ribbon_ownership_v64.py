from __future__ import annotations

import math

import numpy as np

import backend.f1_paired_smooth_ribbon_ownership_v64 as v64


def _extrude_profile(
    profile: np.ndarray,
    *,
    length: float = 80.0,
    displace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.linspace(0.0, length, 81, dtype=np.float64)
    vertices = []
    for x in x_values:
        row = np.asarray(profile, dtype=np.float64).copy()
        if displace:
            tangent = np.gradient(row, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            phase = np.linspace(0.0, math.pi, len(row))
            amplitude = 0.15 * math.sin(2.0 * math.pi * x / length) * np.sin(phase)
            row += amplitude[:, None] * normal
        vertices.extend(np.column_stack((np.full(len(row), x), row)))
    vertices_array = np.asarray(vertices, dtype=np.float64)
    profile_count = len(profile)
    faces = []
    for x_index in range(len(x_values) - 1):
        for profile_index in range(profile_count - 1):
            first = x_index * profile_count + profile_index
            second = (x_index + 1) * profile_count + profile_index
            third = second + 1
            fourth = first + 1
            faces.append((first, second, third))
            faces.append((first, third, fourth))
    return vertices_array, np.asarray(faces, dtype=np.int64)


def _circular_profile(total_degrees: float = 120.0, radius: float = 24.0):
    angles = np.radians(np.linspace(-30.0, total_degrees - 30.0, int(total_degrees) + 1))
    profile = np.column_stack((radius * np.sin(angles), radius * (1.0 - np.cos(angles))))
    return profile, angles


def _integrated_profile() -> tuple[np.ndarray, np.ndarray, int, int]:
    central_length = 30.0
    # Give the displaced source physical margin beyond the certificate's exact
    # 0.25g minimum; the prescribed A-to-B variable-curvature arc is unchanged.
    flank = 0.40 * central_length
    s_values = np.linspace(-flank, central_length + flank, 181, dtype=np.float64)
    tangent_angles = np.zeros_like(s_values)
    for index, s in enumerate(s_values):
        if s <= 0.0:
            tangent_angles[index] = 0.0
        elif s >= central_length:
            tangent_angles[index] = math.radians(75.0)
        else:
            q = s / central_length
            # Integral of 1 + 0.5 cos(2 pi q), normalized because its mean is 1.
            tangent_angles[index] = math.radians(75.0) * (q + 0.5 * math.sin(2.0 * math.pi * q) / (2.0 * math.pi))
    profile = np.zeros((len(s_values), 2), dtype=np.float64)
    for index in range(1, len(s_values)):
        ds = s_values[index] - s_values[index - 1]
        theta = 0.5 * (tangent_angles[index] + tangent_angles[index - 1])
        profile[index] = profile[index - 1] + ds * np.asarray((math.cos(theta), math.sin(theta)))
    first_index = int(np.argmin(np.abs(s_values)))
    second_index = int(np.argmin(np.abs(s_values - central_length)))
    return profile, tangent_angles, first_index, second_index


def _sharp_profile(turns: list[tuple[float, float]], *, before: float = 8.0, after: float = 8.0, step: float = 0.5):
    """Polyline with instantaneous turns after given path lengths.

    ``turns`` contains ``(straight_length, next_tangent_degrees)``.  The first
    segment starts at zero degrees after an initial flank.
    """

    points = [np.asarray((-before, 0.0), dtype=np.float64)]
    candidate_indices = []
    heading = 0.0
    for distance, next_heading in turns:
        count = int(round((distance + (before if len(points) == 1 else 0.0)) / step))
        for _ in range(count):
            theta = math.radians(heading)
            points.append(points[-1] + step * np.asarray((math.cos(theta), math.sin(theta))))
        candidate_indices.append(len(points) - 1)
        heading = next_heading
        before = 0.0
    for _ in range(int(round(after / step))):
        theta = math.radians(heading)
        points.append(points[-1] + step * np.asarray((math.cos(theta), math.sin(theta))))
    return np.asarray(points), candidate_indices


def _row(index: int, point: np.ndarray, *, length: float = 80.0) -> dict:
    return {
        "bend_id": f"B{index:03d}",
        "line_start": [0.0, float(point[0]), float(point[1])],
        "line_end": [length, float(point[0]), float(point[1])],
        "selection_route": "cylindrical_consensus_recovery_v52",
    }


def _baseline(points: list[np.ndarray], *, length: float = 80.0) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "predictions": [_row(index, point, length=length) for index, point in enumerate(points)],
    }


def _decision(result: dict) -> tuple[int, list[int]]:
    return len(result["predictions"]), result["psroc_v1_audit"]["deleted_ordinals"]


def test_circular_smooth_ribbon_deletes_both():
    profile, angles = _circular_profile()
    vertices, faces = _extrude_profile(profile)
    first = profile[int(np.argmin(np.abs(angles - math.radians(0.0))))]
    second = profile[int(np.argmin(np.abs(angles - math.radians(60.0))))]
    result, artifacts = v64.apply_psroc_v1(_baseline([first, second]), vertices, faces)

    assert _decision(result) == (0, [0, 1])
    assert artifacts[0].certified is True
    assert all(station.state == "POSITIVE" for station in artifacts[0].stations)


def test_variable_curvature_ribbon_with_bounded_displacement_deletes_both():
    profile, _angles, first_index, second_index = _integrated_profile()
    vertices, faces = _extrude_profile(profile, length=100.0, displace=True)
    result, artifacts = v64.apply_psroc_v1(
        _baseline([profile[first_index], profile[second_index]], length=100.0),
        vertices,
        faces,
    )

    assert _decision(result) == (0, [0, 1])
    assert artifacts[0].certified is True


def test_sharp_fold_plus_parallel_distractor_deletes_neither():
    profile, candidates = _sharp_profile([(8.0, 60.0), (12.0, 60.0)], before=0.0)
    vertices, faces = _extrude_profile(profile)
    result, artifacts = v64.apply_psroc_v1(
        _baseline([profile[candidates[0]], profile[candidates[1]]]), vertices, faces
    )

    assert _decision(result) == (2, [])
    assert artifacts[0].certified is False
    assert any(station.state == "CONTRADICTORY" for station in artifacts[0].stations)


def test_u_channel_two_structural_bends_delete_neither():
    profile, candidates = _sharp_profile([(8.0, 90.0), (20.0, 180.0)], before=0.0)
    vertices, faces = _extrude_profile(profile)
    result, artifacts = v64.apply_psroc_v1(
        _baseline([profile[candidates[0]], profile[candidates[1]]]), vertices, faces
    )

    assert _decision(result) == (2, [])
    assert artifacts[0].certified is False


def test_return_hem_two_structural_transitions_delete_neither():
    profile, candidates = _sharp_profile([(8.0, 75.0), (20.0, 150.0)], before=0.0)
    vertices, faces = _extrude_profile(profile)
    result, artifacts = v64.apply_psroc_v1(
        _baseline([profile[candidates[0]], profile[candidates[1]]]), vertices, faces
    )

    assert _decision(result) == (2, [])
    assert artifacts[0].certified is False


def test_smooth_roll_singleton_never_mutates():
    profile, angles = _circular_profile()
    vertices, faces = _extrude_profile(profile)
    point = profile[int(np.argmin(np.abs(angles - math.radians(30.0))))]
    result, artifacts = v64.apply_psroc_v1(_baseline([point]), vertices, faces)

    assert _decision(result) == (1, [])
    assert artifacts == ()


def test_three_missing_station_slabs_leave_pair_unchanged():
    profile, angles = _circular_profile()
    vertices, faces = _extrude_profile(profile)
    centroids_x = np.mean(vertices[faces][:, :, 0], axis=1)
    keep = np.ones(len(faces), dtype=bool)
    for station_x in (20.0, 40.0, 60.0):
        keep &= np.abs(centroids_x - station_x) > 0.75
    faces = faces[keep]
    first = profile[int(np.argmin(np.abs(angles - math.radians(0.0))))]
    second = profile[int(np.argmin(np.abs(angles - math.radians(60.0))))]
    result, artifacts = v64.apply_psroc_v1(_baseline([first, second]), vertices, faces)

    assert _decision(result) == (2, [])
    assert artifacts[0].certified is False
    assert sum(station.state == "MISSING" for station in artifacts[0].stations) >= 3
    assert sum(station.state == "POSITIVE" for station in artifacts[0].stations) <= 4


def test_replay_and_reversed_face_winding_are_decision_identical():
    profile, angles = _circular_profile()
    vertices, faces = _extrude_profile(profile)
    first = profile[int(np.argmin(np.abs(angles - math.radians(0.0))))]
    second = profile[int(np.argmin(np.abs(angles - math.radians(60.0))))]
    baseline = _baseline([first, second])
    primary, _ = v64.apply_psroc_v1(baseline, vertices, faces)
    replay, _ = v64.apply_psroc_v1(baseline, vertices, faces)
    reversed_faces = faces[:, ::-1].copy()
    reversed_result, _ = v64.apply_psroc_v1(baseline, vertices, reversed_faces)

    assert primary["psroc_v1_audit_sha256"] == replay["psroc_v1_audit_sha256"]
    assert primary["predictions"] == replay["predictions"]
    assert _decision(primary) == _decision(reversed_result) == (0, [0, 1])
    assert primary["predictions"] == reversed_result["predictions"]
    assert "source" not in str(primary["psroc_v1_audit"]).lower()
