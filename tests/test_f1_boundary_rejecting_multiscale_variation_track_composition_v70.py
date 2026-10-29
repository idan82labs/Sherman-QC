from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

import backend.f1_boundary_rejecting_multiscale_variation_track_composition_v70 as v70


H = 1.0
X = np.arange(65, dtype=np.float64)
S = np.arange(-36, 37, dtype=np.float64)


def _row(name: str, *, start=(0.0, 0.0, 0.0), end=(64.0, 0.0, 0.0), score=0.1) -> dict:
    return {
        "bend_id": name,
        "route_hash": hashlib.sha256(name.encode()).hexdigest()[:32],
        "line_start": list(start),
        "line_end": list(end),
        "selector_score": score,
        "selection_route": "forbidden-and-must-not-be-read",
    }


def _sharp_profile(angle_degrees: float, offset: float = 0.0) -> np.ndarray:
    angle = math.radians(angle_degrees)
    output = []
    for value in S:
        t = value - offset
        if t <= 0:
            output.append((value, 0.0))
        else:
            output.append((offset + t * math.cos(angle), t * math.sin(angle)))
    return np.asarray(output, dtype=np.float64)


def _rounded_profile(angle_degrees: float, radius: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    arc = radius * angle
    half = arc / 2.0
    result = []
    for s in S:
        if s <= -half:
            result.append((s + half, 0.0))
        elif s >= half:
            theta = angle
            end_y = radius * math.sin(angle)
            end_z = radius * (1.0 - math.cos(angle))
            result.append((end_y + (s - half) * math.cos(theta), end_z + (s - half) * math.sin(theta)))
        else:
            theta = (s + half) / radius
            result.append((radius * math.sin(theta), radius * (1.0 - math.cos(theta))))
    result = np.asarray(result, dtype=np.float64)
    result -= result[int(np.argmin(np.linalg.norm(result, axis=1)))]
    return result


def _faceted_profile(angle_degrees: float, facets: int) -> np.ndarray:
    angle = math.radians(angle_degrees)
    headings = np.linspace(0.0, angle, facets + 1)
    result = [np.zeros(2, dtype=np.float64)]
    for step in range(1, 37):
        heading = headings[min(facets, int(step * facets / 12))]
        result.append(result[-1] + np.asarray((math.cos(heading), math.sin(heading))))
    positive = np.asarray(result)
    negative = np.asarray([(-float(step), 0.0) for step in range(36, 0, -1)])
    return np.vstack((negative, positive))


def _opposing_fold_profile(first_arc: float, second_arc: float, angle_degrees: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two true sequential folds and their exact world cross-section points."""
    angle = math.radians(angle_degrees)
    direction = np.asarray((math.cos(angle), math.sin(angle)))
    first_point = np.zeros(2, dtype=np.float64)
    second_point = (second_arc - first_arc) * direction
    output = []
    for value in S:
        if value <= first_arc:
            output.append(first_point + np.asarray((value - first_arc, 0.0)))
        elif value <= second_arc:
            output.append(first_point + (value - first_arc) * direction)
        else:
            output.append(second_point + np.asarray((value - second_arc, 0.0)))
    return np.asarray(output), first_point, second_point


def _extrude_profiles(profiles: list[np.ndarray], *, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    width = len(profiles[0])
    vertices = np.asarray([[float(x), float(y), float(z)] for x, profile in zip(X, profiles) for y, z in profile], dtype=np.float64)
    faces = []
    for layer in range(64):
        for strip in range(width - 1):
            a = layer * width + strip
            b = (layer + 1) * width + strip
            c = a + 1
            d = b + 1
            faces.extend(((a, b, d), (a, d, c)))
    faces = np.asarray(faces, dtype=np.int64)
    if keep is None:
        return vertices, faces
    keep = np.asarray(keep, dtype=bool)
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(keep)] = np.arange(np.count_nonzero(keep))
    valid = np.all(keep[faces], axis=1)
    return vertices[keep], old_to_new[faces[valid]]


def _extrude(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _extrude_profiles([profile for _ in X])


def _run(vertices: np.ndarray, faces: np.ndarray | None, rows: list[dict] | None = None):
    return v70.apply_brmvt_v1({"predictions": rows or [_row("C0")]}, vertices, faces)


def _decisions(result) -> list[str]:
    _payload, evaluations = result
    return [value.decision for value in sorted(evaluations, key=lambda item: item.candidate.ordinal)]


@pytest.mark.parametrize(
    "fixture,profile",
    [
        ("P01", _sharp_profile(90.0)),
        ("P02", _sharp_profile(12.0)),
        ("P03", _rounded_profile(90.0, 12.0)),
        ("P04", _faceted_profile(60.0, 3)),
        ("P05", _rounded_profile(170.0, 4.0)),
    ],
)
def test_p01_p05_complete_positive_morphologies_keep(fixture: str, profile: np.ndarray):
    vertices, faces = _extrude(profile)
    payload, evaluations = _run(vertices, faces, [_row(fixture)])
    assert payload["brmvt_v1_audit"].get("abort_reason") is None
    assert len(evaluations) == 1
    assert evaluations[0].decision == "KEEP"


def test_p06_partial_central_24h_keeps():
    profiles = [_rounded_profile(90.0, 12.0) if 20 <= x < 44 else np.column_stack((S, np.zeros_like(S))) for x in X]
    vertices, faces = _extrude_profiles(profiles)
    _payload, evaluations = _run(vertices, faces, [_row("P06")])
    assert evaluations[0].decision == "KEEP"


def test_p07_dropouts_and_one_sided_interval_keep():
    vertices, faces = _extrude(_rounded_profile(90.0, 12.0))
    width = len(S)
    keep = np.ones(len(vertices), dtype=bool)
    for x in list(range(12, 16)) + list(range(31, 35)):
        keep[x * width:(x + 1) * width] = False
    for x in range(47, 55):
        keep[x * width + np.flatnonzero(S < 0)] = False
    # This acquisition fixture is point-only: the two true gaps and the
    # one-sided interval are scan dropout, not a fabricated open mesh cut.
    _payload, evaluations = _run(vertices[keep], None, [_row("P07")])
    assert evaluations[0].decision == "KEEP"


def test_p08_opposing_folds_keep_two():
    profile, first, second = _opposing_fold_profile(-4.0, 4.0)
    vertices, faces = _extrude(profile)
    rows = [
        _row("P08-A", start=(0, *first), end=(64, *first)),
        _row("P08-B", start=(0, *second), end=(64, *second)),
    ]
    _payload, evaluations = _run(vertices, faces, rows)
    assert _decisions((_payload, evaluations)) == ["KEEP", "KEEP"]


def test_p09_parallel_folds_3_5h_are_not_duplicates():
    profile, first, second = _opposing_fold_profile(-1.75, 1.75)
    vertices, faces = _extrude(profile)
    rows = [
        _row("P09-A", start=(0, *first), end=(64, *first)),
        _row("P09-B", start=(0, *second), end=(64, *second)),
    ]
    _payload, evaluations = _run(vertices, faces, rows)
    assert _decisions((_payload, evaluations)) == ["KEEP", "KEEP"]
    assert all(value.terminal_reason != "DUPLICATE_SUPPRESS" for value in evaluations)


def test_p10_sinusoidally_displaced_crease_keeps():
    profiles = []
    for x in X:
        offset = 2.0 * math.sin(2.0 * math.pi * x / 64.0)
        profiles.append(_sharp_profile(60.0, offset))
    vertices, faces = _extrude_profiles(profiles)
    _payload, evaluations = _run(vertices, faces, [_row("P10")])
    assert evaluations[0].decision == "KEEP"


@pytest.mark.parametrize("fixture", ["N01", "N06"])
def test_flat_interior_and_triangulation_seam_drop(fixture: str):
    vertices, faces = _extrude(np.column_stack((S, np.zeros_like(S))))
    if fixture == "N06":
        faces = np.vstack((faces, faces[:, ::-1]))
    _payload, evaluations = _run(vertices, faces, [_row(fixture)])
    assert evaluations[0].decision == "DROP"
    assert evaluations[0].terminal_reason == "NO_FEATURE_NODE"


def test_n02_flat_free_edge_drops():
    profile = np.column_stack((np.arange(0, 73, dtype=np.float64), np.zeros(73)))
    vertices, faces = _extrude(profile)
    _payload, evaluations = _run(vertices, faces, [_row("N02")])
    assert evaluations[0].decision == "DROP"


def test_n03_flat_hole_like_perimeter_drops():
    # A flat one-sided strip is the local geometry of a circular hole boundary.
    profile = np.column_stack((np.arange(0, 73, dtype=np.float64), np.zeros(73)))
    vertices, faces = _extrude(profile)
    _payload, evaluations = _run(vertices, faces, [_row("N03")])
    assert evaluations[0].decision == "DROP"


def test_n04_eroded_jittered_edge_drops():
    profile = np.column_stack((np.arange(0, 73, dtype=np.float64), np.zeros(73)))
    vertices, faces = _extrude(profile)
    jitter = 0.05 * np.sin(np.arange(len(vertices), dtype=np.float64) * 1.61803398875)
    vertices[:, 2] += jitter
    _payload, evaluations = _run(vertices, faces, [_row("N04")])
    assert evaluations[0].decision == "DROP"


def _cylinder(radius: float, *, complete: bool) -> tuple[np.ndarray, np.ndarray]:
    if complete:
        angles = np.linspace(0.0, 2.0 * math.pi, 73, endpoint=False)
    else:
        angles = np.linspace(-36.0 / radius, 36.0 / radius, 73)
    profile = np.column_stack((radius * np.sin(angles), radius * (1.0 - np.cos(angles))))
    return _extrude(profile)


def test_n05_uniform_radius_80_cylinder_generator_drops():
    vertices, faces = _cylinder(80.0, complete=False)
    _payload, evaluations = _run(vertices, faces, [_row("N05")])
    assert evaluations[0].decision == "DROP"


def test_n07_isolated_six_h_corner_fails_track_length():
    profiles = []
    for x in X:
        angle = 90.0 * math.sin(math.pi * (x - 29.0) / 6.0) ** 2 if 29 <= x <= 35 else 0.0
        profiles.append(_sharp_profile(angle))
    vertices, faces = _extrude_profiles(profiles)
    _payload, evaluations = _run(vertices, faces, [_row("N07")])
    assert evaluations[0].decision == "DROP"
    assert evaluations[0].terminal_reason in {"NO_FEATURE_NODE", "NO_ADMISSIBLE_TRACK"}


def test_n08_offset_duplicate_has_one_owner():
    vertices, faces = _extrude(_sharp_profile(90.0))
    rows = [_row("N08-A"), _row("N08-B", start=(0, 0.75, 0), end=(64, 0.75, 0))]
    _payload, evaluations = _run(vertices, faces, rows)
    assert sum(value.decision == "KEEP" for value in evaluations) == 1
    assert sum(value.terminal_reason == "DUPLICATE_SUPPRESS" for value in evaluations) == 1


def test_n09_complete_cylinder_axial_candidate_drops():
    vertices, faces = _cylinder(12.0, complete=True)
    _payload, evaluations = _run(vertices, faces, [_row("N09")])
    assert evaluations[0].decision == "DROP"


def test_n10_two_short_crossing_features_drop():
    profiles = []
    for x in X:
        angle = 90.0 * math.sin(math.pi * (x - 29.0) / 6.0) ** 2 if 29 <= x <= 35 else 0.0
        profiles.append(_sharp_profile(angle))
    vertices, faces = _extrude_profiles(profiles)
    rows = [_row("N10-A"), _row("N10-B", start=(32, -32, 0), end=(32, 32, 0))]
    _payload, evaluations = _run(vertices, faces, rows)
    assert all(value.decision == "DROP" for value in evaluations)


def test_locked_comparator_values():
    assert np.clip((0.70 - v70.U_LOW) / (v70.U_HIGH - v70.U_LOW), 0, 1) == 0
    assert np.clip((np.nextafter(0.70, math.inf) - v70.U_LOW) / (v70.U_HIGH - v70.U_LOW), 0, 1) > 0
    assert np.clip((0.95 - v70.U_LOW) / (v70.U_HIGH - v70.U_LOW), 0, 1) == 1
    assert np.clip((0.01 - v70.A_LOW) / (v70.A_HIGH - v70.A_LOW), 0, 1) == 0
    assert np.clip((np.nextafter(0.01, math.inf) - v70.A_LOW) / (v70.A_HIGH - v70.A_LOW), 0, 1) > 0
    assert np.clip((0.10 - v70.A_LOW) / (v70.A_HIGH - v70.A_LOW), 0, 1) == 1
    assert np.clip((math.radians(100) - v70.BOUNDARY_START) / v70.BOUNDARY_SPAN, 0, 1) == pytest.approx(0)
    assert np.clip((math.radians(180) - v70.BOUNDARY_START) / v70.BOUNDARY_SPAN, 0, 1) == pytest.approx(1)
    assert 0.20 >= v70.WEAK_THRESHOLD and np.nextafter(0.20, -math.inf) < v70.WEAK_THRESHOLD
    assert 0.55 >= v70.STRONG_THRESHOLD and np.nextafter(0.55, -math.inf) < v70.STRONG_THRESHOLD
    assert 8.0 <= v70.HYSTERESIS_DISTANCE and np.nextafter(8.0, math.inf) > v70.HYSTERESIS_DISTANCE
    assert 1.0 >= 1.0 and np.nextafter(1.0, -math.inf) < 1.0


def test_input_permutation_endpoint_reversal_and_rigid_transform_invariance():
    vertices, faces = _extrude(_sharp_profile(90.0))
    row = _row("INV")
    baseline, _ = _run(vertices, faces, [row])
    rng = np.random.default_rng(7001)
    permutation = rng.permutation(len(vertices))
    inverse = np.empty(len(vertices), dtype=np.int64)
    inverse[permutation] = np.arange(len(vertices))
    permuted_faces = inverse[faces]
    reversed_row = _row("INV", start=(64, 0, 0), end=(0, 0, 0))
    permuted, _ = _run(vertices[permutation], permuted_faces[:, ::-1], [reversed_row])
    assert baseline["brmvt_v1_audit"]["canonical_output_sha256"] == permuted["brmvt_v1_audit"]["canonical_output_sha256"]


def test_forbidden_score_and_route_do_not_change_artifact():
    vertices, faces = _extrude(_sharp_profile(90.0))
    first, _ = _run(vertices, faces, [_row("SAFE", score=0.0)])
    second_row = _row("SAFE", score=999.0)
    second_row["selection_route"] = "different-forbidden-route"
    second, _ = _run(vertices, faces, [second_row])
    assert first["brmvt_v1_audit"]["canonical_output_sha256"] == second["brmvt_v1_audit"]["canonical_output_sha256"]


def test_exact_refinement_preserves_p01_decision():
    vertices, faces = _extrude(_sharp_profile(90.0))
    baseline, baseline_eval = _run(vertices, faces, [_row("REFINE")])
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    output = [point.copy() for point in vertices]

    def midpoint(first: int, second: int) -> int:
        key = tuple(sorted((int(first), int(second))))
        if key not in edge_to_midpoint:
            edge_to_midpoint[key] = len(output)
            output.append(0.5 * (vertices[key[0]] + vertices[key[1]]))
        return edge_to_midpoint[key]

    refined = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    refined_payload, refined_eval = _run(np.asarray(output), np.asarray(refined), [_row("REFINE")])
    assert baseline_eval[0].decision == refined_eval[0].decision == "KEEP"
    assert baseline["predictions"][0]["bend_id"] == refined_payload["predictions"][0]["bend_id"]
