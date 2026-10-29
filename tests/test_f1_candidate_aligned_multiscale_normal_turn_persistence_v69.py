from __future__ import annotations

import hashlib
import math
import time

import numpy as np

import backend.f1_candidate_aligned_multiscale_normal_turn_persistence_v69 as v69


def _profile_from_turns(
    turns: list[tuple[float, float]],
    *,
    extent: float = 0.5,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Integrate (arc position, heading degrees) into a centered PL profile."""

    low = -extent if low is None else low
    high = extent if high is None else high
    breakpoints = sorted({low, high, 0.0, *(position for position, _heading in turns)})
    # Insert a fine regular partition so scale shelves integrate exactly even
    # when a comparator boundary falls inside a profile segment.
    breakpoints.extend(np.linspace(low, high, 101).tolist())
    arc = np.asarray(sorted(set(round(value, 12) for value in breakpoints)), dtype=np.float64)
    headings = []
    for value in arc[:-1]:
        heading = 0.0
        for position, next_heading in turns:
            if value >= position - 1e-14:
                heading = next_heading
        headings.append(math.radians(heading))
    points = [np.zeros(2, dtype=np.float64)]
    origin_index = int(np.where(np.isclose(arc, 0.0, atol=1e-14))[0][0])
    # First integrate from the left, then translate the s=0 point to origin.
    for index, heading in enumerate(headings):
        length = arc[index + 1] - arc[index]
        points.append(points[-1] + length * np.asarray((math.cos(heading), math.sin(heading))))
    result = np.asarray(points, dtype=np.float64)
    result -= result[origin_index]
    return result


def _rounded_profile(angle_deg: float, width: float, facets: int) -> np.ndarray:
    half = width / 2.0
    turns = []
    for index in range(facets):
        position = -half + index * width / facets
        turns.append((position, angle_deg * (index + 1) / facets))
    return _profile_from_turns(turns)


def _extrude(profile: np.ndarray, *, x_limits: tuple[float, float] = (-1.0, 1.0), slabs: int = 37) -> tuple[np.ndarray, np.ndarray]:
    # Unequal deterministic spacing keeps every candidate station off a mesh edge.
    raw = np.linspace(0.0, 1.0, slabs + 1)
    warped = raw + 0.003 * raw * (1.0 - raw)
    x = x_limits[0] + (x_limits[1] - x_limits[0]) * warped
    vertices = np.asarray(
        [[float(axis), float(point[0]), float(point[1])] for axis in x for point in profile],
        dtype=np.float64,
    )
    width = len(profile)
    faces = []
    for layer in range(len(x) - 1):
        for strip in range(width - 1):
            a = layer * width + strip
            b = (layer + 1) * width + strip
            c = layer * width + strip + 1
            d = (layer + 1) * width + strip + 1
            if (layer + strip) & 1:
                faces.extend(((a, b, c), (b, d, c)))
            else:
                faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _point_at_profile_arc(profile: np.ndarray, signed_arc: float, *, low: float = -0.5) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = signed_arc - low
    index = min(len(lengths) - 1, max(0, int(np.searchsorted(cumulative, target, side="right") - 1)))
    fraction = (target - cumulative[index]) / lengths[index]
    return profile[index] + fraction * (profile[index + 1] - profile[index])


def _variable_fold_mesh(angle_at_x, *, slabs: int = 73) -> tuple[np.ndarray, np.ndarray]:
    raw = np.linspace(0.0, 1.0, slabs + 1)
    warped = raw + 0.0017 * raw * (1.0 - raw)
    x_values = -1.0 + 2.0 * warped
    profiles = [_profile_from_turns([(0.0, float(angle_at_x(float(axis))))]) for axis in x_values]
    width = len(profiles[0])
    vertices = np.asarray(
        [[float(axis), float(point[0]), float(point[1])] for axis, profile in zip(x_values, profiles) for point in profile],
        dtype=np.float64,
    )
    faces = []
    for layer in range(len(x_values) - 1):
        for strip in range(width - 1):
            a = layer * width + strip
            b = (layer + 1) * width + strip
            c = layer * width + strip + 1
            d = (layer + 1) * width + strip + 1
            faces.extend(((a, b, d), (a, d, c)) if (layer + strip) % 2 == 0 else ((a, b, c), (b, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _cylindrical_profile(total_angle_deg: float, turn_width: float, *, extent: float = 0.5, facets: int = 100) -> np.ndarray:
    half = turn_width / 2.0
    positions = np.linspace(-half, half, facets + 1)[:-1]
    turns = [(float(position), total_angle_deg * (index + 1) / facets) for index, position in enumerate(positions)]
    return _profile_from_turns(turns, extent=extent)


def _refine(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = [point.copy() for point in vertices]
    midpoint: dict[tuple[int, int], int] = {}

    def mid(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in midpoint:
            midpoint[key] = len(output)
            output.append(0.5 * (vertices[key[0]] + vertices[key[1]]))
        return midpoint[key]

    refined = []
    for a, b, c in np.asarray(faces, dtype=np.int64):
        ab, bc, ca = mid(int(a), int(b)), mid(int(b), int(c)), mid(int(c), int(a))
        refined.extend(((int(a), ab, ca), (ab, int(b), bc), (ca, bc, int(c)), (ab, bc, ca)))
    return np.asarray(output, dtype=np.float64), np.asarray(refined, dtype=np.int64)


def _combine_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for current_vertices, current_faces in meshes:
        offset = len(vertices)
        vertices.extend(np.asarray(current_vertices, dtype=np.float64))
        faces.extend(np.asarray(current_faces, dtype=np.int64) + offset)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _flat_profile(low: float, high: float) -> np.ndarray:
    return np.asarray([[float(value), 0.0] for value in np.linspace(low, high, 101)], dtype=np.float64)


def _row(index: int = 0, *, start=(-0.5, 0.0, 0.0), end=(0.5, 0.0, 0.0), score: float = 0.9) -> dict:
    return {
        "bend_id": f"P-{index}",
        "route_hash": hashlib.sha256(f"P-{index}".encode()).hexdigest()[:32],
        "line_start": list(start),
        "line_end": list(end),
        # Forbidden fields are deliberately present; the implementation must
        # neither branch on them nor copy them into its deterministic audit.
        "selector_score": score,
        "selection_route": "must_not_be_read",
    }


def _run(profile: np.ndarray, rows: list[dict] | None = None):
    vertices, faces = _extrude(profile)
    return v69.apply_camntp_v1({"predictions": rows or [_row()]}, vertices, faces)


def _record(result: dict, index: int = 0) -> dict:
    return result["camntp_v1_decision_record"]["candidate_records"][index]


def test_p01_internal_sharp_30_degree_fold_keeps():
    result, evaluations = _run(_profile_from_turns([(0.0, 30.0)]))
    assert result["camntp_v1_decision_record"]["preprocess_status"] == "OK"
    assert _record(result)["decision"] == "KEEP"
    assert _record(result)["station_pass_count"] == 17
    assert _record(result)["longest_linked_run"] == 17
    assert len(evaluations) == 1


def test_p02_exact_eight_degree_fold_keeps_and_below_fails():
    passing, _ = _run(_profile_from_turns([(0.0, 8.0)]))
    failing, _ = _run(_profile_from_turns([(0.0, 7.999999)]))
    assert _record(passing)["decision"] == "KEEP"
    assert _record(failing)["decision"] == "VETO"
    assert _record(failing)["station_failure_histogram"].get("TURN_TOO_SMALL") == 17


def test_p03_rounded_60_degree_fold_keeps():
    result, _ = _run(_rounded_profile(60.0, 0.006, 16))
    assert _record(result)["decision"] == "KEEP"


def test_p04_faceted_75_degree_fold_keeps():
    result, _ = _run(_rounded_profile(75.0, 0.008, 9))
    assert _record(result)["decision"] == "KEEP"


def test_p05_finite_fold_has_exactly_nine_central_passing_stations():
    def angle(axis: float) -> float:
        absolute = abs(axis)
        if absolute <= 0.23:
            return 30.0
        if absolute >= 0.25:
            return 0.0
        return 30.0 * (0.25 - absolute) / 0.02

    vertices, faces = _variable_fold_mesh(angle)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    record = _record(result)
    assert record["decision"] == "KEEP"
    assert record["station_pass_count"] == 9
    assert record["station_pass_mask_17"] == sum(1 << index for index in range(4, 13))


def test_p06_partially_observed_fold_has_exactly_nine_stations_and_keeps():
    profile = _profile_from_turns([(0.0, 30.0)])
    vertices, faces = _extrude(profile, x_limits=(-0.25, 0.25), slabs=37)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    record = _record(result)
    assert record["decision"] == "KEEP"
    assert record["station_pass_count"] == 9
    assert record["longest_linked_run"] == 9


def test_p07_two_distinct_same_sign_folds_both_keep():
    profile = _profile_from_turns([(0.0, 45.0), (0.06, 90.0)])
    second = _point_at_profile_arc(profile, 0.06)
    rows = [
        _row(0),
        _row(1, start=(-0.5, float(second[0]), float(second[1])), end=(0.5, float(second[0]), float(second[1]))),
    ]
    result, _ = _run(profile, rows)
    assert [row["decision"] for row in result["camntp_v1_decision_record"]["candidate_records"]] == ["KEEP", "KEEP"]


def test_p08_u_channel_opposing_folds_both_keep():
    profile = _profile_from_turns([(0.0, 90.0), (0.12, 0.0)])
    second = _point_at_profile_arc(profile, 0.12)
    rows = [
        _row(0),
        _row(1, start=(-0.5, float(second[0]), float(second[1])), end=(0.5, float(second[0]), float(second[1]))),
    ]
    result, _ = _run(profile, rows)
    assert [row["decision"] for row in result["camntp_v1_decision_record"]["candidate_records"]] == ["KEEP", "KEEP"]


def test_p09_short_observed_return_leg_keeps():
    profile = _profile_from_turns([(0.0, 90.0)], low=-0.5, high=0.06)
    result, _ = _run(profile)
    assert _record(result)["decision"] == "KEEP"


def test_p10_rounded_hem_with_short_return_leg_keeps():
    half = 0.003
    turns = [(-half + index * 0.006 / 16.0, 170.0 * (index + 1) / 16.0) for index in range(16)]
    profile = _profile_from_turns(turns, low=-0.5, high=0.04)
    result, _ = _run(profile)
    assert _record(result)["decision"] == "KEEP"


def test_p11_coarse_longitudinal_triangulation_keeps():
    profile = _rounded_profile(60.0, 0.006, 16)
    vertices, faces = _extrude(profile, slabs=7)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "KEEP"


def test_p12_small_sinusoidal_spine_displacement_keeps():
    profile = _profile_from_turns([(0.0, 30.0)])
    raw = np.linspace(0.0, 1.0, 74)
    x_values = -1.0 + 2.0 * (raw + 0.0017 * raw * (1.0 - raw))
    width = len(profile)
    vertices = np.asarray(
        [
            [float(axis), float(point[0]), float(point[1] + 0.002 * math.sin(2.0 * math.pi * axis))]
            for axis in x_values
            for point in profile
        ],
        dtype=np.float64,
    )
    faces = []
    for layer in range(len(x_values) - 1):
        for strip in range(width - 1):
            a = layer * width + strip
            b = (layer + 1) * width + strip
            c = layer * width + strip + 1
            d = (layer + 1) * width + strip + 1
            faces.extend(((a, b, d), (a, d, c)))
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, np.asarray(faces, dtype=np.int64))
    assert _record(result)["decision"] == "KEEP"


def test_n01_flat_panel_vetoes_turn_too_small():
    result, _ = _run(_profile_from_turns([]))
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["station_failure_histogram"].get("TURN_TOO_SMALL") == 17


def test_n02_open_outer_perimeter_vetoes():
    profile = _profile_from_turns([], low=0.0, high=0.5)
    result, _ = _run(profile)
    assert _record(result)["decision"] == "VETO"
    assert set(_record(result)["station_failure_histogram"]) & {"NO_BILATERAL_ARC", "BOUNDARY_NEAR"}


def test_n03_open_hole_edge_vetoes_boundary_support():
    lower = _extrude(_flat_profile(-0.5, 0.0))
    upper = _extrude(_flat_profile(0.10, 0.5))
    vertices, faces = _combine_meshes([lower, upper])
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert set(_record(result)["station_failure_histogram"]) & {"BOUNDARY_NEAR", "NO_BILATERAL_ARC"}


def test_n04_scan_dropout_slit_vetoes_boundary_support():
    lower = _flat_profile(-0.5, -0.002)
    upper = _flat_profile(0.002, 0.5)
    vertices, faces = _combine_meshes([_extrude(lower), _extrude(upper)])
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert set(_record(result)["station_failure_histogram"]) & {"BOUNDARY_NEAR", "NO_BILATERAL_ARC", "AMBIGUOUS_COMPONENT"}


def test_n05_three_face_nonmanifold_seam_vetoes_specific_support():
    first = _profile_from_turns([], low=-0.5, high=0.0)
    second = _profile_from_turns([(0.0, 30.0)], low=0.0, high=0.5)
    third = _profile_from_turns([(0.0, -30.0)], low=0.0, high=0.5)
    vertices, faces = _combine_meshes([_extrude(first), _extrude(second), _extrude(third)])
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["station_failure_histogram"].get("NONMANIFOLD_SUPPORT") == 17


def test_n06_fold_present_at_only_three_stations_vetoes_count():
    def angle(axis: float) -> float:
        return 30.0 if abs(axis) <= 0.07 else 0.0

    vertices, faces = _variable_fold_mesh(angle)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["terminal_reason"] == "LONGITUDINAL_COUNT"
    assert _record(result)["station_pass_count"] == 3


def test_n07_uniform_cylinder_vetoes_without_scale_plateau():
    result, _ = _run(_cylindrical_profile(math.degrees(0.5 / 0.20), 1.0, facets=160))
    assert _record(result)["decision"] == "VETO"
    assert set(_record(result)["station_failure_histogram"]) & {"TURN_DIFFUSE", "NO_SCALE_PLATEAU", "TURN_TOO_SMALL"}


def test_n08_broad_distributed_transition_vetoes():
    result, _ = _run(_cylindrical_profile(30.0, 0.20, facets=80))
    assert _record(result)["decision"] == "VETO"


def test_n09_alternating_corrugation_vetoes():
    turns = [(-0.012, 6.0), (-0.008, 0.0), (-0.004, 6.0), (0.0, 0.0), (0.004, 6.0), (0.008, 0.0)]
    result, _ = _run(_profile_from_turns(turns))
    assert _record(result)["decision"] == "VETO"


def test_n11_watertight_rim_like_opposing_turn_vetoes():
    result, _ = _run(_profile_from_turns([(0.0, 90.0), (0.004, 0.0)]))
    assert _record(result)["decision"] == "VETO"


def test_n13_axis_misaligned_forty_degrees_vetoes():
    profile = _profile_from_turns([(0.0, 30.0)])
    direction = np.asarray((math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0))
    result, _ = _run(profile, [_row(start=(-0.5 * direction).tolist(), end=(0.5 * direction).tolist())])
    assert _record(result)["decision"] == "VETO"


def test_n14_opposing_bead_turns_vetoes():
    result, _ = _run(_profile_from_turns([(-0.003, 30.0), (0.003, 0.0)]))
    assert _record(result)["decision"] == "VETO"


def test_n15_exactly_eight_passing_stations_vetoes_count():
    stations = [(-0.5 + (index + 1) / 18.0) for index in range(17)]
    active = set(range(4, 12))

    def angle(axis: float) -> float:
        return 30.0 if min((abs(axis - stations[index]) for index in active), default=10.0) < 0.024 else 0.0

    vertices, faces = _variable_fold_mesh(angle, slabs=145)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["station_pass_count"] == 8
    assert _record(result)["longest_linked_run"] == 8


def test_n16_nine_passing_stations_with_maximum_run_six_vetoes_run():
    stations = [(-0.5 + (index + 1) / 18.0) for index in range(17)]
    active = set(range(2, 8)) | set(range(10, 13))

    def angle(axis: float) -> float:
        return 30.0 if min((abs(axis - stations[index]) for index in active), default=10.0) < 0.024 else 0.0

    vertices, faces = _variable_fold_mesh(angle, slabs=145)
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["station_pass_count"] == 9
    assert _record(result)["longest_linked_run"] == 6
    assert _record(result)["terminal_reason"] == "LONGITUDINAL_RUN"


def test_n12_displaced_fold_vetoes_candidate_off_center():
    profile = _profile_from_turns([(0.0, 30.0)])
    vertices, faces = _extrude(profile)
    # The physical fold spine is eight millimetres from the proposed line at
    # every station (normal to the section curve), so q-p exceeds L/128.
    result, _ = v69.apply_camntp_v1(
        {"predictions": [_row(start=(-0.5, 0.0, -0.008), end=(0.5, 0.0, -0.008))]},
        vertices,
        faces,
    )
    assert _record(result)["decision"] == "VETO"
    histogram = _record(result)["station_failure_histogram"]
    assert histogram.get("CANDIDATE_OFF_CENTER", 0) + histogram.get("TURN_OFF_CENTER", 0) == 17


def test_replay_order_endpoint_winding_similarity_and_score_invariance():
    profile = _rounded_profile(60.0, 0.006, 16)
    vertices, faces = _extrude(profile)
    source = {"predictions": [_row(score=0.1)]}
    primary, _ = v69.apply_camntp_v1(source, vertices, faces)
    replay, _ = v69.apply_camntp_v1(source, vertices, faces)
    assert primary["camntp_v1_decision_record"] == replay["camntp_v1_decision_record"]

    vertex_order = np.arange(len(vertices))[::-1]
    inverse = np.empty(len(vertices), dtype=np.int64)
    inverse[vertex_order] = np.arange(len(vertices))
    variants = [
        (vertices, faces[::-1]),
        (vertices, faces[:, ::-1]),
        (vertices[vertex_order], inverse[faces]),
        (0.8 * vertices + np.asarray((3.0, -2.0, 5.0)), faces),
    ]
    variant_rows = [
        [_row(score=99.0)],
        [_row(score=-10.0)],
        [_row(score=0.0)],
        [_row(start=(2.6, -2.0, 5.0), end=(3.4, -2.0, 5.0), score=7.0)],
    ]
    expected = primary["camntp_v1_decision_record"]["candidate_records"]
    for (current_vertices, current_faces), rows in zip(variants, variant_rows):
        result, _ = v69.apply_camntp_v1({"predictions": rows}, current_vertices, current_faces)
        assert result["camntp_v1_decision_record"]["candidate_records"] == expected

    reversed_result, _ = v69.apply_camntp_v1(
        {"predictions": [_row(start=(0.5, 0.0, 0.0), end=(-0.5, 0.0, 0.0))]},
        vertices,
        faces,
    )
    assert reversed_result["camntp_v1_decision_record"]["candidate_records"] == expected


def test_metamorphic_rotation_reflection_refinement_and_distant_component():
    profile = _rounded_profile(60.0, 0.006, 16)
    vertices, faces = _extrude(profile)
    source = {"predictions": [_row()]}
    primary, _ = v69.apply_camntp_v1(source, vertices, faces)
    expected = primary["camntp_v1_decision_record"]["candidate_records"]

    angle = math.radians(31.0)
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle), 0.0), (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    reflection = np.diag((1.0, -1.0, 1.0))
    for matrix in (rotation, reflection @ rotation):
        translation = np.asarray((2.5, -3.25, 4.75))
        transformed_vertices = vertices @ matrix.T + translation
        endpoints = np.asarray((source["predictions"][0]["line_start"], source["predictions"][0]["line_end"])) @ matrix.T + translation
        result, _ = v69.apply_camntp_v1(
            {"predictions": [{**_row(), "line_start": endpoints[0].tolist(), "line_end": endpoints[1].tolist()}]},
            transformed_vertices,
            faces,
        )
        assert result["camntp_v1_decision_record"]["candidate_records"] == expected

    refined_vertices, refined_faces = _refine(vertices, faces)
    refined, _ = v69.apply_camntp_v1(source, refined_vertices, refined_faces)
    assert refined["camntp_v1_decision_record"]["candidate_records"] == expected

    distant_vertices = vertices + np.asarray((100.0, 100.0, 100.0))
    combined_vertices, combined_faces = _combine_meshes([(vertices, faces), (distant_vertices, faces)])
    distant, _ = v69.apply_camntp_v1(source, combined_vertices, combined_faces)
    assert distant["camntp_v1_decision_record"]["candidate_records"] == expected


def test_candidate_stream_permutation_preserves_sorted_decisions_and_survivors():
    profile = _profile_from_turns([(0.0, 45.0), (0.06, 90.0)])
    second = _point_at_profile_arc(profile, 0.06)
    rows = [
        _row(0),
        _row(1, start=(-0.5, float(second[0]), float(second[1])), end=(0.5, float(second[0]), float(second[1]))),
    ]
    vertices, faces = _extrude(profile)
    first, _ = v69.apply_camntp_v1({"predictions": rows}, vertices, faces)
    second_result, _ = v69.apply_camntp_v1({"predictions": rows[::-1]}, vertices, faces)
    assert first["camntp_v1_decision_record"]["candidate_records"] == second_result["camntp_v1_decision_record"]["candidate_records"]
    assert {row["bend_id"] for row in first["predictions"]} == {row["bend_id"] for row in second_result["predictions"]}


def _passing_scale(theta_deg: float, rotation_deg: float = 0.0) -> v69.ScaleResult:
    rotation = math.radians(rotation_deg)
    theta = math.radians(theta_deg)
    minus = np.asarray((0.0, math.cos(rotation), math.sin(rotation)))
    plus = np.asarray((0.0, math.cos(rotation + theta), math.sin(rotation + theta)))
    return v69.ScaleResult(True, "PASS", theta, 1.0, 1.0, 10.0, 0.0, minus, plus)


def test_b01_b24_inclusive_scalar_and_plateau_comparator_boundaries():
    epsilon_ratio = 1e-6
    epsilon_angle = math.radians(1e-6)
    assert v69._ge(math.radians(8.0), v69.ANGLE_MIN)
    assert not v69._ge(math.radians(8.0) - epsilon_angle, v69.ANGLE_MIN)
    for threshold in (0.95, 0.965, 0.70):
        assert v69._ge(threshold, threshold)
        assert not v69._ge(threshold - epsilon_ratio, threshold)
    assert v69._le(0.20, 0.20)
    assert not v69._le(0.20 + epsilon_ratio, 0.20)
    assert v69._ge(1.25, 1.25)
    assert not v69._ge(1.25 - epsilon_ratio, 1.25)
    assert v69._le(1.25 * math.pi, 1.25 * math.pi)
    assert not v69._le(1.25 * math.pi + epsilon_angle, 1.25 * math.pi)

    exact_spread = [_passing_scale(27.0), _passing_scale(30.0), _passing_scale(33.0)] + [v69.ScaleResult(False, "FAIL")] * 12
    above_spread = [_passing_scale(27.0 - 1e-6), _passing_scale(30.0), _passing_scale(33.0)] + [v69.ScaleResult(False, "FAIL")] * 12
    assert v69._first_plateau(exact_spread, 1.0) is not None
    assert v69._first_plateau(above_spread, 1.0) is None

    exact_drift = [_passing_scale(30.0, value) for value in (0.0, 6.0, 12.0)] + [v69.ScaleResult(False, "FAIL")] * 12
    above_drift = [_passing_scale(30.0, value) for value in (0.0, 6.0, 12.000001)] + [v69.ScaleResult(False, "FAIL")] * 12
    assert v69._first_plateau(exact_drift, 1.0) is not None
    assert v69._first_plateau(above_drift, 1.0) is None


def test_b25_b32_exact_plateau_count_and_link_boundaries():
    three = [_passing_scale(30.0)] * 3 + [v69.ScaleResult(False, "FAIL")] * 12
    two = [_passing_scale(30.0)] * 2 + [v69.ScaleResult(False, "FAIL")] * 13
    assert v69._first_plateau(three, 1.0) is not None
    assert v69._first_plateau(two, 1.0) is None

    plateau = v69.Plateau(5, v69.SCALE_RATIOS[6], math.radians(30.0), 1.0, 1.0, 10.0, 0.0)
    limit = 0.5 * plateau.representative_radius + 1.0 / 1024.0
    first = v69.StationResult(0, True, "PASS", 0, np.zeros(3), np.zeros(3), plateau)
    exact = v69.StationResult(1, True, "PASS", 0, np.asarray((0.0, limit, 0.0)), np.asarray((0.0, limit, 0.0)), plateau)
    above = v69.StationResult(1, True, "PASS", 0, np.asarray((0.0, limit + 1e-6, 0.0)), np.asarray((0.0, limit + 1e-6, 0.0)), plateau)
    assert v69._link(first, exact, 1.0)
    assert not v69._link(first, above, 1.0)


def test_b33_equal_distance_valid_components_are_ambiguous():
    profile = _profile_from_turns([(0.0, 30.0)])
    epsilon = 2.0 ** -42
    upper = profile.copy()
    lower = profile.copy()
    upper[:, 1] += epsilon
    lower[:, 1] -= epsilon
    vertices, faces = _combine_meshes([_extrude(upper), _extrude(lower)])
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces)
    assert _record(result)["decision"] == "VETO"
    assert _record(result)["station_failure_histogram"].get("AMBIGUOUS_COMPONENT") == 17


def test_b34_b35_input_caps_are_inclusive_and_plus_one_aborts_atomically():
    vertices = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    face = np.asarray(((0, 1, 2),), dtype=np.int64)
    at_cap = np.repeat(face, v69.MAX_TRIANGLES, axis=0)
    passing, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, at_cap)
    assert passing["camntp_v1_decision_record"]["preprocess_status"] == "OK"
    above = np.repeat(face, v69.MAX_TRIANGLES + 1, axis=0)
    aborted, evaluations = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, above)
    assert evaluations == ()
    assert aborted["camntp_v1_modified_output_committed"] is False
    assert aborted["camntp_v1_decision_record"]["scan_decision_status"] == "SOURCE_TRIANGLE_CAP"

    rows = [_row(index, start=(-0.5, index * 1e-6, 0.0), end=(0.5, index * 1e-6, 0.0)) for index in range(v69.MAX_CANDIDATES + 1)]
    candidate_abort, evaluations = v69.apply_camntp_v1({"predictions": rows}, vertices, face)
    assert evaluations == ()
    assert candidate_abort["camntp_v1_decision_record"]["scan_decision_status"] == "CANDIDATE_CAP"


def test_deterministic_audit_contains_no_forbidden_candidate_or_source_fields():
    result, _ = _run(_profile_from_turns([(0.0, 30.0)]))
    encoded = v69.canonical_bytes(result["camntp_v1_decision_record"]).decode("ascii")
    for forbidden in ("selector_score", "selection_route", "line_start", "line_end", "filename", "part_identity", "truth", "cad"):
        assert forbidden not in encoded.lower()
    assert len(encoded.encode("ascii")) <= v69.MAX_AUDIT_BYTES


def test_frozen_350k_triangle_128_candidate_stress_fixture_meets_caps():
    profile = _profile_from_turns([(0.0, 30.0)])
    local_vertices, local_faces = _extrude(profile, slabs=37)
    required = v69.MAX_TRIANGLES - len(local_faces)
    nx, ny = 420, 410
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
    grid_vertices = np.column_stack((100.0 + gx.ravel(), 100.0 + gy.ravel(), np.full(gx.size, 100.0)))
    grid_faces = []
    for first in range(nx - 1):
        row = first * ny
        next_row = (first + 1) * ny
        for second in range(ny - 1):
            a, b, c, d = row + second, next_row + second, row + second + 1, next_row + second + 1
            grid_faces.extend(((a, b, d), (a, d, c)))
            if len(grid_faces) >= required:
                break
        if len(grid_faces) >= required:
            break
    assert len(grid_faces) >= required
    vertices, faces = _combine_meshes([
        (local_vertices, local_faces),
        (grid_vertices, np.asarray(grid_faces[:required], dtype=np.int64)),
    ])
    assert len(vertices) <= v69.MAX_VERTICES
    assert len(faces) == v69.MAX_TRIANGLES
    rows = []
    for index, signed_arc in enumerate(np.linspace(-0.45, 0.45, v69.MAX_CANDIDATES)):
        point = _point_at_profile_arc(profile, float(signed_arc))
        rows.append(_row(index, start=(-0.5, float(point[0]), float(point[1])), end=(0.5, float(point[0]), float(point[1]))))
    started = time.perf_counter()
    result, evaluations = v69.apply_camntp_v1({"predictions": rows}, vertices, faces)
    elapsed = time.perf_counter() - started
    assert len(evaluations) == v69.MAX_CANDIDATES
    assert result["camntp_v1_decision_record"]["preprocess_status"] == "OK"
    assert elapsed <= v69.MAX_ADDED_SECONDS
    envelope = result["camntp_v1_operational_envelope"]
    assert envelope["additional_peak_rss_bytes"] < v69.MAX_ADDED_RSS_BYTES
    assert envelope["peak_rss_bytes"] < v69.MAX_TOTAL_RSS_BYTES
    assert envelope["audit_bytes"] <= v69.MAX_AUDIT_BYTES
    decisions = {row["decision"] for row in result["camntp_v1_decision_record"]["candidate_records"]}
    assert decisions == {"KEEP", "VETO"}


def test_missing_mesh_and_malformed_candidate_are_atomic_no_mutation():
    rows = [_row()]
    missing, evaluations = v69.apply_camntp_v1({"predictions": rows}, np.zeros((1, 3)), None)
    assert evaluations == ()
    assert missing["camntp_v1_decision_record"]["preprocess_status"] == "ABSTAIN_NO_MESH"
    assert missing["predictions"] == rows
    malformed = [{**_row(), "line_end": [0.0, 0.0, 0.0], "line_start": [0.0, 0.0, 0.0]}]
    vertices, faces = _extrude(_profile_from_turns([(0.0, 30.0)]))
    aborted, evaluations = v69.apply_camntp_v1({"predictions": malformed}, vertices, faces)
    assert evaluations == ()
    assert aborted["camntp_v1_modified_output_committed"] is False
    assert aborted["predictions"] == malformed


def test_ordinary_nms_is_called_exactly_once_after_veto():
    calls = []

    def ordinary(rows):
        calls.append([dict(row) for row in rows])
        return list(rows), [{"called": True}]

    vertices, faces = _extrude(_profile_from_turns([(0.0, 30.0)]))
    result, _ = v69.apply_camntp_v1({"predictions": [_row()]}, vertices, faces, ordinary_nms=ordinary)
    assert len(calls) == 1
    assert result["camntp_v1_operational_envelope"]["ordinary_nms_call_count"] == 1
