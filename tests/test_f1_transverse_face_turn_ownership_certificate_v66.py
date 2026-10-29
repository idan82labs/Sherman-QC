from __future__ import annotations

import hashlib
import math

import numpy as np

import backend.f1_transverse_face_turn_ownership_certificate_v66 as v66


def _extrude(profile: np.ndarray, *, closed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    layers = np.arange(-0.37, 80.64, 1.0, dtype=np.float64)
    vertices = np.asarray(
        [[float(axis), float(point[0]), float(point[1])] for axis in layers for point in profile],
        dtype=np.float64,
    )
    width = len(profile)
    strip_count = width if closed else width - 1
    faces: list[tuple[int, int, int]] = []
    for layer in range(len(layers) - 1):
        for strip in range(strip_count):
            next_strip = (strip + 1) % width
            a = layer * width + strip
            b = (layer + 1) * width + strip
            c = layer * width + next_strip
            d = (layer + 1) * width + next_strip
            if (layer + strip) % 2:
                faces.extend(((a, b, c), (b, d, c)))
            else:
                faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _sharp_profile(turns: list[tuple[int, float]], *, before: int = 20, after: int = 20) -> tuple[np.ndarray, list[float]]:
    points = [np.asarray([-float(before), 0.0], dtype=np.float64)]
    heading = 0.0
    locations: list[float] = []
    for index in range(before):
        points.append(points[-1] + np.asarray([1.0, 0.0]))
    arc = float(before)
    for run, next_heading in turns:
        if run:
            angle = math.radians(heading)
            for _ in range(run):
                points.append(points[-1] + np.asarray([math.cos(angle), math.sin(angle)]))
                arc += 1.0
        locations.append(arc)
        heading = next_heading
    angle = math.radians(heading)
    for _ in range(after):
        points.append(points[-1] + np.asarray([math.cos(angle), math.sin(angle)]))
    return np.asarray(points), locations


def _rounded_profile(angle_deg: float, radius: float, *, before: int = 20, after: int = 20) -> tuple[np.ndarray, float]:
    points = [np.asarray([-float(before), 0.0], dtype=np.float64)]
    for _ in range(before):
        points.append(points[-1] + np.asarray([1.0, 0.0]))
    center = points[-1] + np.asarray([0.0, radius])
    segment_count = int(math.ceil(math.radians(angle_deg) * radius))
    for index in range(1, segment_count + 1):
        phi = math.radians(angle_deg) * index / segment_count
        points.append(center + np.asarray([radius * math.sin(phi), -radius * math.cos(phi)]))
    heading = math.radians(angle_deg)
    for _ in range(after):
        points.append(points[-1] + np.asarray([math.cos(heading), math.sin(heading)]))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1))))
    return np.asarray(points), float(before + 0.5 * math.radians(angle_deg) * radius)


def _point_at_arc(profile: np.ndarray, coordinate: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    index = min(len(lengths) - 1, max(0, int(np.searchsorted(arc, coordinate, side="right") - 1)))
    fraction = (coordinate - arc[index]) / lengths[index]
    return profile[index] + fraction * (profile[index + 1] - profile[index])


def _row(index: int, point: np.ndarray, *, axis: str = "x", centeredness_priority: int = 0) -> dict:
    if axis == "x":
        start, end = [0.0, float(point[0]), float(point[1])], [80.0, float(point[0]), float(point[1])]
    else:
        start, end = [40.0, -20.0, 0.0], [40.0, 20.0, 0.0]
    return {
        "bend_id": f"SF-{index}",
        "route_hash": hashlib.sha256(f"SF-{index}".encode()).hexdigest()[:32],
        "better_first_key": [centeredness_priority, index],
        "line_start": start,
        "line_end": end,
        "selection_route": "source_free_fixture",
    }


def _baseline(points: list[np.ndarray]) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "predictions": [_row(index, point) for index, point in enumerate(points)],
    }


def _run(profile: np.ndarray, offsets: list[float], *, closed: bool = False, drop_stations: set[int] | None = None):
    vertices, faces = _extrude(profile, closed=closed)
    if drop_stations:
        keep = np.ones(len(faces), dtype=bool)
        axis = vertices[:, 0]
        for index, face in enumerate(faces):
            lo, hi = float(np.min(axis[face])), float(np.max(axis[face]))
            if any(lo < station * 8.0 < hi for station in drop_stations):
                keep[index] = False
        faces = faces[keep]
    points = [_point_at_arc(profile, offset) for offset in offsets]
    return v66.apply_tftoc_v1(_baseline(points), vertices, faces)


def _assert_h_one(pairs: tuple[v66.PairResult, ...]) -> None:
    assert pairs
    assert all(pair.scale == 1.0 for pair in pairs)


def test_sf01_sharp_single_fold_duplicates():
    profile, turns = _sharp_profile([(0, 70.0)])
    result, pairs = _run(profile, [turns[0] - 3.0, turns[0] + 1.0, turns[0] + 4.0])
    _assert_h_one(pairs)
    assert len(pairs) == 3
    assert all(pair.pair_class == v66.POSITIVE and pair.same_count == 9 and pair.split_count == 0 for pair in pairs)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["bend_id"] == "SF-1"


def test_sf02_rounded_single_fold_duplicates():
    profile, center = _rounded_profile(70.0, 3.0)
    result, pairs = _run(profile, [center - 3.0, center + 1.0, center + 4.0])
    _assert_h_one(pairs)
    assert len(pairs) == 3
    assert all(pair.pair_class == v66.POSITIVE and pair.same_count == 9 for pair in pairs)
    assert [row["bend_id"] for row in result["predictions"]] == ["SF-1"]


def test_sf03_two_nearby_same_sign_folds():
    profile, turns = _sharp_profile([(0, 55.0), (6, 110.0)])
    offsets = [turns[0] - 1.0, turns[0] + 0.25, turns[1] - 1.0, turns[1] + 0.25]
    result, pairs = _run(profile, offsets)
    _assert_h_one(pairs)
    assert len(pairs) == 6
    positives = [pair for pair in pairs if pair.pair_class == v66.POSITIVE]
    negatives = [pair for pair in pairs if pair.pair_class == v66.NEGATIVE]
    assert len(positives) == 2 and all(pair.same_count == 9 and pair.split_count == 0 for pair in positives)
    assert len(negatives) == 4 and all(pair.same_count == 0 and pair.split_count == 9 for pair in negatives)
    assert len(result["predictions"]) == 2


def test_sf04_u_channel_opposing_folds():
    profile, turns = _sharp_profile([(0, 80.0), (6, 0.0)])
    offsets = [turns[0] - 1.0, turns[0] + 0.25, turns[1] - 1.0, turns[1] + 0.25]
    result, pairs = _run(profile, offsets)
    _assert_h_one(pairs)
    positives = [pair for pair in pairs if pair.pair_class == v66.POSITIVE]
    negatives = [pair for pair in pairs if pair.pair_class == v66.NEGATIVE]
    assert len(positives) == 2 and all(pair.same_count == 9 and pair.split_count == 0 for pair in positives)
    assert len(negatives) == 4 and all(pair.same_count == 0 and pair.split_count == 9 for pair in negatives)
    assert len(result["predictions"]) == 2


def test_sf05_return_hem_double_skin():
    profile, center = _rounded_profile(170.0, 1.5)
    result, pairs = _run(profile, [center - 2.0, center + 2.0])
    _assert_h_one(pairs)
    assert len(pairs) == 1
    assert pairs[0].pair_class != v66.POSITIVE
    assert len(result["predictions"]) == 2


def test_sf06_flat_plate_perimeter_distractor():
    top = [np.asarray([float(x), 0.0]) for x in range(0, 21)]
    bottom = [np.asarray([float(x), -0.8]) for x in range(20, -1, -1)]
    profile = np.asarray(top + bottom)
    result, pairs = _run(profile, [1.0, 4.0], closed=True)
    _assert_h_one(pairs)
    assert len(pairs) == 1
    assert pairs[0].pair_class != v66.POSITIVE
    assert len(result["predictions"]) == 2


def test_sf07_true_fold_plus_adjacent_flat_false_line():
    profile, turns = _sharp_profile([(0, 70.0)])
    result, pairs = _run(profile, [turns[0] + 0.25, turns[0] + 3.0])
    _assert_h_one(pairs)
    assert len(pairs) == 1 and pairs[0].same_count == 9 and pairs[0].pair_class == v66.POSITIVE
    assert [row["bend_id"] for row in result["predictions"]] == ["SF-0"]


def test_sf08_missing_support_fails_closed():
    profile, turns = _sharp_profile([(0, 70.0)])
    result, pairs = _run(profile, [turns[0] - 1.0, turns[0] + 0.25], drop_stations={1, 2, 3})
    _assert_h_one(pairs)
    assert len(pairs) == 1
    assert pairs[0].same_count == 6 and pairs[0].split_count == 0 and pairs[0].pair_class == v66.UNKNOWN
    assert len(result["predictions"]) == 2


def test_sf09_exact_seven_station_boundary():
    profile, turns = _sharp_profile([(0, 70.0)])
    result, pairs = _run(profile, [turns[0] - 1.0, turns[0] + 0.25], drop_stations={1, 9})
    _assert_h_one(pairs)
    assert len(pairs) == 1
    assert pairs[0].same_count == 7 and pairs[0].split_count == 0 and pairs[0].pair_class == v66.POSITIVE
    assert len(result["predictions"]) == 1


def test_sf10_long_flat_face_crossing_is_ineligible():
    profile, turns = _sharp_profile([(0, 70.0)])
    vertices, faces = _extrude(profile)
    point = _point_at_arc(profile, turns[0])
    baseline = _baseline([point])
    baseline["predictions"].append(_row(1, point, axis="y"))
    result, pairs = v66.apply_tftoc_v1(baseline, vertices, faces)
    assert pairs == ()
    assert len(result["predictions"]) == 2


def test_sf11_order_winding_and_replay_invariance():
    profile, turns = _sharp_profile([(0, 70.0)])
    vertices, faces = _extrude(profile)
    points = [_point_at_arc(profile, turns[0] + offset) for offset in (-3.0, 1.0, 4.0)]
    baseline = _baseline(points)
    primary, primary_pairs = v66.apply_tftoc_v1(baseline, vertices, faces)
    replay, replay_pairs = v66.apply_tftoc_v1(baseline, vertices, faces)
    assert primary["tftoc_v1_audit_sha256"] == replay["tftoc_v1_audit_sha256"]
    assert v66.canonical_bytes(primary["tftoc_v1_audit"]) == v66.canonical_bytes(replay["tftoc_v1_audit"])

    face_keys = np.asarray([hashlib.sha256(vertices[face].astype(">f8").tobytes()).hexdigest() for face in faces])
    face_order = np.argsort(face_keys, kind="mergesort")
    variants = [faces[:, ::-1], faces[face_order]]
    vertex_keys = np.asarray([hashlib.sha256(vertex.astype(">f8").tobytes()).hexdigest() for vertex in vertices])
    permutation = np.argsort(vertex_keys, kind="mergesort")
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    variants.append(inverse[faces])
    variant_vertices = [vertices, vertices, vertices[permutation]]
    for current_vertices, current_faces in zip(variant_vertices, variants):
        result, pairs = v66.apply_tftoc_v1(baseline, current_vertices, current_faces)
        assert v66.canonical_bytes(result["tftoc_v1_audit"]) == v66.canonical_bytes(primary["tftoc_v1_audit"])
        assert result["tftoc_v1_audit_sha256"] == primary["tftoc_v1_audit_sha256"]
        assert [(pair.route_hashes, pair.pair_class, pair.same_count, pair.split_count) for pair in pairs] == [
            (pair.route_hashes, pair.pair_class, pair.same_count, pair.split_count) for pair in primary_pairs
        ]
        assert result["tftoc_v1_audit"]["suppressed_hashes"] == primary["tftoc_v1_audit"]["suppressed_hashes"]
        assert result["tftoc_v1_audit"]["mutation_digest"] == primary["tftoc_v1_audit"]["mutation_digest"]


def test_positive_transitive_chain_without_all_pairs_certificate_is_atomic_no_mutation():
    # Near the inclusive 20-degree event floor, the two extreme 4.5 h routes
    # are more than 8 h apart, while each adjacent pair remains eligible and
    # owns the event.
    profile, turns = _sharp_profile([(0, 21.0)])
    result, pairs = _run(profile, [turns[0] - 4.25, turns[0], turns[0] + 4.25])
    assert len(pairs) == 2
    assert all(pair.pair_class == v66.POSITIVE for pair in pairs)
    assert len(result["predictions"]) == 3
    assert result["tftoc_v1_audit"]["cliques"] == []
    assert result["tftoc_v1_audit"]["suppressed_hashes"] == []


def test_point_only_source_is_global_atomic_noop_and_preserves_rows():
    profile, turns = _sharp_profile([(0, 70.0)])
    vertices, _faces = _extrude(profile)
    baseline = _baseline([_point_at_arc(profile, turns[0] - 1.0), _point_at_arc(profile, turns[0] + 0.25)])
    result, pairs = v66.apply_tftoc_v1(baseline, vertices, None)
    assert pairs == ()
    assert result["predictions"] == baseline["predictions"]
    assert result["tftoc_v1_audit"]["status"] == "NO_TRIANGLE_ADJACENCY"
    assert result["tftoc_v1_audit"]["suppressed_hashes"] == []


def test_candidate_cap_is_global_atomic_noop():
    profile, turns = _sharp_profile([(0, 70.0)])
    vertices, faces = _extrude(profile)
    point = _point_at_arc(profile, turns[0])
    baseline = {"predictions": [_row(index, point) for index in range(v66.MAX_CANDIDATES + 1)]}
    result, pairs = v66.apply_tftoc_v1(baseline, vertices, faces)
    assert pairs == ()
    assert result["predictions"] == baseline["predictions"]
    assert result["tftoc_v1_audit"]["status"] == "RESOURCE_ATOMIC_NOOP"
    assert result["tftoc_v1_audit"]["reason"] == "CANDIDATE_CAP"
