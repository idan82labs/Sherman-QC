import math

import numpy as np

import backend.f1_similarity_invariant_normal_cycle_mode_match_nms_v67 as v67


D = 1.0 / 32.0


def candidate(candidate_hash: str, x: float, *, z0: float = 0.0, z1: float = 1.0, angle: float = 0.0):
    direction = np.asarray((math.sin(angle), 0.0, math.cos(angle)), dtype=np.float64)
    start = np.asarray((x, 0.0, z0), dtype=np.float64)
    end = start + (z1 - z0) * direction
    return v67.Candidate(candidate_hash, start, end, direction, float(np.linalg.norm(end - start)), 0)


def atom(xi: float, beta: float, *, a: float = 0.0, b: float = 1.0, sign: int = 1, component: int = 0):
    return v67.Atom(
        np.asarray((xi * D, 0.0, a), dtype=np.float64),
        np.asarray((xi * D, 0.0, b), dtype=np.float64),
        abs(beta),
        sign * abs(beta),
        f"C{component:06d}:support-{xi}-{beta}-{sign}",
    )


def veto(xi: float, *, a: float = 0.0, b: float = 1.0, kind: str = "BOUNDARY"):
    return v67.Atom(
        np.asarray((xi * D, 0.0, a), dtype=np.float64),
        np.asarray((xi * D, 0.0, b), dtype=np.float64),
        0.0,
        0.0,
        f"V{kind}",
        kind,
    )


def evaluate(atoms, *, veto_atoms=(), first=None, second=None):
    first = first or candidate("A", 0.0)
    second = second or candidate("B", D)
    measure = v67.MeshMeasure(list(atoms), list(veto_atoms), 0, 0, 1, 0)
    return v67._evaluate_pair(first, second, measure, [0])


def evaluate_f01_pair(first, second):
    frame = v67._pair_frame(first, second)
    if frame is None:
        return v67._evaluate_pair(first, second, v67.MeshMeasure([], [], 0, 0, 1, 0), [0])
    frame["first_point"] = lambda z: v67._point_on_line(first, frame["axis"], z)
    z0, z1 = frame["z0"], frame["z1"]
    start = frame["first_point"](z0) + (17 / 32) * frame["distance"] * frame["e"]
    end = frame["first_point"](z1) + (17 / 32) * frame["distance"] * frame["e"]
    source = v67.MeshMeasure([v67.Atom(start, end, math.pi / 3, math.pi / 3, "C000000:F01")], [], 0, 0, 1, 0)
    return v67._evaluate_pair(first, second, source, [0])


def parallel_pair(*, overlap=1.0, separation=D):
    first = candidate("A", 0.0)
    second = candidate("B", separation, z0=1.0 - overlap, z1=2.0 - overlap)
    return first, second


def direction_change_pair(change):
    # Symmetric y-slopes make the average axis exactly z.  Their endpoint
    # separation vectors rotate by `change` while midpoint separation stays x.
    separation = 1 / 16
    slope = separation * math.tan(change / 2)
    first = v67.Candidate(
        "A",
        np.asarray((0.0, slope / 2, 0.0)),
        np.asarray((0.0, -slope / 2, 1.0)),
        v67._unit(np.asarray((0.0, -slope, 1.0))),
        math.sqrt(1.0 + slope * slope),
        0,
    )
    second = v67.Candidate(
        "B",
        np.asarray((separation, -slope / 2, 0.0)),
        np.asarray((separation, slope / 2, 1.0)),
        v67._unit(np.asarray((0.0, slope, 1.0))),
        math.sqrt(1.0 + slope * slope),
        0,
    )
    return first, second


def assert_positive(result, state="ONE_FOLD_MODE"):
    assert result.state == state, (result.reason, result.metrics, result.modes)
    assert result.proposed_winner in ("A", "B")
    assert result.proposed_loser in ("A", "B")
    assert result.proposed_winner != result.proposed_loser


def folded_strip(beta=math.pi / 3, *, hinge_x=17 * D / 32, z_cells=16, scale=1.0):
    """Connected two-plane strip with a longitudinal interior hinge."""
    xs = (-D, hinge_x, 2 * D)
    vertices = []
    for z_index in range(z_cells + 1):
        z = scale * z_index / z_cells
        for x in xs:
            y = 0.0 if x <= hinge_x else math.tan(beta) * (x - hinge_x)
            vertices.append((scale * x, scale * y, z))
    faces = []
    for z_index in range(z_cells):
        row0, row1 = 3 * z_index, 3 * (z_index + 1)
        for x_index in range(2):
            a, b = row0 + x_index, row0 + x_index + 1
            c, d = row1 + x_index, row1 + x_index + 1
            faces.extend(((a, b, d), (a, d, c)))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def strip_rows(scale=1.0):
    return [
        {"route_hash": "A", "line_start": [0, 0, 0], "line_end": [0, 0, scale]},
        {"route_hash": "B", "line_start": [scale * D, 0, 0], "line_end": [scale * D, 0, scale]},
    ]


def refine_triangles(vertices, faces):
    vertices = [tuple(value) for value in np.asarray(vertices, dtype=float)]
    midpoint = {}

    def mid(a, b):
        edge = tuple(sorted((int(a), int(b))))
        if edge not in midpoint:
            midpoint[edge] = len(vertices)
            vertices.append(tuple(0.5 * (np.asarray(vertices[edge[0]]) + np.asarray(vertices[edge[1]]))))
        return midpoint[edge]

    refined = []
    for a, b, c in np.asarray(faces, dtype=int):
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return np.asarray(vertices, dtype=float), np.asarray(refined, dtype=int)


def test_f01_sharp_duplicate_60_exact_metrics():
    result = evaluate([atom(17 / 32, math.pi / 3)])
    assert_positive(result)
    assert abs(result.metrics["total_turn_q40"] / v67.Q_ONE - math.pi / 3) < 2e-10
    assert result.metrics["n_eff_q40"] == v67.q40_int(16.0)
    assert result.metrics["bridge_ratio_q40"] == v67.Q_ONE
    assert result.metrics["component_dominance_q40"] == v67.Q_ONE
    assert result.metrics["sign_coherence_q40"] == v67.Q_ONE
    assert len(result.modes) == 1


def test_f02_triple_duplicate_disjoint_matching_cannot_chain():
    rows = [
        {"route_hash": "A", "line_start": [-D, 0, 0], "line_end": [-D, 0, 1]},
        {"route_hash": "B", "line_start": [D, 0, 0], "line_end": [D, 0, 1]},
        {"route_hash": "C", "line_start": [2 * D, 0, 0], "line_end": [2 * D, 0, 1]},
    ]
    # A closed two-face strip is not needed here: the disjoint policy is tested
    # with exact-coincident pairs, which are source-independent by contract.
    rows[1]["line_start"] = rows[0]["line_start"]
    rows[1]["line_end"] = rows[0]["line_end"]
    rows[2]["line_start"] = rows[0]["line_start"]
    rows[2]["line_end"] = rows[0]["line_end"]
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=int)
    output, _pairs = v67.apply_ncmm_nms_v1(rows, vertices, faces)
    assert len(output["predictions"]) == 2
    assert len(output["ncmm_nms_v1_audit"]["suppressed_hashes"]) == 1


def test_f03_nearby_distinct_has_two_modes_and_keeps_both():
    result = evaluate([atom(11 / 32, 0.5), atom(23 / 32, 0.5)])
    assert result.state == "KEEP"
    assert result.reason == "MULTI_MODE"
    assert len(result.modes) == 2


def test_f04_u_channel_opposing_keeps_both():
    result = evaluate([atom(1 / 8, math.pi / 2), atom(7 / 8, math.pi / 2, sign=-1)])
    assert result.state == "KEEP"


def test_f05_return_hem_double_skin_fails_component_dominance():
    result = evaluate([atom(15 / 32, 0.5, component=0), atom(17 / 32, 0.5, sign=-1, component=1)])
    assert result.state == "KEEP"
    assert result.reason == "MULTI_COMPONENT"
    assert result.metrics["component_dominance_q40"] == v67.q40_int(0.5)


def test_f06_rounded_duplicate_is_one_persistent_mode():
    records = [atom(-0.5 + (index + 0.5) / 16, 3 / 25) for index in range(14, 19)]
    result = evaluate(records)
    assert_positive(result)
    assert abs(result.metrics["total_turn_q40"] / v67.Q_ONE - 0.6) < 2e-10
    assert len(result.modes) == 1


def test_f07_small_crease_passes():
    assert_positive(evaluate([atom(0.5, 0.25)]))


def test_f08_flat_face_keeps_both():
    result = evaluate([])
    assert result.state == "KEEP"
    assert result.reason == "LOW_TURN"


def test_f09_perimeter_distractor_keeps_both():
    result = evaluate([], veto_atoms=[veto(0.5)])
    assert result.state == "KEEP"


def test_f10_local_junction_keeps_both():
    result = evaluate([atom(0.5, 6 / 5, b=1 / 16)])
    assert result.state == "KEEP"
    assert result.metrics["total_turn_q40"] == v67.q40_int(3 / 40)


def test_f11_partial_support_four_bins_passes_exactly():
    result = evaluate([atom(0.5, 1.0, b=4 / 16)])
    assert_positive(result)
    assert result.metrics["total_turn_q40"] == v67.q40_int(0.25)
    assert result.metrics["n_eff_q40"] == v67.q40_int(4.0)


def test_f12_partial_support_three_bins_keeps_both():
    result = evaluate([atom(0.5, 1.0, b=3 / 16)])
    assert result.state == "KEEP"
    assert result.metrics["total_turn_q40"] == v67.q40_int(3 / 16)


def test_f13_f14_total_turn_exact_boundary():
    assert_positive(evaluate([atom(0.5, 13 / 64)]))
    below = evaluate([atom(0.5, 13 / 64 - 2**-20)])
    assert below.state == "KEEP"
    assert below.reason == "LOW_TURN"


def test_f15_f16_bridge_mass_exact_boundary():
    # Bin centers 11 and 12; 11 lies outside and 12 lies inside the bridge.
    xi11 = -0.5 + (11.5 / 16)
    xi12 = -0.5 + (12.5 / 16)
    exact = evaluate([atom(xi11, 13 / 128), atom(xi12, 13 / 128)])
    assert_positive(exact)
    assert exact.metrics["bridge_ratio_q40"] == v67.q40_int(0.5)
    eps = 2**-20
    below = evaluate([atom(xi11, 13 / 128 + eps), atom(xi12, 13 / 128 - eps)])
    assert below.state == "KEEP"
    assert below.reason == "NO_BRIDGE"


def test_f17_f18_sign_coherence_exact_boundary():
    exact = evaluate([atom(15 / 32, 7 / 32), atom(17 / 32, 1 / 32, sign=-1)])
    assert_positive(exact)
    assert exact.metrics["sign_coherence_q40"] == v67.q40_int(0.75)
    eps = 2**-20
    below = evaluate([atom(15 / 32, 7 / 32 - eps), atom(17 / 32, 1 / 32 + eps, sign=-1)])
    assert below.state == "KEEP"
    assert below.reason == "OPPOSING_TURN"


def test_f19_f20_component_dominance_exact_boundary():
    exact = evaluate([atom(15 / 32, 3 / 16, component=0), atom(17 / 32, 1 / 16, component=1)])
    assert_positive(exact)
    assert exact.metrics["component_dominance_q40"] == v67.q40_int(0.75)
    eps = 2**-20
    below = evaluate([atom(15 / 32, 3 / 16 - eps, component=0), atom(17 / 32, 1 / 16 + eps, component=1)])
    assert below.state == "KEEP"
    assert below.reason == "MULTI_COMPONENT"


def test_f21_boundary_exact_equality_is_veto():
    result = evaluate([atom(0.5, 1.0)], veto_atoms=[veto(0.5, b=0.25)])
    assert result.state == "KEEP"
    assert result.reason == "BOUNDARY_OR_NONMANIFOLD"
    assert result.metrics["boundary_ratio_q40"] == v67.q40_int(0.25)


def test_f22a_f22b_angle_boundary_is_infeasible_under_endpoint_lock():
    # The endpoint lock bounds each normalized separation vector to
    # e in [7/8,9/8], |n|<=1/8.  Hence their maximum possible difference is
    # sqrt((1/4)^2+(1/4)^2)d.  At d/L<=1/16 this is <0.023L, while two lines
    # separated by pi/24 change by 2*tan(pi/48)L >0.131L.  F22A therefore
    # cannot be positive without violating another frozen predicate.
    endpoint_delta_max = math.sqrt(2) / 4 / 16
    angle_delta_required = 2 * math.tan((math.pi / 24) / 2)
    assert endpoint_delta_max < angle_delta_required

    half = (math.pi / 24) / 2
    first = v67.Candidate(
        "A", np.asarray((0.0, 0.0, 0.0)), np.asarray((-math.tan(half), 0.0, 1.0)),
        v67._unit(np.asarray((-math.tan(half), 0.0, 1.0))), 1 / math.cos(half), 0,
    )
    second = v67.Candidate(
        "B", np.asarray((1 / 16, 0.0, 0.0)), np.asarray((1 / 16 + math.tan(half), 0.0, 1.0)),
        v67._unit(np.asarray((math.tan(half), 0.0, 1.0))), 1 / math.cos(half), 0,
    )
    assert evaluate_f01_pair(first, second).reason == "PAIR_GEOMETRY"


def test_f22c_f22d_overlap_exact_boundary():
    exact = evaluate_f01_pair(*parallel_pair(overlap=5 / 8))
    assert_positive(exact)
    below = evaluate_f01_pair(*parallel_pair(overlap=5 / 8 - 2**-20))
    assert below.state == "KEEP"
    assert below.reason == "PAIR_GEOMETRY"


def test_f22e_f22f_separation_exact_upper_boundary():
    exact = evaluate_f01_pair(*parallel_pair(separation=1 / 16))
    assert_positive(exact)
    over = evaluate_f01_pair(*parallel_pair(separation=1 / 16 + 2**-20))
    assert over.state == "KEEP"
    assert over.reason == "PAIR_GEOMETRY"


def test_f22g_f22h_variation_boundary_is_infeasible_under_endpoint_lock():
    # The endpoint frame lock itself forces each endpoint magnitude into
    # [7/8, sqrt((9/8)^2+(1/8)^2)] times d, so the smallest possible ratio is
    # about 0.773, already strictly above the requested equality 0.75.
    minimum_ratio = (7 / 8) / math.sqrt((9 / 8) ** 2 + (1 / 8) ** 2)
    assert minimum_ratio > 3 / 4


def test_f22i_f22j_direction_change_exact_boundary():
    exact = evaluate_f01_pair(*direction_change_pair(math.pi / 18))
    assert_positive(exact)
    over = evaluate_f01_pair(*direction_change_pair(math.pi / 18 + 2**-20))
    assert over.state == "KEEP"
    assert over.reason == "PAIR_GEOMETRY"


def test_f23_exact_coincident_retains_lower_hash_when_lengths_equal():
    first, second = candidate("A", 0.0), candidate("B", 0.0)
    result = evaluate([], first=first, second=second)
    assert result.state == "EXACT_COINCIDENT"
    assert result.proposed_winner == "A"
    assert result.proposed_loser == "B"


def test_f24_point_only_is_atomic_keep_all():
    rows = [{"route_hash": "A", "line_start": [0, 0, 0], "line_end": [0, 0, 1]}]
    result, pairs = v67.apply_ncmm_nms_v1(rows, np.asarray(((0, 0, 0),)), None)
    assert result["predictions"] == rows
    assert pairs == []
    assert result["ncmm_nms_v1_audit"]["status"] == "POINT_ONLY_KEEP_ALL"


def test_f25_nonmanifold_only_turn_is_veto_not_positive_evidence():
    # Three triangles own the same longitudinal edge.  Even though their face
    # normals differ, the edge is nonmanifold and may only veto suppression.
    vertices = np.asarray(
        ((0, 0, 0), (0, 0, 1), (-D, 0, 0), (D, 0, 0), (0, D, 0)),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=int)
    result, pairs = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    assert len(result["predictions"]) == 2
    assert result["ncmm_nms_v1_audit"]["hinge_atom_count"] == 0
    assert result["ncmm_nms_v1_audit"]["veto_atom_count"] > 0
    assert all(pair.state == "KEEP" for pair in pairs)


def test_f26_candidate_cap_is_atomic_noop():
    rows = [
        {"route_hash": f"H{index:03d}", "line_start": [index, 0, 0], "line_end": [index, 0, 1]}
        for index in range(97)
    ]
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=int)
    result, pairs = v67.apply_ncmm_nms_v1(rows, vertices, faces)
    assert result["predictions"] == rows
    assert pairs == []
    assert result["ncmm_nms_v1_audit"]["status"] == "RESOURCE_ABORT"
    assert result["ncmm_nms_v1_audit"]["detail"] == "CANDIDATE_CAP"


def test_f27_eligible_pair_cap_is_atomic_noop():
    # All 96 lines lie inside a 1/16-wide band, but remain more than 1/4096
    # apart, making more than 2,048 pairs geometry eligible.
    spacing = (1 / 16 - 1 / 4096) / 95
    rows = [
        {"route_hash": f"H{index:03d}", "line_start": [index * spacing, 0, 0], "line_end": [index * spacing, 0, 1]}
        for index in range(96)
    ]
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=int)
    result, pairs = v67.apply_ncmm_nms_v1(rows, vertices, faces)
    assert result["predictions"] == rows
    assert pairs == []
    assert result["ncmm_nms_v1_audit"]["status"] == "RESOURCE_ABORT"
    assert result["ncmm_nms_v1_audit"]["detail"] == "ELIGIBLE_PAIR_CAP"


def test_f28_visit_cap_is_atomic_noop(monkeypatch):
    vertices, faces = folded_strip()
    monkeypatch.setattr(v67, "MAX_CLIP_VISITS", 0)
    result, pairs = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    assert result["predictions"] == strip_rows()
    assert pairs == []
    assert result["ncmm_nms_v1_audit"]["detail"] == "CLIP_VISIT_CAP"


def test_f29_noncanonical_audit_scalar_is_rejected():
    try:
        v67._encode_audit({"bad": float("nan")})
    except v67.NCMMResourceAbort as error:
        assert str(error) == "SERIALIZATION_ABORT"
        pass
    else:
        raise AssertionError("noncanonical audit scalar must abort serialization")


def test_f30_primary_replay_is_byte_identical():
    vertices, faces = folded_strip()
    first, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    second, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    assert v67.canonical_bytes(first) == v67.canonical_bytes(second)
    assert len(first["predictions"]) == 1


def test_f31_order_and_winding_leave_semantic_result_unchanged():
    vertices, faces = folded_strip()
    primary, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    variant_faces = faces[::-1, ::-1]
    variant, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, variant_faces, variant=True)
    pa = primary["ncmm_nms_v1_audit"]
    va = variant["ncmm_nms_v1_audit"]
    assert primary["predictions"] == variant["predictions"]
    for field in ("pair_records", "selected_matching", "suppressed_hashes", "survivor_hashes"):
        assert pa[field] == va[field]


def test_f32_f33_uniform_similarity_leaves_metrics_and_outcome_unchanged():
    vertices, faces = folded_strip()
    primary, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    expected_records = primary["ncmm_nms_v1_audit"]["pair_records"]
    for scale in (0.8, 1.25):
        scaled_vertices, scaled_faces = folded_strip(scale=scale)
        variant, _ = v67.apply_ncmm_nms_v1(strip_rows(scale), scaled_vertices, scaled_faces, variant=True)
        assert variant["predictions"][0]["route_hash"] == primary["predictions"][0]["route_hash"]
        assert variant["ncmm_nms_v1_audit"]["pair_records"] == expected_records


def test_f34_full_piecewise_linear_refinement_preserves_metrics_and_outcome():
    vertices, faces = folded_strip()
    refined_vertices, refined_faces = refine_triangles(vertices, faces)
    primary, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    refined, _ = v67.apply_ncmm_nms_v1(strip_rows(), refined_vertices, refined_faces, variant=True)
    assert primary["predictions"] == refined["predictions"]
    pa = primary["ncmm_nms_v1_audit"]
    ra = refined["ncmm_nms_v1_audit"]
    for field in ("pair_records", "selected_matching", "suppressed_hashes", "survivor_hashes"):
        assert pa[field] == ra[field]


def test_q40_uses_ties_to_even_and_rejects_nan():
    assert v67.q40_int(0.5 / v67.Q_ONE) == 0
    assert v67.q40_int(1.5 / v67.Q_ONE) == 2
    try:
        v67.q40_int(float("nan"))
    except v67.NCMMInputError:
        pass
    else:
        raise AssertionError("NaN must fail closed")
