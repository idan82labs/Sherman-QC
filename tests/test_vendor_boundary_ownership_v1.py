"""Pure coefficient/graph fixtures: no OCP, server, files or physical inputs."""
from dataclasses import replace

import numpy as np
import pytest

from backend.vendor_boundary_ownership_v1 import (
    BoundaryOwnershipUnavailable, Budget, Curve, EdgeUse, Face, SkinPair,
    WallPatch, certify_boundary_ownership,
)


def _fixture(*, split=False, varying=True, transform=None, offset=None, outline=None, holes=()):
    if outline is None:
        outline = [[0., 0., 0.], [4., 0., 0.], [4., 3., 0.], [0., 3., 0.]]
    loops = [np.asarray(loop, dtype=float) for loop in (outline, *holes)]
    vertices = np.concatenate(loops)
    count = len(vertices)
    successors, loop_indices = {}, []
    start = 0
    for loop in loops:
        indices = list(range(start, start + len(loop)))
        loop_indices.append(indices)
        successors.update({i: indices[(j + 1) % len(indices)] for j, i in enumerate(indices)})
        start += len(loop)
    transform = np.eye(3) if transform is None else transform
    offset = np.zeros(3) if offset is None else offset
    normal = transform @ np.array([0., 0., 1.])
    uk, vk = (0., 0., 0., 1., 1., 1.), (0., 0., 0., 0., 1., 1., 1., 1.)
    weights = np.repeat(np.array([1., .7, 1.])[:, None], 4, axis=1)
    levels = [0., .5, 1.] if split else [0., 1.]
    curves, faces, patches = {}, [], []
    # Disjoint native ID spaces for horizontal and vertical complete rails.
    vertical_start = len(levels) * count
    for level_index, level in enumerate(levels):
        for side, a in enumerate(vertices):
            b = vertices[successors[side]]
            p = np.stack([a, a + .4 * (b - a), b]) + np.array([0., 0., level])
            curves[level_index * count + side] = Curve(2, uk, p @ transform.T + offset, weights[:, 0])
    for layer, (low, high) in enumerate(zip(levels, levels[1:])):
        for corner, vertex in enumerate(vertices):
            p = vertex + np.outer(low + (high - low) * np.array([0., .2, .8, 1.]), [0., 0., 1.])
            curves[vertical_start + layer * count + corner] = Curve(3, vk, p @ transform.T + offset, np.ones(4))
        for side, a in enumerate(vertices):
            b = vertices[successors[side]]
            base = np.stack([a, a + .4 * (b - a), b])
            scalar = np.repeat(np.array([[0., .2, .8, 1.]]), 3, axis=0)
            if varying:
                scalar[1] = [0., .1, .5, 1.]
            p = base[:, None] + np.expand_dims(low + (high - low) * scalar, 2) * [0., 0., 1.]
            rails = (EdgeUse(layer * count + side, 1),
                     EdgeUse(vertical_start + layer * count + successors[side], 1),
                     EdgeUse((layer + 1) * count + side, 1),
                     EdgeUse(vertical_start + layer * count + side, 1))
            face_id = 2 + layer * count + side
            uses = tuple(EdgeUse(u.edge_id, u.direction * sign) for u, sign in zip(rails, (1, 1, -1, -1)))
            faces.append(Face(face_id, uses))
            patches.append(WallPatch(face_id, 0, 2, 3, uk, vk,
                                     p @ transform.T + offset, weights.copy(), rails))
    bottom_wires = tuple(tuple(EdgeUse(i, -1) for i in reversed(indices)) for indices in loop_indices)
    top_wires = tuple(tuple(EdgeUse((len(levels) - 1) * count + i, 1) for i in indices)
                      for indices in loop_indices)
    bottom = tuple(u for wire in bottom_wires for u in wire)
    top = tuple(u for wire in top_wires for u in wire)
    faces = [Face(0, bottom, bottom_wires), Face(1, top, top_wires)] + faces
    pair = SkinPair(0, 0, 1, offset.copy(), normal, 1.)
    return faces, [pair], patches, curves


def _prove(fixture, **kwargs):
    return certify_boundary_ownership(*fixture, **kwargs)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("varying", [False, True])
def test_complete_inventory_proves_variable_generator_speed_and_split_walls(split, varying):
    result = _prove(_fixture(split=split, varying=varying))
    assert result["scope"] == "COMPLETE_POLYGONAL_SINGLE_SLAB"
    assert result["native_face_count"] == (10 if split else 6)
    assert len(result["wall_certificates"]) == (8 if split else 4)
    assert result["manufacturing_blank_established"] is False
    assert "EXACT_FULL_CHART_TRIMS" in result["native_adapter_required"]
    assert set(result["source_face_roles"].values()) == {
        "LOWER_PLANE_SKIN", "UPPER_PLANE_SKIN", "CERTIFIED_NORMAL_CUT_WALL"}


def test_pose_invariance_with_nonzero_plane_offset():
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3) + np.sin(.8) * cross + (1 - np.cos(.8)) * cross @ cross
    result = _prove(_fixture(split=True, transform=rotation, offset=np.array([11., -4., 37.])))
    assert max(p["normal_fiber_error_bound"] for p in result["wall_certificates"]) < 1e-12


@pytest.mark.parametrize("fault,reason", [
    ("warp", "UNPROVEN_NORMAL_FIBERS"),
    ("weights", "COUPLED_RATIONAL_WALL_WEIGHTS"),
    ("fold", "NONMONOTONE_OR_SINGULAR_NORMAL_FIBERS"),
    ("endpoint", "VARYING_PHYSICAL_GENERATOR_ENDPOINTS"),
    ("outside", "WALL_OUTSIDE_THICKNESS_SLAB"),
    ("curved", "NONPOLYGONAL_OR_FOLDING_BOUNDARY_SUPPORT"),
])
def test_wall_coefficient_failures_never_become_ignored_material(fault, reason):
    faces, pairs, patches, curves = _fixture()
    p, w = patches[0].poles.copy(), patches[0].weights.copy()
    if fault == "warp": p[1, 1, 1] += .01
    if fault == "weights": w[1, 1] *= .8
    if fault == "fold": p[1, 2, 2] = .05
    if fault == "endpoint": p[1, -1, 2] = .9
    if fault == "outside": p[:, -1, 2] = 1.1
    if fault == "curved": p[1, :, 1] = .1
    patches[0] = replace(patches[0], poles=p, weights=w)
    with pytest.raises(BoundaryOwnershipUnavailable, match=reason):
        _prove((faces, pairs, patches, curves))


def test_native_rail_cannot_be_forged_from_matching_endpoints():
    fixture = _fixture()
    native = fixture[3][0]
    p = native.poles.copy(); p[1, 1] = .01
    fixture[3][0] = replace(native, poles=p)
    with pytest.raises(BoundaryOwnershipUnavailable, match="NATIVE_RAIL_GEOMETRY_MISMATCH"):
        _prove(fixture)


def test_native_curve_weights_cannot_be_silently_replaced():
    fixture = _fixture()
    native = fixture[3][0]
    fixture[3][0] = replace(native, weights=np.ones(3))
    with pytest.raises(BoundaryOwnershipUnavailable, match="NATIVE_RAIL_BASIS_MISMATCH"):
        _prove(fixture)


def test_face_without_coefficient_ownership_remains_unexplained():
    faces, pairs, patches, curves = _fixture()
    with pytest.raises(BoundaryOwnershipUnavailable, match="UNEXPLAINED_NATIVE_MATERIAL_SURFACE"):
        _prove((faces, pairs, patches[:-1], curves))


@pytest.mark.parametrize("fault,reason", [
    ("open", "OPEN_NONMANIFOLD_OR_INCOHERENT_EDGE_INCIDENCE"),
    ("parity", "OPEN_NONMANIFOLD_OR_INCOHERENT_EDGE_INCIDENCE"),
    ("repeat", "REPEATED_OR_INCOMPLETE_NATIVE_EDGE_USES"),
    ("wires", "SKIN_WIRE_COVERAGE_MISMATCH"),
    ("partial", "PARTIAL_OR_NONRECTANGULAR_WALL_TRIM"),
])
def test_complete_native_incidence_and_wires_are_mandatory(fault, reason):
    faces, pairs, patches, curves = _fixture()
    if fault == "open":
        faces[0] = replace(faces[0], uses=faces[0].uses[:-1])
    if fault == "parity":
        uses = list(faces[0].uses); uses[0] = replace(uses[0], direction=1)
        faces[0] = replace(faces[0], uses=tuple(uses))
    if fault == "repeat":
        faces[0] = replace(faces[0], uses=faces[0].uses + (faces[0].uses[0],))
    if fault == "wires":
        faces[0] = replace(faces[0], wires=(faces[0].wires[0][:-1],))
    if fault == "partial":
        patches[0] = replace(patches[0], rails=patches[0].rails[:3])
    with pytest.raises(BoundaryOwnershipUnavailable, match=reason):
        _prove((faces, pairs, patches, curves))


def test_stepped_hole_style_skin_footprint_is_not_uniform_cut_wall():
    faces, pairs, patches, curves = _fixture()
    # A shoulder requires extra source faces; it cannot be hidden as a skin rail.
    p = patches[0].poles.copy(); p[:, -1, 1] += .2
    patches[0] = replace(patches[0], poles=p)
    with pytest.raises(BoundaryOwnershipUnavailable, match="UNPROVEN_NORMAL_FIBERS"):
        _prove((faces, pairs, patches, curves))


@pytest.mark.parametrize("budget,reason", [
    (Budget(max_faces=3), "BOUNDARY_INVENTORY_BUDGET_EXCEEDED"),
    (Budget(max_edges=2), "BOUNDARY_INVENTORY_BUDGET_EXCEEDED"),
    (Budget(max_poles=2), "BOUNDARY_POLE_BUDGET_EXCEEDED"),
    (Budget(max_pairwise_checks=1), "BOUNDARY_INTERSECTION_BUDGET_EXCEEDED"),
    (Budget(tolerance=float("nan")), "INVALID_PROOF_BUDGET"),
])
def test_fixed_proof_budgets_refuse(budget, reason):
    with pytest.raises(BoundaryOwnershipUnavailable, match=reason):
        _prove(_fixture(), budget=budget)


def test_negative_and_nonfinite_weights_refuse():
    for invalid in (-1., float("nan")):
        fixture = _fixture()
        p = fixture[2][0]; weights = p.weights.copy(); weights[1, 1] = invalid
        fixture[2][0] = replace(p, weights=weights)
        with pytest.raises(BoundaryOwnershipUnavailable, match="INVALID_WALL_COEFFICIENTS"):
            _prove(fixture)


def test_degree_or_unclamped_basis_refuses_before_geometric_guessing():
    fixture = _fixture()
    fixture[2][0] = replace(fixture[2][0], u_knots=(-1., 0., 0., 1., 1., 1.))
    with pytest.raises(BoundaryOwnershipUnavailable, match="FULL_CLAMPED_CHART_REQUIRED"):
        _prove(fixture)


def test_source_input_arrays_are_not_modified():
    fixture = _fixture(split=True)
    before = [p.poles.copy() for p in fixture[2]]
    _prove(fixture)
    for a, p in zip(before, fixture[2]):
        assert np.array_equal(a, p.poles)


def test_native_edge_parameter_reversal_preserves_certificate():
    faces, pairs, patches, curves = _fixture(split=True)
    edge_id = 0
    c = curves[edge_id]
    knots = np.asarray(c.knots)
    curves[edge_id] = Curve(c.degree, tuple(knots[0] + knots[-1] - knots[::-1]),
                            c.poles[::-1].copy(), c.weights[::-1].copy())
    def flip(use):
        return replace(use, direction=-use.direction) if use.edge_id == edge_id else use
    faces = [replace(f, uses=tuple(map(flip, f.uses)),
                     wires=tuple(tuple(map(flip, wire)) for wire in f.wires)) for f in faces]
    patches = [replace(p, rails=tuple(map(flip, p.rails))) for p in patches]
    assert _prove((faces, pairs, patches, curves))["native_face_count"] == 10


def test_complete_polygonal_hole_is_retained_not_filled():
    hole = [[1., 1., 0.], [1., 2., 0.], [2., 2., 0.], [2., 1., 0.]]
    result = _prove(_fixture(holes=(hole,), split=True))
    assert result["native_face_count"] == 18
    assert len(result["wall_certificates"]) == 16


@pytest.mark.parametrize("outline,reason", [
    ([[0., 0., 0.], [4., 3., 0.], [0., 3., 0.], [4., 0., 0.]],
     "SELF_INTERSECTING_OR_UNSEPARATED_BOUNDARY"),
    ([[0., 0., 0.], [4., 0., 0.], [2., 0., 0.]],
     "BACKTRACKING_NATIVE_BOUNDARY"),
])
def test_topological_cycle_without_geometric_simple_boundary_refuses(outline, reason):
    with pytest.raises(BoundaryOwnershipUnavailable, match=reason):
        _prove(_fixture(outline=outline))


@pytest.mark.parametrize("hole,reason", [
    ([[5., 1., 0.], [5., 2., 0.], [6., 2., 0.], [6., 1., 0.]],
     "HOLE_OUTSIDE_OUTER_WIRE"),
    ([[1., 1., 0.], [2., 1., 0.], [2., 2., 0.], [1., 2., 0.]],
     "INVALID_NATIVE_WIRE_ORIENTATION"),
    ([[3., 1., 0.], [3., 2., 0.], [5., 2., 0.], [5., 1., 0.]],
     "SELF_INTERSECTING_OR_UNSEPARATED_BOUNDARY"),
])
def test_hole_location_orientation_and_intersection_are_not_adjacency_guesses(hole, reason):
    with pytest.raises(BoundaryOwnershipUnavailable, match=reason):
        _prove(_fixture(holes=(hole,)))


def test_duplicate_wall_patch_cannot_claim_complete_interval_twice():
    faces, pairs, patches, curves = _fixture()
    with pytest.raises(BoundaryOwnershipUnavailable, match="DUPLICATE_OR_MISSING_WALL_OWNERSHIP"):
        _prove((faces, pairs, patches + [patches[0]], curves))


def test_extreme_rational_weights_are_bounded():
    fixture = _fixture()
    native = fixture[3][0]
    fixture[3][0] = replace(native, weights=np.array([1e-300, 1., 1.]))
    with pytest.raises(BoundaryOwnershipUnavailable, match="INVALID_NATIVE_CURVE_COEFFICIENTS"):
        _prove(fixture)


def test_mirror_with_native_face_orientation_transport_preserves_ownership():
    faces, pairs, patches, curves = _fixture(transform=np.diag([-1., 1., 1.]))
    faces = [replace(f, uses=tuple(replace(u, direction=-u.direction) for u in f.uses),
                     wires=tuple(tuple(replace(u, direction=-u.direction) for u in reversed(wire))
                                 for wire in f.wires)) for f in faces]
    assert _prove((faces, pairs, patches, curves))["native_face_count"] == 6


def test_global_inversion_of_solid_orientation_is_not_accepted():
    faces, pairs, patches, curves = _fixture()
    faces = [replace(f, uses=tuple(replace(u, direction=-u.direction) for u in f.uses),
                     wires=tuple(tuple(replace(u, direction=-u.direction) for u in reversed(wire))
                                 for wire in f.wires)) for f in faces]
    with pytest.raises(BoundaryOwnershipUnavailable, match="INVERTED_PAIRED_SKIN_ORIENTATION"):
        _prove((faces, pairs, patches, curves))
