"""Independent numeric charts only; no native CAD imports or file experiments.

These authored face/edge tables exercise the pure graph/chart seam. They are
not imported solids and therefore do not establish native STEP validity,
continuous trim material, physical units, or manufacturing qualification.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from backend import vendor_unfold_finite_width as width
from backend import vendor_unfold_preview as v3
from backend import vendor_unfold_preview_v4 as v4


HASH = "a"*64
RECT_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def plane(index, points, normal, edges):
    points = np.array(points, dtype=float)
    return v3.SkinFace(index, "plane", points, RECT_TRIANGLES.copy(),
                       edges=copy.deepcopy(edges), origin=points[0].copy(),
                       normal=np.array(normal, dtype=float))


def cylinder(index, radius, edges, center=None, rotation=None):
    center = np.zeros(3) if center is None else np.array(center, dtype=float)
    rotation = np.eye(3) if rotation is None else rotation
    angles = [j*math.pi/16 for j in range(9)]
    points = np.array([[radius*math.cos(u), radius*math.sin(u), z]
                       for u in angles for z in (0, 12)]) @ rotation.T+center
    triangles = np.array([row for j in range(8) for row in
                          ([2*j, 2*j+2, 2*j+3], [2*j, 2*j+3, 2*j+1])], dtype=np.int32)
    return v3.SkinFace(index, "cylinder", points, triangles, edges=copy.deepcopy(edges),
        origin=center, axis=rotation @ np.array([0., 0., 1.]),
        x_axis=rotation @ np.array([1., 0., 0.]), y_axis=rotation @ np.array([0., 1., 0.]),
        radius=radius, u0=0., u1=math.pi/2, v0=0., v1=12.)


def axis_and_candidate(indices, midpoint, feature):
    line = [np.array([midpoint[0], midpoint[1], 0.]), np.array([midpoint[0], midpoint[1], 12.])]
    return ({"face_indices": [i+1 for i in indices], "radii_mm": [4., 5.],
             "canonical_line_start": line[0].tolist(), "canonical_line_end": line[1].tolist()},
            {"feature_id": feature, "line_start": line[0].tolist(), "line_end": line[1].tolist()})


def quarter(parent_depth=30., child_depth=10.):
    faces = []
    for offset, radius, edge in [(0, 4., 10), (3, 5., 20)]:
        start = np.array([[radius, 0., 0.], [radius, 0., 12.]])
        end = np.array([[0., radius, 0.], [0., radius, 12.]])
        faces.append(plane(offset, [[radius, -parent_depth, 0], [radius, 0, 0],
                                   [radius, 0, 12], [radius, -parent_depth, 12]],
                           [1, 0, 0], {edge: start}))
        faces.append(cylinder(offset+1, radius, {edge: start, edge+1: end}))
        faces.append(plane(offset+2, [[0, radius, 0], [-child_depth, radius, 0],
                                     [-child_depth, radius, 12], [0, radius, 12]],
                           [0, 1, 0], {edge+1: end}))
    axis, candidate = axis_and_candidate((1, 4), [4.5/math.sqrt(2)]*2, "CAD-native-A")
    return faces, [axis], [candidate]


def add_parent_tab(faces, extent=9.):
    for index in (0, 3):
        face = faces[index]
        face.vertices[2:, 2] = 18
        offset = len(face.vertices)
        r = face.origin[0]
        face.vertices = np.vstack((face.vertices,
            [[r, 0, 14], [r, extent, 14], [r, extent, 18], [r, 0, 18]]))
        face.triangles = np.vstack((face.triangles, RECT_TRIANGLES+offset))


def chain():
    faces, axes, candidates = quarter()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([-10., 0., 0.])
    for parent, radius, index, edge in [(2, 4., 6, 30), (5, 5., 8, 40)]:
        start = np.array([center+[0, radius, 0], center+[0, radius, 12]])
        end = np.array([center+[-radius, 0, 0], center+[-radius, 0, 12]])
        faces[parent].edges[edge] = start
        faces.append(cylinder(index, radius, {edge: start, edge+1: end}, center, rotation))
        faces.append(plane(index+1, [[-10-radius, 0, 0], [-10-radius, -7, 0],
                                    [-10-radius, -7, 12], [-10-radius, 0, 12]],
                           [-1, 0, 0], {edge+1: end}))
    axis, candidate = axis_and_candidate((6, 8), [-10-4.5/math.sqrt(2), 4.5/math.sqrt(2)], "CAD-native-Z")
    axes.append(axis)
    candidates.append(candidate)
    return faces, axes, candidates


def posed(case, mirrored=False):
    faces, axes, candidates = copy.deepcopy(case)
    # Independent Rodrigues construction; do not use the engine's rotation.
    a = np.array([2., -3., 5.]); a /= np.linalg.norm(a)
    angle = .91
    skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rotation = np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(a, a)+math.sin(angle)*skew
    if mirrored:
        rotation = rotation @ np.diag([-1., 1., 1.])
    shift = np.array([73., -21., 44.])
    for face in faces:
        face.vertices = face.vertices @ rotation.T+shift
        face.origin = rotation @ face.origin+shift
        face.edges = {key: points @ rotation.T+shift for key, points in face.edges.items()}
        if face.kind == "plane":
            face.normal = rotation @ face.normal
        else:
            face.axis, face.x_axis, face.y_axis = [rotation @ vector for vector in
                                                  (face.axis, face.x_axis, face.y_axis)]
            if mirrored:
                face.axis = -face.axis
                face.v0, face.v1 = -face.v1, -face.v0
    for row in axes+candidates:
        for key in ("line_start", "line_end", "canonical_line_start", "canonical_line_end"):
            if key in row:
                row[key] = (rotation @ np.array(row[key])+shift).tolist()
    return faces, axes, candidates


def flat_points(payload):
    return np.array(payload["vertices"]).reshape(-1, 3)


def test_quarter_uses_same_native_charts_with_explicit_v4_declarations():
    case = quarter()
    original = v3._develop_faces(*copy.deepcopy(case), HASH)
    payload = v4._develop_faces(*case, HASH)
    assert payload["status"] == "AVAILABLE"
    assert payload["vertices"] == pytest.approx(original["vertices"])
    assert payload["triangles"] == original["triangles"]
    assert payload["bend_lines"] == original["bend_lines"]
    assert np.ptp(flat_points(payload)[:, 1]) == pytest.approx(40+2*math.pi)
    diagnostics = payload["diagnostics"]
    assert diagnostics["mechanism_version"] == 4
    assert diagnostics["finite_width_model"] == "NATIVE_TESSELLATION_FINITE_AXIAL_SLAB"
    assert diagnostics["finite_width_clipping_checked"] is True
    assert diagnostics["finite_width_native_seam_span_checked"] is True
    assert diagnostics["whole_unchanged_mesh_overlap_checked"] is True
    assert diagnostics["finite_width_work"]["triangles"] == 4
    assert not payload["units_verified"] and not payload["manufacturing_ready"]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("mirrored", [False, True])
def test_posed_mirrored_chart_preserves_distances_without_reflected_maps(mirrored):
    posed_case = posed(quarter(), mirrored)
    payload = v4._develop_faces(*posed_case, HASH)
    radius = posed_case[0][payload["diagnostics"]["selected_cylinder_face_indices"][0]].radius
    reference = quarter()
    # V3's exact largest-area policy can break an equal-area skin tie differently
    # after a rigid pose. Match the selected radius, not a neutral-axis fantasy.
    if radius == 5:
        reference[0][:] = reference[0][3:]+reference[0][:3]
        for index, face in enumerate(reference[0]): face.index = index
    baseline = v4._develop_faces(*reference, HASH)
    legacy = v3._develop_faces(*copy.deepcopy(posed_case), HASH)
    assert payload["vertices"] == pytest.approx(legacy["vertices"])
    assert np.ptp(flat_points(payload)[:, 1]) == pytest.approx(40+radius*math.pi/2)
    a, b = flat_points(baseline), flat_points(payload)
    assert np.linalg.norm(a[:, None]-a, axis=2) == pytest.approx(np.linalg.norm(b[:, None]-b, axis=2), abs=1e-8)
    assert np.all(b[:, 2] == 0)
    for row in payload["flange_surfaces"]:
        assert np.all(np.array(row["vertices"]).reshape(-1, 3)[:, 2] == 0)


def test_largest_child_root_reverses_traversal_without_changing_extent():
    payload = v4._develop_faces(*quarter(parent_depth=5, child_depth=30), HASH)
    assert payload["diagnostics"]["root_source_face_index"] == 2
    assert np.ptp(flat_points(payload)[:, 1]) == pytest.approx(35+2*math.pi)
    proof = payload["diagnostics"]["finite_width_proofs"][0]
    assert proof["parent_source_face_index"] == 2 and proof["child_source_face_index"] == 0


def test_multi_bend_chain_shares_one_budget_and_retains_native_ids(monkeypatch):
    budgets = []
    actual = v4.certify_finite_width_flange
    def trace(*args, **kwargs):
        budgets.append(kwargs["budget"])
        return actual(*args, **kwargs)
    monkeypatch.setattr(v4, "certify_finite_width_flange", trace)
    payload = v4._develop_faces(*chain(), HASH)
    assert len(budgets) == 4 and len({id(b) for b in budgets}) == 1
    assert payload["diagnostics"]["finite_width_work"]["triangles"] == 8
    assert [row["feature_id"] for row in payload["bend_lines"]] == ["CAD-native-A", "CAD-native-Z"]
    assert payload["diagnostics"]["selected_source_face_indices"] == [0, 1, 2, 6, 7]
    assert np.ptp(flat_points(payload)[:, 1]) == pytest.approx(47+4*math.pi)


def test_shared_budget_cannot_reset_between_bends(monkeypatch):
    monkeypatch.setattr(width, "MAX_TOTAL_TRIANGLES", 5)
    assert v4._develop_faces(*quarter(), HASH)["status"] == "AVAILABLE"
    with pytest.raises(v4.UnfoldUnavailable, match="FINITE_WIDTH_PARENT:FINITE_WIDTH_WORK_BUDGET_EXCEEDED"):
        v4._develop_faces(*chain(), HASH)


def test_hole_and_original_trimmed_connectivity_are_not_filled():
    faces, axes, candidates = quarter()
    vertices = np.array([[4, -30, 0], [4, 0, 0], [4, 0, 12], [4, -30, 12],
                         [4, -20, 3], [4, -15, 3], [4, -15, 8], [4, -20, 8]], dtype=float)
    triangles = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    for index in (0, 3):
        faces[index].vertices = vertices+[index/3, 0, 0]
        faces[index].triangles = triangles.copy()
    payload = v4._develop_faces(faces, axes, candidates, HASH)
    assert np.array(payload["triangles"][:triangles.size]).reshape(-1, 3).tolist() == triangles.tolist()
    points = flat_points(payload)[:8]
    area = sum(abs(np.cross(points[t[1]]-points[t[0]], points[t[2]]-points[t[0]])[2])/2 for t in triangles)
    assert area == pytest.approx(360-25)


def test_old_parent_false_negative_tab_outside_span_is_retained_entirely():
    case = quarter()
    add_parent_tab(case[0])
    with pytest.raises(v3.UnfoldUnavailable, match="PARENT_FLANGE_EXTENDS"):
        v3._develop_faces(*copy.deepcopy(case), HASH)
    payload = v4._develop_faces(*case, HASH)
    assert payload["status"] == "AVAILABLE"
    assert len(payload["vertices"]) == sum(face.vertices.size for face in case[0][:3])
    assert len(payload["triangles"]) == sum(face.triangles.size for face in case[0][:3])
    assert np.ptp(flat_points(payload)[:, 0]) == pytest.approx(18)
    assert np.array(payload["triangles"][:case[0][0].triangles.size]).reshape(-1, 3).tolist() == case[0][0].triangles.tolist()


def test_old_child_false_negative_tab_is_not_silently_dropped():
    case = quarter()
    for index in (2, 5):
        face = case[0][index]
        r = face.origin[1]
        face.vertices[2:, 2] = 18
        offset = len(face.vertices)
        face.vertices = np.vstack((face.vertices, [[0, r, 14], [9, r, 14], [9, r, 18], [0, r, 18]]))
        face.triangles = np.vstack((face.triangles, RECT_TRIANGLES+offset))
    with pytest.raises(v3.UnfoldUnavailable, match="CHILD_FLANGE_EXTENDS"):
        v3._develop_faces(*copy.deepcopy(case), HASH)
    payload = v4._develop_faces(*case, HASH)
    assert payload["status"] == "AVAILABLE"
    assert len(payload["flange_surfaces"][0]["triangles"]) == case[0][2].triangles.size


def test_triangle_with_all_original_vertices_outside_span_still_refuses():
    faces, axes, candidates = quarter()
    for index in (0, 3):
        r = faces[index].origin[0]
        offset = len(faces[index].vertices)
        faces[index].vertices = np.vstack((faces[index].vertices, [[r, 1, -1], [r, 1, 13], [r, 3, -1]]))
        faces[index].triangles = np.vstack((faces[index].triangles, [[offset, offset+1, offset+2]]))
    with pytest.raises(v4.UnfoldUnavailable, match="FINITE_WIDTH_PARENT:FLANGE_EXTENDS"):
        v4._develop_faces(faces, axes, candidates, HASH)


def test_real_developed_overlap_outside_slab_is_still_refused():
    case = quarter()
    add_parent_tab(case[0], extent=9)
    for index in (2, 5):
        case[0][index].vertices[2:, 2] = 18
    # Independent quarter-chart oracle: tab covers developed distance [0,9]
    # beyond its seam; child begins at r*pi/2=2*pi. Their z[14,18] overlap
    # area is 4*(9-2*pi)>0, outside the bend cylinder's z[0,12] slab.
    assert 4*(9-2*math.pi) > v3.GEOM_TOL**2
    with pytest.raises(v4.UnfoldUnavailable, match="DEVELOPED_SKIN_SELF_OVERLAP"):
        v4._develop_faces(*case, HASH)


def test_truncated_native_seam_span_cannot_receive_v4_certificate():
    case = quarter()
    for face in case[0]:
        for seam in face.edges.values():
            seam[1, 2] = 11
    with pytest.raises(v4.UnfoldUnavailable, match="SEAM_DOES_NOT_COVER_AXIAL_SPAN"):
        v4._develop_faces(*case, HASH)


@pytest.mark.parametrize("change,reason", [
    ("nan", "NONFINITE_GEOMETRY"), ("candidate", "CAD_CANDIDATE_AXIS_MISMATCH"),
    ("axial_span", "UNEQUAL_BEND_SKIN_AXIAL_TRIMS"),
    ("radius", "NATIVE_AXIS_RADIUS_MISMATCH"),
    ("unsupported", "UNSUPPORTED_NATIVE_SURFACE"),
])
def test_legacy_ownership_candidate_and_geometry_refusals_remain(change, reason):
    case = quarter()
    if change == "nan": case[0][0].vertices[0, 0] = np.nan
    elif change == "candidate": case[2][0]["line_start"][0] += .1
    elif change == "axial_span": case[0][4].v1 += 1
    elif change == "radius": case[1][0]["radii_mm"] = [4, 6]
    else: case[0][0].kind = "unsupported"
    with pytest.raises(v4.UnfoldUnavailable, match=reason):
        v4._develop_faces(*case, HASH)


def test_native_boundary_ledger_and_full_overlap_are_called_without_replacements(monkeypatch):
    calls = []
    for name in ("_verify_skin_boundary", "_check_overlap"):
        actual = getattr(v3, name)
        def wrap(*args, _actual=actual, _name=name, **kwargs):
            calls.append(_name)
            return _actual(*args, **kwargs)
        monkeypatch.setattr(v3, name, wrap)
    v4._develop_faces(*quarter(), HASH)
    assert calls == ["_verify_skin_boundary", "_check_overlap"]


def test_body_overlay_projection_and_input_immutability():
    case = posed(quarter())
    before = copy.deepcopy(case)
    payload = v4._develop_faces(*case, HASH)
    points = flat_points(payload)
    for overlay in payload["flange_surfaces"]:
        mesh = np.array(overlay["vertices"]).reshape(-1, 3)
        assert np.all(mesh[:, 2] == 0)
        assert all(any(np.array_equal(point, body_point) for body_point in points) for point in mesh)
    for face, original in zip(case[0], before[0]):
        assert np.array_equal(face.vertices, original.vertices)
        assert np.array_equal(face.triangles, original.triangles)
        for edge in face.edges: assert np.array_equal(face.edges[edge], original.edges[edge])
    assert case[1:] == before[1:]


def test_unavailable_envelope_is_empty_and_no_native_or_path_work_for_invalid_hash(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no source or native work permitted")
    monkeypatch.setattr(v4, "Path", forbidden)
    monkeypatch.setattr(v3, "_read_native", forbidden)
    payload = v4.build_unfold_preview("not-accessed", "invalid-hash", [])
    assert payload["status"] == "UNAVAILABLE" and payload["reason"] == "INVALID_EXPECTED_CAD_HASH"
    assert payload["diagnostics"]["mechanism_version"] == 4
    for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"):
        assert payload[key] == []
    empty = v4.unavailable_unfold_preview("SOURCE_NOT_AVAILABLE")
    assert empty["source_sha256"] is None and empty["manufacturing_ready"] is False


def test_unexplained_face_and_coplanar_split_refusals_remain():
    for coplanar, reason in [(False, "UNEXPLAINED_NATIVE_MATERIAL_SURFACE"),
                              (True, "COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED")]:
        faces, axes, candidates = quarter()
        if coplanar:
            seam = np.array([[4, -30, 0], [4, -30, 12]], dtype=float)
            faces[0].edges[99] = seam
            extra = plane(6, [[4, -31, 0], [4, -30, 0], [4, -30, 12], [4, -31, 12]], [1, 0, 0], {99: seam})
        else:
            extra = plane(6, [[0, 0, 20], [1, 0, 20], [1, 1, 20], [0, 1, 20]], [0, 0, 1], {})
        faces.append(extra)
        with pytest.raises(v4.UnfoldUnavailable, match=reason):
            v4._develop_faces(faces, axes, candidates, HASH)
