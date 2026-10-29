"""Authored numeric topology/mesh charts only; no native or physical inputs."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from backend import vendor_unfold_preview as native
from backend import vendor_unfold_preview_v4 as v4
from backend.vendor_unfold_coplanar_stitch_v1 import stitch_coplanar_native_faces


TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def rectangle(index, x0, x1, z0=0., z1=12., *, y=4., edges=None, normal=None):
    points = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
    return native.SkinFace(index, "plane", points, TRIANGLES.copy(),
        origin=points[0].copy(), normal=np.array([0., 1., 0.]) if normal is None else normal,
        edges=edges or {})


def seam(x, y=4.):
    return np.array([[x, y, 0.], [x, y, 12.]])


def split_rectangles():
    return [rectangle(0, 0., -5., edges={10: seam(-5), 11: seam(0)}),
            rectangle(1, -5., -10., edges={10: seam(-5), 12: seam(-10)})]


def cylinder(index, radius, center, rotation, edges):
    angles = np.linspace(0., math.pi/2, 9)
    points = np.array([[radius*math.cos(u), radius*math.sin(u), z]
                       for u in angles for z in (0., 12.)]) @ rotation.T+center
    triangles = np.array([row for j in range(8) for row in
                          ([2*j, 2*j+2, 2*j+3], [2*j, 2*j+3, 2*j+1])], dtype=np.int32)
    return native.SkinFace(index, "cylinder", points, triangles, edges=edges,
        origin=center, axis=rotation @ np.array([0., 0., 1.]),
        x_axis=rotation @ np.array([1., 0., 0.]), y_axis=rotation @ np.array([0., 1., 0.]),
        radius=radius, u0=0., u1=math.pi/2, v0=0., v1=12.)


def split_two_bend_skin():
    faces = []
    first = np.eye(3)
    second = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for layer, radius in enumerate((4., 5.)):
        base, edge = len(faces), 100+20*layer
        start, end = seam(radius, 0.), seam(0., radius)
        points = np.array([[radius, -30., 0.], [radius, 0., 0.],
                           [radius, 0., 12.], [radius, -30., 12.]])
        faces.append(native.SkinFace(base, "plane", points, TRIANGLES.copy(),
            origin=points[0].copy(), normal=np.array([1., 0., 0.]), edges={edge: start}))
        faces.append(cylinder(base+1, radius, np.zeros(3), first, {edge: start, edge+1: end}))
        faces.append(rectangle(base+2, 0., -5., y=radius,
                              edges={edge+1: end, edge+2: seam(-5., radius)}))
        faces.append(rectangle(base+3, -5., -10., y=radius,
                              edges={edge+2: seam(-5., radius), edge+3: seam(-10., radius)}))
        center = np.array([-10., 0., 0.])
        end2 = seam(-10.-radius, 0.)
        faces.append(cylinder(base+4, radius, center, second,
                              {edge+3: seam(-10., radius), edge+4: end2}))
        points2 = np.array([[-10.-radius, 0., 0.], [-10.-radius, -7., 0.],
                            [-10.-radius, -7., 12.], [-10.-radius, 0., 12.]])
        faces.append(native.SkinFace(base+5, "plane", points2, TRIANGLES.copy(),
            origin=points2[0].copy(), normal=np.array([-1., 0., 0.]), edges={edge+4: end2}))
    axes, candidates = [], []
    for name, indices, midpoint in (("A", [2, 8], [4.5/math.sqrt(2)]*2),
                                     ("B", [5, 11], [-10.-4.5/math.sqrt(2), 4.5/math.sqrt(2)])):
        line = [[*midpoint, 0.], [*midpoint, 12.]]
        axes.append({"face_indices": indices, "radii_mm": [4., 5.],
                     "canonical_line_start": line[0], "canonical_line_end": line[1]})
        candidates.append({"feature_id": name, "line_start": line[0], "line_end": line[1]})
    return faces, axes, candidates


def test_native_connected_rectangles_merge_without_mesh_edits_and_with_lineage():
    faces = split_rectangles()
    original = copy.deepcopy(faces)
    result = stitch_coplanar_native_faces(faces, [])
    assert result.source_to_chart == {0: 0, 1: 0}
    merged = result.faces[0]
    assert np.array_equal(merged.vertices, np.vstack([f.vertices for f in original]))
    assert np.array_equal(merged.triangles, np.vstack([original[0].triangles,
                                                     original[1].triangles+4]))
    assert set(merged.edges) == {11, 12}
    assert merged.area == pytest.approx(120.)
    assert result.ledger["charts"][0]["removed_internal_edge_ids"] == [10]
    assert result.ledger["charts"][0]["members"][1] == {
        "source_face_index": 1, "vertex_start": 4, "vertex_count": 4,
        "triangle_start": 2, "triangle_count": 2}
    merged.vertices[0, 0] = 99.
    assert np.array_equal(faces[0].vertices, original[0].vertices)
    assert set(faces[0].edges) == {10, 11}


def test_split_flange_disconnects_legacy_skin_and_stitched_chart_recovers_both_bends():
    faces, axes, candidates = split_two_bend_skin()
    with pytest.raises(native.UnfoldUnavailable, match="NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS"):
        v4._develop_faces(copy.deepcopy(faces), axes, candidates, "a"*64)
    result = stitch_coplanar_native_faces(faces, axes)
    payload = v4._develop_faces(result.faces, result.axes, candidates, "a"*64)
    assert len(payload["bend_lines"]) == 2
    assert payload["diagnostics"]["selected_source_face_indices"] == [0, 1, 2, 3, 4]
    assert len(payload["vertices"]) == sum(f.vertices.size for f in faces[:6])
    assert len(payload["triangles"]) == sum(f.triangles.size for f in faces[:6])
    points = np.array(payload["vertices"]).reshape(-1, 3)
    assert np.ptp(points[:, 0]) == pytest.approx(12.)
    assert np.ptp(points[:, 1]) == pytest.approx(47.+4.*math.pi)
    assert result.axes[0]["face_indices"] == [2, 7]
    assert result.axes[1]["face_indices"] == [4, 9]
    assert axes[0]["face_indices"] == [2, 8]  # caller input untouched


def test_coincident_positions_without_native_edge_identity_do_not_connect():
    faces = split_rectangles()
    faces[1].edges[13] = faces[1].edges.pop(10)
    assert len(stitch_coplanar_native_faces(faces, []).faces) == 2


def test_opposite_analytic_normals_do_not_lose_native_triangle_winding():
    faces = split_rectangles()
    faces[1].normal *= -1.
    faces[1].triangles = faces[1].triangles[:, ::-1]
    result = stitch_coplanar_native_faces(faces, [])
    assert np.array_equal(result.faces[0].triangles[2:]-4, faces[1].triangles)


def test_shared_edge_cannot_connect_two_thickness_planes():
    faces = split_rectangles()
    faces[1].vertices[:, 1] += 1.
    faces[1].origin[1] += 1.
    faces[1].edges = {k: v+[0., 1., 0.] for k, v in faces[1].edges.items()}
    assert len(stitch_coplanar_native_faces(faces, []).faces) == 2


def test_declared_shared_edge_has_to_be_the_same_rail():
    faces = split_rectangles()
    faces[1].edges[10][0, 2] += .2
    with pytest.raises(native.UnfoldUnavailable, match="SHARED_RAIL_MISMATCH"):
        stitch_coplanar_native_faces(faces, [])


def test_mesh_overlap_cannot_be_hidden_inside_one_supernode():
    faces = split_rectangles()
    faces[1].vertices[:, 0] += 2.
    faces[1].origin = faces[1].vertices[0].copy()
    with pytest.raises(native.UnfoldUnavailable, match="OVERLAPPING_COPLANAR_SOURCE_FACES"):
        stitch_coplanar_native_faces(faces, [])


def test_nonmanifold_native_edge_refuses_before_stitching():
    faces = split_rectangles()
    faces.append(rectangle(2, -5., -10., edges={10: seam(-5.)}))
    with pytest.raises(native.UnfoldUnavailable, match="NON_MANIFOLD_EDGE_INCIDENCE"):
        stitch_coplanar_native_faces(faces, [])


def test_tolerance_creep_is_refused_against_single_anchor_plane():
    faces = [rectangle(i, -5.*i, -5.*(i+1), y=4.+i*.75*native.GEOM_TOL,
                       edges={10+i: seam(-5.*(i+1), 4.+i*.75*native.GEOM_TOL),
                              **({9+i: seam(-5.*i, 4.+i*.75*native.GEOM_TOL)} if i else {})})
             for i in range(3)]
    with pytest.raises(native.UnfoldUnavailable, match="COMPONENT_PLANE_DRIFT"):
        stitch_coplanar_native_faces(faces, [])


def test_remapped_axis_cannot_claim_a_planar_source_as_bend_skin():
    with pytest.raises(native.UnfoldUnavailable, match="AXIS_OWNERSHIP_INVALID"):
        stitch_coplanar_native_faces(split_rectangles(), [{"face_indices": [1, 2]}])


def test_wide_source_triangle_intrusion_still_refuses_after_stitching():
    faces, axes, candidates = split_two_bend_skin()
    # All three new vertices lie outside the bend's axial interval. The full
    # triangle crosses the finite slab at y>0, so vertex-only filtering fails.
    for index in (0, 6):
        face = faces[index]
        radius = face.origin[0]
        offset = len(face.vertices)
        face.vertices = np.vstack([face.vertices,
            [[radius, 1., -1.], [radius, 1., 13.], [radius, 3., -1.]]])
        face.triangles = np.vstack([face.triangles, [[offset, offset+1, offset+2]]])
    result = stitch_coplanar_native_faces(faces, axes)
    with pytest.raises(native.UnfoldUnavailable, match="FINITE_WIDTH_PARENT:FLANGE_EXTENDS"):
        v4._develop_faces(result.faces, result.axes, candidates, "a"*64)


def test_stitching_respects_rigid_pose_and_reversed_native_edge_endpoints():
    faces = split_rectangles()
    # Explicit cyclic coordinate permutation: proper rotation, no engine math.
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    shift = np.array([21., -13., 8.])
    for face in faces:
        face.vertices = face.vertices @ rotation.T+shift
        face.origin = rotation @ face.origin+shift
        face.normal = rotation @ face.normal
        face.edges = {k: v @ rotation.T+shift for k, v in face.edges.items()}
    faces[1].edges[10] = faces[1].edges[10][::-1].copy()
    result = stitch_coplanar_native_faces(faces, [])
    assert len(result.faces) == 1 and result.faces[0].area == pytest.approx(120.)
