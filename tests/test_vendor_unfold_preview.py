"""Source-only tests. All geometry is authored synthetically in this file."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from backend import vendor_unfold_preview as unfold


HASH = "a" * 64
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _plane(index, points, normal, edge, seam):
    points = np.asarray(points, dtype=float)
    return unfold.SkinFace(index, "plane", points, TRIANGLES.copy(),
                           edges={edge: np.asarray(seam)}, origin=points[0].copy(),
                           normal=np.asarray(normal, dtype=float))


def _quarter_skin(radius, start_index, first_edge, parent_depth=10., child_depth=5.):
    r = radius
    p = np.array([[r, 0., 0.], [r, 0., 20.]])
    q = np.array([[0., r, 0.], [0., r, 20.]])
    parent = _plane(start_index, [[r, -parent_depth, 0.], [r, 0., 0.],
                                 [r, 0., 20.], [r, -parent_depth, 20.]], [1., 0., 0.], first_edge, p)
    child = _plane(start_index + 1, [[0., r, 0.], [-child_depth, r, 0.],
                                    [-child_depth, r, 20.], [0., r, 20.]], [0., 1., 0.], first_edge + 1, q)
    angles = np.linspace(0., math.pi / 2, 9)
    points = np.array([[r * math.cos(u), r * math.sin(u), z]
                       for u in angles for z in [0., 20.]])
    triangles = np.array([[2 * i, 2 * i + 2, 2 * i + 3] for i in range(8)]
                        + [[2 * i, 2 * i + 3, 2 * i + 1] for i in range(8)], dtype=np.int32)
    cylinder = unfold.SkinFace(start_index + 2, "cylinder", points, triangles,
                edges={first_edge: p, first_edge + 1: q}, origin=np.zeros(3),
                axis=np.array([0., 0., 1.]), x_axis=np.array([1., 0., 0.]),
                y_axis=np.array([0., 1., 0.]), radius=r, u0=0., u1=math.pi / 2., v0=0., v1=20.)
    return [parent, child, cylinder]


def _case():
    faces = _quarter_skin(2., 0, 0) + _quarter_skin(3., 3, 2)
    center = np.array([2.5 / math.sqrt(2), 2.5 / math.sqrt(2), 10.])
    line = np.array([center - [0., 0., 10.], center + [0., 0., 10.]])
    axis = {"face_indices": [3, 6], "radii_mm": [2., 3.],
            "canonical_line_start": line[0].tolist(), "canonical_line_end": line[1].tolist()}
    candidate = {"feature_id": "native-bend-17", "line_start": line[0].tolist(), "line_end": line[1].tolist()}
    return faces, [axis], [candidate]


def test_quarter_skin_is_complete_planar_and_arc_length_not_chord():
    faces, axes, candidates = _case()
    payload = unfold._develop_faces(faces, axes, candidates, HASH)
    assert payload["status"] == "AVAILABLE"
    assert payload["model"] == "SELECTED_CAD_SKIN_DEVELOPMENT"
    assert not payload["manufacturing_ready"] and not payload["units_verified"]
    points = np.array(payload["vertices"]).reshape(-1, 3)
    assert np.all(points[:, 2] == 0)
    assert payload["diagnostics"]["selected_source_face_indices"] == [0, 1, 2]
    # Root is 10 deep, child 5 deep and the selected radius-2 arc is pi long.
    assert np.ptp(points[:, 1]) == pytest.approx(15. + math.pi)
    line = payload["bend_lines"][0]
    assert line["feature_id"] == "native-bend-17"
    assert line["sweep_deg"] == pytest.approx(90.)
    assert line["bend_direction"] == "unknown"
    assert abs(line["line_start"][1]) == pytest.approx(10. + math.pi / 2.)
    assert len(payload["triangles"]) == sum(face.triangles.size for face in faces[:3])


@pytest.mark.parametrize("mirrored", [False, True])
def test_posed_and_mirrored_cad_preserves_developed_distances(mirrored):
    faces, axes, candidates = _case()
    original = unfold._develop_faces(copy.deepcopy(faces), axes, candidates, HASH)
    rotation = unfold._rotation(np.array([1., 2., 3.]), 0.73)
    if mirrored:
        rotation = rotation @ np.diag([-1., 1., 1.])
    translation = np.array([45., -11., 83.])
    for face in faces:
        face.vertices = face.vertices @ rotation.T + translation
        face.origin = rotation @ face.origin + translation
        face.edges = {key: points @ rotation.T + translation for key, points in face.edges.items()}
        if face.kind == "plane":
            face.normal = rotation @ face.normal
        else:
            face.axis, face.x_axis, face.y_axis = [rotation @ value for value in (face.axis, face.x_axis, face.y_axis)]
            if mirrored:
                # A reflected shape can be represented by a proper native
                # cylinder chart by reversing axis and V, not by reflecting maps.
                face.axis = -face.axis
                face.v0, face.v1 = -face.v1, -face.v0
    for row in axes + candidates:
        for key in ("line_start", "line_end", "canonical_line_start", "canonical_line_end"):
            if key in row:
                row[key] = (rotation @ np.array(row[key]) + translation).tolist()
    payload = unfold._develop_faces(faces, axes, candidates, HASH)
    original_points = np.array(original["vertices"]).reshape(-1, 3)
    points = np.array(payload["vertices"]).reshape(-1, 3)
    assert np.linalg.norm(points[:, None] - points, axis=2) == pytest.approx(
        np.linalg.norm(original_points[:, None] - original_points, axis=2), abs=1e-8)
    assert np.all(points[:, 2] == 0.)


def test_cap_exclusion_is_geometric_not_area_heuristic():
    faces, _, _ = _case()
    cylinder = faces[2]
    end_cap = _plane(6, [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], [0, 0, 1], 0, cylinder.edges[0])
    radial_cap = _plane(7, [[0, 0, 0], [2, 0, 0], [2, 0, 20], [0, 0, 20]], [0, 1, 0], 0, cylinder.edges[0])
    assert unfold._tangent_seam(end_cap, cylinder, {0: {2, 6}, 1: {1, 2}}) is None
    assert unfold._tangent_seam(radial_cap, cylinder, {0: {2, 7}, 1: {1, 2}}) is None


def test_native_trimmed_planar_hole_connectivity_is_preserved():
    faces, axes, candidates = _case()
    # A rectangle with an interior rectangular hole; native triangles are copied,
    # never replaced with an outline fan that would fill the cutout.
    points = np.array([[2, -10, 0], [2, 0, 0], [2, 0, 20], [2, -10, 20],
                       [2, -6, 7], [2, -4, 7], [2, -4, 13], [2, -6, 13]], dtype=float)
    triangles = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    faces[0].vertices, faces[0].triangles = points, triangles
    faces[3].vertices, faces[3].triangles = points + [1., 0., 0.], triangles.copy()
    payload = unfold._develop_faces(faces, axes, candidates, HASH)
    flat = np.array(payload["vertices"]).reshape(-1, 3)[:8]
    assert np.array(payload["triangles"][:triangles.size]).reshape(-1, 3).tolist() == triangles.tolist()
    area = sum(np.linalg.norm(np.cross(flat[t[1]] - flat[t[0]], flat[t[2]] - flat[t[0]])) / 2.
               for t in triangles)
    assert area == pytest.approx(188.)


def _chain_case(thickness=1.):
    faces, axes, candidates = _case()
    for start, radius, parent, edge in [(6, 2., 1, 4), (8, 2. + thickness, 4, 6)]:
        # Same quarter-cylinder chart, rotated 90 degrees and translated to
        # the far edge of the first bend's child flange.
        new = _quarter_skin(radius, start, edge, parent_depth=5., child_depth=4.)
        transform = unfold._rotation(np.array([0., 0., 1.]), math.pi / 2)
        shift = np.array([-5., 0., 0.])
        for face in new:
            face.vertices = face.vertices @ transform.T + shift
            face.origin = transform @ face.origin + shift
            face.edges = {key: points @ transform.T + shift for key, points in face.edges.items()}
            if face.kind == "plane":
                face.normal = transform @ face.normal
            else:
                face.axis, face.x_axis, face.y_axis = [transform @ value for value in (face.axis, face.x_axis, face.y_axis)]
        cylinder, child = new[2], new[1]
        cylinder.index, child.index = start, start + 1
        faces[parent].edges[edge] = new[0].edges[edge]
        faces.extend([cylinder, child])
    center = np.array([-5. - (2. + thickness / 2) / math.sqrt(2),
                       (2. + thickness / 2) / math.sqrt(2), 10.])
    line = np.array([center - [0, 0, 10], center + [0, 0, 10]])
    axes.append({"face_indices": [7, 9], "radii_mm": [2., 2. + thickness],
                 "canonical_line_start": line[0].tolist(), "canonical_line_end": line[1].tolist()})
    candidates.append({"feature_id": "native-bend-22", "line_start": line[0].tolist(), "line_end": line[1].tolist()})
    return faces, axes, candidates


def test_multi_bend_chain_has_one_assignment_per_native_id():
    faces, axes, candidates = _chain_case()
    payload = unfold._develop_faces(faces, axes, candidates, HASH)
    assert payload["status"] == "AVAILABLE"
    assert [row["feature_id"] for row in payload["bend_lines"]] == ["native-bend-17", "native-bend-22"]
    assert payload["diagnostics"]["selected_source_face_indices"] == [0, 1, 2, 6, 7]
    points = np.array(payload["vertices"]).reshape(-1, 3)
    assert np.ptp(points[:, 1]) == pytest.approx(19. + 2. * math.pi)


def test_varying_sheet_thickness_is_unavailable():
    faces, axes, candidates = _chain_case(thickness=2.)
    with pytest.raises(unfold.UnfoldUnavailable, match="VARYING_SHEET_THICKNESS"):
        unfold._develop_faces(faces, axes, candidates, HASH)


def test_split_coplanar_skin_cannot_silently_omit_flange_portion():
    faces, axes, candidates = _case()
    seam = np.array([[2., -10., 0.], [2., -10., 20.]])
    faces[0].edges[80] = seam
    extra = _plane(6, [[2., -12., 0.], [2., -10., 0.], [2., -10., 20.], [2., -12., 20.]],
                   [1., 0., 0.], 80, seam)
    faces.append(extra)
    with pytest.raises(unfold.UnfoldUnavailable, match="COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED"):
        unfold._develop_faces(faces, axes, candidates, HASH)


def test_unexplained_planar_material_is_not_silently_omitted():
    faces, axes, candidates = _case()
    extra = _plane(6, [[0., 0., 30.], [1., 0., 30.], [1., 1., 30.], [0., 1., 30.]],
                   [0., 0., 1.], 80, np.array([[0., 0., 30.], [1., 0., 30.]]))
    faces.append(extra)
    with pytest.raises(unfold.UnfoldUnavailable, match="UNEXPLAINED_NATIVE_MATERIAL_SURFACE"):
        unfold._develop_faces(faces, axes, candidates, HASH)


def test_longer_than_supported_bend_sweep_is_unavailable():
    faces, axes, candidates = _case()
    faces[2].u1 = faces[5].u1 = math.radians(176.)
    with pytest.raises(unfold.UnfoldUnavailable, match="BEND_SWEEP_EXCEEDS_SUPPORTED"):
        unfold._develop_faces(faces, axes, candidates, HASH)


@pytest.mark.parametrize("offset", [-13.7, -3.7, 2.8, 7.1, 15.3])
def test_native_u_interval_wraps_do_not_change_arc_development(offset):
    faces, axes, candidates = _case()
    expected = unfold._develop_faces(copy.deepcopy(faces), axes, candidates, HASH)
    for face in (faces[2], faces[5]):
        face.x_axis = np.array([math.cos(-offset), math.sin(-offset), 0.])
        face.y_axis = np.array([-math.sin(-offset), math.cos(-offset), 0.])
        face.u0 += offset
        face.u1 += offset
    payload = unfold._develop_faces(faces, axes, candidates, HASH)
    assert payload["vertices"] == pytest.approx(expected["vertices"], abs=1e-9)
    assert payload["bend_lines"][0]["line_start"] == pytest.approx(expected["bend_lines"][0]["line_start"], abs=1e-9)


def test_unequal_axial_trims_and_noncoaxial_pairs_are_unavailable():
    faces, axes, candidates = _case()
    faces[5].v1 += 1.
    with pytest.raises(unfold.UnfoldUnavailable, match="UNEQUAL_BEND_SKIN_AXIAL_TRIMS"):
        unfold._develop_faces(faces, axes, candidates, HASH)
    faces, axes, candidates = _case()
    faces[5].origin += [0.01, 0., 0.]
    faces[5].vertices += [0.01, 0., 0.]
    with pytest.raises(unfold.UnfoldUnavailable, match="UNEQUAL_OR_NONCOAXIAL_BEND_SKINS"):
        unfold._develop_faces(faces, axes, candidates, HASH)


@pytest.mark.parametrize("mutation,reason", [
    ("nan", "NONFINITE_GEOMETRY"),
    ("off_plane", "NATIVE_PLANE_OWNERSHIP_MISMATCH"),
    ("off_cylinder", "NATIVE_CYLINDER_OWNERSHIP_MISMATCH"),
    ("radius", "NATIVE_AXIS_RADIUS_MISMATCH"),
    ("opposite_arc", "UNEQUAL_BEND_SKIN_ANGULAR_TRIMS"),
    ("duplicate_candidate", "DUPLICATE_OR_INVALID_CANDIDATE_IDS"),
    ("candidate_line", "CAD_CANDIDATE_AXIS_MISMATCH"),
    ("indirect_frame", "INVALID_CYLINDER_FRAME"),
])
def test_invalid_geometry_never_produces_partial_flat(mutation, reason):
    faces, axes, candidates = _case()
    if mutation == "nan":
        faces[0].vertices[0, 0] = np.nan
    elif mutation == "off_plane":
        faces[0].vertices[0, 0] += 0.1
    elif mutation == "off_cylinder":
        faces[2].vertices[0, 0] += 0.1
    elif mutation == "radius":
        axes[0]["radii_mm"] = [2., 4.]
    elif mutation == "opposite_arc":
        face = faces[5]
        face.vertices[:, :2] *= -1.
        face.u0 += math.pi
        face.u1 += math.pi
    elif mutation == "duplicate_candidate":
        axes *= 2
        candidates *= 2
    elif mutation == "candidate_line":
        candidates[0]["line_start"][0] += 1.
    elif mutation == "indirect_frame":
        faces[2].y_axis *= -1.
    with pytest.raises(unfold.UnfoldUnavailable, match=reason):
        unfold._develop_faces(faces, axes, candidates, HASH)


def test_cycle_and_duplicate_skin_links_refuse():
    faces, _, _ = _case()
    link = unfold.BendLink(2, (0, 1), "a", {}, {})
    duplicate = unfold.BendLink(5, (0, 1), "b", {}, {})
    with pytest.raises(unfold.UnfoldUnavailable, match="CYCLIC_SKIN_GRAPH"):
        unfold._select_skin(faces, [link, duplicate], {0: [link, duplicate], 1: [link, duplicate]}, 2)
    duplicate.feature_id = "a"
    with pytest.raises(unfold.UnfoldUnavailable, match="DUPLICATE_BEND_ASSIGNMENT"):
        unfold._select_skin(faces, [link, duplicate], {0: [link, duplicate], 1: [link, duplicate]}, 2)


def test_overlap_gate_permits_seams_but_refuses_positive_area():
    faces, _, _ = _case()
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    unfold._check_overlap({0: points, 1: points + [1., 0., 0.]}, faces)
    with pytest.raises(unfold.UnfoldUnavailable, match="DEVELOPED_SKIN_SELF_OVERLAP"):
        unfold._check_overlap({0: points, 1: points + [0.5, 0., 0.]}, faces)


def test_source_gate_and_bad_hash_do_not_import_native(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("native reader must not run")
    monkeypatch.setattr(unfold, "_read_native", forbidden)
    _, _, candidates = _case()
    path = tmp_path / "synthetic.step"
    path.write_bytes(b"synthetic bytes, not a STEP")
    payload = unfold.build_unfold_preview(path, HASH, candidates)
    assert payload["reason"] == "CAD_HASH_MISMATCH"
    assert payload["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"):
        assert payload[key] == []
    assert unfold.build_unfold_preview(tmp_path / "unsupported.iges", HASH, candidates)["reason"] == "STEP_SOURCE_REQUIRED"


def test_native_exception_returns_empty_envelope(tmp_path, monkeypatch):
    _, _, candidates = _case()
    path = tmp_path / "synthetic.step"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(unfold, "_read_native", lambda path: (_ for _ in ()).throw(RuntimeError("private kernel detail")))
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["status"] == "UNAVAILABLE" and payload["reason"] == "NATIVE_DEVELOPMENT_UNAVAILABLE"
    assert payload["diagnostics"]["error_type"] == "RuntimeError"
    assert "private" not in str(payload)


def test_complete_refusal_after_geometry_failure_has_no_partial_output(tmp_path, monkeypatch):
    faces, axes, candidates = _case()
    faces[1].vertices[0, 1] += 0.2
    path = tmp_path / "synthetic.step"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(unfold, "_read_native", lambda path: (faces, axes))
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["reason"] == "NATIVE_PLANE_OWNERSHIP_MISMATCH"
    assert all(payload[key] == [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"))


def test_small_synthetic_budget_gates_before_native_parse(tmp_path, monkeypatch):
    _, _, candidates = _case()
    path = tmp_path / "synthetic.step"
    path.write_bytes(b"synthetic exceeds reduced test budget")
    monkeypatch.setattr(unfold, "MAX_CAD_BYTES", 8)
    monkeypatch.setattr(unfold, "_read_native", lambda path: pytest.fail("must refuse before native parse"))
    assert unfold.build_unfold_preview(path, HASH, candidates)["reason"] == "CAD_BYTE_BUDGET_EXCEEDED"
    assert unfold.build_unfold_preview(path, HASH, candidates * 33)["reason"] == "CANDIDATE_BUDGET_OR_SCHEMA_INVALID"
    assert unfold.unavailable_unfold_preview("NO_CAD_SOURCE")["source_sha256"] is None


def test_face_and_triangle_budget_gates_with_tiny_synthetic_limit(monkeypatch):
    faces, axes, candidates = _case()
    monkeypatch.setattr(unfold, "MAX_FACES", 2)
    with pytest.raises(unfold.UnfoldUnavailable, match="GEOMETRY_BUDGET_EXCEEDED"):
        unfold._develop_faces(faces, axes, candidates, HASH)
    monkeypatch.setattr(unfold, "MAX_FACES", 1024)
    monkeypatch.setattr(unfold, "MAX_TRIANGLES", 2)
    with pytest.raises(unfold.UnfoldUnavailable, match="GEOMETRY_BUDGET_EXCEEDED"):
        unfold._develop_faces(faces, axes, candidates, HASH)


def _native_synthetic_step(path: Path, *, sharp_return=False, hole=False, spline_hole=False, rounded_slot=False):
    """Small, valid extruded annular quarter bend plus two planar legs."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
    from scripts.extract_step_topology_bend_axes import extract_step_bend_axes
    wire = BRepBuilderAPI_MakeWire()
    points = [(2., -10.), (2., 0.), (0., 2.), (-5., 2.), (-5., 3.),
              (0., 3.), (3., 0.), (3., -10.)]
    if sharp_return:
        points = [(2., -11.), (2., 0.), (0., 2.), (-5., 2.), (-5., 3.),
                  (0., 3.), (3., 0.), (3., -10.), (9., -10.), (9., -11.)]
    for i, point in enumerate(points):
        following = points[(i + 1) % len(points)]
        if i in (1, 5):
            radius = 2. if i == 1 else 3.
            circle = gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), radius)
            edge = BRepBuilderAPI_MakeEdge(circle, gp_Pnt(*point, 0), gp_Pnt(*following, 0)).Edge()
            if i == 5:
                # Circle traversal from 90° to 0° is the opposite quarter arc.
                edge = BRepBuilderAPI_MakeEdge(circle, 0., math.pi / 2.).Edge()
                edge.Reverse()
        else:
            edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*point, 0), gp_Pnt(*following, 0)).Edge()
        wire.Add(edge)
    shape = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire.Wire()).Face(), gp_Vec(0, 0, 20)).Shape()
    if hole:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        cutter = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(1., -5., 10.), gp_Dir(1, 0, 0)), 1., 3.).Shape()
        shape = BRepAlgoAPI_Cut(shape, cutter).Shape()
    if spline_hole:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.Geom import Geom_BezierCurve
        from OCP.TColgp import TColgp_Array1OfPnt
        cutter_wire = BRepBuilderAPI_MakeWire()
        curves = [
            [(-5., 12.), (-4., 12.), (-3.7, 11.), (-3.7, 10.)],
            [(-3.7, 10.), (-3.7, 9.), (-4., 8.), (-5., 8.)],
            [(-5., 8.), (-6., 8.), (-6.3, 9.), (-6.3, 10.)],
            [(-6.3, 10.), (-6.3, 11.), (-6., 12.), (-5., 12.)],
        ]
        for controls in curves:
            poles = TColgp_Array1OfPnt(1, len(controls))
            for i, (y, z) in enumerate(controls, start=1):
                poles.SetValue(i, gp_Pnt(1., y, z))
            cutter_wire.Add(BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge())
        cutter = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(cutter_wire.Wire()).Face(), gp_Vec(3, 0, 0)).Shape()
        shape = BRepAlgoAPI_Cut(shape, cutter).Shape()
    if rounded_slot:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        first = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(1., -5., 8.), gp_Dir(1, 0, 0)), .6, 3.).Shape()
        second = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(1., -5., 12.), gp_Dir(1, 0, 0)), .6, 3.).Shape()
        middle = BRepPrimAPI_MakeBox(gp_Pnt(1., -5.6, 8.), 3., 1.2, 4.).Shape()
        cutter = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(first, middle).Shape(), second).Shape()
        shape = BRepAlgoAPI_Cut(shape, cutter).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))
    axes = extract_step_bend_axes(path)["bend_axes"]
    candidates = [{"feature_id": axis["bend_id"], "line_start": axis["canonical_line_start"],
                   "line_end": axis["canonical_line_end"]} for axis in axes]
    return candidates


def test_actual_ocp_synthetic_step_through_source_bound_entrypoint(tmp_path):
    pytest.importorskip("OCP")
    path = tmp_path / "authored-quarter-bend.step"
    candidates = _native_synthetic_step(path)
    assert len(candidates) == 1
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["status"] == "AVAILABLE", payload
    assert {line["feature_id"] for line in payload["bend_lines"]} == {candidates[0]["feature_id"]}
    assert np.max(np.abs(np.array(payload["vertices"]).reshape(-1, 3)[:, 2])) == 0.


def test_actual_valid_solid_with_sharp_return_appendage_is_not_partial_available(tmp_path):
    pytest.importorskip("OCP")
    path = tmp_path / "authored-sharp-return.step"
    candidates = _native_synthetic_step(path, sharp_return=True)
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["status"] == "UNAVAILABLE"
    assert payload["reason"] == "UNKNOWN_PLANAR_SKIN_BOUNDARY"
    assert all(payload[key] == [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"))


def test_actual_native_through_hole_preserves_trimmed_skin(tmp_path):
    pytest.importorskip("OCP")
    path = tmp_path / "authored-through-hole.step"
    candidates = _native_synthetic_step(path, hole=True)
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["status"] == "AVAILABLE", payload
    faces, _ = unfold._read_native(path)
    root = payload["diagnostics"]["root_source_face_index"]
    root_native_area = faces[root].area
    assert 196.8 < root_native_area < 197.0
    selected = payload["diagnostics"]["selected_source_face_indices"]
    assert len(payload["triangles"]) == sum(faces[index].triangles.size for index in selected)


@pytest.mark.parametrize("geometry", ["spline_hole", "rounded_slot"])
def test_successor_exact_native_curved_trim_and_boundary_wall(tmp_path, geometry):
    pytest.importorskip("OCP")
    path = tmp_path / f"authored-{geometry}.step"
    candidates = _native_synthetic_step(path, **{geometry: True})
    payload = unfold.build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["status"] == "AVAILABLE", payload
    assert payload["diagnostics"]["native_surface_coverage_checked"]
    faces, _ = unfold._read_native(path)
    selected = payload["diagnostics"]["selected_source_face_indices"]
    assert len(payload["triangles"]) == sum(faces[index].triangles.size for index in selected)
    assert any(curve.kind == "spline" for face in faces for curve in face.edge_curves.values()) if geometry == "spline_hole" else True
    assert any(face.kind == "extruded_boundary" for face in faces) if geometry == "spline_hole" else True


def _bezier_proof(poles, weights=None):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.Geom import Geom_BezierCurve
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfReal
    from OCP.gp import gp_Pnt
    points = TColgp_Array1OfPnt(1, len(poles))
    for i, point in enumerate(poles, start=1):
        points.SetValue(i, gp_Pnt(*point))
    if weights is None:
        curve = Geom_BezierCurve(points)
    else:
        native_weights = TColStd_Array1OfReal(1, len(weights))
        for i, value in enumerate(weights, start=1):
            native_weights.SetValue(i, value)
        curve = Geom_BezierCurve(points, native_weights)
    return unfold._curve_proof(BRepAdaptor_Curve(BRepBuilderAPI_MakeEdge(curve).Edge()))


def test_rational_curve_translation_requires_global_certificate_not_endpoint_samples():
    pytest.importorskip("OCP")
    poles = np.array([[0., 0., 0.], [0., 1e6, 0.], [10., 0., 0.]])
    left = _bezier_proof(poles, [1e-12, 1e-12, 1.])
    right = _bezier_proof(poles + [0., 0., 1.], [9e-11, 1e-12, 1.])
    assert not unfold._translated_curves(left, right, np.array([0., 0., 1.]))
    right = _bezier_proof(poles + [0., 0., 1.], [1e-12, 1e-12, 1.])
    assert unfold._translated_curves(left, right, np.array([0., 0., 1.]))
    reverse = copy.deepcopy(right)
    reverse.poles, reverse.weights = reverse.poles[::-1], reverse.weights[::-1]
    reverse.knots, reverse.multiplicities = 1. - reverse.knots[::-1], reverse.multiplicities[::-1]
    assert unfold._translated_curves(left, reverse, np.array([0., 0., 1.]))


def _circle_proof(radius, phase=0., *, full=False, gap=0., tilt=0.):
    rotation = unfold._rotation(np.array([0., 0., 1.]), phase)
    rotation = unfold._rotation(np.array([1., 0., 0.]), tilt) @ rotation
    return unfold.NativeCurve("circle", np.zeros((2, 3)), center=np.zeros(3),
        axis=rotation @ np.array([0., 0., 1.]), x_axis=rotation @ np.array([1., 0., 0.]),
        y_axis=rotation @ np.array([0., 1., 0.]), radius=radius,
        u0=0. if full else .3, u1=2 * math.pi - gap if full else 1.2)


def test_circle_correspondence_tolerances_are_physical_not_radians():
    left = _circle_proof(10000.)
    assert not unfold._translated_curves(left, _circle_proof(10000., phase=5e-6), np.zeros(3))
    assert not unfold._translated_curves(_circle_proof(10000., full=True),
                                        _circle_proof(10000., full=True, gap=5e-6), np.zeros(3))
    assert not unfold._translated_curves(_circle_proof(10000., full=True),
                                        _circle_proof(10000., full=True, tilt=.001), np.zeros(3))
    assert not unfold._curve_in_perpendicular_plane(_circle_proof(10000., tilt=.001), np.array([0., 0., 1.]))
    assert unfold._translated_curves(_circle_proof(2., full=True), _circle_proof(2., full=True, phase=.7), np.zeros(3))


def test_own_native_knots_and_piecewise_parameterization_are_preserved():
    # Quadratic C0 two-piece curve: two true native knot coordinate systems,
    # identical ordered Bernstein pieces, no global knot movement or merging.
    poles = np.array([[0., 0., 0.], [.5, 1., 0.], [1., 0., 0.], [1.5, -1., 0.], [2., 0., 0.]])
    left = unfold.NativeCurve("spline", poles, 2, np.ones(5), np.array([2., 4., 7.]), np.array([3, 2, 3]))
    right = unfold.NativeCurve("spline", poles + [0., 0., 1.], 2, np.ones(5),
                               np.array([-9., -6.9999999999999, -4.]), np.array([3, 2, 3]))
    original_left, original_right = left.knots.copy(), right.knots.copy()
    assert unfold._translated_curves(left, right, np.array([0., 0., 1.]))
    assert np.array_equal(left.knots, original_left) and np.array_equal(right.knots, original_right)
    right.poles[1, 1] += .01
    assert not unfold._translated_curves(left, right, np.array([0., 0., 1.]))


def test_mismatched_spline_span_topology_abstains():
    poles = np.array([[0., 0., 0.], [1., 2., 0.], [2., 0., 0.]])
    left = unfold.NativeCurve("spline", poles, 2, np.ones(3), np.array([0., 1.]), np.array([3, 3]))
    right = unfold.NativeCurve("spline", np.array([[0., 0., 0.], [.5, 1., 0.], [1., 1., 0.], [2., 0., 0.]]),
                               2, np.ones(4), np.array([0., .5, 1.]), np.array([3, 1, 3]))
    assert not unfold._translated_curves(left, right, np.zeros(3))


def test_nonclamped_straight_spline_cannot_claim_control_poles_as_endpoints():
    curve = unfold.NativeCurve("spline", np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), 2,
        np.ones(3), np.array([0., .25, .75, 1.]), np.array([1, 2, 2, 1]))
    assert curve.straight_endpoints() is None
    assert not unfold._translated_curves(curve, copy.deepcopy(curve), np.zeros(3))


def _ruled_surface(*, warped=False, unequal_weights=False, shifted_knots=False):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    from OCP.gp import gp_Pnt
    controls = [(0., 0., 0.), (0., 2., 1.), (0., 1., 2.)]
    points, weights = TColgp_Array2OfPnt(1, 3, 1, 2), TColStd_Array2OfReal(1, 3, 1, 2)
    for i, point in enumerate(controls, start=1):
        for j in (1, 2):
            delta = np.array([j - 1., 0., 0.])
            if warped and i == 2 and j == 2:
                delta[2] = .05
            points.SetValue(i, j, gp_Pnt(*(np.array(point) + delta)))
            value = [1e-12, 1e-12, 1.][i - 1]
            if unequal_weights and i == 1 and j == 2:
                value = 9e-11
            weights.SetValue(i, j, value)
    uk, vk = TColStd_Array1OfReal(1, 2), TColStd_Array1OfReal(1, 2)
    um, vm = TColStd_Array1OfInteger(1, 2), TColStd_Array1OfInteger(1, 2)
    for i, value in enumerate([2., 7.] if shifted_knots else [0., 1.], start=1):
        uk.SetValue(i, value)
        um.SetValue(i, 3)
    for i, value in enumerate([3., 9.] if shifted_knots else [0., 1.], start=1):
        vk.SetValue(i, value)
        vm.SetValue(i, 2)
    surface = Geom_BSplineSurface(points, weights, uk, vk, um, vm, 2, 1)
    face = (BRepBuilderAPI_MakeFace(surface, 2., 7., 4., 8., 1e-7).Face() if shifted_knots
            else BRepBuilderAPI_MakeFace(surface, 1e-7).Face())
    return BRepAdaptor_Surface(face)


@pytest.mark.parametrize("shifted_knots", [False, True])
def test_exact_affine_tensor_wall_and_parameter_interval(shifted_knots):
    pytest.importorskip("OCP")
    proof = unfold._surface_proof(_ruled_surface(shifted_knots=shifted_knots))
    assert proof["kind"] == "extruded_boundary"
    assert proof["axis"] == pytest.approx([1., 0., 0.])
    assert proof["v0"] == pytest.approx(1. / 6. if shifted_knots else 0.)
    assert proof["v1"] == pytest.approx(5. / 6. if shifted_knots else 1.)


@pytest.mark.parametrize("fault", ["warped", "unequal_weights"])
def test_lookalike_non_affine_ruled_wall_is_rejected(fault):
    pytest.importorskip("OCP")
    with pytest.raises(unfold.UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        unfold._surface_proof(_ruled_surface(**{fault: True}))


def test_native_spline_pole_budget_is_checked_before_segment(monkeypatch):
    pytest.importorskip("OCP")
    monkeypatch.setattr(unfold, "MAX_CURVE_POLES", 2)
    with pytest.raises(unfold.UnfoldUnavailable, match="NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED"):
        _bezier_proof([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]])
