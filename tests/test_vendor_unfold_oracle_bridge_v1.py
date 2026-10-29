"""Pure independently assembled capture bundles; no physical/native inputs."""
import copy
import math

import numpy as np
import pytest

from backend.vendor_unfold_geometry_oracle_v1 import FaceEvidence, GeometryOracleError
from backend.vendor_unfold_oracle_bridge_v1 import CapturedFace, NativeOracleCapture, validate_captured_development
from backend.vendor_unfold_preview_v5 import develop_faces_v5
from tests.test_vendor_unfold_coplanar_stitch_v1 import split_two_bend_skin


def capture_charts(faces):
    capture = NativeOracleCapture()
    for face in faces:
        p, t = face.vertices.copy(), face.triangles.copy()
        cross = np.cross(p[t[:, 1]]-p[t[:, 0]], p[t[:, 2]]-p[t[:, 0]])
        normals = cross/np.linalg.norm(cross, axis=1)[:, None]
        middle = folded = radius = sweep = None
        if face.kind == "plane":
            x = p[t[0, 1]]-p[t[0, 0]]
            x /= np.linalg.norm(x)
            y = np.cross(normals[0], x)
            chart = (p-p[0]) @ np.stack((x, y)).T
        else:
            offset = p-face.origin
            u = np.arctan2(offset @ face.y_axis, offset @ face.x_axis)
            u += 2*math.pi*np.rint(((face.u0+face.u1)/2-u)/(2*math.pi))
            v = offset @ face.axis
            chart = np.column_stack((face.radius*u, v))
            triangle_mid = u[t].mean(axis=1)
            normals = np.cos(triangle_mid)[:, None]*face.x_axis+np.sin(triangle_mid)[:, None]*face.y_axis
            mid = (face.u0+face.u1)/2
            middle = np.array([[face.radius*mid, face.v0], [face.radius*mid, face.v1]])
            folded = np.array([face.origin+face.radius*(math.cos(mid)*face.x_axis+math.sin(mid)*face.y_axis)+v*face.axis
                               for v in (face.v0, face.v1)])
            radius, sweep = face.radius, math.degrees(face.u1-face.u0)
        edge_vertices = {}
        for edge, endpoints in face.edges.items():
            a, b = endpoints
            vector = b-a
            parameter = (p-a) @ vector/np.dot(vector, vector)
            residual = np.linalg.norm(p-a-parameter[:, None]*vector, axis=1)
            ids = np.flatnonzero((residual < 1e-10)&(parameter >= -1e-10)&(parameter <= 1+1e-10))
            edge_vertices[edge] = tuple(int(i) for i in ids[np.argsort(parameter[ids])])
            capture.edge_owners.setdefault(edge, set()).add(face.index)
        capture.observed[face.index] = face.kind
        evidence = FaceEvidence(face.index, p, t, chart, normals, 1)
        capture.faces[face.index] = CapturedFace(evidence, face.kind, edge_vertices,
                                                {"continuous_area_check": "NOT_ESTABLISHED"},
                                                middle, folded, radius, sweep)
    return capture


def case():
    faces, axes, candidates = split_two_bend_skin()
    candidates[0]["feature_id"] = "aef1248e-20fd-4e1d-8d38-ef8e5b34f702"
    candidates[1]["feature_id"] = "49fc1e2c-8e6f-4d56-82ec-0f915775e128"
    payload = develop_faces_v5(faces, axes, candidates, "a"*64)
    return faces, axes, candidates, capture_charts(faces), payload


def test_all_original_member_slices_internal_seams_and_uuid_bends_verify():
    faces, axes, candidates, capture, payload = case()
    before = copy.deepcopy(payload)
    report = validate_captured_development(payload, capture, faces, axes, candidates)
    assert report["face_count"] == 6
    assert report["seam_count"] == 5  # includes native split-flange internal edge
    assert report["bend_count"] == 2
    assert report["continuous_area_check"] == "NOT_ESTABLISHED"
    assert report["all_selected_shared_native_edges_checked"] is True
    assert payload == before


def test_missing_edge_polygon_has_no_sampled_fallback():
    faces, axes, candidates, capture, payload = case()
    capture.faces[2].edge_vertices[102] = None
    with pytest.raises(GeometryOracleError, match="POLYGON_UNAVAILABLE"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_unequal_seam_tessellation_refuses_without_resampling():
    faces, axes, candidates, capture, payload = case()
    capture.faces[2].edge_vertices[102] = capture.faces[2].edge_vertices[102][:1]
    with pytest.raises(GeometryOracleError, match="POLYGON_UNAVAILABLE_OR_UNEQUAL"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_original_native_axis_line_must_match_independent_surface_geometry():
    faces, axes, candidates, capture, payload = case()
    axes[0]["canonical_line_start"][0] += 100.
    with pytest.raises(GeometryOracleError, match="EXTRACTED_AXIS_NATIVE"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_explicit_feature_ownership_cannot_swap_uuid_bends():
    faces, axes, candidates, capture, payload = case()
    ownership = {candidates[0]["feature_id"]: (4, 10), candidates[1]["feature_id"]: (1, 7)}
    with pytest.raises(GeometryOracleError, match="EXPLICIT_NATIVE_FEATURE"):
        validate_captured_development(payload, capture, faces, axes, candidates, feature_face_ownership=ownership)


def test_incomplete_capture_and_changed_source_vertex_refuse():
    faces, axes, candidates, capture, payload = case()
    capture.observed.pop(11)
    with pytest.raises(GeometryOracleError, match="COMPLETE_NATIVE_CAPTURE"):
        validate_captured_development(payload, capture, faces, axes, candidates)
    faces, axes, candidates, capture, payload = case()
    faces[0].vertices[0, 1] += .1
    with pytest.raises(GeometryOracleError, match="CAPTURE_SOURCE_IDENTITY"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_reversed_native_polygon_order_is_handled_by_full_node_identity():
    faces, axes, candidates, capture, payload = case()
    capture.faces[2].edge_vertices[102] = capture.faces[2].edge_vertices[102][::-1]
    assert validate_captured_development(payload, capture, faces, axes, candidates)["seam_count"] == 5


def test_candidate_nominal_metadata_is_independently_checked():
    faces, axes, candidates, capture, payload = case()
    candidates[0]["target_radius"] = 99.
    with pytest.raises(GeometryOracleError, match="CANDIDATE_NATIVE_METADATA"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_selected_shared_edge_cannot_disappear_from_capture_incidence():
    faces, axes, candidates, capture, payload = case()
    capture.edge_owners.pop(102)
    with pytest.raises(GeometryOracleError, match="COMPLETE_NATIVE_EDGE_INCIDENCE"):
        validate_captured_development(payload, capture, faces, axes, candidates)


def test_nonmatching_full_seam_nodes_refuse_even_when_endpoints_look_close():
    faces, axes, candidates, capture, payload = case()
    # Replace one recorded seam endpoint by an actual non-seam source vertex.
    capture.faces[2].edge_vertices[102] = (0, 3)
    with pytest.raises(GeometryOracleError, match="NATIVE_SEAM_NODE_IDENTITY"):
        validate_captured_development(payload, capture, faces, axes, candidates)
