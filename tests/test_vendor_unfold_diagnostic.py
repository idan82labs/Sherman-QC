"""Synthetic-only diagnostic contracts; frozen V2 acceptance remains untouched."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

from backend import vendor_unfold_diagnostic as diagnostic


def _source_fixtures():
    # Only the named source fixture file is loaded. Its geometry-writing helper
    # is called below exclusively for a newly authored synthetic STEP in tmp.
    return runpy.run_path(str(Path(__file__).with_name("test_vendor_unfold_preview.py")))


def _wall_case(*, mismatched_curve=False, wrong_span=False):
    engine = diagnostic.engine
    faces, axes, candidates = _source_fixtures()["_case"]()
    span = 2. if wrong_span else 1.
    angles = np.linspace(0., math.pi / 2., 9)
    points = np.array([[2. + v, -5. + math.cos(u), 10. + math.sin(u)]
                       for u in angles for v in (0., span)])
    triangles = np.array([[2 * i, 2 * i + 2, 2 * i + 3] for i in range(8)]
                        + [[2 * i, 2 * i + 3, 2 * i + 1] for i in range(8)], dtype=np.int32)
    wall = engine.SkinFace(6, "cylinder", points, triangles, edges={80: None, 81: None},
        origin=np.array([2., -5., 10.]), axis=np.array([1., 0., 0.]),
        x_axis=np.array([0., 1., 0.]), y_axis=np.array([0., 0., 1.]),
        radius=1., u0=0., u1=math.pi / 2., v0=0., v1=span)
    faces[0].edges[80], faces[3].edges[81] = None, None
    for face_index, edge, x, phase in [(0, 80, 2., 0.), (3, 81, 3., .1 if mismatched_curve else 0.)]:
        proof = engine.NativeCurve("circle", np.zeros((2, 3)), center=np.array([x, -5., 10.]),
            axis=np.array([1., 0., 0.]), x_axis=np.array([0., math.cos(phase), math.sin(phase)]),
            y_axis=np.array([0., -math.sin(phase), math.cos(phase)]), radius=1., u0=0., u1=math.pi / 2.)
        faces[face_index].edge_curves[edge] = proof
        wall.edge_curves[edge] = proof
    faces.append(wall)
    return faces, axes, candidates


def _call_synthetic(tmp_path, monkeypatch, faces, axes, candidates):
    path = tmp_path / "authored-synthetic.step"
    path.write_bytes(b"synthetic test transport; native reader is explicitly stubbed")
    monkeypatch.setattr(diagnostic.engine, "_read_native", lambda path: (faces, axes))
    monkeypatch.setattr(diagnostic, "_native_types", lambda path, faces: {
        face.index: {"surface_type": "Synthetic_" + face.kind,
                     "edge_curve_types": {edge: "SyntheticCurve" for edge in face.edges}} for face in faces})
    return diagnostic.diagnose_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)


def _assert_no_geometry(payload):
    assert payload["status"] != "AVAILABLE"
    assert payload["preview_status"] == "NOT_PRODUCED_DIAGNOSTIC_ONLY"
    assert not payload["manufacturing_ready"] and not payload["units_verified"]
    assert all(payload[key] == [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"))
    json.dumps(payload, allow_nan=False)


def test_frozen_v2_source_hash_has_not_changed():
    assert hashlib.sha256(Path(diagnostic.engine.__file__).read_bytes()).hexdigest() == diagnostic.FROZEN_ENGINE_SHA256


def test_passing_boundary_diagnostic_does_not_claim_available(tmp_path, monkeypatch):
    faces, axes, candidates = _wall_case()
    payload = _call_synthetic(tmp_path, monkeypatch, faces, axes, candidates)
    _assert_no_geometry(payload)
    assert payload["status"] == "DIAGNOSTIC_ONLY"
    assert payload["engine_boundary_status"] == "PASS"
    assert payload["diagnostics"]["first_failed_wall_record_index"] is None
    assert len(payload["boundary_checks"]) == 2
    assert all(row["frozen_wall_boundary_checks_pass"] for row in payload["boundary_checks"])


def test_mismatched_curved_opposite_rail_is_localized_without_relaxation(tmp_path, monkeypatch):
    faces, axes, candidates = _wall_case(mismatched_curve=True)
    original = copy.deepcopy(faces[0].edge_curves[80].x_axis)
    payload = _call_synthetic(tmp_path, monkeypatch, faces, axes, candidates)
    _assert_no_geometry(payload)
    assert payload["engine_boundary_status"] == "REFUSED"
    assert payload["engine_boundary_reason"] == "UNKNOWN_HOLE_WALL_BOUNDARY"
    first = payload["boundary_checks"][payload["diagnostics"]["first_failed_wall_record_index"]]
    assert first["skin_face_index"] == 0 and first["skin_edge_id"] == 80 and first["wall_face_index"] == 6
    assert first["wall_native_surface_type"] == "Synthetic_cylinder"
    assert first["axis_parallel_check"] and first["span_matches_thickness_check"]
    assert not first["translated_opposite_rail_check"]
    attempt = first["opposite_rail_attempts"][0]
    assert attempt["curve_certificate"]["proof_mode"] == "PHYSICAL_CIRCLE_LOCUS_BOUND"
    assert attempt["curve_certificate"]["reason"] == "TRANSLATED_NATIVE_RAIL_NOT_CERTIFIED"
    assert np.array_equal(original, faces[0].edge_curves[80].x_axis)


def test_wrong_wall_span_is_distinct_from_rail_correspondence_failure(tmp_path, monkeypatch):
    payload = _call_synthetic(tmp_path, monkeypatch, *_wall_case(wrong_span=True))
    _assert_no_geometry(payload)
    first = payload["boundary_checks"][0]
    assert first["wall_axial_span"] == 2. and first["known_thickness"] == 1.
    assert first["span_thickness_residual"] == 1.
    assert not first["span_matches_thickness_check"]
    assert payload["engine_boundary_reason"] == "UNKNOWN_HOLE_WALL_BOUNDARY"


def test_own_span_certificate_explains_topology_refusal_without_infinity_json():
    engine = diagnostic.engine
    left = engine.NativeCurve("spline", np.array([[0., 0., 0.], [1., 2., 0.], [2., 0., 0.]]),
        2, np.ones(3), np.array([2., 7.]), np.array([3, 3]))
    right = engine.NativeCurve("spline", np.array([[0., 0., 1.], [.5, 1., 1.], [1., 1., 1.], [2., 0., 1.]]),
        2, np.ones(4), np.array([2., 4., 7.]), np.array([3, 1, 3]))
    result = diagnostic._certificate(left, right, np.array([0., 0., 1.]), {})
    assert result["left_curve"]["own_native_knots_prefix"] == [2., 7.]
    assert not result["span_count_equal"]
    assert result["forward_deviation_bound"] is None
    assert result["forward_certificate_reason"] == "TOPOLOGY_OR_RELATIVE_WEIGHT_BOUND_UNCERTIFIABLE"
    assert not result["matched_by_frozen_engine"]
    json.dumps(result, allow_nan=False)


def test_diagnostic_record_budget_refuses_without_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "MAX_BOUNDARY_RECORDS", 0)
    payload = _call_synthetic(tmp_path, monkeypatch, *_wall_case(mismatched_curve=True))
    _assert_no_geometry(payload)
    assert payload["status"] == "UNAVAILABLE"
    assert payload["reason"] == "DIAGNOSTIC_BOUNDARY_RECORD_BUDGET_EXCEEDED"
    assert payload["boundary_checks"] == []
    assert payload["engine_boundary_reason"] == "UNKNOWN_HOLE_WALL_BOUNDARY"


def test_diagnostic_certificate_budget_counts_explicit_direction_checks(monkeypatch):
    engine = diagnostic.engine
    curve = engine.NativeCurve("spline", np.array([[0., 0., 0.], [1., 2., 0.], [2., 0., 0.]]),
        2, np.ones(3), np.array([0., 1.]), np.array([3, 3]))
    monkeypatch.setattr(diagnostic, "MAX_EXPLICIT_BOUND_EVALUATIONS", 0)
    counters = {}
    result = diagnostic._certificate(curve, copy.deepcopy(curve), np.zeros(3), counters)
    assert result["matched_by_frozen_engine"]
    assert result["forward_certificate_reason"] == "DIAGNOSTIC_EXPLICIT_BOUND_BUDGET_EXCEEDED"
    assert result["reverse_certificate_reason"] == "DIAGNOSTIC_EXPLICIT_BOUND_BUDGET_EXCEEDED"
    assert counters["certificate_rail_pairs"] == 1 and counters["explicit_bound_evaluations"] == 0


def test_numpy_boolean_line_and_full_circle_certificates_are_strict_json():
    engine = diagnostic.engine
    line = engine.NativeCurve("line", np.array([[0., 0., 0.], [1., 0., 0.]]))
    circle = engine.NativeCurve("circle", np.zeros((2, 3)), center=np.zeros(3),
        axis=np.array([0., 0., 1.]), x_axis=np.array([1., 0., 0.]), y_axis=np.array([0., 1., 0.]),
        radius=1., u0=0., u1=2 * math.pi)
    for curve in (line, circle):
        result = diagnostic._certificate(curve, copy.deepcopy(curve), np.zeros(3), {})
        assert type(result["matched_by_frozen_engine"]) is bool
        assert result["matched_by_frozen_engine"]
        json.dumps(result, allow_nan=False)


def test_bad_hash_and_changed_engine_do_not_parse_native(tmp_path, monkeypatch):
    path = tmp_path / "authored.step"
    path.write_bytes(b"synthetic")
    candidates = _source_fixtures()["_case"]()[2]
    monkeypatch.setattr(diagnostic.engine, "_read_native", lambda path: pytest.fail("native parse forbidden"))
    payload = diagnostic.diagnose_unfold_preview(path, "a" * 64, candidates)
    assert payload["reason"] == "CAD_HASH_MISMATCH"
    _assert_no_geometry(payload)
    monkeypatch.setattr(diagnostic, "FROZEN_ENGINE_SHA256", "b" * 64)
    payload = diagnostic.diagnose_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert payload["reason"] == "FROZEN_ENGINE_SOURCE_HASH_MISMATCH"
    _assert_no_geometry(payload)


def test_actual_authored_native_spline_hole_diagnostic_metadata(tmp_path):
    pytest.importorskip("OCP")
    path = tmp_path / "authored-spline-hole.step"
    candidates = _source_fixtures()["_native_synthetic_step"](path, spline_hole=True)
    payload = diagnostic.diagnose_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    _assert_no_geometry(payload)
    assert payload["status"] == "DIAGNOSTIC_ONLY", payload
    assert payload["engine_boundary_status"] == "PASS"
    assert any("Extrusion" in row["wall_native_surface_type"] or "BSpline" in row["wall_native_surface_type"]
               for row in payload["boundary_checks"])
    assert any(attempt["curve_certificate"].get("proof_mode") == "OWN_NATIVE_KNOT_RATIONAL_BEZIER_CERTIFICATE"
               for row in payload["boundary_checks"] for attempt in row["opposite_rail_attempts"])
