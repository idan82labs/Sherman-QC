"""Synthetic contracts for one shared registration and spatial bend identity.

No server, physical input, model, promotion, or research-status records are loaded.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def detector():
    # Load only the detector, avoiding feature_detection's unrelated import facade.
    name = "_regular_shared_frame_detector_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "backend/feature_detection/bend_detector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def alignment():
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [120, -230, 40]
    return {
        "status": "AVAILABLE",
        "geometry_frame": "reference_mesh_raw",
        "raw_scan_to_reference_transform": transform.tolist(),
    }


def test_aligned_points_are_validated_without_applying_pose_twice():
    from domains.bend.services.inspection_frame import validate_alignment, validate_aligned_points

    pose = alignment()
    # These are already in the CAD frame, despite the nonidentity stored raw pose.
    points = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    validated = validate_alignment(pose)
    np.testing.assert_allclose(validated["raw_scan_to_reference_transform"], pose["raw_scan_to_reference_transform"])
    np.testing.assert_array_equal(validate_aligned_points(points, pose), points)


@pytest.mark.parametrize("invalid", [None, {}, {"status": "AVAILABLE"}])
def test_alignment_requires_explicit_complete_evidence(invalid):
    from domains.bend.services.inspection_frame import validate_alignment
    with pytest.raises(ValueError):
        validate_alignment(invalid)


@pytest.mark.parametrize("field,value", [
    ("status", "UNKNOWN"),
    ("status", "UNAVAILABLE"),
    ("geometry_frame", "scan_raw"),
    ("geometry_frame", None),
    ("alignment_abstained", True),
])
def test_alignment_cannot_claim_an_unverified_shared_frame(field, value):
    from domains.bend.services.inspection_frame import validate_alignment
    pose = alignment()
    pose[field] = value
    with pytest.raises(ValueError):
        validate_alignment(pose)


@pytest.mark.parametrize("mutation", ["missing", "shape", "nan", "inf", "projective", "singular"])
def test_invalid_transform_is_rejected(mutation):
    from domains.bend.services.inspection_frame import validate_alignment
    pose = alignment()
    matrix = np.asarray(pose["raw_scan_to_reference_transform"])
    if mutation == "missing":
        pose.pop("raw_scan_to_reference_transform")
    elif mutation == "shape":
        pose["raw_scan_to_reference_transform"] = np.eye(3).tolist()
    else:
        if mutation == "nan":
            matrix[0, 0] = np.nan
        elif mutation == "inf":
            matrix[1, 3] = np.inf
        elif mutation == "projective":
            matrix[3, 0] = 0.1
        elif mutation == "singular":
            matrix[1, :3] = matrix[0, :3]
        pose["raw_scan_to_reference_transform"] = matrix.tolist()
    with pytest.raises(ValueError):
        validate_alignment(pose)


@pytest.mark.parametrize("points", [[], np.empty((0, 3)), [[1, 2]], [1, 2, 3], [[np.nan, 0, 0]], [[0, np.inf, 0]]])
def test_unusable_aligned_point_cloud_cannot_enter_detection(points):
    from domains.bend.services.inspection_frame import validate_aligned_points
    with pytest.raises(ValueError):
        validate_aligned_points(points, alignment())


def bends(module, center=(0, 0, 0), axis=(1, 0, 0), measured_angle=90):
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    start, end = center - 50 * axis, center + 50 * axis
    cad = module.BendSpecification("CAD-A", 90, 2, np.array([-50., 0, 0]), np.array([50., 0, 0]))
    flange = SimpleNamespace(area=1000., boundary_points=None)
    detected = module.DetectedBend("SCAN-A", measured_angle, 2, start, end, axis, 0.95, flange, flange, inlier_count=200)
    return detected, cad


def matcher(module, **kwargs):
    return module.ProgressiveBendMatcher(strict_spatial_matching=True, max_center_offset=40, **kwargs)


def test_identical_bend_shape_at_other_end_is_not_a_match(detector):
    detected, cad = bends(detector, center=(0, 289.3, 0))
    report = matcher(detector).match([detected], [cad])
    assert report.matches[0].detected_bend is None
    assert report.matches[0].physical_completion_state == "UNKNOWN"
    assert report.unmatched_detections == [detected]


def test_spatial_rejection_survives_disabled_position_cost_and_recovery(detector):
    detected, cad = bends(detector, center=(0, 289.3, 0), measured_angle=115)
    strict = matcher(detector, position_mode="disabled", recovery_enabled=True)
    assert np.isinf(strict._compute_match_cost(detected, cad, use_position_cost=False, angle_limit=32))
    assert strict.match([detected], [cad]).matches[0].detected_bend is None


def test_midpoint_gate_accepts_nearby_and_rejects_beyond_limit(detector):
    near, cad = bends(detector, center=(0, 39.9, 0))
    far, _ = bends(detector, center=(0, 40.1, 0))
    strict = matcher(detector)
    assert np.isfinite(strict._compute_match_cost(near, cad))
    assert np.isinf(strict._compute_match_cost(far, cad))


@pytest.mark.parametrize("angle,accepted", [(0, True), (14.9, True), (15.1, False), (90, False), (180, True)])
def test_bend_axis_is_unoriented_and_has_hard_angular_gate(detector, angle, accepted):
    radians = np.deg2rad(angle)
    detected, cad = bends(detector, axis=(np.cos(radians), np.sin(radians), 0))
    assert np.isfinite(matcher(detector)._compute_match_cost(detected, cad)) == accepted


def test_axis_gate_uses_visible_endpoints_not_stale_direction(detector):
    detected, cad = bends(detector, axis=(0, 1, 0))
    detected.bend_line_direction = np.array([1., 0, 0])
    assert np.isinf(matcher(detector)._compute_match_cost(detected, cad))


@pytest.mark.parametrize("kind", ["zero", "nan", "inf"])
def test_invalid_marker_geometry_never_becomes_a_match(detector, kind):
    detected, cad = bends(detector)
    if kind == "zero":
        detected.bend_line_end = detected.bend_line_start.copy()
    else:
        detected.bend_line_end[0] = np.nan if kind == "nan" else np.inf
    assert np.isinf(matcher(detector)._compute_match_cost(detected, cad))


def test_reversed_endpoints_preserve_identity_and_measured_angle(detector):
    detected, cad = bends(detector, axis=(-1, 0, 0), measured_angle=91.25)
    report = matcher(detector).match([detected], [cad])
    match = report.matches[0]
    assert match.detected_bend is detected
    assert match.angle_deviation == pytest.approx(1.25)
    assert match.line_center_deviation_mm == pytest.approx(0)


def isolated_worker_namespace():
    """Execute the actual worker orchestration without importing its service facade.

    Boundaries are replaced by synthetic functions below; ordering, forwarding,
    persistence, and error handling remain the real run_worker implementation.
    """
    import ast
    import argparse
    import json
    import math
    import re
    import time
    import traceback
    from domains.bend.services.inspection_frame import validate_alignment, validate_aligned_points

    source = ast.parse((ROOT / "backend/bend_analysis_worker.py").read_text())
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {
        "run_worker", "_json_safe", "_sanitize_error_message", "_resolve_scan_state",
    }]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(np=np, argparse=argparse, json=json, math=math, re=re, time=time, traceback=traceback,
                     sys=sys, Path=Path, validate_alignment=validate_alignment, validate_aligned_points=validate_aligned_points)
    exec(compile(module, str(ROOT / "backend/bend_analysis_worker.py"), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("alignment_state", ["available", "unavailable", "abstained"])
def test_worker_uses_one_registration_before_detection_and_fails_closed(tmp_path, alignment_state):
    import json
    namespace = isolated_worker_namespace()
    calls, events = [], []
    points = np.array([[0., 0, 0], [10., 0, 0], [0., 10, 0]])
    pose = alignment()
    cfg = SimpleNamespace(cad_import_deflection=0.1, to_dict=lambda: {})
    report = SimpleNamespace(processing_time_ms=0, to_dict=lambda: {"matches": [], "alignment": pose})
    details = SimpleNamespace(cad_vertices=points.copy(), cad_triangles=np.array([[0, 1, 2]]), scan_state=None,
                              to_dict=lambda: {})

    def aligned(**kwargs):
        calls.append("alignment")
        if alignment_state == "unavailable":
            raise ValueError("Synthetic registration unavailable")
        if alignment_state == "abstained":
            pose["alignment_abstained"] = True
        return {"alignment": pose, "aligned_points": points,
                "aligned_scan_path": tmp_path / "aligned_scan.ply", "deviations_path": None,
                "deviation_stats": {}}

    def pipeline(**kwargs):
        calls.append("pipeline")
        assert calls == ["alignment", "pipeline"]
        np.testing.assert_array_equal(kwargs["aligned_scan_points"], points)
        np.testing.assert_allclose(kwargs["scan_alignment"]["raw_scan_to_reference_transform"],
                                   pose["raw_scan_to_reference_transform"])
        assert kwargs["scan_alignment"]["geometry_frame"] == "reference_mesh_raw"
        return report, details

    def export_reference(**kwargs):
        (tmp_path / "reference_mesh.ply").write_text("synthetic mesh placeholder")

    namespace.update(
        _emit=lambda tag, payload: events.append((tag, payload)),
        load_bend_runtime_config=lambda: cfg,
        _export_aligned_scan_visualization=aligned,
        run_progressive_bend_inspection=pipeline,
        export_reference_mesh_from_vertices=export_reference,
        augment_progressive_bend_report=lambda value, details: value,
        promote_structured_count_reporting=lambda value: value,
        _load_qualification_context=lambda: {"status": "UNKNOWN"},
        finalize_bend_report=lambda value, **kwargs: value,
        format_bend_decision_table=lambda value: "Synthetic bend table",
        build_bend_artifact_bundle=lambda **kwargs: SimpleNamespace(artifacts={"bend_overlay_manifest_url": "synthetic"}),
    )
    args = SimpleNamespace(output_dir=str(tmp_path), cad_path="synthetic-cad", scan_path="synthetic-scan",
                           part_id="synthetic-part", part_name="Synthetic", job_id="synthetic-job",
                           tolerance_angle=1.0, tolerance_radius=0.5)
    result = namespace["run_worker"](args)
    if alignment_state == "available":
        assert result == 0, events
        assert calls == ["alignment", "pipeline"]
        saved = json.loads((tmp_path / "bend_inspection_report.json").read_text())
        assert saved["alignment"]["geometry_frame"] == "reference_mesh_raw"
        assert [tag for tag, _ in events].count("RESULT") == 1
    else:
        assert result != 0
        assert calls == ["alignment"]
        assert not (tmp_path / "bend_inspection_report.json").exists()
        assert "RESULT" not in [tag for tag, _ in events]


def test_local_angle_measurement_does_not_publish_borrowed_cad_geometry_as_measured(detector):
    detected, cad = bends(detector, measured_angle=92.25)
    match = detector.BendMatch(
        cad_bend=cad,
        detected_bend=detected,
        angle_deviation=2.25,
        radius_deviation=0.0,
        assignment_source="CAD_LOCAL_NEIGHBORHOOD",
        expected_line_length_mm=100.0,
        actual_line_length_mm=100.0,
        line_length_deviation_mm=0.0,
        line_start_deviation_mm=0.0,
        line_end_deviation_mm=0.0,
        line_center_deviation_mm=0.0,
        expected_arc_length_mm=np.pi,
        actual_arc_length_mm=np.pi,
        arc_length_deviation_mm=0.0,
        expected_edge_distance_mm=70.0,
        measured_edge_distance_mm=70.0,
        edge_distance_deviation_mm=0.0,
        measured_edge_tip1=np.array([0., 10, 0]),
        measured_edge_tip2=np.array([0., 0, 10]),
    )
    payload = match.to_dict()
    assert payload["measured_angle"] == pytest.approx(92.25)
    assert payload["angle_deviation"] == pytest.approx(2.25)
    assert payload["detected_line_geometry_source"] == "CAD_REFERENCE_PLACEHOLDER"
    for key in (
        "measured_radius", "radius_deviation", "actual_line_length_mm",
        "line_length_deviation_mm", "line_start_deviation_mm", "line_end_deviation_mm",
        "line_center_deviation_mm", "actual_arc_length_mm", "arc_length_deviation_mm",
        "measured_edge_distance_mm", "edge_distance_deviation_mm",
        "measured_edge_tip1", "measured_edge_tip2",
    ):
        assert payload[key] is None, key
    # Nominal values remain useful, explicitly CAD-derived report context.
    assert payload["target_radius"] == 2
    assert payload["expected_line_length_mm"] == 100
    assert payload["expected_edge_distance_mm"] == 70
