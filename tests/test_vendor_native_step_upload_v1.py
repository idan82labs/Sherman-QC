"""Synthetic source-only checks; never parse a physical CAD or touch admission."""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import numpy as np
import pytest

from backend.vendor_native_step_upload_v1 import (
    PROFILE, _assert_native_line_mesh_frame, _axis_rows, _cad_source, _folded_ply, _hash, _validate_result,
    preprocess_vendor_step,
)
from backend.vendor_unfold_contract_v2 import PreviewContractError, canonical_candidates
from backend.vendor_unfold_preview import SkinFace, _matching_axes


HASH = "a" * 64


def _axis(start=(0., 0., 0.), end=(10., 0., 0.), *, sweeps=(90., 90.)):
    return {"radii_mm": [2., 3.], "angular_sweeps_deg": list(sweeps),
        "surface_count": 2, "canonical_line_start": list(start),
        "canonical_line_end": list(end)}


def _result(rows, mesh):
    return {"profile": PROFILE, "source_sha256": HASH, "status": "AVAILABLE",
        "reason": None, "bends": rows, "folded_mesh_sha256": _hash(mesh),
        "folded_mesh_bytes": len(mesh), "folded_vertex_count": 3,
        "folded_triangle_count": 1, "source_pins_sha256": "b" * 64}


def test_native_candidate_lines_match_v4_without_tolerance_repair():
    axes = [_axis(), _axis((0., 20., 0.), (10., 20., 0.), sweeps=(45., 45.))]
    rows = _axis_rows(axes, 1.5)
    assert _matching_axes(rows, axes) == [0, 1]
    assert canonical_candidates(rows)[0]["line_start"] == [0., 0., 0.]
    assert rows[0]["target_angle"] == 90.
    assert rows[0]["target_radius"] == 2.
    mesh = b"ply\nend_header\n"
    specs = _validate_result(_result(rows, mesh), HASH, 1.5)
    assert [(bend.bend_id, bend.target_angle, bend.target_radius,
             bend.tolerance_angle, bend.tolerance_radius) for bend in specs] == [
                 ("B1", 90., 2., 1.5, None), ("B2", 45., 2., 1.5, None)]
    assert list(specs[0].bend_line_start) == [0., 0., 0.]
    assert specs[0].to_dict()["bend_line_end"] == [10., 0., 0.]


@pytest.mark.parametrize("axis, reason", [
    (_axis(sweeps=(90., 92.)), "INCONSISTENT_NATIVE_BEND_METADATA"),
    (_axis(sweeps=(176., 176.)), "INCONSISTENT_NATIVE_BEND_METADATA"),
    (_axis(sweeps=(90., 90.001)), "INCONSISTENT_NATIVE_BEND_METADATA"),
    ({**_axis(), "surface_count": 3}, "SPLIT_OR_AMBIGUOUS_BEND_SKINS"),
    ({**_axis(), "radii_mm": [3., 2.]}, "INCONSISTENT_NATIVE_BEND_METADATA"),
    ({**_axis(), "canonical_line_end": [0., 0., 0.]}, "INVALID_CANDIDATE_LINE"),
])
def test_ambiguous_native_bends_fail_closed(axis, reason):
    with pytest.raises(PreviewContractError, match=reason):
        _axis_rows([axis], 1.)


def test_folded_ply_preserves_source_coordinates_and_triangle_indices():
    faces = [SkinFace(0, "plane", np.array([[1., 2., 3.], [4., 2., 3.], [1., 5., 3.]]),
                      np.array([[0, 1, 2]], dtype=np.int32))]
    mesh, vertex_count, triangle_count = _folded_ply(faces)
    assert mesh.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in mesh
    assert b"element face 1\n" in mesh
    assert vertex_count == 3 and triangle_count == 1
    raw = mesh.split(b"end_header\n", 1)[1]
    assert np.frombuffer(raw[:36], dtype="<f4").reshape(3, 3).tolist() == [
        [1., 2., 3.], [4., 2., 3.], [1., 5., 3.]]
    assert raw[36] == 3
    assert np.frombuffer(raw[37:49], dtype="<i4").tolist() == [0, 1, 2]


def test_folded_mesh_rejects_nonfinite_and_bad_index():
    for points, indices in [
        ([[0., 0., 0.], [1., 0., 0.], [float("nan"), 1., 0.]], [[0, 1, 2]]),
        ([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 3]]),
    ]:
        with pytest.raises(PreviewContractError, match="INVALID_NATIVE_FOLDED_MESH"):
            _folded_ply([SkinFace(0, "plane", np.asarray(points), np.asarray(indices))])


def test_native_line_and_folded_mesh_must_share_source_frame():
    faces = [SkinFace(0, "plane", np.array([[0., 0., 0.], [10., 0., 0.],
                                           [0., 10., 0.], [10., 10., 0.]]),
                      np.array([[0, 1, 2]], dtype=np.int32))]
    _assert_native_line_mesh_frame(_axis_rows([_axis((1., 1., 0.), (9., 1., 0.))], 1.), faces)
    with pytest.raises(PreviewContractError, match="NATIVE_BEND_LINE_FOLDED_FRAME_MISMATCH"):
        _assert_native_line_mesh_frame(_axis_rows([_axis((100., 1., 0.), (109., 1., 0.))], 1.), faces)


def test_float32_folded_triangle_collapse_is_refused():
    faces = [SkinFace(0, "plane", np.array([[100000., 0., 0.], [100000.001, 0., 0.],
                                            [100000., 1., 0.]]),
                      np.array([[0, 1, 2]], dtype=np.int32))]
    with pytest.raises(PreviewContractError, match="FLOAT32_DEGENERATE_FOLDED_TRIANGLE"):
        _folded_ply(faces)


def test_non_step_input_abstains_before_source_or_corpus_access():
    bends, mesh, source = preprocess_vendor_step("not-a-real-file.stl", HASH, 1.)
    assert bends == [] and mesh is None
    assert source["status"] == "UNAVAILABLE"
    assert source["reason"] == "STEP_SOURCE_REQUIRED"
    assert source["units_verified"] is False
    assert source["manufacturing_ready"] is False


def test_boolean_tolerance_is_not_accepted_as_numeric():
    with pytest.raises(PreviewContractError, match="INVALID_ANGLE_TOLERANCE"):
        _axis_rows([_axis()], True)


def test_nonright_angle_is_sweep_nominal_not_included_angle_or_flat_claim():
    rows = _axis_rows([_axis(sweeps=(135., 135.))], 1.)
    assert rows[0]["target_angle"] == 135.
    meta = _cad_source("AVAILABLE", None, HASH)
    assert meta["angle_rule"] == "CYLINDER_SWEEP_ROTATION_FROM_FLAT"
    assert meta["angle_limitation"] == "BEND_DIRECTION_AND_INCLUDED_ANGLE_UNVERIFIED"
    assert meta["flat_preview_status"] == "NOT_RUN_SEPARATE_BOUNDED_V4_GATE"
    assert meta["line_limitation"] == "GEOMETRIC_MID_RADIUS_NOT_PHYSICAL_NEUTRAL_AXIS_OR_K_FACTOR_BLANK"
    assert meta["winding_oriented"] is False


def test_result_rejects_candidate_metadata_or_identity_tampering():
    rows = _axis_rows([_axis()], 1.)
    mesh = b"ply\nend_header\n"
    result = _result(rows, mesh)
    result["bends"][0]["target_angle"] = 0.
    with pytest.raises(PreviewContractError, match="INVALID_NATIVE_STEP_BEND_METADATA"):
        _validate_result(result, HASH, 1.)
    result = _result(_axis_rows([_axis()], 1.), mesh)
    result["source_sha256"] = "b" * 64
    with pytest.raises(PreviewContractError, match="INVALID_NATIVE_STEP_RESULT"):
        _validate_result(result, HASH, 1.)


def test_independent_synthetic_step_through_bounded_upload_and_flat_preview(tmp_path):
    """One real OCC/STEP job, synthetic only; run this test explicitly."""
    pytest.importorskip("OCP")
    from backend import vendor_unfold_worker_v3 as supervision
    try:
        supervision._group_rows(os.getpgrp())
    except (OSError, subprocess.SubprocessError):
        pytest.skip("process-group monitor is unavailable in this sandbox")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
    from backend.vendor_unfold_worker_v3 import build_unfold_preview_bounded

    # Distinct extrusion, radii, and asymmetric legs from the retained tiny
    # synthetic STEP tests. The arc faces are generated by OCC, not mocked.
    profile = [(4., -12.), (4., 0.), (0., 4.), (-8., 4.),
               (-8., 5.), (0., 5.), (5., 0.), (5., -12.)]
    wire = BRepBuilderAPI_MakeWire()
    for i, point in enumerate(profile):
        following = profile[(i + 1) % len(profile)]
        if i in (1, 5):
            radius = 4. if i == 1 else 5.
            circle = gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), radius)
            edge = BRepBuilderAPI_MakeEdge(circle, 0., math.pi / 2.).Edge()
            if i == 5:
                edge.Reverse()
        else:
            edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*point, 0), gp_Pnt(*following, 0)).Edge()
        wire.Add(edge)
    solid = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire.Wire()).Face(),
                                  gp_Vec(0, 0, 16)).Shape()
    source = tmp_path / "independent-quarter-bend.step"
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_AsIs)
    writer.Write(str(source))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    bends, folded_mesh, metadata = preprocess_vendor_step(source, source_hash, 1.0)
    assert metadata["status"] == "AVAILABLE", metadata
    assert len(bends) == 1
    assert bends[0].target_angle == pytest.approx(90., abs=1e-6)
    assert bends[0].target_radius == pytest.approx(4., abs=1e-6)
    assert folded_mesh is not None and folded_mesh.startswith(b"ply\n")
    assert metadata["folded_triangle_count"] > 0
    assert metadata["folded_mesh_sha256"] == hashlib.sha256(folded_mesh).hexdigest()

    candidate = {"feature_id": bends[0].bend_id,
                 "line_start": list(bends[0].bend_line_start),
                 "line_end": list(bends[0].bend_line_end)}
    flat = build_unfold_preview_bounded(source, source_hash, [candidate])
    assert flat["status"] == "AVAILABLE", flat
    assert len(flat["bend_lines"]) == 1
    assert flat["bend_lines"][0]["feature_id"] == bends[0].bend_id
    assert flat["source_sha256"] == source_hash
    assert np.max(np.abs(np.array(flat["vertices"]).reshape(-1, 3)[:, 2])) == 0.
