from __future__ import annotations

import math

import numpy as np

from scripts import compile_cad_bend_template_v2 as compiler
from scripts.extract_step_topology_bend_axes import _cylinder_turn_midpoint_center
from scripts.localize_cad_bends_on_scan_v2 import localize


def test_cylinder_turn_midpoint_center_uses_radius_and_half_turn():
    point = _cylinder_turn_midpoint_center(
        location=np.array([0.0, 0.0, 0.0]),
        native_direction=np.array([1.0, 0.0, 0.0]),
        x_direction=np.array([0.0, 1.0, 0.0]),
        y_direction=np.array([0.0, 0.0, 1.0]),
        radius=4.0,
        u0=0.0,
        u1=math.pi / 2.0,
        v0=0.0,
        v1=10.0,
    )
    assert np.allclose(point, [5.0, 2.0 * math.sqrt(2.0), 2.0 * math.sqrt(2.0)])


def test_v2_template_preserves_event_count_and_selects_canonical_lines(tmp_path, monkeypatch):
    step = tmp_path / "part.step"
    step.write_bytes(b"step")
    monkeypatch.setattr(compiler, "extract_step_bend_axes", lambda _: {
        "bend_axes": [{
            "bend_id": "STEP_BEND_001",
            "face_indices": [1, 2],
            "line_start": [0.0, 0.0, 0.0],
            "line_end": [10.0, 0.0, 0.0],
            "canonical_line_start": [0.0, 2.0, 0.0],
            "canonical_line_end": [10.0, 2.0, 0.0],
            "canonical_line_rule": "NEUTRAL_RADIUS_HALF_ANGULAR_TURN",
            "radii_mm": [1.0, 3.0],
            "angular_sweeps_deg": [90.0, 90.0],
            "axial_span_mm": 10.0,
        }],
        "rejected_groups": [],
        "contract": {},
    })
    payload = compiler.compile_template(step)
    assert payload["event_count"] == 1
    assert payload["events"][0]["cad_line_start_mm"] == [0.0, 2.0, 0.0]
    assert payload["events"][0]["curvature_center_line_start_mm"] == [0.0, 0.0, 0.0]
    assert payload["events"][0]["relation"] == "PAIRED_SKINS_TO_ONE_NEUTRAL_SURFACE_EVENT"


def test_v2_rejects_v1_template_before_scan_read(tmp_path):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    try:
        localize({"schema": "SHERMAN_QC_CAD_BEND_TEMPLATE_V1"}, scan, {})
    except ValueError as exc:
        assert str(exc) == "invalid CAD bend template V2"
    else:
        raise AssertionError("V2 must reject a V1 template")
