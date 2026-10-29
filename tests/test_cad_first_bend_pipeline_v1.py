from __future__ import annotations

import json

import numpy as np

from scripts import compile_cad_bend_template_v1 as compiler
from scripts import localize_cad_bends_on_scan_v1 as localizer


def _template() -> dict:
    return {
        "schema": "SHERMAN_QC_CAD_BEND_TEMPLATE_V1",
        "step_sha256": "a" * 64,
        "event_count": 1,
        "events": [{
            "event_id": "CAD_EVENT_0001",
            "source_bend_id": "BEND_1",
            "step_face_indices": [4, 9],
            "cad_line_start_mm": [0.0, 0.0, 0.0],
            "cad_line_end_mm": [10.0, 0.0, 0.0],
        }],
    }


def test_compile_template_is_compact_source_free_and_deterministic(tmp_path, monkeypatch):
    step = tmp_path / "part.step"
    step.write_bytes(b"STEP-CONTENT")
    monkeypatch.setattr(compiler, "extract_step_bend_axes", lambda _: {
        "bend_axes": [{
            "bend_id": "BEND_1",
            "face_indices": [4, 9],
            "line_start": [0.0, 0.0, 0.0],
            "line_end": [10.0, 0.0, 0.0],
            "radii_mm": [1.0, 2.0],
            "angular_sweeps_deg": [90.0, 90.0],
            "axial_span_mm": 10.0,
        }],
        "rejected_groups": [],
        "contract": {"source_free": True},
    })

    first = compiler.compile_template(step)
    second = compiler.compile_template(step)

    assert first == second
    assert first["event_count"] == 1
    assert first["events"][0]["relation"] == "MANY_FACES_TO_ONE_EVENT"
    assert str(step) not in json.dumps(first)


def test_untrusted_registration_abstains_without_predictions(tmp_path):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    result = localizer.localize(_template(), scan, {
        "alignment_source": "GLOBAL_ICP",
        "alignment_confidence": 0.99,
        "alignment_abstained": False,
        "raw_scan_to_cad_transform": np.eye(4).tolist(),
    })

    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "UNTRUSTED_CAD_SCAN_REGISTRATION"
    assert result["predictions"] == []
    assert result["gold_or_expected_counts_used"] is False


def test_trusted_registration_localizes_only_scan_verified_cad_events(tmp_path, monkeypatch):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    monkeypatch.setattr(localizer, "_scan_points", lambda _: np.array([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0],
    ]))
    monkeypatch.setattr(localizer, "line_scan_support", lambda *_: {
        "frame_integrity_pass": True,
        "trimmed_line_start": [0.0, 0.0, 0.0],
        "trimmed_line_end": [10.0, 0.0, 0.0],
    })
    monkeypatch.setattr(localizer, "_two_surface_normal_turn", lambda *_: {
        "passed": True,
        "normal_turn_deg": 88.0,
    })

    result = localizer.localize(_template(), scan, {
        "alignment_source": "PLANE_DATUM_REFINEMENT",
        "alignment_confidence": 0.8,
        "alignment_abstained": False,
        "raw_scan_to_cad_transform": np.eye(4).tolist(),
    })

    assert result["status"] == "COMPLETE"
    assert result["localized_bend_count"] == 1
    assert result["predictions"][0]["assignment_source"] == "CAD_TEMPLATE_SCAN_LOCALIZED"
    assert result["coordinate_frame"] == "RAW_SCAN"
    assert result["global_scan_proposals_allowed"] is False


def test_invalid_template_is_rejected_before_geometry_processing(tmp_path):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    try:
        localizer.localize({"schema": "WRONG"}, scan, {})
    except ValueError as exc:
        assert str(exc) == "invalid CAD bend template"
    else:
        raise AssertionError("invalid template must be rejected")
