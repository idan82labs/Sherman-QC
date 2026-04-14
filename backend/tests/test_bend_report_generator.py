"""Tests for bend inspection artifact and PDF generation."""

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from bend_report_generator import (
    augment_report_with_overlay_geometry,
    generate_bend_inspection_pdf,
)


def _write_reference_mesh(path: Path) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [40.0, 20.0, 0.0],
                [0.0, 20.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    )
    o3d.io.write_triangle_mesh(str(path), mesh)


def _sample_report() -> dict:
    return {
        "part_id": "PART-001",
        "summary": {
            "completed_bends": 2,
            "expected_bends": 3,
            "completed_in_spec": 1,
            "completed_out_of_spec": 1,
            "remaining_bends": 1,
            "scan_quality_status": "FAIR",
            "overall_result": "WARNING",
            "release_decision": "HOLD",
            "release_blocked_by": [],
            "release_hold_reasons": ["position", "correspondence"],
            "trusted_alignment_for_release": True,
            "trusted_position_evidence": True,
        },
        "release_decision": "HOLD",
        "scan_quality": {
            "coverage_pct": 64.0,
            "density_pts_per_cm2": 18.4,
            "median_spacing_mm": 1.4,
        },
        "operator_brief": {
            "headline": "2/3 completed, 1 in spec, 1 out of spec, 1 remaining",
        },
        "matches": [
            {
                "bend_id": "CAD_B1",
                "status": "FAIL",
                "cad_line_start": [2.0, 2.0, 0.0],
                "cad_line_end": [38.0, 2.0, 0.0],
                "detected_line_start": [4.0, 2.5, 0.0],
                "detected_line_end": [35.0, 2.5, 0.0],
                "target_angle": 90.0,
                "measured_angle": 86.5,
                "angle_deviation": -3.5,
                "target_radius": 2.0,
                "measured_radius": 2.8,
                "radius_deviation": 0.8,
                "line_center_deviation_mm": 0.42,
                "tolerance_angle": 1.0,
                "tolerance_radius": 0.5,
                "action_item": "Re-hit CAD_B1 to nominal angle.",
                "issue_location": "left flange",
                "completion_state": "FORMED",
                "metrology_state": "OUT_OF_TOL",
                "positional_state": "MISLOCATED",
                "position_evidence": {"status": "CONTRADICTED", "reason": "heatmap_contradiction"},
            },
            {
                "bend_id": "CAD_B2",
                "status": "PASS",
                "cad_line_start": [2.0, 18.0, 0.0],
                "cad_line_end": [38.0, 18.0, 0.0],
                "detected_line_start": [2.5, 18.0, 0.0],
                "detected_line_end": [37.4, 18.0, 0.0],
                "target_angle": 90.0,
                "measured_angle": 90.2,
                "angle_deviation": 0.2,
                "target_radius": 2.0,
                "measured_radius": 2.1,
                "radius_deviation": 0.1,
                "line_center_deviation_mm": 0.08,
                "tolerance_angle": 1.0,
                "tolerance_radius": 0.5,
                "action_item": "",
                "issue_location": "none",
                "completion_state": "FORMED",
                "metrology_state": "IN_TOL",
                "positional_state": "ON_POSITION",
                "position_evidence": {"status": "SUPPORTED", "reason": "centerline_offset_within_tolerance"},
            },
            {
                "bend_id": "CAD_B3",
                "status": "NOT_DETECTED",
                "cad_line_start": [20.0, 2.0, 0.0],
                "cad_line_end": [20.0, 18.0, 0.0],
                "target_angle": 90.0,
                "measured_angle": None,
                "angle_deviation": None,
                "target_radius": 2.0,
                "measured_radius": None,
                "radius_deviation": None,
                "line_center_deviation_mm": None,
                "tolerance_angle": 1.0,
                "tolerance_radius": 0.5,
                "action_item": "Form CAD_B3 before final inspection.",
                "issue_location": "center wall",
                "completion_state": "UNKNOWN",
                "metrology_state": "UNMEASURABLE",
                "positional_state": "UNKNOWN_POSITION",
                "position_evidence": {"status": "INCONCLUSIVE", "reason": "no_position_signal"},
            },
        ],
    }


def test_augment_report_with_overlay_geometry_adds_display_lines_and_manifest(tmp_path):
    ref_mesh = tmp_path / "reference_mesh.ply"
    _write_reference_mesh(ref_mesh)
    report = _sample_report()

    augmented, manifest = augment_report_with_overlay_geometry(report, ref_mesh)

    first = augmented["matches"][0]
    assert first["display_cad_line_start"] is not None
    assert first["display_detected_line_start"] is not None
    assert first["callout_anchor"] is not None
    assert manifest["overview_image"] == "bend_overlay_overview.png"
    assert len(manifest["issue_tiles"]) == 1
    assert manifest["issue_tiles"][0]["bend_id"] == "CAD_B1"
    assert any(b["bend_id"] == "CAD_B2" for b in manifest["bends"])


def test_generate_bend_inspection_pdf_writes_output(tmp_path):
    report = _sample_report()
    overview = tmp_path / "bend_overlay_overview.png"
    issue = tmp_path / "bend_overlay_issue_CAD_B1.png"
    Image.new("RGB", (800, 500), color=(240, 244, 248)).save(overview)
    Image.new("RGB", (600, 420), color=(248, 245, 240)).save(issue)
    manifest = {
        "overview_image": overview.name,
        "issue_tiles": [{"bend_id": "CAD_B1", "status": "FAIL", "image": issue.name}],
        "bends": [],
    }

    pdf_path = tmp_path / "bend_report.pdf"
    generate_bend_inspection_pdf(
        report_dict=report,
        manifest=manifest,
        output_path=pdf_path,
        output_dir=tmp_path,
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
