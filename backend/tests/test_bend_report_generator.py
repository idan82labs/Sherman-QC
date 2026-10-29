"""Tests for bend inspection artifact and PDF generation."""

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from pypdf import PdfReader

from bend_report_generator import (
    RESEARCH_WATERMARK,
    _stamp_research_only,
    augment_report_with_overlay_geometry,
    generate_bend_inspection_pdf,
    render_bend_overlay_artifacts,
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
        "alignment": {
            "status": "AVAILABLE",
            "geometry_frame": "reference_mesh_raw",
            "raw_scan_to_reference_transform": np.eye(4).tolist(),
        },
        "summary": {
            "completed_bends": 2,
            "expected_bends": 3,
            "completed_in_spec": 1,
            "completed_out_of_spec": 1,
            "remaining_bends": 1,
            "scan_quality_status": "FAIR",
        },
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
    assert first["display_geometry_frame"] == "reference_mesh_raw"
    assert first["display_geometry_provenance"] == "reference_triangle_surface_validation_v2"
    assert first["display_geometry_valid"] is True
    assert manifest["overview_image"] == "bend_overlay_overview.png"
    assert manifest["qualification_context"]["production_use_allowed"] is False
    assert len(manifest["issue_tiles"]) == 1
    assert manifest["issue_tiles"][0]["bend_id"] == "CAD_B1"
    assert any(b["bend_id"] == "CAD_B2" for b in manifest["bends"])


def test_augment_report_drops_implausibly_far_detected_line(tmp_path):
    ref_mesh = tmp_path / "reference_mesh.ply"
    _write_reference_mesh(ref_mesh)
    report = _sample_report()
    report["matches"][0]["detected_line_start"] = [200.0, 200.0, 50.0]
    report["matches"][0]["detected_line_end"] = [230.0, 205.0, 52.0]

    augmented, _ = augment_report_with_overlay_geometry(report, ref_mesh)

    first = augmented["matches"][0]
    assert first.get("display_detected_line_start") is None
    assert first.get("display_detected_line_end") is None
    assert first["callout_anchor"] == first["display_cad_line_start"] or first["callout_anchor"] == first["display_cad_line_end"] or first["callout_anchor"] is not None
    cad_mid = [
        round((first["display_cad_line_start"][i] + first["display_cad_line_end"][i]) / 2.0, 4)
        for i in range(3)
    ]
    assert first["callout_anchor"] == cad_mid


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
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    assert "RESEARCH ONLY" in text
    assert "NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT" in text


def test_research_watermark_is_baked_into_detached_png():
    import io

    original = io.BytesIO()
    Image.new("RGB", (800, 500), color=(240, 244, 248)).save(original, format="PNG")
    stamped = _stamp_research_only(original.getvalue())
    with Image.open(io.BytesIO(stamped)) as image:
        assert image.size == (800, 500)
        assert image.getpixel((10, 10)) == (136, 19, 55)
    assert RESEARCH_WATERMARK == "RESEARCH ONLY - NOT PRODUCTION QC"


def test_all_rendered_visual_artifacts_are_watermarked_fail_closed(tmp_path, monkeypatch):
    import io

    class FakeRenderer:
        def render_bend_status_overlay(self, **_kwargs):
            buf = io.BytesIO()
            Image.new("RGB", (800, 500), color=(240, 244, 248)).save(buf, format="PNG")
            return buf.getvalue()

    monkeypatch.setattr("bend_report_generator.Model3DSnapshotRenderer", FakeRenderer)
    manifest = {
        "qualification_context": {"production_use_allowed": False},
        "bends": [],
        "issue_tiles": [{"bend_id": "B1", "image": "bend_overlay_issue_B1.png"}],
    }
    result = render_bend_overlay_artifacts(manifest, tmp_path / "unused.ply", tmp_path)
    paths = [Path(result["overview_path"]), *(Path(p) for p in result["issue_paths"]), *(Path(p) for p in result["bender_view_paths"].values())]
    assert len(paths) == 6
    for path in paths:
        with Image.open(path) as image:
            assert image.getpixel((10, 10)) == (136, 19, 55)


def test_qualification_boolean_cannot_remove_research_watermark(tmp_path, monkeypatch):
    import io

    class FakeRenderer:
        def render_bend_status_overlay(self, **_kwargs):
            buf = io.BytesIO()
            Image.new("RGB", (800, 500), color=(240, 244, 248)).save(buf, format="PNG")
            return buf.getvalue()

    monkeypatch.setattr("bend_report_generator.Model3DSnapshotRenderer", FakeRenderer)
    manifest = {
        "qualification_context": {"production_use_allowed": True},
        "bends": [],
        "issue_tiles": [],
    }
    result = render_bend_overlay_artifacts(manifest, tmp_path / "unused.ply", tmp_path)
    with Image.open(result["overview_path"]) as image:
        assert image.getpixel((10, 10)) == (136, 19, 55)
