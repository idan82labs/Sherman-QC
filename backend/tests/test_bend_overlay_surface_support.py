"""Synthetic contracts for reference-frame overlay geometry; no physical inputs."""
from copy import deepcopy

import numpy as np
import open3d as o3d
import pytest

from bend_report_generator import augment_report_with_overlay_geometry


def _reference(path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [40, 0, 0], [40, 20, 0], [0, 20, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def _report():
    return {
        "alignment": {
            "status": "AVAILABLE",
            "geometry_frame": "reference_mesh_raw",
            "raw_scan_to_reference_transform": np.eye(4).tolist(),
        },
        "matches": [{
            "bend_id": "B1", "status": "WARNING",
            "cad_line_start": [3, 10, 0], "cad_line_end": [37, 10, 0],
            "detected_line_start": [3, 10, 1], "detected_line_end": [37, 10, 1],
        }],
    }


def test_sparse_triangle_interiors_support_markers_without_moving_axes(tmp_path):
    original = _report()
    report, manifest = augment_report_with_overlay_geometry(deepcopy(original), _reference(tmp_path / "ref.ply"))
    row = report["matches"][0]
    assert row["display_geometry_valid"] is True
    for kind in ("cad", "detected"):
        for endpoint in ("start", "end"):
            assert row[f"display_{kind}_line_{endpoint}"] == original["matches"][0][f"{kind}_line_{endpoint}"]
    assert row["display_geometry_validation"]["lines"]["detected"]["support_fraction"] == 1
    assert manifest["bends"][0]["display_geometry_validation"]["semantics"] == "reference_supported_axis_not_measured_surface_location"


def test_floating_lines_are_not_certified_even_with_explicit_frame(tmp_path):
    report = _report()
    for kind in ("cad", "detected"):
        report["matches"][0][f"{kind}_line_start"][2] = 35
        report["matches"][0][f"{kind}_line_end"][2] = 35
    row = augment_report_with_overlay_geometry(report, _reference(tmp_path / "ref.ply"))[0]["matches"][0]
    assert row["display_geometry_valid"] is False
    assert "display_cad_line_start" not in row
    assert "display_detected_line_start" not in row
    assert "callout_anchor" not in row
    assert row["cad_line_start"][2] == 35


@pytest.mark.parametrize("alignment", [None, {}, {"status": "AVAILABLE", "geometry_frame": "raw_scan"}, {
    "status": "AVAILABLE", "geometry_frame": "reference_mesh_raw", "raw_scan_to_reference_transform": np.zeros((4, 4)).tolist()
}, {"status": "AVAILABLE", "geometry_frame": "reference_mesh_raw", "raw_scan_to_reference_transform": [[float("nan")]]}])
def test_unproven_frames_clear_stale_display_geometry_without_loading_mesh(alignment, monkeypatch):
    import bend_report_generator as generator
    monkeypatch.setattr(generator, "_load_reference_surface", lambda _: pytest.fail("unproven frame must not load geometry"))
    report = _report()
    report["alignment"] = alignment
    report["matches"][0].update({"display_geometry_valid": True, "display_cad_line_start": [1, 2, 3], "callout_anchor": [1, 2, 3]})
    row = generator.augment_report_with_overlay_geometry(report, "unused.ply")[0]["matches"][0]
    assert row["display_geometry_valid"] is False
    assert row["display_geometry_frame"] is None
    assert "display_cad_line_start" not in row
    assert "callout_anchor" not in row


def test_supported_extent_is_trimmed_on_original_segment_not_recentered(tmp_path):
    report = _report()
    row = report["matches"][0]
    row["cad_line_start"] = [-12, 10, 0]
    row["cad_line_end"] = [38, 10, 0]
    result = augment_report_with_overlay_geometry(report, _reference(tmp_path / "ref.ply"))[0]["matches"][0]
    assert -3 < result["display_cad_line_start"][0] < 0
    assert result["display_cad_line_end"] == [38, 10, 0]
    assert result["cad_line_start"] == [-12, 10, 0]
    assert result["display_cad_line_start"][1:] == [10, 0]


def test_long_line_with_only_small_supported_piece_is_omitted(tmp_path):
    report = _report()
    report["matches"][0]["cad_line_start"] = [-100, 10, 0]
    report["matches"][0]["cad_line_end"] = [100, 10, 0]
    row = augment_report_with_overlay_geometry(report, _reference(tmp_path / "ref.ply"))[0]["matches"][0]
    assert "display_cad_line_start" not in row
    assert row["display_geometry_validation"]["rejections"]["cad"] == "INSUFFICIENT_CONTIGUOUS_REFERENCE_SUPPORT"
    # A separately validated detected marker can still be shown.
    assert row["display_detected_line_start"] == [3, 10, 1]
    assert row["display_geometry_valid"] is True


def test_cpu_reference_surface_occludes_axis_behind_it(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from matplotlib.axes import Axes
    from ai_analyzer import Model3DSnapshotRenderer

    plotted = []
    original_plot = Axes.plot

    def record_plot(self, *args, **kwargs):
        plotted.append((np.asarray(args[0]), kwargs.get("color")))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    png = Model3DSnapshotRenderer()._render_bend_status_overlay_matplotlib(
        bends=[{"bend_id": "B1", "status": "WARNING",
                "cad_line": {"display_start": [3, 10, 0], "display_end": [37, 10, 0]},
                "detected_line": {"display_start": [3, 10, -1], "display_end": [37, 10, -1]}}],
        reference_mesh_path=str(_reference(tmp_path / "ref.ply")), view="top",
        width=640, height=480, focused_bend_ids=None, show_detected_lines=True,
        show_cad_lines=True, compact_bend_labels=False, title_override=None,
    )
    assert Image.open(BytesIO(png)).size == (640, 480)
    assert any(np.isfinite(xs).all() for xs, color in plotted if color == "#0e7490")
    assert any(np.isnan(xs).all() for xs, color in plotted if color == "#d97706")


@pytest.mark.parametrize("point", [[[1, 2, 3]] * 3, [float("nan"), 1, 2], [1, 2]])
def test_malformed_endpoints_cannot_become_display_lines(tmp_path, point):
    report = _report()
    report["matches"][0]["cad_line_start"] = point
    row = augment_report_with_overlay_geometry(report, _reference(tmp_path / "ref.ply"))[0]["matches"][0]
    assert "display_cad_line_start" not in row
    assert row["display_geometry_validation"]["rejections"]["cad"] == "MISSING_LINE"


def test_cad_placeholder_cannot_masquerade_as_detected_scan_line(tmp_path):
    report = _report()
    row = report["matches"][0]
    row["detected_line_geometry_source"] = "CAD_REFERENCE_PLACEHOLDER"
    row["detected_line_start"] = row["cad_line_start"].copy()
    row["detected_line_end"] = row["cad_line_end"].copy()
    row["display_detected_line_start"] = row["cad_line_start"].copy()
    result, manifest = augment_report_with_overlay_geometry(report, _reference(tmp_path / "ref.ply"))
    row = result["matches"][0]
    assert row["display_geometry_valid"] is True
    assert row["display_cad_line_start"] == [3, 10, 0]
    assert "display_detected_line_start" not in row
    assert "display_detected_line_end" not in row
    assert row["display_geometry_validation"]["rejections"]["detected"] == "CAD_REFERENCE_PLACEHOLDER_NOT_SCAN_LINE"
    assert manifest["bends"][0]["detected_line"]["geometry_source"] == "CAD_REFERENCE_PLACEHOLDER"
    assert row["detected_line_start"] == [3, 10, 0]


@pytest.mark.parametrize("change", [
    {"alignment_abstained": True},
    {"raw_scan_to_reference_transform": np.diag([-1, 1, 1, 1]).tolist()},
])
def test_abstained_or_reflected_frame_cannot_certify_markers(tmp_path, change):
    report = _report()
    report["alignment"].update(change)
    row = augment_report_with_overlay_geometry(report, tmp_path / "unused.ply")[0]["matches"][0]
    assert row["display_geometry_valid"] is False
    assert "display_cad_line_start" not in row
    assert row["display_geometry_validation"]["rejections"]["cad"] == "REFERENCE_FRAME_UNPROVEN"
