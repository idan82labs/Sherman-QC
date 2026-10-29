from __future__ import annotations

import numpy as np
import open3d as o3d

from backend.ai_analyzer import Model3DSnapshotRenderer
from backend.zero_shot_bend_overlay import (
    _apply_scan_supported_display_gate,
    build_renderable_bend_objects,
    build_zero_shot_overlay_manifest,
)


def test_bend_overlay_uses_cpu_fallback_without_creating_glfw_window(monkeypatch) -> None:
    monkeypatch.setenv("SHERMAN_QC_FORCE_CPU_RENDERER", "1")
    renderer = Model3DSnapshotRenderer()
    # A sentinel proves the forced branch does not touch visualization.Visualizer.
    renderer._o3d = object()
    image = renderer.render_bend_status_overlay(
        bends=[
            {
                "bend_id": "S1",
                "status": "WARNING",
                "cad_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [10.0, 0.0, 0.0]},
                "detected_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [10.0, 0.0, 0.0]},
                "anchor": [5.0, 0.0, 0.0],
            }
        ],
        reference_mesh_path=None,
    )

    assert image.startswith(b"\x89PNG\r\n\x1a\n")


def test_build_renderable_bend_objects_derives_line_and_metadata() -> None:
    strips = [
        {
            "strip_id": "S1",
            "strip_family": "discrete_fold_strip",
            "axis_direction": [1.0, 0.0, 0.0],
            "axis_line_hint": {"midpoint": [10.0, 20.0, 30.0]},
            "dominant_angle_deg": 91.2,
            "radius_mm": 1.5,
            "support_point_count": 42,
            "cross_fold_width_mm": 4.0,
            "along_axis_span_mm": 20.0,
            "observability_status": "supported",
            "stability_score": 0.8,
            "confidence": 0.9,
            "merge_provenance": {"source_modes": ["multiscale"]},
        }
    ]

    items = build_renderable_bend_objects(strips, profile_family="orthogonal_dense_sheetmetal", part_id="P")

    assert len(items) == 1
    item = items[0]
    assert item["bend_id"] == "S1"
    assert item["status"] == "WARNING"
    assert item["anchor"] == [10.0, 20.0, 30.0]
    assert item["cad_line"]["display_start"] == [0.0, 20.0, 30.0]
    assert item["cad_line"]["display_end"] == [20.0, 20.0, 30.0]
    assert "θ≈91.2°" in item["detail_segments"]
    assert item["metadata"]["strip_family"] == "discrete_fold_strip"


def test_build_zero_shot_overlay_manifest_uses_renderable_objects() -> None:
    renderable = [
        {
            "bend_id": "S1",
            "status": "WARNING",
            "cad_line": {"display_start": [0, 0, 0], "display_end": [1, 0, 0]},
            "detected_line": {"display_start": [0, 0, 0], "display_end": [1, 0, 0]},
            "anchor": [0.5, 0.0, 0.0],
        }
    ]
    manifest = build_zero_shot_overlay_manifest(
        result_payload={
            "profile_family": "conservative_macro",
            "estimated_bend_count": None,
            "estimated_bend_count_range": [1, 2],
            "abstained": True,
            "abstain_reasons": ["policy_margin_low"],
        },
        scan_path="/tmp/scan.ply",
        renderable_bend_objects=renderable,
    )

    assert manifest["source_kind"] == "zero_shot_scan"
    assert manifest["scan_path"] == "/tmp/scan.ply"
    assert manifest["bends"] == renderable
    assert manifest["abstained"] is True


def test_zero_shot_display_gate_suppresses_unsupported_strip_line(tmp_path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray([[x, 0.0, 0.0] for x in range(30)], dtype=np.float64)
    )
    scan_path = tmp_path / "scan.ply"
    o3d.io.write_point_cloud(str(scan_path), pcd)

    bends = [
        {
            "bend_id": "GOOD",
            "status": "WARNING",
            "cad_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [29.0, 0.0, 0.0]},
            "detected_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [29.0, 0.0, 0.0]},
            "anchor": [14.5, 0.0, 0.0],
        },
        {
            "bend_id": "FLOATING",
            "status": "WARNING",
            "cad_line": {"display_start": [100.0, 100.0, 0.0], "display_end": [130.0, 100.0, 0.0]},
            "detected_line": {"display_start": [100.0, 100.0, 0.0], "display_end": [130.0, 100.0, 0.0]},
            "anchor": [115.0, 100.0, 0.0],
        },
    ]

    gated = _apply_scan_supported_display_gate(bends, scan_path)

    assert gated[0]["anchor"] == [14.5, 0.0, 0.0]
    assert gated[0]["detected_line"]["display_start"] == [0.0, 0.0, 0.0]
    assert gated[1]["anchor"] is None
    assert gated[1]["detected_line"]["display_start"] is None
    assert gated[1]["display_suppressed_reason"] == "no_scan_supported_overlay_line"


def test_zero_shot_display_gate_clips_extrapolated_endpoints_to_scan_support(tmp_path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray([[x, 0.0, 0.0] for x in np.linspace(10.0, 20.0, 101)], dtype=np.float64)
    )
    scan_path = tmp_path / "scan.ply"
    o3d.io.write_point_cloud(str(scan_path), pcd)
    bends = [
        {
            "bend_id": "EXTRAPOLATED",
            "status": "WARNING",
            "cad_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [30.0, 0.0, 0.0]},
            "detected_line": {"display_start": [0.0, 0.0, 0.0], "display_end": [30.0, 0.0, 0.0]},
            "anchor": [15.0, 0.0, 0.0],
        }
    ]

    gated = _apply_scan_supported_display_gate(bends, scan_path)

    assert gated[0]["detected_line"]["display_clipped_to_scan_support"] is True
    assert 9.9 <= gated[0]["detected_line"]["display_start"][0] <= 10.1
    assert 19.9 <= gated[0]["detected_line"]["display_end"][0] <= 20.1
