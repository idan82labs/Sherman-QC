from __future__ import annotations

from backend.zero_shot_bend_overlay import (
    build_renderable_bend_objects,
    build_zero_shot_overlay_manifest,
)


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
