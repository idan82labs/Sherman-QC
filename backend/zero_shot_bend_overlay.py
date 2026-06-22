from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _point3(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _normalize_direction(direction: Any) -> np.ndarray:
    vec = np.asarray(direction if direction is not None else [1.0, 0.0, 0.0], dtype=np.float64)
    if vec.shape != (3,):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _line_endpoints_for_strip(strip: Mapping[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    midpoint = _point3((strip.get("axis_line_hint") or {}).get("midpoint")) or [0.0, 0.0, 0.0]
    axis = _normalize_direction(strip.get("axis_direction"))
    span = max(
        _safe_float(strip.get("along_axis_span_mm"), 0.0),
        _safe_float(strip.get("cross_fold_width_mm"), 0.0) * 1.5,
        1.0,
    )
    half = 0.5 * span
    center = np.asarray(midpoint, dtype=np.float64)
    start = center - axis * half
    end = center + axis * half
    return (
        [float(v) for v in start.tolist()],
        [float(v) for v in end.tolist()],
        [float(v) for v in center.tolist()],
    )


def _detail_segments(strip: Mapping[str, Any]) -> List[str]:
    segments: List[str] = []
    strip_family = str(strip.get("strip_family") or "unknown")
    observability = str(strip.get("observability_status") or "unknown")
    angle = _safe_float(strip.get("dominant_angle_deg"), 0.0)
    span = _safe_float(strip.get("along_axis_span_mm"), 0.0)
    confidence = _safe_float(strip.get("confidence"), 0.0)
    segments.append(strip_family.replace("_", " "))
    if angle > 0.0:
        segments.append(f"θ≈{angle:.1f}°")
    if span > 0.0:
        segments.append(f"span {span:.1f}mm")
    segments.append(f"{observability}")
    segments.append(f"conf {confidence:.2f}")
    return segments


def build_renderable_bend_objects(
    bend_support_strips: Sequence[Mapping[str, Any]],
    *,
    profile_family: Optional[str] = None,
    part_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    renderable: List[Dict[str, Any]] = []
    for index, strip in enumerate(bend_support_strips, start=1):
        start, end, anchor = _line_endpoints_for_strip(strip)
        strip_id = str(strip.get("strip_id") or f"S{index}")
        observability = str(strip.get("observability_status") or "")
        status = "WARNING" if observability == "supported" else "NOT_DETECTED"
        dominant_angle = _safe_float(strip.get("dominant_angle_deg"), 0.0)
        renderable.append(
            {
                "bend_id": strip_id,
                "label": strip_id,
                "status": status,
                "source_kind": "zero_shot_support_strip",
                "part_id": part_id,
                "profile_family": profile_family,
                "cad_line": {
                    "raw_start": start,
                    "raw_end": end,
                    "display_start": start,
                    "display_end": end,
                },
                "detected_line": {
                    "raw_start": start,
                    "raw_end": end,
                    "display_start": start,
                    "display_end": end,
                },
                "anchor": anchor,
                "detail_segments": _detail_segments(strip),
                "metadata": {
                    "strip_family": strip.get("strip_family"),
                    "observability_status": strip.get("observability_status"),
                    "dominant_angle_deg": dominant_angle if dominant_angle > 0.0 else None,
                    "radius_mm": strip.get("radius_mm"),
                    "support_point_count": strip.get("support_point_count"),
                    "cross_fold_width_mm": strip.get("cross_fold_width_mm"),
                    "along_axis_span_mm": strip.get("along_axis_span_mm"),
                    "stability_score": strip.get("stability_score"),
                    "confidence": strip.get("confidence"),
                    "merge_provenance": strip.get("merge_provenance"),
                },
            }
        )
    return renderable


def build_zero_shot_overlay_manifest(
    *,
    result_payload: Mapping[str, Any],
    scan_path: str,
    renderable_bend_objects: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    bends = list(renderable_bend_objects or [])
    if not bends:
        bends = build_renderable_bend_objects(
            list(result_payload.get("bend_support_strips") or []),
            profile_family=str(result_payload.get("profile_family") or ""),
            part_id=result_payload.get("part_id"),
        )
    return {
        "generated_at": datetime.now().isoformat(),
        "source_kind": "zero_shot_scan",
        "scan_path": str(scan_path),
        "profile_family": result_payload.get("profile_family"),
        "estimated_bend_count": result_payload.get("estimated_bend_count"),
        "estimated_bend_count_range": result_payload.get("estimated_bend_count_range"),
        "abstained": bool(result_payload.get("abstained")),
        "abstain_reasons": list(result_payload.get("abstain_reasons") or []),
        "overview_image": "zero_shot_bend_overlay_overview.png",
        "bends": bends,
    }


def render_zero_shot_overlay_artifacts(
    *,
    renderer: Any,
    manifest: Mapping[str, Any],
    scan_geometry_path: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_path = out_dir / "zero_shot_bend_overlay_overview.png"
    manifest_path = out_dir / "zero_shot_bend_overlay_manifest.json"
    manifest_path.write_text(_json_dump(manifest), encoding="utf-8")

    overview_bytes = renderer.render_bend_status_overlay(
        bends=list(manifest.get("bends") or []),
        reference_mesh_path=str(scan_geometry_path),
        view="iso",
        width=1600,
        height=1200,
        focused_bend_ids=None,
        show_issue_callouts=False,
    )
    overview_path.write_bytes(overview_bytes)

    focus_paths: List[str] = []
    for bend in list(manifest.get("bends") or []):
        bend_id = str(bend.get("bend_id") or "")
        if not bend_id:
            continue
        filename = f"zero_shot_bend_{bend_id}.png"
        focus_path = out_dir / filename
        focus_bytes = renderer.render_bend_status_overlay(
            bends=list(manifest.get("bends") or []),
            reference_mesh_path=str(scan_geometry_path),
            view="iso",
            width=1200,
            height=900,
            focused_bend_ids=[bend_id],
            show_issue_callouts=False,
        )
        focus_path.write_bytes(focus_bytes)
        focus_paths.append(str(focus_path))

    return {
        "manifest_path": str(manifest_path),
        "overview_path": str(overview_path),
        "focus_paths": focus_paths,
    }


def _json_dump(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
