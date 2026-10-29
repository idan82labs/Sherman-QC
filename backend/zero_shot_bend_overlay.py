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


def _compute_model_max_dim(points: np.ndarray) -> float:
    if points.size < 3:
        return 4.0
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return float(max(1.0, *(maxs - mins)))


def _line_supported_by_reference(
    start: Sequence[float],
    end: Sequence[float],
    reference_positions: np.ndarray,
) -> Tuple[bool, Dict[str, Any]]:
    if len(reference_positions) < 12:
        return True, {"reason": "insufficient_reference_positions"}

    start_np = np.asarray(start[:3], dtype=np.float64)
    end_np = np.asarray(end[:3], dtype=np.float64)
    direction = end_np - start_np
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return False, {"reason": "degenerate_line", "line_length_mm": round(length, 4)}

    model_max_dim = _compute_model_max_dim(reference_positions)
    threshold = max(6.0, model_max_dim * 0.035)
    dir_vec = direction / length
    offsets = reference_positions - start_np
    axial = np.einsum("ij,j->i", offsets, dir_vec)
    clamped = np.clip(axial, 0.0, length)
    closest = start_np + np.outer(clamped, dir_vec)
    distances = np.linalg.norm(reference_positions - closest, axis=1)

    min_distance = float(np.min(distances)) if len(distances) else float("inf")
    near_mask = distances <= threshold
    near_count = int(np.sum(near_mask))
    supported_span = float(np.max(clamped[near_mask]) - np.min(clamped[near_mask])) if near_count else 0.0
    supported = bool(near_count >= 8 or (near_count >= 3 and supported_span >= min(12.0, length * 0.35)))
    close_threshold = min(threshold, max(1.5, model_max_dim * 0.009))
    close_mask = distances <= close_threshold
    close_axial = axial[close_mask & (axial >= 0.0) & (axial <= length)]
    supported_interval = None
    if len(close_axial) >= 12:
        low, high = np.quantile(close_axial, [0.001, 0.999])
        if float(high - low) >= min(3.0, length * 0.25):
            supported_interval = [round(float(low), 4), round(float(high), 4)]
    return supported, {
        "reason": "supported_by_reference" if supported else "line_not_supported_by_scan_geometry",
        "line_length_mm": round(length, 4),
        "support_threshold_mm": round(threshold, 4),
        "nearest_reference_distance_mm": round(min_distance, 4),
        "near_reference_point_count": near_count,
        "supported_span_mm": round(supported_span, 4),
        "close_support_threshold_mm": round(close_threshold, 4),
        "supported_axial_interval_mm": supported_interval,
    }


def _load_reference_positions(scan_geometry_path: str | Path) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)

    mesh_path = Path(scan_geometry_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.has_vertices():
        return np.asarray(mesh.vertices, dtype=np.float64)
    pcd = o3d.io.read_point_cloud(str(mesh_path))
    if pcd.has_points():
        return np.asarray(pcd.points, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def _apply_scan_supported_display_gate(
    bends: Sequence[Mapping[str, Any]],
    scan_geometry_path: str | Path,
) -> List[Dict[str, Any]]:
    reference_positions = _load_reference_positions(scan_geometry_path)
    gated: List[Dict[str, Any]] = []
    for bend in bends:
        item = dict(bend)
        suppressed = False
        for key in ("cad_line", "detected_line"):
            line = dict(item.get(key) or {})
            start = _point3(line.get("display_start") or line.get("raw_start"))
            end = _point3(line.get("display_end") or line.get("raw_end"))
            if start is None or end is None:
                line["display_start"] = None
                line["display_end"] = None
                item[key] = line
                continue
            supported, support = _line_supported_by_reference(start, end, reference_positions)
            item[f"{key}_display_support"] = support
            if supported:
                interval = support.get("supported_axial_interval_mm")
                if isinstance(interval, list) and len(interval) == 2:
                    start_np = np.asarray(start, dtype=np.float64)
                    end_np = np.asarray(end, dtype=np.float64)
                    direction = end_np - start_np
                    line_length = max(float(np.linalg.norm(direction)), 1e-12)
                    direction /= line_length
                    clip_epsilon = max(0.5, line_length * 0.01)
                    if float(interval[0]) > clip_epsilon or line_length - float(interval[1]) > clip_epsilon:
                        line["display_start"] = [float(v) for v in (start_np + direction * float(interval[0])).tolist()]
                        line["display_end"] = [float(v) for v in (start_np + direction * float(interval[1])).tolist()]
                        line["display_clipped_to_scan_support"] = True
                    else:
                        line["display_start"] = start
                        line["display_end"] = end
                else:
                    line["display_start"] = start
                    line["display_end"] = end
            else:
                suppressed = True
                line["display_start"] = None
                line["display_end"] = None
            item[key] = line

        detected = item.get("detected_line") or {}
        cad = item.get("cad_line") or {}
        anchor = _point3(item.get("anchor"))
        if detected.get("display_start") and detected.get("display_end"):
            start_np = np.asarray(detected["display_start"], dtype=np.float64)
            end_np = np.asarray(detected["display_end"], dtype=np.float64)
            anchor = [float(v) for v in ((start_np + end_np) / 2.0).tolist()]
        elif cad.get("display_start") and cad.get("display_end"):
            start_np = np.asarray(cad["display_start"], dtype=np.float64)
            end_np = np.asarray(cad["display_end"], dtype=np.float64)
            anchor = [float(v) for v in ((start_np + end_np) / 2.0).tolist()]
        else:
            anchor = None
        item["anchor"] = anchor
        if suppressed and anchor is None:
            item["display_suppressed_reason"] = "no_scan_supported_overlay_line"
        gated.append(item)
    return gated


def _line_endpoints_for_strip(strip: Mapping[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    supported_endpoints = (strip.get("axis_line_hint") or {}).get("supported_endpoints")
    if isinstance(supported_endpoints, (list, tuple)) and len(supported_endpoints) >= 2:
        supported_start = _point3(supported_endpoints[0])
        supported_end = _point3(supported_endpoints[1])
        if supported_start is not None and supported_end is not None:
            center = (np.asarray(supported_start) + np.asarray(supported_end)) / 2.0
            return supported_start, supported_end, [float(v) for v in center.tolist()]
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
    manifest = dict(manifest)
    manifest["bends"] = _apply_scan_supported_display_gate(
        list(manifest.get("bends") or []),
        scan_geometry_path,
    )
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
