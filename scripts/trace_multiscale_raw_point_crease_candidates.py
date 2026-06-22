#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def _load_tracer_module():
    path = Path(__file__).resolve().parent / "trace_raw_point_crease_candidates.py"
    spec = importlib.util.spec_from_file_location("trace_raw_point_crease_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tracer = _load_tracer_module()


DEFAULT_SCALES: List[Dict[str, Any]] = [
    {
        "scale_id": "fine",
        "k_neighbors": 16,
        "score_quantile": 0.978,
        "min_contrast_deg": 22.0,
        "min_curvature": 0.014,
        "min_points": 22,
        "min_span_mm": 12.0,
        "max_thickness_mm": 10.0,
    },
    {
        "scale_id": "standard",
        "k_neighbors": 24,
        "score_quantile": 0.972,
        "min_contrast_deg": 20.0,
        "min_curvature": 0.016,
        "min_points": 35,
        "min_span_mm": 18.0,
        "max_thickness_mm": 16.0,
    },
    {
        "scale_id": "coarse",
        "k_neighbors": 36,
        "score_quantile": 0.965,
        "min_contrast_deg": 18.0,
        "min_curvature": 0.012,
        "min_points": 45,
        "min_span_mm": 24.0,
        "max_thickness_mm": 22.0,
    },
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_points(candidate: Mapping[str, Any]) -> np.ndarray:
    points = candidate.get("sampled_points") or ()
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)


def _unit(values: Sequence[Any]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    dot = abs(float(np.dot(_unit(lhs), _unit(rhs))))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, dot)))))


def _line_to_line_distance(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> float:
    left_points = _candidate_points(lhs)
    right_points = _candidate_points(rhs)
    if len(left_points) == 0 or len(right_points) == 0:
        return 999.0
    left_center = np.mean(left_points, axis=0)
    right_center = np.mean(right_points, axis=0)
    left_axis = _unit(lhs.get("axis_direction") or (1.0, 0.0, 0.0))
    right_axis = _unit(rhs.get("axis_direction") or (1.0, 0.0, 0.0))
    cross = np.cross(left_axis, right_axis)
    norm = float(np.linalg.norm(cross))
    delta = right_center - left_center
    if norm <= 1e-9:
        return float(np.linalg.norm(delta - float(np.dot(delta, left_axis)) * left_axis))
    return abs(float(np.dot(delta, cross / norm)))


def _projection_interval(candidate: Mapping[str, Any], axis: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    points = _candidate_points(candidate)
    if len(points) == 0:
        return (0.0, 0.0)
    projections = (points - center) @ axis
    return (float(np.min(projections)), float(np.max(projections)))


def _interval_overlap(lhs: tuple[float, float], rhs: tuple[float, float]) -> float:
    lhs_span = max(0.0, lhs[1] - lhs[0])
    rhs_span = max(0.0, rhs[1] - rhs[0])
    if lhs_span <= 1e-9 or rhs_span <= 1e-9:
        return 0.0
    overlap = max(0.0, min(lhs[1], rhs[1]) - max(lhs[0], rhs[0]))
    return overlap / max(1e-9, min(lhs_span, rhs_span))


def _is_duplicate(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    axis_delta = _angle_deg(lhs.get("axis_direction") or (1.0, 0.0, 0.0), rhs.get("axis_direction") or (1.0, 0.0, 0.0))
    if axis_delta > 14.0:
        return False
    line_distance = _line_to_line_distance(lhs, rhs)
    if line_distance > 8.0:
        return False
    left_points = _candidate_points(lhs)
    if len(left_points) == 0:
        return False
    center = np.mean(left_points, axis=0)
    axis = _unit(lhs.get("axis_direction") or (1.0, 0.0, 0.0))
    return _interval_overlap(_projection_interval(lhs, axis, center), _projection_interval(rhs, axis, center)) >= 0.35


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(candidate.get("linearity") or 0.0),
        float(candidate.get("span_mm") or 0.0),
        -float(candidate.get("thickness_mm") or 999.0),
    )


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: _candidate_rank(row), reverse=True):
        duplicate_index = next((index for index, kept_candidate in enumerate(kept) if _is_duplicate(candidate, kept_candidate)), None)
        row = dict(candidate)
        if duplicate_index is None:
            row["duplicate_source_candidate_ids"] = []
            kept.append(row)
            continue
        kept_candidate = kept[duplicate_index]
        kept_candidate.setdefault("duplicate_source_candidate_ids", []).append(row.get("candidate_id"))
        kept_candidate.setdefault("scale_ids_merged", sorted(set([kept_candidate.get("scale_id"), row.get("scale_id")])))
        if _candidate_rank(row) > _candidate_rank(kept_candidate):
            row["duplicate_source_candidate_ids"] = [
                kept_candidate.get("candidate_id"),
                *kept_candidate.get("duplicate_source_candidate_ids", []),
            ]
            row["scale_ids_merged"] = sorted(
                {
                    *(value for value in kept_candidate.get("scale_ids_merged", []) if value),
                    row.get("scale_id"),
                    kept_candidate.get("scale_id"),
                }
            )
            kept[duplicate_index] = row
    kept.sort(key=lambda row: (not bool(row.get("admissible")), -float(row.get("span_mm") or 0.0), float(row.get("thickness_mm") or 999.0), str(row.get("candidate_id"))))
    return kept


def trace_multiscale_raw_point_crease_candidates(
    *,
    ply_path: Path,
    part_id: str,
    max_points: int = 50000,
    scales: Sequence[Mapping[str, Any]] = DEFAULT_SCALES,
) -> Dict[str, Any]:
    scale_reports: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    for scale in scales:
        scale_id = str(scale["scale_id"])
        report = tracer.trace_raw_point_crease_candidates(
            ply_path=ply_path,
            part_id=part_id,
            max_points=max_points,
            k_neighbors=int(scale["k_neighbors"]),
            score_quantile=float(scale["score_quantile"]),
            min_contrast_deg=float(scale["min_contrast_deg"]),
            min_curvature=float(scale["min_curvature"]),
            min_points=int(scale["min_points"]),
            min_span_mm=float(scale["min_span_mm"]),
            max_thickness_mm=float(scale["max_thickness_mm"]),
        )
        scale_reports.append(
            {
                "scale_id": scale_id,
                "parameters": dict(scale),
                "status": report.get("status"),
                "candidate_point_count": report.get("candidate_point_count"),
                "candidate_count": report.get("candidate_count"),
                "admissible_candidate_count": report.get("admissible_candidate_count"),
                "score_threshold": report.get("score_threshold"),
            }
        )
        for candidate in report.get("candidates") or ():
            row = dict(candidate)
            row["candidate_id"] = f"{scale_id.upper()}_{candidate.get('candidate_id')}"
            row["source_candidate_id"] = candidate.get("candidate_id")
            row["scale_id"] = scale_id
            row["scale_parameters"] = dict(scale)
            all_candidates.append(row)
    deduped = _dedupe_candidates([candidate for candidate in all_candidates if candidate.get("admissible")])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": str(ply_path),
        "part_id": part_id,
        "purpose": "Diagnostic-only multiscale raw point crease tracing below the F1 atom graph.",
        "max_points": int(max_points),
        "scale_reports": scale_reports,
        "raw_candidate_count_before_dedupe": len(all_candidates),
        "raw_admissible_candidate_count_before_dedupe": sum(1 for candidate in all_candidates if candidate.get("admissible")),
        "candidate_count": len(deduped),
        "admissible_candidate_count": len(deduped),
        "status": "raw_crease_candidates_found" if deduped else "no_raw_crease_candidate",
        "candidates": deduped,
    }


def _load_scales(path: Path | None) -> Sequence[Mapping[str, Any]]:
    if not path:
        return DEFAULT_SCALES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("scales") or DEFAULT_SCALES
    if isinstance(payload, list):
        return payload
    return DEFAULT_SCALES


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace multiscale raw-point crease candidates below the F1 atom graph.")
    parser.add_argument("--ply-path", required=True, type=Path)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--scales-json", type=Path, default=None)
    args = parser.parse_args()
    report = trace_multiscale_raw_point_crease_candidates(
        ply_path=args.ply_path.expanduser().resolve(),
        part_id=args.part_id,
        max_points=int(args.max_points),
        scales=_load_scales(args.scales_json),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "raw_admissible_candidate_count_before_dedupe": report["raw_admissible_candidate_count_before_dedupe"],
                "candidate_count": report["candidate_count"],
                "admissible_candidate_count": report["admissible_candidate_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
