#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_ply_vertices(path: Path, *, max_points: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    points = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(points) > max_points:
        stride = max(1, int(math.ceil(len(points) / max_points)))
        points = points[::stride]
    return np.asarray(points, dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _estimate_normals_and_curvature(points: np.ndarray, *, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    distances, indexes = tree.query(points, k=min(k_neighbors, len(points)))
    normals = np.zeros_like(points)
    curvature = np.zeros(len(points), dtype=np.float64)
    spacing = np.zeros(len(points), dtype=np.float64)
    for index, neighbor_indexes in enumerate(indexes):
        neighbors = points[np.asarray(neighbor_indexes, dtype=np.int64)]
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = centered.T @ centered / max(len(neighbors) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        normal = eigenvectors[:, order[0]]
        normals[index] = _unit(normal)
        total = float(np.sum(eigenvalues))
        curvature[index] = float(eigenvalues[0] / total) if total > 1e-12 else 0.0
        row_distances = np.asarray(distances[index], dtype=np.float64)
        spacing[index] = float(np.median(row_distances[1:])) if len(row_distances) > 1 else 1.0
    return normals, curvature, spacing


def _normal_contrast(normals: np.ndarray, neighbor_indexes: np.ndarray) -> np.ndarray:
    contrast = np.zeros(len(normals), dtype=np.float64)
    for index, neighbors in enumerate(neighbor_indexes):
        dots = np.abs(normals[np.asarray(neighbors, dtype=np.int64)] @ normals[index])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        contrast[index] = float(np.percentile(angles, 80)) if len(angles) else 0.0
    return contrast


def _connected_components(candidate_indexes: np.ndarray, points: np.ndarray, *, radius: float) -> List[np.ndarray]:
    if len(candidate_indexes) == 0:
        return []
    candidate_points = points[candidate_indexes]
    tree = cKDTree(candidate_points)
    neighbors = tree.query_ball_point(candidate_points, r=radius)
    seen = np.zeros(len(candidate_indexes), dtype=bool)
    components: List[np.ndarray] = []
    for start in range(len(candidate_indexes)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        local: List[int] = []
        while stack:
            current = stack.pop()
            local.append(current)
            for neighbor in neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        components.append(candidate_indexes[np.asarray(local, dtype=np.int64)])
    components.sort(key=lambda values: (-len(values), int(values[0]) if len(values) else 0))
    return components


def _line_stats(points: np.ndarray) -> Dict[str, Any]:
    if len(points) == 0:
        return {"span_mm": 0.0, "thickness_mm": 0.0, "linearity": 0.0, "axis_direction": [1.0, 0.0, 0.0]}
    if len(points) == 1:
        return {"span_mm": 0.0, "thickness_mm": 0.0, "linearity": 1.0, "axis_direction": [1.0, 0.0, 0.0]}
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axis = _unit(eigenvectors[:, order[0]])
    projections = centered @ axis
    distances = np.linalg.norm(centered - np.outer(projections, axis), axis=1)
    total = float(np.sum(eigenvalues))
    return {
        "span_mm": float(np.max(projections) - np.min(projections)),
        "thickness_mm": float(np.percentile(distances, 80)) if len(distances) else 0.0,
        "linearity": float(eigenvalues[0] / total) if total > 1e-12 else 0.0,
        "axis_direction": [float(value) for value in axis.tolist()],
    }


def _quantile_threshold(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, min(max(quantile, 0.0), 1.0)))


def trace_raw_point_crease_candidates(
    *,
    ply_path: Path,
    part_id: str,
    max_points: int = 50000,
    k_neighbors: int = 24,
    score_quantile: float = 0.965,
    min_contrast_deg: float = 18.0,
    min_curvature: float = 0.018,
    min_points: int = 30,
    min_span_mm: float = 18.0,
    max_thickness_mm: float = 18.0,
    top_n_points_per_candidate: int = 900,
) -> Dict[str, Any]:
    points = _load_ply_vertices(ply_path, max_points=max_points)
    if len(points) == 0:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scan_path": str(ply_path),
            "part_id": part_id,
            "status": "empty_scan",
            "point_count": 0,
            "candidates": [],
        }
    normals, curvature, spacing = _estimate_normals_and_curvature(points, k_neighbors=k_neighbors)
    tree = cKDTree(points)
    _, neighbor_indexes = tree.query(points, k=min(k_neighbors, len(points)))
    contrast = _normal_contrast(normals, neighbor_indexes)
    curvature_norm = curvature / max(_quantile_threshold(curvature, 0.98), 1e-9)
    contrast_norm = contrast / 60.0
    score = np.clip(0.58 * curvature_norm + 0.42 * contrast_norm, 0.0, 3.0)
    threshold = max(_quantile_threshold(score, score_quantile), 0.28)
    candidate_mask = (score >= threshold) & (contrast >= min_contrast_deg) & (curvature >= min_curvature)
    candidate_indexes = np.flatnonzero(candidate_mask)
    radius = max(float(np.median(spacing)) * 3.5, 4.0)
    components = _connected_components(candidate_indexes, points, radius=radius)

    rows: List[Dict[str, Any]] = []
    for component_index, indexes in enumerate(components, start=1):
        component_points = points[indexes]
        stats = _line_stats(component_points)
        admissible = (
            len(indexes) >= min_points
            and stats["span_mm"] >= min_span_mm
            and stats["thickness_mm"] <= max_thickness_mm
            and stats["linearity"] >= 0.62
        )
        order = np.argsort(score[indexes])[::-1]
        selected = indexes[order[: min(top_n_points_per_candidate, len(order))]]
        rows.append(
            {
                "candidate_id": f"RPC{component_index}",
                "point_count": int(len(indexes)),
                "sampled_point_indexes": [int(value) for value in selected.tolist()],
                "sampled_points": [[float(coord) for coord in points[index].tolist()] for index in selected],
                "mean_score": float(np.mean(score[indexes])),
                "mean_curvature": float(np.mean(curvature[indexes])),
                "mean_normal_contrast_deg": float(np.mean(contrast[indexes])),
                "span_mm": stats["span_mm"],
                "thickness_mm": stats["thickness_mm"],
                "linearity": stats["linearity"],
                "axis_direction": stats["axis_direction"],
                "admissible": bool(admissible),
                "admissibility_reasons": []
                if admissible
                else [
                    reason
                    for reason, failed in (
                        ("too_few_points", len(indexes) < min_points),
                        ("span_too_short", stats["span_mm"] < min_span_mm),
                        ("too_thick_for_crease_line", stats["thickness_mm"] > max_thickness_mm),
                        ("not_line_like", stats["linearity"] < 0.62),
                    )
                    if failed
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["admissible"]),
            -float(row["span_mm"]),
            float(row["thickness_mm"]),
            str(row["candidate_id"]),
        )
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": str(ply_path),
        "part_id": part_id,
        "purpose": "Diagnostic-only raw point crease tracing below the F1 atom graph.",
        "point_count": int(len(points)),
        "k_neighbors": int(k_neighbors),
        "score_quantile": float(score_quantile),
        "score_threshold": float(threshold),
        "min_contrast_deg": float(min_contrast_deg),
        "min_curvature": float(min_curvature),
        "cluster_radius_mm": float(radius),
        "candidate_point_count": int(len(candidate_indexes)),
        "candidate_count": len(rows),
        "admissible_candidate_count": sum(1 for row in rows if row["admissible"]),
        "status": "raw_crease_candidates_found" if any(row["admissible"] for row in rows) else "no_raw_crease_candidate",
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace raw-point crease candidates below the F1 atom graph.")
    parser.add_argument("--ply-path", required=True, type=Path)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--score-quantile", type=float, default=0.965)
    parser.add_argument("--min-contrast-deg", type=float, default=18.0)
    parser.add_argument("--min-curvature", type=float, default=0.018)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--min-span-mm", type=float, default=18.0)
    parser.add_argument("--max-thickness-mm", type=float, default=18.0)
    args = parser.parse_args()
    report = trace_raw_point_crease_candidates(
        ply_path=args.ply_path.expanduser().resolve(),
        part_id=args.part_id,
        max_points=int(args.max_points),
        k_neighbors=int(args.k_neighbors),
        score_quantile=float(args.score_quantile),
        min_contrast_deg=float(args.min_contrast_deg),
        min_curvature=float(args.min_curvature),
        min_points=int(args.min_points),
        min_span_mm=float(args.min_span_mm),
        max_thickness_mm=float(args.max_thickness_mm),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "point_count": report["point_count"],
                "candidate_point_count": report.get("candidate_point_count", 0),
                "candidate_count": report.get("candidate_count", 0),
                "admissible_candidate_count": report.get("admissible_candidate_count", 0),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
