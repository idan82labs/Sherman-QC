#!/usr/bin/env python3
"""Render high-resolution orthographic and oblique manual bend review sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _stable_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample = points - center
    covariance = sample.T @ sample / max(1, len(sample))
    _values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return center, basis


def _points(path: Path, max_points: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points


def _lines(
    path: Path,
    selected_ids: set[str] | None = None,
    hypotheses_source: str | None = None,
    candidate_source: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hypotheses_source and candidate_source:
        raise ValueError("hypotheses_source and candidate_source are mutually exclusive")
    if candidate_source:
        records = [
            record
            for record in payload.get("candidates", [])
            if record.get("candidate_source") == candidate_source
        ]
    elif hypotheses_source:
        records = [
            record
            for record in payload.get("bend_hypotheses", [])
            if record.get("source_kind") == hypotheses_source
        ]
    elif "manual_bends" in payload:
        records = payload["manual_bends"]
    elif "predictions" in payload:
        records = payload["predictions"]
    else:
        records = payload.get("owned_bend_regions") or []
    lines: list[tuple[str, np.ndarray]] = []
    for record in records:
        bend_id = str(record.get("bend_id") or "B?")
        if selected_ids and bend_id not in selected_ids:
            continue
        start = record.get("line_start")
        end = record.get("line_end")
        if start is None or end is None:
            endpoints = record.get("span_endpoints") or []
            if len(endpoints) >= 2:
                start, end = endpoints[:2]
        if start is None or end is None:
            anchor = record.get("anchor")
            direction = record.get("axis_direction")
            span = record.get("span_mm")
            if anchor is not None and direction is not None and span is not None:
                anchor_array = np.asarray(anchor, dtype=np.float64)
                direction_array = np.asarray(direction, dtype=np.float64)
                direction_array /= max(float(np.linalg.norm(direction_array)), 1e-12)
                half_span = direction_array * (float(span) / 2.0)
                start = (anchor_array - half_span).tolist()
                end = (anchor_array + half_span).tolist()
        if start is None or end is None:
            continue
        lines.append(
            (
                bend_id,
                np.asarray([start, end], dtype=np.float64),
            )
        )
    return lines


def _view_basis(direction: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    right = np.cross(up_hint, direction)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(np.array([0.0, 1.0, 0.0]), direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return np.column_stack((right, up, direction))


def render(
    scan_path: Path,
    labels_path: Path,
    output_path: Path,
    max_points: int,
    selected_ids: set[str] | None = None,
    hypotheses_source: str | None = None,
    candidate_source: str | None = None,
    focus_radius_mm: float | None = None,
) -> None:
    points = _points(scan_path, max_points)
    review_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if candidate_source:
        review_kind = f"Candidate review ({candidate_source})"
    elif hypotheses_source:
        review_kind = f"Hypothesis review ({hypotheses_source})"
    elif "manual_bends" in review_payload:
        review_kind = "Manual bend review"
    else:
        review_kind = "F1 prediction review"
    raw_lines = _lines(
        labels_path, selected_ids, hypotheses_source, candidate_source,
    )
    if focus_radius_mm is not None:
        radius = float(focus_radius_mm)
        if radius <= 0.0 or not raw_lines:
            raise ValueError("focus radius requires at least one selected line and a positive radius")
        keep = np.zeros(len(points), dtype=bool)
        for _bend_id, line in raw_lines:
            start, end = line
            delta = end - start
            denominator = max(float(np.dot(delta, delta)), 1e-12)
            parameter = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
            nearest = start + parameter[:, None] * delta
            keep |= np.linalg.norm(points - nearest, axis=1) <= radius
        if int(keep.sum()) < 100:
            raise ValueError("focus crop contains too few scan points")
        points = points[keep]
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    line_records = [(bend_id, (line - center) @ pca) for bend_id, line in raw_lines]

    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)),
        ("PC2 / PC3", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)),
        ("Oblique front", _view_basis(np.array([1.0, 1.0, 0.75]), np.array([0.0, 0.0, 1.0]))),
        ("Oblique reverse", _view_basis(np.array([-1.0, 0.8, 0.65]), np.array([0.0, 0.0, 1.0]))),
        ("Oblique underside", _view_basis(np.array([0.8, -1.0, -0.7]), np.array([0.0, 0.0, 1.0]))),
    )
    colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, max(1, len(line_records))))
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=160)
    fig.patch.set_facecolor("#07101f")
    for ax, (title, view) in zip(axes.flat, views):
        view_points = projected @ view
        depth = view_points[:, 2]
        order = np.argsort(depth)
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        point_colors = plt.get_cmap("bone")(0.22 + 0.68 * normalized)
        ax.scatter(
            view_points[order, 0],
            view_points[order, 1],
            s=0.28,
            c=point_colors[order],
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
        for index, (bend_id, endpoints) in enumerate(line_records):
            line = endpoints @ view
            color = colors[index]
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=8.0, alpha=0.72, solid_capstyle="round")
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=4.5, solid_capstyle="round")
            ax.scatter(line[:, 0], line[:, 1], s=42, color=[color], edgecolor="white", linewidth=0.8, zorder=5)
            midpoint = np.mean(line, axis=0)
            ax.text(
                midpoint[0], midpoint[1], bend_id,
                color="white", fontsize=10, fontweight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "edgecolor": "white", "alpha": 0.95},
                zorder=6,
            )
        ax.set_title(title, color="white", fontsize=15, fontweight="bold")
        ax.set_facecolor("#07101f")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    fig.suptitle(
        f"{review_kind} — {scan_path.name} — {len(line_records)} axes",
        color="white", fontsize=21, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--ids", help="Comma-separated bend identifiers to render")
    parser.add_argument(
        "--focus-radius-mm", type=float,
        help="Crop scan context to this 3D radius around the selected axes",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--hypotheses-source",
        choices=("detector_bend_seed", "interface_birth", "zero_shot_strip_seed", "scan_supported_corridor_completion_seed"),
        help="Render raw hypotheses of one source kind instead of owned bends",
    )
    source_group.add_argument(
        "--candidate-source",
        help="Render candidate-pool records from one candidate_source value",
    )
    args = parser.parse_args()
    selected_ids = {item.strip() for item in (args.ids or "").split(",") if item.strip()} or None
    render(
        args.scan.resolve(),
        args.labels.resolve(),
        args.output.resolve(),
        args.max_points,
        selected_ids,
        args.hypotheses_source,
        args.candidate_source,
        args.focus_radius_mm,
    )


if __name__ == "__main__":
    main()
