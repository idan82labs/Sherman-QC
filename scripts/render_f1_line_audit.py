#!/usr/bin/env python3
"""Render uncluttered orthographic F1 bend-line audit sheets.

The production snapshot renderer preserves the source mesh's QC colors, which
can obscure short line-local F1 corridors.  This diagnostic renderer deliberately
uses a neutral point cloud and thick, uniquely colored line segments so visual
review can distinguish alignment, duplicates, holes, and face-interior markers.
"""

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


def _select_owned_regions(
    regions: list[dict[str, object]], selected_ids: set[str]
) -> list[dict[str, object]]:
    if not selected_ids:
        return regions
    selected = [
        bend for bend in regions if str(bend.get("bend_id") or "") in selected_ids
    ]
    found_ids = {str(bend.get("bend_id") or "") for bend in selected}
    missing_ids = sorted(selected_ids - found_ids)
    if missing_ids:
        raise ValueError(
            "requested owned-region identifiers are absent: " + ", ".join(missing_ids)
        )
    return selected


def render(
    manifest_path: Path,
    output_path: Path,
    max_points: int,
    all_raw: bool,
    hypothesis_source: str | None,
    hypothesis_ids: str | None = None,
    region_ids: str | None = None,
    geometry_path: Path | None = None,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_geometry_path = geometry_path or Path(str(payload["scan_path"]))
    mesh = o3d.io.read_triangle_mesh(str(source_geometry_path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(source_geometry_path))
        points = np.asarray(cloud.points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    step = max(1, int(np.ceil(len(points) / max(1, max_points))))
    points = points[::step]

    center, basis = _stable_pca(points)
    projected = (points - center) @ basis
    selected_hypothesis_ids = {
        item.strip() for item in str(hypothesis_ids or "").split(",") if item.strip()
    }
    selected_region_ids = {
        item.strip() for item in str(region_ids or "").split(",") if item.strip()
    }
    if (hypothesis_source or selected_hypothesis_ids) and "bend_hypotheses" in payload:
        source_bends = []
        for bend in payload.get("bend_hypotheses") or []:
            if hypothesis_source and str(bend.get("source_kind") or "") != hypothesis_source:
                continue
            if selected_hypothesis_ids and str(bend.get("bend_id") or "") not in selected_hypothesis_ids:
                continue
            anchor = np.asarray(bend["anchor"], dtype=np.float64)
            axis = np.asarray(bend["axis_direction"], dtype=np.float64)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            half_span = 0.5 * float(bend.get("span_mm") or 0.0)
            source_bends.append(
                {
                    "bend_id": bend.get("bend_id"),
                    "detected_line": {
                        "raw_start": (anchor - axis * half_span).tolist(),
                        "raw_end": (anchor + axis * half_span).tolist(),
                    },
                }
            )
    elif "owned_bend_regions" in payload:
        source_bends = _select_owned_regions(
            payload.get("owned_bend_regions") or [], selected_region_ids
        )
    elif "predictions" in payload or "line_objects" in payload:
        source_bends = payload.get("predictions") or payload.get("line_objects") or []
        if selected_hypothesis_ids:
            source_bends = [
                bend
                for bend in source_bends
                if str(
                    bend.get("candidate_id")
                    or bend.get("source_hypothesis_id")
                    or bend.get("bend_id")
                    or ""
                )
                in selected_hypothesis_ids
            ]
    elif all_raw:
        source_bends = payload.get("all_raw_bends") or payload.get("bends") or []
    else:
        source_bends = payload.get("bends") or []
    bend_lines = []
    for bend in source_bends:
        line = bend.get("detected_line") or bend.get("cad_line") or {}
        start = line.get("raw_start") or line.get("display_start")
        end = line.get("raw_end") or line.get("display_end")
        if start is None or end is None:
            start, end = bend.get("line_start"), bend.get("line_end")
        if (start is None or end is None) and bend.get("span_endpoints"):
            start, end = bend["span_endpoints"][:2]
        if start is None or end is None:
            continue
        endpoints = (np.asarray([start, end], dtype=np.float64) - center) @ basis
        bend_lines.append(
            (
                str(
                    bend.get("bend_id")
                    or bend.get("candidate_id")
                    or bend.get("source_hypothesis_id")
                    or "B?"
                ),
                endpoints,
            )
        )

    projections = ((0, 1, "PC1 / PC2"), (0, 2, "PC1 / PC3"), (1, 2, "PC2 / PC3"))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(bend_lines))))
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), dpi=100)
    fig.patch.set_facecolor("white")
    for ax, (left, right, title) in zip(axes, projections):
        ax.scatter(projected[:, left], projected[:, right], s=0.18, c="#64748b", alpha=0.18, linewidths=0)
        for index, (bend_id, endpoints) in enumerate(bend_lines):
            color = colors[index]
            ax.plot(endpoints[:, left], endpoints[:, right], color=color, linewidth=4.0, solid_capstyle="round")
            ax.scatter(endpoints[:, left], endpoints[:, right], s=20, color=[color], edgecolor="white", linewidth=0.6)
            midpoint = np.mean(endpoints, axis=0)
            ax.text(
                midpoint[left], midpoint[right], bend_id,
                fontsize=8, fontweight="bold", color="black",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": color, "edgecolor": "white", "alpha": 0.92},
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    fig.suptitle(
        f"F1 line audit: {source_geometry_path.name} — {len(bend_lines)} rendered regions",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--all-raw", action="store_true")
    parser.add_argument("--hypotheses-source")
    parser.add_argument("--hypothesis-ids", help="Comma-separated bend hypothesis identifiers")
    parser.add_argument("--region-ids", help="Comma-separated owned bend-region identifiers")
    parser.add_argument(
        "--geometry-path",
        type=Path,
        help="Optional scan-frame geometry used only as the neutral render background",
    )
    args = parser.parse_args()
    render(
        args.manifest.resolve(),
        args.output.resolve(),
        args.max_points,
        args.all_raw,
        args.hypotheses_source,
        args.hypothesis_ids,
        args.region_ids,
        args.geometry_path.resolve() if args.geometry_path else None,
    )


if __name__ == "__main__":
    main()
