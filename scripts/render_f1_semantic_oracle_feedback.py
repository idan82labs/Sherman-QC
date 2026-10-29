#!/usr/bin/env python3
"""Render native-resolution candidate ownership feedback without gold geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import _endpoints
from scripts.render_manual_bend_review import _stable_pca, _view_basis


COLORS = {
    "PRECOMMIT": "#38bdf8",
    "TRUE_BEND_OBJECT": "#3dff8b",
    "DUPLICATE_ALIAS": "#d946ef",
    "PERIMETER_OR_TERMINAL": "#f59e0b",
    "HOLE": "#a855f7",
    "NONBEND_CURVED_FEATURE": "#fb7185",
    "UNMATCHED": "#ff334f",
}


def _load_points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError("geometry contains no finite points")
    return points


def _predictions(payload: dict) -> list[dict]:
    for key in ("owned_bend_regions", "manual_bends", "bends", "predictions"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError("prediction payload contains no candidate array")


def render(*, geometry: Path, prediction: Path, feedback: Path | None, output: Path) -> None:
    points = _load_points(geometry)
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    feedback_payload = json.loads(feedback.read_text(encoding="utf-8")) if feedback else None
    rows = _predictions(prediction_payload)
    feedback_by_id = (
        {str(row["candidate_id"]): row for row in feedback_payload["candidate_feedback"]}
        if feedback_payload
        else {}
    )
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    line_rows = []
    for index, row in enumerate(rows):
        candidate_id = str(row.get("bend_id") or row.get("candidate_id") or f"candidate-{index:04d}")
        if feedback_payload and candidate_id not in feedback_by_id:
            raise ValueError(f"candidate missing oracle feedback: {candidate_id}")
        endpoints = (_endpoints(row) - center) @ pca
        ownership = (
            str(feedback_by_id[candidate_id]["ownership_class"])
            if feedback_payload
            else "PRECOMMIT"
        )
        line_rows.append((candidate_id, ownership, endpoints))
    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=float)),
        ("PC2 / PC3", np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float)),
        ("Oblique front", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Oblique reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Oblique underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=256)
    fig.patch.set_facecolor("#07101f")
    for ax, (title, view) in zip(axes.flat[:6], views):
        view_points = projected @ view
        depth = view_points[:, 2]
        order = np.argsort(depth)
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        point_colors = plt.get_cmap("bone")(0.18 + 0.68 * normalized)
        ax.scatter(view_points[order, 0], view_points[order, 1], s=0.22, c=point_colors[order], alpha=0.32, linewidths=0, rasterized=True)
        for candidate_id, ownership, endpoints in line_rows:
            line = endpoints @ view
            color = COLORS.get(ownership, "#ffffff")
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=8.0, alpha=0.74, solid_capstyle="round", zorder=4)
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=4.2, alpha=0.96, solid_capstyle="round", zorder=5)
            midpoint = np.mean(line, axis=0)
            ax.text(midpoint[0], midpoint[1], candidate_id, fontsize=8, fontweight="bold", color="black", bbox={"boxstyle": "round,pad=0.16", "facecolor": color, "edgecolor": "white", "alpha": 0.94}, zorder=6)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_facecolor("#07101f")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    axes[2, 0].axis("off")
    if feedback_payload:
        metrics = feedback_payload["metrics"]
        summary_lines = (
            f"QUARANTINED {str(feedback_payload.get('phase') or '').upper()} ORACLE",
            "",
            f"TP {metrics['true_positive']}",
            f"FP {metrics['false_positive']}",
            f"FN {metrics['false_negative']}",
            f"precision {metrics['aggregate_precision']:.6f}",
            f"recall {metrics['aggregate_recall']:.6f}",
            f"F1 {metrics['aggregate_f1']:.6f}",
            f"exact count {metrics['aggregate_exact_count']}",
            "",
            "No gold geometry is rendered.",
        )
    else:
        summary_lines = (
            "SEALED TRANSFER PRECOMMIT",
            "",
            f"candidate count {len(line_rows)}",
            "oracle feedback NOT OPENED",
            "ownership NOT KNOWN",
            "",
            "All lines use one neutral color.",
            "No gold geometry is rendered.",
        )
    axes[2, 0].text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", color="white", family="monospace", fontsize=12)
    axes[2, 1].axis("off")
    table_lines = ["CANDIDATE OWNERSHIP" if feedback_payload else "PRECOMMITTED CANDIDATES"]
    for candidate_id, ownership, _candidate_endpoints in line_rows:
        table_lines.append(f"{candidate_id:>8}  {ownership}")
    axes[2, 1].text(0.0, 1.0, "\n".join(table_lines), va="top", ha="left", color="white", family="monospace", fontsize=11)
    axes[2, 2].axis("off")
    legend_lines = ["COLORS"] + [f"{key}: {value}" for key, value in COLORS.items()]
    axes[2, 2].text(0.0, 1.0, "\n".join(legend_lines), va="top", ha="left", color="white", family="monospace", fontsize=10)
    title = (
        f"F1 semantic-oracle mismatch audit — {feedback_payload['opaque_submission_id']} — all source vertices"
        if feedback_payload
        else f"F1 sealed-transfer precommit — {prediction_payload.get('profile') or prediction.stem} — all source vertices"
    )
    fig.suptitle(title, color="white", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=256, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        geometry=args.geometry.resolve(),
        prediction=args.prediction.resolve(),
        feedback=args.feedback.resolve() if args.feedback else None,
        output=args.output.resolve(),
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
