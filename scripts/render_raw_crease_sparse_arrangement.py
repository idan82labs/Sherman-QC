#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _points(candidate: Mapping[str, Any]) -> np.ndarray:
    values = candidate.get("sampled_points") or ()
    return np.asarray(values, dtype=np.float64) if values else np.empty((0, 3), dtype=np.float64)


def _load_ply_vertices(path: Path, *, max_points: int = 45000) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        return np.empty((0, 3), dtype=np.float64)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    mesh = trimesh.load(str(path), process=False)
    vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float64)
    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices = vertices[::step]
    return vertices


def _limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins) * 0.56), 1.0)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _draw_line(ax: Any, points: np.ndarray, *, color: str, label: str, width: float) -> None:
    if len(points) == 0:
        return
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=36, label=label, depthshade=False)
        return
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[0]
    projections = (points - centroid) @ axis
    endpoints = np.vstack([centroid + axis * float(np.min(projections)), centroid + axis * float(np.max(projections))])
    ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color=color, linewidth=width, label=f"{label} line")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=18, alpha=0.9, label=label, depthshade=False)


def _panel(
    ax: Any,
    *,
    title: str,
    view: tuple[float, float],
    ply_points: np.ndarray,
    selected: Sequence[Mapping[str, Any]],
    suppressed: Sequence[Mapping[str, Any]],
    limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> None:
    if len(ply_points):
        ax.scatter(ply_points[:, 0], ply_points[:, 1], ply_points[:, 2], c="#71839a", s=0.35, alpha=0.11, label="PLY points", depthshade=False)
    colors = ["#16a34a", "#f97316", "#0891b2", "#22c55e", "#2563eb"]
    for index, candidate in enumerate(selected):
        pair = "-".join(str(value) for value in candidate.get("best_flange_pair") or ())
        label = f"SELECT {candidate.get('raw_candidate_id')} {pair}"
        _draw_line(ax, _points(candidate), color=colors[index % len(colors)], label=label, width=3.4)
    for candidate in suppressed:
        pair = "-".join(str(value) for value in candidate.get("best_flange_pair") or ())
        label = f"suppressed {candidate.get('raw_candidate_id')} {pair}"
        points = _points(candidate)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#a855f7", s=16, alpha=0.55, marker="x", label=label, depthshade=False)
            _draw_line(ax, points, color="#a855f7", label=f"suppressed line {candidate.get('raw_candidate_id')}", width=1.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_raw_crease_sparse_arrangement(
    *,
    arrangement_json: Path,
    bridge_json: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    arrangement = _load_json(arrangement_json)
    bridge = _load_json(bridge_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = {str(row.get("raw_candidate_id")): row for row in bridge.get("candidates") or ()}
    selected_ids = [str(value) for value in arrangement.get("selected_candidate_ids") or ()]
    suppressed_ids = [str(value) for value in arrangement.get("suppressed_duplicate_candidate_ids") or ()]
    families = list(arrangement.get("families") or ())
    selected = [
        row.get("merged_family_candidate") or row.get("selected_candidate")
        for row in families
        if row.get("merged_family_candidate") or row.get("selected_candidate")
    ]
    if not selected:
        selected = [candidates[value] for value in selected_ids if value in candidates]
    # Same-pair duplicate candidates are merged into the selected family support; keep
    # them out of the rejected overlay so the render shows the maximal support.
    suppressed = []

    scan_path = Path(str(arrangement.get("scan_path") or bridge.get("scan_path") or ""))
    ply_points = _load_ply_vertices(scan_path)
    candidate_points = [_points(row) for row in selected + suppressed if len(_points(row))]
    all_points = np.vstack([points for points in [ply_points, *candidate_points] if len(points)])
    limits = _limits(all_points)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    views = [
        ("Overview", (24.0, -58.0)),
        ("Side", (8.0, -88.0)),
        ("Top", (68.0, -43.0)),
    ]
    for index, (title, view) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _panel(
            ax,
            title=title,
            view=view,
            ply_points=ply_points,
            selected=selected,
            suppressed=suppressed,
            limits=limits,
        )
    handles, labels = fig.axes[0].get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"{arrangement.get('part_id')} sparse raw-crease arrangement - {arrangement.get('selected_family_count')} selected / {arrangement.get('safe_candidate_count')} safe",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.955), pad=0.5)
    output_png = output_dir / "raw_crease_sparse_arrangement_1080p.png"
    fig.savefig(output_png, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(fig)
    sidecar = {
        "overview_png": str(output_png),
        "selected_candidate_ids": selected_ids,
        "suppressed_duplicate_candidate_ids": suppressed_ids,
    }
    (output_dir / "raw_crease_sparse_arrangement_render_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Render sparse raw-crease arrangement diagnostics.")
    parser.add_argument("--arrangement-json", required=True, type=Path)
    parser.add_argument("--bridge-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            render_raw_crease_sparse_arrangement(
                arrangement_json=args.arrangement_json,
                bridge_json=args.bridge_json,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
