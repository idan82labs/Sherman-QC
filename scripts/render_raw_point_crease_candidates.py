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


def _load_ply_vertices(path: Path, *, max_points: int = 48000) -> np.ndarray:
    import trimesh

    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    mesh = trimesh.load(str(path), process=False)
    points = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float64)
    if len(points) > max_points:
        points = points[:: max(1, len(points) // max_points)]
    return points


def _axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins) * 0.58), 1.0)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _candidate_points(candidate: Mapping[str, Any]) -> np.ndarray:
    values = candidate.get("sampled_points") or ()
    return np.asarray(values, dtype=np.float64) if values else np.empty((0, 3), dtype=np.float64)


def _scatter(ax: Any, points: np.ndarray, *, color: str, size: float, alpha: float, label: str) -> None:
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, label=label, depthshade=False)


def render_raw_point_crease_candidates(
    *,
    creases_json: Path,
    output_dir: Path,
    top_n: int = 10,
) -> Dict[str, Any]:
    creases = _load_json(creases_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_points = _load_ply_vertices(Path(str(creases.get("scan_path") or "")))
    candidate_arrays = [_candidate_points(candidate) for candidate in (creases.get("candidates") or ())[:top_n]]
    all_points = np.vstack([points for points in [ply_points, *candidate_arrays] if len(points)])
    limits = _axis_limits(all_points)
    selected = list(creases.get("candidates") or [])[:top_n]
    colors = ["#f97316", "#22c55e", "#06b6d4", "#a855f7", "#ef4444", "#0f766e", "#eab308", "#ec4899", "#84cc16", "#14b8a6"]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    views = [("Overview", (22, -57)), ("Side", (8, -88)), ("Top", (63, -43))]
    for index, (title, view) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _scatter(ax, ply_points, color="#64748b", size=0.28, alpha=0.08, label="PLY points")
        for candidate_index, candidate in enumerate(selected):
            color = colors[candidate_index % len(colors)]
            points = _candidate_points(candidate)
            label = (
                f"{candidate.get('candidate_id')} pts={candidate.get('point_count')} "
                f"span={candidate.get('span_mm'):.1f} thick={candidate.get('thickness_mm'):.1f}"
            )
            _scatter(ax, points, color=color, size=7, alpha=0.82, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"{creases.get('part_id')} raw-point crease candidate diagnostic", fontsize=15, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.955), pad=0.25)
    output_path = output_dir / "raw_point_crease_candidates_top_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(fig)
    sidecar = {
        "overview_png": str(output_path),
        "creases_json": str(creases_json),
        "top_candidate_ids": [candidate.get("candidate_id") for candidate in selected],
    }
    (output_dir / "raw_point_crease_candidates_render_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Render raw-point crease candidate diagnostics.")
    parser.add_argument("--creases-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    sidecar = render_raw_point_crease_candidates(
        creases_json=args.creases_json.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        top_n=int(args.top_n),
    )
    print(json.dumps({"overview_png": sidecar["overview_png"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
