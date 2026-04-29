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


def render_raw_point_crease_bridge_validation(
    *,
    bridge_json: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    bridge = _load_json(bridge_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_points = _load_ply_vertices(Path(str(bridge.get("scan_path") or "")))
    candidates = list(bridge.get("candidates") or ())
    candidate_arrays = [_candidate_points(candidate) for candidate in candidates]
    all_points = np.vstack([points for points in [ply_points, *candidate_arrays] if len(points)])
    limits = _axis_limits(all_points)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    views = [("Overview", (22, -57)), ("Side", (8, -88)), ("Top", (63, -43))]
    for index, (title, view) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _scatter(ax, ply_points, color="#64748b", size=0.28, alpha=0.08, label="PLY points")
        for candidate in candidates:
            safe = bool(candidate.get("bridge_safe"))
            color = "#16a34a" if safe else "#ef4444"
            status = "SAFE" if safe else "reject"
            pair = "-".join(str(value) for value in candidate.get("best_flange_pair") or ())
            label = (
                f"{candidate.get('raw_candidate_id')} {status} {pair} "
                f"score={candidate.get('best_pair_score')}"
            )
            _scatter(ax, _candidate_points(candidate), color=color, size=10 if safe else 7, alpha=0.86 if safe else 0.50, label=label)
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
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=2, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"{bridge.get('part_id')} raw-crease to F1 bridge validation", fontsize=15, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.955), pad=0.25)
    output_path = output_dir / "raw_point_crease_bridge_validation_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(fig)
    sidecar = {
        "overview_png": str(output_path),
        "bridge_json": str(bridge_json),
        "bridge_safe_candidate_ids": [candidate.get("raw_candidate_id") for candidate in candidates if candidate.get("bridge_safe")],
    }
    (output_dir / "raw_point_crease_bridge_validation_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Render raw-point crease bridge validation diagnostics.")
    parser.add_argument("--bridge-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sidecar = render_raw_point_crease_bridge_validation(
        bridge_json=args.bridge_json.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps({"overview_png": sidecar["overview_png"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
