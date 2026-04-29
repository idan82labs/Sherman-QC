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


def _vec(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


def _load_ply_vertices(path: Path, *, max_points: int = 42000) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        return np.empty((0, 3), dtype=np.float64)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    mesh = trimesh.load(str(path), process=False)
    vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float64)
    if len(vertices) > max_points:
        vertices = vertices[:: max(1, len(vertices) // max_points)]
    return vertices


def _points_for_atoms(atom_ids: set[str], atoms: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    points = [_vec((atoms.get(atom_id) or {}).get("centroid") or (0.0, 0.0, 0.0)) for atom_id in sorted(atom_ids)]
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)


def _axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins) * 0.58), 1.0)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _padded_limits(points: np.ndarray, *, pad_ratio: float = 0.48, min_radius: float = 28.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins) * (0.5 + pad_ratio)), min_radius)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _scatter(ax: Any, points: np.ndarray, *, color: str, size: float, alpha: float, label: str, marker: str = "o") -> None:
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, label=label, marker=marker, depthshade=False)


def _draw_line(ax: Any, candidate: Mapping[str, Any], *, color: str, label: str, linewidth: float = 3.8) -> None:
    endpoints = candidate.get("endpoints") or ()
    if len(endpoints) != 2:
        return
    points = np.asarray(endpoints, dtype=np.float64)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth, label=label)


def _candidate_lookup(arrangement: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = {str(row.get("candidate_id")): row for row in arrangement.get("candidates") or () if row.get("candidate_id")}
    for row in arrangement.get("top_recovered_contact_birth_candidates") or ():
        if row.get("candidate_id"):
            rows[str(row["candidate_id"])] = row
    return rows


def _same_pair_owned_ids(candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("bend_id"))
        for candidate in candidates
        for row in (candidate.get("duplicate_diagnostics") or {}).get("same_pair_owned_regions") or ()
        if row.get("bend_id")
    }


def _owned_region_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(region.get("bend_id")): region
        for region in decomposition.get("owned_bend_regions") or ()
        if region.get("bend_id")
    }


def render_candidate_comparison(
    *,
    decomposition_json: Path,
    arrangement_json: Path,
    output_dir: Path,
    candidate_ids: Sequence[str],
) -> Dict[str, Any]:
    decomposition = _load_json(decomposition_json)
    arrangement = _load_json(arrangement_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = _atom_lookup(decomposition)
    candidates_by_id = _candidate_lookup(arrangement)
    selected = [candidates_by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in candidates_by_id]
    owned_regions = _owned_region_lookup(decomposition)
    raw_ids = _same_pair_owned_ids(selected)

    ply_points = _load_ply_vertices(Path(str(decomposition.get("scan_path") or "")))
    atom_points = _points_for_atoms(set(atoms), atoms)
    all_points = np.vstack([points for points in (ply_points, atom_points) if len(points)])
    focus_atom_ids = (
        {str(atom_id) for candidate in selected for atom_id in candidate.get("atom_ids") or ()}
        | {str(atom_id) for bend_id in raw_ids for atom_id in (owned_regions.get(bend_id) or {}).get("owned_atom_ids") or ()}
    )
    focus_points = _points_for_atoms(focus_atom_ids, atoms)
    full_limits = _axis_limits(all_points)
    focus_limits = _padded_limits(focus_points if len(focus_points) else atom_points)

    colors = {
        candidate_ids[0] if candidate_ids else "A": "#2563eb",
        candidate_ids[1] if len(candidate_ids) > 1 else "B": "#dc2626",
    }
    views = [
        ("Overview: candidate competition", (23.0, -58.0), full_limits),
        ("Side: support offset check", (8.0, -89.0), full_limits),
        ("Close top: atom/line overlap", (62.0, -42.0), focus_limits),
        ("Close opposite: raw-family separation", (18.0, 122.0), focus_limits),
    ]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    for panel_index, (title, view, limits) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, panel_index, projection="3d")
        _scatter(ax, ply_points, color="#64748b", size=0.35, alpha=0.12, label="PLY points")
        for bend_id in sorted(raw_ids):
            region = owned_regions.get(bend_id)
            if not region:
                continue
            _scatter(
                ax,
                _points_for_atoms({str(value) for value in region.get("owned_atom_ids") or ()}, atoms),
                color="#f97316",
                size=18,
                alpha=0.42,
                label=f"raw {bend_id} same-pair",
            )
        for candidate in selected:
            candidate_id = str(candidate.get("candidate_id"))
            color = colors.get(candidate_id, "#0f766e")
            atom_ids = {str(value) for value in candidate.get("atom_ids") or ()}
            pair = "-".join(str(value) for value in candidate.get("flange_pair") or ())
            label = f"{candidate_id} {candidate.get('source_kind')} {pair}"
            _scatter(ax, _points_for_atoms(atom_ids, atoms), color=color, size=46, alpha=0.96, label=label)
            _draw_line(ax, candidate, color=color, label=f"{candidate_id} line", linewidth=4.2)
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
    fig.suptitle(f"{decomposition.get('part_id')} F1 candidate comparison: {' vs '.join(candidate_ids)}", fontsize=15, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.955), pad=0.25)
    output_path = output_dir / "candidate_comparison_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(fig)

    sidecar = {
        "overview_png": str(output_path),
        "decomposition_json": str(decomposition_json),
        "arrangement_json": str(arrangement_json),
        "candidate_ids": list(candidate_ids),
        "rendered_candidate_ids": [candidate.get("candidate_id") for candidate in selected],
        "same_pair_owned_region_ids": sorted(raw_ids),
    }
    (output_dir / "candidate_comparison_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a focused comparison between F1 local arrangement candidates.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--arrangement-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-ids", nargs="+", required=True)
    args = parser.parse_args()
    sidecar = render_candidate_comparison(
        decomposition_json=args.decomposition_json.expanduser().resolve(),
        arrangement_json=args.arrangement_json.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        candidate_ids=args.candidate_ids,
    )
    print(json.dumps({"overview_png": sidecar["overview_png"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
