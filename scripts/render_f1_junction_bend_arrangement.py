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


def _region_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(region.get("bend_id")): region
        for region in decomposition.get("owned_bend_regions") or ()
        if region.get("bend_id")
    }


def _manual_sets(decomposition: Mapping[str, Any], feature_labels: Mapping[str, Any]) -> Tuple[set[str], set[str]]:
    regions = _region_lookup(decomposition)
    accepted: set[str] = set()
    rejected: set[str] = set()
    for row in feature_labels.get("region_labels") or ():
        bend_id = str(row.get("bend_id"))
        region = regions.get(bend_id)
        if not region:
            continue
        atoms = {str(value) for value in region.get("owned_atom_ids") or ()}
        if row.get("human_feature_label") == "conventional_bend":
            accepted.update(atoms)
        elif row.get("human_feature_label") in {"fragment_or_duplicate", "false_positive", "non_bend"}:
            rejected.update(atoms)
    return accepted, rejected


def _ridge_atom_ids(local_ridge_payload: Mapping[str, Any]) -> set[str]:
    ridges = local_ridge_payload.get("ridges") or ()
    if not ridges:
        return set()
    best = max(ridges, key=lambda row: len(row.get("atom_ids") or ()))
    return {str(value) for value in best.get("atom_ids") or ()}


def _points_for_atoms(atom_ids: set[str], atoms: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    points = [_vec((atoms.get(atom_id) or {}).get("centroid") or (0.0, 0.0, 0.0)) for atom_id in sorted(atom_ids)]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _load_ply_vertices(path: Path, *, max_points: int = 35000) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        return np.empty((0, 3), dtype=np.float64)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    mesh = trimesh.load(str(path), process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        vertices = getattr(mesh, "vertices", np.empty((0, 3)))
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices = vertices[::step]
    return vertices


def _axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.56)
    radius = max(radius, 1.0)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _padded_limits(points: np.ndarray, *, pad_ratio: float = 0.42, min_radius: float = 20.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * (0.5 + pad_ratio))
    radius = max(radius, min_radius)
    return tuple((float(center[index] - radius), float(center[index] + radius)) for index in range(3))  # type: ignore[return-value]


def _scatter(ax: Any, points: np.ndarray, *, color: str, size: float, alpha: float, label: str, marker: str = "o") -> None:
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, label=label, marker=marker, depthshade=False)


def _draw_candidate_line(ax: Any, candidate: Mapping[str, Any], *, color: str, label: str, linewidth: float = 3.0) -> None:
    endpoints_payload = candidate.get("endpoints") or ()
    if len(endpoints_payload) == 2:
        endpoints = np.asarray(endpoints_payload, dtype=np.float64)
        ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color=color, linewidth=linewidth, label=label)
        return
    centroid = _vec(candidate.get("centroid") or (0.0, 0.0, 0.0))
    axis = _vec(candidate.get("axis_direction") or (1.0, 0.0, 0.0))
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return
    axis = axis / norm
    span = max(float(candidate.get("span_mm") or 0.0), 8.0)
    endpoints = np.vstack([centroid - axis * span * 0.55, centroid + axis * span * 0.55])
    ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color=color, linewidth=linewidth, label=label)


def _render_panel(
    ax: Any,
    *,
    title: str,
    view: Tuple[float, float],
    ply_points: np.ndarray,
    atoms: Mapping[str, Mapping[str, Any]],
    accepted_atoms: set[str],
    rejected_atoms: set[str],
    ridge_atoms: set[str],
    selected_candidates: Sequence[Mapping[str, Any]],
    rejected_candidates: Sequence[Mapping[str, Any]],
    limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> None:
    selected_atoms = {str(atom_id) for candidate in selected_candidates for atom_id in candidate.get("atom_ids") or ()}
    rejected_chord_atoms = {str(atom_id) for candidate in rejected_candidates for atom_id in candidate.get("atom_ids") or ()}
    junction_atoms = set(ridge_atoms) - selected_atoms

    _scatter(ax, ply_points, color="#6f8398", size=0.35, alpha=0.11, label="PLY points")
    _scatter(ax, _points_for_atoms(rejected_atoms, atoms), color="#e11d48", size=12, alpha=0.18, label="manual rejected raw OB")
    _scatter(ax, _points_for_atoms(junction_atoms, atoms), color="#facc15", size=18, alpha=0.85, label="uncounted junction/ridge atoms")
    _scatter(ax, _points_for_atoms(rejected_chord_atoms, atoms), color="#a855f7", size=14, alpha=0.40, label="rejected chord support", marker="x")

    colors = ["#10b981", "#f97316", "#06b6d4", "#22c55e"]
    for index, candidate in enumerate(selected_candidates):
        color = colors[index % len(colors)]
        atom_ids = {str(value) for value in candidate.get("atom_ids") or ()}
        pair = "-".join(str(value) for value in candidate.get("flange_pair") or ())
        label = f"{candidate.get('candidate_id')} {pair}"
        _scatter(ax, _points_for_atoms(atom_ids, atoms), color=color, size=36, alpha=0.95, label=label)
        _draw_candidate_line(ax, candidate, color=color, label=f"{label} line")

    for candidate in rejected_candidates:
        _draw_candidate_line(ax, candidate, color="#a855f7", label=f"rejected {candidate.get('candidate_id')}", linewidth=1.8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_junction_arrangement(
    *,
    decomposition_json: Path,
    feature_labels_json: Path,
    local_ridge_json: Path,
    arrangement_json: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    decomposition = _load_json(decomposition_json)
    feature_labels = _load_json(feature_labels_json)
    local_ridge = _load_json(local_ridge_json)
    arrangement = _load_json(arrangement_json)
    part_id = str(decomposition.get("part_id") or arrangement.get("part_id") or "scan")

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = _atom_lookup(decomposition)
    accepted_atoms, rejected_atoms = _manual_sets(decomposition, feature_labels)
    ridge_atoms = _ridge_atom_ids(local_ridge)
    candidate_lookup = {str(row.get("candidate_id")): row for row in arrangement.get("candidates") or ()}
    best_ids = [str(value) for value in (arrangement.get("best_hypothesis") or {}).get("candidate_ids") or ()]
    selected_candidates = [candidate_lookup[value] for value in best_ids if value in candidate_lookup]
    rejected_candidates = [row for row in arrangement.get("candidates") or () if row.get("possible_chord")]

    scan_path = Path(str(decomposition.get("scan_path") or ""))
    ply_points = _load_ply_vertices(scan_path)
    atom_points = _points_for_atoms(set(atoms), atoms)
    all_points = np.vstack([points for points in (ply_points, atom_points) if len(points)])
    limits = _axis_limits(all_points)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    views = [
        ("Overview: sparse bend-line arrangement", (23.0, -58.0)),
        ("Side: selected lines vs junction support", (8.0, -89.0)),
        ("Top/edge: chord rejection check", (62.0, -42.0)),
        ("Opposite: line endpoints and support separation", (18.0, 122.0)),
    ]
    for index, (title, view) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        _render_panel(
            ax,
            title=title,
            view=view,
            ply_points=ply_points,
            atoms=atoms,
            accepted_atoms=accepted_atoms,
            rejected_atoms=rejected_atoms,
            ridge_atoms=ridge_atoms,
            selected_candidates=selected_candidates,
            rejected_candidates=rejected_candidates,
            limits=limits,
        )

    handles, labels = fig.axes[0].get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        f"{part_id} junction-aware bend-line diagnostic: selected local bend-line arrangement",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.965), pad=0.45)
    overview_path = output_dir / "junction_arrangement_overview_1080p.png"
    fig.savefig(overview_path, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(fig)

    focus_ids = set(ridge_atoms) | accepted_atoms | {str(atom_id) for row in selected_candidates for atom_id in row.get("atom_ids") or ()}
    focus_points = _points_for_atoms(focus_ids, atoms)
    focus_limits = _padded_limits(focus_points if len(focus_points) else atom_points, pad_ratio=0.30, min_radius=24.0)
    focus_fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f8fafc")
    focus_views = [
        ("Close overview", (21.0, -58.0)),
        ("Close side", (8.0, -89.0)),
        ("Close top", (63.0, -42.0)),
    ]
    for index, (title, view) in enumerate(focus_views, start=1):
        ax = focus_fig.add_subplot(1, 3, index, projection="3d")
        _render_panel(
            ax,
            title=title,
            view=view,
            ply_points=ply_points,
            atoms=atoms,
            accepted_atoms=accepted_atoms,
            rejected_atoms=rejected_atoms,
            ridge_atoms=ridge_atoms,
            selected_candidates=selected_candidates,
            rejected_candidates=rejected_candidates,
            limits=focus_limits,
        )
    handles, labels = focus_fig.axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    focus_fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    focus_fig.suptitle(
        f"{part_id} close-up: selected conventional bend-line instances and uncounted junction support",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    focus_fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.955), pad=0.25)
    focus_path = output_dir / "junction_arrangement_closeup_1080p.png"
    focus_fig.savefig(focus_path, format="png", dpi=100, facecolor="#f8fafc")
    plt.close(focus_fig)

    sidecar = {
        "decomposition_json": str(decomposition_json),
        "feature_labels_json": str(feature_labels_json),
        "local_ridge_json": str(local_ridge_json),
        "arrangement_json": str(arrangement_json),
        "overview_png": str(overview_path),
        "closeup_png": str(focus_path),
        "best_hypothesis": arrangement.get("best_hypothesis"),
        "selected_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "flange_pair": row.get("flange_pair"),
                "atom_count": row.get("atom_count"),
                "span_mm": row.get("span_mm"),
                "axis_delta_deg": row.get("axis_delta_deg"),
                "median_line_residual_mm": row.get("median_line_residual_mm"),
            }
            for row in selected_candidates
        ],
        "rejected_chord_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "flange_pair": row.get("flange_pair"),
                "reason_codes": row.get("reason_codes"),
            }
            for row in rejected_candidates
        ],
    }
    (output_dir / "junction_arrangement_render_sidecar.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a 1080p visual check for F1 junction-aware bend-line arrangement diagnostics.")
    parser.add_argument("--decomposition-json", required=True)
    parser.add_argument("--feature-labels-json", required=True)
    parser.add_argument("--local-ridge-json", required=True)
    parser.add_argument("--arrangement-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sidecar = render_junction_arrangement(
        decomposition_json=Path(args.decomposition_json).expanduser().resolve(),
        feature_labels_json=Path(args.feature_labels_json).expanduser().resolve(),
        local_ridge_json=Path(args.local_ridge_json).expanduser().resolve(),
        arrangement_json=Path(args.arrangement_json).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print(json.dumps({"overview_png": sidecar["overview_png"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
