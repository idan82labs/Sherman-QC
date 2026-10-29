#!/usr/bin/env python3
"""Render native 4096px V34 pre/post mechanism-review montages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research/f1-v34-opgt-v1"
FIX = OUT / "source_free_fixtures"
PRE = OUT / "preinvoke_visuals"
POST = OUT / "postinvoke_visuals"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_candidate_prism_geometry import (  # noqa: E402
    canonical_triangle_mesh,
    candidate_frame_domain,
    iter_prism_pieces,
    polygon_area,
)
from backend.f1_distributed_harmonic_fold_flux import parse_candidates  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound(fixture_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = load(FIX / f"{fixture_id}.json")
    return descriptor, load(ROOT / descriptor["triangle_binding_path"])


def _source_observations(
    raw: dict[str, Any], candidate_record: dict[str, Any]
) -> tuple[Any, Any, list[tuple[float, float, float, float]]]:
    candidate = parse_candidates([candidate_record])[0]
    mesh = canonical_triangle_mesh(raw, candidate.scale)
    frame = candidate_frame_domain(candidate, mesh)
    rows: list[tuple[float, float, float, float]] = []
    for _index, normal, polygon in iter_prism_pieces(mesh, frame):
        area = polygon_area(polygon)
        if area <= 0.0:
            continue
        centroid = np.mean(polygon, axis=0)
        nb = float(np.dot(normal, frame.transverse))
        nc = float(np.dot(normal, frame.complement))
        phase = 0.5 * np.arctan2(2.0 * nb * nc, nb * nb - nc * nc)
        rows.append((float(centroid[0]), float(centroid[1]), float(phase), area))
    return mesh, frame, rows


def _dark(ax: Any) -> None:
    ax.set_facecolor("#0d1822")
    ax.tick_params(colors="#9fb2c0")
    ax.grid(color="#29404f", alpha=0.25)
    for spine in getattr(ax, "spines", {}).values():
        spine.set_color("#314654")


def render(
    fixture_id: str,
    destination: Path,
    decision: dict[str, Any] | None = None,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    descriptor, raw = load_bound(fixture_id)
    points = np.asarray(raw["positions"], dtype=np.float64)
    faces = np.asarray(raw["triangles"], dtype=np.int64)
    mesh, frame, observations = _source_observations(raw, descriptor["candidate_records"][0])
    rel = points - frame.origin
    u = (points - frame.p0) @ frame.axis
    q = rel @ frame.transverse
    r = rel @ frame.complement
    local = np.column_stack((q, u, r))

    fig = plt.figure(figsize=(16, 16), dpi=256, facecolor="#081018")
    oblique = fig.add_subplot(321, projection="3d")
    axis_view = fig.add_subplot(322, projection="3d")
    cross = fig.add_subplot(323)
    ordered = fig.add_subplot(324)
    cell = fig.add_subplot(325)
    path = fig.add_subplot(326)
    panels = (oblique, axis_view, cross, ordered, cell, path)
    for panel in panels:
        _dark(panel)

    unique_faces: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for face in faces:
        key = tuple(sorted(int(value) for value in face))
        if key not in seen:
            seen.add(key)
            unique_faces.append(face)
    if unique_faces:
        face_array = np.asarray(unique_faces, dtype=np.int64)
        stride = max(1, len(face_array) // 18000)
        shown = face_array[::stride]
        centers = np.mean(u[shown], axis=1)
        colors = plt.get_cmap("viridis")((centers - np.min(centers)) / max(float(np.ptp(centers)), 1e-12))
        oblique.add_collection3d(Poly3DCollection(local[shown], facecolors=colors, edgecolors="none", alpha=0.9))
        axis_view.add_collection3d(Poly3DCollection(local[shown], facecolors="#7fc8d8", edgecolors="#173442", linewidths=0.06, alpha=0.72))
        used = np.unique(shown.reshape(-1))
        for panel in (oblique, axis_view):
            panel.auto_scale_xyz(q[used], u[used], r[used])
            panel.plot([0.0, 0.0], [0.0, frame.length], [0.0, 0.0], color="#ff5b5b", lw=2.5)
            panel.set_xlabel("transverse q")
            panel.set_ylabel("axis u")
            panel.set_zlabel("complement r")
        oblique.view_init(elev=25, azim=-55)
        axis_view.view_init(elev=8, azim=-92)
        oblique.set_box_aspect((1.0, 2.2, 1.0))
        axis_view.set_box_aspect((1.0, 2.2, 1.0))
    oblique.set_title("RAW OBLIQUE TRIANGLE SURFACE", color="white")
    axis_view.set_title("CANDIDATE-AXIS VIEW", color="white")

    surface_points = np.unique(faces.reshape(-1)) if faces.size else np.arange(len(points))
    sample = surface_points[:: max(1, len(surface_points) // 40000)]
    cross.scatter(q[sample] / frame.scale, r[sample] / frame.scale, s=0.45, c="#d7e4ea", alpha=0.55, rasterized=True)
    cross.axvline(0.0, color="#ff5b5b", lw=1.6)
    cross.axhline(0.0, color="#ff5b5b", lw=0.7, alpha=0.65)
    cross.set_aspect("equal", adjustable="datalim")
    cross.set_xlabel("q / s")
    cross.set_ylabel("r / s")
    cross.set_title("TRANSVERSE SECTION · RIMS/RETURNS/SEAMS", color="white")

    if observations:
        obs = np.asarray(observations, dtype=np.float64)
        sizes = np.clip(4.0 + 32.0 * obs[:, 3] / max(float(np.max(obs[:, 3])), 1e-12), 4.0, 36.0)
        ordered.scatter(obs[:, 1] / frame.scale, np.degrees(obs[:, 2]), c=obs[:, 0], cmap="turbo", s=sizes, alpha=0.72, rasterized=True)
    ordered.set_xlim(-12.1, 12.1)
    ordered.set_ylim(-92.0, 92.0)
    ordered.set_xlabel("ordered transverse q / s")
    ordered.set_ylabel("projective normal phase (deg)")
    ordered.set_title("SOURCE-OBSERVED ORDERED PROJECTIVE NORMALS", color="white")

    if decision is None or not visual or visual.get("kind") != "opgt":
        cell.text(0.5, 0.56, "PRE-INVOKE", ha="center", va="center", color="#78dce8", fontsize=25, transform=cell.transAxes)
        cell.text(0.5, 0.44, "no detector output rendered", ha="center", va="center", color="#b8c7d1", fontsize=13, transform=cell.transAxes)
        path.text(0.5, 0.56, "MANUAL CHECK", ha="center", va="center", color="#ffd866", fontsize=24, transform=path.transAxes)
        path.text(0.5, 0.42, "continuity · tails · axial stationarity\nnonmonotone reversal · nuisance geometry", ha="center", va="center", color="#b8c7d1", fontsize=12, transform=path.transAxes)
        cell.set_xticks([]); cell.set_yticks([]); path.set_xticks([]); path.set_yticks([])
    else:
        weights = np.asarray(visual["cell_weights"], dtype=np.float64)
        image = cell.imshow(np.log1p(weights), origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
        fig.colorbar(image, ax=cell, fraction=0.046, pad=0.03, label="log(1 + fixed mass)")
        cell.set_xlabel("transverse strip")
        cell.set_ylabel("axial slab")
        cell.set_title("8×64 ORDERED MASS SUPPORT", color="white")
        strips = np.arange(64)
        phases = np.asarray([np.nan if value is None else np.degrees(value) for value in visual["free_phases"]])
        path.plot(strips, phases, "o-", color="#5ccfe6", lw=1.0, ms=3, label="free strip phase")
        fitted_strips = np.asarray(visual["phase_path_strips"], dtype=int)
        fitted = -90.0 + np.asarray(visual["phase_path"], dtype=float) * 180.0 / 512.0
        path.plot(fitted_strips, fitted, "o-", color="#ffcc66", lw=2.0, ms=3.5, label=f"fitted {visual['phase_direction']}")
        for j in visual["left_tail"]:
            path.axvspan(j - 0.5, j + 0.5, color="#4de3a7", alpha=0.12)
        for j in visual["right_tail"]:
            path.axvspan(j - 0.5, j + 0.5, color="#d77bff", alpha=0.12)
        for j in visual["transition_strips"]:
            path.axvspan(j - 0.5, j + 0.5, color="#ff6b6b", alpha=0.15)
        path.set_xlim(-0.5, 63.5)
        path.set_xlabel("transverse strip")
        path.set_ylabel("projective phase (deg)")
        path.set_title(f"FREE/FITTED PATH · {decision['reason']}", color="white")
        path.legend(loc="best", facecolor="#132431", labelcolor="white", fontsize=9)

    stage = "POST-INVOKE" if decision is not None else "PRE-INVOKE"
    result_text = "" if decision is None else f" · {decision['reason']} · {decision['action']}"
    fig.suptitle(
        f"{fixture_id} · {stage}{result_text}\nvertices={len(points):,} triangles={len(faces):,} canonical={len(mesh.triangles):,}",
        color="white",
        fontsize=13,
        y=0.985,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=256, facecolor=fig.get_facecolor(), pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return {
        "fixture_id": fixture_id,
        "path": str(destination.relative_to(ROOT)),
        "sha256": sha(destination),
        "width": 4096,
        "height": 4096,
        "vertices": len(points),
        "triangles": len(faces),
        "canonical_triangles": len(mesh.triangles),
        "stage": "post" if decision is not None else "pre",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-pre", action="store_true")
    parser.add_argument("--fixture")
    args = parser.parse_args()
    manifest = load(OUT / "fixture_manifest.json")
    fixture_ids = manifest["ordered_fixture_ids"] if args.all_pre else [args.fixture]
    rows = [render(fixture_id, PRE / f"{fixture_id}_pre_4096.png") for fixture_id in fixture_ids]
    value = {
        "schema_version": 1,
        "kind": "v34_opgt_preinvoke_visual_manifest",
        "renderer_sha256": sha(Path(__file__)),
        "rows": rows,
    }
    path = OUT / "preinvoke_visual_manifest.json"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"rendered": len(rows), "manifest": str(path), "sha256": sha(path)}, indent=2))


if __name__ == "__main__":
    main()
