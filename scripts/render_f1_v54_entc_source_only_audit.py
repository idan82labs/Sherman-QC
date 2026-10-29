#!/usr/bin/env python3
"""Render native-resolution source-only V54 ENTC visual audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_exclusive_normal_turn_coverage_v54 import _build_mesh_turn_evidence
from scripts.render_manual_bend_review import _stable_pca, _view_basis


DPI = 220
FIGSIZE = (24, 14)
VIEWS = (
    ("PC1 / PC2", np.eye(3)),
    ("PC1 / PC3", np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=float)),
    ("PC2 / PC3", np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float)),
    ("Oblique", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
    ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
    ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
)


def _limits(ax: plt.Axes, values: np.ndarray) -> None:
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    ax.set_xlim(low[0] - 0.04 * span[0], high[0] + 0.04 * span[0])
    ax.set_ylim(low[1] - 0.04 * span[1], high[1] + 0.04 * span[1])


def _line(row: dict, center: np.ndarray, pca: np.ndarray) -> np.ndarray:
    endpoints = np.asarray((row["line_start"], row["line_end"]), dtype=np.float64)
    return (endpoints - center) @ pca


def _multiview(
    points: np.ndarray,
    lines: list[np.ndarray],
    output: Path,
    title: str,
    *,
    raw_cloud: bool,
    transparent: bool,
    line_color: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    if not transparent:
        fig.patch.set_facecolor("#f8fafc")
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)
    for ax, (view_title, view) in zip(axes.flat, VIEWS):
        cloud = points @ view
        if transparent:
            ax.patch.set_alpha(0.0)
        else:
            ax.set_facecolor("#f8fafc")
        if raw_cloud:
            depth = cloud[:, 2]
            shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                c=plt.get_cmap("Greys")(0.22 + 0.55 * shade),
                s=0.55,
                alpha=0.58,
                linewidths=0,
                rasterized=True,
            )
        for index, line in enumerate(lines, start=1):
            shown = line @ view
            ax.plot(
                shown[:, 0],
                shown[:, 1],
                color="white" if not transparent else line_color,
                alpha=1.0 if not transparent else 0.0,
                linewidth=8.0,
                solid_capstyle="round",
                zorder=7,
            )
            ax.plot(
                shown[:, 0],
                shown[:, 1],
                color=line_color,
                linewidth=4.0,
                solid_capstyle="round",
                zorder=8,
            )
            midpoint = shown.mean(axis=0)
            ax.text(
                midpoint[0],
                midpoint[1],
                str(index),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=9,
                color="#111827" if not transparent else line_color,
                bbox={
                    "boxstyle": "circle,pad=0.18",
                    "facecolor": "white" if not transparent else "none",
                    "edgecolor": line_color,
                    "linewidth": 1.2,
                },
            )
        _limits(ax, cloud)
        if not transparent:
            ax.set_title(view_title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.012, right=0.988, top=0.925, bottom=0.015, wspace=0.018, hspace=0.07)
    fig.savefig(output, facecolor="none" if transparent else fig.get_facecolor(), transparent=transparent)
    plt.close(fig)


def _axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    seed = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(axis, first)
    return first, second


def _axis_stack(
    points: np.ndarray,
    turn_midpoints: np.ndarray,
    turn_mass: np.ndarray,
    addition: dict,
    output: Path,
    case_id: str,
    visual_index: int,
) -> None:
    geometry = addition["line_geometry"]
    line = np.asarray((geometry["line_start"], geometry["line_end"]), dtype=np.float64)
    delta = line[1] - line[0]
    length = float(np.linalg.norm(delta))
    axis = delta / max(length, 1e-12)
    u, v = _axis_basis(axis)
    origin = line[0]
    point_delta = points - origin
    axial = point_delta @ axis
    pu = point_delta @ u
    pv = point_delta @ v
    turn_delta = turn_midpoints - origin
    turn_axial = turn_delta @ axis
    turn_u = turn_delta @ u
    turn_v = turn_delta @ v
    positions = np.linspace(0.05 * length, 0.95 * length, 9)
    slab = max(0.025 * length, 1.5)
    radial = max(15.0, 0.12 * length)
    vmax = max(float(np.quantile(turn_mass[turn_mass > 0.0], 0.98)) if np.any(turn_mass > 0.0) else 1.0, 1e-12)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=180)
    fig.patch.set_facecolor("#07101f")
    for ax, station in zip(axes.flat, positions):
        point_mask = (np.abs(axial - station) <= slab) & (np.hypot(pu, pv) <= radial)
        turn_mask = (np.abs(turn_axial - station) <= slab) & (np.hypot(turn_u, turn_v) <= radial)
        ax.set_facecolor("#07101f")
        ax.scatter(pu[point_mask], pv[point_mask], s=1.0, color="#74879e", alpha=0.28, linewidths=0, rasterized=True)
        ax.scatter(
            turn_u[turn_mask],
            turn_v[turn_mask],
            c=turn_mass[turn_mask],
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            s=4.0,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter([0.0], [0.0], marker="+", s=180, linewidths=3.0, color="#24e5ff")
        ax.set_xlim(-radial, radial)
        ax.set_ylim(-radial, radial)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"station {station / length:.2f}L  •  source n={int(point_mask.sum())}", color="white", fontsize=12)
        ax.tick_params(colors="#6f88a9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#234162")
    fig.suptitle(
        f"{case_id} — ENTC addition {visual_index} axis-normal source-turn stack\ncyan + = proposed axis; inferno = source normal-turn mass",
        color="white",
        fontsize=21,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def _dashboard(
    evidence,
    projected: np.ndarray,
    additions: list[dict],
    output: Path,
    case_id: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(24, 14), dpi=DPI)
    fig.patch.set_facecolor("#07101f")
    for ax in axes.flat:
        ax.set_facecolor("#07101f")
    take = np.linspace(0, len(projected) - 1, min(len(projected), 200_000), dtype=np.int64)
    axes[0, 0].scatter(projected[take, 0], projected[take, 1], s=0.5, color="#91a4bb", alpha=0.45, linewidths=0, rasterized=True)
    axes[0, 0].set_title("Frozen raw-source overview", color="white", fontsize=15, fontweight="bold")
    turn = evidence.edge_h
    positive = turn > 0.0
    center, pca = _stable_pca(evidence.vertices)
    turn_projected = (evidence.edge_midpoints - center) @ pca
    vmax = max(float(np.quantile(turn[positive], 0.98)) if np.any(positive) else 1.0, 1e-12)
    axes[0, 1].scatter(
        turn_projected[:, 0],
        turn_projected[:, 1],
        c=turn,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        s=1.0,
        linewidths=0,
        rasterized=True,
    )
    axes[0, 1].set_title("Source normal-turn intensity (no source IDs)", color="white", fontsize=15, fontweight="bold")
    if additions:
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(additions)))
        x = np.arange(24)
        for index, (addition, color) in enumerate(zip(additions, colors), start=1):
            compatible = np.asarray(addition["bin_compatible_mass"], dtype=float)
            marginal = np.asarray(addition["bin_marginal_mass"], dtype=float)
            scale = max(float(np.max(compatible)), 1e-12)
            axes[1, 0].plot(x, compatible / scale, color=color, linewidth=2.4, label=f"addition {index} compatible")
            axes[1, 0].plot(x, marginal / scale, color=color, linewidth=1.4, linestyle="--", label=f"addition {index} marginal")
            coherent = int(addition["coherent_bin_mask"])
            novel = int(addition["novel_bin_mask"])
            axes[1, 1].scatter(x, [index] * 24, c=["#35f09a" if novel & (1 << k) else "#4e6682" if coherent & (1 << k) else "#17263b" for k in x], s=70, marker="s")
        axes[1, 0].set_ylim(-0.03, 1.08)
        axes[1, 0].legend(loc="upper right", fontsize=8, facecolor="#0e1a2b", labelcolor="white")
        axes[1, 1].set_ylim(0.4, len(additions) + 0.6)
        axes[1, 1].set_yticks(range(1, len(additions) + 1))
    else:
        axes[1, 0].text(0.5, 0.5, "NO ENTC ADDITIONS", transform=axes[1, 0].transAxes, ha="center", va="center", color="#8ba8cf", fontsize=26, fontweight="bold")
        axes[1, 1].text(0.5, 0.5, "V52 UNCHANGED", transform=axes[1, 1].transAxes, ha="center", va="center", color="#8ba8cf", fontsize=26, fontweight="bold")
    axes[1, 0].set_title("24-bin compatible / marginal ownership profiles", color="white", fontsize=15, fontweight="bold")
    axes[1, 1].set_title("24-bin state: green novel, slate coherent", color="white", fontsize=15, fontweight="bold")
    for ax in axes.flat:
        ax.tick_params(colors="#7f9ab9")
        for spine in ax.spines.values():
            spine.set_color("#294766")
    fig.suptitle(f"{case_id} — V54 ENTC native source-only audit", color="white", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def render(scan: Path, v52_path: Path, v54_path: Path, output_dir: Path, case_id: str) -> dict:
    v52 = json.loads(v52_path.read_text(encoding="utf-8"))
    v54 = json.loads(v54_path.read_text(encoding="utf-8"))
    prefix = list(v52.get("predictions") or ())
    if v54.get("predictions", [])[: len(prefix)] != prefix:
        raise ValueError("V52 prefix changed before visual audit")
    audit = dict(v54.get("exclusive_normal_turn_coverage_audit") or {})
    additions = list(audit.get("recovered_candidates") or ())
    mesh, evidence = _build_mesh_turn_evidence(scan, time.perf_counter())
    del mesh
    center, pca = _stable_pca(evidence.vertices)
    projected = (evidence.vertices - center) @ pca
    take = np.linspace(0, len(projected) - 1, min(len(projected), 250_000), dtype=np.int64)
    shown = projected[take]
    prefix_lines = [_line(row, center, pca) for row in prefix]
    addition_lines = [
        _line(row, center, pca) for row in v54.get("predictions", [])[len(prefix):]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_source_overview.png"
    v52_only_path = output_dir / "v52_only_overlay.png"
    addition_layer_path = output_dir / "entc_addition_layer.png"
    composite_path = output_dir / "v54_composite_overlay.png"
    dashboard_path = output_dir / "entc_dashboard.png"
    _multiview(shown, [], raw_path, f"{case_id} — frozen raw source", raw_cloud=True, transparent=False, line_color="#38bdf8")
    _multiview(shown, prefix_lines, v52_only_path, f"{case_id} — immutable V52 layer", raw_cloud=True, transparent=False, line_color="#38bdf8")
    _multiview(shown, addition_lines, addition_layer_path, "", raw_cloud=False, transparent=True, line_color="#f43f5e")
    base = Image.open(v52_only_path).convert("RGBA")
    layer = Image.open(addition_layer_path).convert("RGBA")
    if base.size != layer.size:
        raise ValueError("overlay dimensions differ")
    composite = Image.alpha_composite(base, layer)
    composite.save(composite_path)
    alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
    base_pixels = np.asarray(base)
    composite_pixels = np.asarray(composite)
    outside = alpha == 0
    unchanged_outside = bool(np.array_equal(base_pixels[outside], composite_pixels[outside]))
    if not unchanged_outside:
        raise ValueError("V52-only pixels changed outside ENTC layer")
    _dashboard(evidence, projected, additions, dashboard_path, case_id)
    closeups = []
    for index, addition in enumerate(additions, start=1):
        if len(addition.get("bin_turn_mass") or ()) != 24 or len(addition.get("bin_marginal_mass") or ()) != 24:
            raise ValueError("compact 24-bin record is malformed")
        path = output_dir / f"addition_{index:02d}_axis_normal.png"
        _axis_stack(
            evidence.vertices,
            evidence.edge_midpoints,
            evidence.edge_h,
            addition,
            path,
            case_id,
            index,
        )
        closeups.append(str(path))
    report = {
        "case_id": case_id,
        "native_resolution": True,
        "source_identifier_serialization": False,
        "v52_prefix_count": len(prefix),
        "entc_addition_count": len(additions),
        "displayed_addition_bin_summary_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "bin_turn_mass": row["bin_turn_mass"],
                        "bin_compatible_mass": row["bin_compatible_mass"],
                        "bin_marginal_mass": row["bin_marginal_mass"],
                        "coherent_bin_mask": row["coherent_bin_mask"],
                        "novel_bin_mask": row["novel_bin_mask"],
                    }
                    for row in additions
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "v52_pixels_unchanged_outside_entc_layer": unchanged_outside,
        "raw_source_overview": str(raw_path),
        "v52_only_overlay": str(v52_only_path),
        "entc_addition_layer": str(addition_layer_path),
        "v54_composite_overlay": str(composite_path),
        "entc_dashboard": str(dashboard_path),
        "axis_normal_closeups": closeups,
    }
    (output_dir / "visual_artifacts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--v52", type=Path, required=True)
    parser.add_argument("--v54", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = render(
        args.scan.resolve(),
        args.v52.resolve(),
        args.v54.resolve(),
        args.output_dir.resolve(),
        args.case_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
