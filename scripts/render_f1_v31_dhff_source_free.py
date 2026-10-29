#!/usr/bin/env python3
"""Native-resolution visual evidence renderer for frozen V31 DHFF fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


INK = "#e8edf5"
MUTED = "#8d9bad"
BG = "#070b12"
A_COLOR = "#27d3b2"
B_COLOR = "#ffb454"
PASS_COLOR = "#53e081"
FAIL_COLOR = "#ff5d6c"


def _style_axis(ax: Any, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=13, pad=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#233044")
    ax.set_aspect("equal", adjustable="datalim")


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["candidate_records"][0]


def _frame(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(candidate["center_xyz"], dtype=np.float64)
    axis = np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    n_a = np.asarray(candidate["parent_A"]["unit_normal_xyz"], dtype=np.float64)
    transverse = np.cross(axis, n_a)
    transverse /= np.linalg.norm(transverse)
    depth = np.cross(transverse, axis)
    depth /= np.linalg.norm(depth)
    return center, axis, transverse, depth


def _project(points: np.ndarray, center: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = points - center
    return rel @ horizontal, rel @ vertical


def _scatter(ax: Any, points: np.ndarray, parents: np.ndarray, center: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, size: float = 0.35) -> None:
    x, y = _project(points, center, horizontal, vertical)
    colors = np.where(parents == 1, A_COLOR, np.where(parents == 2, B_COLOR, "#a9b4c2"))
    ax.scatter(x, y, s=size, c=colors, alpha=0.72, linewidths=0, rasterized=True)


def render_raw_native_multiview(payload: dict[str, Any], output_path: Path) -> None:
    source = payload["source"]
    points = np.asarray(source["positions"], dtype=np.float64)
    parents = np.asarray(source["parent_fragment_ids"], dtype=np.uint32)
    candidate = _candidate(payload)
    center, axis, transverse, depth = _frame(candidate)
    views = [
        (0.72 * transverse + 0.31 * axis, 0.85 * depth - 0.20 * axis, "Perspective / material topology"),
        (transverse, depth, "Axis view / fold cross-section"),
        (axis, depth, "Transverse A / axial continuity"),
        (axis, transverse, "Transverse B / tail breadth"),
        (-transverse, depth, "Opposite-skin visibility"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(32, 32), dpi=128, facecolor=BG)
    for ax, (horizontal, vertical, title) in zip(axes.flat, views):
        horizontal = horizontal / np.linalg.norm(horizontal)
        vertical = vertical - np.dot(vertical, horizontal) * horizontal
        vertical /= np.linalg.norm(vertical)
        _style_axis(ax, title)
        _scatter(ax, points, parents, center, horizontal, vertical)
        start = np.asarray(candidate["render_line_start"], dtype=np.float64)
        end = np.asarray(candidate["render_line_end"], dtype=np.float64)
        lx, ly = _project(np.vstack((start, end)), center, horizontal, vertical)
        ax.plot(lx, ly, color="#88aaff", linewidth=1.4, alpha=0.85)
    ax = axes.flat[-1]
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.04, 0.94, payload["fixture_id"], color=INK, fontsize=23, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.04, 0.84, payload["construction"], color=MUTED, fontsize=15, va="top", wrap=True, transform=ax.transAxes)
    ax.text(0.04, 0.57, f"raw samples  {len(points):,}\nmesh edges  {len(source.get('mesh_adjacencies', [])):,}\nscale s  {candidate['local_scale_s']:.3f}\nturn α  {candidate['signed_turn_alpha']:.1f}°", color=INK, fontsize=17, linespacing=1.5, transform=ax.transAxes)
    ax.text(0.04, 0.25, "RAW / PRE-INVOCATION\nNo graph · no potential · no current · no certificate", color="#ffd166", fontsize=16, weight="bold", linespacing=1.5, transform=ax.transAxes)
    fig.suptitle("V31 DHFF · native raw multiview", color=INK, fontsize=28, weight="bold", y=0.995)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=128)
    plt.close(fig)


def render_raw_station_sections(payload: dict[str, Any], output_path: Path) -> None:
    source = payload["source"]
    points = np.asarray(source["positions"], dtype=np.float64)
    parents = np.asarray(source["parent_fragment_ids"], dtype=np.uint32)
    candidate = _candidate(payload)
    center, axis, transverse, depth = _frame(candidate)
    length = float(candidate["segment_length_L"])
    scale = float(candidate["local_scale_s"])
    fig, axes = plt.subplots(3, 4, figsize=(32, 32), dpi=128, facecolor=BG)
    for station in range(11):
        ax = axes.flat[station]
        station_center = center + (-0.40 + 0.08 * station) * length * axis
        axial = np.abs((points - station_center) @ axis) / scale
        local = points[axial <= 0.75]
        local_parents = parents[axial <= 0.75]
        _style_axis(ax, f"station {station:02d} · a={(-0.40 + 0.08 * station):+.2f}L · n={len(local)}")
        _scatter(ax, local, local_parents, station_center, transverse, depth, size=2.2)
        ax.axvline(0.0, color="#88aaff", linewidth=1.0, alpha=0.7)
        ax.axhline(0.0, color="#334155", linewidth=0.7, alpha=0.6)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.05, 0.93, payload["fixture_id"], color=INK, fontsize=22, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.77, "Eleven raw, non-overlapping station slabs.\nTail extent, gaps, opposite skin, nearby routes, and transition support are visible.", color=MUTED, fontsize=15, va="top", linespacing=1.5, wrap=True, transform=ax.transAxes)
    ax.text(0.05, 0.40, "RAW / PRE-INVOCATION\nCandidate axis is shown only as the blue origin line.\nNo DHFF certificate information is present.", color="#ffd166", fontsize=16, weight="bold", linespacing=1.5, transform=ax.transAxes)
    fig.suptitle("V31 DHFF · native raw station sections", color=INK, fontsize=28, weight="bold", y=0.995)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=128)
    plt.close(fig)


def render_raw_fixture(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    multiview = output_dir / "raw_native_multiview_4096.png"
    stations = output_dir / "raw_station_sections_4096.png"
    render_raw_native_multiview(payload, multiview)
    render_raw_station_sections(payload, stations)
    return multiview, stations


def _first_visual(result: dict[str, Any]) -> dict[str, Any]:
    visuals = result["_ephemeral"]["candidate_visuals"]
    return visuals[0] if visuals else {"stations": []}


def render_station_flux(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> None:
    positions = np.asarray(payload["source"]["positions"], dtype=np.float64)
    candidate = _candidate(payload)
    _center, _axis, transverse, depth = _frame(candidate)
    visual = _first_visual(result)
    stations = visual.get("stations", [])
    fig, axes = plt.subplots(3, 4, figsize=(32, 32), dpi=128, facecolor=BG)
    for station_index in range(11):
        ax = axes.flat[station_index]
        station = stations[station_index] if station_index < len(stations) else {"reason": result["decisions"][0]["reason"], "pass": False, "center": candidate["center_xyz"]}
        center = np.asarray(station.get("center", candidate["center_xyz"]), dtype=np.float64)
        indices = np.asarray(station.get("node_indices", []), dtype=np.int64)
        local = positions[indices] if len(indices) else np.empty((0, 3))
        x, y = _project(local, center, transverse, depth) if len(local) else (np.asarray([]), np.asarray([]))
        phi = np.asarray(station.get("phi", np.zeros(len(local))), dtype=np.float64)
        _style_axis(ax, f"station {station_index:02d} · {station.get('reason', 'NO_DATA')}")
        if len(local):
            ax.scatter(x, y, s=7.0, c=phi if len(phi) == len(local) else MUTED, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0, alpha=0.9, rasterized=True)
            segments = []
            widths = []
            currents = {(int(i), int(j)): float(current) for i, j, current in station.get("currents", [])}
            max_current = max(currents.values(), default=1.0)
            for i, j, _weight in station.get("edges", []):
                i, j = int(i), int(j)
                if i < len(local) and j < len(local):
                    segments.append(((x[i], y[i]), (x[j], y[j])))
                    widths.append(0.15 + 3.0 * currents.get((i, j), currents.get((j, i), 0.0)) / max_current)
            if segments:
                ax.add_collection(LineCollection(segments, linewidths=widths, colors="#c6f1ff", alpha=0.34))
            for seed_key, color in (("seeds_a", A_COLOR), ("seeds_b", B_COLOR)):
                seeds = np.asarray(station.get(seed_key, []), dtype=np.int64)
                seeds = seeds[seeds < len(local)]
                if len(seeds):
                    ax.scatter(x[seeds], y[seeds], s=22, facecolors="none", edgecolors=color, linewidths=1.2)
        ax.text(0.02, 0.02, "PASS" if station.get("pass") else "FAIL", color=PASS_COLOR if station.get("pass") else FAIL_COLOR, fontsize=12, weight="bold", transform=ax.transAxes)
    ax = axes.flat[-1]
    ax.axis("off")
    decision = result["decisions"][0]
    ax.text(0.05, 0.92, payload["fixture_id"], color=INK, fontsize=22, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.72, f"station mask  0x{int(decision['station_mask']):03x}\npassing  {decision['passing_stations']}/11\nlongest run  {decision['longest_run']}\naction  {decision['action']}\nreason  {decision['reason']}", color=INK, fontsize=17, linespacing=1.45, transform=ax.transAxes)
    fig.suptitle("V31 DHFF · harmonic station flux", color=INK, fontsize=28, weight="bold", y=0.995)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=128)
    plt.close(fig)


def render_phi_theta(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> None:
    visual = _first_visual(result)
    alpha = abs(float(_candidate(payload)["signed_turn_alpha"]))
    fig, axes = plt.subplots(3, 4, figsize=(32, 32), dpi=128, facecolor=BG)
    for station_index in range(11):
        ax = axes.flat[station_index]
        station = visual.get("stations", [])[station_index] if station_index < len(visual.get("stations", [])) else {}
        _style_axis(ax, f"station {station_index:02d} · {station.get('reason', 'NO_DATA')}")
        phi = np.asarray(station.get("phi", []), dtype=np.float64)
        theta = np.asarray(station.get("theta", []), dtype=np.float64)
        if len(phi) and len(phi) == len(theta):
            ax.scatter(phi, theta, s=7, c=phi, cmap="viridis", alpha=0.75, linewidths=0, rasterized=True)
        for k in range(1, 8):
            ax.axvspan(k / 8.0 - 1 / 16.0, k / 8.0 + 1 / 16.0, color="#4c6fff", alpha=0.035)
            ax.axhspan(k * alpha / 8.0 - alpha / 16.0, k * alpha / 8.0 + alpha / 16.0, color="#ffb454", alpha=0.025)
        medians = station.get("medians", [])
        if len(medians) == 9:
            ax.plot(np.arange(9) / 8.0, medians, color="#ffffff", linewidth=2.0, marker="o", markersize=3)
        ax.axhline(-10.0, color=FAIL_COLOR, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(alpha + 10.0, color=FAIL_COLOR, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-20.0, alpha + 20.0)
        ax.set_xlabel("potential ϕ", color=MUTED)
        ax.set_ylabel("normal angle θ (deg)", color=MUTED)
    ax = axes.flat[-1]
    ax.axis("off")
    decision = result["decisions"][0]
    ax.text(0.05, 0.92, payload["fixture_id"], color=INK, fontsize=22, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.72, f"target turn  {alpha:.1f}°\nmedian backtrack  {decision['median_normal_backtrack']:.4f}°\nmedian off-arc  {decision['median_off_arc_fraction']:.6f}\nprofile  {decision['profile_hash'][:16]}…", color=INK, fontsize=16, linespacing=1.5, transform=ax.transAxes)
    fig.suptitle("V31 DHFF · current-weighted potential / normal certificate", color=INK, fontsize=28, weight="bold", y=0.995)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=128)
    plt.close(fig)


def render_final_overlay(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> None:
    points = np.asarray(payload["source"]["positions"], dtype=np.float64)
    parents = np.asarray(payload["source"]["parent_fragment_ids"], dtype=np.uint32)
    candidates = payload["candidate_records"]
    center, axis, transverse, depth = _frame(candidates[0])
    fig, axes = plt.subplots(2, 2, figsize=(32, 32), dpi=128, facecolor=BG)
    views = [(transverse, depth, "Axis view"), (axis, depth, "Axial / depth"), (axis, transverse, "Axial / breadth"), (0.75 * transverse + 0.25 * axis, depth, "Perspective")]
    final_ids = {int(line["candidate_id"]) for line in result["final_lines"]}
    suppressed = set(int(value) for value in result["_ephemeral"].get("nms_suppressed", []))
    for ax, (horizontal, vertical, title) in zip(axes.flat, views):
        horizontal = horizontal / np.linalg.norm(horizontal)
        vertical = vertical - np.dot(vertical, horizontal) * horizontal
        vertical /= np.linalg.norm(vertical)
        _style_axis(ax, title)
        _scatter(ax, points, parents, center, horizontal, vertical, size=0.55)
        for candidate in candidates:
            line = np.vstack((candidate["render_line_start"], candidate["render_line_end"]))
            lx, ly = _project(line, center, horizontal, vertical)
            cid = int(candidate["candidate_id"])
            if cid in final_ids:
                color, width, style = PASS_COLOR, 5.0, "-"
            elif cid in suppressed:
                color, width, style = "#ff9f43", 2.5, "--"
            else:
                color, width, style = FAIL_COLOR, 1.5, ":"
            ax.plot(lx, ly, color=color, linewidth=width, linestyle=style, alpha=0.95)
    decision = result["decisions"][0]
    fig.suptitle(f"V31 DHFF · final native overlay · {payload['fixture_id']} · {decision['action']} · {decision['reason']}", color=INK, fontsize=24, weight="bold", y=0.995)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=128)
    plt.close(fig)


def render_post_fixture(payload: dict[str, Any], result: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    flux = output_dir / "station_flux_4096.png"
    certificate = output_dir / "phi_theta_certificate_4096.png"
    overlay = output_dir / "final_native_overlay_4096.png"
    render_station_flux(payload, result, flux)
    render_phi_theta(payload, result, certificate)
    render_final_overlay(payload, result, overlay)
    return flux, certificate, overlay
