#!/usr/bin/env python3
"""Deterministic native-resolution V30 AMFC fixture renderer.

Every point or triangle is drawn.  The PNG is encoded by Pillow without text
metadata so exact replays can be compared byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image


W = 4096
H = 4096
DPI = 256
BACKGROUND = "#071018"
SURFACE = "#8ba9ba"
TRIANGLE_FACE = "#4e6f82"
CANDIDATE = "#ffd166"
STATION = "#f4a261"
TAIL0 = "#2ec4b6"
TAIL1 = "#ff5d8f"
GRAPH = "#55758a"
PATH = "#fff275"


def _save_figure(fig: Figure, path: Path) -> None:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    if rgba.shape[1] != W or rgba.shape[0] != H:
        raise RuntimeError(f"native render drift: {rgba.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", compress_level=9, optimize=False)
    with Image.open(path) as image:
        if image.size != (W, H):
            raise RuntimeError("render is not 4096x4096")


def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    padding = 0.04 * float(span.max())
    return lower - padding, upper + padding


def _set_axes(ax: Any, lower: np.ndarray, upper: np.ndarray, title: str, view: tuple[float, float]) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect(tuple(np.maximum(upper - lower, 1.0)))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, color="#ecf4f8", fontsize=12, pad=8)
    ax.tick_params(colors="#78909c", labelsize=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#29404d")


def _draw_source(ax: Any, points: np.ndarray, triangles: np.ndarray) -> None:
    if len(triangles):
        faces = points[triangles]
        collection = Poly3DCollection(
            faces,
            facecolor=TRIANGLE_FACE,
            edgecolor=SURFACE,
            linewidth=0.08,
            alpha=0.72,
            rasterized=True,
        )
        ax.add_collection3d(collection)
    else:
        marker = 0.45 if len(points) < 50_000 else 0.08
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=marker, c=SURFACE, alpha=0.72, depthshade=False, rasterized=True)


def _draw_candidates(ax: Any, candidates: Iterable[dict[str, Any]], *, stations: bool) -> None:
    for candidate in candidates:
        start = np.asarray(candidate["line_start"], dtype=np.float64)
        end = np.asarray(candidate["line_end"], dtype=np.float64)
        line = np.vstack((start, end))
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=CANDIDATE, linewidth=1.2, alpha=0.95)
        if stations:
            center = np.asarray(candidate["line_center"], dtype=np.float64)
            axis = np.asarray(candidate["line_axis"], dtype=np.float64)
            lower, upper = (float(value) for value in candidate["support_interval"])
            positions = [lower + 8.0 + index * (upper - lower - 16.0) / 8.0 for index in range(9)]
            values = np.asarray([center + position * axis for position in positions])
            ax.scatter(values[:, 0], values[:, 1], values[:, 2], s=5.0, c=STATION, depthshade=False)


def _representative_bounds(
    points: np.ndarray,
    fixture_id: str,
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate = candidates[0]
    center = np.asarray(candidate["line_center"], dtype=np.float64)
    axis = np.asarray(candidate["line_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    relative = points - center
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[:, None] * axis[None, :], axis=1)
    radius = 40.0 * float(1.0)
    representative = points[(np.abs(axial) <= 8.0) & (radial <= radius)]
    if fixture_id == "SF10_RESOURCE_STRESS":
        representative = representative[(representative[:, 0] < 40.0) & (representative[:, 1] < 40.0) & (representative[:, 2] < 40.0)]
    if len(representative) == 0:
        representative = points
    return _bounds(representative)


def _figure(fixture_id: str, subtitle: str) -> tuple[Figure, list[Any]]:
    fig = Figure(figsize=(16.0, 16.0), dpi=DPI, facecolor=BACKGROUND)
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    fig.suptitle(f"{fixture_id} — {subtitle}", color="#f7fbfd", fontsize=18, y=0.992)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.965, wspace=0.02, hspace=0.06)
    return fig, axes


def render_raw_fixture(payload: dict[str, Any], path: Path) -> None:
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    triangles = np.asarray(payload["source"]["triangles"], dtype=np.int64)
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    fixture_id = str(payload["fixture_id"])
    fig, axes = _figure(fixture_id, "GATE 1 RAW GEOMETRY — AMFC NOT INVOKED")
    overview_lower, overview_upper = _bounds(points)
    detail_lower, detail_upper = _representative_bounds(points, fixture_id, payload["candidate_records"])
    cross_section_audits = {"SF05_DISCONNECTED_MITER_SEAM", "SF08_NONMONOTONE_WELD_BEAD"}
    detail_view = (0.0, -90.0) if fixture_id in cross_section_audits else (18.0, 25.0)
    views = ((25.0, -55.0), (90.0, -90.0), (0.0, -90.0), detail_view)
    if fixture_id == "SF05_DISCONNECTED_MITER_SEAM":
        detail_title = "cross-section gap audit (candidate hidden)"
    elif fixture_id == "SF08_NONMONOTONE_WELD_BEAD":
        detail_title = "cross-section back-turn audit (candidate hidden)"
    else:
        detail_title = "representative physical detail"
    titles = ("all geometry / isometric", "all geometry / axial", "all geometry / side", detail_title)
    for index, (ax, view, title) in enumerate(zip(axes, views, titles)):
        _draw_source(ax, points, triangles)
        if not (fixture_id in cross_section_audits and index == 3):
            _draw_candidates(ax, payload["candidate_records"], stations=True)
        lower, upper = (detail_lower, detail_upper) if index == 3 else (overview_lower, overview_upper)
        _set_axes(ax, lower, upper, title, view)
    _save_figure(fig, path)


def render_diagnostic_fixture(payload: dict[str, Any], result: dict[str, Any], path: Path, subtitle: str) -> None:
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    triangles = np.asarray(payload["source"]["triangles"], dtype=np.int64)
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    fixture_id = str(payload["fixture_id"])
    fig, axes = _figure(fixture_id, subtitle)
    overview_lower, overview_upper = _bounds(points)
    detail_lower, detail_upper = _representative_bounds(points, fixture_id, payload["candidate_records"])
    views = ((25.0, -55.0), (90.0, -90.0), (0.0, -90.0), (18.0, 25.0))
    titles = ("all source + certificates", "axial certificate coverage", "side / physical crossing", "representative certificate detail")
    visual_rows = result.get("_ephemeral", {}).get("candidates", [])
    for index, (ax, view, title) in enumerate(zip(axes, views, titles)):
        _draw_source(ax, points, triangles)
        _draw_candidates(ax, payload["candidate_records"], stations=True)
        graph_segments: list[np.ndarray] = []
        accepted_paths: list[np.ndarray] = []
        tail0_rows: list[np.ndarray] = []
        tail1_rows: list[np.ndarray] = []
        for candidate_row in visual_rows:
            for station in candidate_row.get("stations", []):
                if not station:
                    continue
                graph_segments.extend(np.asarray(value, dtype=np.float64) for value in station.get("graph_edges", []))
                accepted_paths.extend(np.asarray(value, dtype=np.float64) for value in station.get("accepted_paths", []))
                if len(station.get("tail0", [])):
                    tail0_rows.append(np.asarray(station["tail0"], dtype=np.float64))
                if len(station.get("tail1", [])):
                    tail1_rows.append(np.asarray(station["tail1"], dtype=np.float64))
        if graph_segments:
            ax.add_collection3d(Line3DCollection(graph_segments, colors=GRAPH, linewidths=0.10, alpha=0.16, rasterized=True))
        for collection, color in ((tail0_rows, TAIL0), (tail1_rows, TAIL1)):
            if collection:
                values = np.vstack(collection)
                ax.scatter(values[:, 0], values[:, 1], values[:, 2], s=0.35, c=color, alpha=0.45, depthshade=False, rasterized=True)
        for accepted in accepted_paths:
            ax.plot(accepted[:, 0], accepted[:, 1], accepted[:, 2], color=PATH, linewidth=1.6, alpha=0.95)
        lower, upper = (detail_lower, detail_upper) if index == 3 else (overview_lower, overview_upper)
        _set_axes(ax, lower, upper, title, view)
    decisions = result.get("decisions", [])
    label = ", ".join(f"{row['candidate_key']}:{row['action']}:{row.get('reason')}" for row in decisions[:8])
    if len(decisions) > 8:
        label += f" … +{len(decisions) - 8}"
    fig.text(0.02, 0.004, label, color="#d6e6ee", fontsize=8)
    _save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.fixture.read_text(encoding="utf-8"))
    render_raw_fixture(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
