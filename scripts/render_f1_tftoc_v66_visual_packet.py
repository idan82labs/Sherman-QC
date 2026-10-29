#!/usr/bin/env python3
"""Render the locked V66 native-resolution pretruth visual packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_transverse_face_turn_ownership_certificate_v66 as v66
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    load_identity,
    opaque_source_id,
    sha256_file,
    source_paths,
)


DPI = 128
FIGURE_INCHES = 32
INK = "#17212b"
MESH = "#5b6672"
V52 = "#168aad"
V66 = "#8a3ffc"
SUPPRESSED = "#e03131"
REPRESENTATIVE = "#2b8a3e"
FIRST = "#00a6fb"
SECOND = "#ff922b"
EVENT = "#d6336c"


def _project(points: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points @ horizontal, points @ vertical


def _axis_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    choices = np.eye(3)
    seed = choices[int(np.argmin(np.abs(choices @ direction)))]
    horizontal = np.cross(direction, seed); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(direction, horizontal); vertical /= np.linalg.norm(vertical)
    return horizontal, vertical


def _draw_points(ax: Any, vertices: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, *, size: float = 0.08) -> None:
    x, y = _project(vertices, horizontal, vertical)
    ax.scatter(x, y, s=size, c=MESH, alpha=0.48, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_lines(ax: Any, rows: Sequence[Mapping[str, Any]], horizontal: np.ndarray, vertical: np.ndarray, color: str, *, width: float = 3.0, label: bool = False) -> None:
    for row in rows:
        line = np.asarray((row["line_start"], row["line_end"]), dtype=np.float64)
        x, y = _project(line, horizontal, vertical)
        ax.plot(x, y, color=color, linewidth=width, solid_capstyle="round")
        ax.scatter(x, y, s=16, c=color, edgecolors="white", linewidths=0.5)
        if label:
            ax.text(float(np.mean(x)), float(np.mean(y)), str(row.get("route_hash") or row.get("bend_id"))[:8], color=color, fontsize=8, weight="bold")


def _raw_page(vertices: np.ndarray, output: Path, opaque_id: str) -> None:
    views = (
        (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)), "XY"),
        (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "XZ"),
        (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "YZ"),
        (np.asarray((-1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)), "reverse XY"),
        (np.asarray((-1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "reverse XZ"),
        (np.asarray((0.0, -1.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "reverse YZ"),
        (np.asarray((1.0, -1.0, 0.0)) / math.sqrt(2.0), np.asarray((1.0, 1.0, 2.0)) / math.sqrt(6.0), "fixed oblique"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_INCHES, FIGURE_INCHES), dpi=DPI, facecolor="#f5f7f9")
    for ax, (horizontal, vertical, title) in zip(axes.flat, views):
        _draw_points(ax, vertices, horizontal, vertical)
        ax.set_title(title, fontsize=20, weight="bold", color=INK)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{opaque_id} — complete bounded source mesh — pretruth", fontsize=30, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def _overlay_page(vertices: np.ndarray, identity: Mapping[str, Any], output: Path, opaque_id: str) -> None:
    v52_rows = identity["v52_final_predictions"]
    v66_rows = identity["v66_final_predictions"]
    v66_hashes = {row["route_hash"] for row in v66_rows}
    v52_hashes = {row["route_hash"] for row in v52_rows}
    removed = [row for row in v52_rows if row["route_hash"] not in v66_hashes]
    added = [row for row in v66_rows if row["route_hash"] not in v52_hashes]
    views = (
        (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)), "XY"),
        (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "XZ"),
        (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0)), "YZ"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_INCHES, FIGURE_INCHES), dpi=DPI, facecolor="#f5f7f9")
    for column, (horizontal, vertical, title) in enumerate(views):
        ax = axes[0, column]
        _draw_points(ax, vertices, horizontal, vertical)
        _draw_lines(ax, v52_rows, horizontal, vertical, V52, width=3.2, label=True)
        ax.set_title(f"V52 final — {title} — {len(v52_rows)} lines", fontsize=19, weight="bold", color=INK)
        ax = axes[1, column]
        _draw_points(ax, vertices, horizontal, vertical)
        _draw_lines(ax, v66_rows, horizontal, vertical, V66, width=3.2, label=True)
        _draw_lines(ax, removed, horizontal, vertical, SUPPRESSED, width=6.0, label=True)
        _draw_lines(ax, added, horizontal, vertical, REPRESENTATIVE, width=6.0, label=True)
        ax.set_title(f"V66 final/difference — {title} — {len(v66_rows)} lines; removed={len(removed)}", fontsize=19, weight="bold", color=INK)
    fig.suptitle(f"{opaque_id} — V52/V66 final outputs — evaluator truth sealed", fontsize=29, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def _component_xy(component: Any, horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(component.points)
    if component.closed:
        points = np.vstack((points, points[0]))
    return _project(points, horizontal, vertical)


def _point_at_component_arc(component: Any, coordinate: float) -> np.ndarray:
    coordinate = coordinate % component.length if component.closed else float(np.clip(coordinate, 0.0, component.length))
    arc = np.asarray(component.arc)
    index = min(len(component.points) - (1 if component.closed else 2), max(0, int(np.searchsorted(arc, coordinate, side="right") - 1)))
    first = component.points[index]
    second = component.points[(index + 1) % len(component.points)]
    fraction = (coordinate - arc[index]) / max(float(np.linalg.norm(second - first)), 1e-300)
    return first + fraction * (second - first)


def _pair_geometry(mesh: Any, rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> Any:
    by_hash = {str(row["route_hash"]): row for row in rows}
    first_hash, second_hash = record["route_hashes"]
    first = v66._candidate(by_hash[first_hash], 0)
    second = v66._candidate(by_hash[second_hash], 1)
    return v66._pair_geometry(mesh, first, second)


def _section_page(mesh: Any, rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any], output: Path, opaque_id: str) -> None:
    pair = _pair_geometry(mesh, rows, record)
    horizontal, vertical = _axis_basis(pair.axis)
    budget = v66._Budget()
    fig, axes = plt.subplots(3, 3, figsize=(FIGURE_INCHES, FIGURE_INCHES), dpi=DPI, facecolor="#f5f7f9")
    for station, ax in enumerate(axes.flat, start=1):
        coordinate = pair.overlap_min + station * pair.overlap / 10.0
        first_anchor = v66._point_on_axis(pair.first, pair.axis, coordinate)
        second_anchor = v66._point_on_axis(pair.second, pair.axis, coordinate)
        components, reason = v66._section_components(mesh, pair, coordinate, 0.5 * (first_anchor + second_anchor), budget)
        if not components:
            ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=18, color=SUPPRESSED)
            ax.axis("off")
            continue
        for component in components:
            x, y = _component_xy(component, horizontal, vertical)
            ax.plot(x, y, color=MESH, linewidth=2.2, alpha=0.85)
            events, _event_reason = v66._events(component, pair.axis, pair.h)
            for event in events:
                center = _point_at_component_arc(component, event.center)
                cx, cy = _project(center[None, :], horizontal, vertical)
                ax.scatter(cx, cy, s=100, c=EVENT, marker="*", edgecolors="white", linewidths=1.0, zorder=6)
        x1, y1 = _project(first_anchor[None, :], horizontal, vertical)
        x2, y2 = _project(second_anchor[None, :], horizontal, vertical)
        ax.scatter(x1, y1, s=90, c=FIRST, edgecolors="white", linewidths=1.0, zorder=7)
        ax.scatter(x2, y2, s=90, c=SECOND, edgecolors="white", linewidths=1.0, zorder=7)
        station_audit = record["stations"][station - 1]
        ax.set_title(f"station {station}: {station_audit['reason']}\nd1/h={station_audit['first_anchor_distance_over_h_e6']/1e6:.3f}; d2/h={station_audit['second_anchor_distance_over_h_e6']/1e6:.3f}", fontsize=15, weight="bold")
        ax.set_aspect("equal"); ax.axis("off")
    first_hash, second_hash = record["route_hashes"]
    fig.suptitle(f"{opaque_id} — exact transverse sections — {first_hash[:8]} / {second_hash[:8]} — {record['pair_class']}", fontsize=25, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def _incident_normals(mesh: Any, pair: Any) -> tuple[np.ndarray, np.ndarray]:
    coordinate = pair.overlap_min + 0.5 * pair.overlap
    first_anchor = v66._point_on_axis(pair.first, pair.axis, coordinate)
    second_anchor = v66._point_on_axis(pair.second, pair.axis, coordinate)
    components, _reason = v66._section_components(mesh, pair, coordinate, 0.5 * (first_anchor + second_anchor), v66._Budget())
    if not components:
        horizontal, vertical = _axis_basis(pair.axis)
        return horizontal, vertical
    attachment, _reason = v66._closest_attachment(first_anchor, components, pair.h)
    component = components[attachment.component] if attachment is not None else components[0]
    events, _reason = v66._events(component, pair.axis, pair.h)
    center = events[0].center if events else attachment.arclength if attachment is not None else 0.5 * component.length
    index = min(len(component.normals) - 2, max(0, int(np.searchsorted(np.asarray(component.arc), center, side="right") - 1)))
    first = np.asarray(component.normals[index], dtype=np.float64)
    second = np.asarray(component.normals[min(index + 1, len(component.normals) - 1)], dtype=np.float64)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    return first / np.linalg.norm(first), second / np.linalg.norm(second)


def _close_page(vertices: np.ndarray, rows: Sequence[Mapping[str, Any]], mesh: Any, record: Mapping[str, Any], output: Path, opaque_id: str) -> None:
    pair = _pair_geometry(mesh, rows, record)
    midpoint = 0.25 * (pair.first.start + pair.first.end + pair.second.start + pair.second.end)
    delta = vertices - midpoint
    axial = delta @ pair.axis
    radial = np.linalg.norm(delta - axial[:, None] * pair.axis, axis=1)
    crop = vertices[(np.abs(axial) <= 0.3 * pair.overlap) & (radial <= 20.0 * pair.h)]
    if not len(crop):
        crop = vertices
    normal_first, normal_second = _incident_normals(mesh, pair)
    bisector = normal_first + normal_second
    if np.linalg.norm(bisector) <= 1e-12:
        bisector = normal_first - normal_second
    bisector /= np.linalg.norm(bisector)
    directions = ((pair.axis, "along candidate axis"), (normal_first, "normal to incident face 1"), (normal_second, "normal to incident face 2"), (bisector, "along face-normal bisector"))
    selected_rows = [row for row in rows if row["route_hash"] in record["route_hashes"]]
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_INCHES, FIGURE_INCHES), dpi=DPI, facecolor="#f5f7f9")
    for ax, (direction, title) in zip(axes.flat, directions):
        horizontal, vertical = _axis_basis(direction)
        _draw_points(ax, crop, horizontal, vertical, size=0.18)
        _draw_lines(ax, selected_rows, horizontal, vertical, V66, width=5.0, label=True)
        ax.set_title(title, fontsize=20, weight="bold", color=INK)
    fig.suptitle(f"{opaque_id} — positive-pair close audit — {record['route_hashes'][0][:8]} / {record['route_hashes'][1][:8]}", fontsize=26, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def render(execution_root: Path, output_root: Path) -> Mapping[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite visual packet: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads((execution_root / "summary.json").read_text(encoding="utf-8"))
    binding_by_opaque = {}
    for case_id, pool_name in CASE_BINDINGS:
        source_path, _pool, _v51 = source_paths(case_id, pool_name)
        binding_by_opaque[opaque_source_id(source_path)] = (case_id, pool_name)
    records = []
    for case_row in summary["cases"]:
        opaque_id = case_row["opaque_source_id"]
        case_id, pool_name = binding_by_opaque[opaque_id]
        vertices, faces, _pool, _loaded = load_identity(case_id, pool_name)
        mesh = v66.SourceMesh(vertices, faces)
        payload = json.loads((execution_root / case_row["result"]).read_text(encoding="utf-8"))
        identity = payload["identity"]
        case_root = output_root / opaque_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_path = case_root / "raw_complete_mesh_4096.png"
        overlay_path = case_root / "v52_v66_difference_4096.png"
        _raw_page(vertices, raw_path, opaque_id)
        _overlay_page(vertices, identity, overlay_path, opaque_id)
        artifacts = [raw_path, overlay_path]
        positive_records = [row for row in identity["tftoc_audit"].get("pair_records") or () if row["pair_class"] == "POSITIVE"]
        for index, pair_record in enumerate(positive_records, start=1):
            section_path = case_root / f"positive_pair_{index:02d}_sections_4096.png"
            close_path = case_root / f"positive_pair_{index:02d}_close_4096.png"
            _section_page(mesh, identity["incoming_pre_nms_predictions"], pair_record, section_path, opaque_id)
            _close_page(vertices, identity["incoming_pre_nms_predictions"], mesh, pair_record, close_path, opaque_id)
            artifacts.extend((section_path, close_path))
        records.append({
            "opaque_source_id": opaque_id,
            "positive_pair_count": len(positive_records),
            "artifacts": [{"path": str(path.relative_to(output_root)), "sha256": sha256_file(path), "width": 4096, "height": 4096} for path in artifacts],
        })
    index = {
        "schema_version": 1,
        "profile": "V66_TFTOC_NATIVE_PRETRUTH_VISUAL_PACKET_V1",
        "pseudo_truth_opened": False,
        "source_vertex_decimation_used": False,
        "candidate_decimation_used": False,
        "minimum_resolution": [4096, 4096],
        "cases": records,
        "adjudication_complete": False,
    }
    atomic_json(output_root / "index.json", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    index = render(args.execution_root.resolve(), args.output_root.resolve())
    print(json.dumps(index, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
