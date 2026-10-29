#!/usr/bin/env python3
"""Render mandatory native-resolution, truth-free V55 MSTLR audit suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_mesh_section_turn_locus_reselection_v55 import (
    MeshSectionIndex,
    canonical_bytes,
    canonical_line,
    measure_candidate,
)


EXACT_CASES = frozenset(("S005", "S006", "S007", "S013", "S026", "S027"))
CONTROL_CASES = frozenset(("S011", "S016", "S024", "S029"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload, newline=True))


def _equal_axes(ax, points: np.ndarray) -> None:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.52 * float(np.max(upper - lower))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_lines(ax, rows, color: str, *, linewidth: float, linestyle: str = "-") -> None:
    for row in rows:
        line = canonical_line(row, 0)
        xyz = np.vstack((line.start, line.end))
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=linewidth, linestyle=linestyle)


def _render_overview(vertices: np.ndarray, output: Path) -> None:
    figure = plt.figure(figsize=(12, 9), dpi=200)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.16, c="#b8c0cc", alpha=0.78, linewidths=0, rasterized=True)
    _equal_axes(axis, vertices)
    axis.view_init(elev=28, azim=-52)
    axis.set_title(f"Native-resolution source mesh vertices: {len(vertices):,}")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, facecolor="#0b1020")
    plt.close(figure)


def _render_six_view(vertices: np.ndarray, v52_rows, v55_rows, output: Path) -> None:
    views = ((20, -45), (20, 45), (20, 135), (20, 225), (88, -90), (0, 0))
    figure = plt.figure(figsize=(15, 10), dpi=200)
    for index, (elevation, azimuth) in enumerate(views, 1):
        axis = figure.add_subplot(2, 3, index, projection="3d")
        axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.08, c="#97a3b6", alpha=0.48, linewidths=0, rasterized=True)
        _plot_lines(axis, v52_rows, "#00d4ff", linewidth=2.0, linestyle="--")
        _plot_lines(axis, v55_rows, "#ffb000", linewidth=2.4)
        _equal_axes(axis, vertices)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(f"elev {elevation}°, az {azimuth}°", color="white", fontsize=9)
    figure.suptitle("V52 (cyan dashed) / V55 (amber solid) — all source vertices", color="white")
    figure.patch.set_facecolor("#0b1020")
    figure.tight_layout()
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)


def _section_coordinates(station, direction: np.ndarray):
    assert station.profile is not None
    profile = station.profile
    tangent = profile[-1] - profile[0]
    tangent -= np.dot(tangent, direction) * direction
    tangent /= np.linalg.norm(tangent)
    vertical = np.cross(direction, tangent)
    vertical /= np.linalg.norm(vertical)
    if tuple(vertical.tolist()) < tuple((-vertical).tolist()):
        vertical = -vertical
    relative = profile - station.q
    x = relative @ tangent
    y = relative @ vertical
    return tangent, vertical, x, y


def _render_sections(measurement, output: Path, source_kind: str) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=200)
    tau = float(measurement.parameters["center_tolerance"])
    for station, axis in zip(measurement.stations, axes.flat):
        axis.set_facecolor("#0b1020")
        axis.axvspan(-tau, tau, color="#1fbf75", alpha=0.12, label="±τc")
        axis.axvline(0.0, color="#00d4ff", linewidth=1.5, linestyle="--", label="candidate x=0")
        if station.profile is None:
            axis.text(0.5, 0.5, station.reason, transform=axis.transAxes, ha="center", va="center", color="#ff6577")
        else:
            tangent, vertical, x, y = _section_coordinates(station, measurement.line.direction)
            axis.plot(x, y, color="#d7dee9", linewidth=1.35, marker=".", markersize=2.0, label="source profile")
            if station.turn_width_interval is not None:
                axis.axvspan(station.turn_width_interval[0], station.turn_width_interval[1], color="#bb86fc", alpha=0.17, label="80% turn interval")
            if station.turn_center is not None:
                relative = station.turn_center - station.q
                center_x = float(np.dot(relative, tangent))
                center_y = float(np.dot(relative, vertical))
                axis.scatter([center_x], [center_y], s=38, c="#ff375f", edgecolors="white", linewidths=0.6, zorder=6, label="turn center")
            for flank, color in ((station.left_flank_line, "#59e391"), (station.right_flank_line, "#ffd166")):
                if flank is None:
                    continue
                center, direction = flank
                extent = float(measurement.parameters["profile_half_span"])
                endpoints = np.vstack((center - extent * direction, center + extent * direction)) - station.q
                axis.plot(endpoints @ tangent, endpoints @ vertical, color=color, linewidth=1.0)
        rho_text = "—" if station.rho is None else f"{station.rho:.2f} mm"
        axis.set_title(f"station {station.station_index + 1} · {station.state} · ρ {rho_text}", color="white", fontsize=8)
        axis.tick_params(colors="#aab3c2", labelsize=7)
        axis.grid(color="#263149", alpha=0.45)
        axis.set_aspect("equal", adjustable="datalim")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    figure.suptitle(
        f"{source_kind} ordinal {measurement.line.ordinal} · exact detector stations · geometry {measurement.line.geometry_sha256[:12]}",
        color="white",
    )
    figure.patch.set_facecolor("#0b1020")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(args.output_root)
    manifest = json.loads((args.execution_root / "execution_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SEALED_PRIMARY_REPLAY" or manifest.get("truth_opened") is not False:
        raise ValueError("execution is not ready for its blind visual audit")
    render_records = []
    mandatory_cases = set(EXACT_CASES | CONTROL_CASES)
    changed_by_case: dict[str, list[tuple[str, int]]] = {}
    for case_id in (f"S{index:03d}" for index in range(1, 31)):
        result = json.loads((args.execution_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        audit = result["mesh_section_turn_locus_reselection_audit"]
        changed: list[tuple[str, int]] = []
        for row in audit["v52_line_summaries"]:
            if str(row["decision"]).startswith("PRUNE_"):
                changed.append(("pruned_v52", int(row["ordinal"])))
        for row in audit["accepted_addition_summaries"]:
            changed.append(("addition", int(row["ordinal"])))
        if changed:
            mandatory_cases.add(case_id)
            changed_by_case[case_id] = changed

    for case_id in sorted(mandatory_cases):
        scan_path = args.scan_root / f"{case_id}.ply"
        index = MeshSectionIndex.from_path(scan_path)
        v52 = json.loads((args.v52_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        result = json.loads((args.execution_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        pool = json.loads((args.pool_root / case_id / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        overview = case_root / "raw_source_overview.png"
        comparison = case_root / "six_view_v52_v55.png"
        _render_overview(index.vertices, overview)
        _render_six_view(index.vertices, v52["predictions"], result["predictions"], comparison)
        files = [overview, comparison]
        changed_records = []
        for source_kind, ordinal in changed_by_case.get(case_id, []):
            if source_kind == "pruned_v52":
                row = v52["predictions"][ordinal]
            else:
                row = pool["candidates"][ordinal]
            measurement = measure_candidate(index, canonical_line(row, ordinal))
            line_root = case_root / f"changed_line_{source_kind}_{ordinal:04d}"
            line_root.mkdir(parents=True, exist_ok=True)
            section_path = line_root / "axis_normal_sections.png"
            _render_sections(measurement, section_path, source_kind)
            files.append(section_path)
            changed_records.append(
                {
                    "source_kind": source_kind,
                    "ordinal": ordinal,
                    "geometry_sha256": measurement.line.geometry_sha256,
                    "centered_station_mask": measurement.centered_mask,
                    "offset_station_mask": measurement.offset_mask,
                    "section_render": str(section_path),
                    "section_render_sha256": _sha(section_path),
                }
            )
        render_records.append(
            {
                "case_id": case_id,
                "source_vertex_count": len(index.vertices),
                "source_triangle_count": len(index.triangles),
                "source_point_serialization": False,
                "overview": str(overview),
                "overview_sha256": _sha(overview),
                "comparison": str(comparison),
                "comparison_sha256": _sha(comparison),
                "changed_lines": changed_records,
                "all_files": [{"path": str(path), "sha256": _sha(path)} for path in files],
            }
        )
    output_manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_source_only_visual_manifest",
        "truth_opened": False,
        "native_source_vertex_use": True,
        "required_exact_cases": sorted(EXACT_CASES),
        "required_control_cases": sorted(CONTROL_CASES),
        "changed_cases": sorted(changed_by_case),
        "rendered_cases": sorted(mandatory_cases),
        "records": render_records,
    }
    _write_new(args.output_root / "visual_manifest.json", output_manifest)
    print(json.dumps({"rendered_case_count": len(render_records), "changed_case_count": len(changed_by_case)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
