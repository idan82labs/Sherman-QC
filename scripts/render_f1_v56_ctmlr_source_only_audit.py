#!/usr/bin/env python3
"""Render the mandatory post-freeze, source-only V56 CTMLR visual audit.

This program is deliberately downstream of the immutable V56 predictions.  It
recomputes section evidence only for display and never writes, ranks, or
changes a prediction.  Every mesh vertex is drawn in each overview; no source
vertex or point array is serialized into an audit JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_canonical_turn_median_line_reconstruction_v56 import (
    PROFILE,
    RESULT_KIND,
    STATION_FRACTIONS,
    MeshData,
    Seed,
    _interpolate,
    _section_basis,
    build_station_contours,
    evaluate_seed,
    materialize_seed,
    select_station_evidence,
)


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
EXACT_V52_CASES = frozenset(("S005", "S006", "S007", "S013", "S026", "S027"))
CONTROL_CASES = frozenset(("S011", "S016", "S024", "S029"))
FIGURE_SIZE = (20.48, 11.52)
FIGURE_DPI = 200
EXPECTED_PIXELS = (4096, 2304)


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(row["line_start"], dtype=np.float64), np.asarray(row["line_end"], dtype=np.float64)


def _equal_axes(axis: Any, vertices: np.ndarray) -> None:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.52 * float(np.max(upper - lower)), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _plot_line(axis: Any, start: np.ndarray, end: np.ndarray, *, color: str, width: float, style: str, alpha: float = 1.0) -> None:
    xyz = np.vstack((start, end))
    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=width, linestyle=style, alpha=alpha)


def _render_case_overview(
    mesh: MeshData,
    v52_rows: list[dict[str, Any]],
    v56_rows: list[dict[str, Any]],
    cache_outputs: list[dict[str, Any]],
    removed_ordinals: list[int],
    destination: Path,
    case_id: str,
) -> None:
    views = ((28, -52, "isometric"), (88, -90, "top"), (4, 0, "side"))
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    figure.patch.set_facecolor("#070b13")
    reconstructed = {
        index
        for index, row in enumerate(cache_outputs)
        if str(row.get("class", "")).startswith("RECONSTRUCT")
    }
    for panel, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        axis.set_facecolor("#070b13")
        axis.scatter(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            s=0.10, c="#8e99a8", alpha=0.50, linewidths=0, rasterized=True,
        )
        for ordinal, row in enumerate(v52_rows):
            start, end = _line(row)
            if ordinal in removed_ordinals:
                _plot_line(axis, start, end, color="#ff3b5c", width=2.5, style=":", alpha=0.95)
            else:
                _plot_line(axis, start, end, color="#21d4fd", width=1.5, style="--", alpha=0.65)
        for index, row in enumerate(v56_rows):
            start, end = _line(row)
            color = "#ffb000" if index in reconstructed else "#45e087"
            _plot_line(axis, start, end, color=color, width=2.7, style="-", alpha=1.0)
        _equal_axes(axis, mesh.vertices)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(label, color="white", fontsize=13)
    figure.suptitle(
        f"{case_id} · V56 CTMLR · all {len(mesh.vertices):,} mesh vertices\n"
        "cyan dashed V52 retained · red dotted V52 pruned · amber reconstructed · green byte-held",
        color="white", fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(destination, facecolor=figure.get_facecolor())
    plt.close(figure)


def _line_plane_intersection(start: np.ndarray, end: np.ndarray, station: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    direction = end - start
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1e-12:
        return None
    factor = float(np.dot(station - start, normal) / denominator)
    return start + factor * direction


def _render_seed_sections(
    mesh: MeshData,
    seed: Seed,
    output_line: tuple[np.ndarray, np.ndarray] | None,
    destination: Path,
    *,
    case_id: str,
    line_key: str,
    decision: str,
    output_class: str,
) -> dict[str, Any]:
    proposal, recomputed_decision, audit = evaluate_seed(mesh, seed)
    if recomputed_decision != decision and not (decision == "REJECT_DUPLICATE_LOSER" and proposal is not None):
        raise RuntimeError(f"{case_id}:{line_key}: visual recomputation changed decision {decision} -> {recomputed_decision}")
    figure, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    figure.patch.set_facecolor("#070b13")
    station_records: list[dict[str, Any]] = []
    for station_index, (fraction, axis) in enumerate(zip(STATION_FRACTIONS, axes.flat[:7])):
        station = seed.line_start + fraction * (seed.line_end - seed.line_start)
        contours, section_reason = build_station_contours(mesh, seed, station)
        evidence, evidence_reason = select_station_evidence(mesh, seed, station_index, fraction, station)
        axis.set_facecolor("#09111e")
        for contour in contours:
            axis.plot(contour.points2[:, 0], contour.points2[:, 1], color="#667386", linewidth=0.8, alpha=0.55)
        u, v = _section_basis(seed.direction)
        axis.scatter([0.0], [0.0], marker="x", s=38, c="#21d4fd", linewidths=1.4, label="seed station")
        record: dict[str, Any] = {
            "station_index": station_index,
            "fraction": fraction,
            "contour_count": len(contours),
            "status": evidence_reason,
        }
        if evidence is not None:
            chosen = evidence.contour
            axis.plot(chosen.points2[:, 0], chosen.points2[:, 1], color="#f1f5fb", linewidth=1.5, alpha=0.95)
            markers = []
            for arc, color, marker_label in (
                (evidence.s25, "#bb86fc", "s25"),
                (evidence.s50, "#ff375f", "s50"),
                (evidence.s75, "#bb86fc", "s75"),
            ):
                point = _interpolate(chosen, arc)
                xy = np.asarray((np.dot(point - station, u), np.dot(point - station, v)))
                axis.scatter([xy[0]], [xy[1]], s=44 if marker_label == "s50" else 28, c=color, edgecolors="white", linewidths=0.5, zorder=6)
                markers.append([float(xy[0]), float(xy[1])])
            record.update(
                {
                    "status": "VALID_TURN_LOCUS",
                    "s25_s50_s75_section_xy": markers,
                    "turn_degrees": evidence.turn_degrees,
                    "turn_purity": evidence.turn_purity,
                    "turn_iqr_width": evidence.turn_iqr_width,
                }
            )
        if output_line is not None:
            intersection = _line_plane_intersection(output_line[0], output_line[1], station, seed.direction)
            if intersection is not None:
                xy = np.asarray((np.dot(intersection - station, u), np.dot(intersection - station, v)))
                axis.scatter([xy[0]], [xy[1]], marker="+", s=76, c="#ffb000", linewidths=1.7, zorder=7)
                record["output_intersection_section_xy"] = [float(xy[0]), float(xy[1])]
        axis.axhline(0.0, color="#27344a", linewidth=0.55)
        axis.axvline(0.0, color="#27344a", linewidth=0.55)
        axis.grid(color="#263149", alpha=0.35)
        axis.tick_params(colors="#aab3c2", labelsize=7)
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(f"station {station_index + 1} · {record['status']}", color="white", fontsize=9)
        station_records.append(record)
    summary = axes.flat[7]
    summary.set_facecolor("#09111e")
    summary.axis("off")
    valid = sum(row["status"] == "VALID_TURN_LOCUS" for row in station_records)
    summary.text(
        0.02, 0.98,
        "\n".join(
            (
                f"{case_id} / {line_key}",
                f"source: {seed.origin_class} row {seed.row_ordinal}",
                f"frozen decision: {decision}",
                f"output class: {output_class}",
                f"valid loci: {valid}/7",
                f"B: {mesh.bbox_diagonal:.3f} mm",
                f"h: {mesh.median_mesh_edge:.4f} mm",
                f"W: {mesh.constants.search_half_width:.3f} mm",
                f"tau_f: {mesh.constants.line_consensus_threshold:.3f} mm",
                f"tau_c: {mesh.constants.transverse_center_threshold:.3f} mm",
                "grey: all section contours",
                "white: selected contour",
                "violet/red: s25/s50/s75",
                "cyan ×: seed station",
                "amber +: output-line intersection",
            )
        ),
        transform=summary.transAxes, va="top", ha="left", color="white", fontsize=10, linespacing=1.45,
    )
    figure.suptitle("V56 canonical-turn median evidence · seven frozen stations", color="white", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(destination, facecolor=figure.get_facecolor())
    plt.close(figure)
    return {"station_records": station_records, "recomputed_decision": recomputed_decision}


def _dimensions(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [image.width, image.height]


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
    if manifest.get("status") != "READY_FOR_SINGLE_SCORE" or manifest.get("truth_opened") is not False:
        raise ValueError("V56 execution is not frozen for downstream visual audit")
    if not all((manifest.get("semantic_replay_identity") or {}).values()):
        raise ValueError("V56 replay identity failed")

    case_data: dict[str, dict[str, Any]] = {}
    mandatory_cases = set(EXACT_V52_CASES | CONTROL_CASES)
    changed_cases: set[str] = set()
    for index in range(1, 31):
        case_id = f"S{index:03d}"
        result = json.loads((args.execution_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        if result.get("kind") != RESULT_KIND or result.get("profile") != PROFILE:
            raise ValueError(f"{case_id}: invalid V56 result")
        v52 = json.loads((args.v52_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        cache = json.loads((args.execution_root / "primary" / case_id / "compact_cache.json").read_text(encoding="utf-8"))
        changed = _canonical(result["predictions"]) != _canonical(v52["predictions"])
        if changed:
            changed_cases.add(case_id)
            mandatory_cases.add(case_id)
        case_data[case_id] = {"result": result, "v52": v52, "cache": cache, "changed": changed}

    records: list[dict[str, Any]] = []
    for case_id in sorted(mandatory_cases):
        mesh = MeshData.load(args.scan_root / f"{case_id}.ply")
        pool = json.loads((args.pool_root / "primary" / case_id / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        result = case_data[case_id]["result"]
        v52 = case_data[case_id]["v52"]
        cache = case_data[case_id]["cache"]
        represented_v52 = {
            int(row["source_seed_ordinal"])
            for row in cache["outputs"]
            if row["class"] in ("HOLD_V52", "RECONSTRUCT_V52")
        }
        removed_ordinals = sorted(set(range(len(v52["predictions"]))) - represented_v52)
        diagnostic_specs: list[dict[str, Any]] = []
        for output_index, row in enumerate(cache["outputs"]):
            if row["class"] == "RECONSTRUCT_V52":
                diagnostic_specs.append(
                    {"key": f"reconstruct_v52_{row['source_seed_ordinal']:04d}", "class": row["class"], "origin": "V52", "ordinal": int(row["source_seed_ordinal"]), "output_index": output_index, "decision": "ACCEPT"}
                )
            elif row["class"] == "RECONSTRUCT_POOL":
                diagnostic_specs.append(
                    {"key": f"reconstruct_pool_{row['source_seed_ordinal']:04d}", "class": row["class"], "origin": "POOL", "ordinal": int(row["source_seed_ordinal"]), "output_index": output_index, "decision": "ACCEPT"}
                )
        for ordinal in removed_ordinals:
            diagnostic_specs.append(
                {"key": f"pruned_v52_{ordinal:04d}", "class": "PRUNED_V52", "origin": "V52", "ordinal": ordinal, "output_index": None, "decision": None}
            )

        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        overview = case_root / "three_view_native_overlay_4096x2304.png"
        _render_case_overview(mesh, v52["predictions"], result["predictions"], cache["outputs"], removed_ordinals, overview, case_id)
        if _dimensions(overview) != list(EXPECTED_PIXELS):
            raise RuntimeError(f"{case_id}: overview dimensions changed")
        line_records: list[dict[str, Any]] = []
        for spec in diagnostic_specs:
            source_rows = v52["predictions"] if spec["origin"] == "V52" else pool["line_objects"]
            source_row = source_rows[spec["ordinal"]]
            seed = materialize_seed(
                source_row,
                bbox_diagonal=mesh.bbox_diagonal,
                median_mesh_edge=mesh.median_mesh_edge,
                origin_class=spec["origin"],
                origin_file_ordinal=0 if spec["origin"] == "V52" else 1,
                row_ordinal=spec["ordinal"],
            )
            proposal, recomputed_decision, _ = evaluate_seed(mesh, seed)
            frozen_decision = spec["decision"] or (
                "REJECT_DUPLICATE_LOSER" if recomputed_decision == "ACCEPT" else recomputed_decision
            )
            output_line = None
            if spec["output_index"] is not None:
                output_line = _line(result["predictions"][spec["output_index"]])
            section_path = case_root / f"{spec['key']}_seven_sections_4096x2304.png"
            section_metadata = _render_seed_sections(
                mesh, seed, output_line, section_path,
                case_id=case_id, line_key=spec["key"], decision=frozen_decision, output_class=spec["class"],
            )
            if _dimensions(section_path) != list(EXPECTED_PIXELS):
                raise RuntimeError(f"{case_id}:{spec['key']}: section dimensions changed")
            line_records.append(
                {
                    **spec,
                    "frozen_decision": frozen_decision,
                    "render": str(section_path),
                    "render_sha256": _sha(section_path),
                    "dimensions": _dimensions(section_path),
                    **section_metadata,
                }
            )
        records.append(
            {
                "case_id": case_id,
                "changed": case_id in changed_cases,
                "source_vertex_count": len(mesh.vertices),
                "source_triangle_count": len(mesh.faces),
                "source_point_serialization": False,
                "overview": str(overview),
                "overview_sha256": _sha(overview),
                "overview_dimensions": _dimensions(overview),
                "removed_v52_ordinals": removed_ordinals,
                "diagnostic_line_count": len(line_records),
                "lines": line_records,
            }
        )
    output = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_ctmlr_source_only_visual_manifest",
        "token": TOKEN,
        "profile": PROFILE,
        "truth_opened": False,
        "native_source_vertex_use": True,
        "source_point_serialization": False,
        "pixel_dimensions": list(EXPECTED_PIXELS),
        "required_exact_cases": sorted(EXACT_V52_CASES),
        "required_control_cases": sorted(CONTROL_CASES),
        "changed_cases": sorted(changed_cases),
        "rendered_cases": sorted(mandatory_cases),
        "records": records,
    }
    _write_new(args.output_root / "visual_manifest.json", output)
    print(json.dumps({"rendered_cases": len(records), "changed_cases": len(changed_cases), "diagnostic_lines": sum(row["diagnostic_line_count"] for row in records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
