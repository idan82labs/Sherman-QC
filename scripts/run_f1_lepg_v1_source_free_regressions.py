#!/usr/bin/env python3
"""Execute the frozen LEPG-V1 source-free kill suite exactly once."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_lepg_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json
from research.f1_lepg_v1.core import LEPGInvalid, canonicalize_xyz, evaluate
from research.f1_lepg_v1.fixtures import (
    flat_double_skin,
    full_cylinders,
    sf01,
    sf02,
    sf03,
    sf05_single_skin,
    sf07_open_sectors,
    sf08_short_bend,
)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _axis_metrics(axis, s: float, expected_direction=np.asarray((0.0, 1.0, 0.0))):
    if axis is None:
        return math.inf, math.inf, [math.inf, math.inf], math.inf
    direction_error = math.degrees(math.acos(float(np.clip(abs(np.dot(axis.direction, expected_direction)), 0.0, 1.0))))
    delta = axis.point
    line_distance = float(np.linalg.norm(delta - axis.direction * np.dot(delta, axis.direction)))
    radius_errors = [abs(axis.radii[0] - 8.0) / s, abs(axis.radii[1] - 16.0) / s]
    angle_error = abs(axis.arc_span_degrees - 90.0)
    return direction_error, line_distance / s, radius_errors, angle_error


def _compact(name: str, evaluation) -> dict:
    section = evaluation.cross_sections[0] if evaluation.cross_sections else None
    return {
        "name": name,
        "source": evaluation.source.xyz.copy(),
        "s": evaluation.s,
        "centroid": evaluation.centroid.copy(),
        "section": section,
        "axes": list(evaluation.axes),
        "counts": dict(evaluation.semantic_payload["counts"]),
    }


def _render(records: list[dict], checks: list[dict], decision: str) -> Path:
    output = ROOT / "output/research/lepg-v1-source-free-causal-ruling"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "LEPG_V1_SOURCE_FREE_CAUSAL_RULING_4096.png"
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=256)
    panels = list(axes.ravel())
    panel = 0
    for record in records[:4]:
        source = record["source"]
        ax = panels[panel]
        ax.scatter(source[:, 0], source[:, 2], s=0.30, c="#98a2b3", alpha=0.22, linewidths=0)
        for axis in record["axes"]:
            projected = np.asarray((axis.direction[0], axis.direction[2]))
            if np.linalg.norm(projected) <= 1e-9:
                ax.scatter([axis.point[0]], [axis.point[2]], marker="x", s=100, c="#dc2626", linewidths=2.5)
            else:
                projected /= np.linalg.norm(projected)
                ax.plot([axis.point[0] - 20 * projected[0], axis.point[0] + 20 * projected[0]], [axis.point[2] - 20 * projected[1], axis.point[2] + 20 * projected[1]], c="#dc2626", linewidth=2.2)
        counts = record["counts"]
        ax.set_title(f"{record['name']} source X-Z | dirs {counts['translation_direction_candidates']} | axes {counts['final_axes']}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#d0d5dd", linewidth=0.35, alpha=0.55)
        panel += 1
        section = record["section"]
        ax = panels[panel]
        if section is not None:
            ax.scatter(section.profile_cells[:, 0], section.profile_cells[:, 1], s=5.0, c=section.profile_weights, cmap="viridis", alpha=0.72, linewidths=0)
            for axis in record["axes"]:
                for radius in axis.radii:
                    phi = np.linspace(axis.arc_endpoints[0], axis.arc_endpoints[1], 160)
                    center_x = float(np.dot(axis.point - record["centroid"], section.basis_b))
                    center_z = float(np.dot(axis.point - record["centroid"], section.basis_c))
                    ax.plot(center_x + radius * np.cos(phi), center_z + radius * np.sin(phi), c="#dc2626", linewidth=2.0)
            ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{record['name']} collapsed consensus profile", fontsize=10)
        ax.grid(color="#d0d5dd", linewidth=0.35, alpha=0.55)
        panel += 1
    while panel < 8:
        panels[panel].axis("off")
        panel += 1
    panels[8].axis("off")
    lines = ["LINEAR EXTRUSION PROFILE GRAMMAR V1", "SOURCE-FREE CAUSAL RULING", "", f"decision: {decision}", ""]
    for check in checks:
        lines.append(f"{check['check']}: {'PASS' if check['passed'] else 'FAIL'}")
    panels[8].text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    fig.suptitle("LEPG V1 — repeated raw-coordinate cross-sections must satisfy a two-skin open line–arc–line grammar", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=256, facecolor="white")
    plt.close(fig)
    return path


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = math.radians(degrees)
    cross = np.asarray(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross


def main() -> int:
    checks: list[dict] = []
    rows: list[dict] = []
    visual_records: list[dict] = []
    stopped = False

    def record(name: str, passed: bool, evidence: dict) -> None:
        nonlocal stopped
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})
        stopped = stopped or not passed

    def run(name: str, points: np.ndarray):
        start = perf_counter()
        evaluation = evaluate(canonicalize_xyz(points))
        rows.append({"fixture": name, "wall_seconds": perf_counter() - start, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "semantic": evaluation.semantic_payload})
        if len(visual_records) < 4:
            visual_records.append(_compact(name, evaluation))
        return evaluation

    try:
        sf01_points = sf01()
        base = run("SF01", sf01_points)
        axis = base.axes[0] if len(base.axes) == 1 else None
        de, ld, re, ae = _axis_metrics(axis, base.s)
        record("SF01_90_degree_double_skin_positive", axis is not None and de <= 0.10 and ld <= 0.20 and max(re) <= 0.25 and ae <= 2.0 and min(axis.continuation_coverages) >= CONFIG.ray_bin_coverage_min, {"counts": base.semantic_payload["counts"], "direction_error_degrees": _finite(de), "line_distance_over_s": _finite(ld), "radius_errors_over_s": [_finite(value) for value in re], "angle_error_degrees": _finite(ae), "continuation_coverages": list(axis.continuation_coverages) if axis else None})
        base_semantic = base.semantic_payload
        del base
        gc.collect()

        if not stopped:
            evaluated = run("SF02", sf02())
            axis = evaluated.axes[0] if len(evaluated.axes) == 1 else None
            de, ld, re, ae = _axis_metrics(axis, evaluated.s)
            record("SF02_axial_crop_positive", axis is not None and de <= 0.10 and ld <= 0.20 and max(re) <= 0.25 and ae <= 2.0 and axis.passing_bin_count >= 6, {"counts": evaluated.semantic_payload["counts"], "direction_error_degrees": _finite(de), "line_distance_over_s": _finite(ld), "radius_errors_over_s": [_finite(value) for value in re], "angle_error_degrees": _finite(ae), "passing_bin_count": axis.passing_bin_count if axis else None})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF03", sf03())
            axis = evaluated.axes[0] if len(evaluated.axes) == 1 else None
            if axis is not None:
                de = math.degrees(math.acos(float(np.clip(abs(np.dot(axis.direction, np.asarray((0.0, 1.0, 0.0)))), 0.0, 1.0))))
                delta = axis.point
                ld = float(np.linalg.norm(delta - axis.direction * np.dot(delta, axis.direction))) / evaluated.s
                re = [abs(axis.radii[0] - 21.0) / evaluated.s, abs(axis.radii[1] - 27.0) / evaluated.s]
                ae = abs(axis.arc_span_degrees - 60.0)
            else:
                de, ld, re, ae = math.inf, math.inf, [math.inf, math.inf], math.inf
            record("SF03_larger_radius_60_degree_positive", axis is not None and de <= 0.10 and ld <= 0.20 and max(re) <= 0.25 and ae <= 2.0, {"counts": evaluated.semantic_payload["counts"], "direction_error_degrees": _finite(de), "line_distance_over_s": _finite(ld), "radius_errors_over_s": [_finite(value) for value in re], "angle_error_degrees": _finite(ae)})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF04", flat_double_skin(token="LEPG_SF04"))
            counts = evaluated.semantic_payload["counts"]
            record("SF04_flat_double_skin", counts["translation_direction_candidates"] >= 1 and counts["consensus_profile_cells_max"] >= 96 and counts["open_sector_models"] == 0 and counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF05", sf05_single_skin())
            counts = evaluated.semantic_payload["counts"]
            record("SF05_single_skin_bend", counts["translation_direction_candidates"] >= 1 and counts["open_sector_models"] == 0 and counts["final_axes"] == 0, {"counts": counts, "note": "zero dual-shell fits required; single-shell open arc is visually audited"})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF06", full_cylinders())
            counts = evaluated.semantic_payload["counts"]
            record("SF06_complete_concentric_cylinders", counts["translation_direction_candidates"] >= 1 and counts["dual_circle_models"] >= 1 and counts["open_sector_models"] == 0 and counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF07", sf07_open_sectors())
            counts = evaluated.semantic_payload["counts"]
            record("SF07_open_sectors_without_flanges", counts["open_sector_models"] >= 1 and counts["four_tangent_models"] == 0 and counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF08", sf08_short_bend())
            counts = evaluated.semantic_payload["counts"]
            record("SF08_axially_short_nuisance", counts["translation_direction_candidates"] >= 1 and counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF09", flat_double_skin(token="LEPG_SF09", hole_radius=12.0, hole_wall=True))
            counts = evaluated.semantic_payload["counts"]
            record("SF09_through_hole", counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF10", flat_double_skin(token="LEPG_SF10", terminal_wall=True))
            counts = evaluated.semantic_payload["counts"]
            record("SF10_terminal_wall", counts["translation_direction_candidates"] >= 1 and counts["four_tangent_models"] == 0 and counts["final_axes"] == 0, {"counts": counts})
            del evaluated
            gc.collect()

        if not stopped:
            order = np.argsort(np.asarray([_splitmix64(i ^ 0x4C4550477631) for i in range(len(sf01_points))], dtype=np.uint64))
            permutation = run("SF11_permutation", sf01_points[order])
            permutation_equal = permutation.semantic_payload == base_semantic
            del permutation
            transform_rows = []
            transforms = [
                (_rotation(np.asarray((1.0, 2.0, 3.0)), 37), np.asarray((17.0, -11.0, 23.0))),
                (_rotation(np.asarray((-2.0, 1.0, 4.0)), 103), np.asarray((-31.0, 7.0, 19.0))),
                (_rotation(np.asarray((5.0, -3.0, 2.0)), 211), np.asarray((9.0, 29.0, -13.0))),
            ]
            for index, (rotation, translation) in enumerate(transforms):
                evaluated = run(f"SF11_T{index + 1}", sf01_points @ rotation.T + translation)
                if len(evaluated.axes) == 1:
                    direction = evaluated.axes[0].direction @ rotation
                    point = (evaluated.axes[0].point - translation) @ rotation
                    de = math.degrees(math.acos(float(np.clip(abs(np.dot(direction, np.asarray((0.0, 1.0, 0.0)))), 0.0, 1.0))))
                    ld = float(np.linalg.norm(point - direction * np.dot(point, direction))) / evaluated.s
                    re = [abs(evaluated.axes[0].radii[0] - 8.0) / evaluated.s, abs(evaluated.axes[0].radii[1] - 16.0) / evaluated.s]
                else:
                    de, ld, re = math.inf, math.inf, [math.inf, math.inf]
                transform_rows.append({"axis_count": len(evaluated.axes), "direction_error_degrees": _finite(de), "line_distance_over_s": _finite(ld), "radius_errors_over_s": [_finite(value) for value in re]})
                del evaluated
                gc.collect()
            record("SF11_metamorphic", permutation_equal and all(row["axis_count"] == 1 and row["direction_error_degrees"] <= 0.10 and row["line_distance_over_s"] <= 0.20 and max(row["radius_errors_over_s"]) <= 0.25 for row in transform_rows), {"permutation_byte_semantic_equal": permutation_equal, "rigid_transforms": transform_rows})
    except LEPGInvalid as exc:
        record("source_free_execution", False, {"exception": type(exc).__name__, "message": str(exc)})

    passed = bool(checks) and all(row["passed"] for row in checks)
    decision = "LEPG_V1_SOURCE_FREE_PASS" if passed else "LEPG_V1_SOURCE_FREE_ARCHITECTURE_STOP"
    visual = _render(visual_records, checks, decision)
    evidence = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "checks": checks,
        "results": rows,
        "decision": decision,
        "native_visual": str(visual.relative_to(ROOT)),
        "c006_source_opened": False,
        "c006_claim_created": False,
        "c006_claim_consumed": False,
        "gold_labels_cad_expected_counts_benchmark_opened": False,
    }
    evidence["evidence_sha256"] = sha_json(evidence)
    output = ROOT / "docs/research/LEPG_V1_SOURCE_FREE_REGRESSIONS_20260810.json"
    output.write_bytes(canonical_json_bytes(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
