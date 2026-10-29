#!/usr/bin/env python3
"""Run the frozen LSSC-V1 source-free kill funnel exactly once."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_lssc_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json
from research.f1_lssc_v1.core import LSSCInvalid, LSSCResourceStop, canonicalize_xyz, evaluate
from research.f1_lssc_v1.fixtures import (
    bend,
    complete_cylinders,
    flat_double_skin,
    sf01_double_skin,
    sf02_axial_crop,
    sf03_scale_angle,
    sf04_single_skin,
    sphere,
)


def _axis_metrics(axis, expected_origin=np.zeros(3), expected_direction=np.asarray((0.0, 1.0, 0.0))):
    direction_error = math.degrees(
        math.acos(float(np.clip(abs(np.dot(axis.direction, expected_direction)), 0.0, 1.0)))
    )
    delta = axis.line_point - expected_origin
    distance = float(np.linalg.norm(delta - axis.direction * np.dot(delta, axis.direction)))
    return direction_error, distance


def _finite_or_none(value: float) -> float | None:
    """Keep failure evidence strict-JSON without altering scientific outcomes."""
    return float(value) if math.isfinite(float(value)) else None


def _compact_visual(name: str, evaluation) -> dict:
    raw_slots = np.asarray([value.anchor_id for value in evaluation.raw_candidates], dtype=np.int64)
    localized_slots = np.asarray([value.candidate.anchor_id for value in evaluation.localized_votes], dtype=np.int64)
    tail_rows = []
    for vote in evaluation.localized_votes:
        for slot, normal in (
            (vote.positive_anchor_slot, vote.positive_normal),
            (vote.negative_anchor_slot, vote.negative_normal),
        ):
            tail_rows.append((evaluation.source.xyz[evaluation.field.anchor_ids[slot]].copy(), normal.copy()))
    return {
        "name": name,
        "source": evaluation.source.xyz.copy(),
        "raw": evaluation.source.xyz[raw_slots].copy() if len(raw_slots) else np.empty((0, 3)),
        "localized": evaluation.source.xyz[localized_slots].copy() if len(localized_slots) else np.empty((0, 3)),
        "tails": tail_rows,
        "axes": [(axis.line_point.copy(), axis.direction.copy()) for axis in evaluation.axes],
        "counts": dict(evaluation.semantic_payload["counts"]),
    }


def _draw_fixture(ax, record: dict, title: str) -> None:
    points = record["source"]
    if len(points) > 60_000:
        ids = np.linspace(0, len(points) - 1, 60_000, dtype=int)
        points = points[ids]
    ax.scatter(points[:, 0], points[:, 2], s=0.32, c="#98a2b3", alpha=0.20, linewidths=0)
    if len(record["raw"]):
        ax.scatter(record["raw"][:, 0], record["raw"][:, 2], s=5.0, c="#2563eb", alpha=0.42, linewidths=0)
    if len(record["localized"]):
        ax.scatter(record["localized"][:, 0], record["localized"][:, 2], s=8.0, c="#f59e0b", alpha=0.82, linewidths=0)
    for point, direction in record["axes"]:
        projected = np.asarray((direction[0], direction[2]))
        if np.linalg.norm(projected) < 1e-9:
            ax.scatter([point[0]], [point[2]], marker="x", s=100, c="#dc2626", linewidths=2.5)
        else:
            projected /= np.linalg.norm(projected)
            ax.plot(
                [point[0] - 12 * projected[0], point[0] + 12 * projected[0]],
                [point[2] - 12 * projected[1], point[2] + 12 * projected[1]],
                c="#dc2626", linewidth=2.5,
            )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d0d5dd", linewidth=0.35, alpha=0.55)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")


def _render_visual(records: list[dict], checks: list[dict], decision: str) -> Path:
    output = ROOT / "output/research/lssc-v1-source-free-causal-ruling"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "LSSC_V1_SOURCE_FREE_CAUSAL_RULING_4096.png"
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=256)
    flat_axes = list(axes.ravel())
    required_order = ["SF01", "SF03", "SF04", "SF06", "SF07", "SF08"]
    by_name = {record["name"]: record for record in records}
    panel = 0
    for name in required_order:
        record = by_name.get(name)
        if record is None:
            continue
        counts = record["counts"]
        _draw_fixture(
            flat_axes[panel],
            record,
            f"{name} | raw {counts['raw_two_motion_candidates']} | tails {counts['bilateral_tail_pairs']} | axes {counts['final_axes']}",
        )
        panel += 1
    while panel < 8:
        flat_axes[panel].axis("off")
        panel += 1
    flat_axes[8].axis("off")
    lines = [
        "LOCAL SCREW SLIPPAGE COHERENCE V1",
        "SOURCE-FREE CAUSAL RULING",
        "",
        "neutral: source",
        "blue: raw two-motion anchors",
        "orange: bilateral-tail survivors",
        "red: retained axes",
        "",
        f"decision: {decision}",
        "",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"{check['check']}: {status}")
    flat_axes[8].text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle("LSSC V1 — raw slippage must become bilateral planar-tail-supported lines", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=256, facecolor="white")
    plt.close(fig)
    return path


def _hash_permutation(points: np.ndarray) -> np.ndarray:
    keys = np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(b"LSSC_V1_METAMORPHIC_PERM\0" + np.ascontiguousarray(row, dtype="<f8").tobytes()).digest()[:8],
                "little",
            )
            for row in points
        ),
        dtype=np.uint64,
        count=len(points),
    )
    return np.lexsort((np.arange(len(points)), keys))


def _rotation_x(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.asarray(((1, 0, 0), (0, math.cos(a), -math.sin(a)), (0, math.sin(a), math.cos(a))))


def _rotation_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.asarray(((math.cos(a), 0, math.sin(a)), (0, 1, 0), (-math.sin(a), 0, math.cos(a))))


def _rotation_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.asarray(((math.cos(a), -math.sin(a), 0), (math.sin(a), math.cos(a), 0), (0, 0, 1)))


def main() -> int:
    checks: list[dict] = []
    results: list[dict] = []
    visuals: list[dict] = []
    stopped = False

    def run(name: str, points: np.ndarray):
        start = perf_counter()
        source = canonicalize_xyz(points)
        evaluation = evaluate(source)
        row = {
            "fixture": name,
            "wall_seconds": perf_counter() - start,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "semantic": evaluation.semantic_payload,
        }
        results.append(row)
        if name in {"SF01", "SF03", "SF04", "SF06", "SF07", "SF08"}:
            visuals.append(_compact_visual(name, evaluation))
        return evaluation

    def record(name: str, passed: bool, evidence: dict) -> bool:
        nonlocal stopped
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            stopped = True
        return bool(passed)

    try:
        sf01_points = sf01_double_skin()
        sf01 = run("SF01", sf01_points)
        axis = sf01.axes[0] if len(sf01.axes) == 1 else None
        direction_error, line_distance = _axis_metrics(axis) if axis is not None else (math.inf, math.inf)
        record(
            "SF01_double_skin_90_degree_bend",
            axis is not None
            and direction_error <= 2.0
            and line_distance <= 1.5 * sf01.field.s
            and axis.axial_span >= 60.0 * sf01.field.s,
            {
                "counts": sf01.semantic_payload["counts"],
                "direction_error_degrees": _finite_or_none(direction_error),
                "line_distance_over_s": _finite_or_none(line_distance / sf01.field.s),
                "axial_span_over_s": axis.axial_span / sf01.field.s if axis is not None else None,
            },
        )
        sf01_semantic = sf01.semantic_payload
        del sf01
        gc.collect()

        if not stopped:
            evaluated = run("SF02", sf02_axial_crop())
            axis = evaluated.axes[0] if len(evaluated.axes) == 1 else None
            de, ld = _axis_metrics(axis) if axis is not None else (math.inf, math.inf)
            record("SF02_axial_crop", axis is not None and de <= 2 and ld <= 1.5 * evaluated.field.s and axis.axial_span >= 24 * evaluated.field.s, {"counts": evaluated.semantic_payload["counts"], "direction_error_degrees": _finite_or_none(de), "line_distance_over_s": _finite_or_none(ld / evaluated.field.s), "axial_span_over_s": axis.axial_span / evaluated.field.s if axis else None})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF03", sf03_scale_angle())
            axis = evaluated.axes[0] if len(evaluated.axes) == 1 else None
            de, ld = _axis_metrics(axis) if axis is not None else (math.inf, math.inf)
            record("SF03_scale_angle_positive", axis is not None and de <= 2 and ld <= 2 * evaluated.field.s and axis.axial_span >= 60 * evaluated.field.s, {"counts": evaluated.semantic_payload["counts"], "direction_error_degrees": _finite_or_none(de), "line_distance_over_s": _finite_or_none(ld / evaluated.field.s), "axial_span_over_s": axis.axial_span / evaluated.field.s if axis else None})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF04", sf04_single_skin())
            axis = evaluated.axes[0] if len(evaluated.axes) == 1 else None
            de, ld = _axis_metrics(axis) if axis is not None else (math.inf, math.inf)
            record("SF04_single_skin_positive", axis is not None and de <= 2 and ld <= 1.5 * evaluated.field.s, {"counts": evaluated.semantic_payload["counts"], "direction_error_degrees": _finite_or_none(de), "line_distance_over_s": _finite_or_none(ld / evaluated.field.s)})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF05", flat_double_skin(token="LSSC_SF05"))
            record("SF05_flat_double_skin", len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"]})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF06", complete_cylinders(token="LSSC_SF06", radii=(8.0, 16.0)))
            fraction = evaluated.semantic_payload["raw_two_motion_fraction_of_valid_patch_anchors"]
            record("SF06_complete_concentric_cylinder", fraction >= 0.90 and len(evaluated.localized_votes) == 0 and len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"], "raw_two_motion_fraction": fraction})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF07", complete_cylinders(token="LSSC_SF07", radii=(27.0,)))
            fraction = evaluated.semantic_payload["raw_two_motion_fraction_of_valid_patch_anchors"]
            record("SF07_shallow_complete_cylinder", fraction >= 0.90 and len(evaluated.localized_votes) == 0 and len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"], "raw_two_motion_fraction": fraction})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF08", flat_double_skin(token="LSSC_SF08", hole_radius=12.0, include_hole_wall=True))
            record("SF08_through_hole", len(evaluated.localized_votes) == 0 and len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"]})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF09", flat_double_skin(token="LSSC_SF09", terminal_wall=True))
            record("SF09_terminal_wall", len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"]})
            del evaluated
            gc.collect()

        if not stopped:
            evaluated = run("SF10", sphere())
            record("SF10_sphere", len(evaluated.localized_votes) == 0 and len(evaluated.axes) == 0, {"counts": evaluated.semantic_payload["counts"]})
            del evaluated
            gc.collect()

        if not stopped:
            permutation = run("SF11_permutation", sf01_points[_hash_permutation(sf01_points)])
            permutation_equal = permutation.semantic_payload == sf01_semantic
            del permutation
            transform_rows = []
            transforms = [
                (_rotation_x(17), np.asarray((11.0, -7.0, 5.0))),
                (_rotation_z(-23) @ _rotation_y(71), np.asarray((-23.0, 9.0, 31.0))),
                (_rotation_x(11) @ _rotation_y(29) @ _rotation_z(-43), np.asarray((101.0, -37.0, 19.0))),
            ]
            for index, (rotation, translation) in enumerate(transforms):
                evaluated = run(f"SF11_rigid_{index + 1}", sf01_points @ rotation.T + translation)
                if len(evaluated.axes) == 1:
                    mapped_direction = evaluated.axes[0].direction @ rotation
                    mapped_point = (evaluated.axes[0].line_point - translation) @ rotation
                    direction_disagreement = math.degrees(math.acos(float(np.clip(abs(np.dot(mapped_direction, np.asarray((0.0, 1.0, 0.0)))), 0.0, 1.0))))
                    line_distance = float(np.linalg.norm(mapped_point - mapped_direction * np.dot(mapped_point, mapped_direction)))
                else:
                    direction_disagreement, line_distance = math.inf, math.inf
                transform_rows.append({"axis_count": len(evaluated.axes), "direction_disagreement_degrees": direction_disagreement, "line_distance_over_s": line_distance / evaluated.field.s})
                del evaluated
                gc.collect()
            scale_rows = []
            for factor in (0.5, 3.0):
                evaluated = run(f"SF11_scale_{factor}", sf01_points * factor)
                if len(evaluated.axes) == 1:
                    mapped_point = evaluated.axes[0].line_point / factor
                    direction_disagreement = math.degrees(math.acos(float(np.clip(abs(np.dot(evaluated.axes[0].direction, np.asarray((0.0, 1.0, 0.0)))), 0.0, 1.0))))
                    line_distance = float(np.linalg.norm(mapped_point - evaluated.axes[0].direction * np.dot(mapped_point, evaluated.axes[0].direction)))
                else:
                    direction_disagreement, line_distance = math.inf, math.inf
                scale_rows.append({"factor": factor, "axis_count": len(evaluated.axes), "direction_disagreement_degrees": direction_disagreement, "line_distance_over_s": line_distance / (evaluated.field.s / factor)})
                del evaluated
                gc.collect()
            record(
                "SF11_metamorphic",
                permutation_equal
                and all(row["axis_count"] == 1 and row["direction_disagreement_degrees"] <= 0.75 and row["line_distance_over_s"] <= 0.75 for row in transform_rows)
                and all(row["axis_count"] == 1 and row["direction_disagreement_degrees"] <= 0.75 and row["line_distance_over_s"] <= 0.75 for row in scale_rows),
                {"permutation_byte_semantic_equal": permutation_equal, "rigid_transforms": transform_rows, "uniform_scales": scale_rows},
            )
    except (LSSCInvalid, LSSCResourceStop) as exc:
        record("source_free_execution", False, {"exception": type(exc).__name__, "message": str(exc)})

    passed = bool(checks) and all(check["passed"] for check in checks)
    decision = "LSSC_V1_SOURCE_FREE_REGRESSIONS_PASS" if passed else "LSSC_V1_SOURCE_FREE_REGRESSION_ARCHITECTURE_STOP"
    visual_path = _render_visual(visuals, checks, decision)
    evidence = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "checks": checks,
        "results": results,
        "regressions_passed": passed,
        "decision": decision,
        "native_visual": str(visual_path.relative_to(ROOT)),
        "c006_source_opened": False,
        "c006_claim_created": False,
        "c006_claim_consumed": False,
        "target_gold_cad_benchmark_opened": False,
    }
    evidence["evidence_sha256"] = sha_json(evidence)
    output = ROOT / "docs/research/LSSC_V1_SOURCE_FREE_REGRESSIONS_20260809.json"
    output.write_bytes(canonical_json_bytes(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
