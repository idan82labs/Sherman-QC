#!/usr/bin/env python3
"""Run the seven frozen source-free AMHC-V1 regressions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_amhc_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json
from research.f1_amhc_v1.core import canonicalize_xyz, evaluate


def _hash_u64(row: np.ndarray, tag: bytes) -> int:
    return int.from_bytes(
        hashlib.sha256(tag + np.ascontiguousarray(row, dtype="<f8").tobytes()).digest()[:8],
        "little",
    )


def perturb(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    kept: list[np.ndarray] = []
    for row in values:
        key = _hash_u64(row, b"AMHC_V1_FIXTURE_DELETE\0")
        if key % 20 == 0:
            continue
        digest = hashlib.sha256(
            b"AMHC_V1_FIXTURE_JITTER\0" + np.ascontiguousarray(row, dtype="<f8").tobytes()
        ).digest()
        jitter = (
            np.asarray(
                [int.from_bytes(digest[i : i + 2], "little") for i in (0, 2, 4)],
                dtype=np.float64,
            )
            / 65535.0
            - 0.5
        ) * 0.06
        kept.append(row + jitter)
    return np.asarray(kept, dtype=np.float64)


def flat_double_skin(*, hole: bool = False, end_wall: bool = False) -> np.ndarray:
    axis = np.arange(-42.0, 43.0, 1.0)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    base = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    if hole:
        base = base[np.sum(base[:, :2] ** 2, axis=1) >= 12.0**2]
    top = base + np.asarray((0.0, 0.0, 4.0))
    bottom = base - np.asarray((0.0, 0.0, 4.0))
    rows = [top, bottom]
    if end_wall:
        y = axis
        z = np.arange(-4.0, 5.0, 1.0)
        yy2, zz2 = np.meshgrid(y, z, indexing="ij")
        wall = np.column_stack(
            (np.full(yy2.size, 42.0), yy2.ravel(), zz2.ravel())
        )
        rows.append(wall)
    return perturb(np.vstack(rows))


def bent_double_skin(*, crop: bool = False, single: bool = False) -> np.ndarray:
    radius = 12.0
    y_values = np.arange(-42.0, 43.0, 1.0)
    # Arc-length coordinate: two 42-unit flanks plus a 90-degree radius-12 arc.
    s_values = np.arange(-42.0, 42.0 + radius * math.pi / 2.0 + 1e-9, 1.0)
    rows: list[np.ndarray] = []
    for s in s_values:
        if s < 0.0:
            midpoint = np.asarray((s, 0.0, 0.0))
            normal = np.asarray((0.0, 0.0, 1.0))
        elif s <= radius * math.pi / 2.0:
            angle = s / radius
            midpoint = np.asarray((radius * math.sin(angle), 0.0, radius * (1.0 - math.cos(angle))))
            normal = np.asarray((-math.sin(angle), 0.0, math.cos(angle)))
        else:
            distance = s - radius * math.pi / 2.0
            midpoint = np.asarray((radius, 0.0, radius + distance))
            normal = np.asarray((-1.0, 0.0, 0.0))
        if crop and s < -30.0:
            continue
        for y in y_values:
            center = midpoint + np.asarray((0.0, y, 0.0))
            rows.append(center + 4.0 * normal)
            if not single:
                rows.append(center - 4.0 * normal)
    return perturb(np.asarray(rows, dtype=np.float64))


def _run(name: str, points: np.ndarray) -> tuple[dict, object]:
    source = canonicalize_xyz(points)
    result = evaluate(source)
    semantic = result.semantic_payload
    return {
        "fixture": name,
        "point_count": len(source.xyz),
        "decision": result.decision,
        "counts": semantic["counts"],
        "fractions": semantic["fractions"],
        "scales": semantic["scales"],
        "hashes": semantic["hashes"],
    }, result


def main() -> int:
    results: list[dict] = []
    checks: list[dict] = []

    def record(name: str, passed: bool, evidence: dict) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    flat_row, flat = _run("flat_double_skin", flat_double_skin())
    flat_counts = flat.semantic_payload["counts"]
    record(
        "flat_double_skin",
        flat_counts["reciprocal_pairs"] >= 5_000
        and flat_counts["core_midpoints"] >= 4_000
        and flat_counts["retained_components"] == 0,
        flat_counts,
    )
    results.append(flat_row)

    bend_points = bent_double_skin()
    bend_row, bend = _run("double_skin_90deg_bend", bend_points)
    retained = [row for row in bend.components if row.retained]
    largest = max(bend.components, key=lambda row: row.edge_count)
    record(
        "double_skin_90deg_bend",
        len(retained) == 1 and retained[0].diameter >= 40.0,
        {
            "expected_retained_components": 1,
            "observed_retained_components": len(retained),
            "expected_minimum_length": 40.0,
            "largest_component_edges": largest.edge_count,
            "largest_component_diameter": largest.diameter,
            "largest_component_line_support_fraction": largest.linear_fraction,
            "active_core_edge_fraction": bend.semantic_payload["fractions"]["active_core_edges"],
            "active_core_edge_fraction_wall": CONFIG.active_fraction_maximum,
        },
    )
    results.append(bend_row)

    single_row, single = _run("single_skin", bent_double_skin(single=True))
    record(
        "single_skin",
        single.decision == "AMHC_V1_PAIRABILITY_ARCHITECTURE_STOP",
        {"decision": single.decision},
    )
    results.append(single_row)

    crop_row, crop = _run("bent_sheet_open_crop", bent_double_skin(crop=True))
    record(
        "bent_sheet_open_crop",
        crop.semantic_payload["counts"]["retained_components"] == 1,
        {
            "retained_components": crop.semantic_payload["counts"]["retained_components"],
            "active_core_edge_fraction": crop.semantic_payload["fractions"]["active_core_edges"],
        },
    )
    results.append(crop_row)

    hole_row, hole = _run("flat_double_skin_through_hole", flat_double_skin(hole=True))
    record(
        "flat_double_skin_through_hole",
        hole.semantic_payload["counts"]["retained_components"] == 0,
        {"retained_components": hole.semantic_payload["counts"]["retained_components"]},
    )
    results.append(hole_row)

    end_row, end = _run("flat_double_skin_terminal_wall", flat_double_skin(end_wall=True))
    record(
        "flat_double_skin_terminal_wall",
        end.semantic_payload["counts"]["retained_components"] == 0,
        {"retained_components": end.semantic_payload["counts"]["retained_components"]},
    )
    results.append(end_row)

    permutation = np.asarray(
        sorted(
            range(len(bend_points)),
            key=lambda index: (_hash_u64(bend_points[index], b"AMHC_V1_META_PERM\0"), index),
        ),
        dtype=np.int64,
    )
    perm_row, perm = _run("metamorphic_permutation", bend_points[permutation])
    results.append(perm_row)

    angle = math.radians(37.0)
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transformed = bend_points @ rotation.T + np.asarray((17.0, -23.0, 11.0))
    rot_row, rotated = _run("metamorphic_rigid_transform", transformed)
    results.append(rot_row)
    scalar_deltas = {
        key: abs(
            rotated.semantic_payload["fractions"][key]
            - bend.semantic_payload["fractions"][key]
        )
        for key in ("pilot_mode", "pair_participation", "core_coverage", "active_core_edges")
    }
    record(
        "metamorphic_checks",
        perm.semantic_payload["hashes"] == bend.semantic_payload["hashes"]
        and perm.decision == bend.decision
        and rotated.decision == bend.decision
        and all(value <= 1e-9 for value in scalar_deltas.values()),
        {
            "permutation_hashes_equal": perm.semantic_payload["hashes"]
            == bend.semantic_payload["hashes"],
            "permutation_terminal_equal": perm.decision == bend.decision,
            "rigid_transform_terminal_equal": rotated.decision == bend.decision,
            "rigid_transform_normalized_scalar_deltas": scalar_deltas,
            "maximum_allowed_delta": 1e-9,
        },
    )

    passed = all(row["passed"] for row in checks)
    evidence = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "fixture_run_count": len(results),
        "frozen_check_count": len(checks),
        "checks": checks,
        "regressions_passed": passed,
        "terminal_decision": (
            "AMHC_V1_SOURCE_FREE_REGRESSIONS_PASS"
            if passed
            else "AMHC_V1_SOURCE_FREE_REGRESSION_STOP"
        ),
        "source_opened": False,
        "source_claim_consumed": False,
        "target_gold_cad_benchmark_opened": False,
        "results": results,
    }
    evidence["evidence_sha256"] = sha_json(evidence)
    output = ROOT / "docs/research/AMHC_V1_SOURCE_FREE_REGRESSIONS_V1_20260809.json"
    output.write_bytes(canonical_json_bytes(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
