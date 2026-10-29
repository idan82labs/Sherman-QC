#!/usr/bin/env python3
"""Execute all frozen CFAC-V1 source-free gates, stopping C006 on any failure."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cfac_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json
from research.f1_cfac_v1.core import canonicalize_xyz, evaluate
from research.f1_cfac_v1.fixtures import (
    concentric_bend,
    flat_double_skin,
    positive_plus_hole_stress,
    terminal_geometry,
)


def _axis_metrics(candidate, expected_origin=np.zeros(3), expected_direction=np.asarray((0.0, 1.0, 0.0))):
    direction_error = math.degrees(
        math.acos(float(np.clip(abs(np.dot(candidate.direction, expected_direction)), -1.0, 1.0)))
    )
    delta = candidate.origin - expected_origin
    distance = float(np.linalg.norm(delta - candidate.direction * np.dot(delta, candidate.direction)))
    return direction_error, distance


def _run(name: str, points: np.ndarray):
    start = perf_counter()
    source = canonicalize_xyz(points)
    evaluation = evaluate(source)
    return {
        "fixture": name,
        "point_count": len(source.xyz),
        "wall_seconds": perf_counter() - start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "decision": evaluation.decision,
        "counts": evaluation.semantic_payload["counts"],
        "candidate_table": evaluation.semantic_payload["candidate_table"],
        "semantic_sha256": evaluation.semantic_payload["semantic_sha256"],
    }, evaluation


def _hash_permutation(points: np.ndarray) -> np.ndarray:
    keys = np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(
                    b"CFAC_V1_METAMORPHIC_PERM\0"
                    + np.ascontiguousarray(row, dtype="<f8").tobytes()
                ).digest()[:8],
                "little",
            )
            for row in points
        ),
        dtype=np.uint64,
        count=len(points),
    )
    return np.lexsort((np.arange(len(points)), keys))


def main() -> int:
    rows: list[dict] = []
    checks: list[dict] = []

    def record(name: str, passed: bool, evidence: dict) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    full_points = concentric_bend()
    row, full = _run("full_positive", full_points)
    rows.append(row)
    retained = [value for value in full.candidates if value.retained]
    direction_error, distance = _axis_metrics(retained[0]) if len(retained) == 1 else (math.inf, math.inf)
    radius_errors = (
        [abs(retained[0].radii[0] - 8.0) / full.jets.s, abs(retained[0].radii[1] - 16.0) / full.jets.s]
        if len(retained) == 1 and retained[0].radii is not None
        else [math.inf, math.inf]
    )
    record(
        "full_positive",
        len(retained) == 1 and direction_error <= 2 and distance <= full.jets.s and max(radius_errors) <= 0.75,
        {"final_axes": len(retained), "direction_error_degrees": direction_error, "line_distance_over_s": distance / full.jets.s, "radius_errors_over_s": radius_errors},
    )

    row, crop = _run(
        "axial_crop_positive",
        concentric_bend(axis_low=17.0, axis_high=79.0),
    )
    rows.append(row)
    record("axial_crop_positive", crop.semantic_payload["counts"]["final_axes"] == 1, {"counts": crop.semantic_payload["counts"]})

    row, scaled = _run(
        "scale_angle_positive",
        concentric_bend(radii=(21.0, 27.0), angle_degrees=60.0, flange_length=48.0, pitch=1.2),
    )
    rows.append(row)
    scaled_retained = [value for value in scaled.candidates if value.retained]
    record("scale_angle_positive", len(scaled_retained) == 1, {"counts": scaled.semantic_payload["counts"], "candidates": scaled.semantic_payload["candidate_table"]})

    row, flat = _run("flat_double_skin", flat_double_skin())
    rows.append(row)
    record("flat_double_skin", flat.semantic_payload["counts"]["final_axes"] == 0, {"counts": flat.semantic_payload["counts"]})

    row, single = _run("single_skin_90deg_bend", concentric_bend(radii=(12.0,)))
    rows.append(row)
    record(
        "single_skin_90deg_bend",
        single.semantic_payload["counts"]["provisional_axes"] >= 1 and single.semantic_payload["counts"]["final_axes"] == 0,
        {"counts": single.semantic_payload["counts"], "candidates": single.semantic_payload["candidate_table"]},
    )

    row, hole = _run(
        "flat_double_skin_through_hole",
        flat_double_skin(hole_radius=12.0, include_hole_wall=True),
    )
    rows.append(row)
    record("flat_double_skin_through_hole", hole.semantic_payload["counts"]["final_axes"] == 0, {"counts": hole.semantic_payload["counts"], "candidates": hole.semantic_payload["candidate_table"]})

    row, terminal = _run("terminal_geometry", terminal_geometry())
    rows.append(row)
    record("terminal_geometry", terminal.semantic_payload["counts"]["final_axes"] == 0, {"counts": terminal.semantic_payload["counts"], "candidates": terminal.semantic_payload["candidate_table"]})

    row, cylinder = _run(
        "concentric_full_cylinder_decoy",
        concentric_bend(angle_degrees=360.0, flange_length=0.0),
    )
    rows.append(row)
    record(
        "concentric_full_cylinder_decoy",
        cylinder.semantic_payload["counts"]["provisional_axes"] >= 1
        and cylinder.semantic_payload["counts"]["final_axes"] == 0
        and any(value.rejection == "OCCUPIED_ARC_WIDTH" for value in cylinder.candidates),
        {"counts": cylinder.semantic_payload["counts"], "rejections": [value.rejection for value in cylinder.candidates]},
    )

    permutation_row, permutation = _run("metamorphic_permutation", full_points[_hash_permutation(full_points)])
    rows.append(permutation_row)
    permutation_equal = permutation.semantic_payload == full.semantic_payload
    transform_checks = []
    transforms = [
        (17.0, np.asarray((11.0, -7.0, 23.0))),
        (53.0, np.asarray((-19.0, 31.0, -5.0))),
        (101.0, np.asarray((7.0, 13.0, -29.0))),
    ]
    for index, (degrees, translation) in enumerate(transforms):
        angle = math.radians(degrees)
        rotation = np.asarray(((math.cos(angle), 0.0, math.sin(angle)), (0.0, 1.0, 0.0), (-math.sin(angle), 0.0, math.cos(angle))))
        transformed = full_points @ rotation.T + translation
        row, evaluated = _run(f"metamorphic_rigid_{index}", transformed)
        rows.append(row)
        final = [value for value in evaluated.candidates if value.retained]
        if len(final) == 1:
            mapped_origin = (final[0].origin - translation) @ rotation
            mapped_direction = final[0].direction @ rotation
            direction_error, line_distance = _axis_metrics(final[0], translation, np.asarray((0.0, 1.0, 0.0)))
            radii_relative = [
                abs(final[0].radii[0] - retained[0].radii[0]) / retained[0].radii[0],
                abs(final[0].radii[1] - retained[0].radii[1]) / retained[0].radii[1],
            ]
        else:
            direction_error, line_distance, radii_relative = math.inf, math.inf, [math.inf, math.inf]
        transform_checks.append({"final_axes": len(final), "direction_error_degrees": direction_error, "line_distance_over_s": line_distance / evaluated.jets.s, "radii_relative": radii_relative})
    record(
        "metamorphic",
        permutation_equal
        and all(value["final_axes"] == 1 and value["direction_error_degrees"] <= 2 and value["line_distance_over_s"] <= 1 and max(value["radii_relative"]) <= 0.02 for value in transform_checks),
        {"permutation_byte_semantic_equal": permutation_equal, "rigid_transforms": transform_checks},
    )

    # Stress is last and runs only if every scientific fixture above passed.
    if all(value["passed"] for value in checks):
        stress_start = perf_counter()
        row, stress = _run("positive_plus_hole_stress_approximately_200k", positive_plus_hole_stress())
        stress_wall = perf_counter() - stress_start
        rows.append(row)
        record(
            "resource_stress",
            stress_wall < CONFIG.source_free_wall_seconds
            and row["peak_rss_bytes"] < CONFIG.source_free_rss_ceiling_bytes,
            {"wall_seconds": stress_wall, "peak_rss_bytes": row["peak_rss_bytes"], "point_count": row["point_count"], "counts": stress.semantic_payload["counts"]},
        )
    else:
        checks.append({"check": "resource_stress", "passed": False, "not_evaluated_due_to_upstream_stop": True})

    passed = all(value["passed"] for value in checks)
    evidence = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "checks": checks,
        "results": rows,
        "regressions_passed": passed,
        "decision": "CFAC_V1_SOURCE_FREE_REGRESSIONS_PASS" if passed else "CFAC_V1_SOURCE_FREE_REGRESSION_ARCHITECTURE_STOP",
        "c006_source_opened": False,
        "c006_claim_created": False,
        "c006_claim_consumed": False,
        "target_gold_cad_benchmark_opened": False,
    }
    evidence["evidence_sha256"] = sha_json(evidence)
    output = ROOT / "docs/research/CFAC_V1_SOURCE_FREE_REGRESSIONS_20260809.json"
    output.write_bytes(canonical_json_bytes(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
