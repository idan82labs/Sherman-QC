#!/usr/bin/env python3
"""Fail-fast execution of the first mandatory CFAC-V1 positive fixture."""

from __future__ import annotations

import json
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
from research.f1_cfac_v1.fixtures import concentric_bend


def _axis_error_degrees(direction: np.ndarray) -> float:
    expected = np.asarray((0.0, 1.0, 0.0))
    cosine = float(np.clip(abs(np.dot(direction, expected)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _line_distance(origin: np.ndarray, direction: np.ndarray) -> float:
    expected_origin = np.zeros(3)
    delta = origin - expected_origin
    return float(np.linalg.norm(delta - direction * np.dot(delta, direction)))


def main() -> int:
    start = perf_counter()
    source = canonicalize_xyz(concentric_bend())
    evaluation = evaluate(source)
    retained = [row for row in evaluation.candidates if row.retained]
    details = []
    for row in retained:
        details.append(
            {
                "candidate_id": row.candidate_id,
                "axis_error_degrees": _axis_error_degrees(row.direction),
                "line_distance": _line_distance(row.origin, row.direction),
                "recovered_radii": row.radii,
                "radius_errors_over_s": (
                    None
                    if row.radii is None
                    else [
                        abs(row.radii[0] - 8.0) / evaluation.jets.s,
                        abs(row.radii[1] - 16.0) / evaluation.jets.s,
                    ]
                ),
                "continuation_occupancies": row.continuation_occupancies,
            }
        )
    passed = (
        len(retained) == 1
        and details[0]["axis_error_degrees"] <= 2.0
        and details[0]["line_distance"] <= evaluation.jets.s
        and max(details[0]["radius_errors_over_s"]) <= 0.75
    )
    evidence = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "fixture": "FULL_POSITIVE_RADII_8_16_ANGLE_90_AXIS_96_FLANGES_40_PITCH_1",
        "point_count": len(source.xyz),
        "wall_seconds": perf_counter() - start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "semantic": evaluation.semantic_payload,
        "retained_axis_details": details,
        "passed": passed,
        "decision": (
            "CFAC_V1_FULL_POSITIVE_REGRESSION_PASS"
            if passed
            else "CFAC_V1_SOURCE_FREE_REGRESSION_ARCHITECTURE_STOP"
        ),
        "c006_source_opened": False,
        "c006_claim_created": False,
        "c006_claim_consumed": False,
    }
    evidence["evidence_sha256"] = sha_json(evidence)
    output = ROOT / "docs/research/CFAC_V1_FULL_POSITIVE_REGRESSION_20260809.json"
    output.write_bytes(canonical_json_bytes(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
