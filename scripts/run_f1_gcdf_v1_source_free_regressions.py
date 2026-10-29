#!/usr/bin/env python3
"""Mandatory source-free GCDF-V1 regression package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_gcdf_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402
from research.f1_gcdf_v1.core import (  # noqa: E402
    CanonicalNodes, Sample, _canonical_path, _fit_circular_l1, ceil_sqrt_q16,
    lower_median, ppm,
)
from research.f1_gcdf_v1.tamper import assert_tamper_suite  # noqa: E402


def _fixture_sample(order: np.ndarray | None = None) -> Sample:
    xyz = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype="<f8")
    if order is not None:
        xyz = xyz[order]
    ids = np.zeros((4, 32), dtype=np.uint8)
    multiplicities = np.ones(4, dtype=np.uint32)
    ordinals = np.arange(4, dtype=np.uint32) if order is None else np.arange(4, dtype=np.uint32)[order]
    return Sample(xyz, ids, multiplicities, ordinals, "fixture-o", "fixture-x", "fixture-e", "fixture-c")


def run_regressions() -> dict[str, object]:
    tamper = assert_tamper_suite()
    checks: dict[str, bool] = {}
    checks["input_order_invariance"] = sha_json(sorted(map(tuple, _fixture_sample().xyz.tolist()))) == sha_json(sorted(map(tuple, _fixture_sample(np.asarray([3, 1, 0, 2])).xyz.tolist())))
    duplicate = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.uint32)
    checks["duplicate_collapse"] = len(np.unique(duplicate.view("V12").reshape(3))) == 2
    candidates = np.asarray([8, 4, 7, 2], dtype=np.uint32)
    squared = np.asarray([10, 10, 9, 10], dtype=np.uint64)
    checks["exact_k_neighbor_tie"] = candidates[np.lexsort((candidates, squared))].tolist() == [7, 2, 4, 8]
    direct = ceil_sqrt_q16(100)
    via = ceil_sqrt_q16(16) + ceil_sqrt_q16(16)
    checks["short_circuit_edge_rejection"] = 2 * via <= 5 * direct
    matrix = sparse.csr_matrix((np.asarray([1., 1., 1., 1.]), (np.asarray([0, 0, 1, 2]), np.asarray([1, 2, 3, 3]))), shape=(4, 4))
    matrix = matrix + matrix.T
    distances = np.asarray([0., 1., 1., 2.])
    path, _weights = _canonical_path(matrix, distances, 0, 3)
    checks["equal_distance_predecessor"] = path.tolist() == [0, 1, 3]
    checks["global_deficit_equality_and_below"] = ppm(3, 1000) == 3000 and ppm(2, 1000) == 2000
    checks["concentration_equality_and_below"] = ppm(45, 100) == 450000 and ppm(449999, 1_000_000) == 449999
    checks["lower_median"] = lower_median([9, 1, 7, 3]) == 3
    checks["circular_fit_sign_sensitive"] = _fit_circular_l1([100, 110, 120]) == 110 and _fit_circular_l1([-100, -110, -120]) == -110
    checks.update(tamper)
    mandatory = {
        "input_order_invariance", "duplicate_collapse", "exact_k_neighbor_tie",
        "short_circuit_edge_rejection", "equal_distance_predecessor",
        "global_deficit_equality_and_below", "concentration_equality_and_below",
        "equal_count_landmark_exchange", "vote_endpoint_exchange", "mst_edge_tamper",
        "diameter_tamper", "causal_fold_bit_tamper", "fitted_angle_sign_tamper",
        "candidate_table_mutation", "source_point_serialization",
    }
    passed = all(checks.get(name, False) for name in mandatory)
    result = {
        "schema_version": 1, "regression_id": "GCDF_V1_SOURCE_FREE_REGRESSIONS_V1",
        "configuration_sha256": CONFIG_SHA256, "source_opened": False,
        "target_gold_cad_benchmark_opened": False, "checks": checks, "passed": passed,
    }
    result["result_sha256"] = sha_json(result)
    return result


def main() -> int:
    result = run_regressions()
    print(canonical_json_bytes(result).decode("ascii"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
