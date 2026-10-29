#!/usr/bin/env python3
"""Run local carrier search only for events with incomplete nominal support."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EXP19_PATH = HERE / "run_exp_cad_0019_local_carrier_search.py"
SPEC = importlib.util.spec_from_file_location("exp19_local_search", EXP19_PATH)
EXP19 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP19)
ORIGINAL_SEARCH = EXP19.search_local_carrier


def gated_search_local_carrier(points, event, *, thickness_mm, max_search_points=40000):
    nominal_metric, nominal_mask = EXP19.EXP18.evaluate_carrier_support(
        points, event, thickness_mm=thickness_mm, min_points_per_bin=3
    )
    nominal_coverage = float(nominal_metric["carrier_coverage_fraction"])
    if nominal_coverage >= 1.0:
        result = {
            "event_id": str(event["event_id"]),
            "search_status": "NOMINAL_SUPPORT_SATURATED_SEARCH_FORBIDDEN",
            "search_limit_mm": max(6.0, 6.0 * float(thickness_mm)),
            "coarse_step_mm": max(1.0, float(thickness_mm)),
            "refinement_step_mm": max(1.0, float(thickness_mm)) / 4.0,
            "local_search_point_count": 0,
            "search_sample_point_count": 0,
            "offset_basis_first": None,
            "offset_basis_second": None,
            "best_offset_coordinates_mm": [0.0, 0.0],
            "best_shift_vector_mm": [0.0, 0.0, 0.0],
            "best_shift_norm_mm": 0.0,
            "nominal_coverage_fraction": nominal_coverage,
            "best_coverage_fraction": nominal_coverage,
            "coverage_gain": 0.0,
            "best_median_radial_residual_mm": nominal_metric["median_carrier_radial_residual_mm"],
            "best_carrier_point_count": nominal_metric["carrier_point_count"],
        }
        return result, dict(event), nominal_mask
    result, shifted, mask = ORIGINAL_SEARCH(
        points,
        event,
        thickness_mm=thickness_mm,
        max_search_points=max_search_points,
    )
    minimum_gain = 1.0 / float(nominal_metric["bin_count"])
    if float(result["coverage_gain"]) + 1e-12 < minimum_gain:
        result.update(
            {
                "search_status": "NO_ONE_BIN_GAIN_RETAIN_NOMINAL",
                "best_offset_coordinates_mm": [0.0, 0.0],
                "best_shift_vector_mm": [0.0, 0.0, 0.0],
                "best_shift_norm_mm": 0.0,
                "best_coverage_fraction": nominal_coverage,
                "coverage_gain": 0.0,
                "best_median_radial_residual_mm": nominal_metric["median_carrier_radial_residual_mm"],
                "best_carrier_point_count": nominal_metric["carrier_point_count"],
            }
        )
        return result, dict(event), nominal_mask
    result["search_status"] = "SEARCHED_INCOMPLETE_NOMINAL_SUPPORT_ONE_BIN_GAIN"
    return result, shifted, mask


def main() -> int:
    EXP19.search_local_carrier = gated_search_local_carrier
    argv = sys.argv[1:]
    output_root = Path(argv[argv.index("--output-root") + 1])
    code = EXP19.main(argv)
    summary_path = output_root / "SOURCE_ONLY_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in summary["records"]:
        case_path = output_root / "cases" / record["case_token"] / "semantic_result.json"
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        payload["schema"] = "EXP_CAD_0020_GATED_LOCAL_CARRIER_SEARCH_CASE_V1"
        payload["experiment_id"] = "EXP_CAD_0020"
        payload["search_policy"] = "SEARCH_ONLY_INCOMPLETE_NOMINAL_SUPPORT_REQUIRE_AT_LEAST_ONE_BIN_GAIN_V1"
        EXP19.EXP18.atomic_json(case_path, payload)
        record["semantic_sha256"] = EXP19.EXP18.sha256_path(case_path)
    summary["schema"] = "EXP_CAD_0020_GATED_LOCAL_CARRIER_SEARCH_SUMMARY_V1"
    summary["experiment_id"] = "EXP_CAD_0020"
    summary["search_policy"] = "SEARCH_ONLY_INCOMPLETE_NOMINAL_SUPPORT_REQUIRE_AT_LEAST_ONE_BIN_GAIN_V1"
    summary["generalization_progress_indicator"] = "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE"
    EXP19.EXP18.atomic_json(summary_path, summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
