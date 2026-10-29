#!/usr/bin/env python3
"""Post-inference support-gate sensitivity diagnostic for simple V11 families.

The sweep varies only the recovery support radius.  It uses the fixed three
simple-family cases and the same retained poses, native topology, limits, and
unchanged matcher as the focused diagnostic.  It is diagnostic evidence for a
mechanism, not a promoted candidate or a scorer change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
from backend.cad_guided_v11_topology_recovery_v2 import (
    recover_scan_supported_axes_v2,
    select_pose_v2,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.run_gold11_v11_multi_scan_luna_max_v1 import (
    CATALOG_PATH,
    CATALOG_SHA256,
    REVIEW_PATH,
    REVIEW_SHA256,
    SELECTED_REFERENCE_SHA256,
    V11_REVIEW_RESULT_SHA256,
    _empty_recovery,
    _pose_evidence,
    _read_cases,
    _selected_references,
)


PROTOCOL_PATH = ROOT / "docs/research/2026-09-16_gold11_v11_simple_family_support_radius_sweep_protocol.md"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_simple_family_support_radius_sweep_v1"
SIMPLE_PART_KEYS = ("37361005", "49001000", "49024000")
SIMPLE_CASE_TOKENS = ("CG4BDD1371B6", "CG191DEC7F81", "CG4FAF273093")
SUPPORT_RADII_MM = (3.0, 3.5, 4.0, 4.5, 5.0)
MAX_ENGINE_ROWS = 50_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _simple_cases(all_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row["part_key"]: row for row in all_cases}
    rows = [by_key[key] for key in SIMPLE_PART_KEYS]
    if tuple(row["case_token"] for row in rows) != SIMPLE_CASE_TOKENS:
        raise ValueError("SIMPLE_FAMILY_SCOPE_IDENTITY_MISMATCH")
    return rows


def _compact_recovery(recovery: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in recovery.get("records", []):
        support = record.get("scan_support") or {}
        records.append({
            "bend_id": record.get("bend_id"),
            "accepted": record.get("accepted"),
            "rejection_reason": record.get("rejection_reason"),
            "support_fraction": support.get("support_fraction"),
            "longest_support_run_fraction": support.get("longest_support_run_fraction"),
            "median_distance_mm": support.get("median_distance_mm"),
            "p90_distance_mm": support.get("p90_distance_mm"),
            "supported_span_mm": support.get("supported_span_mm"),
        })
    return {
        "status": recovery.get("status"),
        "accepted_count": recovery.get("accepted_count"),
        "native_axis_count": recovery.get("native_axis_count"),
        "predictions": recovery.get("predictions", []),
        "records": records,
        "reason": recovery.get("reason"),
        "pose_selection": recovery.get("pose_selection"),
    }


def _score(rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in (
        "gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}
    for row in rows:
        score = match_lines(references[row["case_token"]]["bends"],
                            row["recovery"].get("predictions", []), Thresholds())
        row["score"] = {**score, "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST"}
        for key in totals:
            totals[key] += int(score[key])
    precision = totals["true_positive"] / totals["prediction_count"] if totals["prediction_count"] else None
    recall = totals["true_positive"] / totals["gold_count"] if totals["gold_count"] else None
    denominator = 2 * totals["true_positive"] + totals["false_positive"] + totals["false_negative"]
    f1 = 2.0 * totals["true_positive"] / denominator if denominator else None
    return {**totals, "precision": precision, "recall": recall, "f1": f1,
            "bend_count_capture": recall,
            "target_80_percent_met": bool(recall is not None and recall >= 0.80),
            "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST"}


def run(output: Path) -> dict[str, Any]:
    all_cases = _read_cases()
    cases = _simple_cases(all_cases)
    by_radius: dict[str, list[dict[str, Any]]] = {str(radius): [] for radius in SUPPORT_RADII_MM}
    scan_cache: dict[str, dict[str, Any]] = {}
    # No Gold/reference object is loaded in this loop.
    for case in cases:
        token = case["case_token"]
        retained, retained_path, retained_sha = _pose_evidence(token)
        if token == "CG4BDD1371B6" and retained_sha != V11_REVIEW_RESULT_SHA256:
            raise ValueError("RETAINED_V11_CONTROL_RESULT_HASH_MISMATCH")
        registration = retained.get("registration") or {}
        evidence = registration.get("evidence") or {}
        pose = select_pose_v2(evidence)
        if pose.get("status") == "AVAILABLE":
            scan = scan_cache.get(token)
            if scan is None:
                scan = prepare_mesh_scan_v1(scan_bytes=Path(case["scan_path"]).read_bytes(),
                                            max_engine_rows=MAX_ENGINE_ROWS)
                scan_cache[token] = scan
        else:
            scan = None
        for radius in SUPPORT_RADII_MM:
            try:
                if pose.get("status") != "AVAILABLE":
                    recovery = _empty_recovery(pose.get("reason", "NO_V11_POSE"))
                    recovery["pose_selection"] = {key: value for key, value in pose.items()
                                                   if key not in {"rotation", "translation"}}
                elif scan.get("status") != "AVAILABLE":
                    raise ValueError("V11_SCAN_PREPARATION_UNAVAILABLE")
                else:
                    recovery = recover_scan_supported_axes_v2(
                        scan["raw_points"], Path(case["cad_path"]), evidence,
                        support_radius_mm=radius)
            except Exception as error:
                recovery = _empty_recovery("INFERENCE_EXCEPTION:" + type(error).__name__)
            by_radius[str(radius)].append({
                "case_token": token,
                "part_key": case["part_key"],
                "reference_id": case["reference_id"],
                "radius_mm": radius,
                "input_bindings": {
                    "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"],
                    "cad_path": case["cad_path"], "cad_sha256": case["cad_sha256"],
                    "gold_used_for_inference": False,
                },
                "retained_v11_control": {
                    "path": str(retained_path.relative_to(ROOT)),
                    "sha256": retained_sha,
                    "status": retained.get("status"),
                    "registration_status": registration.get("status"),
                    "measurement_context_indices": evidence.get("measurement_context_indices", []),
                },
                "recovery": _compact_recovery(recovery),
            })

    # Gold/reference lines enter only after every radius/case inference.
    references = _selected_references(all_cases)
    score_by_radius = {}
    for radius, rows in by_radius.items():
        score_by_radius[radius] = _score(rows, references)
    result = {
        "schema": "GOLD11_V11_SIMPLE_FAMILY_SUPPORT_RADIUS_SWEEP_RESULT_V1",
        "candidate_implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
        "scope": {
            "selection_rule": "FIXED_PART_KEYS_37361005_49001000_49024000",
            "case_count": len(cases), "family_count": len(SIMPLE_PART_KEYS),
            "case_tokens": list(SIMPLE_CASE_TOKENS), "part_keys": list(SIMPLE_PART_KEYS),
            "radius_values_mm": list(SUPPORT_RADII_MM),
            "full_gold11_case_count_authenticated": len(all_cases),
            "review_source_sha256": REVIEW_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "selected_reference_sha256": SELECTED_REFERENCE_SHA256,
        },
        "inference": {
            "gold_used_for_inference": False,
            "terminal_v11_reopened": False,
            "max_engine_rows": MAX_ENGINE_ROWS,
            "pose_policy": {"max_provisional_p95_upper_mm": 2.0,
                            "min_provisional_coverage_lower": 0.55},
        },
        "score_by_radius": score_by_radius,
        "cases_by_radius": by_radius,
        "historical_score": 0.4580152672,
        "historical_comparability": "UNESTABLISHED",
        "historical_score_beaten": False,
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "uncertainty_status": "UNCALIBRATED_PROVISIONAL_RETAINED_POSE_AND_SCAN_SUPPORT_DISTANCES",
        "limitations": [
            "This is consumed-development mechanism evidence on three fixed families, not a qualification run.",
            "Changing support radius changes candidate acceptance and is not a production recommendation.",
            "Native STEP axes remain CAD-targeted scan-supported contact-line proposals.",
            "The matcher and thresholds remain unchanged; subset scores do not replace the 71-line headline.",
        ],
        "source_bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "recovery_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v2.py"),
            "review_source_sha256": _sha256(REVIEW_PATH),
            "catalog_sha256": _sha256(CATALOG_PATH),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "CHECKSUMS.json").write_text(json.dumps({
        "schema": "GOLD11_V11_SIMPLE_FAMILY_SUPPORT_RADIUS_SWEEP_CHECKSUMS_V1",
        "result_sha256": _sha256(output / "RESULT.json"),
        "source_bindings": result["source_bindings"],
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY_SUPPORT_RADIUS_SWEEP",
                      "score_by_radius": result["score_by_radius"],
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
