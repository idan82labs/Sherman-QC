#!/usr/bin/env python3
"""Run the V11 healing candidate on the fixed low-complexity Gold11 families.

This is a diagnostic development run over an explicit subset of the already
consumed, hash-bound Gold11 cohort.  The subset is fixed by part identity:
37361005, 49001000, and 49024000.  It is not selected from a mutable manifest
or by inspecting outcomes.  The full Gold11 case list and reference snapshot
are still authenticated, and Gold is loaded only after all inference attempts
have returned.  The unchanged matcher is used for the post-inference score.
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
    ATTEMPT_ROOT,
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


PROTOCOL_PATH = ROOT / "docs/research/2026-09-16_gold11_v11_simple_family_diagnostic_protocol.md"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_simple_family_diagnostic_v1"
SIMPLE_PART_KEYS = ("37361005", "49001000", "49024000")
SIMPLE_CASE_TOKENS = ("CG4BDD1371B6", "CG191DEC7F81", "CG4FAF273093")
MAX_ENGINE_ROWS = 50_000
SUPPORT_RADIUS_MM = 3.0
LINE_SAMPLES = 201


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _simple_cases(all_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in all_cases if row["part_key"] in SIMPLE_PART_KEYS]
    if tuple(row["case_token"] for row in rows) != SIMPLE_CASE_TOKENS:
        raise ValueError("SIMPLE_FAMILY_SCOPE_ORDER_OR_IDENTITY_MISMATCH")
    if len(rows) != 3 or len({row["part_key"] for row in rows}) != 3:
        raise ValueError("SIMPLE_FAMILY_SCOPE_CARDINALITY_MISMATCH")
    return rows


def _case_failure_mode(recovery: dict[str, Any], score: dict[str, Any]) -> str:
    if recovery.get("status") == "UNAVAILABLE":
        return str(recovery.get("reason") or "RECOVERY_UNAVAILABLE")
    if int(score["false_positive"]) and int(score["false_negative"]):
        return "PARTIAL_WITH_FALSE_POSITIVES_AND_FALSE_NEGATIVES"
    if int(score["false_negative"]):
        return "PARTIAL_WITH_MISSED_REFERENCES"
    if int(score["false_positive"]):
        return "EXTRA_UNMATCHED_PREDICTIONS"
    return "ALL_REFERENCES_MATCHED"


def _prediction_summary(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for prediction in recovery.get("predictions", []):
        data = prediction.get("bend_data") or {}
        result.append({
            "bend_id": prediction.get("bend_id"),
            "line_start": prediction.get("line_start"),
            "line_end": prediction.get("line_end"),
            "line_span_mm": data.get("line_span_mm"),
            "support_fraction": data.get("support_fraction"),
            "longest_support_run_fraction": data.get("longest_support_run_fraction"),
            "median_scan_distance_mm": data.get("median_scan_distance_mm"),
            "p90_scan_distance_mm": data.get("p90_scan_distance_mm"),
            "observability_state": prediction.get("observability_state"),
        })
    return result


def _score(rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in (
        "gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}
    for row in rows:
        score = match_lines(references[row["case_token"]]["bends"],
                            row["recovery"].get("predictions", []), Thresholds())
        row["score"] = {**score, "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST"}
        row["failure_mode"] = _case_failure_mode(row["recovery"], score)
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
    # Authenticate the complete frozen cohort before narrowing to the fixed
    # diagnostic subset.  No Gold/reference object is opened in this section.
    all_cases = _read_cases()
    cases = _simple_cases(all_cases)
    case_rows: list[dict[str, Any]] = []
    inference_complete = True
    for case in cases:
        token = case["case_token"]
        retained, retained_path, retained_sha = _pose_evidence(token)
        if token == "CG4BDD1371B6" and retained_sha != V11_REVIEW_RESULT_SHA256:
            raise ValueError("RETAINED_V11_CONTROL_RESULT_HASH_MISMATCH")
        registration = retained.get("registration") or {}
        evidence = registration.get("evidence") or {}
        input_bindings = {
            "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"],
            "cad_path": case["cad_path"], "cad_sha256": case["cad_sha256"],
            "gold_used_for_inference": False,
        }
        try:
            pose = select_pose_v2(evidence)
            if pose.get("status") != "AVAILABLE":
                recovery = _empty_recovery(pose.get("reason", "NO_V11_POSE"))
                recovery["pose_selection"] = {key: value for key, value in pose.items()
                                               if key not in {"rotation", "translation"}}
                raw_point_count = None
            else:
                scan_bytes = Path(case["scan_path"]).read_bytes()
                scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes, max_engine_rows=MAX_ENGINE_ROWS)
                if scan.get("status") != "AVAILABLE":
                    raise ValueError("V11_SCAN_PREPARATION_UNAVAILABLE")
                raw_point_count = int(len(scan["raw_points"]))
                recovery = recover_scan_supported_axes_v2(
                    scan["raw_points"], Path(case["cad_path"]), evidence)
            inference_complete = inference_complete and recovery.get("status") in {"PARTIAL", "UNAVAILABLE"}
        except Exception as error:
            recovery = _empty_recovery("INFERENCE_EXCEPTION:" + type(error).__name__)
            raw_point_count = None
            inference_complete = False
        case_rows.append({
            "case_token": token,
            "part_key": case["part_key"],
            "reference_id": case["reference_id"],
            "input_bindings": input_bindings,
            "retained_v11_control": {
                "path": str(retained_path.relative_to(ROOT)),
                "sha256": retained_sha,
                "status": retained.get("status"),
                "phase": retained.get("phase"),
                "registration_status": registration.get("status"),
                "measurement_context_indices": evidence.get("measurement_context_indices", []),
            },
            "raw_point_count_after_preparation": raw_point_count,
            "pose_selection": {key: value for key, value in (recovery.get("pose_selection") or {}).items()
                               if key not in {"rotation", "translation"}},
            "recovery": recovery,
            "prediction_summary": _prediction_summary(recovery),
        })

    # Gold/reference lines enter only after every selected inference attempt.
    references = _selected_references(all_cases)
    selected_references = {case["case_token"]: references[case["case_token"]] for case in cases}
    score = _score(case_rows, selected_references)
    family_rows = []
    for part_key in SIMPLE_PART_KEYS:
        subset = [row for row in case_rows if row["part_key"] == part_key]
        family_score = _score(subset, selected_references)
        family_rows.append({
            "part_key": part_key,
            "case_tokens": [row["case_token"] for row in subset],
            "reference_line_count": family_score["gold_count"],
            "score": family_score,
            "failure_modes": [row["failure_mode"] for row in subset],
        })
    result = {
        "schema": "GOLD11_V11_SIMPLE_FAMILY_DIAGNOSTIC_RESULT_V1",
        "candidate_implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
        "scope": {
            "selection_rule": "FIXED_PART_KEYS_37361005_49001000_49024000",
            "case_count": len(cases),
            "family_count": len(SIMPLE_PART_KEYS),
            "case_tokens": list(SIMPLE_CASE_TOKENS),
            "part_keys": list(SIMPLE_PART_KEYS),
            "reference_line_count": score["gold_count"],
            "full_gold11_case_count_authenticated": len(all_cases),
            "review_source_sha256": REVIEW_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "selected_reference_sha256": SELECTED_REFERENCE_SHA256,
        },
        "inference": {
            "complete": inference_complete,
            "gold_used_for_inference": False,
            "terminal_v11_reopened": False,
            "max_engine_rows": MAX_ENGINE_ROWS,
            "support_radius_mm": SUPPORT_RADIUS_MM,
            "line_samples": LINE_SAMPLES,
        },
        "score": score,
        "per_family": family_rows,
        "cases": case_rows,
        "historical_score": 0.4580152672,
        "historical_comparability": "UNESTABLISHED",
        "historical_score_beaten": False,
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "uncertainty_status": "UNCALIBRATED_PROVISIONAL_RETAINED_POSE_AND_SCAN_SUPPORT_DISTANCES",
        "limitations": [
            "The subset is a fixed consumed-development diagnostic, not an unseen-family test.",
            "Native STEP axes remain CAD-targeted scan-supported contact-line proposals.",
            "The line scorer and thresholds are unchanged; no CAD nominal measurement is scored as observed geometry.",
            "This subset score must not be compared as a replacement for the 71-line Gold11 headline score.",
        ],
        "source_bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "recovery_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v2.py"),
            "recovery_base_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v1.py"),
            "review_source_sha256": _sha256(REVIEW_PATH),
            "catalog_sha256": _sha256(CATALOG_PATH),
            "retained_attempt_root": str(ATTEMPT_ROOT.relative_to(ROOT)),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checksums = {
        "schema": "GOLD11_V11_SIMPLE_FAMILY_DIAGNOSTIC_CHECKSUMS_V1",
        "result_sha256": _sha256(output / "RESULT.json"),
        "source_bindings": result["source_bindings"],
    }
    (output / "CHECKSUMS.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY_SIMPLE_FAMILIES",
                      "scope": result["scope"], "score": result["score"],
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
