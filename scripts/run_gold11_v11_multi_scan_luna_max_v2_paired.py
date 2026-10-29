#!/usr/bin/env python3
"""Paired multi-scan evaluation of the V11 healing candidate.

This is a new development run, not a retry of the terminal V11 attempt.  Both
retained V11 arms are evaluated independently with the same V2 healing policy;
the exact consumed Gold11 scope stays fixed.  Gold is opened only after all 22
inference attempts have returned, then the unchanged matcher is applied to each
arm separately.
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
    _empty_recovery,
    _read_cases,
    _selected_references,
)


ATTEMPT_ROOT = ROOT / "docs/research/gold11_mesh_development_v11_attempt/cases"
PROTOCOL_PATH = ROOT / "docs/research/2026-09-15_gold11_v11_multi_scan_luna_max_v2_paired_protocol.md"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_multi_scan_luna_max_v2_paired"
ARMS = ("control", "challenger")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retained(token: str, arm: str) -> tuple[dict[str, Any], Path, str]:
    path = ATTEMPT_ROOT / token / arm / "RESULT.json"
    blob = path.read_bytes()
    return json.loads(blob), path, hashlib.sha256(blob).hexdigest()


def _score_arm(rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in
              ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}
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


def _case_f1(score: dict[str, Any]) -> float | None:
    """Derive the per-case F1 from the matcher counts (the matcher omits F1)."""
    denominator = 2 * int(score["true_positive"]) + int(score["false_positive"]) + int(score["false_negative"])
    return 2.0 * int(score["true_positive"]) / denominator if denominator else None


def run(output: Path) -> dict[str, Any]:
    cases = _read_cases()
    # The two arm ledgers are independent.  The raw scan is prepared once per
    # case, but no pose or prediction is shared across arms.
    arm_rows = {arm: [] for arm in ARMS}
    inference_complete = True
    for case in cases:
        token = case["case_token"]
        scan = None
        for arm in ARMS:
            retained, retained_path, retained_sha = _retained(token, arm)
            registration = retained.get("registration") or {}
            evidence = registration.get("evidence") or {}
            input_bindings = {
                "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"],
                "cad_path": case["cad_path"], "cad_sha256": case["cad_sha256"],
                "gold_used_for_inference": False,
            }
            try:
                # Gold/reference lines are not loaded in this loop.
                pose = select_pose_v2(evidence)
                if pose.get("status") != "AVAILABLE":
                    recovery = _empty_recovery(pose.get("reason", "NO_V11_POSE"))
                    recovery["pose_selection"] = {key: value for key, value in pose.items()
                                                   if key not in {"rotation", "translation"}}
                else:
                    if scan is None:
                        scan_bytes = Path(case["scan_path"]).read_bytes()
                        scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes, max_engine_rows=50_000)
                        if scan.get("status") != "AVAILABLE":
                            raise ValueError("V11_SCAN_PREPARATION_UNAVAILABLE")
                    recovery = recover_scan_supported_axes_v2(
                        scan["raw_points"], Path(case["cad_path"]), evidence)
                inference_complete = inference_complete and recovery.get("status") in {"PARTIAL", "UNAVAILABLE"}
            except Exception as error:
                recovery = _empty_recovery("INFERENCE_EXCEPTION:" + type(error).__name__)
                inference_complete = False
            arm_rows[arm].append({
                "case_token": token,
                "part_key": case["part_key"],
                "reference_id": case["reference_id"],
                "arm": arm,
                "input_bindings": input_bindings,
                "retained_v11": {
                    "path": str(retained_path.relative_to(ROOT)),
                    "sha256": retained_sha,
                    "status": retained.get("status"),
                    "phase": retained.get("phase"),
                    "registration_status": registration.get("status"),
                    "measurement_context_indices": evidence.get("measurement_context_indices", []),
                },
                "recovery": recovery,
            })

    # Post-inference scoring seam: Gold enters only after all 22 attempts.
    references = _selected_references(cases)
    scores = {arm: _score_arm(arm_rows[arm], references) for arm in ARMS}
    paired = []
    for index, case in enumerate(cases):
        control = arm_rows["control"][index]["score"]
        challenger = arm_rows["challenger"][index]["score"]
        paired.append({"case_token": case["case_token"],
                       "control_tp_fp_fn": [control["true_positive"], control["false_positive"], control["false_negative"]],
                       "challenger_tp_fp_fn": [challenger["true_positive"], challenger["false_positive"], challenger["false_negative"]],
                       "prediction_count_delta": challenger["prediction_count"] - control["prediction_count"],
                       "f1_delta": (_case_f1(challenger) or 0.0) - (_case_f1(control) or 0.0)})
    result = {
        "schema": "GOLD11_V11_MULTI_SCAN_LUNA_MAX_PAIRED_RESULT_V1",
        "candidate_implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
        "scope": {"case_count": len(cases), "arm_count": len(ARMS),
                  "family_count": len({row["part_key"] for row in cases}),
                  "reference_line_count": 71, "review_source_sha256": REVIEW_SHA256,
                  "catalog_sha256": CATALOG_SHA256,
                  "selected_reference_sha256": SELECTED_REFERENCE_SHA256},
        "inference": {"complete": inference_complete, "gold_used_for_inference": False,
                       "terminal_v11_reopened": False,
                       "pose_policy": {"max_provisional_p95_upper_mm": 2.0,
                                       "min_provisional_coverage_lower": 0.55}},
        "scores": scores,
        "paired_case_deltas": paired,
        "historical_score": 0.4580152672,
        "historical_comparability": "UNESTABLISHED",
        "historical_score_beaten": False,
        "cases": {arm: arm_rows[arm] for arm in ARMS},
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "uncertainty_status": "UNCALIBRATED_PROVISIONAL_RETAINED_POSE_AND_SCAN_SUPPORT_DISTANCES",
        "limitations": [
            "This is consumed-development evidence, not unseen-family generalization.",
            "Rejected V11 registration candidates are a provisional pose source, never a verified datum.",
            "Native STEP axes are conditional scan-supported contact-line proposals, not measured manufacturing events.",
            "Cases with no gated finite pose remain explicit abstentions in each 71-line denominator.",
        ],
        "source_bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "recovery_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v2.py"),
            "recovery_base_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v1.py"),
            "single_arm_runner_sha256": _sha256(ROOT / "scripts/run_gold11_v11_multi_scan_luna_max_v1.py"),
            "review_source_sha256": _sha256(REVIEW_PATH),
            "catalog_sha256": _sha256(CATALOG_PATH),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checksums = {"schema": "GOLD11_V11_MULTI_SCAN_LUNA_MAX_PAIRED_CHECKSUMS_V1",
                 "result_sha256": _sha256(output / "RESULT.json"),
                 "source_bindings": result["source_bindings"]}
    (output / "CHECKSUMS.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY_PAIRED",
                      "control": result["scores"]["control"],
                      "challenger": result["scores"]["challenger"],
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
