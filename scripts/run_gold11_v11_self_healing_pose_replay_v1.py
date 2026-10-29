#!/usr/bin/env python3
"""Replay retained finite V11 poses on the failing simple family.

This diagnostic holds the case, scan, CAD, native topology, and matcher fixed
while varying only which retained finite pose is replayed and the raw-support
radius.  It is a mechanism probe for the self-healing behavior, not a new
benchmark baseline or a production candidate.
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
from backend.cad_guided_v11_topology_recovery_v1 import recover_scan_supported_axes_v1
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.run_gold11_v11_multi_scan_luna_max_v1 import (
    CATALOG_PATH,
    CATALOG_SHA256,
    REVIEW_PATH,
    REVIEW_SHA256,
    SELECTED_REFERENCE_SHA256,
    V11_REVIEW_RESULT_SHA256,
    _pose_evidence,
    _read_cases,
    _selected_references,
)


PROTOCOL_PATH = ROOT / "docs/research/2026-09-16_gold11_v11_self_healing_pose_replay_protocol.md"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_self_healing_pose_replay_v1"
CASE_TOKEN = "CG191DEC7F81"
PART_KEY = "49001000"
POSE_INDICES = (0, 1)
SUPPORT_RADII_MM = (3.0, 5.0)
MAX_ENGINE_ROWS = 50_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proper_rigid(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(candidate.get("rotation"), dtype=np.float64)
    translation = np.asarray(candidate.get("translation"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("PROVISIONAL_POSE_SHAPE")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("PROVISIONAL_POSE_NONFINITE")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("PROVISIONAL_POSE_NOT_RIGID")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("PROVISIONAL_POSE_REFLECTION")
    return rotation, translation


def _score(rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    totals = {key: 0 for key in (
        "gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}
    for row in rows:
        score = match_lines(reference["bends"], row["recovery"].get("predictions", []), Thresholds())
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
    case = next(row for row in all_cases if row["case_token"] == CASE_TOKEN)
    if case["part_key"] != PART_KEY:
        raise ValueError("SELF_HEALING_CASE_IDENTITY_MISMATCH")
    retained, retained_path, retained_sha = _pose_evidence(CASE_TOKEN)
    if retained_sha != V11_REVIEW_RESULT_SHA256 and CASE_TOKEN == "CG4BDD1371B6":
        raise ValueError("RETAINED_V11_CONTROL_RESULT_HASH_MISMATCH")
    registration = retained.get("registration") or {}
    evidence = registration.get("evidence") or {}
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or len(candidates) <= max(POSE_INDICES):
        raise ValueError("RETAINED_POSE_CANDIDATE_SCOPE_MISMATCH")
    scan = prepare_mesh_scan_v1(scan_bytes=Path(case["scan_path"]).read_bytes(),
                                max_engine_rows=MAX_ENGINE_ROWS)
    if scan.get("status") != "AVAILABLE":
        raise ValueError("SELF_HEALING_SCAN_PREPARATION_UNAVAILABLE")

    rows_by_setting: dict[str, list[dict[str, Any]]] = {}
    # Gold/reference lines are not loaded in this entire inference loop.
    for pose_index in POSE_INDICES:
        candidate = candidates[pose_index]
        rotation, translation = _proper_rigid(candidate)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        quality = candidate.get("conditional_context_support", {}).get("finite_distance_metrics", {})
        coverage = candidate.get("conditional_context_support", {}).get("geometric_scan_coverage_lower")
        for radius in SUPPORT_RADII_MM:
            key = f"pose_{pose_index}_radius_{radius:g}"
            recovery = recover_scan_supported_axes_v1(
                scan["raw_points"], Path(case["cad_path"]), matrix,
                support_radius_mm=radius)
            rows_by_setting[key] = [{
                "case_token": CASE_TOKEN,
                "part_key": PART_KEY,
                "pose_index": pose_index,
                "support_radius_mm": radius,
                "input_bindings": {
                    "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"],
                    "cad_path": case["cad_path"], "cad_sha256": case["cad_sha256"],
                    "gold_used_for_inference": False,
                },
                "retained_v11_control": {
                    "path": str(retained_path.relative_to(ROOT)), "sha256": retained_sha,
                    "registration_status": registration.get("status"),
                    "measurement_context_indices": evidence.get("measurement_context_indices", []),
                },
                "pose_replay_quality": {
                    "p95_upper_mm": quality.get("p95_upper"),
                    "rms_upper_mm": quality.get("rms_upper"),
                    "coverage_lower": coverage,
                    "replay_semantics": "RETAINED_FINITE_POSE_REPLAY_NOT_RELABELED_ACCEPTED",
                },
                "recovery": recovery,
            }]

    # Post-inference scoring seam.
    references = _selected_references(all_cases)
    reference = references[CASE_TOKEN]
    score_by_setting = {key: _score(rows, reference)
                        for key, rows in rows_by_setting.items()}
    result = {
        "schema": "GOLD11_V11_SELF_HEALING_POSE_REPLAY_RESULT_V1",
        "candidate_implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
        "scope": {
            "case_count": 1, "family_count": 1, "case_token": CASE_TOKEN,
            "part_key": PART_KEY, "reference_line_count": score_by_setting["pose_0_radius_3"]["gold_count"],
            "pose_indices": list(POSE_INDICES), "support_radii_mm": list(SUPPORT_RADII_MM),
            "full_gold11_case_count_authenticated": len(all_cases),
            "review_source_sha256": REVIEW_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "selected_reference_sha256": SELECTED_REFERENCE_SHA256,
        },
        "inference": {
            "gold_used_for_inference": False, "terminal_v11_reopened": False,
            "max_engine_rows": MAX_ENGINE_ROWS,
            "native_topology_and_support_kernel_unchanged": True,
        },
        "score_by_setting": score_by_setting,
        "cases_by_setting": rows_by_setting,
        "historical_score": 0.4580152672,
        "historical_comparability": "UNESTABLISHED",
        "historical_score_beaten": False,
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "limitations": [
            "This is a one-case consumed-development mechanism replay, not a benchmark.",
            "Pose 1 is replayed as a retained finite candidate and is never relabeled as an accepted datum.",
            "Support-radius changes are diagnostic and do not authorize production thresholds.",
            "Gold and matcher enter only after all four inference replays.",
        ],
        "source_bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "recovery_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v1.py"),
            "recovery_v2_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v2.py"),
            "review_source_sha256": _sha256(REVIEW_PATH),
            "catalog_sha256": _sha256(CATALOG_PATH),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "CHECKSUMS.json").write_text(json.dumps({
        "schema": "GOLD11_V11_SELF_HEALING_POSE_REPLAY_CHECKSUMS_V1",
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
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY_SELF_HEALING_POSE_REPLAY",
                      "score_by_setting": result["score_by_setting"],
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
