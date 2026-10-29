#!/usr/bin/env python3
"""Evaluate the precommitted orientation-Hough ridge births on consumed scans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_f1_v51_orientation_hough_turn_birth_v1 import apply_births, canonical_bytes
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
V51_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
RIDGE_LOCK = ROOT / "research/f1_v51_local_turn_locus_birth_v1_precommit.json"
LOCK = ROOT / "research/f1_v51_orientation_hough_turn_birth_v1_precommit.json"
OUTPUT = ROOT / "output/research/f1-v51-orientation-hough-turn-birth-v1-consumed"
ROLE = "2026-08-30_ai_multiview_consensus_development"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    return {
        "case_count": len(rows),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp else 0.0,
        "exact_case_count": sum(bool(row["exact"]) for row in rows),
    }


def evaluate(case_ids: set[str], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output root is not empty")
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    ridge_lock = json.loads(RIDGE_LOCK.read_text(encoding="utf-8"))
    parameter_lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha(RIDGE_LOCK) != parameter_lock["frozen_ridge_field_precommit_sha256"]:
        raise ValueError("orientation-Hough lock does not bind the frozen ridge-field bytes")
    cases = [dict(row) for row in catalog["cases"] if row.get("scan_role") == ROLE]
    if case_ids:
        cases = [row for row in cases if str(row["case_id"]) in case_ids]
    if not cases:
        raise ValueError("no selected consumed cases")

    payloads: dict[str, bytes] = {}
    audits: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        v51 = json.loads((V51_ROOT / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        result = apply_births(v51, Path(case["scan_path"]), ridge_lock, parameter_lock)
        payload = canonical_bytes(result)
        result_path = output / "primary" / case_id / "semantic_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(payload)
        payloads[case_id] = payload
        audits[case_id] = dict(result["orientation_hough_turn_birth_audit"])
        print(json.dumps({"case_id": case_id, "audit": audits[case_id]}, sort_keys=True), flush=True)

    baseline_rows = []
    successor_rows = []
    case_rows = []
    for case in cases:
        case_id = str(case["case_id"])
        v51 = json.loads((V51_ROOT / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        successor = json.loads(payloads[case_id])
        baseline = match_lines(case["bends"], v51["predictions"], THRESHOLDS)
        score = match_lines(case["bends"], successor["predictions"], THRESHOLDS)
        baseline_rows.append(baseline)
        successor_rows.append(score)
        case_rows.append({
            "case_id": case_id,
            "baseline_true_positive": baseline["true_positive"],
            "successor_true_positive": score["true_positive"],
            "baseline_false_positive": baseline["false_positive"],
            "successor_false_positive": score["false_positive"],
            "emitted_addition_count": audits[case_id]["emitted_addition_count"],
            "turn_ridge_point_count": audits[case_id]["turn_ridge_point_count"],
            "local_axis_point_count": audits[case_id]["local_axis_point_count"],
            "qualified_birth_count_before_nms_and_cap": audits[case_id]["qualified_birth_count_before_nms_and_cap"],
        })
    baseline_metrics = _aggregate(baseline_rows)
    successor_metrics = _aggregate(successor_rows)
    folds = []
    if len(cases) == 28:
        dummy = np.zeros(len(cases))
        groups = np.arange(len(cases))
        for fold, (_train, held) in enumerate(GroupKFold(n_splits=7).split(dummy, dummy, groups), 1):
            folds.append({
                "fold": fold,
                "held_out_cases": [str(cases[int(index)]["case_id"]) for index in held],
                "baseline": _aggregate([baseline_rows[int(index)] for index in held]),
                "successor": _aggregate([successor_rows[int(index)] for index in held]),
            })
    no_tp_regression = all(row["successor_true_positive"] >= row["baseline_true_positive"] for row in case_rows)
    required = float(parameter_lock["retention_gate"]["strict_outer_case_group_f1_must_exceed"])
    retained = bool(no_tp_regression and successor_metrics["f1"] > required)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_orientation_hough_turn_birth_v1_consumed_evaluation",
        "catalog_sha256": _sha(CATALOG),
        "ridge_field_parameter_lock_sha256": _sha(RIDGE_LOCK),
        "parameter_lock_sha256": _sha(LOCK),
        "runtime_sha256": _sha(ROOT / "scripts/apply_f1_v51_orientation_hough_turn_birth_v1.py"),
        "source_only_runtime": True,
        "truth_available_to_runtime": False,
        "parameter_search_after_lock": False,
        "frozen_ridge_field_reused_without_change": True,
        "strict_outer_case_group_is_partition_only_for_deterministic_nonlearned_policy": True,
        "baseline_v51": baseline_metrics,
        "orientation_hough_successor": successor_metrics,
        "strict_outer_case_group_folds": folds,
        "case_rows": case_rows,
        "audit_totals": {
            "turn_ridge_point_count": sum(row["turn_ridge_point_count"] for row in case_rows),
            "local_axis_point_count": sum(row["local_axis_point_count"] for row in case_rows),
            "qualified_birth_count_before_nms_and_cap": sum(row["qualified_birth_count_before_nms_and_cap"] for row in case_rows),
            "emitted_addition_count": sum(row["emitted_addition_count"] for row in case_rows),
        },
        "retention_gate": {
            "required_f1_strictly_greater_than": required,
            "no_baseline_tp_regression": no_tp_regression,
            "f1_passed": successor_metrics["f1"] > required,
            "retained": retained,
        },
        "representation_ruling": (
            "FUSED_PLY_SOURCE_REPRESENTATION_STOP"
            if not retained
            else "CONTINUE_ONLY_TO_UNTOUCHED_PROSPECTIVE_VALIDATION"
        ),
        "promotion_eligible": False,
        "generalization_evidence": False,
    }
    report_path = output / "evaluation_report.json"
    report_path.write_bytes(canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(set(args.case_id), args.output.resolve())
    report_path = args.output.resolve() / "evaluation_report.json"
    print(json.dumps({
        "baseline": report["baseline_v51"],
        "successor": report["orientation_hough_successor"],
        "audit_totals": report["audit_totals"],
        "gate": report["retention_gate"],
        "representation_ruling": report["representation_ruling"],
        "report": str(report_path),
        "report_sha256": _sha(report_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
