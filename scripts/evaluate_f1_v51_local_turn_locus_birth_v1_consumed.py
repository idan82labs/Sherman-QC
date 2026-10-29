#!/usr/bin/env python3
"""Evaluate the precommitted local-turn-locus birth lane on consumed scans."""

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

from scripts.apply_f1_v51_local_turn_locus_birth_v1 import apply_births, canonical_bytes
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
V51_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
LOCK = ROOT / "research/f1_v51_local_turn_locus_birth_v1_precommit.json"
OUTPUT = ROOT / "output/research/f1-v51-local-turn-locus-birth-v1-consumed"
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


def evaluate(case_ids: set[str], passes: tuple[str, ...], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output root is not empty")
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    parameter_lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cases = [dict(row) for row in catalog["cases"] if row.get("scan_role") == ROLE]
    if case_ids:
        cases = [row for row in cases if str(row["case_id"]) in case_ids]
    if not cases:
        raise ValueError("no selected consumed cases")
    pass_payloads: dict[str, dict[str, bytes]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for pass_name in passes:
        pass_payloads[pass_name] = {}
        for case in cases:
            case_id = str(case["case_id"])
            v51_path = V51_ROOT / case_id / "semantic_result.json"
            v51 = json.loads(v51_path.read_text(encoding="utf-8"))
            result = apply_births(v51, Path(case["scan_path"]), parameter_lock)
            payload = canonical_bytes(result)
            result_path = output / pass_name / case_id / "semantic_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_bytes(payload)
            pass_payloads[pass_name][case_id] = payload
            if pass_name == passes[0]:
                audits[case_id] = dict(result["local_turn_locus_birth_audit"])
            print(json.dumps({"pass": pass_name, "case_id": case_id, "audit": result["local_turn_locus_birth_audit"]}, sort_keys=True), flush=True)

    baseline_rows = []
    successor_rows = []
    case_rows = []
    primary = passes[0]
    for case in cases:
        case_id = str(case["case_id"])
        v51 = json.loads((V51_ROOT / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        successor = json.loads(pass_payloads[primary][case_id])
        base_score = match_lines(case["bends"], v51["predictions"], THRESHOLDS)
        successor_score = match_lines(case["bends"], successor["predictions"], THRESHOLDS)
        baseline_rows.append(base_score)
        successor_rows.append(successor_score)
        case_rows.append({
            "case_id": case_id,
            "baseline_true_positive": base_score["true_positive"],
            "successor_true_positive": successor_score["true_positive"],
            "baseline_false_positive": base_score["false_positive"],
            "successor_false_positive": successor_score["false_positive"],
            "emitted_addition_count": audits[case_id]["emitted_addition_count"],
        })
    baseline = _aggregate(baseline_rows)
    successor = _aggregate(successor_rows)

    fold_rows = []
    if len(cases) == 28:
        dummy = np.zeros(len(cases))
        groups = np.arange(len(cases))
        for fold, (_train, held) in enumerate(GroupKFold(n_splits=7).split(dummy, dummy, groups), 1):
            selected_base = [baseline_rows[int(index)] for index in held]
            selected_successor = [successor_rows[int(index)] for index in held]
            fold_rows.append({
                "fold": fold,
                "held_out_cases": [str(cases[int(index)]["case_id"]) for index in held],
                "baseline": _aggregate(selected_base),
                "successor": _aggregate(selected_successor),
            })
    replay_identity = {
        case_id: all(pass_payloads[name][case_id] == pass_payloads[primary][case_id] for name in passes[1:])
        for case_id in pass_payloads[primary]
    }
    no_tp_regression = all(row["successor_true_positive"] >= row["baseline_true_positive"] for row in case_rows)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_local_turn_locus_birth_v1_consumed_evaluation",
        "catalog_sha256": _sha(CATALOG),
        "parameter_lock_file_sha256": _sha(LOCK),
        "runtime_sha256": _sha(ROOT / "scripts/apply_f1_v51_local_turn_locus_birth_v1.py"),
        "selected_case_ids": [str(row["case_id"]) for row in cases],
        "source_only_runtime": True,
        "truth_available_to_runtime": False,
        "parameter_search_after_lock": False,
        "strict_outer_case_group_is_partition_only_for_deterministic_nonlearned_policy": True,
        "baseline_v51": baseline,
        "local_turn_locus_successor": successor,
        "strict_outer_case_group_folds": fold_rows,
        "case_rows": case_rows,
        "passes": list(passes),
        "all_pass_byte_identity": all(replay_identity.values()),
        "case_pass_byte_identity": replay_identity,
        "retention_gate": {
            "required_f1_strictly_greater_than": float(parameter_lock["retention_gate"]["strict_outer_case_group_f1_must_exceed"]),
            "no_baseline_tp_regression": no_tp_regression,
            "f1_passed": successor["f1"] > float(parameter_lock["retention_gate"]["strict_outer_case_group_f1_must_exceed"]),
            "retained": no_tp_regression and successor["f1"] > float(parameter_lock["retention_gate"]["strict_outer_case_group_f1_must_exceed"]),
        },
        "promotion_eligible": False,
        "generalization_evidence": False,
    }
    report_path = output / "evaluation_report.json"
    report_path.write_bytes(canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--passes", default="primary")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    passes = tuple(value.strip() for value in args.passes.split(",") if value.strip())
    if not passes:
        raise ValueError("at least one pass is required")
    report = evaluate(set(args.case_id), passes, args.output.resolve())
    print(json.dumps({
        "baseline": report["baseline_v51"],
        "successor": report["local_turn_locus_successor"],
        "gate": report["retention_gate"],
        "report": str(args.output.resolve() / "evaluation_report.json"),
        "report_sha256": _sha(args.output.resolve() / "evaluation_report.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
