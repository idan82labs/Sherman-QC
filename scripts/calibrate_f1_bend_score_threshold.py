#!/usr/bin/env python3
"""Sweep intrinsic F1 evidence thresholds on the frozen development families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def _micro(rows):
    tp = sum(row["true_positive"] for row in rows)
    fp = sum(row["false_positive"] for row in rows)
    fn = sum(row["false_negative"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "line_f1": round(f1, 6),
        "exact_scans": sum(bool(row["exact"]) for row in rows),
        "prediction_count": tp + fp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=BASE / "scorecard_v1/bend_scorecard.json")
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="development")
    parser.add_argument("--output", type=Path, default=BASE / "scorecard_v1/development_threshold_sweep.json")
    args = parser.parse_args()
    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    gold = json.loads((BASE / "freeze/sealed_gold_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((BASE / "freeze/grouped_development_holdout.json").read_text(encoding="utf-8"))
    split_map = {item["case_token"]: item["split"] for item in split["cases"]}
    gold_map = {item["case_token"]: item for item in gold["cases"]}
    cases = {item["case_token"]: item for item in scorecard["cases"]}
    allowed = [
        token for token in sorted(cases)
        if args.split == "all" or split_map[token] == args.split
    ]
    sweep = []
    for threshold in range(0, 101):
        rows = []
        for token in allowed:
            predictions = [
                bend for bend in cases[token]["bends"]
                if float(bend["evidence"]["evidence_score"]) >= threshold
            ]
            rows.append(match_lines(gold_map[token]["manual_bends"], predictions, Thresholds()))
        sweep.append({"threshold": threshold, **_micro(rows)})
    best = max(sweep, key=lambda item: (item["line_f1"], item["precision"], item["threshold"]))
    named = {
        "retain_all": sweep[0],
        "rating_C_or_better": sweep[50],
        "rating_B_or_better": sweep[65],
        "rating_A_only": sweep[80],
        "maximum_development_f1": best,
    }
    precision_operating_points = {}
    for target in (0.70, 0.80, 0.90):
        viable = [item for item in sweep if item["precision"] >= target and item["prediction_count"] > 0]
        precision_operating_points[f"precision_at_least_{target:.2f}"] = (
            max(viable, key=lambda item: (item["recall"], item["line_f1"], -item["threshold"]))
            if viable else None
        )
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_intrinsic_evidence_threshold_sweep",
        "split": args.split,
        "case_tokens": allowed,
        "selection_contract": "threshold selected on development families only",
        "score_semantics": scorecard["score_semantics"],
        "named_operating_points": named,
        "precision_operating_points": precision_operating_points,
        "sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "named_operating_points": named, "precision_operating_points": precision_operating_points, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
