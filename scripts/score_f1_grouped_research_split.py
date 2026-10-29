#!/usr/bin/env python3
"""Score routed F1 predictions on one frozen part-family split."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v1")
    parser.add_argument("--split", choices=("development", "holdout", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = json.loads((BASE / "freeze/sealed_gold_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((BASE / "freeze/grouped_development_holdout.json").read_text(encoding="utf-8"))
    allowed = {
        row["case_token"]
        for row in split["cases"]
        if args.split == "all" or row["split"] == args.split
    }
    gold_map = {row["case_token"]: row for row in gold["cases"]}
    rows = []
    for token in sorted(allowed):
        prediction_path = args.prediction_root / token / "decomposition_result.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        result = match_lines(
            gold_map[token]["manual_bends"],
            prediction.get("owned_bend_regions") or [],
            Thresholds(),
        )
        result.update({"case_token": token, "part_key": gold_map[token]["part_key"], "prediction_path": str(prediction_path)})
        rows.append(result)
    tp = sum(row["true_positive"] for row in rows)
    fp = sum(row["false_positive"] for row in rows)
    fn = sum(row["false_negative"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    line_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "schema_version": 1,
        "split_id": split["split_id"],
        "split": args.split,
        "metrics": {
            "scan_count": len(rows),
            "part_family_count": len({row["part_key"] for row in rows}),
            "gold_bend_count": tp + fn,
            "prediction_count": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "line_f1": round(line_f1, 6),
            "exact_scans": sum(row["exact"] for row in rows),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
