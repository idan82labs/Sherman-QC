#!/usr/bin/env python3
"""Score a selector prediction against a frozen native-visual precommit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    scan = next(row for row in precommit["scans"] if row["opaque_scan_id"] == args.scan_id)
    result = match_lines(scan["manual_bends"], prediction["predictions"], Thresholds())
    tp = int(result["true_positive"])
    fp = int(result["false_positive"])
    fn = int(result["false_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result.update(
        {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2.0 * precision * recall / (precision + recall), 6)
            if precision + recall
            else 0.0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("true_positive", "false_positive", "false_negative", "f1", "exact")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
