#!/usr/bin/env python3
"""Run the locked hidden semantic evaluator and emit only permitted feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_semantic_oracle_v1.evaluator import evaluate_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--phase", choices=("family_A", "family_B"), required=True)
    parser.add_argument("--family-b-claim", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_submission(
        lock_path=args.lock,
        prediction_path=args.prediction,
        phase=args.phase,
        family_b_claim_path=args.family_b_claim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
