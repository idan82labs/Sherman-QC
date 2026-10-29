#!/usr/bin/env python3
"""Score synthetic V51 stress results after the detector boundary is closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.score_f1_v51_new_untouched_corpus as generic

TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"synthetic score is immutable and already exists: {output}")

    generic.TOKEN = TOKEN
    report = generic.score(
        truth_path=args.truth.resolve(),
        truth_sha256=generic._sha(args.truth.resolve()),
        family_path=args.family_map.resolve(),
        family_sha256=generic._sha(args.family_map.resolve()),
        prediction_root=args.prediction_root.resolve(),
    )
    report["kind"] = "sherman_qc_f1_v51_synthetic_stress_score"
    report["evidence_class"] = "engineering_only_non_confirmatory"
    report["confirmatory_eligible"] = False
    generic._atomic_write_json(output, report)
    print(json.dumps({"output": str(output), **report["aggregate_integers"], **report["aggregate_ratios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
