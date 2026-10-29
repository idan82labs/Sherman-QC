#!/usr/bin/env python3
"""Score development-only V61 against private synthetic V1 truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.score_f1_calibrated_raw_synthetic_v60 import TOKEN, _aggregate


ROUTE = "calibrated_birth_ownership_guard_v61"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--v61-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("score output already exists")
    private = json.loads((args.corpus_root / "evaluator_only" / "private_manifest.json").read_text())
    if private.get("token") != TOKEN:
        raise ValueError("private manifest token mismatch")
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    baseline_cases, candidate_cases, births = [], [], []
    for row in private["cases"]:
        case_id = str(row["opaque_case_id"])
        family_id = str(row["family_id"])
        truth = json.loads((args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text())
        v52 = json.loads((args.v52_root / case_id / "semantic_result.json").read_text())
        v61 = json.loads((args.v61_root / case_id / "semantic_result.json").read_text())
        baseline = match_lines(truth["manual_bends"], v52["predictions"], thresholds)
        candidate = match_lines(truth["manual_bends"], v61["predictions"], thresholds)
        baseline.update({"case_id": case_id, "family_id": family_id})
        candidate.update({"case_id": case_id, "family_id": family_id})
        baseline_cases.append(baseline)
        candidate_cases.append(candidate)
        matched = {match["prediction_id"] for match in candidate["matches"]}
        for prediction in v61["predictions"]:
            if prediction.get("selection_route") == ROUTE:
                births.append({"case_id": case_id, "prediction_id": prediction["bend_id"], "matched": prediction["bend_id"] in matched})
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_synthetic_v61_score",
        "token": TOKEN,
        "evidence_class": "development_only_post_hoc_non_confirmatory",
        "confirmatory_eligible": False,
        "v52": {"aggregate": _aggregate(baseline_cases), "cases": baseline_cases},
        "v61": {"aggregate": _aggregate(candidate_cases), "cases": candidate_cases},
        "v61_births": births,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"v52": report["v52"]["aggregate"], "v61": report["v61"]["aggregate"], "v61_births": births, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
