#!/usr/bin/env python3
"""Score frozen V61/V62 on the prospective complex V4 geometry stress corpus."""

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
from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v4 import TOKEN
from scripts.score_f1_calibrated_raw_synthetic_v60 import _aggregate


ROUTE = "calibrated_distributed_turn_consensus_v62"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v61-root", type=Path, required=True)
    parser.add_argument("--v62-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("score output already exists")

    private = json.loads(
        (args.corpus_root / "evaluator_only" / "private_manifest.json").read_text(encoding="utf-8")
    )
    if private.get("token") != TOKEN:
        raise ValueError("private manifest token mismatch")
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    incumbent_cases, candidate_cases, births = [], [], []
    for row in private["cases"]:
        case_id = str(row["opaque_case_id"])
        family_id = str(row["family_id"])
        truth = json.loads(
            (args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text(encoding="utf-8")
        )
        v61 = json.loads((args.v61_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        v62 = json.loads((args.v62_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        incumbent = match_lines(truth["manual_bends"], v61["predictions"], thresholds)
        candidate = match_lines(truth["manual_bends"], v62["predictions"], thresholds)
        incumbent.update({"case_id": case_id, "family_id": family_id})
        candidate.update({"case_id": case_id, "family_id": family_id})
        incumbent_cases.append(incumbent)
        candidate_cases.append(candidate)
        matched = {match["prediction_id"] for match in candidate["matches"]}
        for prediction in v62["predictions"]:
            if prediction.get("selection_route") == ROUTE:
                births.append(
                    {
                        "case_id": case_id,
                        "family_id": family_id,
                        "prediction_id": prediction["bend_id"],
                        "matched": prediction["bend_id"] in matched,
                    }
                )

    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_synthetic_v62_v4_prospective_score",
        "token": TOKEN,
        "evidence_class": "prospective_non_confirmatory_geometry_stress",
        "confirmatory_eligible": False,
        "thresholds_frozen_before_v4_generation": True,
        "truth_free_visual_precommit_completed_before_scoring": True,
        "v61": {"aggregate": _aggregate(incumbent_cases), "cases": incumbent_cases},
        "v62": {"aggregate": _aggregate(candidate_cases), "cases": candidate_cases},
        "v62_births": births,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "v61": report["v61"]["aggregate"],
                "v62": report["v62"]["aggregate"],
                "birth_true_positive": sum(row["matched"] for row in births),
                "birth_false_positive": sum(not row["matched"] for row in births),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
