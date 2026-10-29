#!/usr/bin/env python3
"""Score frozen V52/V62 on the consumed non-confirmatory V5R2 pilot."""

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
from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v5r1_pilot import TOKEN
from scripts.score_f1_calibrated_raw_synthetic_v60 import _aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
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
    model_cases: dict[str, list[dict]] = {"v52": [], "v62": []}
    for row in private["cases"]:
        case_id = str(row["opaque_case_id"])
        family_id = str(row["family_id"])
        truth = json.loads(
            (args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text(encoding="utf-8")
        )
        for model, root in (("v52", args.v52_root), ("v62", args.v62_root)):
            result = json.loads((root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
            scored = match_lines(truth["manual_bends"], result["predictions"], thresholds)
            scored.update({"case_id": case_id, "family_id": family_id})
            model_cases[model].append(scored)

    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_synthetic_v5r2_consumed_pilot_score",
        "token": TOKEN,
        "evidence_class": "non_confirmatory_consumed_morphology_pilot",
        "confirmatory_eligible": False,
        "truth_free_visual_precommit_completed_before_scoring": True,
        "no_tuning_or_rerun_rescue": True,
        "thresholds": {
            "endpoint_distance_mm": 5.0,
            "line_distance_mm": 4.0,
            "direction_cosine": 0.60,
            "overlap_fraction": 0.60,
        },
        "v52": {"aggregate": _aggregate(model_cases["v52"]), "cases": model_cases["v52"]},
        "v62": {"aggregate": _aggregate(model_cases["v62"]), "cases": model_cases["v62"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "v52": report["v52"]["aggregate"],
                "v62": report["v62"]["aggregate"],
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
