#!/usr/bin/env python3
"""Score frozen V60 against private synthetic truth after truth-free audit."""

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


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
ROUTE = "calibrated_per_view_segment_consensus_v60"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(cases: list[dict]) -> dict:
    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_case_count": sum(bool(row["exact"]) for row in cases),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--v60-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("score output already exists")
    private = json.loads((args.corpus_root / "evaluator_only" / "private_manifest.json").read_text(encoding="utf-8"))
    if private.get("token") != TOKEN:
        raise ValueError("private manifest token mismatch")
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    v52_cases, v60_cases, birth_rows = [], [], []
    family_rows: dict[str, dict[str, list[dict]]] = {}
    for private_row in private["cases"]:
        case_id = str(private_row["opaque_case_id"])
        family_id = str(private_row["family_id"])
        truth = json.loads((args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text(encoding="utf-8"))
        v52 = json.loads((args.v52_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        v60 = json.loads((args.v60_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        v52_score = match_lines(truth["manual_bends"], v52["predictions"], thresholds)
        v60_score = match_lines(truth["manual_bends"], v60["predictions"], thresholds)
        v52_score.update({"case_id": case_id, "family_id": family_id})
        v60_score.update({"case_id": case_id, "family_id": family_id})
        v52_cases.append(v52_score)
        v60_cases.append(v60_score)
        family_rows.setdefault(family_id, {"v52": [], "v60": []})["v52"].append(v52_score)
        family_rows[family_id]["v60"].append(v60_score)
        matched_ids = {row["prediction_id"] for row in v60_score["matches"]}
        for prediction in v60["predictions"]:
            if prediction.get("selection_route") == ROUTE:
                birth_rows.append({"case_id": case_id, "prediction_id": prediction["bend_id"], "matched": prediction["bend_id"] in matched_ids})
    families = [
        {"family_id": family_id, "v52": _aggregate(rows["v52"]), "v60": _aggregate(rows["v60"])}
        for family_id, rows in sorted(family_rows.items())
    ]
    birth_tp = sum(row["matched"] for row in birth_rows)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_synthetic_v60_score",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened_after_frozen_truth_free_visual_audit": True,
        "v52": {"aggregate": _aggregate(v52_cases), "cases": v52_cases},
        "v60": {"aggregate": _aggregate(v60_cases), "cases": v60_cases},
        "v60_births": {"true_positive": birth_tp, "false_positive": len(birth_rows) - birth_tp, "precision": _ratio(birth_tp, len(birth_rows)), "rows": birth_rows},
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v52": report["v52"]["aggregate"], "v60": report["v60"]["aggregate"], "v60_births": report["v60_births"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
