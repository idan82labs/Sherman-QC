#!/usr/bin/env python3
"""Score frozen EXP_CAD_0009 semantic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


def _aggregate(rows: list[dict]) -> dict:
    result = score_localized_cases(rows)
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(str(row["family_id"]), []).append(row)
    recalls = {family: sum(row["true_positive"] for row in cases) / sum(row["gold_count"] for row in cases) for family, cases in families.items()}
    result["family_recalls"] = recalls
    result["minimum_family_recall"] = min(recalls.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--retained-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    slice_payload = json.loads(args.slice.read_text(encoding="utf-8"))
    catalog = json.loads(args.gold_catalog.read_text(encoding="utf-8"))
    gold_by_sha = {str(row["scan_sha256"]): row for row in catalog["cases"]}
    rows = []
    for entry in slice_payload["entries"]:
        case_id = str(entry["case_token"])
        gold = gold_by_sha[str(entry["scan_sha256"])]
        payload = json.loads((args.predictions_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        score = match_lines(gold["bends"], payload["predictions"], Thresholds())
        score.update(case_id=case_id, family_id=str(gold["part_id"]), scan_sha256=str(entry["scan_sha256"]))
        rows.append(score)
    retained = json.loads(args.retained_comparison.read_text(encoding="utf-8"))["metrics"]["incumbent_union"]
    matched_ids = {str(row["case_id"]) for row in retained["cases"]}
    matched = _aggregate([row for row in rows if row["case_id"] in matched_ids])
    full = _aggregate(rows)
    delta = {key: matched[key] - retained[key] for key in ("precision", "recall", "f1", "exact_count_accuracy", "exact_localized_target_set_accuracy", "minimum_family_recall")}
    report = {
        "schema": "SHERMAN_QC_CAD_CONTEXTUAL_ASSIGNMENT_SCORE_V1",
        "experiment_id": "EXP_CAD_0009",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "promotion_eligible": False,
        "full13": full,
        "matched11": matched,
        "retained_incumbent11": retained,
        "delta_vs_retained_incumbent11": delta,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"matched11": {key: matched[key] for key in ("precision", "recall", "f1", "exact_count_accuracy", "minimum_family_recall")}, "delta": delta}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
