#!/usr/bin/env python3
"""Score CAD-first scan-localized events on the consumed physical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


def aggregate(rows: list[dict]) -> dict:
    result = score_localized_cases(rows)
    families = {}
    for row in rows: families.setdefault(row["family_id"], []).append(row)
    recalls = {family: sum(r["true_positive"] for r in values) / sum(r["gold_count"] for r in values) for family, values in families.items()}
    result["family_recalls"] = recalls
    result["minimum_family_recall"] = min(recalls.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--retained-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    slice_payload = json.loads(args.slice.read_text())
    summary = json.loads(args.summary.read_text())
    catalog = json.loads(args.gold_catalog.read_text())
    by_path = {str(Path(row["scan_path"]).resolve()): row for row in slice_payload["entries"]}
    gold_by_sha = {row["scan_sha256"]: row for row in catalog["cases"]}
    rows = []
    for result in summary["results"]:
        entry = by_path[str(Path(result["scan_path"]).resolve())]
        gold = gold_by_sha[entry["scan_sha256"]]
        predictions = [{"bend_id": row["bend_id"], "line_start": row["detected_line_start"], "line_end": row["detected_line_end"]} for row in result.get("matches") or []]
        scored = match_lines(gold["bends"], predictions, Thresholds())
        scored.update(case_id=entry["case_token"], family_id=str(gold["part_id"]), scan_sha256=entry["scan_sha256"])
        rows.append(scored)
    baseline_report = json.loads(args.retained_baseline.read_text())
    incumbent = baseline_report["metrics"]["incumbent_union"]
    matched_ids = {row["case_id"] for row in incumbent["cases"]}
    matched = aggregate([row for row in rows if row["case_id"] in matched_ids])
    full = aggregate(rows)
    deltas = {key: matched[key] - incumbent[key] for key in ("precision", "recall", "f1", "exact_count_accuracy", "exact_localized_target_set_accuracy", "minimum_family_recall")}
    report = {"schema_version": 1, "kind": "CAD_FIRST_LOCALIZER_V1_CONSUMED_PHYSICAL_SCORE", "evidence_class": "CONSUMED_PHYSICAL", "promotion_eligible": False, "full13": full, "matched11": matched, "retained_incumbent11": incumbent, "deltas_vs_retained_incumbent11": deltas, "gpi": "C_NO_GENERALIZATION_EVIDENCE", "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"full13_f1": full["f1"], "matched11_f1": matched["f1"], "retained_f1": incumbent["f1"], "delta": deltas["f1"], "gpi": report["gpi"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
