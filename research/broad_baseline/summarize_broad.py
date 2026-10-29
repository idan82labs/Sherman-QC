#!/usr/bin/env python3
"""Combine scores_best_f1.json + scores_runtime.json into one machine-readable broad summary.

Worst-15 ranking: best_f1 per-scan line-F1 ascending, ties broken by more errors (FP+FN),
then runtime F1. Only complete-label line scans are ranked.

  summarize_broad.py --scores-dir docs/production/runlog/broad_baseline \
      --out docs/production/runlog/broad_baseline/BROAD_BASELINE_SUMMARY.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

HEADLINE_COHORTS = ("ALL", "GOLD11", "NON_GOLD11", "HUMAN_GOLD_ALL", "HUMAN_GOLD_NON_GOLD11", "AI_GOLD",
                    "FULL_SCANS", "STAGED_SCANS", "CAD_AVAILABLE", "NO_CAD", "NON_GOLD11_CAD_AVAILABLE_EXCL_LIDAR_DEMO",
                    "LIDAR_DEMO_SYNTHETIC")
KEYS = ("scans", "families", "gold_lines", "predicted_lines", "true_positive", "false_positive", "false_negative",
        "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1",
        "family_macro_f1", "exact_scans", "scans_with_f1_ge_0_8", "scans_with_zero_tp", "scans_with_no_predictions",
        "angle_mae_deg_matched", "position_error_mean_mm_matched", "position_error_median_mm_matched",
        "fn_cause_counts_diagnostic", "fp_cause_counts_diagnostic", "zero_bend_controls")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[], help="extra JSON merged under 'notes'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    lanes = {lane: bb.read_json(args.scores_dir / f"scores_{lane}.json") for lane in ("best_f1", "runtime")}
    per = {lane: {r["case_id"]: r for r in rep["per_scan"]} for lane, rep in lanes.items()}
    ranked = sorted((r for r in per["best_f1"].values() if r.get("disposition") == "SCORED_COMPLETE_LABEL"),
                    key=lambda r: (r["f1"], -(r["false_positive"] + r["false_negative"]),
                                   (per["runtime"].get(r["case_id"]) or {}).get("f1") or 0.0, r["case_id"]))
    worst = []
    for r in ranked[:15]:
        rt = per["runtime"].get(r["case_id"], {})
        worst.append({"case_id": r["case_id"], "family_id": r["family_id"], "tier": r["tier"],
                      "scan_name": r["scan_name"], "staged": r["staged"], "has_cad": r["has_cad"],
                      "gold": r["gold_count"],
                      "best_f1": {k: r.get(k) for k in ("prediction_count", "true_positive", "false_positive",
                                                        "false_negative", "f1", "lane_status", "fn_causes",
                                                        "fp_causes", "angle_mae_deg", "position_error_mean_mm")},
                      "runtime": {k: rt.get(k) for k in ("prediction_count", "true_positive", "false_positive",
                                                         "false_negative", "f1", "lane_status", "worker_error",
                                                         "failure_reason")}})
    summary = {
        "schema": "sherman.broad_baseline.summary.v1",
        "created_at": bb.utc_now(),
        "case_list": lanes["best_f1"]["case_list"],
        "case_list_same_for_both_lanes": lanes["best_f1"]["case_list"] == lanes["runtime"]["case_list"],
        "matcher": lanes["best_f1"]["matcher"],
        "conventions": lanes["best_f1"]["conventions"],
        "lanes": {lane: {"cohorts": {c: {k: rep["cohorts"][c].get(k) for k in KEYS} for c in HEADLINE_COHORTS
                                     if c in rep["cohorts"]},
                         "resources": rep["resources"], "excluded_before_scoring": rep["excluded_before_scoring"]}
                  for lane, rep in lanes.items()},
        "worst_15_by_best_f1": worst,
        "notes": {p.stem: bb.read_json(p) for p in args.extra},
    }
    bb.write_json(args.out, summary)
    for lane, block in summary["lanes"].items():
        for c in ("ALL", "GOLD11", "NON_GOLD11"):
            x = block["cohorts"][c]
            print(f"{lane:8s} {c:12s} scans={x['scans']} TP/FP/FN={x['true_positive']}/{x['false_positive']}/"
                  f"{x['false_negative']} P={x['micro_precision']:.3f} R={x['micro_recall']:.3f} "
                  f"microF1={x['micro_f1']:.4f} macroF1={x['macro_f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
