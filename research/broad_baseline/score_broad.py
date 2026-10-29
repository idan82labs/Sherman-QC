#!/usr/bin/env python3
"""Broad-run scoring (called as `score.py broad ...`): one lane over a broad case list.

Every match decision is score.score_case -> scripts/evaluate_f1_against_manual_gold.match_lines
(frozen 5 deg / 4 mm / 60 % of both spans, one-to-one Hungarian) in the raw-scan frame,
with gold frames from --frame-records (residual <= 0.01 mm or the case is excluded).

Per scan: TP/FP/FN, precision, recall, F1, axis-angle MAE and transverse position error
on matched pairs, lane status / failure reason, wall time and peak RSS of the run, and a
DIAGNOSTIC (not the metric) classification of every FN/FP by its nearest counterpart:
  MISSED / SPURIOUS      nothing within 15 deg and 15 mm transverse
  ANGLE_GT_5 / OFFSET_GT_4MM / SPAN_LT_60PCT   nearest counterpart fails that frozen gate
  TAKEN_BY_OTHER         passes every gate but the one-to-one assignment used it elsewhere
Aggregates (complete-label line cases only; zero-bend controls reported as FP lines):
micro P/R/F1 from summed counts, macro P/R/F1 = mean of per-scan values, exact-part rate,
per family, per tier, Gold11 vs non-Gold11, human vs AI gold, full vs staged scans,
CAD available vs not, and with / without the synthetic LIDAR demo captures.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

NEAR_ANGLE_DEG = 15.0
NEAR_OFFSET_MM = 15.0


def _metrics(gold_line: Mapping[str, Any], pred_line: Mapping[str, Any]) -> dict[str, float]:
    from scripts.evaluate_f1_against_manual_gold import pair_metrics

    return pair_metrics(bb.endpoints(gold_line), bb.endpoints(pred_line))


def _cause(m: Mapping[str, float]) -> list[str]:
    out = []
    if m["axis_angle_deg"] > bb.MATCH_ANGLE_DEG:
        out.append("ANGLE_GT_5")
    if m["transverse_offset_mm"] > bb.MATCH_OFFSET_MM:
        out.append("OFFSET_GT_4MM")
    if m["gold_span_overlap"] < bb.MATCH_GOLD_OVERLAP or m["prediction_span_overlap"] < bb.MATCH_PRED_OVERLAP:
        out.append("SPAN_LT_60PCT")
    return out


def _nearest(metrics_rows: Sequence[tuple[str, dict[str, float]]]) -> tuple[str, dict[str, float]] | None:
    near = [(i, m) for i, m in metrics_rows
            if m["axis_angle_deg"] <= NEAR_ANGLE_DEG and m["transverse_offset_mm"] <= NEAR_OFFSET_MM]
    if not near:
        return None
    return min(near, key=lambda item: item[1]["axis_angle_deg"] / bb.MATCH_ANGLE_DEG
               + item[1]["transverse_offset_mm"] / bb.MATCH_OFFSET_MM
               + (1.0 - min(item[1]["gold_span_overlap"], item[1]["prediction_span_overlap"])))


def diagnose(gold_lines: list[dict[str, Any]], pred_lines: list[dict[str, Any]], matched: Mapping[str, Any]) -> dict:
    gid = lambda r, i: str(r.get("bend_id") or f"G{i + 1:02d}")  # noqa: E731
    pid = lambda r, i: str(r.get("bend_id") or f"P{i + 1:02d}")  # noqa: E731
    matched_gold = {m["gold_id"] for m in matched.get("matches") or []}
    matched_pred = {m["prediction_id"] for m in matched.get("matches") or []}
    table = [[_metrics(g, p) for p in pred_lines] for g in gold_lines]
    fn_rows, fp_rows = [], []
    for gi, g in enumerate(gold_lines):
        if gid(g, gi) in matched_gold:
            continue
        near = _nearest([(pid(p, pj), table[gi][pj]) for pj, p in enumerate(pred_lines)])
        if near is None:
            fn_rows.append({"gold_id": gid(g, gi), "cause": "MISSED"})
            continue
        causes = _cause(near[1])
        fn_rows.append({"gold_id": gid(g, gi), "nearest_prediction": near[0],
                        "cause": "+".join(causes) if causes else "TAKEN_BY_OTHER",
                        **{k: round(v, 3) for k, v in near[1].items()}})
    for pj, p in enumerate(pred_lines):
        if pid(p, pj) in matched_pred:
            continue
        near = _nearest([(gid(g, gi), table[gi][pj]) for gi, g in enumerate(gold_lines)])
        if near is None:
            fp_rows.append({"prediction_id": pid(p, pj), "cause": "SPURIOUS"})
            continue
        causes = _cause(near[1])
        fp_rows.append({"prediction_id": pid(p, pj), "nearest_gold": near[0],
                        "cause": "+".join(causes) if causes else "TAKEN_BY_OTHER",
                        **{k: round(v, 3) for k, v in near[1].items()}})
    return {"false_negatives": fn_rows, "false_positives": fp_rows,
            "fn_cause_counts": dict(Counter(_primary(r["cause"]) for r in fn_rows)),
            "fp_cause_counts": dict(Counter(_primary(r["cause"]) for r in fp_rows))}


def _primary(cause: str) -> str:
    return cause.split("+")[0] if "+" in cause else cause


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.fmean(values)), 4) if values else None


def _median(values: Sequence[float]) -> float | None:
    return round(float(statistics.median(values)), 4) if values else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score_lane(case_list: Mapping[str, Any], *, runs_root: Path, lane: str, pred_file: str,
               frame_records: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    import score as sc

    rows = []
    for entry in case_list["cases"]:
        lane_dir = runs_root / lane / entry["case_id"]
        pred_path = lane_dir / pred_file
        tier = entry["tier"]
        lidar = any(f.startswith("LIDAR_DEMO") for f in entry.get("flags") or [])
        base = {"case_id": entry["case_id"], "family_id": entry["family_id"], "tier": tier,
                "in_gold11": bool(entry.get("in_gold11")), "gold11_id": entry.get("gold11_id"),
                "gold_kind": "HUMAN" if tier in {"GOLD11", "GOLD_LINES_HUMAN_NON_GOLD11"} else "AI",
                "staged": bool(entry.get("staged_or_incomplete")), "stage_hints": entry.get("stage_hints"),
                "has_cad": bool(entry.get("cad_path")), "lidar_demo_synthetic": lidar,
                "flags": entry.get("flags") or [], "scan_path": entry["scan_path"],
                "scan_name": Path(entry["scan_path"]).name, "predictions": str(pred_path)}
        record_path = lane_dir / "run_record.json"
        if record_path.is_file():
            record = bb.read_json(record_path)
            base["run"] = {"status": record.get("status"), "lane_wall_s": record.get("lane_wall_s"),
                           "heavy_step_wall_s": record.get("heavy_step_wall_s"),
                           "peak_rss_gib": round((record.get("peak_rss_bytes") or 0) / 2**30, 3),
                           "timed_out_steps": [s["name"] for s in record.get("steps", []) if s.get("timed_out")],
                           "finished_at": record.get("finished_at")}
        else:
            base["run"] = {"status": "NOT_RUN"}
        row = sc.score_case({"case_id": entry["case_id"], "tier": tier, "family_id": entry["family_id"],
                             "gold": entry["gold"], "label_complete": entry.get("label_complete", True),
                             "predictions": str(pred_path)}, frame_records=frame_records)
        out = {**base, **{k: v for k, v in row.items() if k not in base or k in ("tier",)}}
        if row.get("disposition") in {"SCORED_COMPLETE_LABEL", "SCORED_RECALL_ONLY_PARTIAL_LABEL"}:
            gold = sc.load_gold(entry["gold"], frame_records=frame_records)
            preds = bb.load_eval_lines(pred_path) if pred_path.is_file() else {"owned_bend_regions": []}
            pred_lines = [r for r in preds.get("owned_bend_regions") or []
                          if bb.valid_line(r.get("line_start"), r.get("line_end")) or r.get("span_endpoints")]
            out["failure"] = preds.get("failure")
            matches = row.get("matches") or []
            out["angle_mae_deg"] = _mean([m["axis_angle_deg"] for m in matches])
            out["position_error_mean_mm"] = _mean([m["transverse_offset_mm"] for m in matches])
            out["position_error_median_mm"] = _median([m["transverse_offset_mm"] for m in matches])
            out["gold_span_overlap_mean"] = _mean([m["gold_span_overlap"] for m in matches])
            out["diagnostic"] = diagnose(gold["lines"], pred_lines, row)
        elif row.get("disposition") == "ZERO_BEND_CONTROL":
            preds = bb.load_eval_lines(pred_path) if pred_path.is_file() else {}
            out["failure"] = preds.get("failure")
        rows.append(out)
    return rows


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    partial = [r for r in rows if r.get("disposition") == "SCORED_RECALL_ONLY_PARTIAL_LABEL"]
    controls = [r for r in rows if r.get("disposition") == "ZERO_BEND_CONTROL"]
    tp = sum(r["true_positive"] for r in complete)
    fp = sum(r["false_positive"] for r in complete)
    fn = sum(r["false_negative"] for r in complete)
    micro = _prf(tp, fp, fn)
    matches = [m for r in complete + partial for m in r.get("matches") or []]
    fn_causes: Counter = Counter()
    fp_causes: Counter = Counter()
    for r in complete:
        fn_causes.update((r.get("diagnostic") or {}).get("fn_cause_counts") or {})
        fp_causes.update((r.get("diagnostic") or {}).get("fp_cause_counts") or {})
    return {
        "scans": len(complete), "families": len({r["family_id"] for r in complete}),
        "gold_lines": tp + fn, "predicted_lines": tp + fp,
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "micro_precision": round(micro["precision"], 6), "micro_recall": round(micro["recall"], 6),
        "micro_f1": round(micro["f1"], 10),
        "macro_precision": _mean([r["precision"] for r in complete]),
        "macro_recall": _mean([r["recall"] for r in complete]),
        "macro_f1": _mean([r["f1"] for r in complete]),
        "family_macro_f1": _mean([_prf(sum(x["true_positive"] for x in fr), sum(x["false_positive"] for x in fr),
                                       sum(x["false_negative"] for x in fr))["f1"]
                                  for fr in _by(complete, lambda r: r["family_id"]).values()]),
        "exact_scans": sum(bool(r.get("exact")) for r in complete),
        "scans_with_f1_ge_0_8": sum(r["f1"] >= 0.8 for r in complete),
        "scans_with_zero_tp": sum(r["true_positive"] == 0 for r in complete),
        "scans_with_no_predictions": sum(r["prediction_count"] == 0 for r in complete),
        "angle_mae_deg_matched": _mean([m["axis_angle_deg"] for m in matches]),
        "position_error_mean_mm_matched": _mean([m["transverse_offset_mm"] for m in matches]),
        "position_error_median_mm_matched": _median([m["transverse_offset_mm"] for m in matches]),
        "fn_cause_counts_diagnostic": dict(fn_causes.most_common()),
        "fp_cause_counts_diagnostic": dict(fp_causes.most_common()),
        "lane_status_counts": dict(Counter(str(r.get("lane_status")) for r in complete).most_common()),
        "partial_label_recall_only": {"scans": len(partial),
                                      "true_positive": sum(r["true_positive"] for r in partial),
                                      "gold_lines": sum(r["gold_count"] for r in partial)},
        "zero_bend_controls": {"scans": len(controls),
                               "false_positive_lines": sum(r.get("false_positive_lines", 0) for r in controls),
                               "scans_with_false_positive": sorted(r["case_id"] for r in controls
                                                                   if r.get("false_positive_lines"))},
    }


def _by(rows: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]) -> dict[Any, list]:
    out: dict[Any, list] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


COHORTS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "ALL": lambda r: True,
    "ALL_EXCL_LIDAR_DEMO": lambda r: not r["lidar_demo_synthetic"],
    "GOLD11": lambda r: r["tier"] == "GOLD11",
    "NON_GOLD11": lambda r: r["tier"] != "GOLD11",
    "NON_GOLD11_EXCL_LIDAR_DEMO": lambda r: r["tier"] != "GOLD11" and not r["lidar_demo_synthetic"],
    "HUMAN_GOLD_ALL": lambda r: r["gold_kind"] == "HUMAN",
    "HUMAN_GOLD_NON_GOLD11": lambda r: r["tier"] == "GOLD_LINES_HUMAN_NON_GOLD11",
    "AI_GOLD": lambda r: r["gold_kind"] == "AI",
    "AI_GOLD_EXCL_LIDAR_DEMO": lambda r: r["gold_kind"] == "AI" and not r["lidar_demo_synthetic"],
    "LIDAR_DEMO_SYNTHETIC": lambda r: r["lidar_demo_synthetic"],
    "FULL_SCANS": lambda r: not r["staged"],
    "STAGED_SCANS": lambda r: r["staged"],
    "FULL_SCANS_EXCL_LIDAR_DEMO": lambda r: not r["staged"] and not r["lidar_demo_synthetic"],
    "CAD_AVAILABLE": lambda r: r["has_cad"],
    "NO_CAD": lambda r: not r["has_cad"],
    "CAD_AVAILABLE_EXCL_LIDAR_DEMO": lambda r: r["has_cad"] and not r["lidar_demo_synthetic"],
    "NON_GOLD11_CAD_AVAILABLE_EXCL_LIDAR_DEMO": lambda r: r["tier"] != "GOLD11" and r["has_cad"]
    and not r["lidar_demo_synthetic"],
}


def per_scan_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keep = ("case_id", "family_id", "tier", "gold11_id", "scan_name", "staged", "has_cad", "lidar_demo_synthetic",
            "lane_status", "disposition", "gold_count", "prediction_count", "true_positive", "false_positive",
            "false_negative", "precision", "recall", "f1", "angle_mae_deg", "position_error_mean_mm",
            "position_error_median_mm")
    out = []
    for r in rows:
        item = {k: r.get(k) for k in keep}
        item["failure_reason"] = (r.get("failure") or {}).get("reason") if isinstance(r.get("failure"), dict) else None
        item["worker_error"] = (r.get("failure") or {}).get("worker_error") if isinstance(r.get("failure"), dict) else None
        item["fn_causes"] = (r.get("diagnostic") or {}).get("fn_cause_counts")
        item["fp_causes"] = (r.get("diagnostic") or {}).get("fp_cause_counts")
        item["lane_wall_s"] = (r.get("run") or {}).get("lane_wall_s")
        item["peak_rss_gib"] = (r.get("run") or {}).get("peak_rss_gib")
        out.append(item)
    return out


def worst(rows: Sequence[Mapping[str, Any]], n: int = 15) -> list[dict[str, Any]]:
    complete = [r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    ranked = sorted(complete, key=lambda r: (r["f1"], -r["false_negative"] - r["false_positive"], r["case_id"]))
    return per_scan_table(ranked[:n])


def build(case_list_path: Path, *, runs_root: Path, lane: str, pred_file: str, frame_records_path: Path | None,
          title: str | None = None) -> dict[str, Any]:
    import score as sc

    case_list = bb.read_json(case_list_path)
    frame_records = bb.read_json(frame_records_path) if frame_records_path else None
    rows = score_lane(case_list, runs_root=runs_root, lane=lane, pred_file=pred_file, frame_records=frame_records)
    thresholds, _ = sc._matcher()
    resources = [r["run"] for r in rows if r.get("run", {}).get("lane_wall_s") is not None]
    report = {
        "schema": "sherman.broad_baseline.broad_scores.v1",
        "title": title or f"{lane}:{pred_file}",
        "created_at": bb.utc_now(),
        "lane": lane, "prediction_file": pred_file,
        "case_list": {"path": str(case_list_path), "file_sha256": bb.sha256_file(case_list_path),
                      "case_list_sha256": case_list.get("case_list_sha256"), "cases": len(case_list["cases"])},
        "frame_records": None if frame_records_path is None else {"path": str(frame_records_path),
                                                                   "sha256": bb.sha256_file(frame_records_path)},
        "matcher": {"source": "scripts/evaluate_f1_against_manual_gold.match_lines (via score.score_case)",
                    "axis_angle_deg": thresholds.angle_deg, "transverse_offset_mm": thresholds.transverse_offset_mm,
                    "gold_span_overlap": thresholds.gold_span_overlap,
                    "prediction_span_overlap": thresholds.prediction_span_overlap,
                    "assignment": "one_to_one_minimum_cost Hungarian", "frame": bb.RAW_FRAME},
        "conventions": {"micro": "summed TP/FP/FN over complete-label line scans",
                        "macro": "mean of per-scan precision/recall/F1 (scan with no predictions -> P=R=F1=0)",
                        "family_macro_f1": "mean over families of family-micro F1",
                        "zero_bend_controls": "not in line-F1; every predicted line counted as a false positive",
                        "diagnostic_causes": "nearest-counterpart classification; NOT the metric",
                        "gold_tiers": "GOLD11 (owner-verified except C05/C06 Codex multiview), "
                                      "GOLD_LINES_HUMAN_NON_GOLD11 (owner-verified), AI_GOLD_ACCEPTED (agent-adjudicated)"},
        "cohorts": {name: aggregate([r for r in rows if fn(r)]) for name, fn in COHORTS.items()},
        "by_tier": {t: aggregate(rs) for t, rs in _by(rows, lambda r: r["tier"]).items()},
        "by_family": {f: {k: v for k, v in aggregate(rs).items()
                          if k in ("scans", "gold_lines", "predicted_lines", "true_positive", "false_positive",
                                   "false_negative", "micro_precision", "micro_recall", "micro_f1", "macro_f1")}
                      for f, rs in _by([r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"],
                                       lambda r: r["family_id"]).items()},
        "resources": {"runs_with_record": len(resources),
                      "lane_wall_s_total": round(sum(r["lane_wall_s"] for r in resources), 1),
                      "lane_wall_s_median": _median([r["lane_wall_s"] for r in resources]),
                      "lane_wall_s_max": max([r["lane_wall_s"] for r in resources] or [0]),
                      "peak_rss_gib_max": max([r.get("peak_rss_gib") or 0 for r in resources] or [0]),
                      "runs_over_4_gib": sorted(r_case["case_id"] for r_case in rows
                                                if (r_case.get("run") or {}).get("peak_rss_gib", 0) > 4.0),
                      "timeouts": sorted(r_case["case_id"] for r_case in rows
                                         if (r_case.get("run") or {}).get("timed_out_steps")),
                      "not_run": sorted(r_case["case_id"] for r_case in rows
                                        if (r_case.get("run") or {}).get("status") == "NOT_RUN")},
        "excluded_before_scoring": [{"case_id": r["case_id"], "reason": r.get("reason")} for r in rows
                                    if r.get("disposition") == "EXCLUDED_BEFORE_SCORING"],
        "worst_15": worst(rows, 15),
        "per_scan": per_scan_table(rows),
        "cases": rows,
    }
    return report


def print_summary(report: Mapping[str, Any]) -> None:
    print(f"== {report['title']}  (case list {report['case_list']['case_list_sha256'][:12]})")
    for name, c in report["cohorts"].items():
        print(f"  {name:42s} scans={c['scans']:3d} gold={c['gold_lines']:3d} pred={c['predicted_lines']:3d} "
              f"TP={c['true_positive']:3d} FP={c['false_positive']:3d} FN={c['false_negative']:3d} "
              f"P={c['micro_precision']:.3f} R={c['micro_recall']:.3f} F1={c['micro_f1']:.4f} "
              f"macroF1={c['macro_f1']} zeroFP={c['zero_bend_controls']['false_positive_lines']}")
    if report["excluded_before_scoring"]:
        print("  excluded:", report["excluded_before_scoring"])
