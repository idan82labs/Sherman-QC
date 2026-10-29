#!/usr/bin/env python3
"""Score evaluator-format predictions against gold lines with the frozen F1 matcher.

Matcher: scripts/evaluate_f1_against_manual_gold.match_lines with its default
Thresholds (undirected axis angle <= 5 deg, symmetric transverse midpoint offset
<= 4 mm, overlap >= 60 % of BOTH the gold and the predicted span, one-to-one
minimum-cost Hungarian assignment). Aggregates are micro sums with the
evaluator's conventions (precision/recall 0.0 on an empty denominator, F1 0.0
when P+R = 0). Everything is in the raw-scan frame.

Gold kinds (case-map "gold": {"kind": ..., "path": ...})
  gold11_catalog  data/bend_corpus/annotations/bend_line_gold_catalog_v1.json, selected by
                  "scan_sha256" (or "catalog_case_id"). reference_mesh_raw label sets need a
                  frame record from gold_frames.py (--frame-records) with residual <= 0.01 mm,
                  else the case is excluded NO_RECOVERABLE_TRANSFORM before scoring.
  labels_manual   output/bend_label_*/bend_instance_labels.json (manual_bends); frame as above.
  ai_studio       data/bend_corpus/annotations/ai_bend_lines_v1/<asset>.json (kind
                  sherman_qc_human_bend_labels, frame source_ply_raw == raw coordinates of the
                  labelled file; sha bound through provenance/<asset>.json asset_sha256).
                  no_bends_present=true -> zero-bend FALSE-POSITIVE CONTROL (no line-F1 use).
  bench_lines     docs/production/runlog/benchmark_v2/label_lines*/<scan_id>/lines.json (verified
                  bench_labeling output; bends with status ok; label_line = apex locus; STL-only
                  scans have only this). --locus sharp uses the plane-intersection line.
  generic         {"frame": "scan_raw", "scan_sha256"?, "bends": [{bend_id, line_start, line_end}]}

Label completeness: "label_complete": false (ACCEPT_PARTIAL) scores recall and
position only - matched lines are TP, unmatched gold are FN, unmatched predictions
are NOT false positives (reported as unscored); precision/F1 are undefined for
those cases and they are aggregated separately from complete-label cases.

Tiers are aggregated separately (GOLD_LINES_HUMAN vs AI_GOLD_ACCEPTED ...); a
combined block is labelled as such.

Subcommands
  broad               broad run: --case-list case_list_broad_v2.json --runs-root .../runs --lane best_f1|runtime
                      --frame-records broad_frame_records.json --out-dir .../broad_baseline  (score_broad.py:
                      per-scan P/R/F1, angle MAE, position error, micro + macro, cohorts, worst 15)
  case-list           --case-list case_list_consumed_v1.json --pred-template
                      'docs/production/runlog/broad_baseline/runs/{case_id}/best_f1/predictions_best_f1.json'
                      (tiers split; LIDAR demos and possible CAD meshes get their own tier suffix)
  cases               --case-map MAP.json --out REPORT.json [--frame-records FR.json]
  single              --gold-kind K --gold PATH --pred PRED.json [--scan-sha SHA] [--label-incomplete]
  gold11-historical   reproduce the historical Gold11 UNION line-F1 0.4580152672 from the
                      retained prediction bytes (no inference): --source retained|original
  gold11              score fresh Gold11 predictions: --pred-dir DIR with <CG id>/predictions_<lane>.json
                      or --pred-map {CG id: eval file}, --lane NAME, --frame-records FR.json

Case map schema (sherman.broad_baseline.score_cases.v1):
  {"cases": [{"case_id": "sc_03395097cd96", "family_id": "15651004", "tier": "AI_GOLD_ACCEPTED",
              "gold": {"kind": "ai_studio", "path": "data/.../1c62d21a1fceedefd236259a.json"},
              "label_complete": true, "predictions": ".../predictions_best_f1.json"}]}
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold11_cases as g11  # noqa: E402

CASES_SCHEMA = "sherman.broad_baseline.score_cases.v1"
DEFAULT_FRAME_RECORDS = bb.DEFAULT_RUNLOG / "gold11_frame_recovery.json"


def _frame_records_arg(path: Path | None, *, default: bool = True) -> tuple[dict | None, str | None]:
    if path is None and default and DEFAULT_FRAME_RECORDS.is_file():
        path = DEFAULT_FRAME_RECORDS
    if path is None:
        return None, None
    return bb.read_json(path), str(path)


# ------------------------------------------------------------------ matcher

def _matcher():
    from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines

    return Thresholds(), match_lines


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "gold_bend_count": tp + fn, "prediction_count": tp + fp,
            "precision": precision, "recall": recall, "f1": f1,
            "precision_6": round(precision, 6), "recall_6": round(recall, 6), "f1_6": round(f1, 6)}


# ---------------------------------------------------------------- gold load

class Excluded(Exception):
    def __init__(self, reason: str, **detail: Any):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _bends(rows: Sequence[Mapping[str, Any]], *, prefix: str = "G") -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows, start=1):
        ends = bb.endpoints(row)
        out.append(bb.as_line(str(row.get("bend_id") or f"{prefix}{index:02d}"), ends[0], ends[1]))
    return out


def _frame_record(records: Mapping[str, Any] | None, *, scan_sha: str, catalog_case_id: str | None) -> dict | None:
    if not records:
        return None
    for record in records.get("cases", []):
        if record.get("scan_sha256") == scan_sha and (catalog_case_id is None
                                                       or record.get("catalog_case_id") == catalog_case_id):
            return record
    return None


def _to_raw(lines: list[dict[str, Any]], record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    if record.get("status") != "RECOVERED":
        raise Excluded(record.get("status") or "NO_RECOVERABLE_TRANSFORM")
    residual = float(record.get("residual_max_mm", float("inf")))
    if residual > bb.TRANSFORM_RESIDUAL_MAX_MM:
        raise Excluded("TRANSFORM_RESIDUAL_TOO_LARGE", residual_max_mm=residual)
    transform = bb.validate_h(record["raw_to_reference"])
    inverse = np.linalg.inv(transform)
    identity = bool(np.allclose(transform, np.eye(4), atol=0, rtol=0))
    mapped = []
    for line in lines:
        ends = bb.apply_h(inverse, [line["line_start"], line["line_end"]])
        mapped.append(bb.as_line(line["bend_id"], ends[0], ends[1]))
    return mapped, {"method": record.get("method"), "residual_max_mm": residual, "identity": identity}


def load_gold(spec: Mapping[str, Any], *, frame_records: Mapping[str, Any] | None, locus: str = "apex",
              frame_mode: str = "recovered") -> dict:
    """frame_mode "recovered" (plan s10 6(a), default) or "historical_direct" (labels used as stored,
    exactly how the cached 0.458 control was computed; only for reproducing history)."""
    kind = spec["kind"]
    path = bb.expand(spec.get("path") or bb.GOLD_CATALOG)
    if kind == "gold11_catalog":
        catalog = bb.read_json(path)
        rows = [r for r in catalog["cases"]
                if (spec.get("scan_sha256") and r["scan_sha256"] == spec["scan_sha256"])
                or (spec.get("catalog_case_id") and r["case_id"] == spec["catalog_case_id"])]
        if len(rows) != 1:
            raise Excluded("GOLD_CASE_NOT_FOUND", matches=len(rows))
        row = rows[0]
        lines = _bends(row["bends"])
        frame = row["coordinate_frame"]
        info: dict[str, Any] = {"label_frame": frame, "truth_tier": row.get("truth_tier"),
                                "catalog_case_id": row["case_id"], "catalog_sha256": bb.sha256_file(path)}
        if frame in {"reference_mesh_raw", "scan_raw"} and frame_mode == "historical_direct":
            info["frame_transform"] = {"method": "HISTORICAL_DIRECT_NO_TRANSFORM", "identity": True}
        elif frame in {"reference_mesh_raw", "scan_raw"}:
            record = _frame_record(frame_records, scan_sha=row["scan_sha256"], catalog_case_id=row["case_id"])
            if record is None:
                if frame == "reference_mesh_raw":
                    raise Excluded("NO_RECOVERABLE_TRANSFORM", note="no frame record (run gold_frames.py)")
                info["frame_transform"] = {"method": "DECLARED_SCAN_RAW_UNVERIFIED", "identity": True}
            else:
                lines, info["frame_transform"] = _to_raw(lines, record)
        else:
            raise Excluded(f"UNKNOWN_FRAME:{frame}")
        return {"lines": lines, "scan_sha256": row["scan_sha256"], "family_id": row["part_id"],
                "zero_bend_control": False, "gold_source": str(path), **info}
    if kind == "labels_manual":
        payload = bb.read_json(path)
        lines = _bends(payload.get("manual_bends") or [])
        frame = payload.get("coordinate_frame")
        sha = spec.get("scan_sha256")
        info = {"label_frame": frame}
        record = _frame_record(frame_records, scan_sha=sha, catalog_case_id=None) if sha else None
        if record is None:
            raise Excluded("NO_RECOVERABLE_TRANSFORM", note="labels_manual needs scan_sha256 + frame record")
        lines, info["frame_transform"] = _to_raw(lines, record)
        return {"lines": lines, "scan_sha256": sha, "zero_bend_control": False, "gold_source": str(path), **info}
    if kind == "ai_studio":
        payload = bb.read_json(path)
        if payload.get("coordinate_frame") != "source_ply_raw":
            raise Excluded(f"UNKNOWN_FRAME:{payload.get('coordinate_frame')}")
        provenance_path = path.parent / "provenance" / path.name
        provenance = bb.read_json(provenance_path) if provenance_path.is_file() else {}
        sha = provenance.get("asset_sha256") or spec.get("scan_sha256")
        lines = _bends(payload.get("bends") or [], prefix="B")
        return {"lines": lines, "scan_sha256": sha, "label_frame": "source_ply_raw",
                "frame_transform": {"method": "SOURCE_PLY_RAW_IS_SCAN_RAW", "identity": True},
                "zero_bend_control": bool(payload.get("no_bends_present")) and not lines,
                "review_status": payload.get("review_status"), "annotator": payload.get("annotator"),
                "asset_id": payload.get("asset_id"),
                "asset": f"{payload.get('source')}:{payload.get('source_relative_path')}",
                "gold_source": str(path)}
    if kind == "bench_lines":
        payload = bb.read_json(path)
        if payload.get("coordinate_frame") != "source_ply_raw":
            raise Excluded(f"UNKNOWN_FRAME:{payload.get('coordinate_frame')}")
        lines = []
        for bend in payload.get("bends") or []:
            if bend.get("status") != "ok":
                continue
            segment = bend.get("label_line") if locus == "apex" else bend.get(locus)
            if not segment:
                continue
            lines.append(bb.as_line(bend["bend_id"], segment["start"], segment["end"]))
        return {"lines": lines, "scan_sha256": payload.get("scan_sha256"), "label_frame": "source_ply_raw",
                "frame_transform": {"method": "SOURCE_PLY_RAW_IS_SCAN_RAW", "identity": True},
                "zero_bend_control": bool(payload.get("no_bends_present")) and not lines,
                "locus": locus, "gold_source": str(path)}
    if kind == "generic":
        payload = bb.read_json(path) if spec.get("path") else dict(spec)
        if payload.get("frame", "scan_raw") != "scan_raw":
            raise Excluded(f"UNKNOWN_FRAME:{payload.get('frame')}")
        lines = _bends(payload.get("bends") or [])
        return {"lines": lines, "scan_sha256": payload.get("scan_sha256"), "label_frame": "scan_raw",
                "zero_bend_control": bool(payload.get("no_bends_present")) and not lines,
                "gold_source": str(path) if spec.get("path") else "inline"}
    raise Excluded(f"UNKNOWN_GOLD_KIND:{kind}")


# ------------------------------------------------------------------ scoring

def score_case(case: Mapping[str, Any], *, frame_records: Mapping[str, Any] | None, locus: str = "apex",
               preloaded_predictions: Mapping[str, Any] | None = None,
               frame_mode: str = "recovered") -> dict[str, Any]:
    thresholds, match_lines = _matcher()
    record: dict[str, Any] = {"case_id": case["case_id"], "tier": case.get("tier", "UNSPECIFIED"),
                              "family_id": case.get("family_id"),
                              "label_complete": bool(case.get("label_complete", True)),
                              "predictions_path": case.get("predictions")}
    try:
        gold = load_gold(case["gold"], frame_records=frame_records, locus=locus, frame_mode=frame_mode)
    except Excluded as error:
        record.update({"disposition": "EXCLUDED_BEFORE_SCORING", "reason": error.reason, **error.detail})
        return record
    except (OSError, KeyError, ValueError) as error:
        record.update({"disposition": "EXCLUDED_BEFORE_SCORING", "reason": f"GOLD_UNREADABLE:{error}"})
        return record
    record["family_id"] = record["family_id"] or gold.get("family_id")
    record["gold"] = {k: v for k, v in gold.items() if k != "lines"}
    predictions = preloaded_predictions
    lane_status = None
    if predictions is None:
        pred_path = case.get("predictions")
        if not pred_path or not bb.expand(pred_path).is_file():
            predictions = {"owned_bend_regions": [], "status": "PREDICTIONS_MISSING", "scan": {}}
        else:
            predictions = bb.load_eval_lines(bb.expand(pred_path))
    lane_status = predictions.get("status")
    pred_sha = (predictions.get("scan") or {}).get("sha256")
    if gold.get("scan_sha256") and pred_sha and gold["scan_sha256"] != pred_sha:
        record.update({"disposition": "EXCLUDED_BEFORE_SCORING", "reason": "SCAN_SHA_MISMATCH",
                       "gold_scan_sha256": gold["scan_sha256"], "prediction_scan_sha256": pred_sha})
        return record
    pred_lines = [row for row in predictions.get("owned_bend_regions") or []
                  if bb.valid_line(row.get("line_start"), row.get("line_end")) or row.get("span_endpoints")]
    record["lane_status"] = lane_status
    record["lane"] = predictions.get("lane")
    if gold["zero_bend_control"]:
        record.update({"disposition": "ZERO_BEND_CONTROL", "gold_count": 0,
                       "prediction_count": len(pred_lines), "false_positive_lines": len(pred_lines)})
        return record
    matched = match_lines(gold["lines"], pred_lines, thresholds)
    record.update({k: matched[k] for k in ("gold_count", "prediction_count", "true_positive", "matches",
                                           "unmatched_gold_ids", "unmatched_prediction_ids")})
    if record["label_complete"]:
        record.update({"disposition": "SCORED_COMPLETE_LABEL",
                       "false_positive": matched["false_positive"], "false_negative": matched["false_negative"],
                       "exact": matched["exact"]})
        record.update({k: v for k, v in prf(matched["true_positive"], matched["false_positive"],
                                            matched["false_negative"]).items() if k in ("precision", "recall", "f1")})
    else:
        record.update({"disposition": "SCORED_RECALL_ONLY_PARTIAL_LABEL",
                       "false_negative": matched["false_negative"], "false_positive": None,
                       "unscored_unmatched_predictions": matched["false_positive"],
                       "recall": _ratio(matched["true_positive"], matched["gold_count"]),
                       "precision": None, "f1": None, "exact": None})
    return record


def _position_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [m for row in rows for m in row.get("matches") or []]
    if not matches:
        return {"matched_pairs": 0}

    def summary(key: str) -> dict[str, float]:
        values = sorted(float(m[key]) for m in matches)
        return {"median": round(statistics.median(values), 4),
                "p90": round(values[min(len(values) - 1, int(np.ceil(0.9 * len(values))) - 1)], 4),
                "max": round(values[-1], 4)}

    return {"matched_pairs": len(matches),
            "axis_angle_deg": summary("axis_angle_deg"),
            "transverse_offset_mm": summary("transverse_offset_mm"),
            "gold_span_overlap": summary("gold_span_overlap"),
            "prediction_span_overlap": summary("prediction_span_overlap")}


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    partial = [r for r in rows if r.get("disposition") == "SCORED_RECALL_ONLY_PARTIAL_LABEL"]
    controls = [r for r in rows if r.get("disposition") == "ZERO_BEND_CONTROL"]
    excluded = [r for r in rows if r.get("disposition") == "EXCLUDED_BEFORE_SCORING"]
    tp = sum(r["true_positive"] for r in complete)
    fp = sum(r["false_positive"] for r in complete)
    fn = sum(r["false_negative"] for r in complete)
    complete_block = {"case_count": len(complete), **prf(tp, fp, fn),
                      "exact_part_count": sum(bool(r["exact"]) for r in complete),
                      "exact_part_rate": round(_ratio(sum(bool(r["exact"]) for r in complete), len(complete)), 6),
                      "lane_failures": sorted(r["case_id"] for r in complete
                                              if r.get("lane_status") in {"LANE_FAILED", "PREDICTIONS_MISSING"})}
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in complete:
        families[str(r.get("family_id"))].append(r)
    family_recall = {f: _ratio(sum(r["true_positive"] for r in rs), sum(r["gold_count"] for r in rs))
                     for f, rs in sorted(families.items())}
    complete_block["family_recall"] = family_recall
    complete_block["minimum_family_recall"] = min(family_recall.values()) if family_recall else None
    ptp = sum(r["true_positive"] for r in partial)
    pgold = sum(r["gold_count"] for r in partial)
    partial_block = {"case_count": len(partial), "true_positive": ptp, "gold_bend_count": pgold,
                     "false_negative": pgold - ptp, "recall": _ratio(ptp, pgold),
                     "unscored_unmatched_predictions": sum(r["unscored_unmatched_predictions"] for r in partial),
                     "note": "label_complete=false: recall/position only, no precision/F1"}
    recall_all = {"true_positive": tp + ptp, "gold_bend_count": tp + fn + pgold,
                  "recall": _ratio(tp + ptp, tp + fn + pgold),
                  "note": "complete + partial label cases, recall only"}
    control_block = {"case_count": len(controls),
                     "false_positive_lines": sum(r["false_positive_lines"] for r in controls),
                     "cases_with_false_positive": sorted(r["case_id"] for r in controls if r["false_positive_lines"]),
                     "note": "zero-bend controls: every predicted line is a false positive; not in line-F1"}
    lane_status_counts: dict[str, int] = defaultdict(int)
    for r in complete + partial + controls:
        lane_status_counts[str(r.get("lane_status"))] += 1
    return {"lane_status_counts": dict(sorted(lane_status_counts.items())),
            "complete_labels": complete_block, "partial_labels_recall_only": partial_block,
            "recall_all_scored_cases": recall_all, "zero_bend_controls": control_block,
            "position_of_matches": _position_stats(complete + partial),
            "excluded_before_scoring": [{"case_id": r["case_id"], "reason": r.get("reason")} for r in excluded]}


def build_report(rows: list[dict[str, Any]], *, title: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    tiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tiers[str(row.get("tier"))].append(row)
    thresholds, _ = _matcher()
    return {
        "schema": bb.SCORE_SCHEMA,
        "title": title,
        "created_at": bb.utc_now(),
        "matcher": {"source": "scripts/evaluate_f1_against_manual_gold.match_lines",
                    "axis_angle_deg": thresholds.angle_deg, "transverse_offset_mm": thresholds.transverse_offset_mm,
                    "gold_span_overlap": thresholds.gold_span_overlap,
                    "prediction_span_overlap": thresholds.prediction_span_overlap,
                    "assignment": "one_to_one_minimum_cost (scipy linear_sum_assignment)",
                    "frame": bb.RAW_FRAME},
        "by_tier": {tier: aggregate(tier_rows) for tier, tier_rows in sorted(tiers.items())},
        "all_tiers_combined_NOT_FOR_QUOTING": aggregate(rows) if len(tiers) > 1 else None,
        **dict(extra or {}),
        "cases": rows,
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    for tier, block in report["by_tier"].items():
        c = block["complete_labels"]
        p = block["partial_labels_recall_only"]
        z = block["zero_bend_controls"]
        print(f"[{tier}] complete: cases={c['case_count']} TP={c['true_positive']} FP={c['false_positive']} "
              f"FN={c['false_negative']} P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.10f} "
              f"exact={c['exact_part_count']}/{c['case_count']} minFamR={c['minimum_family_recall']}"
              f" | partial: cases={p['case_count']} R={p['recall']:.4f} | zero-bend FP lines={z['false_positive_lines']}"
              f" | excluded={len(block['excluded_before_scoring'])}")


# ----------------------------------------------------------- subcommands

def cmd_cases(args: argparse.Namespace) -> int:
    case_map = bb.read_json(args.case_map)
    frame_records, frame_path = _frame_records_arg(args.frame_records)
    rows = [score_case(case, frame_records=frame_records, locus=args.locus) for case in case_map["cases"]]
    report = build_report(rows, title=args.title or f"case map {args.case_map}",
                          extra={"case_map": str(args.case_map), "frame_records": frame_path})
    bb.write_json(args.out, report)
    _print_summary(report)
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    case = {"case_id": args.case_id or Path(args.pred).parent.name, "tier": args.tier,
            "family_id": args.family_id, "label_complete": not args.label_incomplete,
            "gold": {"kind": args.gold_kind, "path": str(args.gold), "scan_sha256": args.scan_sha},
            "predictions": str(args.pred)}
    frame_records, frame_path = _frame_records_arg(args.frame_records)
    row = score_case(case, frame_records=frame_records, locus=args.locus)
    report = build_report([row], title=f"single {case['case_id']}", extra={"frame_records": frame_path})
    bb.write_json(args.out, report)
    _print_summary(report)
    return 0


def _retained_payloads(case_id: str, source: str) -> tuple[dict, dict, Path, Path]:
    if source == "retained":
        a0_path = bb.CACHED_CONTROL_INPUTS / "a0" / f"{case_id}.json"
        cad_path = bb.CACHED_CONTROL_INPUTS / "cad" / f"{case_id}.json"
    else:
        a0_path = bb.ROOT / g11.A0_ROOT / g11.token(case_id) / "prediction_owned_regions.json"
        cad_path = bb.ROOT / g11.CAD_ROOT / case_id / "semantic_result.json"
    for path, expected in ((a0_path, g11.a0_sha(case_id)), (cad_path, g11.cad_sha(case_id))):
        actual = bb.sha256_file(path)
        if actual != expected:
            raise ValueError(f"{case_id}: {path} sha256 {actual} != frozen {expected}")
    return bb.read_json(a0_path), bb.read_json(cad_path), a0_path, cad_path


def cmd_gold11_historical(args: argparse.Namespace) -> int:
    import to_eval_format as tef

    frame_records, frame_path = _frame_records_arg(args.frame_records, default=False)
    frame_mode = "recovered" if frame_records else "historical_direct"
    variants = {"scan_only_a0": [], "cad_only_gated": [], "intersection": [], "union": []}
    case_records = []
    for case_id in sorted(g11.CASES):
        a0, cad, a0_path, cad_path = _retained_payloads(case_id, args.source)
        a0_eval = tef.convert_a0(a0, source=a0_path, scan=None)
        cad_eval = tef.convert_cad_semantic(cad, source=cad_path, scan=None)
        for evaluation in (a0_eval, cad_eval):
            evaluation["scan"]["sha256"] = g11.scan_sha(case_id)
        lanes = {"scan_only_a0": a0_eval, "cad_only_gated": cad_eval,
                 "intersection": tef.fuse(a0_eval, cad_eval, mode="intersection"),
                 "union": tef.fuse(a0_eval, cad_eval, mode="union")}
        gold_spec = {"kind": "gold11_catalog", "path": str(args.catalog), "scan_sha256": g11.scan_sha(case_id)}
        for name, payload in lanes.items():
            row = score_case({"case_id": case_id, "tier": name, "family_id": g11.family(case_id),
                              "gold": gold_spec, "label_complete": True}, frame_records=frame_records,
                             preloaded_predictions=payload, frame_mode=frame_mode)
            variants[name].append(row)
        case_records.append({"case_id": case_id, "a0": str(a0_path), "cad": str(cad_path),
                             **{name: {k: variants[name][-1].get(k) for k in ("prediction_count", "true_positive",
                                                                              "false_positive", "false_negative")}
                                for name in variants}})
    results = {}
    for name, rows in variants.items():
        block = aggregate(rows)
        expected = g11.HISTORICAL[name]
        c = block["complete_labels"]
        results[name] = {**block, "historical_expected": expected,
                         "matches_historical": (c["prediction_count"], c["true_positive"], c["false_positive"],
                                                c["false_negative"]) == (expected["lines"], expected["tp"],
                                                                         expected["fp"], expected["fn"])
                         and abs(c["f1"] - expected["f1"]) < 1e-12}
    report = {"schema": bb.SCORE_SCHEMA, "title": "Gold11 historical UNION reproduction (retained bytes)",
              "created_at": bb.utc_now(), "prediction_source": args.source,
              "frame_mode": frame_mode, "frame_records": frame_path,
              "catalog": {"path": str(args.catalog), "sha256": bb.sha256_file(args.catalog),
                          "expected": g11.CATALOG_SHA256},
              "variants": results, "cases": case_records}
    bb.write_json(args.out, report)
    for name, block in results.items():
        c = block["complete_labels"]
        print(f"{name:15s} lines={c['prediction_count']:3d} TP={c['true_positive']:2d} FP={c['false_positive']:2d} "
              f"FN={c['false_negative']:2d} P={c['precision']:.10f} R={c['recall']:.10f} F1={c['f1']:.10f} "
              f"minFamR={c['minimum_family_recall']:.4f} match_historical={block['matches_historical']}")
    return 0 if results["union"]["matches_historical"] else 3


def cmd_gold11(args: argparse.Namespace) -> int:
    frame_records, frame_path = _frame_records_arg(args.frame_records)
    pred_map = bb.read_json(args.pred_map) if args.pred_map else {}
    rows = []
    for case_id in sorted(g11.CASES):
        if args.case and case_id not in args.case:
            continue
        pred = pred_map.get(case_id)
        if pred is None and args.pred_dir:
            pred = str(Path(args.pred_dir) / case_id / f"predictions_{args.lane}.json")
        rows.append(score_case({"case_id": case_id, "tier": "GOLD11_" + args.lane, "family_id": g11.family(case_id),
                                "gold": {"kind": "gold11_catalog", "path": str(args.catalog),
                                         "scan_sha256": g11.scan_sha(case_id)},
                                "label_complete": True, "predictions": pred}, frame_records=frame_records))
    report = build_report(rows, title=f"Gold11 fresh lane {args.lane}",
                          extra={"historical_union_f1": g11.HISTORICAL_UNION_F1, "frame_records": frame_path})
    bb.write_json(args.out, report)
    _print_summary(report)
    return 0


def cmd_case_list(args: argparse.Namespace) -> int:
    """Score one lane over a build_case_list.py case list.

    --pred-template is formatted per case with {case_id}, {gold11_id}, {family_id}, e.g.
    docs/production/runlog/broad_baseline/runs/{case_id}/best_f1/predictions_best_f1.json
    """
    case_list = bb.read_json(args.case_list)
    frame_records, frame_path = _frame_records_arg(args.frame_records)
    rows = []
    for entry in case_list["cases"]:
        if args.only and entry["case_id"] not in args.only:
            continue
        if any(flag.startswith("SCAN_SHA_MISMATCH") for flag in entry.get("flags") or []):
            rows.append({"case_id": entry["case_id"], "tier": entry["tier"], "disposition": "EXCLUDED_BEFORE_SCORING",
                         "reason": "SCAN_SHA_MISMATCH"})
            continue
        pred = args.pred_template.format(case_id=entry["case_id"], gold11_id=entry.get("gold11_id"),
                                         family_id=entry["family_id"])
        tier = entry["tier"]
        if any(flag.startswith("LIDAR_DEMO") for flag in entry.get("flags") or []):
            tier += "_LIDAR_DEMO"
        if any(flag.startswith("EXCLUDE_FROM_SCAN_ACCURACY") for flag in entry.get("flags") or []):
            tier += "_POSSIBLE_CAD_MESH"
        rows.append(score_case({"case_id": entry["case_id"], "tier": tier, "family_id": entry["family_id"],
                                "gold": entry["gold"], "label_complete": entry.get("label_complete", True),
                                "predictions": pred}, frame_records=frame_records, locus=args.locus))
    report = build_report(rows, title=args.title or f"case list {args.case_list} :: {args.pred_template}",
                          extra={"case_list": str(args.case_list), "pred_template": args.pred_template,
                                 "frame_records": frame_path})
    bb.write_json(args.out, report)
    _print_summary(report)
    return 0


def cmd_broad(args: argparse.Namespace) -> int:
    """Broad run: score one lane (all its prediction variants) over a broad case list.

    Writes <out-dir>/scores_<lane>.json for the lane's headline file and
    <out-dir>/scores_<lane>__<variant>.json for its diagnostic component files.
    """
    import score_broad as sb

    headline = {"best_f1": "predictions_best_f1.json", "runtime": "predictions_runtime.json"}[args.lane]
    variants = [headline] + [v for v in (args.variant or []) if v != headline]
    frame_path = args.frame_records or (DEFAULT_FRAME_RECORDS if DEFAULT_FRAME_RECORDS.is_file() else None)
    for pred_file in variants:
        report = sb.build(args.case_list, runs_root=args.runs_root, lane=args.lane, pred_file=pred_file,
                          frame_records_path=frame_path, title=f"broad {args.lane}:{pred_file}")
        name = f"scores_{args.lane}.json" if pred_file == headline else \
            f"scores_{args.lane}__{pred_file.removeprefix('predictions_').removesuffix('.json')}.json"
        bb.write_json(args.out_dir / name, report)
        sb.print_summary(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("broad")
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True, help="<runs-root>/<lane>/<case_id>/predictions_*.json")
    p.add_argument("--lane", required=True, choices=("best_f1", "runtime"))
    p.add_argument("--variant", action="append", help="extra prediction file names to score (diagnostic)")
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_broad)
    p = sub.add_parser("case-list")
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--pred-template", required=True)
    p.add_argument("--only", action="append")
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--locus", choices=("apex", "sharp"), default="apex")
    p.add_argument("--title")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_case_list)
    p = sub.add_parser("cases")
    p.add_argument("--case-map", type=Path, required=True)
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--locus", choices=("apex", "sharp"), default="apex")
    p.add_argument("--title")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cases)
    p = sub.add_parser("single")
    p.add_argument("--gold-kind", required=True,
                   choices=("gold11_catalog", "labels_manual", "ai_studio", "bench_lines", "generic"))
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--scan-sha", help="selects the catalog case / binds labels_manual")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--case-id")
    p.add_argument("--family-id")
    p.add_argument("--tier", default="UNSPECIFIED")
    p.add_argument("--label-incomplete", action="store_true", help="label_complete=false: recall/position only")
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--locus", choices=("apex", "sharp"), default="apex")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_single)
    p = sub.add_parser("gold11-historical")
    p.add_argument("--source", choices=("retained", "original"), default="retained")
    p.add_argument("--catalog", type=Path, default=bb.GOLD_CATALOG)
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_gold11_historical)
    p = sub.add_parser("gold11")
    p.add_argument("--pred-dir", type=Path)
    p.add_argument("--pred-map", type=Path)
    p.add_argument("--lane", default="best_f1")
    p.add_argument("--case", action="append")
    p.add_argument("--catalog", type=Path, default=bb.GOLD_CATALOG)
    p.add_argument("--frame-records", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_gold11)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
