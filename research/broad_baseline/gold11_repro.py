#!/usr/bin/env python3
"""Gold11 'come back green' reproduction: historical rescoring, fresh best-F1 lane, active runtime lane.

Research-only. Reads the authorized Gold11 development inputs (the 11 scans, their CAD, the
gold catalog / label files, the retained historical prediction bytes and the historical
per-token intermediates of these 11 cases only). Writes only under
docs/production/runlog/broad_baseline/. Apart from `replay-typed-union` (one monitored heavy
step) it runs no inference: the fresh lanes are run beforehand by run_benchmark.py /
run_lane.py (one heavy job at a time), and spatial_matching_probe.py holds the runtime probe.

Reproduce the 2026-10-17 record
  run_benchmark.py --case-list docs/production/runlog/broad_baseline/case_list_consumed_v1.json \
      --runs-root docs/production/runlog/broad_baseline/gold11_repro/runs --work-root <scratch> \
      --lanes runtime,best_f1 --no-score --only <the 11 GOLD11 sc_ ids>
  score.py gold11-historical --source original [--frame-records ...] --out .../historical/...
  gold11_repro.py evaluator-crosscheck | replay-typed-union ... | fresh-crosscheck |
                  runtime-determinism --rerun-root ... | summarize [--findings F.json]

Subcommands
  evaluator-crosscheck  Independent check of the historical 0.4580152672 with the frozen
                        evaluator CLI (scripts/evaluate_f1_against_manual_gold.py --case-map ...
                        --output ...). Its own gold loader reads output/bend_label_NN/
                        bend_instance_labels.json (manual_bends) for the 9 corpus cases it knows;
                        the 2 shaked_xv cases (catalog-only gold) use the same match_lines on the
                        catalog rows. The union predictions are rebuilt from the historical bytes.
  replay-typed-union    Diagnostic: solve one existing historical candidate pool with today's typed-union
                        code (one heavy step, engine lock, RSS gate, 600 s cap) and compare with A0
                        case directories (--compare NAME=DIR; used for C001, whose retained A0 was
                        'reused' from an earlier smoke run on a probe pool).
  fresh-crosscheck      Re-score the fresh lane files with score.py gold11 and the frozen evaluator CLI.
  runtime-determinism   Compare two independent runtime-lane passes (status and line endpoints).
  summarize             Build gold11_repro.json: (a) historical rescoring, (b) fresh best_f1
                        (union + components), (c) active runtime lane; per-scan TP/FP/FN,
                        matched-line axis-angle MAE, nominal-relative bend-angle error where the
                        lane measures an angle, runtime, peak RSS, and a per-case
                        fresh-vs-historical reproduction diff with stage-level diagnosis.

Example
  .venv/bin/python -B research/broad_baseline/gold11_repro.py evaluator-crosscheck
  .venv/bin/python -B research/broad_baseline/gold11_repro.py summarize
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold11_cases as g11  # noqa: E402
import score as sc  # noqa: E402
import to_eval_format as tef  # noqa: E402

REPRO_DIR = bb.DEFAULT_RUNLOG / "gold11_repro"
RUNS_ROOT = REPRO_DIR / "runs"
HIST_DIR = REPRO_DIR / "historical"
DEFAULT_WORK_ROOT = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/"
                         "gold11_repro_work")
DEFAULT_OUT = bb.DEFAULT_RUNLOG / "gold11_repro.json"
CASE_LIST = bb.DEFAULT_RUNLOG / "case_list_consumed_v1.json"
FRAME_RECORDS = bb.DEFAULT_RUNLOG / "gold11_frame_recovery.json"
HIST_A0_ROOT = bb.ROOT / "output/research/f1-gold27-20260803/structured_union_exact_v1/cases"
HIST_POOL_ROOT = bb.ROOT / "output/research/f1-gold27-20260803/candidate_pool_bounded_v2"
HIST_CAD_SUMMARY = bb.ROOT / "output/research/f1-cad-guided-gold13-e2e-20260822/summary.json"
HIST_FUSION_DIAGNOSTIC = bb.ROOT / "output/research/f1-cad-guided-gold13-e2e-20260822/FUSION_GOLD11_DIAGNOSTIC.json"
# Historical strict-blind decompositions the bounded pool was built from (DEFAULT_DECOMPOSITIONS of
# scripts/build_f1_bounded_pool_from_freeze.py), per Gold11 catalog case.
HIST_DECOMP_DIRS = {
    "corpus": bb.ROOT / "output/enterprise_benchmark/blind-ply-bounded-20260804-v2/isolated_cases",
    "shaked_xv": bb.ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/baseline/isolated_cases",
}
# Historical prediction bytes are read from their original output/research locations (sha256-checked
# against the frozen identity table); the byte-identical copies inside the closed cached-control
# attempt directory are not needed.
PRED_SOURCE = "original"
G2_GATE = {"line_f1_min": 0.80, "precision_min": 0.90,
           "source": "docs/production/PRODUCTION_LINE_PLAN.md G2(a)"}

FRESH_VARIANTS = {
    "best_f1": ["best_f1", "a0", "cad_gated", "cad_raw_ungated", "f1_zero_shot_decomposition"],
    "runtime": ["runtime", "runtime_with_placeholders", "runtime_gated"],
}


# ------------------------------------------------------------------ helpers

def _gold11_entries() -> list[dict[str, Any]]:
    cases = [c for c in bb.read_json(CASE_LIST)["cases"] if c.get("tier") == "GOLD11"]
    if sorted(c["gold11_id"] for c in cases) != sorted(g11.CASES):
        raise ValueError("case list GOLD11 entries do not equal the frozen Gold11 identity table")
    for c in cases:
        if c["scan_sha256"] != g11.scan_sha(c["gold11_id"]):
            raise ValueError(f"{c['gold11_id']}: case-list scan sha differs from the frozen table")
    return sorted(cases, key=lambda c: c["gold11_id"])


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def _gib(value: float | int | None) -> float | None:
    return None if value is None else round(float(value) / 2**30, 3)


def _match_stats(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    angles = [float(m["axis_angle_deg"]) for m in matches]
    offsets = [float(m["transverse_offset_mm"]) for m in matches]
    return {"matched_pairs": len(matches),
            "line_axis_angle_mae_deg": _mean(angles),
            "line_axis_angle_max_deg": round(max(angles), 6) if angles else None,
            "transverse_offset_mean_mm": _mean(offsets),
            "transverse_offset_max_mm": round(max(offsets), 6) if offsets else None}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bend_angle_errors(matches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """|measured - CAD nominal| per bend id for rows that carry both (runtime / CAD-half rows)."""
    out = {}
    for row in matches or []:
        measured, target = _finite(row.get("measured_angle")), _finite(row.get("target_angle"))
        if measured is not None and target is not None:
            out[str(row.get("bend_id"))] = abs(measured - target)
    return out


def _angle_block(errors: Mapping[str, float], tp_ids: set[str] | None = None) -> dict[str, Any]:
    values = list(errors.values())
    block = {"bends_with_measured_angle": len(values), "bend_angle_mae_vs_cad_nominal_deg": _mean(values),
             "bend_angle_max_abs_dev_deg": round(max(values), 6) if values else None}
    if tp_ids is not None:
        tp_values = [errors[i] for i in tp_ids if i in errors]
        block.update({"tp_lines_with_measured_angle": len(tp_values),
                      "tp_bend_angle_mae_vs_cad_nominal_deg": _mean(tp_values)})
    return block


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(r["true_positive"]) for r in rows)
    fp = sum(int(r["false_positive"]) for r in rows)
    fn = sum(int(r["false_negative"]) for r in rows)
    block = sc.prf(tp, fp, fn)
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        families[str(r["family_id"])].append(r)
    recall = {f: sc._ratio(sum(int(x["true_positive"]) for x in rs), sum(int(x["gold_count"]) for x in rs))
              for f, rs in sorted(families.items())}
    matches = [m for r in rows for m in r.get("matches") or []]
    block.update({
        "case_count": len(rows),
        "exact_part_count": sum(bool(r.get("exact")) for r in rows),
        "family_recall": recall,
        "minimum_family_recall": min(recall.values()) if recall else None,
        **_match_stats(matches),
        "g2a_gate": {"line_f1": block["f1"] >= G2_GATE["line_f1_min"],
                     "precision": block["precision"] >= G2_GATE["precision_min"],
                     "pass": block["f1"] >= G2_GATE["line_f1_min"] and block["precision"] >= G2_GATE["precision_min"]},
    })
    return block


def _label_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gold11 mixes 9 human label sets with 2 Codex multiview label sets (shaked_xv C05/C06)."""
    out = {}
    for name, pick in (("human_labels_9_cases", lambda c: not g11.catalog_case(c).startswith("shaked_xv")),
                       ("codex_multiview_labels_2_cases", lambda c: g11.catalog_case(c).startswith("shaked_xv"))):
        subset = [r for r in rows if pick(r["case_id"])]
        if subset:
            a = _aggregate(subset)
            out[name] = {k: a[k] for k in ("case_count", "true_positive", "false_positive", "false_negative",
                                           "precision", "recall", "f1")}
    out["note"] = "diagnostic split; the historical Gold11 headline is the 11-case micro aggregate"
    return out


def _offset_only_magnitudes(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    offsets = sorted(float(d["transverse_offset_mm"]) for d in items if d.get("failed") == ["offset>4mm"])
    if not offsets:
        return {"count": 0}
    return {"count": len(offsets), "median_mm": round(statistics.median(offsets), 3),
            "min_mm": round(offsets[0], 3), "max_mm": round(offsets[-1], 3),
            "within_6mm": sum(o <= 6.0 for o in offsets), "within_8mm": sum(o <= 8.0 for o in offsets),
            "note": "lines whose nearest counterpart passes angle and both overlaps and fails ONLY the 4 mm offset"}


def _score(case_id: str, predictions: Mapping[str, Any], *, frame_records: Mapping[str, Any] | None,
           frame_mode: str = "recovered") -> dict[str, Any]:
    gold = {"kind": "gold11_catalog", "path": str(bb.GOLD_CATALOG), "scan_sha256": g11.scan_sha(case_id)}
    row = sc.score_case({"case_id": case_id, "tier": "GOLD11", "family_id": g11.family(case_id), "gold": gold,
                         "label_complete": True}, frame_records=frame_records,
                        preloaded_predictions=predictions, frame_mode=frame_mode)
    if row.get("disposition") != "SCORED_COMPLETE_LABEL":
        raise ValueError(f"{case_id}: not scored ({row.get('disposition')}: {row.get('reason')})")
    return row


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative",
            "precision", "recall", "f1", "exact", "unmatched_gold_ids", "unmatched_prediction_ids")
    out = {k: row.get(k) for k in keys}
    out.update(_match_stats(row.get("matches") or []))
    out["matches"] = [{k: m[k] for k in ("gold_id", "prediction_id", "axis_angle_deg", "transverse_offset_mm",
                                         "gold_span_overlap", "prediction_span_overlap")}
                      for m in row.get("matches") or []]
    return out


def _lines_of(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for row in payload.get("owned_bend_regions") or payload.get("predictions") or []:
        out[str(row["bend_id"])] = np.asarray(bb.endpoints(row), dtype=np.float64)
    return out


def _line_set_diff(fresh: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compare two line sets by bend id; endpoint deviation is orientation-free."""
    common = sorted(set(fresh) & set(hist))
    deviations = {}
    for key in common:
        a, b = fresh[key], hist[key]
        deviations[key] = float(min(np.abs(a - b).max(), np.abs(a[::-1] - b).max()))
    result = {"fresh_count": len(fresh), "historical_count": len(hist), "common_ids": len(common),
              "only_fresh": sorted(set(fresh) - set(hist)), "only_historical": sorted(set(hist) - set(fresh)),
              "max_endpoint_abs_diff_mm": round(max(deviations.values()), 9) if deviations else None}
    result["identical"] = (not result["only_fresh"] and not result["only_historical"]
                           and (not deviations or max(deviations.values()) <= 1e-9))
    if not result["identical"] and (result["only_fresh"] or result["only_historical"]):
        # Ids are content hashes for A0 (F1C_<hash>): pair the leftovers geometrically.
        left_f = [fresh[k] for k in result["only_fresh"]]
        left_h = [hist[k] for k in result["only_historical"]]
        pairs = []
        for i, a in enumerate(left_f):
            best = None
            for j, b in enumerate(left_h):
                d = float(min(np.abs(a - b).max(), np.abs(a[::-1] - b).max()))
                if best is None or d < best[0]:
                    best = (d, j)
            if best is not None:
                pairs.append({"fresh": result["only_fresh"][i], "nearest_historical": result["only_historical"][best[1]],
                              "max_endpoint_abs_diff_mm": round(best[0], 6)})
        result["unpaired_nearest"] = pairs
    return result


# ------------------------------------------------------- evaluator crosscheck

def cmd_evaluator_crosscheck(args: argparse.Namespace) -> int:
    from scripts.evaluate_f1_against_manual_gold import CASE_JOBS, Thresholds, match_lines

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "union_predictions_from_historical_bytes"
    pred_dir.mkdir(parents=True, exist_ok=True)
    catalog = bb.read_json(bb.GOLD_CATALOG)
    catalog_by_sha = {row["scan_sha256"]: row for row in catalog["cases"]}
    case_map: dict[str, str] = {}
    shaked_rows = []
    gold_agreement = []
    for case_id in sorted(g11.CASES):
        a0, cad, a0_path, cad_path = sc._retained_payloads(case_id, PRED_SOURCE)
        a0_eval = tef.convert_a0(a0, source=a0_path, scan=None)
        cad_eval = tef.convert_cad_semantic(cad, source=cad_path, scan=None)
        for evaluation in (a0_eval, cad_eval):
            evaluation["scan"]["sha256"] = g11.scan_sha(case_id)
        union = tef.fuse(a0_eval, cad_eval, mode="union")
        path = bb.write_json(pred_dir / f"{case_id}.json", union)
        catalog_case = g11.catalog_case(case_id)
        kind, name = catalog_case.split("/", 1)
        row = catalog_by_sha[g11.scan_sha(case_id)]
        if kind == "corpus" and name in CASE_JOBS:
            case_map[name] = str(path)
            labels = bb.read_json(bb.ROOT / "output" / CASE_JOBS[name] / "bend_instance_labels.json")["manual_bends"]
            cat = {str(b["bend_id"]): np.asarray(bb.endpoints(b)) for b in row["bends"]}
            lab = {str(b["bend_id"]): np.asarray(bb.endpoints(b)) for b in labels}
            diff = _line_set_diff(lab, cat)
            gold_agreement.append({"case_id": case_id, "evaluator_case": name, "label_file_bends": len(lab),
                                   "catalog_bends": len(cat), "identical": diff["identical"],
                                   "max_endpoint_abs_diff_mm": diff["max_endpoint_abs_diff_mm"]})
        else:
            result = match_lines(row["bends"], union["owned_bend_regions"], Thresholds())
            shaked_rows.append({"case_id": case_id, "catalog_case": catalog_case,
                                **{k: result[k] for k in ("gold_count", "prediction_count", "true_positive",
                                                          "false_positive", "false_negative")}})
    map_path = bb.write_json(out_dir / "evaluator_cli_case_map.json", case_map)
    report_path = out_dir / "evaluator_cli_union_9_corpus_cases.json"
    cmd = [str(bb.APP_PYTHON), "-B", str(bb.ROOT / "scripts" / "evaluate_f1_against_manual_gold.py"),
           "--case-map", str(map_path), "--output", str(report_path)]
    done = subprocess.run(cmd, cwd=str(bb.ROOT), capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(done.stderr[-2000:])
    cli = bb.read_json(report_path)["metrics"]
    tp = cli["true_positive"] + sum(r["true_positive"] for r in shaked_rows)
    fp = cli["false_positive"] + sum(r["false_positive"] for r in shaked_rows)
    fn = cli["false_negative"] + sum(r["false_negative"] for r in shaked_rows)
    total = sc.prf(tp, fp, fn)
    expected = g11.HISTORICAL["union"]
    summary = {
        "schema": "sherman.broad_baseline.gold11_evaluator_crosscheck.v1",
        "created_at": bb.utc_now(),
        "method": ("frozen evaluator CLI (own gold loader: output/bend_label_NN/bend_instance_labels.json) on the 9 "
                   "corpus cases + identical match_lines on the catalog gold for shaked_xv C05/C06; union rebuilt "
                   "from the original historical A0/CAD bytes (sha256-checked) with the frozen fusion rule"),
        "command": cmd,
        "evaluator_cli_metrics_9_corpus": cli,
        "shaked_xv_catalog_rows": shaked_rows,
        "label_file_vs_catalog_gold": gold_agreement,
        "total_11": total,
        "historical_expected": expected,
        "matches_historical": (tp, fp, fn) == (expected["tp"], expected["fp"], expected["fn"])
        and abs(total["f1"] - expected["f1"]) < 1e-12,
    }
    bb.write_json(out_dir / "evaluator_crosscheck.json", summary)
    print(f"evaluator CLI (9 corpus): TP={cli['true_positive']} FP={cli['false_positive']} FN={cli['false_negative']}"
          f" | shaked C05/C06: TP={sum(r['true_positive'] for r in shaked_rows)}"
          f" FP={sum(r['false_positive'] for r in shaked_rows)} FN={sum(r['false_negative'] for r in shaked_rows)}"
          f" | total TP={tp} FP={fp} FN={fn} F1={total['f1']:.10f} match_historical={summary['matches_historical']}"
          f" | label-file gold == catalog gold on {sum(g['identical'] for g in gold_agreement)}/{len(gold_agreement)}")
    return 0 if summary["matches_historical"] else 3


# --------------------------------------------------------------- summarize

def _historical_block(frame_records: Mapping[str, Any] | None) -> dict[str, Any]:
    """Re-score the retained bytes per case (per-scan rows incl. match geometry) and attach the
    three historical reproduction reports written by score.py / the original fusion CLI."""
    per_variant: dict[str, list[dict[str, Any]]] = {k: [] for k in ("union", "scan_only_a0", "cad_only_gated")}
    for case_id in sorted(g11.CASES):
        a0, cad, a0_path, cad_path = sc._retained_payloads(case_id, PRED_SOURCE)
        a0_eval = tef.convert_a0(a0, source=a0_path, scan=None)
        cad_eval = tef.convert_cad_semantic(cad, source=cad_path, scan=None)
        for evaluation in (a0_eval, cad_eval):
            evaluation["scan"]["sha256"] = g11.scan_sha(case_id)
        lanes = {"union": tef.fuse(a0_eval, cad_eval, mode="union"), "scan_only_a0": a0_eval,
                 "cad_only_gated": cad_eval}
        gold_rows = _gold_lines(case_id, frame_records)
        for name, payload in lanes.items():
            row = _score(case_id, payload, frame_records=frame_records)
            per_variant[name].append({"case_id": case_id, "family_id": g11.family(case_id), **row,
                                      "diagnosis": _diagnose(gold_rows, payload["owned_bend_regions"], row)})
    reports = {}
    for name in ("score_original_direct.json", "score_original_frame_recovered.json", "score_retained_direct.json",
                 "score_retained_frame_recovered.json",
                 "fusion_gold11_diagnostic_rerun_original_cli.json", "evaluator_crosscheck.json"):
        path = HIST_DIR / name
        if not path.is_file():
            reports[name] = {"status": "MISSING"}
            continue
        payload = bb.read_json(path)
        if name.startswith("score_"):
            union = payload["variants"]["union"]
            reports[name] = {"path": str(path), "frame_mode": payload.get("frame_mode"),
                             "prediction_source": payload.get("prediction_source"),
                             "catalog_sha256": payload["catalog"]["sha256"],
                             "union": {k: union["complete_labels"][k] for k in
                                       ("prediction_count", "true_positive", "false_positive", "false_negative",
                                        "precision", "recall", "f1", "minimum_family_recall")},
                             "all_variants_match_historical": all(v["matches_historical"]
                                                                  for v in payload["variants"].values())}
        elif name.startswith("fusion_"):
            hist = bb.read_json(HIST_FUSION_DIAGNOSTIC)
            reports[name] = {"path": str(path), "union_f1": payload["metrics"]["union"]["f1"],
                             "metrics_identical_to_retained_2026_08_22_diagnostic":
                                 payload["metrics"] == hist["metrics"],
                             "cases_identical_to_retained_2026_08_22_diagnostic":
                                 sorted(payload["cases"], key=lambda c: c["case_id"])
                                 == sorted(hist["cases"], key=lambda c: c["case_id"])}
        else:
            reports[name] = {"path": str(path), "total_11": payload["total_11"],
                             "matches_historical": payload["matches_historical"],
                             "label_file_gold_identical_to_catalog":
                                 f"{sum(g['identical'] for g in payload['label_file_vs_catalog_gold'])}/"
                                 f"{len(payload['label_file_vs_catalog_gold'])}"}
    union_rows = per_variant["union"]
    agg = _aggregate(union_rows)
    expected = g11.HISTORICAL["union"]
    return {
        "question": "(a) re-score the historical prediction artifacts with the evaluator; confirm 0.4580152672",
        "f1": agg["f1"],
        "reproduced_exactly": (agg["true_positive"], agg["false_positive"], agg["false_negative"]) ==
                              (expected["tp"], expected["fp"], expected["fn"]) and abs(agg["f1"] - expected["f1"]) < 1e-12,
        "aggregate": agg,
        "components": {name: _aggregate(rows) for name, rows in per_variant.items() if name != "union"},
        "reproduction_reports": reports,
        "label_source_split": _label_split(union_rows),
        "unmatched_failure_modes": {
            "false_positive_lines": _signature_counts([d for r in union_rows for d in r["diagnosis"]["false_positive_lines"]]),
            "false_negative_lines": _signature_counts([d for r in union_rows for d in r["diagnosis"]["false_negative_lines"]]),
            "offset_only_false_negatives": _offset_only_magnitudes(
                [d for r in union_rows for d in r["diagnosis"]["false_negative_lines"]])},
        "per_scan": [{"case_id": r["case_id"], "family_id": r["family_id"], **_compact(r), "diagnosis": r["diagnosis"]}
                     for r in union_rows],
        "per_scan_components": {name: [{"case_id": r["case_id"], **{k: r[k] for k in (
            "prediction_count", "true_positive", "false_positive", "false_negative")}} for r in rows]
            for name, rows in per_variant.items() if name != "union"},
        "inputs": {"a0": g11.A0_ROOT + "/<token>/prediction_owned_regions.json (sha256 == frozen table)",
                   "cad": g11.CAD_ROOT + "/<CG>/semantic_result.json (sha256 == frozen table)",
                   "gold": str(bb.GOLD_CATALOG), "gold_sha256": bb.sha256_file(bb.GOLD_CATALOG)},
    }


def _historical_resources() -> dict[str, dict[str, Any]]:
    """Historical timing/RSS as recorded (12-way parallel A0 batch, serial-ish CAD batch)."""
    out: dict[str, dict[str, Any]] = {}
    tokens = {g11.token(case_id) for case_id in g11.CASES}
    batch = bb.read_json(HIST_A0_ROOT.parent / "batch_summary.json")
    batch_rows = [row for row in batch["cases"] if row["case_token"] in tokens]  # Gold11 tokens only
    rss = {row["case_token"]: row.get("solver_child_peak_rss_bytes") for row in batch_rows}
    cad_summary = bb.read_json(HIST_CAD_SUMMARY)
    gold11_scans = {Path(e["scan_path"]).name for e in _gold11_entries()}
    cad_time = {}
    for row in cad_summary["results"]:
        if Path(row["scan_path"]).name not in gold11_scans:
            continue
        cad_time[(Path(row["scan_path"]).name, Path(row["cad_path"]).name)] = (row.get("timing") or {}).get("bend_report_sec")
    for case_id in sorted(g11.CASES):
        token = g11.token(case_id)
        case_dir = HIST_A0_ROOT / token
        graph, solution = case_dir / "typed_union_graph_00.json", case_dir / "cpsat_solution_00.json"
        solve_s = None
        if graph.is_file() and solution.is_file():
            solve_s = int(solution.stat().st_mtime - graph.stat().st_mtime)
        reused = next((bool(r.get("reused")) for r in batch_rows if r["case_token"] == token), None)
        out[case_id] = {"a0_typed_union_solve_s_from_mtimes": None if reused else solve_s,
                        "a0_solver_child_peak_rss_gib": _gib(rss.get(token)),
                        "a0_batch_workers": batch.get("workers"),
                        "a0_reused_flag": reused,
                        "note": ("historical batch reused an earlier solution for this token; file mtimes do not "
                                 "time the solve") if reused else "solve time = cpsat_solution_00 mtime - "
                                                                  "typed_union_graph_00 mtime (parallel batch)"}
    return out | {"_cad_time": cad_time}  # type: ignore[operator]


def _run_record(case_entry: Mapping[str, Any], lane: str) -> dict[str, Any] | None:
    path = RUNS_ROOT / case_entry["case_id"] / lane / "run_record.json"
    return bb.read_json(path) if path.is_file() else None


def _step_table(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in record.get("steps") or []:
        if "command" not in step:
            continue
        rows.append({"name": step["name"], "status": step.get("status"), "wall_s": step.get("wall_s"),
                     "timed_out": step.get("timed_out"), "timeout_s": step.get("timeout_s"),
                     "time_max_rss_gib": _gib(step.get("time_max_rss_bytes")),
                     "sampled_group_peak_rss_gib": _gib(step.get("sampled_peak_group_rss_bytes")),
                     "python_rss_gib_at_start": step.get("python_rss_gib_at_start")})
    return rows


def _gold_lines(case_id: str, frame_records: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    gold = sc.load_gold({"kind": "gold11_catalog", "path": str(bb.GOLD_CATALOG), "scan_sha256": g11.scan_sha(case_id)},
                        frame_records=frame_records)
    return gold["lines"]


def _criteria_failed(metrics: Mapping[str, float]) -> list[str]:
    failed = []
    if metrics["axis_angle_deg"] > bb.MATCH_ANGLE_DEG:
        failed.append("angle>5deg")
    if metrics["transverse_offset_mm"] > bb.MATCH_OFFSET_MM:
        failed.append("offset>4mm")
    if metrics["gold_span_overlap"] < bb.MATCH_GOLD_OVERLAP:
        failed.append("gold_overlap<0.6")
    if metrics["prediction_span_overlap"] < bb.MATCH_PRED_OVERLAP:
        failed.append("pred_overlap<0.6")
    return failed or ["passes_all_but_lost_one_to_one_assignment"]


def _diagnose(gold_rows: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]],
              scored: Mapping[str, Any]) -> dict[str, Any]:
    """Why each unmatched line is unmatched: nearest counterpart (matcher cost without thresholds)
    and which frozen criteria that nearest pair fails."""
    from scripts.evaluate_f1_against_manual_gold import pair_metrics

    gold = {str(g["bend_id"]): np.asarray(bb.endpoints(g), dtype=np.float64) for g in gold_rows}
    pred = {str(p["bend_id"]): np.asarray(bb.endpoints(p), dtype=np.float64) for p in pred_rows}
    matched_g = {m["gold_id"] for m in scored.get("matches") or []}
    matched_p = {m["prediction_id"] for m in scored.get("matches") or []}

    def cost(m):
        return (m["axis_angle_deg"] / bb.MATCH_ANGLE_DEG + m["transverse_offset_mm"] / bb.MATCH_OFFSET_MM
                + (1.0 - min(m["gold_span_overlap"], 1.0)) + (1.0 - min(m["prediction_span_overlap"], 1.0)))

    def nearest(line_id, *, is_gold):
        best = None
        others = pred if is_gold else gold
        for other_id, other in others.items():
            m = pair_metrics(gold[line_id], other) if is_gold else pair_metrics(other, pred[line_id])
            if best is None or cost(m) < best[0]:
                best = (cost(m), other_id, m)
        if best is None:
            return {"id": line_id, "nearest": None, "failed": ["no_counterpart_lines"]}
        m = best[2]
        return {"id": line_id, "nearest": best[1], "failed": _criteria_failed(m),
                **{k: round(float(m[k]), 4) for k in ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap",
                                                      "prediction_span_overlap", "gold_length_mm",
                                                      "prediction_length_mm")}}

    fps = [nearest(pid, is_gold=False) for pid in pred if pid not in matched_p]
    fns = [nearest(gid, is_gold=True) for gid in gold if gid not in matched_g]
    return {"false_positive_lines": fps, "false_negative_lines": fns}


def _signature_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts["+".join(item["failed"])] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _fresh_lane_rows(entries: Sequence[Mapping[str, Any]], lane: str, frame_records: Mapping[str, Any] | None,
                     work_root: Path) -> dict[str, Any]:
    variants: dict[str, list[dict[str, Any]]] = {v: [] for v in FRESH_VARIANTS[lane]}
    missing = []
    for entry in entries:
        case_id = entry["gold11_id"]
        record = _run_record(entry, lane)
        lane_dir = RUNS_ROOT / entry["case_id"] / lane
        if record is None:
            missing.append(case_id)
            continue
        gold_rows = _gold_lines(case_id, frame_records)
        work = Path(record.get("work_dir") or work_root / entry["case_id"] / lane)
        angle_rows: list[Mapping[str, Any]] = []
        if lane == "runtime":
            report = work / "runtime" / "bend_inspection_report.json"
            if report.is_file():
                angle_rows = bb.read_json(report).get("matches") or []
        else:
            cad_case = work / "cad" / "cad_case.json"
            if cad_case.is_file():
                angle_rows = bb.read_json(cad_case).get("matches") or []
        errors = _bend_angle_errors(angle_rows)
        primary = lane_dir / f"predictions_{lane}.json"
        for variant in FRESH_VARIANTS[lane]:
            path = lane_dir / f"predictions_{variant}.json"
            fallback = None
            if not path.is_file():
                # A failed lane writes only its primary/failure file: the variant scores as that failure
                # (every gold line a miss), so all variants stay over the same 11 scans.
                if primary.is_file() and bb.load_eval_lines(primary).get("status") == "LANE_FAILED":
                    path, fallback = primary, "LANE_FAILED_PRIMARY_FILE"
                else:
                    continue
            payload = bb.load_eval_lines(path)
            row = _score(case_id, payload, frame_records=frame_records)
            tp_ids = {str(m["prediction_id"]).removeprefix("CADF_") for m in row.get("matches") or []}
            pred_rows = [r for r in payload.get("owned_bend_regions") or []
                         if bb.valid_line(r.get("line_start"), r.get("line_end")) or r.get("span_endpoints")]
            with_angle = variant.startswith("runtime") or variant in {"best_f1", "cad_gated", "cad_raw_ungated"}
            variants[variant].append({
                "case_id": case_id, "sc_id": entry["case_id"], "family_id": entry["family_id"],
                "stage_hints": entry.get("stage_hints"), "lane_status": payload.get("status"),
                "lane_failure": payload.get("failure"), "variant_file_fallback": fallback, **row,
                "angle": _angle_block(errors, tp_ids) if with_angle else None,
                "angle_errors": errors if with_angle else {},
                "diagnosis": _diagnose(gold_rows, pred_rows, row),
                "runtime": {"lane_status": record.get("status"), "lane_wall_s": record.get("lane_wall_s"),
                            "heavy_step_wall_s": record.get("heavy_step_wall_s"),
                            "peak_rss_gib": _gib(record.get("peak_rss_bytes")), "steps": _step_table(record)},
            })
    out: dict[str, Any] = {}
    for variant, rows in variants.items():
        if not rows:
            continue
        agg = _aggregate(rows)
        pooled = [v for r in rows for v in (r.get("angle_errors") or {}).values()]
        tp_pooled = [r["angle_errors"][i] for r in rows for i in
                     {str(m["prediction_id"]).removeprefix("CADF_") for m in r.get("matches") or []}
                     if i in (r.get("angle_errors") or {})]
        walls = [r["runtime"]["lane_wall_s"] for r in rows if r["runtime"]["lane_wall_s"] is not None]
        compute = [r["runtime"]["heavy_step_wall_s"] for r in rows if r["runtime"]["heavy_step_wall_s"] is not None]
        peaks = [r["runtime"]["peak_rss_gib"] for r in rows if r["runtime"]["peak_rss_gib"] is not None]
        out[variant] = {
            "aggregate": agg,
            "complete": len(rows) == len(entries),
            "lane_status_counts": dict(sorted(_count(r["lane_status"] for r in rows).items())),
            "bend_angle_vs_cad_nominal": None if not any(r.get("angle") for r in rows) else {
                "bends_with_measured_angle": len(pooled), "mae_deg": _mean(pooled),
                "max_abs_dev_deg": round(max(pooled), 6) if pooled else None,
                "tp_lines_with_measured_angle": len(tp_pooled), "tp_mae_deg": _mean(tp_pooled),
                "note": "nominal-relative (CAD target), not ground truth; staged/partial/defective scans included"},
            "label_source_split": _label_split(rows),
            "unmatched_failure_modes": {
                "false_positive_lines": _signature_counts([d for r in rows for d in r["diagnosis"]["false_positive_lines"]]),
                "false_negative_lines": _signature_counts([d for r in rows for d in r["diagnosis"]["false_negative_lines"]]),
                "offset_only_false_negatives": _offset_only_magnitudes(
                    [d for r in rows for d in r["diagnosis"]["false_negative_lines"]]),
                "note": "nearest counterpart by matcher cost; criteria that nearest pair fails"},
            "resources": {"compute_s_total": round(sum(compute), 3) if compute else None,
                          "compute_s_median": round(statistics.median(compute), 3) if compute else None,
                          "compute_s_max": round(max(compute), 3) if compute else None,
                          "lane_wall_s_total": round(sum(walls), 3) if walls else None,
                          "lane_wall_s_median": round(statistics.median(walls), 3) if walls else None,
                          "lane_wall_s_max": round(max(walls), 3) if walls else None,
                          "peak_rss_gib_max": max(peaks) if peaks else None,
                          "note": "compute = sum of heavy-step wall times; lane wall also includes lock/RSS-gate "
                                  "waits caused by other jobs' memory"},
            "per_scan": [{"case_id": r["case_id"], "sc_id": r["sc_id"], "family_id": r["family_id"],
                          "stage_hints": r["stage_hints"], "lane_status": r["lane_status"],
                          "lane_failure": r["lane_failure"], "variant_file_fallback": r["variant_file_fallback"],
                          **_compact(r), "angle": r["angle"], "diagnosis": r["diagnosis"],
                          "runtime": r["runtime"]} for r in rows],
        }
    out["_missing_runs"] = missing
    return out


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return counts


def _reproduction_diff(entries: Sequence[Mapping[str, Any]], hist: Mapping[str, Any], work_root: Path,
                       fresh_best: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per case: did the fresh best_f1 run reproduce the retained historical bytes, stage by stage?"""
    hist_union = {r["case_id"]: r for r in hist["per_scan"]}
    fresh_union = {r["case_id"]: r for r in (fresh_best.get("best_f1") or {}).get("per_scan", [])}
    tokens = {g11.token(case_id) for case_id in g11.CASES}
    pool_manifest = {row["case_token"]: row for row in bb.read_json(HIST_POOL_ROOT / "manifest.json")["cases"]
                     if row["case_token"] in tokens}  # Gold11 tokens only
    rows = []
    for entry in entries:
        case_id, token = entry["gold11_id"], g11.token(entry["gold11_id"])
        record = _run_record(entry, "best_f1")
        if record is None:
            rows.append({"case_id": case_id, "status": "NOT_RUN"})
            continue
        work = Path(record["work_dir"])
        lane_dir = RUNS_ROOT / entry["case_id"] / "best_f1"
        item: dict[str, Any] = {"case_id": case_id, "token": token, "family_id": entry["family_id"],
                                "lane_status": record.get("status")}
        # A0 bytes
        fresh_a0 = work / "a0_union" / "prediction_owned_regions.json"
        hist_a0 = HIST_A0_ROOT / token / "prediction_owned_regions.json"
        a0: dict[str, Any] = {"fresh_exists": fresh_a0.is_file()}
        if fresh_a0.is_file():
            a0["fresh_sha256"] = bb.sha256_file(fresh_a0)
            a0["historical_sha256"] = g11.a0_sha(case_id)
            a0["byte_identical"] = a0["fresh_sha256"] == a0["historical_sha256"]
            a0["lines"] = _line_set_diff(_lines_of(bb.read_json(fresh_a0)), _lines_of(bb.read_json(hist_a0)))
            fresh_summary = work / "a0_union" / "inference_summary.json"
            hist_summary = HIST_A0_ROOT / token / "inference_summary.json"
            if fresh_summary.is_file():
                fs, hs = bb.read_json(fresh_summary), bb.read_json(hist_summary)
                a0["typed_union"] = {k: {"fresh": fs.get(k), "historical": hs.get(k)} for k in (
                    "typed_assignment_count", "counted_bend_component_count", "solver_status", "graph_sha256")}
        else:
            a0["failure"] = (bb.read_json(lane_dir / "predictions_a0.json").get("failure")
                             if (lane_dir / "predictions_a0.json").is_file() else None)
        pool_row = work / "pool" / f"BB{entry['scan_sha256'][:12]}.row.json"
        if pool_row.is_file():
            fr, hr = bb.read_json(pool_row), pool_manifest.get(token, {})
            a0["pool"] = {k: {"fresh": fr.get(k), "historical": hr.get(k)}
                          for k in ("candidate_count", "line_object_count", "working_point_count", "plane_point_count")}
            a0["pool"]["counts_identical"] = all(v["fresh"] == v["historical"] for v in a0["pool"].values())
            fresh_pool = work / "pool" / f"BB{entry['scan_sha256'][:12]}.json"
            hist_pool = HIST_POOL_ROOT / f"{token}.json"
            if fresh_pool.is_file() and hist_pool.is_file():
                a0["pool"]["candidate_content"] = _pool_candidate_diff(bb.read_json(fresh_pool), bb.read_json(hist_pool))
        decomposition = work / "a0" / f"BB{entry['scan_sha256'][:12]}" / "result" / "decomposition_result.json"
        hist_decomp = _historical_decomposition(case_id)
        if decomposition.is_file() and hist_decomp is not None:
            item["decomposition"] = _decomposition_diff(bb.read_json(decomposition), bb.read_json(hist_decomp),
                                                         hist_decomp)
        item["a0"] = a0
        # CAD half
        fresh_cad = lane_dir / "predictions_cad_gated.json"
        if fresh_cad.is_file():
            retained = bb.read_json(bb.ROOT / g11.CAD_ROOT / case_id / "semantic_result.json")
            fresh_payload = bb.read_json(fresh_cad)
            cad = {"status": fresh_payload.get("status"),
                   "gated_lines": _line_set_diff(_lines_of(fresh_payload), _lines_of(retained))}
            gate_audit = (fresh_payload.get("conversion") or {}).get("gate_audit") or []
            cad["gate"] = {"evaluated": len(gate_audit), "accepted": sum(bool(a["accepted"]) for a in gate_audit)}
            cad_case = work / "cad" / "cad_case.json"
            if cad_case.is_file():
                cad["raw_runtime_diff"] = _cad_raw_diff(bb.read_json(cad_case), entry)
            item["cad"] = cad
        # union per-case counts
        hu, fu = hist_union.get(case_id), fresh_union.get(case_id)
        if fu is not None and hu is not None:
            keys = ("prediction_count", "true_positive", "false_positive", "false_negative")
            item["union_counts"] = {"fresh": {k: fu[k] for k in keys}, "historical": {k: hu[k] for k in keys},
                                    "identical": all(fu[k] == hu[k] for k in keys)}
        rows.append(item)
    return rows


def _pool_candidate_diff(fresh: Mapping[str, Any], hist: Mapping[str, Any]) -> dict[str, Any]:
    def key(rows):
        return sorted(hashlib.sha256(_canon(r)).hexdigest() for r in rows or [])
    out = {}
    for field in ("candidates", "line_objects", "plane_table", "support_atom_table", "support_atom_edges"):
        fk, hk = key(fresh.get(field)), key(hist.get(field))
        out[field] = {"fresh": len(fk), "historical": len(hk), "identical_multiset": fk == hk}
    out["candidate_source_counts"] = {"fresh": fresh.get("candidate_source_counts"),
                                      "historical": hist.get("candidate_source_counts")}
    return out


def _canon(value: Any) -> bytes:
    def rounded(v):
        if isinstance(v, float):
            return round(v, 9)
        if isinstance(v, dict):
            return {k: rounded(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rounded(x) for x in v]
        return v
    return json.dumps(rounded(value), sort_keys=True, separators=(",", ":")).encode()


def _historical_decomposition(case_id: str) -> Path | None:
    kind, name = g11.catalog_case(case_id).split("/", 1)
    if kind == "corpus":
        root = HIST_DECOMP_DIRS["corpus"] / name
    else:
        root = HIST_DECOMP_DIRS["shaked_xv"] / {"C05": "05_C05", "C06": "06_C06"}[name]
    matches = sorted(root.glob("result/**/decomposition_result.json"))
    for path in matches:
        identity = ((bb.read_json(path).get("provenance") or {}).get("cache_identity") or {})
        if identity.get("scan_sha256") in (None, g11.scan_sha(case_id)):
            return path
    return None


def _decomposition_diff(fresh: Mapping[str, Any], hist: Mapping[str, Any], hist_path: Path) -> dict[str, Any]:
    def owned(payload):
        rows = {}
        for i, row in enumerate(payload.get("owned_bend_regions") or [], start=1):
            try:
                rows[str(row.get("bend_id") or f"D{i:02d}")] = np.asarray(bb.endpoints(row), dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
        return rows
    fp, hp = fresh.get("provenance") or {}, hist.get("provenance") or {}
    return {"historical_path": str(hist_path),
            "exact_bend_count": {"fresh": fresh.get("exact_bend_count"), "historical": hist.get("exact_bend_count")},
            "owned_bend_regions": _line_set_diff(owned(fresh), owned(hist)),
            "semantics_version": {"fresh": fp.get("semantics_version") or fp.get("decomposition_semantics_version"),
                                  "historical": hp.get("semantics_version") or hp.get("decomposition_semantics_version")}}


def _cad_raw_diff(fresh_case: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh blind CAD+PLY runtime matches vs the historical (2026-08-22) case payload, before the gate."""
    hist_dir = HIST_CAD_SUMMARY.parent / "cases"
    scan_name = Path(entry["scan_path"]).name.replace(" ", "_")
    candidates = sorted(hist_dir.glob(f"{entry['family_id']}--{scan_name}--*.json"))
    if len(candidates) != 1:
        return {"status": "HISTORICAL_CASE_NOT_FOUND", "candidates": [str(c) for c in candidates]}
    hist_case = bb.read_json(candidates[0])

    def rows(payload):
        out = {}
        for m in payload.get("matches") or []:
            out[str(m.get("bend_id"))] = {
                "status": m.get("status"), "measured_angle": _finite(m.get("measured_angle")),
                "line": None if m.get("detected_line_start") is None else
                np.asarray([m["detected_line_start"], m["detected_line_end"]], dtype=np.float64),
                "source": m.get("detected_line_geometry_source"), "assignment": m.get("assignment_source")}
        return out
    fr, hr = rows(fresh_case), rows(hist_case)
    per = []
    for key in sorted(set(fr) | set(hr)):
        a, b = fr.get(key), hr.get(key)
        d = None
        if a and b and a["line"] is not None and b["line"] is not None:
            d = float(min(np.abs(a["line"] - b["line"]).max(), np.abs(a["line"][::-1] - b["line"]).max()))
        per.append({"bend_id": key, "fresh_status": a and a["status"], "historical_status": b and b["status"],
                    "fresh_has_line": bool(a and a["line"] is not None),
                    "historical_has_line": bool(b and b["line"] is not None),
                    "fresh_assignment": a and a["assignment"], "historical_assignment": b and b["assignment"],
                    "measured_angle": {"fresh": a and a["measured_angle"], "historical": b and b["measured_angle"]},
                    "line_max_endpoint_abs_diff_mm": None if d is None else round(d, 6)})
    diffs = [p["line_max_endpoint_abs_diff_mm"] for p in per if p["line_max_endpoint_abs_diff_mm"] is not None]
    same_presence = all(p["fresh_has_line"] == p["historical_has_line"] for p in per)
    return {"historical_case": str(candidates[0]), "bends": len(per), "same_line_presence": same_presence,
            "max_line_endpoint_abs_diff_mm": round(max(diffs), 6) if diffs else None,
            "identical": same_presence and (not diffs or max(diffs) <= 1e-6), "per_bend": per,
            "fresh_alignment_fitness": (fresh_case.get("alignment_metadata") or {}).get("fitness"),
            "historical_alignment_fitness": (hist_case.get("alignment_metadata") or {}).get("fitness")}


def _runtime_vs_cad_half(entries: Sequence[Mapping[str, Any]], fresh_best: Mapping[str, Any],
                         fresh_runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Same pipeline function, two call paths: the active worker (shared registration, fitness >= 0.55
    required, lines reported in the CAD frame) vs the blind CAD half of best_f1 (evaluate_bend_case,
    no shared registration, raw-frame global detections, raw-scan support gate)."""
    rt_rows = {r["case_id"]: r for r in (fresh_runtime.get("runtime") or {}).get("per_scan", [])}
    cad_rows = {r["case_id"]: r for r in (fresh_best.get("cad_gated") or {}).get("per_scan", [])}
    out = []
    for entry in entries:
        case_id = entry["gold11_id"]
        row: dict[str, Any] = {"case_id": case_id, "stage_hints": entry.get("stage_hints")}
        rec = _run_record(entry, "runtime")
        if rec is not None:
            report = Path(rec["work_dir"]) / "runtime" / "bend_inspection_report.json"
            if report.is_file():
                payload = bb.read_json(report)
                matches = payload.get("matches") or []
                alignment = payload.get("alignment") or {}
                row["worker"] = {
                    "cad_bends": len(matches),
                    "scan_fit_lines": sum(1 for m in matches if m.get("detected_line_geometry_source") == "SCAN_FIT"
                                          and m.get("detected_line_start") is not None),
                    "cad_reference_placeholders": sum(1 for m in matches if m.get("detected_line_geometry_source")
                                                      == "CAD_REFERENCE_PLACEHOLDER"),
                    "not_detected": sum(1 for m in matches if m.get("detected_line_start") is None),
                    "alignment_fitness": alignment.get("fitness"), "alignment_rmse_mm": alignment.get("rmse_mm"),
                    "alignment_source": alignment.get("alignment_source")}
            else:
                lane = bb.read_json(RUNS_ROOT / entry["case_id"] / "runtime" / "predictions_runtime.json")
                row["worker"] = {"failure": (lane.get("failure") or {}).get("worker_error") or lane.get("status")}
        if case_id in rt_rows:
            row["worker_line_tp"] = rt_rows[case_id]["true_positive"]
        rec = _run_record(entry, "best_f1")
        if rec is not None:
            case_json = Path(rec["work_dir"]) / "cad" / "cad_case.json"
            if case_json.is_file():
                matches = bb.read_json(case_json).get("matches") or []
                gated = bb.read_json(RUNS_ROOT / entry["case_id"] / "best_f1" / "predictions_cad_gated.json")
                row["cad_half"] = {"cad_bends": len(matches),
                                   "lines_before_gate": sum(1 for m in matches if m.get("detected_line_start") is not None),
                                   "lines_after_gate": gated.get("line_count")}
        if case_id in cad_rows:
            row["cad_half_gated_line_tp"] = cad_rows[case_id]["true_positive"]
        out.append(row)
    return out


ENGINE_SOURCES = (
    "backend/bend_analysis_worker.py", "backend/bend_inspection_pipeline.py", "backend/bend_detector.py",
    "backend/cad_bend_extractor.py", "backend/qc_engine.py", "backend/patch_graph_latent_decomposition.py",
    "domains/bend/services/heatmap_evidence_service.py", "domains/bend/services/inspection_frame.py",
    "scripts/evaluate_bend_case.py", "scripts/build_f1_bounded_pool_from_freeze.py",
    "scripts/run_f1_structured_union_e2e.py", "scripts/solve_f1_structured_union_cpsat.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/diagnose_f1_cad_scan_fusion_gold11.py",
    "scripts/apply_f1_cad_guided_scan_frame_gate.py", "scripts/build_additional_line_label_review.py",
    "research/broad_baseline/run_lane.py", "research/broad_baseline/bb_steps.py",
    "research/broad_baseline/bb_common.py", "research/broad_baseline/to_eval_format.py",
    "research/broad_baseline/score.py",
)


def _engine_snapshot() -> dict[str, Any]:
    """sha256 + mtime of the engine/evaluator sources the lanes executed (working tree; the repo HEAD is
    far behind it), and whether any changed after the first fresh run started."""
    records = [_run_record(e, lane) for e in _gold11_entries() for lane in ("runtime", "best_f1")]
    first_start = None
    for rec in records:
        if rec and rec.get("finished_at"):
            started = _utc_epoch(rec["finished_at"]) - float(rec.get("lane_wall_s") or 0)
            first_start = started if first_start is None else min(first_start, started)
    files = {}
    for rel in ENGINE_SOURCES:
        path = bb.ROOT / rel
        if path.is_file():
            mtime = path.stat().st_mtime
            files[rel] = {"sha256": bb.sha256_file(path),
                          "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime)),
                          "modified_after_first_fresh_run_start": bool(first_start is not None and mtime > first_start)}
    return {"first_fresh_run_start_utc": None if first_start is None else
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(first_start)),
            "engine_or_evaluator_modified_during_runs": any(
                v["modified_after_first_fresh_run_start"] for k, v in files.items()
                if not k.startswith("research/")),
            "wrapper_files_modified_during_runs": sorted(
                k for k, v in files.items() if k.startswith("research/") and v["modified_after_first_fresh_run_start"]),
            "note": ("run_lane.py was fixed mid-batch (12:47 local, before case 9): lane budget excludes lock/RSS-gate "
                     "waits, stale prediction files are cleared, failures are written to every variant; engine "
                     "computation and outputs are unaffected"),
            "files": files}


def _utc_epoch(stamp: str) -> float:
    import calendar

    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))


def _per_scan_table(hist: Mapping[str, Any], fresh_best: Mapping[str, Any], fresh_runtime: Mapping[str, Any],
                    diff: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One compact row per Gold11 scan with the three lanes side by side."""
    def by_case(block):
        return {r["case_id"]: r for r in (block or {}).get("per_scan", [])}
    best, rt = by_case(fresh_best.get("best_f1")), by_case(fresh_runtime.get("runtime"))
    diffs = {d["case_id"]: d for d in diff}

    def core(r):
        return {"tp": r["true_positive"], "fp": r["false_positive"], "fn": r["false_negative"],
                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                "line_axis_angle_mae_deg": r["line_axis_angle_mae_deg"],
                "transverse_offset_mean_mm": r["transverse_offset_mean_mm"]}
    rows = []
    for h in hist["per_scan"]:
        case_id = h["case_id"]
        row: dict[str, Any] = {"case_id": case_id, "family_id": h["family_id"],
                               "catalog_case": g11.catalog_case(case_id), "gold_lines": h["gold_count"],
                               "a_historical": {**core(h), "predicted_lines": h["prediction_count"],
                                                "resources_as_recorded": h.get("historical_resources")}}
        if case_id in best:
            b = best[case_id]
            d = diffs.get(case_id) or {}
            row["b_fresh_best_f1"] = {
                **core(b), "predicted_lines": b["prediction_count"], "lane_status": b["lane_status"],
                "cad_half_bend_angle_mae_vs_cad_nominal_deg": (b.get("angle") or {}).get("bend_angle_mae_vs_cad_nominal_deg"),
                "wall_s": b["runtime"]["lane_wall_s"], "compute_s": b["runtime"]["heavy_step_wall_s"],
                "peak_rss_gib": b["runtime"]["peak_rss_gib"],
                "step_wall_s": {s["name"]: s["wall_s"] for s in b["runtime"]["steps"]},
                "union_counts_identical_to_history": (d.get("union_counts") or {}).get("identical"),
                "a0_bytes_identical_to_history": (d.get("a0") or {}).get("byte_identical"),
                "cad_gated_lines_identical_to_history": ((d.get("cad") or {}).get("gated_lines") or {}).get("identical")}
        if case_id in rt:
            c = rt[case_id]
            row["c_active_runtime"] = {
                **core(c), "predicted_lines": c["prediction_count"], "lane_status": c["lane_status"],
                "failure": (c.get("lane_failure") or {}).get("worker_error"),
                "bend_angle_mae_vs_cad_nominal_deg": (c.get("angle") or {}).get("bend_angle_mae_vs_cad_nominal_deg"),
                "bends_with_measured_angle": (c.get("angle") or {}).get("bends_with_measured_angle"),
                "wall_s": c["runtime"]["lane_wall_s"], "compute_s": c["runtime"]["heavy_step_wall_s"],
                "peak_rss_gib": c["runtime"]["peak_rss_gib"]}
        rows.append(row)
    return rows


def cmd_summarize(args: argparse.Namespace) -> int:
    frame_records = bb.read_json(FRAME_RECORDS)
    entries = _gold11_entries()
    work_root = Path(args.work_root)
    hist = _historical_block(frame_records)
    hist_resources = _historical_resources()
    cad_time = hist_resources.pop("_cad_time")
    for row in hist["per_scan"]:
        entry = next(e for e in entries if e["gold11_id"] == row["case_id"])
        res = dict(hist_resources[row["case_id"]])
        res["cad_bend_report_s"] = cad_time.get((Path(entry["scan_path"]).name,
                                                Path(g11.CAD_BY_FAMILY[entry["family_id"]]).name))
        row["historical_resources"] = res
    fresh_best = _fresh_lane_rows(entries, "best_f1", frame_records, work_root)
    fresh_runtime = _fresh_lane_rows(entries, "runtime", frame_records, work_root)
    diff = _reproduction_diff(entries, hist, work_root, fresh_best)

    def headline(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not block:
            return None
        a = block["aggregate"]
        return {k: a[k] for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative",
                                  "prediction_count", "gold_bend_count", "exact_part_count", "minimum_family_recall",
                                  "line_axis_angle_mae_deg", "transverse_offset_mean_mm", "g2a_gate")} | {
            "complete_11": block["complete"], "lane_status_counts": block["lane_status_counts"]}

    report = {
        "schema": "sherman.broad_baseline.gold11_repro.v1",
        "created_at": bb.utc_now(),
        "question": ("Owner 2026-10-17: 'We optimized the engine for those 11 (Gold11), they should come back green.' "
                     "Reproduce Gold11: (a) historical rescoring, (b) fresh best lane, (c) active runtime lane."),
        "gate": G2_GATE,
        "headline": {
            "a_historical_rescored_union": {k: hist["aggregate"][k] for k in (
                "f1", "precision", "recall", "true_positive", "false_positive", "false_negative", "prediction_count",
                "minimum_family_recall", "line_axis_angle_mae_deg", "g2a_gate")} | {
                "reproduced_exactly": hist["reproduced_exactly"]},
            "b_fresh_best_f1_union": headline(fresh_best.get("best_f1")),
            "b_fresh_components": {v: headline(fresh_best.get(v)) for v in FRESH_VARIANTS["best_f1"] if v != "best_f1"},
            "c_active_runtime": headline(fresh_runtime.get("runtime")),
            "c_runtime_diagnostics": {v: headline(fresh_runtime.get(v)) for v in FRESH_VARIANTS["runtime"]
                                      if v != "runtime"},
        },
        "per_scan_table": _per_scan_table(hist, fresh_best, fresh_runtime, diff),
        "a_historical": hist,
        "b_fresh_best_f1": fresh_best,
        "b_reproduction_diff": diff,
        "c_runtime": fresh_runtime,
        "c_runtime_vs_best_f1_cad_half": _runtime_vs_cad_half(entries, fresh_best, fresh_runtime),
        "matcher": {"source": "scripts/evaluate_f1_against_manual_gold.match_lines (5 deg / 4 mm / 0.60 / 0.60, "
                              "one-to-one Hungarian), micro-aggregated", "frame": "scan_raw",
                    "gold": str(bb.GOLD_CATALOG), "frame_records": str(FRAME_RECORDS)},
        "metric_definitions": {
            "line_axis_angle_mae_deg": "mean |angle| between matched predicted and gold line directions (TP pairs only)",
            "bend_angle_mae_vs_cad_nominal_deg": ("mean |measured bend angle - CAD nominal| over the lane's CAD "
                                                  "matches with a measured angle (runtime report / CAD-half case); "
                                                  "Gold11 has no angle truth; staged/partial/defective scans included"),
            "tp_bend_angle_mae_vs_cad_nominal_deg": "same, restricted to lines that matched a gold line",
            "peak_rss_gib": "max over steps of max(/usr/bin/time -l max RSS, sampled process-group RSS)",
        },
        "custody": ("Gold11 development cases only (11 scans, their STEP CAD, catalog gold / label files, retained "
                    "historical predictions and per-token intermediates of these 11). No FRESH, HOLDOUT10, sealed, "
                    "N0xx, EHO4, incoming_codex_fetch or freeze/sealed manifests read."),
        "paths": {"runs_root": str(RUNS_ROOT), "historical_dir": str(HIST_DIR), "work_root": str(work_root)},
        "engine_source_snapshot": _engine_snapshot(),
    }
    extra_paths = {"typed_union_replays": sorted(str(p) for p in REPRO_DIR.glob("replay_*.json")),
                   "runtime_rerun": str(REPRO_DIR / "runtime_determinism.json")}
    report["paths"].update(extra_paths)
    for replay in extra_paths["typed_union_replays"]:
        payload = bb.read_json(replay)
        report.setdefault("b_typed_union_replays", []).append(
            {"path": replay, "purpose": payload.get("purpose"), "pool": payload.get("pool"),
             "pool_sha256": payload.get("pool_sha256"),
             "status": (payload.get("step") or {}).get("status"), "wall_s": (payload.get("step") or {}).get("wall_s"),
             "peak_rss_gib": _gib(max((payload.get("step") or {}).get("time_max_rss_bytes") or 0,
                                      (payload.get("step") or {}).get("sampled_peak_group_rss_bytes") or 0)),
             "replay": payload.get("replay"), "comparisons": payload.get("comparisons")})
    probes = sorted((REPRO_DIR / "spatial_matching_probe").glob("CG*.json"))
    if probes:
        report["c_spatial_matching_probe"] = {
            "question": ("The worker passes a shared registration, which makes run_progressive_bend_inspection force "
                         "position_mode=enabled + strict_spatial_matching=True on the matcher; does that, or the "
                         "frame, drop the scan detections?"),
            "cases": {path.stem: {k: v for k, v in bb.read_json(path).items()
                                  if k in ("cad_bends", "scan_points", "detections", "match",
                                           "shared_frame_detections_vs_nearest_cad_bend", "seconds", "step")}
                      for path in probes}}
    crosscheck = REPRO_DIR / "fresh_crosscheck" / "fresh_crosscheck.json"
    if crosscheck.is_file():
        report["bc_independent_rescoring"] = bb.read_json(crosscheck)
    determinism = REPRO_DIR / "runtime_determinism.json"
    if determinism.is_file():
        report["c_runtime_determinism"] = bb.read_json(determinism)
    if args.findings:
        report["findings"] = bb.read_json(args.findings)
    report["artifact_manifest"] = {
        "root": str(REPRO_DIR),
        "files": {str(path.relative_to(REPRO_DIR)): bb.sha256_file(path)
                  for path in sorted(REPRO_DIR.rglob("*")) if path.is_file()},
        "note": "sha256 of every evidence file under the root at summary time"}
    bb.write_json(args.out, report)
    h = report["headline"]
    for name in ("a_historical_rescored_union", "b_fresh_best_f1_union", "c_active_runtime"):
        block = h[name]
        if block:
            print(f"{name:30s} F1={block['f1']:.10f} P={block['precision']:.4f} R={block['recall']:.4f} "
                  f"TP={block['true_positive']} FP={block['false_positive']} FN={block['false_negative']}"
                  + (f" complete={block.get('complete_11')}" if 'complete_11' in block else ""))
    print(f"missing best_f1 runs: {fresh_best['_missing_runs']} | missing runtime runs: {fresh_runtime['_missing_runs']}")
    print(f"-> {args.out}")
    return 0


def cmd_replay_typed_union(args: argparse.Namespace) -> int:
    """Diagnostic: solve the typed union of one EXISTING candidate pool with today's code (same step
    as run_lane's a0_typed_union) and compare graph/prediction hashes with one or more A0 case
    directories (--compare NAME=DIR, each holding inference_summary.json and
    prediction_owned_regions.json). One heavy step: engine flock, python-RSS gate, /usr/bin/time -l,
    hard timeout."""
    import run_lane

    pool = bb.expand(args.pool)
    out_dir = bb.expand(args.out_dir)
    compares = {}
    for item in args.compare:
        name, _, directory = item.partition("=")
        compares[name] = bb.expand(directory)
    with run_lane._engine_lock(run_lane.DEFAULT_LOCK) as lock_info:
        record = bb.run_monitored(
            [str(bb.APP_PYTHON), "-B", str(bb.ROOT / "scripts" / "run_f1_structured_union_e2e.py"),
             "--candidate-pool", str(pool), "--output-dir", str(out_dir / "a0_union"), "--disable-birth",
             "--solver-python", str(bb.SOLVER_PYTHON)],
            name="replay_typed_union", log_dir=out_dir / "logs", timeout_s=float(args.timeout),
            rss_limit_gib=float(args.rss_limit_gib))
    result: dict[str, Any] = {"schema": "sherman.broad_baseline.typed_union_replay.v1", "created_at": bb.utc_now(),
                              "purpose": args.purpose, "pool": str(pool), "pool_sha256": bb.sha256_file(pool),
                              "step": record, **lock_info, "comparisons": {}}
    pred = out_dir / "a0_union" / "prediction_owned_regions.json"
    summary = out_dir / "a0_union" / "inference_summary.json"
    if record["status"] == "OK" and pred.is_file():
        fs = bb.read_json(summary)
        result["replay"] = {"prediction_sha256": bb.sha256_file(pred), "graph_sha256": fs.get("graph_sha256"),
                            "typed_assignment_count": fs.get("typed_assignment_count"),
                            "counted_bend_component_count": fs.get("counted_bend_component_count"),
                            "solver_status": fs.get("solver_status")}
        for name, directory in compares.items():
            hs = bb.read_json(directory / "inference_summary.json")
            ref_pred = directory / "prediction_owned_regions.json"
            result["comparisons"][name] = {
                "dir": str(directory),
                "prediction_sha256": bb.sha256_file(ref_pred),
                "prediction_byte_identical": bb.sha256_file(ref_pred) == result["replay"]["prediction_sha256"],
                "graph_sha256": hs.get("graph_sha256"),
                "graph_identical": hs.get("graph_sha256") == result["replay"]["graph_sha256"],
                "typed_assignment_count": hs.get("typed_assignment_count"),
                "counted_bend_component_count": hs.get("counted_bend_component_count"),
                "lines": _line_set_diff(_lines_of(bb.read_json(pred)), _lines_of(bb.read_json(ref_pred)))}
    bb.write_json(args.out, result)
    print(json.dumps({"status": record["status"], "wall_s": record["wall_s"], "replay": result.get("replay"),
                      "comparisons": {k: {kk: v[kk] for kk in ("prediction_byte_identical", "graph_identical",
                                                               "counted_bend_component_count")}
                                      for k, v in result["comparisons"].items()}}, default=str))
    return 0 if record["status"] == "OK" else 2


def cmd_fresh_crosscheck(args: argparse.Namespace) -> int:
    """Independent re-scoring of the fresh lane files: score.py's own gold11 subcommand (catalog +
    frame records) and the frozen evaluator CLI (label-file gold) on its 9 corpus cases plus
    match_lines on the 2 shaked_xv cases. Must equal the summarizer's numbers."""
    from scripts.evaluate_f1_against_manual_gold import CASE_JOBS, Thresholds, match_lines

    entries = _gold11_entries()
    catalog_by_sha = {row["scan_sha256"]: row for row in bb.read_json(bb.GOLD_CATALOG)["cases"]}
    out_dir = REPRO_DIR / "fresh_crosscheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lane in ("best_f1", "runtime"):
        pred_map = {}
        for e in entries:
            path = RUNS_ROOT / e["case_id"] / lane / f"predictions_{lane}.json"
            if path.is_file():
                pred_map[e["gold11_id"]] = str(path)
        map_path = bb.write_json(out_dir / f"pred_map_{lane}.json", pred_map)
        score_out = out_dir / f"score_py_gold11_{lane}.json"
        done = subprocess.run([str(bb.APP_PYTHON), "-B", str(bb.HERE / "score.py"), "gold11", "--pred-map", str(map_path),
                               "--lane", lane, "--out", str(score_out)], cwd=str(bb.ROOT), capture_output=True, text=True,
                              check=False)
        if done.returncode:
            raise RuntimeError(done.stderr[-2000:])
        block = bb.read_json(score_out)["by_tier"][f"GOLD11_{lane}"]["complete_labels"]
        case_map, shaked = {}, []
        for e in entries:
            if e["gold11_id"] not in pred_map:
                continue
            kind, name = g11.catalog_case(e["gold11_id"]).split("/", 1)
            if kind == "corpus" and name in CASE_JOBS:
                case_map[name] = pred_map[e["gold11_id"]]
            else:
                lines = bb.read_json(pred_map[e["gold11_id"]]).get("owned_bend_regions") or []
                res = match_lines(catalog_by_sha[e["scan_sha256"]]["bends"], lines, Thresholds())
                shaked.append({k: res[k] for k in ("true_positive", "false_positive", "false_negative")})
        cmap = bb.write_json(out_dir / f"evaluator_case_map_{lane}.json", case_map)
        eval_out = out_dir / f"evaluator_cli_{lane}_9_corpus.json"
        done = subprocess.run([str(bb.APP_PYTHON), "-B", str(bb.ROOT / "scripts" / "evaluate_f1_against_manual_gold.py"),
                               "--case-map", str(cmap), "--output", str(eval_out)], cwd=str(bb.ROOT),
                              capture_output=True, text=True, check=False)
        if done.returncode:
            raise RuntimeError(done.stderr[-2000:])
        m = bb.read_json(eval_out)["metrics"]
        tot = sc.prf(m["true_positive"] + sum(x["true_positive"] for x in shaked),
                     m["false_positive"] + sum(x["false_positive"] for x in shaked),
                     m["false_negative"] + sum(x["false_negative"] for x in shaked))
        results[lane] = {"cases": len(pred_map),
                         "score_py_gold11": {k: block[k] for k in ("true_positive", "false_positive", "false_negative",
                                                                   "precision", "recall", "f1")},
                         "evaluator_cli_9_plus_match_lines_2": {k: tot[k] for k in ("true_positive", "false_positive",
                                                                                    "false_negative", "precision",
                                                                                    "recall", "f1")},
                         "agree": (block["true_positive"], block["false_positive"], block["false_negative"])
                         == (tot["true_positive"], tot["false_positive"], tot["false_negative"])}
    payload = {"schema": "sherman.broad_baseline.gold11_fresh_crosscheck.v1", "created_at": bb.utc_now(),
               "lanes": results}
    bb.write_json(out_dir / "fresh_crosscheck.json", payload)
    print(json.dumps(results, indent=1))
    return 0 if all(v["agree"] for v in results.values()) else 3


def cmd_runtime_determinism(args: argparse.Namespace) -> int:
    """Compare two independent runtime-lane passes over the Gold11 scans (status, line endpoints)."""
    entries = _gold11_entries()
    rerun_root = bb.expand(args.rerun_root)
    rows = []
    for entry in entries:
        row: dict[str, Any] = {"case_id": entry["gold11_id"]}
        def variant_file(root: Path, variant: str) -> Path:
            path = root / entry["case_id"] / "runtime" / f"predictions_{variant}.json"
            primary = root / entry["case_id"] / "runtime" / "predictions_runtime.json"
            if not path.is_file() and primary.is_file() and bb.read_json(primary).get("status") == "LANE_FAILED":
                return primary  # older wrapper wrote a failure only to the primary file
            return path

        for variant in FRESH_VARIANTS["runtime"]:
            a = variant_file(RUNS_ROOT, variant)
            b = variant_file(rerun_root, variant)
            if not a.is_file() or not b.is_file():
                row[variant] = {"first_exists": a.is_file(), "rerun_exists": b.is_file()}
                continue
            pa, pb = bb.read_json(a), bb.read_json(b)
            diff = _line_set_diff(_lines_of(pa), _lines_of(pb))
            row[variant] = {"status": [pa.get("status"), pb.get("status")], "lines": [pa.get("line_count"), pb.get("line_count")],
                            "identical": diff["identical"] and pa.get("status") == pb.get("status"),
                            "max_endpoint_abs_diff_mm": diff["max_endpoint_abs_diff_mm"]}
        ra = _run_record(entry, "runtime")
        rb_path = rerun_root / entry["case_id"] / "runtime" / "run_record.json"
        rb = bb.read_json(rb_path) if rb_path.is_file() else None
        row["lane_wall_s"] = [ra and ra.get("lane_wall_s"), rb and rb.get("lane_wall_s")]
        row["peak_rss_gib"] = [ra and _gib(ra.get("peak_rss_bytes")), rb and _gib(rb.get("peak_rss_bytes"))]
        rows.append(row)
    identical = all(all((row.get(v) or {}).get("identical") for v in FRESH_VARIANTS["runtime"]) for row in rows)
    payload = {"schema": "sherman.broad_baseline.runtime_determinism.v1", "created_at": bb.utc_now(),
               "first_pass": str(RUNS_ROOT), "rerun": str(rerun_root), "all_identical": identical, "cases": rows}
    bb.write_json(args.out, payload)
    print(json.dumps({"all_identical": identical,
                      "non_identical": [r["case_id"] for r in rows
                                        if not all((r.get(v) or {}).get("identical") for v in FRESH_VARIANTS["runtime"])]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluator-crosscheck")
    p.add_argument("--out-dir", type=Path, default=HIST_DIR)
    p.set_defaults(func=cmd_evaluator_crosscheck)
    p = sub.add_parser("replay-typed-union")
    p.add_argument("--pool", required=True)
    p.add_argument("--compare", action="append", default=[], help="NAME=DIR of an A0 case dir to compare with")
    p.add_argument("--purpose", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--rss-limit-gib", type=float, default=9.0)
    p.set_defaults(func=cmd_replay_typed_union)
    p = sub.add_parser("fresh-crosscheck")
    p.set_defaults(func=cmd_fresh_crosscheck)
    p = sub.add_parser("runtime-determinism")
    p.add_argument("--rerun-root", required=True)
    p.add_argument("--out", type=Path, default=REPRO_DIR / "runtime_determinism.json")
    p.set_defaults(func=cmd_runtime_determinism)
    p = sub.add_parser("summarize")
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--findings", type=Path, help="JSON list of narrative findings to embed")
    p.set_defaults(func=cmd_summarize)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
