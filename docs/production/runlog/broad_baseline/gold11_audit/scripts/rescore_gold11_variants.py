#!/usr/bin/env python3
"""Rescore the EXISTING best_f1 and runtime predictions against the Gold11 truth variants.

No engine is re-run: predictions are read from docs/production/runlog/broad_baseline/runs/<lane>/<scan>/.
Every score comes from research/broad_baseline/score.py (subcommand `broad`, the frozen matcher via
score.score_case), run as a subprocess on the case lists written by build_truth_variants.py.

Also computed here, in-process with the same score.score_case:
  - a "perfect apex engine" ceiling: the AI relabel apex lines used as predictions against each
    Gold11 variant (what an engine that draws exactly one outer-apex line per real formed fold
    would score);
  - scan-resampling bootstrap CIs (10,000 resamples, seed 20261017, as the broad baseline report).

Outputs: gold11_audit/rescore/<variant>/scores_<lane>.json and gold11_audit/rescore/RESCORE_SUMMARY.json.

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/rescore_gold11_variants.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g11a_common as C  # noqa: E402

import score as sc  # noqa: E402

TRUTH = C.AUDIT / "truth"
RESCORE = C.AUDIT / "rescore"
RUNS = C.BB / "runs"
LANES = ("best_f1", "runtime")
GOLD11_VARIANTS = ("human", "relabel_v1", "all11_ai_relabel", "human_snapped_apex", "human_snapped_apex_conv")
BROAD_VARIANT = "broad57_gold11_relabel_v1"
COHORTS_BROAD = ("ALL", "GOLD11", "NON_GOLD11", "CAD_AVAILABLE", "NO_CAD", "FULL_SCANS", "STAGED_SCANS",
                 "AI_GOLD", "ALL_EXCL_LIDAR_DEMO", "NON_GOLD11_CAD_AVAILABLE_EXCL_LIDAR_DEMO")


def run_score(case_list: Path, lane: str, out_dir: Path) -> dict:
    cmd = [str(C.ROOT / ".venv" / "bin" / "python"), "-B", "research/broad_baseline/score.py", "broad",
           "--case-list", C.rel(case_list), "--runs-root", C.rel(RUNS), "--lane", lane,
           "--frame-records", C.rel(C.FRAME_RECORDS), "--out-dir", C.rel(out_dir)]
    env = dict(os.environ, **C.bb.THREAD_ENV)
    proc = subprocess.run(cmd, cwd=str(C.ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"score.py failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    (out_dir / f"score_{lane}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    return {"command": " ".join(cmd[1:]), "rc": proc.returncode}


def block(c: dict) -> dict:
    keys = ("scans", "gold_lines", "predicted_lines", "true_positive", "false_positive", "false_negative",
            "micro_precision", "micro_recall", "micro_f1", "macro_f1", "exact_scans", "scans_with_zero_tp",
            "angle_mae_deg_matched", "position_error_median_mm_matched")
    return {k: c.get(k) for k in keys}


def per_scan(report: dict) -> dict:
    return {r["case_id"]: {k: r.get(k) for k in ("gold_count", "prediction_count", "true_positive",
                                                  "false_positive", "false_negative", "precision", "recall", "f1")}
            for r in report["per_scan"]}


def bootstrap(reports: dict, ids: list[str], reps: int = 10000, seed: int = 20261017) -> dict:
    rng = np.random.default_rng(seed)
    rows = {lane: {r["case_id"]: (r["true_positive"], r["false_positive"], r["false_negative"])
                   for r in rep["per_scan"] if r.get("disposition") == "SCORED_COMPLETE_LABEL"}
            for lane, rep in reports.items()}
    ids = [i for i in ids if all(i in rows[lane] for lane in rows)]
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))
    out, f = {"scans": len(ids)}, {}
    for lane in rows:
        arr = np.array([rows[lane][i] for i in ids], dtype=float)
        t = arr[idx].sum(axis=1)
        den = 2 * t[:, 0] + t[:, 1] + t[:, 2]
        f[lane] = np.where(den > 0, 2 * t[:, 0] / np.maximum(den, 1e-9), 0.0)
        out[lane] = [round(float(np.percentile(f[lane], 2.5)), 3), round(float(np.percentile(f[lane], 97.5)), 3)]
    if "best_f1" in f and "runtime" in f:
        d = f["best_f1"] - f["runtime"]
        out["best_minus_runtime"] = [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
    return out


def paired_delta_ci(rep_a: dict, rep_b: dict, ids: list[str], reps: int = 10000, seed: int = 20261017) -> list:
    """CI of F1(variant b) - F1(variant a) for one lane, resampling the same scans."""
    rng = np.random.default_rng(seed)
    a = {r["case_id"]: (r["true_positive"], r["false_positive"], r["false_negative"]) for r in rep_a["per_scan"]}
    b = {r["case_id"]: (r["true_positive"], r["false_positive"], r["false_negative"]) for r in rep_b["per_scan"]}
    A = np.array([a[i] for i in ids], dtype=float)
    B = np.array([b[i] for i in ids], dtype=float)
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))

    def f1(arr):
        t = arr[idx].sum(axis=1)
        den = 2 * t[:, 0] + t[:, 1] + t[:, 2]
        return np.where(den > 0, 2 * t[:, 0] / np.maximum(den, 1e-9), 0.0)

    d = f1(B) - f1(A)
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def ceiling(case_list: Path) -> dict:
    """AI relabel apex lines as predictions against a variant's gold, frozen matcher."""
    cl = C.read_json(case_list)
    frames = C.read_json(C.FRAME_RECORDS)
    rows = []
    for entry in cl["cases"]:
        scan = entry["case_id"]
        label = C.read_json(C.RELABEL_ROOT / f"{C.RELABEL_ASSET[scan]}.json")
        preds = {"owned_bend_regions": [C.bb.as_line(b["bend_id"], b["line_start"], b["line_end"])
                                        for b in label["bends"]],
                 "status": "OK", "lane": "perfect_apex_engine(ai_relabel_lines)",
                 "scan": {"sha256": entry["scan_sha256"]}}
        row = sc.score_case({"case_id": scan, "tier": "GOLD11", "family_id": entry["family_id"],
                             "gold": entry["gold"], "label_complete": True}, frame_records=frames,
                            preloaded_predictions=preds)
        rows.append(row)
    tp = sum(r["true_positive"] for r in rows)
    fp = sum(r["false_positive"] for r in rows)
    fn = sum(r["false_negative"] for r in rows)
    agg = sc.prf(tp, fp, fn)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(agg["precision"], 4), "recall": round(agg["recall"], 4), "f1": round(agg["f1"], 4),
            "per_scan": {r["case_id"]: (r["true_positive"], r["false_positive"], r["false_negative"]) for r in rows}}


def main() -> int:
    runs = {}
    reports: dict[str, dict[str, dict]] = {}
    for variant in GOLD11_VARIANTS + (BROAD_VARIANT,):
        case_list = TRUTH / (f"case_list_gold11__{variant}.json" if variant != BROAD_VARIANT
                             else "case_list_broad_v3__gold11_relabel_v1.json")
        out_dir = RESCORE / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        reports[variant] = {}
        for lane in LANES:
            runs[f"{variant}:{lane}"] = run_score(case_list, lane, out_dir)
            reports[variant][lane] = C.read_json(out_dir / f"scores_{lane}.json")

    published = {lane: C.read_json(C.BB / f"scores_{lane}.json") for lane in LANES}
    g11_ids = sorted(C.SCANS)
    summary = {"schema": "sherman.gold11_audit.rescore_summary.v1", "created_at": C.bb.utc_now(),
               "not_redone": "no engine, prep, lines or labeling step was re-run; existing predictions only",
               "matcher": reports["human"]["best_f1"]["matcher"],
               "predictions": {lane: C.rel(RUNS / lane) + f"/<scan>/predictions_{lane}.json" for lane in LANES},
               "frame_records": C.rel(C.FRAME_RECORDS),
               "score_runs": runs, "gold11": {}, "broad57": {}, "checks": {}}
    for variant in GOLD11_VARIANTS:
        reps = reports[variant]
        summary["gold11"][variant] = {
            "case_list": reps["best_f1"]["case_list"],
            "lanes": {lane: {"gold11": block(reps[lane]["cohorts"]["GOLD11"]),
                             "per_scan": per_scan(reps[lane])} for lane in LANES},
            "bootstrap_f1_ci95": bootstrap(reps, g11_ids),
            "perfect_apex_engine_ceiling": ceiling(TRUTH / f"case_list_gold11__{variant}.json"),
        }
        if variant != "human":
            summary["gold11"][variant]["delta_vs_human_f1_ci95"] = {
                lane: paired_delta_ci(reports["human"][lane], reps[lane], g11_ids) for lane in LANES}

    b = reports[BROAD_VARIANT]
    all_ids = [r["case_id"] for r in b["best_f1"]["per_scan"] if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    cad_ids = [r["case_id"] for r in b["best_f1"]["cases"]
               if r.get("disposition") == "SCORED_COMPLETE_LABEL" and r.get("has_cad")]
    non_ids = [r["case_id"] for r in b["best_f1"]["cases"]
               if r.get("disposition") == "SCORED_COMPLETE_LABEL" and r.get("tier") != "GOLD11"]
    summary["broad57"] = {
        "case_list": b["best_f1"]["case_list"],
        "lanes": {lane: {k: block(b[lane]["cohorts"][k]) for k in COHORTS_BROAD} for lane in LANES},
        "published_v3": {lane: {k: block(published[lane]["cohorts"][k]) for k in COHORTS_BROAD} for lane in LANES},
        "bootstrap_f1_ci95": {"ALL": bootstrap(b, all_ids), "CAD_AVAILABLE": bootstrap(b, cad_ids),
                              "NON_GOLD11": bootstrap(b, non_ids)},
        "delta_vs_published_f1_ci95_ALL": {lane: paired_delta_ci(published[lane], b[lane], all_ids) for lane in LANES},
        "cohort_note": ("cohorts are the scorer's tier-based ones; in this variant HUMAN_GOLD_ALL still counts the "
                        "11 GOLD11-tier scans although 10 of them now carry AI relabels, and AI_GOLD excludes them"),
    }

    # ---- substitution-only broad57: Gold11 rows from the relabel run, every other row from the
    # published v3 scores (frozen gold). Needed because one non-Gold11 AI gold file changed on disk
    # after v3 was frozen (sc_2233372b8df7: revision 2 adds two 15 deg joggle bends, owner request);
    # micro/macro sums are per-scan additive, and the cohorts are recomputed with score_broad's own
    # aggregate() and COHORTS predicates.
    import score_broad as sb  # noqa: E402

    changed = {}
    for lane in LANES:
        pub_rows = {r["case_id"]: r for r in published[lane]["cases"]}
        cur_rows = {r["case_id"]: r for r in b[lane]["cases"]}
        keys = ("true_positive", "false_positive", "false_negative", "matches", "unmatched_gold_ids",
                "unmatched_prediction_ids", "disposition")
        changed[lane] = [k for k, r in pub_rows.items() if r.get("tier") != "GOLD11"
                         and {x: r.get(x) for x in keys} != {x: cur_rows[k].get(x) for x in keys}]
        merged = [cur_rows[k] if pub_rows[k].get("tier") == "GOLD11" else pub_rows[k] for k in pub_rows]
        summary["broad57"].setdefault("substitution_only", {})[lane] = {
            k: block(sb.aggregate([r for r in merged if fn(r)])) for k, fn in sb.COHORTS.items() if k in COHORTS_BROAD}
        summary["broad57"].setdefault("_merged_rows", {})[lane] = merged
    sub_reports = {lane: {"per_scan": [r for r in summary["broad57"]["_merged_rows"][lane]
                                       if r.get("disposition") == "SCORED_COMPLETE_LABEL"]} for lane in LANES}
    summary["broad57"]["substitution_only_bootstrap_f1_ci95"] = {
        "ALL": bootstrap(sub_reports, all_ids), "CAD_AVAILABLE": bootstrap(sub_reports, cad_ids)}
    summary["broad57"]["substitution_only_delta_vs_published_f1_ci95_ALL"] = {
        lane: paired_delta_ci(published[lane], sub_reports[lane], all_ids) for lane in LANES}
    summary["broad57"].pop("_merged_rows")
    summary["broad57"]["non_gold11_rows_changed_since_v3"] = changed
    summary["broad57"]["note"] = (
        "'lanes' = score.py run on case_list_broad_v3__gold11_relabel_v1 with the label files as they are on "
        "disk now; 'substitution_only' = the same Gold11 rows combined with the published v3 rows for every "
        "other scan, i.e. the effect of the Gold11 substitution alone. The two differ only by "
        "sc_2233372b8df7 (AI gold revision 2 added two 15 deg joggle bends after v3 was frozen: +2 FN in "
        "each lane).")

    # ---- reproduction checks
    checks = {}
    for lane in LANES:
        pub = published[lane]["cohorts"]["GOLD11"]
        got = reports["human"][lane]["cohorts"]["GOLD11"]
        checks[f"human_gold11_reproduces_published_{lane}"] = all(
            pub[k] == got[k] for k in ("true_positive", "false_positive", "false_negative", "micro_f1"))
        pub_non = published[lane]["cohorts"]["NON_GOLD11"]
        got_non = summary["broad57"]["substitution_only"][lane]["NON_GOLD11"]
        checks[f"substitution_only_non_gold11_equals_published_{lane}"] = all(
            pub_non[k] == got_non[k] for k in ("true_positive", "false_positive", "false_negative", "micro_f1"))
        checks[f"only_sc_2233372b8df7_changed_outside_gold11_{lane}"] = (
            summary["broad57"]["non_gold11_rows_changed_since_v3"][lane] == ["sc_2233372b8df7"])
        checks[f"broad57_gold11_equals_gold11_relabel_v1_{lane}"] = all(
            b[lane]["cohorts"]["GOLD11"][k] == reports["relabel_v1"][lane]["cohorts"]["GOLD11"][k]
            for k in ("true_positive", "false_positive", "false_negative", "micro_f1"))
        checks[f"excluded_before_scoring_{lane}"] = {v: reports[v][lane]["excluded_before_scoring"]
                                                     for v in GOLD11_VARIANTS + (BROAD_VARIANT,)}
    summary["checks"] = checks
    C.write_json(RESCORE / "RESCORE_SUMMARY.json", summary)

    for variant in GOLD11_VARIANTS:
        s = summary["gold11"][variant]
        line = " | ".join(f"{lane}: {s['lanes'][lane]['gold11']['true_positive']}/{s['lanes'][lane]['gold11']['false_positive']}/"
                          f"{s['lanes'][lane]['gold11']['false_negative']} F1={s['lanes'][lane]['gold11']['micro_f1']:.4f}"
                          for lane in LANES)
        cei = s["perfect_apex_engine_ceiling"]
        print(f"{variant:26s} {line} | ceiling {cei['true_positive']}/{cei['false_positive']}/{cei['false_negative']} "
              f"F1={cei['f1']} | CI {s['bootstrap_f1_ci95']}")
    for lane in LANES:
        for k in ("ALL", "GOLD11", "NON_GOLD11", "CAD_AVAILABLE"):
            c = summary["broad57"]["lanes"][lane][k]
            s_ = summary["broad57"]["substitution_only"][lane][k]
            p = summary["broad57"]["published_v3"][lane][k]
            print(f"broad57 {lane:8s} {k:14s} now {c['true_positive']}/{c['false_positive']}/{c['false_negative']} "
                  f"F1={c['micro_f1']:.4f} | subst-only {s_['true_positive']}/{s_['false_positive']}/"
                  f"{s_['false_negative']} F1={s_['micro_f1']:.4f} | published {p['true_positive']}/"
                  f"{p['false_positive']}/{p['false_negative']} F1={p['micro_f1']:.4f}")
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("excluded")}, indent=1))
    ok = all(v for k, v in checks.items() if not k.startswith("excluded"))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
