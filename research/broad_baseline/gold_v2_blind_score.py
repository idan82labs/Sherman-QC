#!/usr/bin/env python3
"""Score best_f1 (and its scan-only A0 half) on the full GOLD_SET_V2 lists (all / cad / nocad).

Case lists = gold_set_v2/rescore/case_list_gold_v2_<name>__scored.json (cases that already had
predictions; v1 twins keep their STL prediction ids) + every case of case_list_gold_v2_<name>.json
missing from it (predictions filled by gold_v2_fill_driver.sh under their own case id).
Scoring = score.py broad (frozen matcher 5 deg / 4 mm / 60 %, apex locus, raw frame) with the
broad frame records. On top: family-level bootstrap CIs (resample families with replacement),
coverage, FULL vs STAGED, v1-part vs new-part, runtime stats of the fill runs.

  .venv/bin/python -B research/broad_baseline/gold_v2_blind_score.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

ROOT = bb.ROOT
GS2 = ROOT / "docs/production/runlog/gold_set_v2"
RES = GS2 / "rescore"
OUT = GS2 / "rescore_full"
RUNS = ROOT / "docs/production/runlog/broad_baseline/runs"
FRAMES = ROOT / "docs/production/runlog/broad_baseline/broad_frame_records.json"
FILL_LIST = OUT / "case_list_gold_v2_missing46.json"
LISTS = ("all", "cad", "nocad")
VARIANTS = {"best_f1": "scores_best_f1.json", "a0_scan_only": "scores_best_f1__a0.json"}
N_BOOT = 10000
SEED = 20261018


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def family_stats(rows):
    fam = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        f = fam[r["family_id"]]
        f[0] += r["true_positive"]
        f[1] += r["false_positive"]
        f[2] += r["false_negative"]
    return fam


def metrics(rows):
    tp = sum(r["true_positive"] for r in rows)
    fp = sum(r["false_positive"] for r in rows)
    fn = sum(r["false_negative"] for r in rows)
    p, r_, f = prf(tp, fp, fn)
    fam = family_stats(rows)
    fprf = [prf(*v) for v in fam.values()]
    return {"scans": len(rows), "families": len(fam), "gold_lines": tp + fn, "predicted_lines": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "micro_precision": round(p, 4), "micro_recall": round(r_, 4),
            "micro_f1": round(f, 4),
            "family_macro_precision": round(float(np.mean([x[0] for x in fprf])), 4) if fprf else None,
            "family_macro_recall": round(float(np.mean([x[1] for x in fprf])), 4) if fprf else None,
            "family_macro_f1": round(float(np.mean([x[2] for x in fprf])), 4) if fprf else None}


def bootstrap(rows, rng):
    fam = family_stats(rows)
    keys = list(fam)
    if not keys:
        return None
    arr = np.array([fam[k] for k in keys], float)
    n = len(keys)
    micro_f, micro_p, micro_r, macro = [], [], [], []
    fam_f1 = np.array([prf(*v)[2] for v in arr])
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        tp, fp, fn = arr[idx].sum(0)
        p, r, f = prf(tp, fp, fn)
        micro_f.append(f)
        micro_p.append(p)
        micro_r.append(r)
        macro.append(fam_f1[idx].mean())

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    return {"resample_unit": "family", "n_boot": N_BOOT, "seed": SEED, "micro_f1_95ci": ci(micro_f),
            "micro_precision_95ci": ci(micro_p), "micro_recall_95ci": ci(micro_r),
            "family_macro_f1_95ci": ci(macro)}


def case_list_sha(cases):
    """Same definition as gold_set_v2/scripts/build_gold_set_v2.case_list_sha."""
    import hashlib
    keys = ("case_id", "scan_sha256", "gold", "gold_file_sha256", "label_complete", "zero_bend_control", "cad_sha256")
    rows = [{k: c.get(k) for k in keys} for c in cases]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_lists():
    fill_ids = {c["case_id"] for c in bb.read_json(FILL_LIST)["cases"]}
    out = {}
    for name in LISTS:
        base = bb.read_json(GS2 / f"case_list_gold_v2_{name}.json")
        scored = bb.read_json(RES / f"case_list_gold_v2_{name}__scored.json")
        have = {c.get("gold_set_v2_case_id", c["case_id"]) for c in scored["cases"]}
        cases = [copy.deepcopy(c) for c in scored["cases"]]
        added = []
        for e in base["cases"]:
            if e["case_id"] in have:
                continue
            x = copy.deepcopy(e)
            x["gold_set"] = e.get("tier") if e.get("tier") in ("GOLD_SET_V1",) else "GOLD_SET_V2_NEW"
            x["tier"] = "GOLD_SET_V2"
            x["gold_set_v2_case_id"] = e["case_id"]
            x["filled_by"] = "gold_v2_fill_driver.sh (best_f1, 2026-10-18)"
            cases.append(x)
            added.append(e["case_id"])
        assert not set(added) - fill_ids, set(added) - fill_ids
        assert len(cases) == len(base["cases"]), (name, len(cases), len(base["cases"]))
        payload = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": bb.utc_now(),
                   "name": f"case_list_gold_v2_{name}__full",
                   "case_list_sha256": case_list_sha(cases),
                   "variant_note": "every GOLD_SET_V2 case: rescore/__scored cases (existing predictions; v1 twins "
                                   "under their STL prediction id) + the cases filled by the 2026-10-18 best_f1 run",
                   "derived_from": {"case_list": rel(GS2 / f"case_list_gold_v2_{name}.json"),
                                    "case_list_file_sha256": bb.sha256_file(GS2 / f"case_list_gold_v2_{name}.json"),
                                    "case_list_sha256": base.get("case_list_sha256"),
                                    "scored_list": rel(RES / f"case_list_gold_v2_{name}__scored.json")},
                   "summary": {"cases": len(cases), "filled": len(added)}, "cases": cases}
        path = OUT / f"case_list_gold_v2_{name}__full.json"
        bb.write_json(path, payload)
        out[name] = path
    return out


def run_scores(lists):
    cmds = {}
    for name, path in lists.items():
        od = OUT / name
        od.mkdir(parents=True, exist_ok=True)
        cmd = [str(bb.APP_PYTHON), "-B", "research/broad_baseline/score.py", "broad", "--case-list", rel(path),
               "--runs-root", rel(RUNS), "--lane", "best_f1", "--variant", "predictions_a0.json",
               "--frame-records", rel(FRAMES), "--out-dir", rel(od)]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ, **bb.THREAD_ENV), capture_output=True,
                              text=True)
        (od / "score_best_f1.stdout.txt").write_text(proc.stdout + proc.stderr[-4000:])
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:])
        cmds[name] = {"command": " ".join(cmd[1:]), "rc": proc.returncode}
    return cmds


def runtime_stats():
    ids = [c["case_id"] for c in bb.read_json(FILL_LIST)["cases"]]
    rows = []
    for cid in ids:
        rp = RUNS / "best_f1" / cid / "run_record.json"
        if not rp.is_file():
            rows.append({"case_id": cid, "status": "NOT_RUN"})
            continue
        r = bb.read_json(rp)
        if not str(r.get("finished_at", "")).startswith("2026-10-18"):
            rows.append({"case_id": cid, "status": "NOT_RUN_IN_FILL", "run_record_finished_at": r.get("finished_at")})
            continue
        steps = r.get("steps", [])
        failed = [s for s in steps if s.get("returncode") not in (0, None) or s.get("timed_out")]
        rows.append({"case_id": cid, "status": r.get("status"), "cad": bool(r.get("cad")),
                     "line_count": r.get("line_count"), "lane_wall_s": r.get("lane_wall_s"),
                     "peak_rss_gib": round((r.get("peak_rss_bytes") or 0) / 2**30, 3),
                     "failed_steps": [{"name": s["name"], "rc": s.get("returncode"), "timed_out": s.get("timed_out"),
                                       "stderr_tail": (s.get("stderr_tail") or "")[-400:]} for s in failed],
                     "failure": r.get("failure")})
    ran = [x for x in rows if x.get("lane_wall_s") is not None]
    walls = np.array([x["lane_wall_s"] for x in ran]) if ran else np.array([0.0])
    rss = np.array([x["peak_rss_gib"] for x in ran]) if ran else np.array([0.0])
    return {"cases": len(rows), "ran": len(ran), "status_counts": dict(Counter(x["status"] for x in rows)),
            "wall_s": {"median": round(float(np.median(walls)), 1), "p95": round(float(np.percentile(walls, 95)), 1),
                       "max": round(float(walls.max()), 1), "total_h": round(float(walls.sum()) / 3600, 2)},
            "peak_rss_gib": {"median": round(float(np.median(rss)), 2), "p95": round(float(np.percentile(rss, 95)), 2),
                             "max": round(float(rss.max()), 2)},
            "per_case": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lists = build_lists()
    cmds = run_scores(lists)
    rng = np.random.default_rng(SEED)
    results = {}
    for name in LISTS:
        gs_of = {c["case_id"]: c.get("gold_set") for c in bb.read_json(lists[name])["cases"]}
        results[name] = {}
        for vname, fname in VARIANTS.items():
            rep = bb.read_json(OUT / name / fname)
            rows = rep["cases"]
            complete = [r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
            controls = [r for r in rows if r.get("disposition") == "ZERO_BEND_CONTROL"]
            excluded = [r for r in rows if r.get("disposition") not in ("SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL")]
            not_run = [r["case_id"] for r in rows if (r.get("run") or {}).get("status") == "NOT_RUN"]
            failed = [r["case_id"] for r in rows if (r.get("run") or {}).get("status") in ("LANE_FAILED",)]
            covered = [r for r in complete if (r.get("run") or {}).get("status") not in ("NOT_RUN",)]
            cohorts = {}
            for ck, fn in (("ALL", lambda r: True), ("FULL", lambda r: not r["staged"]),
                           ("STAGED", lambda r: r["staged"]),
                           ("GOLD_SET_V1_PART", lambda r: gs_of.get(r["case_id"]) == "GOLD_SET_V1"),
                           ("GOLD_SET_V2_NEW_PART", lambda r: gs_of.get(r["case_id"]) == "GOLD_SET_V2_NEW")):
                sub = [r for r in complete if fn(r)]
                cohorts[ck] = {**metrics(sub), "bootstrap": bootstrap(sub, rng)} if sub else {"scans": 0}
            results[name][vname] = {
                "coverage": {"gold_scans": len(rows), "scored_line_scans": len(complete),
                             "zero_bend_controls": len(controls), "excluded_before_scoring": [
                                 {"case_id": r["case_id"], "reason": r.get("reason")} for r in excluded],
                             "with_run_record": len(rows) - len(not_run), "not_run": not_run,
                             "lane_failed": failed,
                             "lane_status_counts": dict(Counter(str((r.get("run") or {}).get("status")) for r in rows))},
                "cohorts": cohorts,
                "covered_only": metrics(covered) if len(covered) != len(complete) else "same as ALL",
                "zero_bend_controls": {"scans": len(controls),
                                       "fp_lines": sum(r.get("false_positive_lines", 0) for r in controls),
                                       "scans_with_fp": sorted(r["case_id"] for r in controls
                                                               if r.get("false_positive_lines"))},
                "scores_file": rel(OUT / name / fname)}
    summary = {"schema": "sherman.gold_set_v2.blind_track_baseline.v1", "created_at": bb.utc_now(),
               "lane": "best_f1 (headline: A0 scan-only UNION gated CAD half where a STEP exists; "
                       "a0_scan_only = the A0 half alone, i.e. strictly blind on every scan)",
               "runtime_lane": "dropped (owner discarded the shipped runtime engine as a comparator)",
               "matcher": bb.read_json(OUT / "all" / VARIANTS["best_f1"])["matcher"],
               "conventions": "micro = summed TP/FP/FN over complete-label line scans; family-macro = mean over "
                              "families of family-micro P/R/F1; CIs = percentile bootstrap resampling families "
                              "with replacement; zero-bend controls outside line-F1 (every line = FP); a scan "
                              "whose lane failed or produced nothing counts as all-FN",
               "case_lists": {k: rel(v) for k, v in lists.items()}, "score_runs": cmds,
               "fill_runtime": runtime_stats(), "results": results}
    bb.write_json(OUT / "BLIND_BASELINE_SUMMARY.json", summary)
    for name in LISTS:
        for vname in VARIANTS:
            c = results[name][vname]["cohorts"]
            for ck in ("ALL", "FULL", "STAGED"):
                m = c[ck]
                if not m.get("scans"):
                    continue
                b = m["bootstrap"]
                print(f"{name:5s} {vname:12s} {ck:6s} scans {m['scans']:3d} fam {m['families']:2d} gold {m['gold_lines']:3d} "
                      f"P {m['micro_precision']:.3f} R {m['micro_recall']:.3f} F1 {m['micro_f1']:.4f} "
                      f"{b['micro_f1_95ci']} famF1 {m['family_macro_f1']:.4f} {b['family_macro_f1_95ci']}")
        print(name, results[name]["best_f1"]["coverage"]["lane_status_counts"],
              "zero-bend", results[name]["best_f1"]["zero_bend_controls"])
    print("runtime", json.dumps({k: v for k, v in summary["fill_runtime"].items() if k != "per_case"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
