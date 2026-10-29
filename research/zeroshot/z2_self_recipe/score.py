"""Score a Z2 DEV run with the frozen matcher (research/broad_baseline/score.py score_case: 5 deg / 4 mm /
0.6 of both spans, Hungarian, apex locus, raw frame) and the common error budget (score_broad.diagnose).

usage: .venv/bin/python -m research.zeroshot.z2_self_recipe.score --tag it01 [--vs-a0]
Writes z2/runs/<tag>/scores.json and prints micro / family-macro, P/R, error budget, per-fold F1,
and the flange-pair recall proxy (a gold line is 'flange-covered' when some pair of discovered
flanges has an outer-skin intersection within 5 deg / 4 mm of it over >= 60 % of its span).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.zeroshot.z2_self_recipe import common as C
import score as S  # frozen matcher (research/broad_baseline)
import bb_common as bb
import score_broad as SB

A0_PER_CASE = C.HUB / "baseline/scores_dev_per_case.json"


def case(sid, inv):
    r = inv[sid]
    g = r["gold"]
    return {"case_id": sid, "family_id": r["family"], "tier": g["set"], "gold": {"kind": "ai_studio", "path": g["path"]},
            "label_complete": g.get("label_complete", True)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(rows: list[dict]) -> dict:
    rows = [x for x in rows if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    fam = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        f = fam[x["family_id"]]
        f[0] += x["true_positive"]; f[1] += x["false_positive"]; f[2] += x["false_negative"]
    tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
    p, r, f1 = prf(tp, fp, fn)
    macro = float(np.mean([prf(*v)[2] for v in fam.values()])) if fam else 0.0
    fnc, fpc = Counter(), Counter()
    for x in rows:
        fnc.update(x.get("fn_causes") or {}); fpc.update(x.get("fp_causes") or {})
    return {"scans": len(rows), "families": len(fam), "tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r, 4),
            "micro_f1": round(f1, 4), "family_macro_f1": round(macro, 4), "fn_causes": dict(fnc), "fp_causes": dict(fpc)}


def flange_cover(gold_lines, diag, t) -> list[bool]:
    """Proxy flange recall: for each gold line, does some flange pair's outer intersection explain it?"""
    fl = diag.get("flanges") or []
    planes = []
    for f in fl:
        sk = [(np.asarray(s["n"]), s["d"]) for s in f["skins"]]
        planes.append(sk)
    out = []
    for g in gold_lines:
        a, b = np.asarray(g["line_start"]), np.asarray(g["line_end"])
        ga = (b - a) / np.linalg.norm(b - a)
        mid = 0.5 * (a + b)
        ok = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                for (n1, d1) in planes[i]:
                    for (n2, d2) in planes[j]:
                        for o1 in (0.0, t):
                            for o2 in (0.0, t):
                                u = np.cross(n1, n2)
                                s = np.linalg.norm(u)
                                if s < math.sin(math.radians(5)):
                                    continue
                                u = u / s
                                if math.degrees(math.acos(min(1, abs(float(u @ ga))))) > 5:
                                    continue
                                # planes n.x = d (outer or moved by t to the other skin)
                                M = np.array([n1, n2, u])
                                p = np.linalg.solve(M, np.array([d1 - o1, d2 - o2, float(u @ mid)]))
                                dv = p - mid
                                off = np.linalg.norm(dv - ga * (dv @ ga))
                                if off < 4.0 + 2.0:          # sharp vs apex slack
                                    ok = True
                                    break
                            if ok: break
                        if ok: break
                    if ok: break
                if ok: break
            if ok: break
        out.append(ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quiet", action="store_true")
    x = ap.parse_args(argv)
    inv = C.inventory()
    folds = C.folds()
    root = C.OUT / "runs" / x.tag
    a0 = json.loads(A0_PER_CASE.read_text()).get("a0", {}) if A0_PER_CASE.is_file() else {}
    rows, a0rows, per = [], [], {}
    fc_tot = fc_hit = 0
    fc_full_tot = fc_full_hit = 0
    zero_fp = 0
    for sid in C.cohort():
        p = root / sid / "eval_lines.json"
        if not p.is_file():
            continue
        c = case(sid, inv)
        pr = dict(bb.load_eval_lines(p))
        pr["scan"] = {}
        rec = S.score_case(c, frame_records=None, preloaded_predictions=pr)
        gold = S.load_gold(c["gold"], frame_records=None, locus="apex")
        gl = gold["lines"]
        if rec.get("disposition") in ("SCORED_COMPLETE_LABEL", "SCORED_RECALL_ONLY_PARTIAL_LABEL"):
            pl = [q for q in pr.get("owned_bend_regions") or [] if bb.valid_line(q.get("line_start"), q.get("line_end"))]
            dg = SB.diagnose(gl, pl, rec)
            rec["fn_causes"], rec["fp_causes"] = dg["fn_cause_counts"], dg["fp_cause_counts"]
            rec["fn_rows"], rec["fp_rows"] = dg["false_negatives"], dg["false_positives"]
        if rec.get("disposition") == "ZERO_BEND_CONTROL":
            zero_fp += rec.get("false_positive_lines", 0)
        rec["fold"] = folds.get(sid)
        rec["stage"] = inv[sid]["stage"]
        rec["gold_set"] = inv[sid]["gold"]["set"]
        dp = root / sid / "diag.json"
        if dp.is_file() and gl:
            diag = json.loads(dp.read_text())
            cov = flange_cover(gl, diag, diag["t_mm"])
            rec["flange_cover"] = [bool(v) for v in cov]
            fc_tot += len(cov); fc_hit += sum(cov)
            if inv[sid]["stage"] == "full" and inv[sid]["gold"]["set"].startswith("GOLD_SET_V2"):
                fc_full_tot += len(cov); fc_full_hit += sum(cov)
            rec["timing_s"] = diag["timing_s"]["total"]
        rows.append(rec)
        per[sid] = rec
        if sid in a0:
            a0rows.append(a0[sid])
    agg = aggregate(rows)
    out = {"tag": x.tag, "matcher": "score.py score_case 5deg/4mm/0.6", "aggregate": agg,
           "zero_bend_fp_lines": zero_fp,
           "flange_pair_recall": {"all": round(fc_hit / max(fc_tot, 1), 4), "n": fc_tot,
                                  "gold_set_v2_full": round(fc_full_hit / max(fc_full_tot, 1), 4), "n_full": fc_full_tot},
           "per_fold": {}, "a0_same_scans": aggregate(a0rows) if a0rows else None}
    for k in range(5):
        out["per_fold"][k] = aggregate([r for r in rows if r.get("fold") == k])
    partial = [r for r in rows if r.get("disposition") == "SCORED_RECALL_ONLY_PARTIAL_LABEL"]
    out["partial_label_recall"] = (sum(r["true_positive"] for r in partial), sum(r["gold_count"] for r in partial))
    times = [r["timing_s"] for r in rows if r.get("timing_s")]
    if times:
        out["runtime_s"] = {"median": round(float(np.median(times)), 2), "p95": round(float(np.percentile(times, 95)), 2),
                            "max": round(float(np.max(times)), 2)}
    bb.write_json(root / "scores.json", {**out, "per_case": per})
    a = agg
    print(f"[{x.tag}] scans={a['scans']} fam={a['families']} TP={a['tp']} FP={a['fp']} FN={a['fn']} P={a['P']:.3f} R={a['R']:.3f} "
          f"micro={a['micro_f1']:.3f} macro={a['family_macro_f1']:.3f}")
    print("  FN:", a["fn_causes"], " FP:", a["fp_causes"], " zero-bend FP lines:", zero_fp)
    print("  flange-pair recall:", out["flange_pair_recall"], " partial recall:", out["partial_label_recall"])
    print("  folds:", {k: v["micro_f1"] for k, v in out["per_fold"].items()}, " runtime:", out.get("runtime_s"))
    if out["a0_same_scans"]:
        b = out["a0_same_scans"]
        print(f"  A0 same scans: micro={b['micro_f1']:.3f} macro={b['family_macro_f1']:.3f}")
    if not x.quiet:
        worst = sorted([r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"],
                       key=lambda r: (r.get("f1", 0), -r.get("gold_count", 0)))[:15]
        for r in worst:
            print(f"   {r['case_id']} fam={r['family_id']} g={r['gold_count']} p={r['prediction_count']} tp={r['true_positive']} "
                  f"FN={r.get('fn_causes')} FP={r.get('fp_causes')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
