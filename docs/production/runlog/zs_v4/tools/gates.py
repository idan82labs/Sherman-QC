"""PROTOCOL.md section 6/7 report for one v4 LOFO result vs the baselines: G1 (LOFO micro and macro vs A0 and A1),
G2 (paired CIs, 95 % and Bonferroni over the M variants in DESIGN_LOG.jsonl), G3 (per-family change in errors vs
A0), G4 (control false lines), G5 (FE1 subset), G6 (hem recall), lanced-tab / short-ear scans, error taxonomy counts.
usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/gates.py <variant> [--out GATES_<variant>.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring as SC  # noqa: E402
from research.zeroshot.zs_lane_v4 import evalkit as EK  # noqa: E402

RES = EK.V4RUN / "results"
HEM = re.compile(r"180 ?deg[^;|]{0,24}\bhem\b", re.I)
TAB_SCANS = ["gb_68cd06", "gb_959f4a"]
EAR_SCANS = ["gb_138090", "gb_1f3fe0"]


def paired_ci(recs_a, recs_b, alpha: float) -> dict:
    """paired_bootstrap with an arbitrary two-sided level (same families, n, seed as the protocol)."""
    def fam(recs):
        f = defaultdict(lambda: [0, 0, 0])
        for x in recs:
            if x.get("disposition") == "SCORED_COMPLETE_LABEL":
                v = f[x["family_id"]]
                v[0] += x["true_positive"]; v[1] += x["false_positive"]; v[2] += x["false_negative"]
        return f
    fa, fb = fam(recs_a), fam(recs_b)
    keys = sorted(set(fa) & set(fb))
    A = np.array([fa[k] for k in keys], float)
    B = np.array([fb[k] for k in keys], float)
    f1 = lambda v: SC.prf(*v)[2]  # noqa: E731
    fA = np.array([f1(v) for v in A]); fB = np.array([f1(v) for v in B])
    rng = np.random.default_rng(EK.SEED)
    dm, dM = np.empty(EK.N_BOOT), np.empty(EK.N_BOOT)
    for b in range(EK.N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        dm[b] = f1(B[idx].sum(0)) - f1(A[idx].sum(0))
        dM[b] = fB[idx].mean() - fA[idx].mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"alpha": alpha, "d_micro": round(f1(B.sum(0)) - f1(A.sum(0)), 4),
            "d_micro_ci": [round(float(np.percentile(dm, lo)), 4), round(float(np.percentile(dm, hi)), 4)],
            "d_macro": round(float(fB.mean() - fA.mean()), 4),
            "d_macro_ci": [round(float(np.percentile(dM, lo)), 4), round(float(np.percentile(dM, hi)), 4)]}


def hem_recall(pool, lines_by_case) -> dict:
    sc = EK.Scorer(pool)
    hit = tot = 0
    rows = []
    for s in pool:
        raw = {b["bend_id"]: b for b in EK.gold_raw(s)}
        hems = [k for k, b in raw.items() if HEM.search(b.get("notes") or "")]
        if not hems:
            continue
        r = sc.score(s["case_id"], lines_by_case.get(s["case_id"]))
        matched = {m["gold_id"] for m in r.get("matches") or []}
        for h in hems:
            tot += 1
            hit += int(h in matched)
            rows.append((s["case_id"], h, h in matched))
    return {"hem_lines": tot, "matched": hit, "recall": round(hit / tot, 3) if tot else None, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    a = ap.parse_args(argv)
    pool = EK.load_pool()
    base = json.loads((RES / "LOFO_BASELINES.json").read_text())
    v4 = json.loads((RES / f"v4/{a.variant}.json").read_text())
    arms = {k: base["per_case"][k] for k in ("A0_v1_pinned", "A1_v1_retrained", "A2_v3_pinned", "A3_v3_retrained")}
    arms["V4"] = v4["per_case"]
    M = sum(1 for _ in open(RES / "DESIGN_LOG.jsonl")) if (RES / "DESIGN_LOG.jsonl").is_file() else 1
    rep = {"variant": a.variant, "M_variants_scored": M, "arms": {}, "paired": {}}
    for k, recs in arms.items():
        ag = EK.aggregate(recs)
        fe = EK.aggregate([r for r in recs if r.get("origin") == "FE1"])
        rep["arms"][k] = {"ALL": EK.slim(ag), "FE1": EK.slim(fe)}
    for b in ("A0_v1_pinned", "A1_v1_retrained"):
        rep["paired"][f"V4 - {b}"] = {"ci95": paired_ci(arms[b], arms["V4"], 0.05),
                                      "bonferroni": paired_ci(arms[b], arms["V4"], 0.05 / max(M, 1))}
        fe = [r for r in arms[b] if r.get("origin") == "FE1"]
        fv = [r for r in arms["V4"] if r.get("origin") == "FE1"]
        rep["paired"][f"V4 - {b} (FE1)"] = {"ci95": paired_ci(fe, fv, 0.05)}
    A = {k: rep["arms"][k]["ALL"] for k in rep["arms"]}
    g1 = all(A["V4"]["micro_f1"] > A[b]["micro_f1"] and A["V4"]["family_macro_f1"] > A[b]["family_macro_f1"]
             for b in ("A0_v1_pinned", "A1_v1_retrained"))
    # G3: per family, mean over line scans of (FP+FN)(V4) - (FP+FN)(A0)
    def per_fam(recs):
        f = defaultdict(lambda: [0, 0])
        for r in recs:
            if r.get("disposition") == "SCORED_COMPLETE_LABEL":
                f[r["family_id"]][0] += r["false_positive"] + r["false_negative"]
                f[r["family_id"]][1] += 1
        return f
    fa, fv = per_fam(arms["A0_v1_pinned"]), per_fam(arms["V4"])
    fr = per_fam(arms["A1_v1_retrained"])
    worse = []
    fam_rows = []
    for k in sorted(set(fa) | set(fv)):
        da = (fv[k][0] - fa[k][0]) / max(fa[k][1], 1)
        dr = (fv[k][0] - fr[k][0]) / max(fr[k][1], 1)
        fam_rows.append((k, fa[k][0], fr[k][0], fv[k][0], fa[k][1], round(da, 3), round(dr, 3)))
        if da > 1.0:
            worse.append((k, fa[k][0], fv[k][0], fa[k][1], round(da, 3)))
    ctrl = {k: rep["arms"][k]["ALL"]["controls"] for k in rep["arms"]}
    lines = v4["lines"]
    stored = {k: json.loads((RES / f"lines_{k}.json").read_text()) for k in arms if k != "V4"}
    hems = {k: hem_recall(pool, stored[k]) for k in stored}
    hems["V4"] = hem_recall(pool, lines)
    per = {r["case_id"]: r for r in arms["V4"]}
    tabs = {k: {c: next((x for x in (arms[k] if k != "V4" else arms["V4"]) if x["case_id"] == c), {}) for c in
                TAB_SCANS + EAR_SCANS} for k in arms}
    rep.update(G1_lofo_better_than_A0_and_A1=g1, G3_families_worse_than_A0_by_more_than_1_line_per_scan=worse,
               G4_control_false_lines=ctrl, G6_hem=hems,
               tab_and_ear_scans={k: {c: (v.get("true_positive"), v.get("false_positive"), v.get("false_negative"))
                                      for c, v in d.items()} for k, d in tabs.items()},
               family_rows_fam_A0err_A1err_V4err_nscans_dA0_dA1=fam_rows)
    (RES / f"GATES_{a.variant}.json").write_text(json.dumps(rep, indent=1, default=str))
    for k in rep["arms"]:
        x = rep["arms"][k]["ALL"]; y = rep["arms"][k]["FE1"]
        print(f"{k:16s} {x['tp']}/{x['fp']}/{x['fn']} micro {x['micro_f1']} {x['micro_ci95']} macro "
              f"{x['family_macro_f1']} {x['macro_ci95']} ctrl {x['controls']['fp_lines']} | FE1 micro {y['micro_f1']} "
              f"macro {y['family_macro_f1']} | hem {hems[k]['matched']}/{hems[k]['hem_lines']}")
    for k, v in rep["paired"].items():
        print(k, json.dumps(v))
    print("G1", g1, "| G3 worse:", worse, "| M =", M)
    print("tabs/ears", json.dumps(rep["tab_and_ear_scans"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
