"""Auditor-1: recompute fold runs with strict apex deviation (<=0.4 mm from the fitted apex line) and lip-width sanity."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from a1frame import *
xyz, nrm, faces, R, c, G = load()
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}
d = json.load(open(W + "fold_extent_all.json"))

def runs(flags, ss, maxgap=4.0):
    out = []; cur = None
    for f, s in zip(flags, ss):
        if f:
            if cur is None: cur = [s, s]
            elif s - cur[1] <= maxgap: cur[1] = s
            else: out.append(cur); cur = [s, s]
    if cur is not None: out.append(cur)
    return out

summary = {}
for name, r in d.items():
    a, b = gp[name]; L = np.linalg.norm(b - a); u = (b - a) / L
    ca = np.array(r["apex_line_point_part"]); ua = np.array(r["apex_line_dir_part"])
    rows = r["rows"]
    ss = [x["s"] for x in rows]
    fl = [bool(x["ok"] and x.get("apex_dev", 9) <= 0.4) for x in rows]
    rr = runs(fl, ss)
    g0, g1 = r["gold_interval"]
    tot = sum(r1 - r0 for r0, r1 in rr)
    ov = sum(max(0.0, min(g1, r1) - max(g0, r0)) for r0, r1 in rr)
    fold_rows = [x for x, f in zip(rows, fl) if f and g0 <= x["s"] <= g1]
    gb = np.array([x["gold_bis"] for x in fold_rows]); ga = np.array([x["gold_across"] for x in fold_rows]); gd = np.array([x["gold_dist"] for x in fold_rows])
    main = max(rr, key=lambda x: x[1] - x[0])
    f0, f1 = min(x[0] for x in rr), max(x[1] for x in rr)
    pred = np.array([ca + f0 * ua, ca + f1 * ua])
    pm = pair_metrics(np.array([a, b]), pred)
    # per-endpoint offsets: nearest fold station to each gold end (within the fold)
    ends = {}
    for tag, tt in (("start", 0.0), ("end", 1.0)):
        cand = [x for x in fold_rows if "gold_t" in x]
        if cand:
            x = min(cand, key=lambda z: abs(z["gold_t"] - tt))
            ends[tag] = dict(gold_t=x["gold_t"], bis=x["gold_bis"], across=x["gold_across"], dist=x["gold_dist"])
    run_part = [[(ca + r0 * ua).round(1).tolist(), (ca + r1 * ua).round(1).tolist()] for r0, r1 in rr]
    gold_part = [a.round(1).tolist(), b.round(1).tolist()]
    summary[name] = dict(gold_len=float(L), fold_runs=rr, fold_runs_part=run_part, fold_total=float(tot),
                         gold_interval=[g0, g1], gold_part=gold_part,
                         coverage_of_fold_pct=100 * ov / tot, gold_on_fold_pct=100 * ov / (g1 - g0),
                         overshoot_before=max(0, f0 - g0), overshoot_after=max(0, g1 - f1),
                         uncovered_before=max(0, g0 - f0), uncovered_after=max(0, f1 - g1),
                         offset_bis=[float(np.median(gb)), float(gb.min()), float(gb.max())],
                         offset_across=[float(np.median(ga)), float(ga.min()), float(ga.max())],
                         offset_dist=[float(np.median(gd)), float(gd.min()), float(gd.max())],
                         offset_at_fold_ends=ends, n=len(fold_rows), angle_gold_apex_deg=r["angle_gold_apex_deg"],
                         frozen_metrics_vs_apex_fold_line=pm,
                         apex_fold_line_raw=[from_part(pred[0], R, c).tolist(), from_part(pred[1], R, c).tolist()],
                         R_outer_median=r["R_outer_median"], apex_depth_median=r["apex_depth_median"],
                         inner_sharp_bis_median=r["inner_sharp_bis_median"], thick_median=r["thick_median"])
    s = summary[name]
    print(f"== {name}: gold {L:.1f} mm, fold total {tot:.0f} mm, runs(part xyz) {run_part}")
    print(f"   gold endpoints (part) {gold_part}")
    print(f"   coverage_of_fold {s['coverage_of_fold_pct']:.1f}%  gold_on_fold {s['gold_on_fold_pct']:.1f}%  overshoot before/after {s['overshoot_before']:.1f}/{s['overshoot_after']:.1f}  uncovered before/after {s['uncovered_before']:.1f}/{s['uncovered_after']:.1f}")
    print(f"   offset bisector(concave+) med {s['offset_bis'][0]:.2f} [{s['offset_bis'][1]:.2f},{s['offset_bis'][2]:.2f}]  across med {s['offset_across'][0]:.2f} [{s['offset_across'][1]:.2f},{s['offset_across'][2]:.2f}]  dist med {s['offset_dist'][0]:.2f} [{s['offset_dist'][1]:.2f},{s['offset_dist'][2]:.2f}]  n={s['n']}")
    print(f"   ends: {json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in ends.items()})}")
    print(f"   frozen pair metric vs apex fold line: angle {pm['axis_angle_deg']:.2f} transverse {pm['transverse_offset_mm']:.2f} gold_ov {pm['gold_span_overlap']:.2f} pred_ov {pm['prediction_span_overlap']:.2f}")
json.dump(summary, open(W + "strict_summary.json", "w"), indent=1, default=float)
