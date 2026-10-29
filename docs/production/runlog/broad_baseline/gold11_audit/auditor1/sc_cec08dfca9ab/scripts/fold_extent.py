"""Auditor-1 (sc_cec08dfca9ab): robust apex line per gold line from quality-checked corners, then fold extent
trace along the apex line with arc-continuity + plate-reach checks; offsets of gold vs apex at every fold station."""
import json, sys
import numpy as np
from corner import *
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics

def runs(flags, ss, maxgap=4.0):
    out = []; cur = None
    for f, s in zip(flags, ss):
        if f:
            if cur is None: cur = [s, s]
            elif s - cur[1] <= maxgap: cur[1] = s
            else: out.append(cur); cur = [s, s]
    if cur is not None: out.append(cur)
    return out

res_all = {}
for name in (sys.argv[1:] or list(gp)):
    a, b = gp[name]; L = float(np.linalg.norm(b - a)); u = (b - a) / L
    dA3, dB3 = PLATES[name]
    pts = []
    for s in np.arange(0, L + 1e-9, 2.0):
        r = corner_at(a + s * u, u, dA3, dB3, expect=np.zeros(2), max_from_expect=9.0)
        ok, qd = fold_quality(r)
        if ok:
            pts.append(r["apex3"])
    pts = np.array(pts)
    keep = np.ones(len(pts), bool)
    for _ in range(8):
        A = pts[keep]; ca = A.mean(0)
        _, _, vt = np.linalg.svd(A - ca); ua = vt[0] * np.sign(vt[0] @ u)
        rall = np.linalg.norm((pts - ca) - np.outer((pts - ca) @ ua, ua), axis=1)
        thr = max(0.4, 3 * np.median(rall[keep]))
        nk = rall < thr
        if (nk == keep).all(): break
        keep = nk
    A = pts[keep]
    # trace along apex line
    pr = (A - ca) @ ua
    ss = np.arange(pr.min() - 100, pr.max() + 100 + 1e-9, 1.0)
    rows, flags = [], []
    for s in ss:
        q = ca + s * ua
        r = corner_at(q, ua, dA3, dB3, expect=np.zeros(2), max_from_expect=3.0)
        ok, qd = fold_quality(r)
        ok = ok and np.linalg.norm(r["apex"]) < 1.5
        row = dict(s=float(s), ok=bool(ok), status=r.get("status"), **{k: float(v) for k, v in qd.items()})
        if r.get("status") == "corner":
            row["apex_dev"] = float(np.linalg.norm(r["apex"]))
            # gold point in this section: intersection of gold line with plane through q normal ua
            tg = ((q - a) @ ua) / (u @ ua)
            gq = a + tg * u
            g2 = np.array([(gq - q) @ r["e1"], (gq - q) @ r["e2"]])
            d = g2 - r["apex"]
            row.update(gold_t=float(tg / L), gold_bis=float(d @ r["bis"]), gold_across=float(d @ r["perp"]),
                       gold_dist=float(np.linalg.norm(d)), part_xyz=(q).tolist(), apex_part=r["apex3"].tolist(),
                       inner_sharp_bis=float((r["inner_sharp"] - r["X"]) @ r["bis"]) if r["inner_sharp"] is not None else None,
                       apex_depth=float(r["apex_depth"]), thick=[float(t) for t in r["thick"]])
        rows.append(row); flags.append(ok)
    rr = runs(flags, ss)
    g0, g1 = sorted([float((a - ca) @ ua), float((b - ca) @ ua)])
    tot = sum(r1 - r0 for r0, r1 in rr)
    ov = sum(max(0.0, min(g1, r1) - max(g0, r0)) for r0, r1 in rr)
    fr = [r for r in rows if r["ok"] and g0 <= r["s"] <= g1]
    gb = np.array([r["gold_bis"] for r in fr]); ga = np.array([r["gold_across"] for r in fr]); gd = np.array([r["gold_dist"] for r in fr])
    main = max(rr, key=lambda x: x[1] - x[0])
    pred = np.array([ca + main[0] * ua, ca + main[1] * ua])
    pm = pair_metrics(np.array([a, b]), pred)
    # y/x (part frame) coordinates of run ends for readability
    run_xyz = [[(ca + r0 * ua).round(1).tolist(), (ca + r1 * ua).round(1).tolist()] for r0, r1 in rr]
    res = dict(gold_len=L, apex_line_point_part=ca.tolist(), apex_line_dir_part=ua.tolist(), n_apex_fit=int(keep.sum()),
               apex_fit_rms=float(np.sqrt(np.mean(rall[keep] ** 2))),
               angle_gold_apex_deg=float(np.degrees(np.arccos(min(1, abs(u @ ua))))),
               gold_interval=[g0, g1], fold_runs=rr, fold_runs_part_xyz=run_xyz, fold_total=tot,
               coverage_of_fold_pct=100 * ov / tot, gold_on_fold_pct=100 * ov / (g1 - g0),
               offset_bis=dict(median=float(np.median(gb)), min=float(gb.min()), max=float(gb.max())),
               offset_across=dict(median=float(np.median(ga)), min=float(ga.min()), max=float(ga.max())),
               offset_dist=dict(median=float(np.median(gd)), min=float(gd.min()), max=float(gd.max())),
               n_fold_stations_in_gold=len(fr), frozen_metrics_vs_main_fold_apex_line=pm,
               main_fold_apex_line_raw=[from_part(pred[0], R, c).tolist(), from_part(pred[1], R, c).tolist()],
               R_outer_median=float(np.median([r["R_outer"] for r in rows if r["ok"]])),
               thick_median=float(np.median([t for r in rows if r["ok"] for t in r["thick"] if t > 0.5])) if any(r["ok"] for r in rows) else None,
               inner_sharp_bis_median=float(np.nanmedian([r["inner_sharp_bis"] if r.get("inner_sharp_bis") is not None else np.nan for r in rows if r["ok"]])),
               apex_depth_median=float(np.median([r["apex_depth"] for r in rows if r["ok"]])),
               rows=rows)
    res_all[name] = res
    print(f"== {name} gold {L:.1f} mm | apex fit n={keep.sum()} rms={res['apex_fit_rms']:.2f} | angle {res['angle_gold_apex_deg']:.2f} deg | R_o~{res['R_outer_median']:.2f} t~{res['thick_median']:.2f} apex_depth {res['apex_depth_median']:.2f} inner_sharp_bis {res['inner_sharp_bis_median']:.2f}")
    print(f"   gold [{g0:.1f},{g1:.1f}] fold runs {[[round(x,1) for x in r] for r in rr]} total {tot:.0f}")
    print(f"   run ends (part xyz): {run_xyz}")
    print(f"   coverage_of_fold {res['coverage_of_fold_pct']:.1f}%  gold_on_fold {res['gold_on_fold_pct']:.1f}%")
    print(f"   offset bis med {res['offset_bis']['median']:.2f} [{res['offset_bis']['min']:.2f},{res['offset_bis']['max']:.2f}] across med {res['offset_across']['median']:.2f} [{res['offset_across']['min']:.2f},{res['offset_across']['max']:.2f}] dist med {res['offset_dist']['median']:.2f} [{res['offset_dist']['min']:.2f},{res['offset_dist']['max']:.2f}] (n={len(fr)})")
    print(f"   frozen pair metric gold vs apex main-fold line: angle {pm['axis_angle_deg']:.2f} transverse {pm['transverse_offset_mm']:.2f} gold_ov {pm['gold_span_overlap']:.2f} pred_ov {pm['prediction_span_overlap']:.2f}")
tag = "_".join(sys.argv[1:]) if sys.argv[1:] else "all"
json.dump(res_all, open(W + f"fold_extent_{tag}.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
