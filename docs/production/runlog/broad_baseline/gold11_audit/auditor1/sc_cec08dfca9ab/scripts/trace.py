"""Auditor-1 (sc_cec08dfca9ab): per gold line -> apex line fit, fold extent trace, offset stats, coverage."""
import json, math, sys
import numpy as np
from corner import *

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # frozen metric, read-only use


def runs(flags, ss, maxgap=6.0):
    out = []; cur = None
    for f, s in zip(flags, ss):
        if f:
            if cur is None:
                cur = [s, s]
            elif s - cur[1] <= maxgap:
                cur[1] = s
            else:
                out.append(cur); cur = [s, s]
    if cur is not None:
        out.append(cur)
    return out


results = {}
names = sys.argv[1:] or list(gp)
for name in names:
    a, b = gp[name]
    L = float(np.linalg.norm(b - a)); u = (b - a) / L
    dA3, dB3 = PLATES[name]
    # pass 1: corners near gold
    pts, offs = [], []
    for s in np.arange(0, L + 1e-9, 2.0):
        q = a + s * u
        r = corner_at(q, u, dA3, dB3, expect=np.zeros(2), max_from_expect=9.0)
        if r.get("status") == "corner":
            g = -r["apex"]
            pts.append(r["apex3"]); offs.append((s, float(g @ r["bis"]), float(g @ r["perp"])))
    pts = np.array(pts); offs = np.array(offs)
    med = np.median(offs[:, 1]); meda = np.median(offs[:, 2])
    keep = (np.abs(offs[:, 1] - med) < 0.8) & (np.abs(offs[:, 2] - meda) < 0.8)
    for _it in range(5):                      # robust line fit: iterate, drop apex points > 0.5 mm off the line
        A = pts[keep]
        ca = A.mean(0)
        _, _, vt = np.linalg.svd(A - ca)
        ua = vt[0] * np.sign(vt[0] @ u)
        rall = np.linalg.norm((pts - ca) - np.outer((pts - ca) @ ua, ua), axis=1)
        newkeep = keep & (rall < 0.5)
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    A = pts[keep]
    resid = np.linalg.norm((A - ca) - np.outer((A - ca) @ ua, ua), axis=1)
    # pass 2: trace fold along apex line
    proj = (A - ca) @ ua
    s0, s1 = proj.min() - 90, proj.max() + 90
    ss = np.arange(s0, s1 + 1e-9, 1.0)
    flags, rows = [], []
    for s in ss:
        q = ca + s * ua
        r = corner_at(q, ua, dA3, dB3, expect=np.zeros(2), max_from_expect=3.0)
        ok = r.get("status") == "corner" and np.linalg.norm(r["apex"]) < 2.0
        flags.append(ok)
        rows.append(dict(s=float(s), status=r.get("status"), apex_dev=float(np.linalg.norm(r["apex"])) if r.get("status") == "corner" else None,
                         nA=r.get("nA"), nB=r.get("nB"),
                         extA=r.get("extA"), extB=r.get("extB"), thick=r.get("thick")))
    rr = runs(flags, ss)
    main = max(rr, key=lambda x: x[1] - x[0]) if rr else None
    # gold interval on apex line
    g0, g1 = sorted([float((a - ca) @ ua), float((b - ca) @ ua)])
    res = dict(length_gold=L, apex_line_point=ca.tolist(), apex_line_dir=ua.tolist(),
               apex_fit_resid_rms=float(np.sqrt(np.mean(resid ** 2))), n_apex=int(len(A)),
               angle_gold_vs_apex_deg=float(np.degrees(np.arccos(min(1.0, abs(u @ ua))))),
               gold_interval=[g0, g1], fold_runs=rr, fold_main=main,
               offset_bis_median=float(np.median(offs[keep, 1])), offset_bis_min=float(offs[keep, 1].min()),
               offset_bis_max=float(offs[keep, 1].max()), offset_across_median=float(np.median(offs[keep, 2])),
               offset_across_min=float(offs[keep, 2].min()), offset_across_max=float(offs[keep, 2].max()),
               offset_total_median=float(np.median(np.hypot(offs[keep, 1], offs[keep, 2]))),
               n_offset_stations=int(keep.sum()), n_outlier_stations=int((~keep).sum()))
    if rr:
        tot = sum(r1 - r0 for r0, r1 in rr)
        f0, f1 = min(r[0] for r in rr), max(r[1] for r in rr)
        ov_runs = sum(max(0.0, min(g1, r1) - max(g0, r0)) for r0, r1 in rr)
        res.update(fold_span=[f0, f1], fold_len_union=tot, fold_len_span=f1 - f0,
                   coverage_of_fold_pct=100 * ov_runs / tot if tot else None,
                   gold_on_fold_pct=100 * ov_runs / (g1 - g0),
                   gold_overshoot_before=max(0.0, f0 - g0), gold_overshoot_after=max(0.0, g1 - f1),
                   fold_uncovered_before=max(0.0, g0 - f0), fold_uncovered_after=max(0.0, f1 - g1))
        # synthetic apex-located prediction over the fold span -> frozen pair metric vs gold
        pred = np.array([ca + f0 * ua, ca + f1 * ua])
        gold = np.array([a, b])
        res["frozen_pair_metrics_vs_apex_fold_line"] = pair_metrics(gold, pred)
        # same, but gold translated onto apex line (what a pure locus fix would score)
        res["apex_fold_line_raw"] = [from_part(pred[0], R, c).tolist(), from_part(pred[1], R, c).tolist()]
    res["trace_rows"] = rows
    results[name] = res
    print(f"== {name}: gold len {L:.1f}; apex pts {len(A)} (outliers {int((~keep).sum())}), apex-line resid rms {res['apex_fit_resid_rms']:.2f} mm,"
          f" angle gold-vs-apex {res['angle_gold_vs_apex_deg']:.2f} deg")
    print(f"   offset bisector (concave +): median {res['offset_bis_median']:.2f} [{res['offset_bis_min']:.2f},{res['offset_bis_max']:.2f}]"
          f"  across median {res['offset_across_median']:.2f} [{res['offset_across_min']:.2f},{res['offset_across_max']:.2f}]  |d| median {res['offset_total_median']:.2f}")
    print(f"   gold interval on apex line [{g0:.1f},{g1:.1f}]  fold runs {[[round(x,1) for x in r] for r in rr]}")
    if rr:
        print(f"   fold span [{res['fold_span'][0]:.1f},{res['fold_span'][1]:.1f}] union {res['fold_len_union']:.1f}  coverage_of_fold {res['coverage_of_fold_pct']:.1f}%  gold_on_fold {res['gold_on_fold_pct']:.1f}%"
              f"  overshoot before/after {res['gold_overshoot_before']:.1f}/{res['gold_overshoot_after']:.1f}  uncovered before/after {res['fold_uncovered_before']:.1f}/{res['fold_uncovered_after']:.1f}")
        pm = res["frozen_pair_metrics_vs_apex_fold_line"]
        print(f"   frozen metric vs apex fold line: angle {pm['axis_angle_deg']:.2f} transverse {pm['transverse_offset_mm']:.2f} gold_ov {pm['gold_span_overlap']:.2f} pred_ov {pm['prediction_span_overlap']:.2f}")


def conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


tag = "_".join(names) if sys.argv[1:] else "all"
json.dump(results, open(W + f"trace_{tag}.json", "w"), default=conv)
