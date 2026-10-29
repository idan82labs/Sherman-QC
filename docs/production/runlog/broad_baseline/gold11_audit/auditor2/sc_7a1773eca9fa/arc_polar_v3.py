#!/usr/bin/env python3
"""Auditor 2 - v3: per-station outer/inner arc circle fits in the v2 frame; gold in polar coordinates
about the fitted bend axis; ideal outer-apex line per bend and the frozen-matcher metrics gold vs it.

Polar convention: phi = angle of (gold - arc centre) measured from the convex bisector direction (-b);
+phi toward flange A, -phi toward flange B; the bend tangent lines are at phi = +-beta/2.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds, is_match  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from xsec_audit import GOLD  # noqa: E402


def circle_fit(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    R = float(np.sqrt(sol[2] + c @ c))
    res = np.linalg.norm(Q - c, axis=1) - R
    return c, R, float(np.sqrt(np.mean(res ** 2)))


def main():
    d = np.load(sys.argv[1])
    P, N = d["P"], d["N"]
    summ = json.loads((OUT / "v2_summary.json").read_text())
    report = {}
    for bid in GOLD:
        r = json.loads((OUT / f"v2_{bid}.json").read_text())
        Xo0, do, bis, ex = (np.array(r[k]) for k in ("Xo0", "do", "bis", "ex"))
        beta = math.radians(r["bend_deg"])
        a, b = (np.array(v) for v in GOLD[bid])
        s_a, s_b = r["s_gold"]
        f0, f1 = summ[bid]["fold_extent_s"]
        sP = (P - Xo0) @ do
        rows = []
        for st in r["stations"]:
            s = st["s"]
            if not (min(s_a, s_b) <= s <= max(s_a, s_b)) or not st.get("fold") or "apex_h" not in st:
                continue
            m = np.abs(sP - s) <= 0.5
            Q = P[m] - Xo0 - s * do
            q = np.c_[Q @ ex, Q @ bis]
            nb = N[m] @ bis
            nx = N[m] @ ex
            ang = np.degrees(np.arctan2(nx, -nb))       # normal direction angle from -b (outer normals)
            half = math.degrees(beta) / 2
            near = np.linalg.norm(q - np.array([0, st["apex_h"]]), axis=1) < 12
            outer_arc = near & (nb < 0) & (np.abs(ang) < half - 4)
            inner_ang = np.degrees(np.arctan2(-nx, nb))   # inner normals point toward +b
            inner_arc = near & (nb > 0) & (np.abs(inner_ang) < half - 4)
            row = {"s": s}
            if outer_arc.sum() >= 8:
                c, R, rms = circle_fit(q[outer_arc])
                row.update(Co=c.tolist(), Ro=R, rms_o=rms, n_o=int(outer_arc.sum()))
            if inner_arc.sum() >= 8:
                c2, R2, rms2 = circle_fit(q[inner_arc])
                row.update(Ci=c2.tolist(), Ri=R2, rms_i=rms2, n_i=int(inner_arc.sum()))
            lam = (s - s_a) / (s_b - s_a)
            G = a + lam * (b - a) - Xo0 - s * do
            g = np.array([G @ ex, G @ bis])
            row["gold"] = g.tolist()
            if "Co" in row:
                c = np.array(row["Co"])
                v = g - c
                row["r_gold_from_Co"] = float(np.linalg.norm(v))
                row["phi_gold_deg"] = float(np.degrees(np.arctan2(v[0], -v[1])))
            rows.append(row)
        if not rows:
            continue
        arr = lambda k: np.array([x[k] for x in rows if k in x])
        Ro, Ri = arr("Ro"), arr("Ri")
        rg, ph = arr("r_gold_from_Co"), arr("phi_gold_deg")
        rep = {
            "bend_deg": r["bend_deg"], "half_angle_deg": math.degrees(beta) / 2,
            "thickness": r["thickness"],
            "R_outer_median": float(np.median(Ro)) if len(Ro) else None,
            "R_inner_median": float(np.median(Ri)) if len(Ri) else None,
            "n_stations": len(rows), "n_inner_arc_fits": int(len(Ri)),
            "gold_r_from_axis_median": float(np.median(rg)) if len(rg) else None,
            "gold_r_p10_p90": [float(np.percentile(rg, 10)), float(np.percentile(rg, 90))] if len(rg) else None,
            "gold_phi_deg_median": float(np.median(ph)) if len(ph) else None,
            "gold_phi_p10_p90": [float(np.percentile(ph, 10)), float(np.percentile(ph, 90))] if len(ph) else None,
        }
        if rep["R_outer_median"] and rep["gold_r_from_axis_median"]:
            Rin = rep["R_outer_median"] - r["thickness"] if r["thickness"] < 3 else rep["R_outer_median"] - 2.03
            rep["expected_R_inner_from_outer_minus_t"] = Rin
            rep["gold_r_minus_Rinner"] = rep["gold_r_from_axis_median"] - Rin
            # arc length along the inner surface from the inner tangent line (positive = beyond, onto flange)
            phi = abs(rep["gold_phi_deg_median"])
            rep["gold_beyond_tangent_deg"] = phi - rep["half_angle_deg"]
        # ideal outer-apex line over the fold extent
        apex_h = float(np.median([st["apex_h"] for st in r["stations"]
                                  if st.get("fold") and "apex_h" in st and f0 <= st["s"] <= f1]))
        A0 = Xo0 + f0 * do + apex_h * bis
        A1 = Xo0 + f1 * do + apex_h * bis
        X0 = Xo0 + f0 * do
        X1 = Xo0 + f1 * do
        gold = np.array([a, b])
        mt = pair_metrics(gold, np.array([A0, A1]))
        ms = pair_metrics(gold, np.array([X0, X1]))
        rep["ideal_apex_line"] = {"start": A0.tolist(), "end": A1.tolist(), "apex_h_from_outer_sharp": apex_h,
                                  "length": float(f1 - f0)}
        rep["matcher_gold_vs_ideal_apex"] = {k: round(v, 3) for k, v in mt.items()}
        rep["matcher_gold_vs_ideal_apex"]["is_match"] = is_match(mt, Thresholds())
        rep["matcher_gold_vs_outer_sharp_line"] = {k: round(v, 3) for k, v in ms.items()}
        rep["matcher_gold_vs_outer_sharp_line"]["is_match"] = is_match(ms, Thresholds())
        report[bid] = rep
        print(bid, json.dumps(rep, indent=None)[:1400])
    (OUT / "v3_polar_matcher.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
