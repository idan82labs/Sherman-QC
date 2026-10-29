"""Auditor 3: constrained arc fits, tangent-relative gold position, 3D loci lines, frozen-matcher metrics."""
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from a3_measure import run, unit  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

SCR = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad"


def constrained_R(points, Xs, g, half_incl_rad, r_lo=0.3, r_hi=15.0):
    """Circle tangent to both skin lines meeting at Xs with bisector g: centre = Xs + g*R/sin(half)."""
    best = None
    for R in np.linspace(r_lo, r_hi, 1471):
        C = Xs + g * R / np.sin(half_incl_rad)
        res = np.abs(np.hypot(*(points - C).T) - R)
        cost = np.median(res)
        if best is None or cost < best[0]:
            best = (cost, R)
    return best[1], best[0]


def analyse(bid):
    F, res = run(bid)
    lo, hi = F.gold_s_range()
    rows = []
    for r in res:
        if not r.get("ok"):
            continue
        s0 = r["s"]
        m = np.abs(F.s - s0) <= 0.5
        xy, nxy = F.xy[m], F.nxy[m]
        Xo = np.array(r["Xo"]); g = np.array(r["g"]); acr = np.array(r["across"])
        uA = np.array(r["uA"]); uB = np.array(r["uB"])
        half = np.radians(r["included_deg"]) / 2
        rel = xy - Xo
        h = rel @ g
        a = rel @ acr
        nout = -(nxy @ g)
        cosA = np.abs(nxy @ F.a2); cosB = np.abs(nxy @ F.b2)
        flat = (cosA > np.cos(np.radians(6))) | (cosB > np.cos(np.radians(6)))
        # outer arc candidates: outward facing, not flat, near the corner
        arc_o = (~flat) & (nout > 0.2) & (np.hypot(a, h) < 12)
        Ro = cost_o = None
        if arc_o.sum() >= 8:
            Ro, cost_o = constrained_R(xy[arc_o], Xo, g, half)
        row = dict(s=s0, lam=r["gold_lam"], bend=r["bend_deg"], Ro=Ro, cost_o=cost_o, Xo=Xo, g=g, acr=acr, uA=uA, uB=uB,
                   apex=np.array(r["apex"]) if "apex" in r else None, gold=np.array(r["gold_xy"]),
                   th=r["thickness"], inner_apex=np.array(r["inner_apex"]) if "inner_apex" in r else None,
                   Xi=np.array(r["Xi"]) if "Xi" in r else None, nearest=r["gold_nearest_scan"], presA=r["presA"], presB=r["presB"],
                   n_arc_o=r["n_arc_outer"], n_arc_i=r["n_arc_inner"])
        # inner arc: use inner sharp if available
        if row["Xi"] is not None:
            arc_i = (~flat) & (nout < -0.2) & (np.hypot(*(xy - row["Xi"]).T) < 12)
            if arc_i.sum() >= 8:
                Ri, cost_i = constrained_R(xy[arc_i], row["Xi"], g, half, r_lo=0.2)
                row["Ri"], row["cost_i"] = Ri, cost_i
        rows.append(row)
    return F, rows, (lo, hi)


def to3(F, s, p2):
    return F.mid + s * F.d + p2[0] * F.e1 + p2[1] * F.e2


def fit3(points):
    P = np.asarray(points)
    c = P.mean(0)
    w, U = np.linalg.eigh(np.cov((P - c).T))
    d = U[:, -1]
    t = (P - c) @ d
    return c, d, t


if __name__ == "__main__":
    import scipy.ndimage as ndi
    summary = {}
    for bid in ("M01", "M02", "M03", "M04", "M05"):
        F, rows, (lo, hi) = analyse(bid)
        # fold extent: longest run of stations with both plates + arc support
        ss = np.array([r["s"] for r in rows])
        ex = np.array([(r["presA"] > 5 and r["presB"] > 5 and (r["n_arc_o"] + r["n_arc_i"]) > 3) for r in rows])
        lab, nlab = ndi.label(ex)
        runs = [(ss[lab == k].min(), ss[lab == k].max()) for k in range(1, nlab + 1)]
        main = max(runs, key=lambda r: r[1] - r[0])
        f_lo, f_hi = main[0] - 0.5, main[1] + 0.5
        in_g = [r for r in rows if lo <= r["s"] <= hi]
        in_f = [r for r in rows if f_lo <= r["s"] <= f_hi]
        Ro = np.array([r["Ro"] for r in in_g if r["Ro"] is not None])
        Ri = np.array([r.get("Ri") for r in in_g if r.get("Ri") is not None])
        th = np.array([v for r in in_g for v in r["th"].values()])
        # gold position relative to the inner tangent lines (along the plate the gold is on)
        tang = []
        for r in in_g:
            if r.get("Ri") is None or r["Xi"] is None:
                continue
            half = np.arccos(np.clip(r["uA"] @ r["uB"], -1, 1)) / 2
            tdist = r["Ri"] / np.tan(half)  # distance from inner sharp to inner tangent point along each plate
            G = r["gold"]
            alongA = (G - r["Xi"]) @ r["uA"]; alongB = (G - r["Xi"]) @ r["uB"]
            pl = "A" if alongA > alongB else "B"
            tang.append((pl, (alongA if pl == "A" else alongB) - tdist, tdist))
        # 3D loci lines over the fold extent
        apex3 = [to3(F, r["s"], r["apex"]) for r in in_f if r["apex"] is not None]
        Xo3 = [to3(F, r["s"], r["Xo"]) for r in in_f]
        ia3 = [to3(F, r["s"], r["inner_apex"]) for r in in_f if r["inner_apex"] is not None]
        lines = {}
        for name, pts in (("outer_apex", apex3), ("outer_sharp", Xo3), ("inner_apex", ia3)):
            if len(pts) < 5:
                continue
            c, d, t = fit3(pts)
            # extent = fold extent along d
            s_lo = (to3(F, f_lo, np.zeros(2)) - c) @ d
            s_hi = (to3(F, f_hi, np.zeros(2)) - c) @ d
            seg = np.array([c + min(s_lo, s_hi) * d, c + max(s_lo, s_hi) * d])
            resid = np.linalg.norm((np.asarray(pts) - c) - np.outer(t, d), axis=1)
            lines[name] = dict(seg=seg.tolist(), n=len(pts), resid_p95=float(np.percentile(resid, 95)))
        gold_seg = np.array([F.S, F.E])
        metrics = {}
        for name, L in lines.items():
            pm = pair_metrics(gold_seg, np.array(L["seg"]))
            metrics[name] = {k: round(float(vv), 3) for k, vv in pm.items()}
            metrics[name]["match"] = bool(is_match(pm, Thresholds()))
        cov_lo, cov_hi = max(lo, f_lo), min(hi, f_hi)
        coverage = max(0.0, cov_hi - cov_lo) / (f_hi - f_lo)
        overrun = (max(0, f_lo - lo) + max(0, hi - f_hi))
        ga = np.array([np.linalg.norm(r["gold"] - r["apex"]) for r in in_g if r["apex"] is not None])
        ac = np.array([(r["gold"] - r["apex"]) @ r["acr"] for r in in_g if r["apex"] is not None])
        al = np.array([(r["gold"] - r["apex"]) @ r["g"] for r in in_g if r["apex"] is not None])
        gi = np.array([np.linalg.norm(r["gold"] - r["inner_apex"]) for r in in_g if r["inner_apex"] is not None])
        gxo = np.array([np.linalg.norm(r["gold"] - r["Xo"]) for r in in_g])
        gxi = np.array([np.linalg.norm(r["gold"] - r["Xi"]) for r in in_g if r["Xi"] is not None])
        near = np.array([r["nearest"] for r in in_g])
        bend = np.array([r["bend"] for r in in_g])
        tg = np.array([t[1] for t in tang]) if tang else np.array([np.nan])
        tg_pl = sorted(set(t[0] for t in tang))
        summ = dict(
            L_gold=F.L, gold_s=[lo, hi], fold_s=[f_lo, f_hi], fold_len=f_hi - f_lo, fold_runs=[[float(a), float(b)] for a, b in runs],
            coverage_pct=100 * coverage, gold_overrun_mm=overrun,
            angle_gold_vs_axis_deg=float(np.degrees(np.arccos(min(1, abs(F.g3 @ F.d))))),
            bend_deg_med=float(np.median(bend)), bend_deg_range=[float(bend.min()), float(bend.max())],
            R_outer_med=float(np.median(Ro)) if len(Ro) else None, R_inner_med=float(np.median(Ri)) if len(Ri) else None,
            thickness_med=float(np.median(th)) if len(th) else None,
            gold_to_outer_apex=dict(med=float(np.median(ga)), min=float(ga.min()), max=float(ga.max()), n=len(ga)),
            across_med=float(np.median(ac)), along_bisector_med=float(np.median(al)),
            gold_to_inner_apex_med=float(np.median(gi)) if len(gi) else None,
            gold_to_outer_sharp_med=float(np.median(gxo)), gold_to_inner_sharp_med=float(np.median(gxi)) if len(gxi) else None,
            gold_nearest_scan_med=float(np.median(near)), gold_nearest_scan_p90=float(np.percentile(near, 90)),
            gold_plate_vs_inner_tangent=dict(plate=tg_pl, beyond_tangent_med=float(np.nanmedian(tg)),
                                             beyond_tangent_min=float(np.nanmin(tg)), beyond_tangent_max=float(np.nanmax(tg))),
            loci_lines=lines, matcher_vs_loci=metrics,
            plates=dict(A=F.nameA, B=F.nameB))
        summary[bid] = summ
        print(f"\n=== {bid} ({F.nameB}) ===")
        for k in ("L_gold", "fold_len", "coverage_pct", "gold_overrun_mm", "angle_gold_vs_axis_deg", "bend_deg_med", "bend_deg_range",
                  "R_outer_med", "R_inner_med", "thickness_med", "gold_to_outer_apex", "across_med", "along_bisector_med",
                  "gold_to_inner_apex_med", "gold_to_outer_sharp_med", "gold_to_inner_sharp_med", "gold_nearest_scan_med",
                  "gold_nearest_scan_p90", "gold_plate_vs_inner_tangent", "fold_runs"):
            vv = summ[k]
            if isinstance(vv, float):
                vv = round(vv, 2)
            elif isinstance(vv, dict):
                vv = {kk: (round(x, 2) if isinstance(x, float) else x) for kk, x in vv.items()}
            print(f"  {k}: {vv}")
        for name, mm in metrics.items():
            print(f"  matcher gold vs {name}: {mm}")
    json.dump(summary, open(f"{SCR}/a3_loci.json", "w"), indent=1, default=float)
