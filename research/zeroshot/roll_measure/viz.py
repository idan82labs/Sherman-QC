"""Per-scan roll figure: scan with roll axis + run-out lines (end and side views), radius along the length,
and one cross-section with the fitted circle. ~1200 px wide PNG.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.viz RESULT.json RESULT.pkl OUT.png [TITLE]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.roll_measure import geom as G  # noqa: E402

COL = ["#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057", "#8d96a3"]


def render(res: dict, pk: dict, out: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regs = [r for r in res["regions"] if "measure" in r]
    cloud = pk["cloud"]
    fig = plt.figure(figsize=(12, 8.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.32, wspace=0.28)
    axE = fig.add_subplot(gs[0, 0])
    axS = fig.add_subplot(gs[0, 1:])
    axR = fig.add_subplot(gs[1, :2])
    axC = fig.add_subplot(gs[1, 2])
    if not regs:
        axE.text(0.5, 0.5, "no rolled region", ha="center")
        fig.suptitle(title)
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return
    m0 = regs[0]["measure"]
    d = np.array(m0["axis"]["direction"])
    p0 = np.array(m0["axis"]["point_mid"])
    f1, f2 = G.frame(d)
    # end view (looking down the axis of region 0)
    X = cloud - p0
    axE.scatter(X @ f1, X @ f2, s=0.15, c="#9aa0a6", alpha=0.35, linewidths=0, rasterized=True)
    # side view: axis horizontal, vertical = direction to the arc middle of region 0
    pdi = pk["regions"][0]
    S0 = pdi["stations"]
    mid = S0[len(S0) // 2]
    a_mid = mid.get("inner") or mid.get("outer")
    ang_mid = a_mid["start"] + 0.5 * a_mid["sweep"]
    u0 = np.asarray(pdi["u"])
    e1, e2 = G.frame(u0)
    vdir = G.unit(math.cos(ang_mid) * e1 + math.sin(ang_mid) * e2)
    vdir = G.unit(vdir - (vdir @ d) * d)
    wdir = np.cross(d, vdir)
    axS.scatter(X @ d, X @ wdir, s=0.15, c="#9aa0a6", alpha=0.35, linewidths=0, rasterized=True)
    for k, r in enumerate(regs):
        m = r["measure"]
        c = COL[k % len(COL)]
        dk, pk_ = np.array(m["axis"]["direction"]), np.array(m["axis"]["point_mid"])
        L = m["length_mm"]
        A = np.array([pk_ - dk * L / 2, pk_ + dk * L / 2]) - p0
        axE.plot(A @ f1, A @ f2, "+", color=c, ms=14, mew=2)
        axS.plot(A @ d, A @ wdir, "-", color=c, lw=1.6, label=f"roll {k} axis")
        for j, key in enumerate(("start", "end")):
            ro = m["runout_lines"][key]
            Lr = np.array([ro["start"], ro["end"]]) - p0
            ls = "--" if ro["type_majority"] == "tangent" else ":"
            axS.plot(Lr @ d, Lr @ wdir, ls, color=c, lw=2.2, label=f"run-out {key} ({ro['type_majority']})" if k == 0 else None)
            axE.plot(Lr @ f1, Lr @ f2, "o", color=c, ms=6, mfc="white" if j else c)
        pd = pk["regions"][k]
        for s in pd["stations"][:: max(1, len(pd["stations"]) // 12)]:
            for sk in ("outer", "inner"):
                if sk in s:
                    a = s[sk]
                    uu = np.asarray(pd["u"])
                    ee1, ee2 = G.frame(uu)
                    P3 = a["pts"][:, :1] * ee1 + a["pts"][:, 1:2] * ee2 + s["level"] * uu - p0
                    axS.plot(P3 @ d, P3 @ wdir, ".", color=c, ms=0.6, alpha=0.6)
    axE.set_aspect("equal")
    axE.set_title("end view along roll axis (+ axis, o run-outs)", fontsize=9)
    axS.set_aspect("equal")
    axS.set_title("side view: axis (solid), run-out lines (dashed = tangent to flat, dotted = edge/other)", fontsize=9)
    axS.legend(fontsize=7, loc="best", markerscale=4)
    # radius along the length
    for k, r in enumerate(regs):
        m = r["measure"]
        c = COL[k % len(COL)]
        ps = m["per_station"]
        lv = np.array([q["level"] for q in ps])
        lv = lv - lv.mean()
        ri = np.array([np.nan if q["R_in"] is None else q["R_in"] for q in ps])
        ro = np.array([np.nan if q["R_out"] is None else q["R_out"] for q in ps])
        mi = np.array([q["R_in_measured"] for q in ps])
        mo = np.array([q["R_out_measured"] for q in ps])
        axR.plot(lv[mi], ri[mi], ".", color=c, ms=4, label=f"roll {k} R_in (measured)")
        axR.plot(lv[mo], ro[mo], "x", color=c, ms=3, alpha=0.7, label=f"roll {k} R_out (measured)")
        if (~mi).any():
            axR.plot(lv[~mi], ri[~mi], ".", color=c, ms=3, alpha=0.3, label=f"roll {k} R_in = R_out - t")
        rc = np.array([np.nan if q.get("R_in_common") is None else q["R_in_common"] for q in ps])
        if np.isfinite(rc).sum() >= 3:
            w = m["R_inner_common_window"]
            axR.plot(lv, rc, "-", color="k", lw=0.9, alpha=0.8,
                     label=f"roll {k} R_in on common {w['width_deg']:.0f}° window")
        tp = m["taper_mm_per_m"]
        if tp:
            x = np.linspace(lv.min(), lv.max(), 2)
            base = rc if (m.get("taper_basis") == "common angular window") else ri
            axR.plot(x, np.nanmean(base) + tp["value"] / 1000 * x, "--", color=c, lw=0.8)
    axR.set_xlabel("station along axis (mm, centred)")
    axR.set_ylabel("radius (mm)")
    m = regs[0]["measure"]
    axR.set_title(f"radius per station  |  R_in {m['R_inner']['mean']:.2f} ± {m['R_inner']['u_boot']:.2f}, "
                  f"R_out {m['R_outer']['mean']:.2f}, taper {m['taper_mm_per_m']['value']:.2f} ± "
                  f"{m['taper_mm_per_m']['u_boot']:.2f} mm/m, sweep {m['sweep_deg']['mean']:.1f}°", fontsize=9)
    axR.legend(fontsize=7, ncol=2)
    axR.grid(alpha=0.3)
    # one cross-section (middle station of region 0) with the fitted circles
    for sk, c, lab in (("outer", "#d1495b", "outer skin"), ("inner", "#00798c", "inner skin")):
        if sk in mid:
            a = mid[sk]
            P = a["pts"] - a_mid["c"]
            axC.plot(P[:, 0], P[:, 1], ".", ms=1.2, color=c, label=f"{lab} pts")
            phi = np.linspace(a["start"] - 0.15, a["start"] + a["sweep"] + 0.15, 300)
            cc = a["c"] - a_mid["c"]
            axC.plot(cc[0] + a["R"] * np.cos(phi), cc[1] + a["R"] * np.sin(phi), "-", color="k", lw=0.6)
            for e, ph in zip(a["ends"], (a["start"], a["start"] + a["sweep"])):
                q = cc + a["R"] * np.array([math.cos(ph), math.sin(ph)])
                axC.plot(q[0], q[1], "s", color=c, ms=5)
            axC.plot([cc[0]], [cc[1]], "k+")
    axC.set_aspect("equal")
    axC.set_title(f"section @ station {mid['level']:.0f} mm: fitted circle (black), run-outs (squares)", fontsize=8)
    axC.legend(fontsize=7, markerscale=5, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    pk = pickle.loads(Path(sys.argv[2]).read_bytes())
    render(res, pk, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else res.get("name", ""))


if __name__ == "__main__":
    main()
