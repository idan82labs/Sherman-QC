"""Per-region overlay sheet for eye checks: the measured roll drawn on the scan (and on the CAD nominal when a
cad_roll_nominal.json is given).

Panels: (1) iso view: scan points, fitted roll surface as a wireframe, axis, run-out lines in bend-line style;
(2) end-on view along the axis: scan points, fitted inner/outer circles, CAD circles; (3-5) mesh sections at
the start / middle / end stations with the fitted arc and the run-out positions; (6) deviation of the scan
skin from the best-fit CYLINDER (constant R: shows taper + flat spots), (7) from the fitted CONE (taper
removed: flat spots / multi-radius only), (8) from the CAD nominal (when present). Run-out lines are drawn
on the heat maps too, so one can see whether they sit where the deviation from the arc begins.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.overlay RESULT.json RESULT.pkl OUTDIR NAME [CAD_NOMINAL.json]
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

RO_COL = {"start": "#e4572e", "end": "#7b2cbf"}
CAD_COL = "#2a9d8f"
FIT_COL = "#1d3557"
SCAN_COL = "#9aa0a6"


def _model(m):
    """Cone model of the inner skin about the axis: R_in(s) = R0 + taper * s (s from point_mid)."""
    ps = m["per_station"]
    s = np.array([q["s_axis"] for q in ps])
    rin = np.array([np.nan if q["R_in"] is None else q["R_in"] for q in ps])
    rc = np.array([np.nan if q.get("R_in_common") is None else q["R_in_common"] for q in ps])
    base = rc if np.isfinite(rc).sum() >= max(4, 0.5 * len(ps)) else rin
    ok = np.isfinite(base)
    A = np.c_[s[ok], np.ones(ok.sum())]
    sl, R0 = np.linalg.lstsq(A, base[ok], rcond=None)[0]
    # full-arc offset (common window radius can differ from the full-arc one on a multi-radius profile)
    return float(R0), float(sl), s, np.array([q["psi_deg"] for q in ps])


def _iso(P, az=35.0, el=22.0):
    a, e = math.radians(az), math.radians(el)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    Q = P @ (Rx @ Rz).T
    return Q[:, 0], Q[:, 2]


def _ro_segments(m):
    out = []
    for key in ("start", "end"):
        for sg in m["runout_lines"][key].get("segments", [m["runout_lines"][key]]):
            out.append((key, sg))
    return out


def _to_axis(P, fr):
    X = np.asarray(P) - fr["p0"]
    return np.c_[X @ fr["f1"], X @ fr["f2"], X @ fr["d"]]


def render_region(res, pk, k, out_png, title, cad=None, cad_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = res["regions"][k]["measure"]
    ov = pk["regions"][k]["overlay"]
    fr = ov["frame"]
    pts = ov["pts"]
    t = m["thickness"]["t_used_mm"] or 0.0
    R0, taper, s_st, psi_st = _model(m)
    skin_main = -1 if m["R_inner_measured_stations"] >= m["R_outer_measured_stations"] else 1
    fig = plt.figure(figsize=(12, 15.5), dpi=100)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.9, 0.95], hspace=0.3, wspace=0.25)
    # ---------------------------------------------------------------- iso
    ax = fig.add_subplot(gs[0, :2])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts["s"]), min(len(pts["s"]), 40000), replace=False)
    P = np.c_[pts["rho"][sel] * np.cos(pts["phi"][sel]), pts["rho"][sel] * np.sin(pts["phi"][sel]), pts["s"][sel]]
    x, y = _iso(P)
    ax.scatter(x, y, s=0.2, c=SCAN_COL, alpha=0.45, linewidths=0, rasterized=True)
    # wireframe: station arcs (every ~n/10) + generators at 5 angles between the run-outs
    order = np.argsort(s_st)
    s_sorted, psi_sorted = s_st[order], psi_st[order]
    for i in order[:: max(1, len(order) // 10)]:
        R = R0 + taper * s_st[i] + (t if skin_main > 0 else 0)
        ph = np.radians(np.linspace(psi_st[i, 0], psi_st[i, 1], 80))
        W = np.c_[R * np.cos(ph), R * np.sin(ph), np.full(len(ph), s_st[i])]
        ax.plot(*_iso(W), "-", color=FIT_COL, lw=0.8)
    for f in np.linspace(0, 1, 6):
        ph = np.radians(psi_sorted[:, 0] + f * (psi_sorted[:, 1] - psi_sorted[:, 0]))
        R = R0 + taper * s_sorted + (t if skin_main > 0 else 0)
        W = np.c_[R * np.cos(ph), R * np.sin(ph), s_sorted]
        ax.plot(*_iso(W), "-", color=FIT_COL, lw=0.6, alpha=0.8)
    L = m["length_mm"]
    A = np.array([[0, 0, -L / 2 - 10], [0, 0, L / 2 + 10]])
    ax.plot(*_iso(A), "-.", color="k", lw=1.5, label="roll axis")
    seen = set()
    for key, sg in _ro_segments(m):
        Q = _to_axis([sg["start"], sg["end"]], fr)
        ax.plot(*_iso(Q), "-", color=RO_COL[key], lw=3.2, solid_capstyle="round",
                label=None if key in seen else f"run-out {key} ({m['runout_lines'][key].get('n_segments', 1)} segment(s))")
        seen.add(key)
        ax.plot(*_iso(Q), "o", color=RO_COL[key], ms=5)
    if cad is not None:
        for key, ln in cad.get("_runouts_axis", []):
            ax.plot(*_iso(ln), "--", color=CAD_COL, lw=2.2, label=f"CAD run-out {key}")
        for W in cad.get("_wire_axis", []):
            ax.plot(*_iso(W), "-", color=CAD_COL, lw=0.7, alpha=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("iso: scan (grey), fitted roll wireframe (navy), axis (dash-dot), run-out lines (thick)", fontsize=9)
    # ---------------------------------------------------------------- end-on
    ax = fig.add_subplot(gs[0, 2])
    xx, yy = P[:, 0], P[:, 1]
    ax.scatter(xx, yy, s=0.2, c=SCAN_COL, alpha=0.4, linewidths=0, rasterized=True)
    ph = np.radians(np.linspace(np.min(psi_st[:, 0]) - 10, np.max(psi_st[:, 1]) + 10, 300))
    for R, lab in ((R0, "fit R_in"), (R0 + t, "fit R_out")):
        ax.plot(R * np.cos(ph), R * np.sin(ph), "-", color=FIT_COL, lw=0.7, label=f"{lab} {R:.1f}")
    if cad is not None and cad.get("_circles_axis"):
        for (cx, cy, R, lab) in cad["_circles_axis"]:
            ax.plot(cx + R * np.cos(ph), cy + R * np.sin(ph), "--", color=CAD_COL, lw=0.9, label=f"CAD {lab} {R:.1f}")
    for key, sg in _ro_segments(m):
        Q = _to_axis([sg["start"], sg["end"]], fr)
        ax.plot(Q[:, 0], Q[:, 1], "o", color=RO_COL[key], ms=6)
    ax.plot([0], [0], "k+", ms=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="best")
    ax.set_title("end-on along the axis (+ = axis)", fontsize=9)
    # ---------------------------------------------------------------- sections
    for j, sec in enumerate(ov["sections"]):
        ax = fig.add_subplot(gs[1, j])
        p0e = np.array([fr["p0"] @ fr["f1"], fr["p0"] @ fr["f2"]])
        for ch in sec["chains"]:
            C = ch - p0e
            ax.plot(C[:, 0], C[:, 1], "-", color="#555", lw=0.6)
        i = int(np.argmin(np.abs(s_st - sec["s"])))
        Rin = R0 + taper * sec["s"]
        phs = np.radians(np.linspace(psi_st[i, 0], psi_st[i, 1], 200))
        ax.plot(Rin * np.cos(phs), Rin * np.sin(phs), "-", color=FIT_COL, lw=1.4, alpha=0.8, label="fit inner arc")
        ax.plot((Rin + t) * np.cos(phs), (Rin + t) * np.sin(phs), ":", color=FIT_COL, lw=1.2, label="fit outer arc")
        for key, ps_ in (("start", psi_st[i, 0]), ("end", psi_st[i, 1])):
            # the fitted run-out LINE at this station (thick); the raw station run-out (small open circle)
            a_line = None
            for sg in m["runout_lines"][key].get("segments", []):
                Q = _to_axis([sg["start"], sg["end"]], fr)
                z0, z1 = Q[0, 2], Q[1, 2]
                if min(z0, z1) - 1.0 <= sec["s"] <= max(z0, z1) + 1.0 and abs(z1 - z0) > 1e-6:
                    f = (sec["s"] - z0) / (z1 - z0)
                    pnt = Q[0] + f * (Q[1] - Q[0])
                    a_line = math.atan2(pnt[1], pnt[0])
                    break
            L0, L1 = Rin - 4 * t - 3, Rin + 5 * t + 3
            if a_line is not None:
                ax.plot([L0 * math.cos(a_line), L1 * math.cos(a_line)], [L0 * math.sin(a_line), L1 * math.sin(a_line)], "-",
                        color=RO_COL[key], lw=2.2, label=f"run-out line {key}")
            a = math.radians(ps_)
            ax.plot([Rin * math.cos(a)], [Rin * math.sin(a)], "o", mfc="none", color=RO_COL[key], ms=7,
                    label=f"station run-out {key}" + ("" if a_line is not None else " (no line here)"))
        if cad is not None and cad.get("_circles_axis"):
            for (cx, cy, R, lab) in cad["_circles_axis"]:
                pc = np.radians(np.linspace(cad.get("_psi", [psi_st[i, 0], psi_st[i, 1]])[0],
                                            cad.get("_psi", [psi_st[i, 0], psi_st[i, 1]])[1], 200))
                ax.plot(cx + R * np.cos(pc), cy + R * np.sin(pc), "--", color=CAD_COL, lw=1.0)
        ax.set_aspect("equal")
        ax.set_title(f"section s={sec['s']:.0f} mm ({['start', 'middle', 'end'][j]})", fontsize=9)
        if j == 0:
            ax.legend(fontsize=6, loc="best")
    # ---------------------------------------------------------------- heat maps
    skin = pts["skin"]
    msk = skin == skin_main
    s_, ph_, rho_ = pts["s"][msk], np.degrees(pts["phi"][msk]), pts["rho"][msk]
    ref_psi = float(np.median(psi_st.mean(1)))
    ph_ = (ph_ - ref_psi + 180) % 360 - 180 + ref_psi
    off = 0.0 if skin_main < 0 else t
    Rcyl = float(np.nanmean(R0 + taper * s_st)) + off
    maps = [("deviation from best-fit CYLINDER (R const)", rho_ - Rcyl),
            ("deviation from fitted CONE (taper removed)", rho_ - (R0 + taper * s_ + off))]
    if cad is not None and "_dev_fn" in cad:
        maps.append(("deviation from CAD nominal", cad["_dev_fn"](s_, np.radians(ph_), rho_, skin_main)))
    lo_ph, hi_ph = np.min(psi_st) - 12, np.max(psi_st) + 12
    inwin = (ph_ > lo_ph) & (ph_ < hi_ph)
    for j, (lab, dev) in enumerate(maps):
        ax = fig.add_subplot(gs[2, j])
        nb_s = max(10, int((s_.max() - s_.min()) / 3))
        nb_p = max(10, int((hi_ph - lo_ph) / 1.0))
        H, xe, ye = np.histogram2d(s_[inwin], ph_[inwin], bins=[nb_s, nb_p], weights=dev[inwin])
        N, _, _ = np.histogram2d(s_[inwin], ph_[inwin], bins=[xe, ye])
        Mv = np.where(N > 0, H / np.maximum(N, 1), np.nan)
        lim = max(0.3, float(np.nanpercentile(np.abs(Mv), 95)))
        im = ax.imshow(Mv.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest")
        ax.plot(s_st, psi_st[:, 0], ".", color=RO_COL["start"], ms=2.5)
        ax.plot(s_st, psi_st[:, 1], ".", color=RO_COL["end"], ms=2.5)
        ax.set_xlabel("s along axis (mm)")
        ax.set_ylabel("angle about axis (deg)")
        ax.set_title(lab + f"\n{'inner' if skin_main < 0 else 'outer'} skin, mm (+ = away from axis)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.1)
    import textwrap
    fig.suptitle(title + ("\n" + "\n".join(textwrap.wrap(cad_note, 120)) if cad_note else ""), fontsize=10)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    pk = pickle.loads(Path(sys.argv[2]).read_bytes())
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    name = sys.argv[4]
    cadp = sys.argv[5] if len(sys.argv) > 5 else None
    for k, r in enumerate(res["regions"]):
        if "measure" not in r:
            continue
        cad, note = None, "CAD: pending (no cad_roll_nominal.json)"
        if cadp:
            from research.zeroshot.roll_measure import compare as C
            cad, note = C.cad_overlay(json.loads(Path(cadp).read_text()), r["measure"])
        m = r["measure"]
        title = (f"{name} roll {k}: R_in {m['R_inner']['mean']:.2f} mm, sweep {m['sweep_deg']['mean']:.1f}°, "
                 f"L {m['length_mm']:.0f} mm, skins {m['skins']}")
        render_region(res, pk, k, str(out / f"{name}_roll{k}_overlay.png"), title, cad, note)


if __name__ == "__main__":
    main()
