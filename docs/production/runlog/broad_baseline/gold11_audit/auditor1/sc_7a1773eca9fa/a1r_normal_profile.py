#!/usr/bin/env python3
"""Auditor-1 (resumed run): skin-normal angle across each gold bend, pooled over 20-80 % of the span.

Light, read-only.  For every scan point within 0.5 mm of 13 section planes (perpendicular to the true
axis d) the in-section normal angle phi = atan2(N.h, |N.g|) is plotted against the point's position
along its arm (sideways coordinate x, + toward plate P).  A single circular bend shows phi ramping from
0 at the apex to the arm value (90 - included/2) at the tangent and then a flat plateau; an extra
formed fold next to it would show a second ramp + plateau.  Also prints the sideways distance at
which each skin reaches within 1 deg of its far-field arm angle.  Writes a1r_normal_profile.png/.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply")
LABELS = Path("/Users/idantal/ShermanQC/FullProject/output/bend_label_10_49001000/bend_instance_labels.json")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def main():
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(SCAN))
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    tree = cKDTree(V)
    labels = json.loads(LABELS.read_text())
    rep = json.loads((HERE / "audit_sections.json").read_text())
    by = {r["bend_id"]: r for r in rep["lines"]}
    fig, axes = plt.subplots(5, 1, figsize=(14, 24), dpi=80)
    out = {}
    for ax, b in zip(axes, labels["manual_bends"]):
        bid = b["bend_id"]
        s = np.asarray(b["line_start"], float); e = np.asarray(b["line_end"], float)
        L = float(np.linalg.norm(e - s)); dg = (e - s) / L
        f = by[bid]["frame"]
        d = unit(f["d"]); g = unit(f["g"]); g = unit(g - d * (g @ d))
        h0 = unit(f["h"]); h = unit(np.cross(d, g)); h = h if h @ h0 > 0 else -h
        inc = by[bid]["included_deg"]
        arm = 90.0 - inc / 2.0
        # apex reference per station from the outer skin (normal within 12 deg of -g)
        XS, PH, SK = [], [], []
        for q in np.linspace(0.2, 0.8, 13):
            G = s + q * L * dg
            ii = np.array(tree.query_ball_point(G, 45.0))
            Q = V[ii]; NQ = N[ii]
            mm = np.abs((Q - G) @ d) < 0.5
            Q, NQ = Q[mm], NQ[mm]
            x = (Q - G) @ h; y = (Q - G) @ g
            ng = NQ @ g; nh = NQ @ h
            outer = ng < -0.05; inner = ng > 0.05
            ao = outer & (np.hypot(x, y) < 15) & (ng < -np.cos(np.radians(12)))
            if ao.sum() < 3:
                continue
            x0 = np.median(x[ao])
            phi = np.degrees(np.arctan2(nh, np.abs(ng)))
            # outer skin normal on the P arm (x>x0) points away from the V: phi sign convention ->
            # make phi positive on both arms: outer skin P arm has nh<0? use |phi| with arm sign from x
            XS.append(x - x0); PH.append(np.abs(phi)); SK.append(np.where(outer, 1, np.where(inner, 2, 0)))
        X = np.concatenate(XS); P_ = np.concatenate(PH); S = np.concatenate(SK)
        res = {"included_deg": inc, "arm_angle_deg": arm}
        for sk, col, nm in ((1, "#d95f02", "outer"), (2, "#1b9e77", "inner")):
            mm = S == sk
            ax.scatter(X[mm], P_[mm], s=1, c=col, lw=0, alpha=0.5, label=f"{nm} skin")
            # binned median and the distance where |phi| first reaches within 1 deg of the arm angle
            bins = np.arange(-30, 30.01, 0.5)
            med = []
            for lo in bins[:-1]:
                w = mm & (X >= lo) & (X < lo + 0.5)
                med.append(float(np.median(P_[w])) if w.sum() >= 5 else np.nan)
            med = np.asarray(med)
            cx = bins[:-1] + 0.25
            ax.plot(cx, med, "-", color=col, lw=1.6)
            reach = {}
            for side, sgn in (("P", 1), ("Q", -1)):
                sel = (np.sign(cx) == sgn) & np.isfinite(med)
                xs_ = np.abs(cx[sel]); ms_ = med[sel]
                o = np.argsort(xs_); xs_, ms_ = xs_[o], ms_[o]
                hit = np.where(np.abs(ms_ - arm) < 1.0)[0]
                reach[side] = float(xs_[hit[0]]) if len(hit) else None
                # plateau test: bins between 1 and 15 mm whose median sits 3..12 deg below the arm angle
                band = (xs_ > 1) & (xs_ < 20) & (ms_ > arm - 12) & (ms_ < arm - 3)
                reach[side + "_bins_3to12deg_short_of_arm_mm"] = float(0.5 * band.sum())
            res[nm] = {"reach_arm_within_1deg_mm": reach, "binned_median": [[float(a), (None if not np.isfinite(b_) else float(b_))] for a, b_ in zip(cx, med)]}
        ax.axhline(arm, color="k", ls=":", lw=0.8)
        ax.axhline(arm - 6, color="0.5", ls=":", lw=0.6)
        ax.set_xlim(-30, 30); ax.set_ylim(-2, max(arm + 15, 30))
        ax.set_xlabel("sideways position from the outer apex (mm, + toward base plate P)")
        ax.set_ylabel("|normal angle from bisector| (deg)")
        ax.set_title(f"{bid}: included {inc:.1f} deg, arm angle {arm:.1f} deg (dotted); grey dotted = arm-6 deg."
                     f"  outer reaches arm at P {res['outer']['reach_arm_within_1deg_mm']['P']} / Q {res['outer']['reach_arm_within_1deg_mm']['Q']} mm", fontsize=9)
        ax.legend(fontsize=8, markerscale=6)
        ax.grid(alpha=0.25)
        out[bid] = res
        print(bid, "arm", round(arm, 1), "outer", res["outer"]["reach_arm_within_1deg_mm"], "inner", res["inner"]["reach_arm_within_1deg_mm"])
    fig.tight_layout()
    fig.savefig(HERE / "a1r_normal_profile.png")
    (HERE / "a1r_normal_profile.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
