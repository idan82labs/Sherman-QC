#!/usr/bin/env python3
"""Auditor-1 (resumed run) independent cross-check for sc_7a1773eca9fa (49001000).

Light, read-only on all inputs; writes only next to this script (a1r_* files).

Re-uses the bend frames (d = true axis, g = concave bisector, h = sideways toward plate P) already
computed by audit_sections.py, but does NOT re-use its plane-fit skin identification.  Instead each
section point is assigned to a skin by its own mesh normal:  outer (convex) skin <=> N.g < 0,
inner (concave) skin <=> N.g > 0  (valid on both arms and the arc for any bend < 180 deg, given
consistently outward mesh normals, which is checked on the flat arms first).

Per gold line, at 10/25/50/75/90 % of the gold span (section plane through the gold point, normal d):
  * outer apex  = median of outer-skin points whose normal is within 12 deg of -g (arc midline),
    cross-checked by the outer-skin extreme point along -g;
  * inner apex  = same with +g on the inner skin;
  * gold -> outer apex, gold -> inner apex, gold -> nearest outer-skin / inner-skin point;
  * surface-normal angle at the gold's nearest scan point (0 = apex direction, +-90-included/2 = arm).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
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


def load():
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(SCAN))
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    return V, N


def main():
    V, N = load()
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    tree = cKDTree(V)
    labels = json.loads(LABELS.read_text())
    rep = json.loads((HERE / "audit_sections.json").read_text())
    by = {r["bend_id"]: r for r in rep["lines"]}
    out = {"scan": str(SCAN), "method": "normal-sign skin split (N.g<0 outer)", "lines": {}}
    fig, axes = plt.subplots(5, 3, figsize=(15, 25), dpi=80)
    for row, b in enumerate(labels["manual_bends"]):
        bid = b["bend_id"]
        s = np.asarray(b["line_start"], float)
        e = np.asarray(b["line_end"], float)
        L = float(np.linalg.norm(e - s))
        dg = (e - s) / L
        f = by[bid]["frame"]
        d = unit(f["d"]); g = unit(f["g"]); h = unit(f["h"])
        nPi = unit(f["nP_in"]); nQi = unit(f["nQ_in"])
        included = by[bid]["included_deg"]
        # orthonormalise h against d,g (sections are perpendicular to the TRUE axis d)
        g = unit(g - d * (g @ d)); h = unit(np.cross(d, g)) * np.sign(np.cross(d, g) @ h)
        # normal-orientation sanity on the arms: outer skin of P has N ~ -nPi, inner N ~ +nPi
        rows = []
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            G = s + q * L * dg
            ii = np.array(tree.query_ball_point(G, 30.0))
            Q = V[ii]; NQ = N[ii]
            m = np.abs((Q - G) @ d) < 0.5
            Q, NQ = Q[m], NQ[m]
            x = (Q - G) @ h          # sideways, + toward plate P
            y = (Q - G) @ g          # bisector, + toward the concave side
            ng = NQ @ g
            nh = NQ @ h
            outer = ng < -0.05
            inner = ng > 0.05
            ang = np.degrees(np.arctan2(nh, np.abs(ng)))  # 0 = along the bisector
            # apexes: normals within 12 deg of -g (outer) / +g (inner), near the gold (within 15 mm)
            near = np.hypot(x, y) < 15.0
            ao = outer & near & (ng < -math.cos(math.radians(12)))
            ai = inner & near & (ng > math.cos(math.radians(12)))
            apex_o = np.array([np.median(x[ao]), np.median(y[ao])]) if ao.sum() >= 3 else None
            apex_i = np.array([np.median(x[ai]), np.median(y[ai])]) if ai.sum() >= 3 else None
            # extreme-point cross-check (lowest outer-skin point along the bisector, within 8 mm sideways)
            ext = None
            sel = outer & near & (np.abs(x - (apex_o[0] if apex_o is not None else 0.0)) < 8.0)
            if sel.sum() >= 3:
                k = np.argmin(y[sel])
                ext = np.array([x[sel][k], y[sel][k]])
            # nearest outer / inner skin points to the gold (gold at origin)
            r_all = np.hypot(x, y)
            d_out = float(r_all[outer].min()) if outer.any() else None
            d_in = float(r_all[inner].min()) if inner.any() else None
            k = int(np.argmin(r_all))
            near_skin = "outer" if outer[k] else ("inner" if inner[k] else "edge/rim")
            row_ = {
                "q": q,
                "apex_o_xy": None if apex_o is None else apex_o.tolist(),
                "apex_o_extreme_xy": None if ext is None else ext.tolist(),
                "apex_i_xy": None if apex_i is None else apex_i.tolist(),
                "gold_to_outer_apex_mm": None if apex_o is None else float(np.hypot(*apex_o)),
                "gold_to_outer_apex_extreme_mm": None if ext is None else float(np.hypot(*ext)),
                "gold_to_inner_apex_mm": None if apex_i is None else float(np.hypot(*apex_i)),
                "gold_to_nearest_outer_skin_mm": d_out,
                "gold_to_nearest_inner_skin_mm": d_in,
                "nearest_scan_mm": float(r_all[k]),
                "nearest_scan_skin": near_skin,
                "nearest_scan_normal_angle_from_bisector_deg": float(ang[k]),
                "n_outer": int(outer.sum()), "n_inner": int(inner.sum()),
                "inner_skin_near_corner": bool((inner & (np.hypot(x - (apex_o[0] if apex_o is not None else 0), y - (apex_o[1] if apex_o is not None else 0)) < 6.0)).sum() >= 5),
            }
            rows.append(row_)
            if q in (0.25, 0.50, 0.75):
                ax = axes[row, (0.25, 0.50, 0.75).index(q)]
                cx = apex_o[0] if apex_o is not None else 0.0
                cy = apex_o[1] if apex_o is not None else 0.0
                lim = 12.0
                w = (np.abs(x - cx) < lim * 1.2) & (np.abs(y - cy - 4) < lim * 1.2)
                ax.scatter(x[w & outer], y[w & outer], s=2, c="#d95f02", lw=0, label="outer skin (N.g<0)")
                ax.scatter(x[w & inner], y[w & inner], s=2, c="#1b9e77", lw=0, label="inner skin (N.g>0)")
                ax.scatter(x[w & ~outer & ~inner], y[w & ~outer & ~inner], s=2, c="#999999", lw=0, label="rim / ambiguous")
                ax.plot([0], [0], marker="*", ms=16, mfc="#2ca02c", mec="k", ls="none", label="gold")
                if apex_o is not None:
                    ax.plot([apex_o[0]], [apex_o[1]], marker="D", ms=8, mfc="#00bcd4", mec="k", ls="none", label="outer apex (N=-g)")
                    ax.add_patch(plt.Circle(apex_o, 4.0, fill=False, ls=":", color="r"))
                if apex_i is not None:
                    ax.plot([apex_i[0]], [apex_i[1]], marker="D", ms=7, mfc="#ff7f0e", mec="k", ls="none", label="inner apex (N=+g)")
                ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - 4, cy + 2 * lim - 4)
                ax.set_aspect("equal"); ax.grid(alpha=0.25)
                t = (f"{bid} @ {int(q*100)}%  gold->outer apex {row_['gold_to_outer_apex_mm']:.2f} mm" if apex_o is not None else f"{bid} @ {int(q*100)}%  no outer apex")
                t += f"\nnearest scan {row_['nearest_scan_mm']:.2f} mm on {near_skin} skin; ->inner apex " + (f"{row_['gold_to_inner_apex_mm']:.2f}" if apex_i is not None else "n/a")
                ax.set_title(t, fontsize=9)
                ax.set_xlabel("sideways (+ toward base plate P) mm", fontsize=8)
                ax.set_ylabel("bisector (+ concave) mm", fontsize=8)
                if row == 0 and q == 0.25:
                    ax.legend(fontsize=7, loc="upper left", markerscale=2)
        vals = [r_["gold_to_outer_apex_mm"] for r_ in rows if r_["gold_to_outer_apex_mm"] is not None]
        out["lines"][bid] = {
            "included_deg": included, "gold_len_mm": L, "stations": rows,
            "gold_to_outer_apex_mm_median": float(np.median(vals)) if vals else None,
            "gold_to_outer_apex_mm_range": [float(min(vals)), float(max(vals))] if vals else None,
        }
        print(f"== {bid} (included {included:.1f} deg, len {L:.1f})")
        for r_ in rows:
            print(f"   {int(r_['q']*100):3d}%  ->outerApex {r_['gold_to_outer_apex_mm'] if r_['gold_to_outer_apex_mm'] is None else round(r_['gold_to_outer_apex_mm'],2)}"
                  f"  (extreme {None if r_['gold_to_outer_apex_extreme_mm'] is None else round(r_['gold_to_outer_apex_extreme_mm'],2)})"
                  f"  ->innerApex {None if r_['gold_to_inner_apex_mm'] is None else round(r_['gold_to_inner_apex_mm'],2)}"
                  f"  nearest {r_['nearest_scan_mm']:.2f} on {r_['nearest_scan_skin']} (normal {r_['nearest_scan_normal_angle_from_bisector_deg']:+.0f} deg)"
                  f"  d_outerSkin {r_['gold_to_nearest_outer_skin_mm']:.2f} d_innerSkin {r_['gold_to_nearest_inner_skin_mm'] if r_['gold_to_nearest_inner_skin_mm'] is None else round(r_['gold_to_nearest_inner_skin_mm'],2)}"
                  f"  innerSkinAtCorner {r_['inner_skin_near_corner']}  apex_o {None if r_['apex_o_xy'] is None else np.round(r_['apex_o_xy'],2).tolist()}")
    fig.suptitle("auditor1 (resumed) sc_7a1773eca9fa: sections through each gold line, skins split by mesh-normal sign\n"
                 "gold at origin; bisector up (concave side up); red dotted = 4 mm around the outer apex", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "a1r_normal_sections.png")
    (HERE / "a1r_normal_sections.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
