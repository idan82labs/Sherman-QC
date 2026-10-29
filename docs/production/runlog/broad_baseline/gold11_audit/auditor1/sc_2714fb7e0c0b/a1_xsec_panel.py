"""Auditor-1 cross-section panel: every gold line at 3 stations (10/50/90 % of span) with the
auditor's normal-based arc fit (centre C, outer radius R, outer apex), plus gold-minus-apex offset
profiles along each line. Bisector g (concave side) points UP in every panel. Read-only on inputs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import a1common as A
import a1_stations as S

pts, nrm, gold = S.pts, S.nrm, S.gold
tree = cKDTree(pts)
ids = ["B01", "B02", "B03", "B04", "B05", "B07", "B09", "B10", "B06", "B08"]
fig, axes = plt.subplots(len(ids), 4, figsize=(19, 4.3 * len(ids)), dpi=70,
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
data = json.loads((A.OUT / "a1_stations.json").read_text())
for row, bid in enumerate(ids):
    s0, s1 = gold[bid]
    nA, nB = S.CFG[bid]
    d = S.u(s1 - s0); L = float(np.linalg.norm(s1 - s0))
    o = S.u(nA + nB); gdir = -o; v = S.u(np.cross(d, gdir))
    st = {round(r["s"]): r for r in data[bid]["stations"]}
    for k, f in enumerate((0.1, 0.5, 0.9)):
        ax = axes[row, k]
        s = round(f * L)
        c = s0 + s * d
        idx = np.array(tree.query_ball_point(c, 14.0), int)
        p = pts[idx]
        m = np.abs((p - c) @ d) < 0.6
        q = np.c_[(p[m] - c) @ v, (p[m] - c) @ gdir]   # x = across, y = inward (concave up)
        ax.scatter(q[:, 0], q[:, 1], s=3, c="#444")
        ax.plot(0, 0, "*", color="lime", mec="k", ms=16, label=f"gold {bid}", zorder=5)
        ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color="g"))
        r = st.get(s)
        if r and "apex" in r:
            ap = np.array(r["apex"]); C = np.array(r["C"])
            # stored as (g, v) -> plot as (v, g)
            ax.plot(ap[1], ap[0], "D", color="#17becf", mec="k", ms=8, label="outer apex (fit)", zorder=6)
            ax.plot(C[1], C[0], "+", color="#8c564b", ms=13, mew=2, label="arc centre (fit)", zorder=6)
            ax.add_patch(plt.Circle((C[1], C[0]), r["R"], fill=False, color="#8c564b", lw=0.8))
            ttl = (f"{bid} s={s} mm ({int(f*100)}%): gold-apex {r['gold_to_apex']:.2f} mm "
                   f"(in {r['gold_minus_apex_inward']:+.2f}, across {r['gold_minus_apex_across']:+.2f}); R={r['R']:.2f}")
        else:
            ttl = f"{bid} s={s} mm ({int(f*100)}%): NO fold arc found (n_arc={r.get('n_arc') if r else None})"
        ax.set_title(ttl, fontsize=8)
        ax.set_xlim(-8, 8); ax.set_ylim(-6, 10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    ax = axes[row, 3]
    rows = data[bid]["stations"]
    ss = [r["s"] for r in rows if r.get("present")]
    ax.plot(ss, [r["gold_minus_apex_inward"] for r in rows if r.get("present")], ".", ms=3, label="inward (+ = concave side)")
    ax.plot(ss, [r["gold_minus_apex_across"] for r in rows if r.get("present")], ".", ms=3, label="across")
    ax.plot(ss, [r["gold_to_apex"] for r in rows if r.get("present")], ".", ms=3, label="|gold-apex|")
    ax.axvspan(0, L, color="lime", alpha=0.08, label="gold span")
    sm = data[bid]["summary"]
    if sm.get("fold_extent_s_mm"):
        ax.axvspan(*sm["fold_extent_s_mm"], color="k", alpha=0.07, label="fold present (scan)")
    ax.axhline(4, color="r", lw=0.6, ls=":"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-12, L + 12); ax.set_ylim(-4, 6)
    ax.set_title(f"{bid}: offset profile along gold (len {L:.1f} mm), fold extent {sm.get('fold_extent_s_mm')}", fontsize=8)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
fig.suptitle("sc_2714fb7e0c0b auditor-1: gold vs measured outer apex (concave side up; dotted circle = 4 mm)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.995))
fig.savefig(A.OUT / "a1_xsec_panel.png")
print("ok")
