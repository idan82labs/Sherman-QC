import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_08686dc1d5d4")
sys.path.insert(0, str(W / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("fs", str(W / "scripts" / "fold_stations.py"))
P = np.load(W / "plateframe.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy"); B = np.vstack([e1, e2, e3])
def to_plate(p): return ((np.asarray(p) - c) @ B.T) @ R.T
T = json.load(open(W / "true_fold_lines.json"))
TP = {k: [to_plate(np.array(v["world"][0])), to_plate(np.array(v["world"][1]))] for k, v in T.items() if "world" in v}
st = json.load(open(W / "fold_stations.json"))
GC = "#1a9c3a"
# ---------- 1) M01/M02 perpendicular close-ups
xs_in = [-90.0, -30.0, 30.0, 90.0]
fig, axs = plt.subplots(2, 5, figsize=(26, 11))
for i, (side, yw, s) in enumerate((("M01", 152.86, 1), ("M02", -158.66, -1))):
    rows = {r["x"]: r for r in st[side] if r["ok"]}
    for j, x0 in enumerate(xs_in + [140.0]):
        ax = axs[i, j]
        xx = min(rows, key=lambda k: abs(k - x0)) if x0 < 101 else x0
        m = (np.abs(P[:, 0] - xx) < 0.4)
        u = s * (yw - P[m, 1]); z = P[m, 2]
        a, b = G[side]; t = (xx - a[0]) / (b[0] - a[0]); gp = a + t * (b - a); gu = s * (yw - gp[1]); gz = gp[2]
        sel = (np.abs(u - gu) < 9) & (np.abs(z - gz) < 9)
        ax.scatter(u[sel], z[sel], s=3, c="0.25")
        ax.plot(gu, gz, "*", color=GC, ms=18, mec="k", zorder=5, label="gold")
        ax.add_patch(Circle((gu, gz), 4.0, fill=False, ls=":", color=GC))
        if xx in rows:
            r = rows[xx]
            ax.plot(*r["apex_uz"], "D", color="#17becf", mec="k", ms=9, label="outer apex", zorder=6)
            ax.plot(*r["Xo_uz"], "x", color="#1f77b4", ms=11, mew=2.5, label="outer virtual sharp")
            ax.plot(*r["Xi_uz"], "x", color="#9467bd", ms=11, mew=2.5, label="inner virtual sharp")
            ax.plot(*r["C_uz"], "+", color="#8c564b", ms=14, mew=2.5, label="arc centre (bend axis)")
            if r["inner_apex_uz"]: ax.plot(*r["inner_apex_uz"], "o", mfc="none", mec="#e377c2", ms=10, mew=2, label="inner arc apex")
            ax.set_title(f"{side} @ x={xx:.0f} (in fold)  bend {r['bend_deg']:.1f} deg  Ro {r['R_outer']:.2f}\n"
                         f"gold->apex {r['d_apex']:.2f} mm (along bisector {r['gold_minus_apex_along_bisector']:+.2f} concave+, across {r['gold_minus_apex_across']:+.2f})\n"
                         f"->Xo {r['d_outer_sharp']:.2f} ->Xi {r['d_inner_sharp']:.2f} ->axis {r['d_arc_centre']:.2f} ->scan {r['gold_nearest_scan_mm']:.2f}", fontsize=8.5)
        else:
            ax.set_title(f"{side} @ x={xx:.0f}: BEYOND fold end (x>101): no wall, flat plate/joggle strip\n(gold overrun portion)", fontsize=8.5)
        ax.set_xlim(gu - 8, gu + 8); ax.set_ylim(gz - 8, gz + 8); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("u (+ toward plate interior), mm"); ax.set_ylabel("z (plate normal; wall rises +z), mm")
        if j == 0: ax.legend(fontsize=7, loc="upper right")
fig.suptitle("sc_08686dc1d5d4 auditor2: gold M01/M02 vs the plate->wall fold, perpendicular sections (slab +-0.4 mm); dotted circle = 4 mm around gold", fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "a2_xsec_M01_M02_loci.png", dpi=72)
# ---------- 2) M03/M04 sections with joggles
fig, axs = plt.subplots(2, 3, figsize=(26, 9))
for i, (gid, lo, hi, vex) in enumerate((("M03", -182, -78, 6.0), ("M04", 150, 181, 1.0))):
    a, b = G[gid]
    for j, t in enumerate((0.25, 0.5, 0.75)):
        ax = axs[i, j]; gp = a + t * (b - a)
        m = (np.abs(P[:, 1] - gp[1]) < 0.4) & (P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 2] < 3) & (P[:, 2] > -10)
        ax.scatter(P[m, 0], P[m, 2], s=3, c="0.25")
        ax.plot(gp[0], gp[2], "*", color=GC, ms=18, mec="k", zorder=5, label=f"gold {gid}")
        for k in (("LEFT_JOGGLE_A_plate_to_flank", "LEFT_JOGGLE_B_flank_to_strip") if gid == "M03" else ("RIGHT_JOGGLE_A_plate_to_flank", "RIGHT_JOGGLE_B_flank_to_strip")):
            p0, p1 = TP[k]; tt = (gp[1] - p0[1]) / (p1[1] - p0[1]); q = p0 + tt * (p1 - p0)
            ax.plot(q[0], q[2], "v" if "A_" in k else "^", color="#9467bd", ms=10, mec="k", label=k.replace("_", " ").lower())
        ax.set_xlim(lo, hi); ax.set_aspect(vex); ax.grid(alpha=0.3)
        ax.set_title(f"{gid} @ {int(t*100)}% (y={gp[1]:.0f}); section along plate x (perpendicular to {gid}); vertical exaggeration x{vex:g}", fontsize=9)
        ax.set_xlabel("x (plate frame, mm)"); ax.set_ylabel("z (mm)")
        if j == 0: ax.legend(fontsize=7, loc="lower left" if gid == "M03" else "upper left")
fig.suptitle("sc_08686dc1d5d4 auditor2: M03 lies on flat plate ~71 mm inboard of the left joggle; M04 lies on the flat raised strip ~11 mm outboard of the right joggle (3-4 mm from the free edge)", fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "a2_xsec_M03_M04_vs_joggles.png", dpi=72)
# ---------- 3) offset profile along the fold incl. overrun
fig, axs = plt.subplots(1, 2, figsize=(22, 5.5))
for ax, side in zip(axs, ("M01", "M02")):
    rows = [r for r in st[side] if r["ok"]]
    x = [r["x"] for r in rows]
    ax.plot(x, [r["d_apex"] for r in rows], "-o", ms=3, color="k", label="|gold - outer apex| (in-section)")
    ax.plot(x, [r["gold_minus_apex_along_bisector"] for r in rows], "-", color="#d62728", label="along bisector (+ = concave side)")
    ax.plot(x, [r["gold_minus_apex_across"] for r in rows], "-", color="#1f77b4", label="across bisector")
    ax.plot(x, [r["gold_nearest_scan_mm"] for r in rows], ":", color="0.4", label="gold -> nearest scan point")
    # overrun: perpendicular distance from gold to the extended true apex line
    p0, p1 = TP[f"{side}_WALL_FOLD_TRUE_APEX"]; d = (p1 - p0) / np.linalg.norm(p1 - p0)
    a, b = G[side]; xo = np.linspace(101, max(a[0], b[0]), 30)
    dist = []
    for xx in xo:
        gp = a + (xx - a[0]) / (b[0] - a[0]) * (b - a); v = gp - p0; dist.append(np.linalg.norm(v - (v @ d) * d))
    ax.plot(xo, dist, "--", color="k", label="overrun: gold -> extended apex line (no fold here)")
    ax.axhline(4, color="#2ca02c", ls=":", label="4 mm matcher limit")
    ax.axvspan(-100, 101, color="#1f77b4", alpha=0.07, label="true fold extent (wall)")
    ax.axvspan(101, max(a[0], b[0]), color="#d62728", alpha=0.07, label="gold overrun (flat plate edge)")
    ax.set_xlabel("x along fold (plate frame, mm)"); ax.set_ylabel("mm"); ax.grid(alpha=0.3); ax.set_title(f"{side}: gold position relative to the outer apex along the fold")
    ax.legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "a2_offset_profile_M01_M02.png", dpi=72)
# ---------- 4) summary top view
fig, ax = plt.subplots(figsize=(15, 13))
rng = np.random.default_rng(1); sel = rng.choice(len(P), 250000, replace=False)
ax.scatter(P[sel, 0], P[sel, 1], s=0.15, c=np.clip(P[sel, 2], -8, 38), cmap="Greys", alpha=0.5)
col = {"WALL_FOLD": "#1f77b4", "LIP": "#ff7f0e", "JOGGLE": "#9467bd"}
for k, (p0, p1) in TP.items():
    cc = col["WALL_FOLD"] if "WALL_FOLD" in k else (col["LIP"] if "LIP" in k else col["JOGGLE"])
    off = 3.0 if "LIP" in k else 0.0
    ax.plot([p0[0], p1[0]], [p0[1] - (off if "M01" in k else -off), p1[1] - (off if "M01" in k else -off)], "-", color=cc, lw=2.2)
for k, (a, b) in G.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=GC, lw=3.0, alpha=0.85)
    mm = (a + b) / 2; ax.text(mm[0], mm[1], k, color="w", fontsize=12, bbox=dict(fc=GC, ec="none"))
ax.plot([], [], "-", color=GC, lw=3, label="human gold M01-M04")
ax.plot([], [], "-", color=col["WALL_FOLD"], lw=2.2, label="true plate->wall folds (apex)")
ax.plot([], [], "-", color=col["LIP"], lw=2.2, label="true wall->lip folds (drawn offset 3 mm for visibility) - MISSING")
ax.plot([], [], "-", color=col["JOGGLE"], lw=2.2, label="true joggle bends (2 per edge) - MISSING")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="center", fontsize=10)
ax.set_title("sc_08686dc1d5d4 (10839000_01_scan_1) auditor2 top view, plate frame. Perfect apex-locus prediction of the 8 true folds vs this gold: TP 2 / FP 6 / FN 2 (frozen matcher)", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "a2_summary_topview.png", dpi=70)
print("ok")
