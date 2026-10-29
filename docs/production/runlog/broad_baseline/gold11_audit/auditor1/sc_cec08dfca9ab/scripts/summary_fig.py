"""Auditor-1 (sc_cec08dfca9ab): one summary figure (part frame top view + offset bars) and raw-frame joggle flank lines.

Inputs (read-only): cached scan npz, gold labels, prior strict_summary.json (measured apex fold runs).
Output: OUTD/a1_summary_gold_vs_folds.png, W/joggle_lines_a1.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import load, to_part, from_part, W, OUTD

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
S = json.load(open(W + "strict_summary.json"))
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}

# joggle flank mid-lines from scan points (PCA of points within 0.35 mm of flank centre in y, both skins)
J = {}
for name, yc, zr in (("base_joggle", -148.8, (-13.5, -8.0)), ("lip_joggle", 95.1, (24.0, 30.0))):
    m = (np.abs(P[:, 1] - yc) < 0.35) & (P[:, 2] > zr[0]) & (P[:, 2] < zr[1]) & (np.abs(P[:, 0]) < 156)
    Q = P[m]
    cq = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - cq)
    d = vt[0] * np.sign(vt[0][0])
    t = (Q - cq) @ d
    a, b = cq + t.min() * d, cq + t.max() * d
    J[name] = dict(part=[a.round(2).tolist(), b.round(2).tolist()], raw=[from_part(a, R, c).tolist(), from_part(b, R, c).tolist()],
                   length=float(t.max() - t.min()), n=int(m.sum()),
                   note="flank mid-line; the joggle's two bends (knees) lie ~1.2-1.4 mm either side of it in y")
json.dump(J, open(W + "joggle_lines_a1.json", "w"), indent=1)

fig = plt.figure(figsize=(17, 10))
ax = fig.add_axes([0.04, 0.06, 0.55, 0.88])
rng = np.random.default_rng(0)
sub = rng.choice(len(P), 160000, replace=False)
ax.scatter(P[sub, 0], P[sub, 1], s=0.05, c="#9a9a9a", rasterized=True)
for k, (a, b) in gp.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#d000d0", lw=2.2, alpha=.9)
    mid = 0.5 * (a + b)
    ax.text(mid[0] + (4 if mid[0] > 0 else -14), mid[1] + 3, k, color="#d000d0", fontsize=11, weight="bold")
for k, s in S.items():
    for (p0, p1) in s["fold_runs_part"]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color="#0a8f2a", lw=5, alpha=.45, solid_capstyle="butt")
for k, j in J.items():
    a, b = j["part"]
    ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#ff7f00", lw=2.5)
    ax.text(0, a[1] + (3 if k == "lip_joggle" else -7), f"{k.replace('_', ' ')}: 2 bends ~45 deg, step 1.8 mm (NOT in gold)", color="#c05800",
            fontsize=10, ha="center")
ax.set_aspect("equal"); ax.grid(alpha=.25)
ax.set_xlim(-185, 185); ax.set_ylim(-175, 128)
ax.set_title("sc_cec08dfca9ab top view (part frame): magenta = gold HB1-6, green = measured apex fold runs, orange = joggles")
ax.plot([], [], "-", color="#d000d0", lw=2, label="gold (human)")
ax.plot([], [], "-", color="#0a8f2a", lw=5, alpha=.45, label="measured fold run (outer apex)")
ax.plot([], [], "--", color="#ff7f00", lw=2, label="missing joggle (2 bends each)")
ax.legend(loc="lower right", fontsize=9)

ax2 = fig.add_axes([0.66, 0.56, 0.32, 0.36])
names = list(S)
bis = [S[k]["offset_bis"][0] for k in names]
lo = [S[k]["offset_bis"][0] - S[k]["offset_bis"][1] for k in names]
hi = [S[k]["offset_bis"][2] - S[k]["offset_bis"][0] for k in names]
ax2.bar(names, bis, yerr=[lo, hi], color="#d000d0", alpha=.75, capsize=4)
ax2.axhline(4.0, color="r", ls=":", label="matcher 4 mm")
inner = [S[k]["inner_sharp_bis_median"] - S[k]["apex_depth_median"] for k in names]
ax2.plot(names, inner, "s", color="#7b4fb0", label="inner virtual sharp")
th = [S[k]["thick_median"] for k in names]
ax2.plot(names, [1.3] * len(names), "_", ms=30, color="#555", label="inner skin apex (~t = 1.3 mm)")
ax2.axhline(0, color="k", lw=.8)
ax2.plot(names, [-(S[k]["apex_depth_median"]) for k in names], "x", color="#1f77b4", ms=9, label="outer virtual sharp (convex side)")
ax2.set_ylabel("offset from outer apex along bisector, mm (+ = concave side)")
ax2.set_title("gold locus vs bend loci (median over fold, whiskers = min/max)")
ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=.3)

ax3 = fig.add_axes([0.66, 0.08, 0.32, 0.36])
cov = [S[k]["coverage_of_fold_pct"] for k in names]
onf = [S[k]["gold_on_fold_pct"] for k in names]
xx = np.arange(len(names))
ax3.bar(xx - 0.2, cov, 0.4, label="fold covered by gold %", color="#0a8f2a", alpha=.7)
ax3.bar(xx + 0.2, onf, 0.4, label="gold length on a fold %", color="#d000d0", alpha=.7)
for i, k in enumerate(names):
    ob, oa = S[k]["overshoot_before"], S[k]["overshoot_after"]
    if ob + oa > 3:
        ax3.text(i + 0.2, onf[i] + 1.5, f"+{ob:.0f}/{oa:.0f} mm", ha="center", fontsize=8)
ax3.set_xticks(xx); ax3.set_xticklabels(names); ax3.set_ylim(0, 112)
ax3.axhline(60, color="r", ls=":", lw=1)
ax3.set_title("extent: gold overshoot past fold ends (before/after)")
ax3.legend(fontsize=8, loc="lower left"); ax3.grid(alpha=.3)
fig.savefig(OUTD + "a1_summary_gold_vs_folds.png", dpi=80)
print(json.dumps({k: {kk: (vv if kk != "raw" else [[round(x, 2) for x in p] for p in vv]) for kk, vv in j.items()} for k, j in J.items()}, indent=1))
