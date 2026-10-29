"""Whole-part search for formed features: scan points whose normal is NOT within 15 deg of any
part axis (+-L, +-S, +-H) are bend-arc / incline candidates. Plotted in 3 projections with the
gold lines and the auditor-3 apex + joggle lines, to look for formed bends without gold."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *

Q = np.load(SCR / "Q.npy"); NQ = np.load(SCR / "NQ.npy"); C0 = np.load(SCR / "center.npy")
mx = np.abs(NQ).max(1)
cand = mx < np.cos(np.radians(15))
print("candidate points", int(cand.sum()), "of", len(Q))
P = Q[cand]
gold = [(b, to_frame(a, C0), to_frame(e, C0)) for b, a, e in load_gold()]
apx = json.loads((OUT / "a3_apex_lines.json").read_text())["lines"]
apl = {k: (to_frame(np.array(v["start"]), C0), to_frame(np.array(v["end"]), C0)) for k, v in apx.items()}
fig, axes = plt.subplots(3, 1, figsize=(22, 26), dpi=70)
for ax, (i, j, nm) in zip(axes, ((0, 1, "top view L-S"), (0, 2, "front L-H"), (1, 2, "side S-H"))):
    ax.scatter(P[:, i], P[:, j], s=0.3, c="k", linewidths=0, alpha=0.5)
    for k, (a, e) in apl.items():
        ax.plot([a[i], e[i]], [a[j], e[j]], "--", color="#1f77b4" if "joggle" not in k else "#ff7f0e", lw=1.2)
    for b, a, e in gold:
        ax.plot([a[i], e[i]], [a[j], e[j]], "-", color="#1a9c3a", lw=1.5)
        ax.annotate(b, ((a[i] + e[i]) / 2, (a[j] + e[j]) / 2), color="#1a9c3a", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"{nm}: points with off-axis normals (bend arcs / inclines); green = gold, blue dashed = a3 apex lines, orange dashed = joggle bends (no gold)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "a3_all_formed_features.png"); plt.close(fig)
# histogram of candidate points by (S) position in the bottom flange zone of the long sides, to spot any extra lines
for side, sel in (("S-", (P[:, 1] < -55) & (P[:, 2] < -15)), ("S+", (P[:, 1] > 60) & (P[:, 2] < -15))):
    h, e = np.histogram(P[sel, 1], bins=np.arange(-100, 110, 1.0))
    peaks = [(round(float(e[k]), 0), int(h[k])) for k in range(len(h)) if h[k] > 300]
    print(side, "bottom-zone off-axis normal peaks (S bin, count):", peaks)
for side, sel in (("top S-", (P[:, 1] < -55) & (P[:, 2] > 15)), ("top S+", (P[:, 1] > 60) & (P[:, 2] > 15))):
    h, e = np.histogram(P[sel, 1], bins=np.arange(-100, 110, 1.0))
    peaks = [(round(float(e[k]), 0), int(h[k])) for k in range(len(h)) if h[k] > 300]
    print(side, "top-zone peaks:", peaks)
for side, sel in (("L- (short)", (P[:, 0] < -125)), ("L+ (short)", (P[:, 0] > 140))):
    h, e = np.histogram(P[sel, 0], bins=np.arange(-150, 166, 1.0))
    peaks = [(round(float(e[k]), 0), int(h[k])) for k in range(len(h)) if h[k] > 300]
    print(side, "peaks along L:", peaks)
