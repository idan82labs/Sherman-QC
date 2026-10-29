"""PNG 1: datum-fit tilt wells on B_n5 (0.63 mm thinned). usage: plot_wells.py DIAG_TILT_JSON REF_JSON OUT_PNG"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open(sys.argv[1]))
T0 = np.array(json.load(open(sys.argv[2]))["native_best"])


def turn(T):
    T = np.array(T)
    R = T[:3, :3] @ T0[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURF, gridspec_kw={"width_ratios": [1.15, 1]})
for ax in axs:
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

ax = axs[0]
rows = [p for p in d["tilt"] if "deg" in p and p["axis"] == "f1"]
for start, col, lab in (("chosen", C2, "started from the e3-1.13 pose (wrong well)"),
                        ("native_best", C1, "started from the native-density pose")):
    xs = [p["deg"] for p in rows if p["start"] == start]
    ys = [turn(p["T"]) for p in rows if p["start"] == start]
    ax.plot(xs, ys, "o", ms=8, color=col, mec=SURF, mew=1.5, label=lab)
ax.axhline(0.07, color=C1, lw=1, ls=":")
ax.axhline(0.63, color=C2, lw=1, ls=":")
ax.text(1.55, 0.07, "native well", va="bottom", ha="right", color=INK2, fontsize=9)
ax.text(1.55, 0.63, "e3-1.13 well", va="bottom", ha="right", color=INK2, fontsize=9)
ax.set_xlabel("start tilt that lifts the datum's long ends (deg; 460 mm datum)")
ax.set_ylabel("refined pose: tilt from native pose (deg)")
ax.set_title("refine_datum from tilted starts lands in discrete wells", loc="left", fontsize=11, color=INK)
ax.set_ylim(0, 1.45)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)

ax = axs[1]
hy = d["hyps"]
labels = ["e3-1.13 choice\n(no wells)", "native pose\nrefined on thinned"]
tot = [hy[d["chosen"]]["total"], d["ref_eval"][0]["total"] if d.get("ref_eval") else np.nan]
sc = [hy[d["chosen"]]["score"], d["ref_eval"][0]["score"] if d.get("ref_eval") else np.nan]
x = np.arange(2)
ax.bar(x - 0.18, sc, 0.34, color=[C2, C1], alpha=0.55, label="datum score")
ax.bar(x + 0.18, tot, 0.34, color=[C2, C1], label="articulated total")
for i in range(2):
    ax.text(x[i] - 0.18, sc[i] + 0.02, f"{sc[i]:.3f}", ha="center", fontsize=9, color=INK)
    ax.text(x[i] + 0.18, tot[i] + 0.02, f"{tot[i]:.3f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.65)
ax.set_title("same thinned cloud: the objective prefers the native well", loc="left", fontsize=11, color=INK)
ax.text(0.0, -0.22, "light bar = datum score, dark bar = total (score + explained shares); B_n5 thinned to 333,827 points",
        transform=ax.transAxes, fontsize=8, color=INK2)
fig.suptitle("23553000 B, dense B_n5 at 0.63 mm: a search failure (local optimum of the datum plane fit), not an "
             "objective tie", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig(sys.argv[3], dpi=150, facecolor=SURF)
