import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"e3-1.12.0": "#9a9893", "e3-1.13.0": "#eb6834", "candidate": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURF)
def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
def bars(ax, groups, series, fmt="{:g}"):
    x = np.arange(len(groups)); w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, vals, w * 0.92, color=C[name], label=name)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + max(0.02 * ax.get_ylim()[1], 0.05), fmt.format(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, groups)
ax = axs[0]; style(ax, "FAIL->PASS characteristics (lower is better)")
ax.set_ylim(0, 11)
bars(ax, ["survey\n88 scans x 7", "dense\n16 meshes"],
     [("e3-1.12.0", [8, 7]), ("e3-1.13.0", [0, 7]), ("candidate", [0, 0])])
ax.text(0, -0.2, "dense e3-1.12/1.13 = with 0.63 mm thinning (B_n5); live e3-1.13.0 refuses dense input instead",
        transform=ax.transAxes, fontsize=7.5, color=INK2)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax = axs[1]; style(ax, "coverage loss vs e3-1.12.0 (% of decided)")
ax.set_ylim(0, 2.2)
bars(ax, ["native", "all 7 realizations"], [("e3-1.13.0", [1.46, 1.63]), ("candidate", [1.46, 1.49])], "{:.2f}")
ax = axs[2]; style(ax, "corpus-max scan pc_2714, K=2 net (s)")
ax.set_ylim(0, 200)
bars(ax, ["wall, quiet host"], [("e3-1.13.0", [99.0]), ("candidate", [119.7])], "{:.0f}")
ax.axhline(180, color=INK2, lw=1, ls="--"); ax.text(0.45, 182, "180 s hard stop", ha="right", fontsize=8.5, color=INK2)
fig.suptitle("e3-1.14.0-rc pose wells: 0 FAIL->PASS on the survey and on all 16 dense meshes, no added native coverage loss, +21 % runtime",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.03, 1, 0.93)); fig.savefig(sys.argv[1], dpi=150, facecolor=SURF)
