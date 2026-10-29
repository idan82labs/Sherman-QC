"""Panel right-edge check for CADF_CAD_B1 (sc_2233372b8df7), read-only."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("research/broad_baseline/triage/triage_sc_2233372b8df7.py").read().split("# 1) cross sections")[0])
fig, axs = plt.subplots(1, 3, figsize=(22, 9))
m = q[:, 0] > 95
draw(axs[0], 0, 1, m, lanes=("best_f1",)); axs[0].set_xlim(95, 130); axs[0].set_title("top view e1>95 (e1 vs e2), all heights")
for ax, s in zip(axs[1:], [60, 100]):
    m2 = (np.abs(q[:, 1] - s) < 1.5)
    draw(ax, 0, 2, m2, lanes=("best_f1",)); ax.set_xlim(90, 130); ax.set_ylim(-35, 10)
    ax.set_title(f"section e2={s}+-1.5 (e1 vs e3) near panel right edge")
plt.tight_layout(); plt.savefig(OUT + "panel_right_edge.png", dpi=80); plt.close()
print("panel e1 max at e3<-20:", q[q[:, 2] < -20, 0].max(), " points e1>110 & e3>-20 & e2>-30:", int(((q[:, 0] > 110) & (q[:, 2] > -20) & (q[:, 1] > -30)).sum()))
