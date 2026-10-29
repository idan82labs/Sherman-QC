"""Channel close-up sections for sc_2233372b8df7 (read-only). Reuses the frame/setup of triage_sc_2233372b8df7.py."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open("research/broad_baseline/triage/triage_sc_2233372b8df7.py").read()
exec(src.split("# 1) cross sections")[0])
fig, axs = plt.subplots(2, 3, figsize=(21, 13))
sts = [-119, -112, -60, 30, 105, 116]
for ax, s in zip(axs.ravel(), sts):
    xr = (s - 1.0, s + 1.0); m = (q[:, 0] > xr[0]) & (q[:, 0] < xr[1])
    draw(ax, 1, 2, m, lanes=("best_f1", "rt_ph"), xr=xr, lim=((-62, -22), (55, 88)))
    ax.grid(alpha=0.3); ax.set_title(f"channel e1={s}+-1: gold o, best_f1 red x, rt_ph magenta x", fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "channel_zoom_sections2.png", dpi=80); plt.close()
# per-gold nearest prediction in section terms (e2,e3 offsets at the gold midpoint)
for k, (A, B) in G.items():
    gm = L((A + B) / 2)[0]
    best = []
    for ln, dct in LANES.items():
        for pk, (PA, PB) in dct.items():
            la, lb = L(PA)[0], L(PB)[0]
            if abs(lb[0] - la[0]) < 1: continue
            t = (gm[0] - la[0]) / (lb[0] - la[0])
            if t < -0.05 or t > 1.05: continue
            pt = la + t * (lb - la)
            best.append((float(np.hypot(*(pt[1:] - gm[1:]))), ln, pk))
    best.sort()
    print(k, "mid e123", gm.round(1), "nearest along-e1 preds:", [(round(d, 1), l, p) for d, l, p in best[:3]])
