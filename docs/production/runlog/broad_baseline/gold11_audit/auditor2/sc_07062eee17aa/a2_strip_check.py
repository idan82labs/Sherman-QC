"""Auditor 2: (a) tilt of the plate strip between the M04 tab and the M05 flange (and M01/M02 strip) vs s,
looking for a crease (= a formed low-angle bend no gold line covers) vs a smooth twist;
(b) close-up of the s-max corners (reliefs) for any small formed feature."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c
from a2_profiles import G, P

def plate_top(s0, ulo, uhi, half=1.0):
    m = (np.abs(P[:, 2] - s0) < half) & (P[:, 0] > ulo) & (P[:, 0] < uhi) & (np.abs(P[:, 1]) < 4)
    q = P[m]
    if len(q) < 20:
        return None
    # upper skin: per 1 mm u-bin take max w
    bins = np.arange(ulo, uhi + 1e-9, 1.0)
    uu, ww = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mm = (q[:, 0] >= b0) & (q[:, 0] < b1)
        if mm.sum() >= 2:
            uu.append((b0 + b1) / 2); ww.append(q[mm, 1].max())
    return np.array(uu), np.array(ww)

rows = []
fig, axs = plt.subplots(2, 1, figsize=(14, 9))
for s0 in np.arange(-150, 156, 10):
    r = plate_top(s0, 94.5, 107.5)
    rmain = plate_top(s0, 20, 70)
    if r is None or rmain is None:
        continue
    uu, ww = r
    k1 = np.polyfit(uu, ww, 1)
    resid = ww - np.polyval(k1, uu)
    k2 = np.polyfit(uu, ww, 2)
    km = np.polyfit(rmain[0], rmain[1], 1)
    tilt = np.degrees(np.arctan(k1[0])) - np.degrees(np.arctan(km[0]))
    rows.append((s0, tilt, np.abs(resid).max(), k2[0]))
    axs[0].plot(uu, ww - ww[0] + s0 / 10, "-", lw=1)
    axs[0].annotate(f"s={s0:.0f} tilt {tilt:.1f}", (uu[-1], ww[-1] - ww[0] + s0 / 10), fontsize=7)
for s0, t, rr, cc in rows:
    print(f"s={s0:6.0f}  strip(u 94.5-107.5) tilt vs main plate {t:5.2f} deg   max resid from line {rr:.3f} mm  quad coef {cc:+.4f}")
axs[0].set_title("upper skin of strip between M04 tab and M05 flange (offset by s/10); straight = planar strip")
ss = [r[0] for r in rows]; tt = [r[1] for r in rows]
axs[1].plot(ss, tt, "o-")
axs[1].set_xlabel("s (mm)"); axs[1].set_ylabel("strip tilt vs main plate (deg)")
axs[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(c.OUT / "a2_strip_tilt.png", dpi=80)

rows2 = []
for s0 in np.arange(-150, 156, 20):
    r = plate_top(s0, -110.0, -86.0)
    rmain = plate_top(s0, -60, -20)
    if r is None or rmain is None:
        continue
    k1 = np.polyfit(r[0], r[1], 1); km = np.polyfit(rmain[0], rmain[1], 1)
    rows2.append((s0, np.degrees(np.arctan(k1[0])) - np.degrees(np.arctan(km[0]))))
print("M01/M02 strip tilt:", [(int(a), round(b, 2)) for a, b in rows2])

# (b) corner close-ups near s-max and s-min
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for ax, (ulo, uhi, slo, shi) in zip(axs.ravel(), [(-116, -90, 135, 162), (90, 116, 135, 162), (-116, -90, -166, -140), (90, 116, -166, -140)]):
    m = (P[:, 0] > ulo) & (P[:, 0] < uhi) & (P[:, 2] > slo) & (P[:, 2] < shi)
    q = P[m]
    sc = ax.scatter(q[:, 2], q[:, 0], c=q[:, 1], s=1, cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_aspect("equal"); ax.set_title(f"plan u[{ulo},{uhi}] s[{slo},{shi}] colour=w (-4..4)")
    plt.colorbar(sc, ax=ax)
fig.tight_layout(); fig.savefig(c.OUT / "a2_corners.png", dpi=80)
print("ok")
