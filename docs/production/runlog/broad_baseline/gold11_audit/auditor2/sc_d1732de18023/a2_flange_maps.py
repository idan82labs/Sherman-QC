"""Auditor-2: unrolled flange maps for the short down-flanges (M01, M04) and the long
up-flanges (M02, M03): scan points of the bend + flange region shown in (s along true
bend axis, height B from plate mid-plane) for the inner and outer skins separately,
with the human gold trace and the outer/inner apex traces overlaid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_ply, unit  # noqa: E402

pts, nrm, _ = load_ply()
G = load_gold()
X = json.loads((OUT / "a2_xsec.json").read_text())

fig, axes = plt.subplots(4, 1, figsize=(20, 17))
for ax, bid in zip(axes, ["M01", "M04", "M02", "M03"]):
    fr = X[bid]["frame"]
    d, up, f, mid = (np.array(fr[k]) for k in ("d", "e3", "f", "mid"))
    u_in = unit(np.cross(d, up))
    if u_in @ f > 0:
        u_in = -u_in
    w = unit(np.cross(u_in, d))
    if w @ up < 0:
        w = -w
    rel = pts - mid
    S, A, B = rel @ d, rel @ u_in, rel @ w
    nA = nrm @ u_in
    st = [s for s in X[bid]["stations"] if s.get("ok") and "apex" in s]
    aout = np.median([s["Xo"][0] for s in st])
    ain = np.median([s["Xi"][0] for s in st])
    sgn = fr["updown"]
    m = (A > aout - 1.0) & (A < ain + 1.2) & (sgn * B > -1.5) & (np.abs(S) < 160)
    inner = m & (nA > 0.5)
    outer = m & (nA < -0.5)
    other = m & ~inner & ~outer
    ax.scatter(S[outer], B[outer], s=0.3, c="#9ecae1", label="outer face of flange (normal outward)")
    ax.scatter(S[inner], B[inner], s=0.3, c="#fd8d3c", label="inner face of flange (normal toward plate centre)")
    ax.scatter(S[other], B[other], s=0.3, c="#636363", label="arc / edges / hole walls")
    a_, b_ = G[bid]
    gS = [(a_ - mid) @ d, (b_ - mid) @ d]
    gB = [(a_ - mid) @ w, (b_ - mid) @ w]
    ax.plot(gS, gB, "-", c="red", lw=2.5, label=f"HUMAN gold {bid} (projected)")
    ax.plot([s["s"] for s in st], [s["apex"][1] for s in st], "-", c="cyan", lw=1.5, label="outer apex (audit)")
    ax.plot([s["s"] for s in st if "inner_apex" in s], [s["inner_apex"][1] for s in st if "inner_apex" in s],
            "-", c="magenta", lw=1, label="inner apex (audit)")
    ax.set_title(f"{bid}: unrolled bend+flange (A within flange thickness band), height B vs s along true axis. "
                 f"Flange {'rises' if sgn > 0 else 'goes down'}", fontsize=10)
    ax.set_xlabel("s along bend axis (mm)")
    ax.set_ylabel("B (mm, plate mid-plane = 0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right" if sgn > 0 else "upper right", markerscale=12, ncol=2)
    ax.set_ylim((-17, 3) if sgn < 0 else (-3, 21))
fig.suptitle("sc_d1732de18023 auditor 2: where the human lines run on each fold/flange", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "a2_flange_maps.png", dpi=85)
print("wrote", OUT / "a2_flange_maps.png")
