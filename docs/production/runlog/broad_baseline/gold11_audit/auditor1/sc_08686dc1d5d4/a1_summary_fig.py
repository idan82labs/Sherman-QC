"""Summary figure: plan with gold lines vs measured formed bends; full wall profiles at u=0; detector seed regions."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1_common import load_all, gold_lines, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
g = gold_lines()
gp = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in g.items()}
M = json.loads((OUT / "a1_match_check.json").read_text())
MB = json.loads((OUT / "a1_missing_bends.json").read_text())
det = json.loads(open("/Users/idantal/ShermanQC/FullProject/output/experiments/f1-10of13/full-zero-shot-current/10839000/"
                      "10839000_01_scan_1.ply--98b00841230a/decomposition_result.json").read())

fig = plt.figure(figsize=(26, 13))
ax = fig.add_subplot(1, 2, 1)
idx = np.arange(0, len(p), 7)
ax.scatter(p[idx, 0], p[idx, 1], s=0.15, c="0.75")
for k in ("M01", "M02"):
    a, b = [(np.array(x) - o) @ R.T for x in M[k]["apex_line_raw"]]
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1f77b4", lw=6, alpha=0.5)
for k, v in MB.items():
    a, b = [(np.array(x) - o) @ R.T for x in v["line_raw"]]
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#d62728", lw=4, alpha=0.6)
for r in det["owned_bend_regions"]:
    ep = np.array(r.get("span_endpoints") or [r["line_start"], r["line_end"]], float)
    q = (ep - o) @ R.T
    ax.plot(q[:, 0], q[:, 1], ":", color="#ff7f0e", lw=2.5)
    ax.text(q[:, 0].mean(), q[:, 1].mean() + (6 if q[:, 1].mean() > 0 else -9), r["bend_id"], color="#ff7f0e", fontsize=10)
for k, (s, e) in gp.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], "-", color="#1a9c3a", lw=1.8)
    m = (s + e) / 2
    ax.text(m[0] + 3, m[1] + 3, k, fontsize=12, color="#1a9c3a", weight="bold")
ax.plot([], [], "-", color="#1f77b4", lw=6, alpha=0.5, label="measured plate->wall folds (gold M01/M02 targets), 200 mm each")
ax.plot([], [], "-", color="#d62728", lw=4, alpha=0.6, label="formed bends with NO gold: 2 lip folds (~133 mm) + 2 joggles x 2 bends (304 mm)")
ax.plot([], [], ":", color="#ff7f0e", lw=2.5, label="F1 zero-shot regions OB1-OB4 that seeded the gold (adjudicate_remaining_verified_bends.py _case_02)")
ax.plot([], [], "-", color="#1a9c3a", lw=1.8, label="gold M01-M04 (closed rectangle)")
ax.legend(loc="center", fontsize=9)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)")
ax.set_title("plan (part frame): gold vs measured formed bends")

for i, (side, key) in enumerate(((-1, "M01"), (1, "M02"))):
    ax = fig.add_subplot(2, 2, 2 + 2 * i)
    m = (np.abs(p[:, 0] - 0.0) < 0.5) & (side * p[:, 1] > 125)
    ax.scatter(p[m, 1], p[m, 2], s=1.5, c="0.25")
    s, e = gp[key]
    t = (0.0 - s[0]) / (e[0] - s[0]); q = s + t * (e - s)
    ax.plot(q[1], q[2], "*", ms=16, color="#1a9c3a", mec="k")
    ax.text(q[1] + side * -6, q[2] - 4, key, color="#1a9c3a", fontsize=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("full wall profile at u = 0 (%s side): plate->wall fold (gold %s) and wall->lip fold at w~-40 (no gold)" % ("wall 1" if side < 0 else "wall 2", key), fontsize=9)
    ax.set_xlabel("v (mm)"); ax.set_ylabel("w (mm)")
fig.suptitle("sc_08686dc1d5d4 (10839000_01_scan_1) auditor1 summary", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "a1_summary.png", dpi=70)
print("ok")
