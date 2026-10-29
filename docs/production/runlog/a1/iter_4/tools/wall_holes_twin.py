"""wall_holes.py, but the registered scan is first moved by the compiled near-symmetry rotation (the twin image).
usage: wall_holes_twin.py E3_OUT COMPILED PANEL_INDEX OUT.png"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.geom import orthonormal_frame, apply
d = Path(sys.argv[1]); cdir = Path(sys.argv[2]); pi = int(sys.argv[3]); out = sys.argv[4]
cp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
part = PartView(cp, cdir)
M = None
for c in cp.stages[-1].symmetry.candidates:
    if c.kind == "ROTATION":
        M = np.asarray(c.transform, dtype=float); rms = c.rms_mm; break
pv = part.panels[pi]
pts = apply(M, read_scan(d / "registered_scan.ply").points)
n0 = pv.n0; p0 = pv.p0; t = part.t
h = (pts - p0) @ n0
sel = (h < 1.0) & (h > -t - 1.0)
e1, e2 = orthonormal_frame(n0)
q = np.stack([pts[sel] @ e1, pts[sel] @ e2], 1)
ob = np.asarray(pv.model.outer_boundary_mm); obq = np.stack([ob @ e1, ob @ e2], 1)
fig, ax = plt.subplots(figsize=(14, 7))
lo, hi = obq.min(0) - 5, obq.max(0) + 5
m = (q[:, 0] > lo[0]) & (q[:, 0] < hi[0]) & (q[:, 1] > lo[1]) & (q[:, 1] < hi[1])
sub = np.flatnonzero(m)[:: max(1, int(m.sum()) // 150000)]
ax.scatter(q[sub, 0], q[sub, 1], s=0.2, c="k")
ax.plot(np.r_[obq[:, 0], obq[:1, 0]], np.r_[obq[:, 1], obq[:1, 1]], "g-", lw=1)
for hole in pv.model.holes_mm:
    hh = np.asarray(hole); hq = np.stack([hh @ e1, hh @ e2], 1)
    ax.plot(np.r_[hq[:, 0], hq[:1, 0]], np.r_[hq[:, 1], hq[:1, 1]], "r-", lw=1.2)
ax.set_aspect("equal"); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_title(f"{d.parent.name} panel P{pi} under the 180-deg twin image (E1 rotation rms {rms:.2f} mm): scan (black), CAD holes (red)")
fig.tight_layout(); fig.savefig(out, dpi=70)
print("wrote", out)
