"""Render a run's registered scan coloured by segment label (3 orthographic views) plus the CAD final stage cloud."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.scan_io import read_scan

S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
run = Path(sys.argv[1])
out = Path(sys.argv[2])
res = json.loads((run / "result.json").read_text())
cdir = S / "a1_iter0/phys/compiled/dev" / res["sku"]
part = PartView(CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes()), cdir)
pts = read_scan(run / "registered_scan.ply").points
lab = np.fromfile(run / "segment_labels.u16", dtype="<u2") if (run / "segment_labels.u16").is_file() else np.zeros(len(pts), np.uint16)
sub = np.arange(0, len(pts), max(1, len(pts) // 250000))
P, L = pts[sub], lab[sub].astype(np.int64)
cad = part.cloud(f"S{part.n}").points
cad = cad[:: max(1, len(cad) // 60000)]
fig, axs = plt.subplots(2, 3, figsize=(27, 16))
names = {0: "unowned"}
for i, p in enumerate(part.panels):
    names[1 + i] = p.panel_id
for j, b in enumerate(part.bends):
    names[1001 + j] = b.label
uniq = sorted(set(L.tolist()))
cmap = plt.get_cmap("tab20")
colors = {u: ("lightgrey" if u == 0 else cmap(k % 20)) for k, u in enumerate(uniq)}
for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
    ax = axs[0, col]
    for u in uniq:
        m = L == u
        ax.scatter(P[m, a], P[m, b], s=0.15, color=colors[u], label=names.get(u, str(u)))
    ax.set_aspect("equal")
    ax.set_title(f"scan by segment, axes {a},{b}")
    ax = axs[1, col]
    ax.scatter(cad[:, a], cad[:, b], s=0.1, c="k")
    for j, bd in enumerate(part.bends):
        s0, s1 = bd.locus("oml_virtual_sharp")
        ax.plot([s0[a], s1[a]], [s0[b], s1[b]], "r-")
        mid = 0.5 * (s0 + s1)
        ax.text(mid[a], mid[b], bd.label, color="r")
    ax.set_aspect("equal")
    ax.set_title("CAD final stage, nominal OML lines")
axs[0, 0].legend(markerscale=20, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(out, dpi=55)
print("wrote", out)
