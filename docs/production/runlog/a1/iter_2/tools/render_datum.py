"""Render the registered scan's datum-slab points (2D datum frame) against the CAD datum outline/holes/free edges."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for case in sys.argv[3:]:
    res = json.loads((ROOT / "physical/E3" / case / "engine/result.json").read_text())
    sku = res["sku"]
    cdir = ROOT / "compiled/dev" / sku
    comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
    part = PartView(comp, cdir)
    model = DatumModel.from_part(part)
    scan = read_scan(ROOT / "physical/E3" / case / "engine/registered_scan.ply")
    pts = scan.points
    h = (pts - model.p0) @ model.n0
    uv = model.to2d(pts)
    sel = (h > -part.t - 0.6) & (h < 0.6)
    sub = np.nonzero(sel)[0][:: max(1, sel.sum() // 300000)]
    fig, axs = plt.subplots(1, 2, figsize=(22, 11))
    ax = axs[0]
    ax.scatter(uv[sub, 0], uv[sub, 1], c=h[sub], s=0.2, cmap="viridis", vmin=-part.t - 0.6, vmax=0.6)
    poly = np.vstack([model.polygon, model.polygon[:1]])
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8)
    for a, b, _ in model.free_segments():
        ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=1.5)
    for hp in model.holes:
        cl = np.vstack([hp, hp[:1]])
        ax.plot(cl[:, 0], cl[:, 1], "m-", lw=1)
    ax.set_aspect("equal")
    ax.set_title(f"{case} {sku}: scan datum-slab points (colour=height) vs CAD datum (black outline, red free edges, magenta holes)")
    ax = axs[1]
    # all points projected, coloured by height, wide range
    sub2 = np.arange(0, len(pts), max(1, len(pts) // 300000))
    s = ax.scatter(uv[sub2, 0], uv[sub2, 1], c=np.clip(h[sub2], -60, 60), s=0.2, cmap="coolwarm")
    plt.colorbar(s, ax=ax)
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8)
    for a, b, _ in model.free_segments():
        ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("all scan points projected on the datum plane (colour=height, clipped +-60)")
    fig.tight_layout()
    fig.savefig(OUT / f"{case}_datum.png", dpi=70)
    plt.close(fig)
    print("wrote", OUT / f"{case}_datum.png", "bends", part.n, "datum", part.panels[part.datum].panel_id,
          "area", round(part.panels[part.datum].area, 1))
