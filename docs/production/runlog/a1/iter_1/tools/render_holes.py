"""Render the scan around each mismatched datum hole (datum 2D frame), coloured by height; CAD hole outline + CAD samples."""
import json
import math
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
for spec in sys.argv[3:]:
    case, hole = spec.split(":")
    hole = int(hole)
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
    hp = model.holes[hole]
    c = hp.mean(axis=0)
    r = float(np.mean(np.linalg.norm(hp - c, axis=1)))
    span = r + 8.0
    sel = (np.abs(uv[:, 0] - c[0]) < span) & (np.abs(uv[:, 1] - c[1]) < span) & (h > -part.t - 3) & (h < 3)
    final = part.cloud(f"S{part.n}")
    fh = (final.points - model.p0) @ model.n0
    fuv = model.to2d(final.points)
    fsel = (np.abs(fuv[:, 0] - c[0]) < span) & (np.abs(fuv[:, 1] - c[1]) < span) & (fh > -part.t - 3) & (fh < 3)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    ax = axs[0]
    s = ax.scatter(uv[sel, 0], uv[sel, 1], c=np.clip(h[sel], -part.t - 1, 1), s=0.5, cmap="viridis")
    plt.colorbar(s, ax=ax, label="height along n0 (mm); skin0=0, skin1=-t")
    closed = np.vstack([hp, hp[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "r-", lw=1)
    poly = np.vstack([model.polygon, model.polygon[:1]])
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.6)
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_aspect("equal")
    ax.set_title(f"{case} {sku} hole {hole} r={r:.2f} (scan, slab +-3mm)")
    ax = axs[1]
    seg = final.segment_kind[fsel].astype(np.int64) * 1000 + final.segment_index[fsel].astype(np.int64)
    s2 = ax.scatter(fuv[fsel, 0], fuv[fsel, 1], c=fh[fsel], s=0.5, cmap="viridis")
    plt.colorbar(s2, ax=ax)
    ax.plot(closed[:, 0], closed[:, 1], "r-", lw=1)
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_aspect("equal")
    ax.set_title(f"CAD final samples (height) segs={sorted(set(seg.tolist()))[:8]}")
    ax = axs[2]
    # side view: radial distance vs height
    rr = np.linalg.norm(uv[sel] - c, axis=1)
    ax.scatter(rr, h[sel], s=0.3)
    ax.axvline(r, color="r")
    ax.axvline(max(0.5, r - 1.0), color="m")
    ax.set_xlabel("radial distance from hole centre (mm)")
    ax.set_ylabel("height")
    ax.set_title("scan radial profile")
    fig.tight_layout()
    fig.savefig(OUT / f"{case}_hole{hole}.png", dpi=80)
    plt.close(fig)
    print("wrote", OUT / f"{case}_hole{hole}.png")
