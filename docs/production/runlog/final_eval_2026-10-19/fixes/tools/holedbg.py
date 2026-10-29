"""POST-HOC: replay the non-round hole fill test on the registered scan."""
import sys, json
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel, _nonround_hole_filled, _round_hole
from domains.production.engine.eval.plyio import read_ply_vertices
from scipy.spatial import cKDTree
ed, cdir = Path(sys.argv[1]), Path(sys.argv[2])
cp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
part = PartView(cp, cdir); model = DatumModel.from_part(part)
P = read_ply_vertices(ed / "registered_scan.ply").vertices
h = (P - model.p0) @ model.n0
cand = (h < 0.5) & (h > -model.t - 0.5)
uv = model.to2d(P[cand]); print("t", model.t, "slab pts", cand.sum(), "holes", len(model.holes))
spacing = 0.5
for k, hp in enumerate(model.holes):
    rnd = _round_hole(hp)
    cell = max(1.0, 2.0 * spacing)
    core, ring = model.hole_fill_cells(k, cell, cell, 4.0)
    tree = cKDTree(uv)
    dc, _ = tree.query(core, k=1, distance_upper_bound=0.5 * cell) if len(core) else (np.array([]), None)
    dr, _ = tree.query(ring, k=1, distance_upper_bound=0.5 * cell) if len(ring) else (np.array([]), None)
    # where are the core-covering points (height)?
    idx = tree.query_ball_point(core, r=0.5 * cell) if len(core) else []
    hs = h[cand][sorted({i for l in idx for i in l})] if len(core) else np.array([])
    print(k, "round" if rnd else "nonround", "size", np.round(hp.max(0) - hp.min(0), 1), "core", len(core), "cov", round(float(np.mean(np.isfinite(dc))), 2) if len(dc) else None,
          "ring", len(ring), "cov", round(float(np.mean(np.isfinite(dr))), 2) if len(dr) else None,
          "filled", _nonround_hole_filled(uv, model, k, spacing) if not rnd else "-", "core-pt heights", np.round(np.percentile(hs, [5, 50, 95]), 2) if len(hs) else None)
print("--- round-hole replay")
from domains.production.engine.inspect.registration import _open_disc_share
hh = h[cand]
for k, hp in enumerate(model.holes[:3]):
    c = hp.mean(axis=0); rads = np.linalg.norm(hp - c, axis=1); r_h = float(np.mean(rads))
    rr = np.linalg.norm(uv - c, axis=1); r_in = max(0.5, r_h - 1.0)
    inner = rr < r_in
    print(k, "r_h", round(r_h, 2), "rad min/max", round(rads.min(), 2), round(rads.max(), 2), "inner pts", inner.sum(),
          "thr", max(8, 0.05 * np.pi * r_h * r_h / 0.25), "inner heights pct", np.round(np.percentile(hh[inner], [5, 25, 50, 75, 95]), 2) if inner.sum() else None,
          "inner uv spread", np.round(np.ptp(uv[inner], axis=0), 1) if inner.sum() else None)
    print("   hole pts uv sample", np.round(hp[::max(1, len(hp)//8)], 1).tolist())
