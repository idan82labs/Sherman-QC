"""Diagnose DATUM_HOLES_FILLED on iter-0 outputs: per datum hole, points inside it (registered frame)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

ROOT = Path(sys.argv[1])  # a1_iter0/phys
cases = sys.argv[2:]
out = {}
for case in cases:
    rec = json.loads((ROOT / "physical/E3" / case / "record.json").read_text())
    res = json.loads((ROOT / "physical/E3" / case / "engine/result.json").read_text())
    sku = res["sku"]
    cdir = ROOT / "compiled/dev" / sku
    comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
    part = PartView(comp, cdir)
    model = DatumModel.from_part(part)
    scan = read_scan(ROOT / "physical/E3" / case / "engine/registered_scan.ply")
    pts = scan.points
    h = (pts - model.p0) @ model.n0
    slab = (h < 0.5) & (h > -model.t - 0.5)
    uv_all = model.to2d(pts)
    tree = cKDTree(uv_all[slab])
    slab_idx = np.nonzero(slab)[0]
    spacing = 0.5
    rows = []
    for k, hp in enumerate(model.holes):
        c = hp.mean(axis=0)
        rads = np.linalg.norm(hp - c, axis=1)
        r_h = float(np.mean(rads))
        sel = np.asarray(tree.query_ball_point(c, r=r_h + 3.0), dtype=np.int64)
        rel = uv_all[slab_idx[sel]] - c
        rr = np.linalg.norm(rel, axis=1)
        inner = rr < max(0.5, r_h - 1.0)
        thr = max(8, 0.05 * math.pi * r_h * r_h / spacing ** 2)
        hin = h[slab_idx[sel]][inner]
        # all points (any height) over the hole core
        allsel = np.nonzero(np.linalg.norm(uv_all - c, axis=1) < max(0.5, r_h - 1.0))[0]
        rows.append({"hole": k, "c": [round(float(c[0]), 2), round(float(c[1]), 2)], "r": round(r_h, 3),
                     "nverts": len(hp), "circ_std": round(float(np.std(rads)), 3),
                     "inner_slab": int(inner.sum()), "thr": round(thr, 1), "mismatch": bool(inner.sum() > thr),
                     "inner_h_q": [round(float(q), 3) for q in np.quantile(hin, [0, .1, .5, .9, 1])] if len(hin) else None,
                     "inner_any_height": int(len(allsel)),
                     "any_h_q": [round(float(q), 2) for q in np.quantile(h[allsel], [0, .1, .5, .9, 1])] if len(allsel) else None,
                     "ring_pts": int(((rr > r_h - 1.5) & (rr < r_h + 3.0)).sum())})
    out[case] = {"sku": sku, "t": part.t, "holes": len(model.holes), "datum": part.panels[part.datum].panel_id,
                 "rows": rows}
    print(case, sku, "t", part.t, "holes", len(model.holes))
    for r in rows:
        print("   ", r)
json.dump(out, open(Path(__file__).parent.parent / "diag" / "holes.json", "w"), indent=1)
