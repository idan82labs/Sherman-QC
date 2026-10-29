"""Replay E3 up to the metrology of one bend side and print the station values against the nominal station profile."""
import json
import math
import sys
from pathlib import Path

import numpy as np

from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect import metrology as M
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, gross_inliers, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

ROOT = Path(sys.argv[1])
case = sys.argv[2]
targets = [tuple(x.split(":")) for x in sys.argv[3:]]  # B3:B
cdir_case = ROOT / "physical/E3" / case
res = json.loads((cdir_case / "engine/result.json").read_text())
cdir = ROOT / "compiled/dev" / res["sku"]
cfg = InspectConfig()
scanp = next(cdir_case.glob("input/scan.*"))
inputs = load_inputs(cdir, cdir_case / "input/recipe.json", scanp, res["scan"]["sha256"])
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(inputs.scan_path).points
pts = pts[gross_inliers(pts, cfg.extent_trim_percent, cfg.gross_outlier_extent_factor)]
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel)
vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None, median_spacing(reg_src), mirrored=False)
scored = list(proper[:cfg.articulated_hypotheses])
for hh in proper[cfg.articulated_hypotheses:]:
    if any(E._matching_symmetry(hh.fit.T @ np.linalg.inv(g.fit.T), part, cfg) for g in scored):
        scored.append(hh)
for hh in scored:
    E._articulate(hh, pts, vox, part, model, cfg, budget)
proper.sort(key=lambda h: -h.total)
amb, _, _, h = E._ambiguity(proper, part, cfg)
print('ambiguous', amb, 'chosen total', h.total)
art = h.art
reg_pts = apply(h.fit.T, pts)
idx = M.PointIndex(reg_pts, vox)
geoms = {j: M.measure_bend_geometry(idx, part, art, j, cfg) for j in range(part.n) if art.bends[j].found}
for lab, side in targets:
    j = next(i for i, b in enumerate(part.bends) if b.label == lab)
    bend = part.bends[j]
    bm = geoms[j]
    cid = f"{bend.uid}/FLANGE_LENGTH/{side}"
    nom = part.nominals[cid]
    panel_idx = bend.side_panel[side]
    plane = bm.plane_parent if panel_idx == bend.parent else bm.plane_child
    w = bm.w[panel_idx]
    d_t = bend.ro * math.tan(math.radians(min(bm.sweep_deg or 90.0, 170.0)) / 2.0)
    used = []
    if nom.boundary_kind == "ADJACENT_BEND":
        jb = part.bend_index[nom.adjacent_bend_uid]
        other = geoms[jb]
        vals = []
        for i in np.nonzero(bm.valid)[0]:
            ls = M._station_crossing(bm.sharp[i], bm.line_dir, w, other.line_point, other.line_dir)
            if ls is not None:
                vals.append(ls)
                used.append(int(i))
    else:
        mask = M._panel_region(idx, art, panel_idx)
        vals, walls, support, trunc = M._free_edge_stations(idx.pts, mask, bm, plane, float(art.panels[panel_idx].t or part.t),
                                                            w, d_t, spacing, idx.vox.voxel, cfg, used=used)
    prof = M.nominal_station_profile(part, art, bm, j, side, nom.boundary_kind, nom.adjacent_bend_uid, used)
    frame = art.bends[j].frame
    print(f"{lab}:{side} nominal {nom.nominal} kind {nom.boundary_kind} stations {len(bm.z)} width {bm.width:.2f} "
          f"visible {bm.visible} z0 {bend.z0:.2f} z1 {bend.z1:.2f}")
    for k, i in enumerate(used):
        u = float((bm.sharp[i] - frame.o) @ frame.a) - float(bend.z0)
        print(f"   st {i:2d} u {u:7.2f} meas {vals[k]:8.3f} prof {('%8.3f' % prof[k]) if prof is not None else '   none'}"
              f" dev {(vals[k] - prof[k]) if prof is not None else float('nan'):7.3f}")
