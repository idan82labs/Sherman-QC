"""Where does the seeding lose the true pose? usage: seeddiag.py SCAN CDIR RECIPE ICP.npz"""
import sys, math, hashlib
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.inspect import engine as E, registration as RG
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import load_inputs, PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.cloud import median_spacing, adaptive_voxel, voxelize, estimate_normals, neighbour_graph
scan, cdir, rec = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
inp = load_inputs(cdir, rec, scan, hashlib.sha256(Path(scan).read_bytes()).hexdigest())
cfg = InspectConfig(); part = PartView(inp.compiled, inp.compiled_dir); model = RG.DatumModel.from_part(part)
pts = read_scan(scan).points; spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
reg = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
z = np.load(sys.argv[4]); R, t = z["R"], z["t"]
Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
def near(T):  # same pose up to the datum's 2D symmetric images is NOT accepted: exact pose only
    return math.degrees(E._turn(T, Ti)), float(np.linalg.norm(RG.apply(T, model.p0[None] * 0 + reg.mean(0)[None]) - RG.apply(Ti, reg.mean(0)[None])))
# patch coarse_hypotheses' pooled list: re-run with many candidates
cfg2 = InspectConfig()
hyps, segs = RG.coarse_hypotheses(vox, part, model, cfg2, mirrored=False, budget_left=lambda: 1e9)
print("coarse hyps", len(hyps))
for r, h in enumerate(hyps):
    a, d = near(h.T)
    if a < 6 and d < 25:
        print(f"  coarse rank {r} seg {h.segment} fam {h.family} h {h.height:.1f} score {h.score:.3f} turn {a:.1f} d {d:.1f}")
print("top coarse scores", [round(h.score, 3) for h in hyps[:14]], [h.segment for h in hyps[:14]])
fs = []
hyps_all = hyps
for h in [x for x in hyps if not x.segment_seed][:14] + [x for x in hyps if x.segment_seed]:
    Tf, sc = RG.fine_search(reg, h.T, model, cfg, with_score=True, walks=0)
    a, d = near(Tf); fs.append((sc, a, d, h.segment, h.segment_seed))
fs.sort(key=lambda r: -r[0])
for r, x in enumerate(fs[:25]):
    print(f"fine rank {r}: score {x[0]:.4f} turn {x[1]:.1f} d {x[2]:.1f} seg {x[3]} seed {x[4]}")
Tf, sc = RG.fine_search(reg, Ti, model, cfg, with_score=True, walks=0)
print("fine score at ICP pose", round(sc, 4), near(Tf))
