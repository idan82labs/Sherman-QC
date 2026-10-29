"""D2 dev harness (consumed 49207000 scan; diagnostic): seeds + per-hypothesis flange ends vs the ICP pose."""
import sys, math, hashlib, time
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.inspect import engine as E, registration as RG
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import load_inputs, PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.cloud import median_spacing, adaptive_voxel, voxelize, estimate_normals, neighbour_graph
SP = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad"
scan = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming_codex_fetch_2026-10-17/20260901_e8bfd4f6/49207000_B.ply"
C = Path(SP + "/fixes/compile_after1/DEV-7052AE8973")
inp = load_inputs(C / "compiled", C / "dev.recipe.json", scan, hashlib.sha256(Path(scan).read_bytes()).hexdigest())
cfg = InspectConfig(); part = PartView(inp.compiled, inp.compiled_dir); model = RG.DatumModel.from_part(part)
pts = read_scan(scan).points; spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
reg = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
rsp = median_spacing(reg) if reg is not pts else spacing
z = np.load(SP + "/fixes2/icp49207.npz"); R, t = z["R"], z["t"]
Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
c0 = reg.mean(0)
def near(T):
    return round(math.degrees(E._turn(T, Ti)), 1), round(float(np.linalg.norm(RG.apply(T, c0[None]) - RG.apply(Ti, c0[None]))), 1)
budget = E._Budget(1e9)
log = []
note = lambda m: log.append(m)
t0 = time.time()
proper = E._register(reg, vox, part, model, cfg, budget, note, rsp, mirrored=False)
print("register", round(time.time() - t0, 1), "s;", log)
for h in proper:
    print(" fit score %.3f undet %d near %s" % (h.fit.score, h.fit.undetermined.shape[0], near(h.fit.T)))
hyps, segs = RG.coarse_hypotheses(vox, part, model, cfg, mirrored=False, budget_left=lambda: 1e9, plane_seeds=True)
hp = hyps[1]
Tf, sc = RG.fine_search(reg, hp.T, model, cfg, with_score=True, walks=0)
fit = RG.refine_datum(reg, Tf, model, cfg, rsp)
Ti2 = model.symmetric_images(fit.T)[0]
for walks in (0, 2):
    T3, s3 = RG.fine_search(reg, Ti2, model, cfg, with_score=True, walks=walks)
    f2 = RG.refine_datum(reg, T3, model, cfg, rsp)
    for k in range(3):
        h = E._Hyp(fit=f2); E._articulate(h, pts, vox, part, model, cfg, budget)
        print("walks", walks, "iter", k, "fine %.3f fit %.3f near %s expl %.3f tight %.3f total %.3f" % (s3, f2.score, near(f2.T), h.explained, h.explained_tight, h.total), flush=True)
        T3, s3 = RG.fine_search(reg, f2.T, model, cfg, with_score=True, walks=0)
        f2 = RG.refine_datum(reg, T3, model, cfg, rsp)
hI = E._Hyp(fit=RG.refine_datum(reg, Ti, model, cfg, rsp)); E._articulate(hI, pts, vox, part, model, cfg, budget)
print("ICP refine: near %s expl %.3f total %.3f" % (near(hI.fit.T), hI.explained, hI.total))
