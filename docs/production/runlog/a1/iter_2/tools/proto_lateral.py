"""Prototype: lateral cut edges of fitted child panels (perpendicular to every bend axis on their path) as
observations of the datum frame along that axis. Prints per-edge residuals for the best hypothesis."""
import json, math, sys
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, gross_inliers, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.geom import apply, orthonormal_frame

ROOT = Path(sys.argv[1]); case = sys.argv[2]
cdir_case = ROOT / "physical/E3" / case
res = json.loads((cdir_case / "engine/result.json").read_text())
sku = res["sku"]; cdir = ROOT / "compiled/dev" / sku
cfg = InspectConfig()
inputs = load_inputs(cdir, cdir_case / "input/recipe.json", next((cdir_case / "input").glob("scan.*")), res["scan"]["sha256"])
part = PartView(inputs.compiled, inputs.compiled_dir)
scan = read_scan(inputs.scan_path)
pts = scan.points
keep = gross_inliers(pts, cfg.extent_trim_percent, cfg.gross_outlier_extent_factor); pts = pts[keep]
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); slab = cfg.slab_mode(part.t, voxel); vox.slab = slab
normal_voxel = max(voxel, cfg.thin_normal_voxel_factor * part.t) if slab else None
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=normal_voxel)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
reg_spacing = median_spacing(reg_src) if reg_src is not pts else spacing
budget = E._Budget(170); logs = []
proper = E._register(reg_src, vox, part, model, cfg, budget, logs.append, reg_spacing, mirrored=False)
scored = list(proper[:cfg.articulated_hypotheses])
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted(proper, key=lambda h: -h.total)
amb, _, margin, best = E._ambiguity(proper, part, cfg)
fit = best.fit
ev, evec = np.linalg.eigh(fit.information)
print(case, sku, "spacing", round(spacing, 3), "info evals", np.round(ev, 2), "undetermined", np.round(fit.undetermined, 3))
T = fit.T
reg = apply(T, pts)
tree = cKDTree(reg)
art = best.art
t = float(fit.thickness_mm or part.t)
cen2 = model.polygon.mean(axis=0)
L = max(10.0, 0.5 * math.sqrt(model.area))
rows = []
for c, pf in enumerate(art.panels):
    if c == part.datum or pf.A is None or pf.state not in ("FITTED", "FLAT") or pf.n0 is None:
        continue
    path = part.path_bends(c)
    a0 = part.bends[path[0]].axis_dir
    if not all(abs(abs(float(part.bends[j].axis_dir @ a0)) - 1.0) < 1e-6 for j in path):
        continue
    if abs(float(a0 @ model.n0)) > 1e-3:
        continue
    pm = part.panels[c].model
    n_p = part.panels[c].n0
    e1, e2 = orthonormal_frame(n_p)
    outer = np.asarray(pm.outer_boundary_mm, dtype=np.float64)
    poly = MplPath(np.stack([outer @ e1, outer @ e2], axis=1))
    R = pf.A[:3, :3]
    for fe in pm.free_edges:
        if not fe.registration_usable:
            continue
        line = np.asarray(fe.polyline_mm, dtype=np.float64)
        for a, b in zip(line[:-1], line[1:]):
            d = b - a; length = float(np.linalg.norm(d))
            if length < 2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm:
                continue
            e = d / length
            nin = np.cross(n_p, e); nin /= np.linalg.norm(nin)
            mid = 0.5 * (a + b)
            probe = np.array([[(mid + nin * 0.5) @ e1, (mid + nin * 0.5) @ e2], [(mid - nin * 0.5) @ e1, (mid - nin * 0.5) @ e2]])
            inside = poly.contains_points(probe)
            if inside[0] == inside[1]:
                continue
            nout = -nin if inside[0] else nin
            if abs(float(nout @ a0)) < math.cos(math.radians(5.0)):
                continue
            # registered frame
            ar, br = apply(pf.A, a[None])[0], apply(pf.A, b[None])[0]
            er, nr = R @ e, R @ nout
            midr = 0.5 * (ar + br)
            sel = np.asarray(tree.query_ball_point(midr, r=0.5 * length + 2 * cfg.edge_band_mm + 1.0))
            if len(sel) == 0:
                continue
            rel = reg[sel] - ar
            along = rel @ er; off = rel @ nr
            h = reg[sel] @ pf.n0 - pf.d0
            lo, hi = cfg.edge_end_trim_mm, length - cfg.edge_end_trim_mm
            band = cfg.edge_band_mm
            k_ = (along >= lo) & (along <= hi) & (off > -band) & (off < band) & (h < 0.5) & (h > -t - 0.5)
            if k_.sum() < 3:
                print(f"   {pm.panel_id} {fe.edge_id} len {length:.1f}: no points"); continue
            margin_w = cfg.wall_skin_margin_fraction * t
            wall_pt = (h[k_] < -margin_w) & (h[k_] > -t + margin_w)
            al, of = along[k_], off[k_]
            bins = np.floor((al - lo) / cfg.edge_bin_mm).astype(np.int64)
            obs_list = []
            for bi in np.unique(bins):
                m = bins == bi
                if m.sum() < 3: continue
                o_s = np.sort(of[m])[::-1]
                kk = min(cfg.far_edge_rank, int(m.sum())) - 1
                extreme = float(o_s[kk])
                w_ = wall_pt[m] & (of[m] > extreme - cfg.wall_zone_mm)
                walled = int(w_.sum()) >= cfg.reg_wall_min_points
                obs = float(np.median(of[m][w_])) if walled else extreme
                xb = ar + er * float(np.median(al[m])) + nr * obs
                xb2 = model.to2d(xb[None])[0]
                n2 = np.array([nr @ model.f1, nr @ model.f2])
                xr = xb2 - cen2
                rows.append(([float(n2 @ np.array([-xr[1], xr[0]])) / L, n2[0], n2[1]], -obs, walled, pm.panel_id))
                obs_list.append((obs, walled))
            ob = np.array([o for o, _ in obs_list]); wl = np.array([w for _, w in obs_list])
            print(f"   {pm.panel_id} edge {fe.edge_id} len {length:.1f} nout2d {np.round([nr @ model.f1, nr @ model.f2, nr @ model.n0], 3)}"
                  f" bins {len(ob)} walled {wl.sum()} obs median {np.median(ob):+.3f} walled-med {np.median(ob[wl]) if wl.any() else float('nan'):+.3f} spread {np.std(ob):.3f}")
A = np.array([r[0] for r in rows]) if rows else np.zeros((0, 3))
if len(A):
    Ic = A.T @ A
    U = fit.undetermined
    print("child info", np.round(np.linalg.eigvalsh(Ic), 2), "along undetermined", [round(float(u @ Ic @ u), 2) for u in U])
    b = np.array([r[1] for r in rows]); w = np.array([1.0 if r[2] else cfg.edge_unwalled_weight for r in rows])
    for u in U:
        au = A @ u
        z = float((w * au) @ b) / max(float((w * au) @ au), 1e-12)
        print("   LS shift along undetermined", np.round(u, 3), "=", round(z, 3), "mm-equiv;  residual rms",
              round(float(np.sqrt(np.mean((b - au * z) ** 2))), 3))
