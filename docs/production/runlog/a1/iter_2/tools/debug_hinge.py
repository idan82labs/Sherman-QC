"""Debug the hinge theta search of one bend on a probe case (in-process E3 up to the articulated fit)."""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, gross_inliers, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.geom import apply

cdir = Path(sys.argv[1]); label = sys.argv[2]; outpng = sys.argv[3]
scan = next(cdir.glob("scan.*"))
cfg = InspectConfig()
inputs = load_inputs(cdir / "compiled", cdir / "recipe.json", scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
proper = E._register(reg_src, vox, part, model, cfg, budget, print, median_spacing(reg_src), mirrored=False)
h = proper[0]
E._articulate(h, pts, vox, part, model, cfg, budget)
j = next(b.index for b in part.bends if b.label == label)
bend = part.bends[j]
parent = h.art.panels[bend.parent]
print("parent", part.panels[bend.parent].panel_id, parent.state, "child state", h.art.panels[bend.child].state, "bf", h.art.bends[j].theta_deg, h.art.bends[j].rho_iml)
frame = AR.hinge_frame(bend, parent, cfg)
print("rho_iml", frame.rho_iml, "delta", frame.delta, "band", frame.band, "sweep", math.degrees(bend.sweep))
for th in (0, 10, 20, 30, 35, 40, 50):
    c, hw = AR.rho_window(frame, math.radians(th), cfg.hinge_rho_slack_mm, cfg)
    print(f"  theta {th}: centre {c:.3f} half {hw:.3f}")
full = apply(h.fit.T, pts)
reach = frame.band * cfg.hough_fine_reach
widx, wz, wu, wv = AR._band_points(full, frame, 2.0, radius=reach)
inb = np.hypot(wu, wv) < frame.band
th_c = np.radians(np.arange(cfg.theta_min_deg, math.degrees(bend.sweep) + cfg.theta_overbend_deg + 1e-9, 1.0))
sc, rh, ni, no = AR._hough(wz[inb], wu[inb], wv[inb], frame, th_c, cfg, frame.band)
order = np.argsort(-sc)[:8]
for k in order:
    print(f"  coarse theta {math.degrees(th_c[k]):6.1f} score {sc[k]:8.1f} rho {rh[k]:.3f} n_i {ni[k]:.0f} n_o {no[k]:.0f}")
fig, ax = plt.subplots(figsize=(9, 9))
sub = np.arange(0, len(wu), max(1, len(wu) // 40000))
ax.scatter(wu[sub], wv[sub], s=0.3, c="k")
for th in (0.0, math.radians(35)):
    c, hw = AR.rho_window(frame, th, cfg.hinge_rho_slack_mm, cfg)
    d = np.array([math.cos(th), math.sin(th)]); n = np.array([-math.sin(th), math.cos(th)])
    for off, col in ((c, "r"), (c - hw, "r:"), (c + hw, "r:"), (c + frame.delta, "b")):
        p0 = n * off; ax.plot([p0[0], p0[0] + 20 * d[0]], [p0[1], p0[1] + 20 * d[1]], col, lw=0.8)
ax.set_aspect("equal"); ax.set_title(f"{label}: band points in (u, v); red = IML window, blue = OML")
fig.savefig(outpng, dpi=90)
# joint diagnostics
kj, info = AR._short_child_joint(full, part, bend, parent, frame, th_c, reach, cfg)
print("joint ->", None if kj is None else math.degrees(th_c[kj]), info)
# replicate the scoring to list the best pairs
import itertools
from scipy.ndimage import uniform_filter
a0 = AR.unit(bend.axis_dir)
gbk = [kb for kb in part.panels[bend.child].children if abs(abs(float(AR.unit(part.bends[kb].axis_dir) @ a0)) - 1) < 1e-3][0]
gb = part.bends[gbk]
_, _, us, vs = AR._band_points(full, frame, 2.0, radius=reach)
pix = cfg.joint_pixel_mm; n = int(math.ceil(2 * reach / pix)) + 1
iu = np.floor((us + reach) / pix).astype(int); iv = np.floor((vs + reach) / pix).astype(int)
ok = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
img = np.zeros((n, n)); np.add.at(img, (iu[ok], iv[ok]), 1.0)
w = max(1, int(round(cfg.joint_window_mm / pix))); img = uniform_filter(img, size=2 * w + 1, mode="constant") * (2 * w + 1) ** 2
def proj(p):
    q = np.asarray(p, float); return q - float((q - bend.axis_point) @ a0) * a0
n_c = part.panels[bend.child].n0
starts = [proj(bend.locus("inner_tangent_child")[0]), proj(bend.locus("outer_tangent_child")[0])]
ends = [proj(gb.locus("inner_tangent_parent")[0]), proj(gb.locus("outer_tangent_parent")[0])]
segs = [(st, min(ends, key=lambda e: abs(float((e - st) @ n_c)))) for st in starts]
print("child segs (final pose)", [(np.round(a, 2).tolist(), np.round(b, 2).tolist()) for a, b in segs])
g_out = AR.unit(gb.c_out); outline = np.asarray(part.panels[gb.child].model.outer_boundary_mm, float)
g0 = [proj(gb.locus("inner_tangent_child")[0]), proj(gb.locus("outer_tangent_child")[0])]
ext = min(float(np.max((outline - g0[0]) @ g_out)), reach)
print("grand extent", ext)
def samp(segs):
    out = []
    for a, b in segs:
        m = max(2, int(np.linalg.norm(b - a) / (0.5 * pix)) + 1); out.append(a + np.linspace(0, 1, m)[:, None] * (b - a))
    return np.vstack(out)
cp, gp = samp(segs), samp([(g, g + g_out * ext) for g in g0])
def sc_(P):
    rel = P - frame.o; pu = np.floor((rel @ frame.u + reach) / pix).astype(int); pv = np.floor((rel @ frame.v + reach) / pix).astype(int)
    m = (pu >= 0) & (pu < n) & (pv >= 0) & (pv < n); return float(img[pu[m], pv[m]].sum())
rows = []
thk_all = np.radians(np.arange(cfg.theta_min_deg, math.degrees(gb.sweep) + cfg.theta_overbend_deg + 1e-9, 1.0))
for th in th_c:
    A_c = parent.A @ AR.fold_transform(bend, float(th)); s1 = sc_(apply(A_c, cp))
    best = max((sc_(apply(A_c @ AR.fold_transform(gb, float(t2)), gp)), t2) for t2 in thk_all)
    rows.append((s1 + best[0], math.degrees(th), math.degrees(best[1]), s1, best[0]))
rows.sort(reverse=True)
for r in rows[:8]:
    print(f"  total {r[0]:9.0f} theta_j {r[1]:6.1f} theta_k {r[2]:6.1f} child {r[3]:8.0f} grand {r[4]:8.0f}")
tr = [r for r in rows if abs(r[1] - 35) < 0.6]
print("  at 35:", tr[:1])
fig, ax = plt.subplots(figsize=(10, 8))
sub = np.arange(0, len(us), max(1, len(us) // 60000))
ax.scatter(us[sub], vs[sub], s=0.2, c="k")
for (tj, tk, col) in ((35.0, 35.0, "r"), (rows[0][1], rows[0][2], "b"), (0.0, 0.0, "g")):
    A_c = parent.A @ AR.fold_transform(bend, math.radians(tj))
    P1 = apply(A_c, cp) - frame.o
    P2 = apply(A_c @ AR.fold_transform(gb, math.radians(tk)), gp) - frame.o
    for P in (P1, P2):
        ax.scatter(P @ frame.u, P @ frame.v, s=0.3, c=col)
    ax.plot([], [], col, label=f"theta_j {tj:.0f} theta_k {tk:.0f}")
ax.set_xlim(-6, 20); ax.set_ylim(-8, 8); ax.set_aspect("equal"); ax.legend()
fig.savefig(outpng.replace(".png", "_joint.png"), dpi=90)
