import sys, math, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import load_inputs, PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.cloud import median_spacing
from domains.production.engine.inspect.registration import DatumModel, refine_datum, _score_direct, _datum_points
from domains.production.engine.inspect.geom import apply
scan, cdir, rec = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
inp = load_inputs(cdir, rec, scan, hashlib.sha256(Path(scan).read_bytes()).hexdigest())
cfg = InspectConfig(); part = PartView(inp.compiled, inp.compiled_dir); model = DatumModel.from_part(part)
pts = read_scan(scan).points; spacing = median_spacing(pts)
reg = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
z = np.load(sys.argv[4]); R, t = z["R"], z["t"]
T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
fit = refine_datum(reg, T, model, cfg, median_spacing(reg))
print("score", round(fit.score, 3), "edge bins", fit.edge_bins, "/", fit.edge_bins_expected, "datum_points", fit.datum_points, "cov", round(fit.datum_coverage, 3), "two_skins", fit.two_skins, "heights", fit.skin_heights, "detail", {k: v for k, v in fit.detail.items() if not hasattr(v, '__len__') or isinstance(v, str)})
pc = apply(fit.T, reg)
idx, h = _datum_points(pc, model, 0.6, 0.0)
uv = model.to2d(pc[idx])
_, ok_f, kin_f, kf_f = model.kernel_cached(cfg.fine_pixel_mm, cfg.free_edge_band_mm, cfg.violation_weight, clearance=cfg.hole_core_clearance_mm, clearance_window=cfg.window_core_clearance_mm)
print("score_direct(score,cov,viol)", [round(x, 3) for x in _score_direct(kin_f, kf_f, ok_f, cfg.fine_pixel_mm, uv, cfg.violation_weight)])
# where are the datum-height points relative to the outline
hh = (pc - model.p0) @ model.n0
uv_all = model.to2d(pc)
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for ax, (nm, sel) in zip(axs, (("datum slab |h+t/2|<t/2+0.6", np.abs(hh + 0.5 * model.t) < 0.5 * model.t + 0.6), ("near datum plane but off-slab (0.6..15mm)", (np.abs(hh + 0.5 * model.t) >= 0.5 * model.t + 0.6) & (np.abs(hh + 0.5 * model.t) < 15)))):
    q = uv_all[sel][::3]; ax.scatter(q[:, 0], q[:, 1], s=0.1, c=hh[sel][::3], cmap="jet")
    P = np.vstack([model.polygon, model.polygon[:1]]); ax.plot(P[:, 0], P[:, 1], "k-")
    for i in range(len(model.polygon)):
        if model.free[i]:
            a, b = model.polygon[i], model.polygon[(i + 1) % len(model.polygon)]; ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=2)
    ax.set_aspect("equal"); ax.set_title(nm)
plt.savefig(sys.argv[5], dpi=60)
