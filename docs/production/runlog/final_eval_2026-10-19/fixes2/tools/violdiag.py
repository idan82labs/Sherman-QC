import sys, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import load_inputs, PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.cloud import median_spacing
from domains.production.engine.inspect.registration import DatumModel, refine_datum
from domains.production.engine.inspect.geom import apply
scan, cdir, rec = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
inp = load_inputs(cdir, rec, scan, hashlib.sha256(Path(scan).read_bytes()).hexdigest())
cfg = InspectConfig(); part = PartView(inp.compiled, inp.compiled_dir); model = DatumModel.from_part(part)
pts = read_scan(scan).points; sp = median_spacing(pts)
reg = E._decimate(pts, cfg.registration_decimate_mm) if sp < cfg.registration_decimate_mm else pts
z = np.load(sys.argv[4]); R, t = z["R"], z["t"]
T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
fit = refine_datum(reg, T, model, cfg, median_spacing(reg))
pc = apply(fit.T, reg); h = (pc - model.p0) @ model.n0
uv = model.to2d(pc[(h < 0.5) & (h > -model.t - 0.5)])
kern, ok, kin, kf = model.kernel(cfg.fine_pixel_mm, cfg.free_edge_band_mm, cfg.violation_weight, 4.0, clearance=cfg.hole_core_clearance_mm, clearance_window=cfg.window_core_clearance_mm)
pix = cfg.fine_pixel_mm
q = np.floor((uv - ok) / pix + 0.5).astype(int); H, W = kf.shape
m = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H); q = q[m]
hit = np.zeros_like(kf); hit[q[:, 1], q[:, 0]] = 1
print("forbidden cells", int(kf.sum()), "hit", int((hit * kf).sum()))
print("free segments", [(np.round(a, 1).tolist(), np.round(b, 1).tolist()) for a, b, n in model.free_segments()])
ys, xs = np.nonzero(kf); yh, xh = np.nonzero(hit * kf)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(ok[0] + xs * pix, ok[1] + ys * pix, s=1, c="0.7"); ax.scatter(ok[0] + xh * pix, ok[1] + yh * pix, s=1, c="r")
P = np.vstack([model.polygon, model.polygon[:1]]); ax.plot(P[:, 0], P[:, 1], "k-", lw=0.5)
ax.set_xlim(-110, 110); ax.set_ylim(-140, -110); ax.set_aspect("equal"); plt.savefig(sys.argv[5], dpi=80)
