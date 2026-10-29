"""CAD-only: which in-plane datum directions do the datum's free edges and holes determine, and do flange ends
(cut edges perpendicular to a parallel-bend path's axis) observe the free ones? No scan is read."""
import math, sys
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel, _round_hole
from domains.production.engine.inspect.geom import orthonormal_frame, unit

cfg = InspectConfig()
root = Path(sys.argv[1])
for cdir in sorted(root.iterdir()):
    if not (cdir / "compiled_part.json").exists():
        continue
    comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
    if comp.status != "COMPILED":
        print(cdir.name, comp.status); continue
    part = PartView(comp, cdir)
    model = DatumModel.from_part(part)
    cen2 = model.polygon.mean(axis=0)
    L = max(10.0, 0.5 * math.sqrt(model.area))
    rows = []
    min_seg = 2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm
    for a, b, n in list(model.free_segments(min_length=min_seg)) + [s[:3] for s in model.hole_segments(min_seg, 2 * cfg.edge_band_mm)]:
        d = b - a; ln = float(np.linalg.norm(d)); e = d / ln
        for s in np.arange(cfg.edge_end_trim_mm + 1.5, ln - cfg.edge_end_trim_mm, cfg.edge_bin_mm):
            x = a + e * s - cen2
            rows.append([float(n @ np.array([-x[1], x[0]])) / L, n[0], n[1]])
    for hp in model.holes:
        if not _round_hole(hp):
            continue
        c = hp.mean(axis=0); r = float(np.mean(np.linalg.norm(hp - c, axis=1)))
        nb = int(max(8, min(cfg.hole_angular_bins, int(2 * math.pi * r / 0.75))))
        for k in range(nb):
            ang = 2 * math.pi * (k + 0.5) / nb; u = np.array([math.cos(ang), math.sin(ang)])
            x = c + u * r - cen2
            rows.append([float(u @ np.array([-x[1], x[0]])) / L, u[0], u[1]])
    A = np.asarray(rows) if rows else np.zeros((0, 3))
    I = A.T @ A
    ev, evec = np.linalg.eigh(I)
    U = evec[:, ev < cfg.degenerate_min_information].T
    # flange ends (CAD positions)
    ends = []
    for c, pv in enumerate(part.panels):
        if c == part.datum or c in part.rigid_followers:
            continue
        path = part.path_bends(c)
        if not path:
            continue
        a0 = unit(part.bends[path[0]].axis_dir)
        if abs(float(a0 @ model.n0)) > 1e-3 or any(abs(float(unit(part.bends[j].axis_dir) @ a0)) < 1 - 1e-6 for j in path):
            continue
        e1, e2 = orthonormal_frame(pv.n0)
        outer = np.asarray(pv.model.outer_boundary_mm, dtype=float)
        poly = MplPath(np.stack([outer @ e1, outer @ e2], axis=1))
        for fe in pv.model.free_edges:
            if not fe.registration_usable:
                continue
            line = np.asarray(fe.polyline_mm, dtype=float)
            for a, b in zip(line[:-1], line[1:]):
                d = b - a; ln = float(np.linalg.norm(d))
                if ln < min_seg:
                    continue
                e = d / ln; n_in = unit(np.cross(pv.n0, e)); mid = 0.5 * (a + b)
                pr = np.array([mid + n_in * 0.5, mid - n_in * 0.5])
                ins = poly.contains_points(np.stack([pr @ e1, pr @ e2], axis=1))
                if ins[0] == ins[1]:
                    continue
                n_out = -n_in if ins[0] else n_in
                if abs(float(n_out @ a0)) < math.cos(math.radians(cfg.lateral_axis_tol_deg)):
                    continue
                n2 = np.array([n_out @ model.f1, n_out @ model.f2]); n2 /= np.linalg.norm(n2)
                x = model.to2d(mid[None])[0] - cen2
                row = np.array([float(n2 @ np.array([-x[1], x[0]])) / L, n2[0], n2[1]])
                ends.append((pv.panel_id, ln, row))
    sens = [(pid, round(ln, 1), [round(float(abs(row @ u)), 2) for u in U]) for pid, ln, row in ends]
    useful = [s for s in sens if U.size and max(s[2]) >= cfg.lateral_min_sensitivity]
    print(f"{cdir.name}: bends {part.n} datum {part.panels[part.datum].panel_id} area {model.area:.0f} free-edge/hole rows {len(rows)}"
          f" info {np.round(ev, 1).tolist()} undetermined {np.round(U, 2).tolist()} flange ends {len(ends)} useful {len(useful)} {useful[:6]}")
