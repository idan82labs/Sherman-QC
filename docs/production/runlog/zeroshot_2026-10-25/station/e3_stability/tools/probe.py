"""Diagnostic probe (no file edits): run the frozen E3 CLI in-process with pass-through wrappers that record the
metrology intermediates (region panel planes, _reseat_plane, _flat_start, station _fit_side fits, sharp-line move)
for the listed bend labels, plus the registered points of the bends' panels. usage: probe.py SCAN COMPILED_DIR RECIPE OUTDIR LABELS(B5,B6)"""
import json, sys, hashlib, pickle
from pathlib import Path
import numpy as np
REPO = "/Users/idantal/ShermanQC/FullProject"; sys.path[:0] = [REPO, REPO + "/backend"]
from domains.production.engine.inspect import metrology as M, cli
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
scan, cdir, recp, outd, labels = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5].split(",")
outd.mkdir(parents=True, exist_ok=True)
CAP = {"bends": {}}; cur = {"label": None, "rec": None}
def P(pl): return None if pl is None else [np.asarray(pl.n).tolist(), float(pl.d)]
o_fit, o_flat, o_reseat, o_mbg = M._fit_side, M._flat_start, M._reseat_plane, M.measure_bend_geometry
def fit_side(pts, plane, t, cfg):
    r = o_fit(pts, plane, t, cfg)
    if cur["rec"] is not None:
        cur["rec"]["fits"].append({"n_in": int(len(pts)), "plane_in": P(plane), "plane": P(r[0]), "inner": bool(r[1]),
                                   "support": int(r[2]), "rms": float(r[3])})
    return r
def flat_start(s, d, z, start0, depth, t, zlo, zhi, cfg):
    r = o_flat(s, d, z, start0, depth, t, zlo, zhi, cfg)
    if cur["rec"] is not None:
        cur["rec"]["flat"].append({"start0": float(start0), "depth": float(depth), "start": float(r), "n": int(len(s))})
    return r
def reseat(pts, plane, t, cfg):
    r = o_reseat(pts, plane, t, cfg)
    if cur["rec"] is not None:
        cur["rec"]["reseat"].append({"n": int(len(pts)), "plane_in": P(plane), "plane_out": P(r)})
        cur["rec"].setdefault("reseat_pts", []).append(np.asarray(pts, dtype=np.float32))
    return r
def mbg(idx, part, art, j, cfg):
    lab = part.bends[j].label
    if lab in labels:
        rec = {"fits": [], "flat": [], "reseat": []}; cur.update(label=lab, rec=rec)
    try:
        out = o_mbg(idx, part, art, j, cfg)
    finally:
        rec = cur["rec"]; cur.update(label=None, rec=None)
    if lab in labels:
        b = part.bends[j]; par, chi = art.panels[b.parent], art.panels[b.child]
        rec.update(label=lab, parent=int(b.parent), child=int(b.child), ok=bool(out.ok), reason=out.reason,
                   region_parent=[None if par.n0 is None else par.n0.tolist(), par.d0], region_child=[None if chi.n0 is None else chi.n0.tolist(), chi.d0],
                   t=float(par.t or part.t))
        if out.ok:
            rec.update(plane_parent=P(out.plane_parent), plane_child=P(out.plane_child), line_point=out.line_point.tolist(),
                       line_dir=out.line_dir.tolist(), sweep=out.sweep_deg, flat_shift=list(out.flat_shift),
                       sharp=out.sharp.tolist(), valid=out.valid.tolist(), angles=out.angles.tolist(), z=out.z.tolist(),
                       support=[out.support_parent, out.support_child], inner=[out.inner_parent, out.inner_child],
                       depth=[out.depth_parent, out.depth_child], visible=list(out.visible),
                       region_planes=None if out.region_planes is None else [P(x) for x in out.region_planes])
            li = M._intersect(out.plane_parent, out.plane_child)
            if li is not None:
                lp0 = li[0]; a = out.line_dir; rel = out.line_point - lp0
                rec["line_moved_by_station_median_mm"] = float(np.linalg.norm(rel - (rel @ a) * a))
            # the panels' own points (registered frame) near this bend, for plotting
            pts = idx.pts
            for nm, pan in (("parent", par), ("child", chi)):
                if len(pan.voxels):
                    m = idx.region_mask(("probe", lab, nm), pan.voxels)
                    rec[nm + "_pts"] = pts[m].astype(np.float32)
        CAP["bends"][lab] = rec
    return out
M._fit_side, M._flat_start, M._reseat_plane, M.measure_bend_geometry = fit_side, flat_start, reseat, mbg
import domains.production.engine.inspect.engine as E
if hasattr(E, "measure_bend_geometry"): E.measure_bend_geometry = mbg
h = hashlib.sha256(open(scan, "rb").read()).hexdigest()
c = load_compiled(cdir / "compiled_part.json"); r = load_recipe(Path(recp))
rc = cli.main(["--compiled", str(cdir / "compiled_part.json"), "--recipe", recp, "--scan", scan, "--out", str(outd / "engine"),
               "--scan-sha256", h, "--recipe-sha256", r.recipe_sha256, "--compiled-sha256", c.compiled_part_sha256, "--time-budget-s", "170"])
pickle.dump(CAP, open(outd / "probe.pkl", "wb"))
print("rc", rc, "captured", list(CAP["bends"]))
