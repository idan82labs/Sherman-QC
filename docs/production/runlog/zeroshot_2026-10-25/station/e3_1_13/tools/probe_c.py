"""Candidate probe: the e3-1.13 candidate CLI in-process (candidate tree first on sys.path) with pass-through wrappers on
_reseat_plane and measure_bend_geometry for the listed bends; also usable on the live tree with E3C_REPO=<live repo>.
usage: probe_c.py SCAN COMPILED_DIR RECIPE OUTDIR LABELS"""
import hashlib, os, pickle, sys
from pathlib import Path
import numpy as np
C = os.environ.get("E3C_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3c/repo")
sys.path[:0] = [C, C + "/backend"]
from domains.production.engine.inspect import metrology as M, cli
import domains.production.engine.inspect.engine as E
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
scan, cdir, recp, outd, labels = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5].split(",")
outd.mkdir(parents=True, exist_ok=True)
CAP = {"bends": {}, "repo": C, "file": M.__file__}; cur = {"rec": None}
P = lambda pl: None if pl is None else [np.asarray(pl.n).tolist(), float(pl.d)]
o_res, o_mbg = M._reseat_plane, M.measure_bend_geometry
def reseat(pts, plane, t, cfg, *a, **k):
    r = o_res(pts, plane, t, cfg, *a, **k)
    if cur["rec"] is not None:
        cur["rec"]["reseat"].append({"plane_in": P(plane), "plane_out": P(r), "pts": np.asarray(pts, np.float32), "t": float(t)})
    return r
def mbg(idx, part, art, j, cfg):
    lab = part.bends[j].label
    if lab in labels and cur["rec"] is None:
        cur["rec"] = {"reseat": []}
        try: out = o_mbg(idx, part, art, j, cfg)
        finally: rec = cur["rec"]; cur["rec"] = None
        b = part.bends[j]
        rec.update(parent=int(b.parent), child=int(b.child), ok=bool(out.ok))
        if out.ok:
            rec.update(plane_parent=P(out.plane_parent), plane_child=P(out.plane_child), line_point=out.line_point.tolist(),
                       line_dir=out.line_dir.tolist(), visible=list(out.visible),
                       skins_seen=[list(art.panels[b.parent].skins_seen), list(art.panels[b.child].skins_seen)])
        CAP["bends"][lab] = rec
        return out
    return o_mbg(idx, part, art, j, cfg)
M._reseat_plane, M.measure_bend_geometry, E.measure_bend_geometry = reseat, mbg, mbg
h = hashlib.sha256(open(scan, "rb").read()).hexdigest()
c = load_compiled(cdir / "compiled_part.json"); r = load_recipe(Path(recp))
rc = cli.main(["--compiled", str(cdir / "compiled_part.json"), "--recipe", recp, "--scan", scan, "--out", str(outd / "engine"),
               "--scan-sha256", h, "--recipe-sha256", r.recipe_sha256, "--compiled-sha256", c.compiled_part_sha256, "--time-budget-s", "170"])
pickle.dump(CAP, open(outd / "probe.pkl", "wb")); print("rc", rc, list(CAP["bends"]), M.__file__)
