"""Pose diagnostic: run the scratch E3 in-process on one scan (no outputs written) and dump every articulated
proper hypothesis (pose, datum score, explained shares, total, decoded states) plus, optionally, the same
quantities for reference poses (refine_datum from a given T, then the articulated fit) evaluated on THIS cloud.

usage: diag.py SCAN COMPILED_DIR RECIPE OUT_JSON [CFG_JSON|-] [REF_POSES_JSON]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("E3P_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3pose/repo"))
sys.path[:0] = [str(REPO), str(REPO / "backend")]
import numpy as np  # noqa: E402
from domains.production.engine.inspect import engine as E  # noqa: E402
from domains.production.engine.inspect.config import load_config  # noqa: E402
from domains.production.engine.inspect.part import load_inputs  # noqa: E402
from domains.production.engine.inspect.registration import refine_datum  # noqa: E402

scan, cdir, recipe, out = sys.argv[1:5]
cfgp = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
refp = sys.argv[6] if len(sys.argv) > 6 else None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


cap = {}
_reg0, _art0, _amb0 = E._register, E._articulate, E._ambiguity


def reg(pts, vox, part, model, cfg, budget, note, spacing, *, mirrored):
    if not mirrored and "reg_src" not in cap:
        cap.update(reg_src=pts, reg_spacing=spacing)
    return _reg0(pts, vox, part, model, cfg, budget, note, spacing, mirrored=mirrored)


def art(h, pts, vox, part, model, cfg, budget, reversed_search=False):
    if not h.mirrored and "pts" not in cap:
        cap.update(pts=pts, vox=vox, part=part, model=model, cfg=cfg, budget=budget)
    return _art0(h, pts, vox, part, model, cfg, budget, reversed_search=reversed_search)


def hrow(h):
    return {"T": np.asarray(h.fit.T).tolist(), "score": h.fit.score, "explained": h.explained,
            "explained_tight": h.explained_tight, "total": h.total, "free_extra": h.free_extra,
            "articulated": h.art is not None, "states": [d.state for d in (h.decodes or [])],
            "edge_inlier": h.fit.detail.get("edge_inlier"), "hole_inlier": h.fit.detail.get("hole_inlier"),
            "raster": h.fit.detail.get("raster_coverage"), "violation": h.fit.detail.get("violation"),
            "two_skins": h.fit.two_skins, "hole_mismatch": h.fit.hole_mismatch}


def amb(hyps, part, cfg, evidence=None, note=None):
    cap["hyps"] = [hrow(h) for h in hyps]
    r = _amb0(hyps, part, cfg, evidence=evidence, note=note)
    cap["chosen"] = next(i for i, h in enumerate(hyps) if h is r[3])
    return r


E._register, E._articulate, E._ambiguity = reg, art, amb
_rd0, _fs0 = E.refine_datum, E.fine_search
cap["calls"] = []


def rd(points, T0, model, cfg, spacing, **kw):
    t = time.time()
    f = _rd0(points, T0, model, cfg, spacing, **kw)
    cap["calls"].append({"fn": "refine", "T0": np.asarray(T0).tolist(), "T": f.T.tolist(), "score": f.score,
                         "s": round(time.time() - t, 3), "n": len(points)})
    return f


def fs(points, T, model, cfg, with_score=False, walks=None):
    t = time.time()
    r = _fs0(points, T, model, cfg, with_score=True, walks=walks)
    cap["calls"].append({"fn": "fine", "T0": np.asarray(T).tolist(), "T": np.asarray(r[0]).tolist(), "score": r[1],
                         "s": round(time.time() - t, 3), "walks": walks})
    return r if with_score else r[0]


E.refine_datum, E.fine_search = rd, fs
cfg = load_config(cfgp, time_budget_s=float(os.environ.get("E3P_BUDGET", "600")))
inputs = load_inputs(cdir, recipe, scan, sha(scan))
t0 = time.time()
o = E.inspect_scan(inputs, Path(out).with_suffix(".tmpout"), cfg, E.InspectOptions(write_outputs=False))
wall = time.time() - t0
res = {"scan": scan, "wall_s": round(wall, 2), "validity": o.result.inspection_validity.state,
       "summary": o.result.summary.state, "log": o.log, "hyps": cap.get("hyps"), "chosen": cap.get("chosen"),
       "readings": E._readings_of(o.result)["chars"], "calls": cap["calls"]}
if refp and "pts" in cap:
    refs = json.load(open(refp))
    ev = []
    for name, T in refs.items():
        T = np.asarray(T, dtype=np.float64)
        f = refine_datum(cap["reg_src"], T, cap["model"], cap["cfg"], cap["reg_spacing"])
        h = E._Hyp(fit=f)
        cap["budget"].seconds = 1e9
        E._articulate(h, cap["pts"], cap["vox"], cap["part"], cap["model"], cap["cfg"], cap["budget"])
        r = hrow(h)
        r["name"] = name
        r["refined_from_turn_deg"] = float(np.degrees(E._turn(f.T, T)))
        r["refined_from_shift_mm"] = float(np.linalg.norm(f.T[:3, 3] - T[:3, 3]))
        ev.append(r)
    res["ref_eval"] = ev
if os.environ.get("E3P_TILT") and "pts" in cap:
    from domains.production.engine.inspect.registration import rigid, _datum_points
    from domains.production.engine.inspect.geom import apply
    import math
    model, cfg = cap["model"], cap["cfg"]
    c3 = model.p0

    def tilt(T, axis, deg):
        a = np.asarray(axis, float); a = a / np.linalg.norm(a); th = math.radians(deg)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        return rigid(R, c3 - R @ c3) @ T
    starts = {"chosen": np.asarray(res["hyps"][res["chosen"]]["T"])}
    if refp:
        starts.update({k: np.asarray(v) for k, v in json.load(open(refp)).items()})
    pr = []
    for sname, T in starts.items():
        for ax_name, ax in (("f1", model.f1), ("f2", model.f2)):
            for deg in (-1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5):
                t1 = time.time()
                f = refine_datum(cap["reg_src"], tilt(T, ax, deg), model, cfg, cap["reg_spacing"])
                pr.append({"start": sname, "axis": ax_name, "deg": deg, "T": f.T.tolist(), "score": f.score,
                           "cov": f.datum_coverage, "s": round(time.time() - t1, 2)})
        # wide-band plane check at this pose
        pc = apply(T, cap["reg_src"])
        for band in (2.0, 3.0, 5.0):
            idx, h = _datum_points(pc, model, model.t + band, 1.5)
            pr.append({"start": sname, "band": band, "n": int(len(idx)),
                       "h_q": [float(np.percentile(h[idx], q)) for q in (1, 5, 50, 95, 99)] if len(idx) else None})
    res["tilt"] = pr
if os.environ.get("E3P_PERTURB") and "pts" in cap:
    from domains.production.engine.inspect.registration import fine_search, rigid, _rot2
    import math
    model, cfg = cap["model"], cap["cfg"]
    F = np.stack([model.f1, model.f2, model.n0], axis=1)
    cen2 = model.polygon.mean(axis=0)
    c3 = model.p0 + model.f1 * cen2[0] + model.f2 * cen2[1]

    def perturb(T, dth_deg, dx, dy):
        r3 = np.eye(3); r3[:2, :2] = _rot2(math.radians(dth_deg))
        R = F @ r3 @ F.T
        return rigid(R, c3 - R @ c3 + model.f1 * dx + model.f2 * dy) @ T
    starts = {"chosen": np.asarray(res["hyps"][res["chosen"]]["T"])}
    if refp:
        starts.update({k: np.asarray(v) for k, v in json.load(open(refp)).items()})
    pr = []
    for sname, T in starts.items():
        for dth in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dx, dy in ((0, 0), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)):
                t1 = time.time()
                Ts = perturb(T, dth, dx, dy)
                f = refine_datum(cap["reg_src"], Ts, model, cfg, cap["reg_spacing"])
                t2 = time.time()
                Tf, sc = fine_search(cap["reg_src"], Ts, model, cfg, with_score=True, walks=0)
                f2 = refine_datum(cap["reg_src"], Tf, model, cfg, cap["reg_spacing"])
                t3 = time.time()
                pr.append({"start": sname, "dth": dth, "dx": dx, "dy": dy, "T": f.T.tolist(), "score": f.score,
                           "t_refine": round(t2 - t1, 2), "fs_score": sc, "T_fs": f2.T.tolist(), "score_fs": f2.score,
                           "t_fs": round(t3 - t2, 2)})
    res["perturb"] = pr
Path(out).write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: res[k] for k in ("wall_s", "validity", "summary", "chosen")}))
