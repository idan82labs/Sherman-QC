#!/usr/bin/env python3
"""Test whether one similarity (or affine) transform maps a scan's gold lines onto the
measured outer-apex lines (point-to-line residuals, from renders/<id>/measurements.json).
A tiny residual for ALL lines => systematic frame/display-cloud offset, not per-line sloppiness.
Read-only; writes gold11_audit/frame_hypothesis.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

R = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit")
out = {}
for mp in sorted((R / "renders").glob("*/measurements.json")):
    m = json.loads(mp.read_text())
    G, A, D, ids = [], [], [], []
    for l in m["lines"]:
        d = np.array(l["direction"])
        for s in l["stations"]:
            if s.get("fit_ok") and s.get("apex_xyz") and s.get("d_apex") is not None and s["d_apex"] < 15:
                G.append(s["gold_xyz"]); A.append(s["apex_xyz"]); D.append(d); ids.append(l["bend_id"])
    if len(set(ids)) < 3:
        out[m["scan_id"]] = {"note": "fewer than 3 lines with apex fits", "lines": sorted(set(ids))}
        continue
    G, A, D = map(np.array, (G, A, D))
    c0 = G.mean(0)

    def resid(p, mode):
        if mode == "similarity":
            rot = Rotation.from_rotvec(p[:3]).as_matrix(); s = np.exp(p[3]); t = p[4:7]
            X = (s * (G - c0) @ rot.T) + c0 + t
        elif mode == "translation":
            X = G + p[:3]
        else:  # affine
            M = np.eye(3) + p[:9].reshape(3, 3); X = (G - c0) @ M.T + c0 + p[9:12]
        v = A - X
        v = v - (v * D).sum(1)[:, None] * D  # perpendicular to the bend line only
        return v.ravel()

    res = {"n_points": int(len(G)), "lines": sorted(set(ids))}
    before = np.linalg.norm(resid(np.zeros(12), "affine").reshape(-1, 3), axis=1)
    res["before_rms_mm"] = float(np.sqrt((before ** 2).mean()))
    for mode, n in (("translation", 3), ("similarity", 7), ("affine", 12)):
        r = least_squares(resid, np.zeros(n), args=(mode,))
        e = np.linalg.norm(r.fun.reshape(-1, 3), axis=1)
        per = {}
        for i, e_ in zip(ids, e):
            per.setdefault(i, []).append(round(float(e_), 2))
        entry = {"rms_mm": float(np.sqrt((e ** 2).mean())), "max_mm": float(e.max()), "per_line_mm": per}
        if mode == "similarity":
            entry["scale"] = float(np.exp(r.x[3])); entry["rot_deg"] = float(np.degrees(np.linalg.norm(r.x[:3])))
            entry["translation_mm"] = r.x[4:7].tolist()
        if mode == "translation":
            entry["translation_mm"] = r.x.tolist()
        res[mode] = entry
    # leave-one-line-out: fit on the other lines, report held-out line residual (overfit control)
    if len(set(ids)) >= 4:
        loo = {}
        ids_a = np.array(ids)
        for mode, n in (("similarity", 7), ("affine", 12)):
            per = {}
            for lid in sorted(set(ids)):
                keep = ids_a != lid
                G0, A0, D0 = G, A, D
                G, A, D = G0[keep], A0[keep], D0[keep]
                r = least_squares(resid, np.zeros(n), args=(mode,))
                G, A, D = G0[~keep], A0[~keep], D0[~keep]
                e = np.linalg.norm(resid(r.x, mode).reshape(-1, 3), axis=1)
                per[lid] = round(float(np.sqrt((e ** 2).mean())), 2)
                G, A, D = G0, A0, D0
            loo[mode] = per
        res["leave_one_line_out_rms_mm"] = loo
    out[m["scan_id"]] = res
(R / "frame_hypothesis.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if "similarity" in v:
        print(k, "before", round(v["before_rms_mm"], 2), "| T", round(v["translation"]["rms_mm"], 2),
              "| S", round(v["similarity"]["rms_mm"], 2), "max", round(v["similarity"]["max_mm"], 2), "scale", round(v["similarity"]["scale"], 4),
              "rot", round(v["similarity"]["rot_deg"], 2), "| A", round(v["affine"]["rms_mm"], 2), "max", round(v["affine"]["max_mm"], 2))
    else:
        print(k, v)
    if "leave_one_line_out_rms_mm" in v:
        print("    LOO", v["leave_one_line_out_rms_mm"])
