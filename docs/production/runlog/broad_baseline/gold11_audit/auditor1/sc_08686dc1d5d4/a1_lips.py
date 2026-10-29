"""Lip (wall -> return flange) folds at the bottom of both walls: per-station detection + extent."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from a1_common import load_all, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T


def fit_line(P2):
    c = P2.mean(0)
    _, _, vt = np.linalg.svd(P2 - c, full_matrices=False)
    return c, vt[0]


def station(u0, side, half=0.5):
    m = (np.abs(p[:, 0] - u0) < half) & (side * p[:, 1] > 125) & (p[:, 2] < -15)
    P = p[m][:, 1:3]; N = n[m][:, 1:3]
    # wall faces between w -32 and -18
    wall = (np.abs(N[:, 0]) > 0.97) & (P[:, 1] > -36.5) & (P[:, 1] < -28)
    # lip faces: |n_w|>0.97, w<-36, and inboard of the wall by > 4 mm
    vw = np.median(P[wall, 0]) if wall.sum() else side * 156
    lip = (np.abs(N[:, 1]) > 0.97) & (P[:, 1] < -36) & (side * (vw - P[:, 0]) > 3.5)
    out = {"u": float(u0), "n_wall": int(wall.sum()), "n_lip": int(lip.sum())}
    if wall.sum() >= 8 and lip.sum() >= 8:
        lipw = P[lip]
        out["lip_inboard_extent_mm"] = float(np.max(side * (vw - lipw[:, 0])))
        cw, dw = fit_line(P[wall]); cl, dl = fit_line(P[lip])
        ang = np.degrees(np.arccos(abs(dw @ dl)))
        out["wall_lip_angle_deg"] = float(ang)
        out["lip_w"] = float(np.median(lipw[:, 1]))
        out["status"] = "fold"
    else:
        out["status"] = "no_lip"
    return out


res = {}
for lab, side in (("wall1_lip", -1), ("wall2_lip", 1)):
    rows = [station(u0, side) for u0 in np.arange(-106, 107, 1.0)]
    res[lab] = rows
    f = [r for r in rows if r["status"] == "fold"]
    us = np.array([r["u"] for r in f])
    # contiguous runs
    runs = []
    if len(us):
        start = us[0]; prev = us[0]
        for u in us[1:]:
            if u - prev > 1.5:
                runs.append((start, prev)); start = u
            prev = u
        runs.append((start, prev))
    ang = np.array([r["wall_lip_angle_deg"] for r in f]); ext = np.array([r["lip_inboard_extent_mm"] for r in f])
    print(lab, "fold stations", len(f), "runs", runs, "angle med %.1f (%.1f..%.1f)" % (np.median(ang), ang.min(), ang.max()),
          "lip width med %.1f" % np.median(ext), "lip w med %.2f" % np.median([r["lip_w"] for r in f]))
(OUT / "a1_lips.json").write_text(json.dumps(res, indent=1))
