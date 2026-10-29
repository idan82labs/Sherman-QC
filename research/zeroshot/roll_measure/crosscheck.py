"""Independent cross-check: plain RANSAC cylinder on the raw scan points + normals, then a direct nonlinear
least-squares cylinder fit on the inliers (geom.fit_cylinder). Shares nothing with the main method's slicing,
curvature runs, arc tracking or station statistics. An optional hand crop (a slab along a direction, raw scan
coordinates) isolates one portion of a part.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.crosscheck --path SCAN --out OUT.json
          [--crop "nx,ny,nz,lo,hi"] [--seed 0]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.roll_measure import geom as G  # noqa: E402


def load_points(path, n=250000, seed=0):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), min(n, len(V)), replace=False)
    return V[idx], N[idx]


def ransac_cylinder(P, N, iters=4000, tol=0.3, r_min=10.0, r_max=1500.0, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(P), min(30000, len(P)), replace=False)
    Ps, Ns = P[sub], N[sub]
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        a = np.cross(N[i], N[j])
        s = np.linalg.norm(a)
        if s < math.sin(math.radians(4)):
            continue
        a /= s
        e1, e2 = G.frame(a)
        pi, pj = np.array([P[i] @ e1, P[i] @ e2]), np.array([P[j] @ e1, P[j] @ e2])
        ni, nj = G.unit(np.array([N[i] @ e1, N[i] @ e2])), G.unit(np.array([N[j] @ e1, N[j] @ e2]))
        A = np.c_[ni, -nj]
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        tt = np.linalg.solve(A, pj - pi)
        c = pi + tt[0] * ni
        R = np.linalg.norm(pi - c)
        if not (r_min < R < r_max) or abs(np.linalg.norm(pj - c) - R) > tol:
            continue
        X = np.c_[Ps @ e1, Ps @ e2] - c
        rho = np.linalg.norm(X, axis=1)
        radial = X / np.maximum(rho, 1e-9)[:, None]
        nn = np.c_[Ns @ e1, Ns @ e2]
        al = np.abs(np.sum(radial * nn, 1)) / np.maximum(np.linalg.norm(nn, axis=1), 1e-9)
        score = int(np.sum((np.abs(rho - R) < tol) & (al > math.cos(math.radians(12)))))
        if best is None or score > best[0]:
            best = (score, a, c[0] * e1 + c[1] * e2, R)
    return best


def refine(P, N, a, p0, R, tol):
    for _ in range(3):
        r = G.cylinder_residual(P, p0, a, R)
        X = P - p0
        rad = X - np.outer(X @ a, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-9)[:, None]
        ok = (np.abs(r) < tol) & (np.abs(np.sum(rad * N, 1)) > math.cos(math.radians(12)))
        p0, a, R, rms = G.fit_cylinder(P[ok], p0, a, R)
        tol = max(3 * rms, 0.15)
    return p0, a, R, rms, ok


def angular_sweep(P, p0, a, mask, gap_deg=5.0):
    e1, e2 = G.frame(a)
    X = P[mask] - p0
    ph = np.degrees(np.arctan2(X @ e2, X @ e1)) % 360
    h = np.bincount(ph.astype(int) % 360, minlength=360)
    occ = h >= max(3, 0.2 * np.median(h[h > 0])) if (h > 0).any() else h > 0
    # largest circular run of occupied bins, bridging gaps <= gap_deg
    o = occ.copy()
    for k in range(360):
        if not o[k]:
            # bridge short gaps
            j = k
            while j < k + 360 and not occ[j % 360]:
                j += 1
            if j - k <= gap_deg and occ[(k - 1) % 360]:
                for q in range(k, j):
                    o[q % 360] = True
    if o.all():
        return 360.0, 1.0
    start = int(np.argmin(o))  # an empty bin
    best = cur = 0
    for q in range(start, start + 360):
        cur = cur + 1 if o[q % 360] else 0
        best = max(best, cur)
    run_bins = best
    return float(run_bins), float(occ.sum() / max(run_bins, 1))


def check(path, crop=None, seed=0, r_max=1500.0):
    P, N = load_points(path, seed=seed)
    if crop:
        n = G.unit(np.array(crop[:3]))
        h = P @ n
        k = (h >= crop[3]) & (h <= crop[4])
        P, N = P[k], N[k]
    best = ransac_cylinder(P, N, seed=seed, r_max=r_max)
    if best is None:
        return {"found": False}
    score, a, p0, R = best
    p0, a, R, rms, ok = refine(P, N, a, p0, R, 0.3)
    # which skin: outward normals pointing away from the axis = convex = outer skin
    X = P[ok] - p0
    rad = X - np.outer(X @ a, a)
    rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-9)[:, None]
    convex = float(np.mean(np.sum(rad * N[ok], 1) > 0))
    skin = "outer" if convex > 0.5 else "inner"
    sweep, cov = angular_sweep(P, p0, a, ok)
    # the other skin, coaxial: radius from the median distance of opposite-facing points within R +- 8 mm
    X = P - p0
    rho = np.linalg.norm(X - np.outer(X @ a, a), axis=1)
    radl = (X - np.outer(X @ a, a)) / np.maximum(rho, 1e-9)[:, None]
    facing = np.sum(radl * N, 1)
    opp = (facing < -0.9) if skin == "outer" else (facing > 0.9)
    near = opp & (np.abs(rho - R) < 8.0) & (np.abs(rho - R) > 0.1)
    other = None
    if near.sum() > 500:
        r2 = float(np.median(rho[near]))
        k2 = near & (np.abs(rho - r2) < 0.5)
        if k2.sum() > 300:
            p2, a2, R2, rms2 = G.fit_cylinder(P[k2], p0, a, r2)
            other = {"R": R2, "rms": rms2, "n": int(k2.sum()),
                     "axis_angle_deg": float(np.degrees(np.arccos(min(1.0, abs(a2 @ a)))))}
    s = X[ok] @ a
    return {"found": True, "skin": skin, "R": R, "rms": rms, "axis": a.tolist(), "point": (p0 + a * float(np.median(s))).tolist(),
            "n_inliers": int(ok.sum()), "n_points": int(len(P)), "sweep_deg": sweep, "sweep_occupancy": cov,
            "length_mm": float(np.percentile(s, 99) - np.percentile(s, 1)), "other_skin": other}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rmax", type=float, default=1500.0, help="hand limit on R (excludes a dominant flat plate)")
    a = ap.parse_args()
    crop = [float(x) for x in a.crop.split(",")] if a.crop else None
    res = {"path": a.path, "crop": crop, "r_max": a.rmax, "runs": [check(a.path, crop, seed=s, r_max=a.rmax) for s in (a.seed, a.seed + 1, a.seed + 2)]}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for r in res["runs"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("axis", "point")})


if __name__ == "__main__":
    main()
