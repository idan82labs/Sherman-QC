"""Auditor-1 (sc_cec08dfca9ab): generic corner finder in a perpendicular slab + fold tracer.

corner_at(q, u, dA3, dB3): section at q perpendicular to u; plates A (runs along dA3) and B (runs along dB3)
classified by scan normals; skins by histogram peaks; outer skins chosen on the convex side; returns outer
virtual sharp, concave bisector, outer apex (5th percentile of bisector depth of arc points), inner sharp,
plate extents, all in 2D coords (e1, e2) with q at the origin, plus 3D apex.
"""
import numpy as np
from scipy.spatial import cKDTree
from a1frame import *

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
NP = nrm @ R.T
tree3 = cKDTree(P)
EX, EY, EZ = np.eye(3)
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}
PLATES = {"HB1": (EY, EZ), "HB2": (EZ, EY), "HB3": (EX, EZ), "HB4": (EX, EZ), "HB5": (EZ, EX), "HB6": (EZ, EX)}


def unit(v):
    return v / np.linalg.norm(v)


def section(q, u, half=0.75, rad=30.0):
    idx = np.array(tree3.query_ball_point(q, np.sqrt(rad ** 2 + half ** 2 + 1)), dtype=int)
    if len(idx) == 0:
        return idx, None
    d = P[idx] - q
    al = d @ u
    m = np.abs(al) < half
    return idx[m], d[m] - np.outer(al[m], u)


def skins(off, minpts=5):
    if len(off) < 10:
        return []
    e = np.arange(off.min() - 0.3, off.max() + 0.3, 0.1)
    if len(e) < 4:
        return []
    h, e = np.histogram(off, bins=e)
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    pk = [((e[i] + e[i + 1]) / 2, hs[i]) for i in range(1, len(hs) - 1)
          if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= max(4, 0.08 * hs.max())]
    pk.sort()
    mg = []
    for p in pk:
        if mg and p[0] - mg[-1][0] < 0.5:
            if p[1] > mg[-1][1]:
                mg[-1] = p
        else:
            mg.append(p)
    out = []
    for o, _ in mg:
        mm = np.abs(off - o) < 0.35
        if mm.sum() >= minpts:
            out.append((float(np.median(off[mm])), int(mm.sum())))
    return out


def corner_at(q, u, dA3, dB3, expect=None, max_from_expect=6.0, half=0.75, rad=30.0, ang_tol=15.0):
    """expect: optional 2D point (in e1,e2 coords) near which the outer sharp should be."""
    idx, Q3 = section(q, u, half, rad)
    out = {"n": int(len(idx))}
    if Q3 is None or len(idx) < 30:
        out["status"] = "no_points"; return out
    e1 = unit(dA3 - (dA3 @ u) * u)
    t2 = dB3 - (dB3 @ u) * u
    e2 = unit(t2 - (t2 @ e1) * e1)
    Q = np.stack([Q3 @ e1, Q3 @ e2], 1)
    Nn = NP[idx]
    mA = np.abs(Nn @ e2) > np.cos(np.radians(ang_tol))
    mB = np.abs(Nn @ e1) > np.cos(np.radians(ang_tol))
    out["nA"], out["nB"] = int(mA.sum()), int(mB.sum())
    if mA.sum() < 20 or mB.sum() < 20:
        out["status"] = "single_plate_or_none"; return out
    QA, QB = Q[mA], Q[mB]
    skA, skB = skins(QA[:, 1]), skins(QB[:, 0])
    if not skA or not skB:
        out["status"] = "no_skins"; return out
    ref = np.zeros(2) if expect is None else np.asarray(expect, float)
    best = None
    for yA, _ in skA:
        for xB, _ in skB:
            X = np.array([xB, yA])
            pa = QA[np.abs(QA[:, 1] - yA) < 0.4]
            pb = QB[np.abs(QB[:, 0] - xB) < 0.4]
            if len(pa) < 8 or len(pb) < 8:
                continue
            sA = np.sign(np.median(pa[:, 0] - xB)); sB = np.sign(np.median(pb[:, 1] - yA))
            # plates must actually approach the corner (within 6 mm along each plate)
            gA = np.min(sA * (pa[:, 0] - xB)); gB = np.min(sB * (pb[:, 1] - yA))
            if gA > 6.0 or gB > 6.0:
                continue
            dist = np.linalg.norm(X - ref)
            if best is None or dist < best[0]:
                best = (dist, X, sA, sB, yA, xB)
    if best is None:
        out["status"] = "no_corner"; return out
    dist, X0, sA, sB, yA0, xB0 = best
    if expect is not None and dist > max_from_expect:
        out["status"] = "corner_far_from_expected"; out["dist_expect"] = float(dist); return out
    cA = [y for y, _ in skA if abs(y - yA0) < 2.5]
    cB = [x for x, _ in skB if abs(x - xB0) < 2.5]
    yAo = min(cA, key=lambda y: sB * y); xBo = min(cB, key=lambda x: sA * x)
    yAi = max(cA, key=lambda y: sB * y); xBi = max(cB, key=lambda x: sA * x)
    X = np.array([xBo, yAo])
    dA = np.array([sA, 0.0]); dB = np.array([0.0, sB])
    bis = unit(dA + dB); perp = np.array([-bis[1], bis[0]])
    rel = Q - X
    bc, ac = rel @ bis, rel @ perp
    m = (np.abs(ac) < 0.35) & (bc > -1.0) & (bc < 6.0)
    if m.sum() < 3:
        out["status"] = "corner_without_arc_points"; return out
    vals = np.sort(bc[m])
    ab = float(np.percentile(vals, 5)) if len(vals) >= 6 else float(vals[0])
    apex = X + ab * bis
    pa = QA[np.abs(QA[:, 1] - yAo) < 0.4]; pb = QB[np.abs(QB[:, 0] - xBo) < 0.4]
    out.update(status="corner", e1=e1, e2=e2, X=X, bis=bis, perp=perp, apex=apex, apex_depth=ab,
               apex3=q + apex[0] * e1 + apex[1] * e2, sharp3=q + X[0] * e1 + X[1] * e2,
               inner_sharp=np.array([xBi, yAi]) if (len(cA) > 1 and len(cB) > 1) else None,
               thick=(abs(yAi - yAo), abs(xBi - xBo)),
               gapA=float(np.min(sA * (pa[:, 0] - X[0]))), gapB=float(np.min(sB * (pb[:, 1] - X[1]))),
               extA=float(np.max(sA * (pa[:, 0] - X[0]))), extB=float(np.max(sB * (pb[:, 1] - X[1]))),
               n_arc=int(m.sum()), Q=Q, mA=mA, mB=mB)
    return out


def arc_coverage(r, nbins=10, tol=0.45):
    """fraction of angular bins (over the bend angle) of the outer arc that contain scan points."""
    Q, X, bis = r["Q"], r["X"], r["bis"]
    ab = r["apex_depth"]
    dA = np.array([np.sign(r["e1"] @ r["e1"]), 0.0])  # placeholder (unused)
    Rr = ab / (np.sqrt(2) - 1) if ab > 0.05 else 0.3
    Cc = X + Rr * np.sqrt(2) * bis
    rel = Q - Cc
    rad = np.linalg.norm(rel, axis=1)
    ang = np.degrees(np.arctan2(rel @ np.array([-bis[1], bis[0]]), -(rel @ bis)))  # 0 at apex direction
    m = (np.abs(rad - Rr) < tol) & (np.abs(ang) <= 45)
    bins = np.floor((ang[m] + 45) / (90 / nbins)).astype(int).clip(0, nbins - 1)
    return len(set(bins.tolist())) / nbins, float(Rr)


def fold_quality(r, min_ext=5.0, max_gap=2.5, min_arc=0.8):
    if r.get("status") != "corner":
        return False, {}
    cov, Rr = arc_coverage(r)
    q = dict(arc_cov=cov, R_outer=Rr, extA=r["extA"], extB=r["extB"], gapA=r["gapA"], gapB=r["gapB"])
    ok = (cov >= min_arc and r["extA"] >= min_ext and r["extB"] >= min_ext and r["gapA"] <= max_gap
          and r["gapB"] <= max_gap and r["gapA"] >= -0.5 and r["gapB"] >= -0.5)
    return ok, q
