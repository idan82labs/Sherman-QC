"""Auditor-1 (sc_cec08dfca9ab): independent perpendicular cross-section fit of every gold line.

For stations along each gold line (extended 25 mm beyond both ends) take a +-0.75 mm slab perpendicular
to the gold direction, keep points within 30 mm of the gold point in the section plane, split them into
the two plates that the bend should join (by local tangent direction), fit outer/inner skins per plate,
intersect outer skins (outer virtual sharp X), build the concave bisector, locate the outer apex on the
outer arc, and report the gold point offset (along bisector, + = concave side; across) plus distances to
other loci. Also reports whether a real corner exists at that station (both plates present and meeting).
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from a1frame import *

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)                 # part-frame coordinates (x along HB1, z base normal)
NP = nrm @ R.T                          # normals in part frame
EX, EY, EZ = np.eye(3)
# plate directions (part frame) for the two plates of each bend: plate A extends along dirA from the
# corner, plate B along dirB; signs are resolved from the data.
PLATES = {
    "HB1": (EY, EZ),   # base plate runs along y; channel wall runs along z
    "HB2": (EZ, EY),   # channel wall along z; top plate along y
    "HB3": (EX, EZ),   # base along x; right end wall along z
    "HB4": (EX, EZ),   # base along x; left end wall along z
    "HB5": (EZ, EX),   # left end wall along z; its top lip along x
    "HB6": (EZ, EX),   # right end wall along z; its top lip along x
}
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}
tree3 = cKDTree(P)


def unit(v):
    return v / np.linalg.norm(v)


def section(q, u, half=0.75, rad=30.0):
    """points in slab around q perpendicular to u, within rad (in-plane) of q."""
    idx = tree3.query_ball_point(q, np.sqrt(rad ** 2 + half ** 2 + 1))
    idx = np.array(idx, dtype=int)
    if len(idx) == 0:
        return idx, None
    d = P[idx] - q
    along = d @ u
    m = np.abs(along) < half
    return idx[m], d[m] - np.outer(along[m], u)


def tangent_dirs(Q, r=1.2):
    t = cKDTree(Q)
    nb = t.query_ball_point(Q, r)
    T = np.full((len(Q), 2), np.nan)
    lin = np.zeros(len(Q))
    for i, ii in enumerate(nb):
        if len(ii) < 5:
            continue
        X = Q[ii] - Q[ii].mean(0)
        w, v = np.linalg.eigh(X.T @ X)
        T[i] = v[:, 1]
        lin[i] = 1 - w[0] / max(w[1], 1e-12)
    return T, lin


def fit_skins(Q, d, nvec):
    """Q: plate points (2D), d: plate direction, nvec: in-plane normal. Return list of (offset, count) skins."""
    off = Q @ nvec
    if len(off) < 10:
        return []
    h, e = np.histogram(off, bins=np.arange(off.min() - 0.3, off.max() + 0.3, 0.1))
    # smooth
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    peaks = []
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= max(4, 0.08 * hs.max()):
            peaks.append(((e[i] + e[i + 1]) / 2, hs[i]))
    # merge peaks closer than 0.5 mm
    peaks.sort()
    merged = []
    for p in peaks:
        if merged and p[0] - merged[-1][0] < 0.5:
            if p[1] > merged[-1][1]:
                merged[-1] = p
        else:
            merged.append(p)
    # refine each skin offset as median of points within 0.35 mm
    out = []
    for o, _ in merged:
        mm = np.abs(off - o) < 0.35
        if mm.sum() >= 5:
            out.append((float(np.median(off[mm])), int(mm.sum())))
    return out


def analyse_station(name, s):
    a, b = gp[name]
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    q = a + s * u
    idx, Q3 = section(q, u)
    res = dict(s=float(s), t=float(s / L), n=int(len(idx)))
    if Q3 is None or len(idx) < 30:
        res["status"] = "no_points"
        return res
    # 2D basis from projected plate directions
    dA3, dB3 = PLATES[name]
    e1 = unit(dA3 - (dA3 @ u) * u)
    e2 = unit(dB3 - (dB3 @ u) * u - ((dB3 - (dB3 @ u) * u) @ e1) * e1)
    Q = np.stack([Q3 @ e1, Q3 @ e2], 1)           # gold point at origin
    Nn = NP[idx]                                    # scan normals (part frame)
    nA_ = np.abs(Nn @ e2); nB_ = np.abs(Nn @ e1)    # plate A (along e1) has normal ~e2; plate B normal ~e1
    mA = nA_ > np.cos(np.radians(15))
    mB = nB_ > np.cos(np.radians(15))
    res["nA"], res["nB"] = int(mA.sum()), int(mB.sum())
    if mA.sum() < 20 or mB.sum() < 20:
        res["status"] = "single_plate_or_none"
        return res
    QA, QB = Q[mA], Q[mB]
    # plate A runs along e1 -> skins are lines e2 = const ; plate B along e2 -> skins e1 = const
    skA = fit_skins(QA, np.array([1, 0.0]), np.array([0, 1.0]))
    skB = fit_skins(QB, np.array([0, 1.0]), np.array([1, 0.0]))
    res["skinsA"], res["skinsB"] = skA, skB
    if not skA or not skB:
        res["status"] = "no_skins"
        return res
    # Which skin pair forms the corner nearest the gold point: corner candidates = intersections
    # (xB, yA). Plate A extends from the corner along +/-e1: sign = side where most A points lie
    # relative to corner; similarly B.
    best = None
    for yA, nA_ in skA:
        for xB, nB_ in skB:
            X = np.array([xB, yA])
            # A points near line yA
            pa = QA[np.abs(QA[:, 1] - yA) < 0.4]
            pb = QB[np.abs(QB[:, 0] - xB) < 0.4]
            if len(pa) < 8 or len(pb) < 8:
                continue
            sA = np.sign(np.median(pa[:, 0] - xB))
            sB = np.sign(np.median(pb[:, 1] - yA))
            # corner must be "closed": A points should start near the corner (within 6 mm)
            gapA = np.min(np.abs(pa[:, 0] - xB))
            gapB = np.min(np.abs(pb[:, 1] - yA))
            dist = np.linalg.norm(X)
            cand = (dist, X, sA, sB, gapA, gapB, yA, xB)
            if best is None or dist < best[0]:
                best = cand
    if best is None:
        res["status"] = "no_corner"
        return res
    # Now we have the nearest skin-intersection; but we need the OUTER skins specifically.
    # Determine sA, sB from the nearest candidate, then choose for plate A the skin farthest
    # toward -sB (convex side) among skins within 3 mm of the chosen one; same for B.
    _, X0, sA, sB, _, _, yA0, xB0 = best
    candA = [y for y, n in skA if abs(y - yA0) < 3.0]
    candB = [x for x, n in skB if abs(x - xB0) < 3.0]
    yA_out = min(candA, key=lambda y: sB * y)       # outer = most toward -sB
    xB_out = min(candB, key=lambda x: sA * x)
    yA_in = max(candA, key=lambda y: sB * y)
    xB_in = max(candB, key=lambda x: sA * x)
    X = np.array([xB_out, yA_out])
    Xi = np.array([xB_in, yA_in]) if (len(candA) > 1 and len(candB) > 1) else None
    dA = np.array([sA, 0.0]); dB = np.array([0.0, sB])
    bis = unit(dA + dB)                               # points to concave side
    perp = np.array([-bis[1], bis[0]])
    # verify the corner actually exists: points within 4 mm of X on the bisector side
    near = Q[np.linalg.norm(Q - X, axis=1) < 4.0]
    res["n_near_corner"] = int(len(near))
    # outer apex: points near the bisector line from X; min bisector coordinate
    rel = Q - X
    bc = rel @ bis
    ac = rel @ perp
    mbis = (np.abs(ac) < 0.35) & (bc > -1.0) & (bc < 6.0)
    if mbis.sum() < 3:
        res["status"] = "corner_without_arc_points"
        apex = None
    else:
        vals = np.sort(bc[mbis])
        apex_b = float(np.percentile(vals, 5)) if len(vals) >= 6 else float(vals[0])
        apex = X + apex_b * bis
        res["apex_depth_from_sharp"] = apex_b
        res["status"] = "corner"
    # gaps: does each plate actually reach the corner region? (distance of nearest plate point to X along plate)
    pa = QA[np.abs(QA[:, 1] - yA_out) < 0.4]
    pb = QB[np.abs(QB[:, 0] - xB_out) < 0.4]
    res["gapA"] = float(np.min(sA * (pa[:, 0] - X[0]))) if len(pa) else None
    res["gapB"] = float(np.min(sB * (pb[:, 1] - X[1]))) if len(pb) else None
    res["extA"] = float(np.max(sA * (pa[:, 0] - X[0]))) if len(pa) else None
    res["extB"] = float(np.max(sB * (pb[:, 1] - X[1]))) if len(pb) else None
    g = np.zeros(2)  # gold point at origin
    res["sharp_xy"] = X.tolist()
    res["gold_minus_sharp_bis"] = float((g - X) @ bis)
    res["gold_minus_sharp_across"] = float((g - X) @ perp)
    if Xi is not None:
        res["inner_sharp_bis_from_outer_sharp"] = float((Xi - X) @ bis)
        res["thickA"] = abs(yA_in - yA_out); res["thickB"] = abs(xB_in - xB_out)
    if apex is not None:
        res["apex_xy"] = apex.tolist()
        res["gold_minus_apex_bis"] = float((g - apex) @ bis)
        res["gold_minus_apex_across"] = float((g - apex) @ perp)
        res["gold_to_apex"] = float(np.linalg.norm(g - apex))
        res["apex_xyz"] = from_part(q + apex[0] * e1 + apex[1] * e2, R, c).tolist()
    res["gold_to_scan"] = float(tree3.query(q)[0])
    res["e1"] = e1.tolist(); res["e2"] = e2.tolist(); res["q_part"] = q.tolist()
    res["sA"], res["sB"] = float(sA), float(sB)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or list(gp)
    allres = {}
    for name in names:
        a, b = gp[name]
        L = float(np.linalg.norm(b - a))
        ss = np.arange(-25.0, L + 25.0 + 1e-9, 2.0)
        rows = [analyse_station(name, s) for s in ss]
        allres[name] = dict(length=L, rows=rows)
        # compact print
        print(f"== {name} len {L:.1f}")
        for r in rows[::5]:
            if r.get("status") == "corner":
                print(f"  s={r['s']:7.1f} t={r['t']:.3f} corner  gold-apex bis={r['gold_minus_apex_bis']:6.2f} across={r['gold_minus_apex_across']:6.2f}"
                      f"  |d|={r['gold_to_apex']:5.2f} gapA={r['gapA']:.1f} gapB={r['gapB']:.1f} extA={r['extA']:.1f} extB={r['extB']:.1f} near={r['n_near_corner']} toScan={r['gold_to_scan']:.2f}")
            else:
                print(f"  s={r['s']:7.1f} t={r['t']:.3f} {r.get('status')} nA={r.get('nA')} nB={r.get('nB')}")
    json.dump(allres, open(W + "xsec_results.json", "w"), indent=1, default=float)
