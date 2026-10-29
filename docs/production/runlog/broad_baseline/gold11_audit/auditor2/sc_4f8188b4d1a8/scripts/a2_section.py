"""Auditor-2 section analysis: where does each gold line sit relative to the formed bend?

For a gold line, points within a cylinder around the (extended) line are expressed in a frame
(along, e1, e2).  At each station s along the line a thin slab is analysed in 2D:
  * flange axes from the in-plane normal-angle histogram (two dominant directions),
  * per flange two skins (normal sign), each a robust line  dot(q, n) = c,
  * leg directions, convex side, outer/inner virtual sharps, bisector,
  * outer apex = outer-skin point on the bisector ray, inner apex likewise,
  * gold point (origin) relative to all of the above.
Nothing is written except into the auditor2 directory.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import math

import numpy as np

from a2_common import perp_basis, unit


def cyl_select(xyz, nrm, a, b, radius, ext):
    d = unit(b - a)
    L = float(np.linalg.norm(b - a))
    rel = xyz - a
    t = rel @ d
    m = (t > -ext) & (t < L + ext)
    rel2 = rel[m]
    t2 = t[m]
    perp = rel2 - np.outer(t2, d)
    r = np.linalg.norm(perp, axis=1)
    k = r < radius
    idx = np.nonzero(m)[0][k]
    return idx, t2[k], d, L


def frame2d(xyz, nrm, idx, a, d, e1, e2, s):
    p0 = a + s * d
    rel = xyz[idx] - p0
    q = np.stack([rel @ e1, rel @ e2], axis=1)
    m3 = nrm[idx]
    m = np.stack([m3 @ e1, m3 @ e2], axis=1)
    along_n = m3 @ d
    return q, m, along_n


def _peaks_mod180(phi_deg, weights=None, min_sep=25.0):
    h, edges = np.histogram(np.mod(phi_deg, 180.0), bins=180, range=(0, 180), weights=weights)
    # circular smoothing
    k = np.array([1, 2, 3, 2, 1], float)
    hs = np.convolve(np.concatenate([h[-2:], h, h[:2]]), k / k.sum(), mode="valid")
    order = np.argsort(hs)[::-1]
    peaks = []
    for i in order:
        c = (i + 0.5)
        if all(min(abs(c - p), 180 - abs(c - p)) > min_sep for p in peaks):
            peaks.append(c)
        if len(peaks) == 2:
            break
    return peaks, hs


def _angdiff180(a, b):
    d = np.abs(np.mod(a - b + 90.0, 180.0) - 90.0)
    return d


def analyse_section(q, m, along_n, rmin=3.5, rmax=14.0, centre=None, tol_deg=12.0,
                    leg_dirs_hint=None):
    """Return dict with fitted geometry in the 2D frame (gold point = origin)."""
    out = {"ok": False}
    if centre is None:
        centre = np.zeros(2)
    mn = np.linalg.norm(m, axis=1)
    good = (np.abs(along_n) < 0.5) & (mn > 0.3)
    mm = np.zeros_like(m)
    mm[good] = m[good] / mn[good, None]
    rr = np.linalg.norm(q - centre, axis=1)
    phi = np.degrees(np.arctan2(mm[:, 1], mm[:, 0]))
    sel = good & (rr > rmin) & (rr < rmax)
    out["n_pts"] = int(len(q))
    out["n_flange_candidates"] = int(sel.sum())
    if sel.sum() < 30:
        out["why"] = "too_few_points"
        return out
    peaks, _ = _peaks_mod180(phi[sel])
    if len(peaks) < 2:
        out["why"] = "one_axis"
        return out
    flanges = []
    for th in peaks:
        nth = np.array([math.cos(math.radians(th)), math.sin(math.radians(th))])
        tth = np.array([-nth[1], nth[0]])
        f = sel & (_angdiff180(phi, th) < tol_deg)
        if f.sum() < 15:
            out["why"] = "flange_too_small"
            return out
        # refine axis with PCA of the normals (doubled-angle mean)
        ang2 = np.radians(2 * phi[f])
        th_ref = math.degrees(math.atan2(np.sin(ang2).mean(), np.cos(ang2).mean())) / 2.0
        nth = np.array([math.cos(math.radians(th_ref)), math.sin(math.radians(th_ref))])
        tth = np.array([-nth[1], nth[0]])
        f = sel & (_angdiff180(phi, th_ref) < tol_deg)
        sgn = np.sign(mm[f] @ nth)
        c = q[f] @ nth
        along = q[f] @ tth
        skins = {}
        for sg in (+1, -1):
            k = sgn == sg
            if k.sum() >= 5:
                skins[sg] = (float(np.median(c[k])), int(k.sum()), float(np.std(c[k])))
        leg_sign = 1.0 if np.median(along - centre @ tth) > 0 else -1.0
        flanges.append({"theta": th_ref, "n": nth, "t": tth * leg_sign, "skins": skins,
                        "n_pts": int(f.sum()),
                        "along_med": float(np.median(along))})
    A, B = flanges
    # outward (convex-side) normal for each flange: away from the other leg
    for F, G in ((A, B), (B, A)):
        s = -np.sign(G["t"] @ F["n"])
        if s == 0:
            s = 1.0
        F["n_out"] = F["n"] * s
        # skins keyed by sign of the measured normal along F["n"]; convert to offsets along n_out
        offs = []
        for sg, (c, cnt, sd) in F["skins"].items():
            offs.append((c * s, cnt, sd, sg * s))  # offset along n_out, count, std, normal sign along n_out
        offs.sort(key=lambda x: -x[0])
        F["offs"] = offs
    if len(A["offs"]) < 1 or len(B["offs"]) < 1:
        out["why"] = "no_skins"
        return out

    def isect(cA, cB):
        M = np.array([A["n_out"], B["n_out"]])
        if abs(np.linalg.det(M)) < 1e-3:
            return None
        return np.linalg.solve(M, np.array([cA, cB]))

    # outer skin: the largest offset along n_out whose normal points outward (+1); fall back to max
    def pick(F, which):
        cand = [o for o in F["offs"] if (o[3] > 0 if which == "outer" else o[3] < 0)]
        if not cand:
            return None
        return cand[0] if which == "outer" else cand[-1]

    oA, oB = pick(A, "outer"), pick(B, "outer")
    iA, iB = pick(A, "inner"), pick(B, "inner")
    Xo = isect(oA[0], oB[0]) if (oA and oB) else None
    Xi = isect(iA[0], iB[0]) if (iA and iB) else None
    bis = unit(A["t"] + B["t"])
    incl = math.degrees(math.acos(np.clip(A["t"] @ B["t"], -1, 1)))
    out.update({"ok": Xo is not None, "flangeA": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                  for k, v in A.items() if k in ("theta", "n_pts", "offs")},
                "flangeB": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in B.items() if k in ("theta", "n_pts", "offs")},
                "tA": A["t"].tolist(), "tB": B["t"].tolist(), "nA_out": A["n_out"].tolist(),
                "nB_out": B["n_out"].tolist(),
                "included_angle_deg": incl, "bend_angle_deg": 180.0 - incl,
                "bisector": bis.tolist()})
    if oA and iA:
        out["thickA"] = oA[0] - iA[0]
    if oB and iB:
        out["thickB"] = oB[0] - iB[0]
    if Xo is None:
        out["why"] = "parallel"
        return out
    out["Xo"] = Xo.tolist()
    out["Xi"] = Xi.tolist() if Xi is not None else None
    # outer apex: points near the bisector line through Xo, normals pointing to the convex side
    perpb = np.array([-bis[1], bis[0]])
    rel = q - Xo
    sb = rel @ bis
    lat = rel @ perpb
    near = (np.abs(lat) < 0.35) & (sb > -1.0) & (sb < 6.0)
    outer_pts = near & (mm @ bis < -0.3)
    inner_pts = near & (mm @ bis > 0.3)
    if outer_pts.sum() >= 2:
        sv = np.sort(sb[outer_pts])
        s_ap = float(np.median(sv[: max(2, min(5, len(sv)))]))
        apex = Xo + s_ap * bis
        out["apex"] = apex.tolist()
        out["apex_s_from_Xo"] = s_ap
        out["apex_n"] = int(outer_pts.sum())
    else:
        out["apex"] = None
    if inner_pts.sum() >= 2:
        sv = np.sort(sb[inner_pts])
        s_ia = float(np.median(sv[: max(2, min(5, len(sv)))]))
        out["inner_apex"] = (Xo + s_ia * bis).tolist()
        out["inner_apex_s_from_Xo"] = s_ia
    else:
        out["inner_apex"] = None
    # gold point (origin) relative to the outer apex: along bisector (+ = into concave side), lateral
    g = -Xo
    out["gold_s_from_Xo"] = float(g @ bis)
    out["gold_lat_from_Xo"] = float(g @ perpb)
    if out.get("apex") is not None:
        ga = -np.asarray(out["apex"])
        out["gold_minus_apex_bis"] = float(ga @ bis)
        out["gold_minus_apex_lat"] = float(ga @ perpb)
        out["gold_to_apex"] = float(np.linalg.norm(ga))
    out["gold_to_Xo"] = float(np.linalg.norm(Xo))
    if Xi is not None:
        out["gold_to_Xi"] = float(np.linalg.norm(Xi))
    out["gold_to_nearest_scan"] = float(np.min(np.linalg.norm(q, axis=1))) if len(q) else None
    return out
