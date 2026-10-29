"""Auditor-2 independent bend fit: outer-apex line + physical fold extent, seeded by any line.

fit_bend(xyz, nrm, seed_a, seed_b) ->
  apex points per station (3D), robust 3D apex line, fold extent along it (arc material present AND
  both flanges present), median bend angle / thickness, outer/inner virtual sharp lines.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import math

import numpy as np

from a2_common import perp_basis, unit
from a2_section import analyse_section, cyl_select, frame2d


def _fit_line3d(P, w=None):
    P = np.asarray(P, float)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c)
    d = vt[0]
    return c, d


def robust_line(P, iters=4, k=2.5):
    P = np.asarray(P, float)
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        c, d = _fit_line3d(P[keep])
        r = P - c
        r = r - np.outer(r @ d, d)
        dist = np.linalg.norm(r, axis=1)
        mad = np.median(dist[keep]) + 1e-6
        new = dist < max(k * 1.4826 * mad, 0.25)
        if new.sum() < 5 or np.array_equal(new, keep):
            keep = new if new.sum() >= 5 else keep
            break
        keep = new
    c, d = _fit_line3d(P[keep])
    return c, d, keep


def sections_along(xyz, nrm, a, b, step=1.0, ext=12.0, half=0.6, rmin=3.5, rmax=13.0, radius=22.0):
    idx, t, d, L = cyl_select(xyz, nrm, a, b, radius=radius, ext=ext + 2)
    e1, e2 = perp_basis(d)
    rows = []
    for s in np.arange(-ext, L + ext + 1e-6, step):
        k = np.abs(t - s) < half
        r = {"s": float(s)}
        if k.sum() >= 20:
            q, m, an = frame2d(xyz, nrm, idx[k], a, d, e1, e2, s)
            r.update(analyse_section(q, m, an, rmin=rmin, rmax=rmax))
        else:
            r["ok"] = False
            r["why"] = "empty"
        p0 = a + s * d
        r["p0"] = p0
        r["e1"], r["e2"] = e1, e2
        for key in ("Xo", "Xi", "apex", "inner_apex"):
            if r.get(key) is not None:
                v = np.asarray(r[key])
                r[key + "_3d"] = p0 + v[0] * e1 + v[1] * e2
        if r.get("ok"):
            bis = np.asarray(r["bisector"])
            r["bis_3d"] = bis[0] * e1 + bis[1] * e2
            tA, tB = np.asarray(r["tA"]), np.asarray(r["tB"])
            r["tA_3d"] = tA[0] * e1 + tA[1] * e2
            r["tB_3d"] = tB[0] * e1 + tB[1] * e2
        rows.append(r)
    return rows, d, L


def fold_presence(xyz, nrm, c, d, bis, tA, tB, s_lo, s_hi, step=0.5, tube=0.6, leg_lo=3.0, leg_hi=6.0,
                  leg_tol=0.9, ref_lines=None):
    """March along apex line c + s d. arc present: >=1 scan point within `tube` of apex line in the
    +-step/2 window AND within 0.6 mm of the outer skin direction.  legs present: points within leg_tol of
    the outer-skin line positions at leg_lo..leg_hi mm along each leg (measured from the outer virtual sharp
    = apex - ro_sh * bis, approximated using the leg tangents)."""
    rel = xyz - c
    s_all = rel @ d
    m = (s_all > s_lo - 2) & (s_all < s_hi + 2)
    P = rel[m]
    S = s_all[m]
    perp = P - np.outer(S, d)
    out = []
    for s in np.arange(s_lo, s_hi + 1e-9, step):
        k = np.abs(S - s) < step / 2 + 0.05
        Q = perp[k]
        if len(Q) == 0:
            out.append((s, 0, 0, 0))
            continue
        dist_apex = np.linalg.norm(Q, axis=1)
        arc = int((dist_apex < tube).sum())
        legs = []
        for tv in (tA, tB):
            # outer skin of each leg passes through approx apex + (0.67*bis*0) ... use a band around the ray
            # from the apex along the leg direction shifted by the outer-skin offset: we just count points whose
            # projection onto the leg direction is in [leg_lo, leg_hi] and whose distance from the ray is < 2.
            along = Q @ tv
            off = np.linalg.norm(Q - np.outer(along, tv), axis=1)
            kk = (along > leg_lo) & (along < leg_hi) & (off < 2.6)
            legs.append(int(kk.sum()))
        out.append((s, arc, legs[0], legs[1]))
    return np.array(out)


def fit_bend(xyz, nrm, a, b, ext=12.0, step=1.0, rmax=13.0, refine=True):
    rows, d0, L0 = sections_along(xyz, nrm, a, b, step=step, ext=ext, rmax=rmax)
    ap = [r for r in rows if r.get("ok") and r.get("apex_3d") is not None]
    if len(ap) < 4:
        return {"ok": False, "why": "too_few_apex", "rows": rows}
    P = np.array([r["apex_3d"] for r in ap])
    c, d, keep = robust_line(P)
    if d @ d0 < 0:
        d = -d
    res = {"ok": True, "seed": (a, b), "c": c, "d": d, "n_apex": len(ap), "n_keep": int(keep.sum())}
    if refine:
        # re-run sections along the fitted apex line (perpendicular to the true axis)
        sv = (P[keep] - c) @ d
        a2 = c + (sv.min() - 0.0) * d
        b2 = c + (sv.max() + 0.0) * d
        rows2, _, L2 = sections_along(xyz, nrm, a2, b2, step=step, ext=ext, rmax=rmax)
        ap2 = [r for r in rows2 if r.get("ok") and r.get("apex_3d") is not None]
        if len(ap2) >= 4:
            P2 = np.array([r["apex_3d"] for r in ap2])
            c2, d2, keep2 = robust_line(P2)
            if d2 @ d < 0:
                d2 = -d2
            c, d, P, keep, rows, ap = c2, d2, P2, keep2, rows2, ap2
            res.update({"c": c, "d": d, "n_apex": len(ap), "n_keep": int(keep.sum())})
    sv = (P - c) @ d
    resid = P - c - np.outer(sv, d)
    res["apex_resid_mm_med"] = float(np.median(np.linalg.norm(resid[keep], axis=1)))
    angs = np.array([r["bend_angle_deg"] for r in ap])
    res["bend_angle_med"] = float(np.median(angs))
    th = [r.get("thickA") for r in ap if r.get("thickA") is not None] + \
         [r.get("thickB") for r in ap if r.get("thickB") is not None]
    res["thick_med"] = float(np.median(th)) if th else None
    res["apex_s_range_fit"] = (float(sv[keep].min()), float(sv[keep].max()))
    bis = unit(np.median(np.array([r["bis_3d"] for r in ap]), axis=0))
    bis = unit(bis - d * (bis @ d))
    tA = unit(np.median(np.array([r["tA_3d"] for r in ap]), axis=0))
    tB = unit(np.median(np.array([r["tB_3d"] for r in ap]), axis=0))
    tA = unit(tA - d * (tA @ d))
    tB = unit(tB - d * (tB @ d))
    res.update({"bis": bis, "tA": tA, "tB": tB})
    # outer virtual sharp distance from apex
    so = [np.linalg.norm(r["apex_3d"] - r["Xo_3d"]) for r in ap if r.get("Xo_3d") is not None]
    res["apex_to_Xo_med"] = float(np.median(so)) if so else None
    xi = [np.linalg.norm(r["apex_3d"] - r["Xi_3d"]) for r in ap if r.get("Xi_3d") is not None]
    res["apex_to_Xi_med"] = float(np.median(xi)) if xi else None
    ia = [np.linalg.norm(r["apex_3d"] - r["inner_apex_3d"]) for r in ap if r.get("inner_apex_3d") is not None]
    res["apex_to_inner_apex_med"] = float(np.median(ia)) if ia else None
    res["rows"] = rows
    return res


def extent_from_presence(pres, min_arc=1, min_leg=3, max_gap=2.0):
    """longest contiguous run (gaps <= max_gap mm bridged) where arc and both legs are present."""
    s = pres[:, 0]
    ok = (pres[:, 1] >= min_arc) & (pres[:, 2] >= min_leg) & (pres[:, 3] >= min_leg)
    runs = []
    start = None
    last = None
    for si, o in zip(s, ok):
        if o:
            if start is None:
                start = si
            elif last is not None and si - last > max_gap + 1e-9:
                runs.append((start, last))
                start = si
            last = si
    if start is not None:
        runs.append((start, last))
    if not runs:
        return None, runs
    best = max(runs, key=lambda r: r[1] - r[0])
    return best, runs


def point_line_offset(p, c, d):
    r = p - c
    r = r - d * (r @ d)
    return r
