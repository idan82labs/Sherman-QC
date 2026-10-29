"""Auditor 2 (independent): per-station bend-locus measurement for every Gold11 line of sc_71cf0b216f43.

Frame: refined part frame (e1 long, e2 short, e3 face normal) fitted from the scan's own planes.
For each gold line we name the physical fold it refers to (side x top/bottom), then every 5 mm along the
fold we cut a +-0.75 mm slab, fit the outer/inner skins of both legs from normal-filtered points,
build the outer/inner virtual sharps, locate the outer apex (outer-skin point nearest the outer virtual
sharp) and express the gold crossing point in local coords:
   u = depth inward from the WALL outer plane, v = depth inward from the FACE/FLANGE outer plane.
Outer apex ~ (0.29R, 0.29R); inner skin of wall: u ~ t; virtual sharp (0,0).
"""
import json, numpy as np
from scipy.spatial import cKDTree
from a2_frame import *

V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
tree = cKDTree(P)
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}

# fold definitions: axis index along the fold, (y,z) index for the section, wall outward normal (in y), leg2 outward normal (in z)
# wall position guess (y), leg2 level guess (z)
FOLDS = {
 'M01': dict(axis=1, y=0, wall_sign=-1, leg_sign=-1, y0=-149.0, z0=-19.7, name='e1-min short side, bottom bend (wall->short flange)'),
 'M03': dict(axis=1, y=0, wall_sign=-1, leg_sign=+1, y0=-149.0, z0=20.9, name='e1-min short side, top bend (wall->frame face)'),
 'M02': dict(axis=1, y=0, wall_sign=+1, leg_sign=-1, y0=163.9, z0=-19.7, name='e1-max short side, bottom bend (wall->short flange)'),
 'M04': dict(axis=1, y=0, wall_sign=+1, leg_sign=+1, y0=163.9, z0=20.9, name='e1-max short side, top bend (wall->frame face)'),
 'M05': dict(axis=0, y=1, wall_sign=-1, leg_sign=+1, y0=-99.2, z0=20.9, name='e2-min long side, top bend (wall->frame face)'),
 'M06': dict(axis=0, y=1, wall_sign=-1, leg_sign=-1, y0=-99.2, z0=-21.4, name='e2-min long side, bottom bend (wall->joggled flange)'),
 'M07': dict(axis=0, y=1, wall_sign=+1, leg_sign=+1, y0=106.8, z0=20.9, name='e2-max long side, top bend (wall->frame face)'),
 'M08': dict(axis=0, y=1, wall_sign=+1, leg_sign=-1, y0=106.8, z0=-21.6, name='e2-max long side, bottom bend (wall->joggled flange)'),
}

def section(fd, s, half=0.75, win=16.0):
    ax, yi = fd['axis'], fd['y']
    m = np.abs(P[:, ax]-s) < half
    m &= np.abs(P[:, yi]-fd['y0']) < win
    m &= np.abs(P[:, 2]-fd['z0']) < win
    return P[m][:, [yi, 2]], Nn[m][:, [yi, 2]], np.abs(Nn[m][:, ax])

def measure_station(fd, s):
    Q, Qn, nax = section(fd, s)
    if len(Q) < 40: return None
    ws, ls = fd['wall_sign'], fd['leg_sign']
    nw = np.array([ws, 0.0]); nl = np.array([0.0, ls])   # outward normals (y,z)
    inw = -nw; inl = -nl                                  # inward directions
    y0, z0 = fd['y0'], fd['z0']
    # local coords relative to guess corner: a = inward from wall plane guess, b = inward from leg plane guess
    a = (Q - [y0, z0]) @ inw; b = (Q - [y0, z0]) @ inl
    ok = nax < 0.5
    # wall outer skin: normal ~ nw, on wall leg (b between 5 and 15)
    wo = ok & (Qn @ nw > 0.95) & (b > 5) & (b < 15) & (a > -4) & (a < 4)
    wi = ok & (Qn @ inw > 0.95) & (b > 5) & (b < 15) & (a > -3) & (a < 5)
    lo = ok & (Qn @ nl > 0.95) & (a > 5) & (a < 15) & (b > -4) & (b < 4)
    li = ok & (Qn @ inl > 0.95) & (a > 5) & (a < 15) & (b > -3) & (b < 5)
    if wo.sum() < 8 or lo.sum() < 8: return dict(s=float(s), fit_ok=False, n=int(len(Q)), n_wo=int(wo.sum()), n_lo=int(lo.sum()))
    a_wo = float(np.median(a[wo])); b_lo = float(np.median(b[lo]))
    a_wi = float(np.median(a[wi])) if wi.sum() >= 8 else None
    b_li = float(np.median(b[li])) if li.sum() >= 8 else None
    # re-centre on outer virtual sharp
    VSo = np.array([y0, z0]) + inw * a_wo + inl * b_lo
    u = (Q - VSo) @ inw; v = (Q - VSo) @ inl
    # outer arc points: normals between nw and nl, near the corner
    arc = ok & (Qn @ nw > 0.17) & (Qn @ nl > 0.17) & (u > -1.0) & (v > -1.0) & (u < 5) & (v < 5)
    if arc.sum() < 5: return dict(s=float(s), fit_ok=False, n=int(len(Q)), note='no outer arc points', a_wo=a_wo, b_lo=b_lo)
    d = np.hypot(u[arc], v[arc]); j = np.argmin(d)
    apex_uv = np.array([u[arc][j], v[arc][j]]); apex_d = float(d[j])
    R_o = apex_d / (np.sqrt(2)-1)
    t_w = (a_wi - a_wo) if a_wi is not None else None
    t_l = (b_li - b_lo) if b_li is not None else None
    return dict(s=float(s), fit_ok=True, n=int(len(Q)), VSo=VSo.tolist(), apex_uv=apex_uv.tolist(), apex_to_VSo=apex_d, R_outer_est=R_o,
                thick_wall=t_w, thick_leg=t_l, n_arc=int(arc.sum()))

def gold_point(k, s):
    A, B = GL[k]; ax = FOLDS[k]['axis']
    t = (s - A[ax]) / (B[ax]-A[ax])
    return A + t*(B-A), t

def to_uv(fd, st, Pt):
    yi = fd['y']; inw = np.array([-fd['wall_sign'], 0.0]); inl = np.array([0.0, -fd['leg_sign']])
    q = np.array([Pt[yi], Pt[2]]) - np.array(st['VSo'])
    return float(q @ inw), float(q @ inl)

