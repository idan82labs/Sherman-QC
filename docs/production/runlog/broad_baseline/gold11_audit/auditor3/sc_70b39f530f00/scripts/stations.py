"""Per-station exact-section analysis of the three folds and the three gold lines.

For each axis station s (0.5 mm step) the mesh is cut exactly (trimesh mesh_plane, with faces).
Per fold: outer/inner skin arc segments are those whose face-normal angle (in the section plane)
lies strictly between the two adjoining plate skins' normal angles. Circle fits to those arcs
give the bend axis, radii, and the outer/inner apex (= centre + R * bisector normal). A station
counts as 'fold present' when the outer arc covers >= 70 % of the normal-angle range and both
plates appear next to the arc. Gold points are interpolated on each gold line at station s.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import trimesh
from common import *

m = load_mesh(); gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
PL = json.load(open(OUT + '/scripts/plate_planes.json'))
fn = m.face_normals


def plane(name):
    return np.array(PL[name]['n']), np.array(PL[name]['c'])


def line2d_at(name, s):
    """2D line (point, unit normal) of plate skin `name` in the section plane at station s."""
    n, c = plane(name)
    o = c0 + d * s
    # points x in plane: x = o + a u + b w ; n.(x-c)=0 -> a (n.u) + b (n.w) = n.(c-o)
    nu, nw = n @ u, n @ w
    k = n @ (c - o)
    nn = math.hypot(nu, nw)
    return np.array([nu, nw]) / nn, k / nn  # n2 . p = k2


def inter2(l1, l2):
    (n1, k1), (n2, k2) = l1, l2
    M = np.array([n1, n2])
    return np.linalg.solve(M, np.array([k1, k2]))


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def fit_circle(P):
    # algebraic (Kasa) then Gauss-Newton refine
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    cx, cy, cc = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(cc + cx ** 2 + cy ** 2, 1e-9))
    c = np.array([cx, cy])
    for _ in range(20):
        dv = P - c
        dist = np.linalg.norm(dv, axis=1)
        J = np.c_[-dv / dist[:, None], -np.ones(len(P))]
        res = dist - r
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + step[:2]; r = r + step[2]
        if np.linalg.norm(step) < 1e-7:
            break
    res = np.linalg.norm(P - c, axis=1) - r
    return c, r, float(np.sqrt((res ** 2).mean()))


BENDS = {
    # name: (outer skin plate1, outer skin plate2, inner plate1, inner plate2)
    'bend1_A_W': ('A_r1', 'W_r6', 'A_r2', 'W_r4'),
    'bend2_W_Ap': ('W_r4', 'Ap_r5', 'W_r6', 'Ap_r0'),
    'bend3_Ap_F': ('Ap_r5', 'F_r241', 'Ap_r0', 'F_r82'),
}
PHI = {k: v['phi_deg'] for k, v in PL.items()}

stations = np.round(np.arange(-46.0, 4.01, 0.5), 2)
rows = []
for s in stations:
    o = c0 + d * s
    segs, fidx = trimesh.intersections.mesh_plane(m, plane_normal=d, plane_origin=o, return_faces=True)
    if len(segs) == 0:
        rows.append(dict(s=float(s), empty=True)); continue
    P0 = np.c_[(segs[:, 0] - o) @ u, (segs[:, 0] - o) @ w]
    P1 = np.c_[(segs[:, 1] - o) @ u, (segs[:, 1] - o) @ w]
    mid = (P0 + P1) / 2
    L = np.linalg.norm(P1 - P0, axis=1)
    N = fn[fidx]
    phi = np.degrees(np.arctan2(N @ w, N @ u))
    row = dict(s=float(s), empty=False, bends={})
    for bname, (o1, o2, i1, i2) in BENDS.items():
        rec = {}
        for skin, (p1, p2) in (('outer', (o1, o2)), ('inner', (i1, i2))):
            l1, l2 = line2d_at(p1, s), line2d_at(p2, s)
            X = inter2(l1, l2)  # virtual sharp of this skin
            a1, a2 = PHI[p1], PHI[p2]
            span = angdiff(a2, a1)
            bis = a1 + span / 2
            nb = np.array([math.cos(math.radians(bis)), math.sin(math.radians(bis))])
            near = np.linalg.norm(mid - X, axis=1) < 9.0
            # fraction of the way from a1 to a2 for each segment normal
            f = angdiff(phi, a1) / span
            arc = near & (f > 0.06) & (f < 0.94)
            # plates present next to the arc (normal within 2 deg of plate)
            pl1 = near & (np.abs(angdiff(phi, a1)) < 2.0)
            pl2 = near & (np.abs(angdiff(phi, a2)) < 2.0)
            bins = np.zeros(20, bool)
            if arc.any():
                bi = np.clip((f[arc] * 20).astype(int), 0, 19); bins[bi] = True
            cov = bins[1:19].mean()
            r = dict(X=X.tolist(), bis_deg=bis, n_arc=int(arc.sum()), arc_cov=float(cov),
                     plate1=float(L[pl1].sum()), plate2=float(L[pl2].sum()))
            if arc.sum() >= 6:
                C, R, rms = fit_circle(np.r_[P0[arc], P1[arc]])
                apex_c = C + R * nb if skin == 'outer' else C + R * nb * -1 * -1
                # inner skin: the inner arc's 'apex' is the arc point on the bisector: C + R*nb_in where
                # nb_in points from centre toward the convex side, i.e. the same as the outer bisector.
                r.update(C=C.tolist(), R=R, fit_rms=rms)
                # direct apex: arc segment whose normal is closest to the bisector normal
                k = np.where(arc)[0][np.argmin(np.abs(angdiff(phi[arc], bis)))]
                r['apex_direct'] = mid[k].tolist(); r['apex_direct_dphi'] = float(angdiff(phi[k], bis))
            rec[skin] = r
        # unify apex definitions with the outer bisector direction (pointing to convex side)
        bo = math.radians(rec['outer']['bis_deg']); nbo = np.array([math.cos(bo), math.sin(bo)])
        for skin in ('outer', 'inner'):
            r = rec[skin]
            if 'C' in r:
                r['apex_fit'] = (np.array(r['C']) + r['R'] * nbo).tolist()
        present = (rec['outer']['arc_cov'] >= 0.7 and rec['outer']['plate1'] > 0.5 and rec['outer']['plate2'] > 0.5)
        present_in = (rec['inner']['arc_cov'] >= 0.7 and rec['inner']['plate1'] > 0.5 and rec['inner']['plate2'] > 0.5)
        rec['present_outer'] = bool(present); rec['present_inner'] = bool(present_in)
        rec['nbo'] = nbo.tolist()
        row['bends'][bname] = rec
    # gold points
    row['gold'] = {}
    for gid, (a, b) in gold.items():
        ta, tb = (a - c0) @ d, (b - c0) @ d
        t = (s - ta) / (tb - ta)
        p = a + (b - a) * t
        q = np.array([(p - o) @ u, (p - o) @ w])
        # nearest section point (surface) distance
        dseg = np.min(np.linalg.norm(mid - q, axis=1))
        row['gold'][gid] = dict(t=float(t), inside=bool(0 <= t <= 1), q=q.tolist(), d_surface_mid=float(dseg))
    rows.append(row)
json.dump(dict(stations=rows, bends=BENDS), open(OUT + '/scripts/stations.json', 'w'))
print('stations', len(rows))
