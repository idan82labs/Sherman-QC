#!/usr/bin/env python3
"""Auditor 2 (sc_152f1b1aacb2, Gold11 CG64C412CC73, part 40500000 SCAN2): read-only fold measurement.

Reuses the cached part-frame cloud (pf.npz written by a2_load.py + pf.py in the earlier, interrupted
run of this same audit).  P = (V - c) @ R.T, so V = P @ R + c.  e3 = base normal (walls rise to +e3),
e1 = wall/long-bend direction, e2 = across.

For every formed fold found in the scan it slices 1 mm stations along the fold axis (slab +-0.5 mm),
fits the two plates' skins locally in the 2-D section, and measures:
  X_o (outer virtual sharp = outer mold-line intersection), X_i (inner sharp), the outer-skin apex
  (arc midpoint on the bisector), the bend included angle, and whether the fold exists at that station.
Gold lines are intersected with each station plane and their offset from the outer apex is reported
(along the bisector: negative = toward X_o / outside the arc, positive = into the arc toward its centre).
Writes a2r_sections.json next to this script's parent folder.  No writes anywhere else.
"""
import sys, json, math
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree

S = sys.argv[1]            # scratch dir holding pf.npz
OUT = sys.argv[2]          # output json
d = np.load(S + '/pf.npz')
P, NP, GP, ids, R, c = d['P'], d['NP'], d['GP'], [str(x) for x in d['ids']], d['R'], d['c']
tree = cKDTree(P)

# fold definitions in the part frame.
#   axis: unit axis (part frame); o2: function mapping 3-D part-frame pts -> 2-D section coords q=(u,v)
#   with the nominal outer sharp at q=(0,0), plate A extending to +u, plate B extending to +v.
FOLDS = {
    'F_A_base|wall_e2neg': dict(ax=0, rng=(-95, 105), uv=lambda Q: np.c_[Q[:, 1] + 92.3, Q[:, 2] + 7.3],
                                 inv=lambda s, q: np.array([s, q[0] - 92.3, q[1] - 7.3]), gold='M01'),
    'F_B_base|wall_e2pos': dict(ax=0, rng=(-95, 105), uv=lambda Q: np.c_[92.2 - Q[:, 1], Q[:, 2] + 7.3],
                                 inv=lambda s, q: np.array([s, 92.2 - q[0], q[1] - 7.3]), gold='M02'),
    'F_C_wall_e2neg|top_flange': dict(ax=0, rng=(-70, 72), uv=lambda Q: np.c_[Q[:, 1] + 92.3, 36.0 - Q[:, 2]],
                                       inv=lambda s, q: np.array([s, q[0] - 92.3, 36.0 - q[1]]), gold='M07', half=1.0, minpts=10),
    'F_D_wall_e2pos|top_tabs': dict(ax=0, rng=(-70, 72), uv=lambda Q: np.c_[92.2 - Q[:, 1], 36.0 - Q[:, 2]],
                                     inv=lambda s, q: np.array([s, 92.2 - q[0], 36.0 - q[1]]), gold=None, half=1.0, minpts=10),
    'F_E1_base|end_tab_e1neg': dict(ax=1, rng=(-60, -12), uv=lambda Q: np.c_[Q[:, 0] + 105.2, Q[:, 2] + 7.3],
                                     inv=lambda s, q: np.array([q[0] - 105.2, s, q[1] - 7.3]), gold=None),
    'F_E2_base|end_tab_e1pos': dict(ax=1, rng=(-60, -12), uv=lambda Q: np.c_[107.7 - Q[:, 0], Q[:, 2] + 7.3],
                                     inv=lambda s, q: np.array([107.7 - q[0], s, q[1] - 7.3]), gold=None),
}


def two_means(o, iters=20):
    lo, hi = np.percentile(o, 5), np.percentile(o, 95)
    if hi - lo < 0.45:
        return [(float(np.median(o)), len(o))], np.zeros(len(o), int)
    a, b = lo, hi
    for _ in range(iters):
        lab = (np.abs(o - b) < np.abs(o - a)).astype(int)
        if lab.sum() == 0 or lab.sum() == len(o):
            break
        a, b = np.median(o[lab == 0]), np.median(o[lab == 1])
    if b - a < 0.45:
        return [(float(np.median(o)), len(o))], np.zeros(len(o), int)
    return [(float(a), int((lab == 0).sum())), (float(b), int((lab == 1).sum()))], lab


def fit_plate(q, along):  # along: 0 -> plate extends along u, 1 -> along v
    # robust line direction from PCA, sign aligned with the extension axis
    m = q.mean(0)
    w, V = np.linalg.eigh(np.cov((q - m).T))
    dvec = V[:, 1]
    ext = np.array([1.0, 0.0]) if along == 0 else np.array([0.0, 1.0])
    if dvec @ ext < 0:
        dvec = -dvec
    n = np.array([-dvec[1], dvec[0]])  # normal; orient toward the OTHER plate's extension direction
    other = np.array([0.0, 1.0]) if along == 0 else np.array([1.0, 0.0])
    if n @ other < 0:
        n = -n
    o = q @ n
    skins, lab = two_means(o)
    return dvec, n, skins


def intersect(d1, p1, d2, p2):
    A = np.c_[d1, -d2]
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def station(fold, s, half=0.5, win=12.0, dirs=None, minpts=25):
    ax = fold['ax']
    m = np.abs(P[:, ax] - s) < half
    Q = P[m]
    q = fold['uv'](Q)
    near = (np.abs(q[:, 0]) < win) & (np.abs(q[:, 1]) < win) & (q[:, 0] > -3) & (q[:, 1] > -3)
    q = q[near]
    res = dict(s=float(s), n=int(len(q)))
    A = q[(q[:, 0] > 3.5) & (q[:, 0] < 10) & (q[:, 1] > -1.5) & (q[:, 1] < 2.8)]
    B = q[(q[:, 1] > 3.5) & (q[:, 1] < 10) & (q[:, 0] > -1.5) & (q[:, 0] < 2.8)]
    res['nA'], res['nB'] = int(len(A)), int(len(B))
    if len(A) < minpts or len(B) < minpts:
        res['fold'] = False
        res['why'] = 'plate_missing'
        return res, q
    dA, nA, skA = fit_plate(A, 0)
    dB, nB, skB = fit_plate(B, 1)
    if dirs is not None:  # pass 2: fold-wide median plate directions, only skin offsets are local
        dA, dB = np.array(dirs[0]), np.array(dirs[1])
        nA = np.array([-dA[1], dA[0]]); nA = nA if nA[1] > 0 else -nA
        nB = np.array([-dB[1], dB[0]]); nB = nB if nB[0] > 0 else -nB
        skA = two_means(A @ nA)[0]
        skB = two_means(B @ nB)[0]
    res['skinsA'] = skA
    res['skinsB'] = skB
    # outer skin = min offset along the normal that points toward the other plate
    tA = skA[1][0] - skA[0][0] if len(skA) > 1 else None
    tB = skB[1][0] - skB[0][0] if len(skB) > 1 else None
    tt = [x for x in (tA, tB) if x is not None]
    t_nom = float(np.mean(tt)) if tt else 1.0
    # single visible skin: decide outer/inner from the nominal outer sharp at q=(0,0) (outer ~0, inner ~+t)
    def out_in(sk):
        if len(sk) > 1:
            return sk[0][0], sk[1][0], 'both'
        o = sk[0][0]
        if o > 0.5 * t_nom:
            return o - t_nom, o, 'inner_only'
        return o, o + t_nom, 'outer_only'
    oA_out, oA_in, visA = out_in(skA)
    oB_out, oB_in, visB = out_in(skB)
    res['skin_vis'] = [visA, visB]
    Xo = intersect(dA, nA * oA_out, dB, nB * oB_out)
    Xi = intersect(dA, nA * oA_in, dB, nB * oB_in)
    incl = math.degrees(math.acos(np.clip(dA @ dB, -1, 1)))
    b = dA + dB
    b /= np.linalg.norm(b)
    rel = q - Xo
    sb = rel @ b
    lat = rel @ np.array([-b[1], b[0]])
    band = (np.abs(lat) < 0.35) & (sb > -0.6) & (sb < 4.5)
    res.update(dict(dA=dA.tolist(), dB=dB.tolist(), Xo=Xo.tolist(), Xi=Xi.tolist(), incl_deg=incl,
                    t_A=tA, t_B=tB, bis=b.tolist()))
    if band.sum() < 3:
        res['fold'] = False
        res['why'] = 'no_arc_points_on_bisector'
        return res, q
    sv = np.sort(sb[band])
    s0 = sv[0]
    sel = sb[band][(sb[band] <= s0 + 0.25)]
    s_apex = float(np.median(sel))
    apex = Xo + s_apex * b
    half_ang = math.radians(incl) / 2.0
    k = 1.0 / math.sin(half_ang) - 1.0 if incl < 179 else float('nan')
    Ro = s_apex / k if k > 1e-6 else float('nan')
    # arc presence: outer surface points close to Xo region but off both outer lines by > 0.15 mm
    distA = np.abs(q @ nA - oA_out)
    distB = np.abs(q @ nB - oB_out)
    arcpts = (np.linalg.norm(rel, axis=1) < 3.5) & (distA > 0.12) & (distB > 0.12) & (sb < s_apex + 1.2)
    res.update(dict(s_apex=s_apex, apex=apex.tolist(), R_outer_est=Ro, n_arc=int(arcpts.sum())))
    res['fold'] = bool(s_apex < 2.5 and res['n_arc'] >= 3 and 60 < incl < 179)
    if not res['fold']:
        res['why'] = 'arc_test_failed'
    return res, q


def gold_at(gid, ax, s):
    g = GP[ids.index(gid)]
    a, bpt = g[0], g[1]
    if abs(bpt[ax] - a[ax]) < 1e-9:
        return None
    t = (s - a[ax]) / (bpt[ax] - a[ax])
    if t < -1e-6 or t > 1 + 1e-6:
        return None
    return a + t * (bpt - a)


def analyse():
    out = {}
    for name, F in FOLDS.items():
        rows = []
        lo, hi = F['rng']
        half, minpts = F.get('half', 0.5), F.get('minpts', 25)
        p1 = [station(F, s, half=half, minpts=minpts)[0] for s in np.arange(lo, hi + 1e-9, 1.0)]
        good = [r for r in p1 if r.get('fold') and r.get('skin_vis') == ['both', 'both']] or [r for r in p1 if r.get('fold')]
        dA = np.median(np.array([r['dA'] for r in good]), 0); dA /= np.linalg.norm(dA)
        dB = np.median(np.array([r['dB'] for r in good]), 0); dB /= np.linalg.norm(dB)
        F['dirs'] = (dA.tolist(), dB.tolist())
        for s in np.arange(lo, hi + 1e-9, 1.0):
            r, q = station(F, s, half=half, minpts=minpts, dirs=F['dirs'])
            if F['gold'] is not None:
                gp = gold_at(F['gold'], F['ax'], s)
                if gp is not None:
                    qg = F['uv'](gp[None, :])[0]
                    r['gold_q'] = qg.tolist()
                    r['gold_dist_scan'] = float(tree.query(gp)[0])
                    if 'apex' in r:
                        apex = np.array(r['apex'])
                        b = np.array(r['bis'])
                        dv = qg - apex
                        r['gold_minus_apex_bis'] = float(dv @ b)          # + into arc, - toward X_o
                        r['gold_minus_apex_lat'] = float(dv @ np.array([-b[1], b[0]]))
                        r['d_gold_apex'] = float(np.linalg.norm(dv))
                        r['d_gold_Xo'] = float(np.linalg.norm(qg - np.array(r['Xo'])))
                        r['d_gold_Xi'] = float(np.linalg.norm(qg - np.array(r['Xi'])))
            rows.append(r)
        out[name] = rows
    return out


if __name__ == '__main__':
    res = analyse()
    summ = {}
    for name, rows in res.items():
        F = FOLDS[name]
        fs = [r for r in rows if r.get('fold')]
        st = np.array([r['s'] for r in fs])
        # contiguous fold intervals (gap > 1.5 mm splits)
        segs = []
        if len(st):
            a = st[0]
            for i in range(1, len(st)):
                if st[i] - st[i - 1] > 1.5:
                    segs.append((float(a), float(st[i - 1])))
                    a = st[i]
            segs.append((float(a), float(st[-1])))
        apex3 = np.array([F['inv'](r['s'], np.array(r['apex'])) for r in fs]) if fs else np.zeros((0, 3))
        xo3 = np.array([F['inv'](r['s'], np.array(r['Xo'])) for r in fs]) if fs else np.zeros((0, 3))
        summ[name] = dict(n_fold_stations=len(fs), segments=segs,
                          incl_deg_median=float(np.median([r['incl_deg'] for r in fs])) if fs else None,
                          R_outer_median=float(np.median([r['R_outer_est'] for r in fs])) if fs else None,
                          s_apex_median=float(np.median([r['s_apex'] for r in fs])) if fs else None,
                          thickness_median=float(np.median([x for r in fs for x in (r['t_A'], r['t_B']) if x is not None])) if fs else None,
                          apex_pf_mean=apex3.mean(0).tolist() if len(apex3) else None,
                          apex_pf_std=apex3.std(0).tolist() if len(apex3) else None,
                          Xo_pf_mean=xo3.mean(0).tolist() if len(xo3) else None, plate_dirs=F.get('dirs'))
        if F['gold']:
            g = [r for r in fs if 'd_gold_apex' in r]
            if g:
                summ[name]['gold'] = dict(
                    gold_id=F['gold'], n=len(g),
                    d_apex_median=float(np.median([r['d_gold_apex'] for r in g])),
                    d_apex_max=float(np.max([r['d_gold_apex'] for r in g])),
                    bis_median=float(np.median([r['gold_minus_apex_bis'] for r in g])),
                    lat_median=float(np.median([r['gold_minus_apex_lat'] for r in g])),
                    d_Xo_median=float(np.median([r['d_gold_Xo'] for r in g])),
                    d_Xi_median=float(np.median([r['d_gold_Xi'] for r in g])),
                    gold_dist_scan_median=float(np.median([r['gold_dist_scan'] for r in g])))
    json.dump(dict(summary=summ, stations=res, R=R.tolist(), c=c.tolist()), open(OUT, 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
