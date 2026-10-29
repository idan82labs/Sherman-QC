"""Auditor-2 independent fold measurement for sc_f2475286312b.

For every physical fold found in the scan (lips, wall, window tabs, wall ears) and every
2 mm station along it (scanning beyond the expected ends), fit the outer and inner skin
lines of both flanges in a +-0.6 mm slab, then locate:
  X  = outer virtual sharp (intersection of outer-skin lines)
  A  = outer apex (first scan point on the bisector ray from X into the concave side)
  Xi = inner virtual sharp, C = arc centre (from apex distance and included angle)
The fold "exists" at a station when both flanges are present within 6 mm of the corner
and the apex is found within 0..(expected+1.2) mm of X.  The gold line (if any) is
intersected with each station plane and expressed relative to A in the bisector frame
(+along = concave side).  Writes a2_folds.json.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
from a2_common import OUT, load_gold

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
q = np.load(OUT / 'q_cache.npy').astype(float)
G = {k: ((a - c) @ R.T, (b - c) @ R.T) for k, (a, b) in load_gold().items()}

# fold definitions in the plate frame (u=0, v=1, w=2)
FOLDS = {
    'F_M01_lip':   dict(ax=0, ij=(1, 2), C0=(-112.0, 0.0), a=(1, 0), b=(0, -1), rng=(-140, 145), gold=['M01']),
    'F_M02_lip':   dict(ax=0, ij=(1, 2), C0=(112.0, 0.0), a=(-1, 0), b=(0, -1), rng=(-140, 145), gold=['M02']),
    'F_wall_M07':  dict(ax=1, ij=(0, 2), C0=(126.5, 0.0), a=(-1, 0), b=(0, 1), rng=(-118, 118), gold=['M07']),
    'F_left_lip':  dict(ax=1, ij=(0, 2), C0=(-127.0, 0.0), a=(1, 0), b=(0, -1), rng=(-118, 118), gold=[]),
    'F_tab_bottom': dict(ax=0, ij=(1, 2), C0=(-81.0, 0.0), a=(-1, 0), b=(0, 1), rng=(-140, 145), gold=['M03', 'M04']),
    'F_tab_top':   dict(ax=0, ij=(1, 2), C0=(92.0, 0.0), a=(1, 0), b=(0, 1), rng=(-140, 145), gold=['M05', 'M06']),
    'F_ear_minus_v': dict(ax=2, ij=(0, 1), C0=(126.5, -113.0), a=(0, 1), b=(1, 0), rng=(-5, 65), gold=[]),
    'F_ear_plus_v':  dict(ax=2, ij=(0, 1), C0=(126.5, 113.0), a=(0, -1), b=(1, 0), rng=(-5, 65), gold=[]),
}
H = 0.6          # slab half width
T_NOM = 1.4


def perp(d):
    return np.array([-d[1], d[0]])


def fit_line(p):
    m = p.mean(0)
    _, _, vt = np.linalg.svd(p - m, full_matrices=False)
    return m, vt[0]


def intersect(p1, d1, p2, d2):
    M = np.array([d1, -d2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    s, _ = np.linalg.solve(M, p2 - p1)
    return p1 + s * d1


def skins(P, d, n_out, lo=4.0, hi=16.0):
    """Outer/inner skin lines of a flange running along d from the corner (local coords)."""
    al = P @ d
    off = P @ n_out
    m = (al > lo) & (al < hi) & (np.abs(off) < 4.5)
    if m.sum() < 12:
        return None
    o = off[m]
    # two clusters ~t apart: split at the largest gap / midpoint
    srt = np.sort(o)
    if srt[-1] - srt[0] < 0.6:      # one skin only
        mid = None
    else:
        mid = 0.5 * (np.percentile(o, 10) + np.percentile(o, 90))
    out_m = m.copy(); inn_m = m.copy()
    if mid is None:
        out_m &= True; inn_m &= False
    else:
        out_m &= off > mid; inn_m &= off <= mid
    res = {}
    for key, mm in (('outer', out_m), ('inner', inn_m)):
        if mm.sum() >= 6:
            pm, pd = fit_line(P[mm])
            if pd @ d < 0:
                pd = -pd
            # refine: drop residual outliers
            r = (P[mm] - pm) @ perp(pd)
            keep = np.abs(r - np.median(r)) < 0.25
            if keep.sum() >= 6:
                pm, pd = fit_line(P[mm][keep])
                if pd @ d < 0:
                    pd = -pd
            res[key] = (pm, pd, int(mm.sum()))
    return res


def analyse_station(P, a, b):
    """P local 2D coords relative to C0.  Returns dict or None."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na = perp(a); na = na if na @ b < 0 else -na      # convex side of flange a
    nb = perp(b); nb = nb if nb @ a < 0 else -nb
    # recentre on the actual corner: flange planes may sit a few mm from C0
    sa = skins(P, a, na); sb = skins(P, b, nb)
    if not sa or not sb or 'outer' not in sa or 'outer' not in sb:
        return None
    (pa, da, nA), (pb, db, nB) = sa['outer'], sb['outer']
    X = intersect(pa, da, pb, db)
    if X is None:
        return None
    phi = np.degrees(np.arccos(np.clip(da @ db, -1, 1)))   # included angle between flange dirs
    bis = da + db; bis /= np.linalg.norm(bis)               # into concave side
    pr = perp(bis)
    rel = P - X
    al = rel @ bis; ac = rel @ pr
    band = (np.abs(ac) < 0.35) & (al > -0.6) & (al < 6.0)
    apex = None; s_apex = None
    if band.sum() >= 2:
        s_sorted = np.sort(al[band])
        s_apex = float(np.mean(s_sorted[:2]))
        apex = X + s_apex * bis
    Xi = None
    if 'inner' in sa and 'inner' in sb:
        Xi = intersect(sa['inner'][0], sa['inner'][1], sb['inner'][0], sb['inner'][1])
    # flange presence near the corner: points within 1 mm of each outer line at 1..6 mm from X
    def near(pp, dd):
        r = P - pp
        dist = np.abs(r @ perp(dd))
        t = (P - X) @ dd
        return int(((dist < 1.0) & (t > 1.0) & (t < 6.0)).sum())
    out = dict(X=X, bis=bis, phi=phi, apex=apex, s_apex=s_apex, Xi=Xi,
               nA=nA, nB=nB, nearA=near(pa, da), nearB=near(pb, db), da=da, db=db)
    if apex is not None:
        half = np.radians(phi) / 2.0          # angle between each flange dir and the bisector
        # distance sharp->arc midpoint = R (1/sin(half) - 1) for the outer arc
        k = 1.0 / np.sin(half) - 1.0
        Ro = s_apex / k if k > 1e-6 else float('nan')
        out['R_outer'] = Ro
        out['C'] = X + bis * (Ro / np.sin(half))
    return out


def gold_at(gid, ax, s):
    A, B = G[gid]
    if abs(B[ax] - A[ax]) < 1e-9:
        return None, None
    t = (s - A[ax]) / (B[ax] - A[ax])
    return A + t * (B - A), t


results = {}
for name, F in FOLDS.items():
    ax = F['ax']; i, j = F['ij']; C0 = np.array(F['C0'])
    win = (np.abs(q[:, i] - C0[0]) < 22) & (np.abs(q[:, j] - C0[1]) < 22)
    Qw = q[win]
    order = np.argsort(Qw[:, ax]); Qw = Qw[order]
    keys = Qw[:, ax]
    rows = []
    for s in np.arange(F['rng'][0], F['rng'][1] + 1e-9, 2.0):
        lo, hi = np.searchsorted(keys, [s - H, s + H])
        S = Qw[lo:hi]
        row = dict(s=float(s), n=int(len(S)))
        if len(S) >= 30:
            P = S[:, [i, j]] - C0
            st = analyse_station(P, F['a'], F['b'])
        else:
            st = None
        if st is not None:
            exists = (st['apex'] is not None and st['nearA'] >= 4 and st['nearB'] >= 4
                      and -0.6 <= st['s_apex'] <= 3.5 and 70 <= st['phi'] <= 120)
            row.update(phi=round(st['phi'], 2), exists=bool(exists),
                       X=(st['X'] + C0).round(3).tolist(), bis=st['bis'].round(4).tolist(),
                       s_apex=None if st['s_apex'] is None else round(st['s_apex'], 3),
                       R_outer=None if st.get('R_outer') is None else round(float(st['R_outer']), 3),
                       nearA=st['nearA'], nearB=st['nearB'])
            if st['apex'] is not None:
                row['apex'] = (st['apex'] + C0).round(3).tolist()
            if st['Xi'] is not None:
                row['Xi'] = (st['Xi'] + C0).round(3).tolist()
            if st.get('C') is not None:
                row['C'] = (st['C'] + C0).round(3).tolist()
            for gid in F['gold']:
                g, t = gold_at(gid, ax, s)
                if g is None or not (0.0 <= t <= 1.0):
                    continue
                g2 = np.array([g[i], g[j]]) - C0
                ref = st['apex'] if st['apex'] is not None else st['X']
                off = g2 - ref
                row[gid] = dict(t=round(float(t), 4), g2=(g2 + C0).round(3).tolist(),
                                ref='apex' if st['apex'] is not None else 'X',
                                along=round(float(off @ st['bis']), 3),
                                across=round(float(off @ perp(st['bis'])), 3),
                                dist=round(float(np.linalg.norm(off)), 3),
                                d_X=round(float(np.linalg.norm(g2 - st['X'])), 3),
                                d_Xi=None if st['Xi'] is None else round(float(np.linalg.norm(g2 - st['Xi'])), 3),
                                d_C=None if st.get('C') is None else round(float(np.linalg.norm(g2 - st['C'])), 3))
        else:
            row['exists'] = False
        rows.append(row)
    ex = [r['s'] for r in rows if r.get('exists')]
    # contiguous runs of existence (allow 1-station gaps)
    runs = []
    for s in ex:
        if runs and s - runs[-1][1] <= 4.0 + 1e-9:
            runs[-1][1] = s
        else:
            runs.append([s, s])
    results[name] = dict(def_={k: v for k, v in F.items()}, rows=rows, exist_runs=runs)
    tot = sum(r1 - r0 for r0, r1 in runs)
    print(f'{name}: exist runs {runs} total ~{tot:.0f} mm')
    phis = [r['phi'] for r in rows if r.get('exists')]
    Ros = [r['R_outer'] for r in rows if r.get('exists') and r.get('R_outer')]
    if phis:
        print(f'   included angle median {np.median(phis):.1f} deg (bend {180-np.median(phis):.1f}), R_outer median {np.median(Ros):.2f}')
    for gid in F['gold']:
        al = [r[gid]['along'] for r in rows if r.get('exists') and gid in r]
        acr = [r[gid]['across'] for r in rows if r.get('exists') and gid in r]
        dd = [r[gid]['dist'] for r in rows if r.get('exists') and gid in r]
        if al:
            print(f'   {gid}: stations on fold {len(al)}  along med {np.median(al):.2f} [p5 {np.percentile(al,5):.2f}, p95 {np.percentile(al,95):.2f}]  across med {np.median(acr):.2f}  dist med {np.median(dd):.2f} max {np.max(dd):.2f} min {np.min(dd):.2f}')
json.dump(results, open(OUT / 'a2_folds.json', 'w'), indent=0, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
