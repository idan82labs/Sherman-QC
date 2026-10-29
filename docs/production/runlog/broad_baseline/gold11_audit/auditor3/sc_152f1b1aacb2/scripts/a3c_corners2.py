"""Auditor-3 corner analysis v2: outer-skin envelope lines -> outer virtual sharp, outer apex (closest outer-skin
point to the outer sharp = outer arc midline for a symmetric 90-deg bend), inner sharp, arc centre; fold presence;
gold in-section offsets. Stations every 0.5 mm. Read-only on scan/gold."""
import numpy as np, sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
from a3c_sect import *

d = np.load(SCR/'a3c'/'pf.npz'); R, o = d['R'], d['o']
GOLD = {bid: (to_part(s, R, o), to_part(e, R, o)) for bid, s, e in load_gold()}

def gold_at(bid, axis, s):
    a, b = GOLD[bid]
    if abs(b[axis] - a[axis]) < 1e-6: return None
    t = (s - a[axis]) / (b[axis] - a[axis])
    return a + t * (b - a), (0 <= t <= 1)

def envelope_line(Q, along, outer, binw=0.5, want='outer'):
    """Fit the outer (or inner) skin of a plate: bin along `along`, take extreme along `outer` per bin."""
    if len(Q) < 10: return None
    a = Q @ along; h = Q @ outer
    bins = np.floor(a / binw).astype(int)
    pts = []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < 2: continue
        if want == 'outer':
            pts.append(Q[m][np.argmax(h[m])])
        else:
            if h[m].max() - h[m].min() < 0.5: continue   # only one skin present in this bin
            pts.append(Q[m][np.argmin(h[m])])
    if len(pts) < 5: return None
    P = np.array(pts)
    c, u = fit_line2(P)
    for _ in range(2):
        nrm = np.array([-u[1], u[0]]); r = np.abs((P - c) @ nrm)
        keep = r < max(0.25, 2.5 * np.median(r))
        if keep.sum() < 5: break
        c, u = fit_line2(P[keep])
    if u @ along < 0: u = -u
    return c, u

def corner(P, selA, alongA, outA, selB, alongB, outB):
    QA, QB = P[selA], P[selB]
    LAo = envelope_line(QA, alongA, outA); LBo = envelope_line(QB, alongB, outB)
    if LAo is None or LBo is None: return None
    so = intersect2(LAo[0], LAo[1], LBo[0], LBo[1])
    nA = np.array([-LAo[1][1], LAo[1][0]]); nA = nA if nA @ outA > 0 else -nA
    nB = np.array([-LBo[1][1], LBo[1][0]]); nB = nB if nB @ outB > 0 else -nB
    bis = -(nA + nB); bis /= np.linalg.norm(bis)
    ang = np.degrees(np.arccos(np.clip(abs(LAo[1] @ LBo[1]), 0, 1)))
    near = P[np.linalg.norm(P - so, axis=1) < 4.0]
    if len(near) < 5: return None
    apex = near[np.argmin(np.linalg.norm(near - so, axis=1))]
    perp = np.array([-bis[1], bis[0]])
    onbis = near[np.abs((near - so) @ perp) < 0.2]
    apex_bis = onbis[np.argmin((onbis - so) @ bis)] if len(onbis) else apex
    dap = float(np.linalg.norm(apex - so))
    LAi = envelope_line(QA, alongA, outA, want='inner'); LBi = envelope_line(QB, alongB, outB, want='inner')
    tA = float(abs((LAi[0] - LAo[0]) @ nA)) if LAi else None
    tB = float(abs((LBi[0] - LBo[0]) @ nB)) if LBi else None
    si = intersect2(LAi[0], LAi[1], LBi[0], LBi[1]) if (LAi and LBi) else None
    half = np.radians(ang) / 2 if ang > 1 else None
    # outer radius from sharp->apex distance for a bend of deflection `ang`: d = Ro*(1/sin(phi)-1), phi=(180-ang)/2
    phi = np.radians((180 - ang) / 2)
    Ro = dap / (1 / np.sin(phi) - 1) if ang > 5 else None
    centre = so + bis * (Ro / np.sin(phi)) if Ro else None
    return dict(sharp_o=so, apex=apex, apex_bis=apex_bis, bis=bis, angle=float(ang), sharp_to_apex=dap,
                Ro=float(Ro) if Ro else None, centre=centre, sharp_i=si, tA=tA, tB=tB)

def run(axis, stations, box, to2d, selA_fn, alongA, outA, selB_fn, alongB, outB, presence_fn, gold_ids):
    sl = Slicer(axis, box); rows = []
    for s in stations:
        P3 = sl.section(s)
        r = dict(s=float(s), present=False)
        if len(P3):
            P = to2d(P3); r['present'] = bool(presence_fn(P))
            if r['present']:
                c = corner(P, selA_fn(P), alongA, outA, selB_fn(P), alongB, outB)
                if c:
                    r.update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()})
            for gid in gold_ids:
                g = gold_at(gid, axis, s)
                if g is None: continue
                gp, inside = g; g2 = to2d(gp[None])[0]
                r['gold_' + gid] = g2.tolist(); r['gold_in_' + gid] = bool(inside)
                r['gold_scan_' + gid] = float(np.min(np.linalg.norm(P - g2, axis=1)))
                if 'apex' in r:
                    ap = np.array(r['apex']); bis = np.array(r['bis']); v = g2 - ap
                    r['off_apex_' + gid] = float(np.linalg.norm(v))
                    r['off_bis_' + gid] = float(v @ bis)                      # + = toward concave side (inside bend)
                    r['off_perp_' + gid] = float(v @ np.array([-bis[1], bis[0]]))
                    r['off_sharpO_' + gid] = float(np.linalg.norm(g2 - np.array(r['sharp_o'])))
                    if r.get('sharp_i') is not None: r['off_sharpI_' + gid] = float(np.linalg.norm(g2 - np.array(r['sharp_i'])))
                    if r.get('centre') is not None: r['off_centre_' + gid] = float(np.linalg.norm(g2 - np.array(r['centre'])))
        rows.append(r)
    return rows

def pres_p(P, wall_y):
    y, z = P[:, 0], P[:, 1]
    wall_low = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 2.0) & (z < 6.0)).sum()
    base = ((y > wall_y[0] - 14) & (y < wall_y[0] - 4) & (z > -1.5) & (z < 1.0)).sum()
    ear = ((y > wall_y[1] + 0.6) & (z > -1.5) & (z < 0.8)).sum()
    return wall_low >= 6 and base >= 8 and ear <= 2
def pres_top(P, wall_y):
    y, z = P[:, 0], P[:, 1]
    wall_hi = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 37) & (z < 40)).sum()
    fl = ((y > wall_y[0] - 13) & (y < wall_y[0] - 3) & (z > 41) & (z < 44)).sum()
    straight = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 43.4)).sum()
    return wall_hi >= 6 and fl >= 8 and straight <= 2
def pres_lip(P, xw):
    x, z = P[:, 0], P[:, 1]
    lip = ((x > xw[0]) & (x < xw[1]) & (z > 2.0) & (z < 5.0)).sum()
    base = ((x > xw[0] - 14) & (x < xw[0] - 4) & (z > -1.5) & (z < 1.0)).sum()
    beyond = ((x > xw[1] + 0.6) & (z > -1.5) & (z < 0.8)).sum()
    return lip >= 6 and base >= 8 and beyond <= 2

Y, Z = np.array([1.0, 0]), np.array([0, 1.0])
ST1 = np.arange(-112, 109, 0.5); ST2 = np.arange(18, 54, 0.25)
RES = {}
RES['M01'] = run(0, ST1, {1: (60, 110), 2: (-5, 30)}, lambda P3: P3[:, [1, 2]],
    lambda P: (P[:, 0] > 66) & (P[:, 0] < 86) & (np.abs(P[:, 1]) < 1.5), Y, -Z,
    lambda P: (P[:, 0] > 87) & (P[:, 0] < 93.5) & (P[:, 1] > 4.5) & (P[:, 1] < 28), Z, Y,
    lambda P: pres_p(P, (89.0, 92.5)), ['M01'])
RES['M02'] = run(0, ST1, {1: (-110, -60), 2: (-5, 30)}, lambda P3: np.c_[-P3[:, 1], P3[:, 2]],
    lambda P: (P[:, 0] > 68) & (P[:, 0] < 88) & (np.abs(P[:, 1]) < 1.5), Y, -Z,
    lambda P: (P[:, 0] > 89) & (P[:, 0] < 95) & (P[:, 1] > 4.5) & (P[:, 1] < 28), Z, Y,
    lambda P: pres_p(P, (91.0, 94.5)), ['M02'])
RES['M07'] = run(0, ST1, {1: (70, 110), 2: (20, 50)}, lambda P3: P3[:, [1, 2]],
    lambda P: (P[:, 0] > 87) & (P[:, 0] < 93.5) & (P[:, 1] > 24) & (P[:, 1] < 39), Z, Y,
    lambda P: (P[:, 0] > 77) & (P[:, 0] < 87.5) & (P[:, 1] > 40) & (P[:, 1] < 46.5), -Y, Z,
    lambda P: pres_top(P, (89.0, 92.5)), ['M07'])
RES['TABS'] = run(0, ST1, {1: (-110, -70), 2: (20, 50)}, lambda P3: np.c_[-P3[:, 1], P3[:, 2]],
    lambda P: (P[:, 0] > 89) & (P[:, 0] < 95) & (P[:, 1] > 24) & (P[:, 1] < 39), Z, Y,
    lambda P: (P[:, 0] > 79) & (P[:, 0] < 89.5) & (P[:, 1] > 40) & (P[:, 1] < 46.5), -Y, Z,
    lambda P: pres_top(P, (91.0, 94.5)), [])
RES['LIPpos'] = run(1, ST2, {0: (80, 115), 2: (-5, 15)}, lambda P3: P3[:, [0, 2]],
    lambda P: (P[:, 0] > 82) & (P[:, 0] < 100) & (np.abs(P[:, 1]) < 1.5), Y, -Z,
    lambda P: (P[:, 0] > 102.5) & (P[:, 0] < 106.5) & (P[:, 1] > 3) & (P[:, 1] < 9.5), Z, Y,
    lambda P: pres_lip(P, (102.5, 106.5)), [])
RES['LIPneg'] = run(1, ST2, {0: (-115, -80), 2: (-5, 15)}, lambda P3: np.c_[-P3[:, 0], P3[:, 2]],
    lambda P: (P[:, 0] > 85) & (P[:, 0] < 103) & (np.abs(P[:, 1]) < 1.5), Y, -Z,
    lambda P: (P[:, 0] > 105.5) & (P[:, 0] < 109.5) & (P[:, 1] > 3) & (P[:, 1] < 9.5), Z, Y,
    lambda P: pres_lip(P, (105.5, 109.5)), [])
json.dump(RES, open(SCR/'a3c'/'corners2.json', 'w'))

def runs(rows, key):
    out = []; cur = None
    for r in rows:
        if key(r):
            if cur is None: cur = [r['s'], r['s']]
            else: cur[1] = r['s']
        else:
            if cur: out.append(tuple(cur)); cur = None
    if cur: out.append(tuple(cur))
    return out
SUM = {}
for k, rows in RES.items():
    pr = runs(rows, lambda r: r['present']); fit = [r for r in rows if r['present'] and 'apex' in r]
    print('\n==', k, 'fold-present runs:', pr, ' fitted', len(fit), '/', sum(r['present'] for r in rows))
    if not fit: continue
    A = np.array([[r['sharp_to_apex'], r['angle'], r['Ro'] or np.nan, r['tA'] or np.nan, r['tB'] or np.nan] for r in fit])
    print('   sharp->apex med %.2f | plate angle med %.1f | Ro med %.2f | tA %.2f tB %.2f' % tuple(np.nanmedian(A, 0)))
    ap = np.array([r['apex'] for r in fit]); print('   apex locus (local 2D) median', np.round(np.median(ap, 0), 3), 'IQR', np.round(np.percentile(ap, 75, 0) - np.percentile(ap, 25, 0), 3))
    SUM[k] = dict(runs=pr, n_fit=len(fit), sharp_to_apex=float(np.median(A[:, 0])), angle=float(np.median(A[:, 1])), Ro=float(np.nanmedian(A[:, 2])),
                  apex_med=np.median(ap, 0).tolist())
    for gid in ['M01', 'M02', 'M07']:
        g = [r for r in fit if ('off_apex_' + gid) in r and r['gold_in_' + gid]]
        if not g: continue
        f = lambda key: np.array([r.get(key + gid, np.nan) for r in g])
        oa, ob, op, so, si, ce = f('off_apex_'), f('off_bis_'), f('off_perp_'), f('off_sharpO_'), f('off_sharpI_'), f('off_centre_')
        print('   gold %s on %d fitted fold stations: |gold-apex| med %.2f p90 %.2f max %.2f | along-bis(+in) med %.2f | perp med %.2f | to outer sharp %.2f | inner sharp %.2f | arc centre %.2f'
              % (gid, len(g), np.median(oa), np.percentile(oa, 90), oa.max(), np.median(ob), np.median(op), np.median(so), np.nanmedian(si), np.nanmedian(ce)))
        SUM[k]['gold'] = dict(id=gid, n=len(g), off_apex_med=float(np.median(oa)), off_apex_p90=float(np.percentile(oa, 90)), off_apex_max=float(oa.max()),
                              along_bis_med=float(np.median(ob)), perp_med=float(np.median(op)), to_outer_sharp_med=float(np.median(so)),
                              to_inner_sharp_med=float(np.nanmedian(si)), to_centre_med=float(np.nanmedian(ce)))
json.dump(SUM, open(SCR/'a3c'/'corners2_summary.json', 'w'), indent=1)
