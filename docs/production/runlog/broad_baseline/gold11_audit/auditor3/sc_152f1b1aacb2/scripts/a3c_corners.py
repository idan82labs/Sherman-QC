"""Per-station 90-deg corner analysis for every horizontal-axis fold candidate (auditor 3).
For each station: fold present?, outer/inner virtual sharp, outer apex (outer-skin arc midline),
arc radius, bend angle, and the gold line's in-section offset from the apex."""
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

def skin_lines(P, sel, outer_dir):
    """P: (n,2) local section points. sel: mask of plate points. outer_dir: 2-vector pointing to the convex/outer side.
    returns (c_outer,u_outer),(c_inner,u_inner),thickness, n"""
    Q = P[sel]
    if len(Q) < 12: return None
    c, u = fit_line2(Q); nrm = np.array([-u[1], u[0]])
    if nrm @ outer_dir < 0: nrm = -nrm
    dist = (Q - c) @ nrm
    out, inn = Q[dist > 0], Q[dist < 0]
    if len(out) < 5 or len(inn) < 5: return None
    co, uo = fit_line2(out); ci, ui = fit_line2(inn)
    t = abs((co - ci) @ nrm)
    return (co, uo), (ci, ui), t, nrm

def corner(P, selA, outA, selB, outB, near_box):
    """Fit both plates and return corner metrics in local 2D coords."""
    A = skin_lines(P, selA, outA); B = skin_lines(P, selB, outB)
    if A is None or B is None: return None
    (cAo, uAo), (cAi, uAi), tA, nA = A
    (cBo, uBo), (cBi, uBi), tB, nB = B
    try:
        so = intersect2(cAo, uAo, cBo, uBo); si = intersect2(cAi, uAi, cBi, uBi)
    except np.linalg.LinAlgError:
        return None
    inc = np.degrees(np.arccos(abs(uAo @ uBo)))   # angle between plate lines (0..90)
    bis = si - so; bis = bis / (np.linalg.norm(bis) + 1e-12)
    # contour points near the corner
    dd = np.linalg.norm(P - so, axis=1)
    near = P[dd < near_box]
    if len(near) < 5: return None
    # outer apex = contour point near bisector closest to outer sharp
    perp = np.abs((near - so) @ np.array([-bis[1], bis[0]]))
    along = (near - so) @ bis
    cand = near[(perp < 0.35) & (along > -1.0)]
    if len(cand) == 0: return None
    apex = cand[np.argmin(np.linalg.norm(cand - so, axis=1))]
    # outer arc points: outer side of both inner-sharp... take points within R window near apex, on outer skin side
    return dict(sharp_o=so, sharp_i=si, apex=apex, bis=bis, angle_between_lines=inc, tA=tA, tB=tB,
                sharp_to_apex=float(np.linalg.norm(apex - so)), n_near=int(len(near)))

def run(name, axis, stations, box, to2d, selA_fn, outA, selB_fn, outB, presence_fn, gold_ids, near_box=6.0):
    sl = Slicer(axis, box)
    rows = []
    for s in stations:
        P3 = sl.section(s)
        if len(P3) == 0:
            rows.append(dict(s=float(s), present=False)); continue
        P = to2d(P3)
        pres = presence_fn(P)
        r = dict(s=float(s), present=bool(pres))
        if pres:
            c = corner(P, selA_fn(P), outA, selB_fn(P), outB, near_box)
            if c is not None:
                r.update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()})
        for gid in gold_ids:
            g = gold_at(gid, axis, s)
            if g is None: continue
            gp, inside = g
            g2 = to2d(gp[None])[0]
            r['gold_' + gid] = g2.tolist(); r['gold_in_' + gid] = bool(inside)
            # nearest scan contour point to gold in this section
            r['gold_scan_' + gid] = float(np.min(np.linalg.norm(P - g2, axis=1)))
            if 'apex' in r:
                ap = np.array(r['apex']); bis = np.array(r['bis'])
                v = g2 - ap
                r['off_apex_' + gid] = float(np.linalg.norm(v))
                r['off_apex_along_bis_' + gid] = float(v @ bis)          # + = toward concave/inner side
                r['off_apex_perp_' + gid] = float(v @ np.array([-bis[1], bis[0]]))
                so = np.array(r['sharp_o']); si = np.array(r['sharp_i'])
                r['off_sharpO_' + gid] = float(np.linalg.norm(g2 - so))
                r['off_sharpI_' + gid] = float(np.linalg.norm(g2 - si))
        rows.append(r)
    return rows

RES = {}
ST1 = np.arange(-112, 109, 1.0)
# ---------- M01: +e2 base->wall (bend axis e1). local (y,z)=(e2,e3); base outer = -z, wall outer = +y
def pres_p(P, wall_y=(89.0, 92.5), sgn=1):
    """fold present: wall material just above the arc (z 2..6), base inboard, and no flat ear running out past the wall."""
    y, z = P[:, 0], P[:, 1]
    wall_low = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 2.0) & (z < 6.0)).sum()
    base = ((y > wall_y[0] - 14) & (y < wall_y[0] - 4) & (z > -1.5) & (z < 1.0)).sum()
    ear = ((y > wall_y[1] + 0.6) & (z > -1.5) & (z < 0.8)).sum()
    return wall_low >= 6 and base >= 8 and ear <= 2
RES['M01_fold'] = run('M01', 0, ST1, {1: (60, 110), 2: (-5, 30)}, lambda P3: P3[:, [1, 2]],
    lambda P: (P[:, 0] > 66) & (P[:, 0] < 85) & (np.abs(P[:, 1]) < 1.5), np.array([0, -1.0]),
    lambda P: (P[:, 0] > 87) & (P[:, 0] < 93.5) & (P[:, 1] > 5) & (P[:, 1] < 28), np.array([1.0, 0]),
    pres_p, ['M01'])
# ---------- M02: -e2 base->wall. local (y,z)=(-e2,e3)
RES['M02_fold'] = run('M02', 0, ST1, {1: (-110, -60), 2: (-5, 30)}, lambda P3: np.c_[-P3[:, 1], P3[:, 2]],
    lambda P: (P[:, 0] > 68) & (P[:, 0] < 87) & (np.abs(P[:, 1]) < 1.5), np.array([0, -1.0]),
    lambda P: (P[:, 0] > 89) & (P[:, 0] < 95) & (P[:, 1] > 5) & (P[:, 1] < 28), np.array([1.0, 0]),
    lambda P: pres_p(P, (91.0, 94.5)), ['M02'])
# ---------- M07: +e2 wall -> top flange (folds toward -e2). local (y,z)=(e2,e3); wall outer +y, flange outer +z
def pres_top(P, wall_y):
    """flange present: wall just below the arc (z 37..40), flange inboard at z 41..44, and the wall does not run straight up past the flange top."""
    y, z = P[:, 0], P[:, 1]
    wall_hi = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 37) & (z < 40)).sum()
    fl = ((y > wall_y[0] - 13) & (y < wall_y[0] - 3) & (z > 41) & (z < 44)).sum()
    straight = ((y > wall_y[0]) & (y < wall_y[1]) & (z > 43.4)).sum()
    return wall_hi >= 6 and fl >= 8 and straight <= 2
RES['M07_fold'] = run('M07', 0, ST1, {1: (70, 110), 2: (20, 50)}, lambda P3: P3[:, [1, 2]],
    lambda P: (P[:, 0] > 87) & (P[:, 0] < 93.5) & (P[:, 1] > 22) & (P[:, 1] < 38), np.array([1.0, 0]),
    lambda P: (P[:, 0] > 77) & (P[:, 0] < 87) & (P[:, 1] > 40) & (P[:, 1] < 46.5), np.array([0, 1.0]),
    lambda P: pres_top(P, (89.0, 92.5)), ['M07'])
# ---------- -e2 wall top tabs (no gold). local (y,z)=(-e2,e3)
RES['TABS_fold'] = run('TABS', 0, ST1, {1: (-110, -70), 2: (20, 50)}, lambda P3: np.c_[-P3[:, 1], P3[:, 2]],
    lambda P: (P[:, 0] > 89) & (P[:, 0] < 95) & (P[:, 1] > 22) & (P[:, 1] < 38), np.array([1.0, 0]),
    lambda P: (P[:, 0] > 79) & (P[:, 0] < 89) & (P[:, 1] > 40) & (P[:, 1] < 46.5), np.array([0, 1.0]),
    lambda P: pres_top(P, (91.0, 94.5)), [])
# ---------- end lips (bend axis e2). local (x,z)=(+-e1,e3)
ST2 = np.arange(18, 54, 0.5)
def pres_lip(P, xw):
    x, z = P[:, 0], P[:, 1]
    lip = ((x > xw[0]) & (x < xw[1]) & (z > 2.0) & (z < 5.0)).sum()
    base = ((x > xw[0] - 14) & (x < xw[0] - 4) & (z > -1.5) & (z < 1.0)).sum()
    beyond = ((x > xw[1] + 0.6) & (z > -1.5) & (z < 0.8)).sum()
    return lip >= 6 and base >= 8 and beyond <= 2
RES['LIPpos_fold'] = run('LIP+', 1, ST2, {0: (80, 115), 2: (-5, 15)}, lambda P3: P3[:, [0, 2]],
    lambda P: (P[:, 0] > 82) & (P[:, 0] < 99) & (np.abs(P[:, 1]) < 1.5), np.array([0, -1.0]),
    lambda P: (P[:, 0] > 102.5) & (P[:, 0] < 106.5) & (P[:, 1] > 3) & (P[:, 1] < 9), np.array([1.0, 0]),
    lambda P: pres_lip(P, (102.5, 106.5)), [])
RES['LIPneg_fold'] = run('LIP-', 1, ST2, {0: (-115, -80), 2: (-5, 15)}, lambda P3: np.c_[-P3[:, 0], P3[:, 2]],
    lambda P: (P[:, 0] > 85) & (P[:, 0] < 102) & (np.abs(P[:, 1]) < 1.5), np.array([0, -1.0]),
    lambda P: (P[:, 0] > 105.5) & (P[:, 0] < 109.5) & (P[:, 1] > 3) & (P[:, 1] < 9), np.array([1.0, 0]),
    lambda P: pres_lip(P, (105.5, 109.5)), [])
json.dump(RES, open(SCR/'a3c'/'corners.json', 'w'))

def runs(rows, key='present'):
    out = []; cur = None
    for r in rows:
        if r.get(key) and 'apex' in r:
            if cur is None: cur = [r['s'], r['s']]
            else: cur[1] = r['s']
        else:
            if cur: out.append(tuple(cur)); cur = None
    if cur: out.append(tuple(cur))
    return out

for k, rows in RES.items():
    fr = runs(rows)
    print('\n==', k, 'fold-present station runs:', fr)
    ok = [r for r in rows if 'apex' in r]
    if not ok: continue
    A = np.array([[r['sharp_to_apex'], r['angle_between_lines'], r['tA'], r['tB']] for r in ok])
    print('   sharp->apex med %.2f  plate-line angle med %.1f (90=square)  tA %.2f tB %.2f' % tuple(np.median(A, 0)))
    for gid in ['M01', 'M02', 'M07']:
        g = [r for r in ok if ('off_apex_' + gid) in r and r['gold_in_' + gid]]
        if not g: continue
        oa = np.array([r['off_apex_' + gid] for r in g]); ab = np.array([r['off_apex_along_bis_' + gid] for r in g])
        pp = np.array([r['off_apex_perp_' + gid] for r in g]); so = np.array([r['off_sharpO_' + gid] for r in g])
        si = np.array([r['off_sharpI_' + gid] for r in g])
        print('   gold %s on %d fold stations: |gold-apex| med %.2f p90 %.2f max %.2f; along-bisector(+in) med %.2f; perp med %.2f; |gold-outer sharp| med %.2f; |gold-inner sharp| med %.2f'
              % (gid, len(g), np.median(oa), np.percentile(oa, 90), oa.max(), np.median(ab), np.median(pp), np.median(so), np.median(si)))
