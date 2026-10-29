"""Dense apex tracking for the 4 folds of 38172000 scan 1 + per-station gold offsets.
Frame: PCA of raw vertices (PC3 = plate normal). Sections are PC2-PC3 slabs at PC1 = x0 (+-0.5 mm);
gold lines are within 0.3 deg of PC1 so a PC1 slab is perpendicular to every gold line to cos(0.3deg)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
from scipy.spatial import cKDTree

V, F, N = load_mesh()
c, R = pca_frame(V)
P = (V - c) @ R.T
G = load_gold()
GP = {k: ((s - c) @ R.T, (e - c) @ R.T) for k, (s, e) in G.items()}

# fold configs: corner location guess (PC2), outward normals of the two OUTER faces (in PC2,PC3),
# face windows used to fit the outer/inner skin lines
FOLDS = {
  'L_FLANGE': dict(y0=-112.0, n_plate=(0, -1), n_leg=(-1, 0), leg_dir=(0, 1)),
  'R_FLANGE': dict(y0=111.5, n_plate=(0, -1), n_leg=(1, 0), leg_dir=(0, 1)),
  'L_TAB':    dict(y0=-82.0, n_plate=(0, 1), n_leg=(1, 0), leg_dir=(0, -1)),
  'R_TAB':    dict(y0=92.5, n_plate=(0, 1), n_leg=(-1, 0), leg_dir=(0, -1)),
}

def fit_line(pts):
    m = pts.mean(0); u, s, vt = np.linalg.svd(pts - m); return m, vt[0]

def inter(p1, d1, p2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9: return None
    s, _ = np.linalg.solve(A, p2 - p1); return p1 + s * d1

def section(x0, half=0.5):
    m = np.abs(P[:, 0] - x0) < half
    return P[m][:, 1:3]

def analyse(fold, x0, Q):
    cfg = FOLDS[fold]
    y0 = cfg['y0']; ld = np.array(cfg['leg_dir'], float)
    plate_side = -np.array(cfg['n_leg'], float)  # plate extends away from the leg's outer normal
    rel = Q - np.array([y0, -0.75])
    # leg (flange/tab) points: within 3 mm of y0 horizontally and 4..10 mm along leg direction
    along_leg = rel @ ld
    leg = Q[(np.abs(rel[:, 0]) < 3.5) & (along_leg > 4) & (along_leg < 10)]
    # plate points: 5..14 mm from corner on the plate side, |PC3+0.75| < 2.5
    along_plate = rel @ np.array([plate_side[0], 0])
    plate = Q[(along_plate > 5) & (along_plate < 14) & (np.abs(rel[:, 1]) < 2.5)]
    res = dict(x0=float(x0), n_leg=int(len(leg)), n_plate=int(len(plate)))
    if len(leg) < 30 or len(plate) < 30:
        res['bend'] = False; return res
    no = np.array(cfg['n_leg'], float); npl = np.array(cfg['n_plate'], float)
    # split skins by projection on outward normals
    def skins(pts, n):
        s = pts @ n; med = np.median(s)
        return pts[s > med], pts[s <= med]   # outer, inner
    lo, li = skins(leg, no); po, pi = skins(plate, npl)
    if min(len(lo), len(li), len(po), len(pi)) < 10:
        res['bend'] = False; return res
    m_lo, d_lo = fit_line(lo); m_li, d_li = fit_line(li); m_po, d_po = fit_line(po); m_pi, d_pi = fit_line(pi)
    Xo = inter(m_lo, d_lo, m_po, d_po); Xi = inter(m_li, d_li, m_pi, d_pi)
    # actual outward normals of the fitted outer lines
    def outn(d, nref):
        n = np.array([-d[1], d[0]]); return n if n @ nref > 0 else -n
    nlo = outn(d_lo, no); npo = outn(d_po, npl)
    b = unit(nlo + npo)
    ang_between = np.degrees(np.arccos(np.clip(abs(d_lo @ d_po), -1, 1)))
    bend_angle = 180 - ang_between if True else None
    # apex: extreme points along b within 6 mm of the outer sharp
    near = Q[np.linalg.norm(Q - Xo, axis=1) < 6]
    sb = near @ b
    top = near[sb >= sb.max() - 0.08]
    apex = top.mean(0)
    thick_leg = abs((m_lo - m_li) @ nlo); thick_plate = abs((m_po - m_pi) @ npo)
    res.update(bend=True, Xo=Xo.tolist(), Xi=Xi.tolist(), apex=apex.tolist(), b=b.tolist(),
               fold_angle_deg=float(ang_between), t_leg=float(thick_leg), t_plate=float(thick_plate),
               apex_to_Xo=float(np.linalg.norm(apex - Xo)))
    return res

def gold_at(k, x0):
    ps, pe = GP[k]; d = pe - ps; t = (x0 - ps[0]) / d[0]
    g = ps + t * d
    return t, g[1:3]

GOLD_FOR = {'L_FLANGE': ['M02', 'M03'], 'R_FLANGE': ['M05', 'M06'], 'L_TAB': ['M01'], 'R_TAB': ['M04']}
xs = np.arange(-170, 166, 2.0)
out = {}
for fold in FOLDS:
    rows = []
    for x0 in xs:
        Q = section(x0)
        if len(Q) < 50: continue
        r = analyse(fold, x0, Q)
        for k in GOLD_FOR[fold]:
            t, g = gold_at(k, x0)
            r.setdefault('gold', {})[k] = dict(t=float(t), in_span=bool(0 <= t <= 1), g=g.tolist())
            if r.get('bend'):
                apex = np.array(r['apex']); b = np.array(r['b']); Xo = np.array(r['Xo']); Xi = np.array(r['Xi'])
                dv = g - apex
                across = np.array([-b[1], b[0]])
                r['gold'][k].update(d_apex=float(np.linalg.norm(dv)), along_b=float(dv @ b), across=float(dv @ across),
                                    d_Xo=float(np.linalg.norm(g - Xo)), d_Xi=float(np.linalg.norm(g - Xi)))
            # nearest scan point to the gold point in this section
            dd = np.linalg.norm(Q - g, axis=1).min()
            r['gold'][k]['d_scan_section'] = float(dd)
        rows.append(r)
    out[fold] = rows
json.dump(dict(frame=dict(centroid=c.tolist(), R=R.tolist()), folds=out), open(OUT + '/a1_apex_track.json', 'w'), indent=1)

# summaries
for fold, rows in out.items():
    bx = [r['x0'] for r in rows if r.get('bend')]
    # contiguous runs
    runs = []; 
    for x in bx:
        if runs and abs(x - runs[-1][1]) <= 2.01: runs[-1][1] = x
        else: runs.append([x, x])
    print(f'\n== {fold}: bend present at PC1 runs (2 mm step): {runs}')
    ang = [r['fold_angle_deg'] for r in rows if r.get('bend')]
    a2x = [r['apex_to_Xo'] for r in rows if r.get('bend')]
    tl = [r['t_leg'] for r in rows if r.get('bend')]
    print(f'   fold angle med {np.median(ang):.2f} [{np.percentile(ang,5):.1f},{np.percentile(ang,95):.1f}]  apex->outer sharp med {np.median(a2x):.2f}  t_leg med {np.median(tl):.2f}')
    for k in GOLD_FOR[fold]:
        vals = [r['gold'][k] for r in rows if r.get('bend') and r['gold'][k]['in_span']]
        if not vals: print('  ', k, 'no in-span bend stations'); continue
        da = np.array([v['d_apex'] for v in vals]); ab = np.array([v['along_b'] for v in vals]); ac = np.array([v['across'] for v in vals])
        dxo = np.array([v['d_Xo'] for v in vals]); dxi = np.array([v['d_Xi'] for v in vals])
        print(f'   {k}: n={len(vals)} d_apex med {np.median(da):.2f} p90 {np.percentile(da,90):.2f} max {da.max():.2f} | along_b med {np.median(ab):+.2f} [{ab.min():+.2f},{ab.max():+.2f}] across med {np.median(ac):+.2f} [{ac.min():+.2f},{ac.max():+.2f}] | d_outer_sharp med {np.median(dxo):.2f}  d_inner_sharp med {np.median(dxi):.2f}')
