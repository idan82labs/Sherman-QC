"""Build auditor-measured outer-apex lines for every visible fold (from corners2.json), map to the raw scan frame,
and run the FROZEN matcher (scripts/evaluate_f1_against_manual_gold.py) human gold vs these apex lines."""
import numpy as np, sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
sys.path.insert(0, '/Users/idantal/ShermanQC/FullProject/scripts')
sys.dont_write_bytecode = True
from a3c_common import *
from evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds, match_lines
d = np.load(SCR/'a3c'/'pf.npz'); R, o = d['R'], d['o']
C = json.load(open(SCR/'a3c'/'corners2.json'))
def to_raw(P): return (np.asarray(P) + o) @ R
def runs(rows):
    out = []; cur = None
    for r in rows:
        if r['present'] and 'apex' in r:
            if cur is None: cur = [r]
            else: cur.append(r)
        elif r['present']:
            continue
        else:
            if cur: out.append(cur); cur = None
    if cur: out.append(cur)
    return out
def lift(key, r):
    s = r['s']; y, z = r['apex']
    return {'M01': (s, y, z), 'M02': (s, -y, z), 'M07': (s, y, z), 'TABS': (s, -y, z),
            'LIPpos': (y, s, z), 'LIPneg': (-y, s, z)}[key]
def fit3(P):
    keep = np.ones(len(P), bool)
    for _ in range(3):
        c = P[keep].mean(0); _, _, vt = np.linalg.svd(P[keep] - c); u = vt[0]
        res = np.linalg.norm((P - c) - np.outer((P - c) @ u, u), axis=1)
        keep = res < max(0.4, 3 * np.median(res))
    t = (P - c) @ u; return c, u, t   # span from every fold station; axis from inliers
lines = []
spec = [('M01', 'A_base_wall_pos_e2', 'merge'), ('M02', 'A_base_wall_neg_e2', 'merge'), ('M07', 'A_top_flange_pos_e2', 'merge'),
        ('TABS', 'A_tab_neg_e2', 'split'), ('LIPpos', 'A_lip_pos_e1', 'merge'), ('LIPneg', 'A_lip_neg_e1', 'merge')]
for key, name, mode in spec:
    rr = runs(C[key])
    groups = [sum(rr, [])] if mode == 'merge' else rr
    for gi, g in enumerate(groups):
        P = np.array([lift(key, r) for r in g])
        c, u, t = fit3(P)
        half = 0.25 if key.startswith('LIP') else 0.25
        a, b = c + u * (t.min() - half), c + u * (t.max() + half)
        res = np.linalg.norm((P - c) - np.outer((P - c) @ u, u), axis=1)
        nm = name if mode == 'merge' else f'{name}_{"A" if g[0]["s"] < 0 else "B"}'
        lines.append(dict(bend_id=nm, part_start=a.tolist(), part_end=b.tolist(), length_mm=float(np.linalg.norm(b - a)),
                          line_start=to_raw(a).tolist(), line_end=to_raw(b).tolist(), n_stations=len(g),
                          apex_fit_resid_med=float(np.median(res)), apex_fit_resid_max=float(res.max()),
                          runs=[[float(x[0]['s']), float(x[-1]['s'])] for x in (rr if mode == 'merge' else [g])]))
for L in lines:
    print('%-24s len %6.1f  part %s -> %s  resid med %.2f max %.2f  runs %s' % (L['bend_id'], L['length_mm'], np.round(L['part_start'], 2), np.round(L['part_end'], 2),
          L['apex_fit_resid_med'], L['apex_fit_resid_max'], L['runs']))
gold = json.load(open(GOLD))['bends']
th = Thresholds()
print('\npairwise frozen-matcher metrics (gold vs nearest auditor apex line):')
PM = {}
for gb in gold:
    G = np.array([gb['line_start'], gb['line_end']])
    best = None
    for L in lines:
        m = pair_metrics(G, np.array([L['line_start'], L['line_end']]))
        if best is None or (m['axis_angle_deg'] < 10 and m['transverse_offset_mm'] < best[1]['transverse_offset_mm']) or best[1]['axis_angle_deg'] >= 10:
            if m['axis_angle_deg'] < 10 or best is None: best = (L['bend_id'], m)
    PM[gb['bend_id']] = dict(nearest=best[0], **{k: round(v, 3) for k, v in best[1].items()}, match=is_match(best[1], th))
    print(gb['bend_id'], PM[gb['bend_id']])
M = match_lines(gold, [dict(bend_id=L['bend_id'], line_start=L['line_start'], line_end=L['line_end']) for L in lines], th)
print('\nmatch_lines(human gold, auditor apex lines):', {k: v for k, v in M.items() if k != 'matches'})
for m in M['matches']: print('   ', m)
# also: an apex line truncated to the gold span would be what an 'extent-agnostic' reviewer sees; report gold-on-fold fractions
json.dump(dict(apex_lines=lines, pair_metrics=PM, match=M), open(SCR/'a3c'/'lines_match.json', 'w'), indent=1)
