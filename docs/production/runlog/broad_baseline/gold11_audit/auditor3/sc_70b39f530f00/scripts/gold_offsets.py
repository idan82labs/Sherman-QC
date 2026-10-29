import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
from common import *

S = json.load(open(OUT + '/scripts/stations.json'))
rows = [r for r in S['stations'] if not r['empty']]
pairs = {'M01': 'bend3_Ap_F', 'M02': 'bend2_W_Ap', 'M03': 'bend1_A_W'}
res = {}
for gid, b in pairs.items():
    print('=====', gid, 'vs', b)
    recs = []
    for r in rows:
        g = r['gold'][gid]
        rec = r['bends'][b]
        if not g['inside'] or not rec['present_outer'] or 'apex_fit' not in rec['outer']:
            continue
        q = np.array(g['q'])
        ao = np.array(rec['outer']['apex_fit'])
        nb = np.array(rec['nbo'])
        tb = np.array([-nb[1], nb[0]])
        dv = q - ao
        line = dict(s=r['s'], d_outer_apex=float(np.linalg.norm(dv)), along_bis=float(dv @ nb), across=float(dv @ tb),
                    d_surface=g['d_surface_mid'])
        if rec['present_inner'] and 'apex_fit' in rec['inner']:
            ai = np.array(rec['inner']['apex_fit'])
            line['d_inner_apex'] = float(np.linalg.norm(q - ai))
            line['apex_sep'] = float(np.linalg.norm(ao - ai))
        Xo = np.array(rec['outer']['X']); Xi = np.array(rec['inner']['X'])
        line['d_outer_sharp'] = float(np.linalg.norm(q - Xo)); line['d_inner_sharp'] = float(np.linalg.norm(q - Xi))
        recs.append(line)
    for l in recs:
        print(f"s={l['s']:6.1f} |g-outerApex|={l['d_outer_apex']:5.2f} alongBis={l['along_bis']:+5.2f} across={l['across']:+5.2f} "
              f"|g-innerApex|={l.get('d_inner_apex', float('nan')):5.2f} apexSep={l.get('apex_sep', float('nan')):4.2f} "
              f"|g-Xo|={l['d_outer_sharp']:5.2f} |g-Xi|={l['d_inner_sharp']:5.2f} surf={l['d_surface']:4.2f}")
    a = np.array([l['d_outer_apex'] for l in recs]); al = np.array([l['along_bis'] for l in recs]); ac = np.array([l['across'] for l in recs])
    di = np.array([l['d_inner_apex'] for l in recs if 'd_inner_apex' in l])
    print(f"SUMMARY {gid}: n={len(recs)} outer-apex offset mean={a.mean():.2f} min={a.min():.2f} max={a.max():.2f} "
          f"along-bis mean={al.mean():+.2f} across mean={ac.mean():+.2f} (sd {ac.std():.2f}); inner-apex offset mean={di.mean():.2f} max={di.max():.2f}")
    res[gid] = recs
json.dump(res, open(OUT + '/scripts/gold_offsets.json', 'w'), indent=1)
