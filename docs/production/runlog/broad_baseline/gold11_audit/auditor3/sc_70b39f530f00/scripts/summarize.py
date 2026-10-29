import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
from common import *

S = json.load(open(OUT + '/scripts/stations.json'))
rows = [r for r in S['stations'] if not r['empty']]
for b in S['bends']:
    print('=====', b)
    for r in rows:
        rec = r['bends'][b]
        o, i = rec['outer'], rec['inner']
        line = f"s={r['s']:6.1f} out cov={o['arc_cov']:.2f} n={o['n_arc']:3d} p1={o['plate1']:5.1f} p2={o['plate2']:5.1f} pres={int(rec['present_outer'])}"
        if 'R' in o:
            line += f" Ro={o['R']:5.2f} rms={o['fit_rms']:.3f} apex={np.round(o['apex_fit'],2)}"
        line += f" | in cov={i['arc_cov']:.2f} pres={int(rec['present_inner'])}"
        if 'R' in i:
            line += f" Ri={i['R']:5.2f} apex={np.round(i['apex_fit'],2)}"
        print(line)
