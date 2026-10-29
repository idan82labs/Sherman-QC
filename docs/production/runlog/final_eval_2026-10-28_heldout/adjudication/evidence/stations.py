"""Per-station check of bench lines on the raw scan: h = height along bisector from the plane sharp of the outermost
material near the bisector plane (|p|<pw). Formed arc: hmin ~ expected outer apex offset; sharp corner: hmin ~ 0.
usage: stations.py jobs.json out.json   jobs: [{"name","scan","lines_json","bend_ids":[...]}]"""
import json, sys, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
jobs=json.load(open(sys.argv[1])); out={}
cache={}
for j in jobs:
    if j['scan'] not in cache:
        cache.clear(); m=o3d.io.read_triangle_mesh(j['scan']); V=np.asarray(m.vertices).copy(); cache[j['scan']]=(V,cKDTree(V))
    V,tree=cache[j['scan']]
    L=json.load(open(j['lines_json']))
    for b in L['bends']:
        if b['bend_id'] not in j['bend_ids'] or b.get('status')!='ok': continue
        s=np.array(b['sharp']['start']); e=np.array(b['sharp']['end']); d=(e-s)/np.linalg.norm(e-s)
        bi=np.array(b['bisector']); bi=bi-d*(bi@d); bi/=np.linalg.norm(bi); p=np.cross(d,bi)
        Lm=np.linalg.norm(e-s); mid=(s+e)/2
        idx=tree.query_ball_point(mid, Lm/2+8)
        P=V[idx]-s; t=P@d; h=P@bi; q=P@p
        rows=[]
        for ti in np.arange(-3, Lm+3.01, 1.0):
            k=(np.abs(t-ti)<0.5)&(np.abs(q)<j.get('pw',0.6))&(h>-3)&(h<8)
            n=int(k.sum())
            rows.append([round(float(ti),1), n, round(float(np.percentile(h[k],2)),2) if n>=3 else None])
        apx=b.get('apex_offset_mm'); exp=b['expected_apex_offset_mm']['outer'] if b.get('expected_apex_offset_mm') else None
        hs=[r[2] for r in rows if r[2] is not None and 0<=r[0]<=Lm]
        out[f"{j['name']}:{b['bend_id']}"]={'L':round(Lm,1),'apex_offset':apx,'expected_outer':exp,
            'hmin_median':float(np.median(hs)) if hs else None,'rows':rows}
        print(j['name'],b['bend_id'],'L',round(Lm,1),'apx',round(apx,2),'exp',round(exp,2),'median hmin',round(float(np.median(hs)),2) if hs else None)
        print('   ',' '.join(f"{r[0]:.0f}:{r[2]}" for r in rows))
json.dump(out,open(sys.argv[2],'w'),indent=1)
