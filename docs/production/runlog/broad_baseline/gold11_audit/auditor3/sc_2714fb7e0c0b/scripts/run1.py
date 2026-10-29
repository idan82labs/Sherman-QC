import sys, numpy as np, json, math
sys.path.insert(0,'/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714')
from xs import *
from scipy.spatial import cKDTree
tree=cKDTree(V)
out={}
for bid,(A,B) in GOLD.items():
    T=Tube(A,B,rw=35,ext=40)
    L=T.L
    rows=[]
    ref=None
    # mid fit first to get ref dirs
    for frac in [0.5,0.45,0.55,0.4,0.6,0.3,0.7]:
        uv,nuv,na=T.slab(frac*L)
        r=fit_section(uv,nuv,na)
        if r is not None:
            ref=[math.degrees(math.atan2(l['n'][1],l['n'][0]))%180 for l in r['legs']]; break
    print('====',bid,'L=%.1f'%L,'ref normals',None if ref is None else [round(x,1) for x in ref])
    for frac in [0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98]:
        s=frac*L
        uv,nuv,na=T.slab(s)
        r=fit_section(uv,nuv,na,ref_dirs=ref)
        g3=A+(B-A)*frac
        dn=tree.query(g3)[0]
        if r is None:
            print(f'  t={frac:.2f} fit FAILED  n={len(uv)} dnear={dn:.2f}'); rows.append(dict(t=frac,fit=False,dnear=dn)); continue
        o=offsets(r,np.zeros(2))
        th=[None if x is None else round(x,2) for x in r['thick']]
        print(f"  t={frac:.2f} ang={r['bend_angle']:.1f} apex_along_concave(gold-apex)={o.get('along_concave',float('nan')):+.2f} across={o.get('across',float('nan')):+.2f} d_apex={o.get('d_apex',float('nan')):.2f} d_Xo={o['d_Xo']:.2f} (Xo_along={o['Xo_along_concave']:+.2f}) d_Xi={o.get('d_Xi',float('nan')):.2f} apex_to_Xo={r['s_apex'] if r['s_apex'] is not None else float('nan'):.2f} thick={th} dnear={dn:.2f} legs_n={[l['npts'] for l in r['legs']]}")
        rows.append(dict(t=frac,fit=True,angle=r['bend_angle'],dnear=dn,**o,apex_to_Xo=r['s_apex'],thick=th))
    out[bid]=dict(L=L,ref=ref,rows=rows)
json.dump(out,open(W+'run1.json','w'),indent=1,default=float)
