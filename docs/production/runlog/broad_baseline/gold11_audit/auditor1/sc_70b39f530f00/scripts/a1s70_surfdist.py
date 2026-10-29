import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, trimesh
from scipy.spatial import cKDTree
m=load()
pts,fid=trimesh.sample.sample_surface(m,4000000,seed=5)
T=cKDTree(pts)
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
E=json.load(open(OUT+'a1_extent_lines.json'))
res={}
for k,(p,q) in GOLD.items():
    L=np.linalg.norm(q-p); t=np.linspace(0,1,int(L/0.25)+1)
    P=p[None]+t[:,None]*(q-p)[None]; d,_=T.query(P)
    s=P@a
    ss,se=sorted([p@a,q@a])
    # tails: distance at the endpoints and how much of the line is > 0.5 mm off the surface
    off=d>0.5
    res[k]=dict(len=float(L),d_start=float(d[0]),d_end=float(d[-1]),median=float(np.median(d)),p95=float(np.percentile(d,95)),
                frac_within_0p5=float((~off).mean()),off_surface_mm=float(off.sum()*0.25),
                s_of_off_points=[float(x) for x in s[off]])
    print(k,'len %.2f d_start %.2f d_end %.2f median %.2f p95 %.2f within0.5 %.1f%% off-surface length %.2f mm at s=%s'%(L,d[0],d[-1],np.median(d),np.percentile(d,95),100*(~off).mean(),off.sum()*0.25,np.round(s[off],1)))
json.dump(res,open(OUT+'a1_gold_surface_distance.json','w'),indent=1)
