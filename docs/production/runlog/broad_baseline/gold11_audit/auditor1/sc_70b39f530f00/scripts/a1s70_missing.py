import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
from a1s70_render import Splat, draw_line
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
TH=np.degrees(np.arctan2(FN@e2,FN@e1)); NA=FN@a
def wrap(d): return (d+180)%360-180
flats={'strip':-9.43,'mid':-9.73,'incl':-48.36,'frame':-110.54}
# 1) skin faces: normal within 4 deg of +-flat normals and perpendicular to axis
perp=np.abs(NA)<0.07
skin=np.zeros(len(FN),bool)
for t in flats.values():
    skin|=perp&((np.abs(wrap(TH-t))<4)|(np.abs(wrap(TH-t-180))<4))
# 2) bend arcs: perpendicular to axis and normal between two consecutive flats (either orientation)
arc=np.zeros(len(FN),bool)
for t1,t2 in [(-9.43,-48.36),(-48.36,-9.73),(170.27,249.46)]:
    for off in (0,180):
        d=wrap(t2-t1); f=wrap(TH-t1-off)/d
        arc|=perp&(f>-0.05)&(f<1.05)
# 3) edge walls: normal lies within its local plate plane -> |n . n_plate| small for the nearest plate; approximate: |n . n_flat| < 0.2 for some flat
nf={k:np.cos(np.radians(t))*e1+np.sin(np.radians(t))*e2 for k,t in flats.items()}
edge=np.zeros(len(FN),bool)
for k,v in nf.items():
    edge|=np.abs(FN@v)<0.2
unexpl=~(skin|arc|edge)
print('area total %.0f skin %.0f arc-only %.0f edge-only %.0f unexplained %.0f'%(FA.sum(),FA[skin].sum(),FA[arc&~skin].sum(),FA[edge&~skin&~arc].sum(),FA[unexpl].sum()))
# cluster unexplained faces spatially (shared vertices)
idx=np.where(unexpl)[0]
F=m.faces[idx]
# adjacency via shared vertex
rows=np.repeat(np.arange(len(idx)),3); cols=F.ravel()
M=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(idx),len(m.vertices))).tocsr()
Adj=(M@M.T)
nc,lab=connected_components(Adj,directed=False)
clusters=[]
for c in range(nc):
    ii=idx[lab==c]; ar=FA[ii].sum()
    if ar<1.0: continue
    clusters.append(dict(area=float(ar),n=int(len(ii)),centroid=FC[ii].mean(0).tolist(),s=float((FC[ii]@a).mean()),
        ext=float(np.linalg.norm(FC[ii].max(0)-FC[ii].min(0))),mean_normal=unit(FN[ii].sum(0)).tolist()))
clusters.sort(key=lambda z:-z['area'])
print('unexplained clusters >=1 mm2:',len(clusters))
for c in clusters[:25]: print(' area %.1f n %d ext %.1f s %.1f centroid %s normal %s'%(c['area'],c['n'],c['ext'],c['s'],np.round(c['centroid'],1),np.round(c['mean_normal'],2)))
json.dump(dict(area_total=float(FA.sum()),area_unexplained=float(FA[unexpl].sum()),clusters=clusters[:40]),open(OUT+'a1_unexplained_clusters.json','w'),indent=1)
np.save('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_unexpl.npy',unexpl)
# render: color unexplained faces red, arcs blue
S=Splat(m,n=2000000,seed=2)
U=unexpl[S.fid]; Aa=(arc&~skin)[S.fid]
fidmap=S.fid
def color(P,N,_f=None): return np.ones((len(P),3))
fig,axs=plt.subplots(2,2,figsize=(20,18))
v=m.vertices; c=v.mean(0); X=v-c; w,UU=np.linalg.eigh(X.T@X/len(v))
views=[(UU[:,0],UU[:,2]),(-UU[:,0],UU[:,2]),(a,UU[:,0]),(np.cross(a,UU[:,0]),a)]
for ax,(vd,up) in zip(axs.ravel(),views):
    # custom: render with per-sample colors
    Pts=S.p; Nn=S.n
    from a1s70_render import basis
    r,u,d=basis(vd,up); x=Pts@r; y=Pts@u; z=Pts@d; res=0.12
    x0,x1,y0,y1=x.min()-2,x.max()+2,y.min()-2,y.max()+2
    W=int((x1-x0)/res)+1; H=int((y1-y0)/res)+1
    ix=((x-x0)/res).astype(int); iy=((y1-y)/res).astype(int); pix=iy*W+ix
    o=np.lexsort((z,pix)); ps=pix[o]; last=np.r_[ps[1:]!=ps[:-1],True]; sel=o[last]
    sh=0.3+0.7*np.abs(Nn[sel]@d)
    col=np.stack([sh,sh,sh],1)
    col[U[sel]]=np.array([1,0,0])*sh[U[sel],None]+np.array([0.3,0,0])
    col[Aa[sel]]=np.stack([0.55*sh[Aa[sel]],0.75*sh[Aa[sel]],1.0*sh[Aa[sel]]],1)
    img=np.ones((H*W,3)); img[pix[sel]]=np.clip(col,0,1)
    ax.imshow(img.reshape(H,W,3)); ax.set_axis_off()
    for k,(p,q) in GOLD.items():
        P2=np.array([p,q]); X2=(P2@r-x0)/res; Y2=(y1-P2@u)/res; ax.plot(X2,Y2,'-',color='#1a9c3a',lw=1.5); ax.text(X2[1],Y2[1],k,color='#1a9c3a',fontsize=9)
axs[0,0].set_title('red = faces NOT explained by flat skins / parallel bend arcs / edge walls; blue = bend-arc faces')
plt.tight_layout(); plt.savefig(OUT+'a1_missing_bend_scan.png',dpi=60)
