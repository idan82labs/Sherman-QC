import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open(OUT+'a1_bendfit_bins.json'))
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
S=FC@a; X=np.stack([FC@e1,FC@e2],1); NA=FN@a
N2=np.stack([FN@e1,FN@e2],1); TH=np.degrees(np.arctan2(N2[:,1],N2[:,0]))
def wrap(d): return (d+180)%360-180
fig,axs=plt.subplots(3,1,figsize=(18,15))
ext={}
for ax,k in zip(axs,['M01','M02','M03']):
    r=R[k]; T1,T2=r['outer_normal_angles']; d=T2-T1
    apex=np.array(r['apex_mid']); imid=np.array(r['inner_mid'])
    p,q=GOLD[k]; ss,se=sorted([p@a,q@a])
    cen=(apex+imid)/2
    near=np.linalg.norm(X-cen,axis=1)<6.5
    # normalized angular position across bend: 0 at flange1, 1 at flange2 (outer skin); inner skin uses T+180
    fo=wrap(TH-T1)/d; fi=wrap(TH-(T1+180))/d
    on=near&(np.abs(NA)<0.3)&(fo>-0.15)&(fo<1.15)&(np.abs(wrap(TH-T1-d/2))<abs(d)/2+10)
    inn=near&(np.abs(NA)<0.3)&(fi>-0.15)&(fi<1.15)&(np.abs(wrap(TH-T1-180-d/2))<abs(d)/2+10)
    side=near&(np.abs(NA)>0.8)
    ax.scatter(S[on],fo[on],s=1,c='tab:blue',label='outer skin (y = normal-angle fraction across bend; 0..1 = arc)')
    ax.scatter(S[inn],fi[inn]+1.6,s=1,c='tab:purple',label='inner skin (offset +1.6)')
    ax.scatter(S[side],np.full(side.sum(),-0.5),s=2,c='k',label='side/end walls (|n.a|>0.8) near bend')
    for yy in (0,1,1.6,2.6): ax.axhline(yy,color='0.7',lw=0.5)
    ax.plot([ss,se],[3.0,3.0],'-',color='#1a9c3a',lw=5,label='gold extent')
    # arc-present extents: s where faces with fraction in (0.2,0.8) exist
    arc_o=S[on&(fo>0.2)&(fo<0.8)]; arc_i=S[inn&(fi>0.2)&(fi<0.8)]
    def rng(v):
        v=np.sort(v); return float(np.percentile(v,0.5)),float(np.percentile(v,99.5)),float(v.min()),float(v.max())
    eo=rng(arc_o); ei=rng(arc_i); sw=np.sort(S[side])
    ext[k]=dict(gold_s=[float(ss),float(se)],outer_arc_s=eo,inner_arc_s=ei,side_wall_s=[float(x) for x in np.percentile(sw,[0,5,50,95,100])] if len(sw) else None)
    ax.axvline(eo[2],color='tab:blue',ls='--'); ax.axvline(eo[3],color='tab:blue',ls='--'); ax.axvline(ei[2],color='tab:purple',ls=':'); ax.axvline(ei[3],color='tab:purple',ls=':')
    ax.set_title('%s: s (mm along bend axis) vs angular position; gold s=[%.2f, %.2f]; outer arc s=[%.2f, %.2f]; inner arc s=[%.2f, %.2f]'%(k,ss,se,eo[2],eo[3],ei[2],ei[3]))
    ax.set_xlim(min(ss,eo[2])-6,max(se,eo[3])+6); ax.set_ylim(-0.8,3.3); ax.legend(fontsize=7,loc='upper left',ncol=2)
    print(k,json.dumps(ext[k]))
plt.tight_layout(); plt.savefig(OUT+'a1_extent_along_axis.png',dpi=65)
json.dump(ext,open(OUT+'a1_extent.json','w'),indent=1)
