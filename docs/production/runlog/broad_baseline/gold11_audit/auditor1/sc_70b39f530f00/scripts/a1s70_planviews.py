import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
from a1s70_render import Splat, draw_line, basis
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open(OUT+'a1_bendfit_bins.json')); E=json.load(open(OUT+'a1_extent_lines.json'))
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
m=load(); S=Splat(m,n=2500000,seed=1)
def to3(p2,s): return p2[0]*e1+p2[1]*e2+s*a
cols={'M01':'#1a9c3a','M02':'#e07b00','M03':'#c2185b'}
fig,axs=plt.subplots(3,2,figsize=(20,24))
for r,k in enumerate(['M01','M02','M03']):
    rr=R[k]; T1,T2=rr['outer_normal_angles']; tb=np.radians((T1+T2)/2)
    bis3=np.cos(tb)*e1+np.sin(tb)*e2
    apex2=np.array(rr['apex_mid']); im2=np.array(rr['inner_mid'])
    ex=E[k]; s0,s1=ex['outer_apex_line']['s_min'],ex['outer_apex_line']['s_max']; i0,i1=ex['inner_midline']['s_min'],ex['inner_midline']['s_max']
    p,q=GOLD[k]; cen=(p+q)/2
    for c,(side,vd) in enumerate([('from CONCAVE side (inner skin faces camera)',-bis3),('from CONVEX side (outer skin faces camera)',bis3)]):
        # clip: only material within 9 mm (2D) of the bend centre to avoid occlusion by other plates
        c2=(apex2+im2)/2
        clip=lambda P,c2=c2: np.linalg.norm(np.stack([P@e1,P@e2],1)-c2,axis=1)<9
        Rn=S.render(vd,a,res=0.06,center=cen,half=(14,20),clip=clip)
        ax=axs[r,c]; ax.imshow(Rn['img'],interpolation='nearest')
        # apex line over measured fold extent (cyan), inner midline (violet)
        draw_line(ax,Rn,to3(apex2,s0),to3(apex2,s1),'cyan',label='apex',lw=2.5)
        draw_line(ax,Rn,to3(im2,i0),to3(im2,i1),'violet',label='inner mid',lw=2.5)
        draw_line(ax,Rn,p,q,cols[k],label=k+' gold',lw=2.0)
        X,Y,Z=Splat.proj(Rn,np.array([p,q])); ax.plot(X,Y,'o',color=cols[k],ms=6,mec='k')
        ax.set_title('%s %s\n(clip 9 mm around bend; dotted = behind surface; cyan = outer apex line over fold, violet = inner arc midline)'%(k,side),fontsize=9)
        ax.set_axis_off()
plt.tight_layout(); plt.savefig(OUT+'a1_planviews_3bends.png',dpi=60)
print('ok')
