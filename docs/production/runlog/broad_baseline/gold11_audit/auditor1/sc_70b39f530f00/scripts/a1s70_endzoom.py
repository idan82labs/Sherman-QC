import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
from a1s70_render import Splat, draw_line
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open(OUT+'a1_bendfit_bins.json')); E=json.load(open(OUT+'a1_extent_lines.json'))
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
m=load(); S=Splat(m,n=3000000,seed=3)
def to3(p2,s): return p2[0]*e1+p2[1]*e2+s*a
cols={'M01':'#1a9c3a','M02':'#e07b00','M03':'#c2185b'}
fig,axs=plt.subplots(3,4,figsize=(24,19))
for r,k in enumerate(['M01','M02','M03']):
    rr=R[k]; T1,T2=rr['outer_normal_angles']; tb=np.radians((T1+T2)/2)
    bis3=np.cos(tb)*e1+np.sin(tb)*e2
    apex2=np.array(rr['apex_mid']); im2=np.array(rr['inner_mid']); c2=(apex2+im2)/2
    ex=E[k]; s0,s1=ex['outer_apex_line']['s_min'],ex['outer_apex_line']['s_max']; i0,i1=ex['inner_midline']['s_min'],ex['inner_midline']['s_max']
    p,q=GOLD[k]
    ends=sorted([p,q],key=lambda v:v@a)
    clip=lambda P,c2=c2: np.linalg.norm(np.stack([P@e1,P@e2],1)-c2,axis=1)<10
    for c,(endname,E3) in enumerate([('low-s end',ends[0]),('high-s end',ends[1])]):
        for cc,(side,vd) in enumerate([('concave side',-bis3),('convex side',bis3)]):
            ax=axs[r,2*c+cc]
            Rn=S.render(vd,a,res=0.025,center=E3,half=(7,7),clip=clip)
            ax.imshow(Rn['img'],interpolation='nearest')
            draw_line(ax,Rn,to3(apex2,s0-3),to3(apex2,s1+3),'cyan',lw=1.2)
            draw_line(ax,Rn,to3(im2,i0-3),to3(im2,i1+3),'violet',lw=1.2)
            for sv,col in [(s0,'cyan'),(s1,'cyan')]:
                X,Y,_=Splat.proj(Rn,np.array([to3(apex2,sv)])); ax.plot(X,Y,'|',color=col,ms=25,mew=3)
            for sv in (i0,i1):
                X,Y,_=Splat.proj(Rn,np.array([to3(im2,sv)])); ax.plot(X,Y,'|',color='violet',ms=25,mew=3)
            draw_line(ax,Rn,p,q,cols[k],lw=2.5)
            X,Y,_=Splat.proj(Rn,np.array([E3])); ax.plot(X,Y,'o',color=cols[k],ms=10,mec='k')
            ax.set_title('%s %s, viewed from %s\n(cyan = outer apex line, violet = inner arc midline, ticks = measured fold ends; dotted = hidden)'%(k,endname,side),fontsize=8)
            ax.set_axis_off()
plt.tight_layout(); plt.savefig(OUT+'a1_gold_endpoints_zoom.png',dpi=55)
print('ok')
