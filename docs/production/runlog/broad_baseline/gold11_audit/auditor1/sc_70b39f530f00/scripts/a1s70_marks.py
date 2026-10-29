import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); V=m.vertices
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
x=V@e1; y=V@e2; s=V@a
# height maps of each flat skin relative to its fitted plane: incl (both skins), mid (both), frame (both), strip (both)
flats={'strip':(-9.43,lambda X,Y,S:(Y>-330)&(Y<-313)),'incl':(-48.36,lambda X,Y,S:(Y>-344)&(Y<-335)),'mid':(-9.73,lambda X,Y,S:(Y>-355.5)&(Y<-349)),'frame':(-110.54,lambda X,Y,S:(X>52))}
fig,axs=plt.subplots(2,4,figsize=(28,12))
out={}
for c,(nm,(th,reg)) in enumerate(flats.items()):
    n2=np.array([np.cos(np.radians(th)),np.sin(np.radians(th))])
    n3=n2[0]*e1+n2[1]*e2
    u3=np.cross(a,n3)   # in-plane direction perpendicular to axis
    sel=reg(x,y,s)
    h=V[sel]@n3
    # two skins: split by median of h
    med=np.median(h)
    for r,(sk,ss) in enumerate([('+',h>med),('-',h<=med)]):
        P=V[sel][ss]; hh=P@n3
        A=np.c_[P@a,P@u3,np.ones(len(P))]; coef=np.linalg.lstsq(A,hh,rcond=None)[0]
        res=hh-A@coef
        # robust refit
        k=np.abs(res)<0.3; coef=np.linalg.lstsq(A[k],hh[k],rcond=None)[0]; res=hh-A@coef
        ax=axs[r,c]; sc=ax.scatter(P@a,P@u3,c=res,s=1.5,cmap='coolwarm',vmin=-0.4,vmax=0.4); plt.colorbar(sc,ax=ax,label='height above fitted plane (mm)')
        ax.set_aspect('equal'); ax.set_title('%s skin %s: residual to plane (rms %.3f mm, p1 %.2f p99 %.2f)'%(nm,sk,np.sqrt(np.mean(res[k]**2)),np.percentile(res,1),np.percentile(res,99)),fontsize=9)
        ax.set_xlabel('s along bend axis (mm)')
        out['%s%s'%(nm,sk)]=dict(rms=float(np.sqrt(np.mean(res[k]**2))),p1=float(np.percentile(res,1)),p99=float(np.percentile(res,99)),n=int(len(P)))
plt.tight_layout(); plt.savefig(OUT+'a1_flat_skin_residuals.png',dpi=55)
print(json.dumps(out,indent=0))
