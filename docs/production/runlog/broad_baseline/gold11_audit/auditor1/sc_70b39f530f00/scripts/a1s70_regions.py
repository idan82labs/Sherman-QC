import sys; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
s=FC@a; x=FC@e1; y=FC@e2
na=FN@a; th=np.degrees(np.arctan2(FN@e2,FN@e1))
inplane=np.abs(na)<0.2
for lo,hi,name in [(-12,-6,'strip/mid A'),(168,174,'strip/mid B'),(-51,-45,'incl A'),(128,135,'incl B'),(-113,-108,'frame A'),(66,72,'frame B'),(-102,-97,'n6'),(77,82,'n7'),(-24,-18,'n8')]:
    sel=inplane&(th>lo)&(th<hi)
    print('%-12s area %.0f  s[%.1f,%.1f] x[%.1f,%.1f] y[%.1f,%.1f]'%(name,FA[sel].sum(),*np.percentile(s[sel],[1,99]),*np.percentile(x[sel],[1,99]),*np.percentile(y[sel],[1,99])))
fig,axs=plt.subplots(1,3,figsize=(24,8))
sel=inplane
sc=axs[0].scatter(x[sel],y[sel],c=th[sel],s=0.3,cmap='hsv',vmin=-180,vmax=180); axs[0].set_aspect('equal'); plt.colorbar(sc,ax=axs[0]); axs[0].set_title('in-plane-normal faces, color=normal theta')
for k,(p,q) in GOLD.items():
    axs[0].plot([p@e1],[p@e2],'k*',ms=10); axs[0].text(p@e1+0.5,p@e2,k)
axs[0].set_xlim(40,70); axs[0].set_ylim(-365,-325)
for ax,(lo,hi,name) in zip(axs[1:],[(-102,-97,'n6'),(-24,-18,'n8')]):
    selk=inplane&(th>lo)&(th<hi)
    ax.scatter(s[selk],x[selk],s=1,c=y[selk]); ax.set_title(name+' faces: s vs e1 (color e2)'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(OUT+'a1_regions_debug.png',dpi=60)
