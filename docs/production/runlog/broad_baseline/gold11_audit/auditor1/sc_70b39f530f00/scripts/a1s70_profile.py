import sys; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pk=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_peaks.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces; V=m.vertices
fl=pk[[0,2,4,6]]
U,S,Vt=np.linalg.svd(fl); a=Vt[-1]; gd=unit(GOLD['M01'][1]-GOLD['M01'][0]); a=a*np.sign(a@gd)
print('axis from flange normals',np.round(a,5),'sv',S,'angle to gold dirs:',[round(float(np.degrees(np.arccos(abs(unit(GOLD[k][1]-GOLD[k][0])@a)))),2) for k in GOLD])
np.save('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy',a)
# 2D frame
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
for i,n in enumerate(pk):
    print(i,'theta %.2f'%np.degrees(np.arctan2(n@e2,n@e1)),' |n.a| %.3f'%abs(n@a))
# project vertices
s=V@a; x=V@e1; y=V@e2
for k,(p,q) in GOLD.items():
    print(k,'start s=%.2f end s=%.2f  2D start (%.2f,%.2f) end (%.2f,%.2f)'%(p@a,q@a,p@e1,p@e2,q@e1,q@e2))
print('vertex s range',s.min(),s.max())
fig,axs=plt.subplots(1,2,figsize=(20,10))
ax=axs[0]; ax.scatter(x,y,s=0.2,c=s,cmap='viridis'); ax.set_aspect('equal'); ax.set_title('all vertices projected along bend axis a (color = position along a)')
cols={'M01':'#1a9c3a','M02':'#e07b00','M03':'#c2185b'}
for k,(p,q) in GOLD.items():
    ax.plot([p@e1,q@e1],[p@e2,q@e2],'o-',color=cols[k],ms=6); ax.text(p@e1+1,p@e2+1,k,color=cols[k])
ax=axs[1]; ax.scatter(s,x,s=0.2,c=y,cmap='viridis'); ax.set_aspect('equal'); ax.set_title('s (along a) vs e1')
for k,(p,q) in GOLD.items():
    ax.plot([p@a,q@a],[p@e1,q@e1],'o-',color=cols[k],ms=6); ax.text(p@a,p@e1+1,k,color=cols[k])
plt.tight_layout(); plt.savefig(OUT+'a1_profile_projection.png',dpi=70)
