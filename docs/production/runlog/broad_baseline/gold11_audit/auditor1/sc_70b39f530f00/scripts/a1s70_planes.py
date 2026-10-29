import sys; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, json
m=load()
FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces
n=20000; i=np.arange(n)+0.5
phi=np.arccos(1-2*i/n); th=np.pi*(1+5**0.5)*i
D=np.stack([np.cos(th)*np.sin(phi),np.sin(th)*np.sin(phi),np.cos(phi)],1)
score=np.zeros(n)
for s in range(0,len(FN),5000):
    c=FN[s:s+5000]@D.T
    score+=((c>np.cos(np.radians(3)))*FA[s:s+5000,None]).sum(0)
peaks=[]
sc=score.copy()
for _ in range(16):
    j=np.argmax(sc)
    if sc[j]<40: break
    sel=FN@D[j]>np.cos(np.radians(3))
    mn=unit((FN[sel]*FA[sel,None]).sum(0))
    sel=FN@mn>np.cos(np.radians(3))
    mn=unit((FN[sel]*FA[sel,None]).sum(0))
    peaks.append((mn,FA[sel].sum()))
    sc[D@mn>np.cos(np.radians(8))]=0
for k,(mn,a) in enumerate(peaks):
    print(k,np.round(mn,4),'area %.0f'%a)
np.save('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_peaks.npy',np.array([p[0] for p in peaks]))
