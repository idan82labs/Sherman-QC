import numpy as np, hashlib, sys
P='/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply'
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
raw=open(P,'rb').read()
print('sha256', hashlib.sha256(raw).hexdigest())
i0=raw.index(b'end_header'); hend=raw.index(b'\n',i0)+1
hdr=raw[:hend].decode()
nv=int([l for l in hdr.splitlines() if l.startswith('element vertex')][0].split()[-1])
nf=int([l for l in hdr.splitlines() if l.startswith('element face')][0].split()[-1])
v=np.frombuffer(raw,dtype='<f4',count=nv*6,offset=hend).reshape(nv,6).astype(np.float64)
fo=hend+nv*24
fdt=np.dtype([('n','u1'),('i','<i4',(3,))])
f=np.frombuffer(raw,dtype=fdt,count=nf,offset=fo)
assert (f['n']==3).all()
np.save(W+'V.npy',v[:,:3]); np.save(W+'N.npy',v[:,3:]); np.save(W+'F.npy',f['i'].astype(np.int32))
print(nv,nf,v[:,:3].min(0),v[:,:3].max(0), np.linalg.norm(v[:,3:],axis=1)[:5])
