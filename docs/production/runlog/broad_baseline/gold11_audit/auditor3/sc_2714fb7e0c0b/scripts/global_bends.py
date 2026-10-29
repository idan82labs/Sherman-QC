import numpy as np, json
from scipy import ndimage
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
mx=np.abs(NL).max(1)
inter=mx<np.cos(np.radians(15))
P=L[inter]; print('intermediate-normal points',inter.sum(),'of',len(L))
h=1.5
ijk=np.floor((P-P.min(0))/h).astype(int)
shape=ijk.max(0)+1
grid=np.zeros(shape,dtype=bool); grid[tuple(ijk.T)]=True
lab,n=ndimage.label(grid,structure=np.ones((3,3,3)))
pl=lab[tuple(ijk.T)]
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:((np.array(b['line_start'])-o)@R.T,(np.array(b['line_end'])-o)@R.T) for b in G['bends']}
I=json.load(open(W+'ideal_apex_lines_raw.json'))
miss={k:((np.array(v[0])-o)@R.T,(np.array(v[1])-o)@R.T) for k,v in I.items() if 'SF' in k}
def segdist(p,a,b):
    d=b-a; t=np.clip((p-a)@d/(d@d),0,1); return np.linalg.norm(p-(a+t*d))
rows=[]
for c in range(1,n+1):
    q=P[pl==c]
    if len(q)<30: continue
    m=q.mean(0); U,S,Vt=np.linalg.svd(q-m,full_matrices=False)
    proj=(q-m)@Vt[0]; length=proj.max()-proj.min()
    width=np.sqrt(S[1]**2/len(q))*4
    if length<5: continue
    near=sorted([(segdist(m,a,b),k) for k,(a,b) in list(gold.items())+list(miss.items())])[:2]
    rows.append((length,len(q),m.round(1),Vt[0].round(2),round(width,1),[(k,round(d,1)) for d,k in near]))
rows.sort(key=lambda r:-r[0])
for r in rows: print('len %6.1f n=%6d centre %s dir %s width %.1f nearest %s'%r)
print('--- after removing points within 5 mm of the 10 measured apex lines ---')
lines=[((np.array(v[0])-o)@R.T,(np.array(v[1])-o)@R.T) for k,v in I.items()]
keep=np.ones(len(P),bool)
for a,b in lines:
    d=b-a; t=np.clip((P-a)@d/(d@d),0,1); dist=np.linalg.norm(P-(a+np.outer(t,d)),axis=1)
    keep&=dist>5
Q=P[keep]; print('remaining',len(Q))
ijk=np.floor((Q-Q.min(0))/h).astype(int); shape=ijk.max(0)+1
grid=np.zeros(shape,dtype=bool); grid[tuple(ijk.T)]=True
lab,n=ndimage.label(grid,structure=np.ones((3,3,3))); pl=lab[tuple(ijk.T)]
rows=[]
for c in range(1,n+1):
    q=Q[pl==c]
    if len(q)<30: continue
    m=q.mean(0); U,S,Vt=np.linalg.svd(q-m,full_matrices=False)
    proj=(q-m)@Vt[0]; length=proj.max()-proj.min(); w2=((q-m)@Vt[1]); width=w2.max()-w2.min()
    if length<4: continue
    rows.append((length,width,len(q),m.round(1),Vt[0].round(2)))
rows.sort(key=lambda r:-r[0])
for r in rows[:40]: print('len %6.1f width %5.1f n=%6d centre %s dir %s'%r)
