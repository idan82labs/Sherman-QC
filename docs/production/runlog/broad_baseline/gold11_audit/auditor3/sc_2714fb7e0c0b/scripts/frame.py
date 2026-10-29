import numpy as np, json
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy')
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in G['bends']}
# dominant normal = main plate normal (cluster |n| directions)
rng=np.random.default_rng(0); idx=rng.choice(len(N),200000,replace=False)
n=N[idx]
# iterative mode finding on sphere (sign-invariant)
best=None
for seed in n[rng.choice(len(n),200,replace=False)]:
    c=np.abs(n@seed)>0.995
    if best is None or c.sum()>best[0]: best=(c.sum(),seed)
nz=best[1]
for _ in range(5):
    sel=n[np.abs(n@nz)>0.99]; sel=sel*np.sign(sel@nz)[:,None]; nz=sel.mean(0); nz/=np.linalg.norm(nz)
print('plate normal',nz.round(4),'frac',best[0]/len(n))
a01=gold['B01'][1]-gold['B01'][0]; a01/=np.linalg.norm(a01)
eY=a01-nz*(a01@nz); eY/=np.linalg.norm(eY)
eX=np.cross(eY,nz); eX/=np.linalg.norm(eX)
# orient eX along B03 direction (start->end)
a03=gold['B03'][1]-gold['B03'][0]
if eX@a03<0: eX=-eX
eZ=np.cross(eX,eY)
if eZ@nz<0: pass
print('eX',eX.round(4),'eY',eY.round(4),'eZ',eZ.round(4))
R=np.vstack([eX,eY,eZ]); o=V.mean(0)
L=(V-o)@R.T
print('local bbox',L.min(0).round(1),L.max(0).round(1))
np.save(W+'frame.npy',np.vstack([o,eX,eY,eZ]))
for k,(s,e) in gold.items():
    ls=(s-o)@R.T; le=(e-o)@R.T
    print(k,'start',ls.round(1),'end',le.round(1),'len %.1f'%np.linalg.norm(e-s), 'dir',((le-ls)/np.linalg.norm(le-ls)).round(3))
