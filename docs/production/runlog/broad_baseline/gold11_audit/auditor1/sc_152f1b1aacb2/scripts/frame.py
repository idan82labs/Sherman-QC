import numpy as np, json
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/"
V = np.load(S+"V.npy")
gold = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"))
case = [c for c in gold['cases'] if c['case_id']=='corpus/08_40500000'][0]
G = {b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in case['bends']}
c = V.mean(0)
U,Sg,Wt = np.linalg.svd(V-c, full_matrices=False)
e3 = Wt[2].copy()
d1 = G['M01'][1]-G['M01'][0]; d1/=np.linalg.norm(d1)
e1 = d1 - e3*np.dot(d1,e3); e1/=np.linalg.norm(e1)
e2 = np.cross(e3,e1)
R = np.vstack([e1,e2,e3])
P = (V-c)@R.T
# choose e3 sign: walls go up (+e3). base plate is the mode of e3 values with largest count
h,edges = np.histogram(P[:,2], bins=200)
base = edges[np.argmax(h)]
print("base e3 mode", base, "e3 range", P[:,2].min(), P[:,2].max())
if (P[:,2].max()-base) < (base-P[:,2].min()):
    e3=-e3; e2=np.cross(e3,e1); R=np.vstack([e1,e2,e3]); P=(V-c)@R.T
    h,edges = np.histogram(P[:,2], bins=200); base = edges[np.argmax(h)]
    print("flipped; base", base)
print("e1",e1,"e2",e2,"e3",e3)
print("P range", P.min(0), P.max(0))
np.save(S+"P.npy", P); np.save(S+"R.npy", R); np.save(S+"c.npy", c)
for k,(a,b) in G.items():
    A=(a-c)@R.T; B=(b-c)@R.T
    d=(B-A); L=np.linalg.norm(d)
    print(k, "start", np.round(A,2), "end", np.round(B,2), "len %.1f"%L, "dir", np.round(d/L,3))
