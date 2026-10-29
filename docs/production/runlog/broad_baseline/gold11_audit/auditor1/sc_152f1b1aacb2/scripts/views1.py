import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/"
OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_152f1b1aacb2/"
P = np.load(S+"P.npy"); R=np.load(S+"R.npy"); c=np.load(S+"c.npy")
gold = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"))
case = [cc for cc in gold['cases'] if cc['case_id']=='corpus/08_40500000'][0]
G = {b['bend_id']:np.array([(np.array(b['line_start'])-c)@R.T,(np.array(b['line_end'])-c)@R.T]) for b in case['bends']}
rng = np.random.default_rng(0)
idx = rng.choice(len(P), 120000, replace=False)
Q = P[idx]
fig, axs = plt.subplots(2,2, figsize=(20,16))
ax=axs[0,0]
sc=ax.scatter(Q[:,0],Q[:,1],c=Q[:,2],s=0.3,cmap='viridis'); plt.colorbar(sc,ax=ax,label='e3')
for k,L in G.items():
    ax.plot(L[:,0],L[:,1],'r-',lw=1.5); ax.text(L[:,0].mean(),L[:,1].mean(),k,color='r',fontsize=12)
ax.set_aspect('equal'); ax.set_title('top e1-e2, colour=e3 (base plate e3~-7, walls up to 38)'); ax.grid(alpha=.3)
ax=axs[0,1]
m = Q[:,2]>-5.5
sc=ax.scatter(Q[m,0],Q[m,1],c=Q[m,2],s=0.5,cmap='viridis'); plt.colorbar(sc,ax=ax,label='e3')
for k,L in G.items():
    ax.plot(L[:,0],L[:,1],'r-',lw=1); ax.text(L[:,0].mean(),L[:,1].mean(),k,color='r',fontsize=12)
ax.set_aspect('equal'); ax.set_title('top e1-e2, only e3>-5.5 (off base plate)'); ax.grid(alpha=.3)
ax=axs[1,0]
m = Q[:,1]<-70
sc=ax.scatter(Q[m,0],Q[m,2],c=Q[m,1],s=0.5,cmap='coolwarm'); plt.colorbar(sc,ax=ax,label='e2')
for k,L in G.items():
    if L[:,1].mean()<0: ax.plot(L[:,0],L[:,2],'r-',lw=1.5); ax.text(L[:,0].mean()+1,L[:,2].mean()+1,k,color='r',fontsize=12)
ax.set_aspect('equal'); ax.set_title('left wall e2<-70: e1-e3, colour=e2'); ax.grid(alpha=.3)
ax=axs[1,1]
m = Q[:,1]>70
sc=ax.scatter(Q[m,0],Q[m,2],c=Q[m,1],s=0.5,cmap='coolwarm'); plt.colorbar(sc,ax=ax,label='e2')
for k,L in G.items():
    if L[:,1].mean()>0: ax.plot(L[:,0],L[:,2],'r-',lw=1.5); ax.text(L[:,0].mean()+1,L[:,2].mean()+1,k,color='r',fontsize=12)
ax.set_aspect('equal'); ax.set_title('right wall e2>70: e1-e3, colour=e2'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+"a1_overview_partframe.png", dpi=80)
