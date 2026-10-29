import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
heights = [-3, 5, 15, 25, 33]
fig, axs = plt.subplots(4,5, figsize=(26,22))
summary={}
for r,gid in enumerate(["M05","M06","M04","M03"]):
    L=G[gid]; ge1=L[:,0].mean(); ge2=L[:,1].mean()
    sgn = np.sign(ge2)
    summary[gid]=[]
    for j,h in enumerate(heights):
        ax=axs[r,j]
        m=(np.abs(P[:,2]-h)<0.6)&(np.abs(P[:,0]-ge1)<16)&(np.abs(P[:,1]-ge2)<16)
        Q=P[m]
        ax.scatter(Q[:,0],Q[:,1],s=2,c='k')
        g,inside=gold_at(L,2,h)
        ax.plot(g[0],g[1],'*',ms=18,mfc='lime' if inside else 'none',mec='g',mew=2)
        # wall free edge: wall points with |e2-ge2|<2.5
        wm=(np.abs(Q[:,1]-ge2)<2.5)
        if wm.sum():
            if ge1<0: edge=Q[wm,0].min()
            else: edge=Q[wm,0].max()
            ax.axvline(edge,color='r',ls=':')
            # planarity: fit e2 = a + b*e1 on wall points within 3..14 mm inside the edge, then residual near the edge
            inner = wm & ((np.abs(Q[:,0]-edge)>3)&(np.abs(Q[:,0]-edge)<14))
            near = wm & (np.abs(Q[:,0]-edge)<=3)
            # split skins by median e2
            res=None
            if inner.sum()>10 and near.sum()>5:
                A=np.vstack([Q[inner,0],np.ones(inner.sum())]).T
                k,b=np.linalg.lstsq(A,Q[inner,1],rcond=None)[0]
                res=float(np.median(Q[near,1]-(k*Q[near,0]+b)))
                slope_deg=float(np.degrees(np.arctan(k)))
            else: slope_deg=None
            dist_edge = float(abs(g[0]-edge))
            summary[gid].append(dict(e3=h, gold_e1=float(g[0]), gold_e2=float(g[1]), wall_free_edge_e1=float(edge), gold_to_free_edge_mm=dist_edge, near_edge_offset_from_wall_plane=res, wall_slope_deg=slope_deg))
            ax.set_title(f"{gid} slice e3={h}: gold-to-wall-free-edge {dist_edge:.2f} mm\nnear-edge dev from wall plane {res if res is None else round(res,2)} mm", fontsize=10)
        else:
            ax.set_title(f"{gid} slice e3={h}: no wall points", fontsize=10)
        ax.set_xlim(ge1-16,ge1+16); ax.set_ylim(ge2-16,ge2+16); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_xlabel('e1 (along wall)'); ax.set_ylabel('e2')
fig.suptitle("Horizontal slices (e1-e2 plane, |e3-h|<0.6) around the 4 vertical gold lines M03-M06. Red dotted = wall vertical free edge.", fontsize=14)
plt.tight_layout(); plt.savefig(OUT+"a1_xsec_vertical_lines.png", dpi=55); plt.close()
for k,v in summary.items():
    for s in v: print(k, {a:(round(b,2) if isinstance(b,float) else b) for a,b in s.items()})
json.dump(summary, open(OUT+"a1_vertical_lines.json","w"), indent=1)
