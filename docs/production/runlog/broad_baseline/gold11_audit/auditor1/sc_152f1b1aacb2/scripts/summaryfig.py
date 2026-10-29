import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
ML=json.load(open(OUT+"a1_measured_apex_lines.json"))
rng=np.random.default_rng(1); idx=rng.choice(len(P),150000,replace=False); Q=P[idx]
verdict={"M01":"extent overshoot","M02":"extent overshoot","M07":"extent overshoot (fails 60%)","M03":"NOT A BEND","M04":"NOT A BEND","M05":"NOT A BEND","M06":"NOT A BEND"}
fold_extent={"M01":(-63.5,65.9),"M02":(-63.4,66.4),"M07":(-24.0,26.0)}
missing=["B4_right_wall_top_tabs","B5_end_tab_e1min","B6_end_tab_e1max"]
matched=["B1_left_base_wall","B2_right_base_wall","B3_left_wall_top_tab"]
fig=plt.figure(figsize=(26,22))
ax=fig.add_subplot(2,2,1)
ax.scatter(Q[:,0],Q[:,1],c=Q[:,2],s=0.2,cmap='Greys_r',alpha=0.5)
def draw(ax,i,j,showlabels=True,filt=None):
    for k in matched:
        a=np.array(ML[k]['start_part']); b=np.array(ML[k]['end_part'])
        if filt and not filt(a,b): continue
        ax.plot([a[i],b[i]],[a[j],b[j]],'-',color='royalblue',lw=7,alpha=0.45,solid_capstyle='butt')
    for k in missing:
        a=np.array(ML[k]['start_part']); b=np.array(ML[k]['end_part'])
        if filt and not filt(a,b): continue
        ax.plot([a[i],b[i]],[a[j],b[j]],'--',color='red',lw=3)
        if showlabels: ax.text((a[i]+b[i])/2,(a[j]+b[j])/2+3,"MISSING "+k.split('_',1)[0],color='red',fontsize=12,weight='bold')
    for g,L in G.items():
        if filt and not filt(L[0],L[1]): continue
        col='darkorange' if g in ('M01','M02','M07') else 'magenta'
        ax.plot(L[:,i],L[:,j],'-',color=col,lw=2)
        ax.plot(L[:,i],L[:,j],'o',color=col,ms=5)
        if showlabels: ax.text(L[:,i].mean()+1.5,L[:,j].mean()-4,g+": "+verdict[g],color=col,fontsize=11,weight='bold')
draw(ax,0,1)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1 (along walls)'); ax.set_ylabel('e2')
ax.set_title("TOP (e1-e2). blue band = measured fold apex line (extent of real fold); red dashed = formed fold with NO gold;\norange = gold on a real fold (overshoots ends); magenta = gold not on any fold (wall vertical free edges)",fontsize=12)
ax=fig.add_subplot(2,2,2)
m=Q[:,1]<-70
ax.scatter(Q[m,0],Q[m,2],s=0.4,c='grey',alpha=0.6)
draw(ax,0,2,filt=lambda a,b:(a[1]+b[1])/2<0)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1'); ax.set_ylabel('e3'); ax.set_title("LEFT wall (e2<-70) face view e1-e3",fontsize=12); ax.set_ylim(-15,50)
ax=fig.add_subplot(2,2,4)
m=Q[:,1]>70
ax.scatter(Q[m,0],Q[m,2],s=0.4,c='grey',alpha=0.6)
draw(ax,0,2,filt=lambda a,b:(a[1]+b[1])/2>0)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1'); ax.set_ylabel('e3'); ax.set_title("RIGHT wall (e2>70) face view e1-e3",fontsize=12); ax.set_ylim(-15,50)
ax=fig.add_subplot(2,2,3)
m=np.abs(Q[:,1]+36)<12
ax.scatter(Q[m,0],Q[m,2],s=0.4,c='grey',alpha=0.6)
for k in ("B5_end_tab_e1min","B6_end_tab_e1max"):
    a=np.array(ML[k]['start_part']); ax.plot(a[0],a[2],'rx',ms=16,mew=3); ax.text(a[0]+(3 if a[0]<0 else -32),a[2]+6,"MISSING "+k.split('_',1)[0]+"\n(90 deg up-turned end tab)",color='red',fontsize=12,weight='bold')
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1'); ax.set_ylabel('e3'); ax.set_title("Side slice |e2+36|<12 (through the short-end tabs), e1-e3",fontsize=12); ax.set_ylim(-15,20)
fig.suptitle("sc_152f1b1aacb2 (40500000 SCAN2) auditor-1: gold M01-M07 vs measured formed folds (part frame e1/e2/e3; raw-frame lines in a1_measured_apex_lines.json)",fontsize=15)
plt.tight_layout(); plt.savefig(OUT+"a1_summary_gold_vs_folds.png",dpi=55); plt.close()

# close-ups at representative stations for M01, M02, M07
fig,axs=plt.subplots(3,4,figsize=(24,18))
spec=[("M01",-1,"bw",[-50,10,62,-75]),("M02",1,"bw",[-50,10,62,80]),("M07",-1,"top",[-10,10,40,-45])]
for r,(gid,sgn,kind,sts) in enumerate(spec):
    for j,s in enumerate(sts):
        ax=axs[r,j]
        if kind=="bw": m=(np.abs(P[:,0]-s)<0.4)&(sgn*P[:,1]>80)&(P[:,2]<8)
        else: m=(np.abs(P[:,0]-s)<0.4)&(sgn*P[:,1]>78)&(P[:,2]>26)
        S2=P[m]
        ax.scatter(S2[:,1],S2[:,2],s=5,c='k')
        g,inside=gold_at(G[gid],0,s)
        ax.plot(g[1],g[2],'*',ms=22,mfc='lime',mec='g',mew=2)
        ax.add_patch(plt.Circle((g[1],g[2]),4.0,fill=False,ls=':',color='g'))
        cy = -6.8 if kind=="bw" else 36
        ax.set_xlim(g[1]-6,g[1]+6); ax.set_ylim(cy-6,cy+6); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_xlabel('e2'); ax.set_ylabel('e3')
        ext=fold_extent[gid]; onfold = ext[0]<=s<=ext[1]
        ax.set_title(f"{gid} @ e1={s}: "+("ON fold" if onfold else "BEYOND fold end (no fold here)")+"\n(dotted = 4 mm matcher radius)",fontsize=12)
fig.suptitle("Perpendicular close-ups: gold star vs scan cross-section (1 mm = 1 grid unit). On-fold stations: gold sits on the outer apex. Beyond-fold stations: gold lies on a free plate/wall edge.",fontsize=14)
plt.tight_layout(); plt.savefig(OUT+"a1_closeups_M01_M02_M07.png",dpi=55); plt.close()
print("ok")
