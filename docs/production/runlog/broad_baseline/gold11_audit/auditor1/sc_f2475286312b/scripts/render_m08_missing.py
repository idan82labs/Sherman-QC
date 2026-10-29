from render_xsec import *
# M08: u-w profiles at the bottom of the end wall
fig,axs=plt.subplots(2,3,figsize=(19,13)); axs=axs.ravel()
s8,e8=gp['M08']
for ax,v0 in zip(axs[:3],[-250,-166,-80]):
    m=(np.abs(P[:,1]-v0)<0.4); Q=P[m][:,[0,2]]
    t=(v0-s8[1])/(e8[1]-s8[1]); p=s8+t*(e8-s8)
    k=(np.abs(Q[:,0]-p[0])<14)&(np.abs(Q[:,1]-p[2])<14)
    ax.scatter(Q[k,0],Q[k,1],s=4,c='0.25')
    ax.plot(p[0],p[2],'*',color='#1a9c3a',mec='k',ms=15); ax.annotate('M08',(p[0],p[2]),xytext=(5,5),textcoords='offset points',color='#1a9c3a',weight='bold')
    from scipy.spatial import cKDTree
    d=cKDTree(Q[k]).query([p[0],p[2]])[0] if k.sum() else np.nan
    circ(ax,(p[0],p[2]),4.0,color='#1a9c3a',ls=':')
    ax.set_xlim(p[0]-14,p[0]+14); ax.set_ylim(p[2]-14,p[2]+14); ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('u'); ax.set_ylabel('w')
    ax.set_title(f'M08 @ v={v0}: u-w slab |v-({v0})|<0.4\nwall bottom = free cut edge, no return/lip; M08 to nearest scan (in slab) {d:.2f} mm',fontsize=9)
# missing folds: F3 left flange, F7 ear, F8 ear
panel(axs[3],'F3_leftflange_u-477',-166.0,[],title='MISSING F3: left-end up-flange (axis along v), u-w section @ v=-166')
panel(axs[4],'F7_ear_v-54',-90.0,[],title='MISSING F7: ear fold at wall/v=-54 corner (axis along w), u-v plan @ w=-90')
panel(axs[5],'F8_ear_v-278',-90.0,[],title='MISSING F8: ear fold at wall/v=-278 corner (axis along w), u-v plan @ w=-90')
axs[3].legend(loc='lower right',fontsize=7)
fig.suptitle('sc_f2475286312b auditor1: M08 (top row) and formed folds with no gold line (bottom row)',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'a1_xsec_M08_and_missing.png',dpi=72)
