from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
def circ(ax,x,y,r,**kw):
    t=np.linspace(0,2*np.pi,100); ax.plot(x+r*np.cos(t),y+r*np.sin(t),**kw)
fig,axs=plt.subplots(2,3,figsize=(21,14))
# (v,w) zooms at u=-350 and u=-430
for col,u0 in enumerate([-430,-350]):
    m=np.abs(P[:,0]-u0)<0.5
    for row,(vlo,vhi) in enumerate([(-100,-50),(-282,-232)]):
        ax=axs[row,col]
        mm=m&(P[:,1]>vlo)&(P[:,1]<vhi)
        ax.scatter(P[mm,1],P[mm,2],s=3,c='k')
        for k,(s,e) in gp.items():
            if k in ('M07','M08'): continue
            if vlo<s[1]<vhi:
                inside=min(s[0],e[0])<=u0<=max(s[0],e[0])
                ax.plot(s[1],s[2],'*',color='r' if inside else 'orange',ms=14); ax.text(s[1]+0.4,s[2]+0.6,k,color='r',fontsize=11)
                circ(ax,s[1],s[2],4,color='g',ls=':')
        ax.set_aspect('equal'); ax.set_xlim(vlo,vhi); ax.set_ylim(-84,-50); ax.grid(alpha=0.3)
        ax.set_title(f'u={u0} (+-0.5): v-w zoom'); ax.set_xlabel('v'); ax.set_ylabel('w')
# (u,w) zooms at v=-150: left flange and right wall
m=np.abs(P[:,1]+150)<0.5
ax=axs[0,2]; mm=m&(P[:,0]<-455)
ax.scatter(P[mm,0],P[mm,2],s=3,c='k'); ax.set_aspect('equal'); ax.set_xlim(-490,-455); ax.set_ylim(-75,-48); ax.grid(alpha=0.3); ax.set_title('v=-150: left end (u~-477) profile u-w')
ax=axs[1,2]; mm=m&(P[:,0]>-240)
ax.scatter(P[mm,0],P[mm,2],s=3,c='k')
for k in ('M07','M08'):
    s,e=gp[k]; ax.plot(s[0],s[2],'r*',ms=14); ax.text(s[0]+0.4,s[2]+0.6,k,color='r',fontsize=11); circ(ax,s[0],s[2],4,color='g',ls=':')
ax.set_aspect('equal'); ax.set_xlim(-240,-205); ax.set_ylim(-132,-60); ax.grid(alpha=0.3); ax.set_title('v=-150: right wall (u~-224) profile u-w')
plt.tight_layout(); plt.savefig(OUT+'a1_zoom_profiles_1.png',dpi=72)
