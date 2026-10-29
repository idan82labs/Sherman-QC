import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center; FA=m.area_faces; V=m.vertices; VN=m.vertex_normals
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
S=FC@a; X=np.stack([FC@e1,FC@e2],1); NA=FN@a
N2=np.stack([FN@e1,FN@e2],1); TH=np.degrees(np.arctan2(N2[:,1],N2[:,0]))
inpl=np.abs(NA)<0.15
def ang(t): return np.array([np.cos(np.radians(t)),np.sin(np.radians(t))])
def wrap(d): return (d+180)%360-180
def tw(t): return np.abs(wrap(TH-t))
def gold_at(k,s):
    p,q=GOLD[k]; t=(s-p@a)/((q-p)@a); P=p+t*(q-p); return np.array([P@e1,P@e2])
def fit_line(P,n):
    c=P.mean(0); U,Sv,Vt=np.linalg.svd(P-c); d=Vt[0]; nn=np.array([-d[1],d[0]]); nn*=np.sign(nn@n); return c,nn
def isect(c1,n1,c2,n2): return np.linalg.solve(np.array([n1,n2]),np.array([n1@c1,n2@c2]))
FLN={'strip':-9.43,'incl':-48.36,'frame':-110.54,'mid':-9.73}
BENDS={'M03':('strip','incl'),'M02':('incl','mid'),'M01':('mid','frame')}
res={}
fig,axs=plt.subplots(1,3,figsize=(27,9))
for ax,(k,(fa,fb)) in zip(axs,BENDS.items()):
    p,q=GOLD[k]; ss,se=sorted([p@a,q@a]); G=gold_at(k,(ss+se)/2)
    t1=FLN[fa]; t2=FLN[fb]; d=wrap(t2-t1)
    if abs(d)>90: t2=t2+180; d=wrap(t2-t1)
    sk={}
    for flip in (0,180):
        T1=t1+flip; B=ang(T1+d/2)
        win=(np.linalg.norm(X-G,axis=1)<7)&(S>ss)&(S<se)&inpl
        Pm=X[win&(tw(T1+d/2)<2.5)].mean(0)
        Pq=np.vstack([X[win&(tw(T1+d/4)<2.5)].mean(0),X[win&(tw(T1+3*d/4)<2.5)].mean(0)])
        conv=float(((Pq-Pm)@B).mean())   # <0 convex(outer), >0 concave(inner)
        sk[flip]=dict(T1=T1,T2=T1+d,B=B,Pm=Pm,conv=conv)
    o=min(sk.values(),key=lambda z:z['conv']); i=max(sk.values(),key=lambda z:z['conv'])
    bis=o['B']
    L={}
    for nm,t in [('o1',o['T1']),('o2',o['T2']),('i1',i['T1']),('i2',i['T2'])]:
        sel=inpl&(tw(t)<1.2)&(np.linalg.norm(X-G,axis=1)<14)&(S>ss)&(S<se)
        P=X[sel]; nn=ang(t); off=P@nn
        near=np.argsort(np.linalg.norm(P-G,axis=1))[:max(10,len(P)//4)]
        med=np.median(off[near]); P=P[np.abs(off-med)<0.6]
        L[nm]=fit_line(P,nn)
    Xo=isect(*L['o1'],*L['o2']); Xi=isect(*L['i1'],*L['i2'])
    rows=[]
    for b0 in np.arange(np.floor(ss)-4,np.ceil(se)+4,1.0):
        sb=(S>=b0)&(S<b0+1)&inpl&(np.linalg.norm(X-G,axis=1)<7)
        so=sb&(tw(o['T1']+d/2)<2.5); si=sb&(tw(i['T1']+d/2)<2.5)
        apex=X[so].mean(0) if so.sum()>=2 else None; imid=X[si].mean(0) if si.sum()>=2 else None
        gp=gold_at(k,b0+0.5)
        rows.append(dict(s=float(b0+0.5),in_gold=bool(ss<=b0+0.5<=se),n_o=int(so.sum()),n_i=int(si.sum()),
            d_apex=None if apex is None else float(np.linalg.norm(gp-apex)),bis_apex=None if apex is None else float((gp-apex)@bis),
            d_imid=None if imid is None else float(np.linalg.norm(gp-imid)),apex_imid=None if (apex is None or imid is None) else float(np.linalg.norm(apex-imid))))
    ra=[r['d_apex'] for r in rows if r['in_gold'] and r['d_apex'] is not None]
    ri=[r['d_imid'] for r in rows if r['in_gold'] and r['d_imid'] is not None]
    rb=[r['bis_apex'] for r in rows if r['in_gold'] and r['bis_apex'] is not None]
    out=dict(bend=k,deflection_deg=abs(d),outer_normal_angles=[o['T1'],o['T2']],outer_conv=o['conv'],inner_conv=i['conv'],
             apex_mid=o['Pm'].tolist(),inner_mid=i['Pm'].tolist(),outer_sharp=Xo.tolist(),inner_sharp=Xi.tolist(),
             gold_mid=G.tolist(),gold_to_apex_mid=float(np.linalg.norm(G-o['Pm'])),gold_to_inner_mid=float(np.linalg.norm(G-i['Pm'])),
             gold_to_outer_sharp=float(np.linalg.norm(G-Xo)),gold_to_inner_sharp=float(np.linalg.norm(G-Xi)),
             apex_to_outer_sharp=float(np.linalg.norm(o['Pm']-Xo)),inner_mid_to_inner_sharp=float(np.linalg.norm(i['Pm']-Xi)),
             d_apex_bins=dict(median=float(np.median(ra)),p10=float(np.percentile(ra,10)),p90=float(np.percentile(ra,90)),min=float(min(ra)),max=float(max(ra)),n=len(ra)),
             bis_apex_bins_median=float(np.median(rb)),
             d_imid_bins=dict(median=float(np.median(ri)),p10=float(np.percentile(ri,10)),p90=float(np.percentile(ri,90)),min=float(min(ri)),max=float(max(ri)),n=len(ri)),
             rows=rows)
    res[k]=out
    print(k,'defl %.1f conv(o,i)=(%.2f,%.2f)'%(abs(d),o['conv'],i['conv']),'gold->apex %.2f gold->inner_mid %.2f gold->outer_sharp %.2f gold->inner_sharp %.2f apex->outer_sharp %.2f'%(out['gold_to_apex_mid'],out['gold_to_inner_mid'],out['gold_to_outer_sharp'],out['gold_to_inner_sharp'],out['apex_to_outer_sharp']))
    print('   bins d_apex',{kk:round(v,2) for kk,v in out['d_apex_bins'].items()},' bins d_imid',{kk:round(v,2) for kk,v in out['d_imid_bins'].items()},' signed along bisector (gold-apex) median %.2f'%out['bis_apex_bins_median'])
    # ---- plot 2D zoom: faces in gold s-range, colored by skin
    win=(np.linalg.norm(X-G,axis=1)<11)&(S>ss)&(S<se)
    ax.scatter(X[win,0],X[win,1],s=0.6,c='0.6',label='all faces (gold s-range)')
    so=win&inpl&(np.abs(wrap(TH-(o['T1']+d/2)))<abs(d)/2+0.5); si=win&inpl&(np.abs(wrap(TH-(i['T1']+d/2)))<abs(d)/2+0.5)
    ax.scatter(X[so,0],X[so,1],s=0.8,c='tab:blue',label='outer (convex) skin arc')
    ax.scatter(X[si,0],X[si,1],s=0.8,c='tab:purple',label='inner (concave) skin arc')
    for nm,c in [('o1','tab:blue'),('o2','tab:blue'),('i1','tab:purple'),('i2','tab:purple')]:
        c0,nn=L[nm]; dd=np.array([nn[1],-nn[0]]); tt=np.array([-14,14]); ax.plot(c0[0]+tt*dd[0],c0[1]+tt*dd[1],'--',color=c,lw=0.7)
    ax.plot(*o['Pm'],'D',color='cyan',mec='k',ms=9,label='outer apex (arc midline)')
    ax.plot(*i['Pm'],'s',color='violet',mec='k',ms=8,label='inner arc midline')
    ax.plot(*Xo,'x',color='tab:blue',ms=12,mew=2,label='outer virtual sharp'); ax.plot(*Xi,'x',color='tab:purple',ms=12,mew=2,label='inner virtual sharp')
    gs=gold_at(k,ss); ge=gold_at(k,se)
    ax.plot(*gs,'*',color='#1a9c3a',ms=16,mec='k',label='gold @ start'); ax.plot(*G,'*',color='lime',ms=13,mec='k',label='gold @ mid'); ax.plot(*ge,'*',color='darkgreen',ms=10,mec='k',label='gold @ end')
    circ=plt.Circle(o['Pm'],4,fill=False,ls=':',color='r'); ax.add_patch(circ)
    ax.set_xlim(G[0]-9,G[0]+9); ax.set_ylim(G[1]-9,G[1]+9); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('%s: profile projected along bend axis over the gold s-range [%.1f,%.1f]\n deflection %.1f deg; gold->outer apex %.2f mm, gold->inner arc mid %.2f mm\n(red dotted = 4 mm around outer apex)'%(k,ss,se,abs(d),out['gold_to_apex_mid'],out['gold_to_inner_mid']),fontsize=10)
    ax.legend(fontsize=7,loc='lower left' if k!='M01' else 'upper right')
plt.tight_layout(); plt.savefig(OUT+'a1_xsec_projected_all3.png',dpi=75)
json.dump(res,open(OUT+'a1_bendfit_bins.json','w'),indent=1)
