"""Read-only triage renders for sc_d45186653eed (profile slices + plan views)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/idantal/ShermanQC/FullProject/'
OUT=R+'docs/production/runlog/broad_baseline/overlays/sc_d45186653eed/triage/'
import os; os.makedirs(OUT,exist_ok=True)
z=np.load(R+'docs/production/runlog/benchmark_v2/label_prep/sc_d45186653eed/points_ds.npz'); pts=z['points']
L={}
for b in json.load(open(R+'docs/production/runlog/benchmark_v2/label_lines/sc_d45186653eed/lines.json'))['bends']:
    L['G'+b['bend_id']]=(np.array(b['label_line']['start']),np.array(b['label_line']['end']),'g')
for lane,f,c in [('bf_','best_f1/sc_d45186653eed/predictions_best_f1.json','r'),('rt_','runtime/sc_d45186653eed/predictions_runtime.json','b')]:
    for l in json.load(open(R+'docs/production/runlog/broad_baseline/runs/'+f))['owned_bend_regions']:
        L[lane+(l['bend_id'][4:12] if l['bend_id'].startswith('F1C_') else l['bend_id'].replace('CADF_',''))]=(np.array(l['line_start']),np.array(l['line_end']),c)
d=np.array([0.9429,0.0390,-0.3308]); d/=np.linalg.norm(d)
c0=pts.mean(0); X=pts-c0
perp=X-np.outer(X@d,d); w,V=np.linalg.eigh(np.cov(perp.T)); u=V[:,2]; v=np.cross(d,u)
F=lambda p:np.stack([(p-c0)@d,(p-c0)@u,(p-c0)@v],-1)
Q=F(pts)
np.save(OUT+'frame.npy',np.stack([c0,d,u,v]))
# plan views
for name,(i,j) in {'plan_axis_u':(0,1),'plan_axis_v':(0,2),'profile_u_v':(1,2)}.items():
    fig,ax=plt.subplots(figsize=(22,9) if name!='profile_u_v' else (12,12))
    ax.scatter(Q[:,i],Q[:,j],s=0.2,c='0.6',alpha=0.4)
    for k,(s,e,c) in L.items():
        a,b=F(s),F(e); ls='-' if c=='g' else '--'; lw=4 if c=='g' else 1.5
        ax.plot([a[i],b[i]],[a[j],b[j]],ls,color={'g':'tab:green','r':'tab:red','b':'tab:blue'}[c],lw=lw,alpha=.8)
        m=(a+b)/2; ax.text(m[i],m[j],k,fontsize=7,color={'g':'darkgreen','r':'darkred','b':'navy'}[c])
    ax.set_aspect('equal'); ax.set_title(name); fig.savefig(OUT+name+'.png',dpi=90,bbox_inches='tight'); plt.close(fig)
# profile slices at stations along the axis
stations=[-120,-60,-25,-5,10,25,60,120]
fig,axs=plt.subplots(2,4,figsize=(24,12))
for ax,s0 in zip(axs.flat,stations):
    m=np.abs(Q[:,0]-s0)<2.0; ax.scatter(Q[m,1],Q[m,2],s=2,c='0.3')
    for k,(s,e,c) in L.items():
        a,b=F(s),F(e); lo,hi=sorted([a[0],b[0]])
        if lo-1<=s0<=hi+1 and abs(b[0]-a[0])>1e-6:
            f=(s0-a[0])/(b[0]-a[0]); p=a+f*(b-a)
            ax.plot(p[1],p[2],{'g':'o','r':'x','b':'+'}[c],ms=12 if c=='g' else 10,mfc='none',color={'g':'tab:green','r':'tab:red','b':'tab:blue'}[c],mew=2)
            ax.text(p[1]+0.8,p[2]+0.8,k,fontsize=7)
    ax.set_aspect('equal'); ax.set_title('slice axis=%d mm'%s0); ax.grid(alpha=.3)
fig.savefig(OUT+'profile_slices.png',dpi=80,bbox_inches='tight')
print('axis range',Q[:,0].min(),Q[:,0].max())
for k,(s,e,c) in L.items(): print(k,np.round(F(s),1),np.round(F(e),1))
