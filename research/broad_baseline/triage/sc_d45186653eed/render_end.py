"""Read-only close-ups of the left end (B7/B8/B9, CAD_B5) and B6/F1C_24d1 flange for sc_d45186653eed."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/idantal/ShermanQC/FullProject/'
OUT=R+'docs/production/runlog/broad_baseline/overlays/sc_d45186653eed/triage/'
L={}
for b in json.load(open(R+'docs/production/runlog/benchmark_v2/label_lines/sc_d45186653eed/lines.json'))['bends']:
    L['G'+b['bend_id']]=(np.array(b['label_line']['start']),np.array(b['label_line']['end']),'g')
for lane,f,c in [('bf_','best_f1/sc_d45186653eed/predictions_best_f1.json','r'),('rt_','runtime/sc_d45186653eed/predictions_runtime.json','b')]:
    for l in json.load(open(R+'docs/production/runlog/broad_baseline/runs/'+f))['owned_bend_regions']:
        L[lane+(l['bend_id'][4:12] if l['bend_id'].startswith('F1C_') else l['bend_id'].replace('CADF_',''))]=(np.array(l['line_start']),np.array(l['line_end']),c)
import os
z=np.load(R+'docs/production/runlog/benchmark_v2/label_prep/sc_d45186653eed/points_ds.npz'); pts=z['points']; nrm=z['normals']
c0,d,u,v=np.load(OUT+'frame.npy')
F=lambda p:np.stack([(p-c0)@d,(p-c0)@u,(p-c0)@v],-1)
Q=F(pts); N=np.stack([nrm@d,nrm@u,nrm@v],-1)
# end region, several projections
m=Q[:,0]<-95
views={'end_u_v(look along axis)':(1,2),'end_axis_u':(0,1),'end_axis_v':(0,2)}
fig,axs=plt.subplots(1,3,figsize=(30,11))
for ax,(nm,(i,j)) in zip(axs,views.items()):
    ax.scatter(Q[m,i],Q[m,j],s=1,c=np.abs(N[m,0]),cmap='viridis',alpha=.5)
    for k,(s,e,c) in L.items():
        a,b=F(s),F(e)
        if min(a[0],b[0])>-95: continue
        ax.plot([a[i],b[i]],[a[j],b[j]],'-' if c=='g' else '--',color={'g':'tab:green','r':'tab:red','b':'tab:blue'}[c],lw=4 if c=='g' else 1.8,alpha=.8)
        ax.plot([a[i]],[a[j]],'o',color={'g':'tab:green','r':'tab:red','b':'tab:blue'}[c],ms=4)
        mm=(a+b)/2; ax.text(mm[i],mm[j],k,fontsize=8)
    ax.set_aspect('equal'); ax.set_title(nm+'  (color=|normal.axis|)'); ax.grid(alpha=.3)
fig.savefig(OUT+'end_region.png',dpi=70,bbox_inches='tight'); plt.close(fig)
# B7/B8 gap: points within 4 mm of the B7-B8 line, coordinate along line vs distance
s,e=L['GB7'][0],L['GB8'][1]; dd=(e-s)/np.linalg.norm(e-s)
X=pts-s; t=X@dd; r=np.linalg.norm(X-np.outer(t,dd),axis=1)
fig,ax=plt.subplots(figsize=(14,4)); mk=(r<4)&(t>-30)&(t<70)
ax.scatter(t[mk],r[mk],s=2); ax.set_xlabel('mm along B7->B8 line (0=B7 start)'); ax.set_ylabel('dist to line mm')
for k in ['GB7','GB8']:
    a=(L[k][0]-s)@dd; b=(L[k][1]-s)@dd; ax.axvspan(min(a,b),max(a,b),color='g',alpha=.15)
for k in ['bf_CAD_B2','rt_CAD_B2']:
    a=(L[k][0]-s)@dd; b=(L[k][1]-s)@dd; ax.plot([a,b],[3.6 if k[0]=='b' and k[1]=='f' else 3.8]*2,'r-' if k.startswith('bf') else 'b-',lw=3,label=k)
ax.legend(); ax.set_title('material along B7/B8 line (green = gold spans)')
fig.savefig(OUT+'b7_b8_gap.png',dpi=80,bbox_inches='tight'); plt.close(fig)
# B6 / F1C_24d1 left flange zoom slices
fig,axs=plt.subplots(1,5,figsize=(30,6))
for ax,s0 in zip(axs,[-100,-55,0,60,140]):
    mm=np.abs(Q[:,0]-s0)<1.0; ax.scatter(Q[mm,1],Q[mm,2],s=4,c='0.3')
    for k,(s,e,c) in L.items():
        a,b=F(s),F(e); lo,hi=sorted([a[0],b[0]])
        if lo-1<=s0<=hi+1:
            f=(s0-a[0])/(b[0]-a[0]); p=a+f*(b-a); ax.plot(p[1],p[2],{'g':'o','r':'x','b':'+'}[c],ms=12,mfc='none',mew=2,color={'g':'tab:green','r':'tab:red','b':'tab:blue'}[c]); ax.text(p[1]+.5,p[2]+.5,k,fontsize=7)
    ax.set_xlim(-45,-5); ax.set_ylim(-12,25); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title('left flange slice axis=%d'%s0)
fig.savefig(OUT+'left_flange_slices.png',dpi=80,bbox_inches='tight')
