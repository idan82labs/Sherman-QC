import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage/triage_f247.py').read().split('# top & two')[0])
m=(Q[:,0]<-90)&(Q[:,0]>-130)&(Q[:,1]<-95)
fig,ax=plt.subplots(1,3,figsize=(24,8))
for a,(i,j) in zip(ax,[(0,1),(1,2),(0,2)]):
    a.scatter(Q[m,i],Q[m,j],s=1,c=Q[m,3-i-j],cmap='viridis')
    for D,col in [(G,'g'),(B,'r'),(C,'b')]:
        for k,(s0,e0) in D.items():
            s1,e1=T(s0),T(e0); a.plot([s1[i],e1[i]],[s1[j],e1[j]],'-',color=col,lw=2); a.text(*(0.5*(s1+e1))[[i,j]],k[-4:],color=col)
    a.set_xlim(Q[m,i].min()-3,Q[m,i].max()+3); a.set_ylim(Q[m,j].min()-3,Q[m,j].max()+3); a.set_aspect('equal'); a.set_title(f'corner x<-90,y<-95: PC{i+1}-PC{j+1}')
plt.tight_layout(); plt.savefig(OUT+'triage_zoom_corner_63f.png',dpi=60)
h=np.load(S+"h.npy")
w=(Q[:,0]<-114)&(h>35)
print('points x<-114 & h<-35:',w.sum(), 'y range', Q[w,1].min() if w.any() else None, Q[w,1].max() if w.any() else None)
for y0 in range(-100,101,20):
    mm=(np.abs(Q[:,1]-y0)<2)&(h>30)
    print(y0, 'min x of low wall pts', Q[mm,0].min().round(1), 'max x', Q[mm,0].max().round(1), 'min h', h[mm].max().round(1))
