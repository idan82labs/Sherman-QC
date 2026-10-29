import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage/triage_f247.py').read().split('# top & two')[0])
def basis(d,ref):
    d=d/np.linalg.norm(d); e1=ref-ref.dot(d)*d; e1/=np.linalg.norm(e1); e2=np.cross(d,e1); return d,e1,e2
def xsec(name,d,ref,stations,lines,fname,xl=None,yl=None):
    d,e1,e2=basis(np.array(d,float),np.array(ref,float))
    fig,ax=plt.subplots(1,len(stations),figsize=(8*len(stations),7))
    for a,st in zip(np.atleast_1d(ax),stations):
        st=np.array(st,float); m=np.abs((Q-st)@d)<1.5
        X=(Q[m]-st)@e1; Y=(Q[m]-st)@e2
        a.scatter(X,Y,s=1,c='k')
        for lbl,D,col,mk in lines:
            for k,(s0,e0) in D.items():
                s1,e1_=T(s0),T(e0); u=e1_-s1; 
                if abs(u/np.linalg.norm(u)@d)<0.9: continue
                t=((st-s1)@d)/(u@d); p=s1+t*u
                inside = -0.05<=t<=1.05
                a.plot((p-st)@e1,(p-st)@e2,mk,color=col,ms=10 if col=='g' else 7,mfc='none' if not inside else col,mew=2)
                a.annotate(k[-4:] if lbl!='gold' else k,((p-st)@e1,(p-st)@e2),color=col,fontsize=9,xytext=(4,4 if col=='g' else -10),textcoords='offset points')
        a.set_aspect('equal'); a.set_title(f'{name} station {np.round(st,0)}'); a.grid(alpha=.3)
        if xl: a.set_xlim(xl)
        if yl: a.set_ylim(yl)
    plt.tight_layout(); plt.savefig(OUT+fname,dpi=70); plt.close()
L=[('gold',G,'g','o'),('best',B,'r','x'),('ph',C,'b','+')]
dA=np.array([-0.996,-0.013,-0.089])
xsec('lines along PC1',dA,[0,1,0],[[100,0,0],[0,0,-9],[-80,0,-16]],L,'triage_xsec_pc1.png')
xsec('lines along PC2',[0,1,0],[1,0,0],[[0,-60,0],[0,0,0],[0,60,0]],L,'triage_xsec_pc2.png')
# zooms
xsec('top tab (M03/M04) zoom',dA,[0,1,0],[[-60,0,0],[0,0,-9],[40,0,0],[120,0,0]],L,'triage_zoom_M03M04.png',xl=(55,120),yl=(-25,20))
xsec('bottom tab (M05/M06) zoom',dA,[0,1,0],[[-60,0,0],[0,0,-9],[40,0,0],[120,0,0]],L,'triage_zoom_M05M06.png',xl=(-120,-55),yl=(-25,20))
xsec('wall M07/M08 zoom',[0,1,0],[1,0,0],[[0,-90,0],[0,0,0],[0,90,0]],L,'triage_zoom_M07M08.png',xl=(-135,-90),yl=(-55,25))
xsec('+PC1 lip zoom',[0,1,0],[1,0,0],[[0,-60,0],[0,0,0],[0,60,0]],L,'triage_zoom_pc1lip.png',xl=(115,160),yl=(-20,15))
