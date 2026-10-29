import numpy as np, open3d as o3d, sys
m=o3d.io.read_triangle_mesh(sys.argv[1]); m.compute_triangle_normals()
V=np.asarray(m.vertices); T=np.asarray(m.triangles); N=np.asarray(m.triangle_normals)
C=V[T].mean(1); A=np.linalg.norm(np.cross(V[T[:,1]]-V[T[:,0]],V[T[:,2]]-V[T[:,0]]),axis=1)/2
c0=C.mean(0); U,S,Wt=np.linalg.svd((C-c0)*np.sqrt(A)[:,None],full_matrices=False)
u,w,n=Wt
P=C-c0; pu,pw,pn=P@u,P@w,P@n
print('extent u',pu.min(),pu.max(),'w',pw.min(),pw.max(),'n',pn.min(),pn.max(), 'nfaces',len(T))
nn=N@n
for name,sel in (('top',nn>0.8),('bot',nn<-0.8)):
    sel=sel&(np.abs(pw)<2.0)
    st=np.arange(pu.min()+0.25,pu.max(),0.5)
    prof=[]
    for s in st:
        k=sel&(np.abs(pu-s)<0.25)
        if k.sum()>=3: prof.append((s,np.median(pn[k]),k.sum()))
    prof=np.array(prof)
    print(name)
    for s,h,c in prof: print(f'  u={s:6.2f} h={h:7.3f} n={int(c)}')
    # slope via normals: angle of normal in u-n plane
    ang=[]
    for s in st:
        k=sel&(np.abs(pu-s)<0.5)
        if k.sum()>=3:
            nv=N[k]; a=np.degrees(np.arctan2(np.median(nv@u),np.abs(np.median(nv@n))))
            ang.append((s,a))
    print('  normal tilt (deg) by u:',' '.join(f'{s:.1f}:{a:.1f}' for s,a in ang))
