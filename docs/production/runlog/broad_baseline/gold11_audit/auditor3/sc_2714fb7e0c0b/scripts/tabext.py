import numpy as np, json
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
V=np.load(W+'V.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T
E=json.load(open(W+'extent.json'))
for bid in ('B09','B10'):
    c=(np.array(E[bid]['apex_line_point'])-o)@R.T; d=np.array(E[bid]['apex_line_dir'])@R.T
    print(bid,'apex line local point',c.round(2),'dir',d.round(4))
    # apex X,Z at the line; wall plane X at that Y
    for y0 in list(range(-70,-60,1))+list(range(8,22,1)):
        t=(y0-c[1])/d[1]; p=c+t*d
        m=np.abs(L[:,1]-y0)<0.5
        q=L[m]
        dap=np.hypot(q[:,0]-p[0],q[:,2]-p[2])
        nap=(dap<1.0).sum()
        # main wall: points with Z in [-24,-16] within 1.2 mm of the main wall plane X (wall outer/inner skins near apex X)
        wall=((q[:,2]<-16)&(q[:,2]>-24)&(np.abs(q[:,0]-p[0])<1.6)).sum()
        plate=((np.abs(q[:,2]-p[2])<1.5)&(np.abs(q[:,0]-p[0])>4)&(np.abs(q[:,0]-p[0])<10)).sum()
        print(f'  Y={y0:4d} apex_pts={nap:3d} wall_pts={wall:3d} plate_pts={plate:3d}')
