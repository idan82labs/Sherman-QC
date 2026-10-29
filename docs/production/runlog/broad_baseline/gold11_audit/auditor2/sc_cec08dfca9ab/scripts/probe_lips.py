from common import *
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
def side(name, wall_out, inward):
    print("==",name)
    prev=None
    for y0 in np.arange(-100,121,2.0):
        m=np.abs(Y-y0)<0.5
        # wall points: within 2.5 mm inward of the outer wall face
        dx=(X-wall_out)*inward
        w=m&(dx>-0.5)&(dx<2.5)
        wz=Z[w]
        wall_top=wz.max() if w.any() else np.nan
        # lip points: 4..13 mm inward, Z > 24
        l=m&(dx>4)&(dx<13)&(Z>24)
        # arc region points: 0..3.5 mm inward and Z>wall_top-4
        lipz=Z[l]
        # far region 13..40 mm inward Z>24 (back lip / corner overlap)
        f=m&(dx>13)&(dx<45)&(Z>24)
        s="y %6.1f wall_top %5.1f  lip n %4d  lipZ %s  far(13-45) n %4d farZ %s  wallZ>-5 n %d"%(y0,wall_top,l.sum(),
            ("%.1f..%.1f"%(lipz.min(),lipz.max()) if l.any() else "-"), f.sum(), ("%.1f..%.1f"%(Z[f].min(),Z[f].max()) if f.any() else "-"), (w&(Z>-5)).sum())
        print(s)
side("LEFT (HB4/HB5)", -158.1, +1)
side("RIGHT (HB3/HB6)", 155.2, -1)
