import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy"); nrm = np.load(W / "nrm.npy")
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy"); B = np.vstack([e1, e2, e3])
N = (nrm @ B.T) @ R.T
# plate mid-surface model z_mid(x,y) via robust plane on plate points away from walls/joggles
m = (np.abs(P[:, 0]) < 155) & (P[:, 1] > -150) & (P[:, 1] < 145) & (P[:, 2] < 0) & (np.abs(N[:, 2]) > 0.95)
A = np.c_[P[m, 0], P[m, 1], np.ones(m.sum())]; cf, *_ = np.linalg.lstsq(A, P[m, 2], rcond=None)
zmid = P[:, 0] * cf[0] + P[:, 1] * cf[1] + cf[2]
dz = P[:, 2] - zmid
wall_zone = (np.abs(P[:, 0]) <= 102) & ((P[:, 1] > 135) | (P[:, 1] < -140))
jog_zone = np.abs(np.abs(P[:, 0]) - 170) < 9
# 1) anything well above/below the plate outside wall and joggle zones
above = (dz > 2.5) & ~wall_zone & ~jog_zone
below = (dz < -2.5) & ~wall_zone & ~jog_zone
print("points >2.5 mm above plate mid outside wall/joggle zones:", int(above.sum()), "below <-2.5:", int(below.sum()))
for nm, mm in (("above", above), ("below", below)):
    if mm.sum():
        Q = P[mm]; print(f"  {nm}: x {Q[:,0].min():.1f}..{Q[:,0].max():.1f} y {Q[:,1].min():.1f}..{Q[:,1].max():.1f} z {Q[:,2].min():.1f}..{Q[:,2].max():.1f}")
        H, xe, ye = np.histogram2d(Q[:, 0], Q[:, 1], bins=[np.arange(-180, 181, 10), np.arange(-160, 161, 10)])
        idx = np.argwhere(H > 20)
        print("   dense 10mm cells:", [(float(xe[i]), float(ye[j]), int(H[i, j])) for i, j in idx[:20]])
# 2) tabs beyond wall ends: points with z>5 and |x|>103
tab = (P[:, 2] > 5) & (np.abs(P[:, 0]) > 103)
print("points z>5 beyond wall x-extent:", int(tab.sum()))
if tab.sum():
    Q = P[tab]; print("  extent", Q.min(0).round(1), Q.max(0).round(1))
# 3) window edge: any non-horizontal normals along the window rim (flange) beyond end caps
rim = (np.abs(P[:, 0]) < 75) & (np.abs(P[:, 1]) < 128) & (np.abs(N[:, 2]) < 0.5)
print("steep-normal points inside window bbox (rim end caps/flange):", int(rim.sum()))
if rim.sum():
    Q = P[rim]; print("  z range of steep rim points: %.2f..%.2f (plate mid ~%.2f; a flange would extend >3 mm)" % (Q[:, 2].min(), Q[:, 2].max(), np.median(zmid[rim])))
