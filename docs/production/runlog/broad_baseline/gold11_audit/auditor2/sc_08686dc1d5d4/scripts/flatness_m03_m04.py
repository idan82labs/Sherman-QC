import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
def plane(Q):
    A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]; cf, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
    r = Q[:, 2] - A @ cf; n = np.array([-cf[0], -cf[1], 1.0]); return cf, n / np.linalg.norm(n), r
def top_skin(mask):
    Q = P[mask]
    # split by local z: keep upper skin via residual sign to a provisional plane
    cf, n, r = plane(Q)
    return Q[r > 0]
# M03: plate on both sides, away from window (window spans approx x>-67 for |y|<~120) -> use y bands near the walls too
for yb in ((-150, -125), (-60, 60), (125, 148)):
    L = (P[:, 0] > -150) & (P[:, 0] < -100) & (P[:, 1] > yb[0]) & (P[:, 1] < yb[1]) & (P[:, 2] < 0) & (P[:, 2] > -9)
    Rr = (P[:, 0] > -86) & (P[:, 0] < -70) & (P[:, 1] > yb[0]) & (P[:, 1] < yb[1]) & (P[:, 2] < 0) & (P[:, 2] > -9)
    if L.sum() < 100 or Rr.sum() < 100: print("M03 band", yb, "insufficient"); continue
    TL = top_skin(L); TR = top_skin(Rr)
    cfl, nl, rl = plane(TL); cfr, nr, rr = plane(TR)
    ang = math.degrees(math.acos(np.clip(abs(nl @ nr), -1, 1)))
    # step at x=-92 (M03 locus)
    x0 = -92.0; ym = np.mean(yb)
    zl = cfl[0] * x0 + cfl[1] * ym + cfl[2]; zr = cfr[0] * x0 + cfr[1] * ym + cfr[2]
    print(f"M03 band y{yb}: top-skin dihedral across M03 = {ang:.3f} deg ; step at M03 = {zr - zl:+.3f} mm ; resid std L {rl.std():.3f} R {rr.std():.3f}")
# whole-strip flatness of the M03 neighbourhood: residual of top skin to one plane over x in [-150,-70]
m = (P[:, 0] > -150) & (P[:, 0] < -70) & (P[:, 1] > -60) & (P[:, 1] < 60) & (P[:, 2] < 0)
T = top_skin(m); cf, n, r = plane(T)
xb = np.arange(-150, -70, 5.0)
prof = [float(np.median(r[(T[:, 0] >= a) & (T[:, 0] < a + 5)])) for a in xb]
print("top-skin residual profile across M03 region (x bins of 5 mm, median, mm):", [round(v, 3) for v in prof])
# M04: profile of top and bottom skin across the raised strip x in [165,178]
for y0 in (-120, -60, 0, 60, 120):
    mm = (np.abs(P[:, 1] - y0) < 1.0) & (P[:, 0] > 165.5) & (P[:, 0] < 177.3) & (P[:, 2] < 2)
    Q = P[mm]; zc = np.median(Q[:, 2]); top = Q[Q[:, 2] > zc]; bot = Q[Q[:, 2] <= zc]
    out = []
    for nm, S in (("top", top), ("bot", bot)):
        a = S[S[:, 0] < 173.5]; b = S[S[:, 0] > 175.5]
        ca = np.polyfit(a[:, 0], a[:, 2], 1); cb = np.polyfit(b[:, 0], b[:, 2], 1)
        ang = math.degrees(abs(math.atan(ca[0]) - math.atan(cb[0])))
        out.append(f"{nm}: slope-change across M04 {ang:.2f} deg, resid std {np.std(S[:,2]-np.polyval(np.polyfit(S[:,0],S[:,2],1),S[:,0])):.3f}")
    a, b = G["M04"]; t = (y0 - a[1]) / (b[1] - a[1]); gp = a + t * (b - a)
    print(f"M04 strip y={y0}: gold x={gp[0]:.2f} z={gp[2]:.2f} | " + " | ".join(out) + f" | strip top/bot z at gold x ~ {np.median(top[np.abs(top[:,0]-gp[0])<0.6,2]):.2f}/{np.median(bot[np.abs(bot[:,0]-gp[0])<0.6,2]):.2f}")
