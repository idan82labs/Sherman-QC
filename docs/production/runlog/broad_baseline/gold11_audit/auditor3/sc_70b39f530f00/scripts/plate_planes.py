"""Fit precise planes for each plate skin and report them in the common frame."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
from common import *

m = load_mesh()
lab = np.load(OUT + '/scripts/face_region_labels.npy')
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
cent = m.triangles_center; area = m.area_faces; fn = m.face_normals
regions = {'A_r1': [1], 'A_r2': [2], 'W_r6': [6], 'W_r4': [4], 'Ap_r0': [0], 'Ap_r3': [3], 'Ap_r5': [5], 'F_r241': [241], 'F_r82': [82]}
out = {}
for name, rids in regions.items():
    sel = np.isin(lab, rids)
    P = cent[sel]; A = area[sel]
    c = (P * A[:, None]).sum(0) / A.sum()
    X = (P - c) * np.sqrt(A[:, None])
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    if n @ (fn[sel] * A[:, None]).sum(0) < 0:
        n = -n
    res = (P - c) @ n
    s = (P - c0) @ d
    pu = (P - c0) @ u; pw = (P - c0) @ w
    nu, nw, nd = n @ u, n @ w, n @ d
    phi = math.degrees(math.atan2(nw, nu))
    # in-plane 2D line: offset along 2D normal
    n2 = np.array([nu, nw]) / math.hypot(nu, nw)
    off2 = float(np.array([(c - c0) @ u, (c - c0) @ w]) @ n2)
    out[name] = dict(n=n.tolist(), c=c.tolist(), phi_deg=phi, n_dot_d=float(nd), off2=off2,
                     rms=float(np.sqrt((res ** 2).mean())), area=float(A.sum()),
                     s_range=[float(s.min()), float(s.max())], u_range=[float(pu.min()), float(pu.max())],
                     w_range=[float(pw.min()), float(pw.max())])
    print(f"{name:7s} phi={phi:8.2f} n.d={nd:+.4f} off2={off2:7.2f} rms={out[name]['rms']:.3f} area={A.sum():7.1f} "
          f"s[{s.min():6.1f},{s.max():6.1f}] u[{pu.min():6.1f},{pu.max():6.1f}] w[{pw.min():6.1f},{pw.max():6.1f}]")
json.dump(out, open(OUT + '/scripts/plate_planes.json', 'w'), indent=1)
# thickness between paired skins
def thick(a, b):
    na = np.array(out[a]['n']); ca = np.array(out[a]['c']); cb = np.array(out[b]['c'])
    nb = np.array(out[b]['n'])
    return abs((cb - ca) @ na), math.degrees(math.acos(min(1, abs(na @ nb))))
for a, b in [('A_r1', 'A_r2'), ('W_r6', 'W_r4'), ('Ap_r0', 'Ap_r3'), ('Ap_r0', 'Ap_r5'), ('F_r241', 'F_r82'), ('A_r1', 'Ap_r0')]:
    t, ang = thick(a, b)
    print(a, b, 'separation', round(t, 3), 'normal angle', round(ang, 3))
