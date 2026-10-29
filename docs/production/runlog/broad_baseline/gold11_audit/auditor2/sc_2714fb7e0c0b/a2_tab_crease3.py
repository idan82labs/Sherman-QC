"""Tab-end crease angle from 3D skin-plane fits over all heights (w 2..14)."""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, OUT

q, R, O, raw = load()

def plane(P):
    m = P.mean(0); _, s, vt = np.linalg.svd(P - m, full_matrices=False); return m, vt[2], s

def two_skin_normal(P):
    m, n, _ = plane(P)
    off = (P - m) @ n
    # split at the largest gap in the offset histogram near zero
    o = np.sort(off); gaps = np.diff(o); k = np.argmax(gaps * ((o[:-1] > -1.2) & (o[1:] < 1.2)))
    thr = 0.5 * (o[k] + o[k + 1])
    res = []
    for S in (P[off < thr], P[off >= thr]):
        if len(S) >= 20:
            mm, nn, ss = plane(S)
            res.append((mm, nn if nn @ n > 0 else -nn, len(S), float(ss[2] / np.sqrt(len(S)))))
    return m, n, res, float(o[k + 1] - o[k]), float(np.median(off[off >= thr]) - np.median(off[off < thr]))

out = {}
for name, vlo, vhi in (("B09_tab_end", 24.0, 34.0), ("B10_tab_end", 200.0, 212.0)):
    base = (q[:, 2] > 2.0) & (q[:, 2] < 14.0) & (q[:, 1] > vlo) & (q[:, 1] < vhi)
    main = q[base & (q[:, 0] > -80) & (q[:, 0] < -62.0)]
    slant = q[base & (q[:, 0] > -56.8) & (q[:, 0] < -53.4)]
    mA, nA, skA, gapA, thA = two_skin_normal(main)
    mB, nB, skB, gapB, thB = two_skin_normal(slant)
    ang_mid = math.degrees(math.acos(min(1, abs(nA @ nB))))
    ang_sk = [math.degrees(math.acos(min(1, abs(a[1] @ b[1])))) for a in skA for b in skB]
    # crease line = intersection of the two mid-planes; report its direction and position
    d = np.cross(nA, nB); d /= np.linalg.norm(d)
    print(f"{name}: main pts {len(main)} skins {[s[2] for s in skA]} thick {thA:.2f} | slant pts {len(slant)} skins {[s[2] for s in skB]} thick {thB:.2f}")
    print(f"   crease angle (mid-planes) {ang_mid:.1f} deg ; skin-pair angles {[round(a,1) for a in ang_sk]} ; crease dir (u,v,w) {np.round(d,3).tolist()}")
    out[name] = {"angle_mid_deg": ang_mid, "skin_pair_angles": ang_sk, "crease_dir_uvw": d.tolist(),
                 "main_normal": nA.tolist(), "slant_normal": nB.tolist(), "thickness_main": thA, "thickness_slant": thB}
(OUT / "a2_tab_crease3.json").write_text(json.dumps(out, indent=1))
