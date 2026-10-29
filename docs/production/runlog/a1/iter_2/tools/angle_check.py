"""Independent angle check: robust planes on each flange's OML skin at several depth windows from the measured sharp."""
import json, math, sys
from pathlib import Path
import numpy as np
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.geom import robust_plane

run = Path(sys.argv[1])
res = json.loads((run / "result.json").read_text())
geo = json.loads((run / "bend_geometry.json").read_text())
pts = read_scan(run / "registered_scan.ply").points
from scipy.spatial import cKDTree
tree = cKDTree(pts)
rows = {b["bend_uid"]: b for b in res["bends"]}
def unit(v): return v / np.linalg.norm(v)
for g in geo["bends"]:
    r = rows[g["bend_uid"]]
    ang = next((c for c in r["characteristics"] if c["kind"] == "INCLUDED_ANGLE"), None)
    if ang is None or ang["state"] != "MEASURED" or g.get("measured_polyline_mm") is None or g.get("side_a_anchor_mm") is None or g.get("side_b_anchor_mm") is None:
        continue
    p0, p1 = np.asarray(g["measured_polyline_mm"][0]), np.asarray(g["measured_polyline_mm"][-1])
    a = unit(p1 - p0); m = 0.5 * (p0 + p1); L = np.linalg.norm(p1 - p0)
    out = []
    for win in ((3, 10), (5, 20), (10, 40)):
        ws = []
        ok = True
        for key in ("side_a_anchor_mm", "side_b_anchor_mm"):
            anc = np.asarray(g[key]); w = anc - m; w = unit(w - (w @ a) * a)
            n = np.cross(a, w)
            idx = np.asarray(tree.query_ball_point(m + w * 0.5 * (win[0] + win[1]), r=math.hypot(0.5 * L, 0.5 * (win[1] - win[0])) + 3))
            if len(idx) == 0: ok = False; break
            q = pts[idx] - m
            s = q @ w; z = q @ a; h = q @ n
            # OML skin passes through the sharp line: |h| small; take the skin nearest h=0 (exclude other skin at ~t)
            sel = (s > win[0]) & (s < win[1]) & (np.abs(z) < 0.45 * L) & (np.abs(h) < 0.6)
            if sel.sum() < 30: ok = False; break
            pl = robust_plane(pts[idx][sel], iterations=4)
            nn = pl.n
            wf = np.cross(nn, a); wf = unit(wf if wf @ w > 0 else -wf)
            ws.append((wf, int(sel.sum()), pl.rms))
        if not ok:
            out.append(f"{win}: n/a"); continue
        inc = math.degrees(math.acos(max(-1, min(1, float(ws[0][0] @ ws[1][0])))))
        out.append(f"{win}: {inc - ang['expected']:+.2f} (n {ws[0][1]}/{ws[1][1]} rms {ws[0][2]:.3f}/{ws[1][2]:.3f})")
    print(f"{r['label']:>4} eng {ang['deviation']:+.2f}±{ang['uncertainty']:.2f}  " + "  ".join(out))
