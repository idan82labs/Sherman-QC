"""Auditor 2: precise fold extents (s) for the 4 formed folds, coverage of each fold by its gold
line, and the apex-line fit (raw scan coordinates) per fold from a2_fit.json apex points."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c
from a2_profiles import G, P, u, w, d, o

def runs(mask, s, step):
    out = []; cur = None
    for si, mi in zip(s, mask):
        if mi and cur is None: cur = [si, si]
        elif mi: cur[1] = si
        elif cur is not None: out.append(cur); cur = None
    if cur is not None: out.append(cur)
    return [(round(a - step / 2, 1), round(b + step / 2, 1)) for a, b in out]

step = 0.5
sgrid = np.arange(P[:, 2].min(), P[:, 2].max(), step)
order = np.argsort(P[:, 2]); Ps = P[order]; ss = Ps[:, 2]
def present(s0, ulo, uhi, wlo, whi):
    i0, i1 = np.searchsorted(ss, [s0 - step / 2, s0 + step / 2])
    q = Ps[i0:i1]
    return np.sum((q[:, 0] > ulo) & (q[:, 0] < uhi) & (q[:, 1] > wlo) & (q[:, 1] < whi)) >= 3
# second-leg material 4..8 mm from the plate, hugging the bend
spec = {
    "M01": (-85.5, -79.5, -8.0, -4.0),   # tab hanging down at u~-82
    "M04": (88.5, 97.5, -8.0, -4.0),     # tab hanging down at u~92-94
    "M02": (-114.5, -109.0, 4.0, 8.0),   # flange up at u~-112
    "M05": (108.5, 114.5, 4.0, 8.0),     # flange up at u~111
}
res = {}
for k, (ulo, uhi, wlo, whi) in spec.items():
    m = np.array([present(s0, ulo, uhi, wlo, whi) for s0 in sgrid])
    r = runs(m, sgrid, step)
    r = [x for x in r if x[1] - x[0] > 3]
    a, b = G[k]
    g0, g1 = sorted([a[2], b[2]])
    fold_len = sum(x[1] - x[0] for x in r)
    span = (min(x[0] for x in r), max(x[1] for x in r))
    cov = sum(max(0, min(x[1], g1) - max(x[0], g0)) for x in r)
    on = cov / (g1 - g0)
    res[k] = {"fold_runs_s": r, "fold_envelope_s": span, "fold_material_len_mm": round(fold_len, 1),
              "gold_s": (round(g0, 1), round(g1, 1)), "gold_len_mm": round(g1 - g0, 1),
              "fold_covered_by_gold_pct": round(100 * cov / fold_len, 1),
              "envelope_covered_by_gold_pct": round(100 * max(0, min(span[1], g1) - max(span[0], g0)) / (span[1] - span[0]), 1),
              "gold_span_on_fold_pct": round(100 * on, 1),
              "gold_span_inside_envelope_pct": round(100 * max(0, min(span[1], g1) - max(span[0], g0)) / (g1 - g0), 1)}
    print(k, res[k])

# apex-line fit in raw coordinates from a2_fit.json
fit = json.loads((c.OUT / "a2_fit.json").read_text())
R = np.stack([u, w, d])  # rows: frame axes; raw = o + uvw @ R
for k in ["M01", "M02", "M04", "M05"]:
    a, b = G[k]
    pts = []
    for x in fit[k]:
        if x["status"] != "ok":
            continue
        lam = x["lam"]; gp = a + lam * (b - a)
        ap = np.array([gp[0] + x["apex"][0], gp[1] + x["apex"][1], x["s"]])
        pts.append(ap)
    pts = np.array(pts)
    m = pts.mean(0); _, _, vt = np.linalg.svd(pts - m); dv = vt[0]
    if dv[2] < 0: dv = -dv
    resid = np.linalg.norm((pts - m) - np.outer((pts - m) @ dv, dv), axis=1)
    lo, hi = res[k]["fold_envelope_s"]
    e0 = m + (lo - m[2]) / dv[2] * dv; e1 = m + (hi - m[2]) / dv[2] * dv
    raw0 = o + e0 @ R; raw1 = o + e1 @ R
    gdir = (b - a) / np.linalg.norm(b - a)
    ang = np.degrees(np.arccos(abs(gdir @ dv)))
    # distance from apex-line to the gold line at fold mid
    print(k, f"apex-line fit n={len(pts)} rms {np.sqrt(np.mean(resid**2)):.2f} max {resid.max():.2f} mm; angle to gold {ang:.2f} deg")
    print("    apex line raw start", np.round(raw0, 2).tolist(), "end", np.round(raw1, 2).tolist())
    res[k]["apex_line_raw"] = [np.round(raw0, 3).tolist(), np.round(raw1, 3).tolist()]
    res[k]["apex_line_fit_rms_mm"] = round(float(np.sqrt(np.mean(resid**2))), 3)
    res[k]["apex_line_angle_to_gold_deg"] = round(float(ang), 3)
(c.OUT / "a2_extents.json").write_text(json.dumps(res, indent=1))
