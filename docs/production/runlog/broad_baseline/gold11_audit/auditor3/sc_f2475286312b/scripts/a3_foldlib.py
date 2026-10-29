"""Per-fold apex extraction from exact mesh/plane sections and gold-vs-apex measurement.

For every physical fold (found by eye in the overview/zoom sections) the fold axis, the section
plane coordinates and the two flange directions are given; at stations every 2 mm along the axis the
mesh is cut, both flanges' two skins are fitted with lines, giving the outer virtual sharp Xo, inner
virtual sharp Xi, the outer apex (closest outer-skin point to Xo = arc midpoint), radius, angle.
Each gold line is then measured against the fold it is closest to (in its own crossing section),
and the frozen matcher's pair_metrics(gold, apex_line) is computed for a perfect apex line.
Read-only on scan and gold; writes a3_folds.json in the auditor-3 folder.
"""
import json
import math
import sys
import numpy as np
from a3common import load_mesh, load_gold, slice_mesh, unit, OUT

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds, is_match  # noqa: E402

V, N, F = load_mesh()
G = load_gold()
fr = json.loads((OUT / "frame.json").read_text())
o, ea, eb, en = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([ea, eb, en])
VEC = {"a": ea, "b": eb, "h": en}
AX = {"a": 0, "b": 1, "h": 2}
TC = V[F].mean(1)

# fold definitions: axis, 2D coords (x,y), approx corner, flange dirs (away from corner), axis range to scan
FOLDS = {
    "LIP_+b (M01 side)": dict(axis="a", x="b", y="h", c0=(112.9, 0.0), fA=(-1, 0), fB=(0, 1), rng=(-152, 130)),
    "LIP_-b (M02 side)": dict(axis="a", x="b", y="h", c0=(-110.9, 0.0), fA=(1, 0), fB=(0, 1), rng=(-152, 130)),
    "TAB_+b (window b 64..82)": dict(axis="a", x="b", y="h", c0=(82.0, 0.0), fA=(1, 0), fB=(0, -1), rng=(-152, 130)),
    "TAB_-b (window b -90..-73)": dict(axis="a", x="b", y="h", c0=(-90.8, 0.0), fA=(-1, 0), fB=(0, -1), rng=(-152, 130)),
    "WALL a~113 (M07 side)": dict(axis="b", x="a", y="h", c0=(113.0, 0.0), fA=(-1, 0), fB=(0, -1), rng=(-118, 120)),
    "SHORT_LIP a~-140": dict(axis="b", x="a", y="h", c0=(-139.8, 0.0), fA=(1, 0), fB=(0, 1), rng=(-118, 120)),
    "EAR_+b (wall end)": dict(axis="h", x="a", y="b", c0=(113.0, 113.5), fA=(0, -1), fB=(1, 0), rng=(-66, 6)),
    "EAR_-b (wall end)": dict(axis="h", x="a", y="b", c0=(113.2, -111.0), fA=(0, 1), fB=(1, 0), rng=(-66, 6)),
}
STEP = 2.0


def section_pts(axis, val, center3, radius=40.0):
    nrm = VEC[axis]
    segs = slice_mesh(V, F, o + val * nrm, nrm, tri_centroid=TC, radius=radius + 5)
    if not len(segs):
        return np.zeros((0, 2, 3))
    return ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)


def fit_line(P):
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], float(s[1] / math.sqrt(len(P))) if len(P) > 2 else 0.0


def isect(p1, t1, p2, t2):
    M = np.array([t1, -t2]).T
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    s, _ = np.linalg.solve(M, p2 - p1)
    return p1 + s * t1


def closest_on_segs(S2, X):
    a0, a1 = S2[:, 0], S2[:, 1]
    d = a1 - a0
    L2 = (d * d).sum(1)
    t = np.clip(((X - a0) * d).sum(1) / np.where(L2 > 0, L2, 1), 0, 1)
    P = a0 + t[:, None] * d
    dist = np.linalg.norm(P - X, axis=1)
    i = int(np.argmin(dist))
    return P[i], float(dist[i])


def fit_station(fd, val):
    xi, yi = AX[fd["x"]], AX[fd["y"]]
    c0 = np.array(fd["c0"], float)
    fA, fB = unit(fd["fA"]), unit(fd["fB"])
    g = unit(fA + fB)
    # 3D centre of the window for triangle prefilter
    c3 = np.zeros(3)
    c3[AX[fd["axis"]]] = val
    c3[xi], c3[yi] = c0
    centre3 = o + c3 @ R
    segs = slice_mesh(V, F, centre3, VEC[fd["axis"]], tri_centroid=TC, radius=30.0)
    if not len(segs):
        return {"ok": False, "why": "no_section"}
    S = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
    S2 = S[:, :, [xi, yi]]
    P = S2.reshape(-1, 2)
    rel = P - c0
    res = {"ok": False}
    skins = {}
    for name, f in (("A", fA), ("B", fB)):
        nin = unit(g - (g @ f) * f)  # flange normal pointing to the inside
        along = rel @ f
        off = rel @ nin
        m = (along > 4.5) & (along < 10.5) & (np.abs(off) < 4.0)
        if m.sum() < 8:
            res["why"] = f"flange_{name}_missing"
            return res
        offs = off[m]
        # split into two skins at the largest gap in sorted offsets
        so = np.sort(offs)
        gaps = np.diff(so)
        if len(gaps) == 0 or gaps.max() < 0.5:
            res["why"] = f"flange_{name}_single_skin"
            # still use as outer if it's the outer one? mark unknown
            return res
        k = int(np.argmax(gaps))
        thr = 0.5 * (so[k] + so[k + 1])
        outer = m & (off < thr)
        inner = m & (off >= thr)
        if outer.sum() < 4 or inner.sum() < 4:
            res["why"] = f"flange_{name}_skin_split"
            return res
        po, to, ro = fit_line(P[outer])
        pi, ti, ri = fit_line(P[inner])
        skins[name] = dict(po=po, to=to, pi=pi, ti=ti, t=float(np.median(off[inner]) - np.median(off[outer])),
                           rms=max(ro, ri))
    Xo = isect(skins["A"]["po"], skins["A"]["to"], skins["B"]["po"], skins["B"]["to"])
    Xi = isect(skins["A"]["pi"], skins["A"]["ti"], skins["B"]["pi"], skins["B"]["ti"])
    if Xo is None or np.linalg.norm(Xo - c0) > 6:
        res["why"] = "no_corner_near_c0"
        return res
    # flange directions from the fitted skins (pointing away from corner)
    dA = skins["A"]["to"] * np.sign(skins["A"]["to"] @ fA)
    dB = skins["B"]["to"] * np.sign(skins["B"]["to"] @ fB)
    inc = math.degrees(math.acos(np.clip(dA @ dB, -1, 1)))
    gg = unit(dA + dB)
    apex, dap = closest_on_segs(S2, Xo)
    # apex must be on the inside of Xo (along gg) and close
    if (apex - Xo) @ gg < -0.3 or dap > 4.0:
        res["why"] = "apex_not_found"
        return res
    half = math.radians(inc) / 2
    Ro = dap / (1 / math.sin(half) - 1) if half < math.radians(89) else None
    C = Xo + gg * (Ro / math.sin(half)) if Ro is not None else None
    res.update(ok=True, Xo=Xo, Xi=Xi, apex=apex, d_apex_Xo=dap, included_deg=inc, bend_deg=180 - inc, g=gg,
               Ro=Ro, C=C, t=0.5 * (skins["A"]["t"] + skins["B"]["t"]), rms=max(skins["A"]["rms"], skins["B"]["rms"]))
    return res


def to3(fd, val, p2):
    v = np.zeros(3)
    v[AX[fd["axis"]]] = val
    v[AX[fd["x"]]], v[AX[fd["y"]]] = p2
    return o + v @ R


