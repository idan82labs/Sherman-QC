"""Dense per-station apex fit for every fold of sc_2714fb7e0c0b in the part frame (light, read-only).

For each fold the two flange directions are known in the part frame (axis-aligned part). At each
station along the fold axis we take a +-0.75 mm slab, fit the OUTER skin of each flange (points
4..14 mm from the corner, outer skin = the skin farther from the other flange), intersect them
(outer virtual sharp X), take the first scan point along the bisector from X (outer apex), and
report: fold present?, apex position, R_o estimate, and the gold point offset from the apex
(+along = toward the concave side / inside the bend, across = along the outer surface).
"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, gold_lines, OUT

# fold specs: axis index, transverse indices (i, j), nominal corner (ci, cj), flange dirs a, b in (i, j)
FOLDS = {
    "B01": dict(ax=0, tr=(1, 2), c=(0.0, 0.0), a=(1, 0), b=(0, -1), rng=(-226, 6)),
    "B02": dict(ax=0, tr=(1, 2), c=(225.0, 0.0), a=(-1, 0), b=(0, -1), rng=(-226, 6)),
    "B03": dict(ax=1, tr=(0, 2), c=(0.0, 0.0), a=(-1, 0), b=(0, -1), rng=(-6, 231)),
    "B05": dict(ax=1, tr=(0, 2), c=(0.0, -42.5), a=(0, 1), b=(1, 0), rng=(-6, 231)),
    "B04": dict(ax=1, tr=(0, 2), c=(46.0, -42.5), a=(-1, 0), b=(0, -1), rng=(-6, 231)),
    "B07": dict(ax=1, tr=(0, 2), c=(-210.5, 0.0), a=(1, 0), b=(0, -1), rng=(-6, 231)),
    "B09": dict(ax=0, tr=(1, 2), c=(31.0, -0.5), a=(-1, 0), b=(0, 1), rng=(-150, -40)),
    "B10": dict(ax=0, tr=(1, 2), c=(204.0, -0.5), a=(1, 0), b=(0, 1), rng=(-150, -40)),
    # end-flange folds of the lower plate (no gold line)
    "EF_v0": dict(ax=0, tr=(1, 2), c=(0.5, -42.5), a=(1, 0), b=(0, -1), rng=(-6, 52)),
    "EF_v225": dict(ax=0, tr=(1, 2), c=(224.5, -42.5), a=(-1, 0), b=(0, -1), rng=(-6, 52)),
    # vertical corners (gold B06/B08) treated as if folds: flange dirs along v and u in the u-v plane
    "B06": dict(ax=2, tr=(0, 1), c=(46.0, 0.8), a=(-1, 0), b=(0, 1), rng=(-60, -40)),
    "B08": dict(ax=2, tr=(0, 1), c=(46.0, 224.0), a=(-1, 0), b=(0, -1), rng=(-60, -40)),
}


def fit_line(P):
    m = P.mean(0)
    _, _, vt = np.linalg.svd(P - m, full_matrices=False)
    return m, vt[0]


def isect(p1, t1, p2, t2):
    M = np.array([t1, -t2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    s, _ = np.linalg.solve(M, p2 - p1)
    return p1 + s * t1


def outer_skin(Q, corner, fdir, odir, near=4.0, far=14.0):
    """Points of one flange (4..14 mm from the corner along fdir); return the outer skin line."""
    rel = Q - corner
    along = rel @ fdir
    side = rel @ odir  # odir points from this flange toward the other flange's material
    sel = (along > near) & (along < far) & (np.abs(side) < 4.0)
    if sel.sum() < 12:
        return None
    s = side[sel]
    lo = np.percentile(s, 3)
    sk = sel.copy()
    sk[sel] = s < lo + 0.6  # outer skin = smallest projection toward the other flange
    if sk.sum() < 8:
        return None
    m, t = fit_line(Q[sk])
    if t @ fdir < 0:
        t = -t
    if abs(t @ fdir) < math.cos(math.radians(15)):
        return None
    thick = None
    hi = s[(s > lo + 0.9) & (s < lo + 3.0)]
    if len(hi) > 6:
        thick = float(np.median(hi) - np.median(s[s < lo + 0.6]))
    return m, t, int(sk.sum()), thick


def station(Q, spec, gold_pt=None):
    a = np.array(spec["a"], float); b = np.array(spec["b"], float)
    corner = np.array(spec["c"], float)
    res = {"ok": False}
    A = outer_skin(Q, corner, a, b)
    B = outer_skin(Q, corner, b, a)
    if A is None or B is None:
        res["why"] = "flange_missing:" + ("A" if A is None else "") + ("B" if B is None else "")
        return res
    X = isect(A[0], A[1], B[0], B[1])
    if X is None or np.linalg.norm(X - corner) > 8:
        res["why"] = "bad_sharp"
        return res
    ta, tb = A[1], B[1]
    inc = math.degrees(math.acos(np.clip(ta @ tb, -1, 1)))
    g = ta + tb
    g /= np.linalg.norm(g)
    perp = np.array([-g[1], g[0]])
    rel = Q - X
    al = rel @ g
    pp = np.abs(rel @ perp)
    cand = (pp < 0.35) & (al > -0.6) & (al < 8)
    if cand.sum() < 2:
        res["why"] = "no_apex_points"
        return res
    vals = np.sort(al[cand])
    delta = float(np.median(vals[:3]))
    apex = X + g * delta
    half = math.radians(inc) / 2
    k = 1 / math.sin(half) - 1
    Ro = delta / k if k > 1e-6 else None
    C = X + g * (Ro / math.sin(half)) if Ro is not None else None
    # is there material continuity through the corner? count points within 0.6 mm of the expected arc
    arc_pts = 0
    if Ro is not None and Ro > 0.3:
        dC = np.linalg.norm(Q - C, axis=1)
        ang_ok = ((Q - C) @ (-g)) > np.cos(half) * dC * 0.0  # convex half
        arc_pts = int(((np.abs(dC - Ro) < 0.5) & ang_ok & (np.linalg.norm(Q - apex, axis=1) < Ro * 1.2 + 0.5)).sum())
    res.update(ok=True, X=X.tolist(), apex=apex.tolist(), g=g.tolist(), inc_deg=inc, bend_deg=180 - inc, Ro=Ro,
               C=None if C is None else C.tolist(), delta=delta, arc_pts=arc_pts, thick=(A[3], B[3]),
               n_skin=(A[2], B[2]), apex_gap=float(vals[0]))
    if gold_pt is not None:
        v = np.asarray(gold_pt, float) - apex
        res["gold_d_apex"] = float(np.linalg.norm(v))
        res["gold_along"] = float(v @ g)
        res["gold_across"] = float(v @ perp)
        res["gold_d_sharp"] = float(np.linalg.norm(np.asarray(gold_pt) - X))
        if C is not None:
            res["gold_d_centre"] = float(np.linalg.norm(np.asarray(gold_pt) - C))
    return res


def main():
    q, R, O, raw = load()
    G = gold_lines(R, O)
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    out = {}
    for name, spec in FOLDS.items():
        ax = spec["ax"]; i, j = spec["tr"]
        corner = np.array(spec["c"], float)
        win = (np.abs(q[:, i] - corner[0]) < 16) & (np.abs(q[:, j] - corner[1]) < 16)
        Qall = q[win]
        gl = G.get(name)
        rows = []
        for s in np.arange(spec["rng"][0], spec["rng"][1] + 1e-9, step if ax != 2 else 1.0):
            m = np.abs(Qall[:, ax] - s) < 0.75
            Q = Qall[m][:, [i, j]]
            gp = None
            if gl is not None:
                s0, s1 = gl
                d = s1 - s0
                if abs(d[ax]) > 1e-6:
                    t = (s - s0[ax]) / d[ax]
                    if 0 <= t <= 1:
                        x3 = s0 + t * d
                        gp = [x3[i], x3[j]]
            r = station(Q, spec, gp) if len(Q) > 30 else {"ok": False, "why": "few_points"}
            r["s"] = float(s); r["n"] = int(len(Q)); r["gold_in_range"] = gp is not None
            rows.append(r)
        out[name] = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
                     "gold": None if gl is None else [gl[0].tolist(), gl[1].tolist()], "stations": rows}
    (OUT / "a2_stations.json").write_text(json.dumps(out, indent=1, default=float))
    # compact summary
    for name, d in out.items():
        st = d["stations"]
        ok = [r for r in st if r["ok"] and r.get("Ro") is not None and 0.8 < r["Ro"] < 8 and 60 < r["bend_deg"] < 120]
        print(f"== {name}: stations {len(st)}, fold-like {len(ok)}")
        if ok:
            ss = np.array([r["s"] for r in ok])
            print(f"   fold-like s range {ss.min():.1f}..{ss.max():.1f}  median R_o {np.median([r['Ro'] for r in ok]):.2f}  "
                  f"median bend {np.median([r['bend_deg'] for r in ok]):.1f}")
        g = [r for r in ok if "gold_d_apex" in r]
        if g:
            da = np.array([r["gold_d_apex"] for r in g]); al = np.array([r["gold_along"] for r in g]); ac = np.array([r["gold_across"] for r in g])
            print(f"   gold stations {len(g)}: d_apex med {np.median(da):.2f} [{da.min():.2f}..{da.max():.2f}]  along med {np.median(al):.2f} "
                  f"[{al.min():.2f}..{al.max():.2f}]  across med {np.median(ac):.2f} [{ac.min():.2f}..{ac.max():.2f}]  "
                  f"d_centre med {np.median([r.get('gold_d_centre', np.nan) for r in g]):.2f}  d_sharp med {np.median([r['gold_d_sharp'] for r in g]):.2f}")

if __name__ == "__main__":
    main()
