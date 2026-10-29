"""Dense perpendicular stations along every gold line (every 2 mm), local two-skin corner fit,
outer apex, virtual sharps, flange tip; gold point offsets.  Read-only; writes a3x_stations.json."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from a3x_common import *

p = load_points()
F = json.load(open(OUT / "a3x_frame.json"))
O, X, Y, Z = (np.array(F[k]) for k in ("O", "X", "Y", "Z"))
loc = np.c_[(p - O) @ X, (p - O) @ Y, (p - O) @ Z]
order = np.argsort(loc[:, 0])
xs_sorted = loc[order, 0]
gold = load_gold()
GL = {bid: (np.array([(a - O) @ X, (a - O) @ Y, (a - O) @ Z]), np.array([(b - O) @ X, (b - O) @ Y, (b - O) @ Z])) for bid, a, b in gold}

# corner spec per gold line: a = web material direction from the corner, b = flange/tab material direction
SPEC = {
    "M01": dict(a=(+1, 0), b=(0, +1), corner="tab_A"),
    "M02": dict(a=(-1, 0), b=(0, -1), corner="flange_+Y"),
    "M03": dict(a=(-1, 0), b=(0, -1), corner="flange_+Y"),
    "M04": dict(a=(-1, 0), b=(0, +1), corner="tab_B"),
    "M05": dict(a=(+1, 0), b=(0, -1), corner="flange_-Y"),
    "M06": dict(a=(+1, 0), b=(0, -1), corner="flange_-Y"),
}
# corner seed in (Y,Z): side flanges from the outer flange skin, tabs from the gold M01/M04
SEED = {"flange_+Y": (109.6, 0.0), "flange_-Y": (-113.6, 0.0), "tab_A": None, "tab_B": None}


def slab(x0, half=0.6):
    i0, i1 = np.searchsorted(xs_sorted, [x0 - half, x0 + half])
    return loc[order[i0:i1]]


def gold_at(bid, x):
    a, b = GL[bid]
    t = (x - a[0]) / (b[0] - a[0])
    return a + t * (b - a), t


def plate_fit(P, C0, d, lo=3.5, hi=11.0, halfw=3.0):
    d = unit(np.array(d, float))
    n = np.array([-d[1], d[0]])
    r = P - C0
    al = r @ d
    of = r @ n
    m = (al > lo) & (al < hi) & (np.abs(of) < halfw)
    if m.sum() < 20:
        return None
    Q = P[m]
    # refine direction by PCA
    qc = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - qc, full_matrices=False)
    t = vt[0] if vt[0] @ d > 0 else -vt[0]
    if math.degrees(math.acos(min(1, abs(t @ d)))) > 25:
        t = d
    n = np.array([-t[1], t[0]])
    off = (Q - C0) @ n
    # two skins: two dominant peaks of the offset histogram (hole bores do not form peaks)
    h, e = np.histogram(off, bins=np.arange(-halfw, halfw + 0.1001, 0.1))
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    cen = 0.5 * (e[:-1] + e[1:])
    i1 = int(np.argmax(hs))
    cand = [j for j in range(len(hs)) if abs(cen[j] - cen[i1]) > 0.8 and abs(cen[j] - cen[i1]) < 2.6
            and hs[j] >= hs[max(j - 1, 0)] and hs[j] >= hs[min(j + 1, len(hs) - 1)]]
    skins, counts = [], []
    if cand:
        i2 = max(cand, key=lambda j: hs[j])
        if hs[i2] >= 0.15 * hs[i1]:
            for ic in sorted([i1, i2], key=lambda j: cen[j]):
                mm = np.abs(off - cen[ic]) < 0.3
                if mm.sum() >= 6:
                    skins.append(float(np.median(off[mm])))
                    counts.append(int(mm.sum()))
    if len(skins) < 2:
        skins = [float(np.median(off))]
        counts = [len(off)]
    return dict(t=t, n=n, skins=skins, counts=counts, n_pts=int(m.sum()), span=(float(((Q - C0) @ t).min()), float(((Q - C0) @ t).max())))


def isect(p1, t1, p2, t2):
    M = np.array([t1, -t2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    s, _ = np.linalg.solve(M, p2 - p1)
    return p1 + s * t1


def station(bid, x):
    sp = SPEC[bid]
    S = slab(x)
    P = S[:, 1:3]
    g3, tg = gold_at(bid, x)
    G = g3[1:3]
    rec = dict(x=float(x), t_gold=float(tg), gold_yz=G.tolist(), n_slab=int(len(P)))
    if len(P):
        dd = np.linalg.norm(P - G, axis=1)
        rec["d_gold_nearest_scan"] = float(dd.min())
    # corner seed
    if SEED[sp["corner"]] is not None:
        C0 = np.array(SEED[sp["corner"]], float)
    else:
        C0 = np.array([G[0], 0.0])
    a = np.array(sp["a"], float); b = np.array(sp["b"], float)
    # iterate: fit plates, recompute corner
    A = B = None
    for it in range(3):
        A = plate_fit(P, C0, a)
        B = plate_fit(P, C0, b)
        if A is None or B is None:
            break
        # mid-surfaces
        mA = C0 + A["n"] * np.mean(A["skins"]) if len(A["skins"]) == 2 else None
        mB = C0 + B["n"] * np.mean(B["skins"]) if len(B["skins"]) == 2 else None
        if mA is None or mB is None:
            break
        Xm = isect(mA, A["t"], mB, B["t"])
        if Xm is None:
            break
        C0 = Xm
    rec["web_present"] = A is not None
    rec["flange_present"] = B is not None
    if A is None or B is None or len(A["skins"]) < 2 or len(B["skins"]) < 2:
        rec["fold"] = False
        # gold relative to web mid-surface if web there
        Aw = plate_fit(P, np.array([G[0], 0.0]), a, lo=-6, hi=6)
        if Aw is not None and len(Aw["skins"]) == 2:
            rec["gold_minus_web_mid_mm"] = float((G - np.array([G[0], 0.0])) @ Aw["n"] - np.mean(Aw["skins"]))
        return rec
    rec["fold"] = True
    a_t, b_t = A["t"], B["t"]
    nA = A["n"] if A["n"] @ b_t > 0 else -A["n"]
    nB = B["n"] if B["n"] @ a_t > 0 else -B["n"]
    sA = sorted([s * (A["n"] @ nA) for s in A["skins"]])  # along nA: small = outer
    sB = sorted([s * (B["n"] @ nB) for s in B["skins"]])
    Ao = C0 + nA * sA[0]; Ai = C0 + nA * sA[1]
    Bo = C0 + nB * sB[0]; Bi = C0 + nB * sB[1]
    Xo = isect(Ao, a_t, Bo, b_t); Xi = isect(Ai, a_t, Bi, b_t)
    Xm = isect(0.5 * (Ao + Ai), a_t, 0.5 * (Bo + Bi), b_t)
    g = unit(a_t + b_t)
    inc = math.degrees(math.acos(np.clip(a_t @ b_t, -1, 1)))
    rec.update(bend_angle_deg=180 - inc, thick_web=sA[1] - sA[0], thick_flange=sB[1] - sB[0],
               Xo=Xo.tolist(), Xi=Xi.tolist(), Xm=Xm.tolist())
    r = P - Xo
    al = r @ g
    pe = np.abs(r @ np.array([-g[1], g[0]]))
    m = (pe < 0.35) & (al > -0.6) & (al < 4.0)
    if m.sum() >= 2:
        v = np.sort(al[m])
        delta = float(v[min(1, len(v) - 1)])
        apex = Xo + g * max(delta, 0.0)
        half = math.radians(inc) / 2
        k = 1 / math.sin(half) - 1
        rec["apex"] = apex.tolist(); rec["apex_delta"] = delta
        rec["R_outer"] = delta / k if k > 1e-6 else None
        dv = G - apex
        rec["d_apex"] = float(np.linalg.norm(dv))
        rec["gold_minus_apex_along_bisector"] = float(dv @ g)   # + = toward concave side
        rec["gold_minus_apex_across"] = float(dv @ np.array([-g[1], g[0]]))
    rec["d_outer_sharp"] = float(np.linalg.norm(G - Xo))
    rec["d_inner_sharp"] = float(np.linalg.norm(G - Xi)) if Xi is not None else None
    rec["d_mid_sharp"] = float(np.linalg.norm(G - Xm)) if Xm is not None else None
    # flange/tab tip (extreme along b among flange points within 3 mm of its mid-plane)
    rB = P - Xm
    alB = rB @ b_t
    ofB = rB @ nB - 0.5 * (sB[0] + sB[1]) + (Xm - C0) @ nB * 0  # approx
    mid_off = (P - (0.5 * (Bo + Bi))) @ nB
    mB = (np.abs(mid_off) < 2.5) & (alB > 2.0)
    if mB.sum() >= 5:
        tipd = float(np.max(alB[mB]))
        rec["tip_dist_from_mid_sharp"] = tipd
        tip = Xm + b_t * tipd
        rec["tip"] = tip.tolist()
        dv = G - tip
        rec["gold_minus_tip_along_flange"] = float(dv @ b_t)    # + = beyond tip (free air)
        rec["gold_minus_tip_across"] = float(dv @ nB)
        rec["d_tip"] = float(np.linalg.norm(dv))
    return rec


res = {}
for bid in GL:
    a, b = GL[bid]
    x0 = min(a[0], b[0]) - 14; x1 = max(a[0], b[0]) + 14
    xs = np.arange(math.floor(x0), math.ceil(x1) + 0.1, 2.0)
    rows = [station(bid, float(x)) for x in xs]
    res[bid] = rows
json.dump(res, open(OUT / "a3x_stations.json", "w"))

# summary
for bid, rows in res.items():
    a, b = GL[bid]
    gx0, gx1 = sorted([a[0], b[0]])
    fold = [r for r in rows if r.get("fold") and r.get("apex") is not None]
    fx = [r["x"] for r in fold]
    print(f"\n{bid}: gold X [{gx0:.1f},{gx1:.1f}] len {np.linalg.norm(b-a):.1f}")
    # fold intervals
    if fx:
        fx = np.array(fx)
        brk = np.where(np.diff(fx) > 2.5)[0]
        segs = np.split(fx, brk + 1)
        print("   fold present at X segments:", [(round(s[0], 1), round(s[-1], 1)) for s in segs])
    on = [r for r in fold if gx0 <= r["x"] <= gx1]
    if on:
        da = np.array([r["d_apex"] for r in on]); al = np.array([r["gold_minus_apex_along_bisector"] for r in on]); ac = np.array([r["gold_minus_apex_across"] for r in on])
        ang = np.array([r["bend_angle_deg"] for r in on]); Ro = np.array([r["R_outer"] for r in on if r.get("R_outer") is not None])
        print(f"   within gold span & fold: n={len(on)} d_apex med {np.median(da):.2f} p10 {np.percentile(da,10):.2f} p90 {np.percentile(da,90):.2f} max {da.max():.2f}; along-bisector med {np.median(al):+.2f}; across med {np.median(ac):+.2f}; angle med {np.median(ang):.1f}; R_outer med {np.median(Ro):.2f}")
        tips = [r for r in on if r.get("d_tip") is not None]
        if tips:
            print(f"   tip: dist corner->tip med {np.median([r['tip_dist_from_mid_sharp'] for r in tips]):.2f}; gold-tip along {np.median([r['gold_minus_tip_along_flange'] for r in tips]):+.2f} across {np.median([r['gold_minus_tip_across'] for r in tips]):+.2f} d_tip med {np.median([r['d_tip'] for r in tips]):.2f}")
    nof = [r for r in rows if not r.get("fold") and gx0 <= r["x"] <= gx1]
    print(f"   gold stations without fold: {len(nof)} of {len([r for r in rows if gx0 <= r['x'] <= gx1])}; their X:", (round(min(r['x'] for r in nof),1), round(max(r['x'] for r in nof),1)) if nof else None)
    if nof:
        dn = [r.get("d_gold_nearest_scan", np.nan) for r in nof]
        print(f"      no-fold stations: gold->nearest scan med {np.nanmedian(dn):.2f} max {np.nanmax(dn):.2f}")
