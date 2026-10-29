"""Auditor 3: careful per-station measurement of each gold line vs the scanned fold.

Plate normal axes are chosen explicitly per bend (base vs flange/tab/wall), from
axial normal modes near the line, and verified on the section plots.
Per station (slab +-0.5 mm, perpendicular to the fitted fold axis):
  * 2D line fits of each visible skin (plate A outer/inner, plate B outer/inner)
    on the flats 3..22 mm from the corner,
  * outer virtual sharp Xo (outer skin lines), inner virtual sharp Xi,
  * bisector g (toward concave side),
  * outer apex: outer-skin scan points within 0.6 mm of the bisector line, the
    one nearest Xo (median of the near-surface layer), plus a circle fit of the
    outer arc (R_o, centre),
  * inner apex likewise (inner skin, if scanned),
  * gold crossing point, its offset from the outer apex (across, along g), and
    nearest-scan distance, inside-material test.
Fold existence per station: both plates' skins present within 3..15 mm of the corner
and arc points present in the bend zone.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_load import load  # noqa: E402

GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"

# explicit plate normal axes (from axial modes; A = base plate, B = flange/tab/wall)
PLATES = {
    "M01": ([-0.3862, 0.2398, -0.8907], [-0.6883, 0.5907, 0.4210], "base", "long front flange"),
    "M02": ([-0.4076, 0.2264, -0.8846], [-0.5823, -0.5937, 0.5554], "base", "left short tab"),
    "M03": ([-0.3840, 0.2355, -0.8928], [-0.3977, -0.9126, -0.0951], "base", "right short tab"),
    "M04": ([-0.3674, 0.2206, -0.9035], [-0.8071, -0.5582, 0.1924], "base", "left tall wall"),
    "M05": ([-0.3714, 0.2277, -0.9001], [-0.2540, -0.9447, 0.2075], "base", "right tall wall"),
}


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def fit_line2(p):
    c = p.mean(0)
    w, U = np.linalg.eigh(np.cov((p - c).T))
    t = U[:, 1]
    nrm = U[:, 0]
    res = (p - c) @ nrm
    return c, t, nrm, float(np.sqrt(np.mean(res ** 2)))


def robust_line2(p, iters=3):
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        c, t, nrm, rms = fit_line2(p[keep])
        r = np.abs((p - c) @ nrm)
        thr = max(0.15, 2.5 * np.median(r[keep]) * 1.4826)
        keep = r <= thr
        if keep.sum() < 6:
            break
    c, t, nrm, rms = fit_line2(p[keep])
    return c, t, nrm, rms, int(keep.sum())


def intersect2(c1, t1, c2, t2):
    A = np.array([t1, -t2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    lam = np.linalg.solve(A, c2 - c1)
    return c1 + lam[0] * t1


def fit_circle(p):
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    res = np.hypot(x - cx, y - cy) - r
    return np.array([cx, cy]), float(r), float(np.sqrt(np.mean(res ** 2)))


class Fold:
    def __init__(self, v, n, bid, S, E):
        self.bid = bid
        mA, mB, nameA, nameB = PLATES[bid]
        self.mA, self.mB = unit(mA), unit(mB)
        self.nameA, self.nameB = nameA, nameB
        self.t_nominal = 2.05
        self.S, self.E = np.asarray(S, float), np.asarray(E, float)
        self.L = float(np.linalg.norm(self.E - self.S))
        self.g3 = (self.E - self.S) / self.L
        d = unit(np.cross(self.mA, self.mB))
        if d @ self.g3 < 0:
            d = -d
        self.d = d
        self.mid = (self.S + self.E) / 2
        rel = v - self.mid
        s = rel @ d
        w = rel - np.outer(s, d)
        lat = np.linalg.norm(w, axis=1)
        half = self.L / 2 + 60
        sel = (np.abs(s) < half) & (lat < 45)
        self.P, self.N = v[sel], n[sel]
        # in-plane basis: e1 = mA projected, e2 = d x e1
        e1 = unit(self.mA - (self.mA @ d) * d)
        e2 = np.cross(d, e1)
        self.e1, self.e2 = e1, e2
        rel = self.P - self.mid
        self.s = rel @ d
        self.xy = np.c_[rel @ e1, rel @ e2]
        self.nxy = np.c_[self.N @ e1, self.N @ e2]
        self.nd = self.N @ d
        # 2D plate normals
        self.a2 = unit([self.mA @ e1, self.mA @ e2])
        self.b2 = unit([self.mB @ e1, self.mB @ e2])

    def gold2(self, s0):
        a = self.S - self.mid
        b = self.E - self.mid
        sa, sb = a @ self.d, b @ self.d
        lam = (s0 - sa) / (sb - sa)
        p = a + lam * (b - a)
        return np.array([p @ self.e1, p @ self.e2]), lam

    def gold_s_range(self):
        a = (self.S - self.mid) @ self.d
        b = (self.E - self.mid) @ self.d
        return min(a, b), max(a, b)

    def station(self, s0, slab=0.5, prov=None):
        m = np.abs(self.s - s0) <= slab
        xy, nxy = self.xy[m], self.nxy[m]
        out = dict(s=float(s0), n=int(m.sum()))
        if m.sum() < 20:
            out["ok"] = False
            out["why"] = "few points"
            return out
        G, lam = self.gold2(s0)
        out["gold_lam"] = float(lam)
        out["gold_xy"] = G.tolist()
        dd = np.hypot(*(xy - G).T)
        out["gold_nearest_scan"] = float(dd.min())
        cosA = np.abs(nxy @ self.a2) / np.maximum(np.linalg.norm(nxy, axis=1), 1e-9)
        cosB = np.abs(nxy @ self.b2) / np.maximum(np.linalg.norm(nxy, axis=1), 1e-9)
        # provisional corner: from previous station or from the gold point
        P0 = np.asarray(prov) if prov is not None else G
        skins = {}
        for pname, pn, cosv in (("A", self.a2, cosA), ("B", self.b2, cosB)):
            cand = cosv > np.cos(np.radians(8))
            for sgn in (1, -1):
                c2 = cand & ((nxy @ (sgn * pn)) > 0)
                if c2.sum() < 8:
                    continue
                pts = xy[c2]
                # keep points within 3..24 mm of the provisional corner
                r = np.hypot(*(pts - P0).T)
                pts = pts[(r > 2.5) & (r < 24)]
                if len(pts) < 8:
                    continue
                # several parallel surfaces may exist: cluster offsets along the plate normal, keep the one nearest P0
                off = (pts - P0) @ pn
                order = np.sort(off)
                gaps = np.where(np.diff(order) > 0.6)[0]
                bounds = np.r_[order[0] - 1, order[gaps] + 0.3, order[-1] + 1] if len(gaps) else np.r_[order[0] - 1, order[-1] + 1]
                best = None
                for i in range(len(bounds) - 1):
                    sel = (off > bounds[i]) & (off <= bounds[i + 1])
                    if sel.sum() < 8:
                        continue
                    score = abs(np.median(off[sel]))
                    if best is None or score < best[0]:
                        best = (score, sel)
                if best is None:
                    continue
                c, t, nrm, rms, cnt = robust_line2(pts[best[1]])
                # orient line normal along sgn*pn (outward normal of that skin)
                if nrm @ (sgn * pn) < 0:
                    nrm = -nrm
                skins[(pname, sgn)] = dict(c=c, t=t, n=nrm, rms=rms, cnt=cnt)
        out["skins"] = {f"{k[0]}{'+' if k[1] > 0 else '-'}": dict(cnt=vv["cnt"], rms=round(vv["rms"], 3)) for k, vv in skins.items()}
        # need at least one skin per plate to define the corner
        hasA = [k for k in skins if k[0] == "A"]
        hasB = [k for k in skins if k[0] == "B"]
        if not hasA or not hasB:
            out["ok"] = False
            out["why"] = f"plates missing A={len(hasA)} B={len(hasB)}"
            return out
        # side directions: from any skin of each plate, direction from corner toward the plate's points
        kA = max(hasA, key=lambda k: skins[k]["cnt"])
        kB = max(hasB, key=lambda k: skins[k]["cnt"])
        X0 = intersect2(skins[kA]["c"], skins[kA]["t"], skins[kB]["c"], skins[kB]["t"])
        if X0 is None:
            out["ok"] = False
            out["why"] = "parallel"
            return out

        def side_dir(k):
            t = skins[k]["t"]
            return t if (skins[k]["c"] - X0) @ t > 0 else -t

        uA, uB = side_dir(kA), side_dir(kB)
        # classify skins outer/inner: outer outward normal points away from the other plate
        outer, inner = {}, {}
        for k, sk in skins.items():
            u_other = uB if k[0] == "A" else uA
            (outer if sk["n"] @ u_other < 0 else inner)[k[0]] = sk
        out["outer_skins"] = sorted(outer)
        out["inner_skins"] = sorted(inner)
        synth = []
        for pl in ("A", "B"):
            if pl not in outer and pl in inner:
                # synthesize the outer skin from the inner skin offset by the sheet thickness
                th_known = [float(abs((inner[q]["c"] - outer[q]["c"]) @ outer[q]["n"])) for q in ("A", "B") if q in inner and q in outer]
                tk = th_known[0] if th_known else self.t_nominal
                sk = inner[pl]
                outer[pl] = dict(c=sk["c"] - tk * sk["n"], t=sk["t"], n=-sk["n"], rms=sk["rms"], cnt=0)
                synth.append(pl)
        out["outer_synthesized"] = synth
        if "A" not in outer or "B" not in outer:
            out["ok"] = False
            out["why"] = f"outer skin missing (outer={sorted(outer)}, inner={sorted(inner)})"
            out["X0"] = X0.tolist()
            return out
        Xo = intersect2(outer["A"]["c"], outer["A"]["t"], outer["B"]["c"], outer["B"]["t"])
        uA = outer["A"]["t"] if (outer["A"]["c"] - Xo) @ outer["A"]["t"] > 0 else -outer["A"]["t"]
        uB = outer["B"]["t"] if (outer["B"]["c"] - Xo) @ outer["B"]["t"] > 0 else -outer["B"]["t"]
        g = unit(uA + uB)
        across = np.array([-g[1], g[0]])
        if across @ uA > 0:
            across = -across  # plate A at negative 'across'
        included = float(np.degrees(np.arccos(np.clip(uA @ uB, -1, 1))))
        out.update(ok=True, Xo=Xo.tolist(), g=g.tolist(), across=across.tolist(), uA=uA.tolist(), uB=uB.tolist(),
                   included_deg=included, bend_deg=180 - included)
        Xi = None
        if "A" in inner and "B" in inner:
            Xi = intersect2(inner["A"]["c"], inner["A"]["t"], inner["B"]["c"], inner["B"]["t"])
            out["Xi"] = Xi.tolist()
            # thickness from parallel skins
        th = {}
        for pl in ("A", "B"):
            if pl in inner and pl in outer:
                th[pl] = float(abs((inner[pl]["c"] - outer[pl]["c"]) @ outer[pl]["n"]))
        out["thickness"] = th
        # local coordinates relative to Xo: (a = across, h = along g toward concave)
        rel = xy - Xo
        a_ = rel @ across
        h_ = rel @ g
        nout = -(nxy @ g)  # >0 means facing the convex side (outer)
        # outer apex: outer-facing points near bisector
        apex = None
        for lat in (0.6, 1.2):
            c = (np.abs(a_) <= lat) & (nout > 0.3) & (h_ > -1.5) & (h_ < 12)
            if c.sum() >= 3:
                hv = np.sort(h_[c])
                apex_h = float(np.median(hv[hv <= hv[0] + 0.5]))
                apex = Xo + apex_h * g
                out["apex_lat_used"] = lat
                break
        # inner apex: concave-facing points near bisector (beyond the outer apex)
        iapex = None
        if apex is not None:
            for lat in (0.6, 1.2):
                c = (np.abs(a_) <= lat) & (nout < -0.3) & (h_ > apex_h + 0.5) & (h_ < apex_h + 8)
                if c.sum() >= 3:
                    iapex = Xo + float(np.median(np.sort(h_[c])[:max(3, c.sum() // 3)])) * g
                    break
        # arc points: between the outer tangent points, outer-facing, not on the flats
        # tangent-agnostic: points whose normal is >8 deg from both plates, outer-facing, within 12 mm of Xo
        r0 = np.hypot(*rel.T)
        arc_o = (cosA < np.cos(np.radians(8))) & (cosB < np.cos(np.radians(8))) & (nout > 0.2) & (r0 < 14) & (h_ > -1.5)
        arc_i = (cosA < np.cos(np.radians(8))) & (cosB < np.cos(np.radians(8))) & (nout < -0.2) & (r0 < 16) & (h_ > -1.0)
        out["n_arc_outer"] = int(arc_o.sum())
        out["n_arc_inner"] = int(arc_i.sum())
        Co = Ro = None
        if arc_o.sum() >= 8:
            Co, Ro, rr = fit_circle(xy[arc_o])
            out["R_outer"] = Ro
            out["arc_outer_rms"] = rr
            out["C_outer"] = Co.tolist()
        if arc_i.sum() >= 8:
            Ci, Ri, rr = fit_circle(xy[arc_i])
            out["R_inner"] = Ri
            out["arc_inner_rms"] = rr
            out["C_inner"] = Ci.tolist()
        if apex is not None:
            out["apex"] = apex.tolist()
            dv = G - apex
            out["gold_minus_apex_across"] = float(dv @ across)
            out["gold_minus_apex_along_g"] = float(dv @ g)
            out["gold_to_apex"] = float(np.linalg.norm(dv))
        if iapex is not None:
            out["inner_apex"] = iapex.tolist()
            out["gold_to_inner_apex"] = float(np.linalg.norm(G - iapex))
        out["gold_to_Xo"] = float(np.linalg.norm(G - Xo))
        if Xi is not None:
            out["gold_to_Xi"] = float(np.linalg.norm(G - Xi))
        # gold relative to the outer-skin lines (signed: + = inside, toward the material)
        for pl, sk in outer.items():
            out[f"gold_depth_below_outer_{pl}"] = float(-(G - sk["c"]) @ sk["n"])
        for pl, sk in inner.items():
            out[f"gold_height_above_inner_{pl}"] = float((G - sk["c"]) @ sk["n"])
        # flange presence near the corner
        projA = rel @ uA
        projB = rel @ uB
        out["presA"] = int(((projA > 3) & (projA < 15) & (cosA > np.cos(np.radians(10)))).sum())
        out["presB"] = int(((projB > 3) & (projB < 15) & (cosB > np.cos(np.radians(10)))).sum())
        out["prov_next"] = Xo.tolist()
        return out


def run(bid, step=1.0, margin=40.0, verbose=False):
    v, n, f = load()
    gold = {b["bend_id"]: b for b in json.load(open(GOLD))["bends"]}
    b = gold[bid]
    F = Fold(v, n, bid, b["line_start"], b["line_end"])
    lo, hi = F.gold_s_range()
    res = []
    prov = None
    # march from the gold midpoint outward in both directions so the provisional corner tracks
    mids = np.arange(0.0, hi + margin + 1e-9, step)
    for direction in (1, -1):
        prov = None
        rng = np.arange(0.0 if direction == 1 else -step, (hi + margin) if direction == 1 else (lo - margin), direction * step)
        for s0 in rng:
            st = F.station(s0, prov=prov)
            if st.get("ok"):
                prov = st["prov_next"]
            res.append(st)
    res.sort(key=lambda r: r["s"])
    return F, res


if __name__ == "__main__":
    out = {}
    for bid in sys.argv[1:] or list(PLATES):
        F, res = run(bid)
        lo, hi = F.gold_s_range()
        out[bid] = dict(L=F.L, gold_s=[lo, hi], axis=F.d.tolist(),
                        angle_gold_axis_deg=float(np.degrees(np.arccos(min(1, abs(F.g3 @ F.d))))), stations=res)
        ok = [r for r in res if r.get("ok") and "gold_to_apex" in r and lo <= r["s"] <= hi]
        if ok:
            ga = np.array([r["gold_to_apex"] for r in ok])
            ac = np.array([r["gold_minus_apex_across"] for r in ok])
            al = np.array([r["gold_minus_apex_along_g"] for r in ok])
            bd = np.array([r["bend_deg"] for r in ok])
            ro = np.array([r.get("R_outer", np.nan) for r in ok])
            print(f"{bid}: L={F.L:.1f} gold-axis angle={out[bid]['angle_gold_axis_deg']:.2f} deg  stations ok in gold span: {len(ok)}")
            print(f"   gold->outer apex: median {np.median(ga):.2f}  p10 {np.percentile(ga,10):.2f} p90 {np.percentile(ga,90):.2f}  "
                  f"across med {np.median(ac):.2f}  along-g med {np.median(al):.2f}  bend med {np.median(bd):.1f}  R_o med {np.nanmedian(ro):.2f}")
    json.dump(out, open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_measure.json", "w"), default=float)
