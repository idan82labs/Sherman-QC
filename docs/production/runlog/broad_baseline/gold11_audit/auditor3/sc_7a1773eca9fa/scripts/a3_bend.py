"""Auditor 3: independent per-station bend geometry around each gold line.

For each gold line:
  * pick the two plate normal axes near the line (axial normal modes),
  * split each plate into its two skins by normal sign, fit planes on the flats
    (lateral band away from the arc), classify outer/inner skin,
  * build a section frame (s along fold axis, x across, y along bisector toward
    the concave side, origin = outer virtual sharp from the outer-skin planes),
  * march stations along the axis: local outer apex (outer-skin points on the
    bisector), local inner apex, local outer sharp (local 2D line fits),
    fold existence, and the gold point's position in that frame.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_load import load  # noqa: E402

GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"


def unit(a):
    return a / np.linalg.norm(a)


def fit_plane(p, iters=3, keep=0.9):
    c = p.mean(0)
    for _ in range(iters):
        c = p.mean(0)
        w, U = np.linalg.eigh(np.cov((p - c).T))
        nrm = U[:, 0]
        r = np.abs((p - c) @ nrm)
        if len(p) < 30:
            break
        thr = np.quantile(r, keep)
        p = p[r <= thr]
    c = p.mean(0)
    w, U = np.linalg.eigh(np.cov((p - c).T))
    nrm = U[:, 0]
    rms = float(np.sqrt(np.mean(((p - c) @ nrm) ** 2)))
    return nrm, c, rms, len(p)


def axial_modes(normals, k=4, tol_deg=12.0, min_frac=0.03, seed=0):
    rem = normals.copy()
    modes = []
    cos_tol = np.cos(np.radians(tol_deg))
    rng = np.random.default_rng(seed)
    for _ in range(k):
        if len(rem) < 50:
            break
        cand = rem[rng.choice(len(rem), size=min(400, len(rem)), replace=False)]
        score = (np.abs(cand @ rem.T) > cos_tol).sum(1)
        best = cand[int(np.argmax(score))]
        sel = np.abs(rem @ best) > cos_tol
        w, U = np.linalg.eigh(rem[sel].T @ rem[sel])
        m = U[:, -1]
        sel = np.abs(rem @ m) > cos_tol
        if sel.sum() / len(normals) < min_frac:
            break
        modes.append(m)
        rem = rem[~sel]
    return modes


class Bend:
    def __init__(self, v, n, S, E, plate_axes=None, reach=40.0, ext=40.0, flat_band=(9.0, 35.0), tag=""):
        self.tag = tag
        self.S, self.E = np.asarray(S, float), np.asarray(E, float)
        self.L = float(np.linalg.norm(self.E - self.S))
        g = (self.E - self.S) / self.L
        self.g = g
        rel = v - self.S
        t = rel @ g
        w = rel - np.outer(t, g)
        lat = np.linalg.norm(w, axis=1)
        sel = (t > -ext) & (t < self.L + ext) & (lat < reach)
        self.P = v[sel]
        self.N = n[sel]
        self.lat_gold = lat[sel]
        self.t_gold = t[sel]
        if plate_axes is None:
            core = (self.lat_gold > 6) & (self.t_gold > 0) & (self.t_gold < self.L)
            modes = axial_modes(self.N[core], k=5)
            # choose the pair whose cross product is most parallel to the gold direction
            best = None
            for i in range(len(modes)):
                for j in range(i + 1, len(modes)):
                    ax = np.cross(modes[i], modes[j])
                    if np.linalg.norm(ax) < 0.2:
                        continue
                    ang = np.degrees(np.arccos(min(1, abs(unit(ax) @ g))))
                    if best is None or ang < best[0]:
                        best = (ang, i, j)
            plate_axes = (modes[best[1]], modes[best[2]])
        self.mA, self.mB = plate_axes
        self._fit_skins(flat_band)

    def _fit_skins(self, flat_band):
        P, N = self.P, self.N
        cos_tol = np.cos(np.radians(10))
        d0 = unit(np.cross(self.mA, self.mB))
        if d0 @ self.g < 0:
            d0 = -d0
        lo, hi = flat_band
        band = (self.lat_gold > lo) & (self.lat_gold < hi) & (self.t_gold > -5) & (self.t_gold < self.L + 5)
        skins = {}
        for name, m in (("A", self.mA), ("B", self.mB)):
            for sgn in (+1, -1):
                sel = band & ((N @ (sgn * m)) > cos_tol)
                if sel.sum() < 40:
                    continue
                nrm, c, rms, cnt = fit_plane(P[sel])
                if nrm @ (sgn * m) < 0:
                    nrm = -nrm
                skins[(name, sgn)] = dict(n=nrm, c=c, rms=rms, cnt=int(sel.sum()), idx=np.where(sel)[0])
        self.skins = skins
        # side directions: use any available skin of each plate
        def plate_side(name):
            ks = [k for k in skins if k[0] == name]
            k = max(ks, key=lambda kk: skins[kk]["cnt"])
            return k

        kA, kB = plate_side("A"), plate_side("B")
        # provisional line from the most populated skins
        p0, d = self._intersect(skins[kA]["n"], skins[kA]["c"], skins[kB]["n"], skins[kB]["c"])
        if d @ self.g < 0:
            d = -d
        uA = self._side_dir(P[skins[kA]["idx"]], p0, d, skins[kA]["n"])
        uB = self._side_dir(P[skins[kB]["idx"]], p0, d, skins[kB]["n"])
        # outer skin: outward normal points away from the other flange
        self.outer = {}
        self.inner = {}
        for name, u_other in (("A", uB), ("B", uA)):
            for sgn in (+1, -1):
                if (name, sgn) not in skins:
                    continue
                if skins[(name, sgn)]["n"] @ u_other < 0:
                    self.outer[name] = skins[(name, sgn)]
                else:
                    self.inner[name] = skins[(name, sgn)]
        if "A" not in self.outer or "B" not in self.outer:
            raise RuntimeError(f"{self.tag}: outer skins missing: have {list(self.outer)} inner {list(self.inner)}")
        oA, oB = self.outer["A"], self.outer["B"]
        p0, d = self._intersect(oA["n"], oA["c"], oB["n"], oB["c"])
        if d @ self.g < 0:
            d = -d
        uA = self._side_dir(P[oA["idx"]], p0, d, oA["n"])
        uB = self._side_dir(P[oB["idx"]], p0, d, oB["n"])
        self.d, self.p0, self.uA, self.uB = d, p0, uA, uB
        self.bis = unit(uA + uB)  # toward concave side
        x = unit(np.cross(self.bis, d))
        if x @ uA > 0:
            x = -x
        self.x = x  # plate A at negative x
        self.included = float(np.degrees(np.arccos(np.clip(uA @ uB, -1, 1))))
        self.bend_deg = 180.0 - self.included
        self.thick = {}
        for name in ("A", "B"):
            if name in self.outer and name in self.inner:
                o, i = self.outer[name], self.inner[name]
                self.thick[name] = float(abs((i["c"] - o["c"]) @ o["n"]))
        # inner sharp line if both inner skins exist
        self.inner_sharp = None
        if "A" in self.inner and "B" in self.inner:
            iA, iB = self.inner["A"], self.inner["B"]
            q0, dq = self._intersect(iA["n"], iA["c"], iB["n"], iB["c"])
            self.inner_sharp = (q0, dq)

    @staticmethod
    def _intersect(n1, c1, n2, c2):
        d = np.cross(n1, n2)
        d = d / np.linalg.norm(d)
        A = np.stack([n1, n2])
        b = np.array([n1 @ c1, n2 @ c2])
        near = (c1 + c2) / 2
        p0 = near + A.T @ np.linalg.solve(A @ A.T, b - A @ near)
        return p0, d

    @staticmethod
    def _side_dir(pts, p0, d, n):
        rel = pts - p0
        w = rel - np.outer(rel @ d, d)
        w = w - np.outer(w @ n, n)
        return unit(np.median(w, axis=0))

    # ---- section frame
    def to_sec(self, pts):
        rel = np.atleast_2d(pts) - self.p0
        return np.stack([rel @ self.d, rel @ self.x, rel @ self.bis], 1)  # s, x, y

    def gold_at(self, s):
        """Point of the (infinite) gold line whose section coordinate s equals s."""
        a = self.to_sec(self.S)[0]
        b = self.to_sec(self.E)[0]
        lam = (s - a[0]) / (b[0] - a[0])
        return a + lam * (b - a), lam

    def stations(self, step=1.0, slab=0.5, margin=40.0, lateral_apex=0.6):
        sec = self.to_sec(self.P)
        s_all, x_all, y_all = sec[:, 0], sec[:, 1], sec[:, 2]
        out_n = self.N @ (-self.bis)  # >0 : facing convex side
        a = self.to_sec(self.S)[0][0]
        b = self.to_sec(self.E)[0][0]
        s_lo, s_hi = min(a, b) - margin, max(a, b) + margin
        res = []
        # unit side directions in section coords
        uAx, uAy = self.uA @ self.x, self.uA @ self.bis
        uBx, uBy = self.uB @ self.x, self.uB @ self.bis
        nAo = self.outer["A"]["n"]; nBo = self.outer["B"]["n"]
        for s in np.arange(s_lo, s_hi + 1e-9, step):
            m = np.abs(s_all - s) <= slab
            if m.sum() < 5:
                res.append(dict(s=float(s), n=int(m.sum()), apex=None))
                continue
            xs, ys, on = x_all[m], y_all[m], out_n[m]
            Ns = self.N[m]
            # outer apex: points near bisector with outward normal facing convex side
            ap = None
            for lat in (lateral_apex, 2 * lateral_apex):
                c = (np.abs(xs) <= lat) & (on > 0.5) & (ys > -2.0) & (ys < 15.0)
                if c.sum() >= 3:
                    yy = np.sort(ys[c])
                    ap = float(np.median(yy[yy <= yy[0] + 0.6]))
                    break
            ip = None
            for lat in (lateral_apex, 2 * lateral_apex):
                c = (np.abs(xs) <= lat) & (on < -0.5) & (ys > -2.0) & (ys < 20.0)
                if c.sum() >= 3:
                    ip = float(np.median(ys[c]))
                    break
            # flange presence: outer or inner skin points of each plate 3..15 mm from the sharp along the side dir
            projA = xs * uAx + ys * uAy
            projB = xs * uBx + ys * uBy
            nA_ok = np.abs(Ns @ nAo) > np.cos(np.radians(12))
            nB_ok = np.abs(Ns @ nBo) > np.cos(np.radians(12))
            # perpendicular distance from the plate line (outer skin through origin along uA)
            perpA = np.abs(xs * (-uAy) + ys * uAx)
            perpB = np.abs(xs * (-uBy) + ys * uBx)
            hasA = int(((projA > 6) & (projA < 20) & nA_ok & (perpA < 4)).sum())
            hasB = int(((projB > 6) & (projB < 20) & nB_ok & (perpB < 4)).sum())
            gp, lam = self.gold_at(s)
            # nearest scan point to the gold point in this slab (2D)
            dd = np.hypot(xs - gp[1], ys - gp[2])
            res.append(dict(s=float(s), n=int(m.sum()), apex=ap, inner_apex=ip, hasA=hasA, hasB=hasB,
                            gold_x=float(gp[1]), gold_y=float(gp[2]), gold_lam=float(lam),
                            gold_nearest=float(dd.min()) if len(dd) else None))
        return res

    def summary(self):
        out = dict(tag=self.tag, L=self.L, axis=self.d.tolist(),
                   angle_gold_axis_deg=float(np.degrees(np.arccos(min(1, abs(self.g @ self.d))))),
                   included_deg=self.included, bend_deg=self.bend_deg, thickness=self.thick,
                   skins={f"{k[0]}{'+' if k[1] > 0 else '-'}": dict(cnt=vv["cnt"], rms=vv["rms"]) for k, vv in self.skins.items()},
                   outer=list(self.outer), inner=list(self.inner))
        return out


if __name__ == "__main__":
    v, n, f = load()
    gold = json.load(open(GOLD))["bends"]
    for b in gold:
        B = Bend(v, n, b["line_start"], b["line_end"], tag=b["bend_id"])
        sm = B.summary()
        print(json.dumps(sm, indent=None, default=float))
        st = B.stations(step=2.0 if B.L < 100 else 5.0)
        a = B.to_sec(B.S)[0][0]; e = B.to_sec(B.E)[0][0]
        print(f"  gold s-range [{min(a,e):.1f},{max(a,e):.1f}]")
        for r in st:
            if r.get("apex") is None and r.get("hasA") is None:
                continue
            ap = r.get("apex"); ip = r.get("inner_apex")
            dx = r["gold_x"]; dy = (r["gold_y"] - ap) if ap is not None else None
            print(f"  s={r['s']:7.1f} n={r['n']:5d} A={r['hasA']:4d} B={r['hasB']:4d} apex_y={ap if ap is None else round(ap,2)} "
                  f"inner_y={ip if ip is None else round(ip,2)} gold(x,y)=({r['gold_x']:.2f},{r['gold_y']:.2f}) lam={r['gold_lam']:.2f} "
                  f"near={r['gold_nearest']:.2f}")
