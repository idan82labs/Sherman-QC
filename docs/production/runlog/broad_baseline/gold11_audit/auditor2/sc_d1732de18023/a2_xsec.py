"""Auditor-2 independent cross-section measurement for sc_d1732de18023.

For every gold line: estimate the true bend axis from local skin planes, then at
stations every 2 mm along the axis (covering the whole scan, beyond the gold ends)
fit the four skin lines in a +-0.75 mm slab, locate outer apex (normal == bisector on
the convex skin), inner apex, virtual sharps, tangent lines; place the gold line in
that 2-D frame.  Also detects where the fold physically exists (outer-arc points with
bisector normal present) -> fold extent / coverage.  Writes a2_xsec.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_ply, unit  # noqa: E402

pts, nrm, _ = load_ply()
tree = cKDTree(pts)
c0 = pts.mean(0)
_, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
e3 = vt[2]                                   # plate normal; +e3 = side the long flanges rise to
G = load_gold()
COS8 = np.cos(np.radians(8.0))


def line_fit_const(vals):
    """robust constant (1-D) fit: median after MAD trimming."""
    v = np.asarray(vals)
    m = np.median(v)
    mad = np.median(np.abs(v - m)) * 1.4826 + 1e-3
    k = np.abs(v - m) < 3 * mad + 0.05
    return float(np.median(v[k])), float(np.std(v[k])), int(k.sum())


def global_frame(bid):
    a, b = G[bid]
    dg = unit(b - a)
    f = unit(np.cross(dg, e3))
    mid = (a + b) / 2
    if f @ (mid - c0) < 0:      # f points outward (away from part centre)
        f = -f
    # move the frame origin onto the local plate mid-plane (gold may sit off the plate)
    rel0 = pts - mid
    al0 = rel0 @ dg
    in0 = rel0 @ (-f)
    L0 = np.linalg.norm(b - a)
    pm = (np.abs(al0) < L0 / 2) & (in0 > 6) & (in0 < 30) & (np.abs(nrm @ e3) > COS8) & (np.abs(rel0 @ e3) < 12)
    lvl = np.median(rel0[pm] @ e3)
    mid = mid + e3 * lvl
    # refine plate normal and flange normal from points near the bend
    rel = pts - mid
    along = rel @ dg
    L = np.linalg.norm(b - a)
    inside = np.abs(along) < L / 2 + 5
    a2 = rel @ (-f)            # inward along plate
    b2 = rel @ e3
    platem = inside & (np.abs(nrm @ e3) > COS8) & (a2 > 6) & (a2 < 30) & (np.abs(b2) < 4)
    pn = nrm[platem] * np.sign(nrm[platem] @ e3)[:, None]
    n_plate = unit(pn.mean(0))
    flm = inside & (np.abs(nrm @ f) > COS8) & (np.abs(a2) < 8) & (np.abs(b2) > 5) & (np.abs(b2) < 25)
    fn = nrm[flm] * np.sign(nrm[flm] @ f)[:, None]
    n_fl = unit(fn.mean(0))
    d = unit(np.cross(n_plate, n_fl))
    if d @ dg < 0:
        d = -d
    # which way does the flange go (+e3 up or -e3 down)?
    updown = np.sign(np.median(b2[flm]))
    return dict(dg=dg, d=d, f=n_fl, e3=n_plate, mid=mid, L=L, updown=updown,
                ang_gold_axis=float(np.degrees(np.arccos(min(1, abs(dg @ d))))),
                n_plate_pts=int(platem.sum()), n_fl_pts=int(flm.sum()))


def station(fr, s, half=0.75, gold=True):
    d, f, up, mid = fr["d"], fr["f"], fr["e3"], fr["mid"]
    u_in = unit(np.cross(d, up))           # in-plate direction perpendicular to d
    if u_in @ f > 0:
        u_in = -u_in                         # inward
    w = unit(np.cross(u_in, d))              # plate normal orthogonalised
    if w @ up < 0:
        w = -w
    O = mid + d * s
    rel = pts - O
    ax = rel @ d
    m = np.abs(ax) < half
    A = rel[m] @ u_in
    B = rel[m] @ w
    near = (np.abs(A) < 40) & (np.abs(B) < 40)
    A, B = A[near], B[near]
    N = nrm[m][near]
    nA, nB = N @ u_in, N @ w
    idx = np.nonzero(m)[0][near]
    res = {"s": float(s), "n_slab": int(len(A))}
    if len(A) < 30:
        res["ok"] = False
        res["why"] = "few points"
        return res
    sgn = fr["updown"]                       # +1 flange rises (+w), -1 flange goes down
    # plate skins: horizontal faces, 6..30 mm inward, near plate level
    pl = (np.abs(nB) > COS8) & (A > 6) & (A < 30) & (np.abs(B) < 4)
    top = pl & (nB > 0)
    bot = pl & (nB < 0)
    # flange skins: vertical faces, |B| 5..(len) on the flange side, within 8 mm of bend
    fl = (np.abs(nA) > COS8) & (np.abs(A) < 8) & (sgn * B > 4.5) & (sgn * B < 25)
    fin = fl & (nA > 0)                      # faces inward (toward plate centre)  -> inner skin
    fout = fl & (nA < 0)                     # faces outward -> outer skin
    if min(top.sum(), bot.sum(), fin.sum(), fout.sum()) < 4:
        res["ok"] = False
        res["why"] = f"skin pts top{top.sum()} bot{bot.sum()} fin{fin.sum()} fout{fout.sum()}"
        res["has_plate"] = bool(top.sum() >= 4 and bot.sum() >= 4)
        res["has_flange"] = bool(fin.sum() >= 4 and fout.sum() >= 4)
        return res
    btop, sdt, _ = line_fit_const(B[top])
    bbot, sdb, _ = line_fit_const(B[bot])
    ain, sdi, _ = line_fit_const(A[fin])
    aout, sdo, _ = line_fit_const(A[fout])
    t_plate = btop - bbot
    t_fl = ain - aout
    # outer skins: convex side.  flange up (+w): outer = plate bottom + flange outside
    if sgn > 0:
        b_out, b_inn = bbot, btop
        nb_plate_out = np.array([0.0, -1.0])
    else:
        b_out, b_inn = btop, bbot
        nb_plate_out = np.array([0.0, 1.0])
    Xo = np.array([aout, b_out])             # outer virtual sharp (axis-aligned lines in this frame)
    Xi = np.array([ain, b_inn])              # inner virtual sharp
    nf_out = np.array([-1.0, 0.0])
    nbis = unit(nb_plate_out + nf_out)       # outward bisector normal (convex side)
    # small tilt correction: use measured skin normals' mean direction for plate & flange
    P2 = np.stack([A, B], 1)
    N2 = np.stack([nA, nB], 1)
    N2n = N2 / (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-9)
    # outer apex: convex-skin points with normal within 12 deg of bisector, within 4 mm of Xo
    dXo = np.linalg.norm(P2 - Xo, axis=1)
    cand = (N2n @ nbis > np.cos(np.radians(12))) & (dXo < 4.0)
    res.update(ok=True, t_plate=t_plate, t_flange=t_fl, Xo=Xo.tolist(), Xi=Xi.tolist(),
               nbis=nbis.tolist(), sd=[sdt, sdb, sdi, sdo], n_apex_cand=int(cand.sum()))
    if cand.sum() >= 2:
        apex = np.median(P2[cand], axis=0)
        res["apex"] = apex.tolist()
        res["apex_to_Xo"] = float(np.linalg.norm(Xo - apex))
        # R_o estimate for a 90 deg bend: |Xo-apex| = R_o (sqrt2-1)  (general: R(1/cos(th/2)-1))
        res["R_o_est"] = float(np.linalg.norm(Xo - apex) / (np.sqrt(2) - 1))
    # inner apex: concave-skin points with normal within 12 deg of -bisector... concave skin normals
    cand_i = (N2n @ (-nbis) > np.cos(np.radians(12))) & (np.linalg.norm(P2 - Xi, axis=1) < 4.0)
    res["n_inner_apex_cand"] = int(cand_i.sum())
    if cand_i.sum() >= 2:
        res["inner_apex"] = np.median(P2[cand_i], axis=0).tolist()
    # tangent lines on the outer skin: where outer plate skin / outer flange skin stop being flat
    if gold:
        a_, b_ = G_lines[fr["bid"]]
        dg = unit(b_ - a_)
        # gold point on this station plane: solve (a_+dg*t - O).d = 0
        t = ((O - a_) @ d) / (dg @ d)
        Lg = np.linalg.norm(b_ - a_)
        res["gold_t"] = float(t)
        res["gold_inside"] = bool(-1e-6 <= t <= Lg + 1e-6)
        Pg = a_ + dg * t
        g2 = np.array([(Pg - O) @ u_in, (Pg - O) @ w])
        res["gold2d"] = g2.tolist()
        dn, _ = tree.query(Pg)
        res["gold_d_scan"] = float(dn)
        if "apex" in res:
            v = g2 - np.array(res["apex"])
            res["gold_minus_apex"] = v.tolist()
            res["gold_apex_dist"] = float(np.linalg.norm(v))
            res["gold_along_bis_in"] = float(-(v @ nbis))   # + = toward concave side / arc centre
            perp = np.array([-nbis[1], nbis[0]])
            res["gold_across"] = float(v @ perp)
        res["gold_to_Xo"] = float(np.linalg.norm(g2 - Xo))
        res["gold_to_Xi"] = float(np.linalg.norm(g2 - Xi))
        if "inner_apex" in res:
            res["gold_to_inner_apex"] = float(np.linalg.norm(g2 - np.array(res["inner_apex"])))
        # distances to each skin line (signed)
        res["gold_rel_skins"] = {
            "B_minus_plate_top": float(g2[1] - btop), "B_minus_plate_bot": float(g2[1] - bbot),
            "A_minus_flange_in": float(g2[0] - ain), "A_minus_flange_out": float(g2[0] - aout)}
    return res


G_lines = G
out = {}
for bid in sorted(G):
    fr = global_frame(bid)
    fr["bid"] = bid
    a_, b_ = G[bid]
    # stations: cover gold span +-40 mm (fold may extend beyond the gold)
    s0 = (a_ - fr["mid"]) @ fr["d"]
    s1 = (b_ - fr["mid"]) @ fr["d"]
    lo, hi = min(s0, s1) - 40, max(s0, s1) + 40
    st = [station(fr, s) for s in np.arange(lo, hi + 1e-6, 2.0)]
    out[bid] = {"frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in fr.items()},
                "gold_s_range": [float(min(s0, s1)), float(max(s0, s1))],
                "stations": st}
    ok = [x for x in st if x.get("ok") and "apex" in x]
    gin = [x for x in ok if x.get("gold_inside")]
    print(f"{bid}: axis-vs-gold {fr['ang_gold_axis']:.2f} deg updown {fr['updown']:+.0f} "
          f"gold s [{min(s0,s1):.1f},{max(s0,s1):.1f}]  stations ok {len(ok)}/{len(st)} gold-inside {len(gin)}")
    if gin:
        da = np.array([x["gold_apex_dist"] for x in gin])
        bis = np.array([x["gold_along_bis_in"] for x in gin])
        acr = np.array([x["gold_across"] for x in gin])
        ds = np.array([x["gold_d_scan"] for x in gin])
        tp = np.array([x["t_plate"] for x in gin])
        ro = np.array([x.get("R_o_est", np.nan) for x in gin])
        print(f"   gold->apex median {np.median(da):.2f} [p10 {np.percentile(da,10):.2f} p90 {np.percentile(da,90):.2f}]"
              f"  along-bis(in) {np.median(bis):.2f}  across {np.median(acr):.2f}  d_scan {np.median(ds):.2f}"
              f"  t {np.median(tp):.2f}  Ro~{np.nanmedian(ro):.2f}")
        frac4 = np.mean(da <= 4.0)
        print(f"   frac stations gold within 4mm of apex: {frac4:.2f}")
json.dump(out, open(OUT / "a2_xsec.json", "w"), indent=1, default=float)
