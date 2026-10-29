"""Slice-level debug of Z3 corner birth near a line: re-slices one family of the scan (frozen slicer functions) at
the levels crossing a given 3-D segment and reports, per level, the chains passing near the segment, their flats
(a copy of chain_corners' flat extraction with the rejection reason of every run) and the corners found.
usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.n1_diag.chain_debug DIAG_DIR FAMILY 'x0,y0,z0,x1,y1,z1' OUT.pkl"""
from __future__ import annotations
import math, pickle, sys
import numpy as np
from research.zeroshot.z3_slicing import slicer as SL


def flats_debug(ch, prm, t):
    Q = ch["pts"]
    if len(Q) < 6:
        return {"reason": "chain<6pts", "runs": []}
    seglen = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    eps = max(prm.dp_eps_mm, 0.06 * t)
    idx = SL._dp(Q, eps)
    runs = []
    for a, b in zip(idx[:-1], idx[1:]):
        if runs:
            i0, i1 = runs[-1]
            d_prev = Q[i1] - Q[i0]; d_cur = Q[b] - Q[a]
            cosang = (d_prev @ d_cur) / (np.linalg.norm(d_prev) * np.linalg.norm(d_cur) + 1e-12)
            if cosang > math.cos(math.radians(prm.merge_deg)):
                runs[-1] = (i0, b)
                continue
        runs.append((a, b))
    out = []
    for i0, i1 in runs:
        L = arc[i1] - arc[i0]
        r = {"i0": int(i0), "i1": int(i1), "len": float(L), "p0": Q[i0].tolist(), "p1": Q[i1].tolist()}
        if L < prm.min_flat_mm:
            r["reject"] = "short<min_flat"
        else:
            trim = min(0.3, 0.15 * L)
            sel = np.nonzero((arc >= arc[i0] + trim) & (arc <= arc[i1] - trim))[0]
            if len(sel) < 3:
                sel = np.arange(i0, i1 + 1)
            c, d, rms = SL._tls(Q[sel])
            r["rms"] = float(rms)
            if rms > prm.flat_rms_mm:
                r["reject"] = "rms"
            elif len(sel) >= 6:
                X = Q[sel] - c; x = X @ d; rr = d[0] * X[:, 1] - d[1] * X[:, 0]
                A = np.stack([x * x, x, np.ones_like(x)], 1)
                coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
                noise = float(np.sqrt(np.mean((A @ coef - rr) ** 2)))
                sag = abs(coef[0]) * (0.5 * L) ** 2
                if abs(coef[0]) > 1e-9 and sag > max(0.05, 3.0 * noise) and 1.0 / (2.0 * abs(coef[0])) < min(8.0 * L, 40.0):
                    r["reject"] = "arc_bow"
        out.append(r)
    return {"reason": None, "runs": out, "arc_len": float(arc[-1])}


def main(diag_dir, fam, segtxt, out):
    from pathlib import Path
    R = pickle.loads((Path(diag_dir) / "diag.pkl").read_bytes())
    z = np.load(Path(diag_dir) / "mesh.npz")
    mesh = SL.mesh_from_arrays(z["V"], z["F"])
    prm = SL.Params(); t = R["z3"]["t_mm"]
    f = R["z3"]["families"][int(fam)]
    u = np.array(f["axis"]); e1, e2 = SL.orthonormal_frame(u)
    v = [float(x) for x in segtxt.split(",")]
    a, b = np.array(v[:3]), np.array(v[3:])
    levels, chains = SL.slice_family(mesh, u, prm.step_mm)
    h0, h1 = sorted([a @ u, b @ u])
    res = []
    for k, lev in enumerate(levels):
        if not (h0 - 4 <= lev <= h1 + 4):
            continue
        tt = (lev - a @ u) / ((b - a) @ u)
        x3 = a + tt * (b - a)
        x2 = np.array([x3 @ e1, x3 @ e2])
        near = []
        for ch in chains[k]:
            dd = np.linalg.norm(ch["pts"] - x2, axis=1)
            if dd.min() < 12.0:
                fd = flats_debug(ch, prm, t)
                cs = SL.chain_corners(ch, prm, t)
                near.append({"pts": ch["pts"], "nrm": ch["nrm"], "dmin": float(dd.min()), "flats": fd,
                             "corners": [{"apex": c["apex"].tolist(), "incl": c["incl"], "skin": c["skin"], "la": c["la"],
                                          "lb": c["lb"], "joggle": bool(c.get("joggle")),
                                          "d": float(np.linalg.norm(c["apex"] - x2))} for c in cs]})
        res.append({"k": k, "lev": float(lev), "s": float(tt * np.linalg.norm(b - a)), "x2": x2.tolist(), "chains": near})
    pickle.dump({"u": u, "e1": e1, "e2": e2, "t": t, "levels": res}, open(out, "wb"))
    for r in res:
        line = f"k={r['k']:3d} s={r['s']:6.1f} chains={len(r['chains'])}"
        for ch in r["chains"]:
            fl = ch["flats"]["runs"]
            line += f" | d={ch['dmin']:.1f} arc={ch['flats'].get('arc_len', 0):.0f} runs=" + ",".join(
                f"{x['len']:.1f}{'' if 'reject' not in x else '[' + x['reject'] + ']'}" for x in fl if x["len"] > 0.5) + \
                " corners=" + ",".join(f"{c['incl']:.0f}{c['skin'][0]}@{c['d']:.1f}" for c in ch["corners"])
        print(line)


if __name__ == "__main__":
    main(*sys.argv[1:5])
