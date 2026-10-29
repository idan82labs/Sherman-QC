"""Read-only CAD-to-scan rigid registration + per-bend matching of CAD bends
(V10 + lenient candidates) against an AI label's lines.

Usage: align_and_match.py <step> <raw_json> <prep_dir> <lines_json> <out_json> [<out_png>]
The scan side uses the label_prep downsampled points (same raw scan frame as the
AI lines). Nothing outside research/broad_baseline and the runlog is written."""
import itertools, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.broad_baseline.cad_crosscheck.render_cad import tessellate  # noqa: E402


def sample_tris(T, n, rng):
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    idx = rng.choice(len(T), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    t = T[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def rotations24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            out.append(R)
    return out


def icp(src, dst_tree, dst, R, t, iters, max_d):
    for _ in range(iters):
        x = src @ R.T + t
        d, j = dst_tree.query(x, distance_upper_bound=max_d)
        m = np.isfinite(d)
        if m.sum() < 50:
            break
        a, b = x[m], dst[j[m]]
        ca, cb = a.mean(0), b.mean(0)
        U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        dR = Vt.T @ D @ U.T
        R, t = dR @ R, dR @ t + cb - dR @ ca
    return R, t


def register(cad_pts, scan_pts):
    """Find R,t mapping CAD -> scan. Score = fraction of SCAN points within 2 mm
    of CAD (partial scans are subsets of CAD)."""
    rng = np.random.default_rng(0)
    cad = cad_pts[rng.choice(len(cad_pts), min(20000, len(cad_pts)), replace=False)]
    scan = scan_pts[rng.choice(len(scan_pts), min(20000, len(scan_pts)), replace=False)]
    cad_tree, scan_tree = cKDTree(cad), cKDTree(scan)
    cc, sc = cad.mean(0), scan.mean(0)
    _, Vc = np.linalg.eigh(np.cov((cad - cc).T)); _, Vs = np.linalg.eigh(np.cov((scan - sc).T))
    best = None
    cands = []
    for P in rotations24():
        R0 = Vs @ P @ Vc.T
        if np.linalg.det(R0) < 0:
            continue
        cands.append(R0)
    # also try centroid offsets for partial scans: align scan centroid to CAD centroid only
    for R0 in cands:
        t0 = sc - R0 @ cc
        R, t = icp(cad, scan_tree, scan, R0, t0, 30, 30.0)
        R, t = icp(cad, scan_tree, scan, R, t, 30, 8.0)
        R, t = icp(cad, scan_tree, scan, R, t, 30, 3.0)
        # score scan->CAD
        Rinv = R.T; back = (scan - t) @ Rinv.T
        d, _ = cad_tree.query(back)
        frac = float((d < 2.0).mean()); rms = float(np.sqrt(np.mean(np.minimum(d, 10) ** 2)))
        dc, _ = scan_tree.query(cad @ R.T + t)
        cov = float((dc < 2.0).mean())
        score = frac + 0.25 * cov
        if best is None or score > best["score"]:
            best = {"R": R, "t": t, "score": score, "scan_inlier_frac_2mm": frac, "cad_coverage_2mm": cov, "rms_clip10_mm": rms}
    return best


def measure_scan_angle(scan, s, e, rmin, rmax):
    d = e - s; L = np.linalg.norm(d); d = d / L
    rel = scan - s; u = rel @ d
    m = (u > 0.1 * L) & (u < 0.9 * L)
    perp = rel[m] - u[m, None] * d
    r = np.linalg.norm(perp, axis=1)
    k = (r > rmin) & (r < rmax)
    P = perp[k]
    if len(P) < 60:
        return {"status": "TOO_FEW_POINTS", "n": int(len(P))}
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    side = P @ Vt[0] > 0
    outs = []
    for sd in (side, ~side):
        Q = P[sd]
        if len(Q) < 30:
            return {"status": "ONE_SIDE_MISSING", "n": [int(side.sum()), int((~side).sum())]}
        _, _, W = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
        v = W[0] if W[0] @ Q.mean(0) > 0 else -W[0]
        outs.append(v)
    inc = math.degrees(math.acos(float(np.clip(outs[0] @ outs[1], -1, 1))))
    return {"status": "OK", "scan_bend_angle_deg": round(180 - inc, 1), "n": [int(side.sum()), int((~side).sum())],
            "r_window_mm": [round(rmin, 1), round(rmax, 1)]}


def main():
    step, raw, prep, lines_p, out = sys.argv[1:6]
    png = sys.argv[6] if len(sys.argv) > 6 else None
    R = json.loads(Path(raw).read_text())
    T = tessellate(step)
    rng = np.random.default_rng(1)
    cad_pts = sample_tris(T, 80000, rng)
    z = np.load(Path(prep) / "points_ds.npz"); scan = z["points"]
    L = json.loads(Path(lines_p).read_text())
    reg = register(cad_pts, scan)
    Rm, tm = reg["R"], reg["t"]
    scan_tree = cKDTree(scan)
    cad_bends = []
    for i, a in enumerate(R.get("v10_bend_axes", [])):
        cad_bends.append({"cad_id": f"C{i+1}", "source": "V10", "angle_deg": a["angular_sweeps_deg"][0],
                          "span_mm": a["axial_span_mm"], "s": np.array(a["line_start"]), "e": np.array(a["line_end"]),
                          "radii_mm": a["radii_mm"]})
    v10_faces = set()
    for a in R.get("v10_bend_axes", []):
        v10_faces |= {int(x) for x in a["bend_id"].split("_F", 1)[1].split("_")}
    k = 0
    for g in R.get("lenient_coaxial_groups", []):
        if g["kind"] != "BEND_CANDIDATE" or set(g["face_indices"]) & v10_faces:
            continue
        k += 1
        sw = max(g["sweep_by_radius_deg"].values())
        cad_bends.append({"cad_id": f"L{k}", "source": "LENIENT_NOT_V10", "angle_deg": sw, "span_mm": g["max_axial_span_mm"],
                          "s": np.array(g["canonical_line"][0]), "e": np.array(g["canonical_line"][1]), "radii_mm": g["radii_mm"]})
    out_bends = []
    for cb in cad_bends:
        s, e = Rm @ cb["s"] + tm, Rm @ cb["e"] + tm
        ts = np.linspace(0.05, 0.95, 40)
        pts = s[None] + ts[:, None] * (e - s)[None]
        d, _ = scan_tree.query(pts)
        cb.update(s_scan=s, e_scan=e, scan_support=float((d < 3.0).mean()))
    for cb in cad_bends:
        cdir = (cb["e_scan"] - cb["s_scan"]); cdir /= np.linalg.norm(cdir)
        near = 1e9
        for ob in cad_bends:
            if ob is cb: continue
            od = ob["e_scan"] - ob["s_scan"]; od /= np.linalg.norm(od)
            if abs(od @ cdir) > 0.98:
                w = (ob["s_scan"] + ob["e_scan"]) / 2 - cb["s_scan"]
                pd = np.linalg.norm(w - cdir * (w @ cdir))
                # only if axially overlapping
                u0, u1 = sorted([(ob["s_scan"] - cb["s_scan"]) @ cdir, (ob["e_scan"] - cb["s_scan"]) @ cdir])
                if min(u1, np.linalg.norm(cb["e_scan"] - cb["s_scan"])) > max(u0, 0):
                    near = min(near, pd)
        rm = float(np.mean(cb["radii_mm"]))
        rmin = max(3.0, rm * math.radians(cb["angle_deg"]) / 2 + 2.0)
        rmax = min(18.0, 0.8 * near)
        cb["scan_measure"] = measure_scan_angle(scan, cb["s_scan"], cb["e_scan"], rmin, rmax) if rmax - rmin >= 2.5 else {"status": "WINDOW_TOO_NARROW", "r_window_mm": [round(rmin, 1), round(rmax, 1)]}
    ai = []
    for b in L.get("bends", []):
        if b.get("status") != "ok" or not b.get("label_line"):
            continue
        ai.append({"bend_id": b["bend_id"], "s": np.array(b["label_line"]["start"]), "e": np.array(b["label_line"]["end"]),
                   "angle_deg": b.get("bend_angle_deg"), "length_mm": b.get("length_mm")})
    n, m = len(cad_bends), len(ai)
    cost = np.full((n, m), 1e6)
    detail = {}
    for i, cb in enumerate(cad_bends):
        cs, ce = cb["s_scan"], cb["e_scan"]; cdir = (ce - cs) / np.linalg.norm(ce - cs); clen = np.linalg.norm(ce - cs)
        for j, a in enumerate(ai):
            adir = (a["e"] - a["s"]) / max(1e-9, np.linalg.norm(a["e"] - a["s"]))
            ang = math.degrees(math.acos(min(1, abs(float(adir @ cdir)))))
            am = (a["s"] + a["e"]) / 2
            w = am - cs; perp = float(np.linalg.norm(w - cdir * (w @ cdir)))
            u0, u1 = sorted([float((a["s"] - cs) @ cdir), float((a["e"] - cs) @ cdir)])
            ov = max(0.0, min(u1, clen) - max(u0, 0.0)) / max(1e-6, min(clen, u1 - u0))
            detail[(i, j)] = (ang, perp, ov)
            if ang < 12 and perp < 10 and ov > 0.3:
                cost[i, j] = ang / 12 + perp / 10 + (1 - ov)
    ri, ci = linear_sum_assignment(cost) if n and m else ([], [])
    match = {i: j for i, j in zip(ri, ci) if cost[i, j] < 1e5}
    rows = []
    for i, cb in enumerate(cad_bends):
        j = match.get(i)
        rows.append({"cad_id": cb["cad_id"], "source": cb.get("source", "V10"), "scan_measure": cb["scan_measure"], "angle_deg": round(cb["angle_deg"], 2), "span_mm": round(cb["span_mm"], 1),
                     "scan_support_3mm": round(cb["scan_support"], 2),
                     "matched_ai_bend": ai[j]["bend_id"] if j is not None else None,
                     "ai_angle_deg": round(ai[j]["angle_deg"], 1) if j is not None else None,
                     "ai_length_mm": round(ai[j]["length_mm"], 1) if j is not None else None,
                     "match_geom": [round(x, 2) for x in detail[(i, j)]] if j is not None else None})
    matched_ai = set(match.values())
    unmatched_ai = []
    for j, a in enumerate(ai):
        if j in matched_ai:
            continue
        best = min(((detail[(i, j)], cad_bends[i]["cad_id"]) for i in range(n)), key=lambda x: x[0][1] + 10 * x[0][0] / 12, default=None)
        unmatched_ai.append({"bend_id": a["bend_id"], "angle_deg": round(a["angle_deg"], 1), "length_mm": round(a["length_mm"], 1),
                             "nearest_cad": best[1] if best else None, "nearest_ang_perp_ov": [round(x, 2) for x in best[0]] if best else None})
    res = {"registration": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in reg.items() if k not in ("R", "t")},
           "R": Rm.tolist(), "t": tm.tolist(), "cad_bends": rows, "unmatched_ai_lines": unmatched_ai,
           "n_cad": n, "n_ai": m, "n_matched": len(match)}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("registration", "n_cad", "n_ai", "n_matched")}))
    if png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sub = scan[np.random.default_rng(2).choice(len(scan), min(40000, len(scan)), replace=False)]
        c = sub.mean(0); _, V = np.linalg.eigh(np.cov((sub - c).T))
        projs = [(V[:, 2], V[:, 1], "PC1-PC2"), (V[:, 2], V[:, 0], "PC1-PC3"), (V[:, 1], V[:, 0], "PC2-PC3")]
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        for ax, (u, v, nm) in zip(axs, projs):
            ax.scatter((sub - c) @ u, (sub - c) @ v, s=0.3, c="0.6")
            for k, cb in enumerate(cad_bends):
                p = np.array([cb["s_scan"], cb["e_scan"]]) - c
                ax.plot(p @ u, p @ v, color="red" if rows[k]["matched_ai_bend"] is None else "green", lw=3)
                mm = p.mean(0); ax.text(mm @ u, mm @ v, f"{cb['cad_id']} {cb['angle_deg']:.0f}/{cb['span_mm']:.0f} sup{cb['scan_support']:.2f}", fontsize=9, color="darkred")
            for a in ai:
                p = np.array([a["s"], a["e"]]) - c
                ax.plot(p @ u, p @ v, color="blue", lw=1.2, ls="--")
                mm = p.mean(0); ax.text(mm @ u, mm @ v, a["bend_id"], fontsize=9, color="blue")
            ax.set_aspect("equal"); ax.set_title(f"{nm}: grey=scan, green=CAD bend matched, red=CAD bend unmatched, blue dashed=AI line")
        fig.tight_layout(); fig.savefig(png, dpi=60)


if __name__ == "__main__":
    main()
