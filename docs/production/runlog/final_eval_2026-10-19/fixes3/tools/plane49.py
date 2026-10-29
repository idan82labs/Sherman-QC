"""DIAGNOSTIC (consumed FRESH 49207000): independent per-flange OML-skin plane fits of the scan around every CAD bend,
under the E3 pose and the independent ICP pose. Angle = nominal + change of the plane dihedral (CAD planes as reference);
fold = intersection of the two fitted OML planes; flange length = sharp-to-sharp distance or sharp-to-far-edge extent.
The CAD panel planes give the same quantities for the nominal (self-check)."""
import json, sys, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.eval.line_metric import match_lines
SP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
ED = SP / "fixes3/posthoc3/fr_6fa053514df8/engine"; CD = SP / "fixes/compile_after1/DEV-7052AE8973/compiled"
res = json.loads((ED / "result.json").read_text()); cp = json.loads((CD / "compiled_part.json").read_text())
raw = read_scan(str(REPO / "data/bend_corpus/incoming_codex_fetch_2026-10-17/20260901_e8bfd4f6/49207000_B.ply")).points
T = np.asarray(res["inspection_validity"]["registration"]["raw_to_frame"], float); Ti = np.linalg.inv(T)
z = np.load(SP / "fixes2/icp49207.npz"); R, t = z["R"], z["t"]
POSES = {"E3": (raw @ T[:3, :3].T + T[:3, 3], lambda P: P @ Ti[:3, :3].T + Ti[:3, 3]),
         "ICP": ((raw - t) @ R, lambda P: P @ R.T + t)}
cases = {c["case"]: c for c in json.loads(Path(sys.argv[1]).read_text())}
gold = cases["fr_6fa053514df8"]["gold"]
panels = {p["panel_id"]: p for p in cp["panels"]}
bends = {b["bend_uid"]: b for b in cp["bends"]}
eng = {b["bend_uid"]: {(c["kind"], c.get("side")): c for c in b["characteristics"]} for b in res["bends"]}

def line(b, k):
    L = b["loci"][k]; return np.asarray(L["start_mm"], float), np.asarray(L["end_mm"], float)

def oml_plane(b, pid):
    """(normal, point) of the panel's OML skin plane at this bend, normal pointing away from the material."""
    a, e = line(b, "oml_virtual_sharp"); m = (a + e) / 2
    pl = []
    for k in ("skin0_plane", "skin1_plane"):
        n = np.asarray(panels[pid][k]["normal"], float); q = np.asarray(panels[pid][k]["point_mm"], float)
        pl.append((abs((m - q) @ n), n, q))
    pl.sort(key=lambda x: x[0]); (_, n, q), (_, n2, q2) = pl
    if (q2 - q) @ n > 0: n = -n  # the other skin lies on the inner side: n points outward
    return n, q

def skin_peak(Q, n, q, sel, t=3.0):
    """The scan's OML skin near the CAD plane: the outermost of the skin peaks of the signed-distance histogram
    within +-6 mm (a pose off by about t must not swap the skins)."""
    g = Q[sel(Q)]; h = (g - q) @ n; h = h[np.abs(h) < 6.0]
    if len(h) < 200: return None
    hist, edges = np.histogram(h, bins=np.arange(-6.0, 6.01, 0.25)); c = (edges[:-1] + edges[1:]) / 2
    pk = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0.15 * hist.max()]
    if not pk: return None
    # skin pairs: two peaks about t apart -> the outer one; else the strongest peak
    pairs = [(i, j) for i in pk for j in pk if 0.6 * t < c[j] - c[i] < 1.4 * t]
    if pairs:
        i, j = max(pairs, key=lambda ij: hist[ij[0]] + hist[ij[1]]); return q + n * c[j]
    return q + n * c[max(pk, key=lambda i: hist[i])]

def fit_plane(P, n0, q0, sel, tols=(2.0, 1.2, 0.8)):  # noqa
    n, q = n0, q0
    for tol in tols:
        g = P[sel(P) & (np.abs((P - q) @ n) < tol)]
        if len(g) < 200: return None
        m = g.mean(0); u, s, vt = np.linalg.svd(g - m, full_matrices=False); nn = vt[2]
        n, q = (nn if nn @ n0 > 0 else -nn), m
    return n, q, len(g), float(np.median(np.abs((g - q) @ n)))

def plane_line(n1, q1, n2, q2, a):
    d = np.cross(n1, n2); d /= np.linalg.norm(d)
    A = np.stack([n1, n2, d]); b = np.array([n1 @ q1, n2 @ q2, d @ a])
    return np.linalg.solve(A, b), d

out = {}
for k, (P, to_raw) in POSES.items():
    tree = cKDTree(P)
    for uid, b in bends.items():
        a, e = line(b, "oml_virtual_sharp"); dl = (e - a) / np.linalg.norm(e - a); span = float(np.linalg.norm(e - a)); mid = (a + e) / 2
        idx = np.asarray(tree.query_ball_point(mid, span / 2 + 80), int); Q = P[idx]
        fits = {}
        for s in b["sides"]:
            n0, q0 = oml_plane(b, s["panel_id"]); w = np.asarray(s["outward_dir"], float)
            nom = (eng[uid].get(("FLANGE_LENGTH", s["side"])) or {}).get("expected") or 40.0
            wmax = min(60.0, float(nom) - 4.0)
            def sel(X, w=w, wmax=wmax):
                r = X - a; u = r @ dl; ww = r @ w
                return (u > 0.15 * span) & (u < 0.85 * span) & (ww > 6.0) & (ww < max(10.0, wmax))
            qs = skin_peak(Q, n0, q0, sel)
            f = fit_plane(Q, n0, qs, sel, tols=(1.0, 0.8, 0.6)) if qs is not None else None
            fits[s["side"]] = (f, n0, q0, w, s)
        rec = out.setdefault(b["label"], {"nominal_included": round(b["nominal_included_deg"], 2)})
        if any(f is None for f, *_ in fits.values()):
            rec[k] = None; continue
        (fA, nA0, qA0, wA, sA), (fB, nB0, qB0, wB, sB) = fits["A"], fits["B"]
        th0 = math.degrees(math.acos(np.clip(nA0 @ nB0, -1, 1))); th = math.degrees(math.acos(np.clip(fA[0] @ fB[0], -1, 1)))
        sign = 1.0 if abs((180 - th0) - b["nominal_included_deg"]) < abs(th0 - b["nominal_included_deg"]) else -1.0
        inc = b["nominal_included_deg"] - sign * (th - th0)
        p, d = plane_line(fA[0], fA[1], fB[0], fB[1], mid)
        if d @ dl < 0: d = -d
        off = (mid - p) - ((mid - p) @ d) * d
        # scan outer-apex line: the CAD apex-minus-sharp offset carried onto the fitted sharp, then to raw
        ap0, ap1 = line(b, "outer_apex"); dap = (ap0 + ap1) / 2 - mid
        s0 = p + d * ((a - p) @ d) + dap; s1 = p + d * ((e - p) @ d) + dap
        rec[k] = {"included": round(inc, 2), "fold_offset_from_cad_mm": round(float(np.linalg.norm(off)), 2),
                  "plane_rms_mm": [round(fA[3], 3), round(fB[3], 3)], "points": [fA[2], fB[2]],
                  "_sharp": (p.tolist(), d.tolist()), "_apex_raw": [to_raw(s0[None])[0].tolist(), to_raw(s1[None])[0].tolist()],
                  "_planes": {"A": (fA[0].tolist(), fA[1].tolist()), "B": (fB[0].tolist(), fB[1].tolist())}}
    # flange lengths
    for uid, b in bends.items():
        rec = out[b["label"]].get(k)
        if rec is None: continue
        p, d = map(np.asarray, rec["_sharp"])
        for s in b["sides"]:
            c = eng[uid].get(("FLANGE_LENGTH", s["side"])) or {}
            key = f"flange_{s['side']}"
            if s["boundary_kind"] == "ADJACENT_BEND":
                ob = bends[s["adjacent_bend_uid"]]; orec = out[ob["label"]].get(k)
                if orec is None: continue
                p2, d2 = map(np.asarray, orec["_sharp"])
                n, q = map(np.asarray, rec["_planes"][s["side"]]); w0 = np.asarray(s["outward_dir"], float)
                ca, ce = line(b, "oml_virtual_sharp"); oa, oe = line(ob, "oml_virtual_sharp")
                dc = (ce - ca) / np.linalg.norm(ce - ca)
                ncad = oml_plane(b, s["panel_id"])[0]
                def station_lengths(p, d, p2, d2, n, w0, a_, e_):
                    w = w0 - (w0 @ n) * n - ((w0 @ d) * d); w /= np.linalg.norm(w)
                    vals = []
                    for f in np.linspace(0.1, 0.9, 17):
                        x = p + d * (((a_ - p) @ d) + f * ((e_ - a_) @ d))
                        # intersect ray x + s w with the other sharp line (least squares in 3D)
                        A = np.stack([w, -d2], 1); sol, *_ = np.linalg.lstsq(A, p2 - x, rcond=None)
                        if sol[0] > 0: vals.append(float(sol[0]))
                    return vals
                vs = station_lengths(p, d, p2, d2, n, w0, ca, ce)
                vc = station_lengths((ca + ce) / 2, dc, (oa + oe) / 2, (oe - oa) / np.linalg.norm(oe - oa), ncad, w0, ca, ce)
                rec[key] = {"scan_median_mm": round(float(np.median(vs)), 2) if vs else None,
                            "cad_median_mm": round(float(np.median(vc)), 2) if vc else None,
                            "scan_minus_cad_mm": round(float(np.median(vs) - np.median(vc)), 2) if vs and vc else None,
                            "engine": [c.get("expected"), c.get("actual"), c.get("decision")]}
            else:
                n, q = map(np.asarray, rec["_planes"][s["side"]]); w0 = np.asarray(s["outward_dir"], float)
                w = w0 - (w0 @ n) * n; w /= np.linalg.norm(w)
                P_ = POSES[k][0]; tree = cKDTree(P_)
                idx = np.asarray(tree.query_ball_point(p + d * 0.0, 1.0), int)
                ca, ce = line(b, "oml_virtual_sharp"); span = float(np.linalg.norm(ce - ca))
                cand = P_[np.asarray(tree.query_ball_point((ca + ce) / 2, span / 2 + 120), int)]
                r = cand - p; u = r @ d; ww = r @ w; h = r @ n
                keep = (np.abs(h) < 1.0) & (ww > 3) & (ww < float(c.get("expected") or 40) + 25) & (u > (ca - p) @ d + 0.1 * span) & (u < (ce - p) @ d - 0.1 * span)
                # far edge per 5 mm lateral bin (98th percentile), median over bins
                bins = np.floor(u[keep] / 5.0); vals = []
                for bb in np.unique(bins):
                    m_ = bins == bb
                    if m_.sum() > 20: vals.append(np.percentile(ww[keep][m_], 98))
                above = (h > 1.5) & (h < 25) & (ww > 3) & (ww < float(c.get("expected") or 40) + 25) & (u > (ca - p) @ d) & (u < (ce - p) @ d)
                rec[key] = {"engine": [c.get("expected"), c.get("actual"), c.get("decision")], "scan_mm": round(float(np.median(vals)), 2) if vals else None, "far_edge_bins": len(vals),
                            "far_edge_spread_mm": round(float(np.percentile(vals, 90) - np.percentile(vals, 10)), 2) if vals else None,
                            "off_plane_points_outer_side": int(above.sum())}
# gold comparison of the scan apex lines (frozen matcher) for each pose
for k in POSES:
    preds = [{"bend_id": lb, "line_start": r[k]["_apex_raw"][0], "line_end": r[k]["_apex_raw"][1]} for lb, r in out.items() if r.get(k)]
    m = match_lines(gold, preds)
    out[f"_gold_{k}"] = {"tp": m["true_positive"], "fp": m["false_positive"], "fn": m["false_negative"],
                         "pairs": sorted((x["prediction_id"], x["gold_id"], round(x.get("transverse_offset_mm", -1), 2), round(x.get("axis_angle_deg", -1), 2)) for x in m["matches"])}
def seg_metrics(a0, a1, g0, g1):
    da = (a1 - a0) / np.linalg.norm(a1 - a0); dg = (g1 - g0) / np.linalg.norm(g1 - g0)
    ang = math.degrees(math.acos(min(1, abs(da @ dg)))); mid = (a0 + a1) / 2; r = mid - g0
    off = float(np.linalg.norm(r - (r @ dg) * dg)); ta = sorted([(a0 - g0) @ dg, (a1 - g0) @ dg]); L = float(np.linalg.norm(g1 - g0))
    ov = max(0.0, min(ta[1], L) - max(ta[0], 0.0))
    return round(ang, 2), round(off, 2), round(ov / L, 2), round(ov / (ta[1] - ta[0]), 2), round(L, 1), round(ta[1] - ta[0], 1)
for k in POSES:
    r2 = out["B2"].get(k)
    if r2:
        a0, a1 = map(np.asarray, r2["_apex_raw"])
        print("B2 apex (", k, ") vs gold [angle, offset, gold overlap, pred overlap, gold len, pred len]:",
              sorted(((seg_metrics(a0, a1, np.asarray(g["line_start"]), np.asarray(g["line_end"])), g["bend_id"]) for g in gold), key=lambda x: x[0][1])[:4])
for lb, r in out.items():
    if lb.startswith("_"): print(lb, r); continue
    print(lb, "nominal", r["nominal_included"])
    for k in POSES:
        if r.get(k) is None: print("  ", k, None); continue
        x = r[k]; print("  ", k, {kk: v for kk, v in x.items() if not kk.startswith("_")})
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
