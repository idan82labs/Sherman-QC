#!/usr/bin/env python3
"""Gold11 audit renders: gold-overlay 4 views + per-gold-line cross-sections (read-only).

For every Gold11 scan (case_list_broad_v3.json tier GOLD11) this loads the raw scan vertices,
loads the frozen gold through research/broad_baseline/score.load_gold with the recovered frame
records (gold11_frame_recovery.json; all 11 are identity / residual 0), and writes into
gold11_audit/renders/<scan_id>/:

  overview_4views.png        whole part, 4 views, gold lines numbered (best_f1 lines in red)
  xsec_<bend_id>.png         cross-sections perpendicular to the gold line at 25/50/75 % of span:
                             top row wide context (+-40 mm), bottom row close-up (+-12 mm)
                             around the fitted corner, with the gold point, its 4 mm circle,
                             the fitted flange skins, outer/inner virtual sharp, outer apex,
                             fitted arc centre (bend axis), and any best_f1/a0 line crossing
                             the section; strip at the bottom = nearest-scan distance along
                             the whole gold line (extent check).
  measurements.json          per line per station: distances of the gold point to the outer
                             apex, outer virtual sharp, inner virtual sharp, mid-surface sharp,
                             arc centre and nearest scan point, signed offsets in the bisector
                             frame, fitted bend angle, radii, skins seen; provisional locus tag.

The geometric fit is a lightweight 2D heuristic (local-PCA tangents, two dominant directions,
skin offsets, circle fit). It is an aid for a human / second-agent verdict, not a verdict.
Nothing outside gold11_audit/ is written. Gold files are only read.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
BB = ROOT / "research" / "broad_baseline"
sys.path.insert(0, str(BB))
sys.path.insert(0, str(ROOT))
import bb_common as bb  # noqa: E402
import score as sc  # noqa: E402

RUNLOG = ROOT / "docs/production/runlog/broad_baseline"
CASE_LIST = RUNLOG / "case_list_broad_v3.json"
FRAMES = RUNLOG / "gold11_frame_recovery.json"
REPRO_RUNS = RUNLOG / "gold11_repro/runs"
OUT_ROOT = RUNLOG / "gold11_audit/renders"
STATIONS = (0.25, 0.5, 0.75)
GOLD_C = "#1a9c3a"
PRED_C = {"best_f1": "#d62728", "a0": "#ff7f0e"}


# ------------------------------------------------------------------ helpers

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, a))
    w = np.cross(d, u)
    return u, w


def line_intersect(p1, t1, p2, t2):
    """Intersection of 2D lines p1 + s t1 and p2 + r t2 (None if near parallel)."""
    m = np.array([t1, -t2]).T
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    s, _r = np.linalg.solve(m, p2 - p1)
    return p1 + s * t1


def load_preds(case_id):
    out = {}
    for lane, fname in (("best_f1", "predictions_best_f1.json"), ("a0", "predictions_a0.json")):
        p = REPRO_RUNS / case_id / "best_f1" / fname
        if not p.is_file():
            continue
        rows = bb.read_json(p).get("owned_bend_regions") or []
        lines = []
        for r in rows:
            try:
                e = bb.endpoints(r)
            except Exception:  # noqa: BLE001
                continue
            if e is None or len(e) != 2:
                continue
            lines.append((str(r.get("bend_id")), np.asarray(e[0], float), np.asarray(e[1], float)))
        out[lane] = lines
    # a0 lines also appear inside best_f1 (fusion); keep a0 only for ids not already in best_f1
    if "a0" in out and "best_f1" in out:
        ids = {i for i, _, _ in out["best_f1"]}
        out["a0"] = [x for x in out["a0"] if x[0] not in ids]
    return out


# ------------------------------------------------------------------ section fit

def tangents_2d(q, radius):
    from scipy.spatial import cKDTree

    tree = cKDTree(q)
    nb = tree.query_ball_point(q, radius)
    ang = np.full(len(q), np.nan)
    lin = np.zeros(len(q))
    for i, idx in enumerate(nb):
        if len(idx) < 5:
            continue
        x = q[idx] - q[idx].mean(0)
        c = x.T @ x / len(idx)
        w, v = np.linalg.eigh(c)
        if w[1] <= 0:
            continue
        lin[i] = 1 - w[0] / w[1]
        t = v[:, 1]
        ang[i] = math.degrees(math.atan2(t[1], t[0])) % 180.0
    return ang, lin


def circ_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def find_peaks(ang, weights, min_sep=10.0):
    """Two dominant tangent directions: the second must be a separate local maximum with a
    valley (< 50 % of the smaller peak) between it and the first, so one broad peak is not split."""
    good = np.isfinite(ang)
    h, edges = np.histogram(ang[good], bins=90, range=(0, 180), weights=weights[good])
    k = np.array([1, 2, 3, 2, 1], float)
    hs = np.convolve(np.r_[h[-2:], h, h[:2]], k / k.sum(), mode="valid")
    centers = (edges[:-1] + edges[1:]) / 2
    n = len(hs)
    i1 = int(np.argmax(hs))
    p1 = centers[i1]
    loc = [j for j in range(n) if hs[j] > 0 and hs[j] >= hs[(j - 1) % n] and hs[j] >= hs[(j + 1) % n]]
    loc.sort(key=lambda j: -hs[j])
    for j in loc:
        if j == i1 or circ_diff(centers[j], p1) < min_sep or hs[j] < 0.08 * hs[i1]:
            continue
        # valley on the shorter circular path between i1 and j
        fwd = [(i1 + s) % n for s in range((j - i1) % n + 1)]
        bwd = [(j + s) % n for s in range((i1 - j) % n + 1)]
        path = fwd if len(fwd) <= len(bwd) else bwd
        if min(hs[x] for x in path) < 0.5 * hs[j]:
            return p1, centers[j]
    return p1, None


def refine_angle(ang, sel):
    a = np.radians(ang[sel] * 2)
    return (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180.0


def skins_for_class(q, sel, theta, gap):
    """Split one direction class into parallel skin lines by offset along the normal."""
    t = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
    n = np.array([-t[1], t[0]])
    pts = q[sel]
    off = pts @ n
    along = pts @ t
    order = np.argsort(off)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if off[b] - off[a] > gap:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    skins = []
    for g in groups:
        g = np.array(g)
        if len(g) < 8:
            continue
        al = along[g]
        lo, hi = np.percentile(al, [2, 98])
        if hi - lo < 2.0:
            continue
        # split along-direction gaps (> 3 mm) into separate segments
        o = np.argsort(al)
        segs, cs = [], [o[0]]
        for a, b in zip(o[:-1], o[1:]):
            if al[b] - al[a] > 3.0:
                segs.append(cs)
                cs = []
            cs.append(b)
        segs.append(cs)
        for s in segs:
            s = g[np.array(s)]
            if len(s) < 8:
                continue
            a2 = along[s]
            lo, hi = a2.min(), a2.max()
            if hi - lo < 2.0:
                continue
            skins.append({"theta": theta, "t": t, "n": n, "off": float(np.median(off[s])),
                          "lo": float(lo), "hi": float(hi), "count": int(len(s)), "idx": sel.nonzero()[0][s]})
    return skins


def fit_section(q, spacing):
    """Return dict with the corner model in 2D section coords (gold point at origin)."""
    res = {"ok": False}
    if len(q) < 40:
        res["why"] = "too_few_points"
        return res
    radius = float(np.clip(1.6 * spacing, 0.7, 0.9))  # below sheet thickness so the two skins stay separate
    ang, lin = tangents_2d(q, radius)
    w = np.where(lin > 0.85, 1.0, 0.0)
    if w.sum() < 20:
        res["why"] = "no_linear_points"
        return res
    p1, p2 = find_peaks(ang, w)
    if p2 is None:
        res["why"] = "single_direction"
        res["theta1"] = float(p1)
        return res
    gap = max(0.45, 1.2 * spacing)
    cls = []
    for p in (p1, p2):
        sel = (w > 0) & np.array([circ_diff(a, p) <= 5.0 if np.isfinite(a) else False for a in ang])
        th = refine_angle(ang, sel)
        sel = (w > 0) & np.array([circ_diff(a, th) <= 5.0 if np.isfinite(a) else False for a in ang])
        cls.append((th, sel, skins_for_class(q, sel, th, gap)))
    (tha, sela, sa), (thb, selb, sbk) = cls
    if not sa or not sbk:
        res["why"] = "no_skins"
        return res
    best = None
    for A in sa:
        for B in sbk:
            pa = A["off"] * A["n"]
            pb = B["off"] * B["n"]
            X = line_intersect(pa, A["t"], pb, B["t"])
            if X is None:
                continue
            # both skins must approach X (gap from X to the skin's nearest end <= 10 mm),
            # and lie on one side of X
            xa = X @ A["t"]
            xb = X @ B["t"]
            ga = 0 if A["lo"] <= xa <= A["hi"] else min(abs(A["lo"] - xa), abs(A["hi"] - xa))
            gb = 0 if B["lo"] <= xb <= B["hi"] else min(abs(B["lo"] - xb), abs(B["hi"] - xb))
            if ga > 10 or gb > 10:
                continue
            # a skin straddling X by more than 3 mm on both sides is not a flange ending in this corner
            if (A["lo"] < xa - 6 and A["hi"] > xa + 6) or (B["lo"] < xb - 6 and B["hi"] > xb + 6):
                continue
            score = np.linalg.norm(X) + 0.3 * (ga + gb) - 0.002 * (A["count"] + B["count"])
            if best is None or score < best[0]:
                best = (score, A, B, X)
    if best is None:
        res["why"] = "no_corner_pair"
        return res
    _, A, B, X = best

    def material_dir(S, X):
        mid = 0.5 * (S["lo"] + S["hi"])
        s = 1.0 if mid > X @ S["t"] else -1.0
        return s * S["t"]

    a = material_dir(A, X)
    b = material_dir(B, X)

    def sheet(S, all_skins, other_dir):
        # parallel skins of the same sheet: offset within 0.4..4.5 mm and overlapping extent
        mates = [S]
        for T in all_skins:
            if T is S:
                continue
            do = abs(T["off"] - S["off"])
            ov = min(S["hi"], T["hi"]) - max(S["lo"], T["lo"])
            if 0.4 <= do <= 4.5 and ov > -6:
                mates.append(T)
        n = S["n"] if S["n"] @ other_dir > 0 else -S["n"]  # n points toward the other flange (inside)
        offs = [(T["off"] * (S["n"] @ n), T) for T in mates]
        # keep only the closest mate on each side of S, at most two skins
        outer = min(offs, key=lambda x: x[0])
        inner = max(offs, key=lambda x: x[0])
        return n, outer, inner, len(mates)

    na, oa, ia, ka = sheet(A, sa, b)
    nb_, ob, ib, kb = sheet(B, sbk, a)

    def skin_line(n_in, offv, t):
        return offv * n_in, t

    lo_a = skin_line(na, oa[0], A["t"])
    lo_b = skin_line(nb_, ob[0], B["t"])
    li_a = skin_line(na, ia[0], A["t"])
    li_b = skin_line(nb_, ib[0], B["t"])
    Xo = line_intersect(lo_a[0], lo_a[1], lo_b[0], lo_b[1])
    Xi = line_intersect(li_a[0], li_a[1], li_b[0], li_b[1])
    two_skins = ka >= 2 and kb >= 2 and abs(ia[0] - oa[0]) > 0.3 and abs(ib[0] - ob[0]) > 0.3
    thick_a = abs(ia[0] - oa[0]) if ka >= 2 else None
    thick_b = abs(ib[0] - ob[0]) if kb >= 2 else None
    Xm = line_intersect(0.5 * (lo_a[0] + li_a[0]), A["t"], 0.5 * (lo_b[0] + li_b[0]), B["t"])
    g = unit(a + b)  # bisector toward the inside (concave side)
    if np.linalg.norm(a + b) < 1e-6:
        res["why"] = "antiparallel"
        return res
    included = math.degrees(math.acos(np.clip(a @ b, -1, 1)))
    bend_angle = 180.0 - included
    # outer apex = outer-skin scan point on the bisector nearest the outer virtual sharp
    # (for a circular arc tangent to both outer skins this is the arc midpoint);
    # arc centre (bend axis) then follows from the tangent-arc geometry:
    #   apex distance delta = R/sin(alpha/2) - R  ->  R = delta / (1/sin(alpha/2) - 1)
    ref = Xo if Xo is not None else X
    apex = C = Ro = Ri = None
    rel = q - ref
    along = rel @ g
    perp = np.abs(rel @ np.array([-g[1], g[0]]))
    cand = (perp < max(0.5, 1.2 * spacing)) & (along > -1.0) & (np.linalg.norm(rel, axis=1) < 15)
    if cand.sum() >= 1:
        vals = np.sort(along[cand])
        delta = float(vals[min(2, len(vals) - 1)]) if len(vals) >= 3 else float(vals[0])
        delta = max(delta, 0.0)
        apex = ref + g * delta
        half = math.radians(included) / 2
        if 1e-3 < half < math.radians(88):
            k = 1 / math.sin(half) - 1
            Ro = delta / k if k > 1e-6 else None
            if Ro is not None and Ro < 40:
                C = ref + g * (Ro / math.sin(half))
                if two_skins:
                    th = np.nanmean([v for v in (thick_a, thick_b) if v is not None])
                    Ri = max(Ro - th, 0.0)
            else:
                Ro = None
    arc_sel = np.zeros(len(q), bool)
    tangents = {}
    half_inc = math.radians(included) / 2
    if Ro is not None and math.tan(half_inc) > 1e-6:
        if Xo is not None:
            tangents["outer_A"] = Xo + a * Ro / math.tan(half_inc)
            tangents["outer_B"] = Xo + b * Ro / math.tan(half_inc)
        if Ri is not None and Xi is not None and two_skins:
            tangents["inner_A"] = Xi + a * Ri / math.tan(half_inc)
            tangents["inner_B"] = Xi + b * Ri / math.tan(half_inc)
    res.update({"ok": True, "X_pair": X, "Xo": Xo, "Xi": Xi if two_skins else None, "Xm": Xm if two_skins else None,
                "a": a, "b": b, "g": g, "bend_angle_deg": bend_angle, "C": C, "R_outer": Ro, "R_inner": Ri,
                "apex": apex, "tangents": tangents, "two_skins": bool(two_skins), "thick_a": thick_a, "thick_b": thick_b,
                "skins": {"A_outer": lo_a, "A_inner": li_a, "B_outer": lo_b, "B_inner": li_b},
                "A": A, "B": B, "arc_idx": np.nonzero(arc_sel)[0]})
    return res


# ------------------------------------------------------------------ main per scan

def provisional_tag(stations):
    """Provisional, geometry-only locus tag (an aid for the verdict, not the verdict)."""
    ok = [s for s in stations if s.get("fit_ok") and s.get("d_apex") is not None]
    if len(ok) < 2:
        return "NO_CORNER_FOUND_NEAR_GOLD"

    def med(k):
        v = [s[k] for s in ok if s.get(k) is not None]
        return float(np.median(v)) if v else None

    if min(s.get("d_corner", 0) for s in ok) > 12:
        return "FAR_FROM_ANY_CORNER"
    da = med("d_apex")
    if da <= 1.5:
        return "ON_APEX"
    along = [s["gold_minus_apex_along_bisector_mm"] for s in ok if s.get("gold_minus_apex_along_bisector_mm") is not None]
    across = [s["gold_minus_apex_across_mm"] for s in ok if s.get("gold_minus_apex_across_mm") is not None]
    spread = float(max(s["d_apex"] for s in ok) - min(s["d_apex"] for s in ok))
    b = float(np.median(along)) if along else 0.0
    a = float(np.median(across)) if across else 0.0
    side = "CONCAVE" if b > 0 else "CONVEX"
    if spread <= 1.5 and abs(a) <= max(1.5, 0.5 * abs(b)):
        tag = f"CONSISTENT_OFFSET_{side}_{abs(b):.1f}mm"
    elif spread <= 1.5:
        tag = f"CONSISTENT_OFFSET_SIDEWAYS_{da:.1f}mm"
    else:
        tag = f"SLANTED_OR_INCONSISTENT_{min(s['d_apex'] for s in ok):.1f}-{max(s['d_apex'] for s in ok):.1f}mm"
    near = {"outer_sharp": med("d_outer_sharp"), "inner_sharp": med("d_inner_sharp"), "arc_centre": med("d_arc_centre"),
            "tangent_line": med("d_nearest_tangent_line")}
    near = {k: v for k, v in near.items() if v is not None and v <= 1.2}
    if near:
        tag += "_NEAR_" + min(near, key=near.get).upper()
    if med("d_nearest_scan") is not None and med("d_nearest_scan") <= 0.8:
        tag += "_ON_SURFACE"
    return tag


def render_scan(entry, frames, out_root, max_overview=150000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    case_id = entry["case_id"]
    out = out_root / case_id
    out.mkdir(parents=True, exist_ok=True)
    pts = bb.load_scan_vertices(Path(entry["scan_path"]))
    tree = cKDTree(pts)
    nn_d, _ = tree.query(pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 5000), replace=False)], k=2)
    spacing = float(np.median(nn_d[:, 1]))
    gold = sc.load_gold(entry["gold"], frame_records=frames)
    glines = [(g["bend_id"], np.asarray(g["line_start"], float), np.asarray(g["line_end"], float)) for g in gold["lines"]]
    preds = load_preds(case_id)

    # ---------------- overview
    rng = np.random.default_rng(12345)
    sub = pts[rng.choice(len(pts), min(len(pts), max_overview), replace=False)]
    center = sub.mean(0)
    _u, _s, vt = np.linalg.svd(sub[rng.choice(len(sub), min(len(sub), 20000), replace=False)] - center, full_matrices=False)
    e1, e2, e3 = vt
    views = [("top (PC1-PC2)", e1, e2), ("front (PC1-PC3)", e1, e3), ("side (PC2-PC3)", e2, e3),
             ("isometric", (e1 - e2) / math.sqrt(2), (e1 + e2 - 2 * e3) / math.sqrt(6))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=95)
    for ax, (name, vx, vy) in zip(axes.ravel(), views):
        rel = sub - center
        x, y, dep = rel @ vx, rel @ vy, rel @ np.cross(vx, vy)
        o = np.argsort(dep)
        shade = (dep[o] - dep.min()) / max(1e-9, np.ptp(dep))
        ax.scatter(x[o], y[o], s=0.25, c=0.5 + 0.4 * shade, cmap="Greys_r", vmin=0, vmax=1, linewidths=0, rasterized=True)
        for lane, lines in preds.items():
            for pid, s0, s1 in lines:
                e = np.array([s0, s1]) - center
                ax.plot(e @ vx, e @ vy, color=PRED_C[lane], lw=1.1, ls="--", zorder=4)
        for i, (bid, s0, s1) in enumerate(glines):
            e = np.array([s0, s1]) - center
            ax.plot(e @ vx, e @ vy, color=GOLD_C, lw=2.6, zorder=5, solid_capstyle="round")
            m = 0.5 * (e[0] + e[1])
            ax.text(m @ vx, m @ vy, bid, fontsize=8, color="white", zorder=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=GOLD_C, ec="none"))
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    leg = [plt.Line2D([], [], color=GOLD_C, lw=2.6, label=f"gold ({len(glines)}) {gold.get('truth_tier')}")]
    for lane in preds:
        leg.append(plt.Line2D([], [], color=PRED_C[lane], lw=1.1, ls="--", label=f"{lane} lines ({len(preds[lane])})"))
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{case_id}  {entry.get('gold11_id')}  family {entry['family_id']}  {Path(entry['scan_path']).name}  "
                 f"gold {gold.get('catalog_case_id')} (frame {gold.get('label_frame')} -> scan_raw, "
                 f"{gold.get('frame_transform', {}).get('method')})", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "overview_4views.png")
    plt.close(fig)

    # ---------------- per line sections
    meas = {"scan_id": case_id, "gold11_id": entry.get("gold11_id"), "scan_path": entry["scan_path"],
            "n_points": int(len(pts)), "median_spacing_mm": spacing, "gold_source": gold.get("gold_source"),
            "catalog_case_id": gold.get("catalog_case_id"), "truth_tier": gold.get("truth_tier"),
            "frame_transform": gold.get("frame_transform"), "lines": []}
    for bid, s0, s1 in glines:
        d = unit(s1 - s0)
        L = float(np.linalg.norm(s1 - s0))
        u0, w0 = perp_basis(d)
        fig = plt.figure(figsize=(17, 12.5), dpi=90)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.15, 0.38])
        stations = []
        for k, t in enumerate(STATIONS):
            c = s0 + t * (s1 - s0)
            idx = np.array(tree.query_ball_point(c, 42.0), int)
            P = pts[idx] if len(idx) else np.zeros((0, 3))
            along = (P - c) @ d if len(P) else np.zeros(0)
            h_used = None
            for h in (1.0, 1.5, 2.0, 3.0, 4.5):
                m = np.abs(along) < h
                q_all = np.c_[(P[m] - c) @ u0, (P[m] - c) @ w0]
                if (np.linalg.norm(q_all, axis=1) < 20).sum() >= 500 or h == 4.5:
                    h_used = h
                    break
            q_fit = q_all[np.linalg.norm(q_all, axis=1) < 22] if len(q_all) else q_all
            fit = fit_section(q_fit, max(spacing, h_used * 0.0 + spacing))
            # rotate display so the bisector g points up (+y): concave side up, outer apex below
            R = np.eye(2)
            if fit.get("ok"):
                g = fit["g"]
                ang = math.atan2(g[0], g[1])
                R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            rot = lambda v: None if v is None else (R @ np.asarray(v, float))  # noqa: E731
            qa = (R @ q_all.T).T if len(q_all) else q_all
            st = {"t": t, "slab_half_mm": h_used, "points_in_slab": int(len(q_all)), "fit_ok": bool(fit.get("ok")),
                  "fit_note": fit.get("why")}
            nd, _ = tree.query(c)
            st["d_nearest_scan"] = float(nd)
            pred_hits = []
            for lane, lines in preds.items():
                for pid, p0, p1 in lines:
                    pd = unit(p1 - p0)
                    cosang = abs(pd @ d)
                    if cosang < math.cos(math.radians(10)):
                        continue
                    den = pd @ d
                    sp = ((c - p0) @ d) / den
                    if sp < -5 or sp > np.linalg.norm(p1 - p0) + 5:
                        continue
                    x3 = p0 + sp * pd
                    v2 = np.array([(x3 - c) @ u0, (x3 - c) @ w0])
                    if np.linalg.norm(v2) > 40:
                        continue
                    pred_hits.append({"lane": lane, "id": pid, "xy": v2,
                                      "angle_deg": math.degrees(math.acos(min(1, cosang))),
                                      "d_gold": float(np.linalg.norm(v2))})
            if fit.get("ok"):
                for key, name in (("apex", "d_apex"), ("Xo", "d_outer_sharp"), ("Xi", "d_inner_sharp"),
                                  ("Xm", "d_mid_sharp"), ("C", "d_arc_centre")):
                    v = fit.get(key)
                    st[name] = None if v is None else float(np.linalg.norm(v))
                    st[key + "_xy_rot"] = None if v is None else [float(x) for x in rot(v)]
                tg = fit.get("tangents") or {}
                if tg:
                    kmin = min(tg, key=lambda kk: np.linalg.norm(tg[kk]))
                    st["d_nearest_tangent_line"] = float(np.linalg.norm(tg[kmin]))
                    st["nearest_tangent_line"] = kmin
                st["d_corner"] = float(np.linalg.norm(fit["Xo"] if fit["Xo"] is not None else fit["X_pair"]))
                st["bend_angle_deg"] = float(fit["bend_angle_deg"])
                st["R_outer"] = fit["R_outer"]
                st["R_inner"] = fit["R_inner"]
                st["two_skins"] = fit["two_skins"]
                st["thickness_mm"] = [fit["thick_a"], fit["thick_b"]]
                if fit.get("apex") is not None:
                    g = fit["g"]
                    v = -fit["apex"]  # gold relative to apex
                    st["gold_minus_apex_along_bisector_mm"] = float(v @ g)  # + = toward concave side / inside
                    st["gold_minus_apex_across_mm"] = float(v @ np.array([-g[1], g[0]]))
                for ph in pred_hits:
                    if fit.get("apex") is not None:
                        ph["d_apex"] = float(np.linalg.norm(ph["xy"] - fit["apex"]))
            st["pred_hits"] = [{k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2) for k2, v2 in ph.items()} for ph in pred_hits]
            to3 = lambda v: None if v is None else (c + v[0] * u0 + v[1] * w0).tolist()  # noqa: E731
            st["gold_xyz"] = c.tolist()
            if fit.get("ok"):
                st["apex_xyz"] = to3(fit.get("apex"))
                st["outer_sharp_xyz"] = to3(fit.get("Xo"))
                st["arc_centre_xyz"] = to3(fit.get("C"))
            stations.append(st)

            for row, half in ((0, 40.0), (1, 12.0)):
                ax = fig.add_subplot(gs[row, k])
                if row == 1 and fit.get("ok"):
                    cen = rot(fit["apex"] if fit.get("apex") is not None else fit["X_pair"])
                    cen = 0.5 * (cen + np.zeros(2)) if np.linalg.norm(cen) < 10 else cen
                else:
                    cen = np.zeros(2)
                if len(qa):
                    win = (np.abs(qa[:, 0] - cen[0]) < half) & (np.abs(qa[:, 1] - cen[1]) < half)
                    ax.scatter(qa[win, 0], qa[win, 1], s=2.5 if row else 1.2, c="#555555", linewidths=0, rasterized=True)
                ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color=GOLD_C, lw=1.2))
                if fit.get("ok"):
                    for nm, (p0, tv) in fit["skins"].items():
                        p0r, tr = rot(p0), rot(tv)
                        ss = np.array([-60, 60])
                        ax.plot(p0r[0] + ss * tr[0], p0r[1] + ss * tr[1], color="#1f77b4" if "outer" in nm else "#9467bd",
                                lw=0.8, ls="--", alpha=0.8)
                    if fit.get("Xo") is not None:
                        v = rot(fit["Xo"]); ax.plot(*v, "x", color="#1f77b4", ms=9, mew=2, label="outer virtual sharp")
                    if fit.get("Xi") is not None:
                        v = rot(fit["Xi"]); ax.plot(*v, "x", color="#9467bd", ms=8, mew=2, label="inner virtual sharp")
                    if fit.get("C") is not None:
                        v = rot(fit["C"]); ax.plot(*v, "+", color="#8c564b", ms=11, mew=2, label="arc centre (bend axis)")
                        for rr in (fit.get("R_outer"), fit.get("R_inner")):
                            if rr:
                                ax.add_patch(plt.Circle(tuple(v), rr, fill=False, color="#8c564b", lw=0.6, alpha=0.6))
                    for nmk, tv in (fit.get("tangents") or {}).items():
                        v = rot(tv); ax.plot(*v, "|", color="#e377c2", ms=9, mew=2,
                                             label="bend tangent lines" if nmk == "outer_A" else None)
                    if fit.get("apex") is not None:
                        v = rot(fit["apex"]); ax.plot(*v, "D", color="#17becf", ms=7, mec="k", label="outer apex")
                for ph in pred_hits:
                    v = rot(ph["xy"])
                    ax.plot(*v, "o", ms=6, mfc="none", mec=PRED_C[ph["lane"]], mew=1.6)
                    if row == 1 or ph["d_gold"] < 25:
                        ax.annotate(ph["id"][:14], v, fontsize=6, color=PRED_C[ph["lane"]], xytext=(4, -8), textcoords="offset points")
                ax.plot(0, 0, "*", color=GOLD_C, ms=15, mec="k", mew=0.6, label=f"gold {bid}")
                ax.set_xlim(cen[0] - half, cen[0] + half)
                ax.set_ylim(cen[1] - half, cen[1] + half)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)
                if row == 0:
                    ax.set_title(f"{bid} @ {int(t * 100)}%  slab +-{h_used} mm  (wide +-{int(half)} mm)", fontsize=9)
                else:
                    if fit.get("ok"):
                        txt = (f"bend {st['bend_angle_deg']:.0f} deg  2 skins={st['two_skins']}  "
                               f"R_o={st['R_outer'] if st['R_outer'] is None else round(st['R_outer'], 2)}\n"
                               f"gold->apex {st.get('d_apex') if st.get('d_apex') is None else round(st['d_apex'], 2)}  "
                               f"->outer sharp {st.get('d_outer_sharp') if st.get('d_outer_sharp') is None else round(st['d_outer_sharp'], 2)}  "
                               f"->inner sharp {st.get('d_inner_sharp') if st.get('d_inner_sharp') is None else round(st['d_inner_sharp'], 2)}\n"
                               f"->arc centre {st.get('d_arc_centre') if st.get('d_arc_centre') is None else round(st['d_arc_centre'], 2)}  "
                               f"->scan {st['d_nearest_scan']:.2f} mm  ->tangent {st.get('d_nearest_tangent_line') if st.get('d_nearest_tangent_line') is None else round(st['d_nearest_tangent_line'], 2)}")
                    else:
                        txt = f"no corner fit ({fit.get('why')}); gold->scan {st['d_nearest_scan']:.2f} mm"
                    ax.set_title(txt, fontsize=7.5)
                    if k == 0:
                        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.8)
        # extent strip
        ax = fig.add_subplot(gs[2, :])
        ss = np.linspace(0, 1, max(20, int(L / 2)))
        samples = s0[None] + ss[:, None] * (s1 - s0)[None]
        dd, _ = tree.query(samples)
        ax.plot(ss * L, dd, color=GOLD_C, lw=1.5)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.axhline(4.0, color="r", lw=0.6, ls=":")
        for t in STATIONS:
            ax.axvline(t * L, color="#999", lw=0.6)
        ax.set_xlabel(f"arc length along gold {bid} (mm), total {L:.1f} mm", fontsize=8)
        ax.set_ylabel("nearest scan pt (mm)", fontsize=8)
        ax.set_ylim(0, max(6, float(np.percentile(dd, 98)) * 1.1))
        ax.tick_params(labelsize=7)
        tag = provisional_tag(stations)
        frac1 = float((dd <= 1.0).mean())
        fig.suptitle(f"{case_id} {entry.get('gold11_id')}  gold {bid}  len {L:.1f} mm  provisional locus tag: {tag}  "
                     f"(view: bisector up = concave side up; gold at origin; dotted circle = 4 mm matcher limit)",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"xsec_{bid}.png")
        plt.close(fig)
        meas["lines"].append({"bend_id": bid, "length_mm": L, "direction": d.tolist(), "line_start": s0.tolist(), "line_end": s1.tolist(),
                              "frac_within_1mm_of_scan": frac1, "median_dist_to_scan_mm": float(np.median(dd)),
                              "provisional_locus_tag": tag, "stations": stations})
    (out / "measurements.json").write_text(json.dumps(meas, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cases = [c for c in bb.read_json(CASE_LIST)["cases"] if c.get("tier") == "GOLD11"]
    frames = bb.read_json(FRAMES)
    for c in cases:
        if args.only and c["case_id"] not in args.only:
            continue
        o = render_scan(c, frames, Path(args.out_root))
        print("wrote", o, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
