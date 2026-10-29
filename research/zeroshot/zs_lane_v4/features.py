"""zs_lane_v4 selector features (runtime side; label-free, part-agnostic: mm, degrees, multiples of t).

Per pooled candidate of a v4 record (core.process):
  * the 26 pooled features of zs_lane_v1 (21 Z3 + 5 pool flags);
  * the 6 zs_lane_v2 station-sweep summaries (sw_full, sw_bins, st_bend, st_fa, st_fb, st_both);
  * support-profile features (profiles.py) on the candidate's own extent and on the extension that its pooled Z2
    line proposes: apex-touch fraction, longest unsupported runs (touch / sweep / flank A / flank B), extension
    length and its support;
  * interruption evidence (zs_lane_v2 split runs): number of interior interruptions, of tab-type gaps (a flank
    plate absent across the run and not connected around it), total interrupted length;
  * neighbourhood: collinear neighbours (separate segments on one line, e.g. a row of tabs), near-duplicates,
    collinear pairs with both plates connected;
  * the attached Z2 line's own attributes (cover, score, span ratio);
  * scan-level metadata (thickness, mesh spacing, spacing / t, candidate / family / Z2-line counts, area, budget).
No identity, file name, path, CAD or label enters.
"""
from __future__ import annotations

import math

import numpy as np

from research.zeroshot.zs_lane_v4 import profiles as PR

SWEEP_FEATS = ["sw_full", "sw_bins", "st_bend", "st_fa", "st_fb", "st_both"]
PROF_FEATS = ["pr_touch", "pr_touch_gap", "pr_touch_gap_rel", "pr_sw_gap", "pr_fa_gap", "pr_fb_gap",
              "pr_ext_len", "pr_ext_rel", "pr_ext_touch", "pr_ext_sw", "pr_ext_fb", "pr_ext_touch_gap"]
SPLIT_FEATS = ["sp_runs", "sp_tabgaps", "sp_len_rel"]
NB_FEATS = ["nb_collinear", "nb_dups", "nb_dup_longer", "nb_pair_conn", "nb_pair_disc"]
Z2_FEATS = ["z2_cover", "z2_score", "z2_span_rel"]
SCAN_FEATS = ["sc_t", "sc_spacing", "sc_spacing_t", "sc_log_ncand", "sc_nfam", "sc_log_area", "sc_z2_lines",
              "sc_budget"]
V4_FEATS = SWEEP_FEATS + PROF_FEATS + SPLIT_FEATS + NB_FEATS + Z2_FEATS + SCAN_FEATS

DUP_ANG, DUP_OFF, DUP_OV = 4.0, 2.5, 0.5


def _seg(c: dict, key: str = "") -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(c["start" + key], float), np.asarray(c["end" + key], float)


def _geom(a0, a1, b0, b1) -> tuple[float, float, float, float]:
    """(angle deg, transverse offset, overlap fraction of the shorter, gap mm along a) of segments a, b."""
    da, db = a1 - a0, b1 - b0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    ua, ub = da / max(la, 1e-12), db / max(lb, 1e-12)
    ang = math.degrees(math.acos(min(1.0, abs(float(ua @ ub)))))
    ma, mb = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    dm = mb - ma
    off = max(float(np.linalg.norm(dm - ua * (dm @ ua))), float(np.linalg.norm(dm - ub * (dm @ ub))))
    ia = sorted([float((a0 - ma) @ ua), float((a1 - ma) @ ua)])
    ib = sorted([float((b0 - ma) @ ua), float((b1 - ma) @ ua)])
    ov = min(ia[1], ib[1]) - max(ia[0], ib[0])
    return ang, off, max(0.0, ov) / max(min(la, lb), 1e-9), max(0.0, -ov)


def _tab_gap(r: dict) -> bool:
    for f in ("a", "b"):
        if not r[f"c{f}"] and r.get(f"sides_{f}", 0) == 2 and r.get(f"raw_{f}", 1.0) <= 0.34:
            return True
    return False


def candidate_features(rec: dict) -> np.ndarray:
    """(n, len(V4_FEATS)) matrix; NaN where a feature does not apply."""
    C = rec.get("candidates") or []
    n = len(C)
    F = np.full((n, len(V4_FEATS)), np.nan)
    if not n:
        return F
    col = {k: i for i, k in enumerate(V4_FEATS)}
    ev = rec.get("evidence") or {}
    sw = ev.get("sweep")
    prof = rec.get("profiles") or {}
    splits = ev.get("splits") or {}
    z2 = rec.get("z2_lines") or []
    meta = rec.get("meta") or {}
    segs = [_seg(c) for c in C]
    L = np.array([float(np.linalg.norm(b - a)) for a, b in segs])
    # ---- sweep summaries
    if sw is not None:
        for i in range(n):
            if i < len(sw) and sw[i] is not None:
                F[i, [col[k] for k in SWEEP_FEATS]] = sw[i]
    # ---- support profiles
    for i in range(n):
        p = prof.get(str(i))
        if p is None:
            continue
        Li = p["L"]
        F[i, col["pr_touch"]] = PR.frac(p, "touch", 0.0, Li)
        g = PR.longest_gap(p, "touch", 0.0, Li)
        F[i, col["pr_touch_gap"]] = g
        F[i, col["pr_touch_gap_rel"]] = g / max(Li, 1e-9)
        F[i, col["pr_sw_gap"]] = PR.longest_gap(p, "sw", 0.0, Li)
        F[i, col["pr_fa_gap"]] = PR.longest_gap(p, "fa", 0.0, Li)
        F[i, col["pr_fb_gap"]] = PR.longest_gap(p, "fb", 0.0, Li)
        lo = min([0.0] + [e[0] for e in p.get("z2_ext") or []])
        hi = max([Li] + [e[1] for e in p.get("z2_ext") or []])
        ext = (0.0 - lo) + (hi - Li)
        F[i, col["pr_ext_len"]] = ext
        F[i, col["pr_ext_rel"]] = (hi - lo) / max(Li, 1e-9)
        if ext > 1.0:
            parts = [(lo, 0.0), (Li, hi)]
            for key, name in (("touch", "pr_ext_touch"), ("sw", "pr_ext_sw"), ("fb", "pr_ext_fb")):
                vals, wts = [], []
                for a, b in parts:
                    if b - a > 0.5:
                        v = PR.frac(p, key, a, b)
                        if v == v:
                            vals.append(v); wts.append(b - a)
                F[i, col[name]] = float(np.average(vals, weights=wts)) if vals else np.nan
            F[i, col["pr_ext_touch_gap"]] = max(PR.longest_gap(p, "touch", a, b) for a, b in parts if b - a > 0.5) \
                if any(b - a > 0.5 for a, b in parts) else np.nan
    # ---- split runs
    for i in range(n):
        rr = splits.get(str(i))
        if "da" not in C[i]:
            continue
        rr = rr or []
        F[i, col["sp_runs"]] = len(rr)
        F[i, col["sp_tabgaps"]] = sum(1 for r in rr if _tab_gap(r))
        F[i, col["sp_len_rel"]] = sum(r["g1"] - r["g0"] for r in rr) / max(L[i], 1e-9)
    # ---- neighbourhood
    for i in range(n):
        ncol = ndup = nlong = 0
        a0, a1 = segs[i]
        if L[i] < 1e-6:
            continue
        for j in range(n):
            if j == i or L[j] < 1e-6:
                continue
            ang, off, ov, gap = _geom(a0, a1, *segs[j])
            if ang < DUP_ANG and off < DUP_OFF:
                if ov > DUP_OV:
                    ndup += 1
                    nlong += int(L[j] > L[i])
                elif ov <= 0.05 and gap < 250.0:
                    ncol += 1
        F[i, col["nb_collinear"]] = ncol
        F[i, col["nb_dups"]] = ndup
        F[i, col["nb_dup_longer"]] = nlong
    conn = np.zeros(n)
    disc = np.zeros(n)
    for p in ev.get("pairs") or []:
        i, j = int(p["i"]), int(p["j"])
        if not p.get("has_flanks"):
            continue
        both = bool(p.get("ca")) and bool(p.get("cb"))
        for k in (i, j):
            if k < n:
                if both:
                    conn[k] += 1
                else:
                    disc[k] += 1
    F[:, col["nb_pair_conn"]] = conn
    F[:, col["nb_pair_disc"]] = disc
    # ---- attached Z2 line
    for i in range(n):
        p = prof.get(str(i))
        ids = [e[2] for e in (p or {}).get("z2_ext") or []]
        if not ids:
            continue
        best = max(ids, key=lambda k: (z2[k].get("cover") or 0.0) if k < len(z2) else 0.0)
        if best < len(z2):
            z = z2[best]
            F[i, col["z2_cover"]] = z.get("cover") if z.get("cover") is not None else np.nan
            F[i, col["z2_score"]] = z.get("z2_score") if z.get("z2_score") is not None else np.nan
            zl = float(np.linalg.norm(np.subtract(z["end"], z["start"])))
            F[i, col["z2_span_rel"]] = zl / max(L[i], 1e-9)
    # ---- scan level
    t = meta.get("t_mm") or meta.get("t_z2") or np.nan
    sp = meta.get("spacing") or meta.get("z2_spacing") or np.nan
    F[:, col["sc_t"]] = t
    F[:, col["sc_spacing"]] = sp
    F[:, col["sc_spacing_t"]] = sp / t if (t == t and sp == sp and t) else np.nan
    F[:, col["sc_log_ncand"]] = math.log(n + 1)
    F[:, col["sc_nfam"]] = meta.get("n_families") if meta.get("n_families") is not None else np.nan
    F[:, col["sc_log_area"]] = math.log(meta["area_total"] + 1) if meta.get("area_total") else np.nan
    F[:, col["sc_z2_lines"]] = len(z2)
    F[:, col["sc_budget"]] = float(bool(meta.get("budget_applied")))
    return F


def full_matrix(rec: dict, extra: np.ndarray | None = None) -> np.ndarray:
    """[26 pooled | V4_FEATS] (+ optional extra columns)."""
    n = len(rec.get("candidates") or [])
    if not n:
        return np.zeros((0, 26 + len(V4_FEATS) + (extra.shape[1] if extra is not None else 0)))
    X = np.asarray(rec["X"], float).reshape(n, -1)
    F = candidate_features(rec)
    parts = [X, F] + ([extra] if extra is not None else [])
    return np.hstack(parts) if len(X) else np.zeros((0, sum(p.shape[1] for p in parts)))
