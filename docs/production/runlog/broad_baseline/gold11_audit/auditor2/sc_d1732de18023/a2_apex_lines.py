"""Auditor-2: fit the outer-apex line, inner-apex line and fold extent per bend from
a2_xsec.json + the scan, then evaluate every gold line against them with the FROZEN
matcher metrics (scripts/evaluate_f1_against_manual_gold).  Writes a2_apex_lines.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from a2_common import OUT, load_gold, load_ply, unit  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

pts, nrm, _ = load_ply()
tree = cKDTree(pts)
G = load_gold()
X = json.loads((OUT / "a2_xsec.json").read_text())


def to3d(fr, s, p2):
    d, up, mid, f = (np.array(fr[k]) for k in ("d", "e3", "mid", "f"))
    u_in = unit(np.cross(d, up))
    if u_in @ f > 0:
        u_in = -u_in
    w = unit(np.cross(u_in, d))
    if w @ up < 0:
        w = -w
    return mid + d * s + u_in * p2[0] + w * p2[1], (d, u_in, w)


def fit_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    v = vt[0]
    t = (P - c) @ v
    res = np.linalg.norm((P - c) - np.outer(t, v), axis=1)
    return c, v, t, res


def pt_line_dist(p, c, v):
    q = p - c
    return float(np.linalg.norm(q - v * (q @ v)))


out = {}
for bid, rec in X.items():
    fr = rec["frame"]
    st = [s for s in rec["stations"] if s.get("ok") and "apex" in s]
    A3 = np.array([to3d(fr, s["s"], s["apex"])[0] for s in st])
    I3 = np.array([to3d(fr, s["s"], s["inner_apex"])[0] for s in st if "inner_apex" in s])
    Xo3 = np.array([to3d(fr, s["s"], s["Xo"])[0] for s in st])
    Xi3 = np.array([to3d(fr, s["s"], s["Xi"])[0] for s in st])
    c, v, t, res = fit_line(A3)
    ci, vi, ti, resi = fit_line(I3)
    co, vo, to_, reso = fit_line(Xo3)
    cx, vx, tx, resx = fit_line(Xi3)
    a, b = G[bid]
    dg = unit(b - a)
    if v @ dg < 0:
        v = -v
    # fold extent along the apex line: outer-arc points = normal within 20 deg of bisector
    d, u_in, w = to3d(fr, 0.0, [0, 0])[1]
    nb2 = np.array(st[len(st) // 2]["nbis"])
    nbis3 = unit(u_in * nb2[0] + w * nb2[1])
    rel = pts - c
    tt = rel @ v
    perp = np.linalg.norm(rel - np.outer(tt, v), axis=1)
    arc = (perp < 1.2) & (nrm @ nbis3 > np.cos(np.radians(20)))
    ts = np.sort(tt[arc])
    # gaps > 3 mm split the fold into runs
    runs = []
    if len(ts):
        start = ts[0]
        prev = ts[0]
        for x in ts[1:]:
            if x - prev > 3.0:
                runs.append([float(start), float(prev)])
                start = x
            prev = x
        runs.append([float(start), float(prev)])
    runs = [r for r in runs if r[1] - r[0] > 2.0]
    fold_lo, fold_hi = (runs[0][0], runs[-1][1]) if runs else (np.nan, np.nan)
    fold_len = fold_hi - fold_lo
    gs, ge = sorted([float((a - c) @ v), float((b - c) @ v)])
    ov = max(0.0, min(ge, fold_hi) - max(gs, fold_lo))
    coverage = ov / fold_len if fold_len > 0 else np.nan
    # perfect apex prediction spanning the fold extent
    pred_apex = [c + v * fold_lo, c + v * fold_hi]
    pred_inner = [ci + vi * ((c + v * fold_lo - ci) @ vi), ci + vi * ((c + v * fold_hi - ci) @ vi)]
    pred_xo = [co + vo * ((c + v * fold_lo - co) @ vo), co + vo * ((c + v * fold_hi - co) @ vo)]
    pred_xi = [cx + vx * ((c + v * fold_lo - cx) @ vx), cx + vx * ((c + v * fold_hi - cx) @ vx)]
    mm = {}
    for name, pl in (("outer_apex", pred_apex), ("inner_apex", pred_inner),
                     ("outer_virtual_sharp", pred_xo), ("inner_virtual_sharp", pred_xi)):
        met = pair_metrics(np.array([a, b]), np.array(pl))
        met["is_match"] = bool(is_match(met, Thresholds()))
        mm[name] = met
    out[bid] = {
        "apex_line": {"c": c.tolist(), "v": v.tolist(), "fit_rms_mm": float(np.sqrt(np.mean(res ** 2))),
                      "n_stations": len(st)},
        "angle_gold_vs_apex_deg": float(np.degrees(np.arccos(min(1.0, abs(v @ dg))))),
        "gold_start_to_apex_line": pt_line_dist(a, c, v), "gold_end_to_apex_line": pt_line_dist(b, c, v),
        "gold_mid_to_apex_line": pt_line_dist((a + b) / 2, c, v),
        "gold_mid_to_inner_apex_line": pt_line_dist((a + b) / 2, ci, vi),
        "gold_mid_to_outer_sharp_line": pt_line_dist((a + b) / 2, co, vo),
        "gold_mid_to_inner_sharp_line": pt_line_dist((a + b) / 2, cx, vx),
        "apex_to_inner_apex_mm": pt_line_dist(c, ci, vi),
        "apex_to_outer_sharp_mm": pt_line_dist(c, co, vo),
        "fold_runs_t": runs, "fold_extent_t": [fold_lo, fold_hi], "fold_len_mm": fold_len,
        "gold_extent_t": [gs, ge], "gold_len_mm": ge - gs,
        "gold_overlap_mm": ov, "coverage_of_fold": coverage,
        "gold_beyond_fold_mm": [max(0.0, fold_lo - gs), max(0.0, ge - fold_hi)],
        "uncovered_fold_mm": [max(0.0, gs - fold_lo), max(0.0, fold_hi - ge)],
        "pred_apex_line": [p.tolist() for p in pred_apex],
        "frozen_matcher_vs_ideal_line_on_fold_extent": mm,
    }
    o = out[bid]
    print(f"{bid}: fold [{fold_lo:.1f},{fold_hi:.1f}] len {fold_len:.1f} runs {len(runs)} | gold [{gs:.1f},{ge:.1f}] len {ge-gs:.1f} "
          f"cov {coverage*100:.1f}% beyond {np.round(o['gold_beyond_fold_mm'],1)} uncovered {np.round(o['uncovered_fold_mm'],1)} | "
          f"ang {o['angle_gold_vs_apex_deg']:.2f} | d(start,mid,end) to apex {o['gold_start_to_apex_line']:.2f} "
          f"{o['gold_mid_to_apex_line']:.2f} {o['gold_end_to_apex_line']:.2f} | mid->innerapex {o['gold_mid_to_inner_apex_line']:.2f} "
          f"mid->Xo {o['gold_mid_to_outer_sharp_line']:.2f} mid->Xi {o['gold_mid_to_inner_sharp_line']:.2f} | apex fit rms {o['apex_line']['fit_rms_mm']:.2f}")
    print("    runs:", [[round(x, 1) for x in r] for r in runs])
    for k, m in mm.items():
        print(f"    vs ideal {k:20s}: match={m['is_match']} ang {m['axis_angle_deg']:.2f} off {m['transverse_offset_mm']:.2f} gov {m['gold_span_overlap']:.2f} pov {m['prediction_span_overlap']:.2f}")
json.dump(out, open(OUT / "a2_apex_lines.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
