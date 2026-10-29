"""Final per-gold-line numbers for auditor 3 (sc_71cf0b216f43):
- fold extent along its own axis (wall-end to wall-end; short-side bottom folds only between the
  long-side bottom flanges' free edges, verified visually in a3_shortbot_ends_*.png)
- outer-apex line refit inside the true extent; frozen pair_metrics vs that line
- per-station offsets: d_apex, along-bisector (+concave), across (+toward flange/plate), nearest scan
- span coverage of the fold and fraction of the fold where the gold is within 4 mm of the apex
Writes a3_verdict_numbers.json."""
import sys
sys.dont_write_bytecode = True
import json
import math
import numpy as np
from a3_common import *
from a3_stations import SIDES, GOLD_SIDE
from a3_apexlines import uh_to_raw, fit_line3
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds

C0 = np.load(SCR / "center.npy")
st = json.loads((OUT / "a3_stations.json").read_text())
lines_all = json.loads((OUT / "a3_apex_lines.json").read_text())["lines"]
# fold extents along the side axis (part-frame coordinate of that axis), from the wall-end slices
# (a3_corner_slices.png) and the short-bottom transition slices (a3_shortbot_ends_*.png)
EXTENT = {"Sminus_top": (-146.0, 160.6), "Sminus_bot": (-146.0, 160.6),
          "Splus_top": (-146.0, 161.0), "Splus_bot": (-146.0, 161.0),
          "Lminus_top": (-96.0, 103.6), "Lplus_top": (-96.0, 103.6),
          "Lminus_bot": (-62.6, 70.0), "Lplus_bot": (-62.6, 70.5)}
gold = {b: np.array([a, e]) for b, a, e in load_gold()}
th = Thresholds()
out = {}
for bid, (side, corner) in GOLD_SIDE.items():
    key = f"{side}_{corner}"
    lo, hi = EXTENT[key]
    rows = [r for r in st["bends"][key] if r.get("ok") and r.get("apex") and lo + 2 <= r["a"] <= hi - 2]
    pts = [uh_to_raw(side, r["a"], *r["apex"]) for r in rows]
    c, d, t0, t1, rms, n = fit_line3(pts)
    # extend the fitted apex line to the full fold extent along the side axis
    ax = SIDES[side]["ax"]
    axis_vec = [L0, S0][ax]
    def at(aval):
        # point on fitted line whose side-axis coordinate (relative to C0) equals aval
        s = (aval - (c - C0) @ axis_vec) / (d @ axis_vec)
        return c + s * d
    ap = np.array([at(lo), at(hi)])
    m = pair_metrics(gold[bid], ap)
    gr = [r for r in st["bends"][key] if r.get("gold") and lo <= r["a"] <= hi]
    dap = np.array([r["gold"]["d_apex"] for r in gr])
    alo = np.array([r["gold"]["along_bisector_concave"] for r in gr])
    g_ax = np.array([r["a"] for r in gr])
    # recompute across with a clean definition: + toward the flange/plate branch, - toward the wall
    acr = []
    for r in gr:
        g = np.array(r["g"]); q = np.array(r["gold"]["uh"]); apex = np.array(r["apex"])
        gp = np.array([-g[1], g[0]])
        # orient gp toward +u (plate/flange branch is at larger u)
        if gp[0] < 0:
            gp = -gp
        acr.append(float((q - apex) @ gp))
    acr = np.array(acr)
    dsc = np.array([r["gold"]["d_nearest_scan"] for r in gr])
    # gold span along the side axis
    ga = sorted(((gold[bid] - C0) @ axis_vec).tolist())
    ov = max(0.0, min(hi, ga[1]) - max(lo, ga[0]))
    fold_len = hi - lo
    # fraction of fold length with gold within 4 mm of apex (stations every 2 mm)
    within = float(np.mean(dap <= 4.0)) * (len(gr) * 2.0) / fold_len if len(gr) else 0.0
    out[bid] = {"fold": key, "fold_extent_axis_mm": [lo, hi], "fold_length_mm": fold_len,
                "gold_axis_span_mm": ga, "gold_length_mm": float(np.linalg.norm(gold[bid][1] - gold[bid][0])),
                "span_coverage_of_fold_pct": round(100 * ov / fold_len, 1),
                "gold_length_outside_fold_pct": round(100 * (1 - ov / (ga[1] - ga[0])), 1),
                "fold_pct_with_gold_within_4mm_of_apex": round(100 * min(within, 1.0), 1),
                "d_apex_mm": {"min": round(float(dap.min()), 2), "median": round(float(np.median(dap)), 2),
                              "max": round(float(dap.max()), 2),
                              "at_25_50_75pct_of_fold": [round(float(np.interp(lo + f * fold_len, g_ax, dap)), 2) for f in (0.25, 0.5, 0.75)]},
                "along_bisector_concave_mm_median": round(float(np.median(alo)), 2),
                "across_toward_flange_mm_median": round(float(np.median(acr)), 2),
                "nearest_scan_mm_median": round(float(np.median(dsc)), 2),
                "apex_line_scan_raw": {"start": ap[0].tolist(), "end": ap[1].tolist(), "rms_mm": rms, "n_stations": n},
                "frozen_matcher_vs_apex_line": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                "frozen_match": bool(is_match(m, th))}
    o = out[bid]
    print(bid, key, "fold %.0f mm" % fold_len, "| span cov %.0f%% outside %.0f%% | within4 %.0f%%" % (
        o["span_coverage_of_fold_pct"], o["gold_length_outside_fold_pct"], o["fold_pct_with_gold_within_4mm_of_apex"]),
        "| d_apex", o["d_apex_mm"], "| along %.2f across %.2f scan %.2f" % (
        o["along_bisector_concave_mm_median"], o["across_toward_flange_mm_median"], o["nearest_scan_mm_median"]),
        "| matcher ang %.2f off %.2f gov %.2f pov %.2f match %s" % (
        m["axis_angle_deg"], m["transverse_offset_mm"], m["gold_span_overlap"], m["prediction_span_overlap"], o["frozen_match"]))
(OUT / "a3_verdict_numbers.json").write_text(json.dumps(out, indent=1))
