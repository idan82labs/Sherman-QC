"""Fit measured outer-apex lines of the two wall folds; compare gold M01/M02 with them using the
frozen matcher's pair_metrics (scripts/evaluate_f1_against_manual_gold.py). Read-only."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds, is_match
from a1_common import gold_lines, OUT

F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
W = json.loads((OUT / "a1_wall_folds.json").read_text())
g = gold_lines()
EXT = {"M01": (-102.5, 97.5), "M02": (-97.5, 102.5)}
res = {}
for key in ("M01", "M02"):
    rows = [r for r in W[key] if r.get("status") == "fold" and r.get("apex") is not None]
    U = np.array([[r["u"], r["apex"][0], r["apex"][1]] for r in rows])
    S = np.array([[r["u"], r["Xo"][0], r["Xo"][1]] for r in rows])
    # least squares v(u), w(u)
    cv = np.polyfit(U[:, 0], U[:, 1], 1); cw = np.polyfit(U[:, 0], U[:, 2], 1)
    rv = U[:, 1] - np.polyval(cv, U[:, 0]); rw = U[:, 2] - np.polyval(cw, U[:, 0])
    u0, u1 = EXT[key]
    a = np.array([u0, np.polyval(cv, u0), np.polyval(cw, u0)]); b = np.array([u1, np.polyval(cv, u1), np.polyval(cw, u1)])
    A = o + a @ R; B = o + b @ R
    sv = np.polyfit(S[:, 0], S[:, 1], 1); sw = np.polyfit(S[:, 0], S[:, 2], 1)
    As = o + np.array([u0, np.polyval(sv, u0), np.polyval(sw, u0)]) @ R
    Bs = o + np.array([u1, np.polyval(sv, u1), np.polyval(sw, u1)]) @ R
    gold = np.array(g[key])
    m_apex = pair_metrics(gold, np.array([A, B]))
    m_sharp = pair_metrics(gold, np.array([As, Bs]))
    # gold restricted to the fold extent (what the line would be if extent were right)
    s, e = (gold - o) @ R.T
    def at(u):
        t = (u - s[0]) / (e[0] - s[0]); return s + t * (e - s)
    gclip = np.array([o + at(max(u0, min(s[0], e[0]))) @ R, o + at(min(u1, max(s[0], e[0]))) @ R])
    m_clip = pair_metrics(gclip, np.array([A, B]))
    res[key] = {"apex_line_raw": [A.tolist(), B.tolist()], "sharp_line_raw": [As.tolist(), Bs.tolist()],
                "apex_fit_rms_v": float(rv.std()), "apex_fit_rms_w": float(rw.std()),
                "fold_extent_u": [u0, u1], "fold_length_mm": float(np.linalg.norm(B - A)),
                "match_gold_vs_apex": {**m_apex, "is_match": is_match(m_apex, Thresholds())},
                "match_gold_vs_outer_sharp": {**m_sharp, "is_match": is_match(m_sharp, Thresholds())},
                "match_gold_clipped_to_fold_vs_apex": {**m_clip, "is_match": is_match(m_clip, Thresholds())}}
    print(key, "fold len %.1f" % res[key]["fold_length_mm"], "apex rms v/w %.2f/%.2f" % (rv.std(), rw.std()))
    for k in ("match_gold_vs_apex", "match_gold_vs_outer_sharp", "match_gold_clipped_to_fold_vs_apex"):
        m = res[key][k]
        print("   %-36s ang %.2f  transv %.2f  gold_ovl %.2f  pred_ovl %.2f  match=%s" % (k, m["axis_angle_deg"], m["transverse_offset_mm"], m["gold_span_overlap"], m["prediction_span_overlap"], m["is_match"]))
(OUT / "a1_match_check.json").write_text(json.dumps(res, indent=1))
