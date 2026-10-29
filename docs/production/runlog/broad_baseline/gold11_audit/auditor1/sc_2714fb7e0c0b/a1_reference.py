"""Auditor-1 measured apex-locus reference lines for every formed fold of sc_2714fb7e0c0b, and the
frozen matcher (scripts/evaluate_f1_against_manual_gold.match_lines, default thresholds, read-only
import) run gold-vs-reference. Reference lines = least-squares line through the per-station outer
apex points (a1_stations.json / a1_missing.json / a1_tabkink_apex.json), clipped to the measured
fold extent. Output stays in this directory; nothing is written to any label root."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
import numpy as np
import a1common as A
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics

gold = A.load_gold()
ex, ez, en = A.part_frame(gold)
X, Z, N = ex, ez, en
CFG = {"B01": (-N, -X), "B02": (-N, X), "B03": (-N, Z), "B04": (-N, Z), "B05": (N, -Z),
       "B07": (-N, -Z), "B09": (N, X), "B10": (N, -X)}
st = json.loads((HERE / "a1_stations.json").read_text())
ref = []
for bid, (nA, nB) in CFG.items():
    s0, s1 = gold[bid]
    d = A.unit(s1 - s0)
    o = A.unit(nA + nB); g = -o; v = A.unit(np.cross(d, g))
    P3 = []
    f0, f1 = st[bid]["summary"]["fold_extent_s_mm"]  # longest present run only (drops isolated stations)
    for r in st[bid]["stations"]:
        if r.get("present") and f0 <= r["s"] <= f1:
            c = s0 + r["s"] * d
            P3.append(c + r["apex"][0] * g + r["apex"][1] * v)
    P3 = np.array(P3)
    m = P3.mean(0)
    _, _, vt = np.linalg.svd(P3 - m)
    dd = vt[0] if vt[0] @ d > 0 else -vt[0]
    t = (P3 - m) @ dd
    resid = np.linalg.norm((P3 - m) - np.outer(t, dd), axis=1)
    ref.append({"bend_id": f"F_{bid}", "line_start": (m + t.min() * dd).tolist(), "line_end": (m + t.max() * dd).tolist(),
                "note": f"apex line of the fold gold {bid} sits on", "apex_fit_resid_med_mm": float(np.median(resid)),
                "length_mm": float(t.max() - t.min())})
mis = json.loads((HERE / "a1_missing.json").read_text())
for k, lab in (("E_neg", "lower-tray end-flange fold at the B01/B06 end (drawing Section D-D, CAD C9)"),
               ("E_pos", "lower-tray end-flange fold at the B02/B08 end (drawing Section D-D, CAD C1)")):
    a, b = mis[k]["apex_line_raw"]
    ref.append({"bend_id": f"F_{k}", "line_start": a, "line_end": b, "note": lab, "length_mm": mis[k]["apex_line_len"]})
kin = json.loads((HERE / "a1_tabkink_apex.json").read_text())
for k, lab in (("B09_kink_apex", "30-deg kink at the end of the B09 window tab (drawing Section C-C, CAD C7)"),
               ("B10_kink_apex", "30-deg kink at the end of the B10 window tab (drawing Section C-C, CAD C5)")):
    a, b = kin[k]["raw"]
    ref.append({"bend_id": f"F_{k.split('_')[0]}_kink", "line_start": a, "line_end": b, "note": lab, "length_mm": kin[k]["len"]})

gold_rows = [{"bend_id": k, "line_start": v[0].tolist(), "line_end": v[1].tolist()} for k, v in gold.items()]
res = match_lines(gold_rows, ref, Thresholds())
# per gold line: metrics to the reference fold it physically belongs to
pairs = {}
for gr in gold_rows:
    tgt = f"F_{gr['bend_id']}"
    rr = [r for r in ref if r["bend_id"] == tgt]
    if rr:
        pm = pair_metrics(np.array([gr["line_start"], gr["line_end"]]), np.array([rr[0]["line_start"], rr[0]["line_end"]]))
        pairs[gr["bend_id"]] = {k: round(v, 3) for k, v in pm.items()}
# B06/B08 vs the nearest real folds (end flanges) for the record
for gid, tgt in (("B06", "F_E_neg"), ("B08", "F_E_pos")):
    gr = [g_ for g_ in gold_rows if g_["bend_id"] == gid][0]
    rr = [r for r in ref if r["bend_id"] == tgt][0]
    pm = pair_metrics(np.array([gr["line_start"], gr["line_end"]]), np.array([rr["line_start"], rr["line_end"]]))
    pairs[gid + "_vs_" + tgt] = {k: round(v, 3) for k, v in pm.items()}
out = {"reference_lines_scan_raw": ref, "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines default Thresholds",
       "gold_vs_reference": {k: res[k] for k in ("gold_count", "prediction_count", "true_positive", "false_positive",
                                                   "false_negative", "unmatched_gold_ids", "unmatched_prediction_ids")},
       "matches": res["matches"], "pair_metrics_gold_to_own_fold": pairs}
(HERE / "a1_reference_and_match.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["gold_vs_reference"], indent=1))
for k, v in pairs.items():
    print(k, v)
for r in ref:
    print(r["bend_id"], round(r["length_mm"], 1), np.round(r["line_start"], 2).tolist(), np.round(r["line_end"], 2).tolist())
