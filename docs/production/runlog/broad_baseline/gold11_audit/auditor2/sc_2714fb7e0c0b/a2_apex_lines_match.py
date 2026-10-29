"""Build a2 apex reference lines for every physical fold (from a2_stations.json, clipped to the observed fold
extent) + the two tab-end creases, write them (raw scan frame) to a2_apex_lines.json, and score the gold
against them with the FROZEN matcher (scripts/evaluate_f1_against_manual_gold.match_lines, default Thresholds).
Read-only on inputs; output only inside this auditor directory."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, gold_lines, OUT
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics

q, R, O, raw = load()
G = gold_lines(R, O)
st = json.loads((OUT / "a2_stations.json").read_text())
to_raw = lambda x: (np.asarray(x, float) @ R + O)

# observed fold extents along the fold axis (from slices: fold present where both flanges + arc exist)
EXTENT = {"B01": (-219.5, -1.0), "B02": (-219.4, -1.0), "B03": (2.2, 223.0), "B05": (2.2, 223.0), "B04": (2.1, 222.9),
          "B07": (12.0, 213.5), "B09": (-137.2, -59.5), "B10": (-137.2, -59.5), "EF_v0": (0.6, 44.7), "EF_v225": (0.6, 44.7)}
apex = {}
for name, (lo, hi) in EXTENT.items():
    d = st[name]; ax = d["spec"]["ax"]; i, j = d["spec"]["tr"]
    rows = [r for r in d["stations"] if r.get("ok") and r.get("Ro") and 0.8 < r["Ro"] < 8 and lo + 1 <= r["s"] <= hi - 1]
    S = np.array([r["s"] for r in rows]); A = np.array([r["apex"] for r in rows])
    pi = np.polyfit(S, A[:, 0], 1); pj = np.polyfit(S, A[:, 1], 1)
    resid = np.hypot(A[:, 0] - np.polyval(pi, S), A[:, 1] - np.polyval(pj, S))
    ends = []
    for s in (lo, hi):
        x = np.zeros(3); x[ax] = s; x[i] = np.polyval(pi, s); x[j] = np.polyval(pj, s); ends.append(x)
    apex[name] = {"frame_uvw": [e.round(3).tolist() for e in ends], "raw": [to_raw(e).round(3).tolist() for e in ends],
                  "n_stations": len(rows), "apex_fit_resid_med_mm": float(np.median(resid)),
                  "R_o_med": float(np.median([r["Ro"] for r in rows])), "bend_deg_med": float(np.median([r["bend_deg"] for r in rows]))}
# tab-end creases: vertical (w) apex lines on the convex skin at the crease (u ~ -58.5), w 1.7..14.5
for name, v_out in (("CR_B09", 31.45), ("CR_B10", 204.35)):
    ends = [np.array([-58.5, v_out, 1.7]), np.array([-58.5, v_out, 14.5])]
    apex[name] = {"frame_uvw": [e.round(3).tolist() for e in ends], "raw": [to_raw(e).round(3).tolist() for e in ends],
                  "note": "approximate: convex-skin apex of the ~28-30 deg vertical crease at the +u end of the tab"}
(OUT / "a2_apex_lines.json").write_text(json.dumps({"frame": "source_ply_raw (scan raw), mm", "lines": apex}, indent=1))

gold_rec = [{"bend_id": k, "line_start": to_raw(s).tolist(), "line_end": to_raw(e).tolist()} for k, (s, e) in G.items()]
folds_only = [{"bend_id": k, "line_start": v["raw"][0], "line_end": v["raw"][1]} for k, v in apex.items() if not k.startswith("CR_")]
with_creases = [{"bend_id": k, "line_start": v["raw"][0], "line_end": v["raw"][1]} for k, v in apex.items()]
th = Thresholds()
for label, preds in (("a2 apex folds (10)", folds_only), ("a2 apex folds + tab creases (12)", with_creases)):
    res = match_lines(gold_rec, preds, th)
    print(f"== gold (10) vs {label}: TP {res['true_positive']} FP {res['false_positive']} FN {res['false_negative']}")
    for m in res["matches"]:
        print("   ", m["gold_id"], "<->", m["prediction_id"], {k: round(v, 2) for k, v in m.items() if k not in ("gold_id", "prediction_id")})
# per-gold best pair metrics (diagnostic)
print("== per-gold pair metrics vs its physical fold")
pairs = {"B01": "B01", "B02": "B02", "B03": "B03", "B04": "B04", "B05": "B05", "B07": "B07", "B09": "B09", "B10": "B10", "B06": "EF_v0", "B08": "EF_v225"}
for gk, pk in pairs.items():
    gl = np.array([gold_rec[[g["bend_id"] for g in gold_rec].index(gk)]["line_start"], gold_rec[[g["bend_id"] for g in gold_rec].index(gk)]["line_end"]])
    pl = np.array(apex[pk]["raw"])
    print("   ", gk, "vs", pk, {k: round(v, 2) for k, v in pair_metrics(gl, pl).items()})
