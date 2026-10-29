"""Map the Gold11 human lines of sc_71cf0b216f43 to the AI relabel with the frozen matcher (read-only on gold).

Fold identity comes from the pairs notes (human Mk -> AI Bk for k = 1..8); B9-B12 are the joggle bends.
For each human line: frozen-matcher metrics vs its own fold's AI apex line, the offset at the human
line's start / mid / end (perpendicular distance to the AI apex line), and whether the frozen
Hungarian matching (human vs AI, all lines) pairs them.
"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics, is_match

human_path, label_path, out_path = map(Path, sys.argv[1:4])
H = json.loads(human_path.read_text())
A = json.loads(label_path.read_text())
th = Thresholds()
gold = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in H["bends"]]
pred = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in A["bends"]]
res = match_lines(gold, pred, th)
match_of = {m["gold_id"]: m["prediction_id"] for m in res["matches"]}
identity = {f"M0{k}": f"B{k}" for k in range(1, 9)}
fold_desc = {
    "B1": "short wall 8 -> 12 mm lip (lip only between the long flanges' free edges)",
    "B2": "short wall 7 -> 12 mm lip",
    "B3": "short wall 8 -> bottom frame ring",
    "B4": "short wall 7 -> bottom frame ring",
    "B5": "long wall 3 -> bottom frame ring",
    "B6": "long wall 3 -> top flange (pre-joggle)",
    "B7": "long wall 4 -> bottom frame ring",
    "B8": "long wall 4 -> top flange (pre-joggle)",
    "B9": "joggle bend A, wall-3 top flange",
    "B10": "joggle bend B, wall-3 top flange (line on scanned concave upper skin)",
    "B11": "joggle bend A, wall-4 top flange",
    "B12": "joggle bend B, wall-4 top flange",
}
P = {p["bend_id"]: np.array([p["line_start"], p["line_end"]], float) for p in pred}

def perp(q, s, d):
    r = np.asarray(q) - s
    return float(np.linalg.norm(r - (r @ d) * d))

rows = []
for g in gold:
    G = np.array([g["line_start"], g["line_end"]], float)
    aid = identity[g["bend_id"]]
    Q = P[aid]
    s, e = Q
    L = float(np.linalg.norm(e - s)); d = (e - s) / L
    m = pair_metrics(G, Q)
    ts = sorted(float((q - s) @ d) for q in G)
    samples = [G[0] + f * (G[1] - G[0]) for f in np.linspace(0, 1, 101)]
    offs = np.array([perp(q, s, d) for q in samples])
    within4 = float((offs <= 4.0).mean())
    matched_to = match_of.get(g["bend_id"])
    if matched_to == aid:
        status = "same_bend"
    elif matched_to is None:
        status = "moved"   # same physical fold, AI line re-placed on the apex/extent; human line fails the frozen matcher
    else:
        status = f"matched_other:{matched_to}"
    rows.append({
        "human_id": g["bend_id"], "ai_id": aid, "fold": fold_desc[aid], "status": status,
        "frozen_matcher": {k: round(v, 3) for k, v in m.items()}, "frozen_is_match": bool(is_match(m, th)),
        "offset_start_mid_end_mm": [round(float(offs[0]), 2), round(float(offs[50]), 2), round(float(offs[-1]), 2)],
        "human_frac_within_4mm_of_ai_apex": round(within4, 3),
        "human_span_t_on_ai_line": [round(v, 1) for v in ts], "ai_length_mm": round(L, 1),
        "human_length_mm": round(float(np.linalg.norm(G[1] - G[0])), 1),
    })
added = [{"ai_id": p["bend_id"], "fold": fold_desc[p["bend_id"]], "status": "added"} for p in pred
         if p["bend_id"] not in identity.values()]
out = {"frozen_matcher_thresholds": vars(th), "human_count": len(gold), "ai_count": len(pred),
       "frozen_match_human_vs_ai": {k: res[k] for k in ("true_positive", "false_positive", "false_negative",
                                                          "matches", "unmatched_gold_ids", "unmatched_prediction_ids")},
       "per_human_line": rows, "added": added}
out_path.write_text(json.dumps(out, indent=1))
print("frozen matcher human(8) vs AI(12): TP", res["true_positive"], "FP", res["false_positive"], "FN", res["false_negative"])
for r in rows:
    fm = r["frozen_matcher"]
    print(r["human_id"], "->", r["ai_id"], r["status"], "| off", fm["transverse_offset_mm"], "ang", fm["axis_angle_deg"],
          "ovl h/ai", fm["gold_span_overlap"], fm["prediction_span_overlap"], "| off s/m/e", r["offset_start_mid_end_mm"],
          "| within4", r["human_frac_within_4mm_of_ai_apex"], "| len h/ai", r["human_length_mm"], r["ai_length_mm"])
for a in added:
    print(a)
