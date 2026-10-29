"""Map human gold lines of sc_d1732de18023 to the AI relabel with the frozen matcher (read-only on gold)."""
import json, sys
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics, is_match

human_path, label_path, lines_path, out_path = sys.argv[1:5]
H = json.load(open(human_path))
A = json.load(open(label_path))
LD = json.load(open(lines_path))
th = Thresholds()
gold = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in H["bends"]]
pred = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in A["bends"]]
res = match_lines(gold, pred, th)

lines_by_id = {b["bend_id"]: b for b in LD["bends"] if b.get("status") == "ok"}


def dist_to_line(p, o, d):
    r = np.asarray(p) - o
    return float(np.linalg.norm(r - (r @ d) * d))


def fold_frame(bid):
    b = lines_by_id[bid]
    s = np.array(b["label_line"]["start"]); e = np.array(b["label_line"]["end"])
    d = (e - s) / np.linalg.norm(e - s)
    return s, e, d


rows = []
for g in gold:
    G = np.array([g["line_start"], g["line_end"]], dtype=float)
    best = None
    for p in pred:
        P = np.array([p["line_start"], p["line_end"]], dtype=float)
        m = pair_metrics(G, P)
        # same physical fold: parallel within 5 deg and gold midpoint within 8 mm of the AI apex line
        s, e, d = fold_frame(p["bend_id"])
        gm = G.mean(axis=0)
        perp = dist_to_line(gm, s, d)
        t = sorted(float((q - s) @ d) for q in G)
        L = float(np.linalg.norm(e - s))
        cov = max(0.0, min(t[1], L) - max(t[0], 0.0)) / L
        cand = {"ai_id": p["bend_id"], "matcher": {k: round(v, 4) for k, v in m.items()},
                "matcher_is_match": bool(is_match(m, th)),
                "gold_mid_to_ai_apex_mm": round(perp, 3),
                "gold_end_dist_to_ai_apex_mm": [round(dist_to_line(q, s, d), 3) for q in G],
                "gold_span_on_ai_line_t": [round(v, 2) for v in t], "ai_length_mm": round(L, 2),
                "gold_covers_ai_fold_frac": round(cov, 4)}
        if m["axis_angle_deg"] <= 5.0 and perp <= 8.0 and cov > 0.3:
            if best is None or perp < best["gold_mid_to_ai_apex_mm"]:
                best = cand
    matched = next((mm for mm in res["matches"] if mm["gold_id"] == g["bend_id"]), None)
    if best is None:
        status = "removed"
    elif matched and matched["prediction_id"] == best["ai_id"]:
        status = "same_bend"
    else:
        status = "moved"
    rows.append({"human_id": g["bend_id"], "status": status, "frozen_matcher_pair": matched, "same_fold_ai_line": best})
claimed = {r["same_fold_ai_line"]["ai_id"] for r in rows if r["same_fold_ai_line"]}
added = [p["bend_id"] for p in pred if p["bend_id"] not in claimed]
doc = {"scan": "sc_d1732de18023", "human_gold": human_path, "ai_relabel": label_path, "lines": lines_path,
       "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines default Thresholds (5 deg / 4 mm / 60% both spans, Hungarian)",
       "human_count": len(gold), "ai_count": len(pred),
       "frozen_matcher_human_as_gold": {k: res[k] for k in ("true_positive", "false_positive", "false_negative", "unmatched_gold_ids", "unmatched_prediction_ids")},
       "per_human_line": rows, "added_ai_lines": added,
       "status_legend": {"same_bend": "AI line on the same fold and the frozen matcher pairs them",
                          "moved": "AI line on the same fold, but the human line sits off the apex beyond the matcher limit (no match)",
                          "removed": "no formed fold under the human line", "added": "AI fold with no human line"}}
json.dump(doc, open(out_path, "w"), indent=2)
print(json.dumps({"frozen": doc["frozen_matcher_human_as_gold"], "added": added}, indent=1))
for r in rows:
    b = r["same_fold_ai_line"]
    print(r["human_id"], r["status"], "->", b and b["ai_id"], "offset(matcher)", b and b["matcher"]["transverse_offset_mm"],
          "angle", b and b["matcher"]["axis_angle_deg"], "gold_mid_to_apex", b and b["gold_mid_to_ai_apex_mm"],
          "gold covers AI fold", b and b["gold_covers_ai_fold_frac"], "spanOv g/p", b and (b["matcher"]["gold_span_overlap"], b["matcher"]["prediction_span_overlap"]))
