"""Map the human gold lines of sc_08686dc1d5d4 to the AI relabel (read-only on the gold).

Matcher: research/broad_baseline/score.py -> scripts/evaluate_f1_against_manual_gold.match_lines with
its default Thresholds (5 deg / 4 mm / 60% of both spans, Hungarian) -- frozen, not changed here.
Usage: python -I mapping_vs_human.py <human_gold.json> <ai_label.json> <lines.json> <out.json>
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, match_lines, pair_metrics  # noqa: E402

human_path, label_path, lines_path, out_path = sys.argv[1:5]
H = json.load(open(human_path))
A = json.load(open(label_path))
LD = json.load(open(lines_path))
th = Thresholds()
gold = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in H["bends"]]
pred = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in A["bends"]]
res = match_lines(gold, pred, th)
notes_by_id = {b["bend_id"]: b.get("notes", "") for b in A["bends"]}


def perp(p, o, d):
    r = np.asarray(p, float) - o
    return float(np.linalg.norm(r - (r @ d) * d))


rows = []
for g in gold:
    G = np.array([g["line_start"], g["line_end"]], dtype=float)
    gd = (G[1] - G[0]) / np.linalg.norm(G[1] - G[0])
    glen = float(np.linalg.norm(G[1] - G[0]))
    cands = []
    for p in pred:
        P = np.array([p["line_start"], p["line_end"]], dtype=float)
        L = float(np.linalg.norm(P[1] - P[0]))
        d = (P[1] - P[0]) / L
        m = pair_metrics(G, P)
        # human line sampled every 1 mm: distance to the AI apex line, and whether it lies over the AI fold span
        n = max(2, int(round(glen)) + 1)
        samples = G[0] + np.outer(np.linspace(0.0, glen, n), gd)
        t = (samples - P[0]) @ d
        dist = np.array([perp(q, P[0], d) for q in samples])
        on_span = (t >= 0.0) & (t <= L)
        over = dist[on_span]
        ts = sorted(float((q - P[0]) @ d) for q in G)
        covered = max(0.0, min(ts[1], L) - max(ts[0], 0.0))
        cands.append({
            "ai_id": p["bend_id"],
            "ai_note": notes_by_id.get(p["bend_id"], "")[:160],
            "matcher": {k: round(v, 4) for k, v in m.items()},
            "matcher_is_match": bool(is_match(m, th)),
            "human_mid_to_ai_line_mm": round(perp(G.mean(axis=0), P[0], d), 3),
            "human_end_to_ai_line_mm": [round(perp(q, P[0], d), 3) for q in G],
            "human_dist_to_ai_line_over_ai_span_mm": None if not len(over) else {
                "min": round(float(over.min()), 3), "median": round(float(np.median(over)), 3),
                "max": round(float(over.max()), 3), "frac_within_4mm": round(float((over <= 4.0).mean()), 3)},
            "human_span_on_ai_line_t": [round(v, 1) for v in ts], "ai_length_mm": round(L, 1),
            "human_covers_ai_fold_frac": round(covered / L, 4),
            "human_length_off_ai_fold_mm": round(glen - covered, 1),
        })
    cands.sort(key=lambda c: (c["matcher"]["axis_angle_deg"] > 5.0, c["human_mid_to_ai_line_mm"]))
    best = cands[0]
    same_fold = (best["matcher"]["axis_angle_deg"] <= 5.0 and best["human_dist_to_ai_line_over_ai_span_mm"] is not None
                 and best["human_dist_to_ai_line_over_ai_span_mm"]["median"] <= 8.0
                 and best["human_covers_ai_fold_frac"] > 0.3)
    matched = next((mm for mm in res["matches"] if mm["gold_id"] == g["bend_id"]), None)
    if not same_fold:
        status = "removed"
    elif matched and matched["prediction_id"] == best["ai_id"]:
        status = "same_bend"
    else:
        status = "moved"
    rows.append({"human_id": g["bend_id"], "human_length_mm": round(glen, 1), "status": status,
                 "frozen_matcher_pair": matched, "nearest_ai_line": best,
                 "same_fold_ai_line": best["ai_id"] if same_fold else None})
claimed = {r["same_fold_ai_line"] for r in rows if r["same_fold_ai_line"]}
added = [p["bend_id"] for p in pred if p["bend_id"] not in claimed]
doc = {"scan": "sc_08686dc1d5d4", "human_gold": human_path, "ai_relabel": label_path, "lines": lines_path,
       "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines default Thresholds (5 deg / 4 mm / 60% both spans, Hungarian), as used by research/broad_baseline/score.py",
       "human_count": len(gold), "ai_count": len(pred),
       "frozen_matcher_human_as_gold": {k: res[k] for k in ("true_positive", "false_positive", "false_negative",
                                                            "unmatched_gold_ids", "unmatched_prediction_ids")},
       "frozen_matcher_matches": res["matches"],
       "per_human_line": rows, "added_ai_lines": added,
       "status_legend": {"same_bend": "AI line on the same formed fold and the frozen matcher pairs them",
                         "moved": "AI line on the same fold, but the human line is off the apex beyond the matcher limit",
                         "removed": "no formed fold under the human line (no AI line on it)",
                         "added": "formed fold with an AI line and no human line"}}
json.dump(doc, open(out_path, "w"), indent=2)
open(out_path, "a").write("\n")
print(json.dumps({"frozen": doc["frozen_matcher_human_as_gold"], "added": added}))
for r in rows:
    b = r["nearest_ai_line"]
    print(r["human_id"], r["status"], "-> nearest", b["ai_id"], "angle", b["matcher"]["axis_angle_deg"],
          "transverse", b["matcher"]["transverse_offset_mm"], "dist over AI span", b["human_dist_to_ai_line_over_ai_span_mm"],
          "covers AI fold", b["human_covers_ai_fold_frac"], "human mm off the AI fold", b["human_length_off_ai_fold_mm"],
          "spanOv g/p", (b["matcher"]["gold_span_overlap"], b["matcher"]["prediction_span_overlap"]))
