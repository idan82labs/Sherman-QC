"""Map the human Gold11 lines of sc_f2475286312b to the AI relabel (read-only on the gold).

1. Frozen matcher (scripts/evaluate_f1_against_manual_gold.match_lines, default Thresholds:
   5 deg / 4 mm / 60% of both spans, Hungarian), human lines as gold, AI label as prediction.
2. Same-fold association, one to one (min-cost assignment on the matcher's transverse offset).
   Eligible pairs: axis angle <= 5 deg, transverse offset <= 16 mm, and >= 60% overlap of the
   shorter span. A human line the matcher pairs is 'same_bend'. A human line assigned to an AI
   fold but not paired by the matcher is 'moved'. A human line with no eligible fold, or one that
   loses the fold to a closer human line, is 'removed'. An AI line with no human line is 'added'.
3. Locus check of each human line in its AI fold's cross-section frame. Positions are signed
   along the fold bisector from the AI apex (+ = concave side, into the corner; - = convex side,
   toward the outer virtual sharp), and 'across' is perpendicular to the bisector. Reference
   loci on the same axis (90 deg, t, Ri from lines.json): outer sharp, outer apex (0), inner apex
   (+t), inner sharp, bend-arc centre. These locate the human line relative to every standard locus.
Usage: python mapping_vs_human.py <human_label.json> <ai_label.json> <lines.json> <out.json>
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, match_lines, pair_metrics  # noqa: E402

human_path, label_path, lines_path, out_path = sys.argv[1:5]
H = json.load(open(human_path))
A = json.load(open(label_path))
LD = json.load(open(lines_path))
th = Thresholds()
MAX_OFF, MIN_SHORT_OV, MAX_ANG = 16.0, 0.6, 5.0
gold = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in H["bends"]]
pred = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]} for b in A["bends"]]
res = match_lines(gold, pred, th)
lines_by_id = {b["bend_id"]: b for b in LD["bends"] if b.get("status") == "ok"}
t_mm, ri_mm = LD["thickness_mm"], LD["ri_mm"]


def seg(x):
    return np.array([x["line_start"], x["line_end"]], dtype=float)


M = {(g["bend_id"], p["bend_id"]): pair_metrics(seg(g), seg(p)) for g in gold for p in pred}
cost = np.full((len(gold), len(pred)), 1e6)
for i, g in enumerate(gold):
    for j, p in enumerate(pred):
        m = M[(g["bend_id"], p["bend_id"])]
        if (m["axis_angle_deg"] <= MAX_ANG and m["transverse_offset_mm"] <= MAX_OFF
                and m["shorter_span_overlap"] >= MIN_SHORT_OV):
            cost[i, j] = m["transverse_offset_mm"]
rows_i, cols_j = linear_sum_assignment(cost)
assigned = {gold[i]["bend_id"]: pred[j]["bend_id"] for i, j in zip(rows_i, cols_j) if cost[i, j] < 1e6}


def locus_frame(g, ai_id):
    """Human line stations in the AI fold's section frame (apex origin, + bisector = concave)."""
    b = lines_by_id[ai_id]
    s, e = np.asarray(b["label_line"]["start"]), np.asarray(b["label_line"]["end"])
    d = np.asarray(b["direction"])
    bis = np.asarray(b["bisector"])
    e2 = np.cross(d, bis)
    e2 /= np.linalg.norm(e2)
    L = float(np.linalg.norm(e - s))
    G = seg(g)
    n = max(2, int(np.ceil(np.linalg.norm(G[1] - G[0]) / 1.0)) + 1)
    pts = G[0] + np.linspace(0.0, 1.0, n)[:, None] * (G[1] - G[0])
    rel = pts - s
    tt = rel @ d
    w = rel - np.outer(tt, d)
    on = (tt >= 0.0) & (tt <= L)
    along, across = w @ bis, w @ e2
    half = np.radians(b["bend_angle_deg"]) / 2.0
    apex_off = float(b["apex_offset_mm"])            # outer sharp -> outer apex, along +bisector
    inner_sharp = -apex_off + t_mm / np.cos(half)     # inner virtual sharp
    arc_centre = (ri_mm + t_mm) / np.cos(half) - apex_off
    loci = {"outer_sharp": -apex_off, "outer_apex": 0.0, "inner_apex": t_mm,
            "inner_sharp": inner_sharp, "arc_centre": arc_centre}
    med_along = float(np.median(along[on])) if on.any() else None
    med_across = float(np.median(across[on])) if on.any() else None
    span = sorted(float(v) for v in ((G - s) @ d))
    return {
        "human_stations_on_ai_fold": int(on.sum()), "human_stations_total": int(n),
        "human_frac_on_ai_fold_span": round(float(on.mean()), 3),
        "ai_fold_frac_covered_by_human": round(max(0.0, min(span[1], L) - max(span[0], 0.0)) / L, 3),
        "human_overrun_past_fold_mm": [round(max(0.0, -span[0]), 1), round(max(0.0, span[1] - L), 1)],
        "along_bisector_median_mm": None if med_along is None else round(med_along, 2),
        "along_bisector_p5_p95_mm": [round(float(np.percentile(along[on], q)), 2) for q in (5, 95)] if on.any() else None,
        "across_median_mm": None if med_across is None else round(med_across, 2),
        "loci_along_bisector_mm": {k: round(v, 2) for k, v in loci.items()},
        "dist_to_loci_mm": None if med_along is None else {
            k: round(float(np.hypot(med_along - v, med_across)), 2) for k, v in loci.items()},
    }


rows = []
for g in gold:
    gid = g["bend_id"]
    matched = next((mm for mm in res["matches"] if mm["gold_id"] == gid), None)
    ai_id = assigned.get(gid)
    if ai_id is None:
        status = "removed"
    elif matched and matched["prediction_id"] == ai_id:
        status = "same_bend"
    else:
        status = "moved"
    near = sorted(((p["bend_id"], M[(gid, p["bend_id"])]) for p in pred),
                  key=lambda kv: (kv[1]["axis_angle_deg"] > MAX_ANG, kv[1]["transverse_offset_mm"]))[:2]
    row = {"human_id": gid, "status": status, "ai_id": ai_id, "frozen_matcher_pair": matched,
           "nearest_parallel_ai": [{"ai_id": k, **{kk: round(float(vv), 3) for kk, vv in m.items()}} for k, m in near]}
    if ai_id:
        m = M[(gid, ai_id)]
        row["matcher_vs_ai"] = {k: round(float(v), 3) for k, v in m.items()}
        row["matcher_vs_ai_is_match"] = bool(is_match(m, th))
        row["locus"] = locus_frame(g, ai_id)
    rows.append(row)
claimed = set(assigned.values())
added = [p["bend_id"] for p in pred if p["bend_id"] not in claimed]
doc = {"scan": "sc_f2475286312b", "human_gold": human_path, "ai_relabel": label_path, "lines": lines_path,
       "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines default Thresholds (5 deg / 4 mm / 60% both spans, Hungarian)",
       "association_rule": {"max_axis_angle_deg": MAX_ANG, "max_transverse_offset_mm": MAX_OFF,
                            "min_shorter_span_overlap": MIN_SHORT_OV, "assignment": "one-to-one, min transverse offset"},
       "human_count": len(gold), "ai_count": len(pred),
       "frozen_matcher_human_as_gold": {k: res[k] for k in ("true_positive", "false_positive", "false_negative",
                                                            "unmatched_gold_ids", "unmatched_prediction_ids")},
       "per_human_line": rows, "added_ai_lines": added,
       "status_legend": {"same_bend": "AI line on the same fold and the frozen matcher pairs them",
                          "moved": "assigned to the same AI fold, but the human line is off the apex and/or has the wrong extent beyond the matcher limits",
                          "removed": "no AI fold left for it (no formed fold under it, or a closer human line took the fold)",
                          "added": "AI fold with no human line"}}
json.dump(doc, open(out_path, "w"), indent=2)
print(json.dumps({"frozen": doc["frozen_matcher_human_as_gold"], "added": added}))
for r in rows:
    loc = r.get("locus") or {}
    mv = r.get("matcher_vs_ai") or {}
    print(r["human_id"], r["status"], "->", r["ai_id"], "| off", mv.get("transverse_offset_mm"), "ang", mv.get("axis_angle_deg"),
          "gold_ov", mv.get("gold_span_overlap"), "pred_ov", mv.get("prediction_span_overlap"),
          "| along", loc.get("along_bisector_median_mm"), "across", loc.get("across_median_mm"),
          "| on-fold", loc.get("human_frac_on_ai_fold_span"), "covers", loc.get("ai_fold_frac_covered_by_human"),
          "overrun", loc.get("human_overrun_past_fold_mm"), "| loci dist", loc.get("dist_to_loci_mm"),
          "| nearest", [(n["ai_id"], n["transverse_offset_mm"]) for n in r["nearest_parallel_ai"]])
