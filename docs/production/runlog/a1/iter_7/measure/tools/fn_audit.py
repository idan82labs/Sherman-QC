"""BROAD FN/FP audit (report-only): every BROAD gold line the engine misses, and every engine line BROAD calls
false, compared with the gold of record on the same scan (Gold11 relabel v1 on the 9 Gold11 scans, GOLD_SET_V1
elsewhere). Uses the frozen matcher read-only. usage: fn_audit.py PHYS_DIR OUT_JSON"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
from domains.production.engine.eval.line_metric import match_lines, pair_metrics, is_match  # noqa: E402

W = Path(sys.argv[1])
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
relabel = {g["case_id"]: g for g in doc["relabel_gold_cases"]}
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
gs = {}
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    try:
        gold = sc.load_gold(g["gold"], frame_records=None)
        gs[pid] = {"lines": gold["lines"], "case_id": g["case_id"], "zero": gold.get("zero_bend_control")}
    except Exception as e:  # noqa: BLE001
        gs[pid] = {"error": repr(e)}


def L(r):
    return np.asarray([r["line_start"], r["line_end"]], dtype=float)


def best(line, others):
    out = []
    for o in others:
        m = pair_metrics(L(line), L(o))
        out.append((m["axis_angle_deg"] / 5 + m["transverse_offset_mm"] / 4 + (1 - min(1, m["gold_span_overlap"]))
                    + (1 - min(1, m["prediction_span_overlap"])), o["bend_id"], m))
    out.sort(key=lambda x: x[0])
    return out[:2]


rows = []
for g in doc["broad_gold_cases"]:
    cid = g["case_id"]
    bid = g["broad_case_id"]
    pp = W / "analysis" / "lanes" / "e3" / bid / "predictions_e3.json"
    preds = json.loads(pp.read_text())["owned_bend_regions"] if pp.is_file() else []
    if g.get("zero_bend_control"):
        rows.append({"case_id": cid, "zero": True, "preds": len(preds)})
        continue
    gold = g["gold_lines"]
    if g.get("raw_to_reference") is not None and not np.allclose(np.asarray(g["raw_to_reference"]), np.eye(4)):
        T = np.linalg.inv(np.asarray(g["raw_to_reference"]))
        gold = [{"bend_id": x["bend_id"], "line_start": (T @ np.r_[x["line_start"], 1])[:3].tolist(),
                 "line_end": (T @ np.r_[x["line_end"], 1])[:3].tolist()} for x in gold]
    m = match_lines(gold, preds)
    if cid in relabel:
        rec, rec_name = relabel[cid]["gold_lines"], "RELABEL"
    elif bid in gs and "lines" in gs[bid]:
        rec, rec_name = gs[bid]["lines"], "GOLD_SET_V1:" + gs[bid]["case_id"]
    else:
        rec, rec_name = None, "NONE"
    mr = match_lines(rec, preds) if rec is not None else None
    eng_rec = {x["prediction_id"]: x["gold_id"] for x in (mr or {}).get("matches", [])}
    fns = []
    for gid in m["unmatched_gold_ids"]:
        line = next(x for x in gold if str(x["bend_id"]) == gid)
        b = best(line, preds)
        r = best(line, rec) if rec else []
        rec_match = [x[1] for x in r if is_match(x[2])]
        fns.append({"gold_id": gid, "len_mm": round(float(np.linalg.norm(L(line)[1] - L(line)[0])), 1),
                    "nearest_engine": [{"id": x[1], **{k: round(v, 2) for k, v in x[2].items() if k in
                                       ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap", "prediction_length_mm")}} for x in b],
                    "matches_record_gold": rec_match,
                    "nearest_record": [{"id": x[1], **{k: round(v, 2) for k, v in x[2].items() if k in
                                       ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap", "prediction_length_mm")}} for x in r]})
    fps = []
    for pid in m["unmatched_prediction_ids"]:
        fps.append({"pred_id": pid, "matches_record_gold": eng_rec.get(pid)})
    rows.append({"case_id": cid, "bid": bid, "family": g["family"], "tier": g["tier"], "record": rec_name,
                 "broad_gold": len(gold), "preds": len(preds), "tp": m["true_positive"], "fn": len(fns), "fp": len(fps),
                 "record_gold": None if rec is None else len(rec),
                 "record_tp": None if mr is None else mr["true_positive"],
                 "record_fn_ids": None if mr is None else mr["unmatched_gold_ids"],
                 "fn_lines": fns, "fp_lines": fps})
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
tot = {"fn": 0, "fn_in_record": 0, "fn_not_in_record": 0, "fp": 0, "fp_in_record": 0}
for r in rows:
    if r.get("zero"):
        continue
    for f in r["fn_lines"]:
        tot["fn"] += 1
        tot["fn_in_record" if f["matches_record_gold"] else "fn_not_in_record"] += 1
    for f in r["fp_lines"]:
        tot["fp"] += 1
        tot["fp_in_record"] += 1 if f["matches_record_gold"] else 0
print(json.dumps(tot))
for r in rows:
    if r.get("zero"):
        continue
    print(r["case_id"], r["family"], r["tier"], r["record"], "BROAD", r["broad_gold"], "pred", r["preds"], "tp", r["tp"],
          "| record", r["record_gold"], "rec_tp", r["record_tp"], "rec_fn", r["record_fn_ids"])
    for f in r["fn_lines"]:
        print("   FN", f["gold_id"], f["len_mm"], "in_record:", f["matches_record_gold"], "near_eng:", f["nearest_engine"][:1], "near_rec:", f["nearest_record"][:1])
    for f in r["fp_lines"]:
        print("   FP", f["pred_id"], "record:", f["matches_record_gold"])
