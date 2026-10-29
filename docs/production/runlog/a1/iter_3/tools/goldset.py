"""Gold set v1 scoring of the E3 engine vs the best F1 lane on the SAME scans (A1 audience only).

Restricts GOLD_SET_V1 to scans that are A1 cases (cases_A1.json; directly or via the documented
STL->PLY same-mesh twin) and that the engine actually ran on; scores both lanes with the frozen broad
matcher (research/broad_baseline/score.score_case, apex locus, raw frame). Twins: the gold is drawn on the
PLY twin whose vertex set is identical to the STL scan (twin_vertex_check.json), so the scan-sha binding
check is skipped for those (recorded per case).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
import bb_common as bb  # noqa: E402

W = Path(sys.argv[1])
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
a1 = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"] if (c.get("broad") or {}).get("broad_case_id")}
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
lane_root = W / "analysis" / "lanes" / "e3"
ran = {p.name for p in lane_root.iterdir()} if lane_root.is_dir() else set()
e3ok = set()
for c in doc["cases"]:
    rp = W / "physical" / "E3" / c["case_ref"] / "record.json"
    if rp.is_file() and json.loads(rp.read_text()).get("status") == "OK":
        e3ok.add((c.get("broad") or {}).get("broad_case_id"))
rows = {"e3": [], "best_f1": []}
skipped = []
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    if pid not in a1:
        continue  # not an A1 case (may be DEV-CHECK): never scored here
    if not g.get("cad_path"):
        continue
    if pid not in e3ok:
        skipped.append({"case_id": g["case_id"], "a1_case": a1[pid]["case_ref"], "reason": "engine did not run (amendment-7 gate or blocked)"})
        continue
    twin = pid != g["case_id"]
    for lane, path in (("e3", lane_root / pid / "predictions_e3.json"),
                       ("best_f1", REPO / "docs/production/runlog/broad_baseline/runs/best_f1" / pid / "predictions_best_f1.json")):
        pred = bb.load_eval_lines(path) if path.is_file() else {"owned_bend_regions": [], "status": "PREDICTIONS_MISSING", "scan": {}}
        if twin:
            pred = dict(pred, scan={})
        r = sc.score_case({"case_id": g["case_id"], "tier": g.get("tier"), "family_id": g["family_id"], "gold": g["gold"],
                           "label_complete": g.get("label_complete", True)}, frame_records=None, preloaded_predictions=pred)
        r["twin_of"] = pid if twin else None
        rows[lane].append(r)


def agg(rs):
    comp = [r for r in rs if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    tp = sum(r["true_positive"] for r in comp); fp = sum(r["false_positive"] for r in comp); fn = sum(r["false_negative"] for r in comp)
    p = tp / (tp + fp) if tp + fp else 0.0; rr = tp / (tp + fn) if tp + fn else 0.0
    fam = defaultdict(lambda: [0, 0, 0])
    for r in comp:
        f = fam[r["family_id"]]; f[0] += r["true_positive"]; f[1] += r["false_positive"]; f[2] += r["false_negative"]
    def f1(a, b, c):
        p_ = a / (a + b) if a + b else 0.0; r_ = a / (a + c) if a + c else 0.0
        return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    ctrl = [r for r in rs if r.get("disposition") == "ZERO_BEND_CONTROL"]
    return {"scans": len(comp), "families": len(fam), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rr,
            "micro_f1": 2 * p * rr / (p + rr) if p + rr else 0.0,
            "family_macro_f1": sum(f1(*v) for v in fam.values()) / len(fam) if fam else None,
            "zero_bend_controls": len(ctrl), "control_fp_lines": sum(r.get("false_positive_lines", 0) for r in ctrl),
            "other_dispositions": sorted({r.get("disposition") for r in rs} - {"SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL"})}


out = {"definition": "GOLD_SET_V1 cases that are A1 cases with STEP and an OK engine run; same scans for both lanes",
       "e3": agg(rows["e3"]), "best_f1": agg(rows["best_f1"]), "skipped_a1_cases": skipped,
       "per_case": [{"case_id": a["case_id"], "family": a.get("family_id"), "twin_of": a.get("twin_of"),
                     "disposition": a.get("disposition"),
                     "e3": {k: a.get(k) for k in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")},
                     "best_f1": {k: b.get(k) for k in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}}
                    for a, b in zip(rows["e3"], rows["best_f1"])]}
(W / "analysis" / "goldset_scores.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("e3", "best_f1")}, indent=1), len(skipped), "skipped")
