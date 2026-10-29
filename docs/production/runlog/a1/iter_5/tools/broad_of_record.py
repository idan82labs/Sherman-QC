"""BROAD scans scored three ways (report-only; frozen matcher, raw frame, apex locus), engine vs best F1 lane on
the SAME scans: (1) frozen BROAD gold; (2) the Gold11 relabel substituted on the 9 Gold11 scans (owner decision
2026-10-17; the frozen BROAD gold elsewhere); (3) gold of record per scan: relabel on Gold11 scans, GOLD_SET_V1
where the scan is in it, else the frozen BROAD gold. usage: broad_of_record.py PHYS_DIR OUT_JSON"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
from domains.production.engine.eval.line_metric import match_lines  # noqa: E402

W = Path(sys.argv[1])
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
lanes = json.loads((FZ / "broad_reference_lanes_A1.json").read_text())["lanes"]
relabel = {g["case_id"]: g["gold_lines"] for g in doc["relabel_gold_cases"]}
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
gs = {}
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    try:
        gs[pid] = sc.load_gold(g["gold"], frame_records=None)["lines"]
    except Exception:  # noqa: BLE001
        pass


def raw_gold(g):
    gold = g["gold_lines"]
    if g.get("raw_to_reference") is not None and not np.allclose(np.asarray(g["raw_to_reference"]), np.eye(4)):
        T = np.linalg.inv(np.asarray(g["raw_to_reference"]))
        gold = [{"bend_id": x["bend_id"], "line_start": (T @ np.r_[x["line_start"], 1])[:3].tolist(),
                 "line_end": (T @ np.r_[x["line_end"], 1])[:3].tolist()} for x in gold]
    return gold


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(variant, lane):
    fam = defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]
    rows = []
    for g in doc["broad_gold_cases"]:
        if g.get("zero_bend_control"):
            continue
        cid, bid = g["case_id"], g["broad_case_id"]
        if variant == "frozen":
            gold, src = raw_gold(g), "BROAD"
        elif variant == "relabel_substituted":
            gold, src = (relabel[cid], "RELABEL") if cid in relabel else (raw_gold(g), "BROAD")
        else:
            if cid in relabel:
                gold, src = relabel[cid], "RELABEL"
            elif bid in gs:
                gold, src = gs[bid], "GOLD_SET_V1"
            else:
                gold, src = raw_gold(g), "BROAD"
        if lane == "e3":
            p = W / "analysis" / "lanes" / "e3" / bid / "predictions_e3.json"
            preds = json.loads(p.read_text())["owned_bend_regions"] if p.is_file() else []
        else:
            preds = (lanes["best_f1"].get(cid) or {}).get("lines") or []
        m = match_lines(gold, preds)
        f = fam[g["family"]]
        for i, k in enumerate(("true_positive", "false_positive", "false_negative")):
            f[i] += m[k]
            tot[i] += m[k]
        rows.append({"case_id": cid, "family": g["family"], "gold_src": src, "gold": len(gold), "tp": m["true_positive"],
                     "fp": m["false_positive"], "fn": m["false_negative"]})
    p, r, f1 = prf(*tot)
    macro = float(np.mean([prf(*v)[2] for v in fam.values()]))
    return {"tp": tot[0], "fp": tot[1], "fn": tot[2], "precision": round(p, 4), "recall": round(r, 4), "micro_f1": round(f1, 4),
            "family_macro_f1": round(macro, 4), "families": len(fam), "line_cases": len(rows), "rows": rows}


out = {v: {lane: score(v, lane) for lane in ("e3", "best_f1")} for v in ("frozen", "relabel_substituted", "gold_of_record")}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for v in out:
    for lane in out[v]:
        print(v, lane, {k: x for k, x in out[v][lane].items() if k != "rows"})
