"""Score direct E3 runs (runs/<tag>/<case>) with the frozen eval code (read-only): BROAD gold + G2a human/relabel.

usage: score_runs.py TAG [TAG0 for comparison]   (iter-0 outputs are 'iter0')
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]

from domains.production.engine.eval.engine_output import load_compiled, load_engine_output  # noqa: E402
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
from domains.production.engine.eval.freeze.physical import _gold_case_objects  # noqa: E402
from domains.production.engine.eval.line_metric import engine_lines, match_lines, prepare_gold  # noqa: E402

S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
P0 = S / "a1_iter0/phys"
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
lanes = json.loads((FZ / "broad_reference_lanes_A1.json").read_text())["lanes"]


def run_dir(tag: str, cid: str) -> Path:
    if tag == "iter0":
        return P0 / "physical/E3" / cid / "engine"
    return S / "a1_iter1/runs" / tag / cid


def preds_for(tag: str, cid: str):
    d = run_dir(tag, cid)
    if not (d / "result.json").is_file():
        return None, None
    res = json.loads((d / "result.json").read_text())
    compiled = load_compiled(P0 / "compiled/dev" / res["sku"] / "compiled_part.json")
    eo = load_engine_output(d)
    if eo.result is None:
        return None, res
    return engine_lines(eo.result, compiled, locus="OUTER_APEX", frame="raw"), res


broad_gold = {g["case_id"]: g for g in doc["broad_gold_cases"]}
usable_b, _ = prepare_gold(_gold_case_objects(list(broad_gold.values())))
g2h = {g["case_id"]: g for g in doc["gold_cases"]}
g2r_list, _ = relabel_gold_cases(doc["gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
g2r = {g["case_id"]: g for g in g2r_list}
usable_h, _ = prepare_gold(_gold_case_objects(list(g2h.values())))
usable_r, _ = prepare_gold(_gold_case_objects(list(g2r.values())))

tags = sys.argv[1:]
tot = {t: {"b": [0, 0, 0], "h": [0, 0, 0], "r": [0, 0, 0]} for t in tags + ["best_f1"]}
rows = []
for cid in sorted(cases):
    have = [t for t in tags if (run_dir(t, cid) / "result.json").is_file()]
    if len(have) < len(tags):
        continue
    line = [cid, cases[cid]["family"]]
    for t in tags:
        preds, res = preds_for(t, cid)
        val = (res or {}).get("inspection_validity", {}).get("state")
        parts = []
        for key, usable in (("b", usable_b), ("h", usable_h), ("r", usable_r)):
            if cid not in usable:
                continue
            m = match_lines(usable[cid], preds or [])
            tot[t][key][0] += m["true_positive"]
            tot[t][key][1] += m["false_positive"]
            tot[t][key][2] += m["false_negative"]
            parts.append(f"{key}:{m['true_positive']}/{m['gold_count']}(fp{m['false_positive']})")
        line.append(f"{t}[{val} n={len(preds or [])}] " + " ".join(parts))
    bl = (lanes["best_f1"].get(cid) or {}).get("lines") or []
    parts = []
    for key, usable in (("b", usable_b), ("h", usable_h), ("r", usable_r)):
        if cid not in usable:
            continue
        m = match_lines(usable[cid], bl)
        tot["best_f1"][key][0] += m["true_positive"]
        tot["best_f1"][key][1] += m["false_positive"]
        tot["best_f1"][key][2] += m["false_negative"]
        parts.append(f"{key}:{m['true_positive']}/{m['gold_count']}(fp{m['false_positive']})")
    line.append("best_f1 " + " ".join(parts))
    print(" | ".join(line))


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


for t, d in tot.items():
    print(t, {k: (v, round(f1(*v), 3)) for k, v in d.items()})
