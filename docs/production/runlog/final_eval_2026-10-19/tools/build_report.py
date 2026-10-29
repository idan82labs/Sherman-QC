"""Compose results.json (aggregate only; no sealed gold geometry) + headline numbers. usage: build_report.py W OUTDIR"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pb_common as pb  # noqa: E402

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
dc = json.loads((W / "devcheck_analysis.json").read_text())
fr = json.loads((W / "fresh_analysis.json").read_text())
card = json.loads((W / "devcheck_SEALED" / "physical_scorecard.json").read_text())
lm = card["BROAD"]["line_metric"]


def rows_of(blk):
    return [{"case": r["case_id"], "family": r["family"], "tp": r["true_positive"], "fp": r["false_positive"], "fn": r["false_negative"]}
            for r in blk["cases"] if r["disposition"] == "SCORED_COMPLETE_LABEL"]


dc_rows = {"engine": rows_of(lm["engine"]), "best_f1": rows_of(lm["reference_lanes"]["best_f1"])}
dc_ran = {k for k, v in dc["status"].items() if v is None}
fr_ran = {k for k, v in fr["status"].items() if v is None}


def block(rows, ran):
    sub = [r for r in rows if r["case"] in ran]
    return {"all": {**pb.line_stats(rows), "bootstrap": pb.bootstrap_lines(rows)},
            "engine_ran": {**pb.line_stats(sub), "bootstrap": pb.bootstrap_lines(sub) if sub else None}}


line = {}
for arm in ("engine", "best_f1"):
    line[arm] = {"DEVCHECK": block(dc_rows[arm], dc_ran), "FRESH": block(fr["_rows"][arm], fr_ran),
                 "POOL": block(dc_rows[arm] + fr["_rows"][arm], dc_ran | fr_ran)}


def bend_pool():
    fam = {}
    for src in (dc["per_bend"]["per_family"], fr["per_bend"]["per_family"]):
        for f, v in src.items():
            fam.setdefault(f, [0, 0, 0]); fam[f][0] += v["bends"]; fam[f][1] += v["correct_riser_aware"]
    return fam


pool_fam = bend_pool()
pb_pool = {"headline_with_misses": {
    "bends": dc["per_bend"]["headline_with_misses"]["bends"] + fr["per_bend"]["headline_with_misses"]["bends"],
    "correct_riser_aware": dc["per_bend"]["headline_with_misses"]["correct_riser_aware"] + fr["per_bend"]["headline_with_misses"]["correct_riser_aware"],
    "correct_strict": dc["per_bend"]["headline_with_misses"]["correct_strict"] + fr["per_bend"]["headline_with_misses"]["correct_strict"]},
    "engine_ran": {"bends": dc["per_bend"]["engine_ran"]["bends"] + fr["per_bend"]["engine_ran"]["bends"],
                   "correct_riser_aware": dc["per_bend"]["engine_ran"]["correct_riser_aware"] + fr["per_bend"]["engine_ran"]["correct_riser_aware"],
                   "correct_strict": dc["per_bend"]["engine_ran"]["correct_strict"] + fr["per_bend"]["engine_ran"]["correct_strict"]},
    "false_pass_proxy": dc["per_bend"]["false_pass_proxy"] + fr["per_bend"]["false_pass_proxy"],
    "bootstrap_headline": pb.bootstrap_bends(pool_fam)}
for k in ("headline_with_misses", "engine_ran"):
    v = pb_pool[k]
    v["rate_riser_aware"] = v["correct_riser_aware"] / v["bends"] if v["bends"] else None
    v["rate_strict"] = v["correct_strict"] / v["bends"] if v["bends"] else None
fr_pub = {k: v for k, v in fr.items() if not k.startswith("_")}
res = {"schema": "sherman_final_eval_2026-10-19_v1", "line_f1": line,
       "per_bend": {"DEVCHECK": dc["per_bend"], "FRESH": fr["per_bend"], "POOL": pb_pool},
       "devcheck": {k: v for k, v in dc.items() if k not in ("line_f1", "per_bend")}, "fresh": {k: v for k, v in fr_pub.items() if k not in ("line_f1", "per_bend")},
       "devcheck_line_detail": dc["line_f1"], "fresh_line_detail": fr["line_f1"]}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
f = lambda x: "-" if x is None else f"{x:.3f}"
ci = lambda b, k: "-" if not b else f"[{b[k][0]:.2f}, {b[k][1]:.2f}]"
for view in ("all", "engine_ran"):
    print(f"## view={view}")
    for s in ("DEVCHECK", "FRESH", "POOL"):
        for arm in ("engine", "best_f1"):
            x = line[arm][s][view]
            print(f"{s:9s} {arm:8s} scans={x['cases']:2d} fam={x['families']:2d} micro={f(x['micro_f1'])} {ci(x['bootstrap'], 'micro_f1_95ci')} "
                  f"macro={f(x['family_macro_f1'])} {ci(x['bootstrap'], 'family_macro_f1_95ci')} tp/fp/fn={x['tp']}/{x['fp']}/{x['fn']}")
for s, v in (("DEVCHECK", dc["per_bend"]), ("FRESH", fr["per_bend"]), ("POOL", pb_pool)):
    h, e = v["headline_with_misses"], v["engine_ran"]
    print(s, "per-bend headline riser", h["correct_riser_aware"], "/", h["bends"], f(h["rate_riser_aware"]), "strict", f(h["rate_strict"]),
          ci(v["bootstrap_headline"], "rate_riser_aware_95ci"), "| ran", e["correct_riser_aware"], "/", e["bends"], f(e["rate_riser_aware"]),
          "strict", f(e["rate_strict"]), "| falsePASS", v["false_pass_proxy"])
