"""Debug one DEV scan: rerun, overlay (gold green, predictions red), print per-line diagnosis.

usage: .venv/bin/python -m research.zeroshot.z2_self_recipe.debug sc_xxx [out.png]
"""
from __future__ import annotations

import json
import sys

from research.zeroshot.z2_self_recipe import common as C
from research.zeroshot.z2_self_recipe.run import run_scan
from research.zeroshot.z2_self_recipe import viz
import score as S
import score_broad as SB


def main(sid, out=None):
    inv = C.inventory()
    path = C.scan_path(sid, inv)
    payload, diag, scan = run_scan(path)
    g = inv[sid]["gold"]
    gold = S.load_gold({"kind": "ai_studio", "path": g["path"]}, frame_records=None, locus="apex")["lines"]
    pred = payload["owned_bend_regions"]
    rec = S.score_case({"case_id": sid, "family_id": inv[sid]["family"], "gold": {"kind": "ai_studio", "path": g["path"]},
                        "label_complete": True}, frame_records=None, preloaded_predictions={**payload, "scan": {}})
    dg = SB.diagnose(gold, pred, rec)
    print(sid, "t=%.2f" % scan.t, "flanges", len(scan.flanges), "regions", len(scan.regions), "gold", len(gold), "pred", len(pred),
          "tp", rec.get("true_positive"))
    for f in sorted(scan.flanges, key=lambda f: -f.area):
        print("  F%d area=%.0f two=%s w=%.1f L=%.1f n=%s" % (f.fid, f.area, f.two_sided, f.width, f.length, f.normal.round(2)))
    for r in dg["false_negatives"]:
        print("  FN", r)
    for r in dg["false_positives"]:
        print("  FP", r)
    for p in pred:
        print("  P", p["bend_id"], p["flanges"], "sw=%.1f span=%.1f cov=%.2f ro=%s st=%s" % (p["sweep_deg"], p["span_mm"], p["cover"], p["ro_mm"], p["stations"]))
    for m in rec.get("matches") or []:
        print("  M", m["gold_id"], m["prediction_id"], {k: round(v, 2) for k, v in m.items() if isinstance(v, float)})
    if out:
        viz.render(scan, out, gold=gold, pred=pred, title=sid)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
