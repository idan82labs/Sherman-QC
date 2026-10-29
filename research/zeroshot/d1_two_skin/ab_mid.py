#!/usr/bin/env python3
"""A/B of A0 on the D1 mid-surface cloud (and mid + gated outer-apex snap) against A0 on the scan.

Heavy (loads each original mesh for the snap): run through sherman_slot.sh.
Scoring: frozen research/broad_baseline/score.score_case with preloaded predictions. The mid-surface
cloud is in the raw scan frame (points only moved along their normals), so the prediction payload's
scan sha256 is set back to the original scan's before scoring (the SHA guard would otherwise refuse).

usage: ab_mid.py --a0mid DIR --scans d1/scans --out d1/ab_mid.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/broad_baseline"))
import score as frozen  # noqa: E402
from research.zeroshot.d1_two_skin import custody, two_skin as ts  # noqa: E402

RL = REPO / "docs/production/runlog"
SCORED_LIST = RL / "gold_set_v1/rescore/case_list_gold_v1__scored.json"
FRAME_RECORDS = RL / "broad_baseline/broad_frame_records.json"
RUNS = RL / "broad_baseline/runs/best_f1"


def snap(payload, mesh, t):
    out = []
    for r in payload.get("owned_bend_regions") or []:
        s, e = np.asarray(r["line_start"], float), np.asarray(r["line_end"], float)
        prof = ts.bend_profile(mesh.vertices, mesh.vertex_normals, s, e, t)
        r = dict(r)
        if prof.ok:
            cap = t / max(math.sin(math.radians(prof.included_angle_deg) / 2), 0.05) + 1.0
            ns, ne, info = ts.snap_line_to_outer_apex(s, e, prof, max_shift_mm=cap)
            r["line_start"], r["line_end"] = ns.tolist(), ne.tolist()
            r["d1_snap"] = info
        out.append(r)
    p = dict(payload)
    p["owned_bend_regions"] = out
    return p


def agg(rows):
    complete = [x for x in rows if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    tp, fp, fn = (sum(x[k] for x in complete) for k in ("true_positive", "false_positive", "false_negative"))
    fam = defaultdict(list)
    for x in complete:
        fam[x["family_id"]].append(x)
    ff = [frozen.prf(*(sum(x[k] for x in xs) for k in ("true_positive", "false_positive", "false_negative")))["f1"]
          for xs in fam.values()]
    offs = [m["transverse_offset_mm"] for x in complete for m in x.get("matches") or []]
    zero = [x for x in rows if x.get("disposition") == "ZERO_BEND_CONTROL"]
    return {"scans": len(complete), **{k: v for k, v in frozen.prf(tp, fp, fn).items() if not k.endswith("_6")},
            "family_macro_f1": round(float(np.mean(ff)), 6) if ff else None,
            "matched_offset_mm_median": round(statistics.median(offs), 4) if offs else None,
            "zero_bend_fp_lines": sum(x["prediction_count"] for x in zero),
            "per_scan": {x["case_id"]: [x["true_positive"], x["false_positive"], x["false_negative"]] for x in complete}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0mid", type=Path, required=True)
    ap.add_argument("--scans", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    scored = {c["case_id"]: c for c in json.loads(SCORED_LIST.read_text())["cases"]}
    frames = json.loads(FRAME_RECORDS.read_text())
    recs = {}
    for p in args.scans.glob("sc_*.json"):
        r = json.loads(p.read_text())
        if r.get("prediction_case_id"):
            recs[r["prediction_case_id"]] = r
    inv = custody.inventory_custody()
    arms = defaultdict(list)
    status = {}
    for pid, c in sorted(scored.items()):
        midp = args.a0mid / "mid" / pid / "predictions_a0.json"
        basep = RUNS / pid / "predictions_a0.json"
        if not midp.is_file() or not basep.is_file() or pid not in recs:
            status[pid] = "MISSING"
            continue
        base = json.loads(basep.read_text())
        mid = json.loads(midp.read_text())
        status[pid] = mid.get("status")
        mid["scan"] = dict(base.get("scan") or {})       # same raw frame; restore the scan identity
        r = recs[pid]
        mesh = ts.load_mesh(str(custody.check_scan(r["case_id"], r["scan_path"], inv)))
        t = r["thickness"]["t_mm"]
        variants = {"a0_scan": base, "a0_scan+snap": snap(base, mesh, t), "a0_mid": mid, "a0_mid+snap": snap(mid, mesh, t)}
        ctrl = args.a0mid / "orig" / pid / "predictions_a0.json"
        if ctrl.is_file():
            variants["a0_scan_rerun"] = json.loads(ctrl.read_text())
        case = {"case_id": pid, "tier": c["tier"], "family_id": c["family_id"], "gold": c["gold"],
                "label_complete": c.get("label_complete", True)}
        for name, payload in variants.items():
            arms[name].append(frozen.score_case(case, frame_records=frames, preloaded_predictions=payload))
    res = {"schema": "sherman.zeroshot.d1.ab_mid.v1", "lane_status": status,
           "arms": {k: agg(v) for k, v in arms.items()}}
    # determinism control: same cases only
    ctrl_ids = {x["case_id"] for x in arms.get("a0_scan_rerun", [])}
    if ctrl_ids:
        res["determinism_control"] = {
            "cases": sorted(ctrl_ids),
            "original": agg([x for x in arms["a0_scan"] if x["case_id"] in ctrl_ids]),
            "rerun": agg(arms["a0_scan_rerun"])}
        for k in ("original", "rerun"):
            res["determinism_control"][k].pop("per_scan", None)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["arms"].items():
        print(k, v["scans"], v["true_positive"], v["false_positive"], v["false_negative"], round(v["f1"], 4),
              v["family_macro_f1"], v["matched_offset_mm_median"], v["zero_bend_fp_lines"])
    if ctrl_ids:
        print("control", {k: (v["true_positive"], v["false_positive"], v["false_negative"]) for k, v in res["determinism_control"].items() if k != "cases"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
