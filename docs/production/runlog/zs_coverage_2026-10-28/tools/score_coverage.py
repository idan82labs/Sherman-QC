"""Tiered scoring of the zs coverage run (2026-10-28). Frozen matcher (research/broad_baseline/score.py score_case via
research/zeroshot/zs_lane_v1/scoring.py: 5 deg / 4 mm / 0.6 / 0.6, apex, bb_common) and family bootstrap
(n 10000, seed 20261018). Tiers are never blended.

T1 held-out by engine: v1 = the existing frozen one-time per-case records (TEST-B testb/RESULTS_per_case.json
   'zs|testb'; N1 newscans_N1/RESULTS_per_case.json primary), re-aggregated with seed 20261018 (no gold opened).
   v3: none (developed after TEST-B / N1).
T2 post-hoc descriptive, both lanes, coverage outputs vs frozen gold. Gold under a *_SEALED* path is NOT opened
   unless --allow-sealed is given (coordinator authorization); those scans are listed as SEALED_GOLD_NOT_OPENED.
T3 ran-on coverage: status, success rate, wall P50/P95, peak RSS, lines/scan (all 192, labelled or not).
usage: score_coverage.py [--allow-sealed]
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring as SC  # noqa: E402

D = REPO / "docs/production/runlog/zs_coverage_2026-10-28"
ZS = REPO / "docs/production/runlog/zeroshot_2026-10-25"
N, SEED = 10000, 20261018
LANES = ["zs_lane_v1", "zs_lane_v3"]
ALLOW_SEALED = "--allow-sealed" in sys.argv


def agg(recs):
    return SC.aggregate(recs, ci=True, n_boot=N, seed=SEED)


def slim(a):
    keys = ["scans", "families", "gold", "pred", "tp", "fp", "fn", "P", "R", "micro_f1", "micro_ci95", "family_macro_f1",
            "macro_ci95", "failed_or_missing_scans", "controls", "fn_causes", "fp_causes"]
    return {k: a.get(k) for k in keys}


def t1():
    out = {}
    tb = json.loads((ZS / "testb/RESULTS_per_case.json").read_text())["zs|testb"]
    out["TEST-B (61, re-audit gold; one-time 2026-10-25, ledger seq 13-14)"] = slim(agg(tb))
    n1 = json.loads((ZS / "newscans_N1/RESULTS_per_case.json").read_text())["zs_lane_v1:zs (primary)"]
    n1recs = n1 if isinstance(n1, list) else n1.get("per_case") or n1.get("records")
    out["N1 (4: 1 line scan + 3 zero-bend controls; one-time 2026-10-25, ledger seq 18-19)"] = slim(agg(n1recs))
    out["TEST-B + N1 pooled (families disjoint)"] = slim(agg(tb + n1recs))
    return out


def lines_of(lane_rec):
    if not lane_rec or lane_rec.get("status") != "OK":
        return None
    return [{"start": l["start_mm"], "end": l["end_mm"]} for l in lane_rec["lines"]]


def main():
    cov = {r.get("scan_id"): r for r in json.loads((D / "COV_INVENTORY.json").read_text())["rows"]}
    man = json.loads((D / "RUN_MANIFEST.json").read_text())["scans"]
    led = SC._ledger_frozen()
    inv = SC._inv()
    runs = {}
    for s in man:
        p = D / "runs" / f"{s['scan_id']}.json"
        runs[s["scan_id"]] = json.loads(p.read_text()) if p.exists() else None

    # ---- T3
    t3 = {}
    for lane in LANES:
        st, walls, rss, nl, fails = Counter(), [], [], [], []
        for s in man:
            r = runs[s["scan_id"]]
            lr = (r or {}).get("lanes", {}).get(lane)
            if lr is None:
                st["NOT_RUN_YET"] += 1; continue
            st[lr["status"]] += 1
            if lr.get("wall_s") is not None:
                walls.append(lr["wall_s"]); rss.append(lr["peak_rss_gib"])
            if lr["status"] == "OK":
                nl.append(len(lr["lines"]))
            else:
                fails.append({"scan_id": s["scan_id"], "part_id": s["part_id"], "split": s["split"], "status": lr["status"],
                              "reason": json.dumps(lr.get("errors"))[:300], "rc": lr.get("driver_rc")})
        q = lambda a, p: round(float(np.percentile(a, p)), 2) if a else None  # noqa: E731
        t3[lane] = {"scans_in_manifest": len(man), "statuses": dict(st), "success_rate": round(st["OK"] / len(man), 4),
                    "wall_s": {"p50": q(walls, 50), "p95": q(walls, 95), "max": max(walls) if walls else None, "n": len(walls)},
                    "peak_rss_gib": {"p50": q(rss, 50), "p95": q(rss, 95), "max": max(rss) if rss else None},
                    "lines_per_ok_scan": {"mean": round(float(np.mean(nl)), 2) if nl else None, "p50": q(nl, 50),
                                          "max": max(nl) if nl else None, "total": int(sum(nl)), "zero_line_scans": nl.count(0)},
                    "by_split_ok": dict(Counter(s["split"] for s in man if ((runs[s["scan_id"]] or {}).get("lanes", {}).get(lane) or {}).get("status") == "OK")),
                    "failures": fails}

    # ---- T2
    specs, sealed, nogold = {}, [], []
    dev = SC.case_specs("dev")
    for s in man:
        sid = s["scan_id"]
        if sid in dev:
            spec = dict(dev[sid]); grp = "DEV/" + cov[sid]["sub"].split(";")[0]
        elif s["split"] == "TEST-B":
            e = led[sid]
            spec = {"case_id": sid, "family_id": inv[sid]["family"], "tier": "TESTB_POSTHOC",
                    "gold": {"kind": "ai_studio", "path": e["label_path"]}, "label_complete": True}
            grp = "TEST-B"
        elif s["split"] == "N1":
            e = led[sid]
            spec = {"case_id": sid, "family_id": e["part_id"], "tier": "N1_POSTHOC",
                    "gold": {"kind": "ai_studio", "path": e["label_path"]}, "label_complete": True}
            grp = "N1"
        else:
            nogold.append(sid); continue
        if "_SEALED" in str(spec["gold"]["path"]) and not ALLOW_SEALED:
            sealed.append({"scan_id": sid, "group": grp}); continue
        specs[sid] = (spec, grp)
    t2 = {}
    for lane in LANES:
        recs = defaultdict(list)
        for sid, (spec, grp) in specs.items():
            lr = ((runs[sid] or {}).get("lanes") or {}).get(lane)
            if lr is None:
                continue
            rec = SC.score_scan(spec, lines_of(lr), diagnose=True)
            rec["group"] = grp
            recs[grp].append(rec); recs["ALL_SCORED"].append(rec)
            if grp.startswith("DEV/"):
                recs["DEV (all scored)"].append(rec)
        t2[lane] = {g: slim(agg(v)) for g, v in sorted(recs.items())}
        (D / f"scores_T2_{lane}_per_case.json").write_text(json.dumps(
            {g: [{k: x.get(k) for k in ("case_id", "family_id", "disposition", "gold_count", "prediction_count", "true_positive",
                                       "false_positive", "false_negative", "false_positive_lines", "output_missing_or_failed")}
                 for x in v] for g, v in recs.items()}, indent=1))
    doc = {"schema": "sherman.zs_coverage.scores.v1", "matcher": "frozen 5 deg / 4 mm / 0.6 / 0.6 apex (broad_baseline score_case)",
           "bootstrap": {"n": N, "seed": SEED, "unit": "family (part_id)"},
           "T1_heldout_by_engine": {"zs_lane_v1": t1(), "zs_lane_v3": "none: v3 was developed after TEST-B and N1 (FE1 adds it later)"},
           "T2_posthoc_gold_labelled": {"label": "POST-HOC on consumed data; DEV = in-sample for both lanes (production GBMs fit on all 125 DEV); not held-out",
                                        "allow_sealed": ALLOW_SEALED, "lanes": t2,
                                        "sealed_gold_not_opened": {"n": len(sealed), "by_group": dict(Counter(x["group"] for x in sealed)), "scan_ids": [x["scan_id"] for x in sealed]},
                                        "no_frozen_gold": nogold},
           "T3_ran_on_coverage": t3}
    (D / ("SCORES_allow_sealed.json" if ALLOW_SEALED else "SCORES.json")).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"T1": {k: (v["micro_f1"], v["micro_ci95"], v["family_macro_f1"], v["macro_ci95"]) for k, v in doc["T1_heldout_by_engine"]["zs_lane_v1"].items()},
                      "T2": {l: {g: (v["scans"], v["families"], v["micro_f1"], v["micro_ci95"], v["family_macro_f1"], v["macro_ci95"], v["controls"]) for g, v in t2[l].items()} for l in LANES},
                      "sealed": doc["T2_posthoc_gold_labelled"]["sealed_gold_not_opened"]["by_group"],
                      "T3": {l: {k: t3[l][k] for k in ("statuses", "success_rate", "wall_s", "peak_rss_gib", "lines_per_ok_scan")} for l in LANES}}, indent=1))


if __name__ == "__main__":
    main()
