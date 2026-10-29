"""Score the one-time N1 new-family run (newscans_N1/PREREG.json) once, with the frozen matcher through
research/zeroshot/zs_lane_v1/scoring.py (score_scan + aggregate, unchanged). Writes newscans_N1/RESULTS.json and
RESULTS_per_case.json. Zero-gold scans keep the frozen ZERO_BEND_CONTROL convention (FP lines, no F1); a display F1
(1.0 if 0 predicted lines else 0.0) is added for the owner page and labelled as such.
usage (repo root): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/tools/newscans_n1_score.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring  # noqa: E402
import bb_common as bb  # noqa: E402

P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
NB = P / "newscans_N1"
ARMS = {"zs_lane_v1:zs (primary)": NB / "runs/v1/zs", "zs_lane_v2:zs": NB / "runs/v2/zs",
        "zs_lane_v1:z3": NB / "runs/v1/z3", "zs_lane_v1:z2": NB / "runs/v1/z2"}
INCL = re.compile(r"incl ([0-9.]+) deg")
JOGGLE = {("47085000", "B25"), ("47085000", "B26")}   # gold joggle lines (adjudication); flagged, not excluded


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pred_meta(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    d = bb.load_eval_lines(path)
    if d.get("status") != "OK":
        return None
    return [r for r in d.get("owned_bend_regions") or [] if bb.valid_line(r.get("line_start"), r.get("line_end"))]


def main():
    pre = json.loads((NB / "PREREG.json").read_text())
    labels = pre["gold_of_record"]["labels"]
    for s, v in labels.items():
        assert sha(REPO / v["path"]) == v["sha256"], s
    out, per_case = {"schema": "sherman.zeroshot.newscans_results.v1", "claim_label": pre["claim_label"],
                     "prereg_sha256": sha(NB / "PREREG.json"), "arms": {}}, {}
    for arm, root in ARMS.items():
        recs, rows = [], []
        for sid, v in labels.items():
            spec = {"case_id": sid, "family_id": v["part_id"], "tier": "N1_NEW_FAMILY",
                    "gold": {"kind": "ai_studio", "path": v["path"]}, "label_complete": True}
            meta = pred_meta(root / sid / "eval_lines.json")
            lines = None if meta is None else [{"start": r["line_start"], "end": r["line_end"]} for r in meta]
            rec = scoring.score_scan(spec, lines)
            recs.append(rec)
            gold = json.loads((REPO / v["path"]).read_text())
            gincl = {b["bend_id"]: float(INCL.search(b["notes"]).group(1)) for b in gold["bends"]}
            run = json.loads((root / sid / "run_record.json").read_text()) if (root / sid / "run_record.json").is_file() else {}
            row = {"scan_id": sid, "part_id": v["part_id"], "disposition": rec["disposition"],
                   "gold": len(gold["bends"]), "pred": rec["n_pred"], "status_ok": meta is not None,
                   "wall_s": run.get("wall_s"), "peak_rss_gib": run.get("peak_rss_gib_process"),
                   "pins_ok": run.get("pins_ok")}
            if rec["disposition"] == "ZERO_BEND_CONTROL":
                row.update({"tp": 0, "fp": rec["false_positive_lines"], "fn": 0, "f1": None,
                            "f1_display": 1.0 if rec["false_positive_lines"] == 0 else 0.0,
                            "f1_display_rule": "zero-gold control: 1.0 iff 0 predicted lines (display only)",
                            "matches": [], "fp_ids": [f"P{i + 1:03d}" for i in range(rec["n_pred"])], "fn_ids": []})
            else:
                ms = []
                for m in rec["matches"]:
                    i = int(m["prediction_id"][1:]) - 1
                    e = meta[i].get("incl")
                    g = gincl.get(m["gold_id"])
                    ms.append({"gold_id": m["gold_id"], "pred_id": m["prediction_id"],
                               "pred_bend_id": meta[i]["bend_id"],
                               "offset_mm": round(m["transverse_offset_mm"], 3),
                               "axis_angle_deg": round(m["axis_angle_deg"], 3),
                               "span_overlap": [round(m["gold_span_overlap"], 3), round(m["prediction_span_overlap"], 3)],
                               "gold_incl_deg": g, "engine_incl_deg": None if e is None else round(float(e), 2),
                               "incl_diff_deg": None if e is None or g is None else round(float(e) - g, 2),
                               "joggle": (v["part_id"], m["gold_id"]) in JOGGLE})
                row.update({"tp": rec["true_positive"], "fp": rec["false_positive"], "fn": rec["false_negative"],
                            "P": round(rec["precision"], 4), "R": round(rec["recall"], 4), "f1": round(rec["f1"], 4),
                            "f1_display": round(rec["f1"], 4), "matches": ms,
                            "fp_ids": rec["unmatched_prediction_ids"], "fn_ids": rec["unmatched_gold_ids"],
                            "fn_causes": rec.get("fn_causes"), "fp_causes": rec.get("fp_causes")})
            rows.append(row)
        agg = scoring.aggregate(recs)
        ctrl_fp = sum(r["fp"] for r in rows if r["disposition"] == "ZERO_BEND_CONTROL")
        tp, fp, fn = agg["tp"], agg["fp"] + ctrl_fp, agg["fn"]
        p, r_, f = scoring.prf(tp, fp, fn)
        diffs = [m["incl_diff_deg"] for x in rows for m in x["matches"] if m["incl_diff_deg"] is not None]
        nj = [m["incl_diff_deg"] for x in rows for m in x["matches"] if m["incl_diff_deg"] is not None and not m["joggle"]]
        ad = sorted(abs(d) for d in nj)
        out["arms"][arm] = {
            "aggregate_frozen": agg,
            "micro_incl_control_fp": {"tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r_, 4),
                                      "micro_f1": round(f, 4)},
            "controls": {"scans": agg["controls"]["scans"], "fp_lines": ctrl_fp,
                         "clean": sum(1 for x in rows if x["disposition"] == "ZERO_BEND_CONTROL" and x["fp"] == 0)},
            "angle_agreement": {"n_matched": len(diffs), "n_non_joggle": len(nj),
                                "median_abs_non_joggle": round(ad[len(ad) // 2], 2) if ad else None,
                                "max_abs_non_joggle": round(ad[-1], 2) if ad else None,
                                "mean_signed_non_joggle": round(sum(nj) / len(nj), 2) if nj else None,
                                "within_1deg": sum(1 for d in ad if d <= 1.0), "within_2deg": sum(1 for d in ad if d <= 2.0)},
            "per_scan": rows}
        per_case[arm] = [{k: v for k, v in x.items() if k not in ("fn_rows", "fp_rows")} for x in recs]
    (NB / "RESULTS_per_case.json").write_text(json.dumps(per_case, indent=1, default=str) + "\n")
    out["per_case_sha256"] = sha(NB / "RESULTS_per_case.json")
    (NB / "RESULTS.json").write_text(json.dumps(out, indent=1) + "\n")
    print(sha(NB / "RESULTS.json"))
    for arm, a in out["arms"].items():
        print(arm, a["micro_incl_control_fp"], a["controls"], a["angle_agreement"])
        for x in a["per_scan"]:
            print("   ", x["part_id"], x["disposition"][:12], "gold", x["gold"], "pred", x["pred"], "tp/fp/fn",
                  x["tp"], x["fp"], x["fn"], "f1", x["f1"], "disp", x["f1_display"], "wall", x["wall_s"],
                  "rss", x["peak_rss_gib"], "pins", x["pins_ok"])


if __name__ == "__main__":
    main()
