"""Assemble docs/production/runlog/a1/iter_5/scorecard.json from the iter-4 MEASURE outputs (report-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_5")
RAW = R / "measure" / "raw"
LOGS = R / "measure" / "logs"
j = lambda p: json.loads(Path(p).read_text())  # noqa: E731
phys = j(RAW / "physical_scorecard.json")
sub = j(RAW / "subset_scores.json")
gs = j(RAW / "goldset_scores.json")
pbx = j(RAW / "per_bend_ext.json")["summary"]
g1 = j(RAW / "g1_regression_scorecard.json"); g1["gates"] = {g["id"]: g for g in g1["gates"]}
g5 = j(RAW / "g5_report.json")
idx = j(R / "overlays" / "worst20" / "INDEX.json")
prev = j(R.parent / "iter_4" / "scorecard.json")
three = j(RAW / "broad_three_golds.json")
audit = j(RAW / "broad_fn_fp_audit.json")
pba = j(RAW / "per_bend_all.json")["summary"]
fp_start = j(LOGS / "fingerprints_start.json")
fp_end = j(LOGS / "fingerprints_end.json")
extra = j(LOGS / "extra.json")  # rss / runtime / g5 summary / text fields computed by the operator
G2 = phys["G2"]
B = phys["BROAD"]


def brief(s):
    return {"true_positive": s["true_positive"], "false_positive": s["false_positive"], "false_negative": s["false_negative"],
            "precision": s["micro"]["precision"], "recall": s["micro"]["recall"], "micro_f1": s["micro"]["f1"],
            "family_macro_f1": s["family_macro_f1"], "line_cases": s["line_cases"], "families": s["families"]}


def strip(d):
    if not isinstance(d, dict):
        return None
    return {k: v for k, v in d.items() if k not in ("per_case", "per_part", "cases", "per_family", "per_set")}


pb = sub["per_bend_given_cad"]
card = {
    "schema": "sherman_a1_iteration_scorecard_v1",
    "iteration": 5,
    "phase": "MEASURE",
    "created": "2026-10-18",
    "label": extra["label"],
    "comparator_policy": "bar = best F1 lane (old research lane: 0.458 broad-57, 0.510 Gold11 relabel) on the SAME scans; the "
                         "shipped runtime engine is discarded as a comparator; the legacy control arm appears only as "
                         "informational G2 context.",
    "engine": {
        "engine_version": extra["engine_version"], "config": "default (no --config)",
        "config_sha256": extra["config_sha256"],
        "runner_fingerprints_G1": g1["provenance"]["fingerprints"],
        "freeze_style_fingerprints_start": fp_start, "freeze_style_fingerprints_end": fp_end,
        "code_unchanged_during_measure": fp_start == fp_end,
        "matches_change_step": extra["matches_change_step"],
    },
    "ledger": {"anchor": "cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67",
               "ledger_check_ok": phys["ledger_check"]["ok"], "anchor_checked": phys["ledger_check"]["anchor_checked"],
               "g1_ledger_check_ok": g1["provenance"]["ledger_check"]["ok"], "physical_final": phys["final"],
               "physical_qualifying": phys["qualifying"], "g1_qualifying": g1["qualifying"],
               "g1_not_qualifying_reasons": g1["not_qualifying_reasons"]},
    "G1_synthetic_regression_DEV": {
        "tier": "regression DEV, 50 cases / 7 families, e2-synthetic-1.1.1 (iter-0 generated tier reused; generator unchanged), scanner model card (PARTIAL)",
        "verdict": g1["verdict"], "regression_verdict": g1["regression_verdict"], "failed_gates": g1["failed_gates"],
        "gates": {k: {"verdict": v.get("verdict"), "value": v.get("value"), "threshold": v.get("threshold"), "n": v.get("n")}
                  for k, v in g1["gates"].items()},
        "vs_iter4": extra["g1_vs_iter4"],
        "note": "G1.SUITE FAIL is by construction on the regression tier (needs >= 200 cases); runtime gates need the runtime tier",
    },
    "G2": {
        "verdict": G2["verdict"], "final": False, "gates": G2["gates"],
        "a_line_metric": {
            "gate_of_record": "gold11_relabel_v1 (owner 2026-10-17, OWNER_DECISION_2026-10-17.md); human Gold11 = historical diagnostic; both reported",
            "thresholds": {"f1_min": 0.8, "precision_min": 0.9, "matcher": "5 deg / 4 mm / 0.6 span, one-to-one, apex locus, raw frame"},
            "all_9_A1_gold11_cases": sub["G2a"]["all9"],
            "same_scans_engine_ran": sub["G2a"]["same_scans_engine_ran"],
            "legacy_arm_informational": {"human": {k: G2["line_metric"]["legacy"][k] for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative")},
                                         "relabel": {k: G2["line_metric_dual_report"]["relabel_report"]["legacy"][k] for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative")}},
            "relabel_changed_since_freeze": G2["line_metric_dual_report"]["relabel_changed_since_freeze"],
            "relabel_snapshot_sha256": sub["G2a"]["relabel_snapshot_now"],
            "verdict_relabel": G2["line_metric_dual_report"]["relabel"]["verdict"],
            "verdict_human": G2["line_metric_dual_report"]["human_all_cases"]["verdict"],
            "why_fail": extra["g2a_why_fail"],
        },
        "b_coverage": {k: strip(v) for k, v in G2["coverage"].items() if isinstance(v, dict)},
        "c_angle_known_good": strip(G2["angle_known_good"]),
        "d_flange_known_good": G2["flange_known_good"],
        "e_runtime": G2["runtime"],
        "f_repeatability": strip(G2["repeatability"]),
        "compile_rate": {"rate": G2["compile_rate"]["dev"]["rate"], "gate": G2["compile_rate"]["gate"]},
        "population_note": extra["g2_population_note"],
    },
    "G3": phys["G3"],
    "BROAD": {
        "bar": "best F1 lane 0.458 on broad-57 (old definition; the engine runs only on A1 cases); on the frozen A1 BROAD gold the same lane is reported side by side below",
        "verdict": B["verdict"], "target_status": B["line_metric"]["target_status"],
        "all_26_cases_frozen_scorer": {"engine": brief(sub["BROAD_all26"]["engine"]), "best_f1": brief(sub["BROAD_all26"]["best_f1"])},
        "same_scans_engine_ran": {"cases": sub["BROAD_same_scans_engine_ran"]["cases"],
                                  "engine": brief(sub["BROAD_same_scans_engine_ran"]["engine"]),
                                  "best_f1": brief(sub["BROAD_same_scans_engine_ran"]["best_f1"])},
        "per_family_f1": {"engine": {k: v.get("f1") for k, v in B["line_metric"]["engine"]["per_family"].items()},
                          "best_f1": {k: v.get("f1") for k, v in B["line_metric"]["reference_lanes"]["best_f1"]["per_family"].items()}},
        "per_case": sub["BROAD_per_case"],
        "recipe_ceiling_family_macro": B["line_metric"]["recipe_ceiling"]["family_macro_f1_ceiling"],
        "coverage_complete_broad": {k: strip(v) for k, v in B["coverage"].items() if isinstance(v, dict)},
        "engine_bend_states": B["engine_bend_states"], "characteristic_decisions": B["characteristic_decisions"],
        "staged_states": B["staged_states"],
        "runtime": {k: B["runtime"][k] for k in ("p95_wall_s", "max_wall_s", "runs", "crashes")},
        "fp_analysis": extra["fp_analysis"],
        "per_bend_given_cad": {**pb, "extended": pbx, "false_pass_definitions": extra["false_pass_definitions"]},
        "per_bend_given_cad_all_A1": {"definition": extra["per_bend_all_definition"], **pba},
    },
    "BROAD_three_golds_report_only": {
        "definition": "same 24 BROAD line cases, engine vs best F1 lane on the SAME scans, frozen matcher (5 deg / 4 mm / 0.6, one-to-one, apex locus, raw frame): frozen = frozen BROAD gold; relabel_substituted = gold11_relabel_v1 on the 9 Gold11 scans (owner decision 2026-10-17), BROAD elsewhere; gold_of_record = relabel on Gold11 scans, GOLD_SET_V1 where the scan is in it, else BROAD",
        **{v: {lane: {k: x for k, x in three[v][lane].items() if k != "rows"} for lane in three[v]} for v in three}},
    "BROAD_fn_fp_audit": extra["fn_audit"],
    "GOLD_SET_V1": {"scope": extra["goldset_scope"], "e3": gs["e3"], "best_f1": gs["best_f1"],
                    "skipped_a1_cases": len(gs["skipped_a1_cases"])},
    "G5": {**extra["g5"], "report": "measure/raw/g5_report.json"},
    "headline": extra["headline"],
    "runtime_rss": extra["runtime_rss"],
    "vs_iter4": {
        "BROAD_all26_family_macro": [prev["BROAD"]["all_26_cases_frozen_scorer"]["engine"]["family_macro_f1"], sub["BROAD_all26"]["engine"]["family_macro_f1"]],
        "BROAD_all26_micro": [prev["BROAD"]["all_26_cases_frozen_scorer"]["engine"]["micro_f1"] if "micro_f1" in prev["BROAD"]["all_26_cases_frozen_scorer"]["engine"] else prev["BROAD"]["all_26_cases_frozen_scorer"]["engine"]["micro"]["f1"], sub["BROAD_all26"]["engine"]["micro"]["f1"]],
        "BROAD_ran_family_macro": [prev["BROAD"]["same_scans_engine_ran"]["engine"]["family_macro_f1"], sub["BROAD_same_scans_engine_ran"]["engine"]["family_macro_f1"]],
        "G2a_relabel_all9": [prev["G2"]["a_line_metric"]["all_9_A1_gold11_cases"]["relabel"]["engine"]["f1"], sub["G2a"]["all9"]["relabel"]["engine"]["f1"]],
        "G2a_human_all9": [prev["G2"]["a_line_metric"]["all_9_A1_gold11_cases"]["human"]["engine"]["f1"], sub["G2a"]["all9"]["human"]["engine"]["f1"]],
        "goldset_micro": [prev["GOLD_SET_V1"]["e3"]["micro_f1"], gs["e3"]["micro_f1"]],
        "per_bend_correct_rate": [prev["BROAD"]["per_bend_given_cad"]["correct_rate"], pb["correct_rate"]],
        "G3_verdict": [prev["G3"].get("verdict"), phys["G3"].get("verdict")],
        "d_flange_known_good_p95_mm": [prev["G2"]["d_flange_known_good"]["p95_abs_mm"], G2["flange_known_good"]["p95_abs_mm"]],
    },
    "overlays": {"dir": "docs/production/runlog/a1/iter_5/overlays/worst20", "index": "overlays/worst20/INDEX.json", "count": len(idx["bends"]),
                 "selection": idx["selection"],
                 "legend": "NN_<case>_<bend>_closeup.png = frozen G4 close-up; NN_<case>_<bend>_context.png = raw frame, gold green, engine this bend red, other engine lines orange, best lane dotted blue",
                 
                 "bends": [{k: b.get(k) for k in ("rank", "case_id", "family", "label", "truth", "engine_state", "line_matches_gold",
                                                    "angle_dev_deg", "flange_devs_mm", "decisions", "closeup", "context")} for b in idx["bends"]]},
    "custody": "only cases_A1.json cases were run or scored; GOLD_SET_V1 rows only for A1 scans; never read DEV-CHECK, *SEALED*, FRESH, HOLDOUT10, N001-N016, EHO4; no credentials touched; no repo code edited (work dir outside the repo).",
    "files": {"raw": "measure/raw/", "tools": "measure/tools/", "logs": "measure/logs/",
              "work_dir": "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter5m"},
    "top_failures": extra["top_failures"],
    "summary": extra["summary"],
}
(R / "scorecard.json").write_text(json.dumps(card, indent=1, default=str))
print("written", R / "scorecard.json")
