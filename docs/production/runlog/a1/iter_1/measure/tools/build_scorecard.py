"""Assemble docs/production/runlog/a1/iter_1/scorecard.json from the iter-1 MEASURE outputs (report-only)."""
from __future__ import annotations

import json
from pathlib import Path

R = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_1")
RAW = R / "measure" / "raw"
LOGS = R / "measure" / "logs"
j = lambda p: json.loads(Path(p).read_text())  # noqa: E731
phys = j(RAW / "physical_scorecard.json")
sub = j(RAW / "subset_scores.json")
gs = j(RAW / "goldset_scores.json")
pbx = j(RAW / "per_bend_ext.json")["summary"]
g1 = j(RAW / "g1_regression_scorecard.json"); g1["gates"] = {g["id"]: g for g in g1["gates"]}
g5 = j(RAW / "g5_report.json")
idx = j(R / "overlays" / "INDEX.json")
it0 = j(R.parent / "iter_0" / "scorecard.json")
fp_start = j(LOGS / "fingerprints_start.json")
fp_end = j(LOGS / "fingerprints_end.json")
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
    "iteration": 1,
    "phase": "MEASURE",
    "created": "2026-10-18",
    "label": "consumed development (engineering ledger); never unseen-family accuracy. Iteration 1 = engine e3-1.2.0 after the "
             "iter-1 change (CHANGE.md): non-round datum holes, per-direction pose trust, station-profile flange nominal, "
             "displaced-flange search. Re-measured independently, serially under the shared lock.",
    "comparator_policy": "bar = best F1 lane (old research lane) on the SAME scans; the shipped runtime engine is discarded as a "
                         "comparator; the legacy control arm appears only as informational G2 context.",
    "engine": {
        "engine_version": "e3-1.2.0", "config": "default (no --config)",
        "config_sha256": "74c0305091bf0fb900ce1e137603ec1b3176361ca61f6d94a29c72fed044dcc7",
        "runner_fingerprints_G1": g1["provenance"]["fingerprints"],
        "freeze_style_fingerprints_start": fp_start, "freeze_style_fingerprints_end": fp_end,
        "code_unchanged_during_measure": fp_start == fp_end,
        "matches_change_step_G1_fingerprint": g1["provenance"]["fingerprints"]["engine"].startswith("4e63a064"),
        "reproduces_change_step": "subset_scores.json is byte-for-byte equal in content to the change step's raw/subset_scores.json; "
                                  "overlay INDEX identical",
        "note": "runner fingerprints hash compile+contracts together (d4e870e8); freeze-style split gives compile 6147b43c = freeze "
                "value, so compile code is unchanged since the freeze; eval 8aa6860c unchanged (frozen)",
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
        "vs_iter0": "identical gate-for-gate except flange bias 0.009329 -> 0.009377 mm and runtime noise (P95 10.93 -> 11.39 s)",
        "note": "G1.SUITE FAIL is by construction on the regression tier (needs >= 200 cases); runtime gates need the runtime tier",
    },
    "G2": {
        "verdict": G2["verdict"], "final": False, "gates": G2["gates"],
        "a_line_metric": {
            "gate_of_record_per_freeze": "HUMAN_GOLD11 (owner 2026-10-17: gold11_relabel_v1 adopted as gold of record; both reported)",
            "thresholds": {"f1_min": 0.8, "precision_min": 0.9, "matcher": "5 deg / 4 mm / 0.6 span, one-to-one, apex locus, raw frame"},
            "all_9_A1_gold11_cases": sub["G2a"]["all9"],
            "same_scans_engine_ran_4": sub["G2a"]["same_scans_engine_ran"],
            "legacy_arm_informational": {"human": {k: G2["line_metric"]["legacy"][k] for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative")},
                                         "relabel": {k: G2["line_metric_dual_report"]["relabel_report"]["legacy"][k] for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative")}},
            "relabel_changed_since_freeze": G2["line_metric_dual_report"]["relabel_changed_since_freeze"],
            "relabel_snapshot_sha256": sub["G2a"]["relabel_snapshot_now"],
            "bar": "best F1 lane: 0.510 Gold11 relabel (11 scans; 2 are outside A1); on the 9 A1 Gold11 cases it scores 0.547 relabel / 0.473 human",
            "verdict_relabel": G2["line_metric_dual_report"]["relabel"]["verdict"],
            "verdict_human": G2["line_metric_dual_report"]["human_all_cases"]["verdict"],
            "why_fail": "5 of 9 Gold11 cases are refused by the amendment-7 gate (no frozen G3 truth table) and count as misses; on the 4 that run, relabel F1 = 1.000 (31/31, 0 FP)",
        },
        "b_coverage": {k: strip(v) for k, v in G2["coverage"].items() if isinstance(v, dict)},
        "c_angle_known_good": strip(G2["angle_known_good"]),
        "d_flange_known_good": G2["flange_known_good"],
        "e_runtime": G2["runtime"],
        "f_repeatability": strip(G2["repeatability"]),
        "f_repeatability_note": "NOT_EVALUABLE: the 15651004 repeat set has 3 scans, 2 refused by the amendment-7 gate, so 0 E3 pairs",
        "compile_rate": {"rate": G2["compile_rate"]["dev"]["rate"], "gate": G2["compile_rate"]["gate"]},
        "population_note": "5 of 15 G2 cases refused by the amendment-7 gate, 1 compile BLOCKED (pc_b4719db3dd69); refused cases stay in the denominators",
    },
    "G3": phys["G3"],
    "BROAD": {
        "bar": "best F1 lane 0.458 on broad-57 (old definition, engine not run there); on the frozen A1 BROAD gold the same lane scores family-macro 0.543 / micro 0.508",
        "verdict": B["verdict"], "target_status": B["line_metric"]["target_status"],
        "all_26_cases_frozen_scorer": {"engine": brief(sub["BROAD_all26"]["engine"]), "best_f1": brief(sub["BROAD_all26"]["best_f1"])},
        "same_13_scans_engine_ran": {"cases": sub["BROAD_same_scans_engine_ran"]["cases"],
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
        "fp_analysis": "all 12 BROAD FP lines (38172000 X/V 4+4, 37361005 2, 11979003 2) are matched by the Gold11 relabel and/or GOLD_SET_V1: gold-convention differences in the BROAD gold, not spurious lines",
        "per_bend_given_cad": {**pb, "extended": pbx,
                               "false_pass_definitions": {
                                   "loose_proxy (frozen iter-0 definition)": "PASS on a bend whose line matches no BROAD gold line: 17 = 11 convention cases (all confirmed by relabel / gold set) + 6 PASS with position withheld",
                                   "strict_proxy": "PASS on a published line that no gold of record (BROAD, Gold11 relabel, GOLD_SET_V1) confirms: 0",
                                   "pass_without_published_line": "6 bends (49204020 B3/B4/B6/B12, 15651004 B1/B2): FORMED, angle + flanges measured and PASS, position UNKNOWN(REGISTRATION_UNTRUSTED); the bend exists per gold",
                                   "caveat": "no physical dimensional truth exists for these scans; FAIL decisions (27) on 38172000 angles and the 50757000 sc_752e flange chain are unverified and could be false FAILs"}},
    },
    "GOLD_SET_V1": {"scope": "GOLD_SET_V1 scans that are A1 cases with STEP and an OK engine run (13 scans / 11 families); DEV-CHECK never read; 12 A1 gold-set scans not run (amendment-7 gate)",
                    "e3": gs["e3"], "best_f1": gs["best_f1"], "skipped_a1_cases": len(gs["skipped_a1_cases"])},
    "G5": {"verdict": g5["gate"] if isinstance(g5.get("gate"), str) else (g5.get("gate") or {}).get("verdict", "FAIL"),
           "manifest": "PASS (1 allowlisted add: scripts/run_production_line_checks.sh)", "tsc": "PASS (new=0)", "lint": "PASS (new=0)",
           "vendor_scripts": "PASS (5 NOT_RUN matching baseline)",
           "backend_tests": "FAIL: 10 NEW failures in tests/test_vendor_unfold_api_contract.py (same 10 as iter 0)",
           "cause": "apps/api/services/vendor_setup_service.py (mtime 2026-10-18 02:05, another workflow) now calls module helper _natural_label_key; the test AST-extracts get_vendor_setup alone -> NameError. Not engine-related; the A1 engine change touches no backend/, domains/bend/ or scripts/ file.",
           "report": "measure/raw/g5_report.json"},
    "runtime_rss": {"physical_e3_p95_wall_s_G2": G2["runtime"]["p95_wall_s"], "physical_e3_p95_wall_s_BROAD": B["runtime"]["p95_wall_s"],
                    "physical_e3_max_wall_s": B["runtime"]["max_wall_s"], "physical_run_peak_rss_gib": 0.953,
                    "synthetic_p95_wall_s": g1["gates"]["G1.RUNTIME.P95"]["value"],
                    "synthetic_engine_peak_rss_gib": round(g1["gates"]["G1.RUNTIME.RSS"]["value"] / 2 ** 30, 3),
                    "g1_runner_peak_rss_gib": 0.832, "compile_frozen_peak_rss_gib": 0.740, "g5_peak_rss_gib": 1.209,
                    "lock": "every heavy step serial under /private/tmp/claude-501/sherman_geom.lock, OMP_NUM_THREADS=1"},
    "vs_iter0": {
        "BROAD_all26_family_macro": [it0["BROAD"]["all_26_cases_frozen_scorer"]["engine"]["family_macro_f1"], sub["BROAD_all26"]["engine"]["family_macro_f1"]],
        "BROAD_all26_micro": [it0["BROAD"]["all_26_cases_frozen_scorer"]["engine"]["micro"]["f1"], sub["BROAD_all26"]["engine"]["micro"]["f1"]],
        "BROAD_13_family_macro": [it0["BROAD"]["same_13_scans_engine_ran"]["engine"]["family_macro_f1"], sub["BROAD_same_scans_engine_ran"]["engine"]["family_macro_f1"]],
        "G2a_relabel_all9": [it0["G2"]["a_line_metric"]["all_9_A1_gold11_cases"]["relabel"]["engine"]["f1"], sub["G2a"]["all9"]["relabel"]["engine"]["f1"]],
        "G2a_human_all9": [it0["G2"]["a_line_metric"]["all_9_A1_gold11_cases"]["human"]["engine"]["f1"], sub["G2a"]["all9"]["human"]["engine"]["f1"]],
        "goldset_micro": [it0["GOLD_SET_V1"]["e3"]["micro_f1"], gs["e3"]["micro_f1"]],
        "per_bend_correct_rate": [it0["BROAD"]["per_bend_given_cad"]["correct_rate"], pb["correct_rate"]],
        "d_flange_known_good_p95_mm": [1.22306435, G2["flange_known_good"]["p95_abs_mm"]],
    },
    "overlays": {"dir": "docs/production/runlog/a1/iter_1/overlays", "index": "overlays/INDEX.json", "count": len(idx["bends"]),
                 "selection": idx["selection"],
                 "legend": "NN_<case>_<bend>_closeup.png = frozen G4 close-up; NN_<case>_<bend>_context.png = raw frame, gold green, engine this bend red, other engine lines orange, best lane dotted blue",
                 "bends": [{k: b.get(k) for k in ("rank", "case_id", "family", "label", "engine_state", "line_matches_gold",
                                                    "angle_dev_deg", "flange_devs_mm", "decisions", "closeup", "context")} for b in idx["bends"]]},
    "custody": "only cases_A1.json cases were run or scored; GOLD_SET_V1 rows only for A1 scans; never read DEV-CHECK, *SEALED*, FRESH, HOLDOUT10, N001-N016, EHO4; no credentials touched; no repo code edited (work dir outside the repo).",
    "files": {"raw": "measure/raw/ (physical_scorecard, subset_scores, goldset_scores, per_bend, per_bend_ext, g1_regression_scorecard.json/.md, g5_report)",
              "tools": "measure/tools/ (measure.sh, analyse_iter1.py, goldset_iter1.py, overlays_iter1.py, per_bend_ext.py, build_scorecard.py)",
              "logs": "measure/logs/", "work_dir": "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter1m"},
}
card["top_failures"] = [
    "PROCESS: amendment-7 gate refuses 13/26 BROAD cases and 5/9 G2(a) Gold11 cases (no frozen G3 visual truth table); they count as misses: BROAD all-26 recall 0.431 vs 0.795 on the runnable 13; G2(a) relabel 0.574 vs 1.000 on the 4 runnable; G3 NOT_EVALUABLE; G2(f) NOT_EVALUABLE",
    "6 bends with position withheld (REGISTRATION_UNTRUSTED along the free datum direction) yet angle/flange PASS: 49204020 B3/B4/B6/B12, 15651004 B1/B2 (-6 BROAD TPs); needs a decision rule and/or stronger datum-hole information",
    "38172000 X/V angle FAILs: 8 bends at -1.7 to -2.8 deg consistent across both scans, plus V B5 +4.8 deg / flange B -9.17 mm (REVIEW) and V B7 +5.4 deg FAIL: unverified, possible false FAILs; need section-level visual check",
    "50757000 sc_752e flange FAIL chain (B1 +4.9 mm, P6 +4.8 mm, four tabs about -3 mm): plausible real defect, physically unverified",
    "BROAD gold-convention conflicts: 12 FP / 6 FN on 38172000 X/V, 37361005, 11979003 vs the BROAD gold although relabel / GOLD_SET_V1 confirm every published line; 11979003 has 9 gold lines vs 7 recipe bends (ceiling 0.875)",
    "G2(c) angle known-good coverage 14/32 vs legacy 16/32 (refused cases); G2(b) coverage 0.60 angle / 0.61 flange overall but 100% on runnable cases",
    "G5 FAIL from 10 NEW backend failures in tests/test_vendor_unfold_api_contract.py caused by another workflow's vendor_setup_service.py edit (not engine)",
]
(R / "scorecard.json").write_text(json.dumps(card, indent=1, default=str))
print("written", R / "scorecard.json")
