"""Operator-computed fields of the A1 iter-4 MEASURE scorecard (rss/runtime, G1 diff vs iter 3, text)."""
import json
from pathlib import Path

R = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_4")
L = R / "measure" / "logs"
RAW = R / "measure" / "raw"


def rss(name):
    for line in (L / f"{name}.log").read_text().splitlines():
        if "maximum resident set size" in line:
            return round(int(line.split()[0]) / 2**30, 3)


g1 = json.loads((RAW / "g1_regression_scorecard.json").read_text())
gates = {g["id"]: g for g in g1["gates"]}
p3 = json.loads((R.parent / "iter_3" / "scorecard.json").read_text())["G1_synthetic_regression_DEV"]["gates"]
diff = {k: [p3.get(k, {}).get("value"), v.get("value")] for k, v in gates.items() if p3.get(k, {}).get("value") != v.get("value")}
phys = json.loads((RAW / "physical_scorecard.json").read_text())
pba = json.loads((RAW / "per_bend_all.json").read_text())["summary"]
extra = {
    "label": "consumed development (engineering ledger); never unseen-family accuracy. Iteration 4 = engine e3-1.5.0 after the "
             "iter-4 change (CHANGE.md fixes 1-12: hole-pair poses, weak-direction refine, fine-search walk, round-hole core "
             "clearance, pose-on-plane children, station-spread U, distinct refinement slots, articulated fit before WRONG_PART, "
             "joggle step riser, near-twin hole evidence, bowed-panel re-seat, window clearance). First measure step with the G3 "
             "truth table frozen: all 27 A1 cases run (26 compiled; pc_b4719db3dd69 compile BLOCKED). Re-measured independently, "
             "serially under the shared lock.",
    "engine_version": "e3-1.5.0",
    "config_sha256": "687eb815ae0fc4c4e23aab02426e07f4dc57bd6f9e229b760fb4e0bea595d365",
    "matches_change_step": "freeze-style engine fingerprint dd4f039d... = CHANGE.md validation/code_fingerprints_final.json (start = end of "
                           "measure); all 26 E3 result.json carry e3-1.5.0 / config 687eb815...; subset_scores and goldset_scores are "
                           "byte-identical to validation/*_final.json, per_bend and per_bend_ext byte-identical to the change step's "
                           "analysis_v4; per_bend_all summary identical (177/203 riser-aware); physical_scorecard differs only in "
                           "created_at, truth_table.earliest_e3_record_at and runtime (G2 P95 26.77 -> 27.17 s, max 30.59 -> 31.17 s); "
                           "G1 accuracy gates identical to the change step's final G1.",
    "g1_vs_iter3": {"changed_gate_values_[iter3, iter4]": diff,
                    "note": "only flange bias (+6e-6 mm), MEDIAN_U.deg (+1.7e-5, fix 6 station spread), slopes at the 6th decimal and runtime moved"},
    "g2a_why_fail": "G2(a) in the frozen scorer gates on the HUMAN Gold11 (historical diagnostic per owner): F1 0.391 (P 0.342, R 0.458; 27 TP / 52 FP / 32 FN) vs best F1 lane 0.473 -> FAIL. "
                    "On gold11_relabel_v1 (gold of record per OWNER_DECISION_2026-10-17) F1 0.987 (P 0.975, R 1.000; 77/2/0) vs best lane 0.547 -> would PASS both F1>=0.80 and P>=0.90. "
                    "The 52 human-gold FPs are lines the relabel confirms (human labels under audit). All 9 G2(a) cases now run (were 4 of 9 in iter 3).",
    "g2_population_note": "all 13 G2 E3 runs complete (the amendment-7 gate now admits every case: G3 truth table frozen, ledger frz_7344c93bcfe8); 1 compile BLOCKED (pc_b4719db3dd69, ENGINE_EXIT_2 COMPILE_BLOCKED, explicit refusal) stays in the denominators. "
                          "G2(f) evaluable for the first time: FAIL on P95 angle 0.426 deg (15651004's three 'repeats' are three different stages per CHANGE.md); flange 0.254 mm PASS.",
    "fp_analysis": "BROAD (26 A1 BROAD cases, 24 line cases, 14 families): 55 FP / 42 FN vs the older BROAD gold, while the same engine lines score 0.977 micro on GOLD_SET_V1 (3 FP / 4 FN) and 0.987 on the Gold11 relabel: BROAD-gold convention differences dominate the BROAD gap (line_confirmed_any_rate 0.9625 on the 80 complete-scan bends).",
    "false_pass_definitions": {
        "loose_proxy (frozen iter-0 definition, BROAD gold only)": "PASS on a bend whose line matches no BROAD gold line: 11 (complete BROAD scans, 80 bends), all on lines the relabel / GOLD_SET_V1 confirm",
        "strict_proxy (complete BROAD scans)": "PASS on a published line that no gold of record (BROAD, Gold11 relabel, GOLD_SET_V1) confirms: 0; PASS without a published line: 0",
        "all_A1 proxy (per_bend_all)": "PASS on a bend the frozen G3 truth calls NOT_FORMED, or on a formed bend whose line the gold of record does not confirm: 3 (pc_07062eee17aa B1 - 38172000 SCAN_1, line not confirmed by GOLD_SET_V1; pc_cec08dfca9ab B6 and B11 - 10839000 scan_2, truth NOT_FORMED but CHANGE.md's hole evidence says the truth rows are the 180-deg twin and look swapped; disputed)",
        "caveat": "no physical dimensional truth exists for these scans; FAIL decisions are unverified (false FAILs possible)"},
    "per_bend_all_definition": "every A1 case the engine ran (26 OK runs, 203 bends): expected state = frozen G3 truth row (staged scans) else FORMED (complete scans); NOT_DETERMINABLE truth not scored. Correct(FORMED) = engine FORMED-like, angle MEASURED, both flange sides MEASURED (riser-aware: a joggle riser's own side UNKNOWN(UNSUPPORTED_FEATURE) allowed) and line confirmed by GOLD_SET_V1 / Gold11 relabel where they exist. Correct(NOT_FORMED) = NOT_YET_FORMED. This is the primary per-bend-given-CAD figure for iteration 4.",
    "goldset_scope": "GOLD_SET_V1 scans that are A1 cases with STEP and an OK engine run: 23 scans / 14 families (all A1 gold-set scans now run, 0 skipped); DEV-CHECK never read",
    "g5": {"verdict": "FAIL", "manifest": "PASS (1 allowlisted add: scripts/run_production_line_checks.sh)", "tsc": "PASS (new=0)", "lint": "PASS (new=0)",
           "vendor_scripts": "PASS (5 NOT_RUN matching baseline)",
           "backend_tests": "FAIL: 10 NEW failures in tests/test_vendor_unfold_api_contract.py, the same 10 as iterations 0-3",
           "cause": "non-engine: apps/api/services/vendor_setup_service.py (another workflow's edit; NameError on _natural_label_key when the test AST-extracts get_vendor_setup). The iter-4 diff touches only domains/production/engine/inspect/."},
    "runtime_rss": {"physical_e3_p95_wall_s_G2": phys["G2"]["runtime"]["p95_wall_s"], "physical_e3_p95_wall_s_BROAD": phys["BROAD"]["runtime"]["p95_wall_s"],
                    "physical_e3_max_wall_s": phys["BROAD"]["runtime"]["max_wall_s"], "physical_e3_crashes": len(phys["BROAD"]["runtime"]["crashes"]),
                    "physical_e3_runs": phys["BROAD"]["runtime"]["runs"],
                    "legacy_control_p95_s_informational": phys["G2"]["runtime"].get("legacy_control_p95_s"),
                    "physical_run_peak_rss_gib": rss("run_physical"), "g1_runner_peak_rss_gib": rss("g1"),
                    "synthetic_p95_wall_s": gates["G1.RUNTIME.P95"]["value"], "synthetic_engine_peak_rss_gib": round(gates["G1.RUNTIME.RSS"]["value"] / 2**30, 3),
                    "compile_frozen_peak_rss_gib": rss("compile_frozen"), "overlays_peak_rss_gib": rss("overlays"), "g5_peak_rss_gib": rss("g5"),
                    "lock": "every heavy step serial under /private/tmp/claude-501/sherman_geom.lock, OMP_NUM_THREADS=1 (measure/tools/measure.sh; per_bend_all + overlays re-run once under the lock after a ranking fix)"},
    "headline": {
        "BROAD_A1_26_cases": {"engine": {"family_macro_f1": 0.806, "micro_f1": 0.678, "precision": 0.650, "recall": 0.708, "tp_fp_fn": [102, 55, 42]},
                              "best_f1_lane_same_scans": {"family_macro_f1": 0.543, "micro_f1": 0.508, "precision": 0.580, "recall": 0.451, "tp_fp_fn": [65, 47, 79]},
                              "owner_target_family_macro_0.80": "MET (report-only)"},
        "Gold11_relabel_v1_9_A1_cases": {"engine_f1": 0.987, "precision": 0.975, "recall": 1.0, "best_f1_lane": 0.547},
        "Gold11_human_9_A1_cases": {"engine_f1": 0.391, "precision": 0.342, "recall": 0.458, "best_f1_lane": 0.473},
        "GOLD_SET_V1_23_scans": {"engine_micro": 0.977, "engine_family_macro": 0.966, "best_f1_micro": 0.566, "best_f1_family_macro": 0.583},
        "per_bend_given_cad_all_A1_203": {"riser_aware": round(pba["rate_riser_aware"], 4), "strict": round(pba["rate_strict"], 4),
                                          "family_macro_riser_aware": round(pba["family_macro_riser_aware"], 4), "false_pass_proxy": pba["false_pass_proxy"]},
        "per_bend_complete_BROAD_80": {"correct_vs_broad_gold": "59/80 = 0.738 (iter 3: 63/80 = 0.788)", "correct_vs_any_gold": "71/80 (iter 3: 75/80)", "strict_false_pass": 0},
    },
    "regression_found": {
        "case": "pc_752e2bd54aad (50757000_02_X.ply, complete BROAD scan; ran in iter 3 too)",
        "what": "same datum pose (identical raw_to_frame) but B5/B6 now UNKNOWN (all characteristics OCCLUDED, lines unpublished), B4 angle -16.36 deg FAIL (iter 3: +0.08 PASS) with its line no longer gold-matched, B7/B8 angles +15.09/+13.78 deg FAIL (iter 3: +0.15/+0.30 PASS) and their side A ADJACENT_BEND_NOT_MEASURED. 19 characteristic decisions changed; required_unknown 2 -> 12.",
        "when": "already present in the change step's 'first full run (fixes 1-8)' (validation/per_bend_all_fixes1to8.json), so it comes from fixes 1-8 (second start), most likely fix 5 (pose_on_plane turning child poses onto their fitted planes) or fix 7 (refinement slots) - not bisected here (measure step edits no engine). CHANGE.md lists 50757000 among 'identical characteristic values', which is true only between fixes1-8 and final, not vs iteration 3.",
        "effect": "complete-BROAD per-bend correct 63 -> 59 of 80; adds 2 likely false angle FAILs of 14-16 deg and 1 on B4; GOLD_SET_V1 line FN on this scan",
        "overlays": "worst20 ranks 1, 2, 7"},
    "top_failures": [
        "REGRESSION vs iteration 3 on 50757000 sc_752e (pc_752e2bd54aad): same datum pose, but B5/B6 became UNKNOWN (OCCLUDED), B4 angle -16.4 deg FAIL and B7/B8 +15.1/+13.8 deg FAIL (iter 3 read them within 0.3 deg); present since fixes 1-8 (not in CHANGE.md); complete-BROAD per-bend correct 0.788 -> 0.738. Overlays 01, 02, 07.",
        "G3 FAIL (state match 118/123 = 0.959): 4 mismatches on 10839000 scan_2 (pc_cec08dfca9ab B6/B8/B9/B11 - the part's 180-deg near-twin; CHANGE.md's hole evidence 10/0 vs 1/7 says the frozen truth rows look swapped; needs re-adjudication, not an engine change) + 28870000 SCAN_1 (pc_3bde9996f14d) UNKNOWN vs NOT_FORMED (rolled part, WRONG_PART_SUSPECTED). unformed_reported_as_defect = 2 (the disputed cec08 B6/B11). Overlays 03-05.",
        "G3 known-good formed-bend angle MAE 1.186 deg FAIL (common 8 bends 1.323 vs legacy 1.222): dominated by 10839000's 90-deg bends reading 1-2 deg overbent on all three scans (legacy sees the same; likely real).",
        "G2(a) FAIL on the frozen scorer's human Gold11 gate (F1 0.391 vs best lane 0.473); on gold11_relabel_v1 (owner's gold of record) F1 0.987 P 0.975 R 1.000 vs best lane 0.547. Needs the owner's sign-off on the relabel to move the gate of record.",
        "G2(f) FAIL: repeatability P95 angle 0.426 deg on 15651004, whose three 'repeat' scans are three different stages - a case-list problem as much as an engine one.",
        "38172000 SCAN_1 (pc_07062eee17aa) B1 PASS on a line GOLD_SET_V1 does not confirm (false-PASS proxy), B9 UNEXPECTED_FORMED with 3 FAILs; 38172000_01_X (pc_4f8188b4d1a8) B5 flange side B -8.28 mm FAIL + angle +4.76 deg (pre-existing, unchanged since iter 3). Overlays 06, 08, 12.",
        "23553000 SCAN_2 (pc_186463746daa, staged) B2/B6/B10/B14 read UNEXPECTED_FORMED with characteristics UNKNOWN (truth FORMED; state family matches G3 but not counted correct per bend); CHANGE.md risk 7 on 23553000 FULL (hook flanges pose-sensitive, +-1.2 mm FAILs) unverified. Overlays 09, 10, 14.",
        "BROAD vs older BROAD gold: precision 0.650 (55 FP) although GOLD_SET_V1 / relabel confirm the same lines (0.977 / 0.987) - gold-convention, not engine, limited.",
        "G5 FAIL: 10 NEW backend failures in tests/test_vendor_unfold_api_contract.py (vendor_setup_service.py, another workflow; not engine).",
    ],
}
extra["summary"] = (
    "A1 iter-4 MEASURE (e3-1.5.0, config 687eb815; code unchanged during measure; reproduces the change step byte for byte apart from runtime). "
    "First measure with the G3 truth table frozen: all 27 A1 cases run (26 OK, 1 compile BLOCKED). "
    "BROAD frozen A1 list (26 cases / 24 line cases / 14 families): family-macro 0.806, micro 0.678 (P 0.650, R 0.708; TP 102 FP 55 FN 42) vs best F1 lane 0.543 / 0.508 on the same scans - owner target 0.80 family-macro MET (report-only). "
    "Gold11 relabel (gold of record) 0.987 (P 0.975, R 1.00) vs best lane 0.547; human Gold11 0.391 vs 0.473 (frozen G2(a) gate -> FAIL). "
    "GOLD_SET_V1 (23 A1 scans with STEP): 0.977 micro / 0.966 family-macro vs best lane 0.566 / 0.583. "
    "Per bend given CAD (all A1, 203 bends): 0.872 riser-aware, 0.818 strict, family-macro 0.891, false-PASS proxy 3 (2 on disputed truth rows); complete BROAD 80 bends: 0.738 vs BROAD gold, 71/80 vs any gold, strict false PASS 0. "
    "G1 regression DEV PASS (accuracy gates identical to iter 3). G2 FAIL only on (a) human-gold gate and (f) different-stage repeats; (b)(c)(d)(e) + compile PASS (angle MAE 0.795 deg, flange P95 0.392 mm, P95 27.2 s). "
    "G3 FAIL: state match 0.959 (4 disputed 10839000 scan_2 rows + 28870000 SCAN_1), known-good angle MAE 1.186 deg; coverage, flange P95 0.368 mm, blank, counts PASS. G5 FAIL (10 non-engine backend tests). "
    "New finding: a regression vs iter 3 on 50757000 sc_752e (B4/B7/B8 14-16 deg angle FAILs, B5/B6 UNKNOWN) from fixes 1-8, not reported in CHANGE.md. Peak RSS 0.95 GiB.")
(L / "extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps(extra["runtime_rss"], indent=1), json.dumps(diff))
