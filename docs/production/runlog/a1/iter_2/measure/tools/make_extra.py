import json, sys
from pathlib import Path
L = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_2/measure/logs")
def rss(name):
    for line in (L / f"{name}.log").read_text().splitlines():
        if "maximum resident set size" in line:
            return round(int(line.split()[0]) / 2**30, 3)
g1 = json.loads((L.parent / "raw/g1_regression_scorecard.json").read_text())
gates = {g["id"]: g for g in g1["gates"]}
p1 = json.loads(Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_1/scorecard.json").read_text())["G1_synthetic_regression_DEV"]["gates"]
diff = {k: [p1.get(k, {}).get("value"), v.get("value")] for k, v in gates.items() if p1.get(k, {}).get("value") != v.get("value")}
phys = json.loads((L.parent / "raw/physical_scorecard.json").read_text())
extra = {
 "engine_version": "e3-1.3.0",
 "config_sha256": "417f580b2d9369040411137a153b96458c973a3f50401d77e20be4b01a8b9d46",
 "matches_change_step": "engine fingerprint 4b4bdb0f... = CHANGE.md; all 13 E3 result.json carry config_sha256 417f580b...; subset_scores, goldset_scores, per_bend, per_bend_ext are content-identical to the change step's raw/; physical_scorecard differs only in created_at and runtime (P95 19.70 -> 19.64 s); G1 gates identical except runtime noise",
 "g1_vs_iter1": {"changed_gate_values_[iter1, iter2]": diff},
 "g2a_why_fail": "5 of 9 Gold11 cases are refused by the amendment-7 gate (no frozen G3 truth table) and count as misses; on the 4 that run, relabel F1 = 1.000 (31/31, 0 FP) vs best lane 0.561, human 0.724 vs best lane 0.717",
 "g2_population_note": "5 of 15 G2 cases refused by the amendment-7 gate, 1 compile BLOCKED (pc_b4719db3dd69, ENGINE_EXIT_2 COMPILE_BLOCKED); refused cases stay in the denominators. G2(f) NOT_EVALUABLE: 2 of the 3 15651004 repeat scans refused, 0 E3 pairs.",
 "fp_analysis": "all 12 BROAD FP lines on the 13 runnable scans (38172000 X/V 4+4, 37361005 2, 11979003 2) are confirmed by the Gold11 relabel and/or GOLD_SET_V1 (line_confirmed_any_rate 80/80): gold-convention differences in the frozen BROAD gold, not spurious lines",
 "false_pass_definitions": {
   "loose_proxy (frozen iter-0 definition)": "PASS on a bend whose line matches no BROAD gold line: 11, all on lines confirmed by relabel / GOLD_SET_V1 (convention cases)",
   "strict_proxy": "PASS on a published line that no gold of record (BROAD, Gold11 relabel, GOLD_SET_V1) confirms: 0",
   "pass_without_published_line": "0 (was 6 in iteration 1: 49204020 B3/B4/B6/B12, 15651004 B1/B2 now have measured positions)",
   "caveat": "no physical dimensional truth exists for these scans; the 27 FAIL decisions (38172000 angles, 50757000 sc_752e flange chain, 50757000 sc_777c angles) are unverified and could be false FAILs; correct_vs_any_gold 75/80, the 5 misses are compile-side unavailable flange nominals (50757000 B1/B4 side B on both scans, 47266000 B1 UNSUPPORTED_FEATURE)"},
 "goldset_scope": "GOLD_SET_V1 scans that are A1 cases with STEP and an OK engine run (13 scans / 11 families); DEV-CHECK never read; 12 A1 gold-set scans not run (amendment-7 gate)",
 "g5": {"verdict": "FAIL", "manifest": "PASS (1 allowlisted add: scripts/run_production_line_checks.sh)", "tsc": "PASS (new=0)", "lint": "PASS (new=0)",
        "vendor_scripts": "PASS (5 NOT_RUN matching baseline)",
        "backend_tests": "FAIL: 10 NEW failures in tests/test_vendor_unfold_api_contract.py, the same 10 as iterations 0 and 1",
        "cause": "carried over from iteration 1's diagnosis: apps/api/services/vendor_setup_service.py (another workflow) calls module helper _natural_label_key while the test AST-extracts get_vendor_setup alone -> NameError. Not engine-related; the iter-2 diff touches only domains/production/engine/inspect/."},
 "runtime_rss": {"physical_e3_p95_wall_s_G2": phys["G2"]["runtime"]["p95_wall_s"], "physical_e3_p95_wall_s_BROAD": phys["BROAD"]["runtime"]["p95_wall_s"],
                 "physical_e3_max_wall_s": phys["BROAD"]["runtime"]["max_wall_s"], "physical_e3_crashes": len(phys["BROAD"]["runtime"]["crashes"]),
                 "legacy_control_p95_s_informational": phys["G2"]["runtime"].get("legacy_control_p95_s"),
                 "physical_run_peak_rss_gib": rss("run_physical"), "g1_runner_peak_rss_gib": rss("g1"),
                 "synthetic_p95_wall_s": gates["G1.RUNTIME.P95"]["value"], "synthetic_engine_peak_rss_gib": round(gates["G1.RUNTIME.RSS"]["value"] / 2**30, 3),
                 "compile_frozen_peak_rss_gib": rss("compile_frozen"), "overlays_peak_rss_gib": rss("overlays"), "g5_peak_rss_gib": rss("g5"),
                 "lock": "every heavy step serial under /private/tmp/claude-501/sherman_geom.lock, OMP_NUM_THREADS=1 (measure/tools/measure.sh)"},
 "top_failures": [
  "PROCESS (biggest lever): amendment-7 gate still refuses 13/26 BROAD cases and 5/9 G2(a) Gold11 cases (G3 visual truth table not frozen); they count as misses: BROAD all-26 recall 0.472 vs 0.872 on the 13 runnable; G2(a) relabel 0.574 vs 1.000 on the 4 runnable; G3, G2(f) NOT_EVALUABLE; G2(b) coverage 0.60 angle / 0.61 flange (100% on runnable); G2(e) all_cases_run FAIL",
  "BROAD-gold convention conflicts: 12 FP / 10 FN on the 13 runnable scans (38172000 X/V, 37361005, 11979003) although every published line is confirmed by the relabel or GOLD_SET_V1 (80/80); loose false-PASS proxy 11 comes only from these (strict proxy 0); 11979003 has 9 gold lines vs 7 recipe bends",
  "38172000 X/V angle FAILs (X B7 +5.4 deg, X B9 -1.7, V B9 -2.8, V B5 +1.9 ...): independent plane fits in the change step reproduce them (real over/underbend), but no physical truth; possible false FAILs remain unverified",
  "50757000 sc_752e flange FAIL chain (B1 +4.9 mm, B2 side B +4.8 mm, B5 side B -2.9 mm) and sc_777c angle FAILs (B5 -1.3, B6 -1.2 deg): plausible real defects, physically unverified",
  "5 bends not 'correct' vs any gold: compile-side unavailable flange nominals (50757000 B1/B4 side B on both scans, 47266000 B1 UNSUPPORTED_FEATURE); pc_b4719db3dd69 compile BLOCKED",
  "G2(c) known-good angle coverage 14/32 vs legacy 16/32 (refused cases; MAE 0.437 deg, common-bend 0.506 vs legacy 1.360)",
  "Open engine risk: a joggle with a shortened riser (~1 mm) still reads NOT_YET_FORMED; the new 0.4 mm flat window and hinge-continuity rule are unseen on real staged scans (10839000 joggles, 23553000 hooks) until G3 runs",
  "G5 FAIL: 10 NEW backend failures in tests/test_vendor_unfold_api_contract.py from another workflow's vendor_setup_service.py edit (not engine)",
 ],
}
extra["summary"] = ("A1 iter-2 MEASURE (e3-1.3.0, config 417f580b, code unchanged during measure; reproduces the change step exactly). "
  "BROAD frozen A1 list, all 26 cases (refused count as misses): family-macro 0.667 / micro 0.607 (P 0.85, R 0.47) vs best F1 lane 0.543 / 0.508 on the same scans; "
  "on the 13 scans E3 runs: 0.896 / 0.861 (P 0.85, R 0.87) vs best lane 0.649 / 0.620. "
  "G2(a) all 9 Gold11: relabel 0.574 (P 1.00) vs best lane 0.547; human 0.467 vs 0.473; on the 4 runnable: relabel 1.000 vs 0.561. "
  "GOLD_SET_V1 (13 A1 scans with STEP): 1.000 micro / 1.000 family-macro vs best lane 0.597 / 0.653. "
  "Per bend given CAD (80 bends): correct 0.788 vs BROAD gold, 0.938 vs any gold of record; strict false PASS 0, PASS without line 0. "
  "G1 regression DEV PASS (0 failed accuracy gates; non-qualifying by construction). G2 FAIL (a,b,c,e by refused cases; d flange P95 0.52 mm PASS; f NOT_EVALUABLE). "
  "G3 NOT_EVALUABLE (truth table missing). G5 FAIL (10 non-engine backend tests). E3 P95 19.6 s, peak RSS 0.91 GiB. "
  "Biggest lever is process: freeze the G3 truth table so the 13 refused cases run.")
(L / "extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps(extra["runtime_rss"], indent=1), diff)
