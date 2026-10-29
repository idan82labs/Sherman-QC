"""Operator text for the A1 iter-7 MEASURE scorecard; merges logs/extra_auto.json (computed) into logs/extra.json.
Numbers quoted in the text were read from measure/raw (physical_scorecard.json, keynums) of this measure step."""
import json
from pathlib import Path

L = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_7/measure/logs")
e = json.loads((L / "extra_auto.json").read_text())
k = json.loads((L / "keynums.json").read_text())
g5 = json.loads((L.parent / "raw" / "g5_report.json").read_text())
h = e["headline"]
B = h["BROAD_A1_frozen_gold"]
rt = e["runtime_rss"]
r3 = lambda x: round(x, 3)  # noqa: E731
e["fn_audit"]["classification_of_record"] = (
    "unchanged from iteration 6 (the FN/FP line sets are identical line for line to iter_6/measure/raw/broad_fn_fp_audit.json; the file "
    "differs only in measured values carried along): 37 FNs, 36 with no gold-of-record line on the same scan (human-Gold11 lines off the "
    "apex, lines spanning several folds, lines where no fold exists, pre-audit split lines) and 1 gold-of-record fold the recipe excludes "
    "(28870000 FULL); 52 of 54 FPs match a gold-of-record line, 2 are 38172000 SCAN_1 tab lines B1/B9 (stage-level convention).")
e["g5"] = {"verdict": g5["gate"], "manifest": f"{g5['manifest']['status']} (1 allowlisted add: scripts/run_production_line_checks.sh)",
           "tsc": f"{g5['tsc']['status']} (new={len(g5['tsc']['new'])})", "lint": f"{g5['lint']['status']} (new={len(g5['lint']['new'])})",
           "vendor_scripts": f"{g5['vendor_scripts']['status']} ({g5['vendor_scripts']['not_run_count']} NOT_RUN matching baseline)",
           "backend_tests": f"{g5['backend_tests']['status']} ({g5['backend_tests']['counts'].get('passed')} passed, {len(g5['backend_tests']['new'])} new failures)",
           "baseline_integrity": g5["baseline_integrity"]["status"],
           "failing_checks": g5["failing_checks"], "incomplete_checks": g5["incomplete_checks"]}
e["physical_angles_vs_nominal"] = {
    "definition": "every MEASURED included angle in the 26 OK A1 runs, |deviation from nominal|, the two wrong-way hooks (pc_186463746daa B2/B14, ~239 deg off signed nominal, FAIL as they should) excluded",
    "n": 159, "mae_deg": 0.898, "mean_U_deg": 0.956, "iter6": {"mae_deg": 0.946, "mean_U_deg": 1.018},
    "note": "no physical angle truth exists; nominal-relative only. Reproduces CHANGE.md iter-7 exactly."}
e["top_failures"] = [
    "BROAD recall on the frozen BROAD gold 0.743 (107/144) < 0.80, unchanged from iter 6: of 37 FNs, 36 have no matching gold-of-record line on the same scan "
    "(label problems) and 1 is a gold-of-record fold the recipe excludes (28870000 FULL, no recipe bend to report it on). On the gold of record for the same 24 scans "
    "recall is 0.994 (best F1 lane 0.475). Frozen-label recall 0.80 is not reachable without fitting label errors.",
    "G3 FAIL, state match 118/123 = 0.959 (unchanged): 4 mismatches on 10839000 scan_2 (pc_cec08dfca9ab B6/B8/B9/B11; hole evidence says the frozen truth rows are the "
    "180-deg twin, owner re-adjudication needed; overlays 01-03) and 28870000 SCAN_1 (pc_3bde9996f14d B1, rolled part -> UNKNOWN vs NOT_FORMED; overlay 20). "
    "unformed_reported_as_defect = 2 (the disputed cec08 B6/B11).",
    f"G3 known-good formed-bend angle MAE {r3(k['G3_kg_angle']['e3_mae_deg'])} deg FAIL (31 bends; iter 6 1.186): 10839000 {k['G3_kg_per_part']['10839000']} deg "
    f"(CHANGE.md Diagnosis 1: real local overbend/base bow, not an engine bias; joggle risers REVIEW U 7-8 deg), 15651004 {k['G3_kg_per_part']['15651004']} deg. "
    f"Common bends E3 {r3(k['G3_kg_angle']['common_e3_mae_deg'])} vs legacy {r3(k['G3_kg_angle']['common_legacy_mae_deg'])} deg (informational). Owner call whether 10839000 is known-good at +-1 deg.",
    "G2(a) FAIL on the frozen scorer's human Gold11 gate: F1 0.391 (P 0.342, R 0.458) vs best F1 lane 0.473; on gold11_relabel_v1 (gold of record) F1 0.987 (P 0.975, R 1.000) "
    "vs best lane 0.547 -> PASS (F1>=0.80, P>=0.90). Moving the frozen gate of record needs the owner's sign-off.",
    f"G2(f) FAIL: repeatability P95 angle {r3(k['G2f']['e3_p95_angle_deg'])} deg vs gate 0.30 (iter 6 0.426) on 15651004, whose three 'repeats' are three different stages.",
    "38172000 SCAN_1 (pc_07062eee17aa) B1/B9: tab lines collinear with the 214 mm folds; the relabel draws each pair as one 305 mm line -> 2 GOLD_SET_V1 / gold-of-record FPs "
    "and the false-PASS proxy on B1. Stage-level convention question for the owner. Overlays 04, 05.",
    "23553000 SCAN_2 (pc_186463746daa) B2/B14 wrong-way hooks: found and measured (angle FAIL, ~239 deg off signed nominal); flange sides UNKNOWN by contract. Overlays 06, 07.",
    "50757000_02_X (pc_752e2bd54aad) B1/B4 flange/position ~4.9-5.0 mm off nominal FAIL with angles in tolerance (one off-nominal backgauge, unverified); "
    "38172000_01_X (pc_4f8188b4d1a8) B5 side B -8.30 mm + angle +4.69 deg FAIL since iter 3. Overlays 08, 09, 11.",
    "Per-bend correctness given CAD 183/203 = 0.901 riser-aware (0.837 strict), family-macro 0.910, unchanged: remaining not-correct are flange sides ending at a not-yet-formed "
    "adjacent bend (UNKNOWN by contract), NOMINAL_UNAVAILABLE from frozen E1, disputed 10839000 truth rows, stage-convention tab lines, the rolled 28870000 part. "
    "False-PASS proxy 3 (1 stage convention, 2 disputed truth rows); strict false PASS on complete BROAD scans 0.",
    "Ledger check ok=false: CONTRACTS_CODE_CHANGED_SINCE_FREEZE (contracts/production edited by another workflow 2026-10-19; A1 did not touch it) -> physical and G1 "
    "scorecards not qualifying until the freeze is re-anchored.",
]
gor = h["BROAD_scans_gold_of_record_report_only"]
pba = h["per_bend_given_cad_all_A1"]
gs = h["GOLD_SET_V1"]
peak = max(v for kk, v in rt.items() if kk.endswith("_gib") and isinstance(v, float))
e["summary"] = (
    f"A1 iter-7 MEASURE ({e['engine_version']}, config {e['config_sha256'][:8]}; fp.py engine fingerprint 6297d8e3 = change step final, unchanged during measure; "
    "goldset/per-bend/subset outputs byte-identical to the change step's final validation, G1 accuracy gates and physical G2/G3 gate records identical). "
    "All 27 A1 cases (26 OK, 1 compile BLOCKED), serially under the shared lock, OMP_NUM_THREADS=1. "
    f"BROAD frozen A1 list (24 line cases / 14 families): family-macro {B['engine']['family_macro_f1']}, micro {B['engine']['micro_f1']} (P {B['engine']['precision']}, R {B['engine']['recall']}; "
    f"TP/FP/FN {B['engine']['tp_fp_fn']}) vs best F1 lane {B['best_f1_lane_same_scans']['family_macro_f1']} / {B['best_f1_lane_same_scans']['micro_f1']} on the same scans (unchanged from iter 6). "
    f"Same scans on the gold of record (report-only): micro {gor['engine']['micro_f1']}, family-macro {gor['engine']['family_macro_f1']} vs best lane {gor['best_f1']['micro_f1']} / {gor['best_f1']['family_macro_f1']}. "
    f"Gold11 relabel (gold of record) {h['Gold11_relabel_v1_9_A1_cases']['engine_f1']} (P 0.975, R 1.00) vs best lane 0.547; human Gold11 0.391 vs 0.473 (frozen G2(a) gate -> FAIL). "
    f"GOLD_SET_V1 ({gs['scans']} A1 scans): {gs['engine_micro']} micro / {gs['engine_family_macro']} family-macro (TP/FP/FN {gs['tp_fp_fn']}) vs best lane {gs['best_f1_micro']} / {gs['best_f1_family_macro']}. "
    f"Per bend given CAD (all A1, {pba['bends']} bends): {pba['riser_aware']} riser-aware, {pba['strict']} strict, family-macro {pba['family_macro_riser_aware']}, false-PASS proxy {pba['false_pass_proxy']} "
    "(1 stage convention, 2 disputed truth rows); complete-BROAD strict false PASS 0. Iter-7 change shows in angles: physical MAE vs nominal 0.946 -> 0.898 deg (159 angles), mean U 1.018 -> 0.956. "
    f"G1 regression DEV PASS (accuracy gates identical to iter 6 and the change step). G2 FAIL only on (a) human-gold gate and (f) different-stage repeats (P95 {r3(k['G2f']['e3_p95_angle_deg'])} deg, was 0.426); "
    f"(b)(c)(d)(e) + compile PASS (known-good angle MAE {r3(k['G2c']['e3_mae_deg'])} deg vs legacy common {r3(k['G2c']['common_legacy_mae_deg'])} informational, flange P95 {r3(k['G2d']['p95_abs_mm'])} mm, "
    f"P95 {round(rt['physical_e3_p95_wall_s_G2'], 1)} s). G3 FAIL: state match 0.959, known-good angle MAE {r3(k['G3_kg_angle']['e3_mae_deg'])} deg (was 1.186), unformed-as-defect 2 (disputed rows); "
    f"coverage, flange P95 {r3(k['G3_kg_flange_p95'])} mm, blank, counts PASS. G5 {g5['gate']} ({g5['backend_tests']['counts'].get('passed')} backend tests, no new failures). "
    f"Ledger ok=false (contracts edited by another workflow). Peak RSS {peak} GiB.")
(L / "extra.json").write_text(json.dumps(e, indent=1))
print(e["summary"])
