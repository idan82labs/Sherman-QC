"""Operator text for the A1 iter-5 MEASURE scorecard; merges logs/extra_auto.json (computed) into logs/extra.json."""
import json
from pathlib import Path

L = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_5/measure/logs")
e = json.loads((L / "extra_auto.json").read_text())
h = e["headline"]
B = h["BROAD_A1_frozen_gold"]
rt = e["runtime_rss"]
e["g5"] = {"verdict": "FAIL", "manifest": "PASS (1 allowlisted add: scripts/run_production_line_checks.sh)", "tsc": "PASS (new=0)",
           "lint": "PASS (new=0)", "vendor_scripts": "PASS (5 NOT_RUN matching baseline)",
           "backend_tests": "FAIL: 10 NEW failures in tests/test_vendor_unfold_api_contract.py, the same 10 as iterations 0-4",
           "cause": "non-engine: apps/api/services/vendor_setup_service.py (another workflow's edit; the test AST-extracts get_vendor_setup and hits a NameError). The iter-5 diff touches only domains/production/engine/inspect/ and engine config."}
e["top_failures"] = [
    "BROAD recall on the frozen BROAD gold is 0.729 (105/144; owner direction asks >= 0.80): of the 39 FNs, 38 have no matching gold-of-record line on the same scan "
    "(human-Gold11 lines 4-15 mm off the apex, lines spanning several folds, lines where no fold exists, pre-audit split lines on 11979003_C) and 1 is a gold-of-record fold the recipe excludes "
    "(28870000 FULL). The only engine misses a better engine could recover are the 2 wrong-way hooks on 23553000 SCAN_2 (pc_186463746daa B2/B14: UNEXPECTED_FORMED with all characteristics UNKNOWN; overlays 06, 07). "
    "Recall >= 0.80 on the frozen BROAD gold is not reachable without fitting label errors; on the gold of record for the same 24 scans recall is 0.981 (best F1 lane 0.475).",
    "G3 FAIL, state match 118/123 = 0.959: 4 mismatches on 10839000 scan_2 (pc_cec08dfca9ab B6/B8/B9/B11; the hole evidence says the frozen truth rows are the 180-deg twin and look swapped; needs owner re-adjudication, overlays 01-03) "
    "and 28870000 SCAN_1 (pc_3bde9996f14d B1, rolled part read WRONG_PART_SUSPECTED -> UNKNOWN vs NOT_FORMED; overlay 20). unformed_reported_as_defect = 2 (the disputed cec08 B6/B11).",
    "G3 known-good formed-bend angle MAE 1.186 deg FAIL (31 bends, 28 on 10839000: 12 joggle step-equivalent angles MAE 1.53 deg, 16 90-deg bends reading 1-2 deg overbent on all three scans; legacy common-bend MAE 1.22 deg sees the same).",
    "G2(a) FAIL on the frozen scorer's human Gold11 gate: F1 0.391 (P 0.342, R 0.458) vs best F1 lane 0.473; on gold11_relabel_v1 (owner's gold of record) F1 0.987 (P 0.975, R 1.000) vs best lane 0.547. Moving the gate of record needs the owner's sign-off on the relabel.",
    "G2(f) FAIL: repeatability P95 angle 0.426 deg on 15651004, whose three 'repeat' scans are three different stages (B1 station spread is 0.1-0.2 deg within each scan, so the 0.4 deg is systematic between scans).",
    "38172000 SCAN_1 (pc_07062eee17aa) B1/B9: tab lines collinear with the 214 mm folds at that stage; the relabel draws each pair as one 305 mm line, so the one-to-one matcher leaves them unmatched -> 2 GOLD_SET_V1 FPs and the false-PASS proxy on B1 (flange A +0.21 mm PASS on a line the gold contains). Stage-level convention question. Overlays 04, 05.",
    "50757000_02_X (pc_752e2bd54aad) B1/B4: positions and B4 side A read ~4.8-5.0 mm off nominal (FAIL) with angles and other flanges in tolerance - consistent with one off-nominal backgauge, not verified against physical truth; 38172000_01_X (pc_4f8188b4d1a8) B5 side B -8.28 mm FAIL + angle +4.76 deg pre-existing since iter 3. Overlays 08, 09, 11.",
    "Per-bend correctness given CAD 181/203 = 0.892 riser-aware (0.837 strict): 8 not-correct bends are flange sides ending at an adjacent bend not formed at that stage (UNKNOWN by contract), 4 NOMINAL_UNAVAILABLE from frozen E1, 5 disputed 10839000 truth rows, 2 stage-convention lines, 1 rolled part, 2 wrong-way hooks (the only engine-limited ones).",
    "G5 FAIL: 10 NEW backend failures in tests/test_vendor_unfold_api_contract.py (vendor_setup_service.py, another workflow; not engine).",
]
gor = h["BROAD_scans_gold_of_record_report_only"]
pba = h["per_bend_given_cad_all_A1"]
gs = h["GOLD_SET_V1"]
e["summary"] = (
    f"A1 iter-5 MEASURE ({e['engine_version']}, config {e['config_sha256'][:8]}; engine fingerprint {e['matches_change_step']['engine_fingerprint'][:8]} = change step final, unchanged during measure; "
    "every scoring output byte-identical to the change step's final validation). All 27 A1 cases (26 OK, 1 compile BLOCKED), serially under the shared lock. "
    f"BROAD frozen A1 list (24 line cases / 14 families): family-macro {B['engine']['family_macro_f1']}, micro {B['engine']['micro_f1']} (P {B['engine']['precision']}, R {B['engine']['recall']}; TP/FP/FN {B['engine']['tp_fp_fn']}) "
    f"vs best F1 lane {B['best_f1_lane_same_scans']['family_macro_f1']} / {B['best_f1_lane_same_scans']['micro_f1']} on the same scans (iter 4: 0.806 / 0.678). "
    f"Same scans on the gold of record per scan (report-only): P {gor['engine']['precision']}, R {gor['engine']['recall']}, micro {gor['engine']['micro_f1']}, family-macro {gor['engine']['family_macro_f1']} vs best lane {gor['best_f1']['micro_f1']} / {gor['best_f1']['family_macro_f1']}. "
    "Recall 0.80 on the frozen BROAD gold is label-limited: 38 of 39 FNs have no gold-of-record line, the engine-recoverable misses are 2 wrong-way hooks. "
    f"Gold11 relabel (gold of record) {h['Gold11_relabel_v1_9_A1_cases']['engine_f1']} (P 0.975, R 1.00) vs best lane 0.547; human Gold11 0.391 vs 0.473 (frozen G2(a) gate -> FAIL). "
    f"GOLD_SET_V1 ({gs['scans']} A1 scans): {gs['engine_micro']} micro / {gs['engine_family_macro']} family-macro (TP/FP/FN {gs['tp_fp_fn']}) vs best lane {gs['best_f1_micro']} / {gs['best_f1_family_macro']}. "
    f"Per bend given CAD (all A1, {pba['bends']} bends): {pba['riser_aware']} riser-aware, {pba['strict']} strict, family-macro {pba['family_macro_riser_aware']}, false-PASS proxy {pba['false_pass_proxy']} (all explained: 1 stage convention, 2 disputed truth rows); complete BROAD 80 bends 63/80 vs BROAD gold, 75/80 vs any gold, strict false PASS 0. "
    "G1 regression DEV PASS (every accuracy gate identical to iter 4 and the change step). G2 FAIL only on (a) human-gold gate and (f) different-stage repeats; (b)(c)(d)(e) + compile PASS (angle MAE 0.795 deg vs legacy common 1.291, flange P95 0.392 mm, "
    f"P95 {round(rt['physical_e3_p95_wall_s_G2'], 1)} s). G3 FAIL: state match 0.959, known-good angle MAE 1.186 deg, unformed-as-defect 2 (disputed rows); coverage, flange P95 0.368 mm, blank and counts PASS. "
    f"G5 FAIL (10 non-engine backend tests). Peak RSS {max(v for k, v in rt.items() if k.endswith('_gib') and isinstance(v, float))} GiB.")
(L / "extra.json").write_text(json.dumps(e, indent=1))
print(e["summary"])
