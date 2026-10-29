"""Operator-computed fields of the A1 iter-5 MEASURE scorecard (rss/runtime, G1 diff vs iter 4, match to the change
step, FN/FP audit summary, text). Numbers are computed from the measure outputs; the text quotes them."""
import hashlib
import json
from collections import Counter
from pathlib import Path

R = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/a1/iter_6")
L = R / "measure" / "logs"
RAW = R / "measure" / "raw"
V = R / "validation"
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6m")
j = lambda p: json.loads(Path(p).read_text())  # noqa: E731


def rss(name):
    for line in (L / f"{name}.log").read_text().splitlines():
        if "maximum resident set size" in line:
            return round(int(line.split()[0]) / 2**30, 3)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


g1 = j(RAW / "g1_regression_scorecard.json")
gates = {g["id"]: g for g in g1["gates"]}
p4 = j(R.parent / "iter_5" / "scorecard.json")["G1_synthetic_regression_DEV"]["gates"]
diff = {k: [p4.get(k, {}).get("value"), v.get("value")] for k, v in gates.items() if p4.get(k, {}).get("value") != v.get("value")}
c5 = {g["id"]: g for g in j(V / "g1_regression_scorecard_final.json")["gates"]}
acc_ids = [k for k in gates if not k.startswith("G1.RUNTIME") and k not in ("G1.TIER_BUDGET",)]
g1_same_as_change = all(gates[k].get("value") == c5.get(k, {}).get("value") for k in acc_ids)
phys = j(RAW / "physical_scorecard.json")
sub = j(RAW / "subset_scores.json")
gs = j(RAW / "goldset_scores.json")
pba = j(RAW / "per_bend_all.json")["summary"]
pbx = j(RAW / "per_bend_ext.json")["summary"]
three = j(RAW / "broad_three_golds.json")
audit = j(RAW / "broad_fn_fp_audit.json")
fp_s, fp_e = j(L / "fingerprints_start.json"), j(L / "fingerprints_end.json")
fp_change = j(V / "code_fingerprints_chain_end.json")  # different hash recipe for engine; compile/eval/contracts comparable

# byte-identity vs the change step's final validation outputs
same = {}
for mine, theirs in (("goldset_scores.json", "goldset_scores_final.json"), ("per_bend_all.json", "per_bend_all_final.json")):
    same[mine] = sha(RAW / mine) == sha(V / theirs)
same["broad_three_golds.json"] = sha(RAW / "broad_three_golds.json") == sha(V / "broad_three_golds_final.json")
same["broad_fn_fp_audit.json"] = sha(RAW / "broad_fn_fp_audit.json") == sha(V / "broad_fn_fp_audit_final.json")

# engine version / config hash from the E3 results
vers, cfgs = Counter(), Counter()
for rj in (W / "phys" / "physical" / "E3").glob("*/engine/result.json"):
    d = j(rj)
    prov = d.get("provenance") or d.get("engine") or {}
    vers[str(d.get("engine_version") or prov.get("engine_version"))] += 1
    cfgs[str(d.get("config_sha256") or prov.get("config_sha256"))] += 1

# FN / FP audit summary
fn = fn_in = fp = fp_in = 0
fn_not_in = []
for r in audit:
    if r.get("zero"):
        continue
    for f in r["fn_lines"]:
        fn += 1
        if f["matches_record_gold"]:
            fn_in += 1
        else:
            fn_not_in.append(f"{r['case_id']}:{f['gold_id']} ({r['record']})")
    for f in r["fp_lines"]:
        fp += 1
        fp_in += 1 if f["matches_record_gold"] else 0

fz = three["frozen"]
gor = three["gold_of_record"]
rls = three["relabel_substituted"]
B_e, B_b = sub["BROAD_all26"]["engine"], sub["BROAD_all26"]["best_f1"]
G2a = sub["G2a"]["all9"]
G2 = phys["G2"]
G3 = phys["G3"]
pb = sub["per_bend_given_cad"]
r3 = lambda x: round(x, 3)  # noqa: E731

extra = {
    "label": "consumed development (engineering ledger); never unseen-family accuracy. Iteration 5 = engine e3-1.7.0 after the "
             "iter-5 change (CHANGE.md fixes A one-skin re-read, B wide search for astray formed readings, C signed sweep / "
             "wrong-way FAIL semantics). All 27 A1 cases run (26 compiled; pc_b4719db3dd69 compile BLOCKED). Re-measured "
             "independently, serially under the shared lock, OMP_NUM_THREADS=1.",
    "engine_version": "/".join(sorted(vers)) if vers else None,
    "engine_versions_in_results": dict(vers),
    "config_sha256": "/".join(sorted(cfgs)) if cfgs else None,
    "config_hashes_in_results": dict(cfgs),
    "matches_change_step": {
        "fingerprint_start_eq_end": fp_s == fp_e,
        "fingerprint_eq_change_step_final": fp_s.get("engine", "").startswith("b9951fec") and all(fp_s.get(k) == fp_change.get(k) for k in ("compile", "eval", "contracts")),
        "fingerprint_note": "engine fingerprint b9951fec = CHANGE.md iter-6 final (freeze recipe); compile/eval/contracts compared with validation/code_fingerprints_chain_end.json",
        "engine_fingerprint": fp_s.get("engine"),
        "byte_identical_to_change_step": same,
        "g1_accuracy_gates_identical_to_change_step": g1_same_as_change,
    },
    "g1_vs_iter5": {"changed_gate_values_[iter5, iter6]": diff},
    "g2a_why_fail": (
        f"G2(a) in the frozen scorer gates on the HUMAN Gold11 (historical diagnostic per owner): F1 {r3(G2a['human']['engine']['f1'])} "
        f"(P {r3(G2a['human']['engine']['precision'])}, R {r3(G2a['human']['engine']['recall'])}; "
        f"{G2a['human']['engine']['true_positive']}/{G2a['human']['engine']['false_positive']}/{G2a['human']['engine']['false_negative']}) "
        f"vs best F1 lane {r3(G2a['human']['best_f1']['f1'])} -> FAIL. On gold11_relabel_v1 (gold of record, OWNER_DECISION_2026-10-17) "
        f"F1 {r3(G2a['relabel']['engine']['f1'])} (P {r3(G2a['relabel']['engine']['precision'])}, R {r3(G2a['relabel']['engine']['recall'])}; "
        f"{G2a['relabel']['engine']['true_positive']}/{G2a['relabel']['engine']['false_positive']}/{G2a['relabel']['engine']['false_negative']}) "
        f"vs best lane {r3(G2a['relabel']['best_f1']['f1'])} -> would PASS F1>=0.80 and P>=0.90. The gate of record moves only with the owner's sign-off."),
    "g2_population_note": "all 13 G2 E3 runs complete; 1 compile BLOCKED (pc_b4719db3dd69, explicit refusal) stays in the denominators. "
                          "G2(f) FAIL on 15651004 whose three 'repeats' are three different stages (CHANGE.md).",
    "fp_analysis": (f"BROAD frozen gold: {fz['e3']['fp']} FP / {fz['e3']['fn']} FN. FN audit (frozen matcher vs gold of record on the same scan): "
                    f"{fp_in} of {fp} FPs match a gold-of-record line (the older BROAD gold lacks them or places them off the apex); "
                    f"on the gold of record per scan the same engine lines score P {gor['e3']['precision']} / R {gor['e3']['recall']} "
                    f"(best F1 lane {gor['best_f1']['precision']} / {gor['best_f1']['recall']})."),
    "fn_audit": {
        "definition": "each unmatched frozen-BROAD gold line (FN) and unmatched engine line (FP), compared with the gold of record on the same scan (Gold11 relabel v1 on the 9 Gold11 scans, GOLD_SET_V1 where the scan is in it, else none) using the frozen matcher (tools/fn_audit.py, re-run in this measure step)",
        "broad_fn": fn, "broad_fp": fp,
        "fn_that_a_gold_of_record_line_also_matches": fn_in,
        "fn_with_no_matching_gold_of_record_line": fn - fn_in,
        "fp_matching_a_gold_of_record_line": fp_in,
        "fp_not_matching_any_gold_of_record_line": fp - fp_in,
        "fn_lines_without_gold_of_record_match": fn_not_in[:60],
        "classification_of_record": "CHANGE.md 'BROAD recall audit': 36 of 39 FNs are label problems (20 human-Gold11 lines 4.2-14.7 mm off the apex, 6 human lines spanning several folds, 6 human lines where no fold exists, 4 pre-audit bench_lines splits of window-interrupted folds), 3 engine misses (2 wrong-way hooks 23553000 SCAN_2, 1 recipe-excluded fold 28870000 FULL); 52 of 54 FPs match a gold-of-record line, 2 are 38172000 SCAN_1 tab lines B1/B9 (stage-level convention). This measure step reproduces the audit file byte for byte (see matches_change_step) and does not re-classify.",
    },
    "false_pass_definitions": {
        "loose_proxy (frozen iter-0 definition, BROAD gold only)": f"PASS on a bend whose line matches no BROAD gold line: {pb['false_pass_proxy']} (complete BROAD scans, {pb['bends']} bends)",
        "strict_proxy (complete BROAD scans)": f"PASS on a published line that no gold of record (BROAD, Gold11 relabel, GOLD_SET_V1) confirms: {pbx['false_pass_proxy_strict']}; PASS without a published line: {pbx['pass_without_published_line']}",
        "all_A1 proxy (per_bend_all, primary)": f"PASS on a bend the frozen G3 truth calls NOT_FORMED, or on a formed bend whose line the gold of record does not confirm: {pba['false_pass_proxy']} ({', '.join(pba['false_pass_proxy_bends'])}; B1 of pc_07062eee17aa = 38172000 SCAN_1 tab line inside the relabel's 305 mm line, stage-level convention; pc_cec08dfca9ab B6/B11 = disputed 10839000 scan_2 truth rows contradicted by the hole evidence)",
        "caveat": "no physical dimensional truth exists for these scans; FAIL decisions are unverified (false FAILs possible)"},
    "per_bend_all_definition": "every A1 case the engine ran (26 OK runs, 203 bends): expected state = frozen G3 truth row (staged scans) else FORMED (complete scans); NOT_DETERMINABLE truth not scored. Correct(FORMED) = engine FORMED-like, angle MEASURED, both flange sides MEASURED (riser-aware: a joggle riser's own side UNKNOWN(UNSUPPORTED_FEATURE) allowed) and line confirmed by GOLD_SET_V1 / Gold11 relabel where they exist. Correct(NOT_FORMED) = NOT_YET_FORMED. Primary per-bend-given-CAD figure.",
    "goldset_scope": f"GOLD_SET_V1 scans that are A1 cases with STEP and an OK engine run: {gs['e3']['scans']} scans / {gs['e3']['families']} families ({len(gs['skipped_a1_cases'])} skipped); DEV-CHECK never read",
    "runtime_rss": {"physical_e3_p95_wall_s_G2": G2["runtime"]["p95_wall_s"], "physical_e3_p95_wall_s_BROAD": phys["BROAD"]["runtime"]["p95_wall_s"],
                    "physical_e3_max_wall_s": phys["BROAD"]["runtime"]["max_wall_s"], "physical_e3_crashes": len(phys["BROAD"]["runtime"]["crashes"]),
                    "physical_e3_runs": phys["BROAD"]["runtime"]["runs"],
                    "legacy_control_p95_s_informational": G2["runtime"].get("legacy_control_p95_s"),
                    "physical_run_peak_rss_gib": rss("run_physical"), "g1_runner_peak_rss_gib": rss("g1"),
                    "synthetic_p95_wall_s": gates["G1.RUNTIME.P95"]["value"], "synthetic_engine_peak_rss_gib": round(gates["G1.RUNTIME.RSS"]["value"] / 2**30, 3),
                    "compile_frozen_peak_rss_gib": rss("compile_frozen"), "overlays_peak_rss_gib": rss("overlays"), "g5_peak_rss_gib": rss("g5"),
                    "lock": "every heavy step serial under /private/tmp/claude-501/sherman_geom.lock, OMP_NUM_THREADS=1 (measure/tools/measure.sh)"},
    "headline": {
        "BROAD_A1_frozen_gold": {"engine": {"family_macro_f1": r3(B_e["family_macro_f1"]), "micro_f1": r3(B_e["micro"]["f1"]), "precision": r3(B_e["micro"]["precision"]),
                                            "recall": r3(B_e["micro"]["recall"]), "tp_fp_fn": [B_e["true_positive"], B_e["false_positive"], B_e["false_negative"]],
                                            "line_cases": B_e["line_cases"], "families": B_e["families"]},
                                 "best_f1_lane_same_scans": {"family_macro_f1": r3(B_b["family_macro_f1"]), "micro_f1": r3(B_b["micro"]["f1"]), "precision": r3(B_b["micro"]["precision"]),
                                                             "recall": r3(B_b["micro"]["recall"]), "tp_fp_fn": [B_b["true_positive"], B_b["false_positive"], B_b["false_negative"]]}},
        "BROAD_scans_gold_of_record_report_only": {"engine": {k: gor["e3"][k] for k in ("precision", "recall", "micro_f1", "family_macro_f1", "tp", "fp", "fn")},
                                                   "best_f1": {k: gor["best_f1"][k] for k in ("precision", "recall", "micro_f1", "family_macro_f1", "tp", "fp", "fn")}},
        "BROAD_scans_relabel_substituted_report_only": {"engine": {k: rls["e3"][k] for k in ("precision", "recall", "micro_f1", "family_macro_f1")},
                                                        "best_f1": {k: rls["best_f1"][k] for k in ("precision", "recall", "micro_f1", "family_macro_f1")}},
        "Gold11_relabel_v1_9_A1_cases": {"engine_f1": r3(G2a["relabel"]["engine"]["f1"]), "precision": r3(G2a["relabel"]["engine"]["precision"]),
                                         "recall": r3(G2a["relabel"]["engine"]["recall"]), "best_f1_lane": r3(G2a["relabel"]["best_f1"]["f1"])},
        "Gold11_human_9_A1_cases": {"engine_f1": r3(G2a["human"]["engine"]["f1"]), "precision": r3(G2a["human"]["engine"]["precision"]),
                                    "recall": r3(G2a["human"]["engine"]["recall"]), "best_f1_lane": r3(G2a["human"]["best_f1"]["f1"])},
        "GOLD_SET_V1": {"scans": gs["e3"]["scans"], "engine_micro": r3(gs["e3"]["micro_f1"]), "engine_family_macro": r3(gs["e3"]["family_macro_f1"]),
                        "tp_fp_fn": [gs["e3"]["tp"], gs["e3"]["fp"], gs["e3"]["fn"]],
                        "best_f1_micro": r3(gs["best_f1"]["micro_f1"]), "best_f1_family_macro": r3(gs["best_f1"]["family_macro_f1"])},
        "per_bend_given_cad_all_A1": {"bends": pba["bends"], "riser_aware": round(pba["rate_riser_aware"], 4), "strict": round(pba["rate_strict"], 4),
                                      "family_macro_riser_aware": round(pba["family_macro_riser_aware"], 4), "false_pass_proxy": pba["false_pass_proxy"]},
        "per_bend_complete_BROAD": {"bends": pb["bends"], "correct_vs_broad_gold": f"{pb['correct']}/{pb['bends']}",
                                    "correct_vs_any_gold": f"{pbx['correct_vs_any_gold_of_record']}/{pbx['bends']}", "strict_false_pass": pbx["false_pass_proxy_strict"]},
    },
}
(L / "extra_auto.json").write_text(json.dumps(extra, indent=1, default=str))
print(json.dumps({k: extra[k] for k in ("engine_versions_in_results", "config_hashes_in_results", "matches_change_step", "g1_vs_iter5", "runtime_rss", "headline")}, indent=1, default=str))
print(json.dumps(extra["fn_audit"], indent=1)[:1500])
