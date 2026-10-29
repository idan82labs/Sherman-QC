"""Markdown result tables: second start (fixes 1-8) vs final engine on all A1 cases, with the best F1 lane."""
import json, sys
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4"
W = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter4"
FIN = sys.argv[1] if len(sys.argv) > 1 else "v4"


def load(name):
    if name == "prev":
        an, sc = W + "/phys/analysis", W + "/phys/physical_scorecard.json"
    else:
        an, sc = f"{S}/phys/analysis_{name}", f"{S}/phys/physical_scorecard_{name}.json"
    return (json.load(open(f"{an}/goldset_scores.json")), json.load(open(f"{an}/subset_scores.json")), json.load(open(sc)))


P, F = load("prev"), load(FIN)
pba = json.load(open(f"{S}/phys/analysis_{FIN}/per_bend_all.json"))["summary"]
try:
    pbp = json.load(open(f"{S}/phys/analysis_prevall/per_bend_all.json"))["summary"]
except Exception:
    pbp = None


def g2(x):
    return x[2]["G2"]


def row(name, a, b):
    print(f"| {name} | {a} | {b} |")


print("**G2 (physical dev; frozen thresholds)**\n")
print("| Sub-gate | First full run (fixes 1-8) | **Final engine** |")
print("|---|---|---|")
for key, lab in (("a_line_metric", "(a) line metric (gate of record: human Gold11)"), ("b_angle_coverage", "(b) angle coverage"),
                 ("b_flange_coverage", "(b) flange coverage"), ("c_angle_known_good", "(c) known-good angle"),
                 ("d_flange_known_good", "(d) known-good flange"), ("e_runtime_and_failures", "(e) runtime and failures"),
                 ("f_repeatability", "(f) repeatability"), ("compile_rate", "compile rate")):
    def val(x):
        g = g2(x)
        v = g["gates"][key]
        if key == "b_angle_coverage":
            c = g["coverage"]["angle"]; v += f" {c['measured']}/{c['planned']}"
        elif key == "b_flange_coverage":
            c = g["coverage"]["flange_sides"]; v += f" {c['measured']}/{c['planned']}"
        elif key == "c_angle_known_good":
            a = g["angle_known_good"]; v += f", MAE {a['e3_mae_deg']:.3f}° on {a['e3_measured']} bends"
        elif key == "d_flange_known_good":
            d = g["flange_known_good"]; v += f", P95 {d['p95_abs_mm']:.3f} mm, max {d['max_abs_mm']:.3f} mm"
        elif key == "e_runtime_and_failures":
            r = g["runtime"]; v += f", P95 {r['p95_wall_s']:.1f} s, max {r['max_wall_s']:.1f} s, {len(r['crashes'])} crashes"
        elif key == "f_repeatability":
            r = g["repeatability"]; v += f", P95 {r['e3_p95_angle_deg']:.3f}° / {r['e3_p95_flange_mm']:.3f} mm"
        elif key == "a_line_metric":
            h = g["line_metric_dual_report"]["human_all_cases"]; v += f", F1 {h['f1']:.3f}"
        return v
    row(lab, val(P), val(F))
print()
print("**G3 (frozen truth table, 13 cases)**\n")
print("| Gate | First full run (fixes 1-8) | **Final engine** |")
print("|---|---|---|")
for key in F[2]["G3"]["gates"]:
    def val(x):
        g = x[2]["G3"]; v = g["gates"][key]
        if key == "state_match_rate":
            v += f" {g['state_match_rate']:.3f} ({g['matches']}/{g['bends_scored']})"
        elif key == "formed_bends_angle_coverage":
            c = g["formed_bends_coverage"]["angle"]; v += f" {c['measured']}/{c['planned']}"
        elif key == "formed_bends_flange_coverage":
            c = g["formed_bends_coverage"]["flange_sides"]; v += f" {c['measured']}/{c['planned']}"
        elif key == "formed_bends_known_good_angle_mae":
            v += f" {g['formed_bends_known_good_angle']['e3_mae_deg']:.3f}°"
        elif key == "formed_bends_known_good_flange_p95":
            v += f" {g['formed_bends_known_good_flange']['p95_abs_mm']:.3f} mm"
        elif key == "unformed_reported_as_defect":
            v += f" ({len(g['unformed_reported_as_defect'])})"
        return v
    row(key, val(P), val(F))
print()
print("**Line localisation (apex locus, frozen 5°/4 mm/0.6 matcher), engine vs the best F1 lane on the same scans**\n")
print("| Gold | Scans | First full run (fixes 1-8) | **Final engine** | Best F1 lane |")
print("|---|---|---|---|---|")
gp, gf = P[0], F[0]
print(f"| GOLD_SET_V1 (A1 scans with STEP) | {gf['e3']['scans']} scans, {gf['e3']['families']} families | micro {gp['e3']['micro_f1']:.3f}, family-macro {gp['e3']['family_macro_f1']:.3f} (TP {gp['e3']['tp']}, FP {gp['e3']['fp']}, FN {gp['e3']['fn']}) | **micro {gf['e3']['micro_f1']:.3f}, family-macro {gf['e3']['family_macro_f1']:.3f}** (TP {gf['e3']['tp']}, FP {gf['e3']['fp']}, FN {gf['e3']['fn']}) | micro {gf['best_f1']['micro_f1']:.3f}, family-macro {gf['best_f1']['family_macro_f1']:.3f} |")
rp, rf = P[2]["G2"]["line_metric_dual_report"]["relabel"], F[2]["G2"]["line_metric_dual_report"]["relabel"]
bl = F[1]["G2a"]["all9"]["relabel"]["best_f1"]
print(f"| Gold11 relabel v1 (gold of record) | 9 | F1 {rp['f1']:.3f} (P {rp['precision']:.3f}, R {rp['recall']:.3f}) | **F1 {rf['f1']:.3f} (P {rf['precision']:.3f}, R {rf['recall']:.3f})** | F1 {bl['f1']:.3f} (P {bl['precision']:.3f}, R {bl['recall']:.3f}) |")
hp, hf = P[2]["G2"]["line_metric_dual_report"]["human_all_cases"], F[2]["G2"]["line_metric_dual_report"]["human_all_cases"]
bh = F[1]["G2a"]["all9"]["human"]["best_f1"]
print(f"| Human Gold11 (historical diagnostic; G2(a) gate of record in the frozen scorer) | 9 | F1 {hp['f1']:.3f} | F1 {hf['f1']:.3f} (P {hf['precision']:.3f}, R {hf['recall']:.3f}) | F1 {bh['f1']:.3f} |")
bp, bf_ = P[1]["BROAD_all26"], F[1]["BROAD_all26"]
print(f"| BROAD gold (older labels) | {bf_['engine']['line_cases']} scans, {bf_['engine']['families']} families | micro {bp['engine']['micro']['f1']:.3f}, family-macro {bp['engine']['family_macro_f1']:.3f} | **micro {bf_['engine']['micro']['f1']:.3f}, family-macro {bf_['engine']['family_macro_f1']:.3f}** (P {bf_['engine']['micro']['precision']:.3f}, R {bf_['engine']['micro']['recall']:.3f}) | micro {bf_['best_f1']['micro']['f1']:.3f}, family-macro {bf_['best_f1']['family_macro_f1']:.3f} |")
print()
print("**Per-bend correctness given CAD (all A1 cases the engine ran; report-only)**\n")
print("| | First full run (fixes 1-8) | **Final engine** |")
print("|---|---|---|")
if pbp:
    row("bends scored (G3 truth where it exists, else complete-scan FORMED)", pbp["bends"], pba["bends"])
    row("correct, strict (angle + both sides MEASURED, line confirmed)", f"{pbp['correct_strict']} ({pbp['rate_strict']:.3f})", f"**{pba['correct_strict']} ({pba['rate_strict']:.3f})**")
    row("correct, riser-aware (a joggle riser's own length UNSUPPORTED_FEATURE allowed)", f"{pbp['correct_riser_aware']} ({pbp['rate_riser_aware']:.3f})", f"**{pba['correct_riser_aware']} ({pba['rate_riser_aware']:.3f})**")
    row("family-macro, riser-aware", f"{pbp['family_macro_riser_aware']:.3f}", f"**{pba['family_macro_riser_aware']:.3f}**")
    row("false-PASS proxy", f"{pbp['false_pass_proxy']} {pbp['false_pass_proxy_bends']}", f"{pba['false_pass_proxy']} {pba['false_pass_proxy_bends']}")
