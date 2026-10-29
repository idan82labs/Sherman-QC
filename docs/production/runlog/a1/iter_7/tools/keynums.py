"""Key numbers of an A1 physical work dir (report-only). usage: keynums.py PHYS_DIR [TAG]"""
import json, sys
from pathlib import Path
P = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ""
sc = json.loads((P / (f"physical_scorecard_{tag}.json" if tag else "physical_scorecard.json")).read_text())
an = P / (f"analysis_{tag}" if tag else "analysis")
out = {}
g2, g3 = sc["G2"], sc["G3"]
out["G2_gates"] = g2["gates"]
ak = g2["angle_known_good"]
out["G2c"] = {k: ak[k] for k in ("e3_mae_deg", "common_e3_mae_deg", "common_legacy_mae_deg", "e3_measured", "gate")}
out["G2c_per_part"] = {f: round(v["e3_mae_deg"], 3) if v["e3_mae_deg"] is not None else None for f, v in ak["per_part"].items()}
out["G2d"] = {k: g2["flange_known_good"][k] for k in ("p95_abs_mm", "max_abs_mm", "measured_sides", "gate")}
out["G2b"] = {"angle": g2["coverage"]["angle"]["rate"], "flange": g2["coverage"]["flange_sides"]["rate"]}
out["G2e"] = {k: g2["runtime"][k] for k in ("p95_wall_s", "max_wall_s", "gate")}
out["G2f"] = {k: g2["repeatability"][k] for k in ("e3_p95_angle_deg", "e3_p95_flange_mm", "gate")}
out["G3_gates"] = g3["gates"]
out["G3_match"] = f'{g3["matches"]}/{g3["bends_scored"]}'
kg = g3["formed_bends_known_good_angle"]
out["G3_kg_angle"] = {k: kg[k] for k in ("e3_mae_deg", "common_e3_mae_deg", "common_legacy_mae_deg", "e3_measured")}
out["G3_kg_per_part"] = {f: (round(v["e3_mae_deg"], 3) if v["e3_mae_deg"] is not None else None) for f, v in kg["per_part"].items()}
out["G3_kg_flange_p95"] = g3["formed_bends_known_good_flange"]["p95_abs_mm"]
out["G3_cov"] = {"angle": g3["formed_bends_coverage"]["angle"]["rate"], "flange": g3["formed_bends_coverage"]["flange_sides"]["rate"]}
mism = [(r["case_ref"], r["bend_uid"][:10], r["truth"], r["engine_state"]) for r in g3["rows"] if r["result"] != "MATCH"]
out["G3_mismatch"] = mism
try:
    ss = json.loads((an / "subset_scores.json").read_text())
    b = ss["BROAD_all26"]["engine"]
    out["BROAD"] = {"tp": b["true_positive"], "fp": b["false_positive"], "fn": b["false_negative"], **{k: round(v, 4) for k, v in b["micro"].items()}, "fam_macro": round(b["family_macro_f1"], 4)}
    g = ss["G2a"]["all9"]
    out["G2a_all9"] = {k: {kk: round(vv["engine"]["f1"], 4) for kk, vv in [("f1", v)]} for k, v in g.items() if isinstance(v, dict) and "engine" in v}
except Exception as e:
    out["BROAD_err"] = str(e)
try:
    gs = json.loads((an / "goldset_scores.json").read_text())["e3"]
    out["GOLD_SET_V1"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gs.items() if k in ("tp", "fp", "fn", "precision", "recall", "micro_f1", "family_macro_f1")}
except Exception as e:
    out["GS_err"] = str(e)
try:
    pb = json.loads((an / "per_bend_all.json").read_text())["summary"]
    out["per_bend"] = {k: pb[k] for k in ("bends", "correct_strict", "correct_riser_aware", "rate_riser_aware", "family_macro_riser_aware", "false_pass_proxy", "false_pass_proxy_bends")}
except Exception as e:
    out["PB_err"] = str(e)
print(json.dumps(out, indent=1))
