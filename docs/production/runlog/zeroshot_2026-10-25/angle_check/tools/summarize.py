"""Slices for ANGLE_CHECK.md (adds 'slices' and 'sections' to ANGLE_CHECK.json). usage: python3 summarize.py ANGLE_CHECK.json SECTIONS.json"""
import json, sys
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("ac", Path(__file__).with_name("angle_check.py"))
js, sj = Path(sys.argv[1]), Path(sys.argv[2])
D = json.loads(js.read_text())
import numpy as np
def stats(x):
    x = np.asarray([v for v in x if v is not None], float); a = np.abs(x)
    if not len(x): return {"n": 0}
    return {"n": int(len(x)), "median_abs": round(float(np.median(a)), 2), "p90_abs": round(float(np.percentile(a, 90)), 2),
            "max_abs": round(float(a.max()), 2), "bias_mean": round(float(x.mean()), 2), "bias_median": round(float(np.median(x)), 2),
            "within_1deg": round(float((a <= 1).mean()), 3)}
# E3 readings a cross-section shows are wrong (zs and gold agree with the section within 1.5 deg, E3 off by > 8 deg)
E3_WRONG = {("sc_186463746daa", "B5"), ("sc_186463746daa", "B6"), ("sc_6e5d5a2f60f5", "B2"),
            ("sc_2233372b8df7", "B3"), ("sc_2233372b8df7", "B4"), ("sc_2233372b8df7", "B5")}
R = [r for r in D["cli"]["records"] if r["bend_uid"]]
sl = {}
def block(rs):
    return {k: stats([r[k] for r in rs]) for k in ("d_e3", "d_cad", "d_gold", "e3_minus_cad")}
sl["all"] = block(R)
sl["excl_e3_wrong"] = block([r for r in R if (r["scan"], r["gold_id"]) not in E3_WRONG])
sl["excl_e3_wrong_excl_joggle"] = block([r for r in R if (r["scan"], r["gold_id"]) not in E3_WRONG and r["class"] != "joggle"])
sl["e3_pass_fail_only_excl_wrong"] = block([r for r in R if r.get("e3_decision") in ("PASS", "FAIL") and (r["scan"], r["gold_id"]) not in E3_WRONG])
for c in ("~90", "acute", "obtuse", "joggle"):
    sl[f"class={c}"] = block([r for r in R if r["class"] == c])
    sl[f"class={c} excl_e3_wrong"] = block([r for r in R if r["class"] == c and (r["scan"], r["gold_id"]) not in E3_WRONG])
for t in sorted({r["t_mm"] for r in R}):
    rs = [r for r in R if r["t_mm"] == t]
    sl[f"t={t}"] = {**block(rs), "families": sorted({r["family"] for r in rs}), "classes": sorted({r["class"] for r in rs})}
    sl[f"t={t} excl_e3_wrong"] = block([r for r in rs if (r["scan"], r["gold_id"]) not in E3_WRONG])
# tolerance side +-1 about nominal
def tol(rs):
    a = [(abs(r["e3_incl"] - r["nominal"]) <= 1, abs(r["zs_incl"] - r["nominal"]) <= 1) for r in rs if r["e3_incl"] is not None]
    return {"n": len(a), "agree": sum(x == y for x, y in a), "rate": round(sum(x == y for x, y in a) / len(a), 3),
            "both_in": sum(x and y for x, y in a), "both_out": sum(not x and not y for x, y in a),
            "e3_in_zs_out": sum(x and not y for x, y in a), "e3_out_zs_in": sum(y and not x for x, y in a)}
sl["tol_all"] = tol(R)
sl["tol_excl_e3_wrong"] = tol([r for r in R if (r["scan"], r["gold_id"]) not in E3_WRONG])
sl["tol_excl_e3_wrong_excl_joggle"] = tol([r for r in R if (r["scan"], r["gold_id"]) not in E3_WRONG and r["class"] != "joggle"])
sl["tol_class"] = {c: tol([r for r in R if r["class"] == c and (r["scan"], r["gold_id"]) not in E3_WRONG]) for c in ("~90", "acute", "obtuse", "joggle")}
# disagreements near the tolerance edge
dis = [r for r in R if r["e3_incl"] is not None and (r["scan"], r["gold_id"]) not in E3_WRONG and
       (abs(r["e3_incl"] - r["nominal"]) <= 1) != (abs(r["zs_incl"] - r["nominal"]) <= 1)]
sl["tol_disagreements"] = [{"scan": r["scan"], "bend": r["bend_label"], "class": r["class"], "nominal": r["nominal"],
                            "zs": round(r["zs_incl"], 2), "e3": round(r["e3_incl"], 2), "e3_decision": r["e3_decision"],
                            "gold": r["gold_incl"]} for r in dis]
sl["n_gold_with_angle"] = sum(r["gold_incl"] is not None for r in R)
sl["e3_wrong_keys"] = sorted(map(list, E3_WRONG))
D["slices"] = sl
D["sections"] = json.loads(sj.read_text())
js.write_text(json.dumps(D, indent=1, default=str))
for k, v in sl.items():
    print(k, json.dumps(v)[:600])
