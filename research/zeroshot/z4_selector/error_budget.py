"""Error budget for a Z4/Z5 output: FN = NOT_IN_POOL / IN_POOL_SCORED_LOW / IN_POOL_SUPPRESSED; FP = near-gold
(within 15 deg / 15 mm of a gold line: span/offset/duplicate) vs SPURIOUS. -> z4/<name>/error_budget.json"""
import json, sys
from collections import Counter
import numpy as np
from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.custody import OUT, dev_ids, dev_inventory
from research.jev.pool_v2 import pair_metrics_matrix


def main(name, oof_name=None):
    inv = dev_inventory()
    oof_name = oof_name or name
    Z = np.load(OUT / oof_name / "oof_scores.npz")
    fn_c, fp_c, per = Counter(), Counter(), {}
    for sid in dev_ids():
        g = D.load_gold_lines(sid, inv)
        if g["zero_bend_control"]:
            continue
        sel = json.loads((OUT / name / "eval_lines" / f"{sid}.json").read_text())["owned_bend_regions"]
        rec = D.score_lines(sid, inv, sel)
        m = Z["sid"] == sid
        ends = Z["ends"][m]; sc = Z["oof"][m]
        gl = g["lines"]
        matched_g = {x["gold_id"] for x in rec["matches"]}
        mm = D.match_matrix(ends, gl) if len(ends) else np.zeros((0, len(gl)), bool)
        for j, gline in enumerate(gl):
            if gline["bend_id"] in matched_g:
                continue
            if not mm[:, j].any():
                c = "FN_NOT_IN_POOL"
            else:
                best = sc[mm[:, j]].max()
                c = "FN_IN_POOL_SUPPRESSED_OR_DUP" if best >= 0.5 else ("FN_IN_POOL_SCORE_0.2_0.5" if best >= 0.2 else "FN_IN_POOL_SCORE_LT_0.2")
            fn_c[c] += 1
        um = set(rec["unmatched_prediction_ids"])
        if sel and gl:
            pe = np.array([[l["line_start"], l["line_end"]] for l in sel], float)
            ge = np.array([[x["line_start"], x["line_end"]] for x in gl], float)
            pm = pair_metrics_matrix(ge, pe)
        for i, l in enumerate(sel):
            if f"P{i}" not in um:
                continue
            if not gl:
                fp_c["FP_SPURIOUS"] += 1; continue
            near = (pm["axis_angle_deg"][:, i] <= 15) & (pm["transverse_offset_mm"][:, i] <= 15) & (pm["gold_span_overlap"][:, i] > 0.1)
            if not near.any():
                fp_c["FP_SPURIOUS"] += 1
            else:
                j = int(np.argmin(np.where(near, pm["transverse_offset_mm"][:, i], 1e9)))
                if gl[j]["bend_id"] in matched_g:
                    fp_c["FP_DUP_OF_MATCHED_GOLD"] += 1
                elif pm["axis_angle_deg"][j, i] > 5:
                    fp_c["FP_NEAR_ANGLE"] += 1
                elif pm["transverse_offset_mm"][j, i] > 4:
                    fp_c["FP_NEAR_OFFSET"] += 1
                else:
                    fp_c["FP_NEAR_SPAN"] += 1
    res = {"fn": dict(fn_c), "fp": dict(fp_c)}
    (OUT / name / "error_budget.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
