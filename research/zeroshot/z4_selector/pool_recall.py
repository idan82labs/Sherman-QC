"""Per-scan pool recall with the frozen matcher (score_case, every merged candidate kept). -> z4/pool_recall.json"""
import json
from collections import Counter
from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.candidates import FEAT_ROOT
from research.zeroshot.z4_selector.custody import OUT, dev_ids, dev_inventory


def main(tag="base"):
    inv = dev_inventory()
    a0 = D.a0_records()
    out, tot, hit, a0hit = {}, 0, 0, 0
    for sid in dev_ids():
        d = json.loads((FEAT_ROOT / tag / f"{sid}.json").read_text())
        rec = D.score_lines(sid, inv, d["rows"])
        if rec["disposition"] != "SCORED_COMPLETE_LABEL":
            continue
        g, tp = rec["gold_count"], rec["true_positive"]
        tot += g; hit += tp; a0hit += a0[sid].get("true_positive", 0)
        out[sid] = {"gold": g, "pool_tp": tp, "pool_recall": round(tp / g, 3) if g else None, "candidates": len(d["rows"]),
                    "a0_tp": a0[sid].get("true_positive", 0), "a0_empty": not a0[sid].get("prediction_count")}
    e = [v for v in out.values() if v["a0_empty"]]
    res = {"tag": tag, "scans": len(out), "gold_lines": tot, "pool_tp": hit, "pool_recall": round(hit / tot, 4),
           "a0_tp": a0hit, "a0_recall": round(a0hit / tot, 4),
           "median_candidates": sorted(v["candidates"] for v in out.values())[len(out) // 2],
           "a0_empty_scans": {"scans": len(e), "gold": sum(v["gold"] for v in e), "pool_tp": sum(v["pool_tp"] for v in e)},
           "scans_pool_recall_lt_0.5": sorted(k for k, v in out.items() if v["pool_recall"] is not None and v["pool_recall"] < 0.5),
           "per_scan": out}
    (OUT / f"pool_recall_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_scan"}, indent=1))


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:]))
