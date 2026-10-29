"""Nested family-grouped CV over the graph-threshold grid (run.py --grid).

For each held-out fold k (integrator dev_folds.json, family_group5), the grid config with the best micro line-F1
on the other four folds is chosen and scored on fold k; TP/FP/FN are pooled over the five held-out folds.
usage: .venv/bin/python -m research.zeroshot.z2_self_recipe.oof --tag it07
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from research.zeroshot.z2_self_recipe import common as C


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    x = ap.parse_args(argv)
    runs = C.OUT / "runs"
    grids = sorted(p.name for p in runs.glob(f"{x.tag}_*") if (p / "scores.json").is_file())
    per = {}
    for g in grids:
        d = json.loads((runs / g / "scores.json").read_text())["per_case"]
        per[g] = {s: r for s, r in d.items() if r.get("disposition") == "SCORED_COMPLETE_LABEL"}
    sids = sorted(set.intersection(*(set(v) for v in per.values())))
    fold = {s: per[grids[0]][s]["fold"] for s in sids}
    fam = {s: per[grids[0]][s]["family_id"] for s in sids}

    def score(g, ss):
        tp = sum(per[g][s]["true_positive"] for s in ss)
        fp = sum(per[g][s]["false_positive"] for s in ss)
        fn = sum(per[g][s]["false_negative"] for s in ss)
        return tp, fp, fn
    chosen, pooled = {}, defaultdict(lambda: [0, 0, 0])
    famacc = defaultdict(lambda: [0, 0, 0])
    for k in range(5):
        train = [s for s in sids if fold[s] != k]
        test = [s for s in sids if fold[s] == k]
        best = max(grids, key=lambda g: (prf(*score(g, train))[2], g))
        chosen[k] = best
        for s in test:
            r = per[best][s]
            for i, key in enumerate(("true_positive", "false_positive", "false_negative")):
                famacc[fam[s]][i] += r[key]
    tp, fp, fn = (sum(v[i] for v in famacc.values()) for i in range(3))
    p, r, f1 = prf(tp, fp, fn)
    macro = float(np.mean([prf(*v)[2] for v in famacc.values()]))
    table = {g: round(prf(*score(g, sids))[2], 4) for g in grids}
    out = {"tag": x.tag, "scans": len(sids), "families": len(famacc), "chosen_per_fold": chosen,
           "oof": {"tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r, 4), "micro_f1": round(f1, 4),
                   "family_macro_f1": round(macro, 4)},
           "grid_all_dev_micro": table}
    (C.OUT / f"oof_{x.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
