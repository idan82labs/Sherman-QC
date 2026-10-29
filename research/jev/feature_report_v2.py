"""Per-feature separation (true vs false candidate lines) on calibration families (v2 study).

``python -m research.jev.feature_report_v2 --build DIR --out feature_auc.json``: for every
feature, calibration-family AUC (oriented, NaN = worst) on pool and lane candidates,
plus true/false medians. Labels: any-match to the ai_relabel truth (frozen matcher criteria).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.jev import arms_v2 as A
from research.jev.input_opt_v2 import CASE_LIST, _read, load_truth, read_candidates


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = read_candidates(Path(a.build) / "candidates.jsonl")
    truth = load_truth(_read(CASE_LIST), "ai_relabel")
    y = A.any_match_labels(rows, {c: t["lines"] for c, t in truth.items()})
    out = {}
    for set_name in ("pool", "lane"):
        idx = [i for i, r in enumerate(rows) if r["split"] == "calib" and r["set"] == set_name]
        X = A.raw_matrix([rows[i] for i in idx])
        yy = y[idx]
        res = {}
        for j, n in enumerate(A.FEATURE_NAMES):
            col = X[:, j]
            if np.all(np.isnan(col)):
                continue
            filled = np.where(np.isnan(col), np.nanmin(col) - 1, col)
            auc = A.auc(yy, filled)
            if auc is None:
                continue
            res[n] = {"auc_oriented": round(max(auc, 1 - auc), 4), "higher_means_true": auc >= 0.5,
                      "median_true": None if np.all(np.isnan(col[yy == 1])) else round(float(np.nanmedian(col[yy == 1])), 3),
                      "median_false": None if np.all(np.isnan(col[yy == 0])) else round(float(np.nanmedian(col[yy == 0])), 3),
                      "missing_fraction": round(float(np.isnan(col).mean()), 3)}
        out[set_name] = {"n": len(idx), "pos": int(yy.sum()),
                         "features": dict(sorted(res.items(), key=lambda kv: -kv[1]["auc_oriented"]))}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    for s, o in out.items():
        print(s, o["n"], o["pos"])
        for n, v in list(o["features"].items())[:14]:
            print(f"  {n:52s} {v['auc_oriented']:.3f} {'+' if v['higher_means_true'] else '-'} "
                  f"T={v['median_true']} F={v['median_false']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
