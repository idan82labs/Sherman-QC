"""Export the lane's pickled sklearn HistGradientBoostingClassifier models to the versioned JSON tree format
sherman.zs_gbm.v1 read by zs_rust (src/gbm.rs), and check that the exported trees reproduce predict_proba.

Format (one JSON object): format, sklearn_version, source (repo-relative pickle path), source_sha256, n_features,
baseline (the binary model's _baseline_prediction), link "expit", trees: one object per iteration with parallel
node arrays feature, threshold (num_threshold), missing_left, left, right, leaf, value. Only numerical splits
(no categorical features) are supported; the exporter refuses anything else.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.export_models --config CONFIG --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v3_rs.core import REPO


def export(pkl: Path, rel: str) -> dict:
    import sklearn
    raw = pkl.read_bytes()
    obj = pickle.loads(raw)
    m = obj["model"].m                    # selector.GBM wrapper -> HistGradientBoostingClassifier
    if m.n_trees_per_iteration_ != 1:
        raise ValueError("only binary models are supported")
    trees = []
    for it in m._predictors:
        (pred,) = it
        nd = pred.nodes
        if np.any(nd["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        trees.append({"feature": nd["feature_idx"].astype(int).tolist(),
                      "threshold": [float(x) for x in nd["num_threshold"]],
                      "missing_left": nd["missing_go_to_left"].astype(int).tolist(),
                      "left": nd["left"].astype(int).tolist(), "right": nd["right"].astype(int).tolist(),
                      "leaf": nd["is_leaf"].astype(int).tolist(), "value": [float(x) for x in nd["value"]]})
    return {"format": "sherman.zs_gbm.v1", "sklearn_version": sklearn.__version__, "source": rel,
            "source_sha256": hashlib.sha256(raw).hexdigest(), "n_features": int(m.n_features_in_),
            "baseline": float(np.asarray(m._baseline_prediction).ravel()[0]), "link": "expit", "trees": trees}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X))
    for r, x in enumerate(X):
        raw = 0.0 + model["baseline"]
        for t in model["trees"]:
            n = 0
            while not t["leaf"][n]:
                v = x[t["feature"][n]]
                if v != v:
                    n = t["left"][n] if t["missing_left"][n] else t["right"][n]
                else:
                    n = t["left"][n] if v <= t["threshold"][n] else t["right"][n]
            raw += t["value"][n]
        out[r] = 1.0 / (1.0 + np.exp(-raw))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    v = cfg["variants"]["zs"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261025)
    for key in ("model", "z3_model"):
        rel = v[key]
        pkl = REPO / rel
        if hashlib.sha256(pkl.read_bytes()).hexdigest() != v[f"{key}_sha256"]:
            print(f"refusing: {rel} does not match its pinned sha256", file=sys.stderr)
            return 2
        mdl = export(pkl, rel)
        X = rng.normal(size=(2000, mdl["n_features"])) * 3.0
        X[rng.random(X.shape) < 0.1] = np.nan
        ref = pickle.loads(pkl.read_bytes())["model"].predict(X)
        got = predict(mdl, X)
        dmax = float(np.max(np.abs(ref - got)))
        mdl["export_check"] = {"rows": int(len(X)), "max_abs_diff": dmax}
        dst = out / (Path(rel).stem + ".gbm.json")
        dst.write_text(json.dumps(mdl))
        print(f"{dst}: {len(mdl['trees'])} trees, max |proba diff| on {len(X)} random rows = {dmax:g}")
        if dmax > 1e-12:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
