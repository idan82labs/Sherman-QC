"""Out-of-fold candidate selector for Z3 (family-grouped 5 folds from dev_folds.json).

Each candidate gets a label = it matches some gold line under the frozen gates (5 deg / 4 mm / 0.6 of both spans).
A logistic regression on plain geometric features is fit on 4 folds, its threshold is chosen on those same 4 folds
(max micro line-F1 with the frozen one-to-one scorer), and it is applied to the held-out fold. Also reports the
hand gate for comparison. No part-specific inputs (no ids, no family names).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore, gate as G  # noqa: E402
from research.zeroshot.z3_slicing.score_run import load_run, evaluate  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402

FEATS = ["log_len", "log_sup", "support_frac", "bend", "incl_std", "log_fmin", "log_fmax", "res_rms", "axis_dev_deg",
         "frac_outer", "frac_inner", "frac_both", "flat_rms", "gap_med", "strip_frac", "flank_frac", "n_parts",
         "log_dups", "rank_frac", "fam_rank", "len_rel",
         # context (no ids): neighbours that compete for the same fold, scan size, sheet thickness
         "n_par_near", "q_rel_near", "covered_frac", "log_ncand", "t_mm"]


USE_CTX = False      # context features: an ablation (it6: LR 0.910 -> 0.921, GBM 0.928 -> 0.922); off by default
FEATS_BASE = FEATS[:21]


def _context(C: list[dict]) -> np.ndarray:
    n = len(C)
    S = np.array([c["start"] for c in C], float)
    E = np.array([c["end"] for c in C], float)
    D = E - S
    L = np.linalg.norm(D, axis=1)
    U = D / np.maximum(L, 1e-9)[:, None]
    M = 0.5 * (S + E)
    q = np.array([c.get("quality", 0.0) for c in C])
    out = np.zeros((n, 3))
    for i in range(n):
        cosang = np.abs(U @ U[i])
        cand = np.nonzero((cosang > math.cos(math.radians(5))) & (np.arange(n) != i))[0]
        if not len(cand):
            continue
        dm = M[cand] - M[i]
        off = np.linalg.norm(dm - np.outer(dm @ U[i], U[i]), axis=1)
        near = cand[off < 6.0]
        if not len(near):
            continue
        si = sorted([0.0, L[i]])
        cov = 0.0
        npar = 0
        for j in near:
            a = (S[j] - S[i]) @ U[i]
            b = (E[j] - S[i]) @ U[i]
            lo, hi = max(si[0], min(a, b)), min(si[1], max(a, b))
            if hi - lo > 0:
                npar += 1
                if L[j] > L[i]:
                    cov = max(cov, (hi - lo) / max(L[i], 1e-9))
        out[i] = [npar, q[near].max() / max(q[i], 1e-9), cov]
    return out


def feats(d: dict) -> np.ndarray:
    C = d.get("candidates", [])
    if not C:
        return np.zeros((0, len(FEATS)))
    q = np.array([c.get("quality", 0.0) for c in C])
    order = np.argsort(-q)
    rank = np.empty(len(C))
    rank[order] = np.arange(len(C))
    Lmax = max(c["length"] for c in C)
    rows = []
    for i, c in enumerate(C):
        rows.append([math.log(c["length"] + 1), math.log(c["n_support"] + 1), c["support_frac"], 180 - c["incl"],
                     c["incl_std"], math.log(c["flank_min"] + 0.5), math.log(c["flank_max"] + 0.5), c["res_rms"],
                     c["axis_dev_deg"], c["frac_outer"], c["frac_inner"], c["frac_both"], c["flat_rms"],
                     min(c["gap_med"], 30), c.get("strip_frac", 0), c.get("flank_frac", 0), min(c.get("n_parts", 1), 6),
                     math.log(c.get("n_dups", 0) + 1), rank[i] / len(C), min(c["family"], 15), c["length"] / Lmax])
    if not USE_CTX:
        return np.array(rows, float)
    ctx = _context(C)
    extra = np.column_stack([ctx[:, 0], np.minimum(ctx[:, 1], 20.0), ctx[:, 2],
                             np.full(len(C), math.log(len(C) + 1)), np.full(len(C), float(d.get("t_mm", 1.5)))])
    return np.hstack([np.array(rows, float), extra])


def labels(sid: str, row: dict, d: dict) -> np.ndarray:
    gold = zscore.S.load_gold(zscore.case_spec(sid, row)["gold"], frame_records=None)["lines"]
    C = d.get("candidates", [])
    y = np.zeros(len(C))
    for j, c in enumerate(C):
        pe = np.array([c["start"], c["end"]])
        for g in gold:
            m = pair_metrics(np.array([g["line_start"], g["line_end"]]), pe)
            if (m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= 0.6
                    and m["prediction_span_overlap"] >= 0.6):
                y[j] = 1
                break
    return y


class LR:
    def __init__(self, C=1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.s = StandardScaler()
        self.m = LogisticRegression(C=C, max_iter=2000, class_weight=None)

    def fit(self, X, y):
        self.m.fit(self.s.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(self.s.transform(X))[:, 1] if len(X) else np.zeros(0)


class GBM:
    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                                random_state=0)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def select_with(d, p, thr):
    C = d.get("candidates", [])
    return [c for c, pi in zip(C, p) if pi >= thr]


def run(run_dir: str, model: str = "lr", out_json: str = "") -> dict:
    inv = custody.dev_inventory()
    ids = custody.dev_ids()
    data = load_run(Path(run_dir), ids)
    fold = zscore.folds(inv, ids)
    X = {s: feats(d) for s, d in data.items()}
    Y = {s: labels(s, inv[s], d) for s, d in data.items()}
    P = {}
    thr_used = {}
    for f in sorted(set(fold.values())):
        tr = [s for s in data if fold[s] != f]
        te = [s for s in data if fold[s] == f]
        Xtr = np.vstack([X[s] for s in tr if len(X[s])])
        ytr = np.concatenate([Y[s] for s in tr if len(X[s])])
        mdl = (LR() if model == "lr" else GBM()).fit(Xtr, ytr)
        ptr = {s: mdl.predict(X[s]) for s in tr}
        best = (-1, 0.5)
        for thr in np.arange(0.15, 0.8, 0.025):
            agg, _ = evaluate({s: data[s] for s in tr}, inv, lambda d, s: select_with(d, ptr[s], thr), diagnose=False)
            if agg["micro_f1"] > best[0]:
                best = (agg["micro_f1"], thr)
        thr_used[f] = round(float(best[1]), 3)
        for s in te:
            P[s] = (mdl.predict(X[s]), best[1])
    agg, recs = evaluate(data, inv, lambda d, s: select_with(d, P[s][0], P[s][1]))
    ci = zscore.bootstrap_ci(recs)
    res = {"model": model, "thresholds_by_fold": thr_used, **agg, **ci}
    print(f"OOF {model}: P={agg['P']:.3f} R={agg['R']:.3f} F1={agg['micro_f1']:.3f} {ci['micro_ci']} "
          f"macro={agg['family_macro_f1']:.3f} {ci['macro_ci']} tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} thr={thr_used}")
    print("  FN", agg["fn_causes"], " FP", agg["fp_causes"], " controls", agg["controls"])
    if out_json:
        for s in data:
            for c, pi in zip(data[s].get("candidates", []), P[s][0]):
                c["score"] = float(pi)
            data[s]["oof_threshold"] = float(P[s][1])
        Path(out_json).write_text(json.dumps({"summary": res, "per_case": recs,
                                              "thresholds": {s: float(P[s][1]) for s in data},
                                              "scored": {s: data[s].get("candidates", []) for s in data},
                                              "selected": {s: [c for c, pi in zip(data[s].get("candidates", []), P[s][0]) if pi >= P[s][1]] for s in data}},
                                             indent=1, default=str))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="lr")
    ap.add_argument("--json", default="")
    ap.add_argument("--ctx", action="store_true")
    a = ap.parse_args()
    USE_CTX = a.ctx
    run(a.run, a.model, a.json)


def train_final(run_dir: str, model: str, out_path: str, thr: float) -> None:
    """Fit the selector on ALL DEV candidates (for new scans only; DEV numbers are always out-of-fold)."""
    import pickle

    inv = custody.dev_inventory()
    data = load_run(Path(run_dir), custody.dev_ids())
    X = np.vstack([feats(d) for d in data.values() if d.get("candidates")])
    y = np.concatenate([labels(s, inv[s], d) for s, d in data.items() if d.get("candidates")])
    mdl = (LR() if model == "lr" else GBM()).fit(X, y)
    Path(out_path).write_bytes(pickle.dumps({"model": mdl, "threshold": thr, "feats": FEATS if USE_CTX else FEATS_BASE, "kind": model,
                                             "trained_on": f"DEV {len(data)} scans, run {run_dir}"}))
