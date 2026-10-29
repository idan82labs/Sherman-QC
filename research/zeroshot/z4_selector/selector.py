"""Z4 learned selector over the candidate table, with honest out-of-fold DEV numbers.

Outer loop: 5 grouped folds (z4/dev_folds.json, family_group5). For held-out fold k:
  - fit the model on the other 4 folds;
  - choose the keep threshold by an inner grouped CV on those 4 folds (fit on 3, predict 1, pick the
    threshold that maximises micro line-F1 after NMS over the inner out-of-fold scores);
  - score fold k with that model + threshold + NMS. Nothing of fold k is used to fit or choose.
Models: logistic regression (median-imputed, missing indicators, standardised) or sklearn
HistGradientBoosting (native NaN). Labels: candidate matches >= 1 gold line under the frozen matcher.

  .venv/bin/python -B -m research.zeroshot.z4_selector.selector --tag base --model gbm --out-name iter1
Writes z4/<out-name>/{summary.json, per_case.json, oof_scores.npz, eval_lines/<scan>.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.z4_selector.custody import OUT, dev_ids, dev_inventory
from research.zeroshot.z4_selector.candidates import FEAT_ROOT
from research.zeroshot.z4_selector import data as D
from research.jev.pool_v2 import near_duplicate_matrix

THRESH_GRID = np.round(np.concatenate([np.arange(0.05, 0.9, 0.025), np.arange(0.9, 0.995, 0.01)]), 3)
EXCLUDE_FEATURES = {"x_line_length_mm"}  # duplicate of length_mm


# ------------------------------------------------------------------ table
def load_table(tags: list[str], inv: dict) -> dict[str, Any]:
    scans = {}
    for sid in dev_ids():
        rows, metas = [], []
        for tag in tags:
            f = FEAT_ROOT / tag / f"{sid}.json"
            if f.is_file():
                d = json.loads(f.read_text())
                rows = d["rows"]  # last tag wins (tags are alternative candidate sets, e.g. base / merged)
                metas.append(d["meta"])
        scans[sid] = {"rows": rows, "meta": metas[-1] if metas else {}}
    keys = sorted({k for s in scans.values() for r in s["rows"] for k in r["f"]} - EXCLUDE_FEATURES)
    X, y, sid_of, ends = [], [], [], []
    gold = {}
    for sid, s in scans.items():
        g = D.load_gold_lines(sid, inv)
        gold[sid] = g
        if not s["rows"]:
            continue
        e = np.array([[r["line_start"], r["line_end"]] for r in s["rows"]], float)
        mm = D.match_matrix(e, g["lines"])
        for r, lab in zip(s["rows"], mm.any(1) if mm.size else np.zeros(len(e), bool)):
            X.append([_num(r["f"].get(k)) for k in keys])
            y.append(int(lab))
            sid_of.append(sid)
        ends.append(e)
    return {"keys": keys, "X": np.array(X, float), "y": np.array(y, int), "sid": np.array(sid_of),
            "ends": np.concatenate(ends) if ends else np.zeros((0, 2, 3)), "gold": gold, "scans": scans}


def _num(v):
    return float(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else np.nan


# ------------------------------------------------------------------ models
class LR:
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        self.med = np.nanmedian(X, axis=0)
        self.med[~np.isfinite(self.med)] = 0.0
        Z = self._z(X, fit=True)
        self.m = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.m.fit(Z, y)
        return self

    def _z(self, X, fit=False):
        miss = ~np.isfinite(X)
        Xi = np.where(miss, self.med, X)
        # signed log for heavy-tailed counts
        Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
        if fit:
            self.mu, self.sd = Xi.mean(0), Xi.std(0) + 1e-9
            self.miss_cols = np.flatnonzero(miss.any(0))
        return np.hstack([(Xi - self.mu) / self.sd, miss[:, self.miss_cols].astype(float)])

    def predict(self, X):
        return self.m.predict_proba(self._z(X))[:, 1]


class GBM:
    def __init__(self, **kw):
        self.kw = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                       l2_regularization=1.0, random_state=0)
        self.kw.update(kw)

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.m = HistGradientBoostingClassifier(**self.kw)
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def make_model(name: str):
    return LR() if name == "lr" else GBM()


# ------------------------------------------------------------------ selection
def nms(ends: np.ndarray, scores: np.ndarray, thr: float, dedupe_mm: float = 4.0) -> list[int]:
    idx = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] >= thr]
    if not idx:
        return []
    dup = near_duplicate_matrix(ends[idx], angle_deg=5.0, offset_mm=dedupe_mm, overlap=0.6, mutual=False)
    kept: list[int] = []
    for a in range(len(idx)):
        if not any(dup[a, b] for b in kept):
            kept.append(a)
    return [idx[a] for a in kept]


def fast_counts(ends, scores, gold_lines, thr, dedupe_mm=4.0):
    """(tp, fp, fn) for one scan via Hungarian on the match matrix (same rule as score_case)."""
    from scipy.optimize import linear_sum_assignment

    keep = nms(ends, scores, thr, dedupe_mm)
    if not gold_lines:
        return 0, len(keep), 0
    if not keep:
        return 0, 0, len(gold_lines)
    mm = D.match_matrix(ends[keep], gold_lines)
    if not mm.any():
        return 0, len(keep), len(gold_lines)
    r, c = linear_sum_assignment(-mm.astype(float))
    tp = int(mm[r, c].sum())
    return tp, len(keep) - tp, len(gold_lines) - tp


def best_threshold(T, mask_rows, scores, sids, dedupe_mm=4.0):
    """(threshold, dedupe_mm, f1): dedupe_mm may be a list, then it is chosen jointly with the threshold."""
    grid = dedupe_mm if isinstance(dedupe_mm, (list, tuple)) else [dedupe_mm]
    best = (-1.0, 0.5, grid[0])
    for d in grid:
        f1, thr = _best_thr_one(T, mask_rows, scores, sids, d)
        if f1 > best[0] + 1e-9:
            best = (f1, thr, d)
    return best[1], best[2], best[0]


def _best_thr_one(T, mask_rows, scores, sids, dedupe_mm):
    best = (-1.0, 0.5)
    for thr in THRESH_GRID:
        tp = fp = fn = 0
        for sid in sids:
            m = mask_rows[sid]
            g = T["gold"][sid]
            if g["zero_bend_control"]:
                continue
            a, b, c = fast_counts(T["ends"][m], scores[m], g["lines"], thr, dedupe_mm)
            tp += a; fp += b; fn += c
        f1 = D.prf(tp, fp, fn)[2]
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def run(tags, model, out_name, dedupe_mm=4.0, feature_filter=None, inner=True):
    t0 = time.monotonic()
    inv = dev_inventory()
    folds = D.make_folds(inv)
    T = load_table(tags, inv)
    keys = T["keys"]
    cols = [i for i, k in enumerate(keys) if feature_filter is None or feature_filter(k)]
    X = T["X"][:, cols]
    y = T["y"]
    sfold = folds["scan_fold"]
    rows_of = {sid: np.flatnonzero(T["sid"] == sid) for sid in dev_ids()}
    fold_arr = np.array([sfold[s] for s in T["sid"]])
    oof = np.full(len(y), np.nan)
    thresholds = {}
    inner_all = {}
    for k in range(folds["k"]):
        tr = fold_arr != k
        m = make_model(model).fit(X[tr], y[tr])
        oof[~tr] = m.predict(X[~tr])
        if inner:
            inner_scores = np.full(len(y), np.nan)
            for j in range(folds["k"]):
                if j == k:
                    continue
                itr = tr & (fold_arr != j)
                ite = fold_arr == j
                inner_scores[ite] = make_model(model).fit(X[itr], y[itr]).predict(X[ite])
            train_sids = [s for s in dev_ids() if sfold[s] != k and len(rows_of[s])]
            thr, dd, f1i = best_threshold(T, rows_of, inner_scores, train_sids, dedupe_mm)
            inner_all[f"inner_{k}"] = inner_scores
        else:
            thr, dd, f1i = 0.5, (dedupe_mm[0] if isinstance(dedupe_mm, list) else dedupe_mm), None
        thresholds[k] = {"threshold": thr, "dedupe_mm": dd, "inner_f1": f1i}
    # select + score with the frozen matcher
    recs, per_case = [], {}
    ev_dir = OUT / out_name / "eval_lines"
    ev_dir.mkdir(parents=True, exist_ok=True)
    for sid in dev_ids():
        m = rows_of[sid]
        thr = thresholds[sfold[sid]]["threshold"]
        keep = nms(T["ends"][m], oof[m], thr, thresholds[sfold[sid]]["dedupe_mm"]) if len(m) else []
        lines = [{"line_start": T["ends"][m][i][0].tolist(), "line_end": T["ends"][m][i][1].tolist(),
                  "score": float(oof[m][i])} for i in keep]
        rec = D.score_lines(sid, inv, lines)
        recs.append(rec)
        per_case[sid] = {k2: rec.get(k2) for k2 in ("disposition", "family_id", "gold_count", "prediction_count",
                                                     "true_positive", "false_positive", "false_negative",
                                                     "false_positive_lines")}
        per_case[sid]["fold"] = sfold[sid]
        (ev_dir / f"{sid}.json").write_text(json.dumps({
            "schema": "sherman.broad_baseline.eval_lines.v1", "lane": f"z4_{out_name}", "frame": "scan_raw",
            "status": "OK" if lines else "NO_PREDICTIONS", "line_count": len(lines),
            "owned_bend_regions": [{"bend_id": f"Z{i + 1}", **l} for i, l in enumerate(lines)]}))
    from sklearn.metrics import roc_auc_score

    summary = {"tags": tags, "model": model, "dedupe_mm": dedupe_mm, "n_features": len(cols),
               "features": [keys[i] for i in cols], "candidates": int(len(y)), "positives": int(y.sum()),
               "oof_auc": round(float(roc_auc_score(y, oof)), 4) if len(set(y)) > 1 else None,
               "thresholds": thresholds, "oof": D.aggregate(recs), "wall_s": round(time.monotonic() - t0, 1)}
    # pool ceiling (any candidate kept, one-to-one)
    summary["pool_any_match_recall"] = _pool_recall(T)
    (OUT / out_name / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / out_name / "per_case.json").write_text(json.dumps(per_case, indent=1))
    np.savez_compressed(OUT / out_name / "oof_scores.npz", oof=oof, y=y, sid=T["sid"], ends=T["ends"],
                        k=np.array([r["k"] for s_ in dev_ids() for r in T["scans"][s_]["rows"]]), **inner_all)
    return summary, T, oof


def _pool_recall(T) -> dict:
    from scipy.optimize import linear_sum_assignment

    tot = hit = 0
    for sid, g in T["gold"].items():
        if g["zero_bend_control"] or not g["lines"]:
            continue
        m = T["sid"] == sid
        tot += len(g["lines"])
        if m.any():
            mm = D.match_matrix(T["ends"][m], g["lines"])
            if mm.any():
                r, c = linear_sum_assignment(-mm.astype(float))
                hit += int(mm[r, c].sum())
    return {"gold_lines": tot, "hit_one_to_one": hit, "recall": round(hit / tot, 4) if tot else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--model", default="gbm", choices=("gbm", "lr"))
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dedupe-mm", default="4", help="mm, or comma list chosen jointly with the threshold by inner CV")
    a = ap.parse_args(argv)
    dd = [float(x) for x in a.dedupe_mm.split(",")]
    s, _, _ = run(a.tag, a.model, a.out_name, dd if len(dd) > 1 else dd[0])
    print(json.dumps({k: s[k] for k in ("model", "candidates", "positives", "oof_auc", "pool_any_match_recall", "oof")}, indent=1))
    print(s["thresholds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
