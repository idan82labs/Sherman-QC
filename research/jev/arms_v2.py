"""Selection, scoring, classifiers and CIs for the Jev input-optimisation study (v2).

Pure functions (no scan reads, no network). Matching always goes through the
frozen matcher ``scripts/evaluate_f1_against_manual_gold.match_lines`` (5 deg,
4 mm, 0.6/0.6 overlap, one-to-one Hungarian), so a selected set is scored exactly
like a lane's output.

Protocol helpers:

- :func:`fit_threshold` picks the keep threshold that maximises calibration
  micro line-F1 (grid = calibration score quantiles; ties -> lowest threshold).
- classifier scores used to fit a threshold on calibration families are
  leave-one-family-out (:func:`lofo_scores`), so the threshold is not fitted on
  in-sample probabilities.
- :func:`family_bootstrap` resamples families (2000 rounds) for micro / family-
  macro F1, P, R and paired deltas; :func:`auc_ci` does the same for AUC.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from research.jev.pool_v2 import nms_select, pair_metrics_matrix

__all__ = ["FEATURE_NAMES", "feature_matrix", "any_match_labels", "select", "score_cases", "aggregate",
           "fit_threshold", "family_bootstrap", "auc", "auc_ci", "lofo_scores", "make_model"]

FEAT_KEYS = (
    "line_length_mm", "near_support_fraction", "near_support_longest_run_fraction", "median_surface_distance_mm",
    "empty_slice_fraction", "one_flange_slice_fraction", "two_flange_slice_fraction",
    "multi_flange_slice_fraction", "enclosed_slice_fraction", "median_included_angle_deg",
    "included_angle_iqr_deg", "included_angle_stable_fraction", "median_flatter_flange_fit_rms_mm",
    "median_rougher_flange_fit_rms_mm", "median_shorter_flange_width_mm", "median_longer_flange_width_mm",
    "median_flange_balance_ratio", "median_line_to_corner_mm", "median_corner_gap_mm", "median_apex_side_mm",
)
CTX_KEYS = ("near_duplicate_count", "parallel_within_6mm_count", "nearest_parallel_offset_mm",
            "parallel_6_to_40mm_count", "collinear_partner_count", "nearest_collinear_gap_mm")
POOL_KEYS = ("pool_member_count", "pool_radius_mm", "pool_radius_votes_count", "pool_two_sided_transition_score",
             "pool_two_sided_normal_angle_deg", "pool_two_sided_support_balance",
             "pool_two_sided_min_support_fraction", "pool_two_sided_normal_coherence",
             "pool_median_line_distance_mm", "pool_corridor_points_per_mm", "pool_finite_support_fraction",
             "pool_tight_to_wide_normal_second_ratio", "pool_normal_spread_deg_scan")
FEATURE_NAMES = tuple(f"feat.{k}" for k in FEAT_KEYS) + tuple(f"ctx.{k}" for k in CTX_KEYS) + \
    tuple(f"pool.{k}" for k in POOL_KEYS)


def _get(row: Mapping[str, Any], name: str) -> float:
    block, key = name.split(".", 1)
    v = (row.get(block) or {}).get(key)
    if isinstance(v, bool):
        return float(v)
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return float("nan")
    return float(v)


def raw_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    return np.array([[_get(r, n) for n in names] for r in rows], float).reshape(len(rows), len(names))


def feature_matrix(rows, names=FEATURE_NAMES, fill: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """NaN -> ``fill`` (column medians of the training rows) plus a missing indicator per column."""
    X = raw_matrix(rows, names)
    miss = np.isnan(X)
    if fill is None:
        fill = np.nanmedian(np.where(miss, np.nan, X), axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
    X = np.where(miss, fill[None, :], X)
    return np.hstack([X, miss.astype(float)]), fill


def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced"))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=30,
                                              l2_regularization=1.0, random_state=0)
    raise KeyError(kind)


def fit_predict(kind: str, train_rows, y, test_rows, names=FEATURE_NAMES) -> np.ndarray:
    Xtr, fill = feature_matrix(train_rows, names)
    Xte, _ = feature_matrix(test_rows, names, fill)
    model = make_model(kind)
    model.fit(Xtr, y)
    return model.predict_proba(Xte)[:, 1]


def lofo_scores(kind: str, rows, y, groups, names=FEATURE_NAMES) -> np.ndarray:
    """Leave-one-family-out probabilities for the calibration rows."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    out = np.full(len(rows), np.nan)
    for g in sorted(set(groups.tolist())):
        te = groups == g
        tr = ~te
        if len(set(y[tr].tolist())) < 2:
            out[te] = y[tr].mean()
            continue
        out[te] = fit_predict(kind, [rows[i] for i in np.flatnonzero(tr)], y[tr],
                              [rows[i] for i in np.flatnonzero(te)], names)
    return out


def _ends(rows) -> np.ndarray:
    return np.array([[r["line_start"], r["line_end"]] for r in rows], float).reshape(-1, 2, 3)


def any_match_labels(rows, gold_by_case: Mapping[str, list]) -> np.ndarray:
    """1 if the candidate matches any gold line of its scan (matcher criteria, not one-to-one)."""
    y = np.zeros(len(rows), int)
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case_id"]].append(i)
    for cid, idx in by_case.items():
        gold = gold_by_case.get(cid) or []
        if not gold:
            continue
        # gold is the reference, exactly as in the matcher's pair_metrics(gold, prediction)
        m = pair_metrics_matrix(_ends(gold), _ends([rows[i] for i in idx]))
        ok = ((m["axis_angle_deg"] <= 5.0) & (m["transverse_offset_mm"] <= 4.0)
              & (m["gold_span_overlap"] >= 0.6) & (m["prediction_span_overlap"] >= 0.6)).T
        hit = ok.any(axis=1)
        for k, i in enumerate(idx):
            y[i] = int(hit[k])
    return y


def select(rows, scores, threshold: float, *, dedupe: bool = True) -> dict[str, list[int]]:
    """Per case: indices (into ``rows``) kept at ``threshold`` (after one-to-one NMS dedupe)."""
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case_id"]].append(i)
    out = {}
    for cid, idx in by_case.items():
        s = np.asarray([scores[i] for i in idx], float)
        if dedupe:
            kept = nms_select(_ends([rows[i] for i in idx]), s, threshold)
        else:
            kept = [k for k in range(len(idx)) if s[k] >= threshold]
        out[cid] = [idx[k] for k in kept]
    return out


def score_cases(selected: Mapping[str, list[int]], rows, truth: Mapping[str, Mapping[str, Any]],
                cases: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Frozen-matcher TP/FP/FN per case (zero-bend controls give FP line counts only)."""
    from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines

    th = Thresholds()
    out = {}
    for cid in cases:
        t = truth[cid]
        preds = [{"bend_id": f"c{i}", "line_start": rows[i]["line_start"], "line_end": rows[i]["line_end"]}
                 for i in selected.get(cid, [])]
        if t["zero_bend_control"]:
            out[cid] = {"control": True, "fp_lines": len(preds)}
            continue
        m = match_lines(t["lines"], preds, th)
        out[cid] = {"control": False, "tp": m["true_positive"], "fp": m["false_positive"],
                    "fn": m["false_negative"]}
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(per_case: Mapping[str, Mapping[str, Any]], family_of: Mapping[str, str]) -> dict[str, Any]:
    sc = {c: v for c, v in per_case.items() if not v["control"]}
    tp = sum(v["tp"] for v in sc.values())
    fp = sum(v["fp"] for v in sc.values())
    fn = sum(v["fn"] for v in sc.values())
    p, r, f = _prf(tp, fp, fn)
    fam = defaultdict(lambda: [0, 0, 0])
    for c, v in sc.items():
        s = fam[family_of[c]]
        s[0] += v["tp"]
        s[1] += v["fp"]
        s[2] += v["fn"]
    ff = [_prf(*s)[2] for s in fam.values()]
    return {"scans": len(sc), "families": len(fam), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "family_macro_f1": round(float(np.mean(ff)), 4) if ff else 0.0,
            "control_fp_lines": sum(v["fp_lines"] for v in per_case.values() if v["control"])}


def fam_table(per_case, family_of) -> dict[str, tuple[int, int, int]]:
    fam = defaultdict(lambda: [0, 0, 0])
    for c, v in per_case.items():
        if v["control"]:
            continue
        s = fam[family_of[c]]
        s[0] += v["tp"]
        s[1] += v["fp"]
        s[2] += v["fn"]
    return {k: tuple(v) for k, v in fam.items()}


def family_bootstrap(arm: Mapping[str, tuple], base: Mapping[str, tuple] | None = None, rounds: int = 2000,
                     seed: int = 7) -> dict[str, list[float]]:
    fams = sorted(arm)
    if not fams:
        return {}
    a = np.array([arm[f] for f in fams], float)
    b = np.array([base[f] for f in fams], float) if base is not None else None
    rng = np.random.default_rng(seed)

    def met(arr, idx):
        s = arr[idx].sum(axis=0)
        p, r, f = _prf(*s)
        return f, float(np.mean([_prf(*arr[i])[2] for i in idx])), p, r

    draws = []
    for _ in range(rounds):
        idx = rng.integers(0, len(fams), len(fams))
        ma = met(a, idx)
        draws.append(ma + (tuple(x - y for x, y in zip(ma, met(b, idx))) if b is not None else ()))
    d = np.array(draws)
    names = ["f1", "family_macro_f1", "precision", "recall"]
    if b is not None:
        names += ["delta_f1", "delta_family_macro_f1", "delta_precision", "delta_recall"]
    return {n: [round(float(np.percentile(d[:, i], 2.5)), 4), round(float(np.percentile(d[:, i], 97.5)), 4)]
            for i, n in enumerate(names)}


def auc(y, s) -> float | None:
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def auc_ci(y, s, groups, rounds: int = 2000, seed: int = 11) -> list[float] | None:
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    fams = sorted(set(groups.tolist()))
    by = {f: np.flatnonzero(groups == f) for f in fams}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(rounds):
        idx = np.concatenate([by[fams[j]] for j in rng.integers(0, len(fams), len(fams))])
        a = auc(y[idx], s[idx])
        if a is not None:
            vals.append(a)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def fit_threshold(rows, scores, truth, cases, family_of, *, dedupe: bool = True,
                  grid_size: int = 80) -> tuple[float, list[dict[str, Any]]]:
    """Threshold maximising micro line-F1 over ``cases`` (calibration). Ties -> lowest threshold."""
    sub = [i for i, r in enumerate(rows) if r["case_id"] in set(cases)]
    s = np.asarray([scores[i] for i in sub], float)
    qs = np.unique(np.quantile(s, np.linspace(0.0, 1.0, grid_size))) if len(s) else np.array([0.0])
    # high-score tail matters most (few lines per scan): add the top 300 distinct scores
    top = np.unique(np.sort(s)[-300:])
    grid = np.unique(np.concatenate([qs, top]))
    trace, best = [], None
    sub_rows = [rows[i] for i in sub]
    sub_scores = s.tolist()
    for t in grid:
        sel = select(sub_rows, sub_scores, float(t), dedupe=dedupe)
        agg = aggregate(score_cases(sel, sub_rows, truth, cases), family_of)
        trace.append({"threshold": round(float(t), 6), "f1": agg["f1"], "tp": agg["tp"], "fp": agg["fp"]})
        if best is None or agg["f1"] > best[0] + 1e-12:
            best = (agg["f1"], float(t))
    return best[1], trace
