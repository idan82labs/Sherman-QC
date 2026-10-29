"""Z4 stage 2: re-score each candidate with its neighbours' stage-1 scores (stacking), nested out-of-fold.

Context features from stage-1 scores s (per scan): own s; max s and count of parallel neighbours (<= 5 deg,
>= 0.5 shorter-span overlap) within 4 / 8 / 15 mm; own rank among them; offset to the best-scored parallel
neighbour; collinear partner max s; the scan's top score, count above 0.5, and own s / scan top.
Nesting: for held-out fold k the training rows use stage-1 scores `inner_k` (never trained on fold k nor on
themselves), the held-out rows use the outer OOF scores. Stage-2 = GBM on base features + context, its
threshold and dedupe radius chosen by inner grouped CV on the training folds (context from inner_k there too).

  .venv/bin/python -B -m research.zeroshot.z4_selector.stage2 --s1 iter3_gbm_dgrid --tag base --out-name iter5_s2
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.custody import OUT, dev_ids, dev_inventory
from research.zeroshot.z4_selector.selector import (best_threshold, load_table, make_model, nms, _pool_recall)
from research.jev.pool_v2 import pair_metrics_matrix

CTX_KEYS = ["s1", "par4_max", "par8_max", "par15_max", "par8_n_higher", "par15_n_higher", "is_best_par8",
            "is_best_par15", "best_par15_offset", "col_max", "scan_top", "scan_n_gt05", "s1_over_top"]


def context(ends: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = len(s)
    out = np.zeros((n, len(CTX_KEYS)))
    if n == 0:
        return out
    m = pair_metrics_matrix(ends, ends)
    ang, off, ov = m["axis_angle_deg"], m["transverse_offset_mm"], m["shorter_span_overlap"]
    np.fill_diagonal(ov, 0.0)
    par = (ang <= 5.0) & (ov >= 0.5)
    col = (ang <= 3.0) & (off <= 3.0) & (m["gold_span_overlap"] <= 0.05)
    np.fill_diagonal(col, False)
    top = float(s.max())
    for i in range(n):
        row = [s[i]]
        for r in (4.0, 8.0, 15.0):
            msk = par[i] & (off[i] <= r)
            row.append(float(s[msk].max()) if msk.any() else 0.0)
        p8 = par[i] & (off[i] <= 8.0)
        p15 = par[i] & (off[i] <= 15.0)
        row += [float((s[p8] > s[i]).sum()), float((s[p15] > s[i]).sum()),
                float(not (s[p8] > s[i]).any()), float(not (s[p15] > s[i]).any())]
        if p15.any():
            j = np.flatnonzero(p15)[np.argmax(s[p15])]
            row.append(float(off[i, j]))
        else:
            row.append(99.0)
        row.append(float(s[col[i]].max()) if col[i].any() else 0.0)
        row += [top, float((s > 0.5).sum()), s[i] / max(top, 1e-6)]
        out[i] = row
    return out


def ctx_for(T, rows_of, scores):
    C = np.zeros((len(T["y"]), len(CTX_KEYS)))
    for sid, m in rows_of.items():
        if len(m):
            C[m] = context(T["ends"][m], scores[m])
    return C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dedupe-mm", default="4,6,8,10,12")
    a = ap.parse_args(argv)
    dgrid = [float(x) for x in a.dedupe_mm.split(",")]
    inv = dev_inventory()
    folds = D.make_folds(inv)
    sfold = folds["scan_fold"]
    T = load_table(a.tag, inv)
    Z = np.load(OUT / a.s1 / "oof_scores.npz")
    assert len(Z["oof"]) == len(T["y"]) and (Z["sid"] == T["sid"]).all(), "stage-1 table mismatch"
    X, y = T["X"], T["y"]
    rows_of = {sid: np.flatnonzero(T["sid"] == sid) for sid in dev_ids()}
    fa = np.array([sfold[s] for s in T["sid"]])
    C_oof = ctx_for(T, rows_of, Z["oof"])
    s2 = np.full(len(y), np.nan)
    chosen = {}
    for k in range(folds["k"]):
        tr = fa != k
        C_in = ctx_for(T, rows_of, Z[f"inner_{k}"])  # training rows: stage-1 scores never fitted on fold k
        Xtr = np.hstack([X[tr], C_in[tr]])
        mdl = make_model("gbm").fit(Xtr, y[tr])
        s2[~tr] = mdl.predict(np.hstack([X[~tr], C_oof[~tr]]))
        # inner CV for threshold / dedupe on the training folds (context from inner_k)
        inner = np.full(len(y), np.nan)
        for j in range(folds["k"]):
            if j == k:
                continue
            itr, ite = tr & (fa != j), fa == j
            inner[ite] = make_model("gbm").fit(np.hstack([X[itr], C_in[itr]]), y[itr]).predict(
                np.hstack([X[ite], C_in[ite]]))
        train_sids = [s for s in dev_ids() if sfold[s] != k and len(rows_of[s])]
        thr, dd, f1i = best_threshold(T, rows_of, inner, train_sids, dgrid)
        chosen[k] = {"threshold": thr, "dedupe_mm": dd, "inner_f1": round(f1i, 4)}
        print(k, chosen[k], flush=True)
    recs = []
    ev = OUT / a.out_name / "eval_lines"
    ev.mkdir(parents=True, exist_ok=True)
    for sid in dev_ids():
        m = rows_of[sid]
        c = chosen[sfold[sid]]
        keep = nms(T["ends"][m], s2[m], c["threshold"], c["dedupe_mm"]) if len(m) else []
        lines = [{"line_start": T["ends"][m][i][0].tolist(), "line_end": T["ends"][m][i][1].tolist(),
                  "score": float(s2[m][i])} for i in keep]
        recs.append(D.score_lines(sid, inv, lines))
        (ev / f"{sid}.json").write_text(json.dumps({"schema": "sherman.broad_baseline.eval_lines.v1",
                                                     "lane": f"z4_{a.out_name}", "frame": "scan_raw",
                                                     "status": "OK" if lines else "NO_PREDICTIONS",
                                                     "owned_bend_regions": [{"bend_id": f"Z{i+1}", **l} for i, l in enumerate(lines)]}))
    from sklearn.metrics import roc_auc_score
    res = {"s1": a.s1, "tags": a.tag, "chosen": chosen, "oof_auc_s1": round(float(roc_auc_score(y, Z["oof"])), 4),
           "oof_auc_s2": round(float(roc_auc_score(y, s2)), 4), "oof": D.aggregate(recs),
           "pool_any_match_recall": _pool_recall(T)}
    (OUT / a.out_name / "summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / a.out_name / "oof_scores.npz", oof=s2, y=y, sid=T["sid"], ends=T["ends"])
    print(json.dumps({k: res[k] for k in ("oof_auc_s1", "oof_auc_s2", "oof")}, indent=1))


if __name__ == "__main__":
    main()
