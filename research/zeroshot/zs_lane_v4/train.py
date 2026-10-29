"""zs_lane_v4 nested LOFO procedure and final fit (evaluation side; PROTOCOL.md section 3).

Procedure P (what LOFO estimates and what the final freeze runs on the whole pool):
  1. features: features.full_matrix of every pooled candidate (26 pooled + V4_FEATS);
  2. inner CV (5 grouped inner folds of the training families): a GBM fit on 4 inner folds predicts the 5th, giving
     inner out-of-fold probabilities for every training scan; each SelParams of the grid turns them into lines;
     the grid entry with the best pooled micro-F1 (FastScorer == frozen matcher) wins, ties to the first entry;
  3. a GBM fit on all training scans + the chosen SelParams = the fitted procedure.
GBM: sklearn HistGradientBoostingClassifier(max_depth 3, learning_rate 0.05, max_iter 300, l2 1.0,
random_state 0, early_stopping False) -- the zs_lane_v1 recipe with early stopping pinned off.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v4 import evalkit as EK, features as FE, hypotheses as HY, lofo as LO, select as SL
from research.zeroshot.zs_lane_v4.model import GBM


@dataclass
class Spec:
    """A v4 procedure variant: feature columns, GBM settings, the selection grid."""
    name: str
    cols: list[int] | None = None          # columns of full_matrix to use (None = all)
    gbm: dict = field(default_factory=dict)
    grid: list[SL.SelParams] = field(default_factory=lambda: [SL.SelParams()])
    hem: bool = False
    hyp: bool = False
    stab: bool = False


class V4Data:
    """Per-item feature matrices (pooled candidates and, for hypothesis variants, their extent hypotheses), labels
    and records, computed with exactly the runtime code (pipeline.matrices). Columns: 26 pooled | features.V4_FEATS
    | (pipeline.STAB_FEATS when ``stab``) | hypotheses.ALT_FEATS."""

    def __init__(self, items: list[LO.Item], hem: dict | None = None, hyp: bool = False, stab: dict | None = None):
        from research.zeroshot.zs_lane_v4 import pipeline as PL
        self.items = items
        self.by_id = {it.case_id: it for it in items}
        self.hyp = hyp
        self.X: dict[str, np.ndarray] = {}
        self.y: dict[str, np.ndarray] = {}
        self.alts: dict[str, list[dict]] = {}
        self.Xa: dict[str, np.ndarray] = {}
        self.ya: dict[str, np.ndarray] = {}
        self.rec: dict[str, dict] = {}
        self.hem = hem or {}
        cfg = {"stab": stab is not None, "hyp": hyp, "cols": None}
        for it in items:
            r = it.d["v4"]
            self.rec[it.case_id] = r
            C = r.get("candidates") or []
            X, alts, Xa = PL.matrices(r, cfg, (stab or {}).get(it.case_id))
            self.X[it.case_id] = X
            self.y[it.case_id] = it.labels("v4", [(c["start_s"], c["end_s"]) for c in C]) if C else np.zeros(0)
            self.alts[it.case_id] = alts
            self.Xa[it.case_id] = Xa if len(alts) else np.zeros((0, X.shape[1]))
            self.ya[it.case_id] = (EK.match_label([(h["start_s"], h["end_s"]) for h in alts], it.gold)
                                   if alts else np.zeros(0))
        self.fs = EK.FastScorer(items)

    def rows(self, cid: str, cols) -> np.ndarray:
        X = np.vstack([self.X[cid], self.Xa[cid]]) if self.hyp else self.X[cid]
        return X if cols is None else X[:, cols]

    def xy(self, ids, cols):
        Xs, ys = [], []
        for i in ids:
            X = self.rows(i, cols)
            if not len(X):
                continue
            Xs.append(X)
            ys.append(np.concatenate([self.y[i], self.ya[i]]) if self.hyp else self.y[i])
        return np.vstack(Xs), np.concatenate(ys)

    def lines(self, cid: str, prob: np.ndarray, prm: SL.SelParams) -> list[dict] | None:
        r = self.rec[cid]
        if r.get("status") != "OK":
            return None
        n = len(r.get("candidates") or [])
        if self.hyp and prm.hyp:
            return SL.as_eval(SL.select_h(r, self.alts[cid], prob[:n], prob[n:], prm, self.hem.get(cid)))
        return SL.as_eval(SL.select(r, prob[:n], prm, self.hem.get(cid)))


def _predict(m: GBM, data: V4Data, cid: str, cols) -> np.ndarray:
    X = data.rows(cid, cols)
    return m.predict(X) if len(X) else np.zeros(0)


def fit_procedure(data: V4Data, train_ids: list[str], spec: Spec, log=None) -> dict:
    fams = sorted(set(data.by_id[i].family for i in train_ids))
    chosen, scores = spec.grid[0], []
    if len(spec.grid) > 1:
        inf = LO.inner_folds(fams)
        P = {}
        for g in range(LO.K_INNER):
            itr = [i for i in train_ids if inf[data.by_id[i].family] != g]
            ite = [i for i in train_ids if inf[data.by_id[i].family] == g]
            m = GBM(**spec.gbm).fit(*data.xy(itr, spec.cols))
            for i in ite:
                P[i] = _predict(m, data, i, spec.cols)
        for prm in spec.grid:
            pred = {i: data.lines(i, P[i], prm) for i in train_ids}
            scores.append(data.fs.micro(pred))
        k = int(np.argmax(np.round(scores, 6)))
        chosen = spec.grid[k]
        if log:
            log(f"    inner pick #{k} {chosen.key()} micro {scores[k]:.4f} (grid {len(spec.grid)})")
    m = GBM(**spec.gbm).fit(*data.xy(train_ids, spec.cols))
    return {"model": m, "params": chosen, "inner_scores": [round(s, 5) for s in scores]}


def lofo(data: V4Data, spec: Spec, log=print) -> tuple[dict, dict]:
    folds = LO.outer_folds([it.s for it in data.items])
    pred, fitted = {}, {}
    t0 = time.time()
    for f in range(LO.K_OUTER):
        tr = [it.case_id for it in data.items if folds[it.family] != f]
        te = [it.case_id for it in data.items if folds[it.family] == f]
        fp = fit_procedure(data, tr, spec, log=log)
        fitted[f] = {"params": fp["params"].key(), "inner_scores": fp["inner_scores"]}
        for i in te:
            pred[i] = data.lines(i, _predict(fp["model"], data, i, spec.cols), fp["params"])
        if log:
            log(f"  fold {f} done ({time.time() - t0:.0f}s)")
    return pred, fitted


def grid_product(**axes) -> list[SL.SelParams]:
    keys = list(axes)
    out = []
    for vals in itertools.product(*[axes[k] for k in keys]):
        out.append(replace(SL.SelParams(), **dict(zip(keys, vals))))
    return out
