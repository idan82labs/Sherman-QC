"""LOFO arms A1 (v1 retrained) and A3 (v3 retrained) on the v4 cache (PROTOCOL.md section 4).

A1: the pinned v1 blind candidate pipeline (cache "v1"), the zs_lane_v1 GBM recipe (z3_slicing.selector.GBM on the
    26 pooled features), fixed threshold 0.5, lines at the snapped placement. Only the GBM is refit.
A3: the v3 rev3 candidates and evidence (cache "v3x" for budgeted scans, else "v4": exactly the pinned v3 inputs),
    the pooled GBM (26 features) and the Z3-alone GBM (21 Z3 features, has_z3 rows) refit, then v3's frozen
    post-processing (join, dedupe, sweep gate) and fixes (EXT, Z3R) with the pinned values.
"""
from __future__ import annotations

import numpy as np

from research.zeroshot.zs_lane_v4.lofo import Item


def _gbm():
    from research.zeroshot.z3_slicing.selector import GBM
    return GBM()


# ------------------------------------------------------------------ A1
def a1_rec(it: Item) -> dict:
    return it.d["v1"]


def a1_xy(it: Item) -> tuple[np.ndarray, np.ndarray]:
    r = a1_rec(it)
    C = r.get("candidates") or []
    if r.get("status") != "OK" or not C:
        return np.zeros((0, 26)), np.zeros(0)
    y = it.labels("v1", [(c["start_s"], c["end_s"]) for c in C])
    return r["X"], y


def a1_fit(tr: list[Item]):
    X, Y = zip(*[a1_xy(it) for it in tr])
    X = np.vstack([x for x in X if len(x)])
    Y = np.concatenate([y for y in Y if len(y)])
    return _gbm().fit(X, Y)


def a1_predict(m, it: Item) -> list[dict] | None:
    r = a1_rec(it)
    if r.get("status") != "OK":
        return None
    C = r.get("candidates") or []
    if not C:
        return []
    p = m.predict(r["X"])
    return [{"start": c["start_s"], "end": c["end_s"], "score": float(q), "src": c.get("src")}
            for c, q in zip(C, p) if q >= 0.5]


# ------------------------------------------------------------------ A3
def a3_rec(it: Item) -> dict:
    return it.d.get("v3x") or it.d["v4"]


def a3_xy(it: Item):
    r = a3_rec(it)
    C = r.get("candidates") or []
    if r.get("status") != "OK" or not C:
        return np.zeros((0, 26)), np.zeros(0), np.zeros(0, bool)
    key = "v3x" if it.d.get("v3x") else "v4"
    y = it.labels(key, [(c["start_s"], c["end_s"]) for c in C])
    hz = np.array([bool(c.get("has_z3")) for c in C])
    return r["X"], y, hz


def a3_fit(tr: list[Item]):
    xs, ys, hs = zip(*[a3_xy(it) for it in tr])
    X = np.vstack([x for x in xs if len(x)])
    Y = np.concatenate([y for y in ys if len(y)])
    H = np.concatenate([h for h in hs if len(h)])
    mz = _gbm().fit(X, Y)
    m3 = _gbm().fit(X[H][:, :21], Y[H])
    return {"zs": {"model": mz}, "z3": {"model": m3}}


def a3_predict(m, it: Item) -> list[dict] | None:
    from research.zeroshot.zs_lane_v4 import cache_io as CI
    from research.zeroshot.zs_lane_v3.core import z3_probs
    r = a3_rec(it)
    if r.get("status") != "OK":
        return None
    C = r.get("candidates") or []
    if not C:
        return []
    prob = m["zs"]["model"].predict(r["X"])
    p3 = z3_probs(m["z3"], C, r["X"])
    return CI.v3_lines(r, prob=prob, prob_z3=p3)
