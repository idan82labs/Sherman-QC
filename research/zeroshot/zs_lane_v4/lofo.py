"""Leave-family-out cross-validation on the v4 pool (PROTOCOL.md section 3).

Outer folds: K = 10 by family5, families ordered by sha256("zs_v4|" + family5), fold = rank mod 10.
Inner folds: inside an outer-training set, ordered by sha256("zs_v4_inner|" + family5), fold = rank mod 5.
A procedure is a pair (fit(train_items) -> model, predict(model, item) -> lines); ``run`` predicts every pool scan
exactly once with a model fitted without its family. Items are the loaded cache records with labels attached.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np

from research.zeroshot.zs_lane_v4 import cache_io as CI, evalkit as EK

K_OUTER, K_INNER = 10, 5


def _order(fams, salt: str) -> list[str]:
    return sorted(set(fams), key=lambda f: hashlib.sha256((salt + f).encode()).hexdigest())


def outer_folds(pool: list[dict]) -> dict[str, int]:
    return {f: i % K_OUTER for i, f in enumerate(_order([s["family5"] for s in pool], "zs_v4|"))}


def inner_folds(families) -> dict[str, int]:
    return {f: i % K_INNER for i, f in enumerate(_order(families, "zs_v4_inner|"))}


class Item:
    """One pool scan: its pool row, cache record and gold lines (evaluation side)."""

    def __init__(self, s: dict, d: dict):
        self.s = s
        self.case_id = s["case_id"]
        self.family = s["family5"]
        self.origin = s["origin"]
        self.zero_bend = bool(s["zero_bend"])
        self.d = d
        self.gold = EK.gold_lines(s)
        self._labels: dict[str, np.ndarray] = {}

    def labels(self, key: str, segs: list[tuple]) -> np.ndarray:
        if key not in self._labels:
            self._labels[key] = EK.match_label(segs, self.gold) if segs else np.zeros(0)
        return self._labels[key]


def load_items(pool: list[dict] | None = None) -> list[Item]:
    pool = pool or EK.load_pool()
    out = []
    for s in pool:
        d = CI.load(s["case_id"])
        if d is None:
            raise FileNotFoundError(f"cache missing for {s['case_id']}")
        out.append(Item(s, d))
    return out


def run(items: list[Item], fit: Callable[[list[Item]], Any], predict: Callable[[Any, Item], list[dict]],
        folds: dict[str, int] | None = None, log=print) -> tuple[dict[str, list[dict]], dict[int, Any]]:
    folds = folds or outer_folds([it.s for it in items])
    pred, models = {}, {}
    for f in range(K_OUTER):
        tr = [it for it in items if folds[it.family] != f]
        te = [it for it in items if folds[it.family] == f]
        m = fit(tr)
        models[f] = m
        for it in te:
            pred[it.case_id] = predict(m, it)
        if log:
            log(f"  fold {f}: train {len(tr)} scans / test {len(te)}")
    return pred, models


def score(items: list[Item], pred: dict[str, list[dict] | None], sc: EK.Scorer | None = None) -> tuple[dict, list[dict]]:
    sc = sc or EK.Scorer([it.s for it in items])
    recs = [sc.score(it.case_id, pred.get(it.case_id), diagnose=True) for it in items]
    return EK.aggregate(recs), recs


def summary(agg: dict) -> dict:
    return {k: agg.get(k) for k in ("scans", "families", "tp", "fp", "fn", "P", "R", "micro_f1", "micro_ci95",
                                    "family_macro_f1", "macro_ci95", "controls", "fn_causes", "fp_causes")}


def lines_json(pred: dict[str, list[dict] | None]) -> dict:
    return {k: (None if v is None else [{"start": [round(float(x), 6) for x in l["start"]],
                                         "end": [round(float(x), 6) for x in l["end"]],
                                         **{kk: l[kk] for kk in ("score", "op", "fix", "src", "gen") if kk in l}}
                                        for l in v]) for k, v in pred.items()}


def dump(path, obj) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
