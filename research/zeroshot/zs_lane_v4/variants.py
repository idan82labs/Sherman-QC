"""Named v4 procedure variants (each one scored under nested LOFO is logged in results/DESIGN_LOG.jsonl).

A variant fixes: the feature columns of features.full_matrix, the GBM settings and the selection grid that the inner
CV chooses from. Declared here, in code, before it is run.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict

import numpy as np

from research.zeroshot.zs_lane_v4 import features as FE, select as SL
from research.zeroshot.zs_lane_v4.train import Spec, grid_product

N26 = 26
ALL = None
BASE = list(range(N26))


def cols_of(names: list[str]) -> list[int]:
    return BASE + [N26 + FE.V4_FEATS.index(n) for n in names]


NOHYP = list(range(N26 + len(FE.V4_FEATS)))                 # 26 pooled + V4_FEATS (no hypothesis block)
NOHYP_STAB = list(range(N26 + len(FE.V4_FEATS) + 3))        # + the 3 re-slice stability columns

def alt_block(stab: bool = False) -> list[int]:
    from research.zeroshot.zs_lane_v4.hypotheses import ALT_FEATS
    o = N26 + len(FE.V4_FEATS) + (3 if stab else 0)
    return list(range(o, o + len(ALT_FEATS)))


BASE_ALT = BASE + alt_block()

V3SEL = SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="v3")
SAFE = SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="safe")

GRID_MAIN = grid_product(thr=[0.4, 0.5, 0.6], sweep_min=[0.0, 0.25], join=[False, True],
                         ext=["off", "v3", "safe", "safe_sup"], nms=[False, True])

VARIANTS: dict[str, Spec] = {
    # sanity: v4 candidates, the 26 pooled features, v3's selection without Z3R
    "v4_00_base26_v3sel": Spec("v4_00_base26_v3sel", cols=BASE, grid=[V3SEL]),
    # the tab fix alone: v3 selection with the safe EXT
    "v4_01_base26_safe": Spec("v4_01_base26_safe", cols=BASE, grid=[SAFE]),
    # all v4 features, safe selection
    "v4_02_all_safe": Spec("v4_02_all_safe", cols=NOHYP, grid=[SAFE]),
    # all v4 features, nested selection over the main grid
    "v4_03_all_grid": Spec("v4_03_all_grid", cols=NOHYP, grid=GRID_MAIN),
    # feature-group ablations with the safe selection (v4_02 with every group was worse than base26)
    "v4_06_sweep_safe": Spec("v4_06_sweep_safe", cols=cols_of(FE.SWEEP_FEATS), grid=[SAFE]),
    "v4_07_noscan_safe": Spec("v4_07_noscan_safe", cols=cols_of(FE.SWEEP_FEATS + FE.PROF_FEATS + FE.SPLIT_FEATS
                                                              + FE.NB_FEATS + FE.Z2_FEATS), grid=[SAFE]),
    "v4_08_scan_safe": Spec("v4_08_scan_safe", cols=cols_of(FE.SCAN_FEATS), grid=[SAFE]),
    # base26 with the nested selection grid (threshold / sweep gate / join / EXT mode / NMS)
    "v4_09_base26_grid": Spec("v4_09_base26_grid", cols=BASE, grid=GRID_MAIN),
    # extent hypotheses with the base26 + hypothesis features
    "v4_10_hyp_base": Spec("v4_10_hyp_base", cols=BASE_ALT, hyp=True,
                           grid=[SL.SelParams(thr=0.5, sweep_min=0.25, hyp=True, alt_guard=True)]),
    "v4_11_hyp_grid": Spec("v4_11_hyp_grid", cols=BASE_ALT, hyp=True,
                           grid=grid_product(hyp=[True], thr=[0.4, 0.5, 0.6], sweep_min=[0.0, 0.25],
                                             alt_guard=[True, False], nms=[False, True]) + GRID_MAIN),
    # EXT mode at fixed threshold 0.5 (the inner-CV threshold choice of v4_09 did not transfer)
    "v4_12_base26_safesup": Spec("v4_12_base26_safesup", cols=BASE,
                                 grid=[SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="safe_sup")]),
    "v4_13_base26_noext": Spec("v4_13_base26_noext", cols=BASE,
                               grid=[SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="off")]),
    # + the hem generator (pass-2 cache), + matcher-equivalence NMS, + re-slice stability features
    "v4_14_safesup_hem": Spec("v4_14_safesup_hem", cols=BASE, hem=True,
                              grid=[SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="safe_sup", hem=True)]),
    "v4_15_safesup_hem_nms": Spec("v4_15_safesup_hem_nms", cols=BASE, hem=True,
                                  grid=[SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="safe_sup", hem=True,
                                                     nms=True)]),
    "v4_16_safesup_hem_stab": Spec("v4_16_safesup_hem_stab", cols=BASE + [N26 + len(FE.V4_FEATS) + k for k in range(3)],
                                   hem=True, stab=True,
                                   grid=[SL.SelParams(thr=0.5, sweep_min=0.25, join=True, ext="safe_sup", hem=True)]),
    # extent hypotheses compete with the candidates (hypotheses.py, select.select_h), guard on
    "v4_04_hyp_fixed": Spec("v4_04_hyp_fixed", cols=ALL, hyp=True,
                            grid=[SL.SelParams(thr=0.5, sweep_min=0.25, hyp=True, alt_guard=True)]),
    # hypotheses, nested selection over threshold / sweep gate / guard / NMS, plus the non-hypothesis main grid
    "v4_05_hyp_grid": Spec("v4_05_hyp_grid", cols=ALL, hyp=True,
                           grid=grid_product(hyp=[True], thr=[0.4, 0.5, 0.6], sweep_min=[0.0, 0.25],
                                             alt_guard=[True, False], nms=[False, True]) + GRID_MAIN),
}


def get(name: str) -> Spec:
    return VARIANTS[name]


def colnames(stab: bool = False) -> list[str]:
    from research.zeroshot.z3_slicing.pooled import EXTRA
    from research.zeroshot.z3_slicing.selector import FEATS_BASE
    from research.zeroshot.zs_lane_v4.hypotheses import ALT_FEATS
    from research.zeroshot.zs_lane_v4.pipeline import STAB_FEATS
    return (list(FEATS_BASE) + list(EXTRA) + list(FE.V4_FEATS) + (list(STAB_FEATS) if stab else [])
            + list(ALT_FEATS))


def describe(spec: Spec) -> dict:
    names = colnames(spec.stab)
    return {"name": spec.name, "n_cols": len(names) if spec.cols is None else len(spec.cols),
            "cols": "all" if spec.cols is None else ([names[c] for c in spec.cols if c >= N26] or "base26"),
            "gbm": spec.gbm, "grid_size": len(spec.grid), "grid_first": spec.grid[0].key(), "hem": spec.hem,
            "hyp": spec.hyp, "stab": spec.stab}


def hem_lines(spec: Spec, items) -> dict:
    """case_id -> hem lines from the pass-2 cache (cache/<case_id>.p2.json.gz), when the variant uses hems."""
    if not spec.hem:
        return {}
    from research.zeroshot.zs_lane_v4.cache_io import CACHE
    out = {}
    for it in items:
        p = CACHE / f"{it.case_id}.p2.json.gz"
        if p.is_file():
            with gzip.open(p, "rt") as fh:
                out[it.case_id] = json.load(fh).get("hem") or []
    return out


def stab_of(spec: Spec, items) -> dict | None:
    """case_id -> re-slice stability record from the pass-2 cache, when the variant uses it."""
    if not spec.stab:
        return None
    from research.zeroshot.zs_lane_v4.cache_io import CACHE
    out = {}
    for it in items:
        p = CACHE / f"{it.case_id}.p2.json.gz"
        if p.is_file():
            with gzip.open(p, "rt") as fh:
                out[it.case_id] = json.load(fh).get("stab")
    return out
