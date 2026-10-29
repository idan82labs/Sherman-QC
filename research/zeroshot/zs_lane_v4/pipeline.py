"""zs_lane_v4 end-to-end runtime pipeline: scan -> lines (research candidate; production stays zs_lane_v1).

  1. core.process: v3 rev3 detection stages (face budget, Z2 + fallback, Z3 axis-family slicing, pooling, D1 snap
     of every candidate, v2 evidence) + support profiles + scan metadata;
  2. optional generators / checks selected by the frozen config: the hem generator (hem.py) and the Z3 re-slice
     stability check (stability.py), both on the same mesh and thickness;
  3. features (features.py; + stability columns when enabled) and, when enabled, extent hypotheses (hypotheses.py);
  4. the frozen GBM selector, then select.select_h (hypotheses) or select.select, with the frozen SelParams.
Deterministic: every stage is single-threaded numpy/scipy/sklearn with fixed seeds and no randomness at inference.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v4 import core, features as FE, hypotheses as HY, select as SL

STAB_FEATS = ["st_match", "st_len_ratio", "st_support"]


def stab_matrix(rec: dict, stab: dict | None) -> np.ndarray:
    n = len(rec.get("candidates") or [])
    M = np.full((n, len(STAB_FEATS)), np.nan)
    if not stab:
        return M
    for i in range(n):
        m = stab["match"][i] if i < len(stab["match"]) else None
        if m is None:
            continue
        M[i] = [float(bool(m)), stab["len_ratio"][i], stab["support_frac"][i]]
    return M


def matrices(rec: dict, cfg: dict, stab: dict | None) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """(candidate rows, hypotheses, hypothesis rows) with the columns of the frozen config."""
    C = rec.get("candidates") or []
    base = FE.full_matrix(rec, stab_matrix(rec, stab) if cfg.get("stab") else None)
    alts = HY.build(rec) if (cfg.get("hyp") and C) else []
    Hb, Xa = HY.features(rec, alts, base)
    X = np.hstack([base, Hb]) if len(C) else np.zeros((0, base.shape[1] + len(HY.ALT_FEATS)))
    cols = cfg.get("cols")
    if cols is not None:
        X = X[:, cols]
        Xa = Xa[:, cols] if len(alts) else np.zeros((0, len(cols)))
    return X, alts, Xa


def run(path: str, cfg: dict, model, *, thickness: float | None = None, max_families: int = 20) -> dict[str, Any]:
    t0 = time.time()
    rec = core.process(path, thickness=thickness, max_families=max_families, keep_mesh=True)
    stages = dict(rec["stages"])
    if rec.get("status") != "OK":
        return {"status": "FAILED", "lines": [], "stages": stages, "errors": rec.get("errors"), "rec": rec}
    mesh, t = rec.pop("_mesh", None), rec.pop("_t_mm", None)
    hem_lines, stab = None, None
    if mesh is not None and t is not None:
        if cfg.get("hem"):
            from research.zeroshot.zs_lane_v4 import hem
            th = time.time()
            hem_lines = hem.detect(mesh, float(t), hem.HemParams(**(cfg.get("hem_params") or {})))
            stages["hem"] = {"wall": round(time.time() - th, 3), "n": len(hem_lines)}
        if cfg.get("stab"):
            from research.zeroshot.zs_lane_v4 import stability
            ts = time.time()
            stab = stability.restability(mesh, float(t), rec.get("candidates") or [], max_families)
            stages["stab"] = {"wall": round(time.time() - ts, 3), "n_rerun": stab["n_rerun"]}
    del mesh
    t1 = time.time()
    X, alts, Xa = matrices(rec, cfg, stab)
    C = rec.get("candidates") or []
    prob = model.predict(X) if len(X) else np.zeros(0)
    prob_a = model.predict(Xa) if len(Xa) else np.zeros(0)
    prm = SL.SelParams(**cfg["select"])
    if cfg.get("hyp") and prm.hyp:
        lines = SL.select_h(rec, alts, prob, prob_a, prm, hem_lines)
    else:
        lines = SL.select(rec, prob, prm, hem_lines)
    stages["select"] = {"wall": round(time.time() - t1, 3), "n": len(lines), "n_cand": len(C), "n_hyp": len(alts)}
    return {"status": "OK", "lines": lines, "stages": stages, "errors": rec.get("errors"), "rec": rec,
            "wall_s": round(time.time() - t0, 3), "hem_lines": hem_lines, "stab": stab}
