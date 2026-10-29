"""zs_lane_v3 core: zs_lane_v2's pipeline, unchanged, followed by the two v3 line fixes (fixes.py).

Pipeline (no CAD, no labels, no part-specific inputs in the research lane):
  1-6. zs_lane_v2 exactly (frozen modules imported read-only: z3s, Z2 run_scan, pooling, D1 snap of every pooled
       candidate, v2 evidence, the pooled GBM at the fixed threshold 0.5, postproc.postprocess with the frozen
       v2 post-processing). ``process`` mirrors zs_lane_v2.core.process; it only adds two optional inputs used
       by the production CAD-prior path (a thickness and an axis-family cap for Z3). Without them the DEV dump
       is the v2 dump.
  7.   Z3-alone selector probability for every candidate with Z3 features (21 Z3 features, the zs_lane_v1 z3
       recipe), used by Z3R only.
  8.   fixes.apply: EXT, then Z3R.

Revision 2 (runtime and memory only; the same lines on every scan within the budget):
  * Z3 segment merging is ``z3_merge.merge_segments`` (exact: the frozen v2 function's output with bounded work);
  * Z2 runs before Z3 and only its points / normals / t are kept for the snap fallback, so the two stages' peaks do
    not add up (Z2 and Z3 are independent and deterministic: the results do not depend on the order);
  * a face budget (``budget.py``, ``BUDGET_MAX_FACES`` from the frozen config): a scan with more triangles runs on a
    resampled copy of itself; a scan within the budget is used unchanged.

Revision 3 (budget step only): the resampling is density-invariant (``budget.py``: a fixed absolute grid of
``BUDGET_CELL_MM`` = config ``budget.cell_mm``, laddered only to stay within ``max_faces``; one QEF vertex per
(cell, skin), one quad per grid-edge crossing). Every other stage is revision 2's.
"""
from __future__ import annotations

import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v2 import core as V2
from research.zeroshot.zs_lane_v2.core import REPO, sha256_file  # noqa: F401  (REPO also sets sys.path)

LANE = "zs_lane_v3"
Z3_FEATS = 21
BUDGET_MAX_FACES: int | None = None      # revision 2: set by configure() from config["budget"]["max_faces"]
BUDGET_CELL_MM: float | None = None      # revision 3: set by configure() from config["budget"]["cell_mm"]


def configure(cfg: dict, variant: str = "zs") -> None:
    """Apply a frozen config's Z3 / Z2 / evidence settings to the reused v2 stage functions."""
    v = cfg["variants"][variant]
    V2.Z3_PARAMS = cfg.get("z3_params") or None
    V2.Z2_FALLBACK = bool(cfg.get("z2_fallback", False))
    V2.EVIDENCE_SPLITS = bool((v.get("post") or {}).get("split", False))
    global BUDGET_MAX_FACES, BUDGET_CELL_MM
    BUDGET_MAX_FACES = (cfg.get("budget") or {}).get("max_faces")
    BUDGET_CELL_MM = (cfg.get("budget") or {}).get("cell_mm")


def run_z3(path: str, *, thickness: dict | None = None, max_families: int | None = None) -> dict:
    """v2's Z3 stage, with an optional external thickness and family cap (production CAD priors). Revision 2: the
    exact bounded segment merge (z3_merge.detect = the frozen z3s.detect with z3_merge.merge_segments)."""
    from research.zeroshot.zs_lane_v2 import z3s
    from research.zeroshot.zs_lane_v3 import z3_merge
    t0 = time.time()
    mesh = z3s.load_mesh(path)
    kw = dict(V2.Z3_PARAMS or {})
    if max_families is not None:
        kw["max_families"] = int(max_families)
    res = z3_merge.detect(mesh, z3s.Params(**kw), thickness=thickness)
    res["_mesh"] = mesh
    res["timing_s"]["wall"] = time.time() - t0
    return res


def process(path: str, *, thickness: dict | None = None, max_families: int | None = None) -> dict:
    """Detection, pooling, snap of every pooled candidate and the v2 evidence (zs_lane_v2.core.process with
    use_z2 = use_z3 = snap_all = True). The selection is ``select_lines``. Revision 2: face budget, Z2 before Z3."""
    with tempfile.TemporaryDirectory(prefix="zs_v3_budget_") as work:
        return _process(path, work, thickness=thickness, max_families=max_families)


def _process(path: str, work: str, *, thickness: dict | None, max_families: int | None) -> dict:
    from research.zeroshot.zs_lane_v3 import budget
    rec: dict[str, Any] = {"stages": {}, "errors": {}}
    t0 = time.time()
    z3 = z2_lines = z2_scan = None
    try:
        path, binfo = budget.budget_scan(path, BUDGET_MAX_FACES, work, BUDGET_CELL_MM)
    except Exception as e:  # noqa: BLE001  (the budget step failing leaves the scan unchanged)
        binfo = {"applied": False, "error": f"{type(e).__name__}: {e}"}
    rec["stages"]["budget"] = binfo
    try:
        z2_lines, scan, info = V2.run_z2(path)
        z2_scan = SimpleNamespace(P=scan.P, N=scan.N, t=scan.t)      # all the snap fallback reads
        del scan
        rec["stages"]["z2"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
    except Exception as e:  # noqa: BLE001
        rec["errors"]["z2"] = f"{type(e).__name__}: {e}"
        rec["errors"]["z2_trace"] = traceback.format_exc()[-2000:]
    try:
        z3 = run_z3(path, thickness=thickness, max_families=max_families)
        rec["stages"]["z3"] = {"wall": round(z3["timing_s"]["wall"], 3), "t_mm": z3["t_mm"],
                               "n_candidates": len(z3["candidates"]), "area_total": z3.get("area_total"),
                               "n_families": len(z3.get("families", [])),
                               "family_fallback": bool(z3.get("family_fallback"))}
    except Exception as e:  # noqa: BLE001
        rec["errors"]["z3"] = f"{type(e).__name__}: {e}"
        rec["errors"]["z3_trace"] = traceback.format_exc()[-2000:]
    t1 = time.time()
    cands, X = V2.pool(z3, z2_lines, True)
    rec["stages"]["pool"] = {"wall": round(time.time() - t1, 3), "n": len(cands)}
    rec["z2_lines"] = z2_lines or []
    pts = V2.snap_points(z2_scan, z3)
    rec["_snap_pts"] = pts
    t2 = time.time()
    V2.snap(cands, pts)
    rec["stages"]["snap_all"] = {"wall": round(time.time() - t2, 3), "n": len(cands)}
    t3 = time.time()
    try:
        rec["evidence"] = V2.evidence_stage(cands, z3)
    except Exception as e:  # noqa: BLE001
        rec["errors"]["evidence"] = f"{type(e).__name__}: {e}"
        rec["errors"]["evidence_trace"] = traceback.format_exc()[-2000:]
        rec["evidence"] = None
    rec["stages"]["evidence"] = {"wall": round(time.time() - t3, 3)}
    rec["candidates"] = cands
    rec["X"] = X
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["peak_rss_gib_process"] = round(V2._rss_gib(), 3)
    rec["status"] = "OK" if (z3 is not None or z2_lines is not None) else "FAILED"
    return rec


def z3_probs(model: dict | None, cands: list[dict], X: np.ndarray) -> np.ndarray:
    """Z3-alone selector probability per candidate (NaN where the candidate has no Z3 features)."""
    p = np.full(len(cands), np.nan)
    rows = [i for i, c in enumerate(cands) if c.get("has_z3")]
    if model is not None and rows:
        p[rows] = model["model"].predict(np.asarray(X, float)[rows][:, :Z3_FEATS])
    return p


def select(cands: list[dict], X: np.ndarray, ev: dict | None, z2_lines: list[dict], v: dict, zs_model: dict,
           z3_model: dict | None, *, prob: np.ndarray | None = None, prob_z3: np.ndarray | None = None
           ) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """v2 selection + span post-processing, then EXT and Z3R. ``prob`` / ``prob_z3`` may be given (out-of-fold
    evaluation); otherwise they are predicted with the models. Returns (lines, prob, prob_z3)."""
    from research.zeroshot.zs_lane_v2.postproc import PostParams, postprocess, features, sweep_matrix
    from research.zeroshot.zs_lane_v3.fixes import FixParams, apply
    if not cands:
        return [], np.zeros(0), np.zeros(0)
    if prob is None:
        SW = sweep_matrix((ev or {}).get("sweep"), len(cands))
        prob = zs_model["model"].predict(features(X, SW, zs_model.get("feats", "base")))
    if prob_z3 is None:
        prob_z3 = z3_probs(z3_model, cands, X)
    out = postprocess(cands, prob, float(v["threshold"]), ev, PostParams(**(v.get("post") or {})))
    for o in out:
        o.pop("_i", None)
    out = apply(out, cands, z2_lines, prob, prob_z3, (ev or {}).get("sweep") if ev else None,
                FixParams(**(v.get("fix") or {})))
    return out, np.asarray(prob, float), np.asarray(prob_z3, float)


def select_lines(rec: dict, v: dict, zs_model: dict | None, z3_model: dict | None) -> list[dict]:
    if zs_model is None or not rec.get("candidates"):
        return []
    lines, _, _ = select(rec["candidates"], rec["X"], rec.get("evidence"), rec.get("z2_lines") or [], v,
                         zs_model, z3_model)
    return lines


def payload(path: str, scan_sha: str | None, lines: list[dict], *, status: str, conversion: dict,
            lane: str = LANE) -> dict:
    """eval_lines.json (sherman.broad_baseline.eval_lines.v1) via the v2 writer, plus the v3 fix stamp."""
    pay = V2.payload(path, scan_sha, lines, status=status, conversion=conversion, lane=lane)
    for reg, ln in zip(pay.get("owned_bend_regions") or [], lines):
        if "fix" in ln:
            reg["fix"] = ln["fix"]
    return pay
