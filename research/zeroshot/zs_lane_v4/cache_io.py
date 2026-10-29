"""Read the v4 candidate cache (docs/production/runlog/zs_v4/cache/<case_id>.json.gz) and replay the pinned lanes'
selection on it (evaluation-side: this is how "v1 pinned", "v1 retrained" and "v3" are re-created from the cache)."""
from __future__ import annotations

import gzip
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v1.core import REPO

CACHE = REPO / "docs/production/runlog/zs_v4/cache"
ASSETS = REPO / "domains/production/zeroshot/assets"


def path_of(case_id: str) -> Path:
    return CACHE / f"{case_id}.json.gz"


def load(case_id: str) -> dict | None:
    p = path_of(case_id)
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    for key in ("v1", "v4", "v3x"):
        r = d.get(key)
        if not r or "candidates" not in r:
            continue
        n = len(r["candidates"])
        r["X"] = (np.array([[np.nan if x is None else x for x in row] for row in r["X"]], float).reshape(n, -1)
                  if n else np.zeros((0, 26)))
    return d


@lru_cache(maxsize=None)
def pinned_models() -> dict:
    """The pinned production models, sha256-checked against their pins before unpickling."""
    from domains.production.zeroshot import pin as PIN
    out = {}
    for lane in ("zs_lane_v1", "zs_lane_v3"):
        pin = PIN.load_pin(REPO, lane=lane)
        for role, m in PIN.pinned_models(pin).items():
            p = REPO / m["rel_path"]
            assert PIN.sha256_file(p) == m["sha256"], p
            out[(lane, role)] = pickle.loads(p.read_bytes())
        out[(lane, "params")] = pin.get("lane_params")
    return out


def v1_select(r: dict, prob: np.ndarray, thr: float = 0.5) -> list[dict]:
    """zs_lane_v1 selection (lane_cli.run_lane): keep pooled candidates with p >= thr at their snapped placement."""
    out = []
    for c, q in zip(r["candidates"], prob):
        if q >= thr:
            out.append({"start": c["start_s"], "end": c["end_s"], "score": float(q), "src": c.get("src"),
                        "incl": c.get("incl")})
    return out


def v1_pinned_lines(d: dict) -> list[dict]:
    r = d["v1"]
    if r.get("status") != "OK" or not len(r["candidates"]):
        return []
    m = pinned_models()[("zs_lane_v1", "zs")]
    return v1_select(r, m["model"].predict(r["X"]), float(m.get("threshold", 0.5)))


def v3_lines(r: dict, prob: np.ndarray | None = None, prob_z3: np.ndarray | None = None,
             variant: dict | None = None) -> list[dict]:
    """zs_lane_v3 selection + post-processing + fixes on a v4/v3x record (pinned models unless probs given)."""
    from research.zeroshot.zs_lane_v3 import core as V3
    M = pinned_models()
    v = variant or M[("zs_lane_v3", "params")]["variant"]
    if not r.get("candidates"):
        return []
    lines, _, _ = V3.select(r["candidates"], r["X"], r.get("evidence"), r.get("z2_lines") or [], v,
                            M[("zs_lane_v3", "zs")], M[("zs_lane_v3", "z3")], prob=prob, prob_z3=prob_z3)
    return [{"start": l["start_s"], "end": l["end_s"], "score": l.get("score"), "src": l.get("src"),
             "op": l.get("op"), "fix": l.get("fix"), "parts": l.get("parts"), "incl": l.get("incl")} for l in lines]


def same_lines(a: list[dict], b: list[dict], tol: float = 1e-4) -> bool:
    """Equal as unordered sets of segments (either orientation), to ``tol`` mm."""
    if len(a) != len(b):
        return False
    used = set()
    for x in a:
        xa, xb = np.asarray(x["start"], float), np.asarray(x["end"], float)
        hit = None
        for j, y in enumerate(b):
            if j in used:
                continue
            ya, yb = np.asarray(y["start"], float), np.asarray(y["end"], float)
            if (max(np.abs(xa - ya).max(), np.abs(xb - yb).max()) <= tol
                    or max(np.abs(xa - yb).max(), np.abs(xb - ya).max()) <= tol):
                hit = j
                break
        if hit is None:
            return False
        used.add(hit)
    return True
