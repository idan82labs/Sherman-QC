"""Liberal candidate pool for the Jev input-optimisation study (v2, research only).

The old best_f1 lane keeps only a few lines per scan; its scan-only stage
(``research/broad_baseline/bb_steps.py pool``) first builds a much wider pool
(``f1_uniform_scan_only_candidate_pool_v3``: ~1000 ``candidates`` plus ~700
``line_objects`` per scan, scan-only by contract: no part id, no CAD, no truth).
This module turns that pool into a ranked-selection problem:

1. :func:`load_pool_lines` reads every pool row with ``span_endpoints`` (raw scan
   frame) and keeps the scan-only descriptors the generator already computed.
2. :func:`cluster_lines` merges near-identical rows (tight thresholds, default
   2 deg / 1.5 mm / 0.8 mutual overlap) into one candidate. The representative is
   the member with the most corridor support; cluster-level consensus (how many
   rows and how many distinct generators proposed the same line) is kept as a
   feature.
3. :func:`nms_select` is the one-to-one dedupe used at selection time: candidates
   are visited by descending score and a candidate is dropped when it is a
   near-duplicate (matcher thresholds: 5 deg, 4 mm, 0.6 overlap of the shorter
   span) of one already kept.

:func:`pair_metrics_matrix` is a vectorised copy of
``scripts/evaluate_f1_against_manual_gold.pair_metrics`` (checked by a test).
Coordinates never leave this module except as line endpoints for scoring.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_WORK_ROOT",
    "POOL_DESCRIPTOR_KEYS",
    "PoolLine",
    "load_pool_lines",
    "pool_file",
    "pair_metrics_matrix",
    "near_duplicate_matrix",
    "cluster_lines",
    "nms_select",
]

DEFAULT_WORK_ROOT = Path("/private/tmp/claude-501/sherman_broad_baseline_work/broad_v2/best_f1")

#: Scan-only descriptors that the pool generator writes for every row (candidates and line objects).
POOL_DESCRIPTOR_KEYS = (
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "corridor_point_count",
    "corridor_points_per_mm",
    "transverse_angular_span_deg",
    "transverse_angular_bin_fraction",
    "median_corridor_radius_mm",
    "normal_second_eigenvalue",
    "normal_spread_deg_scan",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_normal_second_eigenvalue_r6mm",
    "tight_to_wide_normal_second_ratio",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
)


@dataclass
class PoolLine:
    start: np.ndarray
    end: np.ndarray
    generator: str
    descriptors: dict[str, float] = field(default_factory=dict)
    radius_mm: float | None = None
    angle_deg: float | None = None
    vote_count: float | None = None

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def pool_file(case_id: str, work_root: Path = DEFAULT_WORK_ROOT) -> Path | None:
    d = Path(work_root) / case_id / "pool"
    if not d.is_dir():
        return None
    files = sorted(p for p in d.iterdir() if p.suffix == ".json" and not p.name.endswith(".row.json"))
    return files[0] if files else None


def load_pool_lines(path: Path, *, min_length_mm: float = 5.0) -> tuple[list[PoolLine], dict[str, Any]]:
    """Every pool row with ``span_endpoints``; returns ``(lines, meta)``. The pool must be scan-only."""
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = pool.get("inference_contract") or {}
    if any(contract.get(k) for k in ("part_identifier_exposed", "expected_count_exposed",
                                     "manual_truth_exposed", "cad_exposed")):
        raise ValueError(f"{path}: pool is not scan-only ({contract})")
    if pool.get("coordinate_frame") != "scan_raw":
        raise ValueError(f"{path}: pool frame is {pool.get('coordinate_frame')!r}, not scan_raw")
    out: list[PoolLine] = []
    for block in ("candidates", "line_objects"):
        for row in pool.get(block) or []:
            ends = row.get("span_endpoints")
            if not ends or len(ends) != 2:
                continue
            a, b = np.asarray(ends[0], float), np.asarray(ends[1], float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.linalg.norm(b - a) < min_length_mm:
                continue
            gen = str(row.get("candidate_source") or block)
            if block == "line_objects":
                gen = "line_object"
            desc = {k: _num(row.get(k)) for k in POOL_DESCRIPTOR_KEYS}
            out.append(PoolLine(a, b, gen, {k: v for k, v in desc.items() if v is not None},
                                radius_mm=_num(row.get("radius_mm")), angle_deg=_num(row.get("angle_deg")),
                                vote_count=_num(row.get("vote_count") or row.get("support_count"))))
    meta = {"pool_algorithm": pool.get("algorithm"), "rows": len(out),
            "scan_sha256": pool.get("source_scan_sha256"),
            "candidate_source_counts": pool.get("candidate_source_counts")}
    return out, meta


def _ends(lines: Sequence[Any]) -> np.ndarray:
    arr = []
    for ln in lines:
        if isinstance(ln, PoolLine):
            arr.append([ln.start, ln.end])
        elif isinstance(ln, Mapping):
            arr.append([ln["line_start"], ln["line_end"]])
        else:
            arr.append(ln)
    return np.asarray(arr, float).reshape(-1, 2, 3)


def pair_metrics_matrix(ref: Any, other: Any) -> dict[str, np.ndarray]:
    """Vectorised ``pair_metrics(gold=ref[i], prediction=other[j])`` for all i, j."""
    g = _ends(ref)
    p = _ends(other)
    gm = g.mean(axis=1)[:, None, :]
    pm = p.mean(axis=1)[None, :, :]
    gd = g[:, 1] - g[:, 0]
    pd = p[:, 1] - p[:, 0]
    gl = np.linalg.norm(gd, axis=1)
    pl = np.linalg.norm(pd, axis=1)
    ga = (gd / gl[:, None])[:, None, :]
    pa = (pd / pl[:, None])[None, :, :]
    dot = np.clip((ga * pa).sum(-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(np.abs(dot)))
    delta = pm - gm
    g_off = np.linalg.norm(delta - ga * (delta * ga).sum(-1, keepdims=True), axis=-1)
    pa_s = pa * np.where(dot < 0, -1.0, 1.0)[..., None]
    p_off = np.linalg.norm(delta - pa_s * (delta * pa_s).sum(-1, keepdims=True), axis=-1)
    offset = np.maximum(g_off, p_off)
    # gold interval: [-gl/2, gl/2] along ga (midpoint-centred); prediction ends projected on ga
    p0 = ((p[None, :, 0, :] - gm) * ga).sum(-1)
    p1 = ((p[None, :, 1, :] - gm) * ga).sum(-1)
    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    g0 = ((g[:, None, 0, :] - gm) * ga).sum(-1)
    g1 = ((g[:, None, 1, :] - gm) * ga).sum(-1)
    glo, ghi = np.minimum(g0, g1), np.maximum(g0, g1)
    overlap = np.maximum(0.0, np.minimum(ghi, hi) - np.maximum(glo, lo))
    return {"axis_angle_deg": angle, "transverse_offset_mm": offset,
            "gold_span_overlap": overlap / gl[:, None], "prediction_span_overlap": overlap / pl[None, :],
            "shorter_span_overlap": overlap / np.minimum(gl[:, None], pl[None, :])}


def near_duplicate_matrix(lines: Any, *, angle_deg: float, offset_mm: float, overlap: float,
                          mutual: bool = True) -> np.ndarray:
    m = pair_metrics_matrix(lines, lines)
    ov = np.minimum(m["gold_span_overlap"], m["prediction_span_overlap"]) if mutual else m["shorter_span_overlap"]
    return (m["axis_angle_deg"] <= angle_deg) & (m["transverse_offset_mm"] <= offset_mm) & (ov >= overlap)


def cluster_lines(lines: Sequence[PoolLine], *, angle_deg: float = 2.0, offset_mm: float = 1.5,
                  overlap: float = 0.8) -> list[dict[str, Any]]:
    """Greedy leader clustering. Leaders are visited by descending corridor support."""
    n = len(lines)
    if n == 0:
        return []
    dup = near_duplicate_matrix(lines, angle_deg=angle_deg, offset_mm=offset_mm, overlap=overlap)
    support = np.array([(ln.descriptors.get("corridor_point_count") or 0.0)
                        * (ln.descriptors.get("finite_support_fraction") or 0.0) for ln in lines])
    order = sorted(range(n), key=lambda i: (-support[i], -lines[i].length, i))
    assigned = np.full(n, -1)
    clusters: list[dict[str, Any]] = []
    for i in order:
        if assigned[i] >= 0:
            continue
        members = [j for j in order if assigned[j] < 0 and dup[i, j]]
        if i not in members:
            members = [i] + members
        for j in members:
            assigned[j] = len(clusters)
        gens = [lines[j].generator for j in members]
        clusters.append({"rep": i, "members": members, "generators": sorted(set(gens)),
                         "member_count": len(members)})
    return clusters


def nms_select(ends: Any, scores: Sequence[float], threshold: float, *, angle_deg: float = 5.0,
               offset_mm: float = 4.0, overlap: float = 0.6) -> list[int]:
    """Indices kept: score >= threshold, visited high to low, dropping near-duplicates of kept ones."""
    scores = np.asarray(scores, float)
    idx = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= threshold]
    if not idx:
        return []
    dup = near_duplicate_matrix(_ends(ends)[idx], angle_deg=angle_deg, offset_mm=offset_mm,
                                overlap=overlap, mutual=False)
    kept_local: list[int] = []
    for a in range(len(idx)):
        if not any(dup[a, b] for b in kept_local):
            kept_local.append(a)
    return [int(idx[a]) for a in kept_local]
