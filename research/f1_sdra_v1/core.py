"""Deterministic producer implementation for frozen SDRA-V1.

Only the raw bound scan and frozen code/configuration are accepted.  Neighbor
tables and XYZ coordinates are ephemeral and never enter the evidence payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from research.f1_ciip_v1 import partition, primary_parser
from .config import ACTIVE_SET_ORDER, CONFIG, CONFIG_SHA256, K_VALUES, canonical_json_bytes, sha_json


Q32 = float(1 << 32)
Q48 = float(1 << 48)
EPS48 = 2.0 ** -48


class SDRAStop(RuntimeError):
    def __init__(self, token: str, message: str) -> None:
        super().__init__(message)
        self.token = token


@dataclass(slots=True)
class CanonicalSource:
    integer_xyz: np.ndarray
    metric_xyz: np.ndarray
    point_ids: np.ndarray
    lattice_step: float
    origin: np.ndarray
    raw_source_sha256: str
    raw_source_bytes: int
    sampled_coordinate_sha256: str
    canonical_table_sha256: str
    morton_order: np.ndarray


@dataclass(slots=True)
class PointFeatures:
    point_ids: np.ndarray
    q_q32: np.ndarray
    x_q32: np.ndarray
    mu1_min_q32: np.ndarray
    a_q32: np.ndarray
    c_q32: np.ndarray
    b_q32: np.ndarray
    gain_q32: np.ndarray
    error_q32: np.ndarray
    phi_q32: np.ndarray
    scale_eligible: np.ndarray
    candidate: np.ndarray
    r384_integer: np.ndarray
    h0_sse_q32: np.ndarray
    h1_sse_q32: np.ndarray


@dataclass(slots=True)
class FullEvaluation:
    features: PointFeatures
    candidate_mask: np.ndarray
    supported_mask: np.ndarray
    m48: np.ndarray
    m192: np.ndarray
    d_q32: np.ndarray
    median_d_q32: int
    p90_d_q32: int
    decision: str
    stage_ledger: dict[str, dict[str, str]]
    semantic_payload: dict[str, Any]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_arrays(tag: bytes, arrays: Iterable[np.ndarray | bytes]) -> str:
    digest = hashlib.sha256(tag)
    for value in arrays:
        raw = value if isinstance(value, bytes) else np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def q32(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    safe = np.nan_to_num(array, nan=0.0, posinf=(2**31 - 1), neginf=-(2**31))
    return np.rint(safe * Q32).astype(np.int64)


def _morton6(cell_xyz: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_xyz, dtype=np.uint64)
    value = np.zeros(len(cells), dtype=np.uint64)
    for bit in range(CONFIG.morton_bits_per_axis):
        for axis in range(3):
            value |= ((cells[:, axis] >> bit) & 1) << (3 * bit + axis)
    return value


def morton_order(integer_xyz: np.ndarray) -> np.ndarray:
    cells = np.minimum(
        63,
        (np.asarray(integer_xyz, dtype=np.uint64) * 64) // ((1 << CONFIG.quantization_bits) + 1),
    ).astype(np.uint8)
    codes = _morton6(cells)
    point_ids = np.arange(len(codes), dtype=np.uint32)
    return np.lexsort((point_ids, codes)).astype(np.uint32)


def sentinel_point_ids(source: CanonicalSource) -> np.ndarray:
    ranks = np.linspace(0, len(source.integer_xyz) - 1, CONFIG.sentinel_count, dtype=np.int64)
    if len(np.unique(ranks)) != CONFIG.sentinel_count:
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "sentinel ranks are not unique")
    return np.ascontiguousarray(source.morton_order[ranks], dtype=np.uint32)


def _content_sample(xyz_raw: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(xyz_raw, dtype="<f8").view("V24").reshape(len(xyz_raw))
    _unique, first, _inverse, counts = np.unique(
        packed, return_index=True, return_inverse=True, return_counts=True
    )
    unique_xyz = np.ascontiguousarray(xyz_raw[first], dtype="<f8")
    sampled = partition.canonical_sample(
        unique_xyz, np.asarray(counts, dtype=np.uint32), sample_cap=CONFIG.sample_count
    )
    if len(sampled.xyz) != CONFIG.sample_count:
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "bounded canonical sample is not exactly 200000")
    return np.asarray(sampled.xyz, dtype=np.float64)


def _permuted_raw(xyz_raw: np.ndarray, source_sha: str) -> np.ndarray:
    raw = np.ascontiguousarray(xyz_raw, dtype="<f8").view(np.uint8).reshape(len(xyz_raw), 24)
    keys = np.empty((len(raw), 32), dtype=np.uint8)
    prefix = b"SDRA_V1_PERM" + bytes.fromhex(source_sha)
    for index in range(len(raw)):
        keys[index] = np.frombuffer(hashlib.sha256(prefix + bytes(raw[index])).digest(), dtype=np.uint8)
    key_view = np.ascontiguousarray(keys).view("V32").reshape(len(keys))
    raw_view = np.ascontiguousarray(raw).view("V24").reshape(len(raw))
    order = np.lexsort((raw_view, key_view))
    return np.ascontiguousarray(xyz_raw[order], dtype="<f8")


def canonicalize_sampled_xyz(
    sampled_xyz: np.ndarray,
    *,
    raw_source_sha256: str = "SOURCE_FREE_FIXTURE",
    raw_source_bytes: int = 0,
) -> CanonicalSource:
    xyz = np.asarray(sampled_xyz, dtype=np.float64)
    if xyz.shape != (CONFIG.sample_count, 3) and raw_source_sha256 != "SOURCE_FREE_FIXTURE":
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "source sample shape mismatch")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 385:
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "fewer than 385 usable XYZ points")
    if not np.all(np.isfinite(xyz)):
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "nonfinite coordinate")
    minimum = np.min(xyz, axis=0)
    span = np.max(xyz, axis=0) - minimum
    largest = float(np.max(span))
    if not math.isfinite(largest) or largest <= 0.0:
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "zero or nonfinite largest span")
    lattice = largest / float(1 << CONFIG.quantization_bits)
    integer = np.rint((xyz - minimum) / lattice).astype(np.int64)
    records = np.empty(len(integer), dtype=[("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    records["x"], records["y"], records["z"] = integer[:, 0], integer[:, 1], integer[:, 2]
    order = np.argsort(records, order=("x", "y", "z"), kind="stable")
    ordered = np.ascontiguousarray(integer[order], dtype="<i8")
    if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "duplicate quantized point")
    metric = minimum + lattice * ordered.astype(np.float64)
    point_ids = np.arange(len(ordered), dtype=np.uint32)
    sampled_sha = sha_arrays(b"SDRA_V1_SAMPLED_BINARY64_XYZ\0", (np.asarray(xyz, dtype="<f8"),))
    table_sha = sha_arrays(b"SDRA_V1_CANONICAL_INTEGER_XYZ\0", (ordered,))
    result = CanonicalSource(
        integer_xyz=ordered,
        metric_xyz=metric,
        point_ids=point_ids,
        lattice_step=lattice,
        origin=minimum,
        raw_source_sha256=raw_source_sha256,
        raw_source_bytes=raw_source_bytes,
        sampled_coordinate_sha256=sampled_sha,
        canonical_table_sha256=table_sha,
        morton_order=np.empty(0, dtype=np.uint32),
    )
    result.morton_order = morton_order(result.integer_xyz)
    return result


def load_bound_source(raw_source: Path, *, permute_input: bool = False) -> CanonicalSource:
    observed_sha = sha_file(raw_source)
    observed_bytes = raw_source.stat().st_size
    if observed_sha != CONFIG.raw_source_sha256 or observed_bytes != CONFIG.raw_source_bytes:
        raise SDRAStop("INVALID_SDRA_V1_GOVERNANCE_STOP", "raw source binding mismatch")
    descriptor = os.open(raw_source, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    xyz_raw = np.asarray(parsed.xyz, dtype="<f8")
    if not np.all(np.isfinite(xyz_raw)):
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "raw source contains nonfinite coordinates")
    if permute_input:
        xyz_raw = _permuted_raw(xyz_raw, observed_sha)
    sampled = _content_sample(xyz_raw)
    return canonicalize_sampled_xyz(
        sampled, raw_source_sha256=observed_sha, raw_source_bytes=observed_bytes
    )


def exact_knn_kdtree(
    source: CanonicalSource, *, anchors: np.ndarray | None = None, k: int = 384
) -> tuple[np.ndarray, np.ndarray]:
    """Exact integer-ranked KNN with a balanced kd-tree proposal and tie completion."""
    xyz_i = np.asarray(source.integer_xyz, dtype=np.int64)
    xyz_f = xyz_i.astype(np.float64)
    anchor_ids = source.point_ids if anchors is None else np.asarray(anchors, dtype=np.uint32)
    tree = cKDTree(xyz_f, compact_nodes=True, balanced_tree=True, copy_data=False)
    out_ids = np.empty((len(anchor_ids), k), dtype=np.uint32)
    out_sq = np.empty((len(anchor_ids), k), dtype=np.uint64)
    proposal_k = min(len(xyz_i), k + 9)
    for start in range(0, len(anchor_ids), CONFIG.block_size):
        stop = min(len(anchor_ids), start + CONFIG.block_size)
        block_ids = anchor_ids[start:stop]
        _dist, proposed = tree.query(xyz_f[block_ids], k=proposal_k, workers=1)
        proposed = np.asarray(proposed, dtype=np.int64)
        if proposed.ndim == 1:
            proposed = proposed[None, :]
        for local, anchor in enumerate(block_ids):
            candidate = proposed[local]
            candidate = candidate[candidate != int(anchor)]
            delta = xyz_i[candidate] - xyz_i[int(anchor)]
            sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
            ranked = np.lexsort((candidate, sq))
            candidate, sq = candidate[ranked], sq[ranked]
            if len(candidate) < k:
                raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "insufficient exact neighbors")
            boundary = int(sq[k - 1])
            overflow_tie = len(sq) > k and int(sq[k]) == boundary
            if overflow_tie:
                radius = np.nextafter(math.sqrt(boundary), math.inf)
                candidate = np.asarray(tree.query_ball_point(xyz_f[int(anchor)], radius), dtype=np.int64)
                candidate = candidate[candidate != int(anchor)]
                delta = xyz_i[candidate] - xyz_i[int(anchor)]
                sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
                ranked = np.lexsort((candidate, sq))
                candidate, sq = candidate[ranked], sq[ranked]
            out_ids[start + local] = candidate[:k].astype(np.uint32)
            out_sq[start + local] = sq[:k]
    return out_ids, out_sq


def _fit_active_sets(x: np.ndarray, y: np.ndarray, *, include_c: bool) -> tuple[np.ndarray, np.ndarray]:
    names = ("a", "c", "b")
    columns = np.stack([1.0 / x, np.ones_like(x), x], axis=2)
    allowed = [subset for subset in ACTIVE_SET_ORDER if include_c or "c" not in subset]
    n = len(x)
    best_sse = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    best_coeff = np.zeros((n, 3), dtype=np.int64)
    for subset in allowed:
        indices = [names.index(name) for name in subset]
        design = columns[:, :, indices]
        gram = np.einsum("nki,nkj->nij", design, design)
        rhs = np.einsum("nki,nk->ni", design, y)
        try:
            coeff = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coeff = np.empty_like(rhs)
            for row in range(n):
                coeff[row] = np.linalg.lstsq(design[row], y[row], rcond=None)[0]
        valid = np.all(coeff >= -(2.0 ** -40), axis=1)
        coeff = np.maximum(coeff, 0.0)
        coeff_q = q32(coeff)
        coeff_fixed = coeff_q.astype(np.float64) / Q32
        prediction = np.einsum("nki,ni->nk", design, coeff_fixed)
        sse_q = q32(np.sum((y - prediction) ** 2, axis=1))
        improve = valid & (sse_q < best_sse)
        if np.any(improve):
            best_sse[improve] = sse_q[improve]
            for local, column_index in enumerate(indices):
                best_coeff[improve, column_index] = coeff_q[improve, local]
    if np.any(best_sse == np.iinfo(np.int64).max):
        raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "no valid nonnegative model fit")
    return best_coeff, best_sse


def _producer_symmetric_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Producer-local closed-form symmetric 3x3 eigensolver."""
    output = np.empty((len(matrix), 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for row, value in enumerate(matrix):
        off_diagonal_energy = float(value[0, 1] ** 2 + value[0, 2] ** 2 + value[1, 2] ** 2)
        if off_diagonal_energy == 0.0:
            output[row] = np.sort(np.diag(value))
            continue
        center = float(np.trace(value) / 3.0)
        total_spread = float(np.sum((np.diag(value) - center) ** 2) + 2.0 * off_diagonal_energy)
        scale = math.sqrt(max(total_spread / 6.0, 0.0))
        if scale == 0.0:
            output[row] = center
            continue
        normalized = (value - center * eye) / scale
        phase = math.acos(max(-1.0, min(1.0, float(np.linalg.det(normalized) / 2.0)))) / 3.0
        upper = center + 2.0 * scale * math.cos(phase)
        lower = center + 2.0 * scale * math.cos(phase + 2.0 * math.pi / 3.0)
        output[row] = (lower, 3.0 * center - lower - upper, upper)
    return output


def _features_from_neighbors(
    source: CanonicalSource, anchor_ids: np.ndarray, neighbors: np.ndarray, neighbor_sq: np.ndarray
) -> PointFeatures:
    anchor_ids = np.asarray(anchor_ids, dtype=np.uint32)
    n = len(anchor_ids)
    q_values = np.zeros((n, len(K_VALUES)), dtype=np.float64)
    x_values = np.zeros_like(q_values)
    mu1_min = np.full(n, np.inf, dtype=np.float64)
    r384_integer = np.zeros(n, dtype=np.float64)
    xyz = np.asarray(source.integer_xyz, dtype=np.int64)
    for start in range(0, n, CONFIG.block_size):
        stop = min(n, start + CONFIG.block_size)
        ids = anchor_ids[start:stop]
        block_neighbors = neighbors[start:stop]
        block_sq = neighbor_sq[start:stop].astype(np.float64)
        base_r2 = np.mean(block_sq[:, :24], axis=1)
        if np.any(base_r2 <= 0):
            raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "zero base neighborhood radius")
        for scale_index, k in enumerate(K_VALUES):
            selected = block_neighbors[:, :k]
            delta = xyz[selected].astype(np.float64) - xyz[ids, None, :].astype(np.float64)
            norm = np.sqrt(block_sq[:, :k])
            if np.any(norm <= 0):
                raise SDRAStop("SDRA_V1_SOURCE_CONTRACT_STOP", "zero neighbor displacement")
            direction = delta / norm[..., None]
            moment = np.einsum("nki,nkj->nij", direction, direction) / float(k)
            moment = np.rint(moment * Q48) / Q48
            eigen = _producer_symmetric_eigenvalues(moment)
            eigen_q = q32(eigen).astype(np.float64) / Q32
            q_values[start:stop, scale_index] = eigen_q[:, 0]
            mu1_min[start:stop] = np.minimum(mu1_min[start:stop], eigen_q[:, 1])
            r2 = np.mean(block_sq[:, :k], axis=1)
            x_values[start:stop, scale_index] = q32(r2 / base_r2).astype(np.float64) / Q32
            if k == 384:
                r384_integer[start:stop] = np.sqrt(r2)
    q_q = q32(q_values)
    x_q = q32(x_values)
    ratios_q = q32(x_values[:, 1:] / x_values[:, :-1])
    eligible = (
        np.all(ratios_q >= CONFIG.successive_x_ratio_low_q32, axis=1)
        & np.all(ratios_q <= CONFIG.successive_x_ratio_high_q32, axis=1)
        & (x_q[:, -1] >= CONFIG.x384_low_q32)
        & (x_q[:, -1] <= CONFIG.x384_high_q32)
        & (q32(mu1_min) >= CONFIG.mu1_minimum_q32)
    )
    a_q = np.zeros(n, dtype=np.int64)
    c_q = np.zeros(n, dtype=np.int64)
    b_q = np.zeros(n, dtype=np.int64)
    gain_q = np.zeros(n, dtype=np.int64)
    error_q = np.zeros(n, dtype=np.int64)
    phi_q = np.zeros(n, dtype=np.int64)
    h0_sse_q = np.zeros(n, dtype=np.int64)
    h1_sse_q = np.zeros(n, dtype=np.int64)
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows):
        x = x_q[eligible_rows].astype(np.float64) / Q32
        y = q_q[eligible_rows].astype(np.float64) / Q32
        h0_coeff, h0_sse = _fit_active_sets(x, y, include_c=False)
        h1_coeff, h1_sse = _fit_active_sets(x, y, include_c=True)
        denominator = np.maximum(h0_sse.astype(np.float64) / Q32, EPS48)
        gain = np.maximum(0.0, (h0_sse - h1_sse).astype(np.float64) / Q32 / denominator)
        energy = np.maximum(np.sum(y * y, axis=1), EPS48)
        error = (h1_sse.astype(np.float64) / Q32) / energy
        a = h1_coeff[:, 0].astype(np.float64) / Q32
        c = h1_coeff[:, 1].astype(np.float64) / Q32
        b = h1_coeff[:, 2].astype(np.float64) / Q32
        x_star = x[:, -1]
        phi = c / np.maximum(a / x_star + c + b * x_star + 2.0 ** -24, EPS48)
        a_q[eligible_rows], c_q[eligible_rows], b_q[eligible_rows] = h1_coeff.T
        gain_q[eligible_rows], error_q[eligible_rows], phi_q[eligible_rows] = q32(gain), q32(error), q32(phi)
        h0_sse_q[eligible_rows], h1_sse_q[eligible_rows] = h0_sse, h1_sse
    candidate = (
        eligible
        & (c_q >= CONFIG.c_low_q32)
        & (c_q <= CONFIG.c_high_q32)
        & (gain_q >= CONFIG.gain_minimum_q32)
        & (error_q <= CONFIG.error_maximum_q32)
        & (phi_q >= CONFIG.phi_minimum_q32)
    )
    return PointFeatures(
        point_ids=anchor_ids,
        q_q32=q_q,
        x_q32=x_q,
        mu1_min_q32=q32(mu1_min),
        a_q32=a_q,
        c_q32=c_q,
        b_q32=b_q,
        gain_q32=gain_q,
        error_q32=error_q,
        phi_q32=phi_q,
        scale_eligible=eligible,
        candidate=candidate,
        r384_integer=r384_integer,
        h0_sse_q32=h0_sse_q,
        h1_sse_q32=h1_sse_q,
    )


def evaluate_point_features(
    source: CanonicalSource, *, anchors: np.ndarray | None = None
) -> tuple[PointFeatures, np.ndarray, np.ndarray]:
    anchor_ids = source.point_ids if anchors is None else np.asarray(anchors, dtype=np.uint32)
    neighbors, squared = exact_knn_kdtree(source, anchors=anchor_ids, k=384)
    return _features_from_neighbors(source, anchor_ids, neighbors, squared), neighbors, squared


def _nearest_rank(values: np.ndarray, probability: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.int64), kind="stable")
    if len(ordered) == 0:
        return 0
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return int(ordered[index])


def _stage(execution: str, entity: str, upstream: str = "NONE") -> dict[str, str]:
    return {"execution_state": execution, "entity_state": entity, "upstream_stop": upstream}


def _semantic_hash(features: PointFeatures, supported: np.ndarray, m48: np.ndarray, m192: np.ndarray, d_q: np.ndarray) -> str:
    return sha_arrays(
        b"SDRA_V1_PER_POINT_SEMANTICS\0",
        (
            features.point_ids.astype("<u4"), features.q_q32.astype("<i8"), features.x_q32.astype("<i8"),
            features.mu1_min_q32.astype("<i8"), features.a_q32.astype("<i8"), features.c_q32.astype("<i8"),
            features.b_q32.astype("<i8"), features.gain_q32.astype("<i8"), features.error_q32.astype("<i8"),
            features.phi_q32.astype("<i8"), features.scale_eligible.astype(np.uint8),
            features.candidate.astype(np.uint8), supported.astype(np.uint8), m48.astype("<u2"),
            m192.astype("<u2"), d_q.astype("<i8"),
        ),
    )


def evaluate_full(source: CanonicalSource) -> FullEvaluation:
    sentinel_ids = sentinel_point_ids(source)
    sentinel, _sentinel_neighbors, _sentinel_sq = evaluate_point_features(source, anchors=sentinel_ids)
    sentinel_eligible = int(np.count_nonzero(sentinel.scale_eligible))
    sentinel_candidates = int(np.count_nonzero(sentinel.candidate))
    if sentinel_eligible < CONFIG.sentinel_scale_eligible_minimum:
        decision = "SDRA_V1_SAMPLING_REGIME_ARCHITECTURE_STOP"
    elif sentinel_candidates > CONFIG.sentinel_candidate_maximum:
        decision = "SDRA_V1_NONSELECTIVE_ARCHITECTURE_STOP"
    else:
        decision = "CONTINUE_FULL_SOURCE"
    if decision != "CONTINUE_FULL_SOURCE":
        empty_bool = np.zeros(len(sentinel_ids), dtype=bool)
        empty_u16 = np.zeros(len(sentinel_ids), dtype=np.uint16)
        empty_i64 = np.zeros(len(sentinel_ids), dtype=np.int64)
        ledger = {
            "source": _stage("OBSERVED_COMPLETE", "PRESENT"),
            "sentinel": _stage("OBSERVED_COMPLETE", "PRESENT"),
            "full_features": _stage("NOT_EVALUABLE", "NOT_APPLICABLE", decision),
            "support": _stage("NOT_EVALUABLE", "NOT_APPLICABLE", decision),
        }
        semantic = {
            "schema_version": 1, "attempt_id": CONFIG.attempt_id, "configuration_sha256": CONFIG_SHA256,
            "source_sha256": source.raw_source_sha256, "source_bytes": source.raw_source_bytes,
            "sampled_coordinate_sha256": source.sampled_coordinate_sha256,
            "canonical_table_sha256": source.canonical_table_sha256,
            "point_count": len(source.integer_xyz), "processed_point_count": len(sentinel_ids),
            "sentinel_scale_eligible_count": sentinel_eligible,
            "sentinel_candidate_count": sentinel_candidates, "full_evaluated": False,
            "candidate_count": 0, "supported_candidate_count": 0,
            "median_d_q32": 0, "p90_d_q32": 0, "decision": decision,
            "stage_ledger": ledger,
        }
        semantic["stage_ledger_sha256"] = sha_json(ledger)
        semantic["per_point_semantic_sha256"] = _semantic_hash(sentinel, empty_bool, empty_u16, empty_u16, empty_i64)
        semantic["semantic_sha256"] = sha_json(semantic)
        return FullEvaluation(sentinel, sentinel.candidate, empty_bool, empty_u16, empty_u16, empty_i64, 0, 0, decision, ledger, semantic)

    features, neighbors, _neighbor_sq = evaluate_point_features(source)
    candidate_mask = features.candidate.copy()
    candidate_neighbors48 = candidate_mask[neighbors[:, :48]]
    candidate_neighbors192 = candidate_mask[neighbors[:, :192]]
    m48 = np.sum(candidate_neighbors48, axis=1, dtype=np.uint16)
    m192 = np.sum(candidate_neighbors192, axis=1, dtype=np.uint16)
    supported = candidate_mask & (m48 >= CONFIG.supported_m48_minimum)
    d = np.zeros(len(candidate_mask), dtype=np.float64)
    support_ids = np.flatnonzero(supported)
    if len(support_ids):
        d[support_ids] = np.log2((m192[support_ids].astype(np.float64) + 1.0) / (m48[support_ids].astype(np.float64) + 1.0))
    d_q = q32(d)
    median_d = _nearest_rank(d_q[support_ids], 0.5) if len(support_ids) else 0
    p90_d = _nearest_rank(d_q[support_ids], 0.9) if len(support_ids) else 0
    eligible_count = int(np.count_nonzero(features.scale_eligible))
    candidate_count = int(np.count_nonzero(candidate_mask))
    supported_count = int(np.count_nonzero(supported))
    if eligible_count < CONFIG.full_scale_eligible_minimum:
        decision = "SDRA_V1_SAMPLING_REGIME_ARCHITECTURE_STOP"
    elif candidate_count > CONFIG.candidate_maximum:
        decision = "SDRA_V1_NONSELECTIVE_ARCHITECTURE_STOP"
    elif candidate_count == 0:
        decision = "SDRA_V1_NO_SIGNAL_ARCHITECTURE_STOP"
    elif supported_count * CONFIG.supported_fraction_denominator < candidate_count * CONFIG.supported_fraction_numerator:
        decision = "SDRA_V1_INCOHERENT_SUPPORT_ARCHITECTURE_STOP"
    elif median_d > CONFIG.support_median_d_maximum_q32 or p90_d > CONFIG.support_p90_d_maximum_q32:
        decision = "SDRA_V1_AREA_LIKE_SUPPORT_ARCHITECTURE_STOP"
    else:
        decision = "SDRA_V1_SOURCE_REPRESENTATION_PREFLIGHT_SURVIVES"
    ledger = {
        "source": _stage("OBSERVED_COMPLETE", "PRESENT"),
        "sentinel": _stage("OBSERVED_COMPLETE", "PRESENT"),
        "full_features": _stage("OBSERVED_COMPLETE", "PRESENT"),
        "candidate_mask": _stage("OBSERVED_COMPLETE", "PRESENT" if candidate_count else "NOT_PRESENT"),
        "support": _stage("OBSERVED_COMPLETE", "PRESENT" if supported_count else "NOT_PRESENT"),
    }
    candidate_ids = np.flatnonzero(candidate_mask)
    candidate_rows = [
        {
            "point_id": int(index), "c_q32": int(features.c_q32[index]),
            "gain_q32": int(features.gain_q32[index]), "error_q32": int(features.error_q32[index]),
            "phi_q32": int(features.phi_q32[index]), "m48": int(m48[index]),
            "m192": int(m192[index]), "supported": bool(supported[index]), "d_q32": int(d_q[index]),
        }
        for index in candidate_ids
    ]
    semantic = {
        "schema_version": 1, "attempt_id": CONFIG.attempt_id, "configuration_sha256": CONFIG_SHA256,
        "source_sha256": source.raw_source_sha256, "source_bytes": source.raw_source_bytes,
        "sampled_coordinate_sha256": source.sampled_coordinate_sha256,
        "canonical_table_sha256": source.canonical_table_sha256,
        "point_count": len(source.integer_xyz), "processed_point_count": len(source.integer_xyz),
        "sentinel_scale_eligible_count": sentinel_eligible, "sentinel_candidate_count": sentinel_candidates,
        "full_evaluated": True, "scale_eligible_count": eligible_count,
        "candidate_count": candidate_count, "supported_candidate_count": supported_count,
        "median_d_q32": median_d, "p90_d_q32": p90_d,
        "candidate_rows": candidate_rows, "decision": decision, "stage_ledger": ledger,
    }
    semantic["stage_ledger_sha256"] = sha_json(ledger)
    semantic["per_point_semantic_sha256"] = _semantic_hash(features, supported, m48, m192, d_q)
    semantic["semantic_sha256"] = sha_json(semantic)
    payload_bytes = len(canonical_json_bytes(semantic))
    if payload_bytes > CONFIG.semantic_payload_ceiling_bytes:
        raise SDRAStop("SDRA_V1_RESOURCE_STOP", "semantic payload ceiling exceeded")
    return FullEvaluation(features, candidate_mask, supported, m48, m192, d_q, median_d, p90_d, decision, ledger, semantic)


def source_free_evaluate(xyz: np.ndarray) -> FullEvaluation:
    return evaluate_full(canonicalize_sampled_xyz(xyz))


def evaluate_fixture_geometry(xyz: np.ndarray) -> dict[str, Any]:
    """Run the frozen point and support observables without C006 count walls."""
    source = canonicalize_sampled_xyz(xyz)
    features, neighbors, _squared = evaluate_point_features(source)
    candidate = features.candidate
    m48 = np.sum(candidate[neighbors[:, :48]], axis=1, dtype=np.uint16)
    m192 = np.sum(candidate[neighbors[:, :192]], axis=1, dtype=np.uint16)
    supported = candidate & (m48 >= CONFIG.supported_m48_minimum)
    support_ids = np.flatnonzero(supported)
    d_q = np.zeros(len(candidate), dtype=np.int64)
    if len(support_ids):
        d_q[support_ids] = q32(
            np.log2((m192[support_ids].astype(float) + 1) / (m48[support_ids].astype(float) + 1))
        )
    candidate_ids = np.flatnonzero(candidate)
    return {
        "point_count": len(source.integer_xyz),
        "scale_eligible_count": int(np.count_nonzero(features.scale_eligible)),
        "candidate_count": int(len(candidate_ids)),
        "candidate_fraction": float(len(candidate_ids) / len(candidate)),
        "supported_candidate_count": int(len(support_ids)),
        "median_d_q32": _nearest_rank(d_q[support_ids], 0.5) if len(support_ids) else 0,
        "p90_d_q32": _nearest_rank(d_q[support_ids], 0.9) if len(support_ids) else 0,
        "candidate_integer_xyz": source.integer_xyz[candidate_ids].copy(),
        "candidate_semantic_sha256": sha_arrays(
            b"SDRA_V1_FIXTURE_CANDIDATES\0",
            (
                source.integer_xyz[candidate_ids].astype("<i8"),
                features.c_q32[candidate_ids].astype("<i8"),
                features.gain_q32[candidate_ids].astype("<i8"),
                features.error_q32[candidate_ids].astype("<i8"),
                features.phi_q32[candidate_ids].astype("<i8"),
                supported[candidate_ids].astype(np.uint8),
                d_q[candidate_ids].astype("<i8"),
            ),
        ),
        "known_geometric_confuser_present": False,
    }


def visual_scratch_arrays(evaluation: FullEvaluation) -> dict[str, np.ndarray]:
    f = evaluation.features
    return {
        "point_ids": f.point_ids.astype(np.uint32), "q_q32": f.q_q32.astype(np.int64),
        "x_q32": f.x_q32.astype(np.int64), "c_q32": f.c_q32.astype(np.int64),
        "gain_q32": f.gain_q32.astype(np.int64), "error_q32": f.error_q32.astype(np.int64),
        "phi_q32": f.phi_q32.astype(np.int64), "candidate": f.candidate.astype(np.uint8),
        "supported": evaluation.supported_mask.astype(np.uint8), "d_q32": evaluation.d_q32.astype(np.int64),
        "r384_integer": f.r384_integer.astype(np.float64),
    }


def normalized_semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("role", "wall_seconds", "peak_rss_bytes"):
        result.pop(key, None)
    return result
