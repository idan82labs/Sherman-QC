"""Deterministic, point-native PWTS representation semantic core.

This module is deliberately isolated from the production F1 implementation.  It
consumes only finite XYZ coordinates and emits compact point-ID/statistic tables;
raw coordinates and neighborhoods never enter serialized evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.pwts_v1.config import (  # type: ignore
        CONFIG_SHA256, EXPECTED_CONFIG_SHA256, canonical_json_bytes,
        get_config_copy, require_authorized_config_hash,
    )
    from research.pwts_v1.schema import (  # type: ignore
        CERTIFICATE_REJECTION_ORDER, SCHEMA_SHA256, StructuralCapError,
        TRACE_REJECTION_ORDER, assert_coordinate_free, canonical_row_key,
    )
else:
    from .config import (
        CONFIG_SHA256, EXPECTED_CONFIG_SHA256, canonical_json_bytes,
        get_config_copy, require_authorized_config_hash,
    )
    from .schema import (
        CERTIFICATE_REJECTION_ORDER, SCHEMA_SHA256, StructuralCapError,
        TRACE_REJECTION_ORDER, assert_coordinate_free, canonical_row_key,
    )


POINT_DOMAIN = b"PWTS_POINT_V1\0"
SAMPLE_DOMAIN = b"PWTS_SAMPLE_V1\0"
PERM_DOMAIN = b"PWTS_PERM_V1\0"
TRACE_DOMAIN = b"PWTS_TRACE_V1\0"
CERT_DOMAIN = b"PWTS_CERT_V1\0"
COORDINATE_MULTISET_DOMAIN = b"PWTS_COORDINATE_MULTISET_V1\0"


@dataclass(frozen=True)
class CanonicalPoints:
    points: np.ndarray = field(repr=False)
    point_id_bytes: tuple[bytes, ...]
    point_ids: tuple[str, ...]
    parsed_count: int
    finite_count: int
    unique_count: int
    exact_duplicate_count: int
    sample_count: int
    coordinate_multiset_digest: str
    sampled_point_id_digest: str


@dataclass
class FieldState:
    knn: np.ndarray = field(repr=False)
    knn_dist: np.ndarray = field(repr=False)
    normals: np.ndarray = field(repr=False)
    variation: np.ndarray = field(repr=False)
    normal_valid: np.ndarray = field(repr=False)
    d1: np.ndarray = field(repr=False)
    d2: np.ndarray = field(repr=False)
    response: np.ndarray = field(repr=False)
    balance: np.ndarray = field(repr=False)
    seed_eligible: np.ndarray = field(repr=False)
    trace_eligible: np.ndarray = field(repr=False)
    tangent: np.ndarray = field(repr=False)
    tangent_valid: np.ndarray = field(repr=False)
    tangent_support: np.ndarray = field(repr=False)
    tangent_anisotropy: np.ndarray = field(repr=False)
    h: np.ndarray = field(repr=False)
    local_maxima: np.ndarray = field(repr=False)


@dataclass
class TraceGeometry:
    trace_id: str
    seed_index: int
    node_indices: list[int]
    h_t: float
    valid: bool
    first_rejection_code: str
    segment_starts: np.ndarray = field(repr=False)
    segment_vectors: np.ndarray = field(repr=False)
    segment_lengths: np.ndarray = field(repr=False)
    segment_g: np.ndarray = field(repr=False)
    cumulative_lengths: np.ndarray = field(repr=False)


@dataclass
class RepresentationResult:
    case_token: str
    tables: dict[str, list[dict[str, Any]]]

    def compact_payload(self, process_role: str = "SEMANTIC_CORE") -> dict[str, Any]:
        payload = {
            "preflight_id": "PWTS_REPRESENTATION_PREFLIGHT_V1",
            "case_token": self.case_token,
            "process_role": process_role,
            "source_points_serialized": False,
            "tables": self.tables,
        }
        assert_coordinate_free(payload)
        return payload


def _array_from_records(records: Any) -> np.ndarray:
    if isinstance(records, np.ndarray):
        if records.dtype.names:
            names = set(records.dtype.names)
            if not {"x", "y", "z"}.issubset(names):
                raise ValueError("structured records must contain x, y, z")
            arr = np.column_stack((records["x"], records["y"], records["z"]))
        else:
            arr = np.asarray(records)
    else:
        rows = []
        for record in records:
            if isinstance(record, Mapping):
                rows.append((record["x"], record["y"], record["z"]))
            else:
                rows.append((record[0], record[1], record[2]))
        arr = np.asarray(rows)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    return np.asarray(arr, dtype=np.float64)


def _packed_xyz(row: np.ndarray) -> bytes:
    return struct.pack("<ddd", float(row[0]), float(row[1]), float(row[2]))


def _digest_ids(ids: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for point_id in ids:
        digest.update(point_id)
    return digest.hexdigest()


def canonicalize_points(records: Any, maximum_points: int = 200_000) -> CanonicalPoints:
    """Canonicalize, exactly deduplicate and deterministically sample XYZ records."""
    arr = _array_from_records(records)
    parsed_count = int(len(arr))
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = np.array(arr[finite_mask], dtype=np.float64, order="C", copy=True)
    finite_count = int(len(arr))
    arr[arr == 0.0] = 0.0  # normalize IEEE negative zero

    all_finite_packed = sorted(_packed_xyz(row) for row in arr)
    unique: dict[bytes, np.ndarray] = {}
    for row in arr:
        packed = _packed_xyz(row)
        unique.setdefault(packed, row.copy())
    packed_all = sorted(unique)
    multiset_digest = hashlib.sha256(COORDINATE_MULTISET_DOMAIN + b"".join(all_finite_packed)).hexdigest()

    point_entries: list[tuple[bytes, bytes, np.ndarray]] = []
    seen_ids: dict[bytes, bytes] = {}
    for packed in packed_all:
        point_id = hashlib.sha256(POINT_DOMAIN + packed).digest()
        previous = seen_ids.setdefault(point_id, packed)
        if previous != packed:
            raise RuntimeError("SHA-256 collision between distinct canonical coordinates")
        point_entries.append((point_id, packed, unique[packed]))

    if len(point_entries) > maximum_points:
        ranked = sorted(
            point_entries,
            key=lambda entry: (hashlib.sha256(SAMPLE_DOMAIN + entry[1]).digest(), entry[1]),
        )[:maximum_points]
    else:
        ranked = point_entries
    selected = sorted(ranked, key=lambda entry: entry[0])

    points = np.asarray([entry[2] for entry in selected], dtype=np.float64)
    point_id_bytes = tuple(entry[0] for entry in selected)
    return CanonicalPoints(
        points=points,
        point_id_bytes=point_id_bytes,
        point_ids=tuple(value.hex() for value in point_id_bytes),
        parsed_count=parsed_count,
        finite_count=finite_count,
        unique_count=len(point_entries),
        exact_duplicate_count=finite_count - len(point_entries),
        sample_count=len(selected),
        coordinate_multiset_digest=multiset_digest,
        sampled_point_id_digest=_digest_ids(point_id_bytes),
    )


def deterministic_record_permutation(records: Any) -> np.ndarray:
    arr = _array_from_records(records)
    keys = []
    for index, row in enumerate(arr):
        canonical = np.array(row, dtype=np.float64, copy=True)
        canonical[canonical == 0.0] = 0.0
        packed = _packed_xyz(canonical)
        key = hashlib.sha256(PERM_DOMAIN + struct.pack("<Q", index) + packed).digest()
        keys.append((key, index))
    order = [index for _, index in sorted(keys)]
    return arr[order]


def exact_knn(points: np.ndarray, point_ids: Sequence[bytes], k: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean KNN excluding self, with point-ID distance tie order."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n <= k:
        raise ValueError(f"KNN requires more than {k} unique points")
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float64)

    # Small inputs use an explicit distance matrix, providing a simple exact oracle.
    if n <= 2048:
        for i in range(n):
            delta = points - points[i]
            d2 = np.einsum("ij,ij->i", delta, delta)
            candidates = [j for j in range(n) if j != i]
            candidates.sort(key=lambda j: (float(d2[j]), point_ids[j]))
            chosen = candidates[:k]
            indices[i] = chosen
            distances[i] = np.sqrt(d2[chosen])
        return indices, distances

    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    query_k = min(n, k + 2)  # self + k, plus one to detect a boundary tie
    for start in range(0, n, 4096):
        stop = min(n, start + 4096)
        raw_d, raw_i = tree.query(points[start:stop], k=query_k, eps=0.0, workers=1)
        for local, i in enumerate(range(start, stop)):
            candidates = {int(j) for j in np.atleast_1d(raw_i[local]) if int(j) != i}
            prelim = sorted(
                candidates,
                key=lambda j: (float(np.dot(points[j] - points[i], points[j] - points[i])), point_ids[j]),
            )
            if len(prelim) < k:
                raise RuntimeError("exact KNN query returned too few neighbors")
            kth_d2 = float(np.dot(points[prelim[k - 1]] - points[i], points[prelim[k - 1]] - points[i]))
            if len(prelim) > k:
                next_d2 = float(np.dot(points[prelim[k]] - points[i], points[prelim[k]] - points[i]))
            else:
                next_d2 = math.inf
            if next_d2 == kth_d2:
                radius = np.nextafter(math.sqrt(kth_d2), math.inf)
                candidates = set(int(j) for j in tree.query_ball_point(points[i], radius, eps=0.0))
                candidates.discard(i)
                prelim = sorted(
                    candidates,
                    key=lambda j: (float(np.dot(points[j] - points[i], points[j] - points[i])), point_ids[j]),
                )
            chosen = prelim[:k]
            indices[i] = chosen
            distances[i] = [math.sqrt(float(np.dot(points[j] - points[i], points[j] - points[i]))) for j in chosen]
    return indices, distances


def _canonical_vector_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    result[result == 0.0] = 0.0
    return result


def _lower_quantile(values: Sequence[float] | np.ndarray, q: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) == 0:
        return 0.0
    return float(ordered[int(math.floor((len(ordered) - 1) * q))])


def _angle_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.acos(float(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))))


def compute_field(canonical: CanonicalPoints) -> FieldState:
    cfg = get_config_copy()
    points = canonical.points
    knn, knn_dist = exact_knn(points, canonical.point_id_bytes, 64)
    n = len(points)
    normals = np.zeros((n, 3, 3), dtype=np.float64)
    variation = np.full((n, 3), math.inf, dtype=np.float64)
    valid = np.zeros((n, 3), dtype=bool)
    scales = (16, 32, 64)
    for i in range(n):
        for scale_index, k in enumerate(scales):
            member_indices = np.concatenate(([i], knn[i, :k]))
            members = points[member_indices]
            centered = members - np.mean(members, axis=0)
            covariance = centered.T @ centered / float(k + 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            total = float(np.sum(eigenvalues))
            if eigenvalues[2] > 0.0 and eigenvalues[1] / eigenvalues[2] >= 0.02 and total > 0.0:
                sv = float(eigenvalues[0] / total)
                if sv <= 0.25:
                    normals[i, scale_index] = _canonical_vector_sign(eigenvectors[:, 0])
                    variation[i, scale_index] = sv
                    valid[i, scale_index] = True

    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)
    response = np.zeros(n, dtype=np.float64)
    balance = np.zeros(n, dtype=np.float64)
    all_valid = np.all(valid, axis=1)
    for i in np.flatnonzero(all_valid):
        d1[i] = _angle_abs(normals[i, 0], normals[i, 1])
        d2[i] = _angle_abs(normals[i, 1], normals[i, 2])
        response[i] = math.sqrt(d1[i] * d2[i])
        larger = max(d1[i], d2[i])
        balance[i] = min(d1[i], d2[i]) / larger if larger > 0.0 else 0.0

    transition = cfg["transition_field"]
    seed_eligible = all_valid & (d1 >= transition["seed_min_each_scale_angle_rad"]) & (
        d2 >= transition["seed_min_each_scale_angle_rad"]
    ) & (response >= transition["seed_geomean_angle_rad"]) & (balance >= transition["seed_scale_balance_min"])
    trace_eligible = all_valid & (d1 >= transition["trace_min_each_scale_angle_rad"]) & (
        d2 >= transition["trace_min_each_scale_angle_rad"]
    ) & (response >= transition["trace_geomean_angle_rad"]) & (balance >= transition["trace_scale_balance_min"])

    tangent = np.zeros((n, 3), dtype=np.float64)
    tangent_valid = np.zeros(n, dtype=bool)
    tangent_support = np.zeros(n, dtype=np.int64)
    tangent_anisotropy = np.zeros(n, dtype=np.float64)
    for i in np.flatnonzero(trace_eligible):
        neighbors = knn[i, :64]
        members = neighbors[trace_eligible[neighbors]]
        weights = response[members] ** 2
        positive = weights > 0.0
        members, weights = members[positive], weights[positive]
        tangent_support[i] = len(members)
        if len(members) < 8:
            continue
        offsets = points[members] - points[i]
        normal = normals[i, 2]
        projected = offsets - np.outer(offsets @ normal, normal)
        mean = np.average(projected, axis=0, weights=weights)
        centered = projected - mean
        covariance = (centered * weights[:, None]).T @ centered / float(np.sum(weights))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        denominator = float(eigenvalues[1] + eigenvalues[2])
        anisotropy = float(eigenvalues[2] / denominator) if denominator > 0.0 else 0.0
        tangent_anisotropy[i] = anisotropy
        if anisotropy >= 0.65:
            tangent[i] = _canonical_vector_sign(eigenvectors[:, 2])
            tangent_valid[i] = True

    local_maxima = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(seed_eligible):
        is_maximum = True
        for j in knn[i, :16]:
            if response[j] > response[i] or (
                response[j] == response[i] and canonical.point_id_bytes[j] < canonical.point_id_bytes[i]
            ):
                is_maximum = False
                break
        local_maxima[i] = is_maximum
    return FieldState(
        knn, knn_dist, normals, variation, valid, d1, d2, response, balance,
        seed_eligible, trace_eligible, tangent, tangent_valid, tangent_support,
        tangent_anisotropy, knn_dist[:, 31].copy(), local_maxima,
    )


def _trace_id(case_token: str, seed_id: str) -> str:
    return hashlib.sha256(
        TRACE_DOMAIN + case_token.encode("utf-8") + bytes.fromhex(seed_id) + bytes.fromhex(CONFIG_SHA256)
    ).hexdigest()


def _admissible_extensions(
    current: int,
    directed_tangent: np.ndarray,
    used: set[int],
    canonical: CanonicalPoints,
    field_state: FieldState,
) -> list[tuple[tuple[Any, ...], int, np.ndarray]]:
    points = canonical.points
    rows = []
    for candidate in field_state.knn[current, :32]:
        candidate = int(candidate)
        if candidate in used or not field_state.trace_eligible[candidate] or not field_state.tangent_valid[candidate]:
            continue
        step = points[candidate] - points[current]
        length = float(np.linalg.norm(step))
        if length == 0.0 or length > 1.25 * field_state.h[current]:
            continue
        unit_step = step / length
        forward = float(np.dot(unit_step, directed_tangent))
        if forward < 0.7071067811865476:
            continue
        candidate_tangent = field_state.tangent[candidate].copy()
        if float(np.dot(candidate_tangent, directed_tangent)) < 0.0:
            candidate_tangent *= -1.0
        alignment = float(np.dot(candidate_tangent, directed_tangent))
        if alignment < 0.8191520442889918:
            continue
        turn = float(math.acos(float(np.clip(alignment, -1.0, 1.0))))
        lateral = float(np.linalg.norm(unit_step - forward * directed_tangent))
        score = (
            turn,
            lateral,
            -float(field_state.response[candidate]),
            length / field_state.h[current],
            canonical.point_id_bytes[candidate],
        )
        rows.append((score, candidate, candidate_tangent))
    rows.sort(key=lambda item: item[0])
    return rows


def _extend_arms_symmetric(
    seed: int,
    canonical: CanonicalPoints,
    field_state: FieldState,
) -> tuple[list[int], list[int], bool]:
    """Grow both arms under one symmetric, deterministic 257-node budget."""
    arms = {
        "negative": {
            "nodes": [], "used": {seed}, "current": seed,
            "directed": -field_state.tangent[seed].copy(), "exhausted": False,
        },
        "positive": {
            "nodes": [], "used": {seed}, "current": seed,
            "directed": field_state.tangent[seed].copy(), "exhausted": False,
        },
    }
    total_nodes = 1
    while total_nodes < 257 and not all(bool(arm["exhausted"]) for arm in arms.values()):
        progressed = False
        # The locked symmetric convention gives the negative arm the first slot
        # in each round, then the positive arm; exhausted arms are skipped.
        for name in ("negative", "positive"):
            arm = arms[name]
            if arm["exhausted"] or total_nodes >= 257:
                continue
            choices = _admissible_extensions(
                int(arm["current"]), np.asarray(arm["directed"]), arm["used"],
                canonical, field_state,
            )
            if not choices:
                arm["exhausted"] = True
                continue
            _, candidate, directed = choices[0]
            arm["nodes"].append(candidate)
            arm["used"].add(candidate)
            arm["current"] = candidate
            arm["directed"] = directed
            total_nodes += 1
            progressed = True
        if not progressed and all(bool(arm["exhausted"]) for arm in arms.values()):
            break
    cap_with_extension = False
    if total_nodes == 257:
        for arm in arms.values():
            if arm["exhausted"]:
                continue
            if _admissible_extensions(
                int(arm["current"]), np.asarray(arm["directed"]), arm["used"],
                canonical, field_state,
            ):
                cap_with_extension = True
                break
    return list(arms["negative"]["nodes"]), list(arms["positive"]["nodes"]), cap_with_extension


def _empty_geometry(trace_id: str, seed: int, nodes: list[int], h_t: float, valid: bool, code: str) -> TraceGeometry:
    return TraceGeometry(
        trace_id, seed, nodes, h_t, valid, code,
        np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0),
    )


def build_trace(case_token: str, seed: int, canonical: CanonicalPoints, state: FieldState) -> tuple[TraceGeometry, dict[str, Any]]:
    seed_id = canonical.point_ids[seed]
    trace_id = _trace_id(case_token, seed_id)
    if not state.tangent_valid[seed]:
        code = "TANGENT_SUPPORT_INSUFFICIENT" if state.tangent_support[seed] < 8 else "TANGENT_ANISOTROPY_INSUFFICIENT"
        geometry = _empty_geometry(trace_id, seed, [seed], state.h[seed], False, code)
        return geometry, _trace_row(case_token, geometry, canonical, state, 0.0, 0.0)

    negative, positive, cap_with_extension = _extend_arms_symmetric(seed, canonical, state)
    collision = bool(set(positive) & set(negative))
    nodes = list(reversed(negative)) + [seed] + positive
    h_t = _lower_quantile(state.h[nodes], 0.5)
    points = canonical.points[nodes]
    steps = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1) if len(nodes) > 1 else np.empty(0)
    arc_length = float(np.sum(step_lengths))
    normalized_arc = arc_length / h_t if h_t > 0.0 else 0.0
    maximum_step = float(np.max(step_lengths) / h_t) if len(step_lengths) and h_t > 0.0 else 0.0

    code = "VALID"
    if cap_with_extension:
        code = "TRACE_NODE_CAP_EXCEEDED"
    elif len(nodes) < 9:
        code = "TRACE_NODE_COUNT_INSUFFICIENT"
    elif normalized_arc < 6.0:
        code = "TRACE_LENGTH_INSUFFICIENT"
    elif maximum_step > 1.5:
        code = "TRACE_DISCONTINUITY"
    elif collision or len(set(nodes)) != len(nodes):
        code = "TRACE_ARM_COLLISION"

    valid = code == "VALID"
    if not valid:
        geometry = _empty_geometry(trace_id, seed, nodes, h_t, False, code)
        return geometry, _trace_row(case_token, geometry, canonical, state, normalized_arc, maximum_step)

    segment_vectors = steps.copy()
    segment_lengths = step_lengths.copy()
    segment_tangents = segment_vectors / segment_lengths[:, None]
    segment_g = np.zeros_like(segment_tangents)
    frame_valid = True
    for ordinal, tangent in enumerate(segment_tangents):
        n0 = state.normals[nodes[ordinal], 2].copy()
        n1 = state.normals[nodes[ordinal + 1], 2].copy()
        if float(np.dot(n0, n1)) < 0.0:
            n1 *= -1.0
        summed = n0 + n1
        norm = float(np.linalg.norm(summed))
        if norm == 0.0:
            frame_valid = False
            break
        m = summed / norm
        g = np.cross(m, tangent)
        g_norm = float(np.linalg.norm(g))
        if g_norm == 0.0:
            frame_valid = False
            break
        g /= g_norm
        if ordinal == 0:
            g = _canonical_vector_sign(g)
        elif float(np.dot(g, segment_g[ordinal - 1])) < 0.0:
            g *= -1.0
        segment_g[ordinal] = g
    if not frame_valid:
        geometry = _empty_geometry(trace_id, seed, nodes, h_t, False, "TRACE_DISCONTINUITY")
        return geometry, _trace_row(case_token, geometry, canonical, state, normalized_arc, maximum_step)

    geometry = TraceGeometry(
        trace_id=trace_id,
        seed_index=seed,
        node_indices=nodes,
        h_t=h_t,
        valid=True,
        first_rejection_code="VALID",
        segment_starts=points[:-1].copy(),
        segment_vectors=segment_vectors,
        segment_lengths=segment_lengths,
        segment_g=segment_g,
        cumulative_lengths=np.concatenate(([0.0], np.cumsum(segment_lengths))),
    )
    return geometry, _trace_row(case_token, geometry, canonical, state, normalized_arc, maximum_step)


def _trace_row(
    case_token: str,
    geometry: TraceGeometry,
    canonical: CanonicalPoints,
    state: FieldState,
    normalized_arc: float,
    maximum_step: float,
) -> dict[str, Any]:
    nodes = geometry.node_indices
    maximum_turn = 0.0
    if len(nodes) > 1:
        for a, b in zip(nodes, nodes[1:]):
            if state.tangent_valid[a] and state.tangent_valid[b]:
                maximum_turn = max(maximum_turn, _angle_abs(state.tangent[a], state.tangent[b]))
    node_digest = _digest_ids(sorted(canonical.point_id_bytes[i] for i in nodes))
    return {
        "case_token": case_token,
        "trace_id": geometry.trace_id,
        "seed_point_id": canonical.point_ids[geometry.seed_index],
        "node_count": len(nodes),
        "h_t": geometry.h_t,
        "normalized_arc_length": normalized_arc,
        "maximum_normalized_step": maximum_step,
        "median_response_rad": _lower_quantile(state.response[nodes], 0.5),
        "maximum_turn_rad": maximum_turn,
        "node_table_digest": node_digest,
        "valid": geometry.valid,
        "first_rejection_code": geometry.first_rejection_code,
    }


def _nearest_segments(geometry: TraceGeometry, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    best_d2 = np.full(n, math.inf, dtype=np.float64)
    best_segment = np.zeros(n, dtype=np.int64)
    best_u = np.zeros(n, dtype=np.float64)
    best_c = np.zeros((n, 3), dtype=np.float64)
    for ordinal, (start, vector, length) in enumerate(
        zip(geometry.segment_starts, geometry.segment_vectors, geometry.segment_lengths)
    ):
        denom = float(length * length)
        offset = points - start
        u = np.clip((offset @ vector) / denom, 0.0, 1.0)
        c = points - (start + u[:, None] * vector)
        d2 = np.einsum("ij,ij->i", c, c)
        # Strict improvement preserves the smaller segment ordinal on exact ties.
        update = d2 < best_d2
        best_d2[update] = d2[update]
        best_segment[update] = ordinal
        best_u[update] = u[update]
        best_c[update] = c[update]
    return best_segment, best_u, best_c, np.sqrt(best_d2)


def build_memberships(
    case_token: str,
    geometry: TraceGeometry,
    canonical: CanonicalPoints,
    state: FieldState,
    excluded: set[int] | None = None,
) -> list[dict[str, Any]]:
    if not geometry.valid:
        return []
    excluded = excluded or set()
    segment, u, displacement, rho = _nearest_segments(geometry, canonical.points)
    trace_nodes = set(geometry.node_indices)
    total_length = float(geometry.cumulative_lengths[-1])
    rows = []
    for i in range(len(canonical.points)):
        if i in trace_nodes or i in excluded or not bool(np.all(state.normal_valid[i])):
            continue
        if rho[i] < 1.5 * geometry.h_t or rho[i] > 6.0 * geometry.h_t:
            continue
        side_cosine = float(np.dot(displacement[i], geometry.segment_g[segment[i]]) / rho[i])
        if side_cosine <= -0.35:
            side = "left"
        elif side_cosine >= 0.35:
            side = "right"
        else:
            continue
        arc_position = geometry.cumulative_lengths[segment[i]] + u[i] * geometry.segment_lengths[segment[i]]
        arc_bin = min(7, int(math.floor(8.0 * arc_position / total_length)))
        rows.append({
            "case_token": case_token,
            "trace_id": geometry.trace_id,
            "side": side,
            "point_id": canonical.point_ids[i],
            "nearest_segment_ordinal": int(segment[i]),
            "arc_bin": arc_bin,
            "rho_over_h_t": float(rho[i] / geometry.h_t),
            "signed_side_cosine": side_cosine,
            "response_rad": float(state.response[i]),
            "surface_variation": float(state.variation[i, 0]),
        })
    rows.sort(key=lambda row: canonical_row_key("strip_membership", row))
    return rows


def _strip_summary(
    case_token: str,
    trace_id: str,
    side: str,
    membership: list[dict[str, Any]],
    canonical: CanonicalPoints,
    state: FieldState,
) -> dict[str, Any]:
    rows = [row for row in membership if row["side"] == side]
    lookup = {point_id: i for i, point_id in enumerate(canonical.point_ids)}
    indices = [lookup[row["point_id"]] for row in rows]
    bin_counts = [0] * 8
    for row in rows:
        bin_counts[row["arc_bin"]] += 1
    if indices:
        normals = state.normals[indices, 0]
        second_moment = normals.T @ normals / float(len(indices))
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
        coherence = float(eigenvalues[2])
        dominant = _canonical_vector_sign(eigenvectors[:, 2])
        variation = state.variation[indices, 0]
        responses = state.response[indices]
    else:
        coherence = 0.0
        dominant = np.zeros(3, dtype=np.float64)
        variation = np.empty(0)
        responses = np.empty(0)
    sorted_ids = sorted((canonical.point_id_bytes[i] for i in indices))
    return {
        "case_token": case_token,
        "trace_id": trace_id,
        "side": side,
        "count": len(indices),
        "bin_counts": bin_counts,
        "covered_bins": sum(count >= 3 for count in bin_counts),
        "normal_coherence": coherence,
        "dominant_normal_x": float(dominant[0]),
        "dominant_normal_y": float(dominant[1]),
        "dominant_normal_z": float(dominant[2]),
        "median_surface_variation": _lower_quantile(variation, 0.5),
        "p90_surface_variation": _lower_quantile(variation, 0.9),
        "median_response_rad": _lower_quantile(responses, 0.5),
        "sorted_support_digest": _digest_ids(sorted_ids),
    }


def _certificate_gates(
    geometry: TraceGeometry,
    left: dict[str, Any],
    right: dict[str, Any],
    membership: list[dict[str, Any]],
    state: FieldState,
) -> dict[str, bool]:
    left_ids = {row["point_id"] for row in membership if row["side"] == "left"}
    right_ids = {row["point_id"] for row in membership if row["side"] == "right"}
    smaller = min(left["count"], right["count"])
    larger = max(left["count"], right["count"])
    cocovered = sum(a >= 3 and b >= 3 for a, b in zip(left["bin_counts"], right["bin_counts"]))
    left_n = np.array([left[f"dominant_normal_{axis}"] for axis in "xyz"])
    right_n = np.array([right[f"dominant_normal_{axis}"] for axis in "xyz"])
    separation = _angle_abs(left_n, right_n) if np.linalg.norm(left_n) and np.linalg.norm(right_n) else 0.0
    trace_median = _lower_quantile(state.response[geometry.node_indices], 0.5) if geometry.node_indices else 0.0
    return {
        "TRACE_INVALID": geometry.valid,
        "STRIP_MEMBERSHIP_OVERLAP": not bool(left_ids & right_ids),
        "LEFT_SUPPORT_INSUFFICIENT": left["count"] >= 32,
        "RIGHT_SUPPORT_INSUFFICIENT": right["count"] >= 32,
        "LEFT_COVERAGE_INSUFFICIENT": left["covered_bins"] >= 6,
        "RIGHT_COVERAGE_INSUFFICIENT": right["covered_bins"] >= 6,
        "BILATERAL_COCOVERAGE_INSUFFICIENT": cocovered >= 6,
        "SIDE_COUNT_BALANCE_INSUFFICIENT": larger > 0 and smaller / larger >= 0.25,
        "LEFT_NORMAL_COHERENCE_INSUFFICIENT": left["normal_coherence"] >= 0.8,
        "RIGHT_NORMAL_COHERENCE_INSUFFICIENT": right["normal_coherence"] >= 0.8,
        "LEFT_SURFACE_VARIATION_EXCESSIVE": left["median_surface_variation"] <= 0.12 and left["p90_surface_variation"] <= 0.25,
        "RIGHT_SURFACE_VARIATION_EXCESSIVE": right["median_surface_variation"] <= 0.12 and right["p90_surface_variation"] <= 0.25,
        "SIDE_NORMAL_SEPARATION_INSUFFICIENT": separation >= 0.13962634015954636,
        "TRACE_RESPONSE_INSUFFICIENT": trace_median >= 0.10471975511965978,
        "TRANSITION_CONTRAST_INSUFFICIENT": trace_median - max(left["median_response_rad"], right["median_response_rad"]) >= 0.03490658503988659,
    }


def _first_certificate_rejection(gates: Mapping[str, bool]) -> str:
    for code in CERTIFICATE_REJECTION_ORDER[:-1]:
        if not gates[code]:
            return code
    return "VALID"


def build_certificate(
    case_token: str,
    geometry: TraceGeometry,
    membership: list[dict[str, Any]],
    canonical: CanonicalPoints,
    state: FieldState,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = _strip_summary(case_token, geometry.trace_id, "left", membership, canonical, state)
    right = _strip_summary(case_token, geometry.trace_id, "right", membership, canonical, state)
    gates = _certificate_gates(geometry, left, right, membership, state)
    rejection = _first_certificate_rejection(gates)
    certificate_id = hashlib.sha256(
        CERT_DOMAIN + bytes.fromhex(geometry.trace_id) + bytes.fromhex(left["sorted_support_digest"])
        + bytes.fromhex(right["sorted_support_digest"]) + bytes.fromhex(CONFIG_SHA256)
    ).hexdigest()
    left_n = np.array([left[f"dominant_normal_{axis}"] for axis in "xyz"])
    right_n = np.array([right[f"dominant_normal_{axis}"] for axis in "xyz"])
    separation = _angle_abs(left_n, right_n) if np.linalg.norm(left_n) and np.linalg.norm(right_n) else 0.0
    trace_median = _lower_quantile(state.response[geometry.node_indices], 0.5) if geometry.node_indices else 0.0
    intersection_count = len(
        {r["point_id"] for r in membership if r["side"] == "left"}
        & {r["point_id"] for r in membership if r["side"] == "right"}
    )
    larger = max(left["count"], right["count"])
    side_ratio = min(left["count"], right["count"]) / larger if larger else 0.0
    row: dict[str, Any] = {
        "case_token": case_token,
        "certificate_id": certificate_id,
        "trace_id": geometry.trace_id,
        "trace_valid": geometry.valid,
        "membership_intersection_count": intersection_count,
        "left_support_count": left["count"],
        "right_support_count": right["count"],
        "left_covered_bins": left["covered_bins"],
        "right_covered_bins": right["covered_bins"],
        "bilaterally_covered_bins": sum(
            a >= 3 and b >= 3 for a, b in zip(left["bin_counts"], right["bin_counts"])
        ),
        "side_count_ratio": side_ratio,
        "left_normal_coherence": left["normal_coherence"],
        "right_normal_coherence": right["normal_coherence"],
        "left_median_surface_variation": left["median_surface_variation"],
        "left_p90_surface_variation": left["p90_surface_variation"],
        "right_median_surface_variation": right["median_surface_variation"],
        "right_p90_surface_variation": right["p90_surface_variation"],
        "side_normal_separation_rad": separation,
        "trace_median_response_rad": trace_median,
        "left_median_response_rad": left["median_response_rad"],
        "right_median_response_rad": right["median_response_rad"],
        "valid": rejection == "VALID",
        "first_rejection_code": rejection,
    }
    gate_field = {
        "TRACE_INVALID": "gate_trace_valid",
        "STRIP_MEMBERSHIP_OVERLAP": "gate_membership_disjoint",
        "LEFT_SUPPORT_INSUFFICIENT": "gate_left_support",
        "RIGHT_SUPPORT_INSUFFICIENT": "gate_right_support",
        "LEFT_COVERAGE_INSUFFICIENT": "gate_left_coverage",
        "RIGHT_COVERAGE_INSUFFICIENT": "gate_right_coverage",
        "BILATERAL_COCOVERAGE_INSUFFICIENT": "gate_bilateral_cocoverage",
        "SIDE_COUNT_BALANCE_INSUFFICIENT": "gate_side_count_balance",
        "LEFT_NORMAL_COHERENCE_INSUFFICIENT": "gate_left_normal_coherence",
        "RIGHT_NORMAL_COHERENCE_INSUFFICIENT": "gate_right_normal_coherence",
        "LEFT_SURFACE_VARIATION_EXCESSIVE": "gate_left_surface_variation",
        "RIGHT_SURFACE_VARIATION_EXCESSIVE": "gate_right_surface_variation",
        "SIDE_NORMAL_SEPARATION_INSUFFICIENT": "gate_side_normal_separation",
        "TRACE_RESPONSE_INSUFFICIENT": "gate_trace_response",
        "TRANSITION_CONTRAST_INSUFFICIENT": "gate_transition_contrast",
    }
    for code, field_name in gate_field.items():
        row[field_name] = gates[code]
    return row, [left, right]


def _quantize(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite numeric field cannot be serialized")
        quantized = float(np.rint(number / 1e-10) * 1e-10)
        return 0.0 if quantized == 0.0 else quantized
    if isinstance(value, dict):
        return {key: _quantize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_quantize(child) for child in value]
    return value


def _array_digest(array: np.ndarray) -> str:
    arr = np.asarray(array)
    if arr.dtype.kind == "f":
        arr = np.asarray(arr, dtype="<f8")
    elif arr.dtype.kind in "iu":
        arr = np.asarray(arr, dtype="<i8")
    elif arr.dtype.kind == "b":
        arr = np.asarray(arr, dtype=np.uint8)
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(
        arr.dtype.str.encode("ascii") + canonical_json_bytes(list(arr.shape)) + arr.tobytes(order="C")
    ).hexdigest()


def _table_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _rejection_rows(case_token: str, stage: str, codes: Sequence[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for code in codes:
        counts[code] = counts.get(code, 0) + 1
    denominator = len(codes)
    return [
        {
            "case_token": case_token,
            "stage": stage,
            "rejection_code": code,
            "numerator": count,
            "denominator": denominator,
        }
        for code, count in sorted(counts.items())
    ]


def _required_denominator_rows(
    case_token: str, stage: str, codes: Sequence[str]
) -> list[dict[str, Any]]:
    """Materialize a required denominator, including an explicit zero population."""
    if not codes:
        return [{
            "case_token": case_token,
            "stage": stage,
            "rejection_code": "EMPTY_POPULATION",
            "numerator": 0,
            "denominator": 0,
        }]
    return _rejection_rows(case_token, stage, codes)


def _seed_point_code(i: int, state: FieldState) -> str:
    if not bool(np.all(state.normal_valid[i])):
        return "INVALID_NORMALS"
    if state.d1[i] < 0.05235987755982989 or state.d2[i] < 0.05235987755982989 or state.response[i] < 0.10471975511965978:
        return "RESPONSE_FLOOR"
    if state.balance[i] < 0.35:
        return "SCALE_BALANCE"
    if not state.local_maxima[i]:
        return "NONMAXIMUM"
    return "VALID"


def _build_deletions(
    case_token: str,
    certificate: dict[str, Any],
    geometry: TraceGeometry,
    membership: list[dict[str, Any]],
    canonical: CanonicalPoints,
    state: FieldState,
) -> list[dict[str, Any]]:
    point_lookup = {point_id: i for i, point_id in enumerate(canonical.point_ids)}
    output = []
    for arm, side, required in (
        ("DELETE_LEFT", "left", "LEFT_SUPPORT_INSUFFICIENT"),
        ("DELETE_RIGHT", "right", "RIGHT_SUPPORT_INSUFFICIENT"),
    ):
        deleted_ids = sorted(row["point_id"] for row in membership if row["side"] == side)
        excluded = {point_lookup[point_id] for point_id in deleted_ids}
        rebuilt_membership = build_memberships(case_token, geometry, canonical, state, excluded)
        rebuilt, rebuilt_summaries = build_certificate(
            case_token, geometry, rebuilt_membership, canonical, state
        )
        rebuilt_left, rebuilt_right = rebuilt_summaries
        deleted_digest = _digest_ids(bytes.fromhex(point_id) for point_id in deleted_ids)
        row: dict[str, Any] = {
            "case_token": case_token,
            "certificate_id": certificate["certificate_id"],
            "deletion_arm": arm,
            "trace_id": geometry.trace_id,
            "deleted_id_digest": deleted_digest,
            "deleted_id_count": len(deleted_ids),
            "rebuilt_left_support_count": rebuilt_left["count"],
            "rebuilt_right_support_count": rebuilt_right["count"],
            "rebuilt_valid": rebuilt["valid"],
            "first_rejection_code": rebuilt["first_rejection_code"],
            "required_first_rejection_code": required,
            "raw_id_deletion_applied": True,
        }
        for key, value in rebuilt.items():
            if key.startswith("gate_"):
                row[f"rebuilt_{key}"] = value
        output.append(row)
    return output


def build_representation(
    records: Any,
    case_token: str,
    *,
    source_binding: Mapping[str, Any] | None = None,
) -> RepresentationResult:
    """Construct the complete coordinate-free semantic tables for one locked case."""
    require_authorized_config_hash()
    if case_token not in get_config_copy()["case_tokens"]:
        raise ValueError(f"case token is outside the locked C001-C037 envelope: {case_token}")
    np.seterr(all="raise")
    canonical = canonicalize_points(records)
    if canonical.unique_count < 128:
        raise ValueError("INSUFFICIENT_UNIQUE_POINTS")
    state = compute_field(canonical)

    seed_indices = [int(i) for i in np.flatnonzero(state.local_maxima)]
    caps = get_config_copy()["structural_caps"]
    if len(seed_indices) > caps["maximum_seed_rows_per_case"]:
        raise StructuralCapError("maximum_seed_rows_per_case exceeded before serialization")

    tables: dict[str, list[dict[str, Any]]] = {
        "case_manifest": [], "field_summary": [], "seed_primitive": [],
        "trace_primitive": [], "trace_node": [], "strip_membership": [],
        "strip_summary": [], "certificate": [], "destructive_negative": [],
        "rejection_census": [],
    }
    binding = dict(source_binding or {})
    tables["case_manifest"].append({
        "case_token": case_token,
        "raw_path": binding.get("raw_path"),
        "raw_ply_sha256": binding.get("raw_ply_sha256"),
        "observed_raw_ply_sha256": binding.get("observed_raw_ply_sha256", binding.get("raw_ply_sha256")),
        "raw_byte_count": binding.get("raw_byte_count"),
        "ply_header_sha256": binding.get("ply_header_sha256"),
        "ignored_property_name_digest": binding.get("ignored_property_name_digest"),
        "config_sha256": CONFIG_SHA256,
        "schema_sha256": binding.get("schema_sha256", SCHEMA_SHA256),
        "parsed_vertex_count": canonical.parsed_count,
        "finite_coordinate_count": canonical.finite_count,
        "unique_coordinate_count": canonical.unique_count,
        "exact_duplicate_count": canonical.exact_duplicate_count,
        "canonical_coordinate_multiset_digest": canonical.coordinate_multiset_digest,
        "sample_count": canonical.sample_count,
        "sampled_point_id_digest": canonical.sampled_point_id_digest,
        "validator_sampled_point_id_digest": canonical.sampled_point_id_digest,
        "source_points_serialized": False,
    })
    tables["field_summary"].append({
        "case_token": case_token,
        "sample_count": canonical.sample_count,
        "normal_valid_count_k16": int(np.sum(state.normal_valid[:, 0])),
        "normal_valid_count_k32": int(np.sum(state.normal_valid[:, 1])),
        "normal_valid_count_k64": int(np.sum(state.normal_valid[:, 2])),
        "seed_eligible_count": int(np.sum(state.seed_eligible)),
        "trace_eligible_count": int(np.sum(state.trace_eligible)),
        "deterministic_local_maximum_count": len(seed_indices),
        "dense_knn_digest": _array_digest(state.knn),
        "dense_normal_digest": hashlib.sha256("".join(_array_digest(state.normals[:, i, :]) for i in range(3)).encode()).hexdigest(),
        "dense_response_digest": _array_digest(state.response),
    })

    geometries: dict[str, TraceGeometry] = {}
    membership_by_trace: dict[str, list[dict[str, Any]]] = {}
    for seed in seed_indices:
        geometry, trace_row = build_trace(case_token, seed, canonical, state)
        geometries[geometry.trace_id] = geometry
        tables["trace_primitive"].append(trace_row)
        tables["seed_primitive"].append({
            "case_token": case_token,
            "seed_point_id": canonical.point_ids[seed],
            "d1_rad": float(state.d1[seed]),
            "d2_rad": float(state.d2[seed]),
            "response_rad": float(state.response[seed]),
            "scale_balance": float(state.balance[seed]),
            "tangent_support_count": int(state.tangent_support[seed]),
            "tangent_anisotropy": float(state.tangent_anisotropy[seed]),
            "trace_id": geometry.trace_id,
            "valid": geometry.valid,
            "first_rejection_code": geometry.first_rejection_code,
        })
        for ordinal, node in enumerate(geometry.node_indices):
            tables["trace_node"].append({
                "case_token": case_token,
                "trace_id": geometry.trace_id,
                "ordinal": ordinal,
                "point_id": canonical.point_ids[node],
                "response_rad": float(state.response[node]),
                "tangent_x": float(state.tangent[node, 0]),
                "tangent_y": float(state.tangent[node, 1]),
                "tangent_z": float(state.tangent[node, 2]),
            })
        membership = build_memberships(case_token, geometry, canonical, state)
        membership_by_trace[geometry.trace_id] = membership
        tables["strip_membership"].extend(membership)
        certificate, summaries = build_certificate(case_token, geometry, membership, canonical, state)
        tables["strip_summary"].extend(summaries)
        tables["certificate"].append(certificate)
        if certificate["valid"]:
            tables["destructive_negative"].extend(
                _build_deletions(case_token, certificate, geometry, membership, canonical, state)
            )

    if len(tables["trace_primitive"]) > caps["maximum_trace_rows_per_case"]:
        raise StructuralCapError("maximum_trace_rows_per_case exceeded")
    if len(tables["trace_node"]) > caps["maximum_trace_node_rows_per_case"]:
        raise StructuralCapError("maximum_trace_node_rows_per_case exceeded")
    if len(tables["strip_membership"]) > caps["maximum_strip_membership_rows_per_case"]:
        raise StructuralCapError("maximum_strip_membership_rows_per_case exceeded")
    if len(tables["certificate"]) > caps["maximum_certificate_rows_per_case"]:
        raise StructuralCapError("maximum_certificate_rows_per_case exceeded")

    tables["rejection_census"].extend(
        _rejection_rows(case_token, "sampled_point_seed", [_seed_point_code(i, state) for i in range(canonical.sample_count)])
    )
    tables["rejection_census"].extend(
        _rejection_rows(case_token, "seed_primitive", [row["first_rejection_code"] for row in tables["seed_primitive"]])
    )
    tables["rejection_census"].extend(
        _rejection_rows(case_token, "trace_primitive", [row["first_rejection_code"] for row in tables["trace_primitive"]])
    )
    tables["rejection_census"].extend(
        _rejection_rows(case_token, "certificate", [row["first_rejection_code"] for row in tables["certificate"]])
    )
    tables["rejection_census"].extend(
        _rejection_rows(case_token, "destructive_negative", [row["first_rejection_code"] for row in tables["destructive_negative"]])
    )

    # Literal ruling denominators.  Each denominator is the population named by
    # the stage, not its parent population.  EMPTY 0/0 makes a zero population
    # explicit rather than silently zero-imputing an absent census code.
    required_stage_codes: tuple[tuple[str, list[str]], ...] = (
        ("all_sampled_points", ["VALID"] * canonical.sample_count),
        ("normal_valid_k16", ["VALID"] * int(np.sum(state.normal_valid[:, 0]))),
        ("normal_valid_k32", ["VALID"] * int(np.sum(state.normal_valid[:, 1]))),
        ("normal_valid_k64", ["VALID"] * int(np.sum(state.normal_valid[:, 2]))),
        ("seed_eligible_points", ["VALID"] * int(np.sum(state.seed_eligible))),
        ("deterministic_local_maxima", ["VALID"] * len(seed_indices)),
        ("seed_primitives", [row["first_rejection_code"] for row in tables["seed_primitive"]]),
        ("valid_traces", ["VALID"] * sum(bool(row["valid"]) for row in tables["trace_primitive"])),
        ("certificate_candidates", [row["first_rejection_code"] for row in tables["certificate"]]),
        ("valid_certificates", ["VALID"] * sum(bool(row["valid"]) for row in tables["certificate"])),
        ("destructive_negative_arms", [row["first_rejection_code"] for row in tables["destructive_negative"]]),
    )
    for stage, codes in required_stage_codes:
        tables["rejection_census"].extend(
            _required_denominator_rows(case_token, stage, codes)
        )

    for table_name, rows in tables.items():
        rows.sort(key=lambda row: canonical_row_key(table_name, row))
        tables[table_name] = _quantize(rows)
    assert_coordinate_free(tables)
    return RepresentationResult(case_token, tables)


_PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
}


def parse_ply_xyz(raw: bytes) -> tuple[np.ndarray, bytes, int, list[str]]:
    """Parse only declared vertex x/y/z values while structurally skipping all others."""
    marker = b"end_header"
    location = raw.find(marker)
    if location < 0:
        raise ValueError("PLY header has no end_header")
    line_end = raw.find(b"\n", location)
    header_end = len(raw) if line_end < 0 else line_end + 1
    header = raw[:header_end]
    lines = header.decode("ascii", errors="strict").replace("\r\n", "\n").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY stream")
    format_name = None
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    ignored: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                prop = ("list", parts[2], parts[3], parts[4])
                ignored.append(f"{current['name']}:{parts[4]}")
            else:
                prop = ("scalar", parts[1], parts[2])
                if current["name"] != "vertex" or parts[2] not in {"x", "y", "z"}:
                    ignored.append(f"{current['name']}:{parts[2]}")
            current["properties"].append(prop)
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {format_name}")
    vertex_elements = [element for element in elements if element["name"] == "vertex"]
    if len(vertex_elements) != 1:
        raise ValueError("PLY must declare exactly one vertex element")
    vertex = vertex_elements[0]
    scalar_names = {prop[2] for prop in vertex["properties"] if prop[0] == "scalar"}
    if not {"x", "y", "z"}.issubset(scalar_names):
        raise ValueError("PLY vertex element must declare x, y, z")

    rows: list[tuple[float, float, float]] = []
    if format_name == "ascii":
        body_lines = raw[header_end:].decode("ascii", errors="strict").splitlines()
        cursor = 0
        for element in elements:
            for _ in range(element["count"]):
                if cursor >= len(body_lines):
                    raise ValueError("truncated ASCII PLY body")
                tokens = body_lines[cursor].split()
                cursor += 1
                token_index = 0
                values: dict[str, float] = {}
                for prop in element["properties"]:
                    if prop[0] == "scalar":
                        if token_index >= len(tokens):
                            raise ValueError("truncated ASCII PLY record")
                        if element["name"] == "vertex" and prop[2] in {"x", "y", "z"}:
                            values[prop[2]] = float(tokens[token_index])
                        token_index += 1
                    else:
                        if token_index >= len(tokens):
                            raise ValueError("truncated ASCII PLY list")
                        count = int(tokens[token_index]); token_index += 1 + count
                if token_index != len(tokens):
                    raise ValueError("extra tokens in ASCII PLY record")
                if element["name"] == "vertex":
                    rows.append((values["x"], values["y"], values["z"]))
    else:
        endian = "<" if format_name == "binary_little_endian" else ">"
        cursor = header_end
        for element in elements:
            for _ in range(element["count"]):
                values: dict[str, float] = {}
                for prop in element["properties"]:
                    if prop[0] == "scalar":
                        fmt, size = _PLY_TYPES[prop[1]]
                        if cursor + size > len(raw):
                            raise ValueError("truncated binary PLY record")
                        value = struct.unpack_from(endian + fmt, raw, cursor)[0]; cursor += size
                        if element["name"] == "vertex" and prop[2] in {"x", "y", "z"}:
                            values[prop[2]] = float(value)
                    else:
                        count_fmt, count_size = _PLY_TYPES[prop[1]]
                        if cursor + count_size > len(raw):
                            raise ValueError("truncated binary PLY list count")
                        count = int(struct.unpack_from(endian + count_fmt, raw, cursor)[0]); cursor += count_size
                        _, item_size = _PLY_TYPES[prop[2]]
                        cursor += count * item_size
                        if cursor > len(raw):
                            raise ValueError("truncated binary PLY list")
                if element["name"] == "vertex":
                    rows.append((values["x"], values["y"], values["z"]))
    if len(rows) != vertex["count"]:
        raise ValueError("parsed vertex count disagrees with PLY declaration")
    return np.asarray(rows, dtype=np.float64), header, len(rows), ignored


def _manifest_rows(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return manifest
    for key in ("cases", "rows", "case_manifest"):
        if isinstance(manifest, dict) and isinstance(manifest.get(key), list):
            return manifest[key]
    raise ValueError("source manifest has no case rows")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("BASE_A", "BASE_B", "ORDER_PERMUTED"))
    parser.add_argument("--case-token", required=True)
    parser.add_argument("--raw-ply", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--expected-source-manifest-file-sha256", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--expected-configuration-file-sha256", required=True)
    parser.add_argument("--expected-configuration-sha256", required=True)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--runner-source", required=True, type=Path)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if _sha256_file(Path(__file__).resolve()) != args.expected_executable_sha256:
        raise RuntimeError("semantic executable SHA mismatch")
    if _sha256_file(args.runner_source) != args.expected_runner_sha256:
        raise RuntimeError("runner source SHA mismatch")
    source_bytes = args.source_manifest.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != args.expected_source_manifest_file_sha256:
        raise RuntimeError("source manifest file SHA mismatch")
    manifest = json.loads(source_bytes)
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest must be a JSON object")
    manifest_content = dict(manifest)
    asserted_manifest_sha = manifest_content.pop("input_manifest_sha256", None)
    observed_manifest_sha = hashlib.sha256(canonical_json_bytes(manifest_content)).hexdigest()
    if (
        asserted_manifest_sha != observed_manifest_sha
        or observed_manifest_sha != args.expected_input_manifest_sha256
    ):
        raise RuntimeError("input manifest content SHA mismatch")

    if args.expected_configuration_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("unauthorized expected configuration SHA")
    configuration_bytes = args.configuration.read_bytes()
    if hashlib.sha256(configuration_bytes).hexdigest() != args.expected_configuration_file_sha256:
        raise RuntimeError("configuration file SHA mismatch")
    configuration = json.loads(configuration_bytes)
    observed_configuration_sha = hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
    if observed_configuration_sha != args.expected_configuration_sha256:
        raise RuntimeError("configuration content SHA mismatch")
    if _sha256_file(args.schema) != args.expected_schema_sha256:
        raise RuntimeError("schema file SHA mismatch")

    matching = [row for row in _manifest_rows(manifest) if row.get("case_token") == args.case_token]
    if len(matching) != 1:
        raise RuntimeError("case token must have exactly one source binding")
    binding = dict(matching[0])
    raw = args.raw_ply.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != binding.get("raw_ply_sha256"):
        raise RuntimeError("raw PLY SHA mismatch before parsing")
    points, header, parsed_count, ignored = parse_ply_xyz(raw)
    if hashlib.sha256(header).hexdigest() != binding.get("ply_header_sha256"):
        raise RuntimeError("PLY header SHA mismatch")
    if parsed_count != int(binding.get("parsed_vertex_count", -1)):
        raise RuntimeError("parsed vertex count mismatch")
    ignored_digest = hashlib.sha256(canonical_json_bytes(sorted(ignored))).hexdigest()
    if binding.get("ignored_property_name_digest") not in (None, ignored_digest):
        raise RuntimeError("ignored-property-name digest mismatch")

    if args.role == "ORDER_PERMUTED":
        points = deterministic_record_permutation(points)
    binding.update({
        "raw_path": str(args.raw_ply),
        "raw_byte_count": len(raw),
        "observed_raw_ply_sha256": observed_sha,
        "ignored_property_name_digest": ignored_digest,
        "schema_sha256": args.expected_schema_sha256,
    })
    result = build_representation(points, args.case_token, source_binding=binding)
    payload = result.compact_payload(args.role)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "case_payload.json"
    output.write_bytes(canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
