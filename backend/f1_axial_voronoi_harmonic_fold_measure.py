"""Frozen research-only V32 axial Voronoi harmonic fold measure.

This module is source-only and side-effect free.  It imports only pure frozen
V31 primitives; orchestration, locking, fixtures, and visual evidence live in
the research scripts.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
import os
import struct
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from backend.f1_distributed_harmonic_fold_flux import (
    ACTION_LEGACY,
    ANGULAR_BIN_MINIMUM_MASS,
    ANGULAR_BIN_MINIMUM_NODES,
    Candidate,
    EDGE_MAXIMUM_S,
    MAXIMUM_ANGULAR_VARIATION_DEGREES,
    MAXIMUM_BACKTRACK_DEGREES,
    MAXIMUM_EDGE_CURRENT_FRACTION,
    MAXIMUM_OFF_ARC_MASS,
    MINIMUM_EFFECTIVE_PARTICIPATION,
    MUTUAL_NEIGHBORS,
    OFF_ARC_MARGIN_DEGREES,
    POTENTIAL_BAND_HALF_WIDTH,
    POTENTIAL_BAND_MINIMUM_MASS,
    POTENTIAL_BAND_MINIMUM_NODES,
    Q24,
    SourceGeometry,
    TAIL_BINS,
    TAIL_COMPONENT_FRACTION,
    TAIL_MAX_S,
    TAIL_MINIMUM_PER_BIN,
    TAIL_MINIMUM_TOTAL,
    TAIL_MIN_S,
    _components,
    _edge_allowed,
    _orient_normals,
    _stable_digest,
    _tail_seeds,
    _weighted_median,
    ordinary_nms,
    parse_candidates,
    parse_source,
)

PROFILE = "V32_AXIAL_VORONOI_HARMONIC_FOLD_MEASURE_V1"
PRIMITIVE = "AVHFM_V1"
SCHEMA_VERSION = 1

MAX_SOURCE_SAMPLES = 200_000
MAX_CANDIDATES = 128
MAX_LOCAL_VERTICES = 32_768
MAX_LOCAL_EDGES = 262_144
MAX_AXIAL_EVENTS = 65_536
MAX_WORKERS = 32

AUDIT_HALF_S = 12.0
SOLVE_HALF_S = 13.2
TRANSVERSE_RADIUS_S = 24.0
WINDOW_HALF_S = 1.2
CELL_HALF_CAP_S = 0.75
MEASURE_MINIMUM_S = 19.5
PHASE_MEASURE_MINIMUM_S = 22.5
MAX_GAP_S = 1.2
LONGEST_RUN_MINIMUM_S = 14.4
ENDPOINT_WIDTH_S = 2.4
ENDPOINT_MEASURE_MINIMUM_S = 1.8
ELECTRODE_MEASURE_MINIMUM_S = 22.5
ELECTRODE_MAX_GAP_S = 1.2
AXIAL_LEAKAGE_MAXIMUM = 1.0 / 3.0

Q20 = 1 << 20
Q21 = 1 << 21
Q40 = 1 << 40
CURRENT_ZERO_Q40 = 1 << 16  # 2^-24 in Q40 units
PCG_MAXIMUM_ITERATIONS = 4096
PCG_RESIDUAL = 2.0 ** -36

CACHE_HEADER_BYTES = 256
CACHE_CANDIDATE_BYTES = 1024
MAX_CACHE_BYTES = CACHE_HEADER_BYTES + MAX_CANDIDATES * CACHE_CANDIDATE_BYTES

ACTION_RESCUE = "AVHFM_PASS"
ACTION_NONE = "NO_RESCUE"

STAGES = (
    "CURRENT",
    "AXIAL_LEAKAGE",
    "TAIL_COVERAGE",
    "POTENTIAL_COVERAGE",
    "ANGULAR_COVERAGE",
    "NORMAL_TURN",
    "OFF_ARC_CURRENT",
)

TOKENS = (
    "V32_PARENT_CONTRACT_MISMATCH",
    "V32_FORBIDDEN_INPUT",
    "V32_INVALID_NUMERIC_INPUT",
    "V32_SOURCE_CAP_BREACH",
    "V32_CANDIDATE_CAP_BREACH",
    "V32_FRAME_INVALID",
    "V32_GRAPH_CAP_BREACH",
    "V32_ELECTRODE_0_COVERAGE",
    "V32_ELECTRODE_1_COVERAGE",
    "V32_ELECTRODE_COMPONENT",
    "V32_SOLVER_NONCONVERGENCE",
    "V32_NO_CURRENT_EVENTS",
    "V32_CURRENT_MEASURE",
    "V32_AXIAL_LEAKAGE_MEASURE",
    "V32_TAIL_COVERAGE_MEASURE",
    "V32_POTENTIAL_COVERAGE_MEASURE",
    "V32_ANGULAR_COVERAGE_MEASURE",
    "V32_NORMAL_TURN_MEASURE",
    "V32_OFF_ARC_CURRENT_MEASURE",
    "V32_AXIAL_MAX_GAP",
    "V32_AXIAL_LONGEST_RUN",
    "V32_AXIAL_ENDPOINT_MEASURE",
    "V32_ACCEPT",
    "ORDINARY_NMS_SUPPRESSED",
)
TOKEN_CODE = {value: index for index, value in enumerate(TOKENS)}


class AVHFMContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    return _canonical_bytes({key: value for key, value in result.items() if key != "_ephemeral"}) + b"\n"


def _round_even(value: float) -> int:
    return int(round(float(value)))


def _q(value: float, scale: int) -> float:
    return _round_even(float(value) * scale) / scale


def _pairwise_sum(values: Sequence[float] | np.ndarray) -> float:
    work = [float(v) for v in values]
    if not work:
        return 0.0
    while len(work) > 1:
        work = [work[i] + (work[i + 1] if i + 1 < len(work) else 0.0) for i in range(0, len(work), 2)]
    return work[0]


def _canonical_candidate(candidate: Candidate) -> tuple[Candidate, np.ndarray, float]:
    axis = np.asarray(candidate.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    for component in axis:
        if abs(float(component)) >= 2.0 ** -30:
            if component < 0.0:
                axis = -axis
            break
    parents = candidate.parents
    projected = []
    for parent in parents:
        normal = parent.normal - float(np.dot(parent.normal, axis)) * axis
        norm = float(np.linalg.norm(normal))
        if norm <= 2.0 ** -30:
            raise AVHFMContractError("V32_FRAME_INVALID")
        projected.append(normal / norm)
    cross = float(np.dot(axis, np.cross(projected[0], projected[1])))
    if cross < 0.0:
        parents = (parents[1], parents[0])
        projected = [projected[1], projected[0]]
        cross = -cross
    theta = math.atan2(cross, float(np.dot(projected[0], projected[1])))
    if theta < math.pi / 6.0 or theta > 5.0 * math.pi / 6.0:
        raise AVHFMContractError("V32_FRAME_INVALID")
    return replace(candidate, axis=axis, parents=parents, alpha_degrees=math.degrees(theta)), projected[0], theta


def _candidate_subset(candidate: Candidate, source: SourceGeometry) -> tuple[np.ndarray, np.ndarray]:
    rel = source.positions - candidate.center
    z = rel @ candidate.axis
    radial = np.linalg.norm(rel - z[:, None] * candidate.axis, axis=1)
    indices = np.flatnonzero((z >= -SOLVE_HALF_S * candidate.scale) & (z <= SOLVE_HALF_S * candidate.scale) & (radial <= TRANSVERSE_RADIUS_S * candidate.scale))
    indices = np.asarray(sorted(indices, key=lambda i: (int(source.stable_keys[i]), _stable_digest(rel[i] / candidate.scale, source.normals[i], int(source.parents[i])))), dtype=np.int64)
    if len(indices) > MAX_LOCAL_VERTICES:
        raise AVHFMContractError("V32_GRAPH_CAP_BREACH", "vertices")
    return indices, z[indices]


def _surface_graph(candidate: Candidate, source: SourceGeometry, indices: np.ndarray) -> tuple[list[tuple[int, int, float]], list[list[int]]]:
    points = source.positions[indices]
    global_to_local = {int(value): pos for pos, value in enumerate(indices)}
    proposed: set[tuple[int, int]] = set()
    for ga, gb in source.mesh_edges:
        if ga in global_to_local and gb in global_to_local:
            a, b = global_to_local[ga], global_to_local[gb]
            proposed.add((min(a, b), max(a, b)))
    if len(points) > 1:
        k = min(MUTUAL_NEIGHBORS + 1, len(points))
        _distance, neighbors = cKDTree(points).query(points, k=k, workers=1)
        if k == 1:
            neighbors = neighbors[:, None]
        neighbor_sets = [set(int(v) for v in np.atleast_1d(row) if int(v) != i) for i, row in enumerate(neighbors)]
        for i, values in enumerate(neighbor_sets):
            for j in values:
                if i in neighbor_sets[j]:
                    proposed.add((min(i, j), max(i, j)))
    base: list[tuple[int, int, float]] = []
    degree = np.zeros(len(points), dtype=np.float64)
    for i, j in sorted(proposed):
        allowed, distance = _edge_allowed(points[j] - points[i], source.normals[indices[i]], source.normals[indices[j]], candidate.scale)
        if not allowed:
            continue
        kernel = (1.0 - (distance / (EDGE_MAXIMUM_S * candidate.scale)) ** 2) ** 2
        base.append((i, j, kernel))
        degree[i] += kernel
        degree[j] += kernel
    edges: list[tuple[int, int, float]] = []
    adjacency: list[list[int]] = [[] for _ in points]
    for i, j, kernel in base:
        if degree[i] <= 0.0 or degree[j] <= 0.0:
            continue
        weight = _q(kernel / math.sqrt(degree[i] * degree[j]), Q24)
        if weight <= 0.0:
            continue
        edges.append((i, j, weight))
        adjacency[i].append(j)
        adjacency[j].append(i)
    if len(edges) > MAX_LOCAL_EDGES:
        raise AVHFMContractError("V32_GRAPH_CAP_BREACH", "edges")
    for row in adjacency:
        row.sort()
    return edges, adjacency


def _cells_q21(z_values_s: Sequence[float]) -> list[tuple[int, int]]:
    values = sorted(set(_round_even(float(z) * Q20) for z in z_values_s if -AUDIT_HALF_S <= z <= AUDIT_HALF_S))
    cells: list[tuple[int, int]] = []
    lower_bound, upper_bound = -_round_even(AUDIT_HALF_S * Q21), _round_even(AUDIT_HALF_S * Q21)
    cap = _round_even(CELL_HALF_CAP_S * Q21)
    for k, value_q20 in enumerate(values):
        center = value_q20 * 2
        left = max(lower_bound, center - cap)
        right = min(upper_bound, center + cap)
        if k:
            left = max(left, values[k - 1] + value_q20)
        if k + 1 < len(values):
            right = min(right, value_q20 + values[k + 1])
        if right > left:
            cells.append((left, right))
    return cells


def _merged(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for left, right in sorted(intervals):
        if not out or left > out[-1][1]:
            out.append([left, right])
        else:
            out[-1][1] = max(out[-1][1], right)
    return [(a, b) for a, b in out]


def _interval_metrics(intervals: Sequence[tuple[int, int]]) -> tuple[int, int, int, int, int]:
    merged = _merged(intervals)
    lo, hi = -_round_even(AUDIT_HALF_S * Q21), _round_even(AUDIT_HALF_S * Q21)
    total = sum(b - a for a, b in merged)
    longest = max((b - a for a, b in merged), default=0)
    cursor, gap = lo, 0
    for a, b in merged:
        gap = max(gap, max(0, a - cursor))
        cursor = max(cursor, b)
    gap = max(gap, hi - cursor)
    end_width = _round_even(ENDPOINT_WIDTH_S * Q21)
    left = sum(max(0, min(b, lo + end_width) - max(a, lo)) for a, b in merged)
    right = sum(max(0, min(b, hi) - max(a, hi - end_width)) for a, b in merged)
    return total, longest, gap, left, right


def _electrode_ok(z_s: np.ndarray, local: np.ndarray) -> bool:
    if len(local) < 16:
        return False
    values = z_s[local]
    if any(not np.any((values >= lo) & (values < hi if hi < AUDIT_HALF_S else values <= hi)) for lo, hi in ((-12,-6),(-6,0),(0,6),(6,12))):
        return False
    total, _longest, gap, _left, _right = _interval_metrics(_cells_q21(values.tolist()))
    return total >= _round_even(ELECTRODE_MEASURE_MINIMUM_S * Q21) and gap <= _round_even(ELECTRODE_MAX_GAP_S * Q21)


def _pcg(node_count: int, edges: Sequence[tuple[int, int, float]], boundary0: set[int], boundary1: set[int], active: set[int]) -> tuple[np.ndarray | None, int]:
    phi = np.zeros(node_count, dtype=np.float64)
    phi[list(boundary1)] = 1.0
    interior = sorted(active - boundary0 - boundary1)
    if not interior:
        return None, 0
    mapping = {node: row for row, node in enumerate(interior)}
    diagonal = np.zeros(len(interior), dtype=np.float64)
    rhs = np.zeros(len(interior), dtype=np.float64)
    rows: list[list[tuple[int, float]]] = [[] for _ in interior]
    for i, j, weight in edges:
        if i not in active or j not in active:
            continue
        for node, other in ((i, j), (j, i)):
            if node not in mapping:
                continue
            row = mapping[node]
            diagonal[row] += weight
            if other in mapping:
                rows[row].append((mapping[other], -weight))
            elif other in boundary1:
                rhs[row] += weight
    if np.any(diagonal <= 0.0):
        return None, 0
    for row in rows:
        row.sort()
    def multiply(vector: np.ndarray) -> np.ndarray:
        result = diagonal * vector
        for row, entries in enumerate(rows):
            result[row] += _pairwise_sum([value * vector[column] for column, value in entries])
        return result
    x = np.full(len(interior), 0.5, dtype=np.float64)
    residual = rhs - multiply(x)
    scale = max(1.0, float(np.max(np.abs(rhs))))
    z = residual / diagonal
    direction = z.copy()
    rz = _pairwise_sum(residual * z)
    for iteration in range(PCG_MAXIMUM_ITERATIONS + 1):
        if float(np.max(np.abs(residual))) <= PCG_RESIDUAL * scale:
            phi[interior] = np.rint(x * Q40) / Q40
            return phi, iteration
        if iteration == PCG_MAXIMUM_ITERATIONS:
            break
        product = multiply(direction)
        denominator = _pairwise_sum(direction * product)
        if not math.isfinite(denominator) or denominator <= 0.0:
            break
        step = rz / denominator
        x += step * direction
        residual -= step * product
        z = residual / diagonal
        next_rz = _pairwise_sum(residual * z)
        if not math.isfinite(next_rz) or rz == 0.0:
            break
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    return None, PCG_MAXIMUM_ITERATIONS


def _normal_progress(normals: np.ndarray, axis: np.ndarray, reference: np.ndarray, theta: float) -> np.ndarray:
    values = np.full(len(normals), np.nan, dtype=np.float64)
    for i, normal in enumerate(normals):
        projected = normal - float(np.dot(normal, axis)) * axis
        norm = float(np.linalg.norm(projected))
        if norm <= 2.0 ** -30:
            continue
        projected /= norm
        angle = math.atan2(float(np.dot(axis, np.cross(reference, projected))), float(np.dot(reference, projected)))
        alternatives = (angle - 2 * math.pi, angle, angle + 2 * math.pi)
        angle = min(alternatives, key=lambda v: 0.0 if 0.0 <= v <= theta else min(abs(v), abs(v - theta)))
        values[i] = _q(angle / theta, Q24)
    return values


def _local_stage(
    edges: Sequence[dict[str, Any]],
    node_count: int,
    phi: np.ndarray,
    progress: np.ndarray,
    stable: np.ndarray,
    boundary0: set[int],
    boundary1: set[int],
    tail_bin0: np.ndarray,
    tail_bin1: np.ndarray,
    theta_degrees: float,
) -> tuple[int, str, float]:
    participation = np.zeros(node_count, dtype=np.float64)
    total_edge = _pairwise_sum([edge["current"] for edge in edges])
    for edge in edges:
        mass = 0.5 * edge["current"]
        participation[edge["i"]] += mass
        participation[edge["j"]] += mass
    total_mass = _pairwise_sum(participation)
    effective = total_mass * total_mass / max(_pairwise_sum(participation * participation), 1e-300)
    bottleneck = max((edge["current"] for edge in edges), default=0.0) / max(total_edge, 1e-300)
    if _q(effective, Q24) < MINIMUM_EFFECTIVE_PARTICIPATION or _q(bottleneck, Q24) > MAXIMUM_EDGE_CURRENT_FRACTION:
        return 0, "V32_CURRENT_MEASURE", 1.0
    leakage_num = _pairwise_sum([edge["current"] * edge["axial_fraction"] for edge in edges])
    leakage = leakage_num / max(total_edge, 1e-300)
    if _q(leakage, Q24) > _q(AXIAL_LEAKAGE_MAXIMUM, Q24):
        return 1, "V32_AXIAL_LEAKAGE_MEASURE", leakage
    nodes = set(int(v) for edge in edges for v in (edge["i"], edge["j"]))
    local0 = sorted(nodes & boundary0)
    local1 = sorted(nodes & boundary1)
    bin_counts0 = [sum(int(tail_bin0[node]) == k for node in local0) for k in range(len(TAIL_BINS))]
    bin_counts1 = [sum(int(tail_bin1[node]) == k for node in local1) for k in range(len(TAIL_BINS))]
    if (
        len(local0) < TAIL_MINIMUM_TOTAL
        or len(local1) < TAIL_MINIMUM_TOTAL
        or min(bin_counts0, default=0) < TAIL_MINIMUM_PER_BIN
        or min(bin_counts1, default=0) < TAIL_MINIMUM_PER_BIN
    ):
        return 2, "V32_TAIL_COVERAGE_MEASURE", leakage
    active = participation > 0.0
    bands: list[np.ndarray] = []
    for k in range(1, 8):
        band = np.flatnonzero(active & (np.abs(phi - k / 8.0) <= POTENTIAL_BAND_HALF_WIDTH))
        if len(band) < POTENTIAL_BAND_MINIMUM_NODES or _q(_pairwise_sum(participation[band]) / total_mass, Q24) < POTENTIAL_BAND_MINIMUM_MASS:
            return 3, "V32_POTENTIAL_COVERAGE_MEASURE", leakage
        bands.append(band)
    for k in range(1, 8):
        angular = np.flatnonzero(active & np.isfinite(progress) & (np.abs(progress - k / 8.0) <= 1.0 / 16.0))
        if len(angular) < ANGULAR_BIN_MINIMUM_NODES or _q(_pairwise_sum(participation[angular]) / total_mass, Q24) < ANGULAR_BIN_MINIMUM_MASS:
            return 4, "V32_ANGULAR_COVERAGE_MEASURE", leakage
    medians = [0.0] + [_weighted_median(progress[band], participation[band], stable[band]) for band in bands] + [1.0]
    backtrack = sum(max(0.0, medians[k] - medians[k + 1]) for k in range(8)) * theta_degrees
    variation = (sum(abs(medians[k + 1] - medians[k]) for k in range(8)) - 1.0) * theta_degrees
    if _q(backtrack, Q24) > MAXIMUM_BACKTRACK_DEGREES or _q(variation, Q24) > MAXIMUM_ANGULAR_VARIATION_DEGREES:
        return 5, "V32_NORMAL_TURN_MEASURE", leakage
    margin = OFF_ARC_MARGIN_DEGREES / theta_degrees
    off = _pairwise_sum(participation[active & ((progress < -margin) | (progress > 1.0 + margin) | ~np.isfinite(progress))]) / total_mass
    if _q(off, Q24) > MAXIMUM_OFF_ARC_MASS:
        return 6, "V32_OFF_ARC_CURRENT_MEASURE", leakage
    return len(STAGES), "V32_ACCEPT", leakage


def _base_decision(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "frozen_v16_order": candidate.frozen_order,
        "candidate_sha256": hashlib.sha256(_canonical_bytes(candidate.raw)).hexdigest(),
        "anchor_xyz": candidate.center.tolist(),
        "axis_xyz": candidate.axis.tolist(),
        "line_start_xyz": candidate.line_start.tolist(),
        "line_end_xyz": candidate.line_end.tolist(),
        "scale": float(candidate.scale),
        "theta_q24": 0,
        "total_current_q24": 0,
        "stage_measures_q21": {},
        "reason_counts": {},
        "interval_witnesses": [],
        "interval_witnesses_truncated": False,
    }


def _failure(candidate: Candidate, reason: str, **extra: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = {**_base_decision(candidate), "action": ACTION_NONE, "reason": reason, "accepted_length_q21": 0, "longest_run_q21": 0, "maximum_gap_q21": _round_even(24 * Q21), "endpoint_left_q21": 0, "endpoint_right_q21": 0}
    decision.update(extra)
    return decision, {"candidate_id": candidate.candidate_id, "events": [], "intervals": [], **extra}


def _candidate_decision(original: Candidate, source: SourceGeometry) -> tuple[dict[str, Any], dict[str, Any]]:
    if original.selected_legacy:
        return ({**_base_decision(original), "action": ACTION_LEGACY, "reason": None, "accepted_length_q21": 0, "longest_run_q21": 0, "maximum_gap_q21": 0, "endpoint_left_q21": 0, "endpoint_right_q21": 0}, {"candidate_id": original.candidate_id, "events": [], "intervals": []})
    if not original.eligible:
        return _failure(original, "V32_FRAME_INVALID")
    try:
        candidate, reference, theta = _canonical_candidate(original)
        indices, z = _candidate_subset(candidate, source)
        edges, adjacency = _surface_graph(candidate, source, indices)
    except AVHFMContractError as exc:
        if exc.code == "V32_GRAPH_CAP_BREACH":
            raise
        return _failure(original, exc.code)
    seeds0_array, tail0 = _tail_seeds(candidate, source, indices, candidate.center, 0)
    seeds1_array, tail1 = _tail_seeds(candidate, source, indices, candidate.center, 1)
    seeds0, seeds1 = set(int(v) for v in seeds0_array), set(int(v) for v in seeds1_array)
    tail_bin0 = np.full(len(indices), -1, dtype=np.int8)
    tail_bin1 = np.full(len(indices), -1, dtype=np.int8)
    for node in seeds0:
        x = float(tail0["x"][node])
        for bin_index, (lo, hi) in enumerate(TAIL_BINS):
            if x >= lo and (x < hi if hi < TAIL_MAX_S else x <= hi):
                tail_bin0[node] = bin_index
                break
    for node in seeds1:
        x = float(tail1["x"][node])
        for bin_index, (lo, hi) in enumerate(TAIL_BINS):
            if x >= lo and (x < hi if hi < TAIL_MAX_S else x <= hi):
                tail_bin1[node] = bin_index
                break
    z_s = z / candidate.scale
    if not _electrode_ok(z_s, seeds0_array):
        return _failure(original, "V32_ELECTRODE_0_COVERAGE", graph_vertices=len(indices), graph_edges=len(edges))
    if not _electrode_ok(z_s, seeds1_array):
        return _failure(original, "V32_ELECTRODE_1_COVERAGE", graph_vertices=len(indices), graph_edges=len(edges))
    components = _components(adjacency)
    connecting = [component for component in components if seeds0.intersection(component) and seeds1.intersection(component)]
    qualifying = [component for component in connecting if len(seeds0.intersection(component)) >= math.ceil(TAIL_COMPONENT_FRACTION * len(seeds0)) and len(seeds1.intersection(component)) >= math.ceil(TAIL_COMPONENT_FRACTION * len(seeds1))]
    if len(qualifying) != 1:
        return _failure(original, "V32_ELECTRODE_COMPONENT", graph_vertices=len(indices), graph_edges=len(edges))
    active = set(qualifying[0])
    boundary0, boundary1 = seeds0 & active, seeds1 & active
    phi, iterations = _pcg(len(indices), edges, boundary0, boundary1, active)
    if phi is None:
        return _failure(original, "V32_SOLVER_NONCONVERGENCE", graph_vertices=len(indices), graph_edges=len(edges), solver_iterations=iterations)
    oriented = _orient_normals(source, indices, adjacency, qualifying, boundary0)
    progress = _normal_progress(oriented, candidate.axis, reference, theta)
    current_edges: list[dict[str, Any]] = []
    for i, j, conductance in edges:
        if i not in active or j not in active:
            continue
        delta_q40 = abs(_round_even(phi[j] * Q40) - _round_even(phi[i] * Q40))
        if delta_q40 < CURRENT_ZERO_Q40:
            continue
        delta = source.positions[indices[j]] - source.positions[indices[i]]
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            continue
        if (float(phi[i]), i) > (float(phi[j]), j):
            i, j = j, i
        z_q20 = _round_even(0.5 * (float(z_s[i]) + float(z_s[j])) * Q20)
        current_edges.append({"i": i, "j": j, "current": conductance * abs(float(phi[j] - phi[i])), "z_q20": z_q20, "z": z_q20 / Q20, "axial_fraction": abs(float(np.dot(delta, candidate.axis))) / length})
    current_edges.sort(key=lambda edge: (int(edge["z_q20"]), int(edge["i"]), int(edge["j"])))
    event_values = sorted(set(int(edge["z_q20"]) for edge in current_edges if -_round_even(AUDIT_HALF_S * Q20) <= int(edge["z_q20"]) <= _round_even(AUDIT_HALF_S * Q20)))
    if len(event_values) > MAX_AXIAL_EVENTS:
        raise AVHFMContractError("V32_GRAPH_CAP_BREACH", "axial events")
    if not event_values:
        return _failure(original, "V32_NO_CURRENT_EVENTS", graph_vertices=len(indices), graph_edges=len(edges), solver_iterations=iterations)
    cells = _cells_q21([value / Q20 for value in event_values])
    events: list[dict[str, Any]] = []
    if len(cells) != len(event_values):
        raise AVHFMContractError("V32_INVALID_NUMERIC_INPUT", "event/cell mismatch")
    edge_coordinates = [int(edge["z_q20"]) for edge in current_edges]
    window_half_q20 = _round_even(WINDOW_HALF_S * Q20)
    for value, cell in zip(event_values, cells):
        left = bisect_left(edge_coordinates, value - window_half_q20)
        right = bisect_right(edge_coordinates, value + window_half_q20)
        window = current_edges[left:right]
        stage, reason, leakage = _local_stage(
            window,
            len(indices),
            phi,
            progress,
            source.stable_keys[indices],
            boundary0,
            boundary1,
            tail_bin0,
            tail_bin1,
            math.degrees(theta),
        )
        events.append({"z_q20": value, "cell_q21": list(cell), "stage": stage, "reason": reason, "axial_leakage_q24": _round_even(leakage * Q24)})
    measures: dict[str, int] = {}
    first_failure = None
    minimum = _round_even(MEASURE_MINIMUM_S * Q21)
    for stage_index, stage_name in enumerate(STAGES, start=1):
        intervals = [tuple(event["cell_q21"]) for event in events if int(event["stage"]) >= stage_index]
        measure = sum(b - a for a, b in _merged(intervals))
        measures[stage_name] = measure
        if measure < minimum and first_failure is None:
            first_failure = {
                "CURRENT": "V32_CURRENT_MEASURE",
                "AXIAL_LEAKAGE": "V32_AXIAL_LEAKAGE_MEASURE",
                "TAIL_COVERAGE": "V32_TAIL_COVERAGE_MEASURE",
                "POTENTIAL_COVERAGE": "V32_POTENTIAL_COVERAGE_MEASURE",
                "ANGULAR_COVERAGE": "V32_ANGULAR_COVERAGE_MEASURE",
                "NORMAL_TURN": "V32_NORMAL_TURN_MEASURE",
                "OFF_ARC_CURRENT": "V32_OFF_ARC_CURRENT_MEASURE",
            }[stage_name]
    accepted_intervals = [tuple(event["cell_q21"]) for event in events if event["reason"] == "V32_ACCEPT"]
    merged = _merged(accepted_intervals)
    total, longest, gap, endpoint_left, endpoint_right = _interval_metrics(merged)
    reason = first_failure
    if reason is None and gap > _round_even(MAX_GAP_S * Q21):
        reason = "V32_AXIAL_MAX_GAP"
    if reason is None and longest < _round_even(LONGEST_RUN_MINIMUM_S * Q21):
        reason = "V32_AXIAL_LONGEST_RUN"
    if reason is None and min(endpoint_left, endpoint_right) < _round_even(ENDPOINT_MEASURE_MINIMUM_S * Q21):
        reason = "V32_AXIAL_ENDPOINT_MEASURE"
    accepted = reason is None
    reason = "V32_ACCEPT" if accepted else str(reason)
    accepted_witnesses = [(a, b, "V32_ACCEPT", 1) for a, b in merged]
    rejected_witnesses: list[tuple[int, int, str, int]] = []
    for local_reason in sorted({str(event["reason"]) for event in events if event["reason"] != "V32_ACCEPT"}, key=lambda value: TOKEN_CODE.get(value, 255)):
        local_intervals = _merged([tuple(event["cell_q21"]) for event in events if event["reason"] == local_reason])
        rejected_witnesses.extend((a, b, local_reason, 0) for a, b in local_intervals)
    accepted_ranked = sorted(accepted_witnesses, key=lambda value: (-(value[1] - value[0]), value[0]))
    rejected_ranked = sorted(rejected_witnesses, key=lambda value: (-(value[1] - value[0]), value[0], TOKEN_CODE.get(value[2], 255)))
    witnesses = accepted_ranked[:8] + rejected_ranked[:8]
    reason_counts = {local_reason: sum(event["reason"] == local_reason for event in events) for local_reason in sorted({str(event["reason"]) for event in events}, key=lambda value: TOKEN_CODE.get(value, 255))}
    decision = {
        **_base_decision(original),
        "action": ACTION_RESCUE if accepted else ACTION_NONE,
        "reason": reason,
        "accepted_length_q21": total,
        "longest_run_q21": longest,
        "maximum_gap_q21": gap,
        "endpoint_left_q21": endpoint_left,
        "endpoint_right_q21": endpoint_right,
        "graph_vertices": len(indices),
        "graph_edges": len(edges),
        "event_count": len(events),
        "solver_iterations": iterations,
        "theta_q24": _round_even(theta * Q24),
        "total_current_q24": _round_even(_pairwise_sum([edge["current"] for edge in current_edges]) * Q24),
        "stage_measures_q21": measures,
        "worst_accepted_axial_leakage_q24": max((int(event["axial_leakage_q24"]) for event in events if event["reason"] == "V32_ACCEPT"), default=0),
        "reason_counts": reason_counts,
        "interval_witnesses": [[a, b, local_reason, accepted_flag] for a, b, local_reason, accepted_flag in witnesses],
        "interval_witnesses_truncated": len(accepted_ranked) > 8 or len(rejected_ranked) > 8,
    }
    visual = {"candidate_id": original.candidate_id, "indices": indices.tolist(), "phi": phi.tolist(), "current_edges": current_edges, "events": events, "intervals": [list(v) for v in merged], "z_s": z_s.tolist(), "boundary0": sorted(boundary0), "boundary1": sorted(boundary1), "measures": measures}
    return decision, visual


def deterministic_total_sort(candidates: Sequence[Candidate], decisions: Sequence[dict[str, Any]]) -> list[tuple[Candidate, dict[str, Any]]]:
    rows = list(zip(candidates, decisions))
    legacy = sorted((row for row in rows if row[1]["action"] == ACTION_LEGACY), key=lambda row: (row[0].frozen_order, row[0].candidate_id))
    rescues = sorted((row for row in rows if row[1]["action"] == ACTION_RESCUE), key=lambda row: (-int(row[1]["accepted_length_q21"]), -int(row[1]["longest_run_q21"]), int(row[1]["maximum_gap_q21"]), -int(row[0].frozen_order), str(row[1]["candidate_sha256"])))
    return legacy + rescues


def build_fixed_cache(decisions: Sequence[dict[str, Any]], *, source_count: int = 0, source_sha256: str = "0" * 64, candidate_sha256: str = "0" * 64, parent_sha256: str = "0" * 64, lock_sha256: str = "0" * 64, result_sha256: str = "0" * 64) -> bytes:
    header = bytearray(CACHE_HEADER_BYTES)
    header[:8] = b"V32AVHF1"
    struct.pack_into("<IIIII", header, 8, SCHEMA_VERSION, 0x01020304, CACHE_CANDIDATE_BYTES, int(source_count), len(decisions))
    for offset, digest in zip((32, 64, 96, 128, 160), (source_sha256, candidate_sha256, parent_sha256, lock_sha256, result_sha256)):
        header[offset:offset + 32] = bytes.fromhex(digest)
    records = bytearray()
    for decision in sorted(decisions, key=lambda value: (int(value["frozen_v16_order"]), int(value["candidate_id"]))):
        record = bytearray(CACHE_CANDIDATE_BYTES)
        record[:32] = bytes.fromhex(str(decision["candidate_sha256"]))
        witnesses = list(decision.get("interval_witnesses", []))[:16]
        action_code = {ACTION_NONE: 0, ACTION_RESCUE: 1, ACTION_LEGACY: 2}.get(str(decision.get("action")), 255)
        struct.pack_into("<Q", record, 32, int(decision["candidate_id"]))
        struct.pack_into("<iIIIIIIIII", record, 40, int(decision["frozen_v16_order"]), TOKEN_CODE.get(str(decision.get("reason")), 255), action_code, 0 if decision.get("nms_reason") == "ORDINARY_NMS_SUPPRESSED" else 1, int(decision.get("graph_vertices", 0)), int(decision.get("graph_edges", 0)), int(decision.get("event_count", 0)), int(decision.get("solver_iterations", 0)), 1 if decision.get("interval_witnesses_truncated") else 0, len(witnesses))
        geometry = [float(value) for key in ("anchor_xyz", "axis_xyz", "line_start_xyz", "line_end_xyz") for value in decision.get(key, (0.0, 0.0, 0.0))]
        struct.pack_into("<12d", record, 80, *geometry)
        struct.pack_into("<dqq", record, 176, float(decision.get("scale", 0.0)), int(decision.get("theta_q24", 0)), int(decision.get("total_current_q24", 0)))
        measures = decision.get("stage_measures_q21", {})
        struct.pack_into("<7q", record, 200, *(int(measures.get(stage, 0)) for stage in STAGES))
        struct.pack_into("<6q", record, 256, int(decision.get("accepted_length_q21", 0)), int(decision.get("longest_run_q21", 0)), int(decision.get("maximum_gap_q21", 0)), int(decision.get("endpoint_left_q21", 0)), int(decision.get("endpoint_right_q21", 0)), int(decision.get("worst_accepted_axial_leakage_q24", 0)))
        reason_counts = decision.get("reason_counts", {})
        for index, token in enumerate(TOKENS):
            struct.pack_into("<I", record, 304 + 4 * index, int(reason_counts.get(token, 0)))
        for index, witness in enumerate(witnesses):
            left, right, local_reason, accepted_flag = witness
            struct.pack_into("<qqII", record, 400 + 24 * index, int(left), int(right), TOKEN_CODE.get(str(local_reason), 255), int(accepted_flag))
        semantic = hashlib.sha256(bytes(record[:960])).digest()
        record[960:992] = semantic
        records.extend(record)
    cache = bytes(header + records)
    if len(cache) > MAX_CACHE_BYTES:
        raise AVHFMContractError("V32_CANDIDATE_CAP_BREACH", "cache")
    return cache


def select_unified_v32(raw_source: dict[str, Any], raw_candidates: Sequence[dict[str, Any]], *, worker_count: int = 1) -> dict[str, Any]:
    if len(raw_source.get("positions", [])) > MAX_SOURCE_SAMPLES:
        raise AVHFMContractError("V32_SOURCE_CAP_BREACH")
    if len(raw_candidates) > MAX_CANDIDATES:
        raise AVHFMContractError("V32_CANDIDATE_CAP_BREACH")
    try:
        source = parse_source(raw_source)
        candidates = parse_candidates(raw_candidates)
    except Exception as exc:
        if str(getattr(exc, "code", "")).startswith("PROHIBITED_"):
            raise AVHFMContractError("V32_FORBIDDEN_INPUT", str(exc)) from exc
        raise AVHFMContractError("V32_INVALID_NUMERIC_INPUT", str(exc)) from exc
    workers = int(worker_count)
    physical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if workers < 1 or workers > min(MAX_WORKERS, max(1, physical), max(1, len(candidates))):
        raise AVHFMContractError("V32_INVALID_NUMERIC_INPUT", "worker_count")
    if workers == 1:
        analyzed = [_candidate_decision(candidate, source) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v32-avhfm") as executor:
            analyzed = list(executor.map(lambda candidate: _candidate_decision(candidate, source), candidates))
    decisions = [value[0] for value in analyzed]
    visuals = [value[1] for value in analyzed]
    ordered = deterministic_total_sort(candidates, decisions)
    accepted, suppressed = ordinary_nms(ordered)
    for decision in decisions:
        if int(decision["candidate_id"]) in suppressed:
            decision["nms_reason"] = "ORDINARY_NMS_SUPPRESSED"
    final_lines = [{"candidate_id": candidate.candidate_id, "line_start": candidate.line_start.tolist(), "line_end": candidate.line_end.tolist(), "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v32_avhfm_rescue", "v32_action": decision["action"]} for candidate, decision in accepted]
    cache = build_fixed_cache(decisions, source_count=len(source.positions))
    return {"schema_version": SCHEMA_VERSION, "kind": "sherman_qc_f1_v32_avhfm_research_result", "profile": PROFILE, "primitive": PRIMITIVE, "research_only": True, "source_only": True, "rescue_only": True, "expected_count_used": False, "part_identity_used": False, "filename_identity_used": False, "labels_used": False, "cad_used": False, "decisions": decisions, "final_lines": final_lines, "pre_nms_rescue_count": sum(v["action"] == ACTION_RESCUE for v in decisions), "rescue_count": sum(v["source_lane"] == "v32_avhfm_rescue" for v in final_lines), "ordinary_nms_call_count": 1, "candidate_added_after_nms": False, "cache_bytes": len(cache), "cache_sha256": hashlib.sha256(cache).hexdigest(), "_ephemeral": {"candidate_visuals": visuals, "fixed_cache": cache, "source": source, "candidates": candidates, "nms_suppressed": sorted(suppressed)}}
