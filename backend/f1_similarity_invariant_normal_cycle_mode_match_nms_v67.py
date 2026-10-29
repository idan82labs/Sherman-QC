"""V67: similarity-invariant normal-cycle mode-match final deduplicator.

Research-only implementation of the frozen NCMM_NMS_V1 ruling.  The module is
removal-only: it consumes the complete V52 pre-NMS candidate multiset and the
bounded source mesh, then retains original candidate dictionaries byte for
byte.  Missing or ambiguous source evidence always keeps candidates.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROFILE = "V67_SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1"
ABBREVIATION = "NCMM_NMS_V1"
MECHANISM_LOCK = "V67_NCMM_NMS_V1_MECHANISM_LOCKED"
FIXTURE_LOCK = "V67_NCMM_NMS_V1_FIXTURES_LOCKED_WITH_GEOMETRIC_ERRATUM_E1"
Q_BITS = 40
Q_ONE = 1 << Q_BITS

MAX_SOURCE_VERTICES = 200_000
MAX_PRIMARY_TRIANGLES = 400_000
MAX_VARIANT_TRIANGLES = 500_000
MAX_GRAPH_EDGES = 600_000
MAX_HINGE_ATOMS = 500_000
MAX_CANDIDATES = 96
MAX_TOTAL_PAIRS = 4_560
MAX_ELIGIBLE_PAIRS = 2_048
MAX_CLIP_VISITS = 8_000_000
MAX_AUDIT_BYTES = 4 * 1024 * 1024
MAX_PRIMARY_SECONDS = 30.0

ANGLE_LIMIT = math.pi / 24.0
DIRECTION_LIMIT = math.pi / 18.0
OVERLAP_MIN = 5.0 / 8.0
SEPARATION_MIN = 1.0 / 4096.0
SEPARATION_MAX = 1.0 / 16.0
SEPARATION_VARIATION_MIN = 3.0 / 4.0
FRAME_E_MIN = 7.0 / 8.0
FRAME_E_MAX = 9.0 / 8.0
FRAME_N_MAX = 1.0 / 8.0
TURN_MIN = 13.0 / 64.0
AXIAL_EFFECTIVE_MIN = 4.0
BRIDGE_MASS_MIN = 1.0 / 2.0
COMPONENT_DOMINANCE_MIN = 3.0 / 4.0
SIGN_COHERENCE_MIN = 3.0 / 4.0
BOUNDARY_RATIO_MAX = 1.0 / 4.0


class NCMMInputError(ValueError):
    """Invalid source or candidate contract."""


class NCMMResourceAbort(RuntimeError):
    """Atomic fail-closed resource or serializer abort."""


def q40_int(value: float) -> int:
    if not math.isfinite(float(value)):
        raise NCMMInputError("NUMERIC_INDETERMINATE")
    return int(round(float(value) * Q_ONE))


def q40(value: float) -> float:
    return q40_int(value) / Q_ONE


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _encode_audit(value: Any) -> bytes:
    try:
        return canonical_bytes(value)
    except (TypeError, ValueError, OverflowError, NCMMInputError) as error:
        raise NCMMResourceAbort("SERIALIZATION_ABORT") from error


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _clean_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise NCMMInputError("NUMERIC_INDETERMINATE")
    return 0.0 if result == 0.0 else result


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise NCMMInputError("NUMERIC_INDETERMINATE")
    return np.asarray(vector, dtype=np.float64) / norm


def _candidate_hash(row: Mapping[str, Any]) -> str:
    value = str(row.get("route_hash") or row.get("opaque_candidate_hash") or row.get("bend_id") or "")
    if value:
        return value
    payload = {"line_start": row.get("line_start"), "line_end": row.get("line_end")}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()[:32]


@dataclass(frozen=True)
class Candidate:
    candidate_hash: str
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    ordinal: int


@dataclass(frozen=True)
class Atom:
    start: np.ndarray
    end: np.ndarray
    unsigned_turn: float
    signed_turn: float
    component: str
    kind: str = "HINGE"


@dataclass
class PairEvaluation:
    first_hash: str
    second_hash: str
    state: str
    reason: str
    metrics: dict[str, Any]
    axial_bins: list[int]
    transverse_bins: list[int]
    modes: list[dict[str, Any]]
    confidence_q40: tuple[int, ...] | None
    representative_errors_q40: dict[str, int]
    proposed_winner: str | None
    proposed_loser: str | None


@dataclass
class MeshMeasure:
    hinge_atoms: list[Atom]
    veto_atoms: list[Atom]
    face_count: int
    graph_edge_count: int
    orientable_component_count: int
    nonorientable_component_count: int


def _normal_plane_key(normal: np.ndarray, point: np.ndarray, scale: float) -> tuple[int, ...]:
    n = np.asarray(normal, dtype=np.float64).copy()
    for value in n:
        if abs(float(value)) > 1e-15:
            if value < 0.0:
                n *= -1.0
            break
    offset = -float(np.dot(n, point)) / max(scale, 1e-300)
    return tuple(q40_int(value) for value in (*n.tolist(), offset))


def _canonical_line_key(start: np.ndarray, end: np.ndarray) -> tuple[int, int, int]:
    direction = _unit(end - start)
    for value in direction:
        if abs(float(value)) > 1e-15:
            if value < 0.0:
                direction *= -1.0
            break
    return tuple(q40_int(value) for value in direction.tolist())


def _point_key(point: np.ndarray, scale: float, origin: np.ndarray) -> tuple[int, int, int]:
    normalized = (np.asarray(point, dtype=np.float64) - origin) / max(scale, 1e-300)
    return tuple(q40_int(value) for value in normalized.tolist())


def _face_edge_direction(face: np.ndarray, a: int, b: int) -> int:
    values = [int(value) for value in face]
    for index in range(3):
        x, y = values[index], values[(index + 1) % 3]
        if x == a and y == b:
            return 1
        if x == b and y == a:
            return -1
    raise NCMMInputError("INVALID_TRIANGLE_EDGE")


def _merge_atoms(raw: Sequence[Atom], scale: float, origin: np.ndarray) -> list[Atom]:
    """Merge straight, unbranched fragments with identical support semantics."""
    grouped: dict[tuple[Any, ...], list[Atom]] = defaultdict(list)
    for atom in raw:
        key = (
            atom.kind,
            atom.component,
            _canonical_line_key(atom.start, atom.end),
            q40_int(atom.unsigned_turn),
            q40_int(atom.signed_turn),
        )
        grouped[key].append(atom)
    output: list[Atom] = []
    for group_key in sorted(grouped, key=repr):
        rows = grouped[group_key]
        endpoints: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, atom in enumerate(rows):
            endpoints[_point_key(atom.start, scale, origin)].append(index)
            endpoints[_point_key(atom.end, scale, origin)].append(index)
        used: set[int] = set()

        def walk(seed: int, start_key: tuple[int, int, int]) -> Atom:
            chain = [seed]
            used.add(seed)
            current_key = start_key
            start_point = rows[seed].start if _point_key(rows[seed].start, scale, origin) == current_key else rows[seed].end
            atom = rows[seed]
            other = atom.end if _point_key(atom.start, scale, origin) == current_key else atom.start
            current_key = _point_key(other, scale, origin)
            end_point = other
            while len(endpoints[current_key]) == 2:
                candidates = [value for value in endpoints[current_key] if value not in used]
                if len(candidates) != 1:
                    break
                nxt = candidates[0]
                used.add(nxt)
                chain.append(nxt)
                nxt_atom = rows[nxt]
                other = nxt_atom.end if _point_key(nxt_atom.start, scale, origin) == current_key else nxt_atom.start
                current_key = _point_key(other, scale, origin)
                end_point = other
            base = rows[seed]
            return Atom(np.asarray(start_point), np.asarray(end_point), base.unsigned_turn, base.signed_turn, base.component, base.kind)

        starts = sorted((key for key, values in endpoints.items() if len(values) == 1))
        for start_key in starts:
            for seed in sorted(endpoints[start_key]):
                if seed not in used:
                    output.append(walk(seed, start_key))
        for seed in range(len(rows)):
            if seed not in used:
                # Closed loops or branch-adjacent fragments stay unmerged.
                used.add(seed)
                output.append(rows[seed])
    output.sort(key=lambda atom: (atom.kind, atom.component, _point_key(atom.start, scale, origin), _point_key(atom.end, scale, origin)))
    return output


def build_mesh_measure(vertices: np.ndarray, faces: np.ndarray, *, variant: bool = False) -> MeshMeasure:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise NCMMInputError("INVALID_SOURCE_VERTICES")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise NCMMInputError("INVALID_SOURCE_TRIANGLES")
    if len(vertices) > MAX_SOURCE_VERTICES:
        raise NCMMResourceAbort("SOURCE_VERTEX_CAP")
    triangle_cap = MAX_VARIANT_TRIANGLES if variant else MAX_PRIMARY_TRIANGLES
    if len(faces) > triangle_cap:
        raise NCMMResourceAbort("TRIANGLE_CAP")
    if len(faces) == 0:
        return MeshMeasure([], [], 0, 0, 0, 0)
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise NCMMInputError("INVALID_FACE_INDEX")

    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    scale = float(np.linalg.norm(bbox_max - bbox_min))
    if scale <= 0.0:
        raise NCMMInputError("DEGENERATE_SOURCE")
    raw_normals = np.zeros((len(faces), 3), dtype=np.float64)
    valid = np.ones(len(faces), dtype=bool)
    for index, face in enumerate(faces):
        a, b, c = vertices[face]
        cross = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(cross))
        if not math.isfinite(norm) or norm <= 1e-15 * scale * scale:
            valid[index] = False
        else:
            raw_normals[index] = cross / norm

    incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        if not valid[face_index]:
            continue
        a, b, c = (int(value) for value in face)
        incidence[tuple(sorted((a, b)))].append(face_index)
        incidence[tuple(sorted((b, c)))].append(face_index)
        incidence[tuple(sorted((c, a)))].append(face_index)
    if len(incidence) > MAX_GRAPH_EDGES:
        raise NCMMResourceAbort("INCIDENCE_EDGE_CAP")

    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (a, b), owners in incidence.items():
        if len(owners) != 2:
            continue
        first, second = owners
        d1 = _face_edge_direction(faces[first], a, b)
        d2 = _face_edge_direction(faces[second], a, b)
        relation = -1 if d1 == d2 else 1
        adjacency[first].append((second, relation))
        adjacency[second].append((first, relation))

    flips = np.zeros(len(faces), dtype=np.int8)
    component = np.full(len(faces), -1, dtype=np.int64)
    orientable: dict[int, bool] = {}
    component_members: dict[int, list[int]] = defaultdict(list)
    component_id = 0
    face_seeds = sorted((index for index in range(len(faces)) if valid[index]), key=lambda idx: sha256_json(sorted(_point_key(vertices[int(v)], scale, bbox_min) for v in faces[idx])))
    for seed in face_seeds:
        if component[seed] >= 0:
            continue
        flips[seed] = 1
        component[seed] = component_id
        queue = deque([seed])
        okay = True
        while queue:
            current = queue.popleft()
            component_members[component_id].append(current)
            for neighbor, relation in sorted(adjacency.get(current, ()), key=lambda item: item[0]):
                proposed = int(flips[current]) * relation
                if component[neighbor] < 0:
                    component[neighbor] = component_id
                    flips[neighbor] = proposed
                    queue.append(neighbor)
                elif int(flips[neighbor]) != proposed:
                    okay = False
        orientable[component_id] = okay
        component_id += 1

    raw_hinges: list[Atom] = []
    raw_veto: list[Atom] = []
    plane_keys = {}
    oriented_normals = raw_normals * flips[:, None]
    for face_index in range(len(faces)):
        if valid[face_index]:
            plane_keys[face_index] = _normal_plane_key(oriented_normals[face_index], vertices[int(faces[face_index][0])], scale)

    for edge, owners in sorted(incidence.items()):
        a, b = edge
        start, end = vertices[a], vertices[b]
        if float(np.linalg.norm(end - start)) <= 0.0:
            continue
        if len(owners) != 2:
            kind = "BOUNDARY" if len(owners) == 1 else "NONMANIFOLD"
            raw_veto.append(Atom(start, end, 0.0, 0.0, f"V{kind}", kind))
            continue
        first, second = owners
        cid = int(component[first])
        if cid < 0 or int(component[second]) != cid or not orientable.get(cid, False):
            raw_veto.append(Atom(start, end, 0.0, 0.0, "VNONORIENTABLE", "NONMANIFOLD"))
            continue
        n1, n2 = oriented_normals[first], oriented_normals[second]
        if plane_keys[first] > plane_keys[second]:
            first, second = second, first
            n1, n2 = n2, n1
        direction_sign = _face_edge_direction(faces[first] if flips[first] > 0 else faces[first][::-1], a, b)
        edge_direction = _unit((end - start) * direction_sign)
        unsigned = math.atan2(float(np.linalg.norm(np.cross(n1, n2))), float(np.dot(n1, n2)))
        signed = math.atan2(float(np.dot(edge_direction, np.cross(n1, n2))), float(np.dot(n1, n2)))
        if q40_int(unsigned) == 0:
            continue
        support = sha256_json(sorted((plane_keys[first], plane_keys[second])))[:24]
        raw_hinges.append(Atom(start, end, unsigned, signed, f"C{cid:06d}:{support}", "HINGE"))

    hinges = _merge_atoms(raw_hinges, scale, bbox_min)
    veto = _merge_atoms(raw_veto, scale, bbox_min)
    if len(hinges) > MAX_HINGE_ATOMS:
        raise NCMMResourceAbort("HINGE_ATOM_CAP")
    return MeshMeasure(
        hinges,
        veto,
        int(valid.sum()),
        len(incidence),
        sum(bool(value) for value in orientable.values()),
        sum(not bool(value) for value in orientable.values()),
    )


def _candidates(rows: Sequence[Mapping[str, Any]]) -> list[Candidate]:
    if len(rows) > MAX_CANDIDATES:
        raise NCMMResourceAbort("CANDIDATE_CAP")
    output = []
    seen = set()
    for ordinal, row in enumerate(rows):
        start = np.asarray(row.get("line_start"), dtype=np.float64)
        end = np.asarray(row.get("line_end"), dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
            raise NCMMInputError("INVALID_CANDIDATE")
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise NCMMInputError("INVALID_CANDIDATE")
        candidate_hash = _candidate_hash(row)
        if candidate_hash in seen:
            raise NCMMInputError("DUPLICATE_CANDIDATE_HASH")
        seen.add(candidate_hash)
        output.append(Candidate(candidate_hash, start, end, delta / length, length, ordinal))
    output.sort(key=lambda value: value.candidate_hash)
    return output


def _point_on_line(candidate: Candidate, axis: np.ndarray, z: float) -> np.ndarray:
    denominator = float(np.dot(axis, candidate.direction))
    if abs(denominator) <= 1e-15:
        raise NCMMInputError("NUMERIC_INDETERMINATE")
    return candidate.start + candidate.direction * ((z - float(np.dot(axis, candidate.start))) / denominator)


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def _exact_coincident(first: Candidate, second: Candidate) -> bool:
    angle = math.acos(float(np.clip(abs(np.dot(first.direction, second.direction)), -1.0, 1.0)))
    scale = max(first.length, second.length)
    hausdorff = max(
        _segment_distance(first.start, second.start, second.end),
        _segment_distance(first.end, second.start, second.end),
        _segment_distance(second.start, first.start, first.end),
        _segment_distance(second.end, first.start, first.end),
    ) / scale
    axis = first.direction
    intervals = []
    for candidate in (first, second):
        values = sorted((float(np.dot(axis, candidate.start)), float(np.dot(axis, candidate.end))))
        intervals.append(values)
    overlap = max(0.0, min(intervals[0][1], intervals[1][1]) - max(intervals[0][0], intervals[1][0]))
    ratio = overlap / min(first.length, second.length)
    return q40_int(hausdorff) == 0 and q40_int(angle) == 0 and q40(ratio) >= 15.0 / 16.0


def _pair_frame(first: Candidate, second: Candidate) -> dict[str, Any] | None:
    uj = second.direction if float(np.dot(first.direction, second.direction)) >= 0.0 else -second.direction
    angle = math.acos(float(np.clip(abs(np.dot(first.direction, uj)), -1.0, 1.0)))
    summed = first.direction + uj
    if float(np.linalg.norm(summed)) <= 1e-15:
        return None
    axis = _unit(summed)
    intervals = []
    for candidate in (first, second):
        values = sorted((float(np.dot(axis, candidate.start)), float(np.dot(axis, candidate.end))))
        intervals.append(values)
    z0 = max(intervals[0][0], intervals[1][0])
    z1 = min(intervals[0][1], intervals[1][1])
    overlap = z1 - z0
    if overlap <= 0.0:
        return None
    points_i = [_point_on_line(first, axis, z) for z in (z0, 0.5 * (z0 + z1), z1)]
    points_j = [_point_on_line(second, axis, z) for z in (z0, 0.5 * (z0 + z1), z1)]
    projector = np.eye(3) - np.outer(axis, axis)
    offsets = [projector @ (b - a) for a, b in zip(points_i, points_j)]
    distances = [float(np.linalg.norm(value)) for value in offsets]
    d0, d, d1 = distances
    if min(d0, d, d1) <= 0.0:
        return None
    direction_change = float(np.dot(offsets[0], offsets[2]) / (d0 * d1))
    e = offsets[1] / d
    n = np.cross(axis, e)
    endpoint_e = [float(np.dot(points_j[index] - points_i[index], e) / d) for index in (0, 2)]
    endpoint_n = [abs(float(np.dot(points_j[index] - points_i[index], n) / d)) for index in (0, 2)]
    metrics = {
        "angle": angle,
        "overlap_ratio": overlap / min(first.length, second.length),
        "separation_ratio": d / overlap,
        "variation_ratio": min(distances) / max(distances),
        "direction_cos": direction_change,
        "endpoint_e": endpoint_e,
        "endpoint_n": endpoint_n,
    }
    if not (
        q40(angle) <= q40(ANGLE_LIMIT)
        and q40(metrics["overlap_ratio"]) >= q40(OVERLAP_MIN)
        and q40(metrics["separation_ratio"]) >= q40(SEPARATION_MIN)
        and q40(metrics["separation_ratio"]) <= q40(SEPARATION_MAX)
        and q40(metrics["variation_ratio"]) >= q40(SEPARATION_VARIATION_MIN)
        and q40(direction_change) >= q40(math.cos(DIRECTION_LIMIT))
        and all(q40(value) >= q40(FRAME_E_MIN) and q40(value) <= q40(FRAME_E_MAX) for value in endpoint_e)
        and all(q40(value) <= q40(FRAME_N_MAX) for value in endpoint_n)
    ):
        return None
    return {"axis": axis, "e": e, "n": n, "z0": z0, "z1": z1, "length": overlap, "distance": d, "metrics": metrics}


def _clip_linear_interval(values: Sequence[tuple[float, float, float, float]], low: float = 0.0, high: float = 1.0) -> tuple[float, float] | None:
    lo, hi = low, high
    for base, delta, limit_low, limit_high in values:
        if abs(delta) <= 1e-30:
            if base < limit_low or base > limit_high:
                return None
            continue
        first = (limit_low - base) / delta
        second = (limit_high - base) / delta
        enter, leave = min(first, second), max(first, second)
        lo, hi = max(lo, enter), min(hi, leave)
        if lo >= hi:
            return None
    return max(0.0, lo), min(1.0, hi)


def _profile(measure: MeshMeasure, frame: Mapping[str, Any], visit_counter: list[int]) -> dict[str, Any]:
    axis, e, n = frame["axis"], frame["e"], frame["n"]
    z0, length, distance = frame["z0"], frame["length"], frame["distance"]
    radius = max(distance / 2.0, length / 128.0)
    axial_contrib: list[list[float]] = [[] for _ in range(16)]
    transverse_contrib: list[list[float]] = [[] for _ in range(32)]
    signed_by_component: dict[str, list[float]] = defaultdict(list)
    unsigned_by_component: dict[str, list[float]] = defaultdict(list)
    centroid_contrib: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = defaultdict(list)

    def coordinates(point: np.ndarray) -> tuple[float, float, float]:
        z = float(np.dot(axis, point))
        tau = (z - z0) / length
        pi = frame["first_point"](z)
        return tau, float(np.dot(point - pi, e) / distance), float(np.dot(point - pi, n))

    for atom in measure.hinge_atoms:
        visit_counter[0] += 1
        if visit_counter[0] > MAX_CLIP_VISITS:
            raise NCMMResourceAbort("CLIP_VISIT_CAP")
        c0, c1 = coordinates(atom.start), coordinates(atom.end)
        delta = tuple(c1[index] - c0[index] for index in range(3))
        clipped = _clip_linear_interval(((c0[0], delta[0], 0.0, 1.0), (c0[1], delta[1], -0.5, 1.5), (c0[2], delta[2], -radius, radius)))
        if clipped is None:
            continue
        lo, hi = clipped
        cuts = [lo, hi]
        for boundary in (index / 16.0 for index in range(1, 16)):
            if abs(delta[0]) > 1e-30:
                value = (boundary - c0[0]) / delta[0]
                if lo < value < hi:
                    cuts.append(value)
        for boundary in (-0.5 + index / 16.0 for index in range(1, 32)):
            if abs(delta[1]) > 1e-30:
                value = (boundary - c0[1]) / delta[1]
                if lo < value < hi:
                    cuts.append(value)
        cuts = sorted(set(q40(value) for value in cuts))
        for left, right in zip(cuts, cuts[1:]):
            if right <= left:
                continue
            mid = 0.5 * (left + right)
            tau = c0[0] + mid * delta[0]
            xi = c0[1] + mid * delta[1]
            axial_bin = min(15, max(0, int(math.floor(tau * 16.0))))
            transverse_bin = min(31, max(0, int(math.floor((xi + 0.5) * 16.0))))
            q_left = atom.start + left * (atom.end - atom.start)
            q_right = atom.start + right * (atom.end - atom.start)
            projected = abs(float(np.dot(axis, q_right - q_left))) / length
            unsigned = atom.unsigned_turn * projected
            signed = atom.signed_turn * projected
            axial_contrib[axial_bin].append(unsigned)
            transverse_contrib[transverse_bin].append(unsigned)
            mesh_component = atom.component.split(":", 1)[0]
            unsigned_by_component[mesh_component].append(unsigned)
            signed_by_component[mesh_component].append(signed)
            centroid_contrib[(mesh_component, transverse_bin)].append((unsigned, 0.5 * (q_left + q_right)))

    axial = [q40_int(math.fsum(sorted(values))) for values in axial_contrib]
    transverse = [q40_int(math.fsum(sorted(values))) for values in transverse_contrib]
    total_q = sum(transverse)
    component_q = {key: sum(q40_int(value) for value in values) for key, values in unsigned_by_component.items()}
    signed_component_q = {key: sum(q40_int(value) for value in values) for key, values in signed_by_component.items()}

    boundary_length = 0.0
    for atom in measure.veto_atoms:
        visit_counter[0] += 1
        if visit_counter[0] > MAX_CLIP_VISITS:
            raise NCMMResourceAbort("CLIP_VISIT_CAP")
        c0, c1 = coordinates(atom.start), coordinates(atom.end)
        delta = tuple(c1[index] - c0[index] for index in range(3))
        clipped = _clip_linear_interval(((c0[0], delta[0], 0.0, 1.0), (c0[1], delta[1], 0.25, 0.75), (c0[2], delta[2], -radius, radius)))
        if clipped is not None:
            left, right = clipped
            boundary_length += abs(float(np.dot(axis, (right - left) * (atom.end - atom.start)))) / length
    boundary_q = q40_int(boundary_length)
    return {
        "axial_q40": axial,
        "transverse_q40": transverse,
        "total_q40": total_q,
        "component_q40": component_q,
        "signed_component_q40": signed_component_q,
        "boundary_q40": boundary_q,
        "centroid_contrib": centroid_contrib,
    }


def _persistent_modes(transverse_q: Sequence[int], total_q: int) -> tuple[list[dict[str, Any]], list[int]]:
    h = list(transverse_q)
    g = []
    for index in range(32):
        value = (h[index - 2] if index >= 2 else 0) + 2 * (h[index - 1] if index >= 1 else 0) + 3 * h[index] + 2 * (h[index + 1] if index + 1 < 32 else 0) + (h[index + 2] if index + 2 < 32 else 0)
        g.append(q40_int((value / Q_ONE) / 9.0))
    threshold = q40_int((total_q / Q_ONE) / 16.0)
    plateaus = []
    index = 0
    while index < 32:
        end = index
        while end + 1 < 32 and g[end + 1] == g[index]:
            end += 1
        left = g[index - 1] if index > 0 else -1
        right = g[end + 1] if end + 1 < 32 else -1
        if g[index] >= threshold and g[index] > left and g[index] > right:
            plateaus.append({"left": index, "right": end, "height": g[index]})
        index = end + 1
    if not plateaus:
        return [], g
    parent = list(range(len(plateaus)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for pidx in range(len(plateaus) - 1):
        first, second = plateaus[pidx], plateaus[pidx + 1]
        between = g[first["right"] + 1 : second["left"]]
        valley = min(between) if between else min(first["height"], second["height"])
        if 2 * valley > min(first["height"], second["height"]):
            union(pidx, pidx + 1)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pidx, plateau in enumerate(plateaus):
        groups[find(pidx)].append(plateau)
    modes = []
    for group in groups.values():
        left, right = min(value["left"] for value in group), max(value["right"] for value in group)
        weights = g[left : right + 1]
        centers = [-0.5 + (index + 0.5) / 16.0 for index in range(left, right + 1)]
        denominator = sum(weights)
        center = math.fsum(weight * value for weight, value in zip(weights, centers)) / denominator if denominator else 0.0
        modes.append({"left": left, "right": right, "center_q40": q40_int(center), "height_q40": max(weights) if weights else 0})
    modes.sort(key=lambda row: (row["center_q40"], row["left"], row["right"]))
    return modes, g


def _rms_mode_error(candidate: Candidate, axis: np.ndarray, mode_point: np.ndarray, z0: float, z1: float, length: float) -> float:
    projector = np.eye(3) - np.outer(axis, axis)
    values = []
    for z in (z0, 0.5 * (z0 + z1), z1):
        point = _point_on_line(candidate, axis, z)
        delta = projector @ (point - mode_point)
        values.append(float(np.dot(delta, delta)))
    mean_square = (values[0] + 4.0 * values[1] + values[2]) / 6.0
    return math.sqrt(max(0.0, mean_square)) / length


def _evaluate_pair(first: Candidate, second: Candidate, measure: MeshMeasure, visit_counter: list[int]) -> PairEvaluation:
    hashes = sorted((first.candidate_hash, second.candidate_hash))
    if _exact_coincident(first, second):
        winner = first if (-first.length, first.candidate_hash) < (-second.length, second.candidate_hash) else second
        loser = second if winner is first else first
        return PairEvaluation(hashes[0], hashes[1], "EXACT_COINCIDENT", "EXACT_COINCIDENT", {}, [], [], [], (2 * Q_ONE,), {}, winner.candidate_hash, loser.candidate_hash)
    frame = _pair_frame(first, second)
    if frame is None:
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "PAIR_GEOMETRY", {}, [], [], [], None, {}, None, None)
    frame["first_point"] = lambda z: _point_on_line(first, frame["axis"], z)
    profile = _profile(measure, frame, visit_counter)
    total_q = profile["total_q40"]
    metrics = {key + "_q40": q40_int(value) for key, value in frame["metrics"].items() if not isinstance(value, list)}
    metrics["endpoint_e_q40"] = [q40_int(value) for value in frame["metrics"]["endpoint_e"]]
    metrics["endpoint_n_q40"] = [q40_int(value) for value in frame["metrics"]["endpoint_n"]]
    metrics["total_turn_q40"] = total_q
    if total_q <= 0 or total_q < q40_int(TURN_MIN):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "LOW_TURN", metrics, profile["axial_q40"], profile["transverse_q40"], [], None, {}, None, None)
    # N_eff is an axial-profile statistic.  Use the sum of the already
    # quantized axial cells as its numerator as well as its denominator.
    # Mixing the independently quantized transverse total into this ratio
    # makes a perfectly uniform 16-cell profile drift away from exactly 16.
    axial_total_q = sum(profile["axial_q40"])
    axial_sq = sum(value * value for value in profile["axial_q40"])
    neff = (axial_total_q * axial_total_q / axial_sq) if axial_sq else 0.0
    neff_q = q40_int(neff)
    metrics["n_eff_q40"] = neff_q
    if neff_q < q40_int(AXIAL_EFFECTIVE_MIN):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "AXIAL_LOCALIZED", metrics, profile["axial_q40"], profile["transverse_q40"], [], None, {}, None, None)
    bridge_q = sum(profile["transverse_q40"][12:20])
    bridge_ratio_q = q40_int(bridge_q / total_q)
    metrics["bridge_ratio_q40"] = bridge_ratio_q
    if bridge_ratio_q < q40_int(BRIDGE_MASS_MIN):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "NO_BRIDGE", metrics, profile["axial_q40"], profile["transverse_q40"], [], None, {}, None, None)
    modes, smoothed = _persistent_modes(profile["transverse_q40"], total_q)
    if not modes:
        reason = "NO_SIGNIFICANT_MODE"
        return PairEvaluation(hashes[0], hashes[1], "KEEP", reason, metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    if len(modes) != 1:
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "MULTI_MODE", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    mode = modes[0]
    if not (q40_int(0.25) <= mode["center_q40"] <= q40_int(0.75)):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "MODE_OUTSIDE_BRIDGE", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    components = profile["component_q40"]
    if not components:
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "LOW_TURN", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    dominant = min(components, key=lambda key: (-components[key], key))
    # Component dominance is defined over the component partition itself.
    # Its denominator must therefore be the sum of those same quantized
    # component masses, not an independently binned transverse total.
    component_total_q = sum(components.values())
    dominance_q = q40_int(components[dominant] / component_total_q) if component_total_q else 0
    sign_q = q40_int(abs(profile["signed_component_q40"].get(dominant, 0)) / components[dominant]) if components[dominant] else 0
    boundary_q = profile["boundary_q40"]
    metrics.update({"component_dominance_q40": dominance_q, "sign_coherence_q40": sign_q, "boundary_ratio_q40": boundary_q})
    if dominance_q < q40_int(COMPONENT_DOMINANCE_MIN):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "MULTI_COMPONENT", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    if sign_q < q40_int(SIGN_COHERENCE_MIN):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "OPPOSING_TURN", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    if boundary_q >= q40_int(BOUNDARY_RATIO_MAX):
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "BOUNDARY_OR_NONMANIFOLD", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)

    g = smoothed
    maximum = max(g)
    peak_indices = [index for index, value in enumerate(g) if value == maximum]
    peak = peak_indices[len(peak_indices) // 2]
    basin_left = basin_right = peak
    while basin_left > 0 and 2 * g[basin_left - 1] >= maximum:
        basin_left -= 1
    while basin_right + 1 < 32 and 2 * g[basin_right + 1] >= maximum:
        basin_right += 1
    weighted_points = []
    weights = []
    for bin_index in range(basin_left, basin_right + 1):
        for weight, point in profile["centroid_contrib"].get((dominant, bin_index), ()):
            weights.append(weight)
            weighted_points.append(point)
    if not weights or math.fsum(weights) <= 0.0:
        return PairEvaluation(hashes[0], hashes[1], "KEEP", "NUMERIC_INDETERMINATE", metrics, profile["axial_q40"], profile["transverse_q40"], modes, None, {}, None, None)
    # Each coordinate is accumulated in a deterministic, weight-sorted order.
    ordered = sorted(zip(weights, weighted_points), key=lambda item: (q40_int(item[0]), tuple(q40_int(value) for value in item[1])))
    denominator = math.fsum(weight for weight, _point in ordered)
    mode_point = np.asarray(
        [math.fsum(weight * float(point[axis_index]) for weight, point in ordered) / denominator for axis_index in range(3)],
        dtype=np.float64,
    )
    first_error = _rms_mode_error(first, frame["axis"], mode_point, frame["z0"], frame["z1"], frame["length"])
    second_error = _rms_mode_error(second, frame["axis"], mode_point, frame["z0"], frame["z1"], frame["length"])
    errors = {first.candidate_hash: q40_int(first_error), second.candidate_hash: q40_int(second_error)}
    winner = min((first, second), key=lambda candidate: (errors[candidate.candidate_hash], -q40_int(candidate.length / frame["length"]), candidate.candidate_hash))
    loser = second if winner is first else first
    confidence = (bridge_ratio_q, sign_q, dominance_q, neff_q, total_q, -q40_int(frame["distance"] / frame["length"]))
    metrics["mode_center_q40"] = mode["center_q40"]
    metrics["basin"] = [basin_left, basin_right]
    return PairEvaluation(hashes[0], hashes[1], "ONE_FOLD_MODE", "ONE_FOLD_MODE", metrics, profile["axial_q40"], profile["transverse_q40"], modes, confidence, errors, winner.candidate_hash, loser.candidate_hash)


def _pair_audit(pair: PairEvaluation) -> dict[str, Any]:
    payload = {
        "hashes": [pair.first_hash, pair.second_hash],
        "state": pair.state,
        "reason": pair.reason,
        "metrics": pair.metrics,
        "proposed_winner": pair.proposed_winner,
        "proposed_loser": pair.proposed_loser,
    }
    if pair.state in ("ONE_FOLD_MODE",) or pair.reason not in ("PAIR_GEOMETRY", "LOW_TURN"):
        payload.update({"axial_q40": pair.axial_bins, "transverse_q40": pair.transverse_bins, "modes": pair.modes, "representative_errors_q40": pair.representative_errors_q40})
    return payload


def _atomic_noop(rows: Sequence[Mapping[str, Any]], status: str, detail: str) -> dict[str, Any]:
    return {
        "predictions": [deepcopy(dict(row)) for row in rows],
        "ncmm_nms_v1_audit": {
            "profile": PROFILE,
            "mechanism_lock": MECHANISM_LOCK,
            "fixture_lock": FIXTURE_LOCK,
            "status": status,
            "detail": detail,
            "incoming_candidate_count": len(rows),
            "suppressed_hashes": [],
            "survivor_hashes": [_candidate_hash(row) for row in rows],
        },
    }


def apply_ncmm_nms_v1(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    *,
    variant: bool = False,
    started_at: float | None = None,
) -> tuple[dict[str, Any], list[PairEvaluation]]:
    """Apply V67 exactly once as the sole final deduplicator.

    `source` may be a V52-like mapping with `predictions`, or a prediction
    sequence.  No candidate score or provenance field is read by the decision.
    """
    baseline = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
    rows = [deepcopy(dict(row)) for row in (source.get("predictions") or ())] if isinstance(source, Mapping) else [deepcopy(dict(row)) for row in source]
    started = time.perf_counter() if started_at is None else started_at
    if vertices is None or faces is None or len(np.asarray(faces)) == 0:
        noop = _atomic_noop(rows, "POINT_ONLY_KEEP_ALL", "ORIENTATION_UNAVAILABLE")
        baseline.update(noop)
        return baseline, []
    try:
        candidates = _candidates(rows)
        total_pairs = len(candidates) * (len(candidates) - 1) // 2
        if total_pairs > MAX_TOTAL_PAIRS:
            raise NCMMResourceAbort("PAIR_CAP")
        measure = build_mesh_measure(np.asarray(vertices), np.asarray(faces), variant=variant)
        pairs = []
        visit_counter = [0]
        eligible_count = 0
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                pair = _evaluate_pair(first, second, measure, visit_counter)
                if pair.reason != "PAIR_GEOMETRY":
                    eligible_count += 1
                    if eligible_count > MAX_ELIGIBLE_PAIRS:
                        raise NCMMResourceAbort("ELIGIBLE_PAIR_CAP")
                pairs.append(pair)
                if time.perf_counter() - started > MAX_PRIMARY_SECONDS and not variant:
                    raise NCMMResourceAbort("PRIMARY_WALL_TIME_CAP")

        positives = [pair for pair in pairs if pair.state in ("EXACT_COINCIDENT", "ONE_FOLD_MODE")]
        positives.sort(
            key=lambda pair: (
                0 if pair.state == "EXACT_COINCIDENT" else 1,
                tuple(-value for value in (pair.confidence_q40 or ())),
                pair.first_hash,
                pair.second_hash,
            )
        )
        matched = set()
        selected = []
        suppressed = set()
        for pair in positives:
            if pair.first_hash in matched or pair.second_hash in matched:
                continue
            matched.update((pair.first_hash, pair.second_hash))
            suppressed.add(str(pair.proposed_loser))
            selected.append({"pair": [pair.first_hash, pair.second_hash], "winner": pair.proposed_winner, "loser": pair.proposed_loser, "state": pair.state})
        survivors = [deepcopy(row) for row in rows if _candidate_hash(row) not in suppressed]
        survivor_hashes = [_candidate_hash(row) for row in survivors]
        audit = {
            "profile": PROFILE,
            "mechanism_lock": MECHANISM_LOCK,
            "fixture_lock": FIXTURE_LOCK,
            "status": "OK",
            "source_hash": hashlib.sha256(np.asarray(vertices, dtype=">f8").tobytes() + np.asarray(faces, dtype=">i8").tobytes()).hexdigest(),
            "candidate_input_hash": sha256_json([_candidate_hash(row) for row in rows]),
            "constants_hash": sha256_json({"q_bits": Q_BITS, "turn_min": TURN_MIN, "bridge_min": BRIDGE_MASS_MIN, "dominance_min": COMPONENT_DOMINANCE_MIN, "sign_min": SIGN_COHERENCE_MIN, "boundary_max": BOUNDARY_RATIO_MAX}),
            "mesh_face_count": measure.face_count,
            "graph_edge_count": measure.graph_edge_count,
            "orientable_component_count": measure.orientable_component_count,
            "nonorientable_component_count": measure.nonorientable_component_count,
            "hinge_atom_count": len(measure.hinge_atoms),
            "veto_atom_count": len(measure.veto_atoms),
            "candidate_count": len(candidates),
            "pair_count": len(pairs),
            "geometry_or_exact_pair_count": eligible_count,
            "clip_visit_count": visit_counter[0],
            "pair_records": [_pair_audit(pair) for pair in pairs if pair.reason != "PAIR_GEOMETRY" or pair.state == "EXACT_COINCIDENT"],
            "selected_matching": selected,
            "suppressed_hashes": sorted(suppressed),
            "survivor_hashes": survivor_hashes,
        }
        encoded = _encode_audit(audit)
        if len(encoded) > MAX_AUDIT_BYTES:
            raise NCMMResourceAbort("AUDIT_CAP")
        audit["semantic_sha256"] = hashlib.sha256(encoded).hexdigest()
        baseline.update(
            {
                "schema_version": 1,
                "kind": "sherman_qc_f1_v67_ncmm_nms_research_result",
                "profile": PROFILE,
                "predictions": survivors,
                "candidate_count": len(survivors),
                "ncmm_nms_v1_audit": audit,
                "expected_count_used": False,
                "labels_used": False,
                "cad_used": False,
                "part_identity_used": False,
                "filename_identity_used": False,
                "source_only": True,
                "research_only": True,
                "ordinary_final_nms_call_count": 0,
                "candidate_added_after_nms": False,
            }
        )
        return baseline, pairs
    except (NCMMInputError, NCMMResourceAbort) as error:
        status = "RESOURCE_ABORT" if isinstance(error, NCMMResourceAbort) else "NUMERIC_OR_INPUT_ABORT"
        noop = _atomic_noop(rows, status, str(error))
        baseline.update(noop)
        return baseline, []


__all__ = [
    "ABBREVIATION",
    "FIXTURE_LOCK",
    "MECHANISM_LOCK",
    "PROFILE",
    "Atom",
    "Candidate",
    "MeshMeasure",
    "PairEvaluation",
    "apply_ncmm_nms_v1",
    "build_mesh_measure",
    "canonical_bytes",
    "q40",
    "q40_int",
]
