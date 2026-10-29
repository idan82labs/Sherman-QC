"""Neutral deterministic candidate-prism geometry for frozen F1 studies.

This module contains only source-triangle canonicalization, the frozen
candidate-frame construction, and polygon clipping.  It has no detector
predicate, score, label, identity, topology, adjacency, or NMS access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator

import numpy as np


MAX_VERTICES = 250_000
MAX_TRIANGLES = 500_000
MAX_SUPPORT_TRIANGLES = 65_536


class CandidatePrismError(RuntimeError):
    """Raised when a neutral geometry resource or numeric contract fails."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CandidatePrismError("zero or nonfinite vector")
    return np.asarray(value, dtype=np.float64) / norm


def _quantize(value: float, quantum: float) -> float:
    return int(round(float(value) / quantum)) * quantum


def _jacobi_principal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinned 3x3 symmetric Jacobi solve with sixteen fixed sweeps."""
    work = np.asarray(matrix, dtype=np.float64).copy()
    vectors = np.eye(3, dtype=np.float64)
    for _ in range(16):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            off = float(work[p, q])
            if off == 0.0:
                continue
            angle = 0.5 * math.atan2(
                2.0 * off,
                float(work[q, q] - work[p, p]),
            )
            cosine, sine = math.cos(angle), math.sin(angle)
            rotation = np.eye(3, dtype=np.float64)
            rotation[p, p] = cosine
            rotation[q, q] = cosine
            rotation[p, q] = sine
            rotation[q, p] = -sine
            work = rotation.T @ work @ rotation
            vectors = vectors @ rotation
    values = np.diag(work)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


@dataclass(frozen=True)
class CanonicalTriangleMesh:
    triangles: tuple[np.ndarray, ...]
    normals: tuple[np.ndarray, ...]
    areas: tuple[float, ...]
    source_sha256: str


@dataclass(frozen=True)
class CandidateFrameDomain:
    p0: np.ndarray
    p1: np.ndarray
    origin: np.ndarray
    axis: np.ndarray
    transverse: np.ndarray
    complement: np.ndarray
    scale: float
    length: float
    support_indices: tuple[int, ...]
    frame_gap: float


def canonical_triangle_mesh(
    raw_source: dict[str, Any],
    scale: float,
) -> CanonicalTriangleMesh:
    """Canonicalize raw coordinates/faces without retaining source identities."""
    allowed = {"positions", "triangles"}
    forbidden = set(raw_source) - allowed
    if forbidden:
        raise CandidatePrismError(f"forbidden source fields: {sorted(forbidden)}")
    positions = np.asarray(raw_source.get("positions", []), dtype=np.float64)
    faces = np.asarray(raw_source.get("triangles", []), dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise CandidatePrismError("invalid positions")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise CandidatePrismError("invalid triangles")
    if len(positions) > MAX_VERTICES or len(faces) > MAX_TRIANGLES:
        raise CandidatePrismError("source cap")
    if not np.isfinite(positions).all():
        raise CandidatePrismError("nonfinite positions")
    if np.any(faces < 0) or np.any(faces >= len(positions)):
        raise CandidatePrismError("triangle index")
    if not math.isfinite(scale) or scale <= 0.0:
        raise CandidatePrismError("invalid scale")

    coordinate_quantum = scale / (1 << 16)
    quantized = np.rint(positions / coordinate_quantum).astype(np.int64)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for face in faces:
        vertices = sorted(
            tuple(int(component) for component in quantized[int(index)])
            for index in face
        )
        key = tuple(component for vertex in vertices for component in vertex)
        unique.setdefault(
            key,
            np.asarray(vertices, dtype=np.float64) * coordinate_quantum,
        )

    triangles: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[float] = []
    area_quantum = scale * scale / (1 << 24)
    minimum_area = scale * scale / (1 << 20)
    for key in sorted(unique):
        triangle = unique[key]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        magnitude = float(np.linalg.norm(cross))
        area = _quantize(0.5 * magnitude, area_quantum)
        if magnitude <= 0.0 or area < minimum_area:
            continue
        normal = np.rint((cross / magnitude) * (1 << 30)).astype(np.int64)
        normal = unit(normal.astype(np.float64) / (1 << 30))
        triangles.append(triangle)
        normals.append(normal)
        areas.append(area)

    digest = hashlib.sha256(
        _canonical(
            {
                "positions_q16": quantized.tolist(),
                "triangles": [list(key) for key in sorted(unique)],
                "scale": scale,
            }
        )
    ).hexdigest()
    return CanonicalTriangleMesh(
        tuple(triangles),
        tuple(normals),
        tuple(areas),
        digest,
    )


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    parameter = 0.0 if denominator <= 0.0 else float(
        np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0)
    )
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _segment_segment_distance(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa = float(np.dot(u, u))
    bb = float(np.dot(u, v))
    cc = float(np.dot(v, v))
    dd = float(np.dot(u, w))
    ee = float(np.dot(v, w))
    denominator = aa * cc - bb * bb
    if aa <= 0.0:
        return _point_segment_distance(a0, b0, b1)
    if cc <= 0.0:
        return _point_segment_distance(b0, a0, a1)
    s_num = denominator
    s_den = denominator
    t_num = denominator
    t_den = denominator
    if denominator <= 1e-30:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num = bb * ee - cc * dd
        t_num = aa * ee - bb * dd
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) <= 1e-30 else s_num / s_den
    tc = 0.0 if abs(t_num) <= 1e-30 else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + parameter * ab)))
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + parameter * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + parameter * (c - b))))
    normal = np.cross(ab, ac)
    return abs(float(np.dot(ap, normal))) / max(float(np.linalg.norm(normal)), 1e-30)


def _segment_triangle_distance(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
) -> float:
    a, b, c = triangle
    normal = np.cross(b - a, c - a)
    denominator = float(np.dot(normal, end - start))
    if abs(denominator) > 1e-30:
        parameter = float(np.dot(normal, a - start) / denominator)
        if 0.0 <= parameter <= 1.0:
            hit = start + parameter * (end - start)
            if _point_triangle_distance(hit, triangle) <= 1e-12:
                return 0.0
    distances = [
        _point_triangle_distance(start, triangle),
        _point_triangle_distance(end, triangle),
    ]
    distances.extend(
        _segment_segment_distance(start, end, edge0, edge1)
        for edge0, edge1 in ((a, b), (b, c), (c, a))
    )
    return min(distances)


def candidate_frame_domain(candidate: Any, mesh: CanonicalTriangleMesh) -> CandidateFrameDomain:
    scale = float(candidate.scale)
    p0 = np.asarray(candidate.line_start, dtype=np.float64)
    p1 = np.asarray(candidate.line_end, dtype=np.float64)
    axis = unit(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    origin = 0.5 * (p0 + p1)
    projector = np.eye(3) - np.outer(axis, axis)
    broad_start = p0 - 2.0 * scale * axis
    broad_end = p1 + 2.0 * scale * axis
    tensor = np.zeros((3, 3), dtype=np.float64)
    broad_area = 0.0
    support: list[int] = []
    for index, (triangle, normal, area) in enumerate(
        zip(mesh.triangles, mesh.normals, mesh.areas)
    ):
        axial = (triangle - p0) @ axis
        if float(np.max(axial)) < -2.0 * scale:
            continue
        if float(np.min(axial)) > length + 2.0 * scale:
            continue
        if _segment_triangle_distance(broad_start, broad_end, triangle) > 16.0 * scale:
            continue
        support.append(index)
        projected = projector @ normal
        tensor += area * np.outer(projected, projected)
        broad_area += area
    if len(support) > MAX_SUPPORT_TRIANGLES or broad_area <= 0.0:
        raise CandidatePrismError("support cap or empty support")
    values, vectors = _jacobi_principal(tensor / broad_area)
    complement = unit(projector @ vectors[:, 0])
    for component in complement:
        if abs(float(component)) > 2**-30:
            if component < 0.0:
                complement = -complement
            break
    transverse = unit(np.cross(axis, complement))
    gap = max(0.0, float(values[0] - values[1])) / max(
        1e-30,
        float(values[0] + values[1]),
    )
    return CandidateFrameDomain(
        p0=p0,
        p1=p1,
        origin=origin,
        axis=axis,
        transverse=transverse,
        complement=complement,
        scale=scale,
        length=length,
        support_indices=tuple(support),
        frame_gap=gap,
    )


def local_triangle(triangle: np.ndarray, frame: CandidateFrameDomain) -> np.ndarray:
    return np.column_stack(
        (
            (triangle - frame.p0) @ frame.axis,
            (triangle - frame.origin) @ frame.transverse,
            (triangle - frame.origin) @ frame.complement,
        )
    )


def clip_halfspace(
    polygon: np.ndarray,
    axis: int,
    bound: float,
    keep_greater: bool,
    coordinate_quantum: float,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = bool(previous[axis] >= bound) if keep_greater else bool(previous[axis] <= bound)
    for current in polygon:
        current_inside = bool(current[axis] >= bound) if keep_greater else bool(current[axis] <= bound)
        if current_inside != previous_inside:
            denominator = float(current[axis] - previous[axis])
            parameter = 0.0 if denominator == 0.0 else (
                bound - float(previous[axis])
            ) / denominator
            point = previous + parameter * (current - previous)
            point = np.rint(point / coordinate_quantum) * coordinate_quantum
            point[axis] = bound
            output.append(point)
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=np.float64).reshape((-1, 3))


def clip_candidate_prism(
    polygon: np.ndarray,
    frame: CandidateFrameDomain,
) -> np.ndarray:
    quantum = frame.scale / (1 << 16)
    planes = (
        (0, -frame.scale, True),
        (0, frame.length + frame.scale, False),
        (1, -12.0 * frame.scale, True),
        (1, 12.0 * frame.scale, False),
        (2, -12.0 * frame.scale, True),
        (2, 12.0 * frame.scale, False),
    )
    result = polygon
    for axis, bound, keep_greater in planes:
        result = clip_halfspace(result, axis, bound, keep_greater, quantum)
        if len(result) < 3:
            return np.empty((0, 3), dtype=np.float64)
    return result


def clip_axial_transverse_cell(
    polygon: np.ndarray,
    frame: CandidateFrameDomain,
    axial_lo: float,
    axial_hi: float,
    transverse_lo: float,
    transverse_hi: float,
) -> np.ndarray:
    quantum = frame.scale / (1 << 16)
    result = polygon
    for axis, bound, keep_greater in (
        (0, axial_lo, True),
        (0, axial_hi, False),
        (1, transverse_lo, True),
        (1, transverse_hi, False),
    ):
        result = clip_halfspace(result, axis, bound, keep_greater, quantum)
        if len(result) < 3:
            return np.empty((0, 3), dtype=np.float64)
    return result


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    anchor = polygon[0]
    total = 0.0
    for index in range(1, len(polygon) - 1):
        total += 0.5 * float(
            np.linalg.norm(
                np.cross(
                    polygon[index] - anchor,
                    polygon[index + 1] - anchor,
                )
            )
        )
    return total


def iter_prism_pieces(
    mesh: CanonicalTriangleMesh,
    frame: CandidateFrameDomain,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield canonical face index, face normal, and clipped local polygon."""
    for index in frame.support_indices:
        local = local_triangle(mesh.triangles[index], frame)
        clipped = clip_candidate_prism(local, frame)
        if len(clipped) >= 3 and polygon_area(clipped) > 0.0:
            yield index, mesh.normals[index], clipped
