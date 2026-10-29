"""Deterministic source-only V56 canonical turn-median line reconstruction.

This module is intentionally self-contained.  It does not import or invoke the
terminal V55 implementation or any of its artifacts.  Frozen V52 lines and the
frozen union pool are used only as bounded geometric search seeds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
import resource
import time
from typing import Any, Iterable, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROFILE = "V56_CANONICAL_TURN_MEDIAN_LINE_RECONSTRUCTION_V1"
SHORT_NAME = "CTMLR_V1"
PRECOMMIT_TOKEN = "V56_CTMLR_MECHANISM_LOCKED"
TERMINAL_TOKEN = "V56_CTMLR_TERMINAL_RETIRE"
RESULT_KIND = "sherman_qc_f1_v56_canonical_turn_median_line_reconstruction_result"

STATION_FRACTIONS = tuple(0.10 + index * (0.80 / 6.0) for index in range(7))
MAXIMUM_INPUT_ROWS = 1_024
MAXIMUM_INTERSECTION_SEGMENTS = 16_384
MAXIMUM_WALL_SECONDS = 300.0
RSS_LIMIT_BYTES = 16 * 1024**3
MINIMUM_TURN_DEGREES = 15.0
MAXIMUM_TURN_DEGREES = 170.0
MINIMUM_PROMINENCE_DEGREES = 4.0
MINIMUM_TURN_PURITY = 0.85
MAXIMUM_TAIL_RMS = 0.04
MAXIMUM_TURN_IQR_OVER_W = 0.75
MINIMUM_CERTIFIED_STATIONS = 5
MINIMUM_STATION_COVERAGE = 0.60
MINIMUM_LONGEST_RUN = 3
MAXIMUM_DIRECTION_DRIFT_DEGREES = 7.5
DUPLICATE_DIRECTION_DEGREES = 1.0
DUPLICATE_OVERLAP = 0.70

REJECTION_CODES = (
    "REJECT_SHORT_SEED",
    "REJECT_NO_SECTION",
    "REJECT_COPLANAR_OR_BRANCH_AMBIGUITY",
    "REJECT_NO_PERSISTENT_TURN",
    "REJECT_MULTIPLE_TURN_EVENTS",
    "REJECT_DISTRIBUTED_OR_IMPURE_TURN",
    "REJECT_INSUFFICIENT_STATIONS",
    "REJECT_LINE_CONSENSUS",
    "REJECT_DIRECTION_DRIFT",
    "REJECT_TRANSVERSE_CENTERING",
    "REJECT_OUTPUT_DEGENERACY",
    "REJECT_DUPLICATE_LOSER",
)


class V56TerminalFailure(RuntimeError):
    """Terminal V56 scan or execution failure."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class SeedSchemaError(ValueError):
    """A frozen row cannot be represented by the precommitted total schema."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Constants:
    bbox_diagonal: float
    median_mesh_edge: float
    minimum_seed_span: float
    search_half_width: float
    resampling_step: float
    fine_lag: float
    middle_lag: float
    coarse_lag: float
    crop_radius: float
    join_tolerance: float
    cross_scale_tolerance: float
    line_consensus_threshold: float
    transverse_center_threshold: float
    maximum_median_center_error: float
    duplicate_distance: float

    @classmethod
    def from_scales(cls, bbox_diagonal: float, median_mesh_edge: float) -> "Constants":
        b = float(bbox_diagonal)
        h = float(median_mesh_edge)
        if not math.isfinite(b) or not math.isfinite(h) or b <= 0.0:
            raise V56TerminalFailure("INVALID_BBOX_OR_MEDIAN_EDGE")
        if not (1e-7 * b <= h <= 0.05 * b):
            raise V56TerminalFailure("INVALID_BBOX_OR_MEDIAN_EDGE")

        def clamp(value: float, lower: float, upper: float) -> float:
            return min(max(value, lower), upper)

        w = clamp(20.0 * h, 0.012 * b, 0.035 * b)
        delta = clamp(1.5 * h, 0.0005 * b, 0.002 * b)
        l1 = clamp(4.0 * h, 0.0025 * b, 0.005 * b)
        l2 = clamp(8.0 * h, 0.005 * b, 0.010 * b)
        l3 = clamp(16.0 * h, 0.010 * b, 0.020 * b)
        return cls(
            bbox_diagonal=b,
            median_mesh_edge=h,
            minimum_seed_span=max(12.0 * h, 0.015 * b),
            search_half_width=w,
            resampling_step=delta,
            fine_lag=l1,
            middle_lag=l2,
            coarse_lag=l3,
            crop_radius=w + 2.0 * l3,
            join_tolerance=max(0.05 * h, 1e-6 * b),
            cross_scale_tolerance=max(3.0 * delta, 0.002 * b),
            line_consensus_threshold=max(3.0 * h, 0.002 * b),
            transverse_center_threshold=max(2.0 * h, 0.0015 * b),
            maximum_median_center_error=max(h, 0.00075 * b),
            duplicate_distance=max(4.0 * h, 0.003 * b),
        )

    def to_dict(self) -> dict[str, float]:
        return {field_name: float(getattr(self, field_name)) for field_name in self.__dataclass_fields__}


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    bbox_diagonal: float
    median_mesh_edge: float
    constants: Constants
    face_centers: np.ndarray
    face_radii: np.ndarray
    maximum_face_radius: float
    face_tree: cKDTree

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, faces: np.ndarray) -> "MeshData":
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise V56TerminalFailure("INVALID_SOURCE_GEOMETRY")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise V56TerminalFailure("INVALID_MESH_INDEXING")
        if not np.all(np.isfinite(vertices)):
            raise V56TerminalFailure("INVALID_SOURCE_GEOMETRY")
        if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
            raise V56TerminalFailure("INVALID_MESH_INDEXING")
        if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or np.any(faces[:, 2] == faces[:, 0]):
            raise V56TerminalFailure("INVALID_MESH_INDEXING")
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
        b = float(np.linalg.norm(high - low))
        undirected = np.sort(
            np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0),
            axis=1,
        )
        undirected = np.unique(undirected, axis=0)
        edge_lengths = np.linalg.norm(vertices[undirected[:, 1]] - vertices[undirected[:, 0]], axis=1)
        edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
        if not len(edge_lengths):
            raise V56TerminalFailure("INVALID_BBOX_OR_MEDIAN_EDGE")
        h = float(np.median(edge_lengths))
        constants = Constants.from_scales(b, h)
        tri = vertices[faces]
        centers = tri.mean(axis=1)
        radii = np.linalg.norm(tri - centers[:, None, :], axis=2).max(axis=1)
        return cls(
            vertices=vertices,
            faces=faces,
            bbox_diagonal=b,
            median_mesh_edge=h,
            constants=constants,
            face_centers=centers,
            face_radii=radii,
            maximum_face_radius=float(radii.max(initial=0.0)),
            face_tree=cKDTree(centers, compact_nodes=True, balanced_tree=True),
        )

    @classmethod
    def load(cls, path: Path) -> "MeshData":
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if mesh.is_empty() or not mesh.has_triangles():
            raise V56TerminalFailure("INVALID_SOURCE_GEOMETRY")
        return cls.from_arrays(np.asarray(mesh.vertices), np.asarray(mesh.triangles))


@dataclass(frozen=True)
class Seed:
    direction: np.ndarray
    midpoint: np.ndarray
    span: float
    line_start: np.ndarray
    line_end: np.ndarray
    origin_class: str
    origin_file_ordinal: int
    row_ordinal: int
    schema_code: str
    original_row: dict[str, Any] = field(compare=False, repr=False)

    @property
    def canonical_key(self) -> tuple[float, ...]:
        return (
            float(self.direction[0]),
            float(self.direction[1]),
            float(self.direction[2]),
            float(self.midpoint[0]),
            float(self.midpoint[1]),
            float(self.midpoint[2]),
            float(self.span),
            float(self.origin_file_ordinal),
            float(self.row_ordinal),
        )


@dataclass
class Contour:
    points3: np.ndarray
    points2: np.ndarray
    arc: np.ndarray
    order: int


@dataclass
class StationEvidence:
    station_index: int
    station_fraction: float
    station_point: np.ndarray
    contour: Contour
    s25: float
    s50: float
    s75: float
    locus: np.ndarray
    turn_degrees: float
    turn_purity: float
    turn_iqr_width: float
    median_tail_rms: float


@dataclass
class Proposal:
    output_class: str
    seed: Seed
    line_start: np.ndarray
    line_end: np.ndarray
    certified: list[StationEvidence]
    station_errors: list[float]
    median_fit_error: float
    median_turn_iqr_width: float
    median_turn_degrees: float
    median_turn_purity: float

    @property
    def direction(self) -> np.ndarray:
        return canonical_direction(self.line_end - self.line_start)

    @property
    def midpoint(self) -> np.ndarray:
        return (self.line_start + self.line_end) / 2.0

    @property
    def span(self) -> float:
        return float(np.linalg.norm(self.line_end - self.line_start))

    @property
    def coverage(self) -> float:
        if not self.certified:
            return 0.0
        fractions = [item.station_fraction for item in self.certified]
        return float(max(fractions) - min(fractions))

    @property
    def longest_run(self) -> int:
        return longest_run([item.station_index for item in self.certified])


def canonical_direction(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SeedSchemaError("ZERO_OR_NONFINITE_DIRECTION")
    direction = value / norm
    absolute = np.abs(direction)
    largest = float(absolute.max())
    candidates = np.flatnonzero(absolute == largest)
    index = int(candidates[0])
    if direction[index] < 0.0:
        direction = -direction
    return direction


def _vector(row: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise SeedSchemaError(f"INVALID_{key.upper()}")
    return value


def _canonical_segment(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    span = float(np.linalg.norm(delta))
    if not math.isfinite(span) or span <= 0.0:
        raise SeedSchemaError("ZERO_OR_NONFINITE_SPAN")
    direction = canonical_direction(delta)
    midpoint = (np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64)) / 2.0
    return midpoint - 0.5 * span * direction, midpoint + 0.5 * span * direction, direction, span, midpoint


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    value = abs(float(np.dot(first, second)))
    return math.degrees(math.acos(min(max(value, -1.0), 1.0)))


def _line_distance(point: np.ndarray, midpoint: np.ndarray, direction: np.ndarray) -> float:
    delta = np.asarray(point, dtype=np.float64) - midpoint
    return float(np.linalg.norm(delta - float(np.dot(delta, direction)) * direction))


def materialize_seed(
    row: dict[str, Any],
    *,
    bbox_diagonal: float,
    median_mesh_edge: float,
    origin_class: str,
    origin_file_ordinal: int,
    row_ordinal: int,
) -> Seed:
    """Materialize exactly one frozen row under the total E/S/A contract."""

    b = float(bbox_diagonal)
    h = float(median_mesh_edge)
    ep = max(1e-7 * b, 1e-4 * h)
    complete: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = {}
    has_start = "line_start" in row
    has_end = "line_end" in row
    if has_start and has_end:
        complete["E"] = _canonical_segment(_vector(row, "line_start"), _vector(row, "line_end"))
    if "span_endpoints" in row:
        raw = row["span_endpoints"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise SeedSchemaError("MALFORMED_SPAN_ENDPOINTS")
        span = np.asarray(raw, dtype=np.float64)
        if span.shape != (2, 3) or not np.all(np.isfinite(span)):
            raise SeedSchemaError("INVALID_SPAN_ENDPOINTS")
        complete["S"] = _canonical_segment(span[0], span[1])
    a_keys = ("anchor", "axis_direction", "span_mm")
    if all(key in row for key in a_keys):
        anchor = _vector(row, "anchor")
        direction = canonical_direction(_vector(row, "axis_direction"))
        span = float(row["span_mm"])
        if not math.isfinite(span) or span <= 0.0:
            raise SeedSchemaError("ZERO_OR_NONFINITE_SPAN")
        complete["A"] = _canonical_segment(anchor - 0.5 * span * direction, anchor + 0.5 * span * direction)
    if not complete:
        raise SeedSchemaError("UNSUPPORTED_OR_PARTIAL_SCHEMA")

    selected_code = next(code for code in ("E", "S", "A") if code in complete)
    start, end, direction, span, midpoint = complete[selected_code]
    es = max(1e-7 * b, 1e-6 * span)
    etheta_degrees = math.degrees(1e-5)
    for code, candidate in complete.items():
        if code == selected_code:
            continue
        other_start, other_end, other_direction, other_span, other_midpoint = candidate
        if _angle_degrees(direction, other_direction) > etheta_degrees:
            raise SeedSchemaError("INCONSISTENT_REDUNDANT_DIRECTION")
        if abs(other_span - span) > max(es, max(1e-7 * b, 1e-6 * other_span)):
            raise SeedSchemaError("INCONSISTENT_REDUNDANT_SPAN")
        if code in ("E", "S") and selected_code in ("E", "S"):
            if np.linalg.norm(start - other_start) > max(ep, es) or np.linalg.norm(end - other_end) > max(ep, es):
                raise SeedSchemaError("INCONSISTENT_REDUNDANT_ENDPOINTS")
        else:
            anchor = other_midpoint if code == "A" else midpoint
            line_midpoint = midpoint if code == "A" else other_midpoint
            line_direction = direction if code == "A" else other_direction
            if _line_distance(anchor, line_midpoint, line_direction) > ep:
                raise SeedSchemaError("INCONSISTENT_REDUNDANT_ANCHOR")
            projection = float(np.dot(anchor - line_midpoint, line_direction))
            if abs(projection) > 0.5 * span + ep:
                raise SeedSchemaError("INCONSISTENT_REDUNDANT_ANCHOR")

    if "anchor" in row:
        anchor = _vector(row, "anchor")
        if _line_distance(anchor, midpoint, direction) > ep:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_ANCHOR")
        if abs(float(np.dot(anchor - midpoint, direction))) > 0.5 * span + ep:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_ANCHOR")
    if "axis_direction" in row:
        if _angle_degrees(direction, canonical_direction(_vector(row, "axis_direction"))) > etheta_degrees:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_DIRECTION")
    if "span_mm" in row:
        raw_span = float(row["span_mm"])
        if not math.isfinite(raw_span) or raw_span <= 0.0:
            raise SeedSchemaError("ZERO_OR_NONFINITE_SPAN")
        if abs(raw_span - span) > es:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_SPAN")
    lone_endpoint_key = "line_start" if has_start and not has_end else "line_end" if has_end and not has_start else None
    if lone_endpoint_key is not None:
        endpoint = _vector(row, lone_endpoint_key)
        if _line_distance(endpoint, midpoint, direction) > ep:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_ENDPOINT")
        if abs(float(np.dot(endpoint - midpoint, direction))) > 0.5 * span + ep:
            raise SeedSchemaError("INCONSISTENT_PARTIAL_ENDPOINT")
    return Seed(
        direction=direction,
        midpoint=midpoint,
        span=span,
        line_start=start,
        line_end=end,
        origin_class=origin_class,
        origin_file_ordinal=int(origin_file_ordinal),
        row_ordinal=int(row_ordinal),
        schema_code="".join(code for code in ("E", "S", "A") if code in complete),
        original_row=row,
    )


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _symmetric_midpoint_line_distance(first: Seed | Proposal, second: Seed | Proposal) -> float:
    return max(
        _line_distance(first.midpoint, second.midpoint, second.direction),
        _line_distance(second.midpoint, first.midpoint, first.direction),
    )


def collapse_exact_duplicates(seeds: list[Seed], constants: Constants) -> tuple[list[Seed], list[int]]:
    ordered = sorted(
        seeds,
        key=lambda item: (
            float(item.direction[0]), float(item.direction[1]), float(item.direction[2]),
            float(item.midpoint[0] / constants.bbox_diagonal),
            float(item.midpoint[1] / constants.bbox_diagonal),
            float(item.midpoint[2] / constants.bbox_diagonal),
            float(item.span / constants.bbox_diagonal),
            item.origin_file_ordinal, item.row_ordinal,
        ),
    )
    union = _UnionFind(len(ordered))
    ep = max(1e-7 * constants.bbox_diagonal, 1e-4 * constants.median_mesh_edge)
    etheta = math.degrees(1e-5)
    for index, first in enumerate(ordered):
        for other_index in range(index + 1, len(ordered)):
            second = ordered[other_index]
            if _angle_degrees(first.direction, second.direction) > etheta:
                continue
            es = max(1e-7 * constants.bbox_diagonal, 1e-6 * max(first.span, second.span))
            if _symmetric_midpoint_line_distance(first, second) > ep:
                continue
            if abs(first.span - second.span) > es:
                continue
            if np.linalg.norm(first.line_start - second.line_start) > es or np.linalg.norm(first.line_end - second.line_end) > es:
                continue
            union.union(index, other_index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        groups[union.find(index)].append(index)
    representatives = [ordered[min(indices)] for _, indices in sorted(groups.items())]
    multiplicities = [len(indices) for _, indices in sorted(groups.items())]
    return representatives, multiplicities


def _section_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = np.eye(3, dtype=np.float64)
    alignment = np.abs(axes @ direction)
    selected = int(np.flatnonzero(alignment == alignment.min())[0])
    u = axes[selected] - float(np.dot(axes[selected], direction)) * direction
    u /= float(np.linalg.norm(u))
    v = np.cross(direction, u)
    v /= float(np.linalg.norm(v))
    return u, v


def _quantized_key(point: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.rint(np.asarray(point) / tolerance))


def _triangle_plane_segment(
    triangle: np.ndarray,
    vertex_ids: np.ndarray,
    station: np.ndarray,
    normal: np.ndarray,
) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
    relative = triangle.astype(np.longdouble) - station.astype(np.longdouble)
    signed = np.asarray(relative @ normal.astype(np.longdouble), dtype=np.longdouble)
    zero = signed == 0
    if bool(np.all(zero)):
        return "COPLANAR", None
    points: list[np.ndarray] = []
    for edge_index, (left, right) in enumerate(((0, 1), (1, 2), (2, 0))):
        dl, dr = signed[left], signed[right]
        if dl == 0 and dr == 0:
            points.extend((triangle[left], triangle[right]))
        elif dl == 0:
            points.append(triangle[left])
        elif dr == 0:
            points.append(triangle[right])
        elif (dl < 0 < dr) or (dr < 0 < dl):
            factor = float(dl / (dl - dr))
            points.append(triangle[left] + factor * (triangle[right] - triangle[left]))
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.array_equal(point, current) for current in unique):
            unique.append(np.asarray(point, dtype=np.float64))
    if len(unique) < 2:
        return "NONE", None
    if len(unique) > 2:
        unique.sort(key=lambda point: tuple(float(value) for value in point))
        best = max(
            ((float(np.linalg.norm(unique[j] - unique[i])), i, j) for i in range(len(unique)) for j in range(i + 1, len(unique))),
            key=lambda value: (value[0], -value[1], -value[2]),
        )
        unique = [unique[best[1]], unique[best[2]]]
    if np.array_equal(unique[0], unique[1]):
        return "NONE", None
    return "SEGMENT", (unique[0], unique[1])


def _trace_paths(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    join_tolerance: float,
) -> list[np.ndarray]:
    node_points: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    edge_set: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for first, second in segments:
        key_first = _quantized_key(first, join_tolerance)
        key_second = _quantized_key(second, join_tolerance)
        if key_first == key_second:
            continue
        node_points[key_first].append(first)
        node_points[key_second].append(second)
        edge = tuple(sorted((key_first, key_second)))
        edge_set.add(edge)
    points = {key: np.mean(values, axis=0) for key, values in node_points.items()}
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for first, second in sorted(edge_set):
        adjacency[first].append(second)
        adjacency[second].append(first)
    for key in adjacency:
        adjacency[key].sort()
    unused = set(edge_set)
    paths: list[np.ndarray] = []

    def take_path(start: tuple[int, int, int], first_next: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        path = [start]
        previous, current = start, first_next
        while True:
            edge = tuple(sorted((previous, current)))
            if edge not in unused:
                break
            unused.remove(edge)
            path.append(current)
            if len(adjacency[current]) != 2:
                break
            options = [value for value in adjacency[current] if value != previous and tuple(sorted((current, value))) in unused]
            if not options:
                break
            previous, current = current, options[0]
        return path

    boundary = sorted(key for key, neighbors in adjacency.items() if len(neighbors) != 2)
    for start in boundary:
        for next_node in adjacency[start]:
            edge = tuple(sorted((start, next_node)))
            if edge in unused:
                keys = take_path(start, next_node)
                if len(keys) >= 2:
                    paths.append(np.asarray([points[key] for key in keys], dtype=np.float64))
    while unused:
        first_edge = min(unused)
        keys = take_path(first_edge[0], first_edge[1])
        if len(keys) >= 2:
            paths.append(np.asarray([points[key] for key in keys], dtype=np.float64))
    return paths


def _resample_path(points3: np.ndarray, station: np.ndarray, u: np.ndarray, v: np.ndarray, delta: float, order: int) -> Contour | None:
    lengths = np.linalg.norm(np.diff(points3, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 0.0))
    points3 = points3[keep]
    if len(points3) < 2:
        return None
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points3, axis=0), axis=1))))
    total = float(arc[-1])
    if total <= 0.0:
        return None
    sample_count = max(2, int(math.floor(total / delta)) + 1)
    sample_arc = np.linspace(0.0, total, sample_count, dtype=np.float64)
    result = np.empty((sample_count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(sample_arc, arc, points3[:, axis])
    relative = result - station
    points2 = np.column_stack((relative @ u, relative @ v))
    return Contour(points3=result, points2=points2, arc=sample_arc, order=order)


def build_station_contours(mesh: MeshData, seed: Seed, station: np.ndarray) -> tuple[list[Contour], str | None]:
    c = mesh.constants
    query_radius = c.crop_radius + mesh.maximum_face_radius
    face_ids = sorted(int(value) for value in mesh.face_tree.query_ball_point(station, query_radius))
    if not face_ids:
        return [], "REJECT_NO_SECTION"
    u, v = _section_basis(seed.direction)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    coplanar = False
    for face_id in face_ids:
        if np.linalg.norm(mesh.face_centers[face_id] - station) - mesh.face_radii[face_id] > c.crop_radius:
            continue
        ids = mesh.faces[face_id]
        status, segment = _triangle_plane_segment(mesh.vertices[ids], ids, station, seed.direction)
        if status == "COPLANAR":
            coplanar = True
            continue
        if segment is None:
            continue
        if min(np.linalg.norm(segment[0] - station), np.linalg.norm(segment[1] - station)) > c.crop_radius:
            continue
        segments.append(segment)
        if len(segments) > MAXIMUM_INTERSECTION_SEGMENTS:
            raise V56TerminalFailure("SECTION_SEGMENT_OVERFLOW")
    if coplanar:
        return [], "REJECT_COPLANAR_OR_BRANCH_AMBIGUITY"
    paths = _trace_paths(segments, join_tolerance=c.join_tolerance)
    contours: list[Contour] = []
    for order, path in enumerate(paths):
        projected = (path - station) @ np.column_stack((u, v))
        closest = float(np.linalg.norm(projected, axis=1).min(initial=math.inf))
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if closest > c.search_half_width or length < 2.0 * c.coarse_lag:
            continue
        contour = _resample_path(path, station, u, v, c.resampling_step, order)
        if contour is not None:
            contours.append(contour)
    return contours, None if contours else "REJECT_NO_SECTION"


def _tls_tangent(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    progression = points[-1] - points[0]
    if float(np.dot(tangent, progression)) < 0.0:
        tangent = -tangent
    normal = np.asarray((-tangent[1], tangent[0]))
    rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    return tangent, rms


def _interval_indices(arc: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.flatnonzero((arc >= lower) & (arc <= upper))


def _scale_turns(contour: Contour, lag: float, constants: Constants) -> list[dict[str, float]]:
    raw: list[dict[str, float]] = []
    for index, position in enumerate(contour.arc):
        left = _interval_indices(contour.arc, position - lag, position - 0.5 * lag)
        right = _interval_indices(contour.arc, position + 0.5 * lag, position + lag)
        if len(left) < 2 or len(right) < 2:
            continue
        left_tangent, left_rms = _tls_tangent(contour.points2[left])
        right_tangent, right_rms = _tls_tangent(contour.points2[right])
        value = min(max(float(np.dot(left_tangent, right_tangent)), -1.0), 1.0)
        angle = math.degrees(math.acos(value))
        raw.append(
            {
                "index": float(index),
                "position": float(position),
                "angle": angle,
                "tail_rms": max(left_rms, right_rms) / lag,
            }
        )
    eligible: list[dict[str, float]] = []
    for event in raw:
        local = [item for item in raw if abs(item["position"] - event["position"]) <= 2.0 * constants.resampling_step]
        maximum = max(item["angle"] for item in local)
        if event["angle"] != maximum:
            continue
        if any(item["angle"] == maximum and item["position"] < event["position"] for item in local):
            continue
        wide = [item["angle"] for item in raw if abs(item["position"] - event["position"]) <= 2.0 * lag]
        if event["angle"] < MINIMUM_TURN_DEGREES:
            continue
        if event["tail_rms"] > MAXIMUM_TAIL_RMS:
            continue
        if event["angle"] - float(np.median(wide)) < MINIMUM_PROMINENCE_DEGREES:
            continue
        eligible.append(event)
    return eligible


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = 0.5 * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _interpolate(contour: Contour, position: float) -> np.ndarray:
    return np.asarray([np.interp(position, contour.arc, contour.points3[:, axis]) for axis in range(3)], dtype=np.float64)


def _turn_quantile(positions: np.ndarray, mass: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(mass)
    target = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    if index <= 0:
        return float(positions[0])
    before = float(cumulative[index - 1])
    after = float(cumulative[index])
    if after <= before:
        return float(positions[index])
    fraction = (target - before) / (after - before)
    return float(positions[index - 1] + fraction * (positions[index] - positions[index - 1]))


def _turn_median(contour: Contour, event_position: float, constants: Constants) -> tuple[dict[str, Any] | None, str | None]:
    c = constants
    lower, upper = event_position - c.coarse_lag, event_position + c.coarse_lag
    positions = contour.arc[(contour.arc >= lower) & (contour.arc <= upper)]
    if len(positions) < 3:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    tangents: list[np.ndarray] = []
    tangent_positions: list[float] = []
    half = 0.5 * c.fine_lag
    for position in positions:
        if position - half < contour.arc[0] or position + half > contour.arc[-1]:
            continue
        left = np.asarray([np.interp(position - half, contour.arc, contour.points2[:, axis]) for axis in range(2)])
        right = np.asarray([np.interp(position + half, contour.arc, contour.points2[:, axis]) for axis in range(2)])
        tangent = right - left
        norm = float(np.linalg.norm(tangent))
        if norm <= 0.0:
            continue
        tangents.append(tangent / norm)
        tangent_positions.append(float(position))
    if len(tangents) < 3:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    alpha: list[float] = []
    alpha_positions: list[float] = []
    for index in range(len(tangents) - 1):
        left, right = tangents[index], tangents[index + 1]
        cross = float(left[0] * right[1] - left[1] * right[0])
        dot = min(max(float(np.dot(left, right)), -1.0), 1.0)
        alpha.append(math.atan2(cross, dot))
        alpha_positions.append(0.5 * (tangent_positions[index] + tangent_positions[index + 1]))
    signed = np.asarray(alpha, dtype=np.float64)
    mass = np.abs(signed)
    total = float(mass.sum())
    if total <= 0.0:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    total_degrees = math.degrees(total)
    purity = abs(float(signed.sum())) / total
    if not (MINIMUM_TURN_DEGREES <= total_degrees <= MAXIMUM_TURN_DEGREES) or purity < MINIMUM_TURN_PURITY:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    alpha_positions_array = np.asarray(alpha_positions, dtype=np.float64)
    s25 = _turn_quantile(alpha_positions_array, mass, 0.25)
    s50 = _turn_quantile(alpha_positions_array, mass, 0.50)
    s75 = _turn_quantile(alpha_positions_array, mass, 0.75)
    if s75 - s25 > MAXIMUM_TURN_IQR_OVER_W * c.search_half_width:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    if lower - contour.arc[0] < c.middle_lag or contour.arc[-1] - upper < c.middle_lag:
        return None, "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    return {
        "s25": s25,
        "s50": s50,
        "s75": s75,
        "locus": _interpolate(contour, s50),
        "turn_degrees": total_degrees,
        "turn_purity": purity,
    }, None


def select_station_evidence(
    mesh: MeshData,
    seed: Seed,
    station_index: int,
    station_fraction: float,
    station_point: np.ndarray,
) -> tuple[StationEvidence | None, str]:
    contours, section_reason = build_station_contours(mesh, seed, station_point)
    if not contours:
        return None, section_reason or "REJECT_NO_SECTION"
    events: list[dict[str, Any]] = []
    for contour in contours:
        scales = [
            _scale_turns(contour, mesh.constants.fine_lag, mesh.constants),
            _scale_turns(contour, mesh.constants.middle_lag, mesh.constants),
            _scale_turns(contour, mesh.constants.coarse_lag, mesh.constants),
        ]
        for middle in scales[1]:
            participants = [middle]
            for scale_index in (0, 2):
                nearby = [item for item in scales[scale_index] if abs(item["position"] - middle["position"]) <= mesh.constants.cross_scale_tolerance]
                if nearby:
                    participants.append(max(nearby, key=lambda item: (item["angle"], -item["position"])))
            if len(participants) < 2:
                continue
            positions = [item["position"] for item in participants]
            if max(positions) - min(positions) > mesh.constants.cross_scale_tolerance:
                continue
            location = _weighted_median(positions, [item["angle"] for item in participants])
            events.append(
                {
                    "contour": contour,
                    "position": location,
                    "angle": float(np.median([item["angle"] for item in participants])),
                    "tail_rms": float(np.median([item["tail_rms"] for item in participants])),
                    "distance": float(np.linalg.norm(_interpolate(contour, location) - station_point)),
                }
            )
    if not events:
        return None, "REJECT_NO_PERSISTENT_TURN"
    # Multiple eligible middle-scale maxima can describe the same persistent
    # physical turn.  Collapse those within the frozen cross-scale locus before
    # applying the *between-event* ambiguity rule.
    clustered_events: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["contour"].order, item["position"], -item["angle"])):
        matching = [
            index
            for index, current in enumerate(clustered_events)
            if current["contour"].order == event["contour"].order
            and abs(current["position"] - event["position"]) <= mesh.constants.cross_scale_tolerance
        ]
        if not matching:
            clustered_events.append(event)
            continue
        index = matching[0]
        current = clustered_events[index]
        if (-event["angle"], event["tail_rms"], event["position"]) < (-current["angle"], current["tail_rms"], current["position"]):
            clustered_events[index] = event
    events = clustered_events
    maximum_angle = max(event["angle"] for event in events)
    retained = [event for event in events if event["angle"] >= 0.85 * maximum_angle]
    retained.sort(key=lambda event: (event["distance"], -event["angle"], event["contour"].order, event["position"]))
    chosen = retained[0]
    if len(retained) > 1:
        second = retained[1]
        if abs(second["distance"] - chosen["distance"]) <= mesh.constants.cross_scale_tolerance and second["angle"] >= 0.85 * chosen["angle"]:
            return None, "REJECT_MULTIPLE_TURN_EVENTS"
    median, median_reason = _turn_median(chosen["contour"], chosen["position"], mesh.constants)
    if median is None:
        return None, median_reason or "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
    return StationEvidence(
        station_index=station_index,
        station_fraction=station_fraction,
        station_point=station_point,
        contour=chosen["contour"],
        s25=float(median["s25"]),
        s50=float(median["s50"]),
        s75=float(median["s75"]),
        locus=np.asarray(median["locus"], dtype=np.float64),
        turn_degrees=float(median["turn_degrees"]),
        turn_purity=float(median["turn_purity"]),
        turn_iqr_width=float(median["s75"] - median["s25"]),
        median_tail_rms=float(chosen["tail_rms"]),
    ), "ACCEPT"


def longest_run(indices: Iterable[int]) -> int:
    ordered = sorted(set(int(value) for value in indices))
    best = current = 0
    previous: int | None = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        best = max(best, current)
        previous = value
    return best


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint = points.mean(axis=0)
    centered = points - midpoint
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    direction = canonical_direction(vectors[:, int(np.argmax(values))])
    residuals = np.linalg.norm(centered - (centered @ direction)[:, None] * direction, axis=1)
    return midpoint, direction, residuals


def reconstruct_line(seed: Seed, evidence: list[StationEvidence], constants: Constants) -> tuple[np.ndarray, np.ndarray, list[StationEvidence], float] | None:
    if len(evidence) < MINIMUM_CERTIFIED_STATIONS:
        return None
    loci = np.asarray([item.locus for item in evidence], dtype=np.float64)
    best: tuple[Any, ...] | None = None
    best_inliers: list[int] = []
    for left in range(len(evidence)):
        for right in range(left + 1, len(evidence)):
            if evidence[right].station_fraction - evidence[left].station_fraction < 0.40:
                continue
            delta = loci[right] - loci[left]
            norm = float(np.linalg.norm(delta))
            if norm <= 0.0:
                continue
            direction = delta / norm
            residuals = np.linalg.norm((loci - loci[left]) - (((loci - loci[left]) @ direction)[:, None] * direction), axis=1)
            inliers = [index for index, value in enumerate(residuals) if value <= constants.line_consensus_threshold]
            if not inliers:
                continue
            fractions = [evidence[index].station_fraction for index in inliers]
            coverage = max(fractions) - min(fractions)
            run = longest_run([evidence[index].station_index for index in inliers])
            rank = (
                -len(inliers), -coverage, -run,
                float(np.median(residuals[inliers])), float(np.max(residuals[inliers])),
                evidence[left].station_index, evidence[right].station_index,
            )
            if best is None or rank < best:
                best = rank
                best_inliers = inliers
    if len(best_inliers) < MINIMUM_CERTIFIED_STATIONS:
        return None
    selected = [evidence[index] for index in best_inliers]
    if max(item.station_fraction for item in selected) - min(item.station_fraction for item in selected) < MINIMUM_STATION_COVERAGE:
        return None
    if longest_run(item.station_index for item in selected) < MINIMUM_LONGEST_RUN:
        return None
    midpoint, direction, residuals = _fit_line(np.asarray([item.locus for item in selected]))
    if _angle_degrees(direction, seed.direction) > MAXIMUM_DIRECTION_DRIFT_DEGREES:
        return None
    start_parameter = float(np.dot(seed.line_start - midpoint, direction))
    end_parameter = float(np.dot(seed.line_end - midpoint, direction))
    start = midpoint + min(start_parameter, end_parameter) * direction
    end = midpoint + max(start_parameter, end_parameter) * direction
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)) or np.linalg.norm(end - start) <= 0.0:
        return None
    return start, end, selected, float(np.median(residuals))


def _closest_arc(contour: Contour, point: np.ndarray) -> tuple[float, float]:
    best = (math.inf, 0.0)
    for index in range(len(contour.points3) - 1):
        first = contour.points3[index]
        delta = contour.points3[index + 1] - first
        denominator = float(np.dot(delta, delta))
        factor = 0.0 if denominator <= 0.0 else min(max(float(np.dot(point - first, delta) / denominator), 0.0), 1.0)
        closest = first + factor * delta
        distance = float(np.linalg.norm(point - closest))
        arc = float(contour.arc[index] + factor * (contour.arc[index + 1] - contour.arc[index]))
        candidate = (distance, arc)
        if candidate < best:
            best = candidate
    return best


def certify_line(
    start: np.ndarray,
    end: np.ndarray,
    seed: Seed,
    evidence: list[StationEvidence],
    constants: Constants,
    *,
    allow_refit: bool,
) -> tuple[np.ndarray, np.ndarray, list[StationEvidence], list[float], float] | None:
    direction = canonical_direction(end - start)
    midpoint = (start + end) / 2.0
    certified: list[StationEvidence] = []
    errors: list[float] = []
    for item in evidence:
        denominator = float(np.dot(direction, seed.direction))
        if abs(denominator) <= 1e-12:
            continue
        parameter = float(np.dot(item.station_point - midpoint, seed.direction) / denominator)
        intersection = midpoint + parameter * direction
        distance, arc = _closest_arc(item.contour, intersection)
        center_error = float(np.linalg.norm(intersection - item.locus))
        width = item.s75 - item.s25
        if center_error > constants.transverse_center_threshold:
            continue
        if not (item.s25 <= arc <= item.s75):
            continue
        if abs(arc - item.s50) > 0.25 * width + constants.resampling_step:
            continue
        certified.append(item)
        errors.append(center_error)
    if len(certified) < MINIMUM_CERTIFIED_STATIONS:
        return None
    fractions = [item.station_fraction for item in certified]
    if max(fractions) - min(fractions) < MINIMUM_STATION_COVERAGE:
        return None
    if longest_run(item.station_index for item in certified) < MINIMUM_LONGEST_RUN:
        return None
    if max(errors) > constants.transverse_center_threshold or float(np.median(errors)) > constants.maximum_median_center_error:
        return None
    if not allow_refit:
        return start, end, certified, errors, 0.0
    fit_midpoint, fit_direction, residuals = _fit_line(np.asarray([item.locus for item in certified]))
    if _angle_degrees(fit_direction, seed.direction) > MAXIMUM_DIRECTION_DRIFT_DEGREES:
        return None
    p0 = float(np.dot(seed.line_start - fit_midpoint, fit_direction))
    p1 = float(np.dot(seed.line_end - fit_midpoint, fit_direction))
    refit_start = fit_midpoint + min(p0, p1) * fit_direction
    refit_end = fit_midpoint + max(p0, p1) * fit_direction
    final = certify_line(refit_start, refit_end, seed, certified, constants, allow_refit=False)
    if final is None:
        return None
    return final[0], final[1], final[2], final[3], float(np.median(residuals))


def evaluate_seed(mesh: MeshData, seed: Seed) -> tuple[Proposal | None, str, dict[str, Any]]:
    constants = mesh.constants
    if seed.span < constants.minimum_seed_span:
        return None, "REJECT_SHORT_SEED", {"stations": []}
    evidence: list[StationEvidence] = []
    station_audit: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for station_index, fraction in enumerate(STATION_FRACTIONS):
        point = seed.line_start + fraction * (seed.line_end - seed.line_start)
        item, reason = select_station_evidence(mesh, seed, station_index, fraction, point)
        reason_counts[reason] += 1
        if item is not None:
            evidence.append(item)
            station_audit.append(
                {
                    "station_index": station_index,
                    "fraction": fraction,
                    "status": "VALID_TURN_LOCUS",
                    "station_point": point.tolist(),
                    "locus": item.locus.tolist(),
                    "s25": item.s25,
                    "s50": item.s50,
                    "s75": item.s75,
                    "turn_degrees": item.turn_degrees,
                    "turn_purity": item.turn_purity,
                }
            )
        else:
            station_audit.append({"station_index": station_index, "fraction": fraction, "status": reason, "station_point": point.tolist()})
    if len(evidence) < MINIMUM_CERTIFIED_STATIONS:
        if reason_counts["REJECT_MULTIPLE_TURN_EVENTS"]:
            reason = "REJECT_MULTIPLE_TURN_EVENTS"
        elif reason_counts["REJECT_DISTRIBUTED_OR_IMPURE_TURN"]:
            reason = "REJECT_DISTRIBUTED_OR_IMPURE_TURN"
        elif reason_counts["REJECT_COPLANAR_OR_BRANCH_AMBIGUITY"]:
            reason = "REJECT_COPLANAR_OR_BRANCH_AMBIGUITY"
        elif reason_counts["REJECT_NO_PERSISTENT_TURN"]:
            reason = "REJECT_NO_PERSISTENT_TURN"
        elif reason_counts["REJECT_NO_SECTION"]:
            reason = "REJECT_NO_SECTION"
        else:
            reason = "REJECT_INSUFFICIENT_STATIONS"
        return None, reason, {"stations": station_audit}

    reconstructed = reconstruct_line(seed, evidence, constants)
    reconstructed_certificate = None
    if reconstructed is not None:
        reconstructed_certificate = certify_line(reconstructed[0], reconstructed[1], seed, evidence, constants, allow_refit=True)

    if seed.origin_class == "V52":
        original = certify_line(seed.line_start, seed.line_end, seed, evidence, constants, allow_refit=False)
        if original is not None:
            line_start = np.asarray(seed.original_row["line_start"], dtype=np.float64)
            line_end = np.asarray(seed.original_row["line_end"], dtype=np.float64)
            proposal_class = "HOLD_V52"
            certificate = original
        elif reconstructed_certificate is not None:
            line_start, line_end = reconstructed_certificate[0], reconstructed_certificate[1]
            proposal_class = "RECONSTRUCT_V52"
            certificate = reconstructed_certificate
        else:
            return None, "REJECT_TRANSVERSE_CENTERING", {"stations": station_audit}
    else:
        if reconstructed_certificate is None:
            return None, "REJECT_LINE_CONSENSUS" if reconstructed is None else "REJECT_TRANSVERSE_CENTERING", {"stations": station_audit}
        line_start, line_end = reconstructed_certificate[0], reconstructed_certificate[1]
        proposal_class = "RECONSTRUCT_POOL"
        certificate = reconstructed_certificate
    certified = certificate[2]
    proposal = Proposal(
        output_class=proposal_class,
        seed=seed,
        line_start=np.asarray(line_start, dtype=np.float64),
        line_end=np.asarray(line_end, dtype=np.float64),
        certified=certified,
        station_errors=list(float(value) for value in certificate[3]),
        median_fit_error=float(certificate[4]),
        median_turn_iqr_width=float(np.median([item.turn_iqr_width for item in certified])),
        median_turn_degrees=float(np.median([item.turn_degrees for item in certified])),
        median_turn_purity=float(np.median([item.turn_purity for item in certified])),
    )
    return proposal, "ACCEPT", {"stations": station_audit}


def _projected_overlap(first: Proposal, second: Proposal) -> float:
    direction = canonical_direction(first.direction + (second.direction if np.dot(first.direction, second.direction) >= 0.0 else -second.direction))
    first_values = sorted(float(np.dot(point, direction)) for point in (first.line_start, first.line_end))
    second_values = sorted(float(np.dot(point, direction)) for point in (second.line_start, second.line_end))
    overlap = max(0.0, min(first_values[1], second_values[1]) - max(first_values[0], second_values[0]))
    shorter = min(first_values[1] - first_values[0], second_values[1] - second_values[0])
    return overlap / max(shorter, 1e-12)


def collapse_output_duplicates(proposals: list[Proposal], constants: Constants) -> tuple[list[Proposal], set[int]]:
    union = _UnionFind(len(proposals))
    for index, first in enumerate(proposals):
        for other_index in range(index + 1, len(proposals)):
            second = proposals[other_index]
            if _angle_degrees(first.direction, second.direction) > DUPLICATE_DIRECTION_DEGREES:
                continue
            if _symmetric_midpoint_line_distance(first, second) > constants.duplicate_distance:
                continue
            if _projected_overlap(first, second) < DUPLICATE_OVERLAP:
                continue
            union.union(index, other_index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(proposals)):
        groups[union.find(index)].append(index)
    winners: list[Proposal] = []
    losers: set[int] = set()
    class_rank = {"HOLD_V52": 0, "RECONSTRUCT_V52": 1, "RECONSTRUCT_POOL": 1}
    for _, indices in sorted(groups.items()):
        winner = min(
            indices,
            key=lambda index: (
                class_rank[proposals[index].output_class],
                -len(proposals[index].certified),
                -proposals[index].coverage,
                float(np.median(proposals[index].station_errors)),
                proposals[index].median_fit_error / constants.bbox_diagonal,
                proposals[index].median_turn_iqr_width / constants.search_half_width,
                -proposals[index].median_turn_degrees,
                proposals[index].seed.origin_file_ordinal,
                proposals[index].seed.row_ordinal,
            ),
        )
        winners.append(proposals[winner])
        losers.update(index for index in indices if index != winner)
    winners.sort(key=lambda proposal: (
        0 if proposal.seed.origin_class == "V52" else 1,
        proposal.seed.row_ordinal if proposal.seed.origin_class == "V52" else 0,
        float(proposal.direction[0]), float(proposal.direction[1]), float(proposal.direction[2]),
        float(proposal.midpoint[0]), float(proposal.midpoint[1]), float(proposal.midpoint[2]), float(proposal.span),
    ))
    return winners, losers


def _proposal_output(proposal: Proposal) -> dict[str, Any]:
    if proposal.output_class == "HOLD_V52":
        row = dict(proposal.seed.original_row)
        return row
    return {
        "bend_id": f"V56-{proposal.seed.origin_file_ordinal:01d}-{proposal.seed.row_ordinal:04d}",
        "line_start": [float(value) for value in proposal.line_start],
        "line_end": [float(value) for value in proposal.line_end],
        "selection_route": "canonical_turn_median_line_reconstruction_v56",
        "source_kind": "source_turn_median_locus",
    }


def compose_v56(
    v52_result: dict[str, Any],
    pool: dict[str, Any],
    mesh: MeshData,
    *,
    include_visual_audit: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one deterministic V56 scan without consulting labels or identity."""

    started = time.perf_counter()
    if v52_result.get("source_only") is not True or v52_result.get("research_only") is not True:
        raise V56TerminalFailure("INVALID_V52_CONTRACT")
    rows: list[tuple[dict[str, Any], str, int, int]] = []
    # V52 is assigned the first frozen origin ordinal so exact duplicates retain
    # the safety-hold representation and its byte-preserving endpoints.
    for row_ordinal, row in enumerate(v52_result.get("predictions") or ()):
        rows.append((row, "V52", 0, row_ordinal))
    for row_ordinal, row in enumerate(pool.get("line_objects") or ()):
        rows.append((row, "POOL", 1, row_ordinal))
    if len(rows) > MAXIMUM_INPUT_ROWS:
        raise V56TerminalFailure("INPUT_ROW_CAP_EXCEEDED")
    seeds: list[Seed] = []
    schema_counts: Counter[str] = Counter()
    for row, origin_class, origin_file_ordinal, row_ordinal in rows:
        try:
            seed = materialize_seed(
                row,
                bbox_diagonal=mesh.bbox_diagonal,
                median_mesh_edge=mesh.median_mesh_edge,
                origin_class=origin_class,
                origin_file_ordinal=origin_file_ordinal,
                row_ordinal=row_ordinal,
            )
        except SeedSchemaError as error:
            raise V56TerminalFailure(f"SCHEMA_AUDIT_FAILURE:{error.reason}") from error
        schema_counts[seed.schema_code] += 1
        seeds.append(seed)
    canonical_seeds, multiplicities = collapse_exact_duplicates(seeds, mesh.constants)
    rejection_counts: Counter[str] = Counter()
    proposals: list[Proposal] = []
    visual_rows: list[dict[str, Any]] = []
    for canonical_ordinal, seed in enumerate(canonical_seeds):
        if time.perf_counter() - started >= MAXIMUM_WALL_SECONDS:
            raise V56TerminalFailure("SCAN_WALL_TIME_LIMIT")
        proposal, reason, audit = evaluate_seed(mesh, seed)
        audit.update(
            {
                "canonical_seed_ordinal": canonical_ordinal,
                "origin_class": seed.origin_class,
                "origin_file_ordinal": seed.origin_file_ordinal,
                "row_ordinal": seed.row_ordinal,
                "seed_line_start": seed.line_start.tolist(),
                "seed_line_end": seed.line_end.tolist(),
                "decision": reason,
            }
        )
        if proposal is None:
            rejection_counts[reason] += 1
        else:
            proposals.append(proposal)
            audit["proposal_class"] = proposal.output_class
            audit["output_line_start"] = proposal.line_start.tolist()
            audit["output_line_end"] = proposal.line_end.tolist()
        if include_visual_audit:
            visual_rows.append(audit)
    winners, losers = collapse_output_duplicates(proposals, mesh.constants)
    rejection_counts["REJECT_DUPLICATE_LOSER"] += len(losers)
    outputs = [_proposal_output(proposal) for proposal in winners]
    wall_ms = (time.perf_counter() - started) * 1000.0
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not sys_platform_is_darwin():
        peak_rss *= 1024
    cache_outputs = []
    for proposal in winners:
        cache_outputs.append(
            {
                "class": proposal.output_class,
                "source_seed_ordinal": int(proposal.seed.row_ordinal),
                "line_start": [float(value) for value in proposal.line_start],
                "line_end": [float(value) for value in proposal.line_end],
                "certified_station_count": len(proposal.certified),
                "longitudinal_coverage": proposal.coverage,
                "longest_station_run": proposal.longest_run,
                "median_turn_degrees": proposal.median_turn_degrees,
                "median_turn_purity": proposal.median_turn_purity,
                "median_center_error_over_B": float(np.median(proposal.station_errors)) / mesh.bbox_diagonal,
                "median_fit_error_over_B": proposal.median_fit_error / mesh.bbox_diagonal,
            }
        )
    cache = {
        "version": 1,
        "bbox_diagonal": mesh.bbox_diagonal,
        "median_mesh_edge": mesh.median_mesh_edge,
        "schema_counts": dict(sorted(schema_counts.items())),
        "input_row_count": len(rows),
        "exact_duplicate_count": len(rows) - len(canonical_seeds),
        "canonical_seed_count": len(canonical_seeds),
        "candidate_rejection_counts": {code: int(rejection_counts.get(code, 0)) for code in REJECTION_CODES},
        "proposal_count": len(proposals),
        "output_count": len(outputs),
        "outputs": cache_outputs,
        "wall_ms": wall_ms,
        "peak_rss_bytes": peak_rss,
    }
    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "profile": PROFILE,
        "candidate_count": len(outputs),
        "predictions": outputs,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "filename_used": False,
        "source_only": True,
        "research_only": True,
        "canonical_turn_median_line_reconstruction_audit": cache,
    }
    visual = {
        "schema_version": 1,
        "profile": PROFILE,
        "constants": mesh.constants.to_dict(),
        "seeds": visual_rows,
    }
    return result, visual


def sys_platform_is_darwin() -> bool:
    import sys

    return sys.platform == "darwin"
