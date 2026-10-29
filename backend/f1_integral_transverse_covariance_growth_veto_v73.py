"""V73 integral transverse covariance growth veto (ITCGV_V1).

Research-only, deletion-only selector over a frozen candidate stream.  The
module reads only raw source coordinates and the candidate-local geometric
tuple ``(anchor, axis, endpoint0, endpoint1, spacing)``.  It deliberately does
not read labels, gold, CAD, filenames, identities, candidate scores, route
provenance, benchmark outcomes, other candidates, normals, faces, or mesh
topology.

V73 is independent of V72/CASL.  Its sole scientific action is a conservative
veto when all-scale transverse covariance is persistently compatible with one
flat or smoothly curved surface.  Unsupported and ambiguous evidence keeps the
candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ARCHITECTURE = "V73_INTEGRAL_TRANSVERSE_COVARIANCE_GROWTH_VETO_V1"
SHORT_NAME = "ITCGV_V1"
SCHEMA = "sherman_qc_f1_v73_itcgv_v1_audit"
RULING_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/2026-08-16_gpt56_pro_post_v72_successor_ruling_verbatim.md"
)
RULING_SHA256 = "739164fe9619d3913e332b60053d602f607722483afce738f5775cec877c7e10"

MAX_SOURCE_RECORDS = 200_000
MAX_CANDIDATES = 64
MAX_OCCUPIED_CELLS_PER_CANDIDATE = 200_000
MAX_CANDIDATE_SOURCE_TESTS = 12_800_000
MAX_CELL_FINALIZATIONS = 12_800_000
MAX_CELL_STATION_UPDATES = 38_400_000
MAX_METRIC_TUPLES = 4_608
MAX_RESULT_CANDIDATES = 64
MAX_SCRATCH_BYTES = 536_870_912
MAX_RESULT_BYTES = 1_048_576
MAX_RSS_BYTES = 17_179_869_184
MAX_WALL_SECONDS = 300.0

RADII = (8, 12, 18, 27, 40, 60, 90, 135)
RADII_SQUARED = tuple(value * value for value in RADII)
STATION_COUNT = 9
SCALE_COUNT = 8
AXIAL_HALF_WIDTH = 4.0
LENGTH_FLOOR = 40.0
MIN_INNER_SUPPORT = 24
MIN_OUTER_SUPPORT = 96
MIN_OUTER_HALF_SUPPORT = 24
PSD_EPSILON = 2.0**-40
FLAT_MAX = 2.0**-13
INNER_MIN = 2.0**-24
OUTER_MAX = 2.0**-1
COUPLING_MAX = 2.0**-4
GROWTH_NUMERATOR = 9
GROWTH_DENOMINATOR = 8
NORMALIZED_SPREAD_FACTOR = 2

DECISION_VETO = "VETO_SMOOTH_COVARIANCE"
DECISION_KEEP_FEATURE = "KEEP_FEATURE_INCOMPATIBLE"
DECISION_KEEP_SHORT = "KEEP_ABSTAIN_SHORT"
DECISION_KEEP_SUPPORT = "KEEP_ABSTAIN_SUPPORT"

COMPLETE = "COMPLETE"
ABORT = "ABORT"


class V73InputError(RuntimeError):
    """Input violated the frozen scan-only contract."""


class V73ResourceAbort(RuntimeError):
    """A deterministic hard cap was reached."""


class V73ArithmeticAbort(RuntimeError):
    """A non-finite or invalid arithmetic state was observed."""


class V73SerializationAbort(RuntimeError):
    """Canonical output could not be produced atomically."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Canonical compact ASCII JSON with exactly one final newline."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as error:
        raise V73SerializationAbort("CANONICAL_ENCODING_FAILURE") from error
    if len(encoded) > MAX_RESULT_BYTES:
        raise V73ResourceAbort("RESULT_BYTE_CAP")
    return encoded


def atomic_write_canonical(path: Path, value: Any) -> bytes:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise V73SerializationAbort("CANONICAL_BYTE_VERIFICATION_FAILURE")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_runtime(started: float) -> None:
    if time.perf_counter() - started > MAX_WALL_SECONDS:
        raise V73ResourceAbort("SCAN_WALL_CAP")
    if _rss_bytes() >= MAX_RSS_BYTES:
        raise V73ResourceAbort("PROCESS_RSS_CAP")


def _finite_float(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise V73InputError(code) from error
    if not math.isfinite(result):
        raise V73InputError(code)
    return result


def _vector3(value: Any, code: str) -> tuple[float, float, float]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise V73InputError(code) from error
    if len(values) != 3:
        raise V73InputError(code)
    return tuple(_finite_float(item, code) for item in values)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1] * right[2] - left[2] * right[1]),
        float(left[2] * right[0] - left[0] * right[2]),
        float(left[0] * right[1] - left[1] * right[0]),
    )


def _subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[0] - right[0]),
        float(left[1] - right[1]),
        float(left[2] - right[2]),
    )


def _total_order_key(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits ^ (1 << 63)


def _hex(value: float) -> str:
    if not math.isfinite(value):
        raise V73ArithmeticAbort("NONFINITE_METRIC")
    return float(value).hex()


def _constants_payload() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "axial_half_width": AXIAL_HALF_WIDTH.hex(),
        "coupling_max": COUPLING_MAX.hex(),
        "flat_max": FLAT_MAX.hex(),
        "growth_denominator": GROWTH_DENOMINATOR,
        "growth_numerator": GROWTH_NUMERATOR,
        "inner_min": INNER_MIN.hex(),
        "length_floor": LENGTH_FLOOR.hex(),
        "minimum_inner_support": MIN_INNER_SUPPORT,
        "minimum_outer_half_support": MIN_OUTER_HALF_SUPPORT,
        "minimum_outer_support": MIN_OUTER_SUPPORT,
        "normalized_spread_factor": NORMALIZED_SPREAD_FACTOR,
        "outer_max": OUTER_MAX.hex(),
        "psd_epsilon": PSD_EPSILON.hex(),
        "radii": list(RADII),
        "scale_count": SCALE_COUNT,
        "station_count": STATION_COUNT,
    }


CONSTANTS_SHA256 = _sha256(
    json.dumps(_constants_payload(), sort_keys=True, separators=(",", ":")).encode("ascii")
)


@dataclass(frozen=True)
class CanonicalSource:
    points: tuple[tuple[float, float, float], ...]
    source_sha256: str
    scratch_bytes: int


def canonical_source(vertices: Iterable[Sequence[float]], started: float) -> CanonicalSource:
    points: list[tuple[float, float, float]] = []
    try:
        iterator = iter(vertices)
    except TypeError as error:
        raise V73InputError("MALFORMED_SOURCE") from error
    for ordinal, row in enumerate(iterator):
        if ordinal >= MAX_SOURCE_RECORDS:
            raise V73ResourceAbort("SOURCE_CAP")
        points.append(_vector3(row, "NONFINITE_SOURCE"))
        if ordinal and ordinal % 16_384 == 0:
            _check_runtime(started)
    points.sort(key=lambda point: tuple(_total_order_key(value) for value in point))
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack(">ddd", *point))
    frozen = tuple(points)
    scratch_bytes = sys.getsizeof(frozen)
    for point in frozen:
        scratch_bytes += sys.getsizeof(point)
        scratch_bytes += sum(sys.getsizeof(value) for value in point)
    if scratch_bytes > MAX_SCRATCH_BYTES:
        raise V73ResourceAbort("SCRATCH_BYTE_CAP")
    return CanonicalSource(frozen, digest.hexdigest(), scratch_bytes)


@dataclass(frozen=True)
class CandidateGeometry:
    ordinal: int
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    endpoint0: tuple[float, float, float]
    endpoint1: tuple[float, float, float]
    spacing: float
    closest_origin: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    alpha: float
    beta: float
    length: float
    geometry_sha256: str


def _candidate_components(
    row: Mapping[str, Any],
    ordinal: int,
    default_spacing: float | None,
) -> CandidateGeometry:
    try:
        endpoint0 = _vector3(row["line_start"], "MALFORMED_CANDIDATE")
        endpoint1 = _vector3(row["line_end"], "MALFORMED_CANDIDATE")
    except (KeyError, TypeError) as error:
        raise V73InputError("MALFORMED_CANDIDATE") from error
    anchor = _vector3(row.get("line_anchor", endpoint0), "MALFORMED_CANDIDATE")
    axis_value = row.get("line_axis")
    axis_raw = _subtract(endpoint1, endpoint0) if axis_value is None else _vector3(axis_value, "MALFORMED_CANDIDATE")
    spacing_value = row.get("spacing", default_spacing)
    if spacing_value is None:
        raise V73InputError("MISSING_SPACING")
    spacing = _finite_float(spacing_value, "INVALID_SPACING")
    if spacing <= 0.0:
        raise V73InputError("INVALID_SPACING")

    norm2 = _dot(axis_raw, axis_raw)
    if not math.isfinite(norm2) or norm2 <= 0.0:
        raise V73InputError("ZERO_AXIS")
    norm = math.sqrt(norm2)
    axis = tuple(float(value / norm) for value in axis_raw)
    dominant = 0
    dominant_abs = abs(axis[0])
    for index in (1, 2):
        current = abs(axis[index])
        if current > dominant_abs:
            dominant = index
            dominant_abs = current
    if axis[dominant] < 0.0:
        axis = tuple(float(-value) for value in axis)

    projection = _dot(anchor, axis)
    closest_origin = tuple(float(anchor[index] - projection * axis[index]) for index in range(3))
    t0 = _dot(_subtract(endpoint0, closest_origin), axis)
    t1 = _dot(_subtract(endpoint1, closest_origin), axis)
    alpha = min(t0, t1) / spacing
    beta = max(t0, t1) / spacing
    length = beta - alpha
    if not all(math.isfinite(value) for value in (*axis, *closest_origin, alpha, beta, length)):
        raise V73InputError("NONFINITE_CANDIDATE")
    if length <= 0.0:
        raise V73InputError("ZERO_LENGTH_CANDIDATE")

    q = min((0, 1, 2), key=lambda index: (abs(axis[index]), index))
    denominator2 = 1.0 - axis[q] * axis[q]
    if denominator2 <= 0.0 or not math.isfinite(denominator2):
        raise V73ArithmeticAbort("FRAME_CONSTRUCTION_FAILURE")
    denominator = math.sqrt(denominator2)
    basis = [0.0, 0.0, 0.0]
    basis[q] = 1.0
    u = tuple(float((basis[index] - axis[q] * axis[index]) / denominator) for index in range(3))
    v = _cross(axis, u)
    if not all(math.isfinite(value) for value in (*u, *v)):
        raise V73ArithmeticAbort("FRAME_CONSTRUCTION_FAILURE")

    geometry_payload = {
        "alpha": _hex(alpha),
        "axis": [_hex(value) for value in axis],
        "beta": _hex(beta),
        "closest_origin": [_hex(value) for value in closest_origin],
        "spacing": _hex(spacing),
        "u": [_hex(value) for value in u],
        "v": [_hex(value) for value in v],
    }
    geometry_digest = _sha256(
        json.dumps(geometry_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    return CandidateGeometry(
        ordinal,
        anchor,
        axis,
        endpoint0,
        endpoint1,
        spacing,
        closest_origin,
        u,
        v,
        alpha,
        beta,
        length,
        geometry_digest,
    )


def canonical_candidates(
    rows: Sequence[Mapping[str, Any]],
    default_spacing: float | None,
) -> tuple[CandidateGeometry, ...]:
    if len(rows) > MAX_CANDIDATES:
        raise V73ResourceAbort("CANDIDATE_CAP")
    if not isinstance(rows, Sequence):
        raise V73InputError("MALFORMED_CANDIDATE_STREAM")
    return tuple(_candidate_components(row, ordinal, default_spacing) for ordinal, row in enumerate(rows))


@dataclass(slots=True)
class _Cell:
    count: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0

    def add(self, x: float, y: float, z: float) -> None:
        self.count += 1
        self.sx += x
        self.sy += y
        self.sz += z
        if not all(math.isfinite(value) for value in (self.sx, self.sy, self.sz)):
            raise V73ArithmeticAbort("CELL_ACCUMULATION_NONFINITE")

    def centroid(self) -> tuple[float, float, float]:
        inverse = 1.0 / self.count
        return (self.sx * inverse, self.sy * inverse, self.sz * inverse)


@dataclass(slots=True)
class _Neumaier:
    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        candidate = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - candidate) + value
        else:
            self.correction += (value - candidate) + self.total
        self.total = candidate
        if not math.isfinite(self.total) or not math.isfinite(self.correction):
            raise V73ArithmeticAbort("MOMENT_ACCUMULATION_NONFINITE")

    def value(self) -> float:
        result = self.total + self.correction
        if not math.isfinite(result):
            raise V73ArithmeticAbort("MOMENT_ACCUMULATION_NONFINITE")
        return result


@dataclass(slots=True)
class _Moments:
    count: int = 0
    negative: int = 0
    positive: int = 0
    sx: _Neumaier = field(default_factory=_Neumaier)
    sy: _Neumaier = field(default_factory=_Neumaier)
    sz: _Neumaier = field(default_factory=_Neumaier)
    sxx: _Neumaier = field(default_factory=_Neumaier)
    syy: _Neumaier = field(default_factory=_Neumaier)
    szz: _Neumaier = field(default_factory=_Neumaier)
    sxy: _Neumaier = field(default_factory=_Neumaier)
    sxz: _Neumaier = field(default_factory=_Neumaier)
    syz: _Neumaier = field(default_factory=_Neumaier)

    def add(self, x: float, y: float, z: float) -> None:
        self.count += 1
        if z < 0.0:
            self.negative += 1
        elif z > 0.0:
            self.positive += 1
        self.sx.add(x)
        self.sy.add(y)
        self.sz.add(z)
        self.sxx.add(x * x)
        self.syy.add(y * y)
        self.szz.add(z * z)
        self.sxy.add(x * y)
        self.sxz.add(x * z)
        self.syz.add(y * z)


@dataclass(frozen=True)
class ScaleMetric:
    count: int
    negative: int
    positive: int
    f: float
    g: float
    coupling_left: float
    coupling_right: float
    valid: bool


def _radial_shell(radius2: float) -> int | None:
    for scale, bound2 in enumerate(RADII_SQUARED):
        if radius2 <= bound2:
            return scale
    return None


def _within_axial_slab(zeta: float) -> bool:
    return abs(zeta) <= AXIAL_HALF_WIDTH


def _psd_clamp(raw_determinant: float, g_value: float) -> tuple[float, bool]:
    tolerance = PSD_EPSILON * g_value
    determinant = raw_determinant
    if determinant < 0.0:
        if -determinant <= tolerance:
            determinant = 0.0
        else:
            return determinant, False
    four_d = 4.0 * determinant
    if four_d > g_value:
        if four_d - g_value <= tolerance:
            determinant = g_value / 4.0
        else:
            return determinant, False
    return determinant, True


def _flat_ratio(f_value: float, g_value: float) -> bool:
    return (1 << 13) * f_value <= g_value


def _inner_ratio(f_value: float, g_value: float) -> bool:
    return (1 << 24) * f_value >= g_value


def _outer_ratio(f_value: float, g_value: float) -> bool:
    return 2.0 * f_value <= g_value


def _growth_ratio(
    current_f: float,
    current_g: float,
    following_f: float,
    following_g: float,
) -> bool:
    return (
        GROWTH_DENOMINATOR * following_f * current_g
        >= GROWTH_NUMERATOR * current_f * following_g
    )


def _spread_ratio(
    left_f: float,
    left_g: float,
    left_radius2: float,
    right_f: float,
    right_g: float,
    right_radius2: float,
) -> bool:
    return (
        left_f * right_g * right_radius2
        <= NORMALIZED_SPREAD_FACTOR * right_f * left_g * left_radius2
    )


def _metric(moments: _Moments) -> ScaleMetric:
    count = moments.count
    if count == 0:
        return ScaleMetric(0, 0, 0, 0.0, 0.0, 0.0, 0.0, False)
    inverse = 1.0 / count
    mx = moments.sx.value() * inverse
    my = moments.sy.value() * inverse
    mz = moments.sz.value() * inverse
    cxx = moments.sxx.value() * inverse - mx * mx
    cyy = moments.syy.value() * inverse - my * my
    czz = moments.szz.value() * inverse - mz * mz
    cxy = moments.sxy.value() * inverse - mx * my
    cxz = moments.sxz.value() * inverse - mx * mz
    cyz = moments.syz.value() * inverse - my * mz
    values = (cxx, cyy, czz, cxy, cxz, cyz)
    if not all(math.isfinite(value) for value in values):
        raise V73ArithmeticAbort("NONFINITE_COVARIANCE")
    transverse = cxx + cyy
    raw_determinant = cxx * cyy - cxy * cxy
    g_value = transverse * transverse
    if not all(math.isfinite(value) for value in (transverse, raw_determinant, g_value)):
        raise V73ArithmeticAbort("NONFINITE_COVARIANCE")
    if transverse <= 0.0 or czz <= 0.0 or g_value <= 0.0:
        return ScaleMetric(count, moments.negative, moments.positive, 0.0, g_value, 0.0, 0.0, False)
    determinant, valid = _psd_clamp(raw_determinant, g_value)
    four_d = 4.0 * determinant
    coupling_left = 16.0 * (cxz * cxz + cyz * cyz)
    coupling_right = czz * transverse
    if not all(math.isfinite(value) for value in (four_d, coupling_left, coupling_right)):
        raise V73ArithmeticAbort("NONFINITE_COVARIANCE")
    valid = valid and coupling_left <= coupling_right
    return ScaleMetric(
        count,
        moments.negative,
        moments.positive,
        four_d,
        g_value,
        coupling_left,
        coupling_right,
        valid,
    )


@dataclass(frozen=True)
class CandidateEvaluation:
    ordinal: int
    decision: str
    eligibility_mask: int
    flat_mask: int
    smooth_growth_mask: int
    smoothlike_mask: int
    longest_smoothlike_run: int
    metric_sha256: str
    geometry_sha256: str
    metrics: tuple[tuple[ScaleMetric, ...], ...]

    def record(self) -> dict[str, Any]:
        return {
            "candidate_metric_sha256": self.metric_sha256,
            "canonical_candidate_ordinal": self.ordinal,
            "decision": self.decision,
            "eligibility_mask": self.eligibility_mask,
            "flat_mask": self.flat_mask,
            "longest_smoothlike_run": self.longest_smoothlike_run,
            "smooth_growth_mask": self.smooth_growth_mask,
            "smoothlike_mask": self.smoothlike_mask,
        }


def _longest_run(mask: int) -> int:
    longest = current = 0
    for station in range(STATION_COUNT):
        if mask & (1 << station):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _metric_digest(metrics: tuple[tuple[ScaleMetric, ...], ...]) -> str:
    payload = []
    for station in metrics:
        payload.append(
            [
                {
                    "coupling_left": _hex(metric.coupling_left),
                    "coupling_right": _hex(metric.coupling_right),
                    "f": _hex(metric.f),
                    "g": _hex(metric.g),
                    "n": metric.count,
                    "negative": metric.negative,
                    "positive": metric.positive,
                    "valid": metric.valid,
                }
                for metric in station
            ]
        )
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))


def _short_evaluation(candidate: CandidateGeometry) -> CandidateEvaluation:
    empty = tuple(tuple() for _ in range(STATION_COUNT))
    digest = _sha256(
        json.dumps(
            {"decision": DECISION_KEEP_SHORT, "geometry_sha256": candidate.geometry_sha256},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return CandidateEvaluation(
        candidate.ordinal,
        DECISION_KEEP_SHORT,
        0,
        0,
        0,
        0,
        0,
        digest,
        candidate.geometry_sha256,
        empty,
    )


def _candidate_cells(
    candidate: CandidateGeometry,
    source: CanonicalSource,
    counters: dict[str, int],
    started: float,
) -> dict[tuple[int, int, int], _Cell]:
    counters["candidate_source_tests"] += len(source.points)
    if counters["candidate_source_tests"] > MAX_CANDIDATE_SOURCE_TESTS:
        raise V73ResourceAbort("CANDIDATE_SOURCE_TEST_CAP")
    cells: dict[tuple[int, int, int], _Cell] = {}
    lower = candidate.alpha - AXIAL_HALF_WIDTH
    upper = candidate.beta + AXIAL_HALF_WIDTH
    outer2 = float(RADII_SQUARED[-1])
    spacing = candidate.spacing
    for ordinal, point in enumerate(source.points):
        relative = _subtract(point, candidate.closest_origin)
        xi = _dot(relative, candidate.u) / spacing
        eta = _dot(relative, candidate.v) / spacing
        lam = _dot(relative, candidate.axis) / spacing
        radius2 = xi * xi + eta * eta
        if lower <= lam <= upper and radius2 <= outer2:
            key = (math.floor(xi), math.floor(eta), math.floor(lam))
            cell = cells.get(key)
            if cell is None:
                if len(cells) >= MAX_OCCUPIED_CELLS_PER_CANDIDATE:
                    raise V73ResourceAbort("CANDIDATE_CELL_CAP")
                cell = _Cell()
                cells[key] = cell
            cell.add(xi, eta, lam)
        if ordinal and ordinal % 32_768 == 0:
            _check_runtime(started)
    counters["cell_finalizations"] += len(cells)
    if counters["cell_finalizations"] > MAX_CELL_FINALIZATIONS:
        raise V73ResourceAbort("CELL_FINALIZATION_CAP")
    measured_scratch = source.scratch_bytes + sys.getsizeof(cells)
    for key, cell in cells.items():
        measured_scratch += sys.getsizeof(key)
        measured_scratch += sum(sys.getsizeof(value) for value in key)
        measured_scratch += sys.getsizeof(cell)
    measured_scratch += sys.getsizeof(list(cells))
    measured_scratch += STATION_COUNT * SCALE_COUNT * 1024
    if measured_scratch > MAX_SCRATCH_BYTES:
        raise V73ResourceAbort("SCRATCH_BYTE_CAP")
    return cells


def _evaluate_candidate(
    candidate: CandidateGeometry,
    source: CanonicalSource,
    counters: dict[str, int],
    started: float,
) -> CandidateEvaluation:
    if candidate.length < LENGTH_FLOOR:
        return _short_evaluation(candidate)

    cells = _candidate_cells(candidate, source, counters, started)
    station_positions = tuple(
        candidate.alpha + AXIAL_HALF_WIDTH + station * (candidate.length - 8.0) / 8.0
        for station in range(STATION_COUNT)
    )
    accumulators = [[_Moments() for _ in range(SCALE_COUNT)] for _ in range(STATION_COUNT)]
    for key in sorted(cells):
        xi, eta, lam = cells[key].centroid()
        radius2 = xi * xi + eta * eta
        shell = _radial_shell(radius2)
        if shell is None:
            continue
        for station, position in enumerate(station_positions):
            zeta = lam - position
            if _within_axial_slab(zeta):
                counters["cell_station_updates"] += 1
                if counters["cell_station_updates"] > MAX_CELL_STATION_UPDATES:
                    raise V73ResourceAbort("CELL_STATION_UPDATE_CAP")
                for scale in range(shell, SCALE_COUNT):
                    accumulators[station][scale].add(xi, eta, zeta)

    metrics = tuple(tuple(_metric(value) for value in station) for station in accumulators)
    counters["metric_tuples"] += STATION_COUNT * SCALE_COUNT
    if counters["metric_tuples"] > MAX_METRIC_TUPLES:
        raise V73ResourceAbort("METRIC_TUPLE_CAP")

    eligibility_mask = 0
    flat_mask = 0
    growth_mask = 0
    smoothlike_mask = 0
    for station_index, station in enumerate(metrics):
        eligible = (
            station[0].count >= MIN_INNER_SUPPORT
            and station[-1].count >= MIN_OUTER_SUPPORT
            and station[-1].negative >= MIN_OUTER_HALF_SUPPORT
            and station[-1].positive >= MIN_OUTER_HALF_SUPPORT
            and all(metric.valid for metric in station)
        )
        if not eligible:
            continue
        eligibility_mask |= 1 << station_index
        flat = all(_flat_ratio(metric.f, metric.g) for metric in station)
        if flat:
            flat_mask |= 1 << station_index

        growth = _inner_ratio(station[0].f, station[0].g)
        growth = growth and _outer_ratio(station[-1].f, station[-1].g)
        if growth:
            for scale in range(SCALE_COUNT - 1):
                current = station[scale]
                following = station[scale + 1]
                if not _growth_ratio(current.f, current.g, following.f, following.g):
                    growth = False
                    break
        if growth:
            for left in range(SCALE_COUNT):
                for right in range(SCALE_COUNT):
                    if not _spread_ratio(
                        station[left].f,
                        station[left].g,
                        float(RADII_SQUARED[left]),
                        station[right].f,
                        station[right].g,
                        float(RADII_SQUARED[right]),
                    ):
                        growth = False
                        break
                if not growth:
                    break
        if growth:
            growth_mask |= 1 << station_index
        if flat or growth:
            smoothlike_mask |= 1 << station_index

    smooth_count = smoothlike_mask.bit_count()
    longest = _longest_run(smoothlike_mask)
    if smooth_count >= 8 and longest >= 7:
        decision = DECISION_VETO
    elif eligibility_mask != (1 << STATION_COUNT) - 1:
        decision = DECISION_KEEP_SUPPORT
    else:
        decision = DECISION_KEEP_FEATURE
    return CandidateEvaluation(
        candidate.ordinal,
        decision,
        eligibility_mask,
        flat_mask,
        growth_mask,
        smoothlike_mask,
        longest,
        _metric_digest(metrics),
        candidate.geometry_sha256,
        metrics,
    )


def _input_manifest_sha(source: CanonicalSource, candidates: Sequence[CandidateGeometry]) -> str:
    payload = {
        "candidate_geometry_sha256": [candidate.geometry_sha256 for candidate in candidates],
        "source_count": len(source.points),
        "source_sha256": source.source_sha256,
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))


def _implementation_sha() -> str:
    return _file_sha256(Path(__file__).resolve())


def _abort_artifact(
    code: str,
    source_count: int = 0,
    candidate_count: int = 0,
    counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    observed = dict(counters or {})
    return {
        "architecture": ARCHITECTURE,
        "candidates": [],
        "candidate_count": candidate_count,
        "constants_sha256": CONSTANTS_SHA256,
        "implementation_sha256": _implementation_sha(),
        "source_count": source_count,
        "status": ABORT,
        "abort_code": code,
        "work_counters": {
            "candidate_source_tests": int(observed.get("candidate_source_tests", 0)),
            "cell_finalizations": int(observed.get("cell_finalizations", 0)),
            "cell_station_updates": int(observed.get("cell_station_updates", 0)),
            "metric_tuples": int(observed.get("metric_tuples", 0)),
        },
    }


def evaluate_scan(
    vertices: Iterable[Sequence[float]],
    rows: Sequence[Mapping[str, Any]],
    *,
    spacing: float | None = None,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    """Evaluate one frozen scan atomically.

    The returned audit never contains source coordinates, indices, cells, or
    candidate provenance.  On any abort it contains zero candidate records.
    """
    started = time.perf_counter()
    source_count_hint = len(vertices) if hasattr(vertices, "__len__") else 0  # type: ignore[arg-type]
    candidate_count_hint = len(rows) if hasattr(rows, "__len__") else 0
    counters = {
        "candidate_source_tests": 0,
        "cell_finalizations": 0,
        "cell_station_updates": 0,
        "metric_tuples": 0,
    }
    try:
        default_spacing = None if spacing is None else _finite_float(spacing, "INVALID_SPACING")
        if default_spacing is not None and default_spacing <= 0.0:
            raise V73InputError("INVALID_SPACING")
        source = canonical_source(vertices, started)
        candidates = canonical_candidates(rows, default_spacing)
        _check_runtime(started)
        pending: list[CandidateEvaluation] = []
        for candidate in candidates:
            pending.append(_evaluate_candidate(candidate, source, counters, started))
            _check_runtime(started)
        if len(pending) > MAX_RESULT_CANDIDATES:
            raise V73ResourceAbort("RESULT_CANDIDATE_CAP")
        records = [evaluation.record() for evaluation in pending]
        result_sha = _sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        audit = {
            "architecture": ARCHITECTURE,
            "candidate_count": len(candidates),
            "candidates": records,
            "constants_sha256": CONSTANTS_SHA256,
            "implementation_sha256": _implementation_sha(),
            "input_manifest_sha256": _input_manifest_sha(source, candidates),
            "result_sha256": result_sha,
            "source_count": len(source.points),
            "status": COMPLETE,
            "work_counters": counters,
        }
        canonical_bytes(audit)
        return audit, tuple(pending)
    except (V73InputError, V73ResourceAbort, V73ArithmeticAbort, V73SerializationAbort) as error:
        audit = _abort_artifact(str(error), source_count_hint, candidate_count_hint, counters)
        canonical_bytes(audit)
        return audit, tuple()


def apply_itcgv_v1(
    payload: Mapping[str, Any],
    vertices: Iterable[Sequence[float]],
    *,
    spacing: float | None = None,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    """Apply the independent V73 keep/veto mask to ``payload['predictions']``."""
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not all(isinstance(row, Mapping) for row in predictions):
        audit = _abort_artifact("MALFORMED_CANDIDATE_STREAM", 0, 0)
        output = deepcopy(dict(payload))
        output["itcgv_v1_audit"] = audit
        return output, tuple()
    audit, evaluations = evaluate_scan(vertices, predictions, spacing=spacing)
    output = deepcopy(dict(payload))
    if audit["status"] == COMPLETE:
        output["predictions"] = [
            deepcopy(row)
            for row, evaluation in zip(predictions, evaluations, strict=True)
            if evaluation.decision != DECISION_VETO
        ]
    output["itcgv_v1_audit"] = audit
    return output, evaluations


__all__ = [
    "ABORT",
    "ARCHITECTURE",
    "COMPLETE",
    "CONSTANTS_SHA256",
    "CandidateEvaluation",
    "DECISION_KEEP_FEATURE",
    "DECISION_KEEP_SHORT",
    "DECISION_KEEP_SUPPORT",
    "DECISION_VETO",
    "MAX_CANDIDATES",
    "MAX_RESULT_BYTES",
    "MAX_SOURCE_RECORDS",
    "RADII",
    "RULING_SHA256",
    "apply_itcgv_v1",
    "atomic_write_canonical",
    "canonical_bytes",
    "canonical_candidates",
    "canonical_source",
    "evaluate_scan",
]
