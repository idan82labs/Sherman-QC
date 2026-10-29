"""V72 candidate-centered annular secant law (CASL_V1).

This research-only selector is a conservative, deletion-only successor over a
frozen candidate stream.  It uses source coordinates, source-local eighth
neighbour spacings, and each candidate's unchanged endpoints.  It deliberately
does not read faces, mesh topology, normals, candidate scores, provenance,
identity, labels, CAD, expected counts, or other candidates' geometry.

The scientific decision path is two topology-free source-by-candidate passes:
one scale pass and one fixed annular moment pass.  Missing or ambiguous local
evidence always keeps the candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


ARCHITECTURE = "V72_CANDIDATE_CENTERED_ANNULAR_SECANT_LAW_V1"
SHORT_NAME = "CASL_V1"
SCHEMA = "sherman_qc_f1_v72_casl_v1_audit"
RULING_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/2026-08-16_gpt56_pro_post_v71_successor_ruling_verbatim.md"
)
RULING_SHA256 = "31ea25b5c572f0611620aedc9a95aa212a77bf8b545a418a3b63429513386b8b"

MAX_POINTS = 200_000
MAX_CANDIDATES = 64
STATION_COUNT = 17
CORE_RING_COUNT = 3
SHELL_COUNT = 13
WINDOW_COUNT = 9
MAX_POINT_CANDIDATE_TESTS_PER_PASS = 12_800_000
MAX_POINT_CANDIDATE_TESTS_TOTAL = 25_600_000
MAX_SCRATCH_BYTES = 64 * 1024**2
MAX_RESULT_BYTES = 262_144
MAX_RSS_BYTES = 16 * 1024**3
MAX_WALL_SECONDS = 300.0

DEG = math.pi / 180.0
PSD_EPSILON = 2.0**-40
BISECTOR_EPSILON = 2.0**-20
DIVISION_EPSILON = 2.0**-40

CLASS_POSITIVE_WEDGE = "POSITIVE_WEDGE"
CLASS_POSITIVE_RETURN = "POSITIVE_RETURN"
CLASS_ANTI_ROLL = "ANTI_ROLL"
CLASS_ANTI_FLAT = "ANTI_FLAT"
CLASS_ANTI_EDGE = "ANTI_EDGE"
CLASS_UNKNOWN = "UNKNOWN"

DECISION_CERTIFIED_KEEP = "CERTIFIED_KEEP"
DECISION_ABSTAIN_KEEP = "ABSTAIN_KEEP"
DECISION_SUPPRESS = "SUPPRESS"


class CASLInputError(RuntimeError):
    """A frozen input contract was violated."""


class CASLResourceAbort(RuntimeError):
    """A whole-scan deterministic resource cap was reached."""


class CASLAtomicAbort(RuntimeError):
    """A canonical artifact could not be committed atomically."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CASLInputError("CANONICAL_ENCODING_FAILURE") from error


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_resources(started: float) -> None:
    if time.perf_counter() - started > MAX_WALL_SECONDS:
        raise CASLResourceAbort("SCAN_WALL_CAP")
    if _rss_bytes() >= MAX_RSS_BYTES:
        raise CASLResourceAbort("PROCESS_RSS_CAP")


def _normalize_zero(array: np.ndarray) -> np.ndarray:
    clean = np.asarray(array, dtype=np.float64).copy()
    clean[clean == 0.0] = 0.0
    return clean


def _median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64), kind="stable")
    count = len(ordered)
    if count == 0:
        raise CASLInputError("SOURCE_SCALE_CONTRACT")
    middle = count // 2
    if count & 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


@dataclass(frozen=True)
class CanonicalSource:
    points: np.ndarray
    spacing: np.ndarray
    h0: float
    source_sha256: str


def _eighth_neighbour_spacing(points: np.ndarray, started: float) -> np.ndarray:
    tree = cKDTree(points, balanced_tree=True, compact_nodes=True)
    output = np.empty(len(points), dtype=np.float64)
    for begin in range(0, len(points), 4096):
        _check_resources(started)
        end = min(len(points), begin + 4096)
        distances, _indices = tree.query(points[begin:end], k=9, workers=1)
        output[begin:end] = np.asarray(distances, dtype=np.float64)[:, 8]
    return output


def _canonical_source(
    vertices: Any,
    local_spacings: Any | None,
    started: float,
) -> CanonicalSource:
    try:
        points = _normalize_zero(np.asarray(vertices, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise CASLInputError("NONFINITE_INPUT") from error
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise CASLInputError("NONFINITE_INPUT")
    if len(points) > MAX_POINTS:
        raise CASLResourceAbort("SOURCE_POINT_CAP")
    if len(points) < 9 or not np.isfinite(points).all():
        raise CASLInputError("NONFINITE_INPUT" if not np.isfinite(points).all() else "SOURCE_SCALE_CONTRACT")

    if local_spacings is None:
        spacing = _eighth_neighbour_spacing(points, started)
    else:
        try:
            spacing = _normalize_zero(np.asarray(local_spacings, dtype=np.float64))
        except (TypeError, ValueError) as error:
            raise CASLInputError("SOURCE_SCALE_CONTRACT") from error
        if spacing.shape != (len(points),):
            raise CASLInputError("SOURCE_SCALE_CONTRACT")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise CASLInputError("SOURCE_SCALE_CONTRACT")

    # Canonical mathematical lexicographic order.  The source ordinal created
    # here is the only tie-break used by the scale pass.
    order = np.lexsort((spacing, points[:, 2], points[:, 1], points[:, 0]))
    points = np.ascontiguousarray(points[order], dtype=np.float64)
    spacing = np.ascontiguousarray(spacing[order], dtype=np.float64)
    h0 = _median(spacing)
    if not math.isfinite(h0) or h0 <= 0.0:
        raise CASLInputError("SOURCE_SCALE_CONTRACT")
    payload = points.astype(">f8", copy=False).tobytes() + spacing.astype(">f8", copy=False).tobytes()
    return CanonicalSource(points, spacing, h0, _sha(payload))


@dataclass(frozen=True)
class Candidate:
    opaque_hash: str
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    ordinal: int
    original: Mapping[str, Any]

    @property
    def stations(self) -> np.ndarray:
        fractions = (np.arange(STATION_COUNT, dtype=np.float64) + 0.5) / STATION_COUNT
        return self.start + fractions[:, None] * (self.end - self.start)


def _candidate(row: Mapping[str, Any], ordinal: int) -> Candidate:
    opaque = row.get("candidate_opaque_hash") or row.get("route_hash")
    if opaque is None:
        raise CASLInputError("MALFORMED_CANDIDATE")
    try:
        start = _normalize_zero(np.asarray(row["line_start"], dtype=np.float64))
        end = _normalize_zero(np.asarray(row["line_end"], dtype=np.float64))
    except (KeyError, TypeError, ValueError) as error:
        raise CASLInputError("MALFORMED_CANDIDATE") from error
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise CASLInputError("NONFINITE_INPUT")
    if tuple(float(value) for value in end) < tuple(float(value) for value in start):
        start, end = end.copy(), start.copy()
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length):
        raise CASLInputError("NONFINITE_INPUT")
    if length == 0.0:
        raise CASLInputError("ZERO_LENGTH_CANDIDATE")
    return Candidate(str(opaque), start, end, delta / length, length, ordinal, row)


def _candidate_order(candidate: Candidate) -> tuple[Any, ...]:
    return (
        *(float(value) for value in candidate.start),
        *(float(value) for value in candidate.end),
        candidate.opaque_hash,
    )


@dataclass(frozen=True)
class ScaleRecord:
    hc: float | None
    station_scales: tuple[float | None, ...]
    valid_count: int


def _station_assignment(
    candidate: Candidate,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative = points - candidate.start
    along = relative @ candidate.direction
    valid = (along >= 0.0) & (along <= candidate.length)
    clipped = np.minimum(
        STATION_COUNT - 1,
        np.floor(STATION_COUNT * along[valid] / candidate.length).astype(np.int64),
    )
    return relative, along, np.flatnonzero(valid), clipped


def _scale_candidate(candidate: Candidate, source: CanonicalSource) -> ScaleRecord:
    relative, _along, valid_ids, station_ids = _station_assignment(candidate, source.points)
    stations = candidate.stations
    station_scales: list[float | None] = [None] * STATION_COUNT
    canonical_ordinals = np.arange(len(source.points), dtype=np.int64)
    for station in range(STATION_COUNT):
        current = valid_ids[station_ids == station]
        if len(current) < 8:
            continue
        delta = source.points[current] - stations[station]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort((canonical_ordinals[current], distance2))[:8]
        station_scales[station] = _median(source.spacing[current[order]])
    valid_values = np.asarray([value for value in station_scales if value is not None], dtype=np.float64)
    if len(valid_values) < 9:
        return ScaleRecord(None, tuple(station_scales), len(valid_values))
    estimate = _median(valid_values)
    hc = float(np.clip(estimate, 0.5 * source.h0, 2.0 * source.h0))
    if not math.isfinite(hc) or hc <= 0.0:
        raise CASLInputError("SOURCE_SCALE_CONTRACT")
    return ScaleRecord(hc, tuple(station_scales), len(valid_values))


@dataclass(frozen=True)
class ShellDescriptor:
    base_usable: bool
    two_ray_usable: bool
    model_usable: bool
    one_ray: bool
    flat: bool
    rbar: float
    theta: float
    eta: float
    delta: float
    g: float
    kappa: float
    principal: np.ndarray | None
    bisector: np.ndarray | None
    rays: tuple[np.ndarray, np.ndarray] | None
    one_ray_direction: np.ndarray | None


def _plane_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


def _coverage(weights: np.ndarray, total: float) -> int:
    if total <= 0.0:
        return 0
    return int(np.count_nonzero(weights >= 0.06 * total))


def _effective_count(weight: float, squared_weight: float) -> float:
    return 0.0 if squared_weight <= 0.0 else float(weight * weight / squared_weight)


def _shell_descriptor(
    direction: np.ndarray,
    weight: float,
    squared_weight: float,
    weighted_radius: float,
    weighted_vector: np.ndarray,
    weighted_second: np.ndarray,
    axial_weights: np.ndarray,
) -> ShellDescriptor:
    empty = ShellDescriptor(False, False, False, False, False, 0.0, 0.0, 0.0, math.inf, 0.0, 0.0, None, None, None, None)
    if weight <= 0.0:
        return empty
    if not np.isfinite(
        np.concatenate((np.asarray((weight, squared_weight, weighted_radius)), weighted_vector, weighted_second.ravel(), axial_weights))
    ).all():
        raise CASLResourceAbort("MOMENT_NONFINITE")
    base = _effective_count(weight, squared_weight) >= 12.0 and _coverage(axial_weights, weight) >= 3
    if not base:
        return empty
    rbar = float(weighted_radius / weight)
    mu = weighted_vector / weight
    second_moment = weighted_second / weight
    covariance = 0.5 * ((second_moment - np.outer(mu, mu)) + (second_moment - np.outer(mu, mu)).T)
    first_basis, second_basis = _plane_basis(direction)
    basis = np.column_stack((first_basis, second_basis))
    covariance2 = 0.5 * (basis.T @ covariance @ basis + (basis.T @ covariance @ basis).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance2)
    if not np.isfinite(eigenvalues).all():
        raise CASLResourceAbort("MOMENT_NONFINITE")
    if float(np.min(eigenvalues)) < -PSD_EPSILON:
        raise CASLResourceAbort("MOMENT_PSD_DOMAIN")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambda2, lambda1 = float(eigenvalues[0]), float(eigenvalues[1])
    principal = basis @ eigenvectors[:, 1]
    principal /= np.linalg.norm(principal)
    variation = lambda1 + lambda2
    delta = math.inf if variation == 0.0 else lambda2 / variation
    two_ray = variation >= 0.01 and delta <= 0.12
    mu_perp = mu - float(np.dot(mu, direction)) * direction
    mu_norm = float(np.linalg.norm(mu_perp))
    one_ray = mu_norm >= 0.96 and variation <= 0.06
    one_ray_direction = None if not one_ray else mu_perp / mu_norm
    if not two_ray:
        return ShellDescriptor(True, False, False, one_ray, False, rbar, 0.0, 0.0, delta, 0.0, 0.0, principal, None, None, one_ray_direction)

    principal_mean = float(np.dot(mu_perp, principal))
    b = math.sqrt(float(np.clip(float(np.dot(mu_perp, mu_perp)) - principal_mean * principal_mean, 0.0, 1.0)))
    a = math.sqrt(max(0.0, 1.0 - b * b))
    eta = 0.5 * (1.0 - float(np.clip(abs(principal_mean) / max(a, DIVISION_EPSILON), 0.0, 1.0)))
    theta = 2.0 * math.atan2(a, b)
    model_usable = eta >= 0.08
    bisector = None
    rays = None
    if b >= BISECTOR_EPSILON:
        bisector = (mu_perp - principal_mean * principal) / b
        plus = bisector * b + principal * a
        minus = bisector * b - principal * a
        plus /= np.linalg.norm(plus)
        minus /= np.linalg.norm(minus)
        rays = (plus, minus)
    g = 2.0 * rbar * math.sin(theta / 2.0)
    kappa = 2.0 * math.cos(theta / 2.0) / rbar
    flat = theta >= 176.0 * DEG and delta <= 0.08 and eta >= 0.10
    return ShellDescriptor(True, True, model_usable, one_ray, flat, rbar, theta, eta, delta, g, kappa, principal, bisector, rays, one_ray_direction)


def _safe_angle(first: np.ndarray, second: np.ndarray, *, absolute: bool = False) -> float:
    dot = float(np.dot(first, second))
    if absolute:
        dot = abs(dot)
    return math.acos(float(np.clip(dot, -1.0, 1.0)))


def _pair_drift(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    direct = max(_safe_angle(first[0], second[0]), _safe_angle(first[1], second[1]))
    swapped = max(_safe_angle(first[0], second[1]), _safe_angle(first[1], second[0]))
    return min(direct, swapped)


def _ratio_residual(values: Sequence[float]) -> float:
    low = min(values)
    high = max(values)
    if low <= 0.0 or not math.isfinite(low) or not math.isfinite(high):
        return math.inf
    return (high / low - 1.0) / 0.20


def _window_residuals(shells: Sequence[ShellDescriptor]) -> tuple[float, float, float]:
    theta = [value.theta for value in shells]
    wedge = (max(theta) - min(theta)) / (6.0 * DEG)
    return wedge, _ratio_residual([value.g for value in shells]), _ratio_residual([value.kappa for value in shells])


def _model_window(shells: Sequence[ShellDescriptor], model: str) -> bool:
    if len(shells) != 5:
        return False
    if model in {"WEDGE", "RETURN", "ROLL"} and not all(value.model_usable for value in shells):
        return False
    wedge_error, return_error, roll_error = _window_residuals(shells)
    theta_median = _median(np.asarray([value.theta for value in shells]))
    if model == "WEDGE":
        if not all(value.rays is not None for value in shells):
            return False
        drift = max(_pair_drift(shells[index].rays, shells[index + 1].rays) for index in range(4))  # type: ignore[arg-type]
        return (
            8.0 * DEG <= theta_median <= 174.0 * DEG
            and wedge_error <= 1.0
            and drift <= 6.0 * DEG
            and wedge_error + 0.25 <= min(return_error, roll_error)
        )
    if model in {"RETURN", "ROLL"}:
        if not all(value.bisector is not None for value in shells):
            return False
        drift = max(_safe_angle(shells[index].bisector, shells[index + 1].bisector) for index in range(4))  # type: ignore[arg-type]
        monotone = all(shells[index + 1].theta <= shells[index].theta + 2.0 * DEG for index in range(4))
        if model == "RETURN":
            g_median = _median(np.asarray([value.g for value in shells]))
            return (
                3.0 * DEG <= theta_median <= 80.0 * DEG
                and return_error <= 1.0
                and drift <= 6.0 * DEG
                and monotone
                and return_error + 0.25 <= min(wedge_error, roll_error)
                and math.isfinite(g_median)
            )
        kappa_median = _median(np.asarray([value.kappa for value in shells]))
        radius = math.inf if kappa_median <= 0.0 else 1.0 / kappa_median
        return (
            20.0 * DEG <= theta_median <= 175.0 * DEG
            and roll_error <= 1.0
            and drift <= 6.0 * DEG
            and monotone
            and roll_error + 0.25 <= min(wedge_error, return_error)
            and math.isfinite(radius)
        )
    if model == "FLAT":
        if not all(value.flat and value.principal is not None for value in shells):
            return False
        spread = max(value.theta for value in shells) - min(value.theta for value in shells)
        drift = max(_safe_angle(shells[index].principal, shells[index + 1].principal, absolute=True) for index in range(4))  # type: ignore[arg-type]
        return spread <= 3.0 * DEG and drift <= 4.0 * DEG
    if model == "EDGE":
        if not all(value.one_ray and value.one_ray_direction is not None for value in shells):
            return False
        drift = max(_safe_angle(shells[index].one_ray_direction, shells[index + 1].one_ray_direction) for index in range(4))  # type: ignore[arg-type]
        return drift <= 4.0 * DEG
    raise ValueError(model)


def _window_passes(shells: Sequence[ShellDescriptor], model: str, hc: float) -> bool:
    if not _model_window(shells, model):
        return False
    if model == "RETURN":
        median_g = _median(np.asarray([value.g for value in shells]))
        return 1.0 * hc <= median_g <= 24.0 * hc
    if model == "ROLL":
        median_kappa = _median(np.asarray([value.kappa for value in shells]))
        radius = math.inf if median_kappa <= 0.0 else 1.0 / median_kappa
        return 6.0 * hc <= radius <= 512.0 * hc
    return True


def _adjacent_window_start(starts: Sequence[int]) -> int | None:
    ordered = sorted(set(int(value) for value in starts))
    for first, second in zip(ordered, ordered[1:]):
        if second == first + 1:
            return first
    return None


@dataclass(frozen=True)
class StationDecision:
    station_class: str
    selected_code: str


def _station_decision(shells: Sequence[ShellDescriptor], core_contact: bool, hc: float) -> StationDecision:
    if not core_contact:
        return StationDecision(CLASS_UNKNOWN, "")
    passes: dict[str, list[int]] = {name: [] for name in ("WEDGE", "RETURN", "ROLL", "FLAT", "EDGE")}
    for start in range(WINDOW_COUNT):
        window = shells[start : start + 5]
        for model in passes:
            if _window_passes(window, model, hc):
                passes[model].append(start)
    if passes["WEDGE"]:
        return StationDecision(CLASS_POSITIVE_WEDGE, f"WEDGE:{min(passes['WEDGE'])}")
    if passes["RETURN"]:
        return StationDecision(CLASS_POSITIVE_RETURN, f"RETURN:{min(passes['RETURN'])}")
    for model, station_class in (("ROLL", CLASS_ANTI_ROLL), ("FLAT", CLASS_ANTI_FLAT), ("EDGE", CLASS_ANTI_EDGE)):
        start = _adjacent_window_start(passes[model])
        if start is not None:
            return StationDecision(station_class, f"{model}:{start}")
    return StationDecision(CLASS_UNKNOWN, "")


def _longest_run(values: Sequence[str], allowed: set[str]) -> int:
    longest = current = 0
    for value in values:
        if value in allowed:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def candidate_decision_from_classes(classes: Sequence[str]) -> tuple[str, dict[str, int | str]]:
    positive = {CLASS_POSITIVE_WEDGE, CLASS_POSITIVE_RETURN}
    positive_count = sum(value in positive for value in classes)
    positive_run = _longest_run(classes, positive)
    counts = {
        "ROLL": classes.count(CLASS_ANTI_ROLL),
        "FLAT": classes.count(CLASS_ANTI_FLAT),
        "EDGE": classes.count(CLASS_ANTI_EDGE),
    }
    dominant = min(("ROLL", "FLAT", "EDGE"), key=lambda name: (-counts[name], ("ROLL", "FLAT", "EDGE").index(name)))
    dominant_class = {"ROLL": CLASS_ANTI_ROLL, "FLAT": CLASS_ANTI_FLAT, "EDGE": CLASS_ANTI_EDGE}[dominant]
    dominant_run = _longest_run(classes, {dominant_class})
    anti_total = sum(counts.values())
    if positive_count >= 4 and positive_run >= 3:
        decision = DECISION_CERTIFIED_KEEP
    elif positive_count == 0 and anti_total >= 13 and counts[dominant] >= 11 and dominant_run >= 8:
        decision = DECISION_SUPPRESS
    else:
        decision = DECISION_ABSTAIN_KEEP
    return decision, {
        "positive_count": positive_count,
        "positive_run": positive_run,
        "roll_count": counts["ROLL"],
        "flat_count": counts["FLAT"],
        "edge_count": counts["EDGE"],
        "anti_total": anti_total,
        "dominant_anti_class": dominant,
        "dominant_anti_count": counts[dominant],
        "dominant_anti_run": dominant_run,
    }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    hc: float | None
    station_classes: tuple[str, ...]
    selected_codes: tuple[str, ...]
    decision: str
    summary: Mapping[str, int | str]
    model_window_evaluated: bool


def _profile_candidate(candidate: Candidate, source: CanonicalSource, scale: ScaleRecord) -> CandidateEvaluation:
    if scale.hc is None:
        classes = (CLASS_UNKNOWN,) * STATION_COUNT
        decision, summary = candidate_decision_from_classes(classes)
        return CandidateEvaluation(candidate, None, classes, ("",) * STATION_COUNT, decision, summary, False)
    hc = scale.hc
    axial_half_width = min(3.0 * hc, candidate.length / 34.0)
    if axial_half_width <= 0.0:
        classes = (CLASS_UNKNOWN,) * STATION_COUNT
        decision, summary = candidate_decision_from_classes(classes)
        return CandidateEvaluation(candidate, hc, classes, ("",) * STATION_COUNT, decision, summary, False)

    relative, along, valid_ids, station_ids = _station_assignment(candidate, source.points)
    tau = along[valid_ids] - ((station_ids.astype(np.float64) + 0.5) / STATION_COUNT) * candidate.length
    q = relative[valid_ids] - along[valid_ids, None] * candidate.direction
    radius = np.linalg.norm(q, axis=1)
    finite = np.isfinite(tau) & np.isfinite(radius) & np.isfinite(q).all(axis=1)
    if not finite.all():
        raise CASLResourceAbort("MOMENT_NONFINITE")
    outer_radius = 4.0 * hc * (5.0 / 4.0) ** 13
    accepted = (np.abs(tau) < axial_half_width) & (radius <= outer_radius)
    ids = valid_ids[accepted]
    station_ids = station_ids[accepted]
    tau = tau[accepted]
    q = q[accepted]
    radius = radius[accepted]
    density = np.clip((source.spacing[ids] / hc) ** 2, 0.25, 4.0)
    weight = density * (1.0 - (tau / axial_half_width) ** 2)
    axial_bin = np.searchsorted(np.asarray((-0.5, 0.0, 0.5)), tau / axial_half_width, side="right")

    core_size = STATION_COUNT * CORE_RING_COUNT
    core_w = np.zeros(core_size, dtype=np.float64)
    core_q = np.zeros(core_size, dtype=np.float64)
    core_axial = np.zeros((core_size, 4), dtype=np.float64)
    core_mask = radius < 4.0 * hc
    if np.any(core_mask):
        core_ring = np.searchsorted(np.asarray((hc, 2.0 * hc, 4.0 * hc)), radius[core_mask], side="right")
        core_index = station_ids[core_mask] * CORE_RING_COUNT + core_ring
        core_weight = weight[core_mask]
        core_w += np.bincount(core_index, weights=core_weight, minlength=core_size)
        core_q += np.bincount(core_index, weights=core_weight * core_weight, minlength=core_size)
        core_axial_flat = np.bincount(core_index * 4 + axial_bin[core_mask], weights=core_weight, minlength=core_size * 4)
        core_axial = core_axial_flat.reshape(core_size, 4)

    shell_size = STATION_COUNT * SHELL_COUNT
    shell_w = np.zeros(shell_size, dtype=np.float64)
    shell_q = np.zeros(shell_size, dtype=np.float64)
    shell_r = np.zeros(shell_size, dtype=np.float64)
    shell_vec = np.zeros((shell_size, 3), dtype=np.float64)
    shell_second = np.zeros((shell_size, 3, 3), dtype=np.float64)
    shell_axial = np.zeros((shell_size, 4), dtype=np.float64)
    shell_edges = 4.0 * hc * (5.0 / 4.0) ** np.arange(14, dtype=np.float64)
    shell_mask = (radius >= shell_edges[0]) & (radius <= shell_edges[-1])
    if np.any(shell_mask):
        shell_ids = np.searchsorted(shell_edges, radius[shell_mask], side="right") - 1
        shell_ids = np.minimum(shell_ids, SHELL_COUNT - 1)
        index = station_ids[shell_mask] * SHELL_COUNT + shell_ids
        current_weight = weight[shell_mask]
        current_radius = radius[shell_mask]
        unit = q[shell_mask] / current_radius[:, None]
        shell_w += np.bincount(index, weights=current_weight, minlength=shell_size)
        shell_q += np.bincount(index, weights=current_weight * current_weight, minlength=shell_size)
        shell_r += np.bincount(index, weights=current_weight * current_radius, minlength=shell_size)
        for first in range(3):
            shell_vec[:, first] += np.bincount(index, weights=current_weight * unit[:, first], minlength=shell_size)
            for second in range(first, 3):
                values = np.bincount(index, weights=current_weight * unit[:, first] * unit[:, second], minlength=shell_size)
                shell_second[:, first, second] += values
                if first != second:
                    shell_second[:, second, first] += values
        shell_axial_flat = np.bincount(index * 4 + axial_bin[shell_mask], weights=current_weight, minlength=shell_size * 4)
        shell_axial = shell_axial_flat.reshape(shell_size, 4)

    station_decisions: list[StationDecision] = []
    any_window = False
    for station in range(STATION_COUNT):
        core_index = station * CORE_RING_COUNT
        first_contact = (
            _effective_count(core_w[core_index], core_q[core_index]) >= 4.0
            and _coverage(core_axial[core_index], core_w[core_index]) >= 2
        )
        outer_contact = any(
            _effective_count(core_w[core_index + ring], core_q[core_index + ring]) >= 6.0
            and _coverage(core_axial[core_index + ring], core_w[core_index + ring]) >= 2
            for ring in (1, 2)
        )
        descriptors = []
        for shell in range(SHELL_COUNT):
            offset = station * SHELL_COUNT + shell
            descriptors.append(
                _shell_descriptor(
                    candidate.direction,
                    shell_w[offset],
                    shell_q[offset],
                    shell_r[offset],
                    shell_vec[offset],
                    shell_second[offset],
                    shell_axial[offset],
                )
            )
        if any(sum(value.base_usable for value in descriptors[start : start + 5]) == 5 for start in range(WINDOW_COUNT)):
            any_window = True
        station_decisions.append(_station_decision(descriptors, first_contact and outer_contact, hc))
    classes = tuple(value.station_class for value in station_decisions)
    codes = tuple(value.selected_code for value in station_decisions)
    decision, summary = candidate_decision_from_classes(classes)
    return CandidateEvaluation(candidate, hc, classes, codes, decision, summary, any_window)


def frozen_configuration() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "ruling_sha256": RULING_SHA256,
        "stations": STATION_COUNT,
        "core_rings": CORE_RING_COUNT,
        "analysis_shells": SHELL_COUNT,
        "model_windows": WINDOW_COUNT,
        "shell_radii": [4.0 * (5.0 / 4.0) ** index for index in range(14)],
        "comparators": {
            "effective_count": 12.0,
            "axial_coverage": 3,
            "variation": 0.01,
            "delta": 0.12,
            "eta": 0.08,
            "spread_degrees": 6.0,
            "ratio": 0.20,
            "winning_margin": 0.25,
            "positive_count": 4,
            "positive_run": 3,
            "anti_total": 13,
            "dominant_anti_count": 11,
            "dominant_anti_run": 8,
        },
        "caps": {
            "points": MAX_POINTS,
            "candidates": MAX_CANDIDATES,
            "point_candidate_tests_per_pass": MAX_POINT_CANDIDATE_TESTS_PER_PASS,
            "point_candidate_tests_total": MAX_POINT_CANDIDATE_TESTS_TOTAL,
            "scratch_bytes": MAX_SCRATCH_BYTES,
            "result_bytes": MAX_RESULT_BYTES,
            "rss_bytes": MAX_RSS_BYTES,
            "wall_seconds": MAX_WALL_SECONDS,
        },
        "forbidden_domains": ["graph", "mesh", "adjacency", "normal", "candidate_score", "identity", "gold"],
    }


CONSTANT_SHA256 = _sha(canonical_bytes(frozen_configuration()))


def _implementation_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def _attest_ruling() -> None:
    if not RULING_PATH.is_file() or _sha(RULING_PATH.read_bytes()) != RULING_SHA256:
        raise CASLInputError("RULING_HASH_MISMATCH")


def _record(evaluation: CandidateEvaluation) -> dict[str, Any]:
    return {
        "candidate_hash": evaluation.candidate.opaque_hash,
        "h_c_hex": None if evaluation.hc is None else float(evaluation.hc).hex(),
        "station_classes": list(evaluation.station_classes),
        "selected_model_windows": list(evaluation.selected_codes),
        **dict(evaluation.summary),
        "model_window_evaluated": evaluation.model_window_evaluated,
        "decision": evaluation.decision,
    }


def _abort(
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    reason: str,
    started: float,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    baseline.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v72_casl_v1_result",
            "profile": ARCHITECTURE,
            "predictions": rows,
            "candidate_count": len(rows),
            "source_only": True,
            "research_only": True,
            "casl_v1_audit": {
                "architecture": ARCHITECTURE,
                "status": "SCAN_ATOMIC_ABORT",
                "abort_reason": reason,
                "modified_output_count": 0,
            },
            "casl_v1_operational_envelope": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": _rss_bytes(),
            },
            "casl_v1_modified_output_committed": False,
        }
    )
    return baseline, ()


def apply_casl_v1(
    source_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: Any,
    faces: Any | None = None,
    *,
    local_spacings: Any | None = None,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    """Apply frozen CASL as an unchanged-endpoint subset selector.

    ``faces`` is accepted only to make the ignored topology explicit.  It is
    never inspected, converted, hashed, or used.
    """

    del faces
    started = time.perf_counter()
    baseline = deepcopy(dict(source_payload)) if isinstance(source_payload, Mapping) else {}
    source_rows = source_payload.get("predictions") or () if isinstance(source_payload, Mapping) else source_payload
    rows = [deepcopy(dict(row)) for row in source_rows]
    try:
        _attest_ruling()
        if len(rows) == 0 or len(rows) > MAX_CANDIDATES:
            raise CASLResourceAbort("CANDIDATE_CAP")
        candidates = [_candidate(row, ordinal) for ordinal, row in enumerate(rows)]
        if len({value.opaque_hash for value in candidates}) != len(candidates):
            raise CASLInputError("DUPLICATE_CANDIDATE_HASH")
        source = _canonical_source(vertices, local_spacings, started)
        tests_per_pass = len(source.points) * len(candidates)
        if tests_per_pass > MAX_POINT_CANDIDATE_TESTS_PER_PASS:
            raise CASLResourceAbort("POINT_CANDIDATE_TEST_CAP")

        canonical_candidates = sorted(candidates, key=_candidate_order)
        scales: dict[str, ScaleRecord] = {}
        for candidate in canonical_candidates:
            _check_resources(started)
            scales[candidate.opaque_hash] = _scale_candidate(candidate, source)
        evaluations = []
        for candidate in canonical_candidates:
            _check_resources(started)
            evaluations.append(_profile_candidate(candidate, source, scales[candidate.opaque_hash]))

        by_hash = {value.candidate.opaque_hash: value for value in evaluations}
        survivors = [
            row
            for row, candidate in zip(rows, candidates)
            if by_hash[candidate.opaque_hash].decision != DECISION_SUPPRESS
        ]
        sorted_evaluations = sorted(evaluations, key=lambda value: value.candidate.opaque_hash)
        records = [_record(value) for value in sorted_evaluations]
        suppressed = [value.candidate.opaque_hash for value in sorted_evaluations if value.decision == DECISION_SUPPRESS]
        artifact_core = {
            "schema": SCHEMA,
            "architecture": ARCHITECTURE,
            "constants_sha256": CONSTANT_SHA256,
            "ruling_sha256": RULING_SHA256,
            "source_state_sha256": source.source_sha256,
            "candidate_stream_sha256": _sha(canonical_bytes(sorted(value.opaque_hash for value in candidates))),
            "executable_sha256": _implementation_sha256(),
            "source_count": len(source.points),
            "candidate_count": len(candidates),
            "work_counters": {
                "scale_point_candidate_tests": tests_per_pass,
                "profile_point_candidate_tests": tests_per_pass,
                "total_point_candidate_tests": 2 * tests_per_pass,
                "graph_vertices_visited": 0,
                "graph_edges_visited": 0,
                "mesh_adjacency_reads": 0,
                "q0_count": 0,
                "closure_count": 0,
            },
            "candidates": records,
            "suppression_count": len(suppressed),
            "suppressed_hashes": suppressed,
            "status": "COMPLETE",
            "abort_reason": None,
        }
        artifact_sha = _sha(canonical_bytes(artifact_core))
        artifact = {**artifact_core, "result_sha256": artifact_sha}
        artifact_bytes = canonical_bytes(artifact)
        if len(artifact_bytes) > MAX_RESULT_BYTES:
            raise CASLResourceAbort("RESULT_BYTES_CAP")
        _check_resources(started)
        baseline.update(
            {
                "schema_version": 1,
                "kind": "sherman_qc_f1_v72_casl_v1_result",
                "profile": ARCHITECTURE,
                "predictions": survivors,
                "candidate_count": len(survivors),
                "casl_v1_audit": artifact,
                "casl_v1_operational_envelope": {
                    "wall_seconds": time.perf_counter() - started,
                    "peak_rss_bytes": _rss_bytes(),
                    "result_bytes": len(artifact_bytes),
                },
                "casl_v1_modified_output_committed": True,
                "source_only": True,
                "research_only": True,
                "expected_count_used": False,
                "labels_used": False,
                "cad_used": False,
                "part_identity_used": False,
                "filename_identity_used": False,
                "candidate_score_used": False,
                "route_provenance_used": False,
                "mesh_used": False,
                "adjacency_used": False,
            }
        )
        return baseline, tuple(sorted_evaluations)
    except (
        CASLInputError,
        CASLResourceAbort,
        MemoryError,
        OverflowError,
        FloatingPointError,
        KeyError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return _abort(baseline, rows, str(error), started)


def validate_result_bytes(payload: bytes) -> None:
    if len(payload) > MAX_RESULT_BYTES:
        raise CASLResourceAbort("RESULT_BYTES_CAP")


def commit_canonical_artifact(
    path: Path,
    payload: bytes,
    *,
    inject_pre_rename_failure: bool = False,
) -> str:
    """Flush and atomically rename one already-validated scientific artifact."""

    validate_result_bytes(payload)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if inject_pre_rename_failure:
            raise OSError("injected pre-rename failure")
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise CASLAtomicAbort("ATOMIC_COMMIT_FAILURE") from error
    return _sha(payload)


__all__ = [
    "ARCHITECTURE",
    "SHORT_NAME",
    "CONSTANT_SHA256",
    "CandidateEvaluation",
    "apply_casl_v1",
    "candidate_decision_from_classes",
    "canonical_bytes",
    "commit_canonical_artifact",
    "frozen_configuration",
    "validate_result_bytes",
]
