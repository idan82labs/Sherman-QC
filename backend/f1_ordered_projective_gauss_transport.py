"""Research-only V34 ordered projective Gauss transport.

V34 preserves the joint source observation of transverse position and
projective face-normal direction.  It is a rescue-only consumer of the
frozen V16 pre-NMS candidate stream.  It does not read topology, adjacency,
labels, CAD, expected counts, identity, V32 state, or V33 features/verdicts.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Sequence

import numpy as np

from backend.f1_candidate_prism_geometry import (
    CandidateFrameDomain,
    CandidatePrismError,
    canonical_triangle_mesh,
    candidate_frame_domain,
    clip_axial_transverse_cell,
    iter_prism_pieces,
    polygon_area,
)
from backend.f1_distributed_harmonic_fold_flux import (
    ACTION_LEGACY,
    Candidate,
    ordinary_nms,
    parse_candidates,
)


PROFILE = "V34_ORDERED_PROJECTIVE_GAUSS_TRANSPORT_V1"
PRIMITIVE = "TRANSVERSE_MONOTONE_GAUSS_MEASURE_V1"
SHORT_NAME = "OPGT_V1"
SCHEMA_VERSION = 1

TRANSVERSE_STRIPS = 64
AXIAL_SLABS = 8
OUTER_MASS_FRACTION = 1.0 / 64.0
TAIL_MASS_FRACTION = 3.0 / 16.0
TRANSITION_MASS_FRACTION = 1.0 / 8.0
MIN_EFFECTIVE_AXIAL_SUPPORT = 2.0
PHASE_STEPS_PER_PI = 512
PHASE_STEP = math.pi / PHASE_STEPS_PER_PI
PHASE_COUNT = 2 * PHASE_STEPS_PER_PI + 1
MAX_PHASE_SPAN_STEPS = PHASE_STEPS_PER_PI - 1
PHASE_WINDOW_WIDTH = MAX_PHASE_SPAN_STEPS + 1
PHASE_COSINE = np.cos(
    2.0 * (-0.5 * math.pi + np.arange(PHASE_COUNT, dtype=np.float64) * PHASE_STEP)
)
PHASE_SINE = np.sin(
    2.0 * (-0.5 * math.pi + np.arange(PHASE_COUNT, dtype=np.float64) * PHASE_STEP)
)
PHASE_COSINE.setflags(write=False)
PHASE_SINE.setflags(write=False)
MAX_RESCUE_CANDIDATES = 128
MAX_TRIANGLE_CELL_INTERSECTIONS = 1 << 18
MAX_WORKERS = 8

# Dimensionless area-normal contributions are rounded to this fixed lattice.
# int64 overflow is checked before every addition and is a resource stop.
ACCUMULATOR_Q = 1 << 36
INT64_LIMIT = (1 << 63) - 1

CACHE_HEADER_BYTES = 64
CACHE_RECORD_BYTES = 224
ACTION_RESCUE = "OPGT_PASS"
ACTION_NONE = "NO_RESCUE"

REASONS = (
    "LEGACY_PASS",
    "NUMERICAL_OR_RESOURCE_INVALID",
    "INSUFFICIENT_EFFECTIVE_AXIAL_SUPPORT",
    "LEFT_TAIL_NOT_PLATEAU",
    "RIGHT_TAIL_NOT_PLATEAU",
    "TAIL_NORMALS_NOT_SEPARATED",
    "INSUFFICIENT_TRANSITION_MASS",
    "AXIAL_INCOHERENCE",
    "NONMONOTONE_GAUSS_PATH",
    "OPGT_PASS",
    "ORDINARY_NMS_SUPPRESSED",
)
REASON_CODE = {name: index for index, name in enumerate(REASONS)}


class OPGTContractError(RuntimeError):
    """Terminal V34 contract/resource failure."""

    def __init__(self, token: str, detail: str = "") -> None:
        super().__init__(f"{token}: {detail}" if detail else token)
        self.token = token
        self.detail = detail


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    return _canonical({key: value for key, value in result.items() if key != "_ephemeral"}) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _epsilon(weight: float) -> float:
    return 2.0 * weight * (1.0 - math.cos(PHASE_STEP)) + math.ldexp(weight, -40)


def _e0(weight: float, real: float, imag: float) -> float:
    return max(0.0, 2.0 * weight - 2.0 * math.hypot(real, imag))


def _loss(weight: float, real: float, imag: float, phase: float) -> float:
    return max(
        0.0,
        2.0 * weight
        - 2.0 * (real * math.cos(2.0 * phase) + imag * math.sin(2.0 * phase)),
    )


def _checked_add(array: np.ndarray, index: tuple[int, int], value: int) -> None:
    current = int(array[index])
    if abs(value) > INT64_LIMIT or (
        value > 0 and current > INT64_LIMIT - value
    ) or (
        value < 0 and current < -INT64_LIMIT - 1 - value
    ):
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "fixed accumulator overflow")
    array[index] = current + value


def _cell_bounds(frame: CandidateFrameDomain) -> tuple[np.ndarray, np.ndarray]:
    axial = np.linspace(-frame.scale, frame.length + frame.scale, AXIAL_SLABS + 1)
    transverse = np.linspace(-12.0 * frame.scale, 12.0 * frame.scale, TRANSVERSE_STRIPS + 1)
    return axial, transverse


def _candidate_measure(
    candidate: Candidate,
    mesh: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CandidateFrameDomain, int]:
    """Stream triangle/cell intersections into exact fixed-point cell sums."""
    try:
        frame = candidate_frame_domain(candidate, mesh)
    except CandidatePrismError as exc:
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", str(exc)) from exc
    axial_edges, transverse_edges = _cell_bounds(frame)
    weights = np.zeros((AXIAL_SLABS, TRANSVERSE_STRIPS), dtype=np.int64)
    real = np.zeros_like(weights)
    imag = np.zeros_like(weights)
    intersections = 0
    scale_area = frame.scale * frame.scale

    for _face_index, normal, polygon in iter_prism_pieces(mesh, frame):
        u_min, u_max = float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0]))
        q_min, q_max = float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1]))
        k0 = max(0, min(AXIAL_SLABS - 1, int(np.searchsorted(axial_edges, u_min, side="right") - 1)))
        k1 = max(0, min(AXIAL_SLABS - 1, int(np.searchsorted(axial_edges, u_max, side="left"))))
        j0 = max(0, min(TRANSVERSE_STRIPS - 1, int(np.searchsorted(transverse_edges, q_min, side="right") - 1)))
        j1 = max(0, min(TRANSVERSE_STRIPS - 1, int(np.searchsorted(transverse_edges, q_max, side="left"))))
        nb = float(np.dot(normal, frame.transverse))
        nc = float(np.dot(normal, frame.complement))
        normal_weight = nb * nb + nc * nc
        z_real_factor = nb * nb - nc * nc
        z_imag_factor = 2.0 * nb * nc
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                cell = clip_axial_transverse_cell(
                    polygon,
                    frame,
                    float(axial_edges[k]),
                    float(axial_edges[k + 1]),
                    float(transverse_edges[j]),
                    float(transverse_edges[j + 1]),
                )
                if len(cell) < 3:
                    continue
                area = polygon_area(cell)
                if not math.isfinite(area) or area <= 0.0:
                    continue
                intersections += 1
                if intersections > MAX_TRIANGLE_CELL_INTERSECTIONS:
                    raise OPGTContractError(
                        "V34_OPGT_RESOURCE_STOP",
                        "triangle-cell intersection cap",
                    )
                factor = area / scale_area * ACCUMULATOR_Q
                qw = int(round(factor * normal_weight))
                qr = int(round(factor * z_real_factor))
                qi = int(round(factor * z_imag_factor))
                _checked_add(weights, (k, j), qw)
                _checked_add(real, (k, j), qr)
                _checked_add(imag, (k, j), qi)
    return weights, real, imag, frame, intersections


def _checked_axis_sum(array: np.ndarray, *, axis: int) -> np.ndarray:
    """Sum int64 cells with Python integers, then expose bounded float64 values."""
    if axis == 0:
        values = [sum(int(array[k, j]) for k in range(array.shape[0])) for j in range(array.shape[1])]
    elif axis == 1:
        values = [sum(int(array[k, j]) for j in range(array.shape[1])) for k in range(array.shape[0])]
    else:
        raise ValueError("axis must be 0 or 1")
    if any(value < -INT64_LIMIT - 1 or value > INT64_LIMIT for value in values):
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "fixed aggregate overflow")
    return np.asarray(values, dtype=np.float64)


def _retained_and_tails(strip_weights: np.ndarray) -> tuple[list[int], list[int], list[int]] | None:
    total = sum(int(value) for value in strip_weights)
    if total <= 0:
        return None
    removable = total * OUTER_MASS_FRACTION
    left = 0
    removed = 0
    while left < TRANSVERSE_STRIPS and removed + int(strip_weights[left]) <= removable:
        removed += int(strip_weights[left])
        left += 1
    right = TRANSVERSE_STRIPS - 1
    removed = 0
    while right >= left and removed + int(strip_weights[right]) <= removable:
        removed += int(strip_weights[right])
        right -= 1
    retained = list(range(left, right + 1))
    retained_mass = sum(int(strip_weights[j]) for j in retained)
    if retained_mass <= 0:
        return None
    target = retained_mass * TAIL_MASS_FRACTION
    left_tail: list[int] = []
    mass = 0
    for j in retained:
        left_tail.append(j)
        mass += int(strip_weights[j])
        if mass >= target:
            break
    right_tail: list[int] = []
    mass = 0
    for j in reversed(retained):
        right_tail.append(j)
        mass += int(strip_weights[j])
        if mass >= target:
            break
    right_tail.reverse()
    if not left_tail or not right_tail or left_tail[-1] >= right_tail[0]:
        return None
    return retained, left_tail, right_tail


def _group_loss(
    indexes: Sequence[int],
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
) -> tuple[float, float, float, float, float]:
    weight = float(sum(float(weights[j]) for j in indexes))
    z_real = float(sum(float(real[j]) for j in indexes))
    z_imag = float(sum(float(imag[j]) for j in indexes))
    free = sum(_e0(float(weights[j]), float(real[j]), float(imag[j])) for j in indexes)
    constant = _e0(weight, z_real, z_imag)
    return weight, z_real, z_imag, free, constant


def _emission_row(
    index: int,
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    qlo: int,
) -> np.ndarray:
    qhi = qlo + PHASE_WINDOW_WIDTH
    value = 2.0 * float(weights[index]) - 2.0 * (
        float(real[index]) * PHASE_COSINE[qlo:qhi]
        + float(imag[index]) * PHASE_SINE[qlo:qhi]
    )
    return np.maximum(value, 0.0)


def _window_dp(
    indexes: Sequence[int],
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    qlo: int,
) -> tuple[float, int]:
    """Streaming minimum nondecreasing path with at least one strict increase."""
    infinity = np.inf
    same = _emission_row(indexes[0], weights, real, imag, qlo)
    changed = np.full_like(same, infinity)
    for index in indexes[1:]:
        row = _emission_row(index, weights, real, imag, qlo)
        strict_same = np.empty_like(same)
        np.minimum.accumulate(same, out=strict_same)
        strict_same[1:] = strict_same[:-1]
        strict_same[0] = infinity
        changed_prefix = np.empty_like(changed)
        np.minimum.accumulate(changed, out=changed_prefix)
        np.minimum(changed_prefix, strict_same, out=changed_prefix)
        changed_prefix += row
        same += row
        changed = changed_prefix
    final = int(np.argmin(changed))
    return float(changed[final]), final


def _window_prefix(
    indexes: Sequence[int],
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    qlo: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    same = _emission_row(indexes[0], weights, real, imag, qlo)
    changed = np.full_like(same, np.inf)
    for index in indexes[1:count]:
        row = _emission_row(index, weights, real, imag, qlo)
        strict_same = np.empty_like(same)
        np.minimum.accumulate(same, out=strict_same)
        strict_same[1:] = strict_same[:-1]
        strict_same[0] = np.inf
        changed_prefix = np.empty_like(changed)
        np.minimum.accumulate(changed, out=changed_prefix)
        np.minimum(changed_prefix, strict_same, out=changed_prefix)
        changed_prefix += row
        same += row
        changed = changed_prefix
    return same, changed


def _backtrack_window(
    indexes: Sequence[int],
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    qlo: int,
    final: int,
) -> list[int]:
    """Recover without retaining an O(strips*phases) predecessor matrix."""
    path = [final]
    state = "changed"
    q = final
    for position in range(len(indexes) - 1, 0, -1):
        same, changed = _window_prefix(indexes, weights, real, imag, qlo, position)
        if state == "same":
            predecessor = q
            next_state = "same"
        else:
            changed_slice = changed[: q + 1]
            changed_q = int(np.argmin(changed_slice))
            changed_value = float(changed_slice[changed_q])
            if q == 0:
                same_q, same_value = 0, np.inf
            else:
                same_slice = same[:q]
                same_q = int(np.argmin(same_slice))
                same_value = float(same_slice[same_q])
            if changed_value <= same_value:
                predecessor, next_state = changed_q, "changed"
            else:
                predecessor, next_state = same_q, "same"
        path.append(predecessor)
        q, state = predecessor, next_state
    path.reverse()
    return path


def _monotone_fit(
    indexes: Sequence[int],
    weights: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
) -> tuple[float, list[int], str]:
    """Evaluate both strip traversals and every legal sub-pi phase window."""
    best: tuple[float, int, int, int, list[int], str] | None = None
    for direction_rank, (ordered, direction) in enumerate(
        ((list(indexes), "forward"), (list(reversed(indexes)), "reverse"))
    ):
        for qlo in range(PHASE_COUNT - PHASE_WINDOW_WIDTH + 1):
            cost, final = _window_dp(ordered, weights, real, imag, qlo)
            key = (cost, direction_rank, qlo, final)
            if best is None or key[:4] < best[:4]:
                best = (cost, direction_rank, qlo, final, ordered, direction)
    if best is None or not math.isfinite(best[0]):
        return math.inf, [], "forward"
    cost, _rank, qlo, final, ordered, direction = best
    local_path = _backtrack_window(ordered, weights, real, imag, qlo, final)
    path = [qlo + value for value in local_path]
    if direction == "reverse":
        path.reverse()
    return cost, path, direction


def _base_decision(candidate: Candidate) -> dict[str, Any]:
    line_payload = {
        "line_start": candidate.raw["render_line_start"],
        "line_end": candidate.raw["render_line_end"],
        "score": candidate.raw.get("score"),
    }
    return {
        "candidate_id": candidate.candidate_id,
        "frozen_v16_order": candidate.frozen_order,
        "candidate_sha256": _sha(candidate.raw),
        "passthrough_line_sha256": _sha(line_payload),
        "line_start_xyz": candidate.line_start.tolist(),
        "line_end_xyz": candidate.line_end.tolist(),
        "scale": candidate.scale,
        "nms_reason": "",
    }


def _legacy_decision(candidate: Candidate) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = dict(
        _base_decision(candidate),
        action=ACTION_LEGACY,
        reason="LEGACY_PASS",
        predicate_passes=[True] * 7,
        metrics_i64=[0] * 12,
        phase_path_sha256="0" * 64,
    )
    return decision, {"kind": "legacy", "cell_weights": [], "phase_path": []}


def _analyze_candidate(candidate: Candidate, mesh: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _base_decision(candidate)
    if not candidate.eligible or not (8.0 * candidate.scale <= candidate.length <= 512.0 * candidate.scale):
        decision = dict(
            base,
            action=ACTION_NONE,
            reason="NUMERICAL_OR_RESOURCE_INVALID",
            predicate_passes=[False] * 7,
            metrics_i64=[0] * 12,
            phase_path_sha256="0" * 64,
        )
        return decision, {"kind": "invalid", "cell_weights": [], "phase_path": []}

    weights_i, real_i, imag_i, frame, intersections = _candidate_measure(candidate, mesh)
    weights = weights_i.astype(np.float64)
    real = real_i.astype(np.float64)
    imag = imag_i.astype(np.float64)
    strip_weights = _checked_axis_sum(weights_i, axis=0)
    strip_real = _checked_axis_sum(real_i, axis=0)
    strip_imag = _checked_axis_sum(imag_i, axis=0)
    support = _retained_and_tails(strip_weights)
    if support is None:
        decision = dict(
            base,
            action=ACTION_NONE,
            reason="NUMERICAL_OR_RESOURCE_INVALID",
            predicate_passes=[False] * 7,
            metrics_i64=[intersections] + [0] * 11,
            phase_path_sha256="0" * 64,
        )
        return decision, {"kind": "invalid", "cell_weights": weights.tolist(), "phase_path": []}
    retained, left_tail, right_tail = support
    retained_nonzero = [j for j in retained if strip_weights[j] > 0.0]
    retained_weight = float(sum(strip_weights[j] for j in retained))

    slab_weights = np.asarray(
        [sum(float(weights[k, j]) for j in retained) for k in range(AXIAL_SLABS)],
        dtype=np.float64,
    )
    denominator = float(np.dot(slab_weights, slab_weights))
    effective = 0.0 if denominator <= 0.0 else retained_weight * retained_weight / denominator
    p1 = math.isfinite(effective) and effective >= MIN_EFFECTIVE_AXIAL_SUPPORT

    lw, lr, li, lfree, lconst = _group_loss(left_tail, strip_weights, strip_real, strip_imag)
    rw, rr, ri, rfree, rconst = _group_loss(right_tail, strip_weights, strip_real, strip_imag)
    left_plateau = lconst <= 2.0 * lfree + _epsilon(lw)
    right_plateau = rconst <= 2.0 * rfree + _epsilon(rw)
    p2 = left_plateau and right_plateau

    joint = _e0(lw + rw, lr + rr, li + ri)
    p3 = joint > 2.0 * (lconst + rconst) + _epsilon(lw + rw)

    left_phase = 0.5 * math.atan2(li, lr)
    right_phase = 0.5 * math.atan2(ri, rr)
    tail_set = set(left_tail) | set(right_tail)
    transition: list[int] = []
    for j in retained_nonzero:
        if j in tail_set:
            continue
        wj, rj, ij = float(strip_weights[j]), float(strip_real[j]), float(strip_imag[j])
        free = _e0(wj, rj, ij)
        allowance = 2.0 * free + _epsilon(wj)
        if _loss(wj, rj, ij, left_phase) > allowance and _loss(wj, rj, ij, right_phase) > allowance:
            transition.append(j)
    transition_weight = float(sum(strip_weights[j] for j in transition))
    p4 = transition_weight >= TRANSITION_MASS_FRACTION * retained_weight

    axial_local = 0.0
    for k in range(AXIAL_SLABS):
        for j in retained:
            axial_local += _e0(weights[k, j], real[k, j], imag[k, j])
    axial_pooled = sum(
        _e0(strip_weights[j], strip_real[j], strip_imag[j]) for j in retained
    )
    p5 = axial_pooled <= 2.0 * axial_local + _epsilon(retained_weight)

    monotone_loss, phase_path, path_direction = _monotone_fit(
        retained_nonzero, strip_weights, strip_real, strip_imag
    )
    free_loss = sum(
        _e0(strip_weights[j], strip_real[j], strip_imag[j]) for j in retained_nonzero
    )
    p6 = monotone_loss <= 2.0 * free_loss + _epsilon(retained_weight)

    predicate_passes = [True, p1, p2, p3, p4, p5, p6]
    if not p1:
        reason = "INSUFFICIENT_EFFECTIVE_AXIAL_SUPPORT"
    elif not left_plateau:
        reason = "LEFT_TAIL_NOT_PLATEAU"
    elif not right_plateau:
        reason = "RIGHT_TAIL_NOT_PLATEAU"
    elif not p3:
        reason = "TAIL_NORMALS_NOT_SEPARATED"
    elif not p4:
        reason = "INSUFFICIENT_TRANSITION_MASS"
    elif not p5:
        reason = "AXIAL_INCOHERENCE"
    elif not p6:
        reason = "NONMONOTONE_GAUSS_PATH"
    else:
        reason = "OPGT_PASS"

    phase_path_sha = _sha({"strips": retained_nonzero, "q": phase_path, "direction": path_direction})
    metrics = [
        intersections,
        len(retained),
        len(retained_nonzero),
        len(left_tail),
        len(right_tail),
        len(transition),
        int(round(effective * 1_000_000.0)),
        int(round(lconst - 2.0 * lfree)),
        int(round(rconst - 2.0 * rfree)),
        int(round(axial_pooled - 2.0 * axial_local)),
        int(round(monotone_loss - 2.0 * free_loss)) if math.isfinite(monotone_loss) else INT64_LIMIT,
        sum((1 << index) for index, passed in enumerate(predicate_passes) if passed),
    ]
    decision = dict(
        base,
        action=ACTION_RESCUE if reason == "OPGT_PASS" else ACTION_NONE,
        reason=reason,
        predicate_passes=predicate_passes,
        metrics_i64=metrics,
        phase_path_sha256=phase_path_sha,
    )
    free_phases = [
        0.5 * math.atan2(float(strip_imag[j]), float(strip_real[j]))
        if strip_weights[j] > 0.0
        else None
        for j in range(TRANSVERSE_STRIPS)
    ]
    visual = {
        "kind": "opgt",
        "frame": {
            "axis": frame.axis.tolist(),
            "transverse": frame.transverse.tolist(),
            "complement": frame.complement.tolist(),
            "length": frame.length,
            "scale": frame.scale,
        },
        "cell_weights": weights.tolist(),
        "retained_strips": retained,
        "left_tail": left_tail,
        "right_tail": right_tail,
        "transition_strips": transition,
        "free_phases": free_phases,
        "phase_path_strips": retained_nonzero,
        "phase_path": phase_path,
        "phase_direction": path_direction,
        "first_failing_predicate": reason,
    }
    return decision, visual


def build_fixed_cache(
    decisions: Sequence[dict[str, Any]],
    source_sha256: str,
    config_sha256: str,
) -> bytes:
    header = bytearray(CACHE_HEADER_BYTES)
    header[:8] = b"OPGTV1\0\0"
    struct.pack_into("<HHI", header, 8, SCHEMA_VERSION, CACHE_RECORD_BYTES, len(decisions))
    header[16:48] = bytes.fromhex(source_sha256)
    header[48:64] = bytes.fromhex(config_sha256)[:16]
    records = bytearray()
    for decision in sorted(decisions, key=lambda row: (int(row["frozen_v16_order"]), int(row["candidate_id"]))):
        record = bytearray(CACHE_RECORD_BYTES)
        record[0:32] = bytes.fromhex(decision["candidate_sha256"])
        record[32:64] = bytes.fromhex(decision["passthrough_line_sha256"])
        record[64:96] = bytes.fromhex(decision["phase_path_sha256"])
        flags = (
            (1 if decision["action"] == ACTION_RESCUE else 0)
            | (2 if decision["action"] == ACTION_LEGACY else 0)
            | (4 if decision.get("nms_reason") == "ORDINARY_NMS_SUPPRESSED" else 0)
        )
        struct.pack_into(
            "<QIIHH",
            record,
            96,
            int(decision["candidate_id"]),
            int(decision["frozen_v16_order"]),
            REASON_CODE.get(decision["reason"], 65535),
            flags,
            sum(bool(value) for value in decision["predicate_passes"]),
        )
        struct.pack_into(
            "<12q",
            record,
            116,
            *[max(-INT64_LIMIT - 1, min(INT64_LIMIT, int(value))) for value in decision["metrics_i64"]],
        )
        records.extend(record)
    return bytes(header + records)


def select_unified_v34(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    if worker_count < 1 or worker_count > MAX_WORKERS:
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "worker count")
    candidates = parse_candidates(raw_candidates)
    rescue_candidates = [candidate for candidate in candidates if not candidate.selected_legacy]
    if len(rescue_candidates) > MAX_RESCUE_CANDIDATES:
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "rescue candidate cap")
    scales = {float(candidate.scale) for candidate in candidates}
    if len(scales) != 1:
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "empty or mixed candidate scale")
    try:
        mesh = canonical_triangle_mesh(raw_source, next(iter(scales)))
    except CandidatePrismError as exc:
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", str(exc)) from exc

    analyzed = [
        _legacy_decision(candidate)
        if candidate.selected_legacy
        else _analyze_candidate(candidate, mesh)
        for candidate in candidates
    ]
    decisions = [row[0] for row in analyzed]
    visuals = [row[1] for row in analyzed]
    legacy = [
        (candidate, decision)
        for candidate, decision in zip(candidates, decisions)
        if decision["action"] == ACTION_LEGACY
    ]
    additions = [
        (candidate, decision)
        for candidate, decision in zip(candidates, decisions)
        if decision["action"] == ACTION_RESCUE
    ]
    ordered = sorted(legacy, key=lambda row: (row[0].frozen_order, row[0].candidate_id))
    ordered += sorted(additions, key=lambda row: row[0].candidate_id)
    accepted, suppressed = ordinary_nms(ordered)
    for decision in decisions:
        if decision["candidate_id"] in suppressed:
            decision["nms_reason"] = "ORDINARY_NMS_SUPPRESSED"
    final_lines = [
        {
            "candidate_id": candidate.candidate_id,
            "line_start": candidate.raw["render_line_start"],
            "line_end": candidate.raw["render_line_end"],
            "score": candidate.raw.get("score"),
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v34_opgt_rescue",
        }
        for candidate, decision in accepted
    ]
    cache = build_fixed_cache(decisions, mesh.source_sha256, config_sha256)
    if len(cache) > CACHE_HEADER_BYTES + CACHE_RECORD_BYTES * len(decisions):
        raise OPGTContractError("V34_OPGT_RESOURCE_STOP", "certificate cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sherman_qc_f1_v34_opgt_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "research_only": True,
        "source_only": True,
        "rescue_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "labels_used": False,
        "cad_used": False,
        "v32_state_used": False,
        "v33_features_used": False,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_nms_rescue_count": sum(decision["action"] == ACTION_RESCUE for decision in decisions),
        "rescue_count": sum(line["source_lane"] == "v34_opgt_rescue" for line in final_lines),
        "ordinary_nms_call_count": 1,
        "candidate_added_after_nms": False,
        "cache_bytes": len(cache),
        "cache_record_bytes": CACHE_RECORD_BYTES,
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "source_sha256": mesh.source_sha256,
        "_ephemeral": {
            "candidate_visuals": visuals,
            "fixed_cache": cache,
            "mesh": mesh,
            "candidates": candidates,
        },
    }
