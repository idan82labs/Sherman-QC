"""Bounded large-stream routing to unchanged LOCAL observed-window diagnostics.

Not full-scan discovery, physical coverage, CAD registration or event mapping.
Every selected raw row is retained or its whole window is unsupported. Dense
windows are never downsampled. No geometry runs before successful stream EOF.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from research.geometry.position_plane_regions import MAX_POINTS, _integer
from research.geometry.position_axis_hypothesis import _number
from research.geometry.tangency_audited_bend_candidates import analyze_tangency_audited_bend_candidates


MAX_WINDOWS = 16
MAX_TOTAL_ROWS = 2_000_000
MAX_CHUNK_ROWS = 65_536
MAX_CHUNKS = 4096
MAX_RESULT_BYTES = 16_777_216
_EPS = float(np.finfo(float).eps)


def _windows(windows):
    if not isinstance(windows, (list, tuple)) or not 1 <= len(windows) <= MAX_WINDOWS:
        raise ValueError("windows must explicitly contain one to sixteen observed boxes")
    parsed, names = [], set()
    for window in windows:
        if not isinstance(window, dict) or set(window) != {"window_id", "lower", "upper"}:
            raise ValueError("each window requires exactly window_id, lower and upper")
        name = window["window_id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", name) or name in names:
            raise ValueError("window_id must be a unique bounded identifier")
        bounds = []
        for key in ("lower", "upper"):
            values = window[key]
            if (not isinstance(values, (list, tuple, np.ndarray))
                    or isinstance(values, np.ndarray) and values.ndim != 1
                    or len(values) != 3):
                raise ValueError("window bounds require three explicit finite coordinates")
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                   for v in values):
                raise ValueError("window coordinates must be real numbers")
            try:
                vector = np.array(values, dtype=float, copy=True)
            except (ValueError, OverflowError) as exc:
                raise ValueError("window coordinates must be representable") from exc
            if not np.isfinite(vector).all():
                raise ValueError("window coordinates must be finite")
            bounds.append(vector)
        with np.errstate(over="ignore", invalid="ignore"):
            span = bounds[1]-bounds[0]
        if not np.isfinite(span).all() or not np.all(span > 0):
            raise ValueError("each upper bound must exceed its lower bound by a finite amount")
        parsed.append(dict(window_id=name, lower=bounds[0], upper=bounds[1]))
        names.add(name)
    return parsed


def _fail(result, reason):
    diagnostics = result["diagnostics"]
    diagnostics["discarded_result_payload_bytes"] += diagnostics["result_payload_bytes"]
    diagnostics["result_payload_bytes"] = 0
    result.update(status="ABSTAINED", analyses=[], reasons=[reason],
                  search_complete_within_declared_policy=False)
    return result


def analyze_streamed_window_bends(
    chunks, *, windows, engine_policy, boundary_margin, max_total_rows,
    max_chunk_rows, max_chunks, max_result_bytes,
):
    """Consume cooperative, already-materialized NumPy Nx3 chunks exactly once.

    Budgets bound this consumer's retained work, not a producer's allocation or
    time blocked inside next(). Every coordinate is validated before filtering.
    ANY incomplete/unsupported window clears all batch analyses; no successful
    prefix is returned. Local diagnostics never certify unseen outside geometry.
    """
    boxes = _windows(windows)
    row_cap = _integer(max_total_rows, "max_total_rows", 1, MAX_TOTAL_ROWS)
    chunk_cap = _integer(max_chunk_rows, "max_chunk_rows", 1, MAX_CHUNK_ROWS)
    chunk_count_cap = _integer(max_chunks, "max_chunks", 1, MAX_CHUNKS)
    byte_cap = _integer(max_result_bytes, "max_result_bytes", 1, MAX_RESULT_BYTES)
    margin = _number(boundary_margin, "boundary_margin")
    if not isinstance(engine_policy, dict):
        raise ValueError("engine_policy must explicitly declare the unchanged local policy")
    # Pure source-only empty call validates every inherited policy before reading
    # the producer. It performs no geometry fit and is counted separately.
    validation = analyze_tangency_audited_bend_candidates([], **engine_policy)
    # Snapshot accepted NumPy scalars/short policy arrays into ordinary immutable-
    # by-producer JSON data before calling next(). Window bounds are also copied.
    def scalar(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError("unsupported policy value")
    engine_policy = json.loads(json.dumps(engine_policy, default=scalar, allow_nan=False))
    epsilon = validation["base_analysis"]["plane_discovery"]["position_error_bound"]
    minimum_margin = max(float(engine_policy["max_neighborhood_radius"]),
                         float(engine_policy["max_link_distance"])+2*epsilon)
    if not math.isfinite(minimum_margin) or margin < minimum_margin:
        raise ValueError("boundary_margin must cover seed cap and link cap plus 2 epsilon")
    states = [dict(window_id=box["window_id"], status="PENDING", selected_raw_count=0,
                   boundary_sensitive_raw_count=0, numerical_boundary_allowance=None,
                   local_engine_complete=False) for box in boxes]
    result = {
        "schema_version": 1, "kind": "STREAMED_OBSERVED_WINDOW_BEND_DIAGNOSTICS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": engine_policy["unit"], "analyses": [], "window_dispositions": states,
        "declared_windows": [dict(window_id=box["window_id"], lower=box["lower"].tolist(),
                                  upper=box["upper"].tolist()) for box in boxes],
        "stream_fully_consumed": False, "search_complete_within_declared_policy": False,
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "input_row_count": 0, "reasons": [],
        "policy": {
            "max_total_rows": row_cap, "max_chunk_rows": chunk_cap, "max_chunks": chunk_count_cap,
            "max_windows": MAX_WINDOWS, "max_raw_rows_per_window": MAX_POINTS,
            "max_result_bytes": byte_cap, "result_byte_budget_scope": "SUM_OF_COMPACT_JSON_WINDOW_ANALYSIS_PAYLOADS",
            "boundary_margin": margin, "minimum_declared_boundary_margin": minimum_margin,
            "boundary_rule": "ALL_SELECTED_RAW_ROWS_STRICTLY_BEYOND_INNER_MARGIN_PLUS_ENGINEERING_ALLOWANCE",
            "window_membership": "CLOSED_BOX_ON_OBSERVED_COORDINATES_NOT_LATENT_CORRECTED_MEMBERSHIP",
            "local_analysis_scope": "DECLARED_OBSERVED_WINDOW_ONLY_NOT_GLOBAL_ENGINE_EQUIVALENCE",
            "overlapping_windows_are_independent": True, "cross_window_event_merging": False,
            "downsampling_or_residual_trimming": False, "analysis_before_stream_eof": False,
            "any_window_failure_clears_all_analyses": True,
            "input_row_count_scope": "VALIDATED_AND_PROCESSED_ROWS_EXCLUDES_REJECTED_CHUNK",
            "local_engine_call_count_scope": "NONEMPTY_ENGINE_ATTEMPTS_EMPTY_VALIDATION_SEPARATE",
            "producer_allocation_or_blocking_time_is_bounded": False,
            "engine_policy": dict(engine_policy),
        },
        "diagnostics": {
            "processed_chunks": 0, "received_chunks": 0, "peak_chunk_rows": 0,
            "rows_in_any_window": 0, "rows_outside_all_windows": 0,
            "total_window_row_memberships": 0, "peak_retained_window_rows": 0,
            "retained_window_rows": 0, "policy_validation_empty_engine_calls": 1,
            "local_engine_calls": 0, "result_payload_bytes": 0,
            "discarded_result_payload_bytes": 0, "serialized_window_payload_bytes": 0,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "SCANNER_SCALE_INGESTION_NOT_DENSE_FULL_SCAN_DISCOVERY",
            "DECLARED_WINDOWS_NOT_VERIFIED_CAD_EVENTS_OR_REGISTRATION",
            "EOF_IS_COMPLETE_SUPPLIED_STREAM_NOT_PHYSICAL_CAPTURE_COVERAGE",
            "LOCAL_WINDOW_GRAPH_NOT_GLOBAL_GRAPH_OR_COMPONENT_EQUIVALENCE",
            "BOUNDARY_CENSOR_NOT_PHYSICAL_ISOLATION_OR_CONTINUITY_CERTIFICATE",
            "DENSE_WINDOWS_ABSTAIN_NO_SAMPLED_SUBSET_RESCUE", "RAW_REPEATS_STILL_CONSUME_WINDOW_BUDGET",
            "OVERLAPPING_WINDOW_RESULTS_NOT_ADDITIVE_PHYSICAL_EVENT_COUNTS",
            "PRODUCER_MAY_BLOCK_OR_ALLOCATE_BEFORE_CONSUMER_CAN_CHECK_CHUNK",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ],
    }
    try:
        iterator = iter(chunks)
    except TypeError:
        raise ValueError("chunks must be an explicit cooperative iterable of NumPy arrays") from None
    except Exception:
        return _fail(result, "INPUT_STREAM_FAILED")
    buffers = [[] for _ in boxes]
    diagnostics = result["diagnostics"]
    while True:
        try:
            raw = next(iterator)
        except StopIteration:
            result["stream_fully_consumed"] = True
            break
        except Exception:
            return _fail(result, "INPUT_STREAM_FAILED")
        diagnostics["received_chunks"] += 1
        if diagnostics["received_chunks"] > chunk_count_cap:
            return _fail(result, "CHUNK_COUNT_BUDGET_EXCEEDED")
        if (not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[1] != 3
                or raw.dtype.kind not in "iuf" or raw.dtype.itemsize > 8):
            return _fail(result, "INVALID_CHUNK_ARRAY")
        n = len(raw)
        if n > chunk_cap:
            return _fail(result, "CHUNK_ROW_BUDGET_EXCEEDED")
        if result["input_row_count"]+n > row_cap:
            return _fail(result, "TOTAL_ROW_BUDGET_EXCEEDED")
        if raw.dtype.kind in "iu" and (np.any(raw > 2**53) or np.any(raw < -2**53)):
            return _fail(result, "INTEGER_COORDINATE_PRECISION_UNSUPPORTED")
        block = np.asarray(raw, dtype=float)
        if not np.isfinite(block).all():
            return _fail(result, "NONFINITE_CHUNK_COORDINATE")
        offset = result["input_row_count"]
        result["input_row_count"] += n
        diagnostics["processed_chunks"] += 1
        diagnostics["peak_chunk_rows"] = max(diagnostics["peak_chunk_rows"], n)
        membership = np.zeros(n, dtype=bool)
        for index, box in enumerate(boxes):
            mask = np.all((block >= box["lower"]) & (block <= box["upper"]), axis=1)
            membership |= mask
            selected = int(np.count_nonzero(mask))
            state = states[index]
            state["selected_raw_count"] += selected
            diagnostics["total_window_row_memberships"] += selected
            if state["selected_raw_count"] > MAX_POINTS:
                buffers[index].clear()
                state["status"] = "WINDOW_RAW_CAP_EXCEEDED"
            elif selected:
                buffers[index].extend((offset+int(i), tuple(block[i])) for i in np.flatnonzero(mask))
        inside = int(np.count_nonzero(membership))
        diagnostics["rows_in_any_window"] += inside
        diagnostics["rows_outside_all_windows"] += n-inside
        diagnostics["retained_window_rows"] = sum(map(len, buffers))
        diagnostics["peak_retained_window_rows"] = max(diagnostics["peak_retained_window_rows"],
                                                       diagnostics["retained_window_rows"])
    # All raw rows were validated, including those outside every window. Review
    # EVERY window before any core call; no good-window prefix hides an overflow.
    blocked = False
    for state, box, buffer in zip(states, boxes, buffers):
        if state["status"] == "WINDOW_RAW_CAP_EXCEEDED":
            blocked = True
            continue
        if not buffer:
            state["status"] = "EMPTY_OBSERVED_WINDOW"
            continue
        p = np.asarray([row for _, row in buffer])
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = max(float(np.max(np.abs(p))), float(np.max(np.abs(box["lower"]))),
                            float(np.max(np.abs(box["upper"]))))
            span = math.hypot(*(box["upper"]-box["lower"]))
            guard = 512*_EPS*(magnitude+span)
            clearance = np.min(np.minimum(p-box["lower"], box["upper"]-p), axis=1)
            boundary = clearance <= margin+guard
        if not math.isfinite(guard) or not np.isfinite(clearance).all():
            state["status"] = "WINDOW_BOUNDARY_NUMERICAL_UNSUPPORTED"
            blocked = True
            continue
        state["numerical_boundary_allowance"] = guard
        state["boundary_sensitive_raw_count"] = int(np.count_nonzero(boundary))
        state["status"] = "WINDOW_BOUNDARY_CENSORED" if np.any(boundary) else "READY"
        blocked |= bool(np.any(boundary))
    if blocked:
        return _fail(result, "UNSUPPORTED_OBSERVED_WINDOW")
    for state, box, buffer in zip(states, boxes, buffers):
        if state["status"] == "EMPTY_OBSERVED_WINDOW":
            continue
        p = np.asarray([row for _, row in buffer])
        diagnostics["local_engine_calls"] += 1
        try:
            audit = analyze_tangency_audited_bend_candidates(p, **engine_policy)
        except Exception:
            state["status"] = "WINDOW_ENGINE_FAILED"
            return _fail(result, "WINDOW_ENGINE_FAILED")
        if not audit["search_complete_within_declared_policy"]:
            state["status"] = "WINDOW_ENGINE_INCOMPLETE"
            return _fail(result, "WINDOW_ENGINE_INCOMPLETE")
        row_ids = {}
        for row_id, position in buffer:
            row_ids.setdefault(position, []).append(row_id)
        payload = {
            "window_id": box["window_id"], "lower": box["lower"].tolist(), "upper": box["upper"].tolist(),
            "canonical_to_stream_rows": [row_ids[tuple(position)] for position in audit["base_analysis"]["positions"]],
            "analysis": audit,
            "scope": "LOCAL_OBSERVED_WINDOW_DIAGNOSTIC_NOT_EXPECTED_OR_OBSERVED_PHYSICAL_EVENT",
        }
        try:
            encoded_bytes = len(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        except (ValueError, OverflowError):
            state["status"] = "WINDOW_PAYLOAD_UNAVAILABLE"
            return _fail(result, "WINDOW_PAYLOAD_UNAVAILABLE")
        diagnostics["serialized_window_payload_bytes"] += encoded_bytes
        if diagnostics["result_payload_bytes"]+encoded_bytes > byte_cap:
            state["status"] = "RESULT_PAYLOAD_BUDGET_EXCEEDED"
            return _fail(result, "RESULT_PAYLOAD_BUDGET_EXCEEDED")
        diagnostics["result_payload_bytes"] += encoded_bytes
        state.update(status="LOCAL_DIAGNOSTIC_COMPLETE", local_engine_complete=True)
        result["analyses"].append(payload)
    result["search_complete_within_declared_policy"] = True
    if result["analyses"]:
        result["status"] = "OBSERVED_WINDOW_GEOMETRY_DIAGNOSTICS"
    else:
        result["reasons"] = ["NO_OBSERVED_ROWS_IN_DECLARED_WINDOWS"]
    return result
