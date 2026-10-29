#!/usr/bin/env python3
"""Deterministic source-row intake correction for EXP_CAD_0031 CF-PDG V4.

This module is deliberately scientific-model-free.  It removes only rows
whose point/normal interface values violate the frozen V4 contract, carries
their acquisition-row identities, and reports exact actionability accounting.
It never estimates or replaces a normal and never consumes evaluator data.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np


SCHEMA = "EXP_CAD_0031_SOURCE_INTERFACE_REPORT_V1"
NORMAL_NORM_EPSILON = 1e-12
MAXIMUM_INVALID_FRACTION = 0.01
MINIMUM_VALID_ROW_COUNT = 100_000
REASON_ORDER = (
    "NONFINITE_POINT",
    "NONFINITE_NORMAL",
    "NORMAL_NORM_LE_1E_12",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity_bytes(identities: np.ndarray, *, sort_as_set: bool) -> bytes:
    values = np.asarray(identities)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("original row identities must be a one-dimensional integer array")
    if np.any(values < 0):
        raise ValueError("original row identities must be nonnegative")
    maximum = int(np.iinfo(np.uint64).max)
    if len(values) and max(int(value) for value in values) > maximum:
        raise ValueError("original row identity exceeds uint64")
    canonical = np.asarray(values, dtype="<u8")
    if len(np.unique(canonical)) != len(canonical):
        raise ValueError("original row identities must be unique")
    if sort_as_set:
        canonical = np.sort(canonical, kind="mergesort")
    return np.ascontiguousarray(canonical, dtype="<u8").tobytes()


def identity_sha256(identities: np.ndarray, *, sort_as_set: bool = True) -> str:
    return _sha256(identity_bytes(identities, sort_as_set=sort_as_set))


def numeric_array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("numeric source array must be N-by-3")
    return _sha256(np.ascontiguousarray(array, dtype="<f8").tobytes())


def stable_normal_norm(normals: np.ndarray) -> np.ndarray:
    array = np.asarray(normals, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("normal array must be N-by-3")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.hypot(np.hypot(array[:, 0], array[:, 1]), array[:, 2])


def _validate_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    original_row_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("source points must be N-by-3")
    if normal_array.shape != point_array.shape:
        raise ValueError("source normals must match source point shape")
    if original_row_ids is None:
        identities = np.arange(len(point_array), dtype=np.uint64)
    else:
        identities = np.asarray(original_row_ids)
        if identities.shape != (len(point_array),):
            raise ValueError("original row identity count mismatch")
        # Validation plus canonical uint64 conversion without changing order.
        identity_bytes(identities, sort_as_set=False)
        identities = np.asarray(identities, dtype=np.uint64)
    return point_array, normal_array, identities


def evaluate_actionability(
    *,
    original_count: int,
    retained_count: int,
) -> dict[str, Any]:
    if original_count < 0 or retained_count < 0 or retained_count > original_count:
        raise ValueError("invalid source-interface counts")
    rejected_count = int(original_count - retained_count)
    invalid_fraction = (
        float(rejected_count / original_count) if original_count else None
    )
    reasons = []
    if original_count == 0:
        reasons.append("EMPTY_SOURCE_MEMBER")
    if invalid_fraction is not None and invalid_fraction > MAXIMUM_INVALID_FRACTION:
        reasons.append("INVALID_FRACTION_EXCEEDS_0.01")
    if retained_count < MINIMUM_VALID_ROW_COUNT:
        reasons.append("VALID_ROW_COUNT_BELOW_100000")
    return {
        "actionable": not reasons,
        "actionability_reasons": reasons,
        "invalid_fraction": invalid_fraction,
        "maximum_invalid_fraction": MAXIMUM_INVALID_FRACTION,
        "maximum_invalid_fraction_inclusive": True,
        "minimum_valid_row_count": MINIMUM_VALID_ROW_COUNT,
        "minimum_valid_row_count_inclusive": True,
    }


def filter_source_rows(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    member_token: str,
    original_row_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Filter only frozen invalid predicates and retain acquisition identities."""

    point_array, normal_array, identities = _validate_arrays(
        points, normals, original_row_ids
    )
    point_nonfinite = ~np.all(np.isfinite(point_array), axis=1)
    normal_components_nonfinite = ~np.all(np.isfinite(normal_array), axis=1)
    normal_norm = stable_normal_norm(normal_array)
    # The scientific parent is byte-frozen and canonicalizes through
    # np.linalg.norm.  Reject finite-but-overflowing rows here so every retained
    # normal is guaranteed to cross that unchanged boundary successfully.
    with np.errstate(over="ignore", invalid="ignore"):
        parent_normal_norm = np.linalg.norm(normal_array, axis=1)
    normal_nonfinite = (
        normal_components_nonfinite
        | ~np.isfinite(normal_norm)
        | ~np.isfinite(parent_normal_norm)
    )
    zero_normal = (~normal_nonfinite) & (normal_norm <= NORMAL_NORM_EPSILON)

    reason_code = np.full(len(point_array), "", dtype="<U24")
    reason_code[point_nonfinite] = "NONFINITE_POINT"
    unresolved = reason_code == ""
    reason_code[unresolved & normal_nonfinite] = "NONFINITE_NORMAL"
    unresolved = reason_code == ""
    reason_code[unresolved & zero_normal] = "NORMAL_NORM_LE_1E_12"
    valid = reason_code == ""

    retained_points = np.ascontiguousarray(point_array[valid], dtype=np.float64)
    retained_normals = np.ascontiguousarray(normal_array[valid], dtype=np.float64)
    retained_ids = np.ascontiguousarray(identities[valid], dtype=np.uint64)
    rejected_ids = np.ascontiguousarray(identities[~valid], dtype=np.uint64)
    exclusive_counts = {
        reason: int(np.count_nonzero(reason_code == reason))
        for reason in REASON_ORDER
    }
    raw_predicate_counts = {
        "NONFINITE_POINT": int(np.count_nonzero(point_nonfinite)),
        "NONFINITE_NORMAL": int(np.count_nonzero(normal_nonfinite)),
        "NORMAL_NORM_LE_1E_12": int(np.count_nonzero(zero_normal)),
    }
    if sum(exclusive_counts.values()) != len(rejected_ids):
        raise RuntimeError("exclusive source rejection accounting mismatch")

    actionability = evaluate_actionability(
        original_count=len(point_array),
        retained_count=len(retained_points),
    )
    reason_identity_hashes = {
        reason: identity_sha256(identities[reason_code == reason])
        for reason in REASON_ORDER
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "member_token": str(member_token),
        "original_row_identity_encoding": "LITTLE_ENDIAN_UINT64_ACQUISITION_ORDINAL",
        "original_row_count": int(len(point_array)),
        "retained_row_count": int(len(retained_points)),
        "rejected_row_count": int(len(rejected_ids)),
        "raw_predicate_counts": raw_predicate_counts,
        "exclusive_rejection_counts": exclusive_counts,
        "original_identity_sequence_sha256": identity_sha256(
            identities, sort_as_set=False
        ),
        "original_identity_set_sha256": identity_sha256(identities),
        "retained_identity_sequence_sha256": identity_sha256(
            retained_ids, sort_as_set=False
        ),
        "retained_identity_set_sha256": identity_sha256(retained_ids),
        "rejected_identity_set_sha256": identity_sha256(rejected_ids),
        "rejected_identity_sha256_by_exclusive_reason": reason_identity_hashes,
        "rejected_original_row_ids_sorted": [
            int(value) for value in np.sort(rejected_ids, kind="mergesort")
        ],
        "retained_points_sha256": numeric_array_sha256(retained_points),
        "retained_normals_sha256": numeric_array_sha256(retained_normals),
        "normal_norm_epsilon_inclusive": NORMAL_NORM_EPSILON,
        "replacement_interpolation_or_normal_estimation_performed": False,
        **actionability,
    }
    return {
        "points": retained_points,
        "normals": retained_normals,
        "original_row_ids": retained_ids,
        "report": report,
    }


def pair_actionability(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    reports = [first.get("report"), second.get("report")]
    if any(not isinstance(report, Mapping) for report in reports):
        raise ValueError("pair actionability requires two source-interface reports")
    failed_tokens = [
        str(report["member_token"])
        for report in reports
        if report.get("actionable") is not True
    ]
    return {
        "actionable": not failed_tokens,
        "failed_member_tokens": failed_tokens,
        "status": (
            "PASS_SOURCE_INTERFACE_ACTIONABLE"
            if not failed_tokens
            else "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
        ),
    }
