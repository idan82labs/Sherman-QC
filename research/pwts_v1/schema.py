"""Compact, coordinate-free table schema for the PWTS semantic core."""

from __future__ import annotations

import hashlib

from .config import canonical_json_bytes


TABLE_PRIMARY_KEYS = {
    "case_manifest": ("case_token",),
    "field_summary": ("case_token",),
    "seed_primitive": ("case_token", "seed_point_id"),
    "trace_primitive": ("case_token", "trace_id"),
    "trace_node": ("case_token", "trace_id", "ordinal"),
    "strip_membership": ("case_token", "trace_id", "side", "point_id"),
    "strip_summary": ("case_token", "trace_id", "side"),
    "certificate": ("case_token", "certificate_id"),
    "destructive_negative": ("case_token", "certificate_id", "deletion_arm"),
    "rejection_census": ("case_token", "stage", "rejection_code"),
    "equivalence": ("case_token", "comparison"),
    "resource": ("case_token", "process_role"),
    "criteria": ("criterion_id",),
}

SIDE_ORDER = {"left": 0, "right": 1}
ARM_ORDER = {"DELETE_LEFT": 0, "DELETE_RIGHT": 1}

CERTIFICATE_REJECTION_ORDER = (
    "TRACE_INVALID",
    "STRIP_MEMBERSHIP_OVERLAP",
    "LEFT_SUPPORT_INSUFFICIENT",
    "RIGHT_SUPPORT_INSUFFICIENT",
    "LEFT_COVERAGE_INSUFFICIENT",
    "RIGHT_COVERAGE_INSUFFICIENT",
    "BILATERAL_COCOVERAGE_INSUFFICIENT",
    "SIDE_COUNT_BALANCE_INSUFFICIENT",
    "LEFT_NORMAL_COHERENCE_INSUFFICIENT",
    "RIGHT_NORMAL_COHERENCE_INSUFFICIENT",
    "LEFT_SURFACE_VARIATION_EXCESSIVE",
    "RIGHT_SURFACE_VARIATION_EXCESSIVE",
    "SIDE_NORMAL_SEPARATION_INSUFFICIENT",
    "TRACE_RESPONSE_INSUFFICIENT",
    "TRANSITION_CONTRAST_INSUFFICIENT",
    "VALID",
)

TRACE_REJECTION_ORDER = (
    "TANGENT_SUPPORT_INSUFFICIENT",
    "TANGENT_ANISOTROPY_INSUFFICIENT",
    "TRACE_NODE_COUNT_INSUFFICIENT",
    "TRACE_LENGTH_INSUFFICIENT",
    "TRACE_DISCONTINUITY",
    "TRACE_ARM_COLLISION",
    "TRACE_NODE_CAP_EXCEEDED",
    "VALID",
)

PROHIBITED_SERIALIZED_KEYS = frozenset(
    {
        "x", "y", "z", "xyz", "coordinates", "raw_points", "sampled_points",
        "source_points", "neighborhoods", "rendered_geometry", "point_subset",
    }
)

SCHEMA = {
    "schema_version": 1,
    "table_primary_keys": TABLE_PRIMARY_KEYS,
    "certificate_rejection_order": CERTIFICATE_REJECTION_ORDER,
    "trace_rejection_order": TRACE_REJECTION_ORDER,
    "source_points_serialized": False,
    "id_digest_rule": "sha256_raw_32_byte_ids_ascending_no_framing",
    "prohibited_serialized_keys": sorted(PROHIBITED_SERIALIZED_KEYS),
}
SCHEMA_SHA256 = hashlib.sha256(canonical_json_bytes(SCHEMA)).hexdigest()


class StructuralCapError(RuntimeError):
    """Raised before a required authoritative table would be truncated."""


def canonical_row_key(table: str, row: dict) -> tuple:
    values = []
    for key in TABLE_PRIMARY_KEYS[table]:
        value = row[key]
        if key == "side":
            value = SIDE_ORDER[value]
        elif key == "deletion_arm":
            value = ARM_ORDER[value]
        values.append(value)
    return tuple(values)


def assert_coordinate_free(value: object) -> None:
    """Fail closed if serialized evidence contains raw coordinate-bearing keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in PROHIBITED_SERIALIZED_KEYS:
                raise ValueError(f"prohibited serialized coordinate key: {key}")
            assert_coordinate_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_coordinate_free(child)
