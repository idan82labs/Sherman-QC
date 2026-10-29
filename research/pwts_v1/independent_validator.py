"""Independent, fail-closed validator for PWTS_REPRESENTATION_PREFLIGHT_V1.

This module intentionally contains no imports from the PWTS runner, semantic core,
or any other repository module.  It validates the compact authoritative evidence
tables described by the locked professor ruling.  It does *not* construct point
geometry and it never turns a representation row into a bend prediction.
"""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PREFLIGHT_ID = "PWTS_REPRESENTATION_PREFLIGHT_V1"
RULING_CONFIGURATION_SHA256 = (
    "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
)
VALID_STATUS = "PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
INVALID_STATUS = "INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP"

TABLES = (
    "case_manifest",
    "field_summary",
    "seed_primitive",
    "trace_primitive",
    "trace_node",
    "strip_membership",
    "strip_summary",
    "certificate",
    "destructive_negative",
    "rejection_census",
    "equivalence",
    "resource",
    "criteria",
)

PRIMARY_KEYS: Mapping[str, tuple[str, ...]] = {
    "case_manifest": ("case_token",),
    "field_summary": ("case_token",),
    "seed_primitive": ("case_token", "seed_point_id"),
    "trace_primitive": ("case_token", "trace_id"),
    "trace_node": ("case_token", "trace_id", "ordinal"),
    "strip_membership": ("case_token", "trace_id", "side", "point_id"),
    "strip_summary": ("case_token", "trace_id", "side"),
    "certificate": ("case_token", "certificate_id"),
    "destructive_negative": (
        "case_token",
        "certificate_id",
        "deletion_arm",
    ),
    "rejection_census": ("case_token", "stage", "rejection_code"),
    "equivalence": ("case_token", "comparison"),
    "resource": ("case_token", "process_role"),
    "criteria": ("criterion_id",),
}

SIDE_ORDER = {"left": 0, "right": 1, "LEFT": 0, "RIGHT": 1}
ARM_ORDER = {"DELETE_LEFT": 0, "DELETE_RIGHT": 1}

CERTIFICATE_REJECTIONS = (
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

REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "case_manifest": frozenset(
        {
            "case_token",
            "raw_path",
            "raw_byte_count",
            "raw_ply_sha256",
            "observed_raw_ply_sha256",
            "ply_header_sha256",
            "parsed_vertex_count",
            "finite_coordinate_count",
            "exact_duplicate_count",
            "ignored_property_name_digest",
            "canonical_coordinate_multiset_digest",
            "sample_count",
            "sampled_point_id_digest",
            "validator_sampled_point_id_digest",
            "config_sha256",
            "schema_sha256",
            "source_points_serialized",
        }
    ),
    "field_summary": frozenset(
        {
            "case_token",
            "sample_count",
            "normal_valid_count_k16",
            "normal_valid_count_k32",
            "normal_valid_count_k64",
            "seed_eligible_count",
            "trace_eligible_count",
            "deterministic_local_maximum_count",
            "dense_knn_digest",
            "dense_normal_digest",
            "dense_response_digest",
        }
    ),
    "seed_primitive": frozenset(
        {
            "case_token",
            "seed_point_id",
            "d1_rad",
            "d2_rad",
            "response_rad",
            "scale_balance",
            "tangent_support_count",
            "tangent_anisotropy",
            "trace_id",
            "valid",
            "first_rejection_code",
        }
    ),
    "trace_primitive": frozenset(
        {
            "case_token",
            "trace_id",
            "seed_point_id",
            "node_count",
            "h_t",
            "normalized_arc_length",
            "maximum_normalized_step",
            "median_response_rad",
            "maximum_turn_rad",
            "node_table_digest",
            "valid",
            "first_rejection_code",
        }
    ),
    "trace_node": frozenset(
        {
            "case_token",
            "trace_id",
            "ordinal",
            "point_id",
            "response_rad",
            "tangent_x",
            "tangent_y",
            "tangent_z",
        }
    ),
    "strip_membership": frozenset(
        {
            "case_token",
            "trace_id",
            "side",
            "point_id",
            "nearest_segment_ordinal",
            "arc_bin",
            "rho_over_h_t",
            "signed_side_cosine",
            "response_rad",
            "surface_variation",
        }
    ),
    "strip_summary": frozenset(
        {
            "case_token",
            "trace_id",
            "side",
            "count",
            "covered_bins",
            "normal_coherence",
            "dominant_normal_x",
            "dominant_normal_y",
            "dominant_normal_z",
            "median_surface_variation",
            "p90_surface_variation",
            "median_response_rad",
            "sorted_support_digest",
        }
    ),
    "certificate": frozenset(
        {
            "case_token",
            "certificate_id",
            "trace_id",
            "trace_valid",
            "membership_intersection_count",
            "left_support_count",
            "right_support_count",
            "left_covered_bins",
            "right_covered_bins",
            "bilaterally_covered_bins",
            "side_count_ratio",
            "left_normal_coherence",
            "right_normal_coherence",
            "left_median_surface_variation",
            "left_p90_surface_variation",
            "right_median_surface_variation",
            "right_p90_surface_variation",
            "side_normal_separation_rad",
            "trace_median_response_rad",
            "left_median_response_rad",
            "right_median_response_rad",
            "gate_trace_valid",
            "gate_membership_disjoint",
            "gate_left_support",
            "gate_right_support",
            "gate_left_coverage",
            "gate_right_coverage",
            "gate_bilateral_cocoverage",
            "gate_side_count_balance",
            "gate_left_normal_coherence",
            "gate_right_normal_coherence",
            "gate_left_surface_variation",
            "gate_right_surface_variation",
            "gate_side_normal_separation",
            "gate_trace_response",
            "gate_transition_contrast",
            "valid",
            "first_rejection_code",
        }
    ),
    "destructive_negative": frozenset(
        {
            "case_token",
            "certificate_id",
            "deletion_arm",
            "trace_id",
            "deleted_id_digest",
            "deleted_id_count",
            "rebuilt_left_support_count",
            "rebuilt_right_support_count",
            "rebuilt_valid",
            "first_rejection_code",
            "raw_id_deletion_applied",
        }
    ),
    "rejection_census": frozenset(
        {"case_token", "stage", "rejection_code", "numerator", "denominator"}
    ),
    "equivalence": frozenset(
        {
            "case_token",
            "comparison",
            "per_table_row_counts",
            "per_table_digests",
            "semantic_equivalence",
            "first_mismatch",
        }
    ),
    "resource": frozenset(
        {
            "case_token",
            "process_role",
            "cpu_seconds",
            "wall_seconds",
            "process_tree_peak_rss_bytes",
            "scratch_bytes",
            "serialized_bytes",
            "monitor_available",
            "full_lifetime_accounting",
            "reported_passing",
        }
    ),
    "criteria": frozenset(
        {
            "criterion_id",
            "numerator",
            "denominator",
            "source_table_digest",
            "result",
        }
    ),
}

FORBIDDEN_KEY_PARTS = (
    "source_points",
    "sampled_points",
    "raw_points",
    "point_coordinates",
    "xyz_array",
    "neighborhood_array",
    "neighborhoods",
    "rendered_geometry",
    "point_subset",
    "coordinates",
)

FORBIDDEN_IMPORT_PREFIXES = (
    "backend",
    "scripts",
    "research.pwts_v1.semantic_core",
    "research.pwts_v1.runner",
    "research.pwts_v1.sampler",
    "research.pwts_v1.normals",
    "research.pwts_v1.tracing",
    "research.pwts_v1.certificate",
)


@dataclass(frozen=True)
class ValidationPolicy:
    """External locks that cannot be trusted from the evidence being checked."""

    expected_configuration_sha256: str = RULING_CONFIGURATION_SHA256
    expected_schema_sha256: str | None = None
    expected_source_manifest_sha256: str | None = None
    expected_case_tokens: tuple[str, ...] = tuple(
        f"C{number:03d}" for number in range(1, 38)
    )
    required_process_roles: tuple[str, ...] = (
        "BASE_A",
        "BASE_B",
        "ORDER_PERMUTED",
        "INDEPENDENT_VALIDATOR",
    )
    required_comparisons: tuple[str, ...] = (
        "BASE_A_VS_BASE_B",
        "BASE_A_VS_ORDER_PERMUTED",
        "BASE_A_VS_INDEPENDENT_VALIDATOR",
    )
    per_process_cpu_seconds: float = 900.0
    per_process_wall_seconds: float = 1200.0
    per_process_peak_rss_bytes: int = 12_884_901_888
    per_process_scratch_bytes: int = 4_294_967_296
    per_case_serialized_bytes: int = 100_663_296
    aggregate_cpu_seconds: float = 133_200.0
    aggregate_wall_seconds: float = 177_600.0
    aggregate_serialized_bytes: int = 2_147_483_648
    maximum_sample_points: int = 200_000
    maximum_rows_per_case: Mapping[str, int] = field(
        default_factory=lambda: {
            "seed_primitive": 1024,
            "trace_primitive": 1024,
            "trace_node": 131_072,
            "strip_membership": 1_000_000,
            "certificate": 1024,
        }
    )


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    terminal_status: str
    errors: tuple[str, ...]
    derived: Mapping[str, Any]

    def require_valid(self) -> "ValidationResult":
        if not self.valid:
            raise EvidenceValidationError("; ".join(self.errors))
        return self


class EvidenceValidationError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sorted_id_digest(point_ids: Iterable[str]) -> str:
    raw_ids = sorted(bytes.fromhex(point_id) for point_id in point_ids)
    return hashlib.sha256(b"".join(raw_ids)).hexdigest()


def ordered_id_digest(point_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        b"".join(bytes.fromhex(point_id) for point_id in point_ids)
    ).hexdigest()


def table_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(list(rows))


def _special_order(field_name: str, value: Any) -> Any:
    if field_name == "side":
        return SIDE_ORDER.get(str(value), 99)
    if field_name == "deletion_arm":
        return ARM_ORDER.get(str(value), 99)
    return value


def row_key(table_name: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _special_order(name, row.get(name)) for name in PRIMARY_KEYS[table_name]
    )


def imported_modules(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            elif node.level:
                modules.add("." * node.level)
    return tuple(sorted(modules))


def forbidden_imports(source: str) -> tuple[str, ...]:
    found = imported_modules(source)
    return tuple(
        module
        for module in found
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    )


def inspect_import_boundary(source_or_path: str | Path) -> tuple[str, ...]:
    path = Path(source_or_path)
    source = path.read_text(encoding="utf-8") if path.is_file() else str(source_or_path)
    return forbidden_imports(source)


def _walk_forbidden_payload(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in FORBIDDEN_KEY_PARTS):
                if normalized == "source_points_serialized" and child is False:
                    pass
                else:
                    errors.append(f"forbidden serialized payload at {path}.{key}")
            if normalized in {"x", "y", "z", "xyz"} and isinstance(child, (list, tuple)):
                errors.append(f"serialized coordinate array at {path}.{key}")
            _walk_forbidden_payload(child, f"{path}.{key}", errors)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_forbidden_payload(child, f"{path}[{index}]", errors)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _certificate_truth(row: Mapping[str, Any]) -> tuple[dict[str, bool], str, bool]:
    maximum_side_response = max(
        float(row["left_median_response_rad"]),
        float(row["right_median_response_rad"]),
    )
    gates = {
        "gate_trace_valid": bool(row["trace_valid"]),
        "gate_membership_disjoint": int(row["membership_intersection_count"]) == 0,
        "gate_left_support": int(row["left_support_count"]) >= 32,
        "gate_right_support": int(row["right_support_count"]) >= 32,
        "gate_left_coverage": int(row["left_covered_bins"]) >= 6,
        "gate_right_coverage": int(row["right_covered_bins"]) >= 6,
        "gate_bilateral_cocoverage": int(row["bilaterally_covered_bins"]) >= 6,
        "gate_side_count_balance": float(row["side_count_ratio"]) >= 0.25,
        "gate_left_normal_coherence": float(row["left_normal_coherence"]) >= 0.80,
        "gate_right_normal_coherence": float(row["right_normal_coherence"]) >= 0.80,
        "gate_left_surface_variation": (
            float(row["left_median_surface_variation"]) <= 0.12
            and float(row["left_p90_surface_variation"]) <= 0.25
        ),
        "gate_right_surface_variation": (
            float(row["right_median_surface_variation"]) <= 0.12
            and float(row["right_p90_surface_variation"]) <= 0.25
        ),
        "gate_side_normal_separation": float(row["side_normal_separation_rad"])
        >= 0.13962634015954636,
        "gate_trace_response": float(row["trace_median_response_rad"])
        >= 0.10471975511965978,
        "gate_transition_contrast": float(row["trace_median_response_rad"])
        - maximum_side_response
        >= 0.03490658503988659,
    }
    ordered_gate_names = (
        "gate_trace_valid",
        "gate_membership_disjoint",
        "gate_left_support",
        "gate_right_support",
        "gate_left_coverage",
        "gate_right_coverage",
        "gate_bilateral_cocoverage",
        "gate_side_count_balance",
        "gate_left_normal_coherence",
        "gate_right_normal_coherence",
        "gate_left_surface_variation",
        "gate_right_surface_variation",
        "gate_side_normal_separation",
        "gate_trace_response",
        "gate_transition_contrast",
    )
    for index, name in enumerate(ordered_gate_names):
        if not gates[name]:
            return gates, CERTIFICATE_REJECTIONS[index], False
    return gates, "VALID", True


def _validate_schema(
    evidence: Mapping[str, Any], policy: ValidationPolicy, errors: list[str]
) -> None:
    _require(evidence.get("preflight_id") == PREFLIGHT_ID, "wrong preflight_id", errors)
    _require(evidence.get("schema_version") == 1, "wrong schema_version", errors)
    _require(
        evidence.get("config_sha256") == policy.expected_configuration_sha256,
        "unauthorized configuration hash",
        errors,
    )
    if policy.expected_schema_sha256 is not None:
        _require(
            evidence.get("schema_sha256") == policy.expected_schema_sha256,
            "schema hash mismatch",
            errors,
        )
    if policy.expected_source_manifest_sha256 is not None:
        _require(
            evidence.get("source_manifest_sha256")
            == policy.expected_source_manifest_sha256,
            "source manifest hash mismatch",
            errors,
        )

    for table in TABLES:
        rows = evidence.get(table)
        if not isinstance(rows, list):
            errors.append(f"missing or non-array table: {table}")
            continue
        seen: set[tuple[Any, ...]] = set()
        keys: list[tuple[Any, ...]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"{table}[{index}] is not an object")
                continue
            missing = REQUIRED_FIELDS[table] - set(row)
            if missing:
                errors.append(f"{table}[{index}] missing fields {sorted(missing)}")
                continue
            key = row_key(table, row)
            if key in seen:
                errors.append(f"duplicate primary key in {table}: {key}")
            seen.add(key)
            keys.append(key)
        if keys != sorted(keys):
            errors.append(f"noncanonical row order in {table}")


def _validate_manifests(
    evidence: Mapping[str, Any], policy: ValidationPolicy, errors: list[str]
) -> None:
    rows = evidence.get("case_manifest", [])
    tokens = tuple(row.get("case_token") for row in rows if isinstance(row, Mapping))
    _require(tokens == policy.expected_case_tokens, "case envelope is not exact", errors)
    for row in rows:
        token = row["case_token"]
        _require(
            row["raw_ply_sha256"] == row["observed_raw_ply_sha256"],
            f"{token}: raw source hash mismatch",
            errors,
        )
        _require(
            row["sampled_point_id_digest"]
            == row["validator_sampled_point_id_digest"],
            f"{token}: sampled point-ID digest mismatch",
            errors,
        )
        _require(row["config_sha256"] == evidence.get("config_sha256"), f"{token}: config binding mismatch", errors)
        _require(row["schema_sha256"] == evidence.get("schema_sha256"), f"{token}: schema binding mismatch", errors)
        _require(row["source_points_serialized"] is False, f"{token}: source points serialized", errors)
        sample_count = int(row["sample_count"])
        finite = int(row["finite_coordinate_count"])
        duplicates = int(row["exact_duplicate_count"])
        _require(128 <= sample_count <= policy.maximum_sample_points, f"{token}: sample count outside bounds", errors)
        _require(sample_count <= finite - duplicates, f"{token}: impossible sample count", errors)


def _validate_table_relations(evidence: Mapping[str, Any], errors: list[str]) -> None:
    manifests = {row["case_token"]: row for row in evidence.get("case_manifest", [])}
    fields = {row["case_token"]: row for row in evidence.get("field_summary", [])}
    seeds = evidence.get("seed_primitive", [])
    traces = evidence.get("trace_primitive", [])
    nodes = evidence.get("trace_node", [])
    memberships = evidence.get("strip_membership", [])
    summaries = evidence.get("strip_summary", [])
    certificates = evidence.get("certificate", [])

    _require(set(fields) == set(manifests), "field_summary case completeness mismatch", errors)
    for token, field_row in fields.items():
        _require(int(field_row["sample_count"]) == int(manifests[token]["sample_count"]), f"{token}: sampled denominator mismatch", errors)
        for scale in (16, 32, 64):
            count = int(field_row[f"normal_valid_count_k{scale}"])
            _require(0 <= count <= int(field_row["sample_count"]), f"{token}: invalid k{scale} normal denominator", errors)
        _require(sum(1 for row in seeds if row["case_token"] == token) == int(field_row["deterministic_local_maximum_count"]), f"{token}: seed completeness mismatch", errors)

    nodes_by_trace: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in nodes:
        nodes_by_trace.setdefault((row["case_token"], row["trace_id"]), []).append(row)
    for row in traces:
        key = (row["case_token"], row["trace_id"])
        trace_nodes = nodes_by_trace.get(key, [])
        _require(len(trace_nodes) == int(row["node_count"]), f"{key}: trace-node count mismatch", errors)
        _require(sorted_id_digest(node["point_id"] for node in trace_nodes) == row["node_table_digest"], f"{key}: trace-node digest mismatch", errors)

    members_by_side: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in memberships:
        members_by_side.setdefault((row["case_token"], row["trace_id"], str(row["side"]).lower()), []).append(row)
    summary_index = {
        (row["case_token"], row["trace_id"], str(row["side"]).lower()): row
        for row in summaries
    }
    for key, member_rows in members_by_side.items():
        summary = summary_index.get(key)
        _require(summary is not None, f"{key}: missing strip summary", errors)
        if summary is None:
            continue
        point_ids = [row["point_id"] for row in member_rows]
        _require(len(point_ids) == len(set(point_ids)), f"{key}: duplicate strip point", errors)
        _require(int(summary["count"]) == len(member_rows), f"{key}: strip count mismatch", errors)
        _require(summary["sorted_support_digest"] == sorted_id_digest(point_ids), f"{key}: strip digest mismatch", errors)
        covered = len({int(row["arc_bin"]) for row in member_rows if sum(1 for other in member_rows if int(other["arc_bin"]) == int(row["arc_bin"])) >= 3})
        _require(int(summary["covered_bins"]) == covered, f"{key}: covered-bin mismatch", errors)

    cert_by_key = {(row["case_token"], row["certificate_id"]): row for row in certificates}
    for cert in certificates:
        key = (cert["case_token"], cert["trace_id"])
        left = members_by_side.get((key[0], key[1], "left"), [])
        right = members_by_side.get((key[0], key[1], "right"), [])
        left_ids = {row["point_id"] for row in left}
        right_ids = {row["point_id"] for row in right}
        _require(int(cert["left_support_count"]) == len(left), f"{key}: certificate left count mismatch", errors)
        _require(int(cert["right_support_count"]) == len(right), f"{key}: certificate right count mismatch", errors)
        _require(int(cert["membership_intersection_count"]) == len(left_ids & right_ids), f"{key}: membership intersection mismatch", errors)
        left_bins = {int(row["arc_bin"]) for row in left if sum(1 for other in left if int(other["arc_bin"]) == int(row["arc_bin"])) >= 3}
        right_bins = {int(row["arc_bin"]) for row in right if sum(1 for other in right if int(other["arc_bin"]) == int(row["arc_bin"])) >= 3}
        _require(int(cert["left_covered_bins"]) == len(left_bins), f"{key}: left coverage mismatch", errors)
        _require(int(cert["right_covered_bins"]) == len(right_bins), f"{key}: right coverage mismatch", errors)
        _require(int(cert["bilaterally_covered_bins"]) == len(left_bins & right_bins), f"{key}: bilateral coverage mismatch", errors)
        ratio = min(len(left), len(right)) / max(len(left), len(right)) if left and right else 0.0
        _require(math.isclose(float(cert["side_count_ratio"]), ratio, rel_tol=1e-8, abs_tol=1e-8), f"{key}: side-count ratio mismatch", errors)
        gates, rejection, valid = _certificate_truth(cert)
        for gate, value in gates.items():
            _require(cert[gate] is value, f"{key}: forged gate {gate}", errors)
        _require(cert["first_rejection_code"] == rejection, f"{key}: first rejection mismatch", errors)
        _require(cert["valid"] is valid, f"{key}: forged certificate validity", errors)

    for row in evidence.get("destructive_negative", []):
        cert_key = (row["case_token"], row["certificate_id"])
        cert = cert_by_key.get(cert_key)
        _require(cert is not None and cert.get("valid") is True, f"{cert_key}: deletion references nonvalid certificate", errors)
        if cert is None:
            continue
        side = "left" if row["deletion_arm"] == "DELETE_LEFT" else "right"
        member_rows = members_by_side.get((row["case_token"], cert["trace_id"], side), [])
        _require(row["trace_id"] == cert["trace_id"], f"{cert_key}: deletion changed trace identity", errors)
        _require(row["raw_id_deletion_applied"] is True, f"{cert_key}: raw-ID deletion not evidenced", errors)
        _require(int(row["deleted_id_count"]) == len(member_rows), f"{cert_key}: deleted count mismatch", errors)
        _require(row["deleted_id_digest"] == sorted_id_digest(r["point_id"] for r in member_rows), f"{cert_key}: deleted-ID digest mismatch", errors)
        expected_code = "LEFT_SUPPORT_INSUFFICIENT" if side == "left" else "RIGHT_SUPPORT_INSUFFICIENT"
        rebuilt_deleted = row[f"rebuilt_{side}_support_count"]
        _require(int(rebuilt_deleted) == 0, f"{cert_key}: deleted side survived rebuild", errors)
        _require(row["rebuilt_valid"] is False, f"{cert_key}: destructive negative survived", errors)
        _require(row["first_rejection_code"] == expected_code, f"{cert_key}: wrong destructive rejection", errors)

    valid_certs = [row for row in certificates if row.get("valid") is True]
    _require(bool(valid_certs), "zero valid baseline certificates", errors)
    deletion_index = {
        (row["case_token"], row["certificate_id"], row["deletion_arm"])
        for row in evidence.get("destructive_negative", [])
    }
    for cert in valid_certs:
        for arm in ARM_ORDER:
            _require((cert["case_token"], cert["certificate_id"], arm) in deletion_index, f"{cert['certificate_id']}: missing {arm}", errors)


def _validate_rejection_census(evidence: Mapping[str, Any], errors: list[str]) -> None:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in evidence.get("rejection_census", []):
        groups.setdefault((row["case_token"], row["stage"]), []).append(row)
    for key, rows in groups.items():
        denominators = {int(row["denominator"]) for row in rows}
        _require(len(denominators) == 1, f"{key}: inconsistent rejection denominator", errors)
        if denominators:
            denominator = next(iter(denominators))
            _require(sum(int(row["numerator"]) for row in rows) == denominator, f"{key}: rejection counts do not sum", errors)
            _require(all(int(row["numerator"]) >= 0 for row in rows), f"{key}: negative rejection count", errors)
            if denominator == 0:
                _require(
                    len(rows) == 1
                    and rows[0]["rejection_code"] == "EMPTY_POPULATION"
                    and int(rows[0]["numerator"]) == 0,
                    f"{key}: zero denominator lacks explicit EMPTY_POPULATION row",
                    errors,
                )

    required_denominators: dict[tuple[str, str], int] = {}
    for manifest in evidence.get("case_manifest", []):
        token = manifest["case_token"]
        field_row = next((row for row in evidence.get("field_summary", []) if row["case_token"] == token), None)
        if field_row:
            required_denominators.update(
                {
                    (token, "all_sampled_points"): int(field_row["sample_count"]),
                    (token, "normal_valid_k16"): int(field_row["normal_valid_count_k16"]),
                    (token, "normal_valid_k32"): int(field_row["normal_valid_count_k32"]),
                    (token, "normal_valid_k64"): int(field_row["normal_valid_count_k64"]),
                    (token, "seed_eligible_points"): int(field_row["seed_eligible_count"]),
                    (token, "deterministic_local_maxima"): int(field_row["deterministic_local_maximum_count"]),
                }
            )
        required_denominators[(token, "seed_primitives")] = sum(1 for row in evidence.get("seed_primitive", []) if row["case_token"] == token)
        required_denominators[(token, "valid_traces")] = sum(1 for row in evidence.get("trace_primitive", []) if row["case_token"] == token and row["valid"])
        required_denominators[(token, "certificate_candidates")] = sum(1 for row in evidence.get("certificate", []) if row["case_token"] == token)
        required_denominators[(token, "valid_certificates")] = sum(1 for row in evidence.get("certificate", []) if row["case_token"] == token and row["valid"])
        required_denominators[(token, "destructive_negative_arms")] = sum(1 for row in evidence.get("destructive_negative", []) if row["case_token"] == token)
    for key, expected in required_denominators.items():
        rows = groups.get(key)
        _require(rows is not None, f"{key}: missing rejection denominator stage", errors)
        if rows:
            _require(all(int(row["denominator"]) == expected for row in rows), f"{key}: wrong rejection denominator", errors)


def _validate_equivalence(evidence: Mapping[str, Any], policy: ValidationPolicy, errors: list[str]) -> None:
    semantic_tables = tuple(table for table in TABLES if table not in {"resource", "equivalence", "criteria"})
    rows = evidence.get("equivalence", [])
    by_case: dict[str, set[str]] = {token: set() for token in policy.expected_case_tokens}
    for row in rows:
        token = row["case_token"]
        comparison = row["comparison"]
        by_case.setdefault(token, set()).add(comparison)
        counts = row["per_table_row_counts"]
        digests = row["per_table_digests"]
        for table in semantic_tables:
            variants = digests.get(table)
            variant_counts = counts.get(table)
            _require(isinstance(variants, Mapping) and len(variants) >= 2, f"{token}/{comparison}/{table}: missing equivalence digests", errors)
            _require(isinstance(variant_counts, Mapping) and len(variant_counts) >= 2, f"{token}/{comparison}/{table}: missing equivalence counts", errors)
            if isinstance(variants, Mapping):
                _require(len(set(variants.values())) == 1, f"{token}/{comparison}/{table}: semantic digest mismatch", errors)
                actual_rows = [
                    item for item in evidence.get(table, [])
                    if item.get("case_token") == token
                ]
                _require(
                    variants.get("BASE_A") == table_digest(actual_rows),
                    f"{token}/{comparison}/{table}: BASE_A digest not bound to authoritative table",
                    errors,
                )
            if isinstance(variant_counts, Mapping):
                _require(len(set(variant_counts.values())) == 1, f"{token}/{comparison}/{table}: row-count mismatch", errors)
                _require(
                    int(variant_counts.get("BASE_A", -1))
                    == sum(
                        1 for item in evidence.get(table, [])
                        if item.get("case_token") == token
                    ),
                    f"{token}/{comparison}/{table}: BASE_A count not bound to authoritative table",
                    errors,
                )
        _require(row["semantic_equivalence"] is True, f"{token}/{comparison}: equivalence is false", errors)
        _require(row["first_mismatch"] is None, f"{token}/{comparison}: mismatch location asserted", errors)
    for token, observed in by_case.items():
        _require(observed == set(policy.required_comparisons), f"{token}: incomplete equivalence comparisons", errors)


def _validate_resources(evidence: Mapping[str, Any], policy: ValidationPolicy, errors: list[str]) -> None:
    rows = evidence.get("resource", [])
    by_case: dict[str, set[str]] = {token: set() for token in policy.expected_case_tokens}
    total_cpu = total_wall = 0.0
    total_serialized = 0
    for row in rows:
        token, role = row["case_token"], row["process_role"]
        by_case.setdefault(token, set()).add(role)
        cpu = float(row["cpu_seconds"])
        wall = float(row["wall_seconds"])
        rss = int(row["process_tree_peak_rss_bytes"])
        scratch = int(row["scratch_bytes"])
        serialized = int(row["serialized_bytes"])
        total_cpu += cpu
        total_wall += wall
        total_serialized += serialized
        _require(row["monitor_available"] is True, f"{token}/{role}: monitor unavailable", errors)
        _require(row["full_lifetime_accounting"] is True, f"{token}/{role}: shortened accounting", errors)
        _require(0 <= cpu <= policy.per_process_cpu_seconds, f"{token}/{role}: CPU cap exceeded", errors)
        _require(0 <= wall <= policy.per_process_wall_seconds, f"{token}/{role}: wall cap exceeded", errors)
        _require(0 <= rss <= policy.per_process_peak_rss_bytes, f"{token}/{role}: RSS cap exceeded", errors)
        _require(0 <= scratch <= policy.per_process_scratch_bytes, f"{token}/{role}: scratch cap exceeded", errors)
        _require(0 <= serialized <= policy.per_case_serialized_bytes, f"{token}/{role}: serialized cap exceeded", errors)
        recomputed_pass = (
            row["monitor_available"] is True
            and row["full_lifetime_accounting"] is True
            and 0 <= cpu <= policy.per_process_cpu_seconds
            and 0 <= wall <= policy.per_process_wall_seconds
            and 0 <= rss <= policy.per_process_peak_rss_bytes
            and 0 <= scratch <= policy.per_process_scratch_bytes
            and 0 <= serialized <= policy.per_case_serialized_bytes
        )
        _require(row["reported_passing"] is recomputed_pass, f"{token}/{role}: forged resource pass", errors)
    for token, roles in by_case.items():
        _require(roles == set(policy.required_process_roles), f"{token}: missing or extra resource roles", errors)
    _require(total_cpu <= policy.aggregate_cpu_seconds, "aggregate CPU cap exceeded", errors)
    _require(total_wall <= policy.aggregate_wall_seconds, "aggregate wall cap exceeded", errors)
    _require(total_serialized <= policy.aggregate_serialized_bytes, "aggregate serialization cap exceeded", errors)
    for table, cap in policy.maximum_rows_per_case.items():
        for token in policy.expected_case_tokens:
            count = sum(1 for row in evidence.get(table, []) if row["case_token"] == token)
            _require(count <= cap, f"{token}/{table}: structural cap exceeded", errors)


def _validate_safety(evidence: Mapping[str, Any], errors: list[str]) -> None:
    safety = evidence.get("safety")
    _require(isinstance(safety, Mapping), "missing safety attestation", errors)
    if not isinstance(safety, Mapping):
        return
    required_closed = (
        "gold_truth",
        "expected_counts",
        "manual_labels",
        "cad_reference",
        "prior_f1_candidates",
        "lpf_semantics",
        "mesh_topology",
        "solver",
        "tuned_tracers",
        "heatmap_acceptance",
        "learning",
        "ownership_births",
        "replays",
        "synthetic_rescue",
        "runtime_main",
        "production",
        "benchmark_reporting",
    )
    for lane in required_closed:
        _require(safety.get(lane) is False, f"prohibited lane not closed: {lane}", errors)
    reports = safety.get("import_reports")
    _require(isinstance(reports, Mapping), "missing import reports", errors)
    if isinstance(reports, Mapping):
        for role in ("runner", "semantic_core", "independent_validator"):
            modules = reports.get(role)
            _require(isinstance(modules, list), f"missing {role} import report", errors)
            if isinstance(modules, list):
                bad = [module for module in modules if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)]
                _require(not bad, f"{role} has forbidden imports: {bad}", errors)
    # Statically check this validator itself, rather than trust an asserted report.
    _require(not forbidden_imports(Path(__file__).read_text(encoding="utf-8")), "independent validator imports semantic implementation", errors)


def derive_expected_criteria(
    evidence: Mapping[str, Any], policy: ValidationPolicy
) -> list[dict[str, Any]]:
    """Derive the seven locked criteria without trusting runner assertions.

    The two all-table criteria use the ruling's nonrecursive projection: twelve
    non-criteria table digests plus a digest of all criterion rows with their
    ``source_table_digest`` fields removed.
    """

    noncriteria = tuple(table for table in TABLES if table != "criteria")
    digests = {table: table_digest(evidence.get(table, [])) for table in noncriteria}
    case_count = len(policy.expected_case_tokens)
    census_group_count = len(
        {
            (row.get("case_token"), row.get("stage"))
            for row in evidence.get("rejection_census", [])
            if isinstance(row, Mapping)
        }
    )
    capped = {
        table: digests[table]
        for table in (
            "seed_primitive", "trace_primitive", "trace_node",
            "strip_membership", "certificate",
        )
    }
    identifier_tables = {
        table: digests[table]
        for table in (
            "case_manifest", "seed_primitive", "trace_primitive",
            "certificate", "destructive_negative",
        )
    }
    import_reports = evidence.get("safety", {}).get("import_reports", {})
    rows = [
        {
            "criterion_id": "ALL_SOURCE_BINDINGS_VALID",
            "numerator": case_count, "denominator": case_count,
            "source_table_digest": digests["case_manifest"], "result": True,
        },
        {
            "criterion_id": "COMPLETE_TABLES_PRESENT_AND_CANONICAL",
            "numerator": 13, "denominator": 13,
            "source_table_digest": "", "result": True,
        },
        {
            "criterion_id": "TABLE_DENOMINATORS_EXACT",
            "numerator": census_group_count, "denominator": census_group_count,
            "source_table_digest": digests["rejection_census"], "result": True,
        },
        {
            "criterion_id": "STRUCTURAL_CAPS_RESPECTED",
            "numerator": case_count, "denominator": case_count,
            "source_table_digest": sha256_json(capped), "result": True,
        },
        {
            "criterion_id": "IDENTIFIERS_AND_SOURCE_BINDINGS_RECOMPUTED",
            "numerator": case_count, "denominator": case_count,
            "source_table_digest": sha256_json(identifier_tables), "result": True,
        },
        {
            "criterion_id": "STATIC_IMPORT_ALLOWLIST_VALID",
            "numerator": 3, "denominator": 3,
            "source_table_digest": sha256_json(import_reports), "result": True,
        },
        {
            "criterion_id": "NO_SOURCE_POINT_SERIALIZATION",
            "numerator": 13, "denominator": 13,
            "source_table_digest": "", "result": True,
        },
    ]
    rows.sort(key=lambda row: row["criterion_id"])
    projection = [
        {key: value for key, value in row.items() if key != "source_table_digest"}
        for row in rows
    ]
    all_table_binding = sha256_json(
        {
            "noncriteria_table_digests": digests,
            "criteria_projection_digest": sha256_json(projection),
        }
    )
    for row in rows:
        if row["criterion_id"] in {
            "COMPLETE_TABLES_PRESENT_AND_CANONICAL",
            "NO_SOURCE_POINT_SERIALIZATION",
        }:
            row["source_table_digest"] = all_table_binding
    return rows


def _validate_criteria(
    evidence: Mapping[str, Any], policy: ValidationPolicy, errors: list[str]
) -> None:
    observed = evidence.get("criteria", [])
    expected = derive_expected_criteria(evidence, policy)
    _require(observed == expected, "criteria rows are not independently derived canonical seven", errors)
    for row in observed:
        _require(int(row["denominator"]) > 0, f"{row['criterion_id']}: zero criterion denominator", errors)
        _require(int(row["numerator"]) == int(row["denominator"]), f"{row['criterion_id']}: criterion incomplete", errors)
        _require(row["result"] is True, f"{row['criterion_id']}: criterion false", errors)


def validate_evidence(
    evidence_or_path: Mapping[str, Any] | str | Path,
    *,
    policy: ValidationPolicy | None = None,
) -> ValidationResult:
    """Validate a compact PWTS artifact and derive its only permitted status.

    ``expected_configuration_sha256`` remains external so evidence cannot
    self-authorize.  Production callers use the ruling's verified canonical hash.
    """

    effective_policy = policy or ValidationPolicy()
    if isinstance(evidence_or_path, Mapping):
        evidence = evidence_or_path
    else:
        try:
            evidence = json.loads(Path(evidence_or_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return ValidationResult(False, INVALID_STATUS, (f"unreadable evidence: {exc}",), {})
    errors: list[str] = []
    if not isinstance(evidence, Mapping):
        return ValidationResult(False, INVALID_STATUS, ("evidence root is not an object",), {})

    _walk_forbidden_payload(evidence, "$", errors)
    try:
        _validate_schema(evidence, effective_policy, errors)
        _validate_manifests(evidence, effective_policy, errors)
        _validate_table_relations(evidence, errors)
        _validate_rejection_census(evidence, errors)
        _validate_equivalence(evidence, effective_policy, errors)
        _validate_resources(evidence, effective_policy, errors)
        _validate_safety(evidence, errors)
        _validate_criteria(evidence, effective_policy, errors)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"fail-closed validation exception: {type(exc).__name__}: {exc}")

    derived = {
        "case_count": len(evidence.get("case_manifest", [])),
        "valid_certificate_count": sum(
            1 for row in evidence.get("certificate", []) if row.get("valid") is True
        ),
        "destructive_negative_arm_count": len(evidence.get("destructive_negative", [])),
        "table_row_counts": {
            table: len(evidence.get(table, [])) for table in TABLES
        },
        "table_digests": {
            table: table_digest(evidence.get(table, [])) for table in TABLES
        },
    }
    valid = not errors
    expected_status = VALID_STATUS if valid else INVALID_STATUS
    if evidence.get("terminal_status") != expected_status:
        if valid:
            errors.append("terminal status is not independently derived valid status")
        else:
            # An invalid artifact remains invalid regardless of its asserted status.
            pass
        valid = False
    return ValidationResult(
        valid=valid,
        terminal_status=VALID_STATUS if valid else INVALID_STATUS,
        errors=tuple(dict.fromkeys(errors)),
        derived=derived,
    )


# ---------------------------------------------------------------------------
# Independent raw reconstruction CLI
# ---------------------------------------------------------------------------

_PLY_SCALARS = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _q(value: float) -> float:
    import numpy as np
    quantized = float(np.rint(float(value) / 1e-10) * 1e-10)
    return 0.0 if quantized == 0.0 else quantized


def _lower_quantile(values: Any, fraction: float) -> float:
    import numpy as np
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        return 0.0
    index = max(0, min(len(ordered) - 1, math.floor((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _parse_ply_xyz(raw: bytes) -> tuple[Any, bytes, int, tuple[str, ...]]:
    """Independently parse scalar vertex x/y/z from ASCII or binary PLY."""
    import numpy as np

    marker = b"end_header"
    marker_at = raw.find(marker)
    if marker_at < 0:
        raise EvidenceValidationError("PLY_HEADER_MISSING")
    line_end = raw.find(b"\n", marker_at)
    header_end = len(raw) if line_end < 0 else line_end + 1
    header_bytes = raw[:header_end]
    lines = header_bytes.decode("ascii", errors="strict").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise EvidenceValidationError("PLY_MAGIC_INVALID")
    fmt = None
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    ignored: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                prop = ("list", parts[2], parts[3], parts[4])
                ignored.append(f"{current['name']}:{parts[4]}")
            else:
                prop = ("scalar", parts[1], parts[2])
                if current["name"] != "vertex" or parts[2] not in {"x", "y", "z"}:
                    ignored.append(f"{current['name']}:{parts[2]}")
            current["properties"].append(prop)
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise EvidenceValidationError("PLY_FORMAT_UNSUPPORTED")
    vertex_elements = [element for element in elements if element["name"] == "vertex"]
    if len(vertex_elements) != 1:
        raise EvidenceValidationError("PLY_VERTEX_SCHEMA_INVALID")
    vertex = vertex_elements[0]
    scalar_names = {prop[2] for prop in vertex["properties"] if prop[0] == "scalar"}
    if not {"x", "y", "z"}.issubset(scalar_names):
        raise EvidenceValidationError("PLY_VERTEX_SCHEMA_INVALID")
    rows: list[tuple[float, float, float]] = []
    if fmt == "ascii":
        body_lines = raw[header_end:].decode("ascii", errors="strict").splitlines()
        cursor = 0
        for element in elements:
            for _ in range(element["count"]):
                if cursor >= len(body_lines):
                    raise EvidenceValidationError("PLY_VERTEX_COUNT_SHORT")
                tokens = body_lines[cursor].split(); cursor += 1
                token_index = 0; values: dict[str, float] = {}
                for prop in element["properties"]:
                    if prop[0] == "scalar":
                        if token_index >= len(tokens):
                            raise EvidenceValidationError("PLY_VERTEX_ROW_SHORT")
                        if element["name"] == "vertex" and prop[2] in {"x", "y", "z"}:
                            values[prop[2]] = float(tokens[token_index])
                        token_index += 1
                    else:
                        count = int(tokens[token_index]); token_index += 1 + count
                if token_index != len(tokens):
                    raise EvidenceValidationError("PLY_VERTEX_ROW_EXTRA")
                if element["name"] == "vertex":
                    rows.append((values["x"], values["y"], values["z"]))
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        cursor = header_end
        for element in elements:
            for _ in range(element["count"]):
                values: dict[str, float] = {}
                for prop in element["properties"]:
                    if prop[0] == "scalar":
                        fmt_char = _PLY_SCALARS[prop[1]]
                        value = struct.unpack_from(endian + fmt_char, raw, cursor)[0]
                        cursor += struct.calcsize(fmt_char)
                        if element["name"] == "vertex" and prop[2] in {"x", "y", "z"}:
                            values[prop[2]] = float(value)
                    else:
                        count_fmt = _PLY_SCALARS[prop[1]]
                        count = int(struct.unpack_from(endian + count_fmt, raw, cursor)[0])
                        cursor += struct.calcsize(count_fmt)
                        cursor += count * struct.calcsize(_PLY_SCALARS[prop[2]])
                    if cursor > len(raw):
                        raise EvidenceValidationError("PLY_VERTEX_COUNT_SHORT")
                if element["name"] == "vertex":
                    rows.append((values["x"], values["y"], values["z"]))
    if len(rows) != int(vertex["count"]):
        raise EvidenceValidationError("PLY_VERTEX_COUNT_SHORT")
    return np.asarray(rows, dtype=np.float64), header_bytes, len(rows), tuple(ignored)


def _canonical_sample(points: Any, maximum: int) -> tuple[Any, list[str], int, int, str]:
    import numpy as np

    array = np.asarray(points, dtype=np.float64)
    finite_mask = np.isfinite(array).all(axis=1)
    finite = array[finite_mask].copy()
    finite[finite == 0.0] = 0.0
    unique: dict[bytes, None] = {}
    for point in finite:
        unique.setdefault(struct.pack("<ddd", *map(float, point)), None)
    duplicates = len(finite) - len(unique)
    entries = []
    for packed in unique:
        point_id = _sha(b"PWTS_POINT_V1\0" + packed)
        rank = _sha(b"PWTS_SAMPLE_V1\0" + packed)
        entries.append((point_id, rank, packed))
    if len(entries) > maximum:
        entries = sorted(entries, key=lambda item: (item[1], item[2]))[:maximum]
    entries.sort(key=lambda item: item[0])
    ids = [item[0] for item in entries]
    sampled = np.asarray([struct.unpack("<ddd", item[2]) for item in entries], dtype=np.float64)
    all_finite_packed = sorted(struct.pack("<ddd", *map(float, point)) for point in finite)
    multiset_digest = _sha(b"PWTS_COORDINATE_MULTISET_V1\0" + b"".join(all_finite_packed))
    return sampled, ids, int(finite_mask.sum()), duplicates, multiset_digest


def _exact_knn(points: Any, point_ids: Sequence[str], k: int = 64) -> tuple[Any, Any]:
    import numpy as np
    from scipy.spatial import cKDTree

    if len(points) <= k:
        raise EvidenceValidationError("INSUFFICIENT_UNIQUE_POINTS")
    if len(points) <= 2048:
        out_i = np.empty((len(points), k), dtype=np.int64)
        out_d = np.empty((len(points), k), dtype=np.float64)
        for i in range(len(points)):
            delta = points - points[i]
            d2 = np.einsum("ij,ij->i", delta, delta)
            chosen = sorted((j for j in range(len(points)) if j != i), key=lambda j: (float(d2[j]), bytes.fromhex(point_ids[j])))[:k]
            out_i[i] = chosen
            out_d[i] = np.sqrt(d2[chosen])
        return out_i, out_d
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    out_i = np.empty((len(points), k), dtype=np.int64)
    out_d = np.empty((len(points), k), dtype=np.float64)
    query_k = min(len(points), k + 2)
    for start in range(0, len(points), 4096):
        stop = min(len(points), start + 4096)
        raw_d, raw_i = tree.query(points[start:stop], k=query_k, eps=0.0, workers=1)
        for local, i in enumerate(range(start, stop)):
            candidates = {int(j) for j in np.atleast_1d(raw_i[local]) if int(j) != i}
            prelim = sorted(candidates, key=lambda j: (float(np.dot(points[j] - points[i], points[j] - points[i])), bytes.fromhex(point_ids[j])))
            if len(prelim) < k:
                raise EvidenceValidationError("KNN_FAILURE")
            kth_d2 = float(np.dot(points[prelim[k - 1]] - points[i], points[prelim[k - 1]] - points[i]))
            next_d2 = float(np.dot(points[prelim[k]] - points[i], points[prelim[k]] - points[i])) if len(prelim) > k else math.inf
            if next_d2 == kth_d2:
                radius = np.nextafter(math.sqrt(kth_d2), math.inf)
                candidates = set(int(j) for j in tree.query_ball_point(points[i], radius, eps=0.0))
                candidates.discard(i)
                prelim = sorted(candidates, key=lambda j: (float(np.dot(points[j] - points[i], points[j] - points[i])), bytes.fromhex(point_ids[j])))
            chosen = prelim[:k]
            out_i[i] = chosen
            out_d[i] = [math.sqrt(float(np.dot(points[j] - points[i], points[j] - points[i]))) for j in chosen]
    return out_i, out_d


def _canonical_vector(vector: Any) -> Any:
    import numpy as np
    result = np.asarray(vector, dtype=np.float64).copy()
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0:
        result = -result
    result[result == 0.0] = 0.0
    return result


def _field(points: Any, point_ids: Sequence[str], knn: Any, distances: Any) -> dict[str, Any]:
    import numpy as np

    count = len(points)
    normals = {k: np.zeros((count, 3), dtype=np.float64) for k in (16, 32, 64)}
    variations = {k: np.full(count, np.inf, dtype=np.float64) for k in (16, 32, 64)}
    valid = {k: np.zeros(count, dtype=bool) for k in (16, 32, 64)}
    for i in range(count):
        for k in (16, 32, 64):
            local = np.vstack((points[i], points[knn[i, :k]]))
            centered = local - local.mean(axis=0)
            covariance = centered.T @ centered / len(local)
            values, vectors = np.linalg.eigh(covariance)
            total = float(values.sum())
            if values[2] > 0 and values[1] / values[2] >= 0.02 and total > 0 and values[0] / total <= 0.25:
                valid[k][i] = True
                normals[k][i] = _canonical_vector(vectors[:, 0])
                variations[k][i] = values[0] / total
    all_valid = valid[16] & valid[32] & valid[64]
    d1 = np.zeros(count, dtype=np.float64)
    d2 = np.zeros(count, dtype=np.float64)
    indices = np.flatnonzero(all_valid)
    for i in indices:
        d1[i] = math.acos(float(np.clip(abs(float(np.dot(normals[16][i], normals[32][i]))), 0.0, 1.0)))
        d2[i] = math.acos(float(np.clip(abs(float(np.dot(normals[32][i], normals[64][i]))), 0.0, 1.0)))
    response = np.sqrt(d1 * d2)
    maximum = np.maximum(d1, d2)
    balance = np.divide(np.minimum(d1, d2), maximum, out=np.zeros_like(maximum), where=maximum > 0)
    seed_eligible = all_valid & (d1 >= math.radians(3)) & (d2 >= math.radians(3)) & (response >= math.radians(6)) & (balance >= 0.35)
    trace_eligible = all_valid & (d1 >= math.radians(2)) & (d2 >= math.radians(2)) & (response >= math.radians(4)) & (balance >= 0.25)
    local_max = np.zeros(count, dtype=bool)
    for i in np.flatnonzero(seed_eligible):
        local_max[i] = all(
            response[i] > response[j] or (response[i] == response[j] and point_ids[i] < point_ids[j])
            for j in knn[i, :16]
        )
    tangents = np.zeros((count, 3), dtype=np.float64)
    tangent_valid = np.zeros(count, dtype=bool)
    tangent_support = np.zeros(count, dtype=np.int64)
    tangent_anisotropy = np.zeros(count, dtype=np.float64)
    for i in np.flatnonzero(trace_eligible):
        neighbors = [int(j) for j in knn[i, :64] if trace_eligible[j]]
        tangent_support[i] = len(neighbors)
        if len(neighbors) < 8:
            continue
        offsets = points[neighbors] - points[i]
        normal = normals[64][i]
        projected = offsets - np.outer(offsets @ normal, normal)
        weights = response[neighbors] ** 2
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        centered = projected - np.average(projected, axis=0, weights=weights)
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        values, vectors = np.linalg.eigh(covariance)
        denominator = values[1] + values[2]
        anisotropy = values[2] / denominator if denominator > 0 else 0.0
        tangent_anisotropy[i] = anisotropy
        if anisotropy >= 0.65:
            tangent_valid[i] = True
            tangents[i] = _canonical_vector(vectors[:, 2])
    return {
        "normals": normals, "variations": variations, "valid": valid,
        "d1": d1, "d2": d2, "response": response, "balance": balance,
        "seed_eligible": seed_eligible, "trace_eligible": trace_eligible,
        "local_max": local_max, "tangents": tangents,
        "tangent_valid": tangent_valid, "tangent_support": tangent_support,
        "tangent_anisotropy": tangent_anisotropy, "h": distances[:, 31],
    }


def _trace(seed: int, points: Any, ids: Sequence[str], knn: Any, field: Mapping[str, Any]) -> tuple[list[int], bool, str, bool]:
    import numpy as np

    if not field["tangent_valid"][seed]:
        code = "TANGENT_SUPPORT_INSUFFICIENT" if field["tangent_support"][seed] < 8 else "TANGENT_ANISOTROPY_INSUFFICIENT"
        return [seed], False, code, False

    def choices_for(current: int, direction: Any, used: set[int]) -> list[tuple[Any, ...]]:
            choices: list[tuple[Any, ...]] = []
            for candidate in knn[current, :32]:
                candidate = int(candidate)
                if candidate in used or not field["trace_eligible"][candidate] or not field["tangent_valid"][candidate]:
                    continue
                offset = points[candidate] - points[current]
                length = float(np.linalg.norm(offset))
                if length <= 0 or length > 1.25 * field["h"][current]:
                    continue
                unit = offset / length
                forward = float(np.dot(unit, direction))
                if forward < 0.7071067811865476:
                    continue
                tangent = field["tangents"][candidate].copy()
                if np.dot(tangent, direction) < 0:
                    tangent *= -1
                alignment = float(np.dot(tangent, direction))
                if alignment < 0.8191520442889918:
                    continue
                turn = math.acos(max(-1.0, min(1.0, alignment)))
                lateral = float(np.linalg.norm(offset - np.dot(offset, direction) * direction) / length)
                choices.append((turn, lateral, -float(field["response"][candidate]), length / field["h"][current], ids[candidate], candidate, tangent))
            choices.sort(key=lambda item: item[:5])
            return choices

    arms = {
        "negative": {"current": seed, "direction": -field["tangents"][seed], "used": {seed}, "nodes": [], "exhausted": False},
        "positive": {"current": seed, "direction": field["tangents"][seed].copy(), "used": {seed}, "nodes": [], "exhausted": False},
    }
    while 1 + len(arms["negative"]["nodes"]) + len(arms["positive"]["nodes"]) < 257:
        advanced = False
        for arm_name in ("negative", "positive"):
            if 1 + len(arms["negative"]["nodes"]) + len(arms["positive"]["nodes"]) >= 257:
                break
            arm = arms[arm_name]
            if arm["exhausted"]:
                continue
            choices = choices_for(arm["current"], arm["direction"], arm["used"])
            if not choices:
                arm["exhausted"] = True
                continue
            choice = choices[0]
            arm["current"], arm["direction"] = choice[5], choice[6]
            arm["nodes"].append(choice[5]); arm["used"].add(choice[5])
            advanced = True
        if not advanced and all(arm["exhausted"] for arm in arms.values()):
            break
    negative = arms["negative"]["nodes"]
    positive = arms["positive"]["nodes"]
    cap_with_extension = any(
        choices_for(arm["current"], arm["direction"], arm["used"])
        for arm in arms.values() if not arm["exhausted"]
    )
    nodes = list(reversed(negative)) + [seed] + positive
    h_t = _lower_quantile([field["h"][i] for i in nodes], 0.5)
    steps = [float(np.linalg.norm(points[b] - points[a])) for a, b in zip(nodes, nodes[1:])]
    if cap_with_extension:
        return nodes, False, "TRACE_NODE_CAP_EXCEEDED", True
    if len(nodes) < 9:
        return nodes, False, "TRACE_NODE_COUNT_INSUFFICIENT", cap_with_extension
    if sum(steps) < 6.0 * h_t:
        return nodes, False, "TRACE_LENGTH_INSUFFICIENT", cap_with_extension
    if any(step > 1.5 * h_t for step in steps):
        return nodes, False, "TRACE_DISCONTINUITY", cap_with_extension
    if len(set(positive) & set(negative)) or len(nodes) != len(set(nodes)):
        return nodes, False, "TRACE_ARM_COLLISION", cap_with_extension
    return nodes, True, "VALID", False


def _compact_array_digest(array: Any) -> str:
    import numpy as np
    value = np.asarray(array)
    if value.dtype.kind == "f":
        value = np.array(value, dtype="<f8", copy=True)
        value[value == 0.0] = 0.0
    elif value.dtype.kind in "iu":
        value = np.asarray(value, dtype="<i8")
    elif value.dtype.kind == "b":
        value = np.asarray(value, dtype=np.uint8)
    value = np.ascontiguousarray(value)
    return _sha(
        value.dtype.str.encode("ascii")
        + canonical_json_bytes(list(value.shape))
        + value.tobytes(order="C")
    )


def _quantize_independent(value: Any) -> Any:
    import numpy as np
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise EvidenceValidationError("NONFINITE_SERIALIZED_NUMERIC")
        return _q(float(value))
    if isinstance(value, Mapping):
        return {key: _quantize_independent(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_quantize_independent(child) for child in value]
    return value


def reconstruct_case_from_raw(
    *, case_token: str, raw_bytes: bytes, config: Mapping[str, Any],
    config_sha256: str, schema_sha256: str,
    source_binding: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Independently rebuild the point-native field and primitives from raw XYZ.

    The implementation deliberately duplicates the locked mathematics.  It calls
    no runner semantic, sampler, normal, tracing, strip, or certificate function.
    """
    import numpy as np

    parsed, header, parsed_count, ignored = _parse_ply_xyz(raw_bytes)
    points, ids, finite_count, duplicates, coordinate_digest = _canonical_sample(
        parsed, int(config["input"]["maximum_sample_points"])
    )
    if len(points) < int(config["input"]["minimum_unique_points"]):
        raise EvidenceValidationError("INSUFFICIENT_UNIQUE_POINTS")
    knn, distances = _exact_knn(points, ids)
    field = _field(points, ids, knn, distances)
    tables: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (
            "case_manifest", "field_summary", "seed_primitive", "trace_primitive",
            "trace_node", "strip_membership", "strip_summary", "certificate",
            "destructive_negative", "rejection_census",
        )
    }
    sampled_digest = sorted_id_digest(ids)
    binding = dict(source_binding or {})
    tables["case_manifest"].append({
        "case_token": case_token, "raw_path": binding.get("raw_path"),
        "raw_ply_sha256": _sha(raw_bytes), "observed_raw_ply_sha256": _sha(raw_bytes),
        "raw_byte_count": len(raw_bytes),
        "ply_header_sha256": _sha(header), "parsed_vertex_count": parsed_count,
        "finite_coordinate_count": finite_count, "exact_duplicate_count": duplicates,
        "ignored_property_name_digest": sha256_json(sorted(ignored)),
        "unique_coordinate_count": finite_count - duplicates,
        "canonical_coordinate_multiset_digest": coordinate_digest,
        "sample_count": len(points), "sampled_point_id_digest": sampled_digest,
        "validator_sampled_point_id_digest": sampled_digest,
        "config_sha256": config_sha256, "schema_sha256": schema_sha256,
        "source_points_serialized": False,
    })
    tables["field_summary"].append({
        "case_token": case_token, "sample_count": len(points),
        "normal_valid_count_k16": int(field["valid"][16].sum()),
        "normal_valid_count_k32": int(field["valid"][32].sum()),
        "normal_valid_count_k64": int(field["valid"][64].sum()),
        "seed_eligible_count": int(field["seed_eligible"].sum()),
        "trace_eligible_count": int(field["trace_eligible"].sum()),
        "deterministic_local_maximum_count": int(field["local_max"].sum()),
        "dense_knn_digest": _compact_array_digest(knn),
        "dense_normal_digest": _sha("".join(_compact_array_digest(field["normals"][k]) for k in (16, 32, 64)).encode()),
        "dense_response_digest": _compact_array_digest(field["response"]),
    })
    trace_records: list[tuple[str, list[int], float]] = []
    for seed in np.flatnonzero(field["local_max"]):
        nodes, valid, code, _ = _trace(int(seed), points, ids, knn, field)
        if valid:
            previous_g = None
            for a, b in zip(nodes, nodes[1:]):
                segment = points[b] - points[a]
                segment_length = float(np.linalg.norm(segment))
                if segment_length == 0:
                    valid, code = False, "TRACE_DISCONTINUITY"; break
                tangent = segment / segment_length
                n0, n1 = field["normals"][64][a].copy(), field["normals"][64][b].copy()
                if np.dot(n0, n1) < 0:
                    n1 *= -1
                summed = n0 + n1
                if float(np.linalg.norm(summed)) == 0:
                    valid, code = False, "TRACE_DISCONTINUITY"; break
                normal = summed / np.linalg.norm(summed)
                g = np.cross(normal, tangent)
                if float(np.linalg.norm(g)) == 0:
                    valid, code = False, "TRACE_DISCONTINUITY"; break
                g /= np.linalg.norm(g)
                if previous_g is not None and np.dot(g, previous_g) < 0:
                    g *= -1
                previous_g = _canonical_vector(g) if previous_g is None else g
        trace_id = _sha(b"PWTS_TRACE_V1\0" + case_token.encode() + bytes.fromhex(ids[seed]) + bytes.fromhex(config_sha256))
        seed_row = {
            "case_token": case_token, "seed_point_id": ids[seed],
            "d1_rad": _q(field["d1"][seed]), "d2_rad": _q(field["d2"][seed]),
            "response_rad": _q(field["response"][seed]), "scale_balance": _q(field["balance"][seed]),
            "tangent_support_count": int(field["tangent_support"][seed]),
            "tangent_anisotropy": _q(field["tangent_anisotropy"][seed]),
            "trace_id": trace_id, "valid": valid, "first_rejection_code": code,
        }
        tables["seed_primitive"].append(seed_row)
        h_t = _lower_quantile([field["h"][i] for i in nodes], 0.5)
        node_rows = []
        for ordinal, node in enumerate(nodes):
            node_rows.append({
                "case_token": case_token, "trace_id": trace_id, "ordinal": ordinal,
                "point_id": ids[node], "response_rad": _q(field["response"][node]),
                "tangent_x": _q(field["tangents"][node, 0]),
                "tangent_y": _q(field["tangents"][node, 1]),
                "tangent_z": _q(field["tangents"][node, 2]),
            })
        tables["trace_node"].extend(node_rows)
        steps = [float(np.linalg.norm(points[b] - points[a])) for a, b in zip(nodes, nodes[1:])]
        turns = []
        for a, b in zip(nodes, nodes[1:]):
            turns.append(math.acos(float(np.clip(abs(np.dot(field["tangents"][a], field["tangents"][b])), 0, 1))))
        tables["trace_primitive"].append({
            "case_token": case_token, "trace_id": trace_id, "seed_point_id": ids[seed],
            "node_count": len(nodes), "h_t": _q(h_t),
            "normalized_arc_length": _q(sum(steps) / h_t if h_t else 0),
            "maximum_normalized_step": _q(max(steps, default=0) / h_t if h_t else 0),
            "median_response_rad": _q(_lower_quantile([field["response"][i] for i in nodes], 0.5)),
            "maximum_turn_rad": _q(max(turns, default=0)), "node_table_digest": sorted_id_digest(row["point_id"] for row in node_rows),
            "valid": valid, "first_rejection_code": code,
        })
        if valid:
            trace_records.append((trace_id, nodes, h_t))

    # Explicit strip reconstruction.  It is intentionally direct and point-native.
    for trace_id, nodes, h_t in trace_records:
        trace_set = set(nodes)
        segment_starts = points[nodes[:-1]]
        segment_ends = points[nodes[1:]]
        segment_vectors = segment_ends - segment_starts
        lengths = np.linalg.norm(segment_vectors, axis=1)
        tangents = segment_vectors / lengths[:, None]
        gs = []
        for ordinal, (a, b, tangent) in enumerate(zip(nodes, nodes[1:], tangents)):
            na, nb = field["normals"][64][a].copy(), field["normals"][64][b].copy()
            if np.dot(na, nb) < 0:
                nb *= -1
            normal = na + nb
            normal /= np.linalg.norm(normal)
            g = np.cross(normal, tangent)
            g /= np.linalg.norm(g)
            if ordinal == 0:
                g = _canonical_vector(g)
            elif np.dot(g, gs[-1]) < 0:
                g *= -1
            gs.append(g)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        total_length = float(cumulative[-1])
        membership: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        for point_index in range(len(points)):
            if point_index in trace_set or not all(field["valid"][k][point_index] for k in (16, 32, 64)):
                continue
            best = None
            best_d2 = math.inf
            for ordinal, (start, vector, length) in enumerate(zip(segment_starts, segment_vectors, lengths)):
                # Re-express the locked nearest-segment equation directly in
                # vector/length-squared form.  Strict improvement preserves the
                # smaller ordinal on an exact tie, matching the specification
                # without calling shared semantic code.
                u = float(np.clip(
                    np.dot(points[point_index] - start, vector) / float(length * length),
                    0.0,
                    1.0,
                ))
                closest = start + u * vector
                displacement = points[point_index] - closest
                d2 = float(np.dot(displacement, displacement))
                if d2 < best_d2:
                    best_d2 = d2
                    best = (math.sqrt(d2), ordinal, u * float(length), displacement)
            assert best is not None
            rho, ordinal, projection, displacement = best
            if not (1.5 * h_t <= rho <= 6.0 * h_t) or rho <= 0:
                continue
            cosine = float(np.dot(displacement, gs[ordinal]) / rho)
            side = "left" if cosine <= -0.35 else "right" if cosine >= 0.35 else None
            if side is None:
                continue
            arc = cumulative[ordinal] + projection
            arc_bin = min(7, int(8 * arc / total_length)) if total_length > 0 else 0
            membership[side].append({
                "case_token": case_token, "trace_id": trace_id, "side": side,
                "point_id": ids[point_index], "nearest_segment_ordinal": ordinal,
                "arc_bin": arc_bin, "rho_over_h_t": _q(rho / h_t),
                "signed_side_cosine": _q(cosine), "response_rad": _q(field["response"][point_index]),
                "surface_variation": _q(field["variations"][16][point_index]),
            })
        side_summaries = {}
        for side in ("left", "right"):
            rows = sorted(membership[side], key=lambda row: row["point_id"])
            tables["strip_membership"].extend(rows)
            indices = [ids.index(row["point_id"]) for row in rows]
            bins = {b for b in range(8) if sum(row["arc_bin"] == b for row in rows) >= 3}
            if indices:
                normal_matrix = field["normals"][16][indices]
                values, vectors = np.linalg.eigh(normal_matrix.T @ normal_matrix / len(indices))
                dominant = _canonical_vector(vectors[:, -1])
                coherence = float(values[-1])
                variations = [field["variations"][16][i] for i in indices]
                responses = [field["response"][i] for i in indices]
            else:
                dominant = np.zeros(3); coherence = 0.0; variations = []; responses = []
            summary = {
                "case_token": case_token, "trace_id": trace_id, "side": side,
                "count": len(rows), "bin_counts": [sum(row["arc_bin"] == b for row in rows) for b in range(8)],
                "covered_bins": len(bins), "normal_coherence": _q(coherence),
                "dominant_normal_x": _q(dominant[0]), "dominant_normal_y": _q(dominant[1]),
                "dominant_normal_z": _q(dominant[2]),
                "median_surface_variation": _q(_lower_quantile(variations, 0.5)),
                "p90_surface_variation": _q(_lower_quantile(variations, 0.9)),
                "median_response_rad": _q(_lower_quantile(responses, 0.5)),
                "sorted_support_digest": sorted_id_digest(row["point_id"] for row in rows),
            }
            tables["strip_summary"].append(summary)
            side_summaries[side] = (summary, bins, rows, dominant)
        left, right = side_summaries["left"], side_summaries["right"]
        trace_row = next(row for row in tables["trace_primitive"] if row["trace_id"] == trace_id)
        left_count, right_count = left[0]["count"], right[0]["count"]
        ratio = min(left_count, right_count) / max(left_count, right_count) if left_count and right_count else 0.0
        separation = math.acos(float(np.clip(abs(np.dot(left[3], right[3])), 0, 1))) if left_count and right_count else 0.0
        cert = {
            "case_token": case_token, "trace_id": trace_id, "trace_valid": trace_row["valid"],
            "membership_intersection_count": len({r["point_id"] for r in left[2]} & {r["point_id"] for r in right[2]}),
            "left_support_count": left_count, "right_support_count": right_count,
            "left_covered_bins": len(left[1]), "right_covered_bins": len(right[1]),
            "bilaterally_covered_bins": len(left[1] & right[1]), "side_count_ratio": _q(ratio),
            "left_normal_coherence": left[0]["normal_coherence"], "right_normal_coherence": right[0]["normal_coherence"],
            "left_median_surface_variation": left[0]["median_surface_variation"], "left_p90_surface_variation": left[0]["p90_surface_variation"],
            "right_median_surface_variation": right[0]["median_surface_variation"], "right_p90_surface_variation": right[0]["p90_surface_variation"],
            "side_normal_separation_rad": _q(separation), "trace_median_response_rad": trace_row["median_response_rad"],
            "left_median_response_rad": left[0]["median_response_rad"], "right_median_response_rad": right[0]["median_response_rad"],
        }
        gates, rejection, valid = _certificate_truth(cert)
        cert.update(gates); cert["valid"] = valid; cert["first_rejection_code"] = rejection
        left_digest, right_digest = left[0]["sorted_support_digest"], right[0]["sorted_support_digest"]
        cert_id = _sha(b"PWTS_CERT_V1\0" + bytes.fromhex(trace_id) + bytes.fromhex(left_digest) + bytes.fromhex(right_digest) + bytes.fromhex(config_sha256))
        cert["certificate_id"] = cert_id
        tables["certificate"].append(cert)
        if valid:
            for arm, side, code in (("DELETE_LEFT", "left", "LEFT_SUPPORT_INSUFFICIENT"), ("DELETE_RIGHT", "right", "RIGHT_SUPPORT_INSUFFICIENT")):
                rebuilt_cert = dict(cert)
                rebuilt_cert["left_support_count"] = 0 if side == "left" else left_count
                rebuilt_cert["right_support_count"] = 0 if side == "right" else right_count
                rebuilt_cert["left_covered_bins"] = 0 if side == "left" else len(left[1])
                rebuilt_cert["right_covered_bins"] = 0 if side == "right" else len(right[1])
                rebuilt_cert["bilaterally_covered_bins"] = 0
                rebuilt_cert["side_count_ratio"] = 0.0
                if side == "left":
                    rebuilt_cert.update(left_normal_coherence=0.0, left_median_surface_variation=0.0, left_p90_surface_variation=0.0, left_median_response_rad=0.0)
                else:
                    rebuilt_cert.update(right_normal_coherence=0.0, right_median_surface_variation=0.0, right_p90_surface_variation=0.0, right_median_response_rad=0.0)
                rebuilt_cert["side_normal_separation_rad"] = 0.0
                rebuilt_gates, rebuilt_rejection, rebuilt_valid = _certificate_truth(rebuilt_cert)
                deletion_row = {
                    "case_token": case_token, "certificate_id": cert_id, "deletion_arm": arm,
                    "trace_id": trace_id, "deleted_id_digest": side_summaries[side][0]["sorted_support_digest"],
                    "deleted_id_count": side_summaries[side][0]["count"],
                    "rebuilt_left_support_count": 0 if side == "left" else left_count,
                    "rebuilt_right_support_count": 0 if side == "right" else right_count,
                    "rebuilt_valid": rebuilt_valid, "first_rejection_code": rebuilt_rejection,
                    "required_first_rejection_code": code, "raw_id_deletion_applied": True,
                }
                deletion_row.update({f"rebuilt_{key}": value for key, value in rebuilt_gates.items()})
                tables["destructive_negative"].append(deletion_row)

    empty_support_digest = hashlib.sha256(b"").hexdigest()
    for trace_row in (row for row in tables["trace_primitive"] if not row["valid"]):
        trace_id = trace_row["trace_id"]
        for side in ("left", "right"):
            tables["strip_summary"].append({
                "case_token": case_token, "trace_id": trace_id, "side": side,
                "count": 0, "bin_counts": [0] * 8, "covered_bins": 0,
                "normal_coherence": 0.0, "dominant_normal_x": 0.0,
                "dominant_normal_y": 0.0, "dominant_normal_z": 0.0,
                "median_surface_variation": 0.0, "p90_surface_variation": 0.0,
                "median_response_rad": 0.0,
                "sorted_support_digest": empty_support_digest,
            })
        cert = {
            "case_token": case_token, "trace_id": trace_id, "trace_valid": False,
            "membership_intersection_count": 0, "left_support_count": 0,
            "right_support_count": 0, "left_covered_bins": 0,
            "right_covered_bins": 0, "bilaterally_covered_bins": 0,
            "side_count_ratio": 0.0, "left_normal_coherence": 0.0,
            "right_normal_coherence": 0.0, "left_median_surface_variation": 0.0,
            "left_p90_surface_variation": 0.0, "right_median_surface_variation": 0.0,
            "right_p90_surface_variation": 0.0, "side_normal_separation_rad": 0.0,
            "trace_median_response_rad": trace_row["median_response_rad"],
            "left_median_response_rad": 0.0, "right_median_response_rad": 0.0,
        }
        gates, rejection, valid = _certificate_truth(cert)
        cert.update(gates); cert["valid"] = valid; cert["first_rejection_code"] = rejection
        cert["certificate_id"] = _sha(
            b"PWTS_CERT_V1\0" + bytes.fromhex(trace_id)
            + bytes.fromhex(empty_support_digest) + bytes.fromhex(empty_support_digest)
            + bytes.fromhex(config_sha256)
        )
        tables["certificate"].append(cert)

    def add_rejections(
        stage: str, codes: Sequence[str], *, explicit_empty: bool = True
    ) -> None:
        counts: dict[str, int] = {}
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
        if not counts and explicit_empty:
            counts["EMPTY_POPULATION"] = 0
        for code, numerator in sorted(counts.items()):
            tables["rejection_census"].append({
                "case_token": case_token, "stage": stage, "rejection_code": code,
                "numerator": numerator, "denominator": len(codes),
            })

    sampled_point_codes = []
    for i in range(len(points)):
        if not all(field["valid"][k][i] for k in (16, 32, 64)):
            sampled_point_codes.append("INVALID_NORMALS")
        elif field["d1"][i] < 0.05235987755982989 or field["d2"][i] < 0.05235987755982989 or field["response"][i] < 0.10471975511965978:
            sampled_point_codes.append("RESPONSE_FLOOR")
        elif field["balance"][i] < 0.35:
            sampled_point_codes.append("SCALE_BALANCE")
        elif not field["local_max"][i]:
            sampled_point_codes.append("NONMAXIMUM")
        else:
            sampled_point_codes.append("VALID")
    add_rejections("sampled_point_seed", sampled_point_codes)
    for detailed_stage in (
        "seed_primitive", "trace_primitive", "certificate", "destructive_negative"
    ):
        add_rejections(
            detailed_stage,
            [row["first_rejection_code"] for row in tables[detailed_stage]],
            explicit_empty=False,
        )
    add_rejections("all_sampled_points", ["VALID"] * len(points))
    for k in (16, 32, 64):
        add_rejections(f"normal_valid_k{k}", ["VALID"] * int(field["valid"][k].sum()))
    add_rejections("seed_eligible_points", ["VALID"] * int(field["seed_eligible"].sum()))
    add_rejections("deterministic_local_maxima", ["VALID"] * int(field["local_max"].sum()))
    add_rejections("seed_primitives", [row["first_rejection_code"] for row in tables["seed_primitive"]])
    add_rejections("valid_traces", ["VALID"] * sum(bool(row["valid"]) for row in tables["trace_primitive"]))
    add_rejections("certificate_candidates", [row["first_rejection_code"] for row in tables["certificate"]])
    add_rejections("valid_certificates", ["VALID"] * sum(bool(row["valid"]) for row in tables["certificate"]))
    add_rejections("destructive_negative_arms", [row["first_rejection_code"] for row in tables["destructive_negative"]])
    for name, rows in tables.items():
        if name in PRIMARY_KEYS:
            rows.sort(key=lambda row: row_key(name, row))
        tables[name] = _quantize_independent(rows)
    return tables


def _manifest_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("case_manifest", "cases", "rows"):
            if isinstance(value.get(key), list):
                return value[key]
    raise EvidenceValidationError("SOURCE_MANIFEST_SCHEMA_INVALID")


def independent_validator_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent PWTS semantic reconstruction")
    parser.add_argument("--role", required=True, choices=["INDEPENDENT_VALIDATOR"])
    parser.add_argument("--case-token", required=True)
    parser.add_argument("--raw-ply", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-file-sha256", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--expected-configuration-file-sha256", required=True)
    parser.add_argument("--expected-configuration-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--runner-source", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch-dir", required=True)
    args = parser.parse_args(argv)
    try:
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        if _sha(Path(__file__).read_bytes()) != args.expected_executable_sha256:
            raise EvidenceValidationError("INDEPENDENT_EXECUTABLE_HASH_MISMATCH")
        runner_source = Path(args.runner_source).read_bytes()
        if _sha(runner_source) != args.expected_runner_sha256:
            raise EvidenceValidationError("RUNNER_SOURCE_HASH_MISMATCH")
        if forbidden_imports(runner_source.decode("utf-8")):
            raise EvidenceValidationError("RUNNER_FORBIDDEN_IMPORT")
        source_bytes = Path(args.source_manifest).read_bytes()
        if _sha(source_bytes) != args.expected_source_manifest_file_sha256:
            raise EvidenceValidationError("SOURCE_MANIFEST_HASH_MISMATCH")
        manifest = json.loads(source_bytes)
        manifest_content = dict(manifest)
        asserted_content_hash = manifest_content.pop("input_manifest_sha256", None)
        observed_content_hash = _sha(canonical_json_bytes(manifest_content))
        if (
            asserted_content_hash != observed_content_hash
            or observed_content_hash != args.expected_input_manifest_sha256
        ):
            raise EvidenceValidationError("SOURCE_MANIFEST_CONTENT_HASH_MISMATCH")
        config_bytes = Path(args.configuration).read_bytes()
        config = json.loads(config_bytes)
        if args.expected_configuration_sha256 != RULING_CONFIGURATION_SHA256:
            raise EvidenceValidationError("UNAUTHORIZED_CONFIGURATION_HASH")
        if _sha(config_bytes) != args.expected_configuration_file_sha256:
            raise EvidenceValidationError("CONFIGURATION_FILE_HASH_MISMATCH")
        if _sha(canonical_json_bytes(config)) != args.expected_configuration_sha256:
            raise EvidenceValidationError("CONFIGURATION_HASH_MISMATCH")
        schema_bytes = Path(args.schema).read_bytes()
        schema = json.loads(schema_bytes)
        if _sha(schema_bytes) != args.expected_schema_sha256:
            raise EvidenceValidationError("SCHEMA_HASH_MISMATCH")
        raw_bytes = Path(args.raw_ply).read_bytes()
        matching = [row for row in _manifest_rows(manifest) if row.get("case_token") == args.case_token]
        if len(matching) != 1:
            raise EvidenceValidationError("CASE_BINDING_MISSING")
        binding = dict(matching[0])
        asserted_raw_hash = binding.get("raw_ply_sha256", binding.get("raw_sha256"))
        if asserted_raw_hash != _sha(raw_bytes):
            raise EvidenceValidationError("RAW_SOURCE_HASH_MISMATCH")
        parsed, header, parsed_count, ignored = _parse_ply_xyz(raw_bytes)
        if binding.get("ply_header_sha256") != _sha(header):
            raise EvidenceValidationError("PLY_HEADER_HASH_MISMATCH")
        if int(binding.get("parsed_vertex_count", -1)) != parsed_count:
            raise EvidenceValidationError("PARSED_VERTEX_COUNT_MISMATCH")
        ignored_digest = sha256_json(sorted(ignored))
        if binding.get("ignored_property_name_digest") not in (None, ignored_digest):
            raise EvidenceValidationError("IGNORED_PROPERTY_DIGEST_MISMATCH")
        binding.update({
            "raw_path": str(Path(args.raw_ply)), "raw_byte_count": len(raw_bytes),
            "observed_raw_ply_sha256": _sha(raw_bytes),
            "ignored_property_name_digest": ignored_digest,
            "schema_sha256": args.expected_schema_sha256,
        })
        tables = reconstruct_case_from_raw(
            case_token=args.case_token, raw_bytes=raw_bytes, config=config,
            config_sha256=args.expected_configuration_sha256,
            schema_sha256=args.expected_schema_sha256,
            source_binding=binding,
        )
        payload = {
            "preflight_id": PREFLIGHT_ID, "case_token": args.case_token,
            "process_role": "INDEPENDENT_VALIDATOR", "source_points_serialized": False,
            "tables": tables,
        }
        _walk_forbidden_payload(payload, "$", [])
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)
        (output_dir / "case_payload.json").write_bytes(canonical_json_bytes(payload))
        return 0
    except Exception as exc:
        print(f"{INVALID_STATUS}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "EvidenceValidationError",
    "INVALID_STATUS",
    "PREFLIGHT_ID",
    "RULING_CONFIGURATION_SHA256",
    "TABLES",
    "VALID_STATUS",
    "ValidationPolicy",
    "ValidationResult",
    "canonical_json_bytes",
    "forbidden_imports",
    "derive_expected_criteria",
    "independent_validator_cli",
    "imported_modules",
    "inspect_import_boundary",
    "row_key",
    "reconstruct_case_from_raw",
    "sha256_json",
    "sorted_id_digest",
    "ordered_id_digest",
    "table_digest",
    "validate_evidence",
]


if __name__ == "__main__":
    raise SystemExit(independent_validator_cli())
