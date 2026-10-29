"""End-to-end frozen V7 semantic role over an already-open PLY descriptor."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from .config import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    load_frozen_v7,
    validate_frozen_allocation_plan,
)
from .container import encode_container
from .fields import FieldValidationError, build_fields
from .graph import build_space_scale_adjacency, h0_persistence
from .identifiers import (
    CollisionRegistry,
    case_token_digest,
    validate_v7_known_answer_vectors,
)
from .partition import SampledPoints, build_partition, canonical_sample
from .primary_parser import ParsedPly, parse_ply_fd


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SemanticStop(RuntimeError):
    """A terminal frozen-family STOP; the authorized smoke must not be rerun."""


@dataclass(frozen=True, slots=True)
class RoleResult:
    container_bytes: bytes
    table_bytes: Mapping[str, bytes]
    census: Mapping[str, int]


def verify_source_diagnostics(parsed: ParsedPly, expected: Mapping[str, Any]) -> None:
    diagnostics = parsed.diagnostics
    pairs = {
        "raw_byte_count": diagnostics.raw_byte_count,
        "raw_ply_sha256": diagnostics.raw_ply_sha256,
        "ply_header_byte_count": diagnostics.ply_header_byte_count,
        "ply_header_sha256": diagnostics.ply_header_sha256,
        "ply_format": diagnostics.ply_format,
        "parsed_vertex_count": diagnostics.parsed_vertex_count,
        "finite_coordinate_count": diagnostics.finite_coordinate_count,
        "exact_duplicate_count": diagnostics.exact_duplicate_count,
        "ignored_property_name_count": diagnostics.ignored_property_name_count,
        "ignored_property_name_digest": diagnostics.ignored_property_name_digest,
        "canonical_coordinate_multiset_sha256": diagnostics.canonical_coordinate_multiset_sha256,
    }
    digest_aliases = [
        expected[key]
        for key in ("canonical_coordinate_multiset_digest", "canonical_coordinate_multiset_sha256")
        if key in expected
    ]
    if not digest_aliases or any(value != diagnostics.canonical_coordinate_multiset_sha256 for value in digest_aliases):
        raise SemanticStop("SOURCE_DIAGNOSTIC_MISMATCH:canonical_coordinate_multiset_digest")
    for key, observed in pairs.items():
        if key == "canonical_coordinate_multiset_sha256":
            continue
        if observed != expected.get(key):
            raise SemanticStop(f"SOURCE_DIAGNOSTIC_MISMATCH:{key}")


def _deduplicate(parsed: ParsedPly, *, reverse_input: bool) -> SampledPoints:
    xyz = parsed.xyz[::-1] if reverse_input else parsed.xyz
    packed = np.ascontiguousarray(xyz).view("V24").reshape(len(xyz))
    _, first, counts = np.unique(packed, return_index=True, return_counts=True)
    if np.any(counts > np.iinfo(np.uint32).max):
        raise SemanticStop("DUPLICATE_MULTIPLICITY_U32_OVERFLOW")
    unique_xyz = np.ascontiguousarray(xyz[first], dtype="<f8")
    multiplicities = np.ascontiguousarray(counts, dtype=np.uint32)
    if len(xyz) - len(unique_xyz) != parsed.diagnostics.exact_duplicate_count:
        raise SemanticStop("DUPLICATE_CENSUS_DISAGREEMENT")
    return canonical_sample(unique_xyz, multiplicities, sample_cap=200_000)


def _metadata(
    *,
    raw_sha: str,
    code_sha: str,
    dependency_sha: str,
    census: Mapping[str, int],
    resource: Mapping[str, int | bool],
) -> dict[str, Any]:
    if any(_HEX64.fullmatch(value) is None for value in (raw_sha, code_sha, dependency_sha)):
        raise SemanticStop("INVALID_HASH_BINDING")
    return {
        "schema_version": 1,
        "raw_source_sha256": raw_sha,
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "code_sha256": code_sha,
        "dependency_lock_sha256": dependency_sha,
        "source_points_serialized": False,
        "census": dict(census),
        "resource": dict(resource),
        "attestation": {
            "semantic_core_imported_by_independent_validator": False,
            "filename_path_family_identity_semantics_used": False,
            "gold_truth_cad_render_benchmark_accessed": False,
            "source_points_serialized": False,
            "top_k_reservoir_tail_omission_or_clipping_used": False,
            "complete_baseline_and_isotropic_payload": True,
            "deterministic_rebuild_match": True,
            "input_order_match": True,
            "independent_validator_match": True,
            "all_truncated_digest_collisions_checked": True,
        },
    }


def execute_semantic_role(
    fd: int,
    *,
    expected_source: Mapping[str, Any],
    native_worker: object,
    code_sha256: str,
    dependency_lock_sha256: str,
    role: str,
    resource: Mapping[str, int | bool] | None = None,
) -> RoleResult:
    if role not in {"BASE_A", "BASE_B", "INPUT_ORDER_PERMUTED"}:
        raise SemanticStop("SEMANTIC_ROLE_INVALID")
    # These source-free gates are deliberately before the first parser read.
    validate_v7_known_answer_vectors()
    native_worker.validate_known_answers()
    validate_frozen_allocation_plan(load_frozen_v7())
    parsed = parse_ply_fd(fd)
    verify_source_diagnostics(parsed, expected_source)
    sampled = _deduplicate(parsed, reverse_input=role == "INPUT_ORDER_PERMUTED")
    case_digest = case_token_digest(bytes.fromhex(parsed.diagnostics.raw_ply_sha256))
    registry = CollisionRegistry()
    partition = build_partition(case_digest, sampled, registry=registry)
    fields = build_fields(case_digest, partition, native_worker, registry=registry)
    adjacency = build_space_scale_adjacency(
        case_digest, partition.leaf_mortons, registry=registry, native_worker=native_worker
    )
    o = len(partition.leaf_mortons)
    baseline_persistence, attributed = h0_persistence(
        case_digest, 0, fields.fields[: 3 * o], adjacency,
        registry=registry, native_worker=native_worker,
    )
    isotropic_persistence, isotropic_survivors = h0_persistence(
        case_digest, 1, fields.fields[3 * o :], adjacency,
        registry=registry, native_worker=native_worker,
    )
    if attributed < 1:
        raise SemanticStop("EMPTY_BASELINE_ATTRIBUTED_INTERVAL_DOMAIN")
    if isotropic_survivors != 0:
        raise SemanticStop("ISOTROPIC_ATTRIBUTED_SURVIVOR")
    baseline_positive = int(np.count_nonzero(fields.fields[: 3 * o]["anisotropy_q_u16"] > 0))
    if 4 * baseline_positive > 3 * (3 * o):
        raise SemanticStop("BASELINE_ACTIVE_FIELD_FRACTION_SATURATED")
    if 4 * attributed > 3 * o:
        raise SemanticStop("BASELINE_ATTRIBUTED_INTERVAL_FRACTION_SATURATED")

    persistence = np.concatenate((baseline_persistence, isotropic_persistence))
    native_worker.validate_persistence_collision_domain(
        case_digest, fields.fields, persistence
    )
    census = {
        "raw_records": int(parsed.diagnostics.parsed_vertex_count),
        "finite_records": int(parsed.diagnostics.finite_coordinate_count),
        "unique_records": int(sampled.unique_record_count),
        "sampled_unique_records": int(len(sampled.xyz)),
        "occupied_cells_O": o,
        "partition_rows": int(len(partition.partition_nodes)),
        "field_rows": int(len(fields.fields)),
        "adjacency_rows": int(len(adjacency)),
        "persistence_rows": int(len(persistence)),
        "rejection_rows": int(len(fields.rejections)),
        "moment_valid_rows": int(fields.total_valid_rows),
        "attributed_intervals": int(attributed),
        "isotropic_survivors": int(isotropic_survivors),
    }
    resource_values: dict[str, int | bool] = {
        "cpu_nanoseconds": 0,
        "wall_nanoseconds": 0,
        "peak_process_tree_rss_bytes": 0,
        "scratch_bytes": 0,
        "watchdog_poll_milliseconds": 10,
        "resource_cap_tripped": False,
    }
    if resource is not None:
        resource_values.update(resource)
    tables = {
        "partition_node": partition.partition_nodes,
        "leaf_aggregate": partition.leaf_aggregate,
        "cell_scale_field": fields.fields,
        "space_scale_adjacency": adjacency,
        "persistence_disposition": persistence,
        "rejection": fields.rejections,
    }
    container_bytes = encode_container(
        _metadata(
            raw_sha=parsed.diagnostics.raw_ply_sha256,
            code_sha=code_sha256,
            dependency_sha=dependency_lock_sha256,
            census=census,
            resource=resource_values,
        ),
        tables,
    )
    from .container import decode_container

    decoded = decode_container(
        container_bytes,
        expected_code_sha256=code_sha256,
        expected_dependency_lock_sha256=dependency_lock_sha256,
        materialize_rows=False,
    )
    return RoleResult(container_bytes, decoded.table_bytes, census)


__all__ = ["RoleResult", "SemanticStop", "execute_semantic_role", "verify_source_diagnostics"]
