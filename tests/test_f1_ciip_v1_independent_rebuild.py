from __future__ import annotations

import hashlib

import numpy as np
import pytest

from research.f1_ciip_v1 import container
from research.f1_ciip_v1.fields import build_fields
from research.f1_ciip_v1.graph import build_space_scale_adjacency, h0_persistence
from research.f1_ciip_v1.identifiers import CollisionRegistry, case_token_digest
from research.f1_ciip_v1.independent_semantic_rebuild import (
    compute_independent_known_answer_vectors,
    independent_work_upper_bound,
    rebuild_semantic_tables,
    validate_independent_known_answer_vectors,
)
from research.f1_ciip_v1.independent_validator import (
    IndependentValidationError,
    ParsedSource,
    SourceDiagnostics,
    rebuild_and_validate_container_bytes,
)
from research.f1_ciip_v1.native_worker import NativeWorker, build_native_worker
from research.f1_ciip_v1.partition import build_partition, canonical_sample


RAW_SHA = "11" * 32
CODE_SHA = "22" * 32
DEPENDENCY_SHA = "33" * 32


def test_independent_identifier_and_commitment_known_answers_gate() -> None:
    observed = compute_independent_known_answer_vectors()
    assert observed["morton_u64"] == 53
    assert observed["partition_commitment_sha256"] == (
        "7327b3d33259deef14deda7142f2951470e80606704691de49bedbe41dd84d13"
    )
    assert observed["table_directory_sha256"] == (
        "43055b40d5844fc87cdbd94002f007e9e92f87b8b777dd76d0c19bc1d9510509"
    )
    validate_independent_known_answer_vectors()


def _synthetic_xyz() -> np.ndarray:
    # The centered 9^3 block has enough complete supports to exercise all seven
    # field stages; two extrema freeze a one-unit depth-five cell side.
    rows = [
        (x + 0.1, y + 0.1, z + 0.1)
        for x in range(8, 17)
        for y in range(8, 17)
        for z in range(8, 17)
    ]
    rows.extend(((0.0, 0.0, 0.0), (24.0, 24.0, 24.0)))
    return np.asarray(rows, dtype="<f8")


@pytest.fixture(scope="module")
def exact_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, dict[str, np.ndarray], bytes, ParsedSource]:
    xyz = _synthetic_xyz()
    worker = NativeWorker(build_native_worker(tmp_path_factory.mktemp("independent-native")))
    case = case_token_digest(bytes.fromhex(RAW_SHA))
    registry = CollisionRegistry()
    sampled = canonical_sample(xyz, np.ones(len(xyz), dtype=np.uint32))
    partition = build_partition(case, sampled, registry=registry)
    fields = build_fields(case, partition, worker, registry=registry)
    adjacency = build_space_scale_adjacency(case, partition.leaf_mortons, registry=registry)
    o = len(partition.leaf_mortons)
    baseline, _ = h0_persistence(case, 0, fields.fields[: 3 * o], adjacency, registry=registry)
    isotropic, _ = h0_persistence(case, 1, fields.fields[3 * o :], adjacency, registry=registry)
    persistence = np.concatenate((baseline, isotropic))
    tables = {
        "partition_node": partition.partition_nodes,
        "leaf_aggregate": partition.leaf_aggregate,
        "cell_scale_field": fields.fields,
        "space_scale_adjacency": adjacency,
        "persistence_disposition": persistence,
        "rejection": fields.rejections,
    }
    census = {
        "raw_records": len(xyz), "finite_records": len(xyz), "unique_records": len(xyz),
        "sampled_unique_records": len(xyz), "occupied_cells_O": o,
        "partition_rows": len(partition.partition_nodes), "field_rows": len(fields.fields),
        "adjacency_rows": len(adjacency), "persistence_rows": len(persistence),
        "rejection_rows": len(fields.rejections),
        "moment_valid_rows": int(np.count_nonzero(fields.fields["status_u16"] == 0)),
        "attributed_intervals": int(np.count_nonzero(baseline["reason_u16"] == 0)),
        "isotropic_survivors": int(np.count_nonzero(isotropic["reason_u16"] == 0)),
    }
    metadata = {
        "schema_version": 1, "raw_source_sha256": RAW_SHA,
        "configuration_sha256": container.EXPECTED_CONFIGURATION_SHA256,
        "schema_sha256": container.EXPECTED_SCHEMA_SHA256,
        "code_sha256": CODE_SHA, "dependency_lock_sha256": DEPENDENCY_SHA,
        "source_points_serialized": False, "census": census,
        "resource": {
            "cpu_nanoseconds": 0, "wall_nanoseconds": 0,
            "peak_process_tree_rss_bytes": 0, "scratch_bytes": 0,
            "watchdog_poll_milliseconds": 10, "resource_cap_tripped": False,
        },
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
    encoded = container.encode_container(metadata, tables)
    diagnostics = SourceDiagnostics(
        raw_byte_count=1,
        raw_ply_sha256=RAW_SHA,
        ply_header_byte_count=1,
        ply_header_sha256=hashlib.sha256(b"x").hexdigest(),
        ply_format="binary_little_endian",
        parsed_vertex_count=len(xyz),
        finite_coordinate_count=len(xyz),
        exact_duplicate_count=0,
        ignored_property_name_count=0,
        ignored_property_name_digest=hashlib.sha256(b"[]").hexdigest(),
        canonical_coordinate_multiset_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
    )
    return xyz, tables, encoded, ParsedSource(diagnostics, xyz)


def test_independent_rebuild_matches_all_six_production_tables_byte_exactly(
    exact_fixture: tuple[np.ndarray, dict[str, np.ndarray], bytes, ParsedSource],
) -> None:
    xyz, production, _encoded, _source = exact_fixture
    rebuilt = rebuild_semantic_tables(xyz[::-1], RAW_SHA)
    assert tuple(rebuilt.table_bytes) == container.TABLE_NAMES
    for name in container.TABLE_NAMES:
        assert rebuilt.table_bytes[name] == production[name].tobytes(order="C"), name


def test_runner_validator_requires_independently_rebuilt_exact_table_bytes(
    exact_fixture: tuple[np.ndarray, dict[str, np.ndarray], bytes, ParsedSource],
) -> None:
    _xyz, production, encoded, source = exact_fixture
    summary = rebuild_and_validate_container_bytes(
        encoded,
        source,
        expected_code_sha256=CODE_SHA,
        expected_dependency_lock_sha256=DEPENDENCY_SHA,
    )
    assert summary.occupied_cells == len(production["leaf_aggregate"])

    changed = {name: rows.copy() for name, rows in production.items()}
    changed["leaf_aggregate"]["membership_digest_128"][0] = np.void(bytes.fromhex("ff" * 16))
    decoded = container.decode_container(encoded, materialize_rows=False)
    tampered = container.encode_container(
        {
            key: value
            for key, value in decoded.metadata.items()
            if key not in {"case_token_digest_sha256", "partition_commitment_sha256", "table_directory_sha256", "table_descriptors"}
        },
        changed,
    )
    with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH: leaf_aggregate"):
        rebuild_and_validate_container_bytes(
            tampered,
            source,
            expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


def test_independent_max_o_static_work_bound_is_below_frozen_role_envelope() -> None:
    bound = independent_work_upper_bound(32_768)
    assert bound.support_candidate_checks == 81_199_104
    assert bound.connectivity_neighbor_checks == 81_199_104
    assert bound.jacobi_pivots == 3_538_944
    assert bound.h0_events == 917_504
    assert sum((
        bound.support_candidate_checks,
        bound.connectivity_neighbor_checks,
        bound.jacobi_pivots,
        bound.h0_events,
    )) < 200_000_000
