from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from research.f1_ciip_v1 import container
from research.f1_ciip_v1 import independent_validator


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_f1_ciip_v1_smoke.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "ciip_serialized_bytes_contract_tensor_runner", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)

RAW_SHA = "aa" * 32
CODE_SHA = "11" * 32
DEPENDENCY_SHA = "22" * 32
SERIALIZED_BYTES_SENTINEL = object()


def _container_bytes(raw_sha: str) -> bytes:
    """Build the compact source-free six-table accounting fixture locally."""

    case = container.case_token_digest(bytes.fromhex(raw_sha))
    morton = 5
    partition = [
        {
            "node_id_sha256": container.node_id(case, 5, morton),
            "morton_prefix_u64": morton,
            "subtree_membership_digest_sha256": hashlib.sha256(b"subtree").digest(),
            "subtree_point_count_u32": 1,
            "occupied_leaf_count_u32": 1,
            "depth_u8": 5,
            "child_mask_u8": 0,
            "status_u16": 0,
            "reserved_zero_u32": 0,
        }
    ]
    leaf = [
        {
            "cell_id_sha256": container.cell_id(case, morton),
            "morton_u64": morton,
            "membership_digest_128": hashlib.sha256(b"leaf").digest()[:16],
            "unique_point_count_u32": 1,
            "raw_record_multiplicity_sum_u32": 1,
        }
    ]
    fields = []
    for arm in (0, 1):
        for scale in range(3):
            fields.append(
                {
                    "row_id_digest_128": container.field_row_id(case, arm, scale, morton),
                    "support_membership_digest_128": hashlib.sha256(
                        f"support-{scale}".encode()
                    ).digest()[:16],
                    "morton_u64": morton,
                    "trace_q_i64": 3,
                    "lambda1_q_i64": 2 if arm == 0 else 3,
                    "lambda2_q_i64": 1 if arm == 0 else 3,
                    "lambda3_q_i64": 0 if arm == 0 else 3,
                    "mass_q_i64": 1,
                    "arm_u8": arm,
                    "scale_u8": scale,
                    "status_u16": 0,
                    "support_count_u32": 1,
                    "raw_count_p10_u32": 1,
                    "raw_count_p90_u32": 1,
                    "balance_numerator_u32": 1,
                    "balance_denominator_u32": 1,
                    "growth12_gate_code_i32": 0,
                    "growth23_gate_code_i32": 0,
                    "anisotropy_q_u16": 1 if arm == 0 else 0,
                    "filtration_u16": 65_533 if arm == 0 else 65_534,
                    "stage_mask_u32": 0x7F,
                    "first_rejection_u16": 0,
                    "flags_u16": 0,
                    "eigenvalue_denominator_u32": 1 if arm == 0 else 3,
                }
            )
    adjacency = [
        {
            "edge_id_digest_128": container.edge_id(case, lower, upper, 1, 255),
            "lower_vertex_u32": lower,
            "upper_vertex_u32": upper,
            "edge_kind_u8": 1,
            "axis_u8": 255,
            "status_u16": 0,
            "reserved_zero_u32": 0,
        }
        for lower, upper in ((0, 1), (1, 2))
    ]
    persistence = []
    rejection = []
    for birth_vertex, field in enumerate(fields):
        arm = field["arm_u8"]
        birth = field["row_id_digest_128"]
        death = container.essential_death_row_id(case, arm, birth)
        persistence.append(
            {
                "interval_id_digest_128": container.interval_id(
                    case,
                    arm,
                    birth,
                    death,
                    field["filtration_u16"],
                    65_535,
                ),
                "birth_row_id_digest_128": birth,
                "death_row_id_digest_128": death,
                "arm_u8": arm,
                "status_u8": 1,
                "reason_u16": 0 if arm == 0 else 1,
                "birth_f_u16": field["filtration_u16"],
                "death_f_or_essential_u16": 65_535,
                "lifetime_u16": 65_535 - field["filtration_u16"],
                "scale_mask_u8": 1 << field["scale_u8"],
                "spatial_face_edge_count_u32": 0,
                "flags_u8": 0,
                "birth_vertex_index_u32": birth_vertex % 3,
            }
        )
        rejection.append(
            {
                "row_id_digest_128": birth,
                "arm_u8": arm,
                "scale_u8": field["scale_u8"],
                "first_rejection_u16": 0,
                "stage_mask_u32": 0x7F,
                "denominator_id_u16": 0,
                "reserved_zero_bytes6": bytes(6),
                "morton_u64": morton,
            }
        )
    tables = {
        "partition_node": partition,
        "leaf_aggregate": leaf,
        "cell_scale_field": fields,
        "space_scale_adjacency": adjacency,
        "persistence_disposition": persistence,
        "rejection": rejection,
    }
    metadata = {
        "schema_version": 1,
        "raw_source_sha256": raw_sha,
        "configuration_sha256": container.EXPECTED_CONFIGURATION_SHA256,
        "schema_sha256": container.EXPECTED_SCHEMA_SHA256,
        "code_sha256": CODE_SHA,
        "dependency_lock_sha256": DEPENDENCY_SHA,
        "source_points_serialized": False,
        "census": {
            "raw_records": 1,
            "finite_records": 1,
            "unique_records": 1,
            "sampled_unique_records": 1,
            "occupied_cells_O": 1,
            "partition_rows": 1,
            "field_rows": 6,
            "adjacency_rows": 2,
            "persistence_rows": 6,
            "rejection_rows": 6,
            "moment_valid_rows": 6,
            "attributed_intervals": 3,
            "isotropic_survivors": 0,
        },
        "resource": {
            "cpu_nanoseconds": 0,
            "wall_nanoseconds": 0,
            "peak_process_tree_rss_bytes": 0,
            "scratch_bytes": 0,
            "watchdog_poll_milliseconds": 10,
            "resource_cap_tripped": False,
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
    return container.encode_container(
        metadata,
        tables,
        order_key_hooks={
            "persistence_disposition": lambda row: (
                row["arm_u8"],
                row["birth_vertex_index_u32"],
            ),
            "rejection": lambda row: (
                row["arm_u8"],
                row["scale_u8"],
                row["morton_u64"],
            ),
        },
    )


def _valid_container() -> bytes:
    return _container_bytes(RAW_SHA)


def _reencoding_inputs(encoded: bytes) -> tuple[dict, dict]:
    decoded = container.decode_container(encoded, materialize_rows=False)
    metadata = copy.deepcopy(decoded.metadata)
    for computed in (
        "case_token_digest_sha256",
        "partition_commitment_sha256",
        "table_directory_sha256",
        "table_descriptors",
    ):
        metadata.pop(computed)
    tables = {name: rows.copy() for name, rows in decoded.rows.items()}
    return metadata, tables


def _replace_metadata(encoded: bytes, mutate: object) -> bytes:
    decoded = container.decode_container(encoded, materialize_rows=False)
    metadata = copy.deepcopy(decoded.metadata)
    mutate(metadata)  # type: ignore[operator]
    return container.encode_metadata_region(metadata) + encoded[container.METADATA_REGION_BYTES :]


def _source_diagnostics() -> independent_validator.SourceDiagnostics:
    return independent_validator.SourceDiagnostics(
        raw_byte_count=1,
        raw_ply_sha256=RAW_SHA,
        ply_header_byte_count=1,
        ply_header_sha256="bb" * 32,
        ply_format="binary_little_endian",
        parsed_vertex_count=1,
        finite_coordinate_count=1,
        exact_duplicate_count=0,
        ignored_property_name_count=0,
        ignored_property_name_digest="cc" * 32,
        canonical_coordinate_multiset_sha256="dd" * 32,
    )


def _assert_exact_length_and_commitments(encoded: bytes) -> container.DecodedContainer:
    decoded = container.decode_container(encoded, materialize_rows=False)
    metadata = decoded.metadata
    descriptors = metadata["table_descriptors"]
    table_byte_total = sum(len(decoded.table_bytes[name]) for name in container.TABLE_NAMES)
    descriptor_byte_total = sum(descriptor["byte_length_u64"] for descriptor in descriptors)
    emitted_total = metadata["resource"]["serialized_bytes"]

    assert emitted_total == len(encoded)
    assert emitted_total == container.METADATA_REGION_BYTES + table_byte_total
    assert emitted_total == container.METADATA_REGION_BYTES + descriptor_byte_total

    for name, descriptor in zip(container.TABLE_NAMES, descriptors, strict=True):
        payload = decoded.table_bytes[name]
        assert descriptor["byte_length_u64"] == len(payload)
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["partition_commitment_sha256"] == container.partition_commitment(
        descriptors[0]["row_count_u64"], decoded.table_bytes["partition_node"]
    ).hex()
    assert metadata["table_directory_sha256"] == container.table_directory_digest(
        descriptors
    ).hex()
    assert metadata["case_token_digest_sha256"] == container.case_token_digest(
        bytes.fromhex(metadata["raw_source_sha256"])
    ).hex()
    return decoded


def test_serialized_bytes_omission_is_filled_with_exact_emitted_length() -> None:
    encoded = _valid_container()
    reference = _assert_exact_length_and_commitments(encoded)
    metadata, tables = _reencoding_inputs(encoded)
    metadata["resource"].pop("serialized_bytes")
    rebuilt = container.encode_container(metadata, tables)
    decoded = _assert_exact_length_and_commitments(rebuilt)

    assert rebuilt == encoded
    assert decoded.table_bytes == reference.table_bytes
    for name, current, expected in zip(
        container.TABLE_NAMES,
        decoded.metadata["table_descriptors"],
        reference.metadata["table_descriptors"],
        strict=True,
    ):
        assert current["sha256"] == expected["sha256"], name
    for commitment in (
        "partition_commitment_sha256",
        "table_directory_sha256",
        "case_token_digest_sha256",
    ):
        assert decoded.metadata[commitment] == reference.metadata[commitment]


def test_exact_serialized_bytes_preclaim_preserves_byte_identity() -> None:
    encoded = _valid_container()
    metadata, tables = _reencoding_inputs(encoded)
    metadata["resource"]["serialized_bytes"] = len(encoded)
    assert container.encode_container(metadata, tables) == encoded


def _invalid_preclaim(case: str, total: int) -> object:
    values = {
        "zero": 0,
        "total_minus_one": total - 1,
        "total_plus_one": total + 1,
        "unrelated_positive": 37,
        "configured_cap": container.MAX_SERIALIZED_BYTES,
        "bool": True,
        "negative": -1,
        "u64_overflow": 1 << 64,
        "float": float(total),
        "string": str(total),
        "none": None,
        "list_container": [total],
        "tuple_container": (total,),
        "dict_object": {"serialized_bytes": total},
        "sentinel_object": SERIALIZED_BYTES_SENTINEL,
    }
    if case == "configured_cap":
        assert container.MAX_SERIALIZED_BYTES != total
    return values[case]


@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "total_minus_one",
        "total_plus_one",
        "unrelated_positive",
        "configured_cap",
        "bool",
        "negative",
        "u64_overflow",
        "float",
        "string",
        "none",
        "list_container",
        "tuple_container",
        "dict_object",
        "sentinel_object",
    ],
)
def test_all_wrong_value_type_container_and_sentinel_preclaims_are_rejected(case: str) -> None:
    encoded = _valid_container()
    metadata, tables = _reencoding_inputs(encoded)
    metadata["resource"]["serialized_bytes"] = _invalid_preclaim(case, len(encoded))
    with pytest.raises(container.ContainerValidationError):
        container.encode_container(metadata, tables)


@pytest.mark.parametrize(
    "case",
    ["float", "bool", "negative", "u64_overflow", "list_container", "dict_object"],
)
def test_optional_serialized_bytes_preclaim_requires_non_boolean_u64(case: str) -> None:
    encoded = _valid_container()
    metadata, tables = _reencoding_inputs(encoded)
    metadata["resource"]["serialized_bytes"] = _invalid_preclaim(case, len(encoded))
    with pytest.raises(
        container.ContainerValidationError,
        match=r"resource\.serialized_bytes must be a u64 JSON integer",
    ):
        container.encode_container(metadata, tables)


def test_decoder_requires_the_emitted_serialized_bytes_key() -> None:
    encoded = _valid_container()
    missing = _replace_metadata(
        encoded, lambda metadata: metadata["resource"].pop("serialized_bytes")
    )
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(missing, materialize_rows=False)


@pytest.mark.parametrize("claim", ["zero", "total_minus_one", "total_plus_one"])
def test_decoder_and_independent_validator_reject_all_length_tampers(claim: str) -> None:
    encoded = _valid_container()
    tampered = _replace_metadata(
        encoded,
        lambda metadata: metadata["resource"].__setitem__(
            "serialized_bytes", _invalid_preclaim(claim, len(encoded))
        ),
    )
    with pytest.raises(container.ContainerValidationError, match="serialized-byte"):
        container.decode_container(tampered, materialize_rows=False)
    with pytest.raises(container.ContainerValidationError, match="serialized-byte"):
        independent_validator.validate_container_bytes(
            tampered,
            _source_diagnostics(),
            expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


@pytest.mark.parametrize("descriptor_tamper", ["noncontiguous_offset", "table_length_mismatch"])
def test_exact_serialized_claim_cannot_mask_descriptor_or_table_length_mismatch(
    descriptor_tamper: str,
) -> None:
    encoded = _valid_container()

    def mutate(metadata: dict) -> None:
        metadata["resource"]["serialized_bytes"] = len(encoded)
        descriptor = metadata["table_descriptors"][1]
        if descriptor_tamper == "noncontiguous_offset":
            descriptor["offset_u64"] += 1
        else:
            descriptor["byte_length_u64"] += 1

    masked = _replace_metadata(encoded, mutate)
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(masked, materialize_rows=False)
    with pytest.raises(container.ContainerValidationError):
        independent_validator.validate_container_bytes(
            masked,
            _source_diagnostics(),
            expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


@pytest.mark.parametrize("mode", ["truncated", "trailing"])
def test_claimed_length_cannot_mask_truncation_trailing_or_descriptor_failure(mode: str) -> None:
    encoded = _valid_container()
    claimed_length = len(encoded) - 1 if mode == "truncated" else len(encoded) + 1
    masked = _replace_metadata(
        encoded,
        lambda metadata: metadata["resource"].__setitem__("serialized_bytes", claimed_length),
    )
    masked = masked[:-1] if mode == "truncated" else masked + b"\x00"
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(masked, materialize_rows=False)


def test_serialized_cap_arithmetic_uses_exact_bounded_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _valid_container()
    metadata, tables = _reencoding_inputs(encoded)
    metadata["resource"].pop("serialized_bytes")
    monkeypatch.setattr(container, "MAX_SERIALIZED_BYTES", len(encoded))
    assert len(container.encode_container(metadata, tables)) == len(encoded)
    monkeypatch.setattr(container, "MAX_SERIALIZED_BYTES", len(encoded) - 1)
    with pytest.raises(container.ContainerValidationError, match="exceeds"):
        container.encode_container(metadata, tables)


def test_supervisor_resource_patch_preserves_payload_and_exact_length(
    tmp_path: Path,
) -> None:
    output = tmp_path / "container.bin"
    output.write_bytes(_valid_container())
    before = container.decode_container(output.read_bytes(), materialize_rows=False)
    runner._patch_resource_metadata(
        output,
        {
            "cpu_nanoseconds": 7,
            "wall_nanoseconds": 11,
            "peak_process_tree_rss_bytes": 13,
            "scratch_bytes": 17,
        },
    )
    raw = output.read_bytes()
    after = _assert_exact_length_and_commitments(raw)
    assert after.table_bytes == before.table_bytes
    assert raw[container.METADATA_REGION_BYTES :] == _valid_container()[
        container.METADATA_REGION_BYTES :
    ]
    for name, current, expected in zip(
        container.TABLE_NAMES,
        after.metadata["table_descriptors"],
        before.metadata["table_descriptors"],
        strict=True,
    ):
        assert current["sha256"] == expected["sha256"], name
    for commitment in (
        "partition_commitment_sha256",
        "table_directory_sha256",
        "case_token_digest_sha256",
    ):
        assert after.metadata[commitment] == before.metadata[commitment]


def test_supervisor_resource_patch_rejects_serialized_bytes_change(tmp_path: Path) -> None:
    output = tmp_path / "container.bin"
    original = _valid_container()
    output.write_bytes(original)
    with pytest.raises(RuntimeError, match="exact supervisor-owned set"):
        runner._patch_resource_metadata(
            output,
            {"serialized_bytes": len(original) + 1},
        )
    assert output.read_bytes() == original


@pytest.mark.parametrize("claim", ["zero", "total_minus_one", "total_plus_one"])
def test_independent_resource_accounting_uses_actual_container_length(claim: str) -> None:
    encoded = _valid_container()
    metadata = container.decode_container(encoded, materialize_rows=False).metadata
    independent_validator._validate_resources(metadata, len(encoded))
    tampered = copy.deepcopy(metadata)
    tampered["resource"]["serialized_bytes"] = _invalid_preclaim(claim, len(encoded))
    with pytest.raises(
        independent_validator.IndependentValidationError,
        match="SERIALIZED_BYTE_ACCOUNTING_MISMATCH",
    ):
        independent_validator._validate_resources(tampered, len(encoded))
