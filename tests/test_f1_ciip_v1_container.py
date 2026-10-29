from __future__ import annotations

import copy
import hashlib
import json
import struct

import pytest

from research.f1_ciip_v1 import container


def _digest(label: str, size: int) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()[:size]


def _fixture() -> tuple[dict, dict[str, list[dict]]]:
    raw_sha = hashlib.sha256(b"frozen-v7-container-test").digest()
    case = container.case_token_digest(raw_sha)
    morton = 5

    partition = [{
        "node_id_sha256": container.node_id(case, 5, morton),
        "morton_prefix_u64": morton,
        "subtree_membership_digest_sha256": _digest("subtree", 32),
        "subtree_point_count_u32": 1,
        "occupied_leaf_count_u32": 1,
        "depth_u8": 5,
        "child_mask_u8": 0,
        "status_u16": 0,
        "reserved_zero_u32": 0,
    }]
    leaf = [{
        "cell_id_sha256": container.cell_id(case, morton),
        "morton_u64": morton,
        "membership_digest_128": _digest("leaf-members", 16),
        "unique_point_count_u32": 1,
        "raw_record_multiplicity_sum_u32": 1,
    }]

    fields: list[dict] = []
    for arm in (0, 1):
        for scale in (0, 1, 2):
            q = 1 if arm == 0 else 0
            fields.append({
                "row_id_digest_128": container.field_row_id(case, arm, scale, morton),
                "support_membership_digest_128": _digest(f"support-{scale}", 16),
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
                "anisotropy_q_u16": q,
                "filtration_u16": 65534 - q,
                "stage_mask_u32": 0x7F,
                "first_rejection_u16": 0,
                "flags_u16": 0,
                "eigenvalue_denominator_u32": 1 if arm == 0 else 3,
            })

    adjacency: list[dict] = []
    for lower, upper in ((0, 1), (1, 2)):
        adjacency.append({
            "edge_id_digest_128": container.edge_id(case, lower, upper, 1, 255),
            "lower_vertex_u32": lower,
            "upper_vertex_u32": upper,
            "edge_kind_u8": 1,
            "axis_u8": 255,
            "status_u16": 0,
            "reserved_zero_u32": 0,
        })

    persistence: list[dict] = []
    rejection: list[dict] = []
    for birth_vertex, field in enumerate(fields):
        arm = field["arm_u8"]
        birth_id = field["row_id_digest_128"]
        birth_f = field["filtration_u16"]
        death_id = container.essential_death_row_id(case, arm, birth_id)
        persistence.append({
            "interval_id_digest_128": container.interval_id(case, arm, birth_id, death_id, birth_f, 65535),
            "birth_row_id_digest_128": birth_id,
            "death_row_id_digest_128": death_id,
            "arm_u8": arm,
            "status_u8": 1,
            "reason_u16": 0 if arm == 0 else 1,
            "birth_f_u16": birth_f,
            "death_f_or_essential_u16": 65535,
            "lifetime_u16": 65535 - birth_f,
            "scale_mask_u8": 1 << field["scale_u8"],
            "spatial_face_edge_count_u32": 0,
            "flags_u8": 0,
            "birth_vertex_index_u32": birth_vertex % 3,
        })
        rejection.append({
            "row_id_digest_128": birth_id,
            "arm_u8": arm,
            "scale_u8": field["scale_u8"],
            "first_rejection_u16": 0,
            "stage_mask_u32": 0x7F,
            "denominator_id_u16": 0,
            "reserved_zero_bytes6": bytes(6),
            "morton_u64": morton,
        })

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
        "raw_source_sha256": raw_sha.hex(),
        "configuration_sha256": container.EXPECTED_CONFIGURATION_SHA256,
        "schema_sha256": container.EXPECTED_SCHEMA_SHA256,
        "code_sha256": "11" * 32,
        "dependency_lock_sha256": "22" * 32,
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
            "attributed_intervals": 1,
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
    return metadata, tables


def _encode(metadata: dict | None = None, tables: dict | None = None) -> bytes:
    if metadata is None or tables is None:
        fixture_metadata, fixture_tables = _fixture()
        metadata = fixture_metadata if metadata is None else metadata
        tables = fixture_tables if tables is None else tables
    return container.encode_container(
        metadata,
        tables,
        order_key_hooks={
            "persistence_disposition": lambda row: (row["arm_u8"], row["birth_vertex_index_u32"]),
            "rejection": lambda row: (row["arm_u8"], row["scale_u8"], row["morton_u64"]),
        },
    )


def test_frozen_row_structs_and_numpy_dtypes_have_exact_widths() -> None:
    expected = {
        "partition_node": 88,
        "leaf_aggregate": 64,
        "cell_scale_field": 128,
        "space_scale_adjacency": 32,
        "persistence_disposition": 64,
        "rejection": 32,
    }
    assert {name: spec.struct.size for name, spec in container.TABLE_SPECS.items()} == expected
    assert {name: container.numpy_dtype(name).itemsize for name in container.TABLE_NAMES} == expected


def test_six_descriptor_binary_directory_matches_frozen_known_answer() -> None:
    empty_sha = hashlib.sha256(b"").hexdigest()
    descriptors = [
        {
            "name_enum": spec.name_enum,
            "row_count_u64": 0,
            "row_size_u32": spec.row_size,
            "offset_u64": container.METADATA_REGION_BYTES,
            "byte_length_u64": 0,
            "sha256": empty_sha,
        }
        for spec in container.TABLE_SPECS.values()
    ]
    assert container.table_directory_digest(descriptors).hex() == (
        "43055b40d5844fc87cdbd94002f007e9e92f87b8b777dd76d0c19bc1d9510509"
    )


def test_complete_container_round_trip_and_canonical_order() -> None:
    metadata, tables = _fixture()
    tables["cell_scale_field"] = list(reversed(tables["cell_scale_field"]))
    tables["space_scale_adjacency"] = list(reversed(tables["space_scale_adjacency"]))
    tables["persistence_disposition"] = list(reversed(tables["persistence_disposition"]))
    tables["rejection"] = list(reversed(tables["rejection"]))
    encoded = _encode(metadata, tables)
    decoded = container.decode_container(
        encoded,
        expected_code_sha256="11" * 32,
        expected_dependency_lock_sha256="22" * 32,
    )

    assert len(encoded) < container.MAX_SERIALIZED_BYTES
    assert decoded.metadata["resource"]["serialized_bytes"] == len(encoded)
    assert decoded.metadata["table_descriptors"][0]["offset_u64"] == container.METADATA_REGION_BYTES
    assert [row["arm_u8"] for row in decoded.rows["cell_scale_field"]] == [0, 0, 0, 1, 1, 1]
    assert [row["lower_vertex_u32"] for row in decoded.rows["space_scale_adjacency"]] == [0, 1]
    assert decoded.rows["persistence_disposition"][0]["birth_row_id_digest_128"] == decoded.rows["cell_scale_field"][0]["row_id_digest_128"]
    assert decoded.rows["rejection"][-1]["row_id_digest_128"] == decoded.rows["cell_scale_field"][-1]["row_id_digest_128"]


def test_numpy_structured_arrays_are_accepted_without_object_rows() -> None:
    import numpy as np

    metadata, rows_by_table = _fixture()
    arrays = {}
    for name, rows in rows_by_table.items():
        array = np.zeros(len(rows), dtype=container.numpy_dtype(name))
        for index, row in enumerate(rows):
            for field in container.TABLE_SPECS[name].fields:
                array[field.name][index] = row[field.name]
        arrays[name] = array
    encoded = container.encode_container(metadata, arrays)
    assert container.decode_container(encoded).metadata["census"]["occupied_cells_O"] == 1


def test_metadata_region_is_length_prefixed_canonical_and_zero_padded() -> None:
    encoded = _encode()
    length = struct.unpack_from("<Q", encoded, 0)[0]
    payload = encoded[8: 8 + length]
    metadata = json.loads(payload)
    assert payload == container.canonical_json_bytes(metadata)
    assert encoded[8 + length: container.METADATA_REGION_BYTES] == bytes(
        container.METADATA_REGION_BYTES - 8 - length
    )


@pytest.mark.parametrize("mutation", ["padding", "table", "trailing", "noncanonical_json"])
def test_fail_closed_byte_tamper(mutation: str) -> None:
    encoded = bytearray(_encode())
    if mutation == "padding":
        length = struct.unpack_from("<Q", encoded, 0)[0]
        encoded[8 + length] = 1
    elif mutation == "table":
        encoded[-1] ^= 1
    elif mutation == "trailing":
        encoded.extend(b"\x00")
    else:
        metadata = container.decode_metadata_region(bytes(encoded[:container.METADATA_REGION_BYTES]))
        payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
        encoded[:container.METADATA_REGION_BYTES] = (
            struct.pack("<Q", len(payload))
            + payload
            + bytes(container.MAX_METADATA_JSON_BYTES - len(payload))
        )
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(bytes(encoded))


def test_fail_closed_descriptor_and_metadata_schema_tamper() -> None:
    encoded = _encode()
    decoded = container.decode_container(encoded)
    metadata = copy.deepcopy(decoded.metadata)
    metadata["table_descriptors"][1]["offset_u64"] += 1
    tampered = container.encode_metadata_region(metadata) + encoded[container.METADATA_REGION_BYTES:]
    with pytest.raises(container.ContainerValidationError, match="offset"):
        container.decode_container(tampered)

    metadata = copy.deepcopy(decoded.metadata)
    metadata["unexpected"] = 1
    tampered = container.encode_metadata_region(metadata) + encoded[container.METADATA_REGION_BYTES:]
    with pytest.raises(container.ContainerValidationError, match="keys"):
        container.decode_container(tampered)


def test_fail_closed_wrong_row_width_reserved_bytes_and_metadata_capacity() -> None:
    _, tables = _fixture()
    bad = dict(tables["rejection"][0])
    bad["reserved_zero_bytes6"] = b"\x00" * 5
    with pytest.raises(container.ContainerValidationError, match="exactly 6 bytes"):
        container.pack_row("rejection", bad)

    bad = dict(tables["rejection"][0])
    bad["reserved_zero_bytes6"] = b"\x00" * 5 + b"\x01"
    with pytest.raises(container.ContainerValidationError, match="reserved"):
        container.pack_row("rejection", bad)

    with pytest.raises(container.ContainerValidationError, match="one-MiB"):
        container.encode_metadata_region({"oversized": "x" * container.MAX_METADATA_JSON_BYTES})


def test_fail_closed_signed_binding_mismatch() -> None:
    encoded = _encode()
    with pytest.raises(container.ContainerValidationError, match="code SHA-256"):
        container.decode_container(encoded, expected_code_sha256="33" * 32)


def test_hard_serialized_cap_is_enforced_before_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, tables = _fixture()
    monkeypatch.setattr(container, "MAX_SERIALIZED_BYTES", container.METADATA_REGION_BYTES)
    with pytest.raises(container.ContainerValidationError, match="64 MiB"):
        _encode(metadata, tables)
