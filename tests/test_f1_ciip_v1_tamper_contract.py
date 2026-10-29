"""Executable source-free coverage of all 18 frozen V7 tamper regressions."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from research.f1_ciip_v1 import container, identifiers
from research.f1_ciip_v1.execution_result import (
    ExecutionResultError,
    PASS_TOKEN,
    STOP_TOKEN,
    validate_execution_result,
)
from research.f1_ciip_v1.independent_semantic_rebuild import rebuild_semantic_tables
from research.f1_ciip_v1.independent_validator import (
    IndependentValidationError,
    rebuild_and_validate_container_bytes,
)
from tests.test_f1_ciip_v1_container import _encode as encode_small
from tests.test_f1_ciip_v1_container import _fixture as small_fixture
from tests.test_f1_ciip_v1_independent_rebuild import (
    CODE_SHA,
    DEPENDENCY_SHA,
    RAW_SHA,
    exact_fixture,
)


COMPUTED_METADATA = {
    "case_token_digest_sha256", "partition_commitment_sha256",
    "table_directory_sha256", "table_descriptors",
}


def _metadata_and_arrays(encoded: bytes, arrays: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    decoded = container.decode_container(encoded, materialize_rows=False)
    metadata = {key: copy.deepcopy(value) for key, value in decoded.metadata.items() if key not in COMPUTED_METADATA}
    return metadata, {name: value.copy() for name, value in arrays.items()}


def _reencode(encoded: bytes, arrays: dict[str, np.ndarray], mutate) -> bytes:
    metadata, changed = _metadata_and_arrays(encoded, arrays)
    mutate(metadata, changed)
    return container.encode_container(metadata, changed)


def _patch_metadata(encoded: bytes, mutate) -> bytes:
    metadata = container.decode_metadata_region(encoded[: container.METADATA_REGION_BYTES])
    mutate(metadata)
    return container.encode_metadata_region(metadata) + encoded[container.METADATA_REGION_BYTES :]


def _valid_result() -> dict:
    roles = []
    for role in ("BASE_A", "BASE_B", "INPUT_ORDER_PERMUTED", "INDEPENDENT_VALIDATOR"):
        roles.append({
            "role": role, "return_code": 0, "cpu_nanoseconds": 1,
            "wall_nanoseconds": 1, "peak_process_tree_rss_bytes": 1,
            "scratch_bytes": 1, "stdout_sha256": "11" * 32,
            "stderr_sha256": "22" * 32, "failure": None,
            "container_sha256": "33" * 32,
        })
    return {
        "outcome": PASS_TOKEN, "failure": None,
        "execution_lock_sha256": "44" * 32, "raw_ply_sha256": "55" * 32,
        "roles": roles, "rerun_authorized": False,
        "replacement_input_authorized": False, "gold_or_labels_accessed": False,
    }


def test_v7_tamper_01_source_config_schema_code_dependency_hash_mutation() -> None:
    encoded = encode_small()
    for key in ("raw_source_sha256", "configuration_sha256", "schema_sha256", "code_sha256", "dependency_lock_sha256"):
        tampered = _patch_metadata(encoded, lambda metadata, key=key: metadata.__setitem__(key, "ff" * 32))
        with pytest.raises(container.ContainerValidationError):
            container.decode_container(
                tampered, expected_code_sha256="11" * 32,
                expected_dependency_lock_sha256="22" * 32,
            )


def test_v7_tamper_02_missing_duplicate_reordered_appended_or_trailing_payload() -> None:
    metadata, tables = small_fixture()
    mutations = []
    missing = copy.deepcopy(tables); missing["cell_scale_field"] = missing["cell_scale_field"][:-1]; mutations.append(missing)
    duplicate = copy.deepcopy(tables); duplicate["leaf_aggregate"].append(copy.deepcopy(duplicate["leaf_aggregate"][0])); mutations.append(duplicate)
    appended = copy.deepcopy(tables); appended["rejection"].append(copy.deepcopy(appended["rejection"][-1])); mutations.append(appended)
    for changed in mutations:
        with pytest.raises(container.ContainerValidationError):
            encode_small(metadata, changed)
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(encode_small() + b"\x00")
    # Mutate already-canonical row bytes so the encoder cannot normalize the
    # attack back into order.
    reordered = bytearray(encode_small())
    decoded = container.decode_container(bytes(reordered), materialize_rows=False)
    descriptor = decoded.metadata["table_descriptors"][2]
    offset, width = descriptor["offset_u64"], descriptor["row_size_u32"]
    first, second = bytes(reordered[offset:offset+width]), bytes(reordered[offset+width:offset+2*width])
    reordered[offset:offset+width], reordered[offset+width:offset+2*width] = second, first
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(bytes(reordered))


def test_v7_tamper_03_forged_counts_offsets_lengths_row_sizes_or_container_length() -> None:
    encoded = encode_small()
    for field, delta in (("row_count_u64", 1), ("offset_u64", 1), ("byte_length_u64", 1), ("row_size_u32", 1)):
        tampered = _patch_metadata(
            encoded,
            lambda metadata, field=field, delta=delta: metadata["table_descriptors"][0].__setitem__(
                field, metadata["table_descriptors"][0][field] + delta
            ),
        )
        with pytest.raises(container.ContainerValidationError):
            container.decode_container(tampered)
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(encoded[:-1])


def test_v7_tamper_04_patricia_relation_subtree_digest_or_collision(
    exact_fixture,
) -> None:
    _xyz, arrays, encoded, source = exact_fixture
    for mutate in (
        lambda _m, a: a["partition_node"]["depth_u8"].__setitem__(0, 4),
        lambda _m, a: a["partition_node"]["subtree_membership_digest_sha256"].__setitem__(0, np.void(b"\xff" * 32)),
    ):
        try:
            tampered = _reencode(encoded, arrays, mutate)
        except container.ContainerValidationError:
            continue
        with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH"):
            rebuild_and_validate_container_bytes(
                tampered, source, expected_code_sha256=CODE_SHA,
                expected_dependency_lock_sha256=DEPENDENCY_SHA,
            )
    registry = identifiers.CollisionRegistry()
    registry.record("test_128", b"x" * 16, b"first")
    with pytest.raises(identifiers.DigestCollisionError):
        registry.record("test_128", b"x" * 16, b"second")


def test_v7_tamper_05_sample_census_membership_root_or_presampling_cap_domain(
    exact_fixture,
) -> None:
    _xyz, arrays, encoded, source = exact_fixture
    tampered = _reencode(
        encoded, arrays,
        lambda _m, a: a["leaf_aggregate"]["membership_digest_128"].__setitem__(0, np.void(b"\xaa" * 16)),
    )
    with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH"):
        rebuild_and_validate_container_bytes(
            tampered, source, expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )
    bad = _patch_metadata(encoded, lambda metadata: metadata["census"].__setitem__("sampled_unique_records", 200_000))
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(bad)


def test_v7_tamper_06_actual_o_padding_or_sparse_formula_violation() -> None:
    metadata, tables = small_fixture()
    tables["leaf_aggregate"].append(copy.deepcopy(tables["leaf_aggregate"][0]))
    with pytest.raises(container.ContainerValidationError):
        encode_small(metadata, tables)


def test_v7_tamper_07_omitted_scale_cell_edge_interval_or_rejection() -> None:
    for table_name in (
        "leaf_aggregate", "cell_scale_field", "space_scale_adjacency",
        "persistence_disposition", "rejection",
    ):
        metadata, tables = small_fixture()
        tables[table_name] = tables[table_name][:-1]
        with pytest.raises(container.ContainerValidationError):
            encode_small(metadata, tables)


def test_v7_tamper_08_filtration_tie_owner_status_or_pairing_mutation(exact_fixture) -> None:
    _xyz, arrays, encoded, source = exact_fixture
    tampered = _reencode(
        encoded, arrays,
        lambda _m, a: a["persistence_disposition"]["reason_u16"].__setitem__(0, 5),
    )
    with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH"):
        rebuild_and_validate_container_bytes(
            tampered, source, expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


def test_v7_tamper_09_isotropic_preserved_payload_counterpart_or_dependent_id() -> None:
    metadata, tables = small_fixture()
    cases = []
    drift = copy.deepcopy(tables); drift["cell_scale_field"][3]["trace_q_i64"] += 1; cases.append(drift)
    copied = copy.deepcopy(tables); copied["cell_scale_field"][3]["row_id_digest_128"] = copied["cell_scale_field"][0]["row_id_digest_128"]; cases.append(copied)
    missing = copy.deepcopy(tables); missing["cell_scale_field"] = missing["cell_scale_field"][:-1]; cases.append(missing)
    for changed in cases:
        with pytest.raises(container.ContainerValidationError):
            encode_small(metadata, changed)


def test_v7_tamper_10_denominator_rounded_third_nonzero_q_or_survivor(exact_fixture) -> None:
    metadata, tables = small_fixture()
    for mutate in (
        lambda t: t["cell_scale_field"][0].__setitem__("eigenvalue_denominator_u32", 3),
        lambda t: t["cell_scale_field"][3].__setitem__("lambda1_q_i64", 1),
        lambda t: t["cell_scale_field"][3].__setitem__("anisotropy_q_u16", 1),
    ):
        changed = copy.deepcopy(tables); mutate(changed)
        with pytest.raises(container.ContainerValidationError):
            encode_small(metadata, changed)
    _xyz, arrays, encoded, source = exact_fixture
    arm_size = len(arrays["persistence_disposition"]) // 2
    tampered = _reencode(
        encoded, arrays,
        lambda _m, a: a["persistence_disposition"]["reason_u16"].__setitem__(arm_size, 0),
    )
    with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH"):
        rebuild_and_validate_container_bytes(
            tampered, source, expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


def test_v7_tamper_11_injected_metadata_key_object_string_or_descriptor_order() -> None:
    encoded = encode_small()
    mutations = (
        lambda m: m.__setitem__("unexpected", 1),
        lambda m: m.__setitem__("arbitrary", {"nested": []}),
        lambda m: m.__setitem__("note", "hidden"),
        lambda m: m["table_descriptors"].reverse(),
    )
    for mutate in mutations:
        with pytest.raises(container.ContainerValidationError):
            container.decode_container(_patch_metadata(encoded, mutate))


def test_v7_tamper_12_length_whitespace_padding_or_hidden_xyz_block() -> None:
    encoded = bytearray(encode_small())
    length = struct.unpack_from("<Q", encoded, 0)[0]
    variants = []
    wrong_length = encoded.copy(); struct.pack_into("<Q", wrong_length, 0, length + 1); variants.append(wrong_length)
    whitespace = encoded.copy(); payload = json.dumps(container.decode_metadata_region(bytes(encoded[:container.METADATA_REGION_BYTES])), sort_keys=True).encode(); whitespace[:container.METADATA_REGION_BYTES] = struct.pack("<Q", len(payload)) + payload + bytes(container.MAX_METADATA_JSON_BYTES-len(payload)); variants.append(whitespace)
    padding = encoded.copy(); padding[8 + length] = 1; variants.append(padding)
    hidden = encoded.copy(); hidden[8 + length:8 + length + 24] = struct.pack("<ddd", 1.0, 2.0, 3.0); variants.append(hidden)
    for tampered in variants:
        with pytest.raises(container.ContainerValidationError):
            container.decode_container(bytes(tampered))


def test_v7_tamper_13_known_answer_or_signed_binding_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = dict(identifiers.V7_KNOWN_ANSWER_EXPECTED)
    changed["case_token_digest_sha256"] = "00" * 32
    monkeypatch.setattr(identifiers, "V7_KNOWN_ANSWER_EXPECTED", changed)
    with pytest.raises(ValueError, match="known-answer vector mismatch"):
        identifiers.validate_v7_known_answer_vectors()
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(encode_small(), expected_code_sha256="ff" * 32)


def test_v7_tamper_14_forbidden_xyz_member_variable_width_or_archive_payload() -> None:
    metadata, tables = small_fixture()
    for key in ("xyz", "centroid", "first_moment_xyz", "member_coordinates"):
        changed = copy.deepcopy(metadata); changed[key] = [1.0, 2.0, 3.0]
        with pytest.raises(container.ContainerValidationError):
            encode_small(changed, tables)
    with pytest.raises(container.ContainerValidationError, match="packed table blobs"):
        container.encode_container(metadata, {**tables, "leaf_aggregate": b"PK\x03\x04"})


def test_v7_tamper_15_nonfinite_negative_zero_overflow_q_format_or_eigensolver_disagreement(
    exact_fixture,
) -> None:
    with pytest.raises(ValueError):
        identifiers.canonical_xyz_binary64_le(float("nan"), 0.0, 0.0)
    assert identifiers.canonical_xyz_binary64_le(-0.0, 0.0, 0.0) == bytes(24)
    with pytest.raises(ValueError):
        identifiers.u16_le(1 << 16)
    _xyz, arrays, encoded, source = exact_fixture
    try:
        tampered = _reencode(
            encoded, arrays,
            lambda _m, a: a["cell_scale_field"]["lambda1_q_i64"].__setitem__(0, int(a["cell_scale_field"]["lambda1_q_i64"][0]) + 1),
        )
    except container.ContainerValidationError:
        return
    with pytest.raises(IndependentValidationError, match="INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH"):
        rebuild_and_validate_container_bytes(
            tampered, source, expected_code_sha256=CODE_SHA,
            expected_dependency_lock_sha256=DEPENDENCY_SHA,
        )


def test_v7_tamper_16_forged_work_rss_scratch_time_or_serialized_accounting(exact_fixture) -> None:
    _xyz, _arrays, encoded, source = exact_fixture
    fields = (
        ("cpu_nanoseconds", 300_000_000_001),
        ("wall_nanoseconds", 300_000_000_001),
        ("peak_process_tree_rss_bytes", 12_884_901_889),
        ("scratch_bytes", 268_435_457),
        ("serialized_bytes", len(encoded) + 1),
    )
    for key, value in fields:
        tampered = _patch_metadata(encoded, lambda metadata, key=key, value=value: metadata["resource"].__setitem__(key, value))
        with pytest.raises((container.ContainerValidationError, IndependentValidationError)):
            rebuild_and_validate_container_bytes(
                tampered, source, expected_code_sha256=CODE_SHA,
                expected_dependency_lock_sha256=DEPENDENCY_SHA,
            )


def test_v7_tamper_17_independent_validator_semantic_import_replacement() -> None:
    forbidden = {"primary_parser", "partition", "fields", "graph", "semantic_core", "native_worker"}
    paths = (
        Path(__file__).parents[1] / "research/f1_ciip_v1/independent_validator.py",
        Path(__file__).parents[1] / "research/f1_ciip_v1/independent_semantic_rebuild.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(name.rsplit(".", 1)[-1] in forbidden for name in imports)
    replaced = "from research.f1_ciip_v1.semantic_core import execute_semantic_role\n"
    imports = [node.module or "" for node in ast.walk(ast.parse(replaced)) if isinstance(node, ast.ImportFrom)]
    assert any(name.rsplit(".", 1)[-1] in forbidden for name in imports)


def test_v7_tamper_18_top_k_tail_omission_or_changed_terminal_token() -> None:
    encoded = encode_small()
    tampered = _patch_metadata(
        encoded,
        lambda metadata: metadata["attestation"].__setitem__(
            "top_k_reservoir_tail_omission_or_clipping_used", True
        ),
    )
    with pytest.raises(container.ContainerValidationError):
        container.decode_container(tampered)

    result = _valid_result()
    validate_execution_result(result)
    result["outcome"] = "CII_PERSISTENCE_PREFLIGHT_VALID"
    with pytest.raises(ExecutionResultError, match="terminal outcome token"):
        validate_execution_result(result)
    stopped = _valid_result(); stopped["outcome"] = STOP_TOKEN; stopped["failure"] = "TERMINAL"
    validate_execution_result(stopped)
