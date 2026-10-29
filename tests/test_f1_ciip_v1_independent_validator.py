from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from research.f1_ciip_v1 import container
from research.f1_ciip_v1 import independent_semantic_rebuild as independent_rebuild
from research.f1_ciip_v1 import independent_validator as validator


def _expected_source(raw: bytes, header: bytes, packed_rows: list[bytes], *, fmt: str, ignored: list[str]) -> dict:
    ordered = sorted(packed_rows)
    digest = hashlib.sha256(validator.COORDINATE_MULTISET_DOMAIN + b"".join(ordered)).hexdigest()
    duplicates = len(ordered) - len(set(ordered))
    ignored_digest = hashlib.sha256(
        json.dumps(sorted(ignored), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return {
        "raw_byte_count": len(raw),
        "raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
        "ply_header_byte_count": len(header),
        "ply_header_sha256": hashlib.sha256(header).hexdigest(),
        "ply_format": fmt,
        "parsed_vertex_count": len(packed_rows),
        "finite_coordinate_count": len(packed_rows),
        "exact_duplicate_count": duplicates,
        "ignored_property_name_count": len(ignored),
        "ignored_property_name_digest": ignored_digest,
        "canonical_coordinate_multiset_digest": digest,
    }


def _ascii_fixture() -> tuple[bytes, dict]:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 3\n"
        b"property double x\nproperty double y\nproperty double z\n"
        b"property uchar red\nend_header\n"
    )
    rows = [(0.0, -0.0, 2.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)]
    body = b"0 -0 2 7\n1 2 3 8\n1 2 3 9\n"
    raw = header + body
    packed = [struct.pack("<ddd", *(0.0 if value == 0.0 else value for value in row)) for row in rows]
    return raw, _expected_source(raw, header, packed, fmt="ascii", ignored=["vertex:red"])


def _single_point_fixture() -> tuple[bytes, dict]:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    packed = struct.pack("<ddd", 0.0, 0.0, 0.0)
    raw = header + packed
    return raw, _expected_source(raw, header, [packed], fmt="binary_little_endian", ignored=[])


def _parse(tmp_path: Path, raw: bytes, expected: dict) -> validator.ParsedSource:
    path = tmp_path / "opaque-input"
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, min(3, len(raw)), os.SEEK_SET)
        before = os.lseek(fd, 0, os.SEEK_CUR)
        parsed = validator.parse_and_verify_ply_fd(fd, expected)
        assert os.lseek(fd, 0, os.SEEK_CUR) == before
        return parsed
    finally:
        os.close(fd)


def _container_bytes(raw_sha: str, *, rss: int = 0, isotropic_trace: int = 3) -> bytes:
    case = container.case_token_digest(bytes.fromhex(raw_sha))
    morton = 5
    partition = [{
        "node_id_sha256": container.node_id(case, 5, morton),
        "morton_prefix_u64": morton,
        "subtree_membership_digest_sha256": hashlib.sha256(b"subtree").digest(),
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
        "membership_digest_128": hashlib.sha256(b"leaf").digest()[:16],
        "unique_point_count_u32": 1,
        "raw_record_multiplicity_sum_u32": 1,
    }]
    fields = []
    for arm in (0, 1):
        for scale in range(3):
            trace = 3 if arm == 0 else isotropic_trace
            fields.append({
                "row_id_digest_128": container.field_row_id(case, arm, scale, morton),
                "support_membership_digest_128": hashlib.sha256(f"support-{scale}".encode()).digest()[:16],
                "morton_u64": morton,
                "trace_q_i64": trace,
                "lambda1_q_i64": 2 if arm == 0 else trace,
                "lambda2_q_i64": 1 if arm == 0 else trace,
                "lambda3_q_i64": 0 if arm == 0 else trace,
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
                "filtration_u16": 65533 if arm == 0 else 65534,
                "stage_mask_u32": 0x7F,
                "first_rejection_u16": 0,
                "flags_u16": 0,
                "eigenvalue_denominator_u32": 1 if arm == 0 else 3,
            })
    adjacency = []
    for lower, upper in ((0, 1), (1, 2)):
        adjacency.append({
            "edge_id_digest_128": container.edge_id(case, lower, upper, 1, 255),
            "lower_vertex_u32": lower, "upper_vertex_u32": upper,
            "edge_kind_u8": 1, "axis_u8": 255, "status_u16": 0, "reserved_zero_u32": 0,
        })
    persistence = []
    rejection = []
    for birth_vertex, field in enumerate(fields):
        arm = field["arm_u8"]
        birth = field["row_id_digest_128"]
        death = container.essential_death_row_id(case, arm, birth)
        persistence.append({
            "interval_id_digest_128": container.interval_id(case, arm, birth, death, field["filtration_u16"], 65535),
            "birth_row_id_digest_128": birth,
            "death_row_id_digest_128": death,
            "arm_u8": arm, "status_u8": 1, "reason_u16": 0 if arm == 0 else 1,
            "birth_f_u16": field["filtration_u16"], "death_f_or_essential_u16": 65535,
            "lifetime_u16": 65535 - field["filtration_u16"],
            "scale_mask_u8": 1 << field["scale_u8"], "spatial_face_edge_count_u32": 0,
            "flags_u8": 0, "birth_vertex_index_u32": birth_vertex % 3,
        })
        rejection.append({
            "row_id_digest_128": birth, "arm_u8": arm, "scale_u8": field["scale_u8"],
            "first_rejection_u16": 0, "stage_mask_u32": 0x7F, "denominator_id_u16": 0,
            "reserved_zero_bytes6": bytes(6), "morton_u64": morton,
        })
    tables = {
        "partition_node": partition, "leaf_aggregate": leaf, "cell_scale_field": fields,
        "space_scale_adjacency": adjacency, "persistence_disposition": persistence,
        "rejection": rejection,
    }
    metadata = {
        "schema_version": 1, "raw_source_sha256": raw_sha,
        "configuration_sha256": container.EXPECTED_CONFIGURATION_SHA256,
        "schema_sha256": container.EXPECTED_SCHEMA_SHA256,
        "code_sha256": "11" * 32, "dependency_lock_sha256": "22" * 32,
        "source_points_serialized": False,
        "census": {
            "raw_records": 1, "finite_records": 1, "unique_records": 1,
            "sampled_unique_records": 1, "occupied_cells_O": 1, "partition_rows": 1,
            "field_rows": 6, "adjacency_rows": 2, "persistence_rows": 6,
            "rejection_rows": 6, "moment_valid_rows": 6,
            "attributed_intervals": 3, "isotropic_survivors": 0,
        },
        "resource": {
            "cpu_nanoseconds": 0, "wall_nanoseconds": 0,
            "peak_process_tree_rss_bytes": rss, "scratch_bytes": 0,
            "watchdog_poll_milliseconds": 10, "resource_cap_tripped": False,
        },
        "attestation": {
            "semantic_core_imported_by_independent_validator": False,
            "filename_path_family_identity_semantics_used": False,
            "gold_truth_cad_render_benchmark_accessed": False,
            "source_points_serialized": False,
            "top_k_reservoir_tail_omission_or_clipping_used": False,
            "complete_baseline_and_isotropic_payload": True,
            "deterministic_rebuild_match": True, "input_order_match": True,
            "independent_validator_match": True, "all_truncated_digest_collisions_checked": True,
        },
    }
    return container.encode_container(
        metadata, tables,
        order_key_hooks={
            "persistence_disposition": lambda row: (row["arm_u8"], row["birth_vertex_index_u32"]),
            "rejection": lambda row: (row["arm_u8"], row["scale_u8"], row["morton_u64"]),
        },
    )


def test_independent_ascii_parser_verifies_frozen_diagnostics_and_preserves_fd_offset(tmp_path: Path) -> None:
    raw, expected = _ascii_fixture()
    parsed = _parse(tmp_path, raw, expected)
    assert parsed.diagnostics.exact_duplicate_count == 1
    assert parsed.points.shape == (3, 3)
    assert not bool(__import__("numpy").signbit(parsed.points[0, 1]))


def test_independent_binary_parser_and_container_relationship_validation(tmp_path: Path) -> None:
    raw, expected = _single_point_fixture()
    parsed = _parse(tmp_path, raw, expected)
    encoded = _container_bytes(parsed.diagnostics.raw_ply_sha256)
    summary = validator.validate_container_bytes(
        encoded, parsed.diagnostics,
        expected_code_sha256="11" * 32,
        expected_dependency_lock_sha256="22" * 32,
    )
    assert summary.occupied_cells == 1
    assert summary.sampled_unique_records == 1


def test_parser_stops_on_frozen_diagnostic_mismatch_and_nonfinite(tmp_path: Path) -> None:
    raw, expected = _ascii_fixture()
    bad = dict(expected)
    bad["ply_header_sha256"] = "00" * 32
    with pytest.raises(validator.IndependentValidationError, match="FROZEN_SOURCE_DIAGNOSTIC_MISMATCH"):
        _parse(tmp_path, raw, bad)

    header = (
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty double x\n"
        b"property double y\nproperty double z\nend_header\n"
    )
    nonfinite = header + b"nan 0 0\n"
    expected_nonfinite = _expected_source(
        nonfinite, header, [struct.pack("<ddd", float("nan"), 0.0, 0.0)],
        fmt="ascii", ignored=[],
    )
    with pytest.raises(validator.IndependentValidationError, match="NONFINITE_COORDINATE"):
        _parse(tmp_path, nonfinite, expected_nonfinite)


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_parser_requires_exact_real_lock_multiset_digest_key(tmp_path: Path, mode: str) -> None:
    raw, expected = _ascii_fixture()
    bad = dict(expected)
    if mode == "missing":
        bad.pop("canonical_coordinate_multiset_digest")
    else:
        bad["canonical_coordinate_multiset_digest"] = "00" * 32
    with pytest.raises(
        validator.IndependentValidationError,
        match="FROZEN_SOURCE_DIAGNOSTIC_MISMATCH: canonical_coordinate_multiset_digest",
    ):
        _parse(tmp_path, raw, bad)


def test_parser_rejects_float_list_count_type_before_body(tmp_path: Path) -> None:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty double x\n"
        b"property double y\nproperty double z\nproperty list float int junk\nend_header\n"
    )
    raw = header + b"0 0 0 0\n"
    expected = _expected_source(raw, header, [bytes(24)], fmt="ascii", ignored=["vertex:junk"])
    with pytest.raises(validator.IndependentValidationError, match="PLY_LIST_TYPE_UNSUPPORTED"):
        _parse(tmp_path, raw, expected)


def test_independent_validator_rejects_preserved_arm_drift_and_resource_overage(tmp_path: Path) -> None:
    raw, expected = _single_point_fixture()
    parsed = _parse(tmp_path, raw, expected)
    # The neutral fixed-array codec now rejects this malformed cross-arm row
    # even earlier than the independent semantic validator.
    with pytest.raises(container.ContainerValidationError, match="preserved payload"):
        _container_bytes(parsed.diagnostics.raw_ply_sha256, isotropic_trace=4)
    oversized = _container_bytes(parsed.diagnostics.raw_ply_sha256, rss=validator.RSS_WATCHDOG_BYTES + 1)
    with pytest.raises(validator.IndependentValidationError, match="RESOURCE_CAP_EXCEEDED"):
        validator.validate_container_bytes(
            oversized, parsed.diagnostics,
            expected_code_sha256="11" * 32, expected_dependency_lock_sha256="22" * 32,
        )


def test_independent_validator_has_no_forbidden_imports() -> None:
    forbidden = {"primary_parser", "partition", "fields", "graph", "semantic_core", "native_worker"}
    for module in (validator, independent_rebuild):
        source_path = Path(module.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(name.rsplit(".", 1)[-1] in forbidden for name in imports), (source_path, imports)
