from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

import scripts.build_professor_f1_cubical_iip_v1_preflight_lock_package_20260807 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    output = tmp_path / "packet"
    result = builder.build_package(
        output_dir=output,
        audit_path=tmp_path / "audit.json",
        verify_raw_sources=False,
    )
    return output, result


def _valid_closed_metadata(
    config: dict[str, object],
    schema: dict[str, object],
    *,
    occupied_cells: int = 1,
) -> dict[str, object]:
    scale_count = int(config["scales"]["scale_count"]["value"])
    census = {
        "raw_records": 10,
        "finite_records": 10,
        "unique_records": 10,
        "sampled_unique_records": 10,
        "occupied_cells_O": occupied_cells,
        "partition_rows": occupied_cells,
        "field_rows": 2 * scale_count * occupied_cells,
        "adjacency_rows": (scale_count - 1) * occupied_cells,
        "persistence_rows": 2 * scale_count * occupied_cells,
        "rejection_rows": 2 * scale_count * occupied_cells,
        "moment_valid_rows": 2 * scale_count * occupied_cells,
        "attributed_intervals": 1,
        "isotropic_survivors": 0,
    }
    count_by_name = {
        "partition_node": census["partition_rows"],
        "leaf_aggregate": census["occupied_cells_O"],
        "cell_scale_field": census["field_rows"],
        "space_scale_adjacency": census["adjacency_rows"],
        "persistence_disposition": census["persistence_rows"],
        "rejection": census["rejection_rows"],
    }
    offset = int(schema["metadata_reserve_bytes"])
    descriptors: list[dict[str, object]] = []
    for table in schema["tables"]:
        name = str(table["name"])
        rows = int(count_by_name[name])
        row_size = int(table["row_size_bytes"])
        length = rows * row_size
        descriptors.append(
            {
                "name_enum": name,
                "row_count_u64": rows,
                "row_size_u32": row_size,
                "offset_u64": offset,
                "byte_length_u64": length,
                "sha256": "0" * 64,
            }
        )
        offset += length
    metadata = {
        "schema_version": schema["schema_version"],
        "case_token_digest_sha256": builder._domain_digest(
            "TAG_CIIP_CASE_V1", b"\x00" * 32
        ).hex(),
        "raw_source_sha256": "0" * 64,
        "configuration_sha256": config["configuration_sha256"],
        "schema_sha256": schema["schema_sha256"],
        "code_sha256": "0" * 64,
        "dependency_lock_sha256": "0" * 64,
        "partition_commitment_sha256": "0" * 64,
        "table_directory_sha256": "0" * 64,
        "source_points_serialized": False,
        "census": census,
        "resource": {
            "cpu_nanoseconds": 0,
            "wall_nanoseconds": 0,
            "peak_process_tree_rss_bytes": 0,
            "scratch_bytes": 0,
            "serialized_bytes": offset,
            "watchdog_poll_milliseconds": 10,
            "resource_cap_tripped": False,
        },
        "attestation": dict(schema["closed_metadata_schema"]["attestation_expected_values"]),
        "table_descriptors": descriptors,
    }
    directory_parts = [struct.pack("<I", len(descriptors))]
    table_enum = config["canonical_binary_encoding"]["table_name_enum"]
    for descriptor in descriptors:
        directory_parts.append(
            struct.pack(
                "<BQIQQ",
                int(table_enum[descriptor["name_enum"]]),
                int(descriptor["row_count_u64"]),
                int(descriptor["row_size_u32"]),
                int(descriptor["offset_u64"]),
                int(descriptor["byte_length_u64"]),
            )
            + bytes.fromhex(str(descriptor["sha256"]))
        )
    metadata["table_directory_sha256"] = builder._domain_digest(
        "TAG_CIIP_TABLE_DIRECTORY_V1", *directory_parts
    ).hex()
    return metadata


def _metadata_expected_bindings(metadata: dict[str, object]) -> dict[str, str]:
    return {
        key: str(metadata[key])
        for key in (
            "raw_source_sha256",
            "code_sha256",
            "dependency_lock_sha256",
            "partition_commitment_sha256",
        )
    }


def test_builds_exact_ten_file_nonexecution_paper_lock_packet(tmp_path: Path) -> None:
    output, result = _build(tmp_path)
    entries = sorted(output.iterdir())
    assert tuple(path.name for path in entries) == tuple(sorted(builder.UPLOAD_FILENAMES))
    assert len(entries) == 10
    assert result["individual_file_count"] == 10
    assert result["paper_lock_review_only"] is True
    assert result["execution_authority"] == "NONE"
    assert result["geometry_execution_authorized"] is False
    assert result["raw_scan_semantic_processing_authorized"] is False
    assert result["gold_or_truth_consumed"] is False
    assert result["benchmark_authorized"] is False
    assert result["runtime_or_production_authorized"] is False
    assert result["architecture_token"] == builder.ARCHITECTURE_TOKEN
    assert result["primitive_token"] == builder.PRIMITIVE_TOKEN


def test_preserves_selected_pro_response_byte_exact_and_hash_bound(tmp_path: Path) -> None:
    output, result = _build(tmp_path)
    prompt = (output / builder.UPLOAD_FILENAMES[0]).read_text(encoding="utf-8")
    embedded = prompt.split(builder.RESPONSE_BEGIN, 1)[1].split(builder.RESPONSE_END, 1)[0]
    assert embedded.encode("utf-8") == builder.SELECTION_RESPONSE_PATH.read_bytes()
    assert hashlib.sha256(embedded.encode("utf-8")).hexdigest() == builder.SELECTION_RESPONSE_SHA256
    assert result["selection_response_sha256"] == builder.SELECTION_RESPONSE_SHA256
    assert result["embedded_selection_response_byte_exact"] is True


def test_preserves_v5_correction_pro_response_byte_exact_and_hash_bound(tmp_path: Path) -> None:
    output, result = _build(tmp_path)
    prompt = (output / builder.UPLOAD_FILENAMES[0]).read_text(encoding="utf-8")
    embedded = prompt.split(builder.CORRECTION_RESPONSE_BEGIN, 1)[1].split(
        builder.CORRECTION_RESPONSE_END, 1
    )[0]
    assert embedded.encode("utf-8") == builder.CORRECTION_RESPONSE_PATH.read_bytes()
    assert hashlib.sha256(embedded.encode("utf-8")).hexdigest() == builder.CORRECTION_RESPONSE_SHA256
    assert result["correction_response_sha256"] == builder.CORRECTION_RESPONSE_SHA256
    assert result["embedded_correction_response_byte_exact"] is True


def test_carries_source_manifest_byte_identical_and_validates_internal_binding(tmp_path: Path) -> None:
    output, result = _build(tmp_path)
    carried = output / builder.SOURCE_FILENAME
    assert carried.read_bytes() == builder.SOURCE_MANIFEST_PATH.read_bytes()
    assert _sha(carried) == builder.SOURCE_MANIFEST_SHA256
    payload = json.loads(carried.read_text(encoding="utf-8"))
    content = dict(payload)
    asserted = content.pop("input_manifest_sha256")
    assert hashlib.sha256(builder._canonical_json_bytes(content)).hexdigest() == asserted
    assert asserted == builder.SOURCE_MANIFEST_INTERNAL_SHA256
    assert result["source_manifest_internal_sha256"] == asserted


def test_configuration_resolves_scientific_claim_boundary_and_h0_only() -> None:
    config = builder._configuration()
    assert config["architecture_token"] == builder.ARCHITECTURE_TOKEN
    assert config["primitive_token"] == builder.PRIMITIVE_TOKEN
    assert config["claim_boundary"]["published_pottmann_solid_volume_invariant_claimed"] is False
    assert config["claim_boundary"]["primitive_is_digital_occupied_shell_engineering_extrapolation"] is True
    assert config["digital_measure"]["kind"] == "UNION_OF_FULL_OCCUPIED_GRID_CUBES"
    assert config["digital_measure"]["semantic_mass_independent_of_raw_point_count"] is True
    assert config["complex"]["homology_dimensions"]["value"] == [0]
    assert config["complex"]["higher_homology_closed"] is True
    assert config["complex"]["algorithm"].startswith("DETERMINISTIC_UNION_FIND")
    assert config["complex"]["coefficient_field"] == "F2"
    builder._assert_no_unresolved(config)


def test_configuration_locks_partition_scales_gates_and_ties() -> None:
    config = builder._configuration()
    assert config["input"]["sample_cap"]["value"] == 200_000
    assert config["partition"]["grid_depth"]["value"] == 5
    assert config["partition"]["cells_per_axis"]["value"] == 32
    assert config["partition"]["logical_cell_cap"]["value"] == 32_768
    assert config["partition"]["partition_row_cap"]["value"] == 65_535
    assert config["partition"]["boundary_tie_rule"] == "EXACT_BOUNDARY_TO_UPPER_INDEX_CELL"
    assert config["scales"]["radius_multipliers"]["value"] == [2, 3, 4]
    assert config["scales"]["minimum_support_counts"]["value"] == [9, 21, 37]
    assert config["coverage_density_gates"]["opposite_offset_balance_minimum"]["numerator"] == 11
    assert config["coverage_density_gates"]["opposite_offset_balance_minimum"]["denominator"] == 20
    assert config["tensor_and_scalar"]["quantization_multiplier"]["value"] == 65_534
    assert config["tensor_and_scalar"]["dimensionless_tensor_evidence_format"] == "SIGNED_Q24_40_I64"
    assert config["tensor_and_scalar"]["baseline_trace_evidence_rule"].startswith("EXACT_SUM")
    assert config["tensor_and_scalar"]["isotropic_eigenvalue_evidence_rule"].endswith("DENOMINATOR_3")
    assert config["tensor_and_scalar"]["essential_death_sentinel"]["value"] == 65_535
    assert config["complex"]["order"].startswith("(f_u16,event_kind_u8")
    assert config["complex"]["zero_lifetime_intervals_retained"] is True
    assert config["input"]["point_id_collision_policy"] == "TERMINAL_STOP"
    assert "UNEQUAL_CANONICAL_PREIMAGE" in config["input"]["all_truncated_digest_collision_policy"]
    assert config["tensor_and_scalar"]["jacobi_similarity_direction"] == "A_NEW=TRANSPOSE(J)*A_OLD*J"
    assert "FOR_I_0_TO_2_FOR_J_0_TO_2" in config["tensor_and_scalar"]["jacobi_multiply_loop_order"]
    assert config["tensor_and_scalar"]["anisotropy_source"].startswith("FINAL_UNQUANTIZED_BINARY64")
    assert config["tensor_and_scalar"]["q24_40_evidence_not_used_to_compute_anisotropy_or_filtration"] is True
    assert config["complex"]["vertex_index_rule"] == "scale_u8_TIMES_O_PLUS_occupied_cell_ordinal_U32"
    assert config["complex"]["vertex_event_tie_key"] == "vertex_index_u32"
    assert config["complex"]["elder_rule"].endswith("NUMERIC_BIRTH_VERTEX_INDEX_U32")
    assert config["component_semantics"]["finite_snapshot_excludes_death_edge"] is True
    assert config["component_semantics"]["scale_mask_u8_rule"].startswith("BIT_scale_u8")
    assert config["component_semantics"]["finite_lifetime_rule"].endswith("death_f_u16_MINUS_birth_f_u16")
    assert config["component_semantics"]["essential_lifetime_rule"].endswith("65535_MINUS_birth_f_u16")
    assert config["row_codebooks_and_sentinels"]["persistence_status_u8"] == {"FINITE": 0, "ESSENTIAL": 1}
    assert config["row_codebooks_and_sentinels"]["reserved_and_flags_rule"].endswith("ALL_ZERO")
    assert config["isotropic_counterfactual"]["zero_anisotropy_survivors_expected_by_construction"] is True
    assert config["isotropic_counterfactual"]["scientific_nuisance_origin_falsification_claimed"] is False
    assert config["isotropic_counterfactual"]["field_row_pairing_key"] == "case_token_digest_sha256_THEN_scale_u8_THEN_morton_u64"
    assert config["isotropic_counterfactual"]["arm_specific_field_row_ids_recomputed_from_locked_preimages"] is True
    assert config["isotropic_counterfactual"]["baseline_and_isotropic_field_row_ids_byte_identical"] is False
    assert "row_ids_and_order" not in config["isotropic_counterfactual"]["preserved_domains"]
    assert config["claim_boundary"]["baseline_anisotropy_physical_origin_resolved"] is False


def test_identifier_and_commitment_known_answer_vectors_recompute_independently() -> None:
    config = builder._configuration()
    vectors = config["identifier_and_commitment_rules"]["known_answer_vectors"]
    fixture = vectors["fixture"]
    observed = vectors["expected"]
    empty_sha = hashlib.sha256(b"").hexdigest()
    assert len(fixture["table_directory_descriptor_fixtures"]) == 6
    for enum, descriptor in enumerate(fixture["table_directory_descriptor_fixtures"]):
        assert descriptor == {
            "name_enum_u8": enum,
            "row_count_u64": 0,
            "row_size_u32": (88, 64, 128, 32, 64, 32)[enum],
            "offset_u64": 1_048_576,
            "byte_length_u64": 0,
            "table_sha256": empty_sha,
        }

    def digest(tag: str, *parts: bytes) -> bytes:
        return hashlib.sha256(tag.encode("ascii") + b"\x00" + b"".join(parts)).digest()

    raw_source = bytes.fromhex(fixture["raw_source_sha256"])
    case_digest = digest("CIIP_CASE_V1", raw_source)
    xyz = bytes.fromhex(fixture["xyz_binary64_le24"])
    point_id = digest("CIIP_POINT_V1", xyz)
    sample_rank = digest("CIIP_SAMPLE_RANK_V1", point_id)
    x, y, z = fixture["grid_xyz_u5"]
    morton = 0
    for bit in range(5):
        morton |= ((x >> bit) & 1) << (3 * bit)
        morton |= ((y >> bit) & 1) << (3 * bit + 1)
        morton |= ((z >> bit) & 1) << (3 * bit + 2)
    node_id = digest("CIIP_NODE_V1", case_digest, struct.pack("<B", 5), struct.pack("<Q", morton))
    subtree = digest("CIIP_SUBTREE_MEMBERS_V1", struct.pack("<I", 1), point_id)
    cell_id = digest("CIIP_CELL_V1", case_digest, struct.pack("<Q", morton))
    leaf = digest("CIIP_LEAF_MEMBERS_V1", struct.pack("<I", 1), point_id)[:16]
    field = digest("CIIP_FIELD_ROW_V1", case_digest, struct.pack("<BBQ", 0, 1, morton))[:16]
    isotropic_field = digest("CIIP_FIELD_ROW_V1", case_digest, struct.pack("<BBQ", 1, 1, morton))[:16]
    support = digest("CIIP_SUPPORT_MEMBERS_V1", struct.pack("<BI", 1, 1), cell_id)[:16]
    edge = digest("CIIP_EDGE_V1", case_digest, struct.pack("<IIBB", 1, 2, 0, 0))[:16]
    essential = digest("CIIP_ESSENTIAL_DEATH_V1", case_digest, struct.pack("<B", 0), field)[:16]
    interval = digest(
        "CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 0),
        field,
        edge,
        struct.pack("<HH", 10, 20),
    )[:16]
    essential_interval = digest(
        "CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 0),
        field,
        essential,
        struct.pack("<HH", 10, 65_535),
    )[:16]
    isotropic_essential = digest(
        "CIIP_ESSENTIAL_DEATH_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field,
    )[:16]
    isotropic_interval = digest(
        "CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field,
        edge,
        struct.pack("<HH", 10, 20),
    )[:16]
    isotropic_essential_interval = digest(
        "CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field,
        isotropic_essential,
        struct.pack("<HH", 10, 65_535),
    )[:16]
    partition = digest(
        "CIIP_PARTITION_COMMIT_V1",
        struct.pack("<Q", 1),
        bytes.fromhex(fixture["partition_row_fixture_hex"]),
    )
    directory_parts = [struct.pack("<I", 6)]
    for descriptor in fixture["table_directory_descriptor_fixtures"]:
        directory_parts.append(
            struct.pack(
                "<BQIQQ",
                descriptor["name_enum_u8"],
                descriptor["row_count_u64"],
                descriptor["row_size_u32"],
                descriptor["offset_u64"],
                descriptor["byte_length_u64"],
            )
            + bytes.fromhex(descriptor["table_sha256"])
        )
    directory = digest("CIIP_TABLE_DIRECTORY_V1", *directory_parts)
    assert observed == {
        "case_token_digest_sha256": case_digest.hex(),
        "point_id_sha256": point_id.hex(),
        "sample_rank_sha256": sample_rank.hex(),
        "morton_u64": morton,
        "node_id_sha256": node_id.hex(),
        "subtree_membership_digest_sha256": subtree.hex(),
        "cell_id_sha256": cell_id.hex(),
        "leaf_membership_digest_128": leaf.hex(),
        "field_row_id_digest_128": field.hex(),
        "isotropic_field_row_id_digest_128": isotropic_field.hex(),
        "support_membership_digest_128": support.hex(),
        "edge_id_digest_128": edge.hex(),
        "essential_death_row_id_digest_128": essential.hex(),
        "finite_interval_id_digest_128": interval.hex(),
        "essential_interval_id_digest_128": essential_interval.hex(),
        "isotropic_essential_death_row_id_digest_128": isotropic_essential.hex(),
        "isotropic_finite_interval_id_digest_128": isotropic_interval.hex(),
        "isotropic_essential_interval_id_digest_128": isotropic_essential_interval.hex(),
        "partition_commitment_sha256": partition.hex(),
        "table_directory_sha256": directory.hex(),
    }
    assert observed["field_row_id_digest_128"] == "ae35aeb5777676ad6ac61aba20f90caa"
    assert observed["isotropic_field_row_id_digest_128"] == "72dc258326cb1f01014240070cc9e59b"
    assert observed["isotropic_essential_death_row_id_digest_128"] == "9b7ee31a830d9fbf9c08dd1f25205fce"
    assert observed["isotropic_finite_interval_id_digest_128"] == "1bbcb2248cdaadadaf40c12ebbd7535c"
    assert observed["isotropic_essential_interval_id_digest_128"] == "0a1538fb562f41615668cf31c77f8f3e"
    assert observed["field_row_id_digest_128"] != observed["isotropic_field_row_id_digest_128"]


def test_every_explicit_engineering_constant_is_provenance_typed() -> None:
    config = builder._configuration()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "provenance" in value:
                assert value["provenance"] == builder.ENGINEERING_PROVENANCE
                assert "value" in value or {"numerator", "denominator"} <= set(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(config)
    assert builder.ENGINEERING_PROVENANCE in config["numeric_provenance_rule"]


def test_schema_is_fixed_width_complete_and_coordinate_free() -> None:
    schema = builder._schema()
    assert schema["source_points_serialized"] is False
    assert schema["encoding"]["unknown_fields_allowed"] is False
    assert schema["encoding"]["trailing_bytes_allowed"] is False
    assert schema["encoding"]["archive_or_object_payload_allowed"] is False
    names = {table["name"] for table in schema["tables"]}
    assert names == {
        "partition_node",
        "leaf_aggregate",
        "cell_scale_field",
        "space_scale_adjacency",
        "persistence_disposition",
        "rejection",
    }
    for table in schema["tables"]:
        assert table["fixed_width"] is True
        assert table["variable_width_extensions_allowed"] is False
        assert sum(field["bytes"] for field in table["fields"]) == table["row_size_bytes"]
    field_names = {
        field["name"]
        for table in schema["tables"]
        for field in table["fields"]
    }
    forbidden = set(builder.FORBIDDEN_SCHEMA_TOKENS)
    assert field_names.isdisjoint(forbidden)
    assert "morton_u64" in field_names
    assert "leaf_centroid" not in field_names
    assert "first_moment_xyz" not in field_names
    tables = {table["name"]: table for table in schema["tables"]}
    partition_fields = {field["name"]: field for field in tables["partition_node"]["fields"]}
    field_fields = {field["name"]: field for field in tables["cell_scale_field"]["fields"]}
    adjacency_fields = {field["name"]: field for field in tables["space_scale_adjacency"]["fields"]}
    assert tables["partition_node"]["row_size_bytes"] == 88
    assert partition_fields["node_id_sha256"]["bytes"] == 32
    assert partition_fields["subtree_membership_digest_sha256"]["bytes"] == 32
    assert "eigenvalue_denominator_u32" not in partition_fields
    assert field_fields["eigenvalue_denominator_u32"]["bytes"] == 4
    assert tables["space_scale_adjacency"]["row_size_bytes"] == 32
    assert adjacency_fields["edge_id_digest_128"]["bytes"] == 16


def test_resource_proof_recomputes_exact_rows_bytes_work_and_caps() -> None:
    config = builder._configuration()
    schema = builder._schema()
    proof = builder._resource_proof(config, schema)
    assert proof["maximum_rows_per_complete_role"] == {
        "partition_node": 65_535,
        "leaf_aggregate": 32_768,
        "cell_scale_field": 196_608,
        "space_scale_adjacency": 360_448,
        "persistence_disposition": 196_608,
        "rejection": 196_608,
    }
    assert proof["calculated_rows_including_metadata_record"] == 1_048_576
    assert proof["maximum_bytes_per_complete_role"]["partition_node"] == 5_767_080
    assert proof["maximum_bytes_per_complete_role"]["space_scale_adjacency"] == 11_534_336
    assert proof["calculated_payload_bytes_per_complete_role"] == 64_487_336
    assert proof["hard_payload_cap_bytes_per_complete_role"] == 67_108_864
    assert proof["payload_headroom_bytes"] == 2_621_528
    assert proof["constants"]["stencil_widths"] == [125, 343, 729]
    assert proof["constants"]["M_observed_max_finite_records"] == 1_140_033
    assert proof["constants"]["raw_bytes_observed_max"] == 64_382_075
    assert proof["work_upper_bounds"]["moment_lattice_offset_checks"] == 39_223_296
    assert proof["work_upper_bounds"]["coordinate_radix_byte_passes"] == 27_360_792
    assert proof["work_upper_bounds"]["stable_point_id_and_rank_radix_byte_passes"] == 72_962_112
    assert proof["work_upper_bounds"]["baseline_plus_isotropic_union_find_parent_steps"] == 27_852_800
    assert proof["work_upper_bounds"]["major_operation_upper_bound_per_role"] == 4_600_208_856
    assert proof["generic_h1_h2_boundary_matrix_reduction_authorized"] is False
    assert proof["h0_union_find_linear_domain_only"] is True
    assert proof["scratch_required_bytes"] <= proof["scratch_cap_bytes"]
    assert proof["semantic_allocation_ledger_total_bytes"] == 596_856_088
    assert proof["semantic_allocation_ledger_total_bytes"] <= proof["application_array_cap_bytes"]
    assert proof["fixed_reserve_total_bytes"] == 1_140_850_688
    assert proof["static_process_tree_calculated_bytes"] == 1_737_706_776
    assert proof["static_process_tree_calculated_bytes"] <= proof["static_process_tree_configured_cap_bytes"]
    assert proof["static_process_tree_calculated_bytes"] < proof["enterprise_rss_forbidden_threshold_bytes"]
    assert "runtime_allocator_library_reserve_bytes" not in proof


def test_source_manifest_observed_envelope_is_recomputed_not_assumed() -> None:
    payload = json.loads(builder.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == builder.SOURCE_MANIFEST_CASE_COUNT
    assert max(case["finite_coordinate_count"] for case in payload["cases"]) == builder.SOURCE_MAX_FINITE_RECORDS
    assert max(case["raw_byte_count"] for case in payload["cases"]) == builder.SOURCE_MAX_RAW_BYTES
    assert builder._source_manifest_bytes() == builder.SOURCE_MANIFEST_PATH.read_bytes()


@pytest.mark.parametrize("occupied_cells", [1, 7, 32_768])
def test_actual_sparse_domain_counts_are_o_parameterized(occupied_cells: int) -> None:
    config = builder._configuration()
    scale_count = int(config["scales"]["scale_count"]["value"])
    census = {
        "raw_records": occupied_cells,
        "finite_records": occupied_cells,
        "unique_records": occupied_cells,
        "sampled_unique_records": min(occupied_cells, 200_000),
        "occupied_cells_O": occupied_cells,
        "partition_rows": 2 * occupied_cells - 1,
        "field_rows": 2 * scale_count * occupied_cells,
        "adjacency_rows": (4 * scale_count - 1) * occupied_cells,
        "persistence_rows": 2 * scale_count * occupied_cells,
        "rejection_rows": 2 * scale_count * occupied_cells,
        "moment_valid_rows": 2 * scale_count * occupied_cells,
        "attributed_intervals": 1,
        "isotropic_survivors": 0,
    }
    builder._validate_actual_domain_counts(census, config)


def test_actual_sparse_domain_rejects_padding_to_logical_cap() -> None:
    config = builder._configuration()
    metadata = _valid_closed_metadata(config, builder._schema(), occupied_cells=7)
    census = metadata["census"]
    census["field_rows"] = 2 * 3 * 32_768
    with pytest.raises(ValueError, match="complete-domain count mismatch"):
        builder._validate_actual_domain_counts(census, config)


def test_exact_rational_trace_evidence_does_not_require_divisibility_by_three() -> None:
    builder._validate_valid_trace_evidence(
        arm="BASELINE",
        trace_numerator=7,
        eigenvalue_numerators=(4, 2, 1),
        eigenvalue_denominator=1,
    )
    builder._validate_valid_trace_evidence(
        arm="ISOTROPIC",
        trace_numerator=7,
        eigenvalue_numerators=(7, 7, 7),
        eigenvalue_denominator=3,
    )
    with pytest.raises(ValueError, match="isotropic same-trace rational mismatch"):
        builder._validate_valid_trace_evidence(
            arm="ISOTROPIC",
            trace_numerator=7,
            eigenvalue_numerators=(2, 2, 3),
            eigenvalue_denominator=1,
        )


def test_closed_metadata_accepts_exact_schema_and_rejects_injection_and_forgery() -> None:
    config = builder._configuration()
    schema = builder._schema()
    metadata = _valid_closed_metadata(config, schema)
    expected = _metadata_expected_bindings(metadata)
    builder._validate_closed_metadata(metadata, schema, config, expected)

    injected = deepcopy(metadata)
    injected["root_origin"] = [0, 0, 0]
    with pytest.raises(ValueError, match="top-level key set mismatch"):
        builder._validate_closed_metadata(injected, schema, config, expected)

    forged = deepcopy(metadata)
    forged["attestation"]["gold_truth_cad_render_benchmark_accessed"] = True
    with pytest.raises(ValueError, match="attestation value mismatch"):
        builder._validate_closed_metadata(forged, schema, config, expected)

    offset_tamper = deepcopy(metadata)
    offset_tamper["table_descriptors"][1]["offset_u64"] += 1
    with pytest.raises(ValueError, match="offset is not contiguous"):
        builder._validate_closed_metadata(offset_tamper, schema, config, expected)

    wrong_binding = dict(expected)
    wrong_binding["code_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="signed binding mismatch: code_sha256"):
        builder._validate_closed_metadata(metadata, schema, config, wrong_binding)

    wrong_case = deepcopy(metadata)
    wrong_case["case_token_digest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="case-token digest recomputation mismatch"):
        builder._validate_closed_metadata(wrong_case, schema, config, expected)

    wrong_directory = deepcopy(metadata)
    wrong_directory["table_directory_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="table-directory commitment mismatch"):
        builder._validate_closed_metadata(wrong_directory, schema, config, expected)


def test_metadata_region_is_length_prefixed_canonical_and_zero_padded() -> None:
    config = builder._configuration()
    schema = builder._schema()
    metadata = _valid_closed_metadata(config, schema)
    expected = _metadata_expected_bindings(metadata)
    region = builder._encode_metadata_region(metadata, schema)
    assert len(region) == 1_048_576
    encoded_length = struct.unpack("<Q", region[:8])[0]
    assert region[8 : 8 + encoded_length] == builder._canonical_json_bytes(metadata)
    assert set(region[8 + encoded_length :]) <= {0}
    assert builder._validate_metadata_region_bytes(region, schema, config, expected) == metadata

    nonzero_padding = bytearray(region)
    nonzero_padding[-1] = 1
    with pytest.raises(ValueError, match="padding must be all zero"):
        builder._validate_metadata_region_bytes(bytes(nonzero_padding), schema, config, expected)

    bad_length = bytearray(region)
    bad_length[:8] = struct.pack("<Q", encoded_length + 1)
    with pytest.raises(ValueError):
        builder._validate_metadata_region_bytes(bytes(bad_length), schema, config, expected)

    alternate_json = bytearray(region)
    alternate_json[8 : 8 + encoded_length] = b" " + region[8 : 8 + encoded_length - 1]
    with pytest.raises(ValueError):
        builder._validate_metadata_region_bytes(bytes(alternate_json), schema, config, expected)


def test_prompt_has_exact_tokens_pairings_bindings_and_none_authority(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)
    prompt = (output / builder.UPLOAD_FILENAMES[0]).read_text(encoding="utf-8")
    for token in builder.LOCK_DECISION_TOKENS + builder.NEXT_STEP_TOKENS:
        assert token in prompt
    for decision, next_step in builder.VALID_DECISION_PAIRS.items():
        assert f"`{decision}` with `{next_step}`" in prompt
    assert "EXECUTION_AUTHORITY: NONE" in prompt
    assert "EXECUTION_AUTHORITY=NONE" in prompt
    assert builder.ARCHITECTURE_TOKEN in prompt
    assert builder.PRIMITIVE_TOKEN in prompt
    assert builder.SELECTION_RESPONSE_SHA256 in prompt
    assert builder.CORRECTION_RESPONSE_SHA256 in prompt
    assert builder.SOURCE_MANIFEST_SHA256 in prompt
    assert builder.CLASSIFICATION_RESPONSE_SHA256 in prompt
    assert "No second correction outcome" in prompt


def test_corrected_identity_contract_is_arm_bound_and_fail_closed() -> None:
    config = builder._configuration()
    iso = config["isotropic_counterfactual"]
    assert iso["baseline_to_isotropic_arm_map"] == "BASELINE_0_TO_ISOTROPIC_1"
    assert iso["arm_specific_rejection_death_and_interval_ids_recomputed_transitively"] is True
    assert iso["rejection_row_id_rule"] == "EXACT_OWN_ARM_FIELD_ROW_ID_DIGEST_128"
    assert iso["persistence_birth_row_id_rule"] == "EXACT_OWN_ARM_FIELD_ROW_ID_DIGEST_128"
    safety = builder._safety_contract(config, builder._schema(), builder._resource_proof(config, builder._schema()))
    assert "ARM_SPECIFIC_IDENTIFIER_OR_SOURCE_CELL_CORRESPONDENCE_FAILURE" in safety["terminal_stop_conditions"]
    spec = builder._normative_spec(config, builder._schema(), builder._resource_proof(config, builder._schema()))
    assert "Do not preserve arm-bound IDs byte-identically" in spec
    assert "copied baseline ID in an isotropic row" in spec.replace("\n", " ")


def test_prompt_hash_table_matches_all_nine_attachments(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)
    prompt = (output / builder.UPLOAD_FILENAMES[0]).read_text(encoding="utf-8")
    parsed = builder._parse_attachment_hashes(prompt)
    assert parsed == {
        name: _sha(output / name) for name in builder.UPLOAD_FILENAMES[1:]
    }


@pytest.mark.parametrize("name", builder.UPLOAD_FILENAMES)
def test_standalone_validator_rejects_every_attachment_mutation(
    tmp_path: Path, name: str
) -> None:
    output, _ = _build(tmp_path)
    path = output / name
    path.write_bytes(path.read_bytes() + b"X")
    with pytest.raises(ValueError, match="deterministic expected bytes"):
        builder._validate_packet(output)


def test_rejects_existing_output_raw_verification_and_inside_audit(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        builder.build_package(
            output_dir=existing,
            audit_path=None,
            verify_raw_sources=False,
        )
    with pytest.raises(ValueError, match="may not read or verify raw PLY"):
        builder.build_package(
            output_dir=tmp_path / "raw",
            audit_path=None,
            verify_raw_sources=True,
        )
    output = tmp_path / "inside"
    with pytest.raises(ValueError, match="audit path must remain outside"):
        builder.build_package(
            output_dir=output,
            audit_path=output / "audit.json",
            verify_raw_sources=False,
        )


def test_validator_rejects_extra_hidden_directory_symlink_and_archive(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)
    (output / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 10"):
        builder._validate_packet(output)

    (output / "extra.txt").unlink()
    (output / ".hidden").write_text("hidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden"):
        builder._validate_packet(output)

    (output / ".hidden").unlink()
    (output / "subdir").mkdir()
    with pytest.raises(ValueError, match="regular non-symlink"):
        builder._validate_packet(output)

    (output / "subdir").rmdir()
    (output / "archive.zip").write_bytes(b"zip")
    with pytest.raises(ValueError, match="exactly 10"):
        builder._validate_packet(output)

    (output / "archive.zip").unlink()
    (output / "link").symlink_to(output / builder.UPLOAD_FILENAMES[1])
    with pytest.raises(ValueError, match="regular non-symlink"):
        builder._validate_packet(output)


def test_context_validation_rejects_recomputed_authority_escalation() -> None:
    _, context = builder._expected_payloads()
    changed = deepcopy(context)
    changed_config = dict(changed["configuration"])
    changed_config.pop("configuration_sha256")
    changed_config["authority"] = dict(changed_config["authority"])
    changed_config["authority"]["geometry_execution_authorized"] = True
    changed["configuration"] = builder._self_bound_json(
        changed_config, "configuration_sha256"
    )
    with pytest.raises(ValueError, match="authority state mismatch"):
        builder._validate_context(changed)


def test_context_validation_rejects_recomputed_resource_arithmetic_mutation() -> None:
    _, context = builder._expected_payloads()
    changed = deepcopy(context)
    proof = dict(changed["resource_proof"])
    proof.pop("resource_proof_sha256")
    proof["calculated_payload_bytes_per_complete_role"] += 1
    changed["resource_proof"] = builder._self_bound_json(
        proof, "resource_proof_sha256"
    )
    changed["safety"] = builder._safety_contract(
        changed["configuration"], changed["schema"], changed["resource_proof"]
    )
    with pytest.raises(ValueError, match="does not recompute"):
        builder._validate_context(changed)


@pytest.mark.parametrize(
    "vector_name",
    sorted(builder._identifier_known_answer_vectors()["expected"]),
)
def test_context_rejects_each_identifier_or_commitment_vector_tamper(
    vector_name: str,
) -> None:
    _, context = builder._expected_payloads()
    changed = deepcopy(context)
    config = dict(changed["configuration"])
    config.pop("configuration_sha256")
    config["identifier_and_commitment_rules"] = deepcopy(
        config["identifier_and_commitment_rules"]
    )
    vectors = config["identifier_and_commitment_rules"]["known_answer_vectors"]
    value = vectors["expected"][vector_name]
    vectors["expected"][vector_name] = value + 1 if isinstance(value, int) else ("f" if value[0] != "f" else "e") + value[1:]
    changed["configuration"] = builder._self_bound_json(config, "configuration_sha256")
    changed["resource_proof"] = builder._resource_proof(
        changed["configuration"], changed["schema"]
    )
    changed["safety"] = builder._safety_contract(
        changed["configuration"], changed["schema"], changed["resource_proof"]
    )
    with pytest.raises(ValueError, match="known-answer vector mismatch"):
        builder._validate_context(changed)


def test_context_validation_rejects_h1_h2_or_zero_falsifier_denominator() -> None:
    _, context = builder._expected_payloads()
    changed = deepcopy(context)
    config = dict(changed["configuration"])
    config.pop("configuration_sha256")
    config["complex"] = dict(config["complex"])
    config["complex"]["homology_dimensions"] = builder._engineering([0, 1])
    config["isotropic_counterfactual"] = dict(config["isotropic_counterfactual"])
    config["isotropic_counterfactual"]["baseline_attributed_denominator_minimum"] = builder._engineering(0)
    changed["configuration"] = builder._self_bound_json(config, "configuration_sha256")
    changed["resource_proof"] = builder._resource_proof(changed["configuration"], changed["schema"])
    with pytest.raises(ValueError):
        builder._validate_context(changed)


def test_context_rejects_scientific_overclaim_for_tautological_isotropic_negative() -> None:
    _, context = builder._expected_payloads()
    changed = deepcopy(context)
    config = dict(changed["configuration"])
    config.pop("configuration_sha256")
    config["isotropic_counterfactual"] = dict(config["isotropic_counterfactual"])
    config["isotropic_counterfactual"]["scientific_nuisance_origin_falsification_claimed"] = True
    changed["configuration"] = builder._self_bound_json(config, "configuration_sha256")
    changed["resource_proof"] = builder._resource_proof(changed["configuration"], changed["schema"])
    changed["safety"] = builder._safety_contract(
        changed["configuration"], changed["schema"], changed["resource_proof"]
    )
    with pytest.raises(ValueError, match="may not claim baseline nuisance falsification"):
        builder._validate_context(changed)


def test_audit_is_external_and_upload_stays_exactly_ten(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    audit = tmp_path / "external" / "audit.json"
    builder.build_package(
        output_dir=output,
        audit_path=audit,
        verify_raw_sources=False,
    )
    assert audit.is_file()
    assert len(list(output.iterdir())) == 10
    builder._validate_packet(output)
