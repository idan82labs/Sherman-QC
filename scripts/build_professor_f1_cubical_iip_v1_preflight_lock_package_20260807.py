#!/usr/bin/env python3
"""Build the corrected non-executable Cubical-IIP-V1 paper-lock review packet.

The selected architecture is represented only as a declarative paper lock.  This
builder never reads raw PLY geometry and grants no authority to implement or run
the representation.  GPT-5.6 Pro may approve the paper lock for one further
pre-execution review packet or stop the family.  This V7 packet applies the one
paper-only correction authorized by the V5 GPT-5.6 Pro audit and uses unique
upload filenames so the ChatGPT UI cannot alias repeated attachment names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_professor_f1_pwts_stop_decision_package_20260807 as stop_packet


PREFLIGHT_ROOT = stop_packet.PREFLIGHT_ROOT
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7"
)
DEFAULT_AUDIT_PATH = (
    PREFLIGHT_ROOT
    / "ciip-v1-preflight-lock-package-v7"
    / "package_integrity_audit.json"
)

ARCHITECTURE_TOKEN = "SELECT_CUBICAL_INTEGRAL_INVARIANT_PERSISTENCE_V1"
PRIMITIVE_TOKEN = "CIIP_DIGITAL_OCCUPIED_SHELL_H0_V1"
MAX_FILES = 10

SELECTION_RESPONSE_PATH = (
    ROOT
    / "docs/research/PROFESSOR_F1_POST_PWTS_ARCHITECTURE_SELECTION_GPT56_PRO_RESPONSE_20260808.txt"
)
SELECTION_RESPONSE_SHA256 = (
    "20380ec3cc4c84c8daa6ebdf539d18298f22f41d08200433d040fd07f71325e5"
)
CORRECTION_RESPONSE_PATH = (
    ROOT
    / "docs/research/PROFESSOR_F1_CUBICAL_IIP_V1_PREFLIGHT_LOCK_V5_GPT56_PRO_RESPONSE_20260809.txt"
)
CORRECTION_RESPONSE_SHA256 = (
    "f15e2d775a23405bc5f593dd1c260e6bb6d7b6d0718e6a8b635935f4d73fd23b"
)
CLASSIFICATION_RESPONSE_SHA256 = (
    "ffd8c2a9259e7fcb4be96c966d771d4c8e2b73afa653cab5d7ea83bef8e980cf"
)
PRIOR_SELECTION_PROMPT_SHA256 = (
    "dafbee73bcc920a1c4414d5cf11490e0889a428c51fd99f358c9d3ed9416c901"
)
SOURCE_MANIFEST_PATH = Path(
    stop_packet.SOURCE_BINDINGS["02_SOURCE_LOCK_INPUT_MANIFEST.json"]["path"]
)
SOURCE_MANIFEST_SHA256 = str(
    stop_packet.SOURCE_BINDINGS["02_SOURCE_LOCK_INPUT_MANIFEST.json"]["sha256"]
)
SOURCE_MANIFEST_INTERNAL_SHA256 = (
    "a87b10e9b14119a543d5d114d84579007a1272616b29c317f505812c4f4667fd"
)
SOURCE_MANIFEST_CASE_COUNT = 37
SOURCE_MAX_FINITE_RECORDS = 1_140_033
SOURCE_MAX_RAW_BYTES = 64_382_075

LOCK_PASS = "CUBICAL_IIP_V1_CORRECTION_AUDIT_PASS_NO_EXECUTION"
LOCK_STOP = "CUBICAL_IIP_V1_CORRECTION_AUDIT_STOP_NO_EXECUTION"
LOCK_DECISION_TOKENS = (LOCK_PASS, LOCK_STOP)

NEXT_PREFLIGHT = "BUILD_CUBICAL_IIP_V1_PREEXECUTION_REVIEW_PACKET_ONLY"
NEXT_STOP = "FORMAL_CUBICAL_IIP_V1_FAMILY_STOP_NO_EXECUTION"
NEXT_STEP_TOKENS = (NEXT_PREFLIGHT, NEXT_STOP)
VALID_DECISION_PAIRS = {
    LOCK_PASS: NEXT_PREFLIGHT,
    LOCK_STOP: NEXT_STOP,
}

ENGINEERING_PROVENANCE = "PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED"
SOURCE_OBSERVATION_PROVENANCE = "BOUND_SOURCE_MANIFEST_OBSERVATION_NOT_ALGORITHM_CONSTANT"
EXECUTION_AUTHORITY = "NONE"

PROMPT_FILENAME = "00_V7_GPT_PRO_CUBICAL_IIP_V1_PREFLIGHT_LOCK_REVIEW.md"
SOURCE_FILENAME = "01_V7_SOURCE_LOCK_INPUT_MANIFEST.json"
LINEAGE_FILENAME = "02_V7_LINEAGE_CLAIM_AND_CROSSBINDING_LEDGER.md"
LITERATURE_FILENAME = "03_V7_PRIMARY_SOURCE_AND_ENGINEERING_ASSUMPTION_LEDGER.md"
SPEC_FILENAME = "04_V7_NORMATIVE_CUBICAL_IIP_V1_PREFLIGHT_SPEC.md"
CONFIG_FILENAME = "05_V7_LOCKED_CUBICAL_IIP_V1_CONFIGURATION.json"
SCHEMA_FILENAME = "06_V7_LOCKED_CUBICAL_IIP_V1_SCHEMA.json"
PROOF_FILENAME = "07_V7_STATIC_CARDINALITY_EVIDENCE_RESOURCE_PROOF.json"
VALIDATOR_FILENAME = "08_V7_INDEPENDENT_VALIDATOR_AND_TAMPER_CONTRACT.md"
SAFETY_FILENAME = "09_V7_SAFETY_AUTHORITY_AND_TERMINAL_CONTRACT.json"

UPLOAD_FILENAMES = (
    PROMPT_FILENAME,
    SOURCE_FILENAME,
    LINEAGE_FILENAME,
    LITERATURE_FILENAME,
    SPEC_FILENAME,
    CONFIG_FILENAME,
    SCHEMA_FILENAME,
    PROOF_FILENAME,
    VALIDATOR_FILENAME,
    SAFETY_FILENAME,
)

RESPONSE_BEGIN = "-----BEGIN VERBATIM CUBICAL IIP ARCHITECTURE SELECTION RESPONSE-----\n"
RESPONSE_END = "-----END VERBATIM CUBICAL IIP ARCHITECTURE SELECTION RESPONSE-----"
CORRECTION_RESPONSE_BEGIN = "-----BEGIN VERBATIM CUBICAL IIP V5 LOCK AUDIT RESPONSE-----\n"
CORRECTION_RESPONSE_END = "-----END VERBATIM CUBICAL IIP V5 LOCK AUDIT RESPONSE-----"

FORBIDDEN_SCHEMA_TOKENS = (
    "source_xyz",
    "source_points",
    "point_coordinates",
    "point_membership_list",
    "neighborhood_array",
    "member_coordinates",
    "leaf_centroid",
    "first_moment_xyz",
    "gold",
    "truth",
    "expected_count",
    "manual_label",
    "cad",
    "reference_geometry",
    "render",
)

ID_DOMAIN_TAGS = {
    "TAG_CIIP_CASE_V1": "CIIP_CASE_V1",
    "TAG_CIIP_POINT_V1": "CIIP_POINT_V1",
    "TAG_CIIP_SAMPLE_RANK_V1": "CIIP_SAMPLE_RANK_V1",
    "TAG_CIIP_NODE_V1": "CIIP_NODE_V1",
    "TAG_CIIP_SUBTREE_MEMBERS_V1": "CIIP_SUBTREE_MEMBERS_V1",
    "TAG_CIIP_CELL_V1": "CIIP_CELL_V1",
    "TAG_CIIP_LEAF_MEMBERS_V1": "CIIP_LEAF_MEMBERS_V1",
    "TAG_CIIP_FIELD_ROW_V1": "CIIP_FIELD_ROW_V1",
    "TAG_CIIP_SUPPORT_MEMBERS_V1": "CIIP_SUPPORT_MEMBERS_V1",
    "TAG_CIIP_EDGE_V1": "CIIP_EDGE_V1",
    "TAG_CIIP_ESSENTIAL_DEATH_V1": "CIIP_ESSENTIAL_DEATH_V1",
    "TAG_CIIP_INTERVAL_V1": "CIIP_INTERVAL_V1",
    "TAG_CIIP_PARTITION_COMMIT_V1": "CIIP_PARTITION_COMMIT_V1",
    "TAG_CIIP_TABLE_DIRECTORY_V1": "CIIP_TABLE_DIRECTORY_V1",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_line(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _self_bound_json(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    if key in result:
        raise ValueError(f"self-hash key already present: {key}")
    result[key] = _sha_bytes(_canonical_json_bytes(result))
    return result


def _verify_self_bound_json(value: Mapping[str, Any], key: str, label: str) -> str:
    content = dict(value)
    asserted = content.pop(key, None)
    observed = _sha_bytes(_canonical_json_bytes(content))
    if asserted != observed:
        raise ValueError(f"{label} canonical self-hash mismatch")
    return observed


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        raise ValueError("fraction denominator must be positive")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "provenance": ENGINEERING_PROVENANCE,
    }


def _engineering(value: Any) -> dict[str, Any]:
    return {"value": value, "provenance": ENGINEERING_PROVENANCE}


def _authority_state() -> dict[str, Any]:
    return {
        "execution_authority": EXECUTION_AUTHORITY,
        "geometry_execution_authorized": False,
        "raw_scan_semantic_processing_authorized": False,
        "gold_or_truth_access_authorized": False,
        "expected_count_access_authorized": False,
        "manual_label_access_authorized": False,
        "cad_or_reference_access_authorized": False,
        "render_access_authorized": False,
        "benchmark_authorized": False,
        "synthetic_or_replay_authorized": False,
        "runtime_or_main_integration_authorized": False,
        "production_or_release_authorized": False,
        "next_action": "PROFESSOR_PACKET_ONLY",
    }


def _domain_digest(tag_key: str, *parts: bytes) -> bytes:
    tag = ID_DOMAIN_TAGS[tag_key].encode("ascii") + b"\x00"
    return hashlib.sha256(tag + b"".join(parts)).digest()


def _morton5(x: int, y: int, z: int) -> int:
    if any(value < 0 or value >= 32 for value in (x, y, z)):
        raise ValueError("known-answer Morton coordinate outside five-bit domain")
    result = 0
    for bit in range(5):
        result |= ((x >> bit) & 1) << (3 * bit)
        result |= ((y >> bit) & 1) << (3 * bit + 1)
        result |= ((z >> bit) & 1) << (3 * bit + 2)
    return result


def _identifier_known_answer_vectors() -> dict[str, Any]:
    raw_source = b"\x00" * 32
    case_digest = _domain_digest("TAG_CIIP_CASE_V1", raw_source)
    xyz = struct.pack("<ddd", 0.0, 0.0, 0.0)
    point_id = _domain_digest("TAG_CIIP_POINT_V1", xyz)
    sample_rank = _domain_digest("TAG_CIIP_SAMPLE_RANK_V1", point_id)
    morton = _morton5(1, 2, 3)
    node_id = _domain_digest(
        "TAG_CIIP_NODE_V1", case_digest, struct.pack("<B", 5), struct.pack("<Q", morton)
    )
    subtree = _domain_digest(
        "TAG_CIIP_SUBTREE_MEMBERS_V1", struct.pack("<I", 1), point_id
    )
    cell_id = _domain_digest("TAG_CIIP_CELL_V1", case_digest, struct.pack("<Q", morton))
    leaf_membership = _domain_digest(
        "TAG_CIIP_LEAF_MEMBERS_V1", struct.pack("<I", 1), point_id
    )[:16]
    field_row = _domain_digest(
        "TAG_CIIP_FIELD_ROW_V1",
        case_digest,
        struct.pack("<BBQ", 0, 1, morton),
    )[:16]
    isotropic_field_row = _domain_digest(
        "TAG_CIIP_FIELD_ROW_V1",
        case_digest,
        struct.pack("<BBQ", 1, 1, morton),
    )[:16]
    support = _domain_digest(
        "TAG_CIIP_SUPPORT_MEMBERS_V1",
        struct.pack("<BI", 1, 1),
        cell_id,
    )[:16]
    edge = _domain_digest(
        "TAG_CIIP_EDGE_V1",
        case_digest,
        struct.pack("<IIBB", 1, 2, 0, 0),
    )[:16]
    essential_death = _domain_digest(
        "TAG_CIIP_ESSENTIAL_DEATH_V1", case_digest, struct.pack("<B", 0), field_row
    )[:16]
    finite_interval = _domain_digest(
        "TAG_CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 0),
        field_row,
        edge,
        struct.pack("<HH", 10, 20),
    )[:16]
    essential_interval = _domain_digest(
        "TAG_CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 0),
        field_row,
        essential_death,
        struct.pack("<HH", 10, 65_535),
    )[:16]
    isotropic_essential_death = _domain_digest(
        "TAG_CIIP_ESSENTIAL_DEATH_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field_row,
    )[:16]
    isotropic_finite_interval = _domain_digest(
        "TAG_CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field_row,
        edge,
        struct.pack("<HH", 10, 20),
    )[:16]
    isotropic_essential_interval = _domain_digest(
        "TAG_CIIP_INTERVAL_V1",
        case_digest,
        struct.pack("<B", 1),
        isotropic_field_row,
        isotropic_essential_death,
        struct.pack("<HH", 10, 65_535),
    )[:16]
    partition_row_fixture = b"\x00" * 88
    partition_commitment = _domain_digest(
        "TAG_CIIP_PARTITION_COMMIT_V1", struct.pack("<Q", 1), partition_row_fixture
    )
    row_sizes = (88, 64, 128, 32, 64, 32)
    empty_table_sha256 = hashlib.sha256(b"").digest()
    descriptor_fixtures = [
        {
            "name_enum_u8": name_enum,
            "row_count_u64": 0,
            "row_size_u32": row_size,
            "offset_u64": 1_048_576,
            "byte_length_u64": 0,
            "table_sha256": empty_table_sha256.hex(),
        }
        for name_enum, row_size in enumerate(row_sizes)
    ]
    directory_parts = [struct.pack("<I", 6)]
    for descriptor in descriptor_fixtures:
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
    table_directory = _domain_digest(
        "TAG_CIIP_TABLE_DIRECTORY_V1", *directory_parts
    )
    return {
        "fixture": {
            "raw_source_sha256": raw_source.hex(),
            "xyz_binary64_le24": xyz.hex(),
            "grid_xyz_u5": [1, 2, 3],
            "depth_u8": 5,
            "baseline_arm_u8": 0,
            "isotropic_arm_u8": 1,
            "scale_u8": 1,
            "edge_fixture": {"lower_vertex_u32": 1, "upper_vertex_u32": 2, "edge_kind_u8": 0, "axis_u8": 0},
            "finite_interval_birth_f_u16": 10,
            "finite_interval_death_f_u16": 20,
            "essential_interval_death_f_u16": 65_535,
            "partition_row_fixture_hex": partition_row_fixture.hex(),
            "table_directory_descriptor_fixtures": descriptor_fixtures,
        },
        "expected": {
            "case_token_digest_sha256": case_digest.hex(),
            "point_id_sha256": point_id.hex(),
            "sample_rank_sha256": sample_rank.hex(),
            "morton_u64": morton,
            "node_id_sha256": node_id.hex(),
            "subtree_membership_digest_sha256": subtree.hex(),
            "cell_id_sha256": cell_id.hex(),
            "leaf_membership_digest_128": leaf_membership.hex(),
            "field_row_id_digest_128": field_row.hex(),
            "isotropic_field_row_id_digest_128": isotropic_field_row.hex(),
            "support_membership_digest_128": support.hex(),
            "edge_id_digest_128": edge.hex(),
            "essential_death_row_id_digest_128": essential_death.hex(),
            "finite_interval_id_digest_128": finite_interval.hex(),
            "essential_interval_id_digest_128": essential_interval.hex(),
            "isotropic_essential_death_row_id_digest_128": isotropic_essential_death.hex(),
            "isotropic_finite_interval_id_digest_128": isotropic_finite_interval.hex(),
            "isotropic_essential_interval_id_digest_128": isotropic_essential_interval.hex(),
            "partition_commitment_sha256": partition_commitment.hex(),
            "table_directory_sha256": table_directory.hex(),
        },
    }


def _configuration() -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": 1,
        "package_id": "SHERMAN_QC_F1_CUBICAL_IIP_V1_PREFLIGHT_LOCK_REVIEW_20260809_V7",
        "architecture_token": ARCHITECTURE_TOKEN,
        "primitive_token": PRIMITIVE_TOKEN,
        "claim_boundary": {
            "published_pottmann_solid_volume_invariant_claimed": False,
            "primitive_is_digital_occupied_shell_engineering_extrapolation": True,
            "inside_outside_or_mesh_constructed": False,
            "literal_bounded_domain_invariant_unavailable_from_unoriented_xyz": True,
            "isotropic_negative_claimed_to_distinguish_bends_from_boundary_hole_density_or_quantization_artifacts": False,
            "baseline_anisotropy_physical_origin_resolved": False,
        },
        "authority": _authority_state(),
        "source_envelope_observed": {
            "maximum_finite_coordinate_records_per_scan": SOURCE_MAX_FINITE_RECORDS,
            "maximum_raw_bytes_per_scan": SOURCE_MAX_RAW_BYTES,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_observation_provenance": SOURCE_OBSERVATION_PROVENANCE,
        },
        "input": {
            "allowed_semantic_properties": ["x", "y", "z"],
            "finite_only": True,
            "numeric_type": "IEEE754_BINARY64",
            "byte_order": "LITTLE_ENDIAN",
            "negative_zero_policy": "NORMALIZE_TO_POSITIVE_ZERO",
            "exact_duplicate_policy": "COLLAPSE_PACKED_XYZ_IGNORE_MULTIPLICITY_SEMANTICALLY",
            "point_id_rule": "SHA256(TAG_CIIP_POINT_V1 || canonical_xyz_binary64_le24)",
            "sample_rank_rule": "SHA256(TAG_CIIP_SAMPLE_RANK_V1 || point_id_sha256_raw32)",
            "sample_order_rule": "ASCENDING_SAMPLE_RANK_THEN_POINT_ID",
            "point_id_collision_policy": "TERMINAL_STOP",
            "all_truncated_digest_collision_policy": "SAME_DIGEST_WITH_UNEQUAL_CANONICAL_PREIMAGE_TERMINAL_STOP",
            "sample_cap": _engineering(200_000),
            "filename_path_family_identity_semantics": False,
            "source_normals_colors_semantics": False,
        },
        "canonical_binary_encoding": {
            "ascii_domain_tag_rule": "EXACT_LISTED_ASCII_BYTES_FOLLOWED_BY_ONE_0X00_BYTE",
            "unsigned_integer_rule": "FIXED_WIDTH_LITTLE_ENDIAN_U8_U16_U32_U64",
            "signed_integer_rule": "FIXED_WIDTH_TWOS_COMPLEMENT_LITTLE_ENDIAN_I32_I64",
            "binary64_xyz_rule": "X_THEN_Y_THEN_Z_IEEE754_BINARY64_LITTLE_ENDIAN_AFTER_NEGATIVE_ZERO_NORMALIZATION",
            "sha256_rule": "RAW_32_DIGEST_BYTES",
            "digest_truncation_rule": "FIRST_N_RAW_SHA256_DIGEST_BYTES_NO_INTEGER_REINTERPRETATION",
            "counted_sequence_rule": "U32_LE_COUNT_THEN_FIXED_WIDTH_ITEMS_WITH_NO_SEPARATOR",
            "morton_coordinate_bits": "GRID_XYZ_INDICES_ARE_UNSIGNED_5_BIT_VALUES",
            "morton_interleave_rule": "FOR_I_0_TO_4_PUT_X_BIT_I_AT_3I_Y_BIT_I_AT_3I_PLUS_1_Z_BIT_I_AT_3I_PLUS_2",
            "morton_prefix_rule": "DEPTH_D_PREFIX_IS_MORTON_U64_SHIFT_RIGHT_BY_3_TIMES_5_MINUS_D_STORED_RIGHT_ALIGNED_HIGH_BITS_ZERO",
            "patricia_prefix_order": "DEPTH_ASCENDING_THEN_MORTON_PREFIX_U64_ASCENDING",
            "occupied_cell_order": "MORTON_U64_ASCENDING",
            "arm_enum": {"BASELINE": 0, "ISOTROPIC": 1},
            "edge_kind_enum": {"SPATIAL_FACE": 0, "SCALE": 1},
            "axis_enum": {"X": 0, "Y": 1, "Z": 2, "NOT_APPLICABLE": 255},
            "table_name_enum": {
                "partition_node": 0,
                "leaf_aggregate": 1,
                "cell_scale_field": 2,
                "space_scale_adjacency": 3,
                "persistence_disposition": 4,
                "rejection": 5,
            },
        },
        "identifier_and_commitment_rules": {
            "case_token_digest_sha256": "SHA256(TAG_CIIP_CASE_V1 || raw_source_sha256_raw32)",
            "point_id_sha256": "SHA256(TAG_CIIP_POINT_V1 || canonical_xyz_binary64_le24)",
            "sample_rank_sha256": "SHA256(TAG_CIIP_SAMPLE_RANK_V1 || point_id_sha256_raw32)",
            "node_id_sha256": "SHA256(TAG_CIIP_NODE_V1 || case_token_digest_sha256_raw32 || depth_u8 || morton_prefix_u64_le)",
            "subtree_membership_digest_sha256": "SHA256(TAG_CIIP_SUBTREE_MEMBERS_V1 || point_count_u32_le || point_id_sha256_raw32_sorted_ascending)",
            "cell_id_sha256": "SHA256(TAG_CIIP_CELL_V1 || case_token_digest_sha256_raw32 || morton_u64_le)",
            "leaf_membership_digest_128": "FIRST16(SHA256(TAG_CIIP_LEAF_MEMBERS_V1 || point_count_u32_le || point_id_sha256_raw32_sorted_ascending))",
            "field_row_id_digest_128": "FIRST16(SHA256(TAG_CIIP_FIELD_ROW_V1 || case_token_digest_sha256_raw32 || arm_u8 || scale_u8 || morton_u64_le))",
            "support_membership_digest_128": "FIRST16(SHA256(TAG_CIIP_SUPPORT_MEMBERS_V1 || scale_u8 || support_count_u32_le || cell_id_sha256_raw32_sorted_ascending))",
            "edge_id_digest_128": "FIRST16(SHA256(TAG_CIIP_EDGE_V1 || case_token_digest_sha256_raw32 || lower_vertex_u32_le || upper_vertex_u32_le || edge_kind_u8 || axis_u8))",
            "finite_death_row_id_digest_128": "EXACT_EDGE_ID_DIGEST_128_OF_DEATH_CAUSING_EDGE",
            "essential_death_row_id_digest_128": "FIRST16(SHA256(TAG_CIIP_ESSENTIAL_DEATH_V1 || case_token_digest_sha256_raw32 || arm_u8 || birth_row_id_digest_128))",
            "interval_id_digest_128": "FIRST16(SHA256(TAG_CIIP_INTERVAL_V1 || case_token_digest_sha256_raw32 || arm_u8 || birth_row_id_digest_128 || death_row_id_digest_128 || birth_f_u16_le || death_f_or_essential_u16_le))",
            "rejection_row_id_digest_128": "EXACT_FIELD_ROW_ID_DIGEST_128",
            "partition_commitment_sha256": "SHA256(TAG_CIIP_PARTITION_COMMIT_V1 || partition_row_count_u64_le || canonical_partition_table_bytes)",
            "table_directory_sha256": "SHA256(TAG_CIIP_TABLE_DIRECTORY_V1 || descriptor_count_u32_le || SIX_DESCRIPTORS_IN_TABLE_ENUM_ORDER)",
            "table_directory_descriptor_binary": "name_enum_u8 || row_count_u64_le || row_size_u32_le || offset_u64_le || byte_length_u64_le || table_sha256_raw32",
            "all_tags": dict(ID_DOMAIN_TAGS),
            "truncated_digest_collision_registry": "FOR_EACH_ID_CLASS_MAP_TRUNCATED_BYTES_TO_FULL_CANONICAL_PREIMAGE_AND_STOP_ON_UNEQUAL_PREIMAGE",
            "known_answer_vectors_required_before_geometry": True,
            "known_answer_vectors": _identifier_known_answer_vectors(),
        },
        "canonical_table_orders": {
            "partition_node": "depth_u8_THEN_morton_prefix_u64",
            "leaf_aggregate": "morton_u64",
            "cell_scale_field": "arm_u8_THEN_scale_u8_THEN_morton_u64",
            "space_scale_adjacency": "edge_kind_u8_THEN_axis_u8_THEN_lower_vertex_u32_THEN_upper_vertex_u32",
            "persistence_disposition": "arm_u8_THEN_birth_vertex_index_u32",
            "rejection": "arm_u8_THEN_scale_u8_THEN_morton_u64",
        },
        "partition": {
            "kind": "SPARSE_COMPLETE_DEPTH_DYADIC_CUBICAL_GRID",
            "root_axis_range_rule": "D=max(max_a-min_a); require D>0",
            "root_side_rule": "R=(D*4.0)/3.0 with each operation rounded binary64",
            "root_center_rule": "c_a=min_a+(max_a-min_a)/2",
            "root_origin_rule": "o_a=c_a-R/2",
            "cell_side_rule": "h=R/32",
            "coordinate_assignment_rule": "floor((x_a-o_a)/h) without epsilon",
            "boundary_tie_rule": "EXACT_BOUNDARY_TO_UPPER_INDEX_CELL",
            "out_of_range_policy": "TERMINAL_STOP",
            "logical_split_rule": "BREADTH_FIRST_ALL_OCCUPIED_NODES_TO_DEPTH_5",
            "within_level_order": "MORTON_ASCENDING_THEN_POINT_ID",
            "serialized_tree": "PATRICIA_OCCUPIED_RADIX_TREE_UNARY_PATHS_COLLAPSED",
            "grid_depth": _engineering(5),
            "cells_per_axis": _engineering(32),
            "logical_cell_cap": _engineering(32_768),
            "occupied_leaf_symbol": "O",
            "occupied_leaf_bound": "0 < O <= 32768",
            "partition_row_cap": _engineering(65_535),
            "minimum_root_margin_cells": _engineering(4),
            "all_sampled_points_assigned_exactly_once": True,
            "empty_cells_serialized": False,
        },
        "digital_measure": {
            "kind": "UNION_OF_FULL_OCCUPIED_GRID_CUBES",
            "semantic_mass_per_occupied_cell": "h^3",
            "semantic_mass_independent_of_raw_point_count": True,
            "raw_point_count_is_diagnostic_only": True,
            "cube_covariance_term": "h^2*I/12",
            "integer_accumulation": "SIGNED_INT128_LATTICE_COUNTS_SUMS_AND_SECOND_SUMS",
            "division_policy": "DELAY_UNTIL_CANONICAL_TENSOR_CONSTRUCTION",
            "source_coordinate_serialization": False,
        },
        "scales": {
            "scale_count": _engineering(3),
            "radius_multipliers": _engineering([2, 3, 4]),
            "physical_radius_rule": "R_s=m_s*h in raw coordinate units",
            "support_rule": "occupied cell-center integer offset squared <= m_s^2",
            "cube_ball_intersection_claimed": False,
            "minimum_support_counts": _engineering([9, 21, 37]),
        },
        "coverage_density_gates": {
            "support_face_graph_components_required": _engineering(1),
            "opposite_offset_balance_minimum": _fraction(11, 20),
            "opposite_offset_partition_rule": "EXCLUDE_ZERO_OFFSET_THEN_PLUS_IF_FIRST_NONZERO_COMPONENT_IN_X_Y_Z_ORDER_IS_POSITIVE_ELSE_MINUS",
            "opposite_offset_balance_evidence_rule": "P=OCCUPIED_PLUS_COUNT_N=OCCUPIED_MINUS_COUNT_NUMERATOR=MIN(P,N)_DENOMINATOR=MAX(P,N)_REQUIRE_DENOMINATOR_GT_ZERO",
            "opposite_offset_balance_integer_test": "20*balance_numerator_u32 >= 11*balance_denominator_u32",
            "growth_dimension_minimum": _fraction(5, 4),
            "growth_dimension_maximum": _fraction(11, 4),
            "growth12_lower_integer_test": "2^5*n2^4 >= 3^5*n1^4",
            "growth12_upper_integer_test": "2^11*n2^4 <= 3^11*n1^4",
            "growth23_lower_integer_test": "3^5*n3^4 >= 4^5*n2^4",
            "growth23_upper_integer_test": "3^11*n3^4 <= 4^11*n2^4",
            "growth_integer_arithmetic": "UNSIGNED_INT128_NO_LOGARITHM",
            "raw_count_p90_to_max1_p10_maximum": _engineering(32),
            "percentile_convention": "NEAREST_RANK_ASCENDING_COUNTS",
            "p10_zero_based_index": "ceil(n/10)-1",
            "p90_zero_based_index": "ceil(9*n/10)-1",
            "density_gate_integer_test": "P90 <= 32*max(1,P10)",
            "case_invalid_field_fraction_maximum": _fraction(9, 10),
            "ordered_first_rejections": [
                "INSUFFICIENT_SUPPORT",
                "DISCONNECTED_SUPPORT",
                "OPPOSITE_BALANCE_FAILURE",
                "GROWTH_DIMENSION_FAILURE",
                "DENSITY_HETEROGENEITY_FAILURE",
                "NONPOSITIVE_TRACE",
                "NONFINITE_OR_NUMERIC_FAILURE",
            ],
        },
        "row_codebooks_and_sentinels": {
            "scale_u8": {"RADIUS_2H": 0, "RADIUS_3H": 1, "RADIUS_4H": 2},
            "partition_status_u16": {"VALID": 0},
            "field_status_u16": {"VALID": 0, "REJECTED": 1},
            "adjacency_status_u16": {"ACTIVE": 0},
            "persistence_status_u8": {"FINITE": 0, "ESSENTIAL": 1},
            "first_rejection_u16": {
                "NONE_VALID": 0,
                "INSUFFICIENT_SUPPORT": 1,
                "DISCONNECTED_SUPPORT": 2,
                "OPPOSITE_BALANCE_FAILURE": 3,
                "GROWTH_DIMENSION_FAILURE": 4,
                "DENSITY_HETEROGENEITY_FAILURE": 5,
                "NONPOSITIVE_TRACE": 6,
                "NONFINITE_OR_NUMERIC_FAILURE": 7,
            },
            "stage_mask_u32_bits": {
                "SUPPORT": 0,
                "CONNECTIVITY": 1,
                "OPPOSITE_BALANCE": 2,
                "GROWTH_DIMENSION": 3,
                "DENSITY_HETEROGENEITY": 4,
                "TRACE": 5,
                "NUMERIC": 6,
            },
            "stage_mask_rule": "SET_STAGE_BIT_IMMEDIATELY_AFTER_THAT_STAGE_IS_EVALUATED_PASS_OR_FAIL_STOP_EVALUATING_AFTER_FIRST_FAILURE",
            "growth_gate_code_i32": {
                "NOT_EVALUATED": -2_147_483_648,
                "FAIL_LOWER": -1,
                "PASS": 0,
                "FAIL_UPPER": 1,
            },
            "persistence_reason_u16_first_failure_order": {
                "ANISOTROPY_ATTRIBUTED": 0,
                "BIRTH_Q_NOT_POSITIVE": 1,
                "LIFETIME_TOO_SHORT": 2,
                "DISTINCT_SCALE_COUNT_TOO_SMALL": 3,
                "SPATIAL_FACE_EDGE_COUNT_TOO_SMALL": 4,
                "COMPONENT_CONTAINS_INVALID_FIELD_ROW": 5,
            },
            "rejection_denominator_id_u16": {
                "NONE": 0,
                "SUPPORT_COUNT": 1,
                "SUPPORT_COMPONENT_COUNT": 2,
                "OPPOSITE_BALANCE": 3,
                "GROWTH_12": 4,
                "GROWTH_23": 5,
                "DENSITY_P10_P90": 6,
                "TRACE": 7,
                "NUMERIC": 8,
            },
            "reserved_and_flags_rule": "EVERY_RESERVED_OR_FLAGS_FIELD_MUST_BE_ALL_ZERO",
            "rejected_field_row_rule": "PRESERVE_VALUES_FROM_COMPLETED_STAGES_SET_ALL_NOT_YET_COMPUTED_NUMERIC_FIELDS_TO_ZERO_EXCEPT_GROWTH_CODES_NOT_EVALUATED_Q_EQ_0_FILTRATION_EQ_65534_AND_EIGENVALUE_DENOMINATOR_EQ_1_BASELINE_OR_3_ISOTROPIC",
            "valid_field_row_rule": "ALL_SEVEN_STAGE_BITS_SET_FIRST_REJECTION_NONE_VALID_STATUS_VALID",
            "rejection_table_rule": "ONE_ROW_PER_FIELD_ROW_REPEATING_EXACT_FIRST_REJECTION_STAGE_MASK_AND_APPLICABLE_DENOMINATOR_ID_VALID_ROWS_USE_ALL_PASS_SENTINELS",
        },
        "tensor_and_scalar": {
            "covariance_rule": "h^2*(mean((z-zbar)(z-zbar)^T)+I/12)",
            "eigenvalue_order": "lambda1>=lambda2>=lambda3>=0",
            "eigensolver": "FIXED_12_SWEEP_SYMMETRIC_3X3_CYCLIC_JACOBI",
            "jacobi_sweep_count": _engineering(12),
            "jacobi_pivot_order_each_sweep": [[0, 1], [0, 2], [1, 2]],
            "jacobi_exact_zero_rule": "IF_APQ_IS_EXACTLY_ZERO_USE_IDENTITY_ROTATION",
            "jacobi_nonzero_rotation_rule": "tau=(aqq-app)/(2*apq); t=sign_nonzero(tau)/(abs(tau)+sqrt(1+tau^2)); tau_zero_uses_t=1; c=1/sqrt(1+t^2); s=t*c",
            "jacobi_rotation_matrix": "J_IS_IDENTITY_EXCEPT_JPP_EQ_C_JQQ_EQ_C_JPQ_EQ_S_JQP_EQ_MINUS_S",
            "jacobi_similarity_direction": "A_NEW=TRANSPOSE(J)*A_OLD*J",
            "jacobi_update_rule": "FORM_B=A_OLD*J_THEN_A_TMP=TRANSPOSE_J*B_USING_EXACT_LOCKED_LOOPS",
            "jacobi_multiply_loop_order": "FOR_I_0_TO_2_FOR_J_0_TO_2_ACC_POSITIVE_ZERO_FOR_K_0_TO_2_ACC=RN_EVEN(ACC+RN_EVEN(LEFT_IK*RIGHT_KJ))",
            "jacobi_authoritative_triangle": "COPY_A_TMP_UPPER_TRIANGLE_I_LE_J_THEN_SET_LOWER_AJI_EXACTLY_EQUAL_TO_UPPER_AIJ",
            "jacobi_scalar_evaluation_order": "EACH_LISTED_SUBEXPRESSION_LEFT_TO_RIGHT_WITH_BINARY64_ROUND_AFTER_EACH_ADD_SUBTRACT_MULTIPLY_DIVIDE_OR_SQRT",
            "sqrt_rule": "IEEE754_BINARY64_CORRECTLY_ROUNDED_SQRT_ROUND_TO_NEAREST_TIES_EVEN",
            "binary64_environment": "ROUND_TO_NEAREST_TIES_EVEN_NO_FMA_NO_EXTENDED_PRECISION",
            "offdiagonal_residual_relative_power_of_two": _engineering(-40),
            "negative_eigenvalue_relative_power_of_two": _engineering(-40),
            "small_negative_eigenvalue_policy": "CLAMP_TO_ZERO_IF_ABS(lambda)<=(2^-40)*trace_ELSE_STOP",
            "eigenvalue_sort_tie_rule": "DESCENDING_VALUE_THEN_ASCENDING_ORIGINAL_AXIS",
            "dimensionless_tensor_evidence_fraction_bits": _engineering(40),
            "dimensionless_tensor_evidence_format": "SIGNED_Q24_40_I64",
            "dimensionless_tensor_evidence_rule": "round_half_even((eigenvalue/h^2)*2^40)_TO_I64",
            "baseline_trace_evidence_rule": "EXACT_SUM_OF_THREE_BASELINE_EIGENVALUE_NUMERATORS_DENOMINATOR_1",
            "isotropic_eigenvalue_evidence_rule": "EACH_NUMERATOR_EQUALS_TRACE_NUMERATOR_DENOMINATOR_3",
            "mass_evidence_rule": "support_count_in_units_of_h^3",
            "blas_threads": _engineering(1),
            "anisotropy_scalar": "((lambda1-lambda2)/trace)*((lambda2-lambda3)/trace)",
            "anisotropy_source": "FINAL_UNQUANTIZED_BINARY64_JACOBI_EIGENVALUES_AFTER_NEGATIVE_CLAMP_AND_DESCENDING_SORT",
            "anisotropy_operation_order": "trace=RN(RN(lambda1+lambda2)+lambda3);d12=RN(lambda1-lambda2);d23=RN(lambda2-lambda3);L=RN(d12/trace);P=RN(d23/trace);a=RN(L*P);scaled=RN(a*65534);q=ROUND_HALF_EVEN_SCALED_TO_U16",
            "anisotropy_range_rule": "REQUIRE_FINITE_TRACE_GT_ZERO_AND_0_LE_A_LE_1_ELSE_NUMERIC_STOP",
            "q24_40_evidence_not_used_to_compute_anisotropy_or_filtration": True,
            "isotropic_anisotropy_rule": "EXACT_RATIONAL_EQUAL_EIGENVALUES_BYPASS_BINARY64_SCALAR_AND_SET_A_EQ_0_Q_EQ_0",
            "scalar_claimed_as_published_sherman_bend_invariant": False,
            "quantization_multiplier": _engineering(65_534),
            "quantization_rounding": "ROUND_HALF_TO_EVEN",
            "filtration_rule": "f=65534-q",
            "essential_death_sentinel": _engineering(65_535),
            "invalid_row_q": _engineering(0),
        },
        "complex": {
            "kind": "OCCUPIED_CELL_X_THREE_SCALE_PATH_FACE_GRAPH",
            "homology_dimensions": _engineering([0]),
            "higher_homology_closed": True,
            "coefficient_field": "F2",
            "spatial_adjacency": "SIX_FACE_SAME_SCALE",
            "scale_adjacency": "SAME_CELL_ADJACENT_SCALE",
            "occupied_cell_ordinal": "ZERO_BASED_INDEX_IN_LEAF_AGGREGATE_MORTON_ASCENDING_ORDER",
            "vertex_index_rule": "scale_u8_TIMES_O_PLUS_occupied_cell_ordinal_U32",
            "spatial_edge_rule": "EMIT_ONLY_POSITIVE_X_Y_Z_FACE_NEIGHBORS_LOWER_VERTEX_IS_NUMERIC_MIN_AXIS_0_1_2",
            "scale_edge_rule": "EMIT_SCALE_S_TO_S_PLUS_1_LOWER_VERTEX_IS_SMALLER_SCALE_AXIS_255",
            "edge_table_order": "edge_kind_THEN_axis_THEN_lower_vertex_THEN_upper_vertex",
            "filtration": "LOWER_STAR_ON_INTEGER_F",
            "vertex_filtration_rule": "VERTEX_EVENT_F_EQUALS_VERTEX_FIELD_F_U16",
            "edge_filtration_rule": "EDGE_EVENT_F_EQUALS_MAX_OF_ENDPOINT_VERTEX_F_U16",
            "event_kind_enum": {"VERTEX": 0, "EDGE": 1},
            "vertex_event_tie_key": "vertex_index_u32",
            "edge_event_tie_key": "edge_kind_u8_THEN_axis_u8_THEN_lower_vertex_u32_THEN_upper_vertex_u32",
            "order": "(f_u16,event_kind_u8,vertex_index_u32_IF_VERTEX_ELSE_edge_kind_u8_axis_u8_lower_vertex_u32_upper_vertex_u32)",
            "elder_rule": "SMALLER_BIRTH_F_SURVIVES_THEN_SMALLER_NUMERIC_BIRTH_VERTEX_INDEX_U32",
            "zero_lifetime_intervals_retained": True,
            "essential_interval_per_final_connected_component": True,
            "algorithm": "DETERMINISTIC_UNION_FIND_UNION_BY_RANK_PATH_COMPRESSION",
        },
        "component_semantics": {
            "output_is_bend_prediction": False,
            "output_is_line_or_owner": False,
            "output_semantics": "CONNECTED_ANISOTROPY_REGION_IN_SPACE_SCALE_GRAPH",
            "attributed_requires_q_positive": True,
            "minimum_integer_lifetime": _engineering(2),
            "minimum_distinct_scales": _engineering(2),
            "minimum_spatial_face_edges": _engineering(1),
            "membership_snapshot_for_finite_interval": "LOSING_OWNER_COMPONENT_IMMEDIATELY_BEFORE_DEATH_EDGE_IN_TOTAL_ORDER",
            "membership_snapshot_for_essential_interval": "SURVIVING_OWNER_COMPONENT_AFTER_ALL_VERTICES_AND_EDGES",
            "same_filtration_tie_block_policy": "NO_BLOCK_COLLAPSE_EACH_ELEMENT_PROCESSED_IN_FULL_TOTAL_ORDER",
            "component_summary_state": "SCALE_BITMASK_SPATIAL_FACE_EDGE_COUNT_ALL_VALID_BIT",
            "scale_mask_u8_rule": "BIT_scale_u8_IS_SET_FOR_EACH_PRESENT_SCALE_BITS_3_THROUGH_7_MUST_BE_ZERO",
            "spatial_face_edge_count_rule": "FOR_EACH_PROCESSED_SPATIAL_EDGE_EXACTLY_ONCE_IF_ROOTS_DIFFER_MERGED_COUNT=COUNT_A+COUNT_B+1_ELSE_INCREMENT_COMPONENT_COUNT_BY_1_SCALE_EDGES_ADD_COUNTS_WITHOUT_INCREMENT",
            "finite_snapshot_excludes_death_edge": True,
            "structural_dsu_parent_and_persistence_owner_are_separate": True,
            "structural_union_rule": "UNION_BY_RANK_THEN_STRUCTURAL_ROOT_ID",
            "persistence_owner_rule": "ELDER_RULE_INDEPENDENT_OF_STRUCTURAL_PARENT",
            "all_contributing_field_rows_must_be_valid": True,
            "disposition_row_order": "arm_u8_THEN_birth_vertex_index_u32",
            "finite_death_row_rule": "death_row_id_digest_128_EQUALS_DEATH_CAUSING_EDGE_ID_DIGEST_128",
            "finite_lifetime_rule": "lifetime_u16_EQUALS_death_f_u16_MINUS_birth_f_u16",
            "essential_death_row_rule": "DOMAIN_SEPARATED_ESSENTIAL_SENTINEL_PER_CASE_ARM_AND_BIRTH_ROW",
            "essential_death_f_rule": "death_f_or_essential_u16_EQUALS_65535",
            "essential_lifetime_rule": "lifetime_u16_EQUALS_65535_MINUS_birth_f_u16",
            "minimum_attributed_intervals_per_case": _engineering(1),
            "active_field_fraction_maximum": _fraction(3, 4),
            "attributed_interval_fraction_maximum": _fraction(1, 4),
        },
        "isotropic_counterfactual": {
            "kind": "OCCUPANCY_TRACE_ADJACENCY_SCALAR_MASS_PRESERVING_ISOTROPIZATION_V1",
            "purpose": "IMPLEMENTATION_CAUSAL_DEPENDENCY_AND_HIDDEN_CHANNEL_INTEGRITY_NEGATIVE_ONLY",
            "zero_anisotropy_survivors_expected_by_construction": True,
            "scientific_nuisance_origin_falsification_claimed": False,
            "does_not_distinguish_physical_bend_anisotropy_from_boundary_hole_density_or_quantization_anisotropy": True,
            "separate_nuisance_interventions_require_new_professor_authority": True,
            "tensor_rule": "C_iso=(trace(C)/3)*I represented as exact rational trace_over_3",
            "preserved_domains": [
                "partition",
                "occupied_cells",
                "support_membership_digest",
                "raw_count_diagnostics",
                "scalar_mass",
                "tensor_trace",
                "validity_and_rejection_state",
                "space_scale_adjacency",
                "arm_independent_source_cell_relation",
                "within_arm_canonical_order",
                "arm_independent_node_cell_support_and_edge_ids",
            ],
            "field_row_pairing_key": "case_token_digest_sha256_THEN_scale_u8_THEN_morton_u64",
            "baseline_to_isotropic_arm_map": "BASELINE_0_TO_ISOTROPIC_1",
            "arm_specific_field_row_ids_recomputed_from_locked_preimages": True,
            "arm_specific_rejection_death_and_interval_ids_recomputed_transitively": True,
            "baseline_and_isotropic_field_row_ids_byte_identical": False,
            "cross_arm_field_row_id_equality_policy": "UNEQUAL_CANONICAL_PREIMAGE_TRUNCATED_DIGEST_COLLISION_TERMINAL_STOP",
            "rejection_row_id_rule": "EXACT_OWN_ARM_FIELD_ROW_ID_DIGEST_128",
            "persistence_birth_row_id_rule": "EXACT_OWN_ARM_FIELD_ROW_ID_DIGEST_128",
            "finite_death_edge_ids_arm_independent": True,
            "isotropic_q_exact": _engineering(0),
            "baseline_attributed_denominator_minimum": _engineering(1),
            "allowed_anisotropy_attributed_survivors": _engineering(0),
            "complete_field_and_h0_rebuild_required": True,
            "same_source_not_new_scan": True,
        },
        "deterministic_roles": {
            "roles": [
                "BASE_A",
                "BASE_B",
                "INPUT_ORDER_PERMUTED",
                "INDEPENDENT_VALIDATOR",
            ],
            "role_count": _engineering(4),
            "each_role_emits_complete_baseline_and_isotropic_payload": True,
            "independent_validator_imports_semantic_core": False,
            "canonical_table_bytes_must_match": True,
            "disagreement_policy": "TERMINAL_STOP_NO_TOLERANCE_REPAIR",
        },
        "resources": {
            "worker_count_per_role": _engineering(1),
            "blas_thread_count": _engineering(1),
            "wall_seconds_per_role_maximum": _engineering(300),
            "cpu_seconds_per_role_maximum": _engineering(300),
            "application_array_bytes_maximum": _engineering(1_073_741_824),
            "static_process_tree_bytes_maximum": _engineering(4_294_967_296),
            "measured_process_tree_watchdog_bytes": _engineering(12_884_901_888),
            "enterprise_rss_forbidden_threshold_bytes": _engineering(17_179_869_184),
            "watchdog_poll_milliseconds": _engineering(10),
            "scratch_bytes_per_role_maximum": _engineering(268_435_456),
            "serialized_bytes_per_role_maximum": _engineering(67_108_864),
            "darwin_watchdog_is_not_kernel_cgroup": True,
            "bounded_parallelism_before_one_role_measured": False,
        },
        "work_model": {
            "coordinate_radix_byte_passes": _engineering(24),
            "stable_point_id_and_rank_radix_byte_passes": _engineering(64),
            "support_probe_scalar_steps_per_offset": _engineering(16),
            "support_connectivity_neighbor_steps_per_offset": _engineering(6),
            "opposite_balance_steps_per_offset": _engineering(1),
            "percentile_radix_byte_passes_per_offset": _engineering(4),
            "int128_moment_scalar_steps_per_offset": _engineering(64),
            "growth_and_gate_scalar_steps_per_vertex": _engineering(64),
            "jacobi_scalar_steps_per_baseline_vertex": _engineering(8_192),
            "field_quantization_and_disposition_steps_per_vertex_arm": _engineering(256),
        },
        "allocation_model": {
            "raw_parser_and_chunk_buffers_bytes": _engineering(67_108_864),
            "canonical_xyz_bytes_per_record": _engineering(24),
            "coordinate_radix_buffer_copies": _engineering(2),
            "coordinate_radix_index_bytes_per_record_per_buffer": _engineering(8),
            "coordinate_radix_index_buffer_copies": _engineering(2),
            "point_id_and_sample_rank_bytes_per_record": _engineering(64),
            "hash_radix_bytes_per_record_per_buffer": _engineering(64),
            "hash_radix_buffer_copies": _engineering(2),
            "sampled_xyz_and_ids_bytes_per_sample": _engineering(88),
            "partition_work_bytes_per_occupied_cell": _engineering(256),
            "field_and_moment_work_bytes_per_vertex": _engineering(256),
            "adjacency_work_bytes_per_edge": _engineering(32),
            "union_find_work_bytes_per_vertex_per_arm": _engineering(64),
            "baseline_plus_isotropic_arm_count": _engineering(2),
            "serializer_buffer_count": _engineering(2),
        },
        "static_allocation_reserves": {
            "python_or_native_runtime_bytes": _engineering(536_870_912),
            "allocator_fragmentation_bytes": _engineering(268_435_456),
            "native_library_workspace_bytes": _engineering(268_435_456),
            "semantic_process_stack_and_misc_bytes": _engineering(67_108_864),
            "preallocate_all_semantic_workspaces": True,
            "python_object_per_point_cell_edge_or_row_allowed": False,
            "allocation_plan_mismatch_policy": "TERMINAL_STOP_BEFORE_GEOMETRY",
        },
        "terminal": {
            "sole_pass_token": "CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET",
            "sole_stop_token": "INVALID_CII_PERSISTENCE_PREFLIGHT_STOP",
            "pass_authorizes_gold_benchmark_or_runtime": False,
            "no_modified_rerun_after_terminal_stop": True,
        },
        "numeric_provenance_rule": (
            "Every algorithmic number in an object carrying a provenance field is "
            + ENGINEERING_PROVENANCE
        ),
    }
    return _self_bound_json(config, "configuration_sha256")


def _field(name: str, type_name: str, size: int) -> dict[str, Any]:
    return {"name": name, "type": type_name, "bytes": size}


def _table(name: str, row_size: int, maximum_rows_formula: str, fields: Iterable[dict[str, Any]]) -> dict[str, Any]:
    field_list = list(fields)
    if sum(int(item["bytes"]) for item in field_list) != row_size:
        raise ValueError(f"schema field widths do not sum to row size: {name}")
    return {
        "name": name,
        "row_size_bytes": row_size,
        "maximum_rows_formula": maximum_rows_formula,
        "fixed_width": True,
        "variable_width_extensions_allowed": False,
        "fields": field_list,
    }


def _schema() -> dict[str, Any]:
    tables = [
        _table(
            "partition_node",
            88,
            "2*O-1",
            [
                _field("node_id_sha256", "BYTES32", 32),
                _field("morton_prefix_u64", "U64_LE", 8),
                _field("subtree_membership_digest_sha256", "BYTES32", 32),
                _field("subtree_point_count_u32", "U32_LE", 4),
                _field("occupied_leaf_count_u32", "U32_LE", 4),
                _field("depth_u8", "U8", 1),
                _field("child_mask_u8", "U8", 1),
                _field("status_u16", "U16_LE", 2),
                _field("reserved_zero_u32", "U32_LE", 4),
            ],
        ),
        _table(
            "leaf_aggregate",
            64,
            "O",
            [
                _field("cell_id_sha256", "BYTES32", 32),
                _field("morton_u64", "U64_LE", 8),
                _field("membership_digest_128", "BYTES16", 16),
                _field("unique_point_count_u32", "U32_LE", 4),
                _field("raw_record_multiplicity_sum_u32", "U32_LE", 4),
            ],
        ),
        _table(
            "cell_scale_field",
            128,
            "2*S*O",
            [
                _field("row_id_digest_128", "BYTES16", 16),
                _field("support_membership_digest_128", "BYTES16", 16),
                _field("morton_u64", "U64_LE", 8),
                _field("trace_q_i64", "I64_LE", 8),
                _field("lambda1_q_i64", "I64_LE", 8),
                _field("lambda2_q_i64", "I64_LE", 8),
                _field("lambda3_q_i64", "I64_LE", 8),
                _field("mass_q_i64", "I64_LE", 8),
                _field("arm_u8", "U8", 1),
                _field("scale_u8", "U8", 1),
                _field("status_u16", "U16_LE", 2),
                _field("support_count_u32", "U32_LE", 4),
                _field("raw_count_p10_u32", "U32_LE", 4),
                _field("raw_count_p90_u32", "U32_LE", 4),
                _field("balance_numerator_u32", "U32_LE", 4),
                _field("balance_denominator_u32", "U32_LE", 4),
                _field("growth12_gate_code_i32", "I32_LE", 4),
                _field("growth23_gate_code_i32", "I32_LE", 4),
                _field("anisotropy_q_u16", "U16_LE", 2),
                _field("filtration_u16", "U16_LE", 2),
                _field("stage_mask_u32", "U32_LE", 4),
                _field("first_rejection_u16", "U16_LE", 2),
                _field("flags_u16", "U16_LE", 2),
                _field("eigenvalue_denominator_u32", "U32_LE", 4),
            ],
        ),
        _table(
            "space_scale_adjacency",
            32,
            "(4*S-1)*O",
            [
                _field("edge_id_digest_128", "BYTES16", 16),
                _field("lower_vertex_u32", "U32_LE", 4),
                _field("upper_vertex_u32", "U32_LE", 4),
                _field("edge_kind_u8", "U8", 1),
                _field("axis_u8", "U8", 1),
                _field("status_u16", "U16_LE", 2),
                _field("reserved_zero_u32", "U32_LE", 4),
            ],
        ),
        _table(
            "persistence_disposition",
            64,
            "2*S*O",
            [
                _field("interval_id_digest_128", "BYTES16", 16),
                _field("birth_row_id_digest_128", "BYTES16", 16),
                _field("death_row_id_digest_128", "BYTES16", 16),
                _field("arm_u8", "U8", 1),
                _field("status_u8", "U8", 1),
                _field("reason_u16", "U16_LE", 2),
                _field("birth_f_u16", "U16_LE", 2),
                _field("death_f_or_essential_u16", "U16_LE", 2),
                _field("lifetime_u16", "U16_LE", 2),
                _field("scale_mask_u8", "U8", 1),
                _field("spatial_face_edge_count_u32", "U32_LE", 4),
                _field("flags_u8", "U8", 1),
            ],
        ),
        _table(
            "rejection",
            32,
            "2*S*O",
            [
                _field("row_id_digest_128", "BYTES16", 16),
                _field("arm_u8", "U8", 1),
                _field("scale_u8", "U8", 1),
                _field("first_rejection_u16", "U16_LE", 2),
                _field("stage_mask_u32", "U32_LE", 4),
                _field("denominator_id_u16", "U16_LE", 2),
                _field("reserved_zero_bytes6", "BYTES6", 6),
            ],
        ),
    ]
    schema: dict[str, Any] = {
        "schema_version": 1,
        "architecture_token": ARCHITECTURE_TOKEN,
        "primitive_token": PRIMITIVE_TOKEN,
        "encoding": {
            "byte_order": "LITTLE_ENDIAN",
            "metadata_json": "U64_LE_LENGTH_THEN_UTF8_SORTED_KEY_NO_NAN_CANONICAL_JSON_THEN_ZERO_FILL_TO_FIXED_REGION_END",
            "float_policy": "FINITE_ONLY_NEGATIVE_ZERO_NORMALIZED_ROUND_HALF_EVEN",
            "unknown_fields_allowed": False,
            "trailing_bytes_allowed": False,
            "archive_or_object_payload_allowed": False,
        },
        "tables": tables,
        "metadata_reserve_bytes": 1_048_576,
        "metadata_region_encoding": {
            "total_bytes": 1_048_576,
            "length_prefix_offset": 0,
            "length_prefix_type": "U64_LE",
            "json_payload_offset": 8,
            "json_payload_length_rule": "EXACT_VALUE_OF_U64_LE_PREFIX",
            "json_payload_rule": "EXACT_CANONICAL_JSON_BYTES_NO_BOM_NO_TRAILING_WHITESPACE",
            "padding_rule": "EVERY_BYTE_AFTER_JSON_THROUGH_OFFSET_1048575_MUST_EQUAL_0X00",
            "first_table_offset": 1_048_576,
        },
        "required_metadata_domains": [
            "census",
            "resource",
            "attestation",
            "table_descriptors",
            "source_points_serialized",
        ],
        "closed_metadata_schema": {
            "unknown_keys_allowed": False,
            "arbitrary_strings_allowed": False,
            "arbitrary_arrays_allowed": False,
            "arbitrary_nested_objects_allowed": False,
            "maximum_encoded_bytes": 1_048_568,
            "top_level_keys_exact": [
                "schema_version",
                "case_token_digest_sha256",
                "raw_source_sha256",
                "configuration_sha256",
                "schema_sha256",
                "code_sha256",
                "dependency_lock_sha256",
                "partition_commitment_sha256",
                "table_directory_sha256",
                "source_points_serialized",
                "census",
                "resource",
                "attestation",
                "table_descriptors",
            ],
            "digest_fields": {
                "type": "LOWERCASE_HEX_SHA256",
                "encoded_length": 64,
            },
            "census_keys_exact": [
                "raw_records",
                "finite_records",
                "unique_records",
                "sampled_unique_records",
                "occupied_cells_O",
                "partition_rows",
                "field_rows",
                "adjacency_rows",
                "persistence_rows",
                "rejection_rows",
                "moment_valid_rows",
                "attributed_intervals",
                "isotropic_survivors",
            ],
            "census_value_type": "U64_JSON_INTEGER",
            "resource_keys_exact": [
                "cpu_nanoseconds",
                "wall_nanoseconds",
                "peak_process_tree_rss_bytes",
                "scratch_bytes",
                "serialized_bytes",
                "watchdog_poll_milliseconds",
                "resource_cap_tripped",
            ],
            "resource_value_types": "FIXED_U64_JSON_INTEGERS_PLUS_ONE_BOOLEAN",
            "attestation_keys_exact": [
                "semantic_core_imported_by_independent_validator",
                "filename_path_family_identity_semantics_used",
                "gold_truth_cad_render_benchmark_accessed",
                "source_points_serialized",
                "top_k_reservoir_tail_omission_or_clipping_used",
                "complete_baseline_and_isotropic_payload",
                "deterministic_rebuild_match",
                "input_order_match",
                "independent_validator_match",
                "all_truncated_digest_collisions_checked",
            ],
            "attestation_expected_values": {
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
            "table_descriptors": {
                "fixed_length": 6,
                "canonical_order": [
                    "partition_node",
                    "leaf_aggregate",
                    "cell_scale_field",
                    "space_scale_adjacency",
                    "persistence_disposition",
                    "rejection",
                ],
                "keys_exact": [
                    "name_enum",
                    "row_count_u64",
                    "row_size_u32",
                    "offset_u64",
                    "byte_length_u64",
                    "sha256",
                ],
            },
            "coordinate_valued_case_metadata_forbidden": [
                "root_origin",
                "bounding_box",
                "axis_extrema",
                "cell_side",
                "physical_radius",
                "xyz",
                "centroid",
                "first_moment",
            ],
            "binding_recomputation_rules": {
                "case_token_digest_sha256": "RECOMPUTE_FROM_RAW_SOURCE_SHA256_USING_CONFIGURATION_RULE",
                "raw_source_sha256": "RECOMPUTE_FROM_EXACT_RAW_PLY_BYTES",
                "configuration_sha256": "COMPARE_TO_SIGNED_PREEXECUTION_LOCK",
                "schema_sha256": "COMPARE_TO_SIGNED_PREEXECUTION_LOCK",
                "code_sha256": "COMPARE_TO_SIGNED_PREEXECUTION_LOCK",
                "dependency_lock_sha256": "COMPARE_TO_SIGNED_PREEXECUTION_LOCK",
                "partition_commitment_sha256": "RECOMPUTE_FROM_CANONICAL_PARTITION_TABLE_BYTES",
                "table_directory_sha256": "RECOMPUTE_FROM_CANONICAL_SIX_DESCRIPTOR_BINARY_DIRECTORY",
            },
            "trailing_bytes_allowed": False,
        },
        "complete_domain_assertions": [
            "sum(leaf_aggregate.unique_point_count)==sampled_unique_point_count",
            "O<=partition_node_rows<=2*O-1",
            "leaf_aggregate_rows==O",
            "cell_scale_field_rows==2*S*O",
            "(S-1)*O<=space_scale_adjacency_rows<=(4*S-1)*O",
            "persistence_disposition_rows==2*S*O",
            "rejection_rows==2*S*O",
            "declared_table_bytes==row_count*row_size_bytes",
            "declared_container_bytes==header_plus_all_tables_no_trailing_bytes",
        ],
        "source_points_serialized": False,
        "forbidden_field_tokens": list(FORBIDDEN_SCHEMA_TOKENS),
        "explicitly_forbidden_payloads": [
            "XYZ_COLUMNS_OR_ARRAYS",
            "PER_POINT_IDS_OR_MEMBERSHIP_ROWS",
            "LEAF_CENTROIDS_OR_FIRST_MOMENTS",
            "RAW_OR_CENTERED_POINT_SUBSETS",
            "NEIGHBOR_OR_MEMBER_LISTS",
            "SINGLETON_RECONSTRUCTABLE_MOMENTS",
            "PICKLE_NPZ_ARCHIVE_OBJECT_OR_EXTENSION_PAYLOAD",
            "VARIABLE_WIDTH_HIDDEN_TAIL",
        ],
        "authority": _authority_state(),
    }
    return _self_bound_json(schema, "schema_sha256")


def _resource_proof(config: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    o_max = int(config["partition"]["logical_cell_cap"]["value"])
    s_count = int(config["scales"]["scale_count"]["value"])
    n_max = int(config["input"]["sample_cap"]["value"])
    m_max = int(config["source_envelope_observed"]["maximum_finite_coordinate_records_per_scan"])
    raw_bytes_max = int(config["source_envelope_observed"]["maximum_raw_bytes_per_scan"])
    role_count = int(config["deterministic_roles"]["role_count"]["value"])
    radius_multipliers = list(config["scales"]["radius_multipliers"]["value"])
    arm_count = int(config["allocation_model"]["baseline_plus_isotropic_arm_count"]["value"])

    rows = {
        "partition_node": 2 * o_max - 1,
        "leaf_aggregate": o_max,
        "cell_scale_field": 2 * s_count * o_max,
        "space_scale_adjacency": (4 * s_count - 1) * o_max,
        "persistence_disposition": 2 * s_count * o_max,
        "rejection": 2 * s_count * o_max,
    }
    table_by_name = {str(row["name"]): row for row in schema["tables"]}
    bytes_by_table = {
        name: count * int(table_by_name[name]["row_size_bytes"])
        for name, count in rows.items()
    }
    metadata = int(schema["metadata_reserve_bytes"])
    payload_bytes = sum(bytes_by_table.values()) + metadata
    hard_cap = int(config["resources"]["serialized_bytes_per_role_maximum"]["value"])

    stencil_widths = [(2 * int(value) + 1) ** 3 for value in radius_multipliers]
    vertices = s_count * o_max
    edges = (4 * s_count - 1) * o_max
    moment_offsets = o_max * sum(stencil_widths)
    parent_depth_bound = math.ceil(math.log2(max(1, vertices)))
    work_model = config["work_model"]
    work_value = lambda name: int(work_model[name]["value"])
    work = {
        "coordinate_radix_byte_passes": work_value("coordinate_radix_byte_passes") * m_max,
        "stable_point_id_and_rank_radix_byte_passes": work_value("stable_point_id_and_rank_radix_byte_passes") * m_max,
        "partition_level_assignments": (int(config["partition"]["grid_depth"]["value"]) + 1) * n_max,
        "moment_lattice_offset_checks": moment_offsets,
        "support_probe_scalar_steps": work_value("support_probe_scalar_steps_per_offset") * moment_offsets,
        "support_connectivity_neighbor_steps": work_value("support_connectivity_neighbor_steps_per_offset") * moment_offsets,
        "opposite_balance_steps": work_value("opposite_balance_steps_per_offset") * moment_offsets,
        "percentile_radix_byte_passes": work_value("percentile_radix_byte_passes_per_offset") * moment_offsets,
        "int128_moment_scalar_steps": work_value("int128_moment_scalar_steps_per_offset") * moment_offsets,
        "growth_and_gate_scalar_steps": work_value("growth_and_gate_scalar_steps_per_vertex") * vertices,
        "jacobi_scalar_steps_baseline_only": work_value("jacobi_scalar_steps_per_baseline_vertex") * vertices,
        "baseline_plus_isotropic_quantization_and_disposition_steps": work_value("field_quantization_and_disposition_steps_per_vertex_arm") * arm_count * vertices,
        "adjacency_candidate_checks": edges,
        "baseline_plus_isotropic_union_find_parent_steps": arm_count * (vertices + 2 * edges) * parent_depth_bound,
    }
    work["major_operation_upper_bound_per_role"] = sum(work.values())

    allocation_model = config["allocation_model"]
    allocation_value = lambda name: int(allocation_model[name]["value"])
    allocation_ledger = {
        "raw_parser_and_chunk_buffers": allocation_value("raw_parser_and_chunk_buffers_bytes"),
        "all_canonical_xyz": m_max * allocation_value("canonical_xyz_bytes_per_record"),
        "coordinate_radix_primary_and_buffer": m_max * allocation_value("canonical_xyz_bytes_per_record") * allocation_value("coordinate_radix_buffer_copies"),
        "coordinate_radix_indices_primary_and_buffer": m_max * allocation_value("coordinate_radix_index_bytes_per_record_per_buffer") * allocation_value("coordinate_radix_index_buffer_copies"),
        "point_id_and_sample_rank": m_max * allocation_value("point_id_and_sample_rank_bytes_per_record"),
        "hash_radix_primary_and_buffer": m_max * allocation_value("hash_radix_bytes_per_record_per_buffer") * allocation_value("hash_radix_buffer_copies"),
        "sampled_xyz_and_ids": n_max * allocation_value("sampled_xyz_and_ids_bytes_per_sample"),
        "partition_work": o_max * allocation_value("partition_work_bytes_per_occupied_cell"),
        "field_and_moment_work": vertices * allocation_value("field_and_moment_work_bytes_per_vertex"),
        "adjacency_work": edges * allocation_value("adjacency_work_bytes_per_edge"),
        "baseline_plus_isotropic_union_find_work": vertices * allocation_value("union_find_work_bytes_per_vertex_per_arm") * arm_count,
        "serializer_double_buffer": hard_cap * allocation_value("serializer_buffer_count"),
        "fixed_logs_and_metadata": metadata,
    }
    allocation_total = sum(allocation_ledger.values())
    application_cap = int(config["resources"]["application_array_bytes_maximum"]["value"])
    fixed_reserve_ledger = {
        "python_or_native_runtime": int(config["static_allocation_reserves"]["python_or_native_runtime_bytes"]["value"]),
        "allocator_fragmentation": int(config["static_allocation_reserves"]["allocator_fragmentation_bytes"]["value"]),
        "native_library_workspace": int(config["static_allocation_reserves"]["native_library_workspace_bytes"]["value"]),
        "semantic_process_stack_and_misc": int(config["static_allocation_reserves"]["semantic_process_stack_and_misc_bytes"]["value"]),
    }
    fixed_reserve_total = sum(fixed_reserve_ledger.values())
    static_process_calculated = allocation_total + fixed_reserve_total
    static_process_cap = int(config["resources"]["static_process_tree_bytes_maximum"]["value"])
    if allocation_total > application_cap:
        raise ValueError("semantic allocation ledger exceeds application-array cap")
    if static_process_calculated > static_process_cap:
        raise ValueError("enumerated static process-tree proof exceeds configured cap")

    scratch_cap = int(config["resources"]["scratch_bytes_per_role_maximum"]["value"])
    scratch_required = 2 * hard_cap + metadata

    proof: dict[str, Any] = {
        "schema_version": 1,
        "architecture_token": ARCHITECTURE_TOKEN,
        "configuration_sha256": config["configuration_sha256"],
        "schema_sha256": schema["schema_sha256"],
        "constants": {
            "M_observed_max_finite_records": m_max,
            "raw_bytes_observed_max": raw_bytes_max,
            "N": n_max,
            "O_max": o_max,
            "S": s_count,
            "arms": arm_count,
            "roles": role_count,
            "stencil_widths": stencil_widths,
            "V_max": vertices,
            "E_max": edges,
            "union_find_parent_depth_bound": parent_depth_bound,
        },
        "maximum_rows_per_complete_role": rows,
        "actual_domain_formulas": {
            "partition_node": "O<=rows<=2*O-1",
            "leaf_aggregate": "O",
            "cell_scale_field": "2*S*O",
            "space_scale_adjacency": "(S-1)*O<=rows<=(4*S-1)*O",
            "persistence_disposition": "2*S*O",
            "rejection": "2*S*O",
        },
        "maximum_bytes_per_complete_role": bytes_by_table,
        "metadata_reserve_bytes": metadata,
        "calculated_payload_bytes_per_complete_role": payload_bytes,
        "hard_payload_cap_bytes_per_complete_role": hard_cap,
        "payload_headroom_bytes": hard_cap - payload_bytes,
        "calculated_rows_including_metadata_record": sum(rows.values()) + 1,
        "payloads_include_baseline_and_complete_isotropic_rebuild": True,
        "all_roles_emit_complete_payload": True,
        "maximum_hard_cap_bytes_per_case_all_roles": hard_cap * role_count,
        "maximum_hard_cap_bytes_source_manifest_all_roles": hard_cap * role_count * SOURCE_MANIFEST_CASE_COUNT,
        "work_upper_bounds": work,
        "semantic_allocation_ledger_bytes": allocation_ledger,
        "semantic_allocation_ledger_total_bytes": allocation_total,
        "application_array_cap_bytes": application_cap,
        "semantic_allocation_headroom_bytes": application_cap - allocation_total,
        "fixed_reserve_ledger_bytes": fixed_reserve_ledger,
        "fixed_reserve_total_bytes": fixed_reserve_total,
        "static_process_tree_calculated_bytes": static_process_calculated,
        "static_process_tree_configured_cap_bytes": static_process_cap,
        "static_process_tree_headroom_bytes": static_process_cap - static_process_calculated,
        "measured_process_tree_watchdog_bytes": config["resources"]["measured_process_tree_watchdog_bytes"]["value"],
        "enterprise_rss_forbidden_threshold_bytes": config["resources"]["enterprise_rss_forbidden_threshold_bytes"]["value"],
        "scratch_required_bytes": scratch_required,
        "scratch_cap_bytes": scratch_cap,
        "generic_h1_h2_boundary_matrix_reduction_authorized": False,
        "h0_union_find_linear_domain_only": True,
        "top_k_reservoir_tail_omission_or_posthoc_clipping": False,
        "provenance": ENGINEERING_PROVENANCE,
    }
    return _self_bound_json(proof, "resource_proof_sha256")


def _safety_contract(config: Mapping[str, Any], schema: Mapping[str, Any], proof: Mapping[str, Any]) -> dict[str, Any]:
    safety: dict[str, Any] = {
        "schema_version": 1,
        "architecture_token": ARCHITECTURE_TOKEN,
        "primitive_token": PRIMITIVE_TOKEN,
        "configuration_sha256": config["configuration_sha256"],
        "schema_sha256": schema["schema_sha256"],
        "resource_proof_sha256": proof["resource_proof_sha256"],
        "authority": _authority_state(),
        "lock_decision_tokens": list(LOCK_DECISION_TOKENS),
        "next_step_tokens": list(NEXT_STEP_TOKENS),
        "valid_decision_pairs": VALID_DECISION_PAIRS,
        "required_execution_authority_line": "EXECUTION_AUTHORITY: NONE",
        "pass_meaning": "PAPER_LOCK_COHERENT_FOR_ONE_FURTHER_PREEXECUTION_PROFESSOR_PACKET_ONLY",
        "terminal_tokens_for_any_later_preflight": {
            "pass": config["terminal"]["sole_pass_token"],
            "stop": config["terminal"]["sole_stop_token"],
        },
        "terminal_stop_conditions": [
            "DIGITAL_SHELL_MISREPRESENTED_AS_PUBLISHED_SOLID_VOLUME_INVARIANT",
            "SOURCE_CONFIG_SCHEMA_CODE_OR_DEPENDENCY_HASH_MISMATCH",
            "ANY_SAMPLED_POINT_NOT_ASSIGNED_EXACTLY_ONCE",
            "MISSING_DUPLICATE_OR_EXTRA_COMPLETE_DOMAIN_ROW",
            "TRUNCATED_DIGEST_COLLISION_OR_SUBTREE_COMMITMENT_FAILURE",
            "ARM_SPECIFIC_IDENTIFIER_OR_SOURCE_CELL_CORRESPONDENCE_FAILURE",
            "CLOSED_METADATA_SCHEMA_ATTESTATION_OR_OFFSET_FAILURE",
            "NONZERO_METADATA_PADDING_OR_LENGTH_PREFIX_FAILURE",
            "IDENTIFIER_COMMITMENT_KNOWN_ANSWER_OR_SIGNED_BINDING_FAILURE",
            "CARDINALITY_BYTE_SCRATCH_WORK_OR_RSS_BOUND_VIOLATION",
            "SOURCE_POINT_OR_FORBIDDEN_PAYLOAD_SERIALIZATION",
            "NONFINITE_OVERFLOW_OR_EIGENSOLVER_DISAGREEMENT",
            "JACOBI_OR_ANISOTROPY_OPERATION_ORDER_DISAGREEMENT",
            "BASE_REBUILD_PERMUTATION_OR_INDEPENDENT_VALIDATOR_DISAGREEMENT",
            "TAMPER_REGRESSION_FAILURE",
            "ZERO_BASELINE_ANISOTROPY_ATTRIBUTED_DENOMINATOR",
            "TRACE_EIGENVALUE_RATIONAL_DENOMINATOR_MISMATCH",
            "ANY_ISOTROPIC_ANISOTROPY_ATTRIBUTED_SURVIVOR",
            "ISOTROPIC_NEGATIVE_MISREPRESENTED_AS_BASELINE_NUISANCE_OR_SELECTIVITY_FALSIFIER",
            "CHANGED_NEGATIVE_OCCUPANCY_TRACE_MASS_VALIDITY_OR_ADJACENCY",
            "NONVACUITY_OR_SELECTIVITY_GATE_FAILURE",
            "GOLD_CAD_LABEL_RENDER_BENCHMARK_RUNTIME_OR_PRODUCTION_ACCESS",
        ],
        "no_modified_rerun_after_terminal_stop": True,
    }
    return _self_bound_json(safety, "safety_contract_sha256")


def _selection_response() -> str:
    if not SELECTION_RESPONSE_PATH.is_file():
        raise FileNotFoundError(SELECTION_RESPONSE_PATH)
    observed = _sha_file(SELECTION_RESPONSE_PATH)
    if observed != SELECTION_RESPONSE_SHA256:
        raise ValueError("preserved Cubical-IIP selection response SHA-256 mismatch")
    text = SELECTION_RESPONSE_PATH.read_text(encoding="utf-8")
    required = (
        f"ARCHITECTURE: {ARCHITECTURE_TOKEN}",
        "NEXT_STEP: BUILD_SELECTED_ARCHITECTURE_PREFLIGHT_LOCK_PACKET_ONLY",
        "EXECUTION_AUTHORITY=NONE",
        "exact isotropic same-trace falsifier",
        "complete—not truncated—evidence semantics",
        "sole terminal statuses",
    )
    for item in required:
        if item not in text:
            raise ValueError(f"selection response lacks required ruling: {item}")
    return text


def _correction_response() -> str:
    if not CORRECTION_RESPONSE_PATH.is_file():
        raise FileNotFoundError(CORRECTION_RESPONSE_PATH)
    observed = _sha_file(CORRECTION_RESPONSE_PATH)
    if observed != CORRECTION_RESPONSE_SHA256:
        raise ValueError("preserved Cubical-IIP V5 lock-audit response SHA-256 mismatch")
    text = CORRECTION_RESPONSE_PATH.read_text(encoding="utf-8")
    required = (
        "LOCK_DECISION: CUBICAL_IIP_V1_LOCK_AUDIT_REQUIRES_ONE_NONEXECUTION_CORRECTION",
        "NEXT_STEP: BUILD_ONE_CUBICAL_IIP_V1_LOCK_CORRECTION_PACKET_ONLY",
        "EXECUTION_AUTHORITY: NONE",
        "Reconcile the isotropic field-row identity contract",
        "no implementation could simultaneously obey both locked clauses",
    )
    for item in required:
        if item not in text:
            raise ValueError(f"correction response lacks required ruling: {item}")
    return text


def _source_manifest_bytes() -> bytes:
    if not SOURCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST_PATH)
    raw = SOURCE_MANIFEST_PATH.read_bytes()
    if _sha_bytes(raw) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest external SHA-256 mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source manifest must be a JSON object")
    internal = stop_packet._validate_content_binding(
        payload,
        hash_key="input_manifest_sha256",
        label="source manifest",
    )
    if internal != SOURCE_MANIFEST_INTERNAL_SHA256:
        raise ValueError("source manifest internal canonical hash mismatch")
    if payload.get("source_points_serialized") is not False:
        raise ValueError("source manifest safety field changed")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != SOURCE_MANIFEST_CASE_COUNT:
        raise ValueError("source manifest case domain changed")
    if any(not isinstance(case, Mapping) for case in cases):
        raise ValueError("source manifest case row must be an object")
    observed_max_finite = max(int(case["finite_coordinate_count"]) for case in cases)
    observed_max_raw = max(int(case["raw_byte_count"]) for case in cases)
    if observed_max_finite != SOURCE_MAX_FINITE_RECORDS:
        raise ValueError("source manifest maximum finite-record envelope changed")
    if observed_max_raw != SOURCE_MAX_RAW_BYTES:
        raise ValueError("source manifest maximum raw-byte envelope changed")
    return raw


def _lineage_ledger(config: Mapping[str, Any]) -> str:
    return f"""# Lineage, claims, and cross-bindings

## Exact decision lineage

- The prior PWTS stop-classification response is bound at SHA-256
  `{CLASSIFICATION_RESPONSE_SHA256}` and began with
  `CLASSIFICATION: PWTS_V1_PROTOCOL_INVALID_REPRESENTATION_UNRESOLVED` and
  `NEXT_STEP: BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY`.
- The exact prior post-PWTS selection prompt is bound at SHA-256
  `{PRIOR_SELECTION_PROMPT_SHA256}`.
- The selected GPT-5.6 Pro response is embedded byte-exact in file 00 and bound
  at SHA-256 `{SELECTION_RESPONSE_SHA256}`.
- It selected `{ARCHITECTURE_TOKEN}` and authorized only
  `BUILD_SELECTED_ARCHITECTURE_PREFLIGHT_LOCK_PACKET_ONLY`.
- The V5 GPT-5.6 Pro lock audit is also embedded byte-exact in file 00 and
  bound at SHA-256 `{CORRECTION_RESPONSE_SHA256}`. It found one isolated
  row-identity contradiction and authorized only
  `BUILD_ONE_CUBICAL_IIP_V1_LOCK_CORRECTION_PACKET_ONLY`.
- V7 keeps arm-bound row identifiers, pairs baseline and isotropic rows by the
  arm-neutral `(case_token_digest_sha256, scale_u8, morton_u64)` key, and
  recomputes each arm-specific dependent identifier from its unchanged locked
  preimage. No algorithmic, resource, schema-width, or scientific-claim change
  is made.
- The current configuration binds to `{config['configuration_sha256']}`.

## Frozen observed / inference / unknown ledger

| Class | Permitted statement |
| --- | --- |
| Observed | PWTS-V1 attempted only C001/BASE_A and exceeded its locked 1,024 seed-row evidence envelope before serialization; the only supported count is `seed_count >= 1025`. No trace, strip, certificate, deletion, replay, independent-validator, or accuracy result completed. |
| Observed | The source manifest binds C001-C037, 13,991,575 finite XYZ records, 448,084 exact duplicates, 699,881,650 raw bytes, no sampling, no semantics, and no source-point serialization. These are 37 scans, not evidence of 37 parts or families. |
| Inference | PWTS-V1's frozen seed definition and evidence envelope were incompatible on C001, so its protocol STOP was valid. |
| Unknown | Exact PWTS seed count and distribution, physical-bend meaning, downstream selectivity, behavior on other scans, and all bend accuracy remain unknown. |

## New architecture claim boundary

The selected paper family is narrowed to `{PRIMITIVE_TOKEN}`. Pottmann's
published ball-volume invariant assumes a domain bounded by a surface. An
unoriented raw surface point cloud does not provide that inside/outside domain.
This packet therefore specifies a digital union of equally weighted occupied
grid cubes. That occupied shell is an engineering extrapolation, not the
published solid-volume curvature invariant and not a claimed bend detector.

No PWTS or LPF semantic primitive, plane/fragment pair, seed maximum, tangent
trace, strip, certificate, mesh, CAD, heatmap, learned model, global solver,
ownership rule, or bend birth is inherited.

`EXECUTION_AUTHORITY=NONE`.
"""


def _literature_ledger() -> str:
    return f"""# Primary-source mechanism and engineering-assumption ledger

| Claim | Classification | Primary source / boundary |
| --- | --- | --- |
| Local kernel integrals and moments can provide stable multiscale geometric descriptors. | Literature-backed mechanism | Pottmann et al., DOI `10.1016/j.cagd.2008.01.002`. The paper does not establish this packet's occupied-shell discretization, constants, or Sherman accuracy. |
| Moment eigenstructure of kernel neighborhoods can encode curvature/principal-direction information. | Literature-backed mechanism | Pottmann et al., DOI `10.1016/j.cagd.2007.07.004`. Its bounded-domain/surface-patch mathematics is not silently transferred to sparse occupied cubes. |
| Digital integral-invariant curvature estimators can converge for defined digital shape boundaries. | Literature-backed mechanism | Coeurjolly, Lachaud, and Levallois, DOI `10.1016/j.cviu.2014.04.013`. This does not prove sparse-shell validity. |
| Persistent homology is defined for finite filtered complexes. | Literature-backed mechanism | Zomorodian and Carlsson, DOI `10.1007/s00454-004-1146-y`. |
| Cubical data can be processed without first triangulating it. | Literature-backed mechanism | Wagner, Chen, and Vuçini, DOI `10.1007/978-3-642-23175-9_7`. |
| Persistence diagrams are stable to bounded perturbations of a filtration. | Literature-backed mechanism | Cohen-Steiner, Edelsbrunner, and Harer, DOI `10.1007/s00454-006-1276-5`. This does not establish grid-phase, occupancy, or density invariance. |
| Equal mass per occupied cube, the three support radii, the `L*P` scalar, all coverage/density gates, H0-only space-scale graph, quantization, nonvacuity/selectivity gates, evidence caps, and runtime caps are appropriate for Sherman scans. | Engineering hypothesis | `{ENGINEERING_PROVENANCE}`; no cited paper establishes these choices or bend accuracy. |

The configuration contains no literature-derived Sherman threshold. Every
algorithmic numerical envelope is explicitly typed
`{ENGINEERING_PROVENANCE}`. The packet asks Pro to audit a paper specification,
not to infer accuracy or authorize execution.
"""


def _normative_spec(config: Mapping[str, Any], schema: Mapping[str, Any], proof: Mapping[str, Any]) -> str:
    return f"""# Normative `{PRIMITIVE_TOKEN}` paper lock

Configuration SHA-256: `{config['configuration_sha256']}`  
Schema SHA-256: `{schema['schema_sha256']}`  
Resource-proof SHA-256: `{proof['resource_proof_sha256']}`

## 1. Input and canonical sampling

Read raw finite PLY XYZ only; normalize negative zero; exact-deduplicate packed
binary64 triples; assign coordinate-only domain-separated SHA-256 IDs; retain
the 200,000 lowest sample ranks; sort by rank then point ID. A hash collision,
nonfinite coordinate, parser disagreement, or any semantic access to paths,
filenames, families, normals, colors, labels, CAD, renders, previous candidates,
or expected counts is terminal STOP. Every retained point is assigned once.
The bound source manifest contains at most
{proof['constants']['M_observed_max_finite_records']:,} finite records and
{proof['constants']['raw_bytes_observed_max']:,} raw bytes in one scan. The
pre-sampling coordinate and digest radix domains are bounded by that observed
record maximum, not by the 200,000 retained-sample cap.
The case token is content-derived as SHA-256 of the `CIIP_CASE_V1` tag and raw
source SHA-256 bytes; paths and filenames never enter it. File 05 fixes every
ASCII-plus-NUL domain tag, little-endian preimage, counted sequence, first-byte
digest truncation, and known-answer vector.

## 2. Deterministic cubical partition

Let `D` be the maximum sampled axis range, require `D>0`, set `R=(4/3)D`,
center a 32-by-32-by-32 depth-5 dyadic root cube on the sampled bounding-box
center, and set `h=R/32`. The construction leaves at least four cells of margin
on the widest axis. Assign each coordinate by exact `floor((x-o)/h)`; an exact
cell-boundary point goes to the upper-index cell and no epsilon branch exists.
An out-of-range result is STOP.
Morton encoding interleaves five-bit grid indices LSB-first as X at bit `3i`, Y
at `3i+1`, and Z at `3i+2`; a depth-`d` prefix is the high `3d` Morton bits
right-aligned in a zero-filled u64. Node IDs bind case digest, depth, and prefix;
cell IDs bind case digest and leaf Morton code. Subtree and leaf membership
preimages contain a u32 count followed by full point IDs sorted bytewise.

Logically split all occupied nodes breadth-first to depth five; order within a
level by Morton code then point ID. Serialize a Patricia occupied radix tree
with unary paths collapsed. At most 32,768 occupied leaf cells and 65,535 tree
rows exist. Every sampled point contributes to exactly one leaf aggregate; no
point, cell row, or rejection is selected by top-k, reservoir, tail omission,
or clipping. Partition node IDs and subtree membership commitments are full
32-byte SHA-256 values. Every remaining 128-bit row or membership digest is
collision-checked against its canonical preimage; an unequal-preimage collision
is terminal STOP.

## 3. Digital occupied-shell moment measure

Each occupied cell represents its full cube with equal semantic mass `h^3`,
regardless of raw point count. Raw count remains diagnostic. This defines a
voxelized surface shell, not the solid interior and not a literal Pottmann
ball-volume invariant. For multipliers `m=(2,3,4)`, support contains every
occupied cell whose integer center offset satisfies squared norm `<=m^2`.
Cube centers are included as whole cubes; exact cube-ball intersection is not
claimed.

Use signed int128 lattice counts, sums, and second sums. For support `J`,
construct `C=h^2*(mean((z-zbar)(z-zbar)^T)+I/12)`. The `I/12` term is the exact
central covariance of a full cube. Delay division until tensor construction.
Use exactly 12 cyclic Jacobi sweeps with pivot order `(0,1),(0,2),(1,2)`.
An exactly zero off-diagonal term uses the identity rotation; otherwise use the
locked tau/t/c/s formula in file 05, a full temporary 3-by-3 similarity update,
and exact symmetry mirroring. The rotation is fixed as `Jpp=Jqq=c`, `Jpq=s`,
`Jqp=-s`; each pivot computes `A_new=J^T*A_old*J` through two full 3-by-3
matrix products with `i,j,k` loops ascending, positive-zero accumulators, one
rounded multiply then one rounded add per term. The upper triangle is
authoritative and is copied exactly to the lower triangle. Correctly rounded
binary64 square root and round-to-nearest-ties-even are required, with no FMA
or extended precision. Reject a final maximum off-diagonal residual
above `2^-40*trace`; clamp a negative eigenvalue only when its magnitude is at
most `2^-40*trace`, otherwise STOP. Sort eigenvalues by value then original axis.
Quantize dimensionless trace/eigenvalues to signed Q24.40 by round-half-even;
mass is stored as integer support count in units of `h^3`. Nonfinite output,
overflow, or validator eigensolver disagreement is STOP.
For baseline rows, the three Q24.40 eigenvalue numerators and trace numerator
have denominator one and the numerators sum exactly. For isotropic rows, each
eigenvalue numerator equals the unchanged baseline trace numerator with
denominator three. No divisibility-by-three premise or rounded one-third is
allowed; the schema carries the denominator explicitly.

## 4. Coverage, boundary, hole, and density gates

Retain every cell-scale row. Apply first-rejection order: minimum support counts
9/21/37; exactly one connected component in the induced six-face support graph;
opposite-offset balance at least 11/20; both nested support-growth dimensions in
[5/4,11/4], tested by the four exact unsigned-int128 power inequalities in file
05 without logarithms; raw cell-count P90/max(1,P10) at most 32, using ascending
nearest-rank indices `ceil(n/10)-1` and `ceil(9n/10)-1` and the integer comparison
`P90 <= 32*max(1,P10)`; positive finite trace; then numeric validity. More than
9/10 invalid rows in a case is STOP. These are engineering gates and may reject
real physical-edge bends; no accuracy claim is made.

## 5. Integer anisotropy field

Order eigenvalues `lambda1>=lambda2>=lambda3>=0`, set
`L=(lambda1-lambda2)/trace`, `P=(lambda2-lambda3)/trace`, and `a=L*P`.
This product suppresses purely planar, line-like, and isotropic neighborhoods;
it is not a published Sherman bend invariant. Quantize
`q=round_half_even(65534*a)` and filter by `f=65534-q`; 65535 is reserved as
the essential-death sentinel. Invalid rows use q=0. All comparisons use integer
filtration values.
Compute this scalar only from the final unquantized binary64 Jacobi eigenvalues,
never from Q24.40 evidence. File 05 locks the exact rounded sequence for trace,
two differences, two divisions, multiplication, scaling by 65,534, and final
ties-even integer rounding. The isotropic arm bypasses that floating path and
sets `a=0,q=0` from its exact rational equal-eigenvalue construction.

## 6. H0-only finite space-scale complex

Vertices are all `(occupied_cell,scale)` pairs, at most 98,304. Edges connect
six-face spatial neighbors at the same scale and identical cells at adjacent
scales, at most 360,448. Compute lower-star H0 only over F2 by deterministic
union-find. H1/H2 and generic boundary-matrix reduction are closed because no
static sub-12-GiB worst-case reduction proof is supplied and shell holes have no
selected bend semantics.
Occupied-cell ordinal is the zero-based Morton-sorted leaf index; vertex index
is `scale*O+ordinal`. Emit only positive-axis spatial neighbors and adjacent
scale pairs. Lower/upper are numeric vertex minima/maxima; spatial axes are
0/1/2 and scale-edge axis is 255. Edge IDs are the first 16 bytes of the locked
domain-separated preimage hash and every unequal-preimage collision stops.

Order filtration events by `(f, vertex-before-edge, exact vertex-or-edge tie key)`.
Vertex ties use numeric vertex index; edge ties use kind, axis, lower numeric
vertex index, then upper numeric vertex index. The elder rule keeps smaller
birth-f, then smaller numeric birth vertex index.
Emit one persistence disposition per vertex, including zero-lifetime and every
essential class; the number of essential classes equals final graph connected
components.

## 7. Representation semantics and gold-free gates

An interval is anisotropy-attributed only if its birth q is positive, lifetime
is at least two integer units, its component scale bitmask contains at least two
scales, it has accumulated at least one spatial face edge, and its all-valid bit
is true. For a finite interval, snapshot that summary on the losing persistence
owner immediately before the death-causing edge in the full element total order;
do not collapse equal-filtration blocks. For an essential interval, snapshot
the surviving owner after all elements. Union-by-rank structural parents and
elder-rule persistence owners are separate state; structural attachment cannot
change ownership. It denotes a connected anisotropy region, not a bend, line,
owner, count, or prediction.
Every processed spatial edge contributes exactly once: a cross-component edge
merges counts as `countA+countB+1`, while an intra-component spatial edge
increments its component; scale edges merge counts without increment. A finite
pre-death snapshot excludes its death edge. Persistence rows are ordered by arm
then birth vertex. Finite death IDs equal the death-edge digest. Essential death
IDs use the locked case/arm/birth sentinel preimage, store death `65535`, and
store lifetime `65535-birth_f`.
File 05 fixes every status, rejection, reason, denominator, growth, stage-bit,
scale-mask, flag, reserved, and not-evaluated code. Bits `0,1,2` of the scale
mask correspond exactly to radii `2h,3h,4h`; higher bits and all flags/reserved
bytes are zero. Finite lifetime is exactly `death_f-birth_f`.

Each case must contain at least one attributed interval. Rows with positive q
may not exceed 3/4 of all field vertices, attributed intervals may not exceed
1/4 of vertices, and invalid rows may not exceed 9/10. Every numerator and
denominator is retained. Empty and saturated outputs are terminal representation
STOPs, not successful selectivity.

## 8. Direct same-trace isotropic implementation negative

For every field row replace `C` by the exact rational tensor `(trace(C)/3)I`.
Preserve partition, occupied cells, support membership, counts, scalar mass,
trace, validity/rejection state, scale, adjacency, and the one-to-one source-cell
relation byte-identically. Pair the baseline row `(arm=0,scale=s,morton=m)` with
the isotropic row `(arm=1,scale=s,morton=m)` by
`(case_token_digest_sha256,scale_u8,morton_u64)`. Preserve canonical
`(scale_u8,morton_u64)` order within each arm and the global
`(arm_u8,scale_u8,morton_u64)` order.

Do not preserve arm-bound IDs byte-identically. Recompute every isotropic field,
rejection, essential-death, and interval identifier from its unchanged locked
preimage using `arm_u8=1`; rejection and persistence birth IDs alias the field
ID from their own arm. Node, cell, support-membership, adjacency-edge, and finite
death-edge IDs remain byte-identical because their preimages are arm-independent.
The paired baseline and isotropic field IDs must differ. Equality for these
unequal canonical preimages is a truncated-digest collision and terminal STOP.
Rebuild the complete field and H0 complex; q_iso is exactly zero. Structural
zero-lifetime and essential background classes may remain but are never
anisotropy-attributed.

The baseline attributed denominator must be positive and the isotropic
anisotropy-attributed survivor count must be zero. Any changed arm-independent
payload, missing or duplicated counterpart key, copied baseline ID in an
isotropic row, noncanonical arm-specific identifier, broken transitive
rejection/death/interval binding, changed within-arm order, zero denominator,
or survivor is terminal STOP. This is a same-source intervention, not a
synthetic scan or replay rescue.
Zero anisotropy survivors are expected by construction because equal
eigenvalues force q=0 and attribution requires positive q. This negative can
therefore falsify only implementation dependency integrity or reveal a hidden
channel; it cannot distinguish physical-bend anisotropy from boundary, hole,
density, or voxel-quantization anisotropy in the baseline. Those scientific
nuisance origins remain unresolved, no selectivity claim follows from a clean
negative, and any separate nuisance intervention requires new Pro authority.

## 9. Complete schemas, bounds, and determinism

The fixed-width allowlisted schema contains every partition node, occupied leaf,
baseline and isotropic field row, shared edge, baseline and isotropic persistence
disposition, and baseline and isotropic rejection row. Its calculated maximum is
{proof['calculated_payload_bytes_per_complete_role']:,} bytes and
{proof['calculated_rows_including_metadata_record']:,} rows per complete role,
below the 64 MiB hard cap. Four roles—BASE_A, BASE_B, input-order permutation,
and an independently implemented raw-source validator—each emit the complete
baseline-plus-negative payload. No raw coordinate, point ID, centroid, first
moment, member list, variable-width field, hidden tail, or archive is legal.
For actual occupied-cell count `O`, the complete domains are exactly `O` leaf
rows, `2*S*O` field rows, `2*S*O` persistence rows, and `2*S*O` rejection rows;
partition rows are in `[O,2*O-1]`, and adjacency rows are in
`[(S-1)*O,(4*S-1)*O]`. A sparse role may not pad any table to the 32,768-cell
maximum. Metadata uses a closed exact key schema, exact six-table directory,
fixed booleans, contiguous offsets, canonical hashes, and no coordinate-valued
case metadata.
The one-MiB metadata region begins with an exact u64-le JSON byte length, then
the byte-exact canonical JSON, then only zero bytes through offset 1,048,575;
the first table starts at 1,048,576. Any alternate whitespace, bad length, or
nonzero padding byte is a hidden-payload STOP. Case, partition, and directory
commitments are recomputed from their locked preimages; raw-source, code, and
dependency hashes must equal the signed preexecution lock. File 05 contains
known-answer vectors for every identifier/commitment class.

Exact table bytes and hashes must match after canonical quantization. The
independent validator may share only the declarative configuration/schema and
neutral hash/container utilities; it may not import the semantic implementation.
Any mismatch is terminal and cannot be repaired by tolerance or reordering.

The work ledger charges {proof['work_upper_bounds']['major_operation_upper_bound_per_role']:,}
major scalar/radix/graph steps per maximum role, including 24 coordinate-radix
passes and 64 stable point-ID/rank passes over the full observed finite-record
envelope, all support/gate/moment work, 12-sweep baseline Jacobi work, and
{proof['work_upper_bounds']['baseline_plus_isotropic_union_find_parent_steps']:,}
bounded union-find parent steps. The semantic allocation ledger is
{proof['semantic_allocation_ledger_total_bytes']:,} bytes. Independently listed
runtime, fragmentation, native-workspace, and stack reserves bring the static
process-tree calculation to {proof['static_process_tree_calculated_bytes']:,}
bytes, below the configured 4 GiB cap; no reserve is defined as cap-minus-arrays.
Measured process-tree watchdog is 12 GiB and the enterprise forbidden threshold
is 16 GiB. Per-role CPU and wall limits are 300 seconds, scratch is 256 MiB, one
worker and one BLAS thread are required, and no parallelism is allowed before
one role is measured. The 10 ms Darwin watchdog is measured supervision, not an
in-kernel cgroup; the enumerated static allocation proof closes that sampling
gap.

## 10. Sole outcomes

Any later representation preflight may emit only
`CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET` or
`INVALID_CII_PERSISTENCE_PREFLIGHT_STOP`. A pass would authorize only another
professor packet. Any contamination, domain omission, resource failure,
determinism mismatch, tamper failure, source-point leakage, nonvacuity/selectivity
failure, or falsifier failure stops the family with no modified rerun.

This document is declarative. `EXECUTION_AUTHORITY=NONE`.
"""


def _validator_contract(config: Mapping[str, Any], schema: Mapping[str, Any], proof: Mapping[str, Any]) -> str:
    return f"""# Independent validator and tamper contract

Bindings: configuration `{config['configuration_sha256']}`, schema
`{schema['schema_sha256']}`, resource proof `{proof['resource_proof_sha256']}`.

## Independent reconstruction

The independent validator rereads the raw PLY and reconstructs canonical finite
XYZ parsing, deduplication, sample ranks, the cubical partition, digital-shell
aggregates, all support digests, tensors, integer filtration values, the complete
space-scale adjacency graph, H0 pairing, every rejection, and the isotropic
negative. It may not import the semantic core. BASE_A, BASE_B, input-order
permutation, and the independent implementation must match exact row domains,
canonical bytes, and table hashes. A mismatch is terminal; no tolerance repair,
row suppression, or alternate ordering is allowed.
Before raw geometry may be read, both implementations must independently match
every identifier/commitment known-answer vector in file 05. They must recompute
case, partition, table-directory, source, code, and dependency bindings from the
locked preimages or signed preexecution lock; digest syntax alone never passes.

## Mandatory tamper regressions

The verifier must independently reject all of the following:

1. source/config/schema/code/dependency hash mutation;
2. missing, duplicated, reordered, appended, or trailing row/payload bytes;
3. forged table counts, offsets, lengths, row sizes, or container length;
4. invalid Patricia parent/child/depth/Morton relation, full subtree digest, or unequal-preimage collision hidden behind any truncated digest;
5. sampled-point census mismatch, membership-root substitution, or use of the 200,000 sample cap for a pre-sampling full-record work/allocation domain;
6. actual-`O` table padding or any row count outside the exact sparse-domain formulas;
7. omitted scale, cell, edge, zero-lifetime interval, essential interval, or rejection;
8. changed filtration value, vertex/edge tie key, elder numeric birth index, status/reason/stage/scale-mask code, persistence pairing, structural parent, or persistence owner;
9. changed arm-independent negative occupancy, support digest, mass, trace,
   validity, scale, or adjacency; missing or duplicated
   `(case_token_digest_sha256,scale_u8,morton_u64)` counterpart; copied baseline
   ID in an isotropic row; isotropic field ID not equal to its canonical arm-1
   recomputation; changed arm without dependent-ID recomputation; rejection,
   birth, essential-death, or interval ID not derived from its own arm-specific
   field ID; cross-arm field-ID equality for unequal preimages; or changed
   within-arm order;
10. baseline denominator other than one, isotropic denominator other than three, rounded trace thirds, a divisibility-by-three premise, nonzero q_iso, zero baseline attributed denominator, or any survivor;
11. injected metadata key, array, object, arbitrary string, coordinate-valued metadata, reordered table descriptor, noncontiguous offset, or forged attestation;
12. mismatched metadata u64 length, alternate JSON whitespace, one nonzero reserve-padding byte, or an XYZ block hidden in metadata padding;
13. any identifier/commitment known-answer mismatch or signed raw-source/code/dependency binding mismatch;
14. injected XYZ, centroid, first moment, point/member list, neighborhood array, variable-width payload, unknown field, pickle, NPZ, or archive;
15. NaN, infinity, negative zero, integer overflow, quantization overflow, Q24.40 format mutation, or eigensolver disagreement;
16. forged work multiplier, union-find edge accounting, semantic allocation row, fixed reserve, RSS, scratch, time, or serialized-byte record;
17. independent-validator source replaced by or importing the semantic core;
18. top-k, reservoir, omitted tail, post-hoc clipping, or changed terminal token.

Every tamper test must prove a specific fail-closed reason. A missing tamper case
or false accept is terminal STOP. No geometry execution is authorized by this
contract. `EXECUTION_AUTHORITY=NONE`.
"""


def _prompt(
    attachment_hashes: Mapping[str, str],
    selection_text: str,
    correction_text: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    proof: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> str:
    attachment_rows = "\n".join(
        f"- `{name}` — `{digest}`" for name, digest in sorted(attachment_hashes.items())
    )
    decisions = "\n".join(f"- `{token}`" for token in LOCK_DECISION_TOKENS)
    next_steps = "\n".join(f"- `{token}`" for token in NEXT_STEP_TOKENS)
    return f"""# GPT-5.6 Pro extended-thinking — corrected Cubical-IIP-V1 paper-lock audit

Read all nine attachments completely and independently recompute their SHA-256
values. Audit the cross-bindings, mathematical completeness, static arithmetic,
authority state, and scientific claim boundary before answering. This is a
fresh, non-execution consultation. Prior-chat authority does not carry forward.

The exact selected architecture response is bound at SHA-256
`{SELECTION_RESPONSE_SHA256}` and embedded below byte-for-byte, including its
trailing newline. It selected `{ARCHITECTURE_TOKEN}` but selected no constants
and granted no execution authority.

{RESPONSE_BEGIN}{selection_text}{RESPONSE_END}

The exact V5 lock-audit response is bound at SHA-256
`{CORRECTION_RESPONSE_SHA256}` and embedded below byte-for-byte, including its
trailing newline. It authorized exactly one non-execution correction packet and
no implementation or run.

{CORRECTION_RESPONSE_BEGIN}{correction_text}{CORRECTION_RESPONSE_END}

## Question to decide

Files 02-09 specify the single corrected declarative lock for
`{PRIMITIVE_TOKEN}`. V7 resolves the V5 contradiction by retaining arm-bound
identifier preimages, pairing cross-arm rows by
`(case_token_digest_sha256,scale_u8,morton_u64)`, and recomputing every
arm-specific dependent identifier. It does not remove `arm_u8`, change a schema
width or cardinality, alter a resource bound, or modify a scientific claim.
The central scientific boundary remains explicit: raw XYZ does
not provide Pottmann's surface-bounded solid domain, so the proposed primitive
is an equal-mass digital occupied shell. Its H0-only space-scale persistence,
all constants, gates, and caps are engineering hypotheses. Decide whether the
corrected paper lock now has one satisfiable identity rule and is internally
complete enough to justify exactly one further pre-execution professor packet—
without execution—or whether the family must stop. No second correction outcome
is authorized by this packet.
In particular, audit the explicit admission that equal-eigenvalue isotropization
guarantees q=0 and therefore tests only implementation dependency/hidden-channel
integrity; it does not identify whether baseline anisotropy came from a bend or
from boundary, hole, density, or voxel quantization. Do not treat a clean
isotropic negative as scientific selectivity evidence.

Do not authorize implementation, raw-scan semantic processing, geometry,
parameter trials, gold/truth/expected counts, labels, CAD/reference, renders,
benchmarks, replay/synthetic rescue, runtime/main, production, or promotion.

## Exact authoritative bindings

- Configuration: `{config['configuration_sha256']}`
- Schema: `{schema['schema_sha256']}`
- Static proof: `{proof['resource_proof_sha256']}`
- Safety contract: `{safety['safety_contract_sha256']}`
- Source-manifest external file: `{SOURCE_MANIFEST_SHA256}`
- Source-manifest internal canonical object: `{SOURCE_MANIFEST_INTERNAL_SHA256}`
- Previous classification response: `{CLASSIFICATION_RESPONSE_SHA256}`
- Previous architecture-selection prompt: `{PRIOR_SELECTION_PROMPT_SHA256}`
- Embedded selected response: `{SELECTION_RESPONSE_SHA256}`
- Embedded V5 correction ruling: `{CORRECTION_RESPONSE_SHA256}`

## Required answer

Return exactly these three lines first:

`LOCK_DECISION: <exactly one lock token>`

{decisions}

`NEXT_STEP: <exactly one next-step token>`

{next_steps}

`EXECUTION_AUTHORITY: NONE`

Only these pairings are valid:

- `{LOCK_PASS}` with `{NEXT_PREFLIGHT}`
- `{LOCK_STOP}` with `{NEXT_STOP}`

If passing, state that pass means corrected paper coherence only and authorizes
one further pre-execution review packet, not code or a run. If stopping, make
the family STOP explicit. Do not request or invent a second correction lane.

`EXECUTION_AUTHORITY=NONE` for this packet and its response.

## Attachment SHA-256

{attachment_rows}
"""


def _expected_payloads() -> tuple[dict[str, bytes], dict[str, Any]]:
    selection_text = _selection_response()
    correction_text = _correction_response()
    source_manifest = _source_manifest_bytes()
    config = _configuration()
    schema = _schema()
    proof = _resource_proof(config, schema)
    safety = _safety_contract(config, schema, proof)

    payloads: dict[str, bytes] = {
        SOURCE_FILENAME: source_manifest,
        LINEAGE_FILENAME: _lineage_ledger(config).rstrip().encode("utf-8") + b"\n",
        LITERATURE_FILENAME: _literature_ledger().rstrip().encode("utf-8") + b"\n",
        SPEC_FILENAME: _normative_spec(config, schema, proof).rstrip().encode("utf-8") + b"\n",
        CONFIG_FILENAME: _json_line(config),
        SCHEMA_FILENAME: _json_line(schema),
        PROOF_FILENAME: _json_line(proof),
        VALIDATOR_FILENAME: _validator_contract(config, schema, proof).rstrip().encode("utf-8") + b"\n",
        SAFETY_FILENAME: _json_line(safety),
    }
    attachment_hashes = {name: _sha_bytes(raw) for name, raw in payloads.items()}
    prompt = _prompt(
        attachment_hashes,
        selection_text,
        correction_text,
        config,
        schema,
        proof,
        safety,
    )
    payloads[PROMPT_FILENAME] = (
        prompt.rstrip().encode("utf-8") + b"\n"
    )
    ordered = {name: payloads[name] for name in UPLOAD_FILENAMES}
    context = {
        "configuration": config,
        "schema": schema,
        "resource_proof": proof,
        "safety": safety,
        "attachment_hashes": attachment_hashes,
    }
    return ordered, context


def _parse_attachment_hashes(prompt: str) -> dict[str, str]:
    marker = "## Attachment SHA-256\n\n"
    if prompt.count(marker) != 1:
        raise ValueError("prompt attachment-hash section mismatch")
    section = prompt.split(marker, 1)[1]
    rows = re.findall(r"^- `([^`]+)` — `([0-9a-f]{64})`$", section, flags=re.MULTILINE)
    if len(rows) != 9:
        raise ValueError("prompt must bind exactly nine attachment hashes")
    if len({name for name, _ in rows}) != 9:
        raise ValueError("duplicate attachment hash row")
    return dict(rows)


def _assert_no_unresolved(value: Any, path: str = "root") -> None:
    if value is None:
        raise ValueError(f"unresolved null configuration value at {path}")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"tbd", "todo", "auto", "default", "inherit", "unknown"}:
            raise ValueError(f"unresolved configuration token at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_unresolved(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_unresolved(child, f"{path}[{index}]")


def _validate_actual_domain_counts(
    census: Mapping[str, int], config: Mapping[str, Any]
) -> None:
    o = int(census.get("occupied_cells_O", -1))
    s = int(config["scales"]["scale_count"]["value"])
    o_max = int(config["partition"]["logical_cell_cap"]["value"])
    n_max = int(config["input"]["sample_cap"]["value"])
    if not 1 <= o <= o_max:
        raise ValueError("occupied-leaf count O is outside the locked domain")
    raw_records = int(census.get("raw_records", -1))
    finite_records = int(census.get("finite_records", -1))
    unique_records = int(census.get("unique_records", -1))
    sampled_records = int(census.get("sampled_unique_records", -1))
    if not 1 <= sampled_records <= unique_records <= finite_records <= raw_records:
        raise ValueError("raw/finite/unique/sampled census ordering mismatch")
    if sampled_records > n_max:
        raise ValueError("sampled unique-record count exceeds cap")
    exact = {
        "field_rows": 2 * s * o,
        "persistence_rows": 2 * s * o,
        "rejection_rows": 2 * s * o,
    }
    for key, expected in exact.items():
        if int(census.get(key, -1)) != expected:
            raise ValueError(f"actual complete-domain count mismatch: {key}")
    if not o <= int(census.get("partition_rows", -1)) <= 2 * o - 1:
        raise ValueError("actual Patricia partition-row count mismatch")
    if not (s - 1) * o <= int(census.get("adjacency_rows", -1)) <= (4 * s - 1) * o:
        raise ValueError("actual space-scale adjacency count exceeds bound")
    if not 0 <= int(census.get("moment_valid_rows", -1)) <= 2 * s * o:
        raise ValueError("moment-valid row count exceeds complete field")
    if not 0 <= int(census.get("attributed_intervals", -1)) <= 2 * s * o:
        raise ValueError("attributed interval count exceeds persistence domain")
    if not 0 <= int(census.get("isotropic_survivors", -1)) <= int(census.get("attributed_intervals", -1)):
        raise ValueError("isotropic survivor census is inconsistent")


def _validate_valid_trace_evidence(
    *,
    arm: str,
    trace_numerator: int,
    eigenvalue_numerators: Iterable[int],
    eigenvalue_denominator: int,
) -> None:
    values = tuple(eigenvalue_numerators)
    if len(values) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("trace evidence requires exactly three integer eigenvalue numerators")
    if isinstance(trace_numerator, bool) or not isinstance(trace_numerator, int) or trace_numerator <= 0:
        raise ValueError("valid trace numerator must be a positive integer")
    if any(value < 0 for value in values):
        raise ValueError("valid eigenvalue numerators must be nonnegative")
    if arm == "BASELINE":
        if eigenvalue_denominator != 1 or sum(values) != trace_numerator:
            raise ValueError("baseline Q24.40 trace/eigenvalue rational mismatch")
        return
    if arm == "ISOTROPIC":
        if eigenvalue_denominator != 3 or any(value != trace_numerator for value in values):
            raise ValueError("isotropic same-trace rational mismatch")
        return
    raise ValueError("unknown trace-evidence arm")


def _validate_closed_metadata(
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> None:
    contract = schema["closed_metadata_schema"]
    expected_keys = set(contract["top_level_keys_exact"])
    if set(metadata) != expected_keys:
        raise ValueError("metadata top-level key set mismatch")
    if metadata.get("schema_version") != schema["schema_version"]:
        raise ValueError("metadata schema version mismatch")
    if metadata.get("source_points_serialized") is not False:
        raise ValueError("metadata source-point safety field mismatch")

    digest_keys = {
        "case_token_digest_sha256",
        "raw_source_sha256",
        "configuration_sha256",
        "schema_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "partition_commitment_sha256",
        "table_directory_sha256",
    }
    for key in digest_keys:
        value = metadata.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"metadata digest field invalid: {key}")
    required_expected_bindings = {
        "raw_source_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "partition_commitment_sha256",
    }
    if set(expected_bindings) != required_expected_bindings:
        raise ValueError("metadata expected-binding key set mismatch")
    for key in required_expected_bindings:
        expected = expected_bindings[key]
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"expected metadata binding invalid: {key}")
        if metadata[key] != expected:
            raise ValueError(f"metadata signed binding mismatch: {key}")
    if metadata["configuration_sha256"] != config["configuration_sha256"]:
        raise ValueError("metadata configuration binding mismatch")
    if metadata["schema_sha256"] != schema["schema_sha256"]:
        raise ValueError("metadata schema binding mismatch")
    case_expected = _domain_digest(
        "TAG_CIIP_CASE_V1", bytes.fromhex(str(metadata["raw_source_sha256"]))
    ).hex()
    if metadata["case_token_digest_sha256"] != case_expected:
        raise ValueError("metadata case-token digest recomputation mismatch")

    census = metadata.get("census")
    if not isinstance(census, Mapping) or set(census) != set(contract["census_keys_exact"]):
        raise ValueError("metadata census key set mismatch")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in census.values()):
        raise ValueError("metadata census values must be nonnegative integers")
    _validate_actual_domain_counts(census, config)

    resource = metadata.get("resource")
    if not isinstance(resource, Mapping) or set(resource) != set(contract["resource_keys_exact"]):
        raise ValueError("metadata resource key set mismatch")
    for key, value in resource.items():
        if key == "resource_cap_tripped":
            if not isinstance(value, bool):
                raise ValueError("metadata resource_cap_tripped must be boolean")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"metadata resource value invalid: {key}")
    if resource["resource_cap_tripped"] is not False:
        raise ValueError("metadata records a resource-cap trip")
    if resource["cpu_nanoseconds"] > int(config["resources"]["cpu_seconds_per_role_maximum"]["value"]) * 1_000_000_000:
        raise ValueError("metadata CPU duration exceeds cap")
    if resource["wall_nanoseconds"] > int(config["resources"]["wall_seconds_per_role_maximum"]["value"]) * 1_000_000_000:
        raise ValueError("metadata wall duration exceeds cap")
    if resource["peak_process_tree_rss_bytes"] > int(config["resources"]["measured_process_tree_watchdog_bytes"]["value"]):
        raise ValueError("metadata RSS exceeds watchdog cap")
    if resource["scratch_bytes"] > int(config["resources"]["scratch_bytes_per_role_maximum"]["value"]):
        raise ValueError("metadata scratch exceeds cap")
    if resource["serialized_bytes"] > int(config["resources"]["serialized_bytes_per_role_maximum"]["value"]):
        raise ValueError("metadata serialization exceeds cap")
    if resource["watchdog_poll_milliseconds"] != int(config["resources"]["watchdog_poll_milliseconds"]["value"]):
        raise ValueError("metadata watchdog cadence mismatch")

    attestation = metadata.get("attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != set(contract["attestation_keys_exact"]):
        raise ValueError("metadata attestation key set mismatch")
    if dict(attestation) != dict(contract["attestation_expected_values"]):
        raise ValueError("metadata attestation value mismatch")

    descriptors = metadata.get("table_descriptors")
    descriptor_contract = contract["table_descriptors"]
    if not isinstance(descriptors, list) or len(descriptors) != int(descriptor_contract["fixed_length"]):
        raise ValueError("metadata table descriptor length mismatch")
    expected_names = list(descriptor_contract["canonical_order"])
    table_by_name = {str(table["name"]): table for table in schema["tables"]}
    census_by_name = {
        "partition_node": int(census["partition_rows"]),
        "leaf_aggregate": int(census["occupied_cells_O"]),
        "cell_scale_field": int(census["field_rows"]),
        "space_scale_adjacency": int(census["adjacency_rows"]),
        "persistence_disposition": int(census["persistence_rows"]),
        "rejection": int(census["rejection_rows"]),
    }
    expected_offset = int(schema["metadata_reserve_bytes"])
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping) or set(descriptor) != set(descriptor_contract["keys_exact"]):
            raise ValueError("metadata table descriptor key set mismatch")
        if descriptor.get("name_enum") != expected_names[index]:
            raise ValueError("metadata table descriptor order mismatch")
        for key in ("row_count_u64", "row_size_u32", "offset_u64", "byte_length_u64"):
            value = descriptor.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metadata descriptor integer invalid: {key}")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("metadata table descriptor digest invalid")
        name = expected_names[index]
        expected_rows = census_by_name[name]
        expected_row_size = int(table_by_name[name]["row_size_bytes"])
        if descriptor["row_count_u64"] != expected_rows:
            raise ValueError("metadata table descriptor row-count mismatch")
        if descriptor["row_size_u32"] != expected_row_size:
            raise ValueError("metadata table descriptor row-size mismatch")
        if descriptor["byte_length_u64"] != expected_rows * expected_row_size:
            raise ValueError("metadata table descriptor byte-length mismatch")
        if descriptor["offset_u64"] != expected_offset:
            raise ValueError("metadata table descriptor offset is not contiguous")
        expected_offset += descriptor["byte_length_u64"]

    if resource["serialized_bytes"] != expected_offset:
        raise ValueError("metadata serialized-byte census mismatches closed container length")

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
    directory_expected = _domain_digest(
        "TAG_CIIP_TABLE_DIRECTORY_V1", *directory_parts
    ).hex()
    if metadata["table_directory_sha256"] != directory_expected:
        raise ValueError("metadata table-directory commitment mismatch")

    encoded = _canonical_json_bytes(metadata)
    if len(encoded) > int(contract["maximum_encoded_bytes"]):
        raise ValueError("metadata exceeds closed-schema byte cap")


def _encode_metadata_region(metadata: Mapping[str, Any], schema: Mapping[str, Any]) -> bytes:
    encoded = _canonical_json_bytes(metadata)
    contract = schema["closed_metadata_schema"]
    region = schema["metadata_region_encoding"]
    maximum = int(contract["maximum_encoded_bytes"])
    total = int(region["total_bytes"])
    if len(encoded) > maximum or 8 + len(encoded) > total:
        raise ValueError("metadata exceeds fixed region")
    return struct.pack("<Q", len(encoded)) + encoded + b"\x00" * (total - 8 - len(encoded))


def _validate_metadata_region_bytes(
    region_bytes: bytes,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> dict[str, Any]:
    region = schema["metadata_region_encoding"]
    total = int(region["total_bytes"])
    if len(region_bytes) != total:
        raise ValueError("metadata region length mismatch")
    encoded_length = struct.unpack("<Q", region_bytes[:8])[0]
    maximum = int(schema["closed_metadata_schema"]["maximum_encoded_bytes"])
    if encoded_length > maximum or 8 + encoded_length > total:
        raise ValueError("metadata length prefix exceeds fixed region")
    encoded = region_bytes[8 : 8 + encoded_length]
    padding = region_bytes[8 + encoded_length :]
    if any(padding):
        raise ValueError("metadata interstitial padding must be all zero")
    try:
        metadata = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata canonical JSON parse failure") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata canonical JSON must encode one object")
    if _canonical_json_bytes(metadata) != encoded:
        raise ValueError("metadata JSON is not byte-exact canonical encoding")
    _validate_closed_metadata(metadata, schema, config, expected_bindings)
    return dict(metadata)


def _validate_context(context: Mapping[str, Any]) -> None:
    config = context["configuration"]
    schema = context["schema"]
    proof = context["resource_proof"]
    safety = context["safety"]

    _verify_self_bound_json(config, "configuration_sha256", "configuration")
    _verify_self_bound_json(schema, "schema_sha256", "schema")
    _verify_self_bound_json(proof, "resource_proof_sha256", "resource proof")
    _verify_self_bound_json(safety, "safety_contract_sha256", "safety contract")
    _assert_no_unresolved(config)

    if config["authority"] != _authority_state() or schema["authority"] != _authority_state():
        raise ValueError("authority state mismatch")
    if safety["authority"] != _authority_state():
        raise ValueError("safety authority state mismatch")
    if safety["valid_decision_pairs"] != VALID_DECISION_PAIRS:
        raise ValueError("decision-pair mapping mismatch")
    if proof["configuration_sha256"] != config["configuration_sha256"]:
        raise ValueError("resource-proof configuration cross-binding mismatch")
    if proof["schema_sha256"] != schema["schema_sha256"]:
        raise ValueError("resource-proof schema cross-binding mismatch")
    if safety["configuration_sha256"] != config["configuration_sha256"]:
        raise ValueError("safety configuration cross-binding mismatch")
    if safety["schema_sha256"] != schema["schema_sha256"]:
        raise ValueError("safety schema cross-binding mismatch")
    if safety["resource_proof_sha256"] != proof["resource_proof_sha256"]:
        raise ValueError("safety resource-proof cross-binding mismatch")
    source_envelope = config["source_envelope_observed"]
    if int(source_envelope["maximum_finite_coordinate_records_per_scan"]) != SOURCE_MAX_FINITE_RECORDS:
        raise ValueError("source finite-record envelope binding mismatch")
    if int(source_envelope["maximum_raw_bytes_per_scan"]) != SOURCE_MAX_RAW_BYTES:
        raise ValueError("source raw-byte envelope binding mismatch")
    if source_envelope["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
        raise ValueError("source-envelope manifest binding mismatch")
    identifier_rules = config["identifier_and_commitment_rules"]
    if identifier_rules["all_tags"] != ID_DOMAIN_TAGS:
        raise ValueError("identifier domain-tag lock mismatch")
    if identifier_rules["known_answer_vectors"] != _identifier_known_answer_vectors():
        raise ValueError("identifier/commitment known-answer vector mismatch")

    if config["complex"]["homology_dimensions"]["value"] != [0]:
        raise ValueError("only H0 is authorized by the bounded paper lock")
    if config["complex"]["higher_homology_closed"] is not True:
        raise ValueError("higher homology must remain explicitly closed")
    if config["complex"]["coefficient_field"] != "F2":
        raise ValueError("coefficient-field lock mismatch")
    if not str(config["complex"]["algorithm"]).startswith("DETERMINISTIC_UNION_FIND"):
        raise ValueError("bounded H0 union-find algorithm lock mismatch")
    if config["complex"]["vertex_event_tie_key"] != "vertex_index_u32":
        raise ValueError("vertex event tie-key lock mismatch")
    if not str(config["complex"]["elder_rule"]).endswith("NUMERIC_BIRTH_VERTEX_INDEX_U32"):
        raise ValueError("elder-rule numeric birth-index lock mismatch")
    if config["row_codebooks_and_sentinels"]["persistence_status_u8"] != {"FINITE": 0, "ESSENTIAL": 1}:
        raise ValueError("persistence status codebook mismatch")
    if not str(config["component_semantics"]["scale_mask_u8_rule"]).startswith("BIT_scale_u8"):
        raise ValueError("scale-mask bit lock mismatch")
    if int(
        config["isotropic_counterfactual"][
            "baseline_attributed_denominator_minimum"
        ]["value"]
    ) < 1:
        raise ValueError("falsifier denominator must be provably nonzero")
    if int(
        config["isotropic_counterfactual"][
            "allowed_anisotropy_attributed_survivors"
        ]["value"]
    ) != 0:
        raise ValueError("isotropic anisotropy-attributed survivors are forbidden")
    if config["isotropic_counterfactual"]["complete_field_and_h0_rebuild_required"] is not True:
        raise ValueError("counterfactual must rebuild the complete field and H0 complex")
    if config["isotropic_counterfactual"]["zero_anisotropy_survivors_expected_by_construction"] is not True:
        raise ValueError("isotropic zero-survivor construction boundary mismatch")
    if config["isotropic_counterfactual"]["scientific_nuisance_origin_falsification_claimed"] is not False:
        raise ValueError("isotropic negative may not claim baseline nuisance falsification")
    isotropic = config["isotropic_counterfactual"]
    if isotropic["field_row_pairing_key"] != "case_token_digest_sha256_THEN_scale_u8_THEN_morton_u64":
        raise ValueError("cross-arm field-row pairing key mismatch")
    if isotropic["arm_specific_field_row_ids_recomputed_from_locked_preimages"] is not True:
        raise ValueError("arm-specific field-row IDs must be recomputed")
    if isotropic["baseline_and_isotropic_field_row_ids_byte_identical"] is not False:
        raise ValueError("cross-arm field-row IDs may not be byte-identical")
    if "row_ids_and_order" in isotropic["preserved_domains"]:
        raise ValueError("contradictory cross-arm row-ID preservation remains")
    kat = identifier_rules["known_answer_vectors"]["expected"]
    if kat["field_row_id_digest_128"] == kat["isotropic_field_row_id_digest_128"]:
        raise ValueError("cross-arm field-row KATs must be distinct")
    if "ARM_SPECIFIC_IDENTIFIER_OR_SOURCE_CELL_CORRESPONDENCE_FAILURE" not in safety["terminal_stop_conditions"]:
        raise ValueError("corrected identity terminal condition missing")
    if config["claim_boundary"]["baseline_anisotropy_physical_origin_resolved"] is not False:
        raise ValueError("paper lock may not claim resolved baseline anisotropy origin")
    if config["deterministic_roles"]["role_count"]["value"] != 4:
        raise ValueError("deterministic role-count lock mismatch")
    if config["deterministic_roles"]["each_role_emits_complete_baseline_and_isotropic_payload"] is not True:
        raise ValueError("every deterministic role must emit a complete payload")
    if config["deterministic_roles"]["independent_validator_imports_semantic_core"] is not False:
        raise ValueError("independent validator may not import the semantic core")

    schema_text = json.dumps(schema, sort_keys=True).lower()
    for forbidden in FORBIDDEN_SCHEMA_TOKENS:
        if f'"name": "{forbidden}"' in schema_text:
            raise ValueError(f"forbidden schema field present: {forbidden}")
    if schema.get("source_points_serialized") is not False:
        raise ValueError("schema permits source-point serialization")

    recomputed = _resource_proof(config, schema)
    if recomputed != proof:
        raise ValueError("resource proof does not recompute from configuration and schema")
    hard_cap = int(proof["hard_payload_cap_bytes_per_complete_role"])
    payload = int(proof["calculated_payload_bytes_per_complete_role"])
    if payload > hard_cap or int(proof["payload_headroom_bytes"]) != hard_cap - payload:
        raise ValueError("payload proof exceeds cap or has bad headroom")
    if int(proof["scratch_required_bytes"]) > int(proof["scratch_cap_bytes"]):
        raise ValueError("scratch proof exceeds cap")
    if int(proof["semantic_allocation_ledger_total_bytes"]) > int(proof["application_array_cap_bytes"]):
        raise ValueError("application allocation proof exceeds cap")
    if int(proof["static_process_tree_calculated_bytes"]) > int(proof["static_process_tree_configured_cap_bytes"]):
        raise ValueError("static process-tree proof exceeds configured cap")
    if int(proof["static_process_tree_calculated_bytes"]) >= int(proof["enterprise_rss_forbidden_threshold_bytes"]):
        raise ValueError("static process-tree proof violates enterprise ceiling")
    if proof["generic_h1_h2_boundary_matrix_reduction_authorized"] is not False:
        raise ValueError("H1/H2 reduction must remain closed")


def _audit_path_inside_output(output_dir: Path, audit_path: Path) -> bool:
    output = output_dir.resolve(strict=False)
    audit = audit_path.resolve(strict=False)
    return audit == output or output in audit.parents


def _validate_packet(output_dir: Path) -> dict[str, Any]:
    expected, context = _expected_payloads()
    _validate_context(context)

    entries = sorted(output_dir.iterdir())
    if any(path.name.startswith(".") for path in entries):
        raise ValueError("packet may not contain hidden files")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("packet may contain only regular non-symlink files")
    if len(entries) != MAX_FILES:
        raise ValueError(f"expected exactly {MAX_FILES} individual files")
    if tuple(path.name for path in entries) != tuple(sorted(UPLOAD_FILENAMES)):
        raise ValueError("packet filename set mismatch")
    if any(path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".npz", ".pkl"} for path in entries):
        raise ValueError("archives and object containers are forbidden")

    for name, expected_bytes in expected.items():
        observed = (output_dir / name).read_bytes()
        if observed != expected_bytes:
            raise ValueError(f"packet file differs from deterministic expected bytes: {name}")

    prompt_path = output_dir / UPLOAD_FILENAMES[0]
    prompt = prompt_path.read_text(encoding="utf-8")
    if prompt.count(RESPONSE_BEGIN) != 1 or prompt.count(RESPONSE_END) != 1:
        raise ValueError("embedded selection-response marker count mismatch")
    embedded = prompt.split(RESPONSE_BEGIN, 1)[1].split(RESPONSE_END, 1)[0]
    if embedded.encode("utf-8") != SELECTION_RESPONSE_PATH.read_bytes():
        raise ValueError("embedded selection response is not byte-exact")
    if _sha_bytes(embedded.encode("utf-8")) != SELECTION_RESPONSE_SHA256:
        raise ValueError("embedded selection response SHA mismatch")
    if prompt.count(CORRECTION_RESPONSE_BEGIN) != 1 or prompt.count(CORRECTION_RESPONSE_END) != 1:
        raise ValueError("embedded correction-response marker count mismatch")
    correction_embedded = prompt.split(CORRECTION_RESPONSE_BEGIN, 1)[1].split(
        CORRECTION_RESPONSE_END, 1
    )[0]
    if correction_embedded.encode("utf-8") != CORRECTION_RESPONSE_PATH.read_bytes():
        raise ValueError("embedded correction response is not byte-exact")
    if _sha_bytes(correction_embedded.encode("utf-8")) != CORRECTION_RESPONSE_SHA256:
        raise ValueError("embedded correction response SHA mismatch")

    observed_hashes = {
        name: _sha_file(output_dir / name) for name in UPLOAD_FILENAMES[1:]
    }
    if _parse_attachment_hashes(prompt) != observed_hashes:
        raise ValueError("prompt attachment hash table mismatch")
    if observed_hashes[SOURCE_FILENAME] != SOURCE_MANIFEST_SHA256:
        raise ValueError("carried source manifest differs from frozen hash")

    required_prompt_tokens = (
        ARCHITECTURE_TOKEN,
        PRIMITIVE_TOKEN,
        SELECTION_RESPONSE_SHA256,
        CORRECTION_RESPONSE_SHA256,
        context["configuration"]["configuration_sha256"],
        context["schema"]["schema_sha256"],
        context["resource_proof"]["resource_proof_sha256"],
        context["safety"]["safety_contract_sha256"],
        "EXECUTION_AUTHORITY: NONE",
        "EXECUTION_AUTHORITY=NONE",
    )
    for token in (*required_prompt_tokens, *LOCK_DECISION_TOKENS, *NEXT_STEP_TOKENS):
        if token not in prompt:
            raise ValueError(f"prompt missing required token: {token}")

    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="strict") for path in entries
    )
    for contradictory in (
        "EXECUTION_AUTHORITY=GRANTED",
        "EXECUTION_AUTHORITY: GRANTED",
        '"execution_authority":"GRANTED"',
        '"geometry_execution_authorized":true',
        '"raw_scan_semantic_processing_authorized":true',
        '"benchmark_authorized":true',
        '"runtime_or_main_integration_authorized":true',
        '"production_or_release_authorized":true',
    ):
        if contradictory in all_text:
            raise ValueError(f"contradictory authority present: {contradictory}")

    packet_hashes = {path.name: _sha_file(path) for path in entries}
    return {
        "schema_version": 1,
        "package_id": "SHERMAN_QC_F1_CUBICAL_IIP_V1_PREFLIGHT_LOCK_REVIEW_20260809_V7",
        "individual_file_count": len(entries),
        "no_archive": True,
        "paper_lock_review_only": True,
        "execution_authority": EXECUTION_AUTHORITY,
        "geometry_execution_authorized": False,
        "raw_scan_semantic_processing_authorized": False,
        "gold_or_truth_consumed": False,
        "benchmark_authorized": False,
        "runtime_or_production_authorized": False,
        "selection_response_sha256": SELECTION_RESPONSE_SHA256,
        "embedded_selection_response_byte_exact": True,
        "correction_response_sha256": CORRECTION_RESPONSE_SHA256,
        "embedded_correction_response_byte_exact": True,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_internal_sha256": SOURCE_MANIFEST_INTERNAL_SHA256,
        "architecture_token": ARCHITECTURE_TOKEN,
        "primitive_token": PRIMITIVE_TOKEN,
        "lock_decision_tokens": list(LOCK_DECISION_TOKENS),
        "next_step_tokens": list(NEXT_STEP_TOKENS),
        "valid_decision_pairs": VALID_DECISION_PAIRS,
        "configuration_sha256": context["configuration"]["configuration_sha256"],
        "schema_sha256": context["schema"]["schema_sha256"],
        "resource_proof_sha256": context["resource_proof"]["resource_proof_sha256"],
        "safety_contract_sha256": context["safety"]["safety_contract_sha256"],
        "calculated_payload_bytes_per_complete_role": context["resource_proof"]["calculated_payload_bytes_per_complete_role"],
        "hard_payload_cap_bytes_per_complete_role": context["resource_proof"]["hard_payload_cap_bytes_per_complete_role"],
        "packet_sha256": packet_hashes,
    }


def build_package(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    audit_path: Path | None = DEFAULT_AUDIT_PATH,
    verify_raw_sources: bool = False,
) -> dict[str, Any]:
    if verify_raw_sources:
        raise ValueError("paper-lock builder may not read or verify raw PLY sources")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if audit_path is not None and _audit_path_inside_output(output_dir, audit_path):
        raise ValueError("audit path must remain outside the ten-file upload directory")

    expected, context = _expected_payloads()
    _validate_context(context)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in UPLOAD_FILENAMES:
        destination = output_dir / name
        if name == SOURCE_FILENAME:
            shutil.copyfile(SOURCE_MANIFEST_PATH, destination)
        else:
            destination.write_bytes(expected[name])

    result = _validate_packet(output_dir)
    if audit_path is not None:
        if audit_path.exists():
            raise FileExistsError(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_bytes(_json_line(result))
        result = _validate_packet(output_dir)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_package(
        output_dir=args.output_dir,
        audit_path=args.audit_path,
        verify_raw_sources=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
