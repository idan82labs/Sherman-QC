from __future__ import annotations

import hashlib
import struct

import pytest

from research.f1_ciip_v1 import identifiers as ids


def _kat_case() -> bytes:
    return ids.case_token_digest(bytes(32))


def _empty_descriptors() -> tuple[ids.TableDirectoryDescriptor, ...]:
    empty_sha = hashlib.sha256(b"").digest()
    return tuple(
        ids.TableDirectoryDescriptor(index, 0, row_size, ids.METADATA_REGION_BYTES, 0, empty_sha)
        for index, row_size in enumerate(ids.TABLE_ROW_SIZES)
    )


def test_exact_ascii_nul_domain_tags() -> None:
    assert len(ids.DOMAIN_TAGS) == 14
    for key, literal in ids.DOMAIN_TAGS.items():
        assert ids.domain_tag_bytes(key) == literal.encode("ascii") + b"\x00"
        assert ids.domain_tag_bytes(key).count(b"\x00") == 1
    with pytest.raises(ValueError, match="unknown CIIP-V1 domain tag"):
        ids.domain_tag_bytes("CIIP_CASE_V1")


def test_unsigned_integer_encodings_are_exact_little_endian() -> None:
    assert ids.u8_le(0xA5) == b"\xa5"
    assert ids.u16_le(0x1234) == b"\x34\x12"
    assert ids.u32_le(0x12345678) == b"\x78\x56\x34\x12"
    assert ids.u64_le(0x0123456789ABCDEF) == b"\xef\xcd\xab\x89\x67\x45\x23\x01"
    for function in (ids.u8_le, ids.u16_le, ids.u32_le, ids.u64_le):
        with pytest.raises((TypeError, ValueError)):
            function(True)
        with pytest.raises(ValueError):
            function(-1)


def test_canonical_xyz_is_finite_little_endian_and_normalizes_negative_zero() -> None:
    assert ids.canonical_xyz_binary64_le(-0.0, 0.0, -0.0) == bytes(24)
    assert ids.canonical_xyz_binary64_le(1.0, 2.0, 3.0) == struct.pack("<ddd", 1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="finite"):
        ids.canonical_xyz_binary64_le(float("nan"), 0.0, 0.0)


def test_morton5_and_right_aligned_prefix_rules() -> None:
    assert ids.morton5(1, 2, 3) == 53
    full = ids.morton5(31, 31, 31)
    assert full == (1 << 15) - 1
    assert [ids.morton5_prefix(full, depth) for depth in range(6)] == [
        0,
        0b111,
        0b111111,
        0b111111111,
        0b111111111111,
        0b111111111111111,
    ]
    with pytest.raises(ValueError):
        ids.morton5(32, 0, 0)
    with pytest.raises(ValueError):
        ids.morton5_prefix(0, 6)


def test_full_and_first16_helpers_use_first_raw_digest_bytes() -> None:
    preimage = b"CIIP_CASE_V1\x00" + bytes(32)
    expected = hashlib.sha256(preimage).digest()
    assert ids.tagged_preimage("TAG_CIIP_CASE_V1", bytes(32)) == preimage
    assert ids.full_digest("TAG_CIIP_CASE_V1", bytes(32)) == expected
    assert ids.domain_digest("TAG_CIIP_CASE_V1", bytes(32)) == expected
    assert ids.first16_digest("TAG_CIIP_CASE_V1", bytes(32)) == expected[:16]


def test_concise_public_api_names_bind_the_exact_implementations() -> None:
    required = {
        "domain_digest",
        "case_token_digest",
        "point_id",
        "sample_rank",
        "node_id",
        "subtree_membership_digest",
        "cell_id",
        "leaf_membership_digest",
        "field_row_id",
        "support_membership_digest",
        "edge_id",
        "essential_death_id",
        "interval_id",
        "partition_commitment",
        "table_directory_commitment",
        "morton5",
        "morton_prefix",
        "CollisionRegistry",
    }
    assert required <= set(ids.__all__)
    assert ids.point_id is ids.point_id_sha256
    assert ids.sample_rank is ids.sample_rank_sha256
    assert ids.node_id is ids.node_id_sha256
    assert ids.subtree_membership_digest is ids.subtree_membership_digest_sha256
    assert ids.cell_id is ids.cell_id_sha256
    assert ids.leaf_membership_digest is ids.leaf_membership_digest_128
    assert ids.field_row_id is ids.field_row_id_digest_128
    assert ids.support_membership_digest is ids.support_membership_digest_128
    assert ids.edge_id is ids.edge_id_digest_128
    assert ids.essential_death_id is ids.essential_death_row_id_digest_128
    assert ids.interval_id is ids.interval_id_digest_128
    assert ids.partition_commitment is ids.partition_commitment_sha256
    assert ids.table_directory_commitment is ids.table_directory_commitment_sha256
    assert ids.morton_prefix is ids.morton5_prefix


def test_membership_preimages_sort_items_and_bind_u32_count() -> None:
    first = hashlib.sha256(b"first").digest()
    second = hashlib.sha256(b"second").digest()
    expected = hashlib.sha256(
        b"CIIP_SUBTREE_MEMBERS_V1\x00"
        + struct.pack("<I", 2)
        + b"".join(sorted((first, second)))
    ).digest()
    assert ids.subtree_membership_digest_sha256([second, first]) == expected
    assert ids.subtree_membership_digest_sha256([first, second]) == expected


def test_field_rows_are_arm_bound_and_dependent_aliases_are_exact() -> None:
    case = _kat_case()
    morton = ids.morton5(1, 2, 3)
    baseline = ids.field_row_id_digest_128(case, 0, 1, morton)
    isotropic = ids.field_row_id_digest_128(case, 1, 1, morton)
    assert baseline.hex() == "ae35aeb5777676ad6ac61aba20f90caa"
    assert isotropic.hex() == "72dc258326cb1f01014240070cc9e59b"
    assert baseline != isotropic
    assert ids.rejection_row_id_digest_128(baseline) == baseline


def test_edge_enum_relation_and_finite_death_alias() -> None:
    case = _kat_case()
    spatial = ids.edge_id_digest_128(case, 1, 2, 0, 0)
    scale = ids.edge_id_digest_128(case, 1, 2, 1, 255)
    assert spatial.hex() == "2e5aed76eaf0270e31d51c6027870360"
    assert scale != spatial
    assert ids.finite_death_row_id_digest_128(spatial) == spatial
    with pytest.raises(ValueError, match="kind and axis"):
        ids.edge_id_digest_128(case, 1, 2, 1, 0)


def test_partition_commitment_binds_row_count_and_exact_table_length() -> None:
    observed = ids.partition_commitment_sha256(1, bytes(ids.PARTITION_ROW_SIZE_BYTES))
    assert observed.hex() == "7327b3d33259deef14deda7142f2951470e80606704691de49bedbe41dd84d13"
    with pytest.raises(ValueError, match="length does not match"):
        ids.partition_commitment_sha256(1, b"")


def test_table_directory_descriptor_encoding_and_commitment() -> None:
    descriptors = _empty_descriptors()
    first = descriptors[0]
    assert len(first.canonical_bytes()) == 61
    assert first.canonical_bytes() == (
        struct.pack("<BQIQQ", 0, 0, 88, ids.METADATA_REGION_BYTES, 0)
        + hashlib.sha256(b"").digest()
    )
    assert ids.table_directory_commitment_sha256(descriptors).hex() == (
        "43055b40d5844fc87cdbd94002f007e9e92f87b8b777dd76d0c19bc1d9510509"
    )


def test_table_directory_rejects_wrong_order_width_length_or_offset() -> None:
    descriptors = list(_empty_descriptors())
    descriptors[0] = ids.TableDirectoryDescriptor(
        1, 0, 88, ids.METADATA_REGION_BYTES, 0, hashlib.sha256(b"").digest()
    )
    with pytest.raises(ValueError, match="enum order"):
        ids.table_directory_commitment_sha256(descriptors)

    descriptors = list(_empty_descriptors())
    descriptors[3] = ids.TableDirectoryDescriptor(
        3, 1, 32, ids.METADATA_REGION_BYTES, 0, hashlib.sha256(b"").digest()
    )
    with pytest.raises(ValueError, match="byte length"):
        ids.table_directory_commitment_sha256(descriptors)


def test_collision_registry_is_scoped_by_identifier_class_and_preimage() -> None:
    registry = ids.CollisionRegistry()
    digest = bytes.fromhex("11" * 16)
    assert registry.record("field", digest, b"one") == digest
    assert registry.record("field", digest, b"one") == digest
    assert registry.record("edge", digest, b"other-class") == digest
    assert len(registry) == 2
    with pytest.raises(ids.DigestCollisionError, match="unequal canonical preimages"):
        registry.record("field", digest, b"two")


def test_truncated_identifier_helpers_register_full_tagged_preimages() -> None:
    registry = ids.CollisionRegistry()
    case = _kat_case()
    digest = ids.field_row_id_digest_128(case, 0, 1, 53, registry=registry)
    expected_preimage = b"CIIP_FIELD_ROW_V1\x00" + case + struct.pack("<BBQ", 0, 1, 53)
    assert registry.preimage_for("field_row_id_digest_128", digest) == expected_preimage


def test_all_twenty_v7_known_answer_vectors_match_frozen_values() -> None:
    observed = ids.compute_v7_known_answer_vectors()
    assert len(observed) == 20
    assert observed == dict(ids.V7_KNOWN_ANSWER_EXPECTED)
    ids.validate_v7_known_answer_vectors()


def test_identifier_inputs_fail_closed_on_invalid_widths_and_domains() -> None:
    case = _kat_case()
    with pytest.raises(ValueError, match="exactly 32"):
        ids.case_token_digest(b"short")
    with pytest.raises(ValueError, match="BASELINE=0 or ISOTROPIC=1"):
        ids.field_row_id_digest_128(case, 2, 0, 0)
    with pytest.raises(ValueError, match="one of 0, 1, 2"):
        ids.field_row_id_digest_128(case, 0, 3, 0)
    with pytest.raises(ValueError, match="must not precede"):
        ids.interval_id_digest_128(case, 0, bytes(16), bytes(16), 20, 10)
