"""Frozen V7 identifier and commitment layer for CIIP-V1.

This module is deliberately limited to byte encodings, hashes, Morton keys,
commitments, and collision accounting.  It performs no scan I/O or geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from types import MappingProxyType
from typing import Final, Iterable, Mapping, Sequence


DOMAIN_TAGS: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)

MORTON_BITS_PER_AXIS: Final = 5
MORTON_AXIS_SIZE: Final = 1 << MORTON_BITS_PER_AXIS
MORTON_VALUE_LIMIT: Final = 1 << (3 * MORTON_BITS_PER_AXIS)
METADATA_REGION_BYTES: Final = 1_048_576
PARTITION_ROW_SIZE_BYTES: Final = 88
TABLE_ROW_SIZES: Final[tuple[int, ...]] = (88, 64, 128, 32, 64, 32)


class DigestCollisionError(ValueError):
    """Raised when one truncated digest is assigned unequal preimages."""


def _bytes(value: bytes | bytearray | memoryview, *, name: str, length: int | None = None) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    result = bytes(value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} bytes")
    return result


def _uint(value: int, *, bits: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value >= 1 << bits:
        raise ValueError(f"{name} is outside the unsigned {bits}-bit domain")
    return value


def u8_le(value: int, *, name: str = "value") -> bytes:
    return struct.pack("<B", _uint(value, bits=8, name=name))


def u16_le(value: int, *, name: str = "value") -> bytes:
    return struct.pack("<H", _uint(value, bits=16, name=name))


def u32_le(value: int, *, name: str = "value") -> bytes:
    return struct.pack("<I", _uint(value, bits=32, name=name))


def u64_le(value: int, *, name: str = "value") -> bytes:
    return struct.pack("<Q", _uint(value, bits=64, name=name))


def domain_tag_bytes(tag_key: str) -> bytes:
    """Return the exact listed ASCII domain tag followed by one NUL byte."""

    try:
        tag = DOMAIN_TAGS[tag_key]
    except KeyError as exc:
        raise ValueError(f"unknown CIIP-V1 domain tag: {tag_key}") from exc
    return tag.encode("ascii") + b"\x00"


def tagged_preimage(tag_key: str, *parts: bytes) -> bytes:
    return domain_tag_bytes(tag_key) + b"".join(
        _bytes(part, name=f"preimage part {index}") for index, part in enumerate(parts)
    )


def full_digest(tag_key: str, *parts: bytes) -> bytes:
    return hashlib.sha256(tagged_preimage(tag_key, *parts)).digest()


def first16_digest(tag_key: str, *parts: bytes) -> bytes:
    return full_digest(tag_key, *parts)[:16]


@dataclass(slots=True)
class CollisionRegistry:
    """Per-identifier-class registry for truncated digest preimages."""

    _preimages: dict[str, dict[bytes, bytes]] = field(default_factory=dict)

    def record(self, identifier_class: str, digest: bytes, canonical_preimage: bytes) -> bytes:
        if not isinstance(identifier_class, str) or not identifier_class:
            raise ValueError("identifier_class must be a nonempty string")
        digest_bytes = _bytes(digest, name="digest")
        if not digest_bytes:
            raise ValueError("digest must not be empty")
        preimage_bytes = _bytes(canonical_preimage, name="canonical_preimage")
        class_registry = self._preimages.setdefault(identifier_class, {})
        prior = class_registry.get(digest_bytes)
        if prior is not None and prior != preimage_bytes:
            raise DigestCollisionError(
                f"unequal canonical preimages share {identifier_class} digest {digest_bytes.hex()}"
            )
        class_registry[digest_bytes] = preimage_bytes
        return digest_bytes

    def preimage_for(self, identifier_class: str, digest: bytes) -> bytes | None:
        return self._preimages.get(identifier_class, {}).get(_bytes(digest, name="digest"))

    def __len__(self) -> int:
        return sum(len(values) for values in self._preimages.values())


def _truncated_identifier(
    identifier_class: str,
    tag_key: str,
    *parts: bytes,
    registry: CollisionRegistry | None,
) -> bytes:
    preimage = tagged_preimage(tag_key, *parts)
    digest = hashlib.sha256(preimage).digest()[:16]
    if registry is not None:
        registry.record(identifier_class, digest, preimage)
    return digest


def canonical_xyz_binary64_le(x: float, y: float, z: float) -> bytes:
    """Pack finite XYZ as little-endian binary64 after negative-zero normalization."""

    values: list[float] = []
    for name, value in zip(("x", "y", "z"), (x, y, z), strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        values.append(0.0 if converted == 0.0 else converted)
    return struct.pack("<ddd", *values)


def case_token_digest(raw_source_sha256: bytes) -> bytes:
    return full_digest("TAG_CIIP_CASE_V1", _bytes(raw_source_sha256, name="raw_source_sha256", length=32))


def point_id_sha256(canonical_xyz_binary64_le24: bytes) -> bytes:
    return full_digest(
        "TAG_CIIP_POINT_V1",
        _bytes(canonical_xyz_binary64_le24, name="canonical_xyz_binary64_le24", length=24),
    )


def sample_rank_sha256(point_id: bytes) -> bytes:
    return full_digest("TAG_CIIP_SAMPLE_RANK_V1", _bytes(point_id, name="point_id", length=32))


def morton5(x: int, y: int, z: int) -> int:
    coordinates = (
        _uint(x, bits=MORTON_BITS_PER_AXIS, name="x"),
        _uint(y, bits=MORTON_BITS_PER_AXIS, name="y"),
        _uint(z, bits=MORTON_BITS_PER_AXIS, name="z"),
    )
    result = 0
    for bit in range(MORTON_BITS_PER_AXIS):
        result |= ((coordinates[0] >> bit) & 1) << (3 * bit)
        result |= ((coordinates[1] >> bit) & 1) << (3 * bit + 1)
        result |= ((coordinates[2] >> bit) & 1) << (3 * bit + 2)
    return result


def morton5_prefix(morton_u64: int, depth: int) -> int:
    morton = _uint(morton_u64, bits=15, name="morton_u64")
    prefix_depth = _uint(depth, bits=8, name="depth")
    if prefix_depth > MORTON_BITS_PER_AXIS:
        raise ValueError("depth must be in the closed interval [0, 5]")
    return morton >> (3 * (MORTON_BITS_PER_AXIS - prefix_depth))


def node_id_sha256(case_digest: bytes, depth: int, morton_prefix_u64: int) -> bytes:
    prefix_depth = _uint(depth, bits=8, name="depth")
    if prefix_depth > MORTON_BITS_PER_AXIS:
        raise ValueError("depth must be in the closed interval [0, 5]")
    prefix = _uint(morton_prefix_u64, bits=64, name="morton_prefix_u64")
    if prefix >= 1 << (3 * prefix_depth):
        raise ValueError("morton_prefix_u64 has nonzero bits outside its depth")
    return full_digest(
        "TAG_CIIP_NODE_V1",
        _bytes(case_digest, name="case_digest", length=32),
        u8_le(prefix_depth, name="depth"),
        u64_le(prefix, name="morton_prefix_u64"),
    )


def _counted_sorted(items: Iterable[bytes], *, item_size: int, name: str) -> tuple[bytes, tuple[bytes, ...]]:
    canonical = tuple(sorted(_bytes(item, name=name, length=item_size) for item in items))
    return u32_le(len(canonical), name=f"{name}_count") + b"".join(canonical), canonical


def subtree_membership_digest_sha256(point_ids: Iterable[bytes]) -> bytes:
    counted, _ = _counted_sorted(point_ids, item_size=32, name="point_id")
    return full_digest("TAG_CIIP_SUBTREE_MEMBERS_V1", counted)


def cell_id_sha256(case_digest: bytes, morton_u64: int) -> bytes:
    morton = _uint(morton_u64, bits=15, name="morton_u64")
    return full_digest(
        "TAG_CIIP_CELL_V1",
        _bytes(case_digest, name="case_digest", length=32),
        u64_le(morton, name="morton_u64"),
    )


def leaf_membership_digest_128(
    point_ids: Iterable[bytes], *, registry: CollisionRegistry | None = None
) -> bytes:
    counted, _ = _counted_sorted(point_ids, item_size=32, name="point_id")
    return _truncated_identifier(
        "leaf_membership_digest_128",
        "TAG_CIIP_LEAF_MEMBERS_V1",
        counted,
        registry=registry,
    )


def field_row_id_digest_128(
    case_digest: bytes,
    arm_u8: int,
    scale_u8: int,
    morton_u64: int,
    *,
    registry: CollisionRegistry | None = None,
) -> bytes:
    arm = _uint(arm_u8, bits=8, name="arm_u8")
    scale = _uint(scale_u8, bits=8, name="scale_u8")
    if arm not in (0, 1):
        raise ValueError("arm_u8 must be BASELINE=0 or ISOTROPIC=1")
    if scale not in (0, 1, 2):
        raise ValueError("scale_u8 must be one of 0, 1, 2")
    morton = _uint(morton_u64, bits=15, name="morton_u64")
    return _truncated_identifier(
        "field_row_id_digest_128",
        "TAG_CIIP_FIELD_ROW_V1",
        _bytes(case_digest, name="case_digest", length=32),
        struct.pack("<BBQ", arm, scale, morton),
        registry=registry,
    )


def support_membership_digest_128(
    scale_u8: int,
    cell_ids: Iterable[bytes],
    *,
    registry: CollisionRegistry | None = None,
) -> bytes:
    scale = _uint(scale_u8, bits=8, name="scale_u8")
    if scale not in (0, 1, 2):
        raise ValueError("scale_u8 must be one of 0, 1, 2")
    counted, _ = _counted_sorted(cell_ids, item_size=32, name="cell_id")
    return _truncated_identifier(
        "support_membership_digest_128",
        "TAG_CIIP_SUPPORT_MEMBERS_V1",
        u8_le(scale, name="scale_u8"),
        counted,
        registry=registry,
    )


def edge_id_digest_128(
    case_digest: bytes,
    lower_vertex_u32: int,
    upper_vertex_u32: int,
    edge_kind_u8: int,
    axis_u8: int,
    *,
    registry: CollisionRegistry | None = None,
) -> bytes:
    lower = _uint(lower_vertex_u32, bits=32, name="lower_vertex_u32")
    upper = _uint(upper_vertex_u32, bits=32, name="upper_vertex_u32")
    if lower >= upper:
        raise ValueError("lower_vertex_u32 must be less than upper_vertex_u32")
    kind = _uint(edge_kind_u8, bits=8, name="edge_kind_u8")
    axis = _uint(axis_u8, bits=8, name="axis_u8")
    if (kind == 0 and axis not in (0, 1, 2)) or (kind == 1 and axis != 255):
        raise ValueError("edge kind and axis do not match the frozen enum relation")
    if kind not in (0, 1):
        raise ValueError("edge_kind_u8 must be SPATIAL_FACE=0 or SCALE=1")
    return _truncated_identifier(
        "edge_id_digest_128",
        "TAG_CIIP_EDGE_V1",
        _bytes(case_digest, name="case_digest", length=32),
        struct.pack("<IIBB", lower, upper, kind, axis),
        registry=registry,
    )


def essential_death_row_id_digest_128(
    case_digest: bytes,
    arm_u8: int,
    birth_row_id_digest_128: bytes,
    *,
    registry: CollisionRegistry | None = None,
) -> bytes:
    arm = _uint(arm_u8, bits=8, name="arm_u8")
    if arm not in (0, 1):
        raise ValueError("arm_u8 must be BASELINE=0 or ISOTROPIC=1")
    return _truncated_identifier(
        "essential_death_row_id_digest_128",
        "TAG_CIIP_ESSENTIAL_DEATH_V1",
        _bytes(case_digest, name="case_digest", length=32),
        u8_le(arm, name="arm_u8"),
        _bytes(birth_row_id_digest_128, name="birth_row_id_digest_128", length=16),
        registry=registry,
    )


def interval_id_digest_128(
    case_digest: bytes,
    arm_u8: int,
    birth_row_id_digest_128: bytes,
    death_row_id_digest_128: bytes,
    birth_f_u16: int,
    death_f_or_essential_u16: int,
    *,
    registry: CollisionRegistry | None = None,
) -> bytes:
    arm = _uint(arm_u8, bits=8, name="arm_u8")
    if arm not in (0, 1):
        raise ValueError("arm_u8 must be BASELINE=0 or ISOTROPIC=1")
    birth = _uint(birth_f_u16, bits=16, name="birth_f_u16")
    death = _uint(death_f_or_essential_u16, bits=16, name="death_f_or_essential_u16")
    if death < birth:
        raise ValueError("death filtration must not precede birth filtration")
    return _truncated_identifier(
        "interval_id_digest_128",
        "TAG_CIIP_INTERVAL_V1",
        _bytes(case_digest, name="case_digest", length=32),
        u8_le(arm, name="arm_u8"),
        _bytes(birth_row_id_digest_128, name="birth_row_id_digest_128", length=16),
        _bytes(death_row_id_digest_128, name="death_row_id_digest_128", length=16),
        struct.pack("<HH", birth, death),
        registry=registry,
    )


def rejection_row_id_digest_128(field_row_id: bytes) -> bytes:
    return _bytes(field_row_id, name="field_row_id", length=16)


def finite_death_row_id_digest_128(edge_id: bytes) -> bytes:
    return _bytes(edge_id, name="edge_id", length=16)


def partition_commitment_sha256(partition_row_count_u64: int, canonical_partition_table_bytes: bytes) -> bytes:
    row_count = _uint(partition_row_count_u64, bits=64, name="partition_row_count_u64")
    table_bytes = _bytes(canonical_partition_table_bytes, name="canonical_partition_table_bytes")
    if len(table_bytes) != row_count * PARTITION_ROW_SIZE_BYTES:
        raise ValueError("canonical partition table length does not match row count")
    return full_digest(
        "TAG_CIIP_PARTITION_COMMIT_V1",
        u64_le(row_count, name="partition_row_count_u64"),
        table_bytes,
    )


@dataclass(frozen=True, slots=True)
class TableDirectoryDescriptor:
    name_enum_u8: int
    row_count_u64: int
    row_size_u32: int
    offset_u64: int
    byte_length_u64: int
    table_sha256: bytes

    def __post_init__(self) -> None:
        _uint(self.name_enum_u8, bits=8, name="name_enum_u8")
        _uint(self.row_count_u64, bits=64, name="row_count_u64")
        _uint(self.row_size_u32, bits=32, name="row_size_u32")
        _uint(self.offset_u64, bits=64, name="offset_u64")
        _uint(self.byte_length_u64, bits=64, name="byte_length_u64")
        object.__setattr__(self, "table_sha256", _bytes(self.table_sha256, name="table_sha256", length=32))

    def canonical_bytes(self) -> bytes:
        return struct.pack(
            "<BQIQQ",
            self.name_enum_u8,
            self.row_count_u64,
            self.row_size_u32,
            self.offset_u64,
            self.byte_length_u64,
        ) + self.table_sha256


def table_directory_commitment_sha256(descriptors: Sequence[TableDirectoryDescriptor]) -> bytes:
    rows = tuple(descriptors)
    if len(rows) != len(TABLE_ROW_SIZES):
        raise ValueError("table directory must contain exactly six descriptors")
    expected_offset = METADATA_REGION_BYTES
    for index, (descriptor, expected_row_size) in enumerate(zip(rows, TABLE_ROW_SIZES, strict=True)):
        if not isinstance(descriptor, TableDirectoryDescriptor):
            raise TypeError("every directory entry must be a TableDirectoryDescriptor")
        if descriptor.name_enum_u8 != index:
            raise ValueError("table descriptors must be in exact enum order")
        if descriptor.row_size_u32 != expected_row_size:
            raise ValueError("table descriptor row size does not match the frozen schema")
        if descriptor.byte_length_u64 != descriptor.row_count_u64 * descriptor.row_size_u32:
            raise ValueError("table descriptor byte length does not match row count and width")
        if descriptor.offset_u64 != expected_offset:
            raise ValueError("table descriptor offsets must be contiguous after metadata")
        expected_offset += descriptor.byte_length_u64
    return full_digest(
        "TAG_CIIP_TABLE_DIRECTORY_V1",
        u32_le(len(rows), name="descriptor_count_u32"),
        *(descriptor.canonical_bytes() for descriptor in rows),
    )


V7_KNOWN_ANSWER_EXPECTED: Final[Mapping[str, str | int]] = MappingProxyType(
    {
        "case_token_digest_sha256": "6392da896a8301fd064f0c7193e192ecefb52d4849dfd249b2c561817aabce47",
        "point_id_sha256": "14f0ea7a432441933b005b92cfe832ecc6a717bece85f9f3671fea8cda98ec93",
        "sample_rank_sha256": "0b2b76b21d689acb8c8e4f81ebd201aaed20d2138380357c3ef3a0b1fc31310f",
        "morton_u64": 53,
        "node_id_sha256": "400ad68152473affe03a3a26fb24f02d48e260032603b51d80aeb3d3fa06b796",
        "subtree_membership_digest_sha256": "2be1a640bfdc6f49db8832e48b6003262b850fde9ef4ebf159550bea9c429a76",
        "cell_id_sha256": "e7761de38cf71fd21ce6f1e60cf27d9a4cf20153f416b68123bcc86b479aec7a",
        "leaf_membership_digest_128": "1370de04d0e5a8e3573f836a823d0e94",
        "field_row_id_digest_128": "ae35aeb5777676ad6ac61aba20f90caa",
        "isotropic_field_row_id_digest_128": "72dc258326cb1f01014240070cc9e59b",
        "support_membership_digest_128": "5833e832aedcf3ee259f0f10d16b0c5d",
        "edge_id_digest_128": "2e5aed76eaf0270e31d51c6027870360",
        "essential_death_row_id_digest_128": "52b54371e52653ac86315d8262321c7e",
        "finite_interval_id_digest_128": "4d597be02e71f80571efc2b60495c56e",
        "essential_interval_id_digest_128": "cf4e90a43b10c1316c240aa309e4cf17",
        "isotropic_essential_death_row_id_digest_128": "9b7ee31a830d9fbf9c08dd1f25205fce",
        "isotropic_finite_interval_id_digest_128": "1bbcb2248cdaadadaf40c12ebbd7535c",
        "isotropic_essential_interval_id_digest_128": "0a1538fb562f41615668cf31c77f8f3e",
        "partition_commitment_sha256": "7327b3d33259deef14deda7142f2951470e80606704691de49bedbe41dd84d13",
        "table_directory_sha256": "43055b40d5844fc87cdbd94002f007e9e92f87b8b777dd76d0c19bc1d9510509",
    }
)


def compute_v7_known_answer_vectors() -> dict[str, str | int]:
    raw_source = bytes(32)
    case_digest = case_token_digest(raw_source)
    xyz = canonical_xyz_binary64_le(0.0, 0.0, 0.0)
    point = point_id_sha256(xyz)
    rank = sample_rank_sha256(point)
    morton = morton5(1, 2, 3)
    node = node_id_sha256(case_digest, 5, morton)
    subtree = subtree_membership_digest_sha256([point])
    cell = cell_id_sha256(case_digest, morton)
    leaf = leaf_membership_digest_128([point])
    field_row = field_row_id_digest_128(case_digest, 0, 1, morton)
    isotropic_field_row = field_row_id_digest_128(case_digest, 1, 1, morton)
    support = support_membership_digest_128(1, [cell])
    edge = edge_id_digest_128(case_digest, 1, 2, 0, 0)
    essential_death = essential_death_row_id_digest_128(case_digest, 0, field_row)
    finite_interval = interval_id_digest_128(case_digest, 0, field_row, edge, 10, 20)
    essential_interval = interval_id_digest_128(
        case_digest, 0, field_row, essential_death, 10, 65_535
    )
    isotropic_essential_death = essential_death_row_id_digest_128(
        case_digest, 1, isotropic_field_row
    )
    isotropic_finite_interval = interval_id_digest_128(
        case_digest, 1, isotropic_field_row, edge, 10, 20
    )
    isotropic_essential_interval = interval_id_digest_128(
        case_digest, 1, isotropic_field_row, isotropic_essential_death, 10, 65_535
    )
    partition_commitment = partition_commitment_sha256(1, bytes(PARTITION_ROW_SIZE_BYTES))
    empty_sha = hashlib.sha256(b"").digest()
    descriptors = tuple(
        TableDirectoryDescriptor(index, 0, row_size, METADATA_REGION_BYTES, 0, empty_sha)
        for index, row_size in enumerate(TABLE_ROW_SIZES)
    )
    directory_commitment = table_directory_commitment_sha256(descriptors)
    return {
        "case_token_digest_sha256": case_digest.hex(),
        "point_id_sha256": point.hex(),
        "sample_rank_sha256": rank.hex(),
        "morton_u64": morton,
        "node_id_sha256": node.hex(),
        "subtree_membership_digest_sha256": subtree.hex(),
        "cell_id_sha256": cell.hex(),
        "leaf_membership_digest_128": leaf.hex(),
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
        "table_directory_sha256": directory_commitment.hex(),
    }


def validate_v7_known_answer_vectors() -> None:
    observed = compute_v7_known_answer_vectors()
    expected = dict(V7_KNOWN_ANSWER_EXPECTED)
    if observed != expected:
        mismatches = sorted(key for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key))
        raise ValueError(f"V7 known-answer vector mismatch: {', '.join(mismatches)}")


# Concise public API names used by the later CIIP paper-only layers.  The
# explicit ``*_sha256``/``*_digest_128`` names above remain available where a
# caller benefits from seeing the encoded width at the call site.
domain_digest = full_digest
point_id = point_id_sha256
sample_rank = sample_rank_sha256
node_id = node_id_sha256
subtree_membership_digest = subtree_membership_digest_sha256
cell_id = cell_id_sha256
leaf_membership_digest = leaf_membership_digest_128
field_row_id = field_row_id_digest_128
support_membership_digest = support_membership_digest_128
edge_id = edge_id_digest_128
essential_death_id = essential_death_row_id_digest_128
interval_id = interval_id_digest_128
partition_commitment = partition_commitment_sha256
table_directory_commitment = table_directory_commitment_sha256
morton_prefix = morton5_prefix


__all__ = [
    "CollisionRegistry",
    "DOMAIN_TAGS",
    "DigestCollisionError",
    "METADATA_REGION_BYTES",
    "PARTITION_ROW_SIZE_BYTES",
    "TABLE_ROW_SIZES",
    "TableDirectoryDescriptor",
    "V7_KNOWN_ANSWER_EXPECTED",
    "canonical_xyz_binary64_le",
    "case_token_digest",
    "cell_id",
    "cell_id_sha256",
    "compute_v7_known_answer_vectors",
    "domain_digest",
    "domain_tag_bytes",
    "edge_id",
    "edge_id_digest_128",
    "essential_death_id",
    "essential_death_row_id_digest_128",
    "field_row_id",
    "field_row_id_digest_128",
    "finite_death_row_id_digest_128",
    "first16_digest",
    "full_digest",
    "interval_id_digest_128",
    "interval_id",
    "leaf_membership_digest",
    "leaf_membership_digest_128",
    "morton5",
    "morton5_prefix",
    "morton_prefix",
    "node_id",
    "node_id_sha256",
    "partition_commitment",
    "partition_commitment_sha256",
    "point_id",
    "point_id_sha256",
    "rejection_row_id_digest_128",
    "sample_rank",
    "sample_rank_sha256",
    "subtree_membership_digest",
    "subtree_membership_digest_sha256",
    "support_membership_digest",
    "support_membership_digest_128",
    "table_directory_commitment",
    "table_directory_commitment_sha256",
    "tagged_preimage",
    "u8_le",
    "u16_le",
    "u32_le",
    "u64_le",
    "validate_v7_known_answer_vectors",
]
