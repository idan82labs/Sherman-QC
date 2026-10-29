"""Canonical coordinate-only sampling and sparse depth-five CIIP partition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np

from .container import numpy_dtype
from .identifiers import (
    CollisionRegistry,
    cell_id_sha256,
    domain_tag_bytes,
    node_id_sha256,
    point_id_sha256,
    sample_rank_sha256,
)


class PartitionValidationError(ValueError):
    """Raised when sampling or grid assignment violates the frozen domain."""


@dataclass(frozen=True, slots=True)
class SampledPoints:
    xyz: np.ndarray
    point_ids: np.ndarray
    multiplicities: np.ndarray
    unique_record_count: int


@dataclass(frozen=True, slots=True)
class PartitionResult:
    partition_nodes: np.ndarray
    leaf_aggregate: np.ndarray
    leaf_mortons: np.ndarray
    leaf_xyz: np.ndarray
    sampled_point_leaf_ordinal: np.ndarray
    root_origin: np.ndarray
    root_side: float
    cell_side: float


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(xyz)
    if values.dtype.kind != "f" or values.shape[-1:] != (3,) or values.ndim != 2:
        raise PartitionValidationError("XYZ must be one finite N-by-3 floating array")
    values = np.asarray(values, dtype="<f8", order="C").copy()
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise PartitionValidationError("XYZ must be nonempty and finite")
    values[values == 0.0] = 0.0
    return values


def canonical_sample(
    unique_xyz: np.ndarray,
    multiplicities: np.ndarray,
    *,
    sample_cap: int = 200_000,
) -> SampledPoints:
    """Assign coordinate-only IDs and retain the lowest rank/ID sequence."""

    xyz = _normalize_xyz(unique_xyz)
    raw_multiplicity = np.asarray(multiplicities)
    if (
        raw_multiplicity.shape != (len(xyz),)
        or raw_multiplicity.dtype.kind not in "iu"
        or raw_multiplicity.dtype.kind == "b"
        or np.any(raw_multiplicity < 1)
        or np.any(raw_multiplicity > np.iinfo(np.uint32).max)
    ):
        raise PartitionValidationError("multiplicities must be positive u32 values aligned to XYZ")
    multiplicity = np.asarray(raw_multiplicity, dtype=np.uint32)
    if isinstance(sample_cap, bool) or not isinstance(sample_cap, int) or not 0 < sample_cap <= 200_000:
        raise PartitionValidationError("sample cap must be in the frozen 1..200000 domain")

    n = len(xyz)
    point_ids = np.empty((n, 32), dtype=np.uint8)
    ranks = np.empty((n, 32), dtype=np.uint8)
    packed = xyz.view(np.uint8).reshape(n, 24)
    for index in range(n):
        pid = point_id_sha256(bytes(packed[index]))
        point_ids[index] = np.frombuffer(pid, dtype=np.uint8)
        ranks[index] = np.frombuffer(sample_rank_sha256(pid), dtype=np.uint8)
    point_order = np.argsort(point_ids.view("V32").reshape(n), kind="stable")
    ordered_ids = point_ids[point_order]
    equal_point_ids = np.all(ordered_ids[1:] == ordered_ids[:-1], axis=1)
    if np.any(equal_point_ids):
        collision_positions = np.flatnonzero(equal_point_ids)
        ordered_xyz = packed[point_order]
        if any(
            not np.array_equal(ordered_xyz[position], ordered_xyz[position + 1])
            for position in collision_positions
        ):
            raise PartitionValidationError("unequal XYZ preimages share one point ID")
    rank_order = np.argsort(ranks.view("V32").reshape(n), kind="stable")
    ordered_ranks = ranks[rank_order]
    equal_ranks = np.all(ordered_ranks[1:] == ordered_ranks[:-1], axis=1)
    if np.any(equal_ranks):
        collision_positions = np.flatnonzero(equal_ranks)
        ordered_point_ids = point_ids[rank_order]
        if any(
            not np.array_equal(ordered_point_ids[position], ordered_point_ids[position + 1])
            for position in collision_positions
        ):
            raise PartitionValidationError("unequal point IDs share one sample-rank digest")
    combined = np.empty((n, 64), dtype=np.uint8)
    combined[:, :32] = ranks
    combined[:, 32:] = point_ids
    order = np.argsort(combined.view("V64").reshape(n), kind="stable")
    take = order[: min(n, sample_cap)]
    return SampledPoints(
        xyz=np.ascontiguousarray(xyz[take]),
        point_ids=np.ascontiguousarray(point_ids[take]),
        multiplicities=np.ascontiguousarray(multiplicity[take]),
        unique_record_count=n,
    )


def _morton_array(indices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(indices), dtype=np.uint64)
    for bit in range(5):
        result |= ((indices[:, 0] >> bit) & 1).astype(np.uint64) << (3 * bit)
        result |= ((indices[:, 1] >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        result |= ((indices[:, 2] >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    return result


def _sorted_point_id_payload(point_ids: np.ndarray) -> bytes:
    if len(point_ids) == 0:
        return (0).to_bytes(4, "little")
    order = np.argsort(point_ids.view("V32").reshape(len(point_ids)), kind="stable")
    ordered = np.ascontiguousarray(point_ids[order])
    return len(ordered).to_bytes(4, "little") + ordered.tobytes(order="C")


def build_partition(
    case_digest: bytes,
    sampled: SampledPoints,
    *,
    registry: CollisionRegistry | None = None,
) -> PartitionResult:
    xyz = _normalize_xyz(sampled.xyz)
    point_ids = np.asarray(sampled.point_ids, dtype=np.uint8)
    multiplicity = np.asarray(sampled.multiplicities, dtype=np.uint32)
    if point_ids.shape != (len(xyz), 32) or multiplicity.shape != (len(xyz),):
        raise PartitionValidationError("sample arrays are not aligned")

    minimum = np.min(xyz, axis=0)
    maximum = np.max(xyz, axis=0)
    ranges = maximum - minimum
    diameter = float(np.max(ranges))
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise PartitionValidationError("sampled axis range must be positive and finite")
    root_side = float(float(diameter * 4.0) / 3.0)
    center = minimum + (ranges / 2.0)
    origin = center - (root_side / 2.0)
    cell_side = float(root_side / 32.0)
    assigned = np.floor((xyz - origin) / cell_side).astype(np.int64)
    if np.any(assigned < 0) or np.any(assigned >= 32):
        raise PartitionValidationError("sampled point assigned outside depth-five root")
    morton = _morton_array(assigned)

    point_key = point_ids.view("V32").reshape(len(point_ids))
    order = np.lexsort((point_key, morton))
    morton_sorted = morton[order]
    point_sorted = point_ids[order]
    multiplicity_sorted = multiplicity[order]
    starts = np.flatnonzero(np.r_[True, morton_sorted[1:] != morton_sorted[:-1]])
    ends = np.r_[starts[1:], len(morton_sorted)]
    leaf_mortons = morton_sorted[starts]
    o = len(leaf_mortons)
    if not 0 < o <= 32_768:
        raise PartitionValidationError("occupied leaf count violates frozen bound")
    point_leaf = np.empty(len(morton_sorted), dtype=np.uint32)
    leaves = np.zeros(o, dtype=numpy_dtype("leaf_aggregate"))
    for ordinal, (start, end) in enumerate(zip(starts, ends, strict=True)):
        value = int(leaf_mortons[ordinal])
        members = _sorted_point_id_payload(point_sorted[start:end])
        leaves["cell_id_sha256"][ordinal] = np.void(cell_id_sha256(case_digest, value))
        leaves["morton_u64"][ordinal] = value
        leaf_preimage = domain_tag_bytes("TAG_CIIP_LEAF_MEMBERS_V1") + members
        leaf_digest = hashlib.sha256(leaf_preimage).digest()[:16]
        if registry is not None:
            registry.record("leaf_membership_digest_128", leaf_digest, leaf_preimage)
        leaves["membership_digest_128"][ordinal] = np.void(leaf_digest)
        leaves["unique_point_count_u32"][ordinal] = end - start
        leaves["raw_record_multiplicity_sum_u32"][ordinal] = int(
            np.sum(multiplicity_sorted[start:end], dtype=np.uint64)
        )
        point_leaf[order[start:end]] = ordinal

    # Logical trie prefixes and point/leaf ranges.  Retain leaves and every
    # branching node; unary logical paths are omitted (Patricia compression).
    node_spec_dtype = np.dtype(
        [
            ("depth", "u1"), ("prefix", "<u8"), ("leaf_start", "<u4"),
            ("leaf_end", "<u4"), ("point_start", "<u4"), ("point_end", "<u4"),
            ("child_mask", "u1"),
        ]
    )
    node_specs = np.empty(2 * o - 1, dtype=node_spec_dtype)
    node_cursor = 0
    for depth in range(6):
        shift = 3 * (5 - depth)
        prefixes = leaf_mortons >> shift if shift else leaf_mortons
        prefix_starts = np.flatnonzero(np.r_[True, prefixes[1:] != prefixes[:-1]])
        prefix_ends = np.r_[prefix_starts[1:], o]
        for leaf_start, leaf_end in zip(prefix_starts, prefix_ends, strict=True):
            prefix = int(prefixes[leaf_start])
            if depth < 5:
                child_shift = 3 * (4 - depth)
                children = np.unique((leaf_mortons[leaf_start:leaf_end] >> child_shift) & 7)
                child_mask = sum(1 << int(child) for child in children)
                if len(children) == 1:
                    continue
            else:
                child_mask = 0
            point_start = int(starts[leaf_start])
            point_end = int(ends[leaf_end - 1])
            node_specs[node_cursor] = (
                depth, prefix, leaf_start, leaf_end, point_start, point_end, child_mask
            )
            node_cursor += 1
    node_specs = node_specs[:node_cursor]
    node_specs.sort(order=("depth", "prefix"), kind="stable")
    nodes = np.zeros(node_cursor, dtype=numpy_dtype("partition_node"))
    for index, spec in enumerate(node_specs):
        depth = int(spec["depth"])
        prefix = int(spec["prefix"])
        leaf_start = int(spec["leaf_start"])
        leaf_end = int(spec["leaf_end"])
        point_start = int(spec["point_start"])
        point_end = int(spec["point_end"])
        child_mask = int(spec["child_mask"])
        members = _sorted_point_id_payload(point_sorted[point_start:point_end])
        nodes["node_id_sha256"][index] = np.void(node_id_sha256(case_digest, depth, prefix))
        nodes["morton_prefix_u64"][index] = prefix
        nodes["subtree_membership_digest_sha256"][index] = np.void(
            hashlib.sha256(domain_tag_bytes("TAG_CIIP_SUBTREE_MEMBERS_V1") + members).digest()
        )
        nodes["subtree_point_count_u32"][index] = point_end - point_start
        nodes["occupied_leaf_count_u32"][index] = leaf_end - leaf_start
        nodes["depth_u8"][index] = depth
        nodes["child_mask_u8"][index] = child_mask
    if not o <= len(nodes) <= 2 * o - 1:
        raise PartitionValidationError("Patricia row count violates actual-O bound")
    leaf_xyz = assigned[order][starts].astype(np.uint8)
    return PartitionResult(nodes, leaves, leaf_mortons, leaf_xyz, point_leaf, origin, root_side, cell_side)


__all__ = [
    "PartitionResult",
    "PartitionValidationError",
    "SampledPoints",
    "build_partition",
    "canonical_sample",
]
