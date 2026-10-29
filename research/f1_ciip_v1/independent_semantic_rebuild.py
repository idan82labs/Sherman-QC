"""Independent raw-XYZ reconstruction of the six frozen CIIP-V1 tables.

This implementation intentionally shares no parser, partition, field, graph,
semantic-core, or native-worker code.  Only the neutral fixed-width schema and
identifier codecs are imported.  It is deliberately straightforward: the
independent lane is an oracle, not a production shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping

import numpy as np

from .container import TABLE_NAMES, numpy_dtype
from .identifiers import (
    case_token_digest,
    cell_id_sha256,
    domain_tag_bytes,
    edge_id_digest_128,
    essential_death_row_id_digest_128,
    field_row_id_digest_128,
    interval_id_digest_128,
    node_id_sha256,
    point_id_sha256,
    sample_rank_sha256,
    support_membership_digest_128,
)


class IndependentRebuildError(ValueError):
    """The independent implementation could not reproduce the frozen domain."""


@dataclass(frozen=True, slots=True)
class IndependentTables:
    arrays: Mapping[str, np.ndarray]
    table_bytes: Mapping[str, bytes]
    unique_records: int
    sampled_unique_records: int
    occupied_cells: int


@dataclass(frozen=True, slots=True)
class IndependentWorkBound:
    occupied_cells: int
    support_candidate_checks: int
    connectivity_neighbor_checks: int
    jacobi_pivots: int
    h0_events: int


@dataclass(frozen=True, slots=True)
class _Partition:
    nodes: np.ndarray
    leaves: np.ndarray
    mortons: np.ndarray
    leaf_xyz: np.ndarray


NOT_EVALUATED = -2_147_483_648


_INDEPENDENT_KAT_EXPECTED = {
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


def compute_independent_known_answer_vectors() -> dict[str, str | int]:
    """Recompute all V7 identifier/commitment KATs without production codecs."""

    def tagged(tag: bytes, *parts: bytes) -> bytes:
        return hashlib.sha256(tag + b"\0" + b"".join(parts)).digest()

    raw = bytes(32)
    case = tagged(b"CIIP_CASE_V1", raw)
    point = tagged(b"CIIP_POINT_V1", struct.pack("<ddd", 0.0, 0.0, 0.0))
    rank = tagged(b"CIIP_SAMPLE_RANK_V1", point)
    morton = sum(
        ((coordinate >> bit) & 1) << (3 * bit + axis)
        for axis, coordinate in enumerate((1, 2, 3))
        for bit in range(5)
    )
    node = tagged(b"CIIP_NODE_V1", case, struct.pack("<BQ", 5, morton))
    counted_point = struct.pack("<I", 1) + point
    subtree = tagged(b"CIIP_SUBTREE_MEMBERS_V1", counted_point)
    cell = tagged(b"CIIP_CELL_V1", case, struct.pack("<Q", morton))
    leaf = tagged(b"CIIP_LEAF_MEMBERS_V1", counted_point)[:16]
    field = tagged(b"CIIP_FIELD_ROW_V1", case, struct.pack("<BBQ", 0, 1, morton))[:16]
    isotropic_field = tagged(b"CIIP_FIELD_ROW_V1", case, struct.pack("<BBQ", 1, 1, morton))[:16]
    support = tagged(b"CIIP_SUPPORT_MEMBERS_V1", struct.pack("<BI", 1, 1), cell)[:16]
    edge = tagged(b"CIIP_EDGE_V1", case, struct.pack("<IIBB", 1, 2, 0, 0))[:16]
    death = tagged(b"CIIP_ESSENTIAL_DEATH_V1", case, b"\0", field)[:16]
    finite = tagged(b"CIIP_INTERVAL_V1", case, b"\0", field, edge, struct.pack("<HH", 10, 20))[:16]
    essential = tagged(
        b"CIIP_INTERVAL_V1", case, b"\0", field, death, struct.pack("<HH", 10, 65_535)
    )[:16]
    isotropic_death = tagged(b"CIIP_ESSENTIAL_DEATH_V1", case, b"\1", isotropic_field)[:16]
    isotropic_finite = tagged(
        b"CIIP_INTERVAL_V1", case, b"\1", isotropic_field, edge, struct.pack("<HH", 10, 20)
    )[:16]
    isotropic_essential = tagged(
        b"CIIP_INTERVAL_V1", case, b"\1", isotropic_field, isotropic_death,
        struct.pack("<HH", 10, 65_535),
    )[:16]
    partition = tagged(b"CIIP_PARTITION_COMMIT_V1", struct.pack("<Q", 1), bytes(88))
    empty = hashlib.sha256(b"").digest()
    directory = bytearray(b"CIIP_TABLE_DIRECTORY_V1\0" + struct.pack("<I", 6))
    for enum, row_size in enumerate((88, 64, 128, 32, 64, 32)):
        directory.extend(struct.pack("<BQIQQ32s", enum, 0, row_size, 1_048_576, 0, empty))
    return {
        "case_token_digest_sha256": case.hex(),
        "point_id_sha256": point.hex(),
        "sample_rank_sha256": rank.hex(),
        "morton_u64": morton,
        "node_id_sha256": node.hex(),
        "subtree_membership_digest_sha256": subtree.hex(),
        "cell_id_sha256": cell.hex(),
        "leaf_membership_digest_128": leaf.hex(),
        "field_row_id_digest_128": field.hex(),
        "isotropic_field_row_id_digest_128": isotropic_field.hex(),
        "support_membership_digest_128": support.hex(),
        "edge_id_digest_128": edge.hex(),
        "essential_death_row_id_digest_128": death.hex(),
        "finite_interval_id_digest_128": finite.hex(),
        "essential_interval_id_digest_128": essential.hex(),
        "isotropic_essential_death_row_id_digest_128": isotropic_death.hex(),
        "isotropic_finite_interval_id_digest_128": isotropic_finite.hex(),
        "isotropic_essential_interval_id_digest_128": isotropic_essential.hex(),
        "partition_commitment_sha256": partition.hex(),
        "table_directory_sha256": hashlib.sha256(directory).hexdigest(),
    }


def validate_independent_known_answer_vectors() -> None:
    observed = compute_independent_known_answer_vectors()
    if observed != _INDEPENDENT_KAT_EXPECTED:
        mismatches = sorted(
            key for key in observed.keys() | _INDEPENDENT_KAT_EXPECTED.keys()
            if observed.get(key) != _INDEPENDENT_KAT_EXPECTED.get(key)
        )
        raise IndependentRebuildError(f"INDEPENDENT_KAT_MISMATCH:{','.join(mismatches)}")


def _sphere_offsets(radius: int) -> np.ndarray:
    values = np.arange(-radius, radius + 1, dtype=np.int16)
    dx, dy, dz = np.meshgrid(values, values, values, indexing="ij")
    keep = dx * dx + dy * dy + dz * dz <= radius * radius
    return np.ascontiguousarray(np.column_stack((dx[keep], dy[keep], dz[keep])), dtype=np.int16)


_OFFSETS = {radius: _sphere_offsets(radius) for radius in (2, 3, 4)}
_FACE_DIRECTIONS = np.asarray(
    ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)),
    dtype=np.int16,
)


def independent_work_upper_bound(occupied_cells: int) -> IndependentWorkBound:
    """Return a source-free worst-case operation census for one rebuild role."""

    if isinstance(occupied_cells, bool) or not isinstance(occupied_cells, int) or not 0 < occupied_cells <= 32_768:
        raise IndependentRebuildError("OCCUPIED_CELL_BOUND")
    stencil_sum = sum(len(_OFFSETS[radius]) for radius in (2, 3, 4))
    # Five gated support passes plus one support-membership reconstruction pass.
    support_checks = 6 * stencil_sum * occupied_cells
    return IndependentWorkBound(
        occupied_cells=occupied_cells,
        support_candidate_checks=support_checks,
        connectivity_neighbor_checks=6 * stencil_sum * occupied_cells,
        jacobi_pivots=3 * occupied_cells * 12 * 3,
        h0_events=2 * (3 * occupied_cells + 11 * occupied_cells),
    )


def _canonical_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or xyz.dtype.kind != "f" or len(xyz) == 0:
        raise IndependentRebuildError("XYZ_DOMAIN_INVALID")
    xyz = np.asarray(xyz, dtype="<f8", order="C").copy()
    if not bool(np.isfinite(xyz).all()):
        raise IndependentRebuildError("NONFINITE_COORDINATE")
    xyz[xyz == 0.0] = 0.0

    packed = xyz.view("V24").reshape(len(xyz))
    _values, first, counts = np.unique(packed, return_index=True, return_counts=True)
    unique_xyz = np.ascontiguousarray(xyz[first], dtype="<f8")
    multiplicity = np.ascontiguousarray(counts, dtype=np.uint32)
    unique_count = len(unique_xyz)

    point_ids = np.empty((unique_count, 32), dtype=np.uint8)
    ranks = np.empty((unique_count, 32), dtype=np.uint8)
    unique_packed = unique_xyz.view(np.uint8).reshape(unique_count, 24)
    for index in range(unique_count):
        point_id = point_id_sha256(bytes(unique_packed[index]))
        point_ids[index] = np.frombuffer(point_id, dtype=np.uint8)
        ranks[index] = np.frombuffer(sample_rank_sha256(point_id), dtype=np.uint8)

    point_order = np.argsort(point_ids.view("V32").reshape(unique_count), kind="stable")
    ordered_ids = point_ids[point_order]
    for position in np.flatnonzero(np.all(ordered_ids[1:] == ordered_ids[:-1], axis=1)):
        if not np.array_equal(
            unique_packed[point_order[position]], unique_packed[point_order[position + 1]]
        ):
            raise IndependentRebuildError("POINT_ID_COLLISION")
    rank_order = np.argsort(ranks.view("V32").reshape(unique_count), kind="stable")
    ordered_ranks = ranks[rank_order]
    for position in np.flatnonzero(np.all(ordered_ranks[1:] == ordered_ranks[:-1], axis=1)):
        if not np.array_equal(point_ids[rank_order[position]], point_ids[rank_order[position + 1]]):
            raise IndependentRebuildError("SAMPLE_RANK_COLLISION")

    rank_and_id = np.empty((unique_count, 64), dtype=np.uint8)
    rank_and_id[:, :32] = ranks
    rank_and_id[:, 32:] = point_ids
    sample_order = np.argsort(rank_and_id.view("V64").reshape(unique_count), kind="stable")
    take = sample_order[: min(unique_count, 200_000)]
    sampled = np.ascontiguousarray(unique_xyz[take])
    sampled_ids = np.ascontiguousarray(point_ids[take])
    sampled_multiplicity = np.ascontiguousarray(multiplicity[take])
    return sampled, sampled_ids, sampled_multiplicity, unique_count


def _morton(xyz: np.ndarray) -> np.ndarray:
    result = np.zeros(len(xyz), dtype=np.uint64)
    for bit in range(5):
        for axis in range(3):
            result |= ((xyz[:, axis] >> bit) & 1).astype(np.uint64) << (3 * bit + axis)
    return result


def _decode_morton(values: np.ndarray) -> np.ndarray:
    mortons = np.asarray(values, dtype=np.uint64)
    xyz = np.zeros((len(mortons), 3), dtype=np.uint8)
    for bit in range(5):
        for axis in range(3):
            xyz[:, axis] |= ((mortons >> (3 * bit + axis)) & 1).astype(np.uint8) << bit
    return xyz


def _counted_ids(point_ids: np.ndarray) -> bytes:
    values = np.ascontiguousarray(point_ids, dtype=np.uint8).reshape(-1, 32)
    ordered = np.sort(values.view("V32").reshape(-1), kind="stable")
    return len(ordered).to_bytes(4, "little") + ordered.tobytes()


def _partition(
    case: bytes,
    xyz: np.ndarray,
    point_ids: np.ndarray,
    multiplicity: np.ndarray,
    registry: object | None,
) -> _Partition:
    minimum = np.min(xyz, axis=0)
    ranges = np.max(xyz, axis=0) - minimum
    diameter = float(np.max(ranges))
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise IndependentRebuildError("ROOT_RANGE_INVALID")
    root_side = float(float(diameter * 4.0) / 3.0)
    center = minimum + (ranges / 2.0)
    origin = center - (root_side / 2.0)
    cell_side = float(root_side / 32.0)
    grid_xyz = np.floor((xyz - origin) / cell_side).astype(np.int64)
    if bool(np.any(grid_xyz < 0) or np.any(grid_xyz >= 32)):
        raise IndependentRebuildError("POINT_OUTSIDE_ROOT")
    mortons = _morton(grid_xyz)
    order = np.lexsort((point_ids.view("V32").reshape(len(xyz)), mortons))
    sorted_mortons = mortons[order]
    sorted_ids = point_ids[order]
    sorted_mult = multiplicity[order]
    starts = np.flatnonzero(np.r_[True, sorted_mortons[1:] != sorted_mortons[:-1]])
    ends = np.r_[starts[1:], len(sorted_mortons)]
    leaves_morton = sorted_mortons[starts]
    occupied = len(leaves_morton)
    if not 0 < occupied <= 32_768:
        raise IndependentRebuildError("OCCUPIED_CELL_BOUND")

    leaves = np.zeros(occupied, dtype=numpy_dtype("leaf_aggregate"))
    for ordinal, (start, end) in enumerate(zip(starts, ends, strict=True)):
        morton = int(leaves_morton[ordinal])
        counted = _counted_ids(sorted_ids[start:end])
        preimage = domain_tag_bytes("TAG_CIIP_LEAF_MEMBERS_V1") + counted
        digest = hashlib.sha256(preimage).digest()[:16]
        if registry is not None:
            registry.record("leaf_membership_digest_128", digest, preimage)
        leaves["cell_id_sha256"][ordinal] = np.void(cell_id_sha256(case, morton))
        leaves["morton_u64"][ordinal] = morton
        leaves["membership_digest_128"][ordinal] = np.void(digest)
        leaves["unique_point_count_u32"][ordinal] = end - start
        leaves["raw_record_multiplicity_sum_u32"][ordinal] = int(
            np.sum(sorted_mult[start:end], dtype=np.uint64)
        )

    spec_dtype = np.dtype([
        ("depth", "u1"), ("prefix", "<u8"), ("leaf_start", "<u4"),
        ("leaf_end", "<u4"), ("point_start", "<u4"), ("point_end", "<u4"),
        ("child_mask", "u1"),
    ])
    specs = np.empty(2 * occupied - 1, dtype=spec_dtype)
    spec_count = 0
    for depth in range(6):
        shift = 3 * (5 - depth)
        prefixes = leaves_morton >> shift if shift else leaves_morton
        prefix_starts = np.flatnonzero(np.r_[True, prefixes[1:] != prefixes[:-1]])
        prefix_ends = np.r_[prefix_starts[1:], occupied]
        for leaf_start, leaf_end in zip(prefix_starts, prefix_ends, strict=True):
            prefix = int(prefixes[leaf_start])
            if depth < 5:
                child_shift = 3 * (4 - depth)
                children = np.unique((leaves_morton[leaf_start:leaf_end] >> child_shift) & 7)
                if len(children) == 1:
                    continue
                child_mask = sum(1 << int(child) for child in children)
            else:
                child_mask = 0
            specs[spec_count] = (
                depth, prefix, int(leaf_start), int(leaf_end), int(starts[leaf_start]),
                int(ends[leaf_end - 1]), child_mask,
            )
            spec_count += 1
    specs = specs[:spec_count]
    specs.sort(order=("depth", "prefix"), kind="stable")
    nodes = np.zeros(len(specs), dtype=numpy_dtype("partition_node"))
    for index, spec in enumerate(specs):
        depth, prefix = int(spec["depth"]), int(spec["prefix"])
        leaf_start, leaf_end = int(spec["leaf_start"]), int(spec["leaf_end"])
        point_start, point_end = int(spec["point_start"]), int(spec["point_end"])
        child_mask = int(spec["child_mask"])
        counted = _counted_ids(sorted_ids[point_start:point_end])
        nodes["node_id_sha256"][index] = np.void(node_id_sha256(case, depth, prefix))
        nodes["morton_prefix_u64"][index] = prefix
        nodes["subtree_membership_digest_sha256"][index] = np.void(
            hashlib.sha256(domain_tag_bytes("TAG_CIIP_SUBTREE_MEMBERS_V1") + counted).digest()
        )
        nodes["subtree_point_count_u32"][index] = point_end - point_start
        nodes["occupied_leaf_count_u32"][index] = leaf_end - leaf_start
        nodes["depth_u8"][index] = depth
        nodes["child_mask_u8"][index] = child_mask
    if not occupied <= len(nodes) <= 2 * occupied - 1:
        raise IndependentRebuildError("PATRICIA_ROW_BOUND")
    return _Partition(nodes, leaves, leaves_morton, grid_xyz[order][starts].astype(np.uint8))


def _support_stage(
    center: np.ndarray,
    radius: int,
    occupied: np.ndarray,
    raw_counts: np.ndarray,
    stage: int,
) -> object:
    offsets = _OFFSETS[radius]
    candidates = offsets + np.asarray(center, dtype=np.int16)
    in_range = np.all((candidates >= 0) & (candidates < 32), axis=1)
    candidates = candidates[in_range]
    offsets = offsets[in_range]
    mortons = _morton(candidates.astype(np.int64, copy=False))
    present = occupied[mortons.astype(np.intp)] != 0
    coords_array = np.ascontiguousarray(candidates[present], dtype=np.int16)
    offsets = offsets[present]
    mortons = mortons[present]
    if not len(coords_array):
        raise IndependentRebuildError("EMPTY_SUPPORT")
    if stage == 0:
        return len(coords_array)
    if stage == 4:
        count_values = np.sort(raw_counts[mortons.astype(np.intp)])
        n = len(count_values)
        return int(count_values[(n + 9) // 10 - 1]), int(count_values[(9 * n + 9) // 10 - 1])
    if stage == 5:
        values = coords_array.astype(np.int64, copy=False)
        x, y, z = values[:, 0], values[:, 1], values[:, 2]
        return np.asarray(
            [x.sum(), y.sum(), z.sum(), (x*x).sum(), (y*y).sum(), (z*z).sum(),
             (x*y).sum(), (x*z).sum(), (y*z).sum()], dtype=np.int64,
        )
    if stage == 2:
        oriented = offsets[np.any(offsets != 0, axis=1)]
        first_nonzero = np.where(
            oriented[:, 0] != 0,
            oriented[:, 0],
            np.where(oriented[:, 1] != 0, oriented[:, 1], oriented[:, 2]),
        )
        plus = int(np.count_nonzero(first_nonzero > 0))
        minus = int(np.count_nonzero(first_nonzero < 0))
        return min(plus, minus), max(plus, minus)
    if stage != 1:
        raise IndependentRebuildError("SUPPORT_STAGE_INVALID")

    width = 2 * radius + 1
    local = coords_array - (np.asarray(center, dtype=np.int16) - radius)
    local_indices = (
        local[:, 0] + width * (local[:, 1] + width * local[:, 2])
    ).astype(np.int32)
    local_mask = np.zeros(width**3, dtype=np.uint8)
    visited = np.zeros(width**3, dtype=np.uint8)
    queue = np.empty(len(local_indices), dtype=np.int32)
    local_mask[local_indices] = 1
    components = 0
    for seed in range(width**3):
        if local_mask[seed] == 0 or visited[seed] != 0:
            continue
        components += 1
        head, tail = 0, 1
        queue[0] = seed
        visited[seed] = 1
        while head < tail:
            current = int(queue[head])
            head += 1
            lx = current % width
            ly = (current // width) % width
            lz = current // (width * width)
            for direction in _FACE_DIRECTIONS:
                nx, ny, nz = lx + int(direction[0]), ly + int(direction[1]), lz + int(direction[2])
                if min(nx, ny, nz) < 0 or max(nx, ny, nz) >= width:
                    continue
                neighbor = nx + width * (ny + width * nz)
                if local_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue[tail] = neighbor
                    tail += 1
    return components


def _growth(n1: int, n2: int, n3: int) -> tuple[int, int]:
    p1, p2, p3 = n1**4, n2**4, n3**4
    g12 = -1 if 32*p2 < 243*p1 else (1 if 2048*p2 > 177147*p1 else 0)
    g23 = -1 if 243*p3 < 1024*p2 else (1 if 177147*p3 > 4194304*p2 else 0)
    return g12, g23


def _support_ordinals(center: np.ndarray, radius: int, lookup: np.ndarray) -> np.ndarray:
    candidates = _OFFSETS[radius] + np.asarray(center, dtype=np.int16)
    candidates = candidates[np.all((candidates >= 0) & (candidates < 32), axis=1)]
    ordinals = lookup[_morton(candidates.astype(np.int64, copy=False)).astype(np.intp)]
    return ordinals[ordinals >= 0]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            accumulator = 0.0
            for k in range(3):
                accumulator = float(accumulator + float(left[i][k] * right[k][j]))
            out[i][j] = accumulator
    return out


def _jacobi(upper: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    matrix = [[upper[0], upper[1], upper[2]], [upper[1], upper[3], upper[4]], [upper[2], upper[4], upper[5]]]
    for _sweep in range(12):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = matrix[p][q]
            if apq == 0.0:
                continue
            tau = float(float(matrix[q][q] - matrix[p][p]) / float(2.0 * apq))
            if not math.isfinite(tau):
                raise IndependentRebuildError("JACOBI_NONFINITE")
            t = 1.0
            if tau != 0.0:
                root = math.sqrt(float(1.0 + float(tau * tau)))
                t = float((-1.0 if math.copysign(1.0, tau) < 0 else 1.0) / float(abs(tau) + root))
            c = float(1.0 / math.sqrt(float(1.0 + float(t * t))))
            s = float(t * c)
            rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            rotation[p][p] = rotation[q][q] = c
            rotation[p][q], rotation[q][p] = s, -s
            first = _matmul(matrix, rotation)
            updated = _matmul([[rotation[j][i] for j in range(3)] for i in range(3)], first)
            for i in range(3):
                for j in range(i, 3):
                    matrix[i][j] = updated[i][j]
                    matrix[j][i] = matrix[i][j]
    trace = float(float(matrix[0][0] + matrix[1][1]) + matrix[2][2])
    tolerance = math.ldexp(trace, -40)
    if (
        not math.isfinite(trace)
        or trace <= 0.0
        or not math.isfinite(tolerance)
        or max(abs(matrix[0][1]), abs(matrix[0][2]), abs(matrix[1][2])) > tolerance
    ):
        raise IndependentRebuildError("JACOBI_RESIDUAL")
    eigen = []
    for axis in range(3):
        value = matrix[axis][axis]
        if value < 0.0:
            if abs(value) <= tolerance:
                value = 0.0
            else:
                raise IndependentRebuildError("NEGATIVE_EIGENVALUE")
        eigen.append((value, axis))
    eigen.sort(key=lambda item: (-item[0], item[1]))
    return eigen[0][0], eigen[1][0], eigen[2][0]


def _q24_40(value: float) -> int:
    if not math.isfinite(value):
        raise IndependentRebuildError("Q24_40_NONFINITE")
    quantized = round(math.ldexp(value, 40))
    if not -(1 << 63) <= quantized < 1 << 63:
        raise IndependentRebuildError("Q24_40_OVERFLOW")
    return quantized


def _field_tables(case: bytes, partition: _Partition, registry: object | None) -> tuple[np.ndarray, np.ndarray]:
    o = len(partition.mortons)
    occupied = np.zeros(1 << 15, dtype=np.uint8)
    raw_counts = np.zeros(1 << 15, dtype="<u4")
    occupied[partition.mortons.astype(np.intp)] = 1
    raw_counts[partition.mortons.astype(np.intp)] = partition.leaves["raw_record_multiplicity_sum_u32"]
    cell_ids = partition.leaves["cell_id_sha256"]
    lookup = np.full(1 << 15, -1, dtype=np.int32)
    lookup[partition.mortons.astype(np.intp)] = np.arange(o, dtype=np.int32)
    support_count = np.zeros((3, o), dtype=np.uint32)
    moments = np.zeros((3, o, 9), dtype=np.int64)
    p10 = np.zeros((3, o), dtype=np.uint32)
    p90 = np.zeros((3, o), dtype=np.uint32)
    balance_num = np.zeros((3, o), dtype=np.uint32)
    balance_den = np.zeros((3, o), dtype=np.uint32)
    components = np.zeros((3, o), dtype=np.uint32)
    growth12 = np.full((3, o), NOT_EVALUATED, dtype=np.int32)
    growth23 = np.full((3, o), NOT_EVALUATED, dtype=np.int32)
    trace_q = np.zeros((3, o), dtype=np.int64)
    lambdas = np.zeros((3, o, 3), dtype=np.int64)
    anisotropy_q = np.zeros((3, o), dtype=np.uint16)
    filtration = np.full((3, o), 65_534, dtype=np.uint16)
    stage_mask = np.zeros((3, o), dtype=np.uint32)
    first_rejection = np.zeros((3, o), dtype=np.uint16)
    active = np.ones((3, o), dtype=np.bool_)
    minimum_support = (9, 21, 37)

    # Stages 0..2: no later support evidence is computed for a rejected row.
    for scale, radius in enumerate((2, 3, 4)):
        for ordinal, center in enumerate(partition.leaf_xyz):
            support_count[scale, ordinal] = int(_support_stage(center, radius, occupied, raw_counts, 0))
            stage_mask[scale, ordinal] |= 1
            if support_count[scale, ordinal] < minimum_support[scale]:
                first_rejection[scale, ordinal] = 1
                active[scale, ordinal] = False
        for ordinal in np.flatnonzero(active[scale]):
            stage_mask[scale, ordinal] |= 2
            components[scale, ordinal] = int(
                _support_stage(partition.leaf_xyz[ordinal], radius, occupied, raw_counts, 1)
            )
            if components[scale, ordinal] != 1:
                first_rejection[scale, ordinal] = 2
                active[scale, ordinal] = False
        for ordinal in np.flatnonzero(active[scale]):
            stage_mask[scale, ordinal] |= 4
            numerator, denominator = _support_stage(
                partition.leaf_xyz[ordinal], radius, occupied, raw_counts, 2
            )
            balance_num[scale, ordinal], balance_den[scale, ordinal] = numerator, denominator
            if denominator == 0 or 20 * numerator < 11 * denominator:
                first_rejection[scale, ordinal] = 3
                active[scale, ordinal] = False

    # Stage 3: nested growth uses complete stage-0 counts, but emits evidence
    # only for the row whose prior three gates survived.
    for scale in range(3):
        for ordinal in np.flatnonzero(active[scale]):
            stage_mask[scale, ordinal] |= 8
            g12, g23 = _growth(*(int(support_count[index, ordinal]) for index in range(3)))
            growth12[scale, ordinal], growth23[scale, ordinal] = g12, g23
            if g12 or g23:
                first_rejection[scale, ordinal] = 4
                active[scale, ordinal] = False

    # Stages 4..6: density, moments/trace, then numeric tensor evidence.
    for scale, radius in enumerate((2, 3, 4)):
        for ordinal in np.flatnonzero(active[scale]):
            stage_mask[scale, ordinal] |= 16
            low, high = _support_stage(partition.leaf_xyz[ordinal], radius, occupied, raw_counts, 4)
            p10[scale, ordinal], p90[scale, ordinal] = low, high
            if high > 32 * max(1, low):
                first_rejection[scale, ordinal] = 5
                active[scale, ordinal] = False
        for ordinal in np.flatnonzero(active[scale]):
            stage_mask[scale, ordinal] |= 32
            moments[scale, ordinal] = _support_stage(
                partition.leaf_xyz[ordinal], radius, occupied, raw_counts, 5
            )
            count = int(support_count[scale, ordinal])
            moment = moments[scale, ordinal]
            denominator = float(count)
            means = [float(int(moment[index])) / denominator for index in range(3)]
            def centered(second: int, left: float, right: float) -> float:
                return float(float(float(second) / denominator) - float(left * right))
            diagonal = 1.0 / 12.0
            upper = (
                float(centered(int(moment[3]), means[0], means[0]) + diagonal),
                centered(int(moment[6]), means[0], means[1]),
                centered(int(moment[7]), means[0], means[2]),
                float(centered(int(moment[4]), means[1], means[1]) + diagonal),
                centered(int(moment[8]), means[1], means[2]),
                float(centered(int(moment[5]), means[2], means[2]) + diagonal),
            )
            trace = float(float(upper[0] + upper[3]) + upper[5])
            if not math.isfinite(trace) or trace <= 0.0:
                first_rejection[scale, ordinal] = 6
                active[scale, ordinal] = False
                continue
            stage_mask[scale, ordinal] |= 64
            try:
                trace_q[scale, ordinal] = _q24_40(trace)
                eigen = _jacobi(upper)
                trace_e = float(float(eigen[0] + eigen[1]) + eigen[2])
                scalar = float(
                    float((eigen[0] - eigen[1]) / trace_e)
                    * float((eigen[1] - eigen[2]) / trace_e)
                )
                if not math.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
                    raise IndependentRebuildError("ANISOTROPY_INVALID")
                q = round(float(scalar * 65_534.0))
                if not 0 <= q <= 65_534:
                    raise IndependentRebuildError("ANISOTROPY_QUANTIZATION_INVALID")
                quantized = tuple(_q24_40(value) for value in eigen)
            except IndependentRebuildError:
                first_rejection[scale, ordinal] = 7
                active[scale, ordinal] = False
            else:
                lambdas[scale, ordinal] = quantized
                trace_q[scale, ordinal] = sum(quantized)
                anisotropy_q[scale, ordinal], filtration[scale, ordinal] = q, 65_534 - q

    rows = np.zeros(6 * o, dtype=numpy_dtype("cell_scale_field"))
    baseline = rows[:3*o]
    baseline["support_count_u32"] = support_count.reshape(-1)
    baseline["mass_q_i64"] = support_count.reshape(-1)
    baseline["raw_count_p10_u32"], baseline["raw_count_p90_u32"] = p10.reshape(-1), p90.reshape(-1)
    baseline["balance_numerator_u32"], baseline["balance_denominator_u32"] = balance_num.reshape(-1), balance_den.reshape(-1)
    baseline["growth12_gate_code_i32"], baseline["growth23_gate_code_i32"] = growth12.reshape(-1), growth23.reshape(-1)
    baseline["trace_q_i64"] = trace_q.reshape(-1)
    baseline["lambda1_q_i64"], baseline["lambda2_q_i64"], baseline["lambda3_q_i64"] = (
        lambdas[:, :, 0].reshape(-1), lambdas[:, :, 1].reshape(-1), lambdas[:, :, 2].reshape(-1)
    )
    baseline["anisotropy_q_u16"], baseline["filtration_u16"] = anisotropy_q.reshape(-1), filtration.reshape(-1)
    baseline["stage_mask_u32"], baseline["first_rejection_u16"] = stage_mask.reshape(-1), first_rejection.reshape(-1)
    baseline["status_u16"] = (first_rejection.reshape(-1) != 0).astype(np.uint16)
    baseline["eigenvalue_denominator_u32"] = 1
    cursor = 0
    for scale, radius in enumerate((2, 3, 4)):
        for ordinal in range(o):
            row = baseline[cursor]
            morton = int(partition.mortons[ordinal])
            row["row_id_digest_128"] = np.void(field_row_id_digest_128(case, 0, scale, morton, registry=registry))
            support_ordinals = _support_ordinals(partition.leaf_xyz[ordinal], radius, lookup)
            row["support_membership_digest_128"] = np.void(
                support_membership_digest_128(scale, (bytes(cell_ids[i]) for i in support_ordinals), registry=registry)
            )
            row["morton_u64"] = morton
            row["arm_u8"], row["scale_u8"] = 0, scale
            cursor += 1

    isotropic = rows[3*o:]
    for index in range(3*o):
        for name in rows.dtype.names:
            isotropic[index][name] = baseline[index][name]
        source, target = baseline[index], isotropic[index]
        target["arm_u8"] = 1
        target["row_id_digest_128"] = np.void(field_row_id_digest_128(
            case, 1, int(source["scale_u8"]), int(source["morton_u64"]), registry=registry,
        ))
        for name in ("lambda1_q_i64", "lambda2_q_i64", "lambda3_q_i64"):
            target[name] = source["trace_q_i64"]
        target["eigenvalue_denominator_u32"] = 3
        target["anisotropy_q_u16"], target["filtration_u16"] = 0, 65_534

    rejections = np.zeros(6*o, dtype=numpy_dtype("rejection"))
    for index, field in enumerate(rows):
        rejection = rejections[index]
        rejection["row_id_digest_128"] = field["row_id_digest_128"]
        rejection["arm_u8"], rejection["scale_u8"] = field["arm_u8"], field["scale_u8"]
        rejection["first_rejection_u16"], rejection["stage_mask_u32"] = field["first_rejection_u16"], field["stage_mask_u32"]
        first = int(field["first_rejection_u16"])
        if first == 0: denominator_id = 0
        elif first == 1: denominator_id = 1
        elif first == 2: denominator_id = 2
        elif first == 3: denominator_id = 3
        elif first == 4: denominator_id = 4 if int(field["growth12_gate_code_i32"]) != 0 else 5
        elif first == 5: denominator_id = 6
        elif first == 6: denominator_id = 7
        else: denominator_id = 8
        rejection["denominator_id_u16"] = denominator_id
    return rows, rejections


def _adjacency(case: bytes, mortons: np.ndarray, registry: object | None) -> np.ndarray:
    o = len(mortons)
    xyz = _decode_morton(mortons)
    ordinal = np.full(1 << 15, -1, dtype=np.int32)
    ordinal[mortons.astype(np.intp)] = np.arange(o, dtype=np.int32)
    neighbors_by_axis = np.empty((3, o), dtype=np.int32)
    spatial_count = 0
    for axis in range(3):
        candidate = xyz.astype(np.int16)
        valid = candidate[:, axis] < 31
        candidate[:, axis] += 1
        candidate[:, axis] = np.minimum(candidate[:, axis], 31)
        neighbors = np.where(valid, ordinal[_morton(candidate).astype(np.intp)], -1).astype(np.int32)
        neighbors_by_axis[axis] = neighbors
        spatial_count += int(np.count_nonzero(neighbors >= 0))
    rows = np.zeros(3 * spatial_count + 2 * o, dtype=numpy_dtype("space_scale_adjacency"))
    cursor = 0
    for axis in range(3):
        neighbors = neighbors_by_axis[axis]
        source = np.flatnonzero(neighbors >= 0).astype(np.uint32)
        target = neighbors[source].astype(np.uint32)
        for scale in range(3):
            count = len(source)
            segment = rows[cursor:cursor+count]
            lower, upper = source + np.uint32(scale*o), target + np.uint32(scale*o)
            segment["lower_vertex_u32"] = np.minimum(lower, upper)
            segment["upper_vertex_u32"] = np.maximum(lower, upper)
            segment["edge_kind_u8"], segment["axis_u8"] = 0, axis
            cursor += count
    spatial = rows[:cursor]
    spatial_order = np.lexsort((spatial["upper_vertex_u32"], spatial["lower_vertex_u32"], spatial["axis_u8"]))
    rows[:cursor] = spatial[spatial_order]
    for scale in range(2):
        segment = rows[cursor:cursor+o]
        segment["lower_vertex_u32"] = np.arange(o,dtype=np.uint32) + np.uint32(scale*o)
        segment["upper_vertex_u32"] = np.arange(o,dtype=np.uint32) + np.uint32((scale+1)*o)
        segment["edge_kind_u8"], segment["axis_u8"] = 1, 255
        cursor += o
    for index in range(len(rows)):
        rows["edge_id_digest_128"][index] = np.void(edge_id_digest_128(
            case, int(rows["lower_vertex_u32"][index]), int(rows["upper_vertex_u32"][index]),
            int(rows["edge_kind_u8"][index]), int(rows["axis_u8"][index]), registry=registry,
        ))
    return rows


def _reason(birth_q: int, lifetime: int, scale_mask: int, spatial: int, valid: bool) -> int:
    if birth_q <= 0: return 1
    if lifetime < 2: return 2
    if scale_mask.bit_count() < 2: return 3
    if spatial < 1: return 4
    if not valid: return 5
    return 0


def _persistence(case: bytes, arm: int, fields: np.ndarray, edges: np.ndarray, registry: object | None) -> np.ndarray:
    v = len(fields)
    edge_f = np.maximum(fields["filtration_u16"][edges["lower_vertex_u32"]], fields["filtration_u16"][edges["upper_vertex_u32"]]).astype(np.uint16)
    event_dtype = np.dtype([("f","<u2"),("kind","u1"),("a","<u4"),("b","<u4"),("c","u1"),("d","u1"),("payload","<u4")])
    events = np.empty(v+len(edges), dtype=event_dtype)
    events[:v]["f"], events[:v]["kind"], events[:v]["a"], events[:v]["b"], events[:v]["c"], events[:v]["d"], events[:v]["payload"] = fields["filtration_u16"], 0, np.arange(v,dtype=np.uint32), 0, 0, 0, np.arange(v,dtype=np.uint32)
    ee = events[v:]
    ee["f"], ee["kind"], ee["a"], ee["b"], ee["c"], ee["d"], ee["payload"] = edge_f, 1, edges["lower_vertex_u32"], edges["upper_vertex_u32"], edges["edge_kind_u8"], edges["axis_u8"], np.arange(len(edges),dtype=np.uint32)
    order = np.argsort(events, order=("f","kind","c","d","a","b"), kind="stable")
    parent=np.arange(v,dtype=np.int32); rank=np.zeros(v,dtype=np.uint8); active=np.zeros(v,dtype=bool)
    owner=np.arange(v,dtype=np.uint32); masks=np.zeros(v,dtype=np.uint8); spatial=np.zeros(v,dtype=np.uint32); valid=np.ones(v,dtype=bool)
    output=np.zeros(v,dtype=numpy_dtype("persistence_disposition")); emitted=np.zeros(v,dtype=bool)
    def find(value: int) -> int:
        root=value
        while int(parent[root]) != root: root=int(parent[root])
        while int(parent[value]) != value:
            nxt=int(parent[value]); parent[value]=root; value=nxt
        return root
    def emit(birth: int, death_f: int, status: int, death_id: bytes, root: int) -> None:
        birth_f=int(fields["filtration_u16"][birth]); lifetime=65_535-birth_f if status else death_f-birth_f
        birth_id=bytes(fields["row_id_digest_128"][birth]); row=output[birth]
        row["interval_id_digest_128"]=np.void(interval_id_digest_128(case,arm,birth_id,death_id,birth_f,death_f,registry=registry))
        row["birth_row_id_digest_128"], row["death_row_id_digest_128"] = np.void(birth_id), np.void(death_id)
        row["arm_u8"], row["status_u8"] = arm, status
        row["reason_u16"]=_reason(int(fields["anisotropy_q_u16"][birth]),lifetime,int(masks[root]),int(spatial[root]),bool(valid[root]))
        row["birth_f_u16"], row["death_f_or_essential_u16"], row["lifetime_u16"] = birth_f, death_f, lifetime
        row["scale_mask_u8"], row["spatial_face_edge_count_u32"] = masks[root], spatial[root]
        emitted[birth]=True
    for event_index in order:
        event=events[int(event_index)]
        if int(event["kind"]) == 0:
            vertex=int(event["payload"]); active[vertex]=True; masks[vertex]=1<<int(fields["scale_u8"][vertex]); valid[vertex]=int(fields["status_u16"][vertex])==0; continue
        edge=edges[int(event["payload"])]; left,right=int(edge["lower_vertex_u32"]),int(edge["upper_vertex_u32"])
        rl,rr=find(left),find(right); is_spatial=int(edge["edge_kind_u8"])==0
        if rl == rr:
            if is_spatial: spatial[rl]+=1
            continue
        ol,orr=int(owner[rl]),int(owner[rr])
        if (int(fields["filtration_u16"][ol]),ol) <= (int(fields["filtration_u16"][orr]),orr): survivor,loser,loser_root=ol,orr,rr
        else: survivor,loser,loser_root=orr,ol,rl
        emit(loser,int(event["f"]),0,bytes(edge["edge_id_digest_128"]),loser_root)
        if rank[rl] < rank[rr] or (rank[rl] == rank[rr] and rr < rl): rl,rr=rr,rl
        parent[rr]=rl
        if rank[rl] == rank[rr]: rank[rl]+=1
        owner[rl]=survivor; masks[rl]=int(masks[rl])|int(masks[rr]); spatial[rl]=int(spatial[rl])+int(spatial[rr])+int(is_spatial); valid[rl]=bool(valid[rl] and valid[rr])
    roots=np.unique(np.fromiter((find(i) for i in range(v)),dtype=np.int32,count=v))
    for root in roots:
        birth=int(owner[int(root)]); birth_id=bytes(fields["row_id_digest_128"][birth])
        death=essential_death_row_id_digest_128(case,arm,birth_id,registry=registry)
        emit(birth,65_535,1,death,int(root))
    if not bool(emitted.all()): raise IndependentRebuildError("INCOMPLETE_PERSISTENCE")
    return output


def _require_unique_digest(rows: np.ndarray, field: str, code: str) -> None:
    digests = np.ascontiguousarray(rows[field]).reshape(len(rows)).view("V16")
    if len(digests) > 1:
        ordered = np.sort(digests, kind="stable")
        if np.any(ordered[1:] == ordered[:-1]):
            raise IndependentRebuildError(code)


def _support_preimage(partition: _Partition, scale: int, ordinal: int, lookup: np.ndarray) -> bytes:
    support = _support_ordinals(partition.leaf_xyz[ordinal], scale + 2, lookup)
    cell_ids = np.ascontiguousarray(partition.leaves["cell_id_sha256"][support]).view("V32").reshape(-1)
    ordered = np.sort(cell_ids, kind="stable")
    return bytes((scale,)) + len(ordered).to_bytes(4, "little") + ordered.tobytes()


def _audit_truncated_collisions(partition: _Partition, arrays: Mapping[str, np.ndarray]) -> None:
    """Audit digest classes with fixed arrays; reconstruct support preimages only on a tie."""

    _require_unique_digest(partition.leaves, "membership_digest_128", "LEAF_MEMBERSHIP_DIGEST_COLLISION")
    fields = arrays["cell_scale_field"]
    _require_unique_digest(fields, "row_id_digest_128", "FIELD_ROW_ID_COLLISION")
    _require_unique_digest(arrays["space_scale_adjacency"], "edge_id_digest_128", "EDGE_ID_COLLISION")
    _require_unique_digest(arrays["persistence_disposition"], "interval_id_digest_128", "INTERVAL_ID_COLLISION")
    persistence = arrays["persistence_disposition"]
    essential = persistence[persistence["status_u8"] == 1]
    _require_unique_digest(essential, "death_row_id_digest_128", "ESSENTIAL_DEATH_ID_COLLISION")

    o = len(partition.mortons)
    baseline = fields[: 3 * o]
    digests = np.ascontiguousarray(baseline["support_membership_digest_128"]).view("V16").reshape(-1)
    order = np.argsort(digests, kind="stable")
    lookup = np.full(1 << 15, -1, dtype=np.int32)
    lookup[partition.mortons.astype(np.intp)] = np.arange(o, dtype=np.int32)
    for left, right in zip(order[:-1], order[1:]):
        if digests[left] != digests[right]:
            continue
        left_index, right_index = int(left), int(right)
        if _support_preimage(partition, left_index // o, left_index % o, lookup) != _support_preimage(
            partition, right_index // o, right_index % o, lookup
        ):
            raise IndependentRebuildError("SUPPORT_MEMBERSHIP_DIGEST_COLLISION")


def rebuild_semantic_tables(points: np.ndarray, raw_source_sha256: str) -> IndependentTables:
    """Reconstruct every frozen table from raw parsed XYZ without semantic imports."""
    validate_independent_known_answer_vectors()
    if not isinstance(raw_source_sha256, str) or len(raw_source_sha256) != 64:
        raise IndependentRebuildError("RAW_SHA256_INVALID")
    try: raw_digest=bytes.fromhex(raw_source_sha256)
    except ValueError as exc: raise IndependentRebuildError("RAW_SHA256_INVALID") from exc
    case=case_token_digest(raw_digest); registry = None
    xyz, point_ids, multiplicity, unique_count = _canonical_points(points)
    partition=_partition(case,xyz,point_ids,multiplicity,registry)
    fields,rejections=_field_tables(case,partition,registry)
    adjacency=_adjacency(case,partition.mortons,registry)
    o=len(partition.mortons)
    persistence=np.concatenate((
        _persistence(case,0,fields[:3*o],adjacency,registry),
        _persistence(case,1,fields[3*o:],adjacency,registry),
    ))
    arrays={
        "partition_node":partition.nodes, "leaf_aggregate":partition.leaves,
        "cell_scale_field":fields, "space_scale_adjacency":adjacency,
        "persistence_disposition":persistence, "rejection":rejections,
    }
    _audit_truncated_collisions(partition, arrays)
    table_bytes={name:np.ascontiguousarray(arrays[name],dtype=numpy_dtype(name)).tobytes() for name in TABLE_NAMES}
    return IndependentTables(arrays,table_bytes,unique_count,len(xyz),o)


__all__ = [
    "compute_independent_known_answer_vectors",
    "IndependentRebuildError",
    "IndependentTables",
    "IndependentWorkBound",
    "independent_work_upper_bound",
    "rebuild_semantic_tables",
    "validate_independent_known_answer_vectors",
]
