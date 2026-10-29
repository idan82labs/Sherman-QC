"""Deterministic finite space-scale graph and H0 persistence for CIIP-V1.

The routines operate only on fixed-width arrays derived from the frozen V7
partition/field domains.  They do not parse source files and do not access any
labels, CAD, expected counts, or benchmark metadata.
"""

from __future__ import annotations

import numpy as np

from .container import numpy_dtype
from .identifiers import (
    CollisionRegistry,
    edge_id_digest_128,
    essential_death_row_id_digest_128,
    interval_id_digest_128,
)


SPATIAL_EDGE = 0
SCALE_EDGE = 1
ESSENTIAL = 1
FINITE = 0


class GraphValidationError(ValueError):
    """Raised when the fixed V7 graph domain or persistence state is invalid."""


def _decode_morton5_array(mortons: np.ndarray) -> np.ndarray:
    values = np.asarray(mortons, dtype=np.uint64)
    xyz = np.zeros((len(values), 3), dtype=np.uint8)
    for bit in range(5):
        xyz[:, 0] |= ((values >> (3 * bit)) & 1).astype(np.uint8) << bit
        xyz[:, 1] |= ((values >> (3 * bit + 1)) & 1).astype(np.uint8) << bit
        xyz[:, 2] |= ((values >> (3 * bit + 2)) & 1).astype(np.uint8) << bit
    return xyz


def _python_build_space_scale_adjacency(
    case_digest: bytes,
    leaf_mortons: np.ndarray,
    *,
    registry: CollisionRegistry | None = None,
) -> np.ndarray:
    """Build exact three-scale positive-face and adjacent-scale edges."""

    mortons = np.asarray(leaf_mortons, dtype=np.uint64)
    if mortons.ndim != 1 or not 0 < len(mortons) <= 32_768:
        raise GraphValidationError("leaf Morton domain must contain 1..32768 rows")
    if np.any(mortons >= (1 << 15)) or np.any(mortons[1:] <= mortons[:-1]):
        raise GraphValidationError("leaf Morton rows must be strict ascending 15-bit values")
    o = len(mortons)
    xyz = _decode_morton5_array(mortons)
    ordinal = np.full(1 << 15, -1, dtype=np.int32)
    ordinal[mortons.astype(np.intp)] = np.arange(o, dtype=np.int32)

    spatial_count = 0
    neighbor_ordinals: list[np.ndarray] = []
    for axis in range(3):
        candidate = xyz.copy()
        valid = candidate[:, axis] < 31
        candidate[:, axis] += 1
        neighbor_morton = np.zeros(o, dtype=np.uint64)
        for bit in range(5):
            neighbor_morton |= ((candidate[:, 0] >> bit) & 1).astype(np.uint64) << (3 * bit)
            neighbor_morton |= ((candidate[:, 1] >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
            neighbor_morton |= ((candidate[:, 2] >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
        neighbors = np.where(valid, ordinal[neighbor_morton.astype(np.intp)], -1).astype(np.int32)
        neighbor_ordinals.append(neighbors)
        spatial_count += int(np.count_nonzero(neighbors >= 0))

    total = 3 * spatial_count + 2 * o
    rows = np.zeros(total, dtype=numpy_dtype("space_scale_adjacency"))
    cursor = 0
    # Frozen canonical edge order: spatial kind, axis, lower, upper.
    for axis, neighbors in enumerate(neighbor_ordinals):
        source = np.flatnonzero(neighbors >= 0).astype(np.uint32)
        target = neighbors[source].astype(np.uint32)
        for scale in range(3):
            count = len(source)
            lower = source + np.uint32(scale * o)
            upper = target + np.uint32(scale * o)
            segment = rows[cursor : cursor + count]
            segment["lower_vertex_u32"] = np.minimum(lower, upper)
            segment["upper_vertex_u32"] = np.maximum(lower, upper)
            segment["edge_kind_u8"] = SPATIAL_EDGE
            segment["axis_u8"] = axis
            cursor += count
    # Sort spatial rows because loops above are axis,scale,ordinal and canonical
    # order is axis,lower,upper.
    spatial = rows[:cursor]
    spatial_order = np.lexsort((spatial["upper_vertex_u32"], spatial["lower_vertex_u32"], spatial["axis_u8"]))
    rows[:cursor] = spatial[spatial_order]

    for scale in range(2):
        count = o
        segment = rows[cursor : cursor + count]
        segment["lower_vertex_u32"] = np.arange(o, dtype=np.uint32) + np.uint32(scale * o)
        segment["upper_vertex_u32"] = np.arange(o, dtype=np.uint32) + np.uint32((scale + 1) * o)
        segment["edge_kind_u8"] = SCALE_EDGE
        segment["axis_u8"] = 255
        cursor += count
    if cursor != total:
        raise GraphValidationError("adjacency preallocation count mismatch")

    for index in range(total):
        rows["edge_id_digest_128"][index] = np.void(
            edge_id_digest_128(
                case_digest,
                int(rows["lower_vertex_u32"][index]),
                int(rows["upper_vertex_u32"][index]),
                int(rows["edge_kind_u8"][index]),
                int(rows["axis_u8"][index]),
                registry=registry,
            )
        )
    if not 2 * o <= total <= 11 * o:
        raise GraphValidationError("adjacency row count violates actual-O bound")
    return rows


def _reason(
    birth_q: int,
    lifetime: int,
    scale_mask: int,
    spatial_edges: int,
    all_valid: bool,
) -> int:
    if birth_q <= 0:
        return 1
    if lifetime < 2:
        return 2
    if int(scale_mask).bit_count() < 2:
        return 3
    if spatial_edges < 1:
        return 4
    if not all_valid:
        return 5
    return 0


def _python_h0_persistence(
    case_digest: bytes,
    arm_u8: int,
    field_rows_for_arm: np.ndarray,
    adjacency: np.ndarray,
    *,
    registry: CollisionRegistry | None = None,
) -> tuple[np.ndarray, int]:
    """Compute one disposition per vertex with frozen union/elder semantics."""

    fields = np.asarray(field_rows_for_arm)
    edges = np.asarray(adjacency)
    v = len(fields)
    if arm_u8 not in (0, 1) or v == 0 or v % 3:
        raise GraphValidationError("field arm must contain exactly 3*O vertices")
    if np.any(fields["arm_u8"] != arm_u8):
        raise GraphValidationError("field rows do not belong to requested arm")
    if np.any(edges["upper_vertex_u32"] >= v):
        raise GraphValidationError("adjacency endpoint outside field domain")

    edge_f = np.maximum(
        fields["filtration_u16"][edges["lower_vertex_u32"]],
        fields["filtration_u16"][edges["upper_vertex_u32"]],
    ).astype(np.uint16)
    # One stable total order over V vertices and E edges. Kind 0 is vertex,
    # kind 1 is edge. Edge subkeys exactly match the frozen tie rule.
    event_dtype = np.dtype(
        [("f", "<u2"), ("kind", "u1"), ("a", "<u4"), ("b", "<u4"), ("c", "u1"), ("d", "u1"), ("payload", "<u4")]
    )
    events = np.empty(v + len(edges), dtype=event_dtype)
    events[:v]["f"] = fields["filtration_u16"]
    events[:v]["kind"] = 0
    events[:v]["a"] = np.arange(v, dtype=np.uint32)
    events[:v]["b"] = 0
    events[:v]["c"] = 0
    events[:v]["d"] = 0
    events[:v]["payload"] = np.arange(v, dtype=np.uint32)
    event_edges = events[v:]
    event_edges["f"] = edge_f
    event_edges["kind"] = 1
    event_edges["a"] = edges["lower_vertex_u32"]
    event_edges["b"] = edges["upper_vertex_u32"]
    event_edges["c"] = edges["edge_kind_u8"]
    event_edges["d"] = edges["axis_u8"]
    event_edges["payload"] = np.arange(len(edges), dtype=np.uint32)
    order = np.argsort(events, order=("f", "kind", "c", "d", "a", "b"), kind="stable")

    parent = np.arange(v, dtype=np.int32)
    rank = np.zeros(v, dtype=np.uint8)
    active = np.zeros(v, dtype=np.bool_)
    owner = np.arange(v, dtype=np.uint32)
    scale_mask = np.zeros(v, dtype=np.uint8)
    spatial_count = np.zeros(v, dtype=np.uint32)
    all_valid = np.ones(v, dtype=np.bool_)
    output = np.zeros(v, dtype=numpy_dtype("persistence_disposition"))
    emitted = np.zeros(v, dtype=np.bool_)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def emit(birth_vertex: int, death_f: int, status: int, death_id: bytes, root: int) -> int:
        if emitted[birth_vertex]:
            raise GraphValidationError("persistence owner emitted more than once")
        birth_f = int(fields["filtration_u16"][birth_vertex])
        lifetime = (65_535 - birth_f) if status == ESSENTIAL else (death_f - birth_f)
        reason = _reason(
            int(fields["anisotropy_q_u16"][birth_vertex]),
            lifetime,
            int(scale_mask[root]),
            int(spatial_count[root]),
            bool(all_valid[root]),
        )
        birth_id = bytes(fields["row_id_digest_128"][birth_vertex])
        row = output[birth_vertex]
        row["interval_id_digest_128"] = np.void(
            interval_id_digest_128(
                case_digest, arm_u8, birth_id, death_id, birth_f, death_f, registry=registry
            )
        )
        row["birth_row_id_digest_128"] = np.void(birth_id)
        row["death_row_id_digest_128"] = np.void(death_id)
        row["arm_u8"] = arm_u8
        row["status_u8"] = status
        row["reason_u16"] = reason
        row["birth_f_u16"] = birth_f
        row["death_f_or_essential_u16"] = death_f
        row["lifetime_u16"] = lifetime
        row["scale_mask_u8"] = scale_mask[root]
        row["spatial_face_edge_count_u32"] = spatial_count[root]
        emitted[birth_vertex] = True
        return int(reason == 0)

    attributed = 0
    for event_index in order:
        event = events[int(event_index)]
        if int(event["kind"]) == 0:
            vertex = int(event["payload"])
            active[vertex] = True
            scale = int(fields["scale_u8"][vertex])
            scale_mask[vertex] = np.uint8(1 << scale)
            all_valid[vertex] = bool(fields["status_u16"][vertex] == 0)
            continue
        edge_index = int(event["payload"])
        edge = edges[edge_index]
        left = int(edge["lower_vertex_u32"])
        right = int(edge["upper_vertex_u32"])
        if not active[left] or not active[right]:
            raise GraphValidationError("lower-star edge processed before an endpoint")
        root_left, root_right = find(left), find(right)
        spatial = int(edge["edge_kind_u8"]) == SPATIAL_EDGE
        if root_left == root_right:
            if spatial:
                spatial_count[root_left] += np.uint32(1)
            continue

        owner_left, owner_right = int(owner[root_left]), int(owner[root_right])
        elder_left = (int(fields["filtration_u16"][owner_left]), owner_left)
        elder_right = (int(fields["filtration_u16"][owner_right]), owner_right)
        if elder_left <= elder_right:
            survivor_owner, loser_owner, loser_root = owner_left, owner_right, root_right
        else:
            survivor_owner, loser_owner, loser_root = owner_right, owner_left, root_left
        death_id = bytes(edge["edge_id_digest_128"])
        attributed += emit(loser_owner, int(event["f"]), FINITE, death_id, loser_root)

        # Structural union by rank, then smaller structural root id.
        if rank[root_left] < rank[root_right] or (
            rank[root_left] == rank[root_right] and root_right < root_left
        ):
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += np.uint8(1)
        owner[root_left] = survivor_owner
        scale_mask[root_left] = np.uint8(int(scale_mask[root_left]) | int(scale_mask[root_right]))
        spatial_count[root_left] = np.uint32(
            int(spatial_count[root_left]) + int(spatial_count[root_right]) + int(spatial)
        )
        all_valid[root_left] = bool(all_valid[root_left] and all_valid[root_right])

    roots = np.unique(np.fromiter((find(index) for index in range(v)), dtype=np.int32, count=v))
    for root in roots:
        birth_vertex = int(owner[int(root)])
        birth_id = bytes(fields["row_id_digest_128"][birth_vertex])
        death_id = essential_death_row_id_digest_128(
            case_digest, arm_u8, birth_id, registry=registry
        )
        attributed += emit(birth_vertex, 65_535, ESSENTIAL, death_id, int(root))
    if not np.all(emitted):
        raise GraphValidationError("persistence did not emit exactly one row per vertex")
    return output, attributed


def build_space_scale_adjacency(
    case_digest: bytes,
    leaf_mortons: np.ndarray,
    *,
    registry: CollisionRegistry | None = None,
    native_worker: object | None = None,
) -> np.ndarray:
    """Build adjacency natively in production; retain Python as a test oracle."""

    if native_worker is None:
        return _python_build_space_scale_adjacency(
            case_digest, leaf_mortons, registry=registry
        )
    return native_worker.build_adjacency(case_digest, leaf_mortons)


def h0_persistence(
    case_digest: bytes,
    arm_u8: int,
    field_rows_for_arm: np.ndarray,
    adjacency: np.ndarray,
    *,
    registry: CollisionRegistry | None = None,
    native_worker: object | None = None,
) -> tuple[np.ndarray, int]:
    """Run H0 natively in production; retain Python as a byte-exact oracle."""

    if native_worker is None:
        return _python_h0_persistence(
            case_digest,
            arm_u8,
            field_rows_for_arm,
            adjacency,
            registry=registry,
        )
    return native_worker.h0_persistence(
        case_digest, arm_u8, field_rows_for_arm, adjacency
    )


__all__ = ["GraphValidationError", "build_space_scale_adjacency", "h0_persistence"]
