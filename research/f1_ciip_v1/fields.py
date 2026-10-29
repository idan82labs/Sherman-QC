"""Frozen V7 digital support gates and baseline/isotropic field construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .container import numpy_dtype
from .identifiers import (
    CollisionRegistry,
    field_row_id_digest_128,
    support_membership_digest_128,
)
from .partition import PartitionResult


NOT_EVALUATED = -2_147_483_648
VALID = 0
REJECTED = 1


class FieldValidationError(ValueError):
    """Raised when a frozen field/gate invariant cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class FieldResult:
    fields: np.ndarray
    rejections: np.ndarray
    baseline_valid_rows: int
    total_valid_rows: int


def _morton_grid_ordinal(partition: PartitionResult) -> np.ndarray:
    lookup = np.full(1 << 15, -1, dtype=np.int32)
    lookup[partition.leaf_mortons.astype(np.intp)] = np.arange(
        len(partition.leaf_mortons), dtype=np.int32
    )
    return lookup


def _support_ordinals(center: np.ndarray, radius: int, lookup: np.ndarray) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.int16)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    mask = dx * dx + dy * dy + dz * dz <= radius * radius
    candidates = np.column_stack((dx[mask], dy[mask], dz[mask])) + center.astype(np.int16)
    in_range = np.all((candidates >= 0) & (candidates < 32), axis=1)
    candidates = candidates[in_range].astype(np.uint8)
    mortons = np.zeros(len(candidates), dtype=np.uint64)
    for bit in range(5):
        mortons |= ((candidates[:, 0] >> bit) & 1).astype(np.uint64) << (3 * bit)
        mortons |= ((candidates[:, 1] >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        mortons |= ((candidates[:, 2] >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    ordinals = lookup[mortons.astype(np.intp)]
    return ordinals[ordinals >= 0]


def _denominator_id(first_rejection: int, growth12: int, growth23: int) -> int:
    if first_rejection in (0,):
        return 0
    if first_rejection == 1:
        return 1
    if first_rejection == 2:
        return 2
    if first_rejection == 3:
        return 3
    if first_rejection == 4:
        return 4 if growth12 != 0 else 5
    if first_rejection == 5:
        return 6
    if first_rejection == 6:
        return 7
    return 8


def build_fields(
    case_digest: bytes,
    partition: PartitionResult,
    native_worker: object,
    *,
    registry: CollisionRegistry | None = None,
) -> FieldResult:
    """Construct all 2*3*O fixed-width field and rejection rows."""

    o = len(partition.leaf_mortons)
    if not 0 < o <= 32_768:
        raise FieldValidationError("occupied leaf domain is empty or too large")
    occupied = np.zeros(1 << 15, dtype=np.uint8)
    raw_counts = np.zeros(1 << 15, dtype="<u4")
    morton_index = partition.leaf_mortons.astype(np.intp)
    occupied[morton_index] = 1
    raw_counts[morton_index] = partition.leaf_aggregate["raw_record_multiplicity_sum_u32"]
    lookup = _morton_grid_ordinal(partition)
    leaf_xyz = np.ascontiguousarray(partition.leaf_xyz, dtype=np.uint8)
    evidence = native_worker.staged_field_evidence(occupied, raw_counts, leaf_xyz)

    field_rows = np.zeros(6 * o, dtype=numpy_dtype("cell_scale_field"))
    baseline = field_rows[: 3 * o]
    baseline["support_count_u32"] = evidence.support_count
    baseline["mass_q_i64"] = evidence.support_count
    baseline["raw_count_p10_u32"] = evidence.raw_p10
    baseline["raw_count_p90_u32"] = evidence.raw_p90
    baseline["balance_numerator_u32"] = evidence.balance_numerator
    baseline["balance_denominator_u32"] = evidence.balance_denominator
    baseline["growth12_gate_code_i32"] = evidence.growth12
    baseline["growth23_gate_code_i32"] = evidence.growth23
    baseline["trace_q_i64"] = evidence.trace_q
    baseline["lambda1_q_i64"] = evidence.lambda1_q
    baseline["lambda2_q_i64"] = evidence.lambda2_q
    baseline["lambda3_q_i64"] = evidence.lambda3_q
    baseline["anisotropy_q_u16"] = evidence.anisotropy_q
    baseline["filtration_u16"] = evidence.filtration
    baseline["stage_mask_u32"] = evidence.stage_mask
    baseline["first_rejection_u16"] = evidence.first_rejection
    baseline["status_u16"] = np.where(evidence.first_rejection == 0, VALID, REJECTED)
    baseline["eigenvalue_denominator_u32"] = 1

    cell_ids = partition.leaf_aggregate["cell_id_sha256"]
    cursor = 0
    for scale, radius in enumerate((2, 3, 4)):
        for ordinal in range(o):
            row = baseline[cursor]
            morton = int(partition.leaf_mortons[ordinal])
            row["row_id_digest_128"] = np.void(
                field_row_id_digest_128(case_digest, 0, scale, morton, registry=registry)
            )
            support = _support_ordinals(leaf_xyz[ordinal], radius, lookup)
            if len(support) != int(evidence.support_count[cursor]):
                raise FieldValidationError("native/Python support membership disagreement")
            row["support_membership_digest_128"] = np.void(
                support_membership_digest_128(
                    scale, (bytes(cell_ids[index]) for index in support), registry=registry
                )
            )
            row["morton_u64"] = morton
            row["arm_u8"] = 0
            row["scale_u8"] = scale
            cursor += 1

    # Complete same-source isotropic arm.  Preserve every arm-independent
    # quantity and rejection decision; recompute all arm-bound identifiers.
    isotropic = field_rows[3 * o :]
    for index in range(3 * o):
        source = baseline[index]
        target = isotropic[index]
        for name in baseline.dtype.names:
            target[name] = source[name]
        target["arm_u8"] = 1
        target["row_id_digest_128"] = np.void(
            field_row_id_digest_128(
                case_digest,
                1,
                int(source["scale_u8"]),
                int(source["morton_u64"]),
                registry=registry,
            )
        )
        target["lambda1_q_i64"] = source["trace_q_i64"]
        target["lambda2_q_i64"] = source["trace_q_i64"]
        target["lambda3_q_i64"] = source["trace_q_i64"]
        target["eigenvalue_denominator_u32"] = 3
        target["anisotropy_q_u16"] = 0
        target["filtration_u16"] = 65_534

    rejections = np.zeros(6 * o, dtype=numpy_dtype("rejection"))
    for index, field in enumerate(field_rows):
        rejection = rejections[index]
        rejection["row_id_digest_128"] = field["row_id_digest_128"]
        rejection["arm_u8"] = field["arm_u8"]
        rejection["scale_u8"] = field["scale_u8"]
        rejection["first_rejection_u16"] = field["first_rejection_u16"]
        rejection["stage_mask_u32"] = field["stage_mask_u32"]
        rejection["denominator_id_u16"] = _denominator_id(
            int(field["first_rejection_u16"]),
            int(field["growth12_gate_code_i32"]),
            int(field["growth23_gate_code_i32"]),
        )
    baseline_valid = int(np.count_nonzero(baseline["status_u16"] == VALID))
    total_valid = int(np.count_nonzero(field_rows["status_u16"] == VALID))
    if (3 * o - baseline_valid) * 10 > 9 * (3 * o):
        raise FieldValidationError("invalid baseline field fraction exceeds 9/10")
    return FieldResult(field_rows, rejections, baseline_valid, total_valid)


__all__ = ["FieldResult", "FieldValidationError", "build_fields"]
