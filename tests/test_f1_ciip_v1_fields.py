from __future__ import annotations

import numpy as np

from research.f1_ciip_v1.container import numpy_dtype
from research.f1_ciip_v1.fields import build_fields
from research.f1_ciip_v1.identifiers import case_token_digest, cell_id_sha256, morton5
from research.f1_ciip_v1.native_worker import StagedFieldEvidence, build_and_load
from research.f1_ciip_v1.partition import PartitionResult


def _plane_partition() -> tuple[bytes, PartitionResult]:
    case = case_token_digest(bytes.fromhex("11" * 32))
    points = [(x, y, 16) for x in range(8, 23) for y in range(8, 23)]
    ordered = sorted((morton5(*xyz), xyz) for xyz in points)
    mortons = np.asarray([item[0] for item in ordered], dtype=np.uint64)
    xyz = np.asarray([item[1] for item in ordered], dtype=np.uint8)
    leaves = np.zeros(len(mortons), dtype=numpy_dtype("leaf_aggregate"))
    for index, morton in enumerate(mortons):
        leaves["cell_id_sha256"][index] = np.void(cell_id_sha256(case, int(morton)))
        leaves["morton_u64"][index] = morton
        leaves["membership_digest_128"][index] = np.void(bytes([index % 251]) * 16)
        leaves["unique_point_count_u32"][index] = 1
        leaves["raw_record_multiplicity_sum_u32"][index] = 1
    nodes = np.zeros(len(mortons), dtype=numpy_dtype("partition_node"))
    result = PartitionResult(
        nodes,
        leaves,
        mortons,
        xyz,
        np.arange(len(mortons), dtype=np.uint32),
        np.zeros(3, dtype=np.float64),
        32.0,
        1.0,
    )
    return case, result


def test_field_builder_emits_complete_baseline_and_isotropic_domains(tmp_path) -> None:
    case, partition = _plane_partition()
    worker = build_and_load(tmp_path)
    result = build_fields(case, partition, worker)
    o = len(partition.leaf_mortons)
    assert len(result.fields) == 6 * o
    assert len(result.rejections) == 6 * o
    assert result.baseline_valid_rows > 0
    baseline = result.fields[: 3 * o]
    isotropic = result.fields[3 * o :]
    assert np.all(baseline["arm_u8"] == 0)
    assert np.all(isotropic["arm_u8"] == 1)
    assert np.array_equal(baseline["support_count_u32"], isotropic["support_count_u32"])
    assert np.array_equal(baseline["trace_q_i64"], isotropic["trace_q_i64"])
    assert np.all(isotropic["anisotropy_q_u16"] == 0)
    assert np.all(isotropic["filtration_u16"] == 65_534)
    assert all(
        bytes(left) != bytes(right)
        for left, right in zip(baseline["row_id_digest_128"], isotropic["row_id_digest_128"], strict=True)
    )
    assert np.array_equal(result.rejections["row_id_digest_128"], result.fields["row_id_digest_128"])


def test_all_first_rejections_have_exact_stage_masks_and_sentinels(tmp_path) -> None:
    case, partition = _plane_partition()
    worker = build_and_load(tmp_path)
    occupied = np.zeros(1 << 15, dtype=np.uint8)
    raw_counts = np.zeros(1 << 15, dtype="<u4")
    morton_index = partition.leaf_mortons.astype(np.intp)
    occupied[morton_index] = 1
    raw_counts[morton_index] = 1
    base = worker.staged_field_evidence(occupied, raw_counts, partition.leaf_xyz)

    arrays = {
        name: np.array(getattr(base, name), copy=True)
        for name in StagedFieldEvidence.__dataclass_fields__
    }
    row_count = 3 * len(partition.leaf_mortons)
    arrays["first_rejection"].fill(0)
    arrays["stage_mask"].fill(0x7F)
    arrays["growth12"].fill(0)
    arrays["growth23"].fill(0)
    arrays["raw_p10"].fill(1)
    arrays["raw_p90"].fill(1)
    arrays["balance_numerator"].fill(11)
    arrays["balance_denominator"].fill(20)
    arrays["trace_q"].fill(3)
    arrays["lambda1_q"].fill(1)
    arrays["lambda2_q"].fill(1)
    arrays["lambda3_q"].fill(1)
    arrays["anisotropy_q"].fill(1)
    arrays["filtration"].fill(65_533)
    assert row_count > 7

    masks = (0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F)
    for index, rejection in enumerate(range(1, 8)):
        arrays["first_rejection"][index] = rejection
        arrays["stage_mask"][index] = masks[index]
        arrays["anisotropy_q"][index] = 0
        arrays["filtration"][index] = 65_534
        arrays["lambda1_q"][index] = 0
        arrays["lambda2_q"][index] = 0
        arrays["lambda3_q"][index] = 0
        if rejection <= 2:
            arrays["balance_numerator"][index] = 0
            arrays["balance_denominator"][index] = 0
        if rejection <= 3:
            arrays["growth12"][index] = -2_147_483_648
            arrays["growth23"][index] = -2_147_483_648
        if rejection <= 4:
            arrays["raw_p10"][index] = 0
            arrays["raw_p90"][index] = 0
        if rejection <= 6:
            arrays["trace_q"][index] = 0
    arrays["growth12"][3] = -1
    arrays["growth23"][3] = 0

    evidence = StagedFieldEvidence(**arrays)

    class FakeWorker:
        def staged_field_evidence(self, *_args: object) -> StagedFieldEvidence:
            return evidence

    result = build_fields(case, partition, FakeWorker())
    baseline = result.fields[:row_count]
    rejections = result.rejections[:row_count]
    denominator_ids = (1, 2, 3, 4, 6, 7, 8)
    for index, rejection in enumerate(range(1, 8)):
        row = baseline[index]
        assert int(row["status_u16"]) == 1
        assert int(row["first_rejection_u16"]) == rejection
        assert int(row["stage_mask_u32"]) == masks[index]
        assert int(row["anisotropy_q_u16"]) == 0
        assert int(row["filtration_u16"]) == 65_534
        assert int(row["lambda1_q_i64"]) == 0
        assert int(row["lambda2_q_i64"]) == 0
        assert int(row["lambda3_q_i64"]) == 0
        assert int(rejections[index]["first_rejection_u16"]) == rejection
        assert int(rejections[index]["stage_mask_u32"]) == masks[index]
        assert int(rejections[index]["denominator_id_u16"]) == denominator_ids[index]
    assert int(baseline[0]["growth12_gate_code_i32"]) == -2_147_483_648
    assert int(baseline[2]["growth23_gate_code_i32"]) == -2_147_483_648
    assert int(baseline[3]["growth12_gate_code_i32"]) == -1
    assert int(baseline[4]["raw_count_p10_u32"]) == 1
    assert int(baseline[5]["trace_q_i64"]) == 0
    assert int(baseline[6]["trace_q_i64"]) == 3
