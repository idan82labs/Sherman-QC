from __future__ import annotations

import numpy as np

from research.f1_ciip_v1.container import numpy_dtype
from research.f1_ciip_v1.target_attribution import (
    build_merge_witnesses,
    decision_from_metrics,
    locked_grid_transform,
)


def _id(value: int) -> np.void:
    return np.void(bytes([value]) * 16)


def test_locked_grid_transform_matches_depth_five_definition() -> None:
    origin, side = locked_grid_transform(np.asarray([[0.0, 0.0, 0.0], [24.0, 12.0, 6.0]]))
    assert np.allclose(origin, [-4.0, -10.0, -13.0])
    assert side == 1.0


def test_every_finite_merge_emits_one_witness_without_cutoff() -> None:
    fields = np.zeros(4, dtype=numpy_dtype("cell_scale_field"))
    fields["row_id_digest_128"] = [_id(1), _id(2), _id(3), _id(4)]
    fields["arm_u8"] = [0, 0, 1, 1]
    fields["scale_u8"] = 0
    fields["morton_u64"] = [0, 1, 0, 1]
    fields["filtration_u16"] = [1, 2, 1, 2]
    edges = np.zeros(1, dtype=numpy_dtype("space_scale_adjacency"))
    edges["edge_id_digest_128"] = _id(9)
    edges["lower_vertex_u32"] = 0
    edges["upper_vertex_u32"] = 1
    persistence = np.zeros(2, dtype=numpy_dtype("persistence_disposition"))
    persistence["interval_id_digest_128"] = [_id(7), _id(8)]
    persistence["birth_row_id_digest_128"] = [_id(2), _id(4)]
    persistence["death_row_id_digest_128"] = _id(9)
    persistence["arm_u8"] = [0, 1]
    persistence["status_u8"] = 0
    persistence["birth_f_u16"] = 2
    persistence["death_f_or_essential_u16"] = 2
    source = np.asarray([[0.0, 0.0, 0.0], [24.0, 12.0, 6.0]])
    witnesses, exceptions = build_merge_witnesses(
        field_rows=fields, adjacency_rows=edges, persistence_rows=persistence,
        source_xyz=source, source_sha256="a" * 64,
    )
    assert len(witnesses) == 2
    assert exceptions == []
    assert {row["birth_filtration_value"] for row in witnesses} == {2}
    assert all(row["persistence_value"] == 0 for row in witnesses)


def test_scientific_zero_is_terminal_architecture_stop() -> None:
    metrics = {
        "baseline_hash_reproduction_ok": True,
        "producer_validator_consensus_ok": True,
        "aggregate_added_ciip_rung_count": 1,
        "target_reached_by_ciip_rung_count": 1,
        "target_side_gate_advanced_count": 0,
        "target_novel_ciip_attributable_certificate_count": 1,
    }
    assert decision_from_metrics(metrics) == "CIIP_V1_NO_NOVEL_TARGET_ATTRIBUTION_ARCHITECTURE_STOP"
