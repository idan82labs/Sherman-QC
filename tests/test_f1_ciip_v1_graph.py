from __future__ import annotations

import numpy as np

from research.f1_ciip_v1.container import numpy_dtype
from research.f1_ciip_v1.graph import build_space_scale_adjacency, h0_persistence
from research.f1_ciip_v1.identifiers import (
    case_token_digest,
    edge_id_digest_128,
    field_row_id_digest_128,
)
from research.f1_ciip_v1.native_worker import build_and_load


CASE = case_token_digest(bytes.fromhex("05" * 32))


def _fields(mortons: list[int], arm: int, q: int) -> np.ndarray:
    o = len(mortons)
    rows = np.zeros(3 * o, dtype=numpy_dtype("cell_scale_field"))
    cursor = 0
    for scale in range(3):
        for morton in mortons:
            rows["row_id_digest_128"][cursor] = np.void(
                field_row_id_digest_128(CASE, arm, scale, morton)
            )
            rows["morton_u64"][cursor] = morton
            rows["arm_u8"][cursor] = arm
            rows["scale_u8"][cursor] = scale
            rows["status_u16"][cursor] = 0
            rows["anisotropy_q_u16"][cursor] = q
            rows["filtration_u16"][cursor] = 65_534 - q
            rows["eigenvalue_denominator_u32"][cursor] = 1 if arm == 0 else 3
            cursor += 1
    return rows


def test_adjacency_exact_actual_o_domain_and_order() -> None:
    rows = build_space_scale_adjacency(CASE, np.array([0, 1], dtype=np.uint64))
    assert len(rows) == 7  # three face edges plus four scale edges
    keys = [
        (int(row["edge_kind_u8"]), int(row["axis_u8"]), int(row["lower_vertex_u32"]), int(row["upper_vertex_u32"]))
        for row in rows
    ]
    assert keys == sorted(keys)
    assert len({bytes(row["edge_id_digest_128"]) for row in rows}) == len(rows)


def test_h0_emits_one_row_per_vertex_and_isotropic_has_no_survivor() -> None:
    mortons = [0, 1]
    adjacency = build_space_scale_adjacency(CASE, np.asarray(mortons, dtype=np.uint64))
    baseline = _fields(mortons, arm=0, q=10)
    baseline_rows, attributed = h0_persistence(CASE, 0, baseline, adjacency)
    assert len(baseline_rows) == 6
    assert attributed == 1
    assert np.count_nonzero(baseline_rows["status_u8"] == 1) == 1
    assert np.count_nonzero(baseline_rows["reason_u16"] == 0) == 1

    isotropic = _fields(mortons, arm=1, q=0)
    isotropic_rows, isotropic_attributed = h0_persistence(CASE, 1, isotropic, adjacency)
    assert isotropic_attributed == 0
    assert np.all(isotropic_rows["reason_u16"] == 1)


def test_h0_is_byte_deterministic() -> None:
    mortons = [0, 1, 2]
    adjacency = build_space_scale_adjacency(CASE, np.asarray(mortons, dtype=np.uint64))
    fields = _fields(mortons, arm=0, q=100)
    first, first_count = h0_persistence(CASE, 0, fields, adjacency)
    second, second_count = h0_persistence(CASE, 0, fields, adjacency)
    assert first_count == second_count
    assert first.tobytes() == second.tobytes()


def _spatial_edges(pairs: list[tuple[int, int]]) -> np.ndarray:
    rows = np.zeros(len(pairs), dtype=numpy_dtype("space_scale_adjacency"))
    for index, (lower, upper) in enumerate(pairs):
        rows["edge_id_digest_128"][index] = np.void(
            edge_id_digest_128(CASE, lower, upper, 0, 0)
        )
        rows["lower_vertex_u32"][index] = lower
        rows["upper_vertex_u32"][index] = upper
        rows["edge_kind_u8"][index] = 0
        rows["axis_u8"][index] = 0
    return rows


def test_h0_same_filtration_uses_vertex_first_then_numeric_elder() -> None:
    fields = _fields([0], arm=0, q=65_514)  # f=20 for all three vertices
    adjacency = _spatial_edges([(0, 1)])
    rows, _ = h0_persistence(CASE, 0, fields, adjacency)
    assert int(rows[1]["status_u8"]) == 0
    assert int(rows[1]["birth_f_u16"]) == 20
    assert int(rows[1]["death_f_or_essential_u16"]) == 20
    assert int(rows[1]["lifetime_u16"]) == 0
    assert bytes(rows[1]["death_row_id_digest_128"]) == bytes(
        adjacency[0]["edge_id_digest_128"]
    )
    assert int(rows[0]["status_u8"]) == 1


def test_h0_owner_is_separate_from_structural_root_and_predeath_excludes_edge() -> None:
    fields = _fields([0, 1], arm=0, q=65_514)
    # Vertex 4 is born earlier than the rank-one component rooted at vertex 0.
    # The final 3-4 merge must preserve owner 4 while retaining structural root
    # 0. Edge 1-2 is an intra-component cycle and must increment exactly once.
    fields["anisotropy_q_u16"][4] = 65_524
    fields["filtration_u16"][4] = 10
    adjacency = _spatial_edges([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    rows, attributed = h0_persistence(CASE, 0, fields, adjacency)

    # The younger component owner 0 dies at 3-4. Its pre-death summary includes
    # the three cross-component spatial edges plus the 1-2 cycle, but excludes
    # the death-causing fifth edge.
    assert int(rows[0]["status_u8"]) == 0
    assert bytes(rows[0]["death_row_id_digest_128"]) == bytes(
        adjacency[4]["edge_id_digest_128"]
    )
    assert int(rows[0]["spatial_face_edge_count_u32"]) == 4
    assert int(rows[0]["scale_mask_u8"]) == 0b011

    # Structural root 0 now carries persistence owner 4; the essential row must
    # still be emitted at birth vertex 4 with the post-merge summary.
    assert int(rows[4]["status_u8"]) == 1
    assert int(rows[4]["spatial_face_edge_count_u32"]) == 5
    assert int(rows[4]["scale_mask_u8"]) == 0b111
    assert int(rows[4]["reason_u16"]) == 0
    assert attributed >= 1


def test_native_adjacency_and_h0_are_byte_exact_with_python_oracle(tmp_path) -> None:
    worker = build_and_load(tmp_path)
    mortons = np.asarray([0, 1, 2, 3], dtype=np.uint64)
    python_adjacency = build_space_scale_adjacency(CASE, mortons)
    native_adjacency = build_space_scale_adjacency(
        CASE, mortons, native_worker=worker
    )
    assert native_adjacency.dtype == python_adjacency.dtype
    assert native_adjacency.tobytes() == python_adjacency.tobytes()

    fields = _fields(mortons.tolist(), arm=0, q=100)
    # Exercise multiple filtration buckets plus same-filtration elder ties.
    fields["anisotropy_q_u16"][[0, 4, 8]] = (300, 300, 200)
    fields["filtration_u16"] = 65_534 - fields["anisotropy_q_u16"]
    python_rows, python_attributed = h0_persistence(
        CASE, 0, fields, python_adjacency
    )
    native_rows, native_attributed = h0_persistence(
        CASE, 0, fields, native_adjacency, native_worker=worker
    )
    assert native_attributed == python_attributed
    assert native_rows.dtype == python_rows.dtype
    assert native_rows.tobytes() == python_rows.tobytes()

    isotropic = _fields(mortons.tolist(), arm=1, q=0)
    python_iso, python_iso_count = h0_persistence(
        CASE, 1, isotropic, python_adjacency
    )
    native_iso, native_iso_count = h0_persistence(
        CASE, 1, isotropic, native_adjacency, native_worker=worker
    )
    assert native_iso_count == python_iso_count == 0
    assert native_iso.tobytes() == python_iso.tobytes()
    worker.validate_persistence_collision_domain(
        CASE,
        np.concatenate((fields, isotropic)),
        np.concatenate((native_rows, native_iso)),
    )
