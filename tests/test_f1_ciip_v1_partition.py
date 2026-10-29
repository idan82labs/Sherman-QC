from __future__ import annotations

import numpy as np
import pytest

import research.f1_ciip_v1.partition as partition_module
from research.f1_ciip_v1.identifiers import case_token_digest
from research.f1_ciip_v1.partition import PartitionValidationError, build_partition, canonical_sample


CASE = case_token_digest(bytes(32))


def test_sampling_is_input_order_invariant() -> None:
    xyz = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype="<f8")
    multiplicity = np.asarray([1, 2, 3], dtype=np.uint32)
    first = canonical_sample(xyz, multiplicity)
    permutation = np.asarray([2, 1, 0])
    second = canonical_sample(xyz[permutation], multiplicity[permutation])
    assert first.xyz.tobytes() == second.xyz.tobytes()
    assert first.point_ids.tobytes() == second.point_ids.tobytes()
    assert first.multiplicities.tobytes() == second.multiplicities.tobytes()


def test_partition_assigns_every_sample_once_and_serializes_sparse_tree() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]],
        dtype="<f8",
    )
    sampled = canonical_sample(xyz, np.ones(4, dtype=np.uint32))
    result = build_partition(CASE, sampled)
    o = len(result.leaf_aggregate)
    assert 1 <= o <= 4
    assert o <= len(result.partition_nodes) <= 2 * o - 1
    assert int(np.sum(result.leaf_aggregate["unique_point_count_u32"])) == 4
    assert np.all(result.leaf_mortons[1:] > result.leaf_mortons[:-1])
    keys = [
        (int(row["depth_u8"]), int(row["morton_prefix_u64"]))
        for row in result.partition_nodes
    ]
    assert keys == sorted(keys)
    assert result.root_side > 0.0 and result.cell_side == result.root_side / 32.0


def test_negative_zero_is_canonicalized() -> None:
    xyz = np.asarray([[-0.0, 0.0, 0.0], [1.0, -0.0, 0.0]], dtype="<f8")
    sampled = canonical_sample(xyz, np.ones(2, dtype=np.uint32))
    assert not np.signbit(sampled.xyz[sampled.xyz == 0.0]).any()


@pytest.mark.parametrize("multiplicity", [np.asarray([-1]), np.asarray([0]), np.asarray([1.5])])
def test_sampling_rejects_multiplicity_before_u32_cast(multiplicity: np.ndarray) -> None:
    with pytest.raises(PartitionValidationError, match="multiplicities"):
        canonical_sample(np.asarray([[0.0, 0.0, 0.0]], dtype="<f8"), multiplicity)


@pytest.mark.parametrize("sample_cap", [True, 0, 200_001, 1.0])
def test_sampling_rejects_noncanonical_cap(sample_cap: object) -> None:
    with pytest.raises(PartitionValidationError, match="sample cap"):
        canonical_sample(
            np.asarray([[0.0, 0.0, 0.0]], dtype="<f8"),
            np.ones(1, dtype=np.uint32),
            sample_cap=sample_cap,  # type: ignore[arg-type]
        )


def test_sampling_stops_on_unequal_point_id_preimages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(partition_module, "point_id_sha256", lambda _packed: bytes(32))
    monkeypatch.setattr(partition_module, "sample_rank_sha256", lambda _point_id: bytes(32))
    with pytest.raises(PartitionValidationError, match="unequal XYZ"):
        canonical_sample(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8"),
            np.ones(2, dtype=np.uint32),
        )


def test_sampling_stops_on_unequal_rank_preimages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(partition_module, "sample_rank_sha256", lambda _point_id: bytes(32))
    with pytest.raises(PartitionValidationError, match="unequal point IDs"):
        canonical_sample(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8"),
            np.ones(2, dtype=np.uint32),
        )


def test_sampled_point_leaf_ordinals_align_with_sampled_order() -> None:
    xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype="<f8")
    sampled = canonical_sample(xyz, np.ones(4, dtype=np.uint32))
    result = build_partition(CASE, sampled)
    assigned = np.floor((sampled.xyz - result.root_origin) / result.cell_side).astype(np.int64)
    mortons = partition_module._morton_array(assigned)
    expected = np.searchsorted(result.leaf_mortons, mortons).astype(np.uint32)
    assert np.array_equal(result.sampled_point_leaf_ordinal, expected)
