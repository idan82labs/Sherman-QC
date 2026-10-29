from __future__ import annotations

import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

from scripts.audit_f1_aug30_retrospective_pseudo_unseen_split_v1 import (
    DESCRIPTORS_NAME,
    HASHES_NAME,
    PREREG_NAME,
    SEAL_NAME,
    SPLIT_NAME,
    TRAIN_ALLOWLIST_NAME,
    audit_artifacts,
    select_tiers,
)


def test_geometry_only_selector_is_deterministic_and_retains_neighbors() -> None:
    positions = [index * 0.01 for index in range(18)]
    positions += [2.0 + index * 0.25 for index in range(7)]
    positions += [10.0, 20.0, 35.0]
    rows = [
        {"case_id": f"T-{index:03d}", "normalized_vector": [position, position * 0.1]}
        for index, position in enumerate(positions, start=1)
    ]
    first = select_tiers(rows)
    second = select_tiers(list(reversed(rows)))
    assert first == second
    assert len(first["train"]) == 18
    assert {tier: len(case_ids) for tier, case_ids in first["tiers"].items()} == {
        "SIMILAR_FAMILY": 4,
        "ADJACENT_FAMILY": 3,
        "DISTANT_FAMILY": 3,
    }
    train = set(first["train"])
    for case_id in first["tiers"]["SIMILAR_FAMILY"]:
        assert len(first["retained_neighbor_anchors"][case_id]) == 2
        assert set(first["retained_neighbor_anchors"][case_id]).issubset(train)
    for case_id in first["tiers"]["ADJACENT_FAMILY"]:
        assert len(first["retained_neighbor_anchors"][case_id]) == 1
        assert set(first["retained_neighbor_anchors"][case_id]).issubset(train)


def test_selector_rejects_non_28_case_input() -> None:
    with pytest.raises(ValueError, match="28 unique opaque cases"):
        select_tiers([{"case_id": "ONLY", "normalized_vector": [0.0]}])


def test_sealed_artifacts_pass_without_source_tree(tmp_path: Path) -> None:
    artifact_dir = ROOT / "research"
    for name in (
        PREREG_NAME,
        DESCRIPTORS_NAME,
        SPLIT_NAME,
        TRAIN_ALLOWLIST_NAME,
        HASHES_NAME,
        SEAL_NAME,
    ):
        shutil.copyfile(artifact_dir / name, tmp_path / name)
    result = audit_artifacts(tmp_path, verify_code_bindings=False)
    assert result["status"] == "PASS"
    assert result["source_free"] is True
    assert result["train_count"] == 18
    assert result["holdout_count"] == 10
