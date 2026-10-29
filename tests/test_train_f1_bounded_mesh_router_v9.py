from __future__ import annotations

import numpy as np

from scripts.train_f1_bounded_mesh_router_v9 import (
    family_class_balanced_weights,
)


def test_family_class_balanced_weights_equalize_families_and_classes() -> None:
    groups = np.asarray(["large"] * 6 + ["small"] * 3)
    target = np.asarray([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)

    weights = family_class_balanced_weights(groups, target)

    assert np.isclose(weights[groups == "large"].sum(), 0.5)
    assert np.isclose(weights[groups == "small"].sum(), 0.5)
    for family in ("large", "small"):
        family_mask = groups == family
        assert np.isclose(weights[family_mask & (target == 0)].sum(), 0.25)
        assert np.isclose(weights[family_mask & (target == 1)].sum(), 0.25)


def test_family_class_balanced_weights_support_single_class_family() -> None:
    groups = np.asarray(["one_class", "one_class", "mixed", "mixed"])
    target = np.asarray([0, 0, 0, 1], dtype=np.uint8)

    weights = family_class_balanced_weights(groups, target)

    assert np.isclose(weights[groups == "one_class"].sum(), 0.5)
    assert np.isclose(weights[groups == "mixed"].sum(), 0.5)
