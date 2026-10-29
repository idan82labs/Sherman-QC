from __future__ import annotations

import numpy as np

from research.f1_cfac_v1.config import CONFIG, CONFIG_SHA256, validate_config
from research.f1_cfac_v1.core import canonicalize_xyz


def test_frozen_config_and_clarified_gates() -> None:
    validate_config()
    assert len(CONFIG_SHA256) == 64
    assert CONFIG.radius_minimum_s == 2.5
    assert CONFIG.radius_maximum_s == 64.0
    assert CONFIG.anisotropy_maximum == 0.18


def test_canonical_source_is_permutation_invariant() -> None:
    points = np.asarray([(x, y, (x + 2 * y) % 5) for x in range(9) for y in range(9)], dtype=float)
    assert canonicalize_xyz(points).canonical_sha256 == canonicalize_xyz(points[::-1]).canonical_sha256
