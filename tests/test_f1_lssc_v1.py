from __future__ import annotations

import numpy as np

from research.f1_lssc_v1.config import CONFIG, CONFIG_SHA256, validate_config
from research.f1_lssc_v1.core import canonicalize_xyz
from research.f1_lssc_v1.fixtures import sf01_double_skin


def test_frozen_config() -> None:
    validate_config()
    assert len(CONFIG_SHA256) == 64
    assert CONFIG.patch_radii_over_s == (6.0, 10.0, 16.0, 24.0)
    assert CONFIG.cylindrical_lambda2_max == 0.004
    assert CONFIG.target_capability_granted is False


def test_canonical_source_is_permutation_invariant() -> None:
    points = sf01_double_skin()[:1500]
    forward = canonicalize_xyz(points)
    reverse = canonicalize_xyz(points[::-1])
    assert forward.canonical_sha256 == reverse.canonical_sha256
    assert np.array_equal(forward.xyz, reverse.xyz)
