from __future__ import annotations

import numpy as np

from research.f1_lepg_v1.config import CONFIG, CONFIG_SHA256, validate_config
from research.f1_lepg_v1.core import canonicalize_xyz
from research.f1_lepg_v1.fixtures import sf01


def test_frozen_config() -> None:
    validate_config()
    assert len(CONFIG_SHA256) == 64
    assert CONFIG.signature_ranks == (1, 2, 4, 8, 16, 32)
    assert CONFIG.gpu_enabled is False
    assert CONFIG.target_capability_granted is False


def test_canonicalization_is_permutation_invariant() -> None:
    points = sf01()[:2500]
    forward = canonicalize_xyz(points)
    reverse = canonicalize_xyz(points[::-1])
    assert np.array_equal(forward.xyz, reverse.xyz)
    assert forward.canonical_sha256 == reverse.canonical_sha256
