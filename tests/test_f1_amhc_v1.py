from __future__ import annotations

import numpy as np

from research.f1_amhc_v1.config import CONFIG, CONFIG_SHA256, validate_config
from research.f1_amhc_v1.core import canonicalize_xyz, sha_arrays


def test_config_is_frozen_and_source_only() -> None:
    validate_config()
    assert len(CONFIG_SHA256) == 64
    assert CONFIG.authoritative_png_count == 30
    assert CONFIG.target_capability_granted is False
    assert CONFIG.source_point_serialization is False


def test_canonicalization_is_permutation_invariant() -> None:
    points = np.asarray(
        [(float(x), float(y), float((x + y) % 3)) for x in range(8) for y in range(8)]
    )
    first = canonicalize_xyz(points)
    second = canonicalize_xyz(points[::-1])
    assert np.array_equal(first.xyz, second.xyz)
    assert first.canonical_table_sha256 == second.canonical_table_sha256


def test_hash_domain_separation() -> None:
    values = np.arange(12, dtype=np.uint32)
    assert sha_arrays(b"A", (values,)) != sha_arrays(b"B", (values,))
