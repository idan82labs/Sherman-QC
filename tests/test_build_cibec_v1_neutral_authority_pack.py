from __future__ import annotations

import numpy as np

from scripts.build_cibec_v1_neutral_authority_pack import _fixtures


def test_neutral_fixture_geometry_is_exact_complete_and_finite() -> None:
    fixtures = _fixtures()
    assert list(fixtures) == [f"S{index:02d}" for index in range(1, 15)]
    for points in fixtures.values():
        assert points.ndim == 2 and points.shape[1] == 3
        assert len(points) >= 1000
        assert np.all(np.isfinite(points))


def test_partial_fixture_has_bounded_contiguous_x_support() -> None:
    fixtures = _fixtures()
    assert np.ptp(fixtures["S13"][:, 0]) < np.ptp(fixtures["S01"][:, 0])


def test_no_fixture_is_empty_or_identical_except_control_pairs() -> None:
    fixtures = _fixtures()
    hashes = {key: hash(value.tobytes()) for key, value in fixtures.items()}
    assert hashes["S01"] == hashes["S14"]
    assert len(set(hashes.values())) >= 12
