from __future__ import annotations

import numpy as np

from scripts.extract_step_topology_bend_axes import _cylinder_axis_endpoints


def test_negative_native_axis_evaluates_signed_v_before_canonical_ordering():
    endpoints = _cylinder_axis_endpoints(
        np.array([10.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        2.0,
        5.0,
    )
    assert np.allclose(endpoints[0], [5.0, 0.0, 0.0])
    assert np.allclose(endpoints[1], [8.0, 0.0, 0.0])


def test_positive_native_axis_preserves_expected_segment():
    endpoints = _cylinder_axis_endpoints(
        np.array([10.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        -5.0,
        -2.0,
    )
    assert np.allclose(endpoints[0], [5.0, 0.0, 0.0])
    assert np.allclose(endpoints[1], [8.0, 0.0, 0.0])
