"""Pure display preflight checks; no native shapes or physical input reads."""
import numpy as np
import pytest
from backend.vendor_native_folded_mesh_v1 import validate_control_net
from backend.vendor_unfold_preview import UnfoldUnavailable


def test_nonextrusion_control_net_is_allowed_for_display_only():
    p = np.array([[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 2]]], dtype=float)
    # Twisted/varying generators: this makes no developability assertion.
    assert validate_control_net(p, np.ones((2, 2))) == 4


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_weights_refused(bad):
    weights = np.ones((2, 2)); weights[0, 0] = bad
    with pytest.raises(UnfoldUnavailable):
        validate_control_net(np.zeros((2, 2, 3)), weights)


def test_coordinate_and_shape_limits():
    with pytest.raises(UnfoldUnavailable):
        validate_control_net(np.zeros((2, 2, 2)), np.ones((2, 2)))
    p = np.zeros((2, 2, 3)); p[0, 0, 0] = float("nan")
    with pytest.raises(UnfoldUnavailable):
        validate_control_net(p, np.ones((2, 2)))
