import numpy as np

import backend.f1_calibrated_multiview_turn_consensus_v58 as v58
from backend.f1_calibrated_per_view_segment_consensus_v60 import _per_view_segments


def test_per_view_segment_fits_collinear_tokens() -> None:
    tokens = [
        v58.Token(np.array([float(i), 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 20.0, 30.0, 1.0, 1.0, 0, 10, i, 0)
        for i in range(6)
    ]
    segments = _per_view_segments(tokens, 20.0)
    assert len(segments) == 1
    assert np.linalg.norm(segments[0]["end"] - segments[0]["start"]) == 5.0
