from __future__ import annotations

from scripts.run_f1_manual_gold_calibration import FEATURE_NAMES, _duplicate, _features


def test_feature_schema_contains_no_source_points_or_part_identity():
    assert "source_points" not in " ".join(FEATURE_NAMES)
    assert "part" not in " ".join(FEATURE_NAMES)
    candidate = {
        "span_endpoints": [[0, 0, 0], [20, 0, 0]],
        "calibration_source": "fragment_merge",
        "support_count": 2,
    }
    assert len(_features(candidate)) == len(FEATURE_NAMES)


def test_duplicate_gate_merges_tilted_estimates_but_not_neighboring_lines():
    first = {"span_endpoints": [[0, 0, 0], [20, 0, 0]]}
    tilted_same = {"span_endpoints": [[0, 2, 0], [19.7, 5.0, 0]]}
    neighboring = {"span_endpoints": [[0, 11, 0], [20, 11, 0]]}
    assert _duplicate(first, tilted_same)
    assert not _duplicate(first, neighboring)
