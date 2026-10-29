from scripts.score_f1_real_scan_affine_resample_self_consistency_v1 import _as_matcher_truth


def test_as_matcher_truth_renames_development_line_ids() -> None:
    result = _as_matcher_truth([{
        "line_id": "PT01",
        "line_start": [0, 0, 0],
        "line_end": [1, 0, 0],
        "source_reference_id": "ignored",
    }])
    assert result == [{"bend_id": "PT01", "line_start": [0, 0, 0], "line_end": [1, 0, 0]}]
