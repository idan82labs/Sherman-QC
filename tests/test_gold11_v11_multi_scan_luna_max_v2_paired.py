from scripts.run_gold11_v11_multi_scan_luna_max_v2_paired import _case_f1


def test_case_f1_is_derived_from_matcher_counts():
    assert _case_f1({"true_positive": 3, "false_positive": 1, "false_negative": 2}) == 2 / 3


def test_zero_denominator_remains_undefined():
    assert _case_f1({"true_positive": 0, "false_positive": 0, "false_negative": 0}) is None
