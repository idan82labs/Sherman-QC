import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_f1_feature_semantics.py"
    spec = importlib.util.spec_from_file_location("audit_f1_feature_semantics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def test_audit_flags_total_feature_match_but_conventional_miss():
    report = audit.audit_feature_semantics(
        {
            "records": [
                {
                    "track": "known",
                    "part_key": "10839000",
                    "scan_name": "full.ply",
                    "human_observed_bend_count": 8,
                    "human_full_part_bend_count": 10,
                    "feature_breakdown": {
                        "conventional_bends": 8,
                        "raised_form_features": 2,
                        "legacy_expected_total_features": 10,
                    },
                    "algorithm_range": [9, 10],
                    "algorithm_exact": None,
                    "algorithm_raw_exact_field": 10,
                    "feature_count_semantics": {"algorithm_feature_split_status": "not_modeled"},
                }
            ]
        }
    )

    assert report["blockers"][0]["feature_semantics_status"] == "total_feature_match_but_conventional_miss"
    assert report["blockers"][0]["range_contains_total_features"] is True
    assert report["blockers"][0]["range_contains_conventional"] is False
    assert report["feature_split_not_modeled_count"] == 1


def test_audit_marks_clean_contained_case_promotion_safe():
    report = audit.audit_feature_semantics(
        {
            "records": [
                {
                    "track": "known",
                    "part_key": "clean",
                    "scan_name": "clean.ply",
                    "human_observed_bend_count": 4,
                    "human_full_part_bend_count": 4,
                    "algorithm_range": [4, 4],
                    "algorithm_exact": 4,
                }
            ]
        }
    )

    case = report["cases"][0]
    assert case["feature_semantics_status"] == "clean_or_contained"
    assert case["promotion_safe_under_conventional_semantics"] is True
    assert report["promotion_safe_case_count"] == 1


def test_audit_keeps_rolled_special_out_of_promotion_safe_bucket():
    report = audit.audit_feature_semantics(
        {
            "records": [
                {
                    "track": "known",
                    "part_key": "rolled",
                    "scan_name": "rolled.ply",
                    "human_observed_bend_count": 2,
                    "human_full_part_bend_count": 2,
                    "part_family": "rolled_or_rounded_mixed_transition",
                    "algorithm_range": [2, 3],
                }
            ]
        }
    )

    case = report["cases"][0]
    assert case["feature_semantics_status"] == "special_feature_range_contains_conventional_truth"
    assert case["promotion_safe_under_conventional_semantics"] is False
