import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_f1_feature_region_labels.py"
    spec = importlib.util.spec_from_file_location("summarize_f1_feature_region_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summarizer = _load_module()


def _label(bend_id, human_feature_label):
    return {
        "bend_id": bend_id,
        "human_feature_label": human_feature_label,
        "suggested_role": "candidate",
        "diagnostic_tags": [],
    }


def test_summary_resolves_expected_counts_when_labels_match():
    payload = {
        "part_key": "10839000",
        "scan_name": "full.ply",
        "truth_conventional_bends": 2,
        "truth_raised_form_features": 1,
        "truth_total_features": 3,
        "region_labels": [
            _label("OB1", "conventional_bend"),
            _label("OB2", "conventional_bend"),
            _label("OB3", "raised_form_feature"),
            _label("OB4", "fragment_or_duplicate"),
        ],
    }

    summary = summarizer.summarize_feature_region_labels(payload)

    assert summary["computed_counts"]["conventional_bend_count"] == 2
    assert summary["computed_counts"]["raised_form_feature_count"] == 1
    assert summary["computed_counts"]["fragment_or_duplicate_count"] == 1
    assert summary["blocker_status"] == "resolved_by_region_labels"
    assert summary["label_capacity"]["owned_region_capacity_deficit"] == 0


def test_summary_reports_pending_and_invalid_labels():
    payload = {
        "truth_conventional_bends": 1,
        "region_labels": [
            _label("OB1", None),
            _label("OB2", "bad_label"),
        ],
    }

    summary = summarizer.summarize_feature_region_labels(payload)

    assert summary["pending_region_ids"] == ["OB1"]
    assert summary["invalid_labels"] == [{"bend_id": "OB2", "label": "bad_label"}]
    assert summary["blocker_status"] == "invalid_labels"


def test_summary_complete_but_not_resolved():
    payload = {
        "truth_conventional_bends": 2,
        "truth_raised_form_features": 0,
        "region_labels": [
            _label("OB1", "conventional_bend"),
            _label("OB2", "fragment_or_duplicate"),
        ],
    }

    summary = summarizer.summarize_feature_region_labels(payload)

    assert summary["blocker_status"] == "labels_do_not_resolve_expected_counts"
    assert summary["resolves_expected_counts"]["conventional_bends"] is False
    assert summary["label_capacity"]["valid_conventional_region_deficit"] == 1
    assert summary["label_capacity"]["requires_missing_conventional_region_recovery"] is True


def test_summary_reports_owned_region_capacity_deficit_before_labels():
    payload = {
        "truth_conventional_bends": 8,
        "truth_raised_form_features": 2,
        "truth_total_features": 10,
        "region_labels": [
            _label(f"OB{index}", None)
            for index in range(1, 10)
        ],
    }

    summary = summarizer.summarize_feature_region_labels(payload)

    assert summary["blocker_status"] == "owned_region_capacity_deficit"
    assert summary["label_capacity"]["owned_region_count"] == 9
    assert summary["label_capacity"]["expected_counted_conventional_plus_raised"] == 10
    assert summary["label_capacity"]["owned_region_capacity_deficit"] == 1
    assert summary["label_capacity"]["valid_counted_feature_deficit"] == 10
    assert summary["label_capacity"]["requires_split_or_missing_region_recovery"] is True
