import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_f1_manual_validation_labels.py"
    spec = importlib.util.spec_from_file_location("import_f1_manual_validation_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


importer = _load_module()


def test_importer_writes_conventional_count_and_raised_feature_metadata():
    payload = importer.build_manual_validation_payload(
        labels_payload={
            "generated_at": "2026-04-25T00:00:00",
            "labels": [
                {
                    "part_key": "10839000",
                    "scan_name": "scan_1.ply",
                    "track": "partial",
                    "human_observed_bend_count": 4,
                    "human_full_part_bend_count": 10,
                    "algorithm_range": [2, 2],
                    "feature_breakdown": {
                        "conventional_bends": 4,
                        "raised_form_features": 2,
                        "counted_target": "conventional_bends",
                    },
                    "notes": "4 conventional bends plus raises",
                }
            ],
        },
        scan_path_index={("10839000", "scan_1.ply"): "/tmp/scan_1.ply"},
    )

    entry = payload["slices"]["f1_manual_partial_observed_bends"]["entries"][0]
    assert entry["scan_path"] == "/tmp/scan_1.ply"
    assert entry["expectation"]["expected_bend_count"] == 4
    assert entry["expectation"]["expected_conventional_bend_count"] == 4
    assert entry["expectation"]["expected_raised_form_feature_count"] == 2
    assert entry["expectation"]["expected_total_feature_count"] == 6
    assert entry["expectation"]["known_whole_part_total_context"] == 10
    assert "feature_breakdown" in entry["audit_tags"]


def test_importer_separates_clean_known_from_special_and_excluded():
    payload = importer.build_manual_validation_payload(
        labels_payload={
            "labels": [
                {
                    "part_key": "clean",
                    "scan_name": "clean.ply",
                    "track": "known",
                    "human_observed_bend_count": 4,
                    "human_full_part_bend_count": 4,
                    "algorithm_range": [4, 6],
                },
                {
                    "part_key": "rolled",
                    "scan_name": "rolled.ply",
                    "track": "known",
                    "human_observed_bend_count": 2,
                    "human_full_part_bend_count": 2,
                    "part_family": "rolled_or_rounded_mixed_transition",
                    "legacy_expected_disputed": True,
                    "algorithm_range": [2, 3],
                },
                {
                    "part_key": "bad",
                    "scan_name": "bad.ply",
                    "track": "known",
                    "exclude_from_accuracy": True,
                    "exclusion_reason": "bad_scan_quality",
                    "algorithm_range": [9, 12],
                },
            ]
        },
        scan_path_index={},
    )

    assert [entry["part_key"] for entry in payload["slices"]["f1_manual_clean_known_count_plys"]["entries"]] == ["clean"]
    special_keys = {entry["part_key"] for entry in payload["slices"]["f1_manual_special_cases"]["entries"]}
    assert special_keys == {"rolled", "bad"}
    bad_entry = next(entry for entry in payload["slices"]["f1_manual_special_cases"]["entries"] if entry["part_key"] == "bad")
    assert "expected_bend_count" not in bad_entry["expectation"]
    assert bad_entry["expectation"]["exclude_from_accuracy"] is True
    assert all(entry["part_key"] != "bad" for entry in payload["slices"]["f1_manual_clean_known_count_plys"]["entries"])
