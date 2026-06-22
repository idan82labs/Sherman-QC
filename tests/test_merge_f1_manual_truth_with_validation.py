import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_f1_manual_truth_with_validation.py"
    spec = importlib.util.spec_from_file_location("merge_f1_manual_truth_with_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


merge = _load_module()


def test_merge_updates_algorithm_fields_and_contact_sheet(tmp_path):
    manual_path = tmp_path / "manual.json"
    summary_path = tmp_path / "summary.json"
    manual_path.write_text(
        json.dumps(
            {
                "label_semantics": "visible bends",
                "records": [
                    {
                        "track": "partial",
                        "part_key": "demo",
                        "scan_name": "demo.ply",
                        "human_observed_bend_count": 5,
                        "algorithm_range": [1, 9],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "track": "partial",
                        "part_key": "demo",
                        "scan_path": "/tmp/demo.ply",
                        "algorithm": {
                            "accepted_bend_count_range": [4, 6],
                            "observed_bend_count_prediction": None,
                            "accepted_exact_bend_count": None,
                            "candidate_source": "partial_observed_control_f1_envelope",
                        },
                        "render_artifacts": {
                            "contact_sheet_1080p_path": "/tmp/contact.png",
                        },
                        "feature_count_semantics": {
                            "counted_target": "conventional_bends",
                            "algorithm_feature_split_status": "not_modeled",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = merge.merge_manual_truth_with_validation(
        manual_truth_json=manual_path,
        validation_summary_jsons=[summary_path],
    )

    record = payload["records"][0]
    assert record["algorithm_range"] == [4, 6]
    assert record["algorithm_candidate_source"] == "partial_observed_control_f1_envelope"
    assert record["feature_count_semantics"]["counted_target"] == "conventional_bends"
    assert record["contact_sheet"] == "/tmp/contact.png"
    assert payload["merge_diagnostics"]["matched_record_count"] == 1


def test_merge_keeps_non_singleton_exact_field_out_of_scored_exact(tmp_path):
    manual_path = tmp_path / "manual.json"
    summary_path = tmp_path / "summary.json"
    manual_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "track": "known",
                        "part_key": "raised",
                        "scan_name": "raised.ply",
                        "human_observed_bend_count": 8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "track": "known",
                        "part_key": "raised",
                        "scan_path": "/tmp/raised.ply",
                        "algorithm": {
                            "accepted_bend_count_range": [9, 10],
                            "accepted_exact_bend_count": None,
                            "raw_exact_bend_count_field": 10,
                        },
                        "render_artifacts": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = merge.merge_manual_truth_with_validation(
        manual_truth_json=manual_path,
        validation_summary_jsons=[summary_path],
    )

    record = payload["records"][0]
    assert record["algorithm_exact"] is None
    assert record["algorithm_raw_exact_field"] == 10


def test_merge_reports_unmatched_rows(tmp_path):
    manual_path = tmp_path / "manual.json"
    summary_path = tmp_path / "summary.json"
    manual_path.write_text(
        json.dumps({"records": [{"track": "known", "part_key": "a", "scan_name": "a.ply"}]}),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"results": [{"track": "known", "part_key": "b", "scan_path": "/tmp/b.ply"}]}),
        encoding="utf-8",
    )

    payload = merge.merge_manual_truth_with_validation(
        manual_truth_json=manual_path,
        validation_summary_jsons=[summary_path],
    )

    assert payload["merge_diagnostics"]["unmatched_manual_count"] == 1
    assert payload["merge_diagnostics"]["unmatched_validation_count"] == 1
