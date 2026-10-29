import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_f1_split_candidate_labels.py"
    spec = importlib.util.spec_from_file_location("summarize_f1_split_candidate_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summarizer = _load_module()


def _candidate(bend_id, human_split_label):
    return {
        "bend_id": bend_id,
        "suggested_split_rank": "strong",
        "human_split_label": human_split_label,
        "notes": "",
    }


def test_summary_resolves_capacity_deficit_when_one_candidate_splits():
    payload = {
        "part_key": "10839000",
        "capacity_deficit": 1,
        "candidate_labels": [
            _candidate("OB2", "split_into_two_features"),
            _candidate("OB1", "keep_single"),
        ],
    }

    summary = summarizer.summarize_split_candidate_labels(payload)

    assert summary["blocker_status"] == "resolved_by_split_labels"
    assert summary["capacity_resolution"]["recovered_capacity_from_splits"] == 1
    assert summary["capacity_resolution"]["remaining_capacity_deficit"] == 0


def test_summary_reports_pending_and_invalid_split_labels():
    payload = {
        "capacity_deficit": 1,
        "candidate_labels": [
            _candidate("OB2", None),
            _candidate("OB1", "bad_label"),
        ],
    }

    summary = summarizer.summarize_split_candidate_labels(payload)

    assert summary["pending_candidate_ids"] == ["OB2"]
    assert summary["invalid_labels"] == [{"bend_id": "OB1", "label": "bad_label"}]
    assert summary["blocker_status"] == "invalid_split_labels"


def test_summary_reports_unresolved_when_no_candidate_splits():
    payload = {
        "capacity_deficit": 1,
        "candidate_labels": [
            _candidate("OB2", "keep_single"),
            _candidate("OB1", "keep_single"),
        ],
    }

    summary = summarizer.summarize_split_candidate_labels(payload)

    assert summary["blocker_status"] == "split_labels_do_not_resolve_capacity_deficit"
    assert summary["capacity_resolution"]["remaining_capacity_deficit"] == 1


def test_summary_reports_over_split_risk():
    payload = {
        "capacity_deficit": 1,
        "candidate_labels": [
            _candidate("OB2", "split_into_two_features"),
            _candidate("OB1", "split_into_two_features"),
        ],
    }

    summary = summarizer.summarize_split_candidate_labels(payload)

    assert summary["blocker_status"] == "split_labels_over_recover_capacity"
    assert summary["capacity_resolution"]["over_split_risk"] is True
