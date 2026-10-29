from __future__ import annotations

from copy import deepcopy

from scripts.audit_f1_gold37_split_integrity import audit_split_integrity


def _fixture() -> tuple[dict, dict, dict]:
    gold = {
        "corpus_freeze_id": "freeze",
        "cases": [
            {
                "case_token": "C1",
                "part_key": "P1",
                "source_scan_sha256": "scan1",
                "label_sha256": "label1",
                "gold_bend_count": 2,
            },
            {
                "case_token": "C2",
                "part_key": "P2",
                "source_scan_sha256": "scan2",
                "label_sha256": "label2",
                "gold_bend_count": 3,
            },
        ],
    }
    inference = {
        "corpus_freeze_id": "freeze",
        "cases": [
            {"case_token": "C1", "source_scan_sha256": "scan1"},
            {"case_token": "C2", "source_scan_sha256": "scan2"},
        ],
    }
    split = {
        "corpus_freeze_id": "freeze",
        "split_id": "split",
        "cases": [
            {"case_token": "C1", "part_key": "P1", "fold": 0},
            {"case_token": "C2", "part_key": "P2", "fold": 1},
        ],
        "folds": [
            {
                "fold": 0,
                "locked_part_families": ["P1"],
                "development_part_families": ["P2"],
            },
            {
                "fold": 1,
                "locked_part_families": ["P2"],
                "development_part_families": ["P1"],
            },
        ],
    }
    return gold, inference, split


def test_valid_grouped_split_passes() -> None:
    report = audit_split_integrity(*_fixture())

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["case_count"] == 2
    assert report["part_family_count"] == 2
    assert report["inference_metadata_exposure"] is False


def test_family_or_exact_scan_cross_fold_fails_closed() -> None:
    gold, inference, split = _fixture()
    gold["cases"].append(
        {
            "case_token": "C3",
            "part_key": "P1",
            "source_scan_sha256": "scan1",
            "label_sha256": "label3",
            "gold_bend_count": 1,
        }
    )
    inference["cases"].append({"case_token": "C3", "source_scan_sha256": "scan1"})
    split["cases"].append({"case_token": "C3", "part_key": "P1", "fold": 1})
    split["folds"][1]["locked_part_families"].append("P1")
    split["folds"][1]["development_part_families"] = []

    report = audit_split_integrity(gold, inference, split)

    assert report["passed"] is False
    assert report["checks"]["part_families_confined_to_one_fold"] is False
    assert report["checks"]["byte_identical_scans_confined_to_one_fold"] is False
    assert report["cross_fold_part_families"] == {"P1": [0, 1]}


def test_case_or_hash_drift_fails_closed() -> None:
    gold, inference, split = _fixture()
    broken_inference = deepcopy(inference)
    broken_inference["cases"][0]["source_scan_sha256"] = "changed"
    broken_split = deepcopy(split)
    broken_split["cases"].pop()

    hash_report = audit_split_integrity(gold, broken_inference, split)
    case_report = audit_split_integrity(gold, inference, broken_split)

    assert hash_report["passed"] is False
    assert any("scan hash" in error for error in hash_report["errors"])
    assert case_report["passed"] is False
    assert case_report["checks"]["case_sets_equal"] is False
