from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.f1_semantic_oracle_v1.evaluator import OracleContractError, evaluate_submission


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    catalog_path = tmp_path / "data/catalog.json"
    catalog = {
        "cases": [
            {
                "case_id": "A/1",
                "part_id": "A",
                "truth_tier": "authoritative_user_verified",
                "promotion_eligible": True,
                "bends": [
                    {"bend_id": "gold-secret", "line_start": [0, 0, 0], "line_end": [10, 0, 0]}
                ],
            },
            {
                "case_id": "B/1",
                "part_id": "B",
                "truth_tier": "authoritative_user_verified",
                "promotion_eligible": True,
                "bends": [
                    {"bend_id": "gold-b-secret", "line_start": [0, 5, 0], "line_end": [10, 5, 0]}
                ],
            },
        ]
    }
    _write(catalog_path, catalog)
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    lock_path = tmp_path / "docs/research/lock.json"
    lock = {
        "lock_id": "test-lock",
        "catalog": {
            "path": "data/catalog.json",
            "sha256": catalog_sha,
            "required_truth_tier": "authoritative_user_verified",
        },
        "family_A": {"part_id": "A", "case_ids": ["A/1"]},
        "family_B": {"part_id": "B", "case_ids": ["B/1"]},
        "matching": {
            "axis_angle_degrees_max": 5.0,
            "transverse_offset_mm_max": 4.0,
            "gold_span_overlap_min": 0.6,
            "prediction_span_overlap_min": 0.6,
        },
    }
    _write(lock_path, lock)
    return lock_path, catalog_path


def test_family_a_feedback_hides_gold_geometry(tmp_path: Path) -> None:
    lock_path, _catalog_path = _fixture(tmp_path)
    prediction_path = tmp_path / "prediction.json"
    _write(
        prediction_path,
        {
            "owned_bend_regions": [
                {"bend_id": "candidate-1", "line_start": [0, 0, 0], "line_end": [10, 0, 0]},
                {"bend_id": "candidate-2", "line_start": [2, 0, 0], "line_end": [4, 0, 0]},
            ]
        },
    )
    report = evaluate_submission(lock_path=lock_path, prediction_path=prediction_path, phase="family_A")
    encoded = json.dumps(report, sort_keys=True)
    assert "gold-secret" not in encoded
    assert "line_start" not in encoded
    assert report["metrics"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 0,
        "aggregate_exact_count": False,
        "aggregate_precision": 0.5,
        "aggregate_recall": 1.0,
        "aggregate_f1": 0.666667,
    }
    assert {row["ownership_class"] for row in report["candidate_feedback"]} == {
        "TRUE_BEND_OBJECT",
        "DUPLICATE_ALIAS",
    }


def test_family_b_requires_and_consumes_exact_claim(tmp_path: Path) -> None:
    lock_path, _catalog_path = _fixture(tmp_path)
    prediction_path = tmp_path / "prediction.json"
    _write(prediction_path, {"owned_bend_regions": []})
    with pytest.raises(OracleContractError, match="fresh one-shot claim"):
        evaluate_submission(lock_path=lock_path, prediction_path=prediction_path, phase="family_B")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    submission_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    claim_path = tmp_path / "claim.json"
    _write(claim_path, {"status": "created", "lock_sha256": lock_sha, "submission_sha256": submission_sha})
    report = evaluate_submission(
        lock_path=lock_path,
        prediction_path=prediction_path,
        phase="family_B",
        family_b_claim_path=claim_path,
    )
    assert report["metrics"]["false_negative"] == 1
    assert json.loads(claim_path.read_text())["status"] == "consumed"
    with pytest.raises(OracleContractError, match="not fresh"):
        evaluate_submission(
            lock_path=lock_path,
            prediction_path=prediction_path,
            phase="family_B",
            family_b_claim_path=claim_path,
        )
