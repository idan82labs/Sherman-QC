from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json"
OUTPUT = ROOT / "output/research/f1-v51-today-axis-span-recovery-v1-20260831-r3"
REPORT = OUTPUT / "cached_axis_span_development_report.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_report_is_bound_to_current_final_model_bytes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert _sha(MODEL) == "8689db67f8a7f1df04c4cc43b104ef116e0cecb9004d5aca580764091318ec7b"
    assert report["model_sha256"] == _sha(MODEL)
    assert report["axis_span_successor_resubstitution"] == {
        "case_count": 28,
        "exact_case_count": 5,
        "f1": 0.632479,
        "false_negative": 28,
        "false_positive": 15,
        "gold_bend_count": 65,
        "precision": 0.711538,
        "prediction_count": 52,
        "recall": 0.569231,
        "true_positive": 37,
    }
    assert report["case_group_held_out_diagnostic"]["f1"] == 0.483333
    assert report["generalization_evidence"] is False
    assert report["promotion_eligible"] is False


def test_final_primary_and_replay_artifacts_are_byte_identical() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["all_case_primary_replay_byte_identity"] is True
    assert set(report["case_primary_replay_byte_identity"]) == {
        f"T{index:03d}" for index in range(1, 29)
    }
    for case_id, identical in report["case_primary_replay_byte_identity"].items():
        assert identical is True
        primary = OUTPUT / "primary" / case_id / "semantic_result.json"
        replay = OUTPUT / "replay" / case_id / "semantic_result.json"
        assert primary.read_bytes() == replay.read_bytes()
