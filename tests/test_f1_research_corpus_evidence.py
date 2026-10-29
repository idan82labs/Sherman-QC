from __future__ import annotations

import json
from pathlib import Path

from scripts.build_f1_research_corpus_freeze import build_freeze
from scripts.score_zero_shot_research_corpus import _micro, _wilson


def test_freeze_separates_inference_inputs_from_manual_truth(tmp_path: Path) -> None:
    scan = tmp_path / "named_part_123.ply"
    scan.write_bytes(b"ply\n")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "manual_bends": [
                    {"bend_id": "M1", "line_start": [0, 0, 0], "line_end": [1, 0, 0]}
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "part::scan",
                        "part_key": "part-123",
                        "scan_name": scan.name,
                        "scan_path": str(scan),
                        "scan_state": "full",
                        "truth_status": "manual_verified",
                        "release_metric_eligible": True,
                        "counts": {"observed_conventional_bends": 1},
                        "instance_geometry": {"source_file": str(labels)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    inference_path, gold_path = build_freeze(registry, tmp_path / "freeze")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    assert inference["cases"][0]["case_token"] == "C01"
    assert inference["cases"][0]["opaque_input_name"] == "input.ply"
    assert "manual_bends" not in inference["cases"][0]
    assert "gold_bend_count" not in inference["cases"][0]
    assert "part_key" not in inference["cases"][0]
    assert gold["cases"][0]["gold_bend_count"] == 1
    assert len(gold["cases"][0]["manual_bends"]) == 1


def test_micro_metrics_penalize_wrong_geometry_even_when_counts_match() -> None:
    metrics = _micro(
        [
            {"true_positive": 1, "false_positive": 2, "false_negative": 2},
            {"true_positive": 2, "false_positive": 0, "false_negative": 0},
        ]
    )
    assert metrics["gold_bend_count"] == metrics["prediction_count"] == 5
    assert metrics["true_positive"] == 3
    assert metrics["line_f1"] == 0.6


def test_wilson_interval_is_bounded_for_small_corpus() -> None:
    interval = _wilson(1, 13)
    assert interval is not None
    assert 0.0 <= interval[0] < 1 / 13 < interval[1] <= 1.0
