import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_zero_shot_bends.py"
spec = importlib.util.spec_from_file_location("evaluate_zero_shot_bends", MODULE_PATH)
evaluate_zero_shot_bends = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(evaluate_zero_shot_bends)


def test_score_result_tracks_exact_count_range_angle_duplicate_and_abstention():
    result = {
        "estimated_bend_count": 4,
        "estimated_bend_count_range": [3, 5],
        "dominant_angle_families_deg": [90.8],
        "abstained": False,
        "profile_family": "repeated_90_sheetmetal",
        "support_breakdown": {
            "duplicate_line_family_rate": 0.125,
            "bend_support_strip_count": 3,
            "strip_duplicate_merge_count": 1,
        },
    }
    expectation = {
        "expected_bend_count": 4,
        "dominant_angle_families_deg": [90.0],
        "expected_profile_family": "repeated_90_sheetmetal",
        "expected_abstained": False,
    }

    metrics = evaluate_zero_shot_bends._score_result(result, expectation)

    assert metrics["predicted_exact_count"] == 4
    assert metrics["exact_count"] == 1
    assert metrics["count_range_contains_truth"] == 1
    assert metrics["dominant_angle_family_match"] == 1
    assert metrics["profile_family_match"] == 1
    assert metrics["abstention_correct"] == 1
    assert metrics["duplicate_line_family_rate"] == 0.125
    assert metrics["bend_support_strip_count"] == 3
    assert metrics["strip_duplicate_merge_count"] == 1
    assert metrics["abstained"] is False


def test_run_entry_is_label_blind_at_inference_time():
    captured = {}

    def fake_loader(_):
        return SimpleNamespace(local_refinement_kwargs={"deterministic_seed": 17, "retry_seeds": [29]})

    def fake_backend(*, scan_path, runtime_config, part_id=None):
        captured["scan_path"] = scan_path
        captured["part_id"] = part_id
        captured["runtime_config"] = runtime_config
        return {
            "profile_family": "repeated_90_sheetmetal",
            "estimated_bend_count": 12,
            "estimated_bend_count_range": [11, 13],
            "dominant_angle_families_deg": [90.0],
            "abstained": False,
            "duplicate_line_family_rate": 0.2,
        }

    entry = {
        "part_key": "YSHERMAN",
        "scan_path": "/tmp/ysherman.ply",
        "expectation": {"expected_bend_count": 12, "dominant_angle_families_deg": [90.0]},
    }

    item = evaluate_zero_shot_bends._run_entry(
        entry,
        run_zero_shot_bend_profile=fake_backend,
        load_runtime_config=fake_loader,
        deterministic_seed=101,
        retry_seeds=[211, 307],
    )

    assert captured["scan_path"] == "/tmp/ysherman.ply"
    assert captured["part_id"] == "YSHERMAN"
    assert item["metrics"]["exact_count"] == 1
    assert item["runtime_seed_profile"] == {"deterministic_seed": 101, "retry_seeds": [211, 307]}


def test_main_writes_summary_and_per_scan_results(monkeypatch, tmp_path):
    slice_json = tmp_path / "slice.json"
    output_dir = tmp_path / "out"
    slice_json.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "part_key": "P1",
                        "scan_path": "/tmp/scan_1.ply",
                        "expectation": {
                            "expected_bend_count": 4,
                            "dominant_angle_families_deg": [90.0],
                            "expected_profile_family": "repeated_90_sheetmetal",
                            "expected_abstained": False,
                        },
                    },
                    {
                        "part_key": "P2",
                        "scan_path": "/tmp/scan_2.ply",
                        "expectation": {
                            "expected_bend_count": 2,
                            "dominant_angle_families_deg": [45.0],
                            "expected_profile_family": "scan_limited_partial",
                            "expected_abstained": True,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_loader(_):
        return SimpleNamespace(local_refinement_kwargs={})

    def fake_backend(*, scan_path, runtime_config, part_id=None):
        if scan_path.endswith("scan_1.ply"):
            return {
                "profile_family": "repeated_90_sheetmetal",
                "estimated_bend_count": 4,
                "estimated_bend_count_range": [4, 4],
                "dominant_angle_families_deg": [90.0],
                "abstained": False,
                "support_breakdown": {"duplicate_line_family_rate": 0.1},
            }
        return {
            "profile_family": "scan_limited_partial",
            "estimated_bend_count": None,
            "estimated_bend_count_range": [1, 3],
            "dominant_angle_families_deg": [45.0],
            "abstained": True,
            "support_breakdown": {"duplicate_line_family_rate": 0.3},
        }

    monkeypatch.setattr(
        evaluate_zero_shot_bends,
        "_load_zero_shot_backend",
        lambda: (fake_backend, fake_loader),
    )
    monkeypatch.setattr(
        evaluate_zero_shot_bends,
        "DEFAULT_DEV_SLICE",
        slice_json,
    )
    monkeypatch.setattr(
        evaluate_zero_shot_bends,
        "argparse",
        SimpleNamespace(
            ArgumentParser=lambda **kwargs: SimpleNamespace(
                add_argument=lambda *args, **kwargs: None,
                parse_args=lambda: SimpleNamespace(
                    slice_json=str(slice_json),
                    output_dir=str(output_dir),
                    deterministic_seed=None,
                    retry_seeds_json=None,
                ),
            )
        ),
    )

    assert evaluate_zero_shot_bends.main() == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["scans_processed"] == 2
    assert summary["aggregate"]["successful_scans"] == 2
    assert summary["aggregate"]["exact_count_accuracy"] == 1.0
    assert summary["aggregate"]["count_range_contains_truth_rate"] == 1.0
    assert summary["aggregate"]["dominant_angle_family_match_rate"] == 1.0
    assert summary["aggregate"]["profile_family_match_rate"] == 1.0
    assert summary["aggregate"]["abstention_correctness_rate"] == 1.0
    assert summary["aggregate"]["mean_bend_support_strip_count"] is None
    assert summary["aggregate"]["abstained_count"] == 1

    result_files = sorted(output_dir.glob("*.json"))
    assert any(path.name == "summary.json" for path in result_files)
    assert any(path.name == "manifest.json" for path in result_files)
    per_scan = [path for path in result_files if path.name not in {"summary.json", "manifest.json"}]
    assert len(per_scan) == 2
    payload = json.loads(per_scan[0].read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "exact_count" in payload["metrics"]
    assert "count_range_contains_truth" in payload["metrics"]
