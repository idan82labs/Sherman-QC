import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_cad_gold_set.py"
spec = importlib.util.spec_from_file_location("evaluate_cad_gold_set", MODULE_PATH)
evaluate_cad_gold_set = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(evaluate_cad_gold_set)


def test_project_result_extracts_structured_row():
    row = evaluate_cad_gold_set._project_result(
        {
            "part_key": "P1",
            "scan_name": "scan_1.ply",
            "details": {
                "cad_strategy": "detector_mesh",
                "profile_mode": "cad_nominal",
                "nominal_source_class": "semantic_nominal_cad",
                "cad_structural_target": {
                    "expected_bend_count": 4,
                    "dominant_angle_families_deg": [90.0],
                },
                "scan_only_diagnostic_mismatch": {"flag": "none"},
            },
            "metrics": {
                "blocker_attribution_breakdown": {"engine_gap": 1},
                "correspondence_decision_ready_bends": 3,
                "position_eligible_bends": 2,
            },
            "matches": [
                {"bend_id": "CAD_B1", "status": "PASS", "countable_in_regression": True},
                {"bend_id": "CAD_B2", "status": "NOT_DETECTED", "countable_in_regression": True},
                {"bend_id": "CAD_R1", "status": "PASS", "countable_in_regression": False},
            ],
        }
    )

    assert row["cad_strategy"] == "detector_mesh"
    assert row["profile_mode"] == "cad_nominal"
    assert row["expected_bend_count"] == 4
    assert row["dominant_angle_families_deg"] == [90.0]
    assert row["matched_bend_ids"] == ["CAD_B1"]


def test_compare_rows_detects_field_mismatches():
    comparison = evaluate_cad_gold_set._compare_rows(
        [
            {
                "part_key": "P1",
                "scan_name": "scan_1.ply",
                "cad_strategy": "detector_mesh",
                "profile_mode": "cad_nominal",
                "nominal_source_class": "semantic_nominal_cad",
                "expected_bend_count": 4,
                "dominant_angle_families_deg": [90.0],
                "blocker_attribution_breakdown": {"engine_gap": 1},
                "correspondence_ready_bends": 3,
                "position_eligible_bends": 2,
                "matched_bend_ids": ["CAD_B1"],
                "scan_only_diagnostic_mismatch": None,
            }
        ],
        [
            {
                "part_key": "P1",
                "scan_name": "scan_1.ply",
                "cad_strategy": "legacy_mesh",
                "profile_mode": "cad_nominal",
                "nominal_source_class": "semantic_nominal_cad",
                "expected_bend_count": 4,
                "dominant_angle_families_deg": [90.0],
                "blocker_attribution_breakdown": {"engine_gap": 1},
                "correspondence_ready_bends": 3,
                "position_eligible_bends": 2,
                "matched_bend_ids": ["CAD_B1"],
                "scan_only_diagnostic_mismatch": None,
            }
        ],
    )

    assert comparison["rows_evaluated"] == 1
    assert comparison["exact_matches"] == 0
    assert comparison["mismatch_count"] == 1
    assert comparison["mismatches"][0]["fields"]["cad_strategy"]["actual"] == "detector_mesh"


def test_filter_projected_rows_respects_explicit_scan_selectors():
    projected_rows = [
        {"part_key": "P1", "scan_name": "scan_1.ply"},
        {"part_key": "P1", "scan_name": "scan_2.ply"},
        {"part_key": "P2", "scan_name": "scan_3.ply"},
    ]

    filtered = evaluate_cad_gold_set._filter_projected_rows(
        projected_rows,
        [("P1", "scan_2.ply"), ("P2", None)],
    )

    assert filtered == [
        {"part_key": "P1", "scan_name": "scan_2.ply"},
        {"part_key": "P2", "scan_name": "scan_3.ply"},
    ]


def test_main_writes_summary_and_comparison(monkeypatch, tmp_path):
    summary_json = tmp_path / "input_summary.json"
    gold_json = tmp_path / "gold.json"
    output_dir = tmp_path / "out"

    summary_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "P1",
                        "scan_name": "scan_1.ply",
                        "details": {
                            "cad_strategy": "detector_mesh",
                            "profile_mode": "cad_nominal",
                            "nominal_source_class": "semantic_nominal_cad",
                            "cad_structural_target": {
                                "expected_bend_count": 4,
                                "dominant_angle_families_deg": [90.0],
                            },
                        },
                        "metrics": {
                            "blocker_attribution_breakdown": {"engine_gap": 1},
                            "correspondence_decision_ready_bends": 3,
                            "position_eligible_bends": 2,
                        },
                        "matches": [{"bend_id": "CAD_B1", "status": "PASS", "countable_in_regression": True}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    gold_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "part_key": "P1",
                        "scan_name": "scan_1.ply",
                        "cad_strategy": "detector_mesh",
                        "profile_mode": "cad_nominal",
                        "nominal_source_class": "semantic_nominal_cad",
                        "expected_bend_count": 4,
                        "dominant_angle_families_deg": [90.0],
                        "blocker_attribution_breakdown": {"engine_gap": 1},
                        "correspondence_ready_bends": 3,
                        "position_eligible_bends": 2,
                        "matched_bend_ids": ["CAD_B1"],
                        "scan_only_diagnostic_mismatch": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_cad_gold_set,
        "argparse",
        SimpleNamespace(
            ArgumentParser=lambda **kwargs: SimpleNamespace(
                add_argument=lambda *args, **kwargs: None,
                parse_args=lambda: SimpleNamespace(
                    summary_json=str(summary_json),
                    run_evaluator=False,
                    slice_json="unused",
                    gold_json=str(gold_json),
                    output_dir=str(output_dir),
                    write_gold=False,
                    deterministic_seed=None,
                    retry_seeds_json=None,
                ),
            )
        ),
    )

    assert evaluate_cad_gold_set.main() == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["projected_row_count"] == 1
    assert comparison["exact_matches"] == 1
    assert len(manifest["entries"]) == 1
