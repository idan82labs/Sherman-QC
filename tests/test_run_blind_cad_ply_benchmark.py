import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_blind_cad_ply_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_blind_cad_ply_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_case_output_name_distinguishes_same_basename(tmp_path):
    first = {"part_key": "P1", "scan_path": str(tmp_path / "a" / "scan.ply")}
    second = {"part_key": "P1", "scan_path": str(tmp_path / "b" / "scan.ply")}

    assert MODULE._case_output_name(first) != MODULE._case_output_name(second)


def test_run_slice_forces_blind_mode_and_scores_partial_visible_count(monkeypatch, tmp_path):
    captured = {}

    def fake_run_case_subprocess(**kwargs):
        captured.update(kwargs)
        return {
            "inference_mode": "blind_cad_ply",
            "expectation": kwargs["expectation"],
            "metrics": {"expected_total_bends": 7},
        }

    monkeypatch.setattr(MODULE, "_run_case_subprocess", fake_run_case_subprocess)
    summary = MODULE.run_slice(
        slice_payload={
            "entries": [
                {
                    "part_key": "P1",
                    "cad_path": "/tmp/part.stp",
                    "scan_path": "/tmp/scan.ply",
                    "expectation": {
                        "state": "partial",
                        "expected_conventional_bend_count": 3,
                        "known_whole_part_total_context": 7,
                        "truth_record_id": "P1::scan.ply",
                    },
                }
            ]
        },
        output_dir=tmp_path,
        timeout_sec=10,
        selected_parts=set(),
        limit=None,
    )

    assert captured["inference_mode"] == "blind_cad_ply"
    assert captured["inference_part_id"].startswith("blind-")
    assert captured["inference_part_id"] != "P1"
    assert captured["expectation"] == {
        "state": "unknown",
        "expected_total_bends": None,
        "expected_completed_bends": None,
    }
    assert captured["spec_bend_angles"] == []
    assert captured["spec_schedule_type"] is None
    assert summary["results"][0]["expectation"] == {
        "state": "partial",
        "expected_total_bends": 7,
        "expected_completed_bends": 3,
    }
    assert summary["results"][0]["blind_inference_contract"]["scoring_truth_crossed_subprocess_boundary"] is False
    assert summary["results"][0]["blind_inference_contract"]["corpus_part_identifier_exposed_to_pipeline"] is False
    assert summary["inference_mode"] == "blind_cad_ply"
