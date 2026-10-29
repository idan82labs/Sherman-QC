from pathlib import Path

from scripts.run_f1_structured_union_batch import _run_case


def test_batch_case_failure_is_reported_not_hidden(tmp_path: Path):
    result = _run_case(
        {
            "case_token": "C001",
            "candidate_pool": str(tmp_path / "missing.json"),
            "output_dir": str(tmp_path / "out"),
            "app_python": "/usr/bin/false",
            "solver_python": "/usr/bin/false",
            "disable_birth": True,
        }
    )

    assert result["success"] is False
    assert result["returncode"] != 0

