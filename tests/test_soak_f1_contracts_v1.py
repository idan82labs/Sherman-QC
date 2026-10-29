from __future__ import annotations

import json
from pathlib import Path

from scripts.soak_f1_contracts_v1 import run


def test_short_soak_is_deterministic_and_source_free(tmp_path: Path) -> None:
    first = run(output=tmp_path / "first", duration_seconds=60, seed=17, max_iterations=25, child_interval=10, checkpoint_seconds=0.001)
    second = run(output=tmp_path / "second", duration_seconds=60, seed=17, max_iterations=25, child_interval=10, checkpoint_seconds=0.001)
    assert first["status"] == second["status"] == "PASS"
    assert first["iterations"] == second["iterations"] == 25
    assert first["isolated_child_jobs"] == second["isolated_child_jobs"] == 2
    assert first["isolated_abstention_jobs"] == second["isolated_abstention_jobs"] == 0
    assert first["digest_chain_sha256"] == second["digest_chain_sha256"]
    assert first["physical_assets_opened"] == first["prediction_assets_opened"] == 0
    assert json.loads((tmp_path / "first/final_report.json").read_text())["status"] == "PASS"


def test_soak_repeatedly_exercises_declared_abstention(tmp_path: Path) -> None:
    result = run(
        output=tmp_path / "abstain",
        duration_seconds=60,
        seed=19,
        max_iterations=100,
        child_interval=10,
        checkpoint_seconds=60,
    )
    assert result["isolated_child_jobs"] == 10
    assert result["isolated_abstention_jobs"] == 1
