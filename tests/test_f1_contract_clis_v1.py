from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tests.test_f1_unified_gate_v1 import evidence


def test_candidate_cli_is_atomic_and_bare_script_safe(tmp_path: Path) -> None:
    source = tmp_path / "rows.json"; output = tmp_path / "canonical.json"
    source.write_text(json.dumps([{"line_start": [20, 0, 0], "line_end": [0, 0, 0], "source_route": "mesh", "source_stage": "birth", "source_ids": ["x"]}]))
    result = subprocess.run([sys.executable, "scripts/canonicalize_f1_candidate_ir_v1.py", "--input", str(source), "--output", str(output)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0
    payload = json.loads(output.read_text())
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["line_start_mm"] == [0.0, 0.0, 0.0]
    assert not list(tmp_path.glob(".*.tmp"))


def test_unified_gate_cli_fails_closed_but_can_write_development_report(tmp_path: Path) -> None:
    row = evidence(); row["evidence_class"] = "CONSUMED_PHYSICAL"
    source = tmp_path / "evidence.json"; output = tmp_path / "report.json"
    source.write_text(json.dumps(row))
    base = [sys.executable, "scripts/run_f1_unified_gate_v1.py", "--evidence", str(source), "--output", str(output)]
    result = subprocess.run(base, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    result = subprocess.run([*base, "--allow-non-a"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0
