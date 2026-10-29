from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.run_cibec_v1_source_free_preflight import run


def test_unsigned_preflight_blocks_without_opening_physical_assets(tmp_path: Path) -> None:
    code, report = run(
        Path("research/cibec_v1/CIBEC_V1_CHARTER.yaml"),
        Path("research/cibec_v1/source_free_fixture_manifest.json"),
        tmp_path / "report.json",
    )
    assert code == 2
    assert report["status"] == "BLOCKED_PRE_PHYSICAL"
    assert report["fixture_count"] == 14
    assert report["physical_assets_opened"] == 0
    assert report["prediction_assets_opened"] == 0
    assert {item.split(":", 1)[0] for item in report["blockers"]} == {
        "CHARTER_SIGNATURE_REQUIRED",
        "SIGNED_FIXTURE_EXPECTATIONS_REQUIRED",
        "FIXTURE_AUTHORITY_SIGNATURE_REQUIRED",
    }
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == report


def test_preflight_cli_runs_without_pythonpath(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cibec_v1_source_free_preflight.py",
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": ""},
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "BLOCKED_PRE_PHYSICAL"
    assert report["physical_assets_opened"] == 0
