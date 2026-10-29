from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tests.test_f1_custody_v1 import payload


ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path: Path, value: dict) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(value))
    return subprocess.run(
        [sys.executable, "scripts/build_f1_blind_custody_assignment_v1.py", "--input", str(source),
         "--public-output", str(tmp_path / "public.json"), "--custodian-output", str(tmp_path / "custodian.json")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def test_cli_separates_public_assignment_from_scan_hash_map(tmp_path: Path) -> None:
    value = payload()
    result = run(tmp_path, value)
    assert result.returncode == 0
    public_bytes = (tmp_path / "public.json").read_text()
    custodian = json.loads((tmp_path / "custodian.json").read_text())
    assert all(row["scan_sha256"] not in public_bytes for row in value["cases"])
    assert all(row["morphology_group_sha256"] not in public_bytes for row in value["cases"])
    assert value["blinding_salt_sha256"] not in public_bytes
    assert len(custodian["cases"]) == 30


def test_cli_capacity_abstain_is_explicit_nonzero(tmp_path: Path) -> None:
    result = run(tmp_path, payload(29, 15))
    assert result.returncode == 2
    public = json.loads((tmp_path / "public.json").read_text())
    assert public["status"] == "CORPUS_CAPACITY_ABSTAIN"
    assert public["assignments"] == []
