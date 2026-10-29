from __future__ import annotations

import json
from pathlib import Path

from scripts.build_f1_qualification_readiness_v1 import build


ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path: Path, soak: Path) -> dict:
    missing = tmp_path / "missing.json"
    return build(
        ROOT,
        soak_report=soak,
        authority_pack=ROOT / "output/research/cibec-v1-authority-candidate-v2",
        source_free_report=missing,
        physical_report=missing,
        cad_grouping_report=missing,
        custody_public_report=missing,
        unified_gate_report=missing,
    )


def test_invalid_soak_and_unsigned_authority_are_explicit(tmp_path: Path) -> None:
    soak = tmp_path / "soak.json"
    soak.write_text(json.dumps({"schema": "F1_CONTRACT_SOAK_V1", "status": "PASS", "elapsed_seconds": 10_800}))
    result = _build(tmp_path, soak)
    assert result["status"] == "BLOCKED"
    assert result["first_blocked_stage"] == "SOURCE_FREE_CONTRACT_SOAK"
    assert result["physical_generalization_progress"] is False
    authority = next(row for row in result["stages"] if row["stage"] == "TWO_AUTHORITY_CIBEC_ACCEPTANCE")
    assert authority["status"] == "BLOCKED"
    assert authority["blockers"] == [
        "MANUFACTURING_AUTHORITY_NAME_REQUIRED",
        "COUNTERSIGNING_AUTHORITY_NAME_REQUIRED",
    ]
    assert all(row["status"] == "NOT_REACHED" for row in result["stages"][2:])
