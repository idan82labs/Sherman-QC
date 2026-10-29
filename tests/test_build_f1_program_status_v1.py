from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_f1_program_status_v1", ROOT / "scripts/build_f1_program_status_v1.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_status_separates_infrastructure_from_physical_progress(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.json").write_text(json.dumps({"status": "RUNNING", "elapsed_seconds": 10.0, "iterations": 5, "isolated_child_jobs": 1}))
    result = MODULE.build(ROOT, tmp_path)
    assert result["infrastructure_progress"] is True
    assert result["physical_generalization_progress"] is False
    assert result["production_promotion_authorized"] is False
    assert result["lane_status"]["scan_only"]["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    assert "FROZEN_THREE_HOUR_CONTRACT_SOAK_NOT_YET_QUALIFIED" in result["active_blockers"]
    assert "TWO_REAL_CIBEC_AUTHORITY_ACCEPTANCES_MISSING" in result["active_blockers"]
    assert result["next_model_attack_after_evidence_unlock"]["new_candidate_generator_funded"] is False


def test_status_rejects_superficial_pass_without_frozen_manifest(tmp_path: Path) -> None:
    (tmp_path / "final_report.json").write_text(json.dumps({
        "schema": "F1_CONTRACT_SOAK_V1",
        "status": "PASS",
        "elapsed_seconds": 10_800.0,
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "isolated_child_jobs": 100,
        "isolated_abstention_jobs": 10,
        "module_sha256": {},
    }))
    result = MODULE.build(ROOT, tmp_path)
    assert result["contract_soak"]["qualified"] is False
    assert result["contract_soak"]["validation_error"]
    assert "FROZEN_THREE_HOUR_CONTRACT_SOAK_NOT_YET_QUALIFIED" in result["active_blockers"]
