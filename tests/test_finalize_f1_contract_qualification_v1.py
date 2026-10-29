from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_f1_contract_qualification_v1",
    ROOT / "scripts/finalize_f1_contract_qualification_v1.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_report() -> dict:
    return {
        "schema": "F1_CONTRACT_SOAK_V1",
        "status": "PASS",
        "elapsed_seconds": 10_800.001,
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "isolated_child_jobs": 100,
        "isolated_abstention_jobs": 10,
        "module_sha256": {relative: MODULE._sha(ROOT / relative) for relative in MODULE.REQUIRED_MODULES},
    }


def test_rejects_short_soak() -> None:
    report = valid_report()
    report["elapsed_seconds"] = 10_799.999
    with pytest.raises(MODULE.QualificationError, match="below 10800"):
        MODULE.validate_report(report, ROOT)


def test_rejects_source_exposure_and_hash_drift() -> None:
    report = valid_report()
    report["physical_assets_opened"] = 1
    with pytest.raises(MODULE.QualificationError, match="opened physical"):
        MODULE.validate_report(report, ROOT)


def test_rejects_soak_without_controlled_abstention_trials() -> None:
    report = valid_report()
    report["isolated_abstention_jobs"] = 0
    with pytest.raises(MODULE.QualificationError, match="completion and abstention"):
        MODULE.validate_report(report, ROOT)
    report = valid_report()
    report["module_sha256"][MODULE.REQUIRED_MODULES[0]] = "0" * 64
    with pytest.raises(MODULE.QualificationError, match="changed"):
        MODULE.validate_report(report, ROOT)


def test_accepts_exact_frozen_manifest() -> None:
    report = valid_report()
    report["module_sha256"] = dict(reversed(list(report["module_sha256"].items())))
    MODULE.validate_report(report, ROOT)


def test_finalizer_manifest_matches_the_modules_recorded_by_the_soak() -> None:
    soak_source = (ROOT / "scripts/soak_f1_contracts_v1.py").read_text(encoding="utf-8")
    expected = {
        "scripts/soak_f1_contracts_v1.py",
        "research/f1_candidate_ir_v1.py",
        "research/f1_unified_gate_v1.py",
        "research/f1_isolated_job_v1.py",
        "research/cibec_v1/agreement.py",
    }
    assert set(MODULE.REQUIRED_MODULES) == expected
    for relative in expected:
        assert f'ROOT / "{relative}"' in soak_source
