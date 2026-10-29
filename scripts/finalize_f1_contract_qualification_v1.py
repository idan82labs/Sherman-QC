#!/usr/bin/env python3
"""Fail-closed finalizer for the frozen F1 contract qualification soak."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MINIMUM_SOAK_SECONDS = 10_800.0
REQUIRED_MODULES = (
    "scripts/soak_f1_contracts_v1.py",
    "research/f1_candidate_ir_v1.py",
    "research/f1_unified_gate_v1.py",
    "research/f1_isolated_job_v1.py",
    "research/cibec_v1/agreement.py",
)
RELEASE_FILES = (
    *REQUIRED_MODULES,
    "research/cibec_v1/contract.py",
    "research/f1_custody_v1.py",
    "research/f1_cibec_qualification_v1.py",
    "research/f1_cad_grouping_agreement_v1.py",
    "research/f1_shadow_batch_v1.py",
    "scripts/canonicalize_f1_candidate_ir_v1.py",
    "scripts/run_f1_unified_gate_v1.py",
    "scripts/run_f1_isolated_job_v1.py",
    "scripts/build_f1_retained_stage_graph_v1.py",
    "scripts/build_f1_current_unified_gate_evidence_v1.py",
    "scripts/build_f1_blind_custody_assignment_v1.py",
    "scripts/score_f1_cibec_physical_qualification_v1.py",
    "scripts/score_f1_cad_grouping_qualification_v1.py",
    "scripts/build_f1_program_status_v1.py",
    "scripts/build_f1_qualification_readiness_v1.py",
    "scripts/build_cibec_v1_neutral_authority_pack.py",
    "scripts/build_cibec_v1_authority_candidate.py",
    "scripts/validate_cibec_v1_authority_review.py",
    "scripts/materialize_cibec_v1_signed_authority.py",
    "scripts/run_cibec_v1_source_free_qualification.py",
    "scripts/finalize_f1_contract_qualification_v1.py",
    "tests/test_f1_candidate_ir_v1.py",
    "tests/test_f1_unified_gate_v1.py",
    "tests/test_f1_isolated_job_v1.py",
    "tests/test_soak_f1_contracts_v1.py",
    "tests/test_f1_contract_clis_v1.py",
    "tests/test_build_f1_retained_stage_graph_v1.py",
    "tests/test_build_f1_current_unified_gate_evidence_v1.py",
    "tests/test_f1_custody_v1.py",
    "tests/test_f1_custody_cli_v1.py",
    "tests/test_f1_cibec_qualification_v1.py",
    "tests/test_f1_cad_grouping_agreement_v1.py",
    "tests/test_score_f1_cad_grouping_qualification_v1.py",
    "tests/test_f1_shadow_batch_v1.py",
    "tests/test_build_f1_program_status_v1.py",
    "tests/test_build_f1_qualification_readiness_v1.py",
    "tests/test_build_cibec_v1_neutral_authority_pack.py",
    "tests/test_build_cibec_v1_authority_candidate.py",
    "tests/test_validate_cibec_v1_authority_review.py",
    "tests/test_cibec_v1_signed_qualification.py",
    "tests/test_cibec_v1_contract.py",
    "docs/research/artifacts/2026-08-25_f1_retained_stage_graph_v1.json",
    "docs/research/artifacts/2026-08-25_f1_current_lane_separated_unified_gate_v1.json",
    "docs/research/artifacts/2026-08-25_f1_program_status_v1.json",
    "docs/research/artifacts/2026-08-25_f1_qualification_readiness_v1.json",
    "docs/research/artifacts/2026-08-25_cibec-v1_neutral-authority-pack-v4_visual-audit.json",
    "docs/research/artifacts/2026-08-25_cibec-v1_authority-candidate-v2_visual-audit.json",
    "docs/research/artifacts/2026-08-25_cibec-v1_authority-candidate-v2_validation.json",
    "docs/research/2026-08-25_cibec-v1_neutral-authority-pack-v4_visual-audit.md",
    "docs/research/2026-08-25_cibec-v1_authority-candidate-v2_visual-audit.md",
    "docs/research/2026-08-25_f1_cibec_authority_handoff_v2.md",
    "docs/research/2026-08-25_f1_cibec_qualification_input_contract_v1.md",
    "docs/research/2026-08-25_f1_post_cibec_dual_lane_preregistration_v1.md",
    "docs/research/2026-08-25_f1_three_hour_contract_qualification_and_next_action.md",
)


class QualificationError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_report(report: dict[str, Any], root: Path) -> None:
    if report.get("schema") != "F1_CONTRACT_SOAK_V1" or report.get("status") != "PASS":
        raise QualificationError("soak did not produce F1_CONTRACT_SOAK_V1 PASS")
    elapsed = report.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or float(elapsed) < MINIMUM_SOAK_SECONDS:
        raise QualificationError("soak elapsed time is below 10800 seconds")
    if report.get("physical_assets_opened") != 0 or report.get("prediction_assets_opened") != 0:
        raise QualificationError("soak opened physical or prediction assets")
    child_jobs = report.get("isolated_child_jobs")
    abstention_jobs = report.get("isolated_abstention_jobs")
    if (
        isinstance(child_jobs, bool) or not isinstance(child_jobs, int) or child_jobs <= 0
        or isinstance(abstention_jobs, bool) or not isinstance(abstention_jobs, int) or abstention_jobs <= 0
        or abstention_jobs > child_jobs
    ):
        raise QualificationError("soak did not exercise both isolated completion and abstention paths")
    hashes = report.get("module_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_MODULES):
        raise QualificationError("soak module manifest is missing or contains extra entries")
    for relative in REQUIRED_MODULES:
        if hashes[relative] != _sha(root / relative):
            raise QualificationError(f"frozen module changed during or after soak: {relative}")


def build(root: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report, root)
    release_hashes = {}
    for relative in RELEASE_FILES:
        path = root / relative
        if not path.is_file():
            raise QualificationError(f"release file missing: {relative}")
        release_hashes[relative] = _sha(path)
    current_gate = json.loads((root / "docs/research/artifacts/2026-08-25_f1_current_lane_separated_unified_gate_v1.json").read_text(encoding="utf-8"))
    lanes = {
        key: {
            "grade": value["grade"],
            "direction": value["direction"],
            "operational_badge": value["operational_badge"],
        }
        for key, value in current_gate["gates"].items()
    }
    return {
        "schema_version": 1,
        "kind": "F1_CONTRACT_QUALIFICATION_RELEASE_V1",
        "status": "PASS",
        "qualification_scope": "SOURCE_FREE_CONTRACT_AND_OPERATIONS_INFRASTRUCTURE_ONLY",
        "physical_generalization_claim": False,
        "production_promotion_authorized": False,
        "soak_report_relative": str(report_path.resolve().relative_to(root)),
        "soak_report_sha256": _sha(report_path),
        "soak_elapsed_seconds": report["elapsed_seconds"],
        "soak_iterations": report["iterations"],
        "soak_isolated_child_jobs": report["isolated_child_jobs"],
        "soak_isolated_abstention_jobs": report["isolated_abstention_jobs"],
        "soak_digest_chain_sha256": report["digest_chain_sha256"],
        "lane_status": lanes,
        "release_file_sha256": release_hashes,
    }


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--soak-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.root.resolve(), args.soak_report.resolve())
    _atomic(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "qualification_scope", "soak_elapsed_seconds", "soak_iterations")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
