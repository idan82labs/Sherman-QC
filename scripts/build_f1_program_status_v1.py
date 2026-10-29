#!/usr/bin/env python3
"""Build a machine-readable Sherman F1 program status and critical path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_cibec_v1_authority_review import validate as validate_authority
from scripts.finalize_f1_contract_qualification_v1 import QualificationError, validate_report


CURRENT_GATE = Path("docs/research/artifacts/2026-08-25_f1_current_lane_separated_unified_gate_v1.json")
AUTHORITY_PACK = Path("output/research/cibec-v1-authority-candidate-v2")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build(root: Path, soak_directory: Path) -> dict[str, Any]:
    current = _load(root / CURRENT_GATE)
    authority_code, authority = validate_authority(root / AUTHORITY_PACK)
    final_path = soak_directory / "final_report.json"
    checkpoint_path = soak_directory / "checkpoint.json"
    soak = _load(final_path if final_path.exists() else checkpoint_path) if (final_path.exists() or checkpoint_path.exists()) else {"status": "NOT_STARTED"}
    soak_validation_error = None
    soak_valid = False
    if final_path.exists():
        try:
            validate_report(soak, root)
            soak_valid = True
        except (QualificationError, KeyError, TypeError, ValueError) as exc:
            soak_validation_error = str(exc)
    authority_ready = authority_code == 0
    blockers = []
    if not soak_valid:
        blockers.append("FROZEN_THREE_HOUR_CONTRACT_SOAK_NOT_YET_QUALIFIED")
    if not authority_ready:
        blockers.append("TWO_REAL_CIBEC_AUTHORITY_ACCEPTANCES_MISSING")
    lane_status = {
        key: {
            "grade": gate["grade"],
            "direction": gate["direction"],
            "operational_badge": gate["operational_badge"],
        }
        for key, gate in current["gates"].items()
    }
    return {
        "schema_version": 1,
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V1",
        "objective": "TRUSTWORTHY_ZERO_SHOT_SCAN_ONLY_AND_SEPARATE_CAD_ASSISTED_BEND_INSPECTION",
        "physical_generalization_progress": False,
        "infrastructure_progress": True,
        "production_promotion_authorized": False,
        "lane_status": lane_status,
        "contract_soak": {
            "status": soak.get("status"),
            "elapsed_seconds": soak.get("elapsed_seconds"),
            "iterations": soak.get("iterations"),
            "isolated_child_jobs": soak.get("isolated_child_jobs"),
            "isolated_abstention_jobs": soak.get("isolated_abstention_jobs"),
            "qualified": soak_valid,
            "validation_error": soak_validation_error,
        },
        "cibec_authority": {
            "status": authority["status"],
            "review_payload_sha256": authority["review_payload_sha256"],
            "blockers": authority["blockers"],
            "ready_to_materialize": authority_ready,
        },
        "active_blockers": blockers,
        "critical_path": [
            "COMPLETE_FROZEN_THREE_HOUR_CONTRACT_SOAK",
            "OBTAIN_TWO_REAL_EXACT_HASH_CIBEC_ACCEPTANCES",
            "MATERIALIZE_SIGNED_CIBEC_AND_PASS_14_CASE_SOURCE_FREE_QUALIFICATION",
            "RUN_13_CONSUMED_CASE_DUAL_ANNOTATOR_CIBEC_QUALIFICATION_WITH_NATIVE_VISUAL_PASS",
            "VERIFY_NEW_CORPUS_CAPACITY_AT_LEAST_30_SCANS_AND_15_MORPHOLOGIES_FROM_METADATA_ONLY",
            "ALLOCATE_AND_ANNOTATE_SEQUESTERED_NEW_CORPUS_WITHOUT_INFERENCE",
            "FREEZE_ONE_SCAN_ONLY_AND_ONE_CAD_ASSISTED_CANDIDATE_BEFORE_TRUTH_OPENING",
            "RUN_UNTOUCHED_30_BY_15_PHYSICAL_COMPARISON_AND_UNIFIED_GATE",
        ],
        "next_model_attack_after_evidence_unlock": {
            "scan_only": "EVENT_OWNERSHIP_CONSOLIDATION_AND_FLAT_FACE_VETO_ON_RETAINED_V51",
            "cad_assisted": "MANY_TO_ONE_CAD_EVENT_CONTRACTION_THEN_SCAN_LOCALIZED_LINE",
            "new_candidate_generator_funded": False,
        },
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
    parser.add_argument("--soak-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root.resolve(), args.soak_directory.resolve())
    _atomic(args.output, result)
    print(json.dumps({"active_blockers": result["active_blockers"], "lane_status": result["lane_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
