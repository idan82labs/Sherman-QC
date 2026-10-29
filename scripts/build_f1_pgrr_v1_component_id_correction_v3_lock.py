#!/usr/bin/env python3
"""Build the immutable PGRR component-ID correction-V3 execution lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_pgrr_v1.config import CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402


ATTEMPT_ID = "PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_COMPONENT_ID_CORRECTION_V3"
V2_LOCK_RELATIVE = "docs/research/PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2_LOCK_20260809.json"
V2_CLAIM_RELATIVE = "docs/research/PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2_CLAIM_20260809.json"
V2_OUTPUT_RELATIVE = "output/research/pgrr-v1-c006-frozen-target-causal-preflight-correction-v2"
RULING_RELATIVE = "docs/research/PGRR_V1_COMPONENT_ID_EXECUTION_STOP_CORRECTION_GPT56_PRO_RESPONSE_20260809.txt"
REGRESSION_RELATIVE = "docs/research/PGRR_V1_COMPONENT_ID_CORRECTION_V3_SOURCE_FREE_REGRESSION_V1_20260809.json"

CODE_PATHS = (
    "research/f1_pgrr_v1/__init__.py",
    "research/f1_pgrr_v1/config.py",
    "research/f1_pgrr_v1/geometry.py",
    "research/f1_pgrr_v1/independent_validator.py",
    "research/f1_pgrr_v1/frozen_handoff.py",
    "research/f1_pgrr_v1/tamper_suite.py",
    "research/f1_pgrr_v1/visual.py",
    "scripts/run_f1_pgrr_v1_candidate_worker.py",
    "scripts/run_f1_pgrr_v1_causal_preflight.py",
    "scripts/run_f1_pgrr_v1_source_free_import_regression.py",
    "scripts/run_f1_pgrr_v1_component_id_regression.py",
    "scripts/build_f1_pgrr_v1_preexecution_lock.py",
    "scripts/build_f1_pgrr_v1_correction_v2_lock.py",
    "scripts/build_f1_pgrr_v1_component_id_correction_v3_lock.py",
    "tests/test_f1_pgrr_v1.py",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_lock() -> dict[str, Any]:
    v2_lock = _read(ROOT / V2_LOCK_RELATIVE)
    v2_claim = _read(ROOT / V2_CLAIM_RELATIVE)
    v2_result = _read(ROOT / V2_OUTPUT_RELATIVE / "result.json")
    regression = _read(ROOT / REGRESSION_RELATIVE)
    if v2_claim.get("claim_state") != "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH":
        raise ValueError("V2 claim is not immutably consumed")
    if v2_result.get("decision") != "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP":
        raise ValueError("V2 result classification changed")
    if v2_result.get("architecture_scientifically_evaluated") is not False:
        raise ValueError("V2 architecture status changed")
    if regression.get("passed") is not True or not all(regression.get("checks", {}).values()):
        raise ValueError("component-ID source-free regression did not pass")
    if regression.get("implementation_sha256") != _sha_file(ROOT / "scripts/run_f1_pgrr_v1_component_id_regression.py"):
        raise ValueError("component-ID regression implementation changed")
    corrected = regression.get("corrected_implementation_sha256", {})
    if corrected.get("producer_geometry") != _sha_file(ROOT / "research/f1_pgrr_v1/geometry.py"):
        raise ValueError("producer changed after component-ID regression")
    if corrected.get("independent_validator") != _sha_file(ROOT / "research/f1_pgrr_v1/independent_validator.py"):
        raise ValueError("validator changed after component-ID regression")

    geometry_source = (ROOT / "research/f1_pgrr_v1/geometry.py").read_text(encoding="utf-8")
    validator_source = (ROOT / "research/f1_pgrr_v1/independent_validator.py").read_text(encoding="utf-8")
    if 'local = int(np.argsort(ids, kind="stable")[0])' not in geometry_source:
        raise ValueError("authorized producer correction absent")
    if "key=lambda offset: bytes(ids[indices[offset]])," not in validator_source:
        raise ValueError("authorized validator correction absent")

    code_sha = {relative: _sha_file(ROOT / relative) for relative in CODE_PATHS}
    v2_evidence_paths = (
        "producer-a-process.json",
        "00_binding_census.tsv",
        "09_role_consensus.tsv",
        "10_visual_audit_manifest.tsv",
        "failure_report.md",
        "visual/visual_critique.md",
        "result.json",
    )
    v2_evidence_sha = {
        f"{V2_OUTPUT_RELATIVE}/{relative}": _sha_file(ROOT / V2_OUTPUT_RELATIVE / relative)
        for relative in v2_evidence_paths
    }
    unchanged_binding = {
        "configuration": v2_lock["configuration"],
        "configuration_sha256": v2_lock["configuration_sha256"],
        "resource_proof": v2_lock["resource_proof"],
        "raw_source_relative": v2_lock["raw_source_relative"],
        "input_sha256": v2_lock["input_sha256"],
        "safety": v2_lock["safety"],
        "failure_policy": v2_lock["failure_policy"],
    }
    lock = {
        "schema_version": 3,
        "profile": ATTEMPT_ID,
        "authorization": {
            "ruling": "AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_COMPONENT_ID_CORRECTION_V3",
            "professor_response_relative": RULING_RELATIVE,
            "professor_response_sha256": _sha_file(ROOT / RULING_RELATIVE),
            "execution_count": "EXACTLY_ONE_FRESH_V3_ATTEMPT",
            "claim_consumption": "BEFORE_FIRST_WORKER_LAUNCH",
        },
        "v2_lineage": {
            "lock_relative": V2_LOCK_RELATIVE,
            "lock_file_sha256": _sha_file(ROOT / V2_LOCK_RELATIVE),
            "lock_self_sha256": v2_lock["execution_lock_sha256"],
            "consumed_claim_relative": V2_CLAIM_RELATIVE,
            "consumed_claim_sha256": _sha_file(ROOT / V2_CLAIM_RELATIVE),
            "classification": v2_result["decision"],
            "architecture_scientifically_evaluated": False,
            "evidence_sha256": v2_evidence_sha,
            "evidence_aggregate_sha256": sha_json(v2_evidence_sha),
            "immutable_no_overwrite": True,
        },
        "authorized_correction": {
            "semantics": "LEXICOGRAPHICALLY_SMALLEST_COMPLETE_32_BYTE_POINT_ID_HEX",
            "producer_old": "int(np.argmin(ids))",
            "producer_new": "int(np.argsort(ids, kind=\"stable\")[0])",
            "validator_old": "int(np.argmin(values))",
            "validator_new": "min(range(len(indices)), key=lambda offset: bytes(ids[indices[offset]]))",
            "independent_implementations": True,
            "additional_producer_or_validator_changes": False,
            "retry_authorized": False,
        },
        "source_free_regression": {
            "relative": REGRESSION_RELATIVE,
            "file_sha256": _sha_file(ROOT / REGRESSION_RELATIVE),
            "record": regression,
        },
        "configuration": v2_lock["configuration"],
        "configuration_sha256": v2_lock["configuration_sha256"],
        "resource_proof": v2_lock["resource_proof"],
        "raw_source_relative": v2_lock["raw_source_relative"],
        "code_sha256": code_sha,
        "corrected_source_aggregate_sha256": sha_json(code_sha),
        "input_sha256": v2_lock["input_sha256"],
        "unchanged_original_bindings_sha256": sha_json(unchanged_binding),
        "safety": v2_lock["safety"],
        "failure_policy": v2_lock["failure_policy"],
        "v3_terminal_rule": "NO_AUTOMATIC_V4_NO_RERUN_NO_REPLACEMENT_NO_FALLBACK_NO_SECOND_CORRECTION_LANE",
        "visual_rule": "VALID_RENDER_REQUIRES_MANUAL_CANONICAL_PERMUTED_NEGATIVE_AND_CAUSAL_CONTROL_RECONCILIATION",
    }
    if lock["configuration_sha256"] != CONFIG_SHA256:
        raise ValueError("PGRR configuration changed")
    lock["execution_lock_sha256"] = sha_json(lock)
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("component-ID correction-V3 lock builder accepts no arguments")
    print(canonical_json_bytes(build_lock()).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
