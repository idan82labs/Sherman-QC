#!/usr/bin/env python3
"""Build the fresh immutable PGRR correction-V2 lock after regression success."""

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
from scripts.build_f1_pgrr_v1_preexecution_lock import allocation_proof  # noqa: E402


ATTEMPT_ID = "PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2"
PRIOR_LOCK_RELATIVE = "docs/research/PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_V1_LOCK_20260809.json"
PRIOR_CLAIM_RELATIVE = "docs/research/PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_V1_CLAIM_20260809.json"
PRIOR_RESULT_RELATIVE = "output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/result.json"
PRIOR_PROCESS_RELATIVE = "output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/producer-a-process.json"
PRIOR_VISUAL_RELATIVE = "output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/visual/visual_critique.md"
RULING_RELATIVE = "docs/research/PGRR_V1_PRE_SOURCE_EXECUTION_STOP_CORRECTION_GPT56_PRO_RESPONSE_20260809.txt"
REGRESSION_RELATIVE = "docs/research/PGRR_V1_SOURCE_FREE_WORKER_IMPORT_REGRESSION_V1_20260809.json"

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
    "scripts/build_f1_pgrr_v1_preexecution_lock.py",
    "scripts/build_f1_pgrr_v1_correction_v2_lock.py",
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
    prior_lock = _read(ROOT / PRIOR_LOCK_RELATIVE)
    prior_claim = _read(ROOT / PRIOR_CLAIM_RELATIVE)
    regression = _read(ROOT / REGRESSION_RELATIVE)
    if prior_claim.get("claim_state") != "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH":
        raise ValueError("prior claim is not immutably consumed")
    if regression.get("passed") is not True or not all(regression.get("checks", {}).values()):
        raise ValueError("source-free import regression did not pass")
    if regression.get("implementation_sha256") != _sha_file(ROOT / "scripts/run_f1_pgrr_v1_source_free_import_regression.py"):
        raise ValueError("regression implementation changed after execution")
    code_sha = {relative: _sha_file(ROOT / relative) for relative in CODE_PATHS}
    corrected_source_aggregate = sha_json(code_sha)
    prior_runner_sha = str(prior_lock["code_sha256"]["scripts/run_f1_pgrr_v1_causal_preflight.py"])
    corrected_runner_sha = code_sha["scripts/run_f1_pgrr_v1_causal_preflight.py"]
    if prior_runner_sha == corrected_runner_sha:
        raise ValueError("authorized runner correction is absent")
    lock = {
        "schema_version": 2,
        "profile": ATTEMPT_ID,
        "authorization": {
            "ruling": "AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2",
            "professor_response_relative": RULING_RELATIVE,
            "professor_response_sha256": _sha_file(ROOT / RULING_RELATIVE),
            "execution_count": "EXACTLY_ONE_FRESH_CORRECTION_ATTEMPT",
            "claim_consumption": "BEFORE_FIRST_WORKER_LAUNCH",
        },
        "prior_attempt": {
            "classification": "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP",
            "architecture_scientifically_evaluated": False,
            "lock_relative": PRIOR_LOCK_RELATIVE,
            "lock_file_sha256": _sha_file(ROOT / PRIOR_LOCK_RELATIVE),
            "lock_self_sha256": prior_lock["execution_lock_sha256"],
            "consumed_claim_relative": PRIOR_CLAIM_RELATIVE,
            "consumed_claim_sha256": _sha_file(ROOT / PRIOR_CLAIM_RELATIVE),
            "result_relative": PRIOR_RESULT_RELATIVE,
            "result_sha256": _sha_file(ROOT / PRIOR_RESULT_RELATIVE),
            "producer_process_relative": PRIOR_PROCESS_RELATIVE,
            "producer_process_sha256": _sha_file(ROOT / PRIOR_PROCESS_RELATIVE),
            "visual_critique_relative": PRIOR_VISUAL_RELATIVE,
            "visual_critique_sha256": _sha_file(ROOT / PRIOR_VISUAL_RELATIVE),
            "immutable_no_overwrite": True,
        },
        "authorized_correction": {
            "old_line": "python = Path(sys.executable).resolve()",
            "new_line": "python = Path(sys.executable)",
            "prior_runner_sha256": prior_runner_sha,
            "corrected_runner_sha256": corrected_runner_sha,
            "additional_scientific_path_changes": False,
            "fallback_interpreter_search": False,
            "retry_authorized": False,
        },
        "source_free_regression": {
            "relative": REGRESSION_RELATIVE,
            "file_sha256": _sha_file(ROOT / REGRESSION_RELATIVE),
            "record": regression,
        },
        "configuration": prior_lock["configuration"],
        "configuration_sha256": prior_lock["configuration_sha256"],
        "resource_proof": allocation_proof(),
        "raw_source_relative": prior_lock["raw_source_relative"],
        "code_sha256": code_sha,
        "corrected_source_aggregate_sha256": corrected_source_aggregate,
        "input_sha256": prior_lock["input_sha256"],
        "unchanged_original_bindings_sha256": sha_json({
            "configuration": prior_lock["configuration"],
            "configuration_sha256": prior_lock["configuration_sha256"],
            "raw_source_relative": prior_lock["raw_source_relative"],
            "input_sha256": prior_lock["input_sha256"],
            "resource_proof": prior_lock["resource_proof"],
            "safety": prior_lock["safety"],
            "failure_policy": prior_lock["failure_policy"],
        }),
        "safety": prior_lock["safety"],
        "failure_policy": prior_lock["failure_policy"],
        "correction_v2_terminal_rule": "NO_RERUN_NO_REPLACEMENT_NO_SECOND_CORRECTION_LANE",
    }
    if lock["configuration_sha256"] != CONFIG_SHA256:
        raise ValueError("PGRR configuration changed")
    lock["execution_lock_sha256"] = sha_json(lock)
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("correction-V2 lock builder accepts no arguments")
    print(canonical_json_bytes(build_lock()).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
