#!/usr/bin/env python3
"""Audit the V34 post-return serializer waiver without invoking the detector."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research/f1-v34-opgt-v1"
ORIGINAL_PRECOMMIT = OUT / "precommit_manifest.json"
WAIVER_PRECOMMIT = OUT / "waiver_precommit_manifest.json"
COUNTER = OUT / "invocations.jsonl"
INCIDENT = OUT / "terminal_stop.json"
WAIVER_RECORD = OUT / "exact_rerun_waiver_record.json"
FIXTURE_MANIFEST = OUT / "fixture_manifest.json"
VISUAL_MANIFEST = OUT / "preinvoke_visual_manifest.json"
ENGINE = ROOT / "backend/f1_ordered_projective_gauss_transport.py"
RUNNER = ROOT / "scripts/run_f1_v34_opgt_one_shot.py"
NORMALIZER = ROOT / "backend/f1_v34_artifact_scalar_normalizer.py"
TEST = ROOT / "tests/test_f1_v34_artifact_scalar_normalizer.py"
RULING = ROOT / "docs/research/2026-08-14_gpt56_sol_pro_v34_exact_rerun_waiver_verbatim.txt"
AUDIT = Path(__file__).resolve()

ORIGINAL_PRECOMMIT_SHA256 = "471ccf818ee212cfa0c89c2fa1f3d17f3add8cbc388298cce27eda2bf480b1e2"
ORIGINAL_COUNTER_SHA256 = "cc224d2c975fffaae3905a4272800b2317e8fd9b05b5ecffa7c60b0db78a8cd9"
ORIGINAL_ENGINE_SHA256 = "5874f59f6d6e98dfbcfe9d5113bf5e1876ac5c228b86deda743d943a68c06124"
ORIGINAL_RUNNER_SHA256 = "4a9c928ce6230f611e603ec4f6a850b0f0c365bc30f67a7ad205c6d04ae49a97"
WAIVER_ID = "V34_OPGT_EXACT_RERUN_WAIVER_2026_08_14"
FIXTURE_ID = "SF01_FOLD60_MESH_C2"
REPLACEMENT_COMMAND = (
    ".venv/bin/python scripts/run_f1_v34_opgt_one_shot.py "
    "replace-aborted --fixture SF01_FOLD60_MESH_C2"
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_manifest(value: dict[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "manifest_payload_sha256"}
    value["manifest_payload_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    WAIVER_PRECOMMIT.write_bytes(canonical(value))


def invocation_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in COUNTER.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    original = load(ORIGINAL_PRECOMMIT)
    waiver = load(WAIVER_RECORD)
    incident = load(INCIDENT)
    fixture_manifest = load(FIXTURE_MANIFEST)
    rows_before = invocation_rows()

    check(
        "original_mechanism_precommit_preserved",
        sha(ORIGINAL_PRECOMMIT) == ORIGINAL_PRECOMMIT_SHA256
        and original.get("passed") is True
        and original.get("token") == "V34_OPGT_MECHANISM_LOCKED",
        sha(ORIGINAL_PRECOMMIT),
    )
    check("frozen_engine_byte_identical", sha(ENGINE) == ORIGINAL_ENGINE_SHA256, sha(ENGINE))
    check(
        "original_invocation_checkpoint_preserved",
        sha(COUNTER) == ORIGINAL_COUNTER_SHA256 and len(rows_before) == 1,
        {"sha256": sha(COUNTER), "count": len(rows_before)},
    )
    first = rows_before[0] if rows_before else {}
    check(
        "aborted_physical_one_identity",
        first.get("fixture_id") == FIXTURE_ID and first.get("ordinal") == 1,
        first,
    )

    frozen_drift: list[dict[str, str]] = []
    for relative, expected in original["frozen_hashes"].items():
        actual = sha(ROOT / relative)
        if actual != expected:
            frozen_drift.append({"path": relative, "original_sha256": expected, "current_sha256": actual})
    check(
        "only_original_frozen_harness_runner_differs",
        len(frozen_drift) == 1
        and frozen_drift[0]["path"] == "scripts/run_f1_v34_opgt_one_shot.py"
        and frozen_drift[0]["original_sha256"] == ORIGINAL_RUNNER_SHA256,
        frozen_drift,
    )

    descriptor_row = fixture_manifest["rows"][0]
    descriptor_path = ROOT / descriptor_row["descriptor_path"]
    descriptor = load(descriptor_path)
    source_path = ROOT / descriptor["source_payload_path"]
    binding_path = ROOT / descriptor["triangle_binding_path"]
    check(
        "replacement_fixture_and_inputs_byte_identical",
        descriptor_row["fixture_id"] == FIXTURE_ID
        and sha(descriptor_path) == waiver["unchanged_fixture_descriptor_sha256"]
        and sha(source_path) == waiver["unchanged_source_payload_sha256"]
        and sha(binding_path) == waiver["unchanged_triangle_binding_sha256"],
        {
            "descriptor_sha256": sha(descriptor_path),
            "source_payload_sha256": sha(source_path),
            "triangle_binding_sha256": sha(binding_path),
        },
    )
    check(
        "configuration_and_sequence_unchanged",
        sha(ROOT / "research/f1_v34_opgt_v1_lock.json")
        == original["frozen_hashes"]["research/f1_v34_opgt_v1_lock.json"]
        and sha(FIXTURE_MANIFEST)
        == original["frozen_hashes"]["output/research/f1-v34-opgt-v1/fixture_manifest.json"]
        and sha(VISUAL_MANIFEST)
        == original["frozen_hashes"]["output/research/f1-v34-opgt-v1/preinvoke_visual_manifest.json"]
        and fixture_manifest["ordered_fixture_ids"][0] == FIXTURE_ID,
        {
            "lock_sha256": sha(ROOT / "research/f1_v34_opgt_v1_lock.json"),
            "fixture_manifest_sha256": sha(FIXTURE_MANIFEST),
            "visual_manifest_sha256": sha(VISUAL_MANIFEST),
        },
    )

    forbidden_artifacts = [
        OUT / "source_free_results" / f"001_{FIXTURE_ID}.json",
        OUT / "source_free_results" / f"001_{FIXTURE_ID}.cache",
        OUT / "source_free_results" / f"001_{FIXTURE_ID}.record.json",
        OUT / "postinvoke_visuals" / f"001_{FIXTURE_ID}_post_4096.png",
    ]
    check(
        "aborted_call_has_no_empirical_artifacts",
        not any(path.exists() for path in forbidden_artifacts),
        [str(path.relative_to(ROOT)) for path in forbidden_artifacts if path.exists()],
    )
    check(
        "incident_declares_outcome_unobserved",
        incident.get("mechanism_outcome_known") is False
        and incident.get("public_result_preserved") is False
        and incident.get("post_visual_preserved") is False
        and waiver.get("mechanism_outcome_from_aborted_call_inspected") is False
        and waiver.get("mechanism_outcome_from_aborted_call_recovered") is False,
        {
            "incident_sha256": sha(INCIDENT),
            "waiver_record_sha256": sha(WAIVER_RECORD),
        },
    )
    check(
        "pro_ruling_preserved_verbatim",
        sha(RULING) == waiver["ruling_sha256"] == "bb2d050bb3c3f6aad87285707417f2bcee050b65eea439e0be8b58650f1d1987",
        sha(RULING),
    )

    engine_source = ENGINE.read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")
    test_source = TEST.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    detector_calls = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "select_unified_v34"
    ]
    normalizer_calls = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_numpy_scalars"
    ]
    check(
        "normalizer_is_post_return_harness_only",
        len(detector_calls) == 1
        and len(normalizer_calls) == 1
        and normalizer_calls[0].lineno > detector_calls[0].lineno
        and "f1_v34_artifact_scalar_normalizer" not in engine_source
        and 'if key != "_ephemeral"' in runner_source,
        {
            "detector_call_lines": [node.lineno for node in detector_calls],
            "normalizer_call_lines": [node.lineno for node in normalizer_calls],
        },
    )
    check(
        "serializer_tests_cannot_invoke_detector",
        "select_unified_v34" not in test_source
        and "f1_ordered_projective_gauss_transport" not in test_source,
        str(TEST.relative_to(ROOT)),
    )

    test_run = subprocess.run(
        [str(ROOT / ".venv/bin/pytest"), "-q", str(TEST.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    rows_after = invocation_rows()
    check("serializer_only_tests", test_run.returncode == 0, (test_run.stdout + test_run.stderr).strip())
    check(
        "audit_kept_detector_invocation_count_one",
        rows_after == rows_before and sha(COUNTER) == ORIGINAL_COUNTER_SHA256,
        {"before": len(rows_before), "after": len(rows_after), "sha256": sha(COUNTER)},
    )

    harness_paths = [RUNNER, NORMALIZER, TEST, WAIVER_RECORD, RULING, INCIDENT, AUDIT]
    harness_hashes = {str(path.relative_to(ROOT)): sha(path) for path in harness_paths}
    environment = {
        "command": REPLACEMENT_COMMAND,
        "cwd": str(ROOT),
        "python_executable": str(ROOT / ".venv/bin/python"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "worker_count": 1,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "platform": platform.platform(),
    }
    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "kind": "v34_opgt_exact_rerun_waiver_precommit",
        "passed": passed,
        "token": "V34_OPGT_EXACT_RERUN_WAIVER_LOCKED" if passed else "V34_OPGT_EXACT_RERUN_WAIVER_FAILED",
        "waiver_id": WAIVER_ID,
        "original_mechanism_token": "V34_OPGT_MECHANISM_LOCKED",
        "original_precommit_sha256": ORIGINAL_PRECOMMIT_SHA256,
        "original_invocation_counter_sha256": ORIGINAL_COUNTER_SHA256,
        "original_runner_sha256": ORIGINAL_RUNNER_SHA256,
        "unchanged_engine_sha256": ORIGINAL_ENGINE_SHA256,
        "authorized_physical_invocation": 2,
        "authorized_logical_ordinal": 1,
        "replacement_of_physical_invocation": 1,
        "fixture_id": FIXTURE_ID,
        "environment_manifest": environment,
        "waiver_harness_hashes": harness_hashes,
        "checks": checks,
    }
    write_manifest(result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
