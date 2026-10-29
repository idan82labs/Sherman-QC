#!/usr/bin/env python3
"""Execute the single authorized V72 A1 primary lock and internal replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests/test_f1_candidate_centered_annular_secant_law_v72.py"
IMPLEMENTATION_PATH = ROOT / "backend/f1_candidate_centered_annular_secant_law_v72.py"
AMENDMENT_PATH = ROOT / "docs/research/2026-08-16_gpt56_pro_post_v72_lock_correction_ruling_verbatim.md"
ORIGINAL_RULING_PATH = ROOT / "docs/research/2026-08-16_gpt56_pro_post_v71_successor_ruling_verbatim.md"
ORIGINAL_FIXTURE_PATH = ROOT / "docs/research/artifacts/2026-08-16_v72_original_failed_lock_fixture.py"
ORIGINAL_RUNNER_PATH = ROOT / "scripts/run_f1_v72_casl_u01_u04_mechanical_gate.py"
FAILED_JUNIT_PATH = ROOT / "output/research/f1-v72-casl-source-free-lock-20260816.junit.xml"

EXPECTED_IMMUTABLE_HASHES = {
    "implementation_sha256": "b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a",
    "original_ruling_sha256": "31ea25b5c572f0611620aedc9a95aa212a77bf8b545a418a3b63429513386b8b",
    "original_fixture_sha256": "8d7ddece77b4f56b17883e8b8f0ae436ce5a38bc301572bdb7ec9e33e8233147",
    "original_runner_sha256": "254a465ceba4bf493dce963e2e35941d7eab970338d1d192ce33a84bfdc4848f",
    "failed_junit_sha256": "e609c9f6221448e7017d667a49470be8cd7ca41a43b1b9adc2185fc7fbe4cde8",
    "amendment_sha256": "d7ec349d5aec67c9e8a1b2e4a73ce3be82b7275c487b598d2ac409963712df5a",
}

EXPECTED_SOURCE_RECORD_COUNTS = {
    "G10": 199_826,
    "G15": 199_485,
    "G16": 199_826,
    "refinement_coarse": 99_913,
    "refinement_refined": 199_485,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _junit_result(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    testcases = list(root.iter("testcase"))
    records = []
    failures = errors = skipped = 0
    for testcase in testcases:
        failure = testcase.find("failure")
        error = testcase.find("error")
        skip = testcase.find("skipped")
        if failure is not None:
            outcome = "failed"
            failures += 1
        elif error is not None:
            outcome = "error"
            errors += 1
        elif skip is not None:
            outcome = "skipped"
            skipped += 1
        else:
            outcome = "passed"
        records.append(
            {
                "classname": testcase.attrib.get("classname", ""),
                "name": testcase.attrib.get("name", ""),
                "outcome": outcome,
            }
        )
    return {
        "schema_version": 1,
        "ordered_cases": records,
        "tests": len(testcases),
        "passed": len(testcases) - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _run_suite(label: str, output_root: Path) -> dict[str, Any]:
    junit = output_root / f"{label}.junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(TEST_PATH.relative_to(ROOT)),
        f"--junitxml={junit}",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    _atomic_write(output_root / f"{label}.stdout.txt", completed.stdout.encode("utf-8"))
    _atomic_write(output_root / f"{label}.stderr.txt", completed.stderr.encode("utf-8"))
    result = _junit_result(junit)
    canonical = _canonical_bytes(result)
    _atomic_write(output_root / f"{label}.canonical.json", canonical)
    return {
        "exit_code": completed.returncode,
        "junit_path": str(junit.relative_to(ROOT)),
        "junit_sha256": _sha256(junit),
        "canonical_path": str((output_root / f"{label}.canonical.json").relative_to(ROOT)),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_bytes": canonical,
        "result": result,
    }


def _validate_precommit(precommit_path: Path) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    expected = {
        "implementation_sha256": _sha256(IMPLEMENTATION_PATH),
        "fixture_sha256": _sha256(TEST_PATH),
        "lock_runner_sha256": _sha256(Path(__file__).resolve()),
        "u01_u04_runner_sha256": _sha256(ORIGINAL_RUNNER_PATH),
        "amendment_sha256": _sha256(AMENDMENT_PATH),
    }
    for key, value in expected.items():
        if precommit.get(key) != value:
            raise RuntimeError(f"PRECOMMIT_HASH_MISMATCH:{key}")
    return precommit


def execute(output_root: Path, precommit_path: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite lock output: {output_root}")

    immutable_paths = {
        "implementation_sha256": IMPLEMENTATION_PATH,
        "original_ruling_sha256": ORIGINAL_RULING_PATH,
        "original_fixture_sha256": ORIGINAL_FIXTURE_PATH,
        "original_runner_sha256": ORIGINAL_RUNNER_PATH,
        "failed_junit_sha256": FAILED_JUNIT_PATH,
        "amendment_sha256": AMENDMENT_PATH,
    }
    observed_immutable = {key: _sha256(path) for key, path in immutable_paths.items()}
    if observed_immutable != EXPECTED_IMMUTABLE_HASHES:
        raise RuntimeError("IMMUTABLE_EVIDENCE_HASH_MISMATCH")
    precommit = _validate_precommit(precommit_path)

    output_root.mkdir(parents=True, exist_ok=True)
    primary = _run_suite("primary", output_root)
    replay = _run_suite("replay", output_root)
    byte_identical = primary["canonical_bytes"] == replay["canonical_bytes"]
    primary_clean = primary["result"] == {
        "schema_version": 1,
        "ordered_cases": primary["result"]["ordered_cases"],
        "tests": 43,
        "passed": 43,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    replay_clean = replay["result"] == {
        "schema_version": 1,
        "ordered_cases": replay["result"]["ordered_cases"],
        "tests": 43,
        "passed": 43,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    gate_passed = (
        primary["exit_code"] == 0
        and replay["exit_code"] == 0
        and primary_clean
        and replay_clean
        and byte_identical
    )
    summary = {
        "schema_version": 1,
        "profile": "V72_CASL_SOURCE_FREE_LOCK_AMENDMENT_A1",
        "single_top_level_invocation": True,
        "primary": {key: value for key, value in primary.items() if key != "canonical_bytes"},
        "internal_replay": {key: value for key, value in replay.items() if key != "canonical_bytes"},
        "primary_replay_canonical_byte_identical": byte_identical,
        "expected_source_record_counts": EXPECTED_SOURCE_RECORD_COUNTS,
        "precommit": precommit,
        "immutable_evidence": observed_immutable,
        "implementation_byte_identical": observed_immutable["implementation_sha256"]
        == EXPECTED_IMMUTABLE_HASHES["implementation_sha256"],
        "source_free_lock_passed": gate_passed,
        "terminal_disposition": "V72_LOCKED" if gate_passed else "V72_CASL_TERMINAL_RETIRE",
        "u01_u04_authorized": gate_passed,
        "native_visual_gate_opened": False,
        "reserved_physical_scan_status": "SEALED",
        "rerun_or_rescue_authorized": False,
    }
    payload = _canonical_bytes(summary)
    _atomic_write(output_root / "summary.json", payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    arguments = parser.parse_args()
    summary = execute(arguments.output_root.resolve(), arguments.precommit.resolve())
    return 0 if summary["source_free_lock_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
