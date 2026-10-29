#!/usr/bin/env python3
"""Execute the single authorized V73 source-free primary+replay lock."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATH = ROOT / "backend/f1_integral_transverse_covariance_growth_veto_v73.py"
FIXTURE_PATH = ROOT / "tests/fixtures/f1_v73_itcgv_v1_fixture_oracle.py"
TEST_PATH = ROOT / "tests/test_f1_integral_transverse_covariance_growth_veto_v73.py"
RUNNER_PATH = Path(__file__).resolve()
RULING_PATH = ROOT / "docs/research/2026-08-16_gpt56_pro_post_v72_successor_ruling_verbatim.md"
CONSTANTS_PATH = ROOT / "docs/research/artifacts/2026-08-20_v73_itcgv_v1_constants_manifest.json"
SCHEMA_PATH = ROOT / "docs/research/artifacts/2026-08-20_v73_itcgv_v1_serialization_schema.json"

EXPECTED_RULING_SHA256 = "739164fe9619d3913e332b60053d602f607722483afce738f5775cec877c7e10"
EXPECTED_CASES = tuple(f"G{index:02d}" for index in range(1, 33))
EXPECTED_TEST_COUNT = 32
BANNED_AST_NAMES = {"pi", "DEG", "radians", "sin", "cos", "tan"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.read_bytes() != payload:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("ATOMIC_BYTE_VERIFICATION_FAILURE")
    temporary.replace(path)


def _load_precommit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "implementation_sha256": _sha256(IMPLEMENTATION_PATH),
        "fixture_generator_sha256": _sha256(FIXTURE_PATH),
        "test_suite_sha256": _sha256(TEST_PATH),
        "lock_runner_sha256": _sha256(RUNNER_PATH),
        "ruling_sha256": _sha256(RULING_PATH),
        "constants_manifest_sha256": _sha256(CONSTANTS_PATH),
        "serialization_schema_sha256": _sha256(SCHEMA_PATH),
    }
    if value.get("profile") != "V73_ITCGV_V1_SOURCE_FREE_LOCK":
        raise RuntimeError("PRECOMMIT_PROFILE_MISMATCH")
    if value.get("frozen_hashes") != expected:
        raise RuntimeError("PRECOMMIT_HASH_MISMATCH")
    if expected["ruling_sha256"] != EXPECTED_RULING_SHA256:
        raise RuntimeError("RULING_HASH_MISMATCH")
    return value


def _validate_constants(precommit: dict[str, Any]) -> dict[str, Any]:
    constants = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))
    digest = _sha256_bytes(
        json.dumps(constants, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    expected = precommit.get("constants_payload_sha256")
    if digest != expected:
        raise RuntimeError("CONSTANTS_PAYLOAD_HASH_MISMATCH")
    literal_expectations = {
        "architecture": "V73_INTEGRAL_TRANSVERSE_COVARIANCE_GROWTH_VETO_V1",
        "axial_half_width": "0x1.0000000000000p+2",
        "coupling_max": "0x1.0000000000000p-4",
        "flat_max": "0x1.0000000000000p-13",
        "growth_denominator": 8,
        "growth_numerator": 9,
        "inner_min": "0x1.0000000000000p-24",
        "length_floor": "0x1.4000000000000p+5",
        "minimum_inner_support": 24,
        "minimum_outer_half_support": 24,
        "minimum_outer_support": 96,
        "normalized_spread_factor": 2,
        "outer_max": "0x1.0000000000000p-1",
        "psd_epsilon": "0x1.0000000000000p-40",
        "radii": [8, 12, 18, 27, 40, 60, 90, 135],
        "scale_count": 8,
        "station_count": 9,
    }
    if constants != literal_expectations:
        raise RuntimeError("CONSTANTS_LITERAL_MISMATCH")
    return constants


def _validate_production_ast() -> dict[str, Any]:
    source = IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(IMPLEMENTATION_PATH))
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_AST_NAMES:
            violations.append({"kind": "name", "line": node.lineno, "value": node.id})
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_AST_NAMES:
            violations.append({"kind": "attribute", "line": node.lineno, "value": node.attr})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node).lower()
            if "v72" in rendered or "casl" in rendered:
                violations.append({"kind": "forbidden_import", "line": node.lineno, "value": rendered})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "0x" in lowered and ("degree" in lowered or "radian" in lowered):
                violations.append({"kind": "angle_hex_literal", "line": node.lineno, "value": node.value})
    if violations:
        raise RuntimeError(f"PRODUCTION_AST_VIOLATION:{violations}")
    return {
        "banned_names": sorted(BANNED_AST_NAMES),
        "production_ast_clean": True,
        "v72_or_casl_imports": 0,
    }


def _validate_fixture_independence() -> dict[str, Any]:
    source = FIXTURE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FIXTURE_PATH))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node).lower()
            if "f1_integral_transverse_covariance_growth_veto_v73" in rendered or "backend" in rendered:
                forbidden.append({"line": node.lineno, "value": rendered})
    if forbidden:
        raise RuntimeError(f"FIXTURE_INDEPENDENCE_VIOLATION:{forbidden}")
    return {
        "fixture_imports_production": False,
        "fixture_oracle_independent": True,
    }


def _validate_environment(precommit: dict[str, Any]) -> dict[str, str]:
    import numpy
    import pytest

    observed = {
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "pytest": pytest.__version__,
    }
    if precommit.get("execution_environment") != observed:
        raise RuntimeError("EXECUTION_ENVIRONMENT_MISMATCH")
    return observed


def _junit_result(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    records = []
    failures = errors = skipped = 0
    for testcase in root.iter("testcase"):
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
        "errors": errors,
        "failures": failures,
        "ordered_cases": records,
        "passed": len(records) - failures - errors - skipped,
        "skipped": skipped,
        "tests": len(records),
    }


def _case_artifacts(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records = []
    payloads = {}
    for identifier in EXPECTED_CASES:
        path = root / f"{identifier}.json"
        if not path.is_file():
            continue
        payload = path.read_bytes()
        decoded = json.loads(payload)
        if _canonical_bytes(decoded) != payload:
            raise RuntimeError(f"NONCANONICAL_CASE_ARTIFACT:{identifier}")
        records.append(
            {
                "bytes": len(payload),
                "case": identifier,
                "sha256": _sha256_bytes(payload),
            }
        )
        payloads[identifier] = payload
    return records, payloads


def _run_suite(label: str, output_root: Path) -> dict[str, Any]:
    run_root = output_root / label
    case_root = run_root / "cases"
    case_root.mkdir(parents=True, exist_ok=False)
    junit_path = run_root / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(TEST_PATH.relative_to(ROOT)),
        f"--junitxml={junit_path}",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "V73_LOCK_ARTIFACT_ROOT": str(case_root),
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=3_600,
    )
    _atomic_write(run_root / "stdout.txt", completed.stdout.encode("utf-8"))
    _atomic_write(run_root / "stderr.txt", completed.stderr.encode("utf-8"))
    junit = _junit_result(junit_path)
    case_manifest, case_payloads = _case_artifacts(case_root)
    canonical_result = {
        "case_artifacts": case_manifest,
        "pytest": junit,
    }
    canonical = _canonical_bytes(canonical_result)
    _atomic_write(run_root / "canonical.json", canonical)
    return {
        "canonical": canonical,
        "canonical_sha256": _sha256_bytes(canonical),
        "case_payloads": case_payloads,
        "exit_code": completed.returncode,
        "result": canonical_result,
    }


def execute(output_root: Path, precommit_path: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite lock output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    precommit = _load_precommit(precommit_path)
    constants = _validate_constants(precommit)
    ast_audit = _validate_production_ast()
    fixture_audit = _validate_fixture_independence()
    execution_environment = _validate_environment(precommit)
    _atomic_write(output_root / "precommit.snapshot.json", _canonical_bytes(precommit))

    primary = _run_suite("primary", output_root)
    replay = _run_suite("replay", output_root)
    primary_clean = (
        primary["exit_code"] == 0
        and primary["result"]["pytest"]["tests"] == EXPECTED_TEST_COUNT
        and primary["result"]["pytest"]["passed"] == EXPECTED_TEST_COUNT
        and primary["result"]["pytest"]["failures"] == 0
        and primary["result"]["pytest"]["errors"] == 0
        and primary["result"]["pytest"]["skipped"] == 0
        and [record["case"] for record in primary["result"]["case_artifacts"]] == list(EXPECTED_CASES)
    )
    replay_clean = (
        replay["exit_code"] == 0
        and replay["result"]["pytest"]["tests"] == EXPECTED_TEST_COUNT
        and replay["result"]["pytest"]["passed"] == EXPECTED_TEST_COUNT
        and replay["result"]["pytest"]["failures"] == 0
        and replay["result"]["pytest"]["errors"] == 0
        and replay["result"]["pytest"]["skipped"] == 0
        and [record["case"] for record in replay["result"]["case_artifacts"]] == list(EXPECTED_CASES)
    )
    canonical_identical = primary["canonical"] == replay["canonical"]
    case_bytes_identical = primary["case_payloads"] == replay["case_payloads"]
    gate_passed = primary_clean and replay_clean and canonical_identical and case_bytes_identical

    summary = {
        "ast_audit": ast_audit,
        "constants": constants,
        "execution_environment": execution_environment,
        "fixture_audit": fixture_audit,
        "internal_replay": {
            "canonical_sha256": replay["canonical_sha256"],
            "exit_code": replay["exit_code"],
            "result": replay["result"],
        },
        "primary": {
            "canonical_sha256": primary["canonical_sha256"],
            "exit_code": primary["exit_code"],
            "result": primary["result"],
        },
        "primary_replay_canonical_byte_identical": canonical_identical,
        "primary_replay_case_artifacts_byte_identical": case_bytes_identical,
        "precommit": precommit,
        "profile": "V73_ITCGV_V1_SOURCE_FREE_LOCK",
        "rerun_or_rescue_authorized": False,
        "reserved_physical_scan_status": "SEALED",
        "single_top_level_invocation": True,
        "source_free_lock_passed": gate_passed,
        "terminal_disposition": "V73_LOCKED" if gate_passed else "V73_TERMINAL_RETIRE",
        "u01_u04_authorized": gate_passed,
    }
    _atomic_write(output_root / "summary.json", _canonical_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--precommit", required=True, type=Path)
    arguments = parser.parse_args()
    summary = execute(arguments.output_root.resolve(), arguments.precommit.resolve())
    return 0 if summary["source_free_lock_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
