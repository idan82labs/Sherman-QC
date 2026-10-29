#!/usr/bin/env python3
"""Source-free gate for the authorized PGRR component-ID correction V3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ID = "PGRR_V1_COMPONENT_ID_CORRECTION_V3_SOURCE_FREE_REGRESSION_V1"
PROHIBITED_PATH_TOKENS = (
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply",
    "output/research/f1-gold27-20260803/candidate_pool_transition_v2_probe/C006.json",
    "output/research/f1-gold27-20260803/local_plane_fragment_probe_v2/C006.json",
    "output/research/f1-gold27-20260803/v2_complete_same_surface_edge_probe_v1/C006.json",
    "output/research/f1-gold27-20260803/v2_cross_parent_rung_mapping_preflight_v1.json",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


CHILD_PROGRAM = r'''
import json
from pathlib import Path
import sys

PROHIBITED = (
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply",
    "output/research/f1-gold27-20260803/candidate_pool_transition_v2_probe/C006.json",
    "output/research/f1-gold27-20260803/local_plane_fragment_probe_v2/C006.json",
    "output/research/f1-gold27-20260803/v2_complete_same_surface_edge_probe_v1/C006.json",
    "output/research/f1-gold27-20260803/v2_cross_parent_rung_mapping_preflight_v1.json",
)
accesses = []

def audit(event, args):
    if event != "open" or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes)):
        return
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    normalized = text.replace("\\", "/")
    if any(token in normalized for token in PROHIBITED):
        accesses.append(normalized)
        raise RuntimeError("prohibited scientific input access during component-ID regression")

sys.addaudithook(audit)
import numpy as np
from research.f1_pgrr_v1.geometry import _component_id
from research.f1_pgrr_v1.independent_validator import _min_id

rows = [
    bytes([0xFF] * 32),
    bytes([0x00] * 32),
    bytes([0x01] + [0x00] * 31),
    bytes([0x00] * 16 + [0x01] + [0x00] * 15),
    bytes([0x00] * 31 + [0x01]),
    bytes.fromhex("10" * 12 + "00" + "10" * 19),
    bytes.fromhex("10" * 12 + "01" + "00" * 19),
    bytes.fromhex("10" * 31 + "00"),
    bytes.fromhex("10" * 31 + "01"),
    bytes.fromhex("00" * 31 + "00"),
    bytes.fromhex("00" * 31 + "00"),
    bytes.fromhex("7f" * 32),
]
permutations = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0),
    (2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11),
    (3, 0, 9, 6, 1, 10, 4, 7, 2, 11, 8, 5),
    (4, 8, 0, 5, 9, 1, 6, 10, 2, 7, 11, 3),
    (5, 1, 9, 2, 10, 3, 11, 4, 0, 6, 8, 7),
    (6, 11, 4, 9, 2, 7, 0, 5, 10, 3, 8, 1),
    (7, 3, 10, 6, 2, 11, 0, 8, 4, 9, 5, 1),
    (8, 5, 2, 11, 7, 4, 1, 10, 6, 3, 0, 9),
]
oracle = min(rows).hex()
cases = []
all_correct = True
for case_id, permutation in enumerate(permutations):
    permuted = [rows[index] for index in permutation]
    point_ids = np.asarray([list(value) for value in permuted], dtype=np.uint8)
    indices = np.arange(len(point_ids), dtype=np.uint32)
    producer = _component_id(indices, point_ids)
    validator = _min_id(indices, point_ids)
    correct = producer == oracle and validator == oracle and producer == validator and len(producer) == 64
    all_correct = all_correct and correct
    cases.append({
        "case_id": case_id,
        "permutation": list(permutation),
        "producer": producer,
        "validator": validator,
        "oracle": oracle,
        "correct": correct,
    })

empty = np.asarray([], dtype=np.uint32)
empty_ids = np.empty((0, 32), dtype=np.uint8)
empty_results = {}
for name, function in (("producer", _component_id), ("validator", _min_id)):
    try:
        function(empty, empty_ids)
    except Exception as error:
        empty_results[name] = {"raised": True, "exception_type": type(error).__name__}
    else:
        empty_results[name] = {"raised": False, "exception_type": ""}

record = {
    "child_sys_executable": sys.executable,
    "child_sys_prefix": sys.prefix,
    "child_sys_base_prefix": sys.base_prefix,
    "numpy_version": np.__version__,
    "numpy_module_origin": str(Path(np.__file__).resolve()),
    "geometry_module_origin": str(Path(sys.modules[_component_id.__module__].__file__).resolve()),
    "validator_module_origin": str(Path(sys.modules[_min_id.__module__].__file__).resolve()),
    "oracle": oracle,
    "cases": cases,
    "all_permutations_correct": all_correct,
    "permutation_count": len(permutations),
    "canonical_and_reversed_included": permutations[0] == tuple(range(12)) and permutations[1] == tuple(reversed(range(12))),
    "duplicate_minimum_present": rows.count(min(rows)) == 3,
    "empty_results": empty_results,
    "raw_or_target_accesses": accesses,
}
print(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
'''.strip()


def run_regression() -> dict[str, Any]:
    python = Path(sys.executable)
    command = [str(python), "-c", CHILD_PROGRAM]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    child: dict[str, Any] = {}
    parse_error = ""
    try:
        parsed = json.loads(completed.stdout.decode("utf-8"))
        if isinstance(parsed, dict):
            child = parsed
        else:
            parse_error = "child stdout was not a JSON object"
    except Exception as error:
        parse_error = f"{type(error).__name__}: {error}"

    parent_prefix = str(Path(sys.prefix))
    parent_base_prefix = str(Path(sys.base_prefix))
    numpy_origin = str(child.get("numpy_module_origin") or "")
    try:
        numpy_within_parent_prefix = Path(numpy_origin).resolve().is_relative_to(Path(parent_prefix).resolve())
    except (OSError, ValueError):
        numpy_within_parent_prefix = False
    geometry_source = (ROOT / "research/f1_pgrr_v1/geometry.py").read_text(encoding="utf-8")
    validator_source = (ROOT / "research/f1_pgrr_v1/independent_validator.py").read_text(encoding="utf-8")
    checks = {
        "exact_unresolved_launcher_used": command[0] == str(Path(sys.executable)),
        "launcher_not_canonicalized": command[0] != str(Path(sys.executable).resolve()),
        "child_return_code_zero": completed.returncode == 0,
        "child_record_parsed": not parse_error and bool(child),
        "same_virtual_environment": (
            child.get("child_sys_executable") == str(Path(sys.executable))
            and child.get("child_sys_prefix") == parent_prefix
            and child.get("child_sys_base_prefix") == parent_base_prefix
            and parent_prefix != parent_base_prefix
            and numpy_within_parent_prefix
        ),
        "producer_exact_authorized_primitive": 'local = int(np.argsort(ids, kind="stable")[0])' in geometry_source,
        "validator_exact_authorized_primitive": (
            "local = min(" in validator_source
            and "key=lambda offset: bytes(ids[indices[offset]])," in validator_source
        ),
        "oracle_agreement_all_permutations": child.get("all_permutations_correct") is True,
        "canonical_reverse_and_eight_fixed_permutations": (
            child.get("canonical_and_reversed_included") is True
            and child.get("permutation_count") == 10
        ),
        "byte_position_prefix_and_extreme_adversaries": child.get("oracle") == (b"\x00" * 32).hex(),
        "duplicate_minimum_value_invariant": child.get("duplicate_minimum_present") is True,
        "empty_producer_fails_closed": child.get("empty_results", {}).get("producer", {}).get("raised") is True,
        "empty_validator_fails_closed": child.get("empty_results", {}).get("validator", {}).get("raised") is True,
        "no_raw_target_gold_cad_benchmark_or_outcome_access": child.get("raw_or_target_accesses") == [],
        "stdout_is_single_json_record": completed.stdout.count(b"\n") == 1,
    }
    result = {
        "schema_version": 1,
        "regression_id": REGRESSION_ID,
        "scientific_evidence": False,
        "definition": {
            "oracle": "min(bytes(row)).hex()",
            "producer_function": "research.f1_pgrr_v1.geometry._component_id",
            "validator_function": "research.f1_pgrr_v1.independent_validator._min_id",
            "fixed_permutation_count": 10,
            "randomness": False,
            "prohibited_path_tokens": list(PROHIBITED_PATH_TOKENS),
        },
        "definition_sha256": _sha_bytes(_canonical_bytes({
            "child_program": CHILD_PROGRAM,
            "prohibited_path_tokens": PROHIBITED_PATH_TOKENS,
        })),
        "implementation_sha256": _sha_file(Path(__file__).resolve()),
        "corrected_implementation_sha256": {
            "producer_geometry": _sha_file(ROOT / "research/f1_pgrr_v1/geometry.py"),
            "independent_validator": _sha_file(ROOT / "research/f1_pgrr_v1/independent_validator.py"),
        },
        "command_record": {
            "argv": [command[0], "-c", "<CHILD_PROGRAM_BOUND_BY_DEFINITION_SHA256>"],
            "launch_argv0": command[0],
            "cwd": str(ROOT),
            "shell": False,
            "stdin": "DEVNULL",
        },
        "environment_identity": {
            "parent_sys_executable": str(Path(sys.executable)),
            "child_launch_argv0": command[0],
            "parent_sys_prefix": parent_prefix,
            "child_sys_prefix": child.get("child_sys_prefix"),
            "parent_sys_base_prefix": parent_base_prefix,
            "child_sys_base_prefix": child.get("child_sys_base_prefix"),
            "child_sys_executable": child.get("child_sys_executable"),
            "numpy_version": child.get("numpy_version"),
            "numpy_module_origin": numpy_origin,
            "geometry_module_origin": child.get("geometry_module_origin"),
            "validator_module_origin": child.get("validator_module_origin"),
        },
        "child_return_code": completed.returncode,
        "child_stdout_sha256": _sha_bytes(completed.stdout),
        "child_stderr_sha256": _sha_bytes(completed.stderr),
        "child_stdout_bytes": len(completed.stdout),
        "child_stderr_bytes": len(completed.stderr),
        "child_parse_error": parse_error,
        "checks": checks,
        "cases_sha256": _sha_bytes(_canonical_bytes(child.get("cases", []))),
        "empty_results": child.get("empty_results", {}),
        "passed": all(checks.values()),
    }
    result["result_sha256"] = _sha_bytes(_canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("component-ID regression accepts no arguments")
    result = run_regression()
    print(_canonical_bytes(result).decode("ascii"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
