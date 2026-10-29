#!/usr/bin/env python3
"""Verify the exact unrevolved PGRR worker interpreter before claim creation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ID = "PGRR_V1_SOURCE_FREE_WORKER_IMPORT_REGRESSION_V1"
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
import hashlib
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
        raise RuntimeError("prohibited scientific input access during import regression")

sys.addaudithook(audit)
import scripts.run_f1_pgrr_v1_candidate_worker as worker
import numpy
import research.f1_pgrr_v1.geometry as geometry

record = {
    "child_sys_executable": sys.executable,
    "child_sys_prefix": sys.prefix,
    "child_sys_base_prefix": sys.base_prefix,
    "numpy_module_origin": str(Path(numpy.__file__).resolve()),
    "worker_module_origin": str(Path(worker.__file__).resolve()),
    "geometry_module_origin": str(Path(geometry.__file__).resolve()),
    "worker_imported": "scripts.run_f1_pgrr_v1_candidate_worker" in sys.modules,
    "geometry_imported": "research.f1_pgrr_v1.geometry" in sys.modules,
    "numpy_imported": "numpy" in sys.modules,
    "worker_main_called": False,
    "raw_or_target_accesses": accesses,
    "scientific_arguments_supplied": False,
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
    stdout_sha = _sha_bytes(completed.stdout)
    stderr_sha = _sha_bytes(completed.stderr)
    child: dict[str, Any] = {}
    parse_error = ""
    try:
        parsed = json.loads(completed.stdout.decode("utf-8"))
        if isinstance(parsed, dict):
            child = parsed
        else:
            parse_error = "child stdout was not a JSON object"
    except Exception as error:  # fail closed and retain only the error class/message
        parse_error = f"{type(error).__name__}: {error}"

    parent_prefix = str(Path(sys.prefix))
    parent_base_prefix = str(Path(sys.base_prefix))
    numpy_origin = str(child.get("numpy_module_origin") or "")
    try:
        numpy_within_parent_prefix = Path(numpy_origin).resolve().is_relative_to(Path(parent_prefix).resolve())
    except (OSError, ValueError):
        numpy_within_parent_prefix = False

    checks = {
        "exact_unresolved_launcher_used": command[0] == str(Path(sys.executable)),
        "launcher_not_canonicalized": command[0] != str(Path(sys.executable).resolve()),
        "child_return_code_zero": completed.returncode == 0,
        "child_record_parsed": not parse_error and bool(child),
        "child_executable_matches_parent": child.get("child_sys_executable") == str(Path(sys.executable)),
        "child_prefix_matches_parent": child.get("child_sys_prefix") == parent_prefix,
        "child_base_prefix_matches_parent": child.get("child_sys_base_prefix") == parent_base_prefix,
        "parent_is_virtual_environment": parent_prefix != parent_base_prefix,
        "numpy_origin_within_parent_prefix": numpy_within_parent_prefix,
        "worker_imported": child.get("worker_imported") is True,
        "geometry_imported_transitively": child.get("geometry_imported") is True,
        "numpy_imported_transitively": child.get("numpy_imported") is True,
        "worker_main_not_called": child.get("worker_main_called") is False,
        "no_scientific_arguments_supplied": child.get("scientific_arguments_supplied") is False,
        "no_raw_or_target_access": child.get("raw_or_target_accesses") == [],
        "stdout_is_single_json_record": completed.stdout.count(b"\n") == 1,
    }
    result = {
        "schema_version": 1,
        "regression_id": REGRESSION_ID,
        "definition": {
            "launcher_expression": "str(Path(sys.executable))",
            "shell": False,
            "child_mode": "-c import-only",
            "required_import": "scripts.run_f1_pgrr_v1_candidate_worker",
            "required_transitive_imports": ["research.f1_pgrr_v1.geometry", "numpy"],
            "prohibited_path_tokens": list(PROHIBITED_PATH_TOKENS),
            "scientific_execution": False,
        },
        "definition_sha256": _sha_bytes(_canonical_bytes({
            "child_program": CHILD_PROGRAM,
            "prohibited_path_tokens": PROHIBITED_PATH_TOKENS,
        })),
        "implementation_sha256": _sha_file(Path(__file__).resolve()),
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
            "numpy_module_origin": numpy_origin,
            "worker_module_origin": child.get("worker_module_origin"),
            "geometry_module_origin": child.get("geometry_module_origin"),
        },
        "child_return_code": completed.returncode,
        "child_stdout_sha256": stdout_sha,
        "child_stderr_sha256": stderr_sha,
        "child_stdout_bytes": len(completed.stdout),
        "child_stderr_bytes": len(completed.stderr),
        "child_parse_error": parse_error,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["result_sha256"] = _sha_bytes(_canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("source-free import regression accepts no arguments")
    result = run_regression()
    print(_canonical_bytes(result).decode("ascii"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
