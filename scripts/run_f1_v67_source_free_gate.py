#!/usr/bin/env python3
"""Run and freeze the V67 source-free implementation gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_f1_similarity_invariant_normal_cycle_mode_match_nms_v67.py"
IMPLEMENTATION = ROOT / "backend/f1_similarity_invariant_normal_cycle_mode_match_nms_v67.py"
ERRATUM = ROOT / "docs/research/2026-08-15_f1_v67_pair_domain_geometric_erratum_e1.md"
OUTPUT = ROOT / "output/research/f1-ncmm-v67-source-free-gate-v1-20260815/summary.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    command = [str(ROOT / ".venv/bin/pytest"), "-q", str(TEST)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v67_source_free_gate",
        "profile": "V67_SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1",
        "fixture_lock": "V67_NCMM_NMS_V1_FIXTURES_LOCKED_WITH_GEOMETRIC_ERRATUM_E1",
        "status": "PASS_WITH_GEOMETRIC_ERRATUM_E1" if completed.returncode == 0 else "FAIL",
        "pytest_command": command,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "pytest_returncode": completed.returncode,
        "implementation_sha256": digest(IMPLEMENTATION),
        "fixture_sha256": digest(TEST),
        "erratum_sha256": digest(ERRATUM),
        "original_constants_changed_by_erratum": False,
        "synthetic_morphology_used": False,
        "real_source_read": False,
        "gold_used": False,
        "labels_used": False,
        "expected_count_used": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
