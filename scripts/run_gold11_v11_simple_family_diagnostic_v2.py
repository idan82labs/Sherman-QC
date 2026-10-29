#!/usr/bin/env python3
"""Successor entrypoint for the fixed simple-family V11 diagnostic.

The predecessor entrypoint was terminally stopped before raw input access by
an order assertion in its fixed-scope guard.  This wrapper keeps that source
and output untouched, corrects only the deterministic review-order mapping,
and binds the successor result to this entrypoint and protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gold11_v11_simple_family_diagnostic_v1 as base


PROTOCOL_PATH = ROOT / "docs/research/2026-09-16_gold11_v11_simple_family_diagnostic_v2_protocol.md"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_simple_family_diagnostic_v2"
SIMPLE_PART_KEYS = ("37361005", "49001000", "49024000")
SIMPLE_CASE_TOKENS = ("CG4BDD1371B6", "CG191DEC7F81", "CG4FAF273093")


def _simple_cases(all_cases):
    by_key = {row["part_key"]: row for row in all_cases}
    if set(by_key) & set(SIMPLE_PART_KEYS) != set(SIMPLE_PART_KEYS):
        raise ValueError("SIMPLE_FAMILY_SCOPE_PART_KEY_MISSING")
    rows = [by_key[key] for key in SIMPLE_PART_KEYS]
    if tuple(row["case_token"] for row in rows) != SIMPLE_CASE_TOKENS:
        raise ValueError("SIMPLE_FAMILY_SCOPE_IDENTITY_MISMATCH")
    if len(rows) != 3 or len({row["part_key"] for row in rows}) != 3:
        raise ValueError("SIMPLE_FAMILY_SCOPE_CARDINALITY_MISMATCH")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path):
    # Reuse the reviewed implementation while replacing only its predecessor
    # scope-order guard.  Every inference/scoring rule remains identical.
    base._simple_cases = _simple_cases
    base.PROTOCOL_PATH = PROTOCOL_PATH
    result = base.run(output)
    bindings = dict(result["source_bindings"])
    bindings["runner_sha256"] = _sha256(Path(__file__).resolve())
    bindings["predecessor_runner_sha256"] = _sha256(
        ROOT / "scripts/run_gold11_v11_simple_family_diagnostic_v1.py")
    result["source_bindings"] = bindings
    result["predecessor_attempt"] = {
        "entrypoint": "scripts/run_gold11_v11_simple_family_diagnostic_v1.py",
        "status": "TERMINAL_PRE_INFERENCE_SCOPE_ORDER_GUARD",
        "raw_bytes_accessed": False,
        "reason": "review-order assertion expected a different fixed tuple",
    }
    output = output.resolve()
    (output / "RESULT.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checksums = {
        "schema": "GOLD11_V11_SIMPLE_FAMILY_DIAGNOSTIC_CHECKSUMS_V2",
        "result_sha256": _sha256(output / "RESULT.json"),
        "source_bindings": result["source_bindings"],
    }
    (output / "CHECKSUMS.json").write_text(
        json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY_SIMPLE_FAMILIES_V2",
                      "scope": result["scope"], "score": result["score"],
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
