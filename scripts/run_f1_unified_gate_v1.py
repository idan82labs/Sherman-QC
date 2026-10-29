#!/usr/bin/env python3
"""Evaluate the fail-closed lane-aware F1 generalization gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_candidate_ir_v1 import canonical_json_bytes
from research.f1_unified_gate_v1 import evaluate


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-a", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    thresholds = None if args.thresholds is None else json.loads(args.thresholds.read_text())
    report = evaluate(evidence, thresholds)
    _atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if args.allow_non_a or report["grade"] == "A_VERIFIED_PHYSICAL_PROGRESS" else 2)


if __name__ == "__main__":
    main()
