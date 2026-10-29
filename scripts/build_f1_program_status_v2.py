#!/usr/bin/env python3
"""Materialize the current F1 critical path from frozen artifacts."""

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

from research.f1_program_status_v2 import build


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-gate", type=Path, required=True)
    parser.add_argument("--soak-report", type=Path, required=True)
    parser.add_argument("--authority-validation", type=Path, required=True)
    parser.add_argument("--pending-intake", type=Path, required=True)
    parser.add_argument("--pending-intake-root", type=Path, required=True)
    parser.add_argument("--additional-pending-intake", type=Path, action="append", default=[])
    parser.add_argument("--additional-pending-intake-root", type=Path, action="append", default=[])
    parser.add_argument("--lineage-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-output", type=Path)
    args = parser.parse_args()
    if len(args.additional_pending_intake) != len(args.additional_pending_intake_root):
        parser.error("additional pending intake paths and roots must have equal counts")
    result = build(
        unified_gate=_load(args.unified_gate),
        soak_report=_load(args.soak_report),
        authority_validation=_load(args.authority_validation),
        pending_intake=_load(args.pending_intake),
        pending_intake_root=args.pending_intake_root,
        lineage_registry=_load(args.lineage_registry),
        additional_pending_intakes=[
            (_load(path), root)
            for path, root in zip(args.additional_pending_intake, args.additional_pending_intake_root)
        ],
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    for target in (args.output, args.current_output):
        if target is None:
            continue
        _atomic_write(target, rendered)
    print(json.dumps({
        "active_blockers": result["active_blockers"],
        "pending_new_physical_intake": result["pending_new_physical_intake"],
        "gpi": result["generalization_progress_indicator"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
