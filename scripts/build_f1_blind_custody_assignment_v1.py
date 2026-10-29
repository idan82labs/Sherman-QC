#!/usr/bin/env python3
"""CLI for deterministic, split-output physical annotation custody."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_custody_v1 import allocate


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--custodian-output", type=Path, required=True)
    args = parser.parse_args()
    result = allocate(json.loads(args.input.read_text(encoding="utf-8")))
    _atomic(args.public_output, result["public_assignment"])
    _atomic(args.custodian_output, result["custodian_only_case_map"])
    print(json.dumps({
        "status": result["public_assignment"]["status"],
        "case_count": result["public_assignment"].get("case_count", 0),
        "morphology_family_count": result["public_assignment"].get("morphology_family_count", 0),
        "public_and_custodian_outputs_separated": True,
    }, sort_keys=True))
    return 0 if result["public_assignment"]["status"] == "READY_FOR_BLIND_ANNOTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
