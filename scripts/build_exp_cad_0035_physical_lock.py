#!/usr/bin/env python3
"""Materialize the one immutable EXP_CAD_0035 physical execution lock.

The builder is source-free: it never opens or hashes either physical PLY.
Every regular static dependency is provided explicitly as KEY=PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp_cad_0035_execution_contract import (
    REQUIRED_BOUND_FILES,
    build_lock_payload,
    canonical_bytes,
)


def _binding(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or key not in REQUIRED_BOUND_FILES or not raw_path:
        raise argparse.ArgumentTypeError("binding must be a required KEY=PATH pair")
    return key, Path(raw_path)


def create_immutable_lock(path: Path, payload: object) -> str:
    """Publish one durable lock without a check/replace race."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short EXP35 immutable-lock write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", action="append", required=True, type=_binding)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    bindings = dict(args.binding)
    if len(bindings) != len(args.binding):
        raise ValueError("duplicate EXP35 lock binding key")
    payload = build_lock_payload(bindings=bindings, lock_path=args.output)
    digest = create_immutable_lock(args.output, payload)
    print(json.dumps({"status": "LOCKED_AWAITING_DETACHED_APPROVAL", "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
