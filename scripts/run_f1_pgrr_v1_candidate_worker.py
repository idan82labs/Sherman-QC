#!/usr/bin/env python3
"""Target-blind candidate worker for the authorized PGRR-V1 preflight.

This executable intentionally imports no frozen-target or downstream graph
module and accepts no target, case, filename-feature, gold, CAD, or benchmark
path.  Its process capability is limited to the exact raw source, immutable
configuration/code, commitment destination, and ephemeral bitset scratch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_pgrr_v1.config import canonical_json_bytes  # noqa: E402
from research.f1_pgrr_v1 import geometry  # noqa: E402
from research.f1_pgrr_v1.independent_validator import build_semantic_bundle  # noqa: E402


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("producer", "validator"), required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--reverse-input", action="store_true")
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kind == "producer":
        result = geometry.construct_and_commit(
            raw_source=args.raw_source.resolve(),
            commitment_path=args.commitment.resolve(),
            scratch_path=args.scratch.resolve(),
            role=args.role,
            reverse_input=args.reverse_input,
        )
        print(json.dumps({
            "decision": result["commitment"]["decision"],
            "commitment": str(args.commitment.resolve()),
            "scratch": result["scratch"],
        }, sort_keys=True))
        return 0
    result = build_semantic_bundle(args.raw_source.resolve())
    _atomic_write(args.commitment.resolve(), canonical_json_bytes(result) + b"\n")
    print(json.dumps({
        "decision": result["decision"],
        "commitment": str(args.commitment.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
