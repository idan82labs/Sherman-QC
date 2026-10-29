#!/usr/bin/env python3
"""Canonicalize a source-free candidate JSON array into F1_CANDIDATE_SET_V1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_candidate_ir_v1 import canonical_json_bytes, canonicalize_candidate_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-route")
    parser.add_argument("--source-stage")
    args = parser.parse_args()
    import json
    payload = json.loads(args.input.read_text())
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    result = canonicalize_candidate_set(rows, source_route=args.source_route, source_stage=args.source_stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(result)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, args.output)
    finally:
        if temp.exists(): temp.unlink()
    print(canonical_json_bytes({"status": "PASS", "candidate_count": result["candidate_count"], "output_sha256": __import__("hashlib").sha256(args.output.read_bytes()).hexdigest()}).decode(), end="")


if __name__ == "__main__":
    main()
