#!/usr/bin/env python3
"""Split a neutral candidate slate into small native-review chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = list(payload["predictions"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(rows), args.chunk_size):
        chunk = dict(payload)
        chunk["predictions"] = rows[offset : offset + args.chunk_size]
        chunk["candidate_count"] = len(chunk["predictions"])
        chunk["chunk_index"] = offset // args.chunk_size + 1
        chunk["chunk_count"] = (len(rows) + args.chunk_size - 1) // args.chunk_size
        path = args.output_dir / f"chunk_{chunk['chunk_index']:02d}.json"
        path.write_text(
            json.dumps(chunk, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
