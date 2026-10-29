#!/usr/bin/env python3
"""Compile independent CAD ontology reviews into consensus or adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cad_line_ontology_consensus_v1 import compile_consensus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manufacturing", type=Path, required=True)
    parser.add_argument("--countersigning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_consensus(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.manufacturing.read_text(encoding="utf-8")),
        json.loads(args.countersigning.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("READY_TO_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
