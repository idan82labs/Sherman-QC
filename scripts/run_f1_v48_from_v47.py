#!/usr/bin/env python3
"""Apply deterministic nested-partial NMS to frozen V47 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_nested_partial_duplicate_nms_v48 import compose_v48


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v47", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose_v48(json.loads(args.v47.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "prediction_count": len(result["predictions"]),
        "nested_partial_suppression_count": len(result["nested_partial_duplicate_nms_changes"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
