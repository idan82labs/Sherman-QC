#!/usr/bin/env python3
"""Run source-only single-axis parallel-return recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_single_axis_parallel_return_recovery_v51 import compose_v51


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v49", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose_v51(
        json.loads(args.v49.read_text(encoding="utf-8")),
        json.loads(args.pool.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audit = result["single_axis_parallel_return_recovery_audit"]
    print(json.dumps({
        "prediction_count": len(result["predictions"]),
        "gate_active": audit["gate_active"],
        "selected_candidate_id": audit["selected_candidate_id"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
