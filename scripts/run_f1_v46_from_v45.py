#!/usr/bin/env python3
"""Apply the bounded elongated-lane consolidation to frozen V45 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_elongated_lane_consolidation_v46 import compose_v46


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v45", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v45 = json.loads(args.v45.read_text(encoding="utf-8"))
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    result = compose_v46(v45, pool["scan_bbox_mm"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audit = result["elongated_lane_consolidation_audit"]
    print(json.dumps({
        "prediction_count": len(result["predictions"]),
        "gate_active": audit["gate_active"],
        "suppressed_terminal_count": len(audit["suppressed_terminal_candidate_ids"]),
        "stitched_component_count": len(audit["stitch_changes"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
