#!/usr/bin/env python3
"""Run longest-first incumbent ownership and retained V45 completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_longest_incumbent_ownership_v50 import compose_v50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v44 = json.loads(args.v44.read_text(encoding="utf-8"))
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    result = compose_v50(v44, pool["line_objects"], max(pool["scan_bbox_mm"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "prediction_count": len(result["predictions"]),
        "paired_return_addition_count": len(result["paired_return_completion_changes"]),
        "post_completion_nms_suppressed_count": len(
            result["post_completion_nms_suppressed_candidate_ids"]
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
