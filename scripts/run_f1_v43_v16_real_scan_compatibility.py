#!/usr/bin/env python3
"""Execute V43's two-sided support guard on retained V16 scan pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_two_sided_parent_support_guard_v43 import (
    MIN_TWO_SIDED_SUPPORT_FRACTION,
    select_unified_v43,
)
from scripts.run_f1_v42_v16_real_scan_compatibility import (
    CONTRACT as V42_CONTRACT,
    run_case,
)


PROFILE = "F1_V43_V16_REAL_SCAN_COMPATIBILITY_V1"
CONTRACT = {
    **V42_CONTRACT,
    "profile": PROFILE,
    "minimum_two_sided_support_fraction": MIN_TWO_SIDED_SUPPORT_FRACTION,
    "support_guard_scope": "v42_restorations_only",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(
        pool_path=args.pool.resolve(), retained_path=args.retained.resolve(),
        scan_path=args.scan.resolve(), output_path=args.output.resolve(),
        selector=select_unified_v43, contract=CONTRACT, profile=PROFILE,
        result_label="v43",
    )
    print(json.dumps({
        "compatible_rejected_candidate_count": result["compatible_rejected_candidate_count"],
        "v43_resource_stop_count": result["v43_resource_stop_count"],
        "v43_pre_merge_addition_count": result["v43_pre_merge_addition_count"],
        "v43_post_merge_addition_count": result["v43_post_merge_addition_count"],
        "candidate_count": result["candidate_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
