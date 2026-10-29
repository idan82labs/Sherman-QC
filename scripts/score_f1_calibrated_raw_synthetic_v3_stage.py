#!/usr/bin/env python3
"""Score V3 with the frozen V60/V61 matchers under the V3 namespace token.

This adapter changes only the corpus namespace token.  Matching thresholds,
assignment, aggregation, and birth accounting remain in the frozen scorers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import score_f1_calibrated_raw_synthetic_v60 as score_v60


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V3_COMPLEX_GEOMETRY_NON_CONFIRMATORY"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("v60", "v61"), required=True)
    args, remaining = parser.parse_known_args()
    score_v60.TOKEN = TOKEN
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == "v60":
        return score_v60.main()

    from scripts import score_f1_calibrated_raw_synthetic_v61 as score_v61

    score_v61.TOKEN = TOKEN
    return score_v61.main()


if __name__ == "__main__":
    raise SystemExit(main())
