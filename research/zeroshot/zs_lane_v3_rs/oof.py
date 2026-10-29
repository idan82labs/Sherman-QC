"""DEV out-of-fold evaluation of a zs_lane_v3_rs DEV dump with the frozen zs_lane_v3 revision-2 procedure
(zs_lane_v3/rev2_eval.py, unchanged; run from the revision-2 byte copies, rev2ref): the v3 procedure refit on the dump, and the production-faithful arm that keeps
the v3 fold models and post settings. Python training code only; nothing is refit for production.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.oof --dump RS_DUMP --v3-dump V3_REV1_DUMP --out DIR
"""
from __future__ import annotations

import hashlib
import sys


PINNED = {"research/zeroshot/zs_lane_v3/train.py": "cce19ecfe675fad5e3958593816d8bb5d3b4c95b1437703ae108f9642359fe74"}


def main(argv=None) -> int:
    from research.zeroshot.zs_lane_v3_rs import rev2ref
    for f, sha in PINNED.items():
        if hashlib.sha256(rev2ref.resolve(f).read_bytes()).hexdigest() != sha:
            print(f"refusing: {f} is not the frozen file", file=sys.stderr)
            return 2
    rev2ref.install()
    from research.zeroshot.zs_lane_v3 import rev2_eval
    return rev2_eval.main(argv)


if __name__ == "__main__":
    sys.exit(main())
