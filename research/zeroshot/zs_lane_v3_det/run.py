"""zs_lane_v3 CLI under the deterministic-math research patch (patch.apply()), then lane.main(argv) unchanged.
usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3_det.run <zs_lane_v3.lane arguments>"""
import json
import sys

from research.zeroshot.zs_lane_v3_det import patch, dm


def main():
    applied = patch.apply()
    from research.zeroshot.zs_lane_v3 import lane
    rc = lane.main(sys.argv[1:])
    print("DETMATH", json.dumps({"applied": applied, "calls": dm.CALLS}), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
