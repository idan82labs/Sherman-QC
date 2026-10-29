"""DEV dump for zs_lane_v3: run the lane's detection + pooling + snap + v2 evidence on every DEV scan (121
SPLIT_FROZEN dev + the 4 newly frozen ai_bend_lines_zs_v1 scans) and write <out>/<sid>.json for out-of-fold
evaluation (train.py). Every scan passes the gold guard (lane.run). The Z3 / Z2 settings are zs_lane_v2's frozen
ones (config_v2.json: Z3 fallback_frac_only, Z2 fallback), which v3 inherits unchanged; revision 2 adds the face
budget (budget.MAX_FACES; --no-budget for the diagnostic without it); revision 3 resamples on budget.CELL_MM. Run under the slot
semaphore (one process holds one slot).

usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.zs_lane_v3.dev_dump --out DIR [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from research.zeroshot.zs_lane_v3 import lane
from research.zeroshot.zs_lane_v3.core import REPO
from research.zeroshot.zs_lane_v2 import core as V2, scoring

V2_CONFIG = REPO / "research/zeroshot/zs_lane_v2/config_v2.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default="")
    ap.add_argument("--no-budget", action="store_true", help="revision 2 without the face budget (diagnostic)")
    a = ap.parse_args(argv)
    cfg = json.loads(V2_CONFIG.read_text())
    V2.Z3_PARAMS = cfg.get("z3_params") or None
    V2.Z2_FALLBACK = bool(cfg.get("z2_fallback", False))
    from research.zeroshot.zs_lane_v3 import budget, core
    core.BUDGET_MAX_FACES = None if a.no_budget else budget.MAX_FACES      # revision 2 face budget
    specs = scoring.case_specs("dev")
    ids = [s for s in a.ids.split(",") if s] or scoring.dev_ids()
    i, n = (int(x) for x in a.shard.split("/"))
    ids = [s for j, s in enumerate(ids) if j % n == i]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        dst = out / f"{sid}.json"
        if dst.is_file():
            continue
        try:
            lane.run(str(Path(specs[sid]["scan_path"]).expanduser()), "", cfg_path=None, variant=None,
                     dump=str(dst), record=None)
        except Exception as e:  # noqa: BLE001
            print(f"{sid} DUMP_ERROR {type(e).__name__}: {e}", flush=True)
            (out / f"{sid}.error.txt").write_text(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
