"""DEV dump for zs_lane_v1: run the lane's detection + pooling + snap on every DEV scan (121 SPLIT_FROZEN dev + the 4
newly frozen ai_bend_lines_zs_v1 scans) and write <out>/<sid>.json for out-of-fold training (train.py).
Every scan passes the gold guard (lane.run). Run under the slot semaphore (one process holds one slot).

usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.zs_lane_v1.dev_dump --out DIR [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from research.zeroshot.zs_lane_v1 import lane, scoring


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
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
                     final_testb=False, dump=str(dst), record=None)
        except Exception as e:  # noqa: BLE001
            print(f"{sid} DUMP_ERROR {type(e).__name__}: {e}", flush=True)
            (out / f"{sid}.error.txt").write_text(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
