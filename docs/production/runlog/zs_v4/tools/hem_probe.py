"""Hem-generator probe on a few pool scans (through the slot): run zs_lane_v4.hem.detect on the v4 (budgeted) mesh
with the cached thickness and print the hem lines; evaluation-side comparison with the gold hem lines is done
afterwards from the written JSON. usage: sherman_slot.sh .venv/bin/python -B hem_probe.py <out.json> <case_id>...
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))


def main(out: str, ids: list[str]) -> None:
    from research.zeroshot.zs_lane_v2 import z3s
    from research.zeroshot.zs_lane_v3 import budget
    from research.zeroshot.zs_lane_v4 import cache_io as CI, hem
    pool = {s["case_id"]: s for s in json.loads((REPO / "docs/production/runlog/zs_v4/POOL.json").read_text())["scans"]}
    res = {}
    for cid in ids:
        d = CI.load(cid)
        t = float(d["v4"]["meta"]["t_mm"])
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="zs_v4_hem_") as work:
            path, _ = budget.budget_scan(pool[cid]["path"], 3_000_000, work, 0.30)
            mesh = z3s.load_mesh(path)
        t1 = time.time()
        lines = hem.detect(mesh, t)
        res[cid] = {"t": t, "load_s": round(t1 - t0, 2), "hem_s": round(time.time() - t1, 2), "lines": lines}
        print(cid, len(lines), f"load {t1 - t0:.1f}s hem {time.time() - t1:.1f}s",
              [(round(l["hem"]["len"], 1), l["hem"]["stack_t"], l["hem"]["frac"]) for l in lines], flush=True)
        del mesh
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
