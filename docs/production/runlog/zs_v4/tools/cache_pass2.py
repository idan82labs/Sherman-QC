"""v4 candidate cache, pass 2 (one pool scan; through the slot): the hem generator and a Z3 re-slice stability check
on the same (budgeted) mesh and thickness as the pass-1 "v4" record.

Writes docs/production/runlog/zs_v4/cache/<case_id>.p2.json.gz:
  * "hem": zs_lane_v4.hem.detect lines;
  * "stab": Z3 re-run with slice step 1.5 mm (instead of 2.0) on the same mesh; per pass-1 v4 candidate with Z3
            features: does the re-slice reproduce it (zs_lane_v3.fixes.is_dup on the raw lines), and the best
            matching re-slice candidate's support / quality ratio. A real fold is found at any slice phase; a
            spurious track often is not.
usage (repo root): sherman_slot.sh .venv/bin/python -B docs/production/runlog/zs_v4/tools/cache_pass2.py <case_id>
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
CACHE = REPO / "docs/production/runlog/zs_v4/cache"


def main(case_id: str) -> int:
    from research.zeroshot.zs_lane_v4 import cache_io as CI, hem, stability as ST
    from research.zeroshot.zs_lane_v2 import z3s
    from research.zeroshot.zs_lane_v3 import budget
    pool = {s["case_id"]: s for s in json.loads((REPO / "docs/production/runlog/zs_v4/POOL.json").read_text())["scans"]}
    s = pool[case_id]
    d = CI.load(case_id)
    r = d["v4"]
    t = (r.get("meta") or {}).get("t_mm")
    doc = {"schema": "sherman.zs_v4.cache_p2.v1", "case_id": case_id, "sha256": s["sha256"], "wall": {}}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="zs_v4_p2_") as work:
        path, binfo = budget.budget_scan(s["path"], 3_000_000, work, 0.30)
        mesh = z3s.load_mesh(path)
    doc["wall"]["load"] = round(time.time() - t0, 3)
    if t is None:
        doc.update(hem=[], stab=None, note="no thickness in pass 1")
    else:
        t1 = time.time()
        doc["hem"] = hem.detect(mesh, float(t))
        doc["wall"]["hem"] = round(time.time() - t1, 3)
        t2 = time.time()
        doc["stab"] = ST.restability(mesh, float(t), r.get("candidates") or [])
        doc["wall"]["stab"] = round(time.time() - t2, 3)
    doc["wall"]["total"] = round(time.time() - t0, 3)
    import resource
    doc["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 3)
    tmp = CACHE / f"{case_id}.p2.json.gz.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, CACHE / f"{case_id}.p2.json.gz")
    print(f"{case_id} hem={len(doc.get('hem') or [])} stab={'-' if not doc.get('stab') else sum(1 for x in doc['stab']['match'] if x)}"
          f" wall={doc['wall']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
