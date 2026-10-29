"""Run the Z3 slicer over DEV scans in one process (use via sherman_slot.sh). Writes per-scan candidate
files (all candidates + features + timing) to <out>/<sid>.json. Custody-checked: DEV only.

usage: .venv/bin/python -B -m research.zeroshot.z3_slicing.run_dev --out DIR [--ids a,b] [--shard i/n] [--param k=v ...]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path

_HERE = Path(__file__).resolve().parents[3]
REPO = _HERE if (_HERE / "research/bench_labeling").is_dir() else Path("/Users/idantal/ShermanQC/FullProject")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, slicer  # noqa: E402


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)
    inv = custody.dev_inventory()
    ids = [s for s in a.ids.split(",") if s] or custody.dev_ids()
    i, n = (int(x) for x in a.shard.split("/"))
    ids = [s for j, s in enumerate(ids) if j % n == i]
    prm = slicer.Params()
    types = {f.name: f.type for f in fields(prm)}
    for kv in a.param:
        k, v = kv.split("=")
        setattr(prm, k, int(v) if types[k] in ("int", int) else float(v))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        dst = out / f"{sid}.json"
        if a.skip_existing and dst.is_file():
            continue
        path = custody.check_dev_scan(sid, inv)
        t0 = time.time()
        try:
            mesh = slicer.load_mesh(str(path))
            tl = time.time() - t0
            res = slicer.detect(mesh, prm)
            res["timing_s"]["load"] = tl
            res["timing_s"]["wall"] = time.time() - t0
            res["status"] = "OK"
            res["mesh"] = {"vertices": int(len(mesh.V)), "faces": int(len(mesh.F))}
        except Exception as e:  # noqa: BLE001
            res = {"status": "FAILED", "error": repr(e), "trace": traceback.format_exc(),
                   "candidates": [], "timing_s": {"wall": time.time() - t0}}
        res["scan_id"] = sid
        res["scan_path"] = str(path)
        res["peak_rss_gib"] = round(_rss_gib(), 3)
        dst.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"{sid} {res['status']} cands={len(res['candidates'])} wall={res['timing_s']['wall']:.1f}s "
              f"rss={res['peak_rss_gib']:.2f}G", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
