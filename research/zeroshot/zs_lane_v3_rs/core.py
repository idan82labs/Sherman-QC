"""zs_lane_v3_rs adapter core: run the zs_rust binary's `lane` command on one scan.

The binary runs the whole of zs_lane_v3 revision 2 (code fingerprint ab31d061..., config 1c3f8cb7...): face budget,
Z2, Z3, pooling, D1 snap, v2 evidence, the two GBMs (exported trees, export_models.py), v2 post-processing, EXT and
Z3R, and writes eval_lines.json. Python keeps: the gold guard and custody (lane.py), the pins, and one fallback:
scans stored without faces (point clouds) exit with EXIT_POINT_CLOUD, and the adapter then runs the frozen Python
revision 2 lane for that scan (open3d's normal estimation and ball-pivoting reconstruction are not ported).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from research.zeroshot.zs_lane_v3_rs import rs

REPO = rs.REPO
LANE = "zs_lane_v3_rs"
EXIT_POINT_CLOUD = 10


class PointCloudScan(RuntimeError):
    pass


def run_binary(scan: str, cfg_path: str, *, out: str | None = None, record: str | None = None,
               dump: str | None = None, splits: bool = False, conversion: dict | None = None,
               lane: str = "zs_lane_v3_rs:zs", cad: dict | None = None) -> dict:
    """One `zs_rust lane` call. Raises PointCloudScan for a scan without faces, RuntimeError on any other failure.
    ``cad``: the production CAD priors {"t_mm": float, "max_families": int} (lane_cli.run_lane_v3's thickness and
    family cap for Z3); the n_bends count cap stays with the caller, as in lane_cli."""
    if not rs.available():
        raise RuntimeError(f"zs_rust binary not found: {rs.binary()}")
    with tempfile.TemporaryDirectory(prefix="zs_rust_lane_") as d:
        job = {"scan": str(scan), "config": str(cfg_path), "repo": str(REPO), "lane": lane, "splits": bool(splits),
               "conversion": conversion or {}}
        if out:
            job["out"] = str(out)
        if record:
            job["record"] = str(record)
        if dump:
            job["dump"] = str(dump)
        if cad:
            job["cad"] = {"t_mm": float(cad["t_mm"]), "max_families": int(cad["max_families"])}
        jp, op = Path(d) / "job.json", Path(d) / "out.json"
        jp.write_text(json.dumps(job))
        t0 = time.time()
        r = subprocess.run([str(rs.binary()), "lane", "--job", str(jp), "--out", str(op), "--threads", str(rs.threads())],
                           capture_output=True, text=True)
        if r.returncode == EXIT_POINT_CLOUD:
            raise PointCloudScan(r.stderr.strip())
        if r.returncode != 0 or not op.is_file():
            raise RuntimeError(f"zs_rust lane failed (rc {r.returncode}): {r.stderr[-2000:]}")
        res = json.loads(op.read_text())
        res["call_wall_s"] = round(time.time() - t0, 3)
        return res
