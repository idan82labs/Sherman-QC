"""Rebuild the A0 scan-only candidate pools (decompose + pool, NO typed union) for DEV scans.

One process per slot: run the whole worker share under one sherman_slot.sh hold:
  /private/tmp/claude-501/sherman_slot.sh .venv/bin/python -B -m research.zeroshot.z4_selector.rebuild_pools \
      --worker 0 --workers 3
Each scan: copy to an opaque staged name, `bb_steps.py decompose`, `bb_steps.py pool --voxel-mm 1.8`
(exactly the run_lane.py A0 steps 1-2), each in its own subprocess with a timeout; the staged copy is
deleted afterwards. Scans that already have a pool (here or in the phase-1 dev work dir) are skipped.
Writes POOL_ROOT/<scan>/{pool/BB*.json, status.json}. Labels are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.zeroshot.z4_selector.custody import OLD_DEV_ROOT, POOL_ROOT, REPO, check_dev_scan, dev_ids, dev_inventory

PY = str(REPO / ".venv/bin/python")
STEPS = str(REPO / "research/broad_baseline/bb_steps.py")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def existing_pool(sid: str) -> Path | None:
    for root in (POOL_ROOT, OLD_DEV_ROOT):
        d = root / sid / "pool"
        if d.is_dir():
            files = sorted(p for p in d.glob("BB*.json") if not p.name.endswith(".row.json"))
            if files:
                return files[0]
    return None


def run(cmd: list[str], log: Path, timeout: float) -> dict:
    t0 = time.monotonic()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    with open(log, "w") as fo:
        try:
            rc = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, timeout=timeout, cwd=REPO, env=env).returncode
            to = False
        except subprocess.TimeoutExpired:
            rc, to = -9, True
    return {"rc": rc, "timed_out": to, "wall_s": round(time.monotonic() - t0, 1)}


def build(sid: str, inv: dict, timeout: float) -> dict:
    src = check_dev_scan(sid, inv)
    work = POOL_ROOT / sid
    (work / "logs").mkdir(parents=True, exist_ok=True)
    sha = sha256(src)
    token = "BB" + sha[:12]
    staged = work / "staged" / f"scan_{sha[:12]}{src.suffix.lower()}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, staged)
    st: dict = {"scan_id": sid, "token": token, "scan_sha256": sha}
    dec = work / "a0" / token / "result" / "decomposition_result.json"
    st["decompose"] = run([PY, "-B", STEPS, "decompose", "--scan", str(staged), "--out-json", str(dec),
                           "--cache-dir", str(work / "a0_cache")], work / "logs/decompose.log", timeout)
    if dec.is_file():
        st["pool"] = run([PY, "-B", STEPS, "pool", "--decomposition-root", str(work / "a0"), "--token", token,
                          "--source-scan", str(staged), "--source-sha", sha, "--out-dir", str(work / "pool"),
                          "--voxel-mm", "1.8"], work / "logs/pool.log", timeout)
    pool = work / "pool" / f"{token}.json"
    st["status"] = "OK" if pool.is_file() else ("DECOMPOSE_FAILED" if not dec.is_file() else "POOL_FAILED")
    staged.unlink(missing_ok=True)
    shutil.rmtree(work / "a0_cache", ignore_errors=True)
    (work / "status.json").write_text(json.dumps(st, indent=1))
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    inv = dev_inventory()
    # stride over a FIXED list (all DEV ids, or --only): a list filtered by "pool exists" at start-up differs
    # between workers that acquire their slot at different times and made shares overlap (iter 0 bug).
    ids = a.only or dev_ids()
    mine = ids[a.worker::a.workers]
    log = POOL_ROOT / f"driver_{a.worker}.log"
    POOL_ROOT.mkdir(parents=True, exist_ok=True)
    for sid in mine:
        if existing_pool(sid) is not None and not a.force:
            continue
        t0 = time.monotonic()
        st = build(sid, inv, a.timeout)
        with open(log, "a") as f:
            f.write(f"{time.strftime('%FT%T')} {sid} {st['status']} {round(time.monotonic() - t0)}s\n")
    with open(log, "a") as f:
        f.write("DONE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
