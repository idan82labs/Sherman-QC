"""Custody for Z4/Z5: DEV scans of SPLIT_FROZEN.json only. Path/name checks never open the file.

A scan is admitted only when (1) its scan_id is in SPLIT_FROZEN.json ``dev.scans``, (2) it is not in
``test_b.scans``, and (3) its path passes the D1 path guard (holdout / sealed / EHO4 / N001-N016 /
rhino-preview tokens). INVENTORY rows of TEST-B scans are dropped at load time and never returned.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNLOG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
OUT = RUNLOG / "z4"
WORK = Path("/private/tmp/claude-501/sherman_zeroshot_work")
POOL_ROOT = WORK / "z4_pools"           # rebuilt pools (this WP)
OLD_DEV_ROOT = WORK / "dev"            # the 18 phase-1 DEV re-runs (pool + typed union)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.d1_two_skin.custody import CustodyError, check_path  # noqa: E402


def _split() -> tuple[list[str], set[str]]:
    sp = json.loads((RUNLOG / "SPLIT_FROZEN.json").read_text())
    return list(sp["dev"]["scans"]), set(sp["test_b"]["scans"])


def dev_inventory() -> dict[str, dict]:
    """INVENTORY rows of DEV scans only (TEST-B rows are never kept)."""
    dev, test = _split()
    devset = set(dev) - test
    rows = json.loads((RUNLOG / "INVENTORY.json").read_text())["scans"]
    return {r["scan_id"]: r for r in rows if r["scan_id"] in devset}


def dev_ids() -> list[str]:
    dev, test = _split()
    return [s for s in dev if s not in test]


def check_dev_scan(scan_id: str, inv: dict[str, dict]) -> Path:
    dev, test = _split()
    if scan_id in test or scan_id not in set(dev) or scan_id not in inv:
        raise CustodyError(f"{scan_id}: not a DEV scan")
    return check_path(os.path.expanduser(inv[scan_id]["path"]))


def check_source_path(path: str | Path) -> Path:
    """For extra candidate sources (Z2/Z3 eval_lines): refuse sealed / test paths."""
    p = check_path(path)
    low = p.as_posix().casefold()
    for tok in ("testb", "test_b", "test-b"):
        if tok in low:
            raise CustodyError(f"TEST-B path refused: {p}")
    return p
