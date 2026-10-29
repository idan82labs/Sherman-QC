"""Custody guard for Z3 (axis-family slicing). Path/name checks only; never opens a file it refuses.

A scan is admitted only when
  1. its scan_id is in SPLIT_FROZEN.json ``dev.scans`` and not in ``test_b.scans``;
  2. its INVENTORY row says usable and has gold;
  3. its path passes research/bench_labeling/custody.check_path (holdout / sealed / EHO4 / one-shot /
     N001-N014 / rhino-preview) plus this programme's extras (N001-N016, EHO4 part numbers, _sealed,
     holdout10, eyal50 quarantine, test_b tokens).
INVENTORY rows of TEST-B scans are dropped when loaded and never returned.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[3]
REPO = _HERE if (_HERE / "research/bench_labeling").is_dir() else Path("/Users/idantal/ShermanQC/FullProject")
RUNLOG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
OUT = RUNLOG / "z3"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.bench_labeling.custody import CustodyError, check_path as _base_check  # noqa: E402

EHO4_PARTS = ("16231000_03", "38168001_02", "38173001_01", "26687013_A00", "31409000_01",
              "47215000_B", "42637000_B", "50476000_A", "31367000_A00", "30820000_A01")
_N016 = re.compile(r"(?<![A-Za-z0-9])N0(0[1-9]|1[0-6])(?![0-9])")
_EXTRA = ("holdout10", "_sealed", "rhino-preview", "rhino_preview", "quarantine", "testb", "test_b",
          "test-b", "38173000_01")


def check_path(path: str | Path) -> Path:
    p = _base_check(path)
    low = p.as_posix().casefold()
    for tok in _EXTRA:
        if tok in low:
            raise CustodyError(f"forbidden token {tok!r}: {p.name}")
    if _N016.search(p.as_posix()):
        raise CustodyError(f"opaque N001-N016 identifier: {p.name}")
    for part in EHO4_PARTS:
        if part.casefold() in low:
            raise CustodyError(f"EHO4 part {part}: {p.name}")
    return p


def _split() -> tuple[list[str], set[str]]:
    sp = json.loads((RUNLOG / "SPLIT_FROZEN.json").read_text())
    return list(sp["dev"]["scans"]), set(sp["test_b"]["scans"])


def dev_ids() -> list[str]:
    dev, test = _split()
    return [s for s in dev if s not in test]


def dev_inventory() -> dict[str, dict]:
    dev, test = _split()
    keep = set(dev) - test
    rows = json.loads((RUNLOG / "INVENTORY.json").read_text())["scans"]
    return {r["scan_id"]: r for r in rows if r["scan_id"] in keep}


def check_dev_scan(scan_id: str, inv: dict[str, dict]) -> Path:
    dev, test = _split()
    if scan_id in test or scan_id not in set(dev) or scan_id not in inv:
        raise CustodyError(f"{scan_id}: not a DEV scan")
    row = inv[scan_id]
    if not row.get("usable") or not row.get("has_gold"):
        raise CustodyError(f"{scan_id}: not usable / no gold")
    try:   # programme harness guard (gold ledger + TEST-B seal), when present
        from research.zeroshot.common import gold_guard
    except ImportError:
        gold_guard = None
    if gold_guard is not None:
        try:
            gold_guard.require_runnable(scan_id)
        except Exception as e:  # noqa: BLE001  (GoldGuardRefusal)
            raise CustodyError(f"{scan_id}: gold guard refused: {e}") from e
    return check_path(os.path.expanduser(row["path"]))


def check_source_path(path: str | Path) -> Path:
    """Extra candidate sources (A0 / Z2 eval_lines): refuse sealed / test paths."""
    return check_path(path)
