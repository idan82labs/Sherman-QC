"""Custody guard for the D1 study. Path/name checks only; never opens the file it checks.

Wraps research/bench_labeling/custody.check_path (holdout / sealed / EHO4 / one-shot /
authorization tokens, N001-N014, rhino-preview) and adds what this programme forbids on top:
N001-N016, the ten EHO4 part numbers by name, and the zero-shot INVENTORY custody class
(a scan is usable only when INVENTORY.json marks it usable and DEV_CONSUMED / CONSUMED_DEVCHECK).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.bench_labeling.custody import CustodyError, check_path as _base_check  # noqa: E402

EHO4_PARTS = ("16231000_03", "38168001_02", "38173001_01", "26687013_A00", "31409000_01",
              "47215000_B", "42637000_B", "50476000_A", "31367000_A00", "30820000_A01")
_N016 = re.compile(r"(?<![A-Za-z0-9])N0(0[1-9]|1[0-6])(?![0-9])")
_EXTRA_TOKENS = ("holdout10", "_sealed", "rhino-preview", "rhino_preview", "fresh_v1_sealed")
INVENTORY = REPO / "docs/production/runlog/zeroshot_2026-10-25/INVENTORY.json"
ALLOWED_CUSTODY = {"DEV_CONSUMED", "CONSUMED_DEVCHECK"}


def check_path(path: str | Path) -> Path:
    resolved = _base_check(path)
    text = resolved.as_posix()
    low = text.casefold()
    for tok in _EXTRA_TOKENS:
        if tok in low:
            raise CustodyError(f"forbidden token {tok!r}: {resolved.name}")
    if _N016.search(text):
        raise CustodyError(f"opaque N001-N016 identifier: {resolved.name}")
    for part in EHO4_PARTS:
        if part.casefold() in low:
            raise CustodyError(f"EHO4 part {part}: {resolved.name}")
    return resolved


def inventory_custody() -> dict[str, dict]:
    inv = json.loads(INVENTORY.read_text())
    rows = {s["scan_id"]: s for s in inv["scans"]}
    for s in inv.get("outside_benchmark", []):
        rows.setdefault(s["scan_id"], s)
    return rows


def check_scan(scan_id: str, path: str | Path, inventory: dict[str, dict]) -> Path:
    resolved = check_path(path)
    row = inventory.get(scan_id)
    if row is None or not row.get("usable") or row.get("custody") not in ALLOWED_CUSTODY:
        raise CustodyError(f"{scan_id}: not a usable dev scan in INVENTORY "
                           f"({None if row is None else row.get('custody')})")
    return resolved
