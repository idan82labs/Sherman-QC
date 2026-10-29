"""Z2 shared plumbing: paths, DEV cohort, custody guard, family-grouped folds.

Custody: only DEV scans whose INVENTORY custody is DEV_CONSUMED / CONSUMED_DEVCHECK are opened
(the D1 guard, imported read-only). The 18 DEV scans whose gold lives under a *_SEALED* path
(CONSUMED_FRESH) are not opened or scored here; TEST-B is never listed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HUB = REPO / "docs/production/runlog/zeroshot_2026-10-25"
OUT = HUB / "z2"
for p in (str(REPO), str(REPO / "research/broad_baseline"), str(REPO / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.zeroshot.d1_two_skin.custody import check_path, ALLOWED_CUSTODY  # noqa: E402  (read-only reuse)
from research.zeroshot.common.gold_guard import require_runnable  # noqa: E402  (programme harness guard)

N_FOLDS = 5


def inventory() -> dict[str, dict]:
    return {r["scan_id"]: r for r in json.loads((HUB / "INVENTORY.json").read_text())["scans"]}


def dev_ids() -> list[str]:
    split = json.loads((HUB / "SPLIT_FROZEN.json").read_text())
    return list(split["dev"]["scans"])


def cohort() -> list[str]:
    """DEV scans this WP may open (custody-allowed, gold path not sealed)."""
    inv = inventory()
    out = []
    for sid in dev_ids():
        r = inv[sid]
        if not r.get("usable") or r.get("custody") not in ALLOWED_CUSTODY:
            continue
        if "sealed" in (r.get("gold") or {}).get("path", "").lower():
            continue
        out.append(sid)
    return sorted(out)


def scan_path(sid: str, inv: dict | None = None) -> Path:
    inv = inv or inventory()
    p = Path(inv[sid]["path"]).expanduser()
    require_runnable(sid)
    return check_path(p)


def folds(ids: list[str] | None = None) -> dict[str, int]:
    """Family-grouped 5-fold over the 5-digit family group (dev_folds.json if the integrator wrote it)."""
    f = HUB / "dev_folds.json"
    if f.is_file():
        d = json.loads(f.read_text())
        m = d.get("scan_fold") or {}
        if isinstance(m, dict) and m:
            return {k: int(v) for k, v in m.items()}
    inv = inventory()
    ids = ids or cohort()
    groups = sorted({inv[s]["family_group5"] for s in ids})
    # deterministic: sort groups by sha1, deal round-robin
    order = sorted(groups, key=lambda g: hashlib.sha1(("z2fold:" + g).encode()).hexdigest())
    gf = {g: i % N_FOLDS for i, g in enumerate(order)}
    return {s: gf[inv[s]["family_group5"]] for s in ids}
