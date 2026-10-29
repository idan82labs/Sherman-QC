"""The frozen zs_lane_v3 revision 2 Python, independent of the live research/zeroshot/zs_lane_v3 package.

Revision 3 is built in place in research/zeroshot/zs_lane_v3 (budget.py and core.py changed at 23:45 on
2026-10-25), so the reference that zs_rust must equal is kept here as byte copies (rev2_snapshot/), each checked
against the revision-2 config's code fingerprint (config sha256 1c3f8cb7..., code fingerprint ab31d061...).
``install()`` makes ``research.zeroshot.zs_lane_v3.<module>`` resolve to those copies for the rest of the process
(the package ``__init__`` is unchanged between revisions), so the point-cloud fallback, the equivalence tests and
the harness run exactly revision 2. Call it before anything imports a zs_lane_v3 submodule.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SNAP = Path(__file__).resolve().parent / "rev2_snapshot"
REV2_CONFIG = SNAP / "config_v3.json"
REV2_CONFIG_SHA = "1c3f8cb7787185f33a61922dda8a2fced900c2e3011ea0105612f4fd5cc0fc53"
REV2_FINGERPRINT = "ab31d061c6231f5f7581f4cf03d451e77dbc5667e0193e66f3acac955e30a03d"
PKG = "research.zeroshot.zs_lane_v3"
# import order: leaves first, so each module's own `from research.zeroshot.zs_lane_v3 import x` finds the copy
MODULES = ["budget", "fixes", "z3_merge", "core", "fingerprint", "lane", "dev_dump", "train", "rev2_eval"]
# not in the revision-2 code fingerprint, pinned here (train.py as rev2_eval checks it)
EXTRA_SHA = {"train.py": "cce19ecfe675",
             "rev2_eval.py": "5374f030fc19", "dev_dump.py": "5e3eb26468fb"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pinned() -> dict[str, str]:
    raw = REV2_CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REV2_CONFIG_SHA:
        raise RuntimeError("rev2_snapshot/config_v3.json is not the frozen revision 2 config")
    cfg = json.loads(raw)
    if cfg["code_fingerprint"]["sha256"] != REV2_FINGERPRINT:
        raise RuntimeError("revision 2 config carries an unexpected code fingerprint")
    return cfg["code_fingerprint"]["files"]


def verify() -> None:
    files = pinned()
    for m in MODULES:
        p = SNAP / f"{m}.py"
        rel = f"research/zeroshot/zs_lane_v3/{m}.py"
        h = _sha(p)
        want = files.get(rel)
        if want is not None and h != want:
            raise RuntimeError(f"{p} differs from revision 2 ({rel})")
        if want is None and not h.startswith(EXTRA_SHA[f"{m}.py"]):
            raise RuntimeError(f"{p} differs from the pinned copy")
    live_init = Path(__file__).resolve().parents[1] / "zs_lane_v3" / "__init__.py"
    if _sha(live_init) != files["research/zeroshot/zs_lane_v3/__init__.py"]:
        raise RuntimeError("research/zeroshot/zs_lane_v3/__init__.py changed since revision 2")


def resolve(rel: str) -> Path:
    """Repo-relative path of a revision-2 file -> the file holding its revision-2 bytes."""
    from research.zeroshot.zs_lane_v3_rs.rs import REPO
    pre = "research/zeroshot/zs_lane_v3/"
    if rel.startswith(pre) and (SNAP / rel[len(pre):]).is_file():
        return SNAP / rel[len(pre):]
    return REPO / rel


def install() -> None:
    if getattr(sys.modules.get(PKG), "_rev2_snapshot", False):
        return
    verify()
    loaded = [m for m in MODULES if f"{PKG}.{m}" in sys.modules]
    if loaded:
        raise RuntimeError(f"live zs_lane_v3 modules already imported: {loaded}")
    pkg = importlib.import_module(PKG)
    for m in MODULES:
        name = f"{PKG}.{m}"
        spec = importlib.util.spec_from_file_location(name, SNAP / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        setattr(pkg, m, mod)
        spec.loader.exec_module(mod)
    # revision 2's own fingerprint reads files by repo path: point it at the revision-2 bytes
    fp = sys.modules[f"{PKG}.fingerprint"]
    fp.file_hashes = lambda files: {f: fp.sha256_file(resolve(f)) for f in sorted(files)}
    pkg._rev2_snapshot = True
