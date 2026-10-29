"""The pinned zs_lane_v1: fingerprint and model checks (``assets/zs_lane_v1.pin.json``).

The pin copies the frozen research config's code fingerprint (``research/zeroshot/zs_lane_v1/config_v1.json``,
ledger PREREG seq 12): sha256 per file over every source file the lane executes -- its own files, the reused
Z3 / Z2 / D1 modules, the E3 metrology modules Z2 imports, the contracts, the guard and the eval writer -- and
the combined sha256 (sha256 over sorted ``path \\0 sha \\n`` lines, as ``research/zeroshot/zs_lane_v1/
fingerprint.py``). The selector model is vendored byte-identical at ``assets/zs_lane_v1_zs_gbm.pkl`` and
pinned by sha256; it is unpickled only after that check.

:func:`verify` recomputes everything under a root (the engine bundle, or the dev repo) and returns the
mismatches; an empty list is the only state in which the lane may run. Changing any pinned file (including an
E3 engine bump: Z2 reuses E3's metrology code) disables the lane until it is re-pinned, which is the intended
fail-closed behaviour.

Pin selection. Two pins exist: ``zs_lane_v1`` (the live default) and ``zs_lane_v3`` (``assets/zs_lane_v3.pin.json``,
zs_lane_v2 + EXT + Z3R, prepared for a re-pin after its fresh-scan test; it pins two models, the pooled selector
and the Z3-alone selector, and carries the frozen lane parameters). :data:`DEFAULT_LANE` decides which one runs;
it is ``zs_lane_v1`` so production behaviour is unchanged. A non-default pin is selected only explicitly, with
``lane=`` or the environment variable :data:`PIN_ENV` (rehearsals); an unknown value fails closed (no overlay).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PIN_REL = "domains/production/zeroshot/assets/zs_lane_v1.pin.json"
#: Every pin this package can run, by lane id.
PIN_RELS = {"zs_lane_v1": PIN_REL, "zs_lane_v3": "domains/production/zeroshot/assets/zs_lane_v3.pin.json"}
#: The pin production runs. Flip to "zs_lane_v3" only per docs/production/runlog/zeroshot_2026-10-25/v3/REPIN_PLAN.md.
DEFAULT_LANE = "zs_lane_v1"
#: Explicit, non-default pin selection (rehearsals only): the lane id, e.g. ``SHERMAN_ZS_LANE_PIN=zs_lane_v3``.
PIN_ENV = "SHERMAN_ZS_LANE_PIN"
PIN_SCHEMA = "sherman_zeroshot_lane_pin_v1"
#: This package's own sources: allowed to be loaded next to the pinned files.
OWN_PACKAGE = "domains/production/zeroshot/"

__all__ = ["PIN_REL", "PIN_RELS", "DEFAULT_LANE", "PIN_ENV", "PIN_SCHEMA", "root_of_package", "selected_lane",
           "pin_rel", "load_pin", "pinned_files", "pinned_models", "combined", "sha256_file", "verify",
           "unpinned_loaded"]


def root_of_package() -> Path:
    """The tree this package runs from (engine bundle or dev repo): ``<root>/domains/production/zeroshot``."""
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def combined(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(k.encode())
        h.update(b"\0")
        h.update(hashes[k].encode())
        h.update(b"\n")
    return h.hexdigest()


def selected_lane(lane: str | None = None) -> str:
    """The lane whose pin runs: ``lane`` if given, else :data:`PIN_ENV` if set, else :data:`DEFAULT_LANE`."""
    name = lane or os.environ.get(PIN_ENV, "").strip() or DEFAULT_LANE
    if name not in PIN_RELS:
        raise ValueError(f"unknown zero-shot lane pin {name!r} (known: {sorted(PIN_RELS)})")
    return name


def pin_rel(lane: str | None = None) -> str:
    return PIN_RELS[selected_lane(lane)]


def load_pin(root: Path | None = None, lane: str | None = None) -> dict[str, Any]:
    root = root or root_of_package()
    name = selected_lane(lane)
    pin = json.loads((root / PIN_RELS[name]).read_text(encoding="utf-8"))
    if pin.get("schema") != PIN_SCHEMA:
        raise ValueError(f"pin schema is {pin.get('schema')!r}, expected {PIN_SCHEMA}")
    if pin.get("lane") != name:
        raise ValueError(f"pin {PIN_RELS[name]} names lane {pin.get('lane')!r}, expected {name!r}")
    return pin


def pinned_models(pin: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The pin's model files by role: ``zs`` (the pooled selector) and, for v3, ``z3`` (the Z3-alone selector)."""
    out = {"zs": pin["model"]}
    out.update(pin.get("extra_models") or {})
    return out


def pinned_files(pin: dict[str, Any]) -> list[str]:
    """Every repo-relative file the pin covers (sources + the pin + the models)."""
    return sorted(set(pin["code_fingerprint"]["files"]) | {PIN_RELS[pin["lane"]]}
                  | {m["rel_path"] for m in pinned_models(pin).values()})


def verify(root: Path | None = None, pin: dict[str, Any] | None = None) -> list[str]:
    """Mismatches between the files under ``root`` and the pin (empty = the lane may run)."""
    root = root or root_of_package()
    try:
        pin = pin or load_pin(root)
    except (OSError, ValueError) as exc:
        return [f"PIN_UNREADABLE:{type(exc).__name__}"]
    bad: list[str] = []
    pinned = pin["code_fingerprint"]["files"]
    now: dict[str, str] = {}
    for rel, want in sorted(pinned.items()):
        p = root / rel
        if not p.is_file() or p.is_symlink():
            bad.append(f"CODE_MISSING:{rel}")
            continue
        now[rel] = sha256_file(p)
        if now[rel] != want:
            bad.append(f"CODE:{rel}")
    if not bad and combined(now) != pin["code_fingerprint"]["sha256"]:
        bad.append("CODE_FINGERPRINT")
    for m in pinned_models(pin).values():
        model = root / m["rel_path"]
        if not model.is_file() or model.is_symlink():
            bad.append(f"MODEL_MISSING:{m['rel_path']}")
        elif sha256_file(model) != m["sha256"]:
            bad.append(f"MODEL:{m['rel_path']}")
    return bad


def unpinned_loaded(root: Path | None = None, pin: dict[str, Any] | None = None) -> list[str]:
    """Source files under ``root`` loaded in this process that neither the pin nor this package covers."""
    root = (root or root_of_package()).resolve()
    pin = pin or load_pin(root)
    cover = set(pin["code_fingerprint"]["files"])
    out: set[str] = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.split("/", 1)[0] in (".venv", ".venv-f1-structured") or rel.startswith(OWN_PACKAGE):
            continue
        if rel not in cover:
            out.add(rel)
    return sorted(out)
