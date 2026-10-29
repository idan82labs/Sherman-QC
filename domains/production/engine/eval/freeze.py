"""Freeze ledger (plan §10 amendment 4: "freeze before A1 starts, hashes recorded in the ledger").

The ledger is an append-only JSONL file. Each entry records the sha256 of what
is frozen and chains to the previous entry, so an edited or reordered ledger is
detected by :func:`verify_ledger`:

- **thresholds**: every document of :mod:`.thresholds` (version + sha256);
- **seeds**: per (tier, seed set) the planned case ids and seeds (E2
  ``plan_cases``; QUAL needs an explicit ``allow_qual`` -- planning seeds builds
  no geometry and reads no output);
- **recipes**: the DEV_AUTO_CONFIRMED recipe files (path + sha256);
- **case lists**: e.g. the G2 / G3 / DEV-CHECK lists (path + sha256 + count);
- **engine config** and **scanner model card** (path + sha256; the card also
  with E2's model sha256);
- **code fingerprints** of the engine, compile, eval and contracts packages;
- any extra file.

Entry kinds: FREEZE (the items above), RUN (one entry per QUAL run, written by
the runner before the first engine call, amendment 4c), WAIVER (an **owner**
decision accepting one named, waivable mismatch; ``approved_by`` and ``reason``
are required; agents never write waivers) and NOTE.

Entry hash: sha256 of the entry's canonical JSON without ``entry_sha256``;
``previous_sha256`` is the previous entry's ``entry_sha256`` (null for the
first). A chain only proves internal consistency (anyone can recompute it), so
the sha256 of the first FREEZE entry must also be recorded **outside** the
ledger (the run log / ``docs/CURRENT_STATUS.md``) and passed back as the
*anchor*; a scorecard is qualifying only when the anchor matches.

:func:`check_against_ledger` -- "thresholds never change mid-run":

- the **first** FREEZE entry that defines an item is authoritative for the
  pre-A1 items (thresholds, seeds, recipes, case lists, scanner card, eval and
  contracts code, extra files); a later FREEZE entry that redefines one with a
  different value is reported (``THRESHOLDS_REFROZEN:<doc>``,
  ``SEEDS_REFROZEN:...``, ...), and the current values are compared with the
  authoritative ones (``THRESHOLDS_CHANGED:<doc>``, ...);
- the eval and contracts code fingerprints must equal the frozen ones
  (``EVAL_CODE_CHANGED_SINCE_FREEZE`` / ``CONTRACTS_CODE_CHANGED_SINCE_FREEZE``,
  waivable by an owner WAIVER naming the new fingerprint);
- the engine config is A1's final knob: it is compared with the **latest**
  FREEZE entry that defines it (one config must pass every gate);
- QUAL RUN entries for the same tier with another engine identity set
  ``qual_seeds_reused`` (the QUAL seeds then evaluated two engines).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.production.canonical import sha256_hex

from .plyio import sha256_file
from .thresholds import ALL_THRESHOLDS, thresholds_sha256

__all__ = [
    "LEDGER_SCHEMA",
    "CODE_ROOTS",
    "FROZEN_CODE",
    "WAIVABLE_SCOPES",
    "freeze_items",
    "append_entry",
    "read_ledger",
    "verify_ledger",
    "latest_freeze",
    "first_freeze",
    "current_code_fingerprints",
    "check_against_ledger",
]

LEDGER_SCHEMA = "sherman_e4_freeze_ledger_v1"
_KINDS = ("FREEZE", "NOTE", "RUN", "WAIVER")
#: Code roots fingerprinted by a freeze (the CLI records all four).
CODE_ROOTS: dict[str, tuple[str, ...]] = {
    "engine": ("domains/production/engine/inspect",),
    "compile": ("domains/production/engine/compile",),
    "eval": ("domains/production/engine/eval",),
    "contracts": ("contracts/production",),
}
#: Code that must not change after the freeze: thresholds, scoring and gate logic, contracts.
FROZEN_CODE = ("eval", "contracts")
#: Mismatches an owner WAIVER entry may accept (never a threshold, seed or case-list change).
WAIVABLE_SCOPES = ("EVAL_CODE_CHANGED_SINCE_FREEZE", "CONTRACTS_CODE_CHANGED_SINCE_FREEZE", "SCANNER_MODEL_PARTIAL")
_UNCHECKED = object()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_item(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def _seed_item(tier: str, seedset: str, *, allow_qual: bool) -> dict[str, Any]:
    from domains.production.engine.synthetic.cases import GENERATOR_VERSION, plan_cases

    specs = plan_cases(tier, seedset, allow_qual=allow_qual)
    rows = [[s.case_id, s.seed, s.family, s.scenario] for s in specs]
    return {"tier": tier, "seedset": seedset, "generator_version": GENERATOR_VERSION, "count": len(rows),
            "case_ids_sha256": sha256_hex([r[0] for r in rows]), "seeds_sha256": sha256_hex(rows)}


def current_code_fingerprints(repo_root: str | os.PathLike | None = None,
                              names: Sequence[str] = tuple(CODE_ROOTS)) -> dict[str, str]:
    """Code fingerprints (:func:`.runner.code_fingerprint`) of the named :data:`CODE_ROOTS`."""
    from .runner import REPO_ROOT, code_fingerprint

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return {name: code_fingerprint(root, CODE_ROOTS[name]) for name in names}


def freeze_items(
    *,
    seeds: Sequence[tuple[str, str]] = (),
    allow_qual: bool = False,
    recipes: Sequence[str | os.PathLike] = (),
    case_lists: Mapping[str, str | os.PathLike] | None = None,
    engine_config: str | os.PathLike | None = None,
    scanner_model_card: str | os.PathLike | None = None,
    code_roots: Mapping[str, Sequence[str]] | None = None,
    repo_root: str | os.PathLike | None = None,
    extra_files: Mapping[str, str | os.PathLike] | None = None,
) -> dict[str, Any]:
    """The hashed items of one FREEZE entry (module docstring)."""
    items: dict[str, Any] = {
        "thresholds": {name: {"version": doc["version"], "sha256": thresholds_sha256(doc)}
                       for name, doc in ALL_THRESHOLDS.items()},
    }
    if seeds:
        items["seeds"] = [_seed_item(t, s, allow_qual=allow_qual) for t, s in seeds]
    if recipes:
        files = sorted((_file_item(r) for r in recipes), key=lambda d: d["path"])
        items["recipes"] = {"count": len(files), "list_sha256": sha256_hex([f["sha256"] for f in files]),
                            "files": files}
    if case_lists:
        lists = {}
        for name, path in sorted(case_lists.items()):
            entry = _file_item(path)
            try:
                doc = json.loads(Path(path).read_text(encoding="utf-8"))
                entry["count"] = len(doc if isinstance(doc, list) else doc.get("cases", []))
            except (OSError, ValueError):
                entry["count"] = None
            lists[name] = entry
        items["case_lists"] = lists
    if engine_config is not None:
        items["engine_config"] = _file_item(engine_config)
    if scanner_model_card is not None:
        card = _file_item(scanner_model_card)
        from domains.production.engine.synthetic.scanner import ScannerModel

        model = ScannerModel.from_json(scanner_model_card)
        card.update(model_sha256=model.sha256(), provenance=model.provenance)
        items["scanner_model_card"] = card
    if code_roots:
        from .runner import REPO_ROOT, code_fingerprint

        root = Path(repo_root) if repo_root is not None else REPO_ROOT
        items["code_fingerprints"] = {name: code_fingerprint(root, roots) for name, roots in sorted(code_roots.items())}
    if extra_files:
        items["files"] = {name: _file_item(path) for name, path in sorted(extra_files.items())}
    return items


def read_ledger(path: str | os.PathLike) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entry_hash(entry: Mapping[str, Any]) -> str:
    return sha256_hex({k: v for k, v in entry.items() if k != "entry_sha256"})


def _validate_items(kind: str, items: Mapping[str, Any]) -> None:
    if kind == "WAIVER":
        if items.get("scope") not in WAIVABLE_SCOPES:
            raise ValueError(f"a WAIVER scope must be one of {WAIVABLE_SCOPES}")
        if not str(items.get("approved_by") or "").strip() or not str(items.get("reason") or "").strip():
            raise ValueError("a WAIVER needs approved_by (the owner) and a reason")
        if items["scope"] == "SCANNER_MODEL_PARTIAL" and not (items.get("scanner_model_sha256") and items.get("fields")):
            raise ValueError("a SCANNER_MODEL_PARTIAL waiver names the scanner_model_sha256 and the fields")
        if items["scope"] != "SCANNER_MODEL_PARTIAL" and not items.get("fingerprint"):
            raise ValueError("a code-change waiver names the accepted code fingerprint")
    if kind == "RUN" and not (items.get("tier") and items.get("seedset") and items.get("case_ids_sha256")):
        raise ValueError("a RUN entry names tier, seedset and case_ids_sha256")


def append_entry(path: str | os.PathLike, *, label: str, items: Mapping[str, Any], kind: str = "FREEZE",
                 recorded_by: str = "e4a") -> dict[str, Any]:
    """Append one chained entry (refuses to append to a ledger that does not verify)."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    _validate_items(kind, items)
    check = verify_ledger(path)
    if not check["ok"]:
        raise ValueError(f"refusing to append to a broken ledger: {check['problems']}")
    entries = read_ledger(path)
    prev = entries[-1]["entry_sha256"] if entries else None
    entry = {"schema": LEDGER_SCHEMA, "seq": len(entries) + 1, "kind": kind, "label": label,
             "recorded_at": _now(), "recorded_by": recorded_by, "items": dict(items), "previous_sha256": prev}
    entry["entry_id"] = "frz_" + _entry_hash(entry)[:12]
    entry["entry_sha256"] = _entry_hash(entry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return entry


def verify_ledger(path: str | os.PathLike) -> dict[str, Any]:
    """Every entry's hash, sequence number and chain link."""
    problems: list[str] = []
    try:
        entries = read_ledger(path)
    except ValueError as exc:
        return {"ok": False, "entries": 0, "problems": [f"UNREADABLE:{exc}"]}
    prev = None
    for i, e in enumerate(entries, start=1):
        if e.get("schema") != LEDGER_SCHEMA:
            problems.append(f"{i}:SCHEMA")
        if e.get("seq") != i:
            problems.append(f"{i}:SEQUENCE")
        if e.get("previous_sha256") != prev:
            problems.append(f"{i}:CHAIN")
        if e.get("entry_sha256") != _entry_hash(e):
            problems.append(f"{i}:HASH")
        if e.get("kind") not in _KINDS:
            problems.append(f"{i}:KIND")
        prev = e.get("entry_sha256")
    first = next((e for e in entries if e.get("kind") == "FREEZE"), None)
    return {"ok": not problems, "entries": len(entries), "problems": problems,
            "first_freeze_sha256": None if first is None else first.get("entry_sha256")}


def latest_freeze(path: str | os.PathLike) -> dict[str, Any] | None:
    for e in reversed(read_ledger(path)):
        if e.get("kind") == "FREEZE":
            return e
    return None


def first_freeze(path: str | os.PathLike) -> dict[str, Any] | None:
    return next((e for e in read_ledger(path) if e.get("kind") == "FREEZE"), None)


def _flatten(items: Mapping[str, Any]) -> dict[str, tuple[str, str | None]]:
    """``item key -> (mismatch family, value sha)`` of the pre-A1 items of one FREEZE entry."""
    out: dict[str, tuple[str, str | None]] = {}
    for name, d in (items.get("thresholds") or {}).items():
        out[f"thresholds:{name}"] = ("THRESHOLDS", d.get("sha256"))
    for s in items.get("seeds") or ():
        out[f"seeds:{s['tier']}:{s['seedset']}"] = ("SEEDS", s.get("seeds_sha256"))
    if items.get("recipes"):
        out["recipes"] = ("RECIPES", items["recipes"].get("list_sha256"))
    for name, d in (items.get("case_lists") or {}).items():
        out[f"case_lists:{name}"] = ("CASE_LISTS", d.get("sha256"))
    if items.get("scanner_model_card"):
        out["scanner_model_card"] = ("SCANNER_CARD", items["scanner_model_card"].get("model_sha256"))
    for name in FROZEN_CODE:
        fp = (items.get("code_fingerprints") or {}).get(name)
        if fp is not None:
            out[f"code:{name}"] = ("CODE", fp)
    for name, d in (items.get("files") or {}).items():
        out[f"files:{name}"] = ("FILES", d.get("sha256"))
    return out


def check_against_ledger(
    path: str | os.PathLike,
    *,
    case_ids: Sequence[str] | None = None,
    tier: str | None = None,
    seedset: str | None = None,
    scanner_model_sha256: str | None = None,
    engine_config: str | os.PathLike | None = None,
    engine_config_sha256: Any = _UNCHECKED,
    code_fingerprints: Mapping[str, str] | None = None,
    engine_identity: Mapping[str, Any] | None = None,
    anchor_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare the current state with the ledger (module docstring).

    ``engine_config`` (a path) or ``engine_config_sha256`` (a sha, or None for
    "no config") enables the engine-config check; ``code_fingerprints`` (from
    :func:`current_code_fingerprints` over :data:`FROZEN_CODE`) the frozen-code
    check; ``engine_identity`` the QUAL reuse check; ``anchor_sha256`` the
    external anchor of the first FREEZE entry."""
    v = verify_ledger(path)
    base: dict[str, Any] = {"ok": False, "entry_id": None, "first_entry_sha256": None, "latest_freeze_entry_id": None,
                            "anchor_checked": False, "mismatches": [], "waived": [], "waivers": [], "qual_runs": [],
                            "qual_seeds_reused": False}
    if not v["ok"]:
        return dict(base, mismatches=[f"LEDGER_BROKEN:{p}" for p in v["problems"]])
    entries = read_ledger(path)
    waivers = [dict(e["items"], entry_id=e["entry_id"], recorded_at=e["recorded_at"])
               for e in entries if e.get("kind") == "WAIVER"]
    qual_runs = [dict(e["items"], entry_id=e["entry_id"], recorded_at=e["recorded_at"])
                 for e in entries if e.get("kind") == "RUN" and (e.get("items") or {}).get("seedset") == "QUAL"
                 and (tier is None or e["items"].get("tier") == tier)]
    reused = False
    if engine_identity is not None:
        ident = {k: engine_identity.get(k) for k in ("engine_fingerprint", "engine_config_sha256")}
        reused = any({k: r.get(k) for k in ident} != ident for r in qual_runs)
    base.update(waivers=waivers, qual_runs=qual_runs, qual_seeds_reused=reused)
    freezes = [e for e in entries if e.get("kind") == "FREEZE"]
    if not freezes:
        return dict(base, mismatches=["NO_FREEZE_ENTRY"])
    first, latest = freezes[0], freezes[-1]
    mism: list[str] = []
    anchor_checked = False
    if anchor_sha256 is not None:
        anchor_checked = True
        if anchor_sha256 != first["entry_sha256"]:
            mism.append("LEDGER_ANCHOR_MISMATCH")

    # authority: the first definition of every pre-A1 item; later redefinitions are re-freezes
    authority: dict[str, tuple[str, str | None]] = {}
    for e in freezes:
        for key, (family, value) in _flatten(e["items"]).items():
            if key not in authority:
                authority[key] = (family, value)
            elif authority[key][1] != value:
                mism.append(f"{family}_REFROZEN:{key.split(':', 1)[-1]}")

    for name, doc in ALL_THRESHOLDS.items():
        frozen = authority.get(f"thresholds:{name}")
        if frozen is None:
            mism.append(f"THRESHOLDS_NOT_FROZEN:{name}")
        elif frozen[1] != thresholds_sha256(doc):
            mism.append(f"THRESHOLDS_CHANGED:{name}")
    if case_ids is not None:
        seed_items = [s for e in freezes for s in e["items"].get("seeds", ())
                      if (tier is None or s["tier"] == tier) and (seedset is None or s["seedset"] == seedset)]
        if not seed_items:
            mism.append("SEEDS_NOT_FROZEN")
        else:
            frozen_ids = {s["case_ids_sha256"] for s in seed_items
                          if authority.get(f"seeds:{s['tier']}:{s['seedset']}", (None, None))[1] == s["seeds_sha256"]}
            if sha256_hex(list(case_ids)) not in frozen_ids:
                mism.append("CASE_LIST_DIFFERS_FROM_FROZEN")
    if scanner_model_sha256 is not None:
        frozen = authority.get("scanner_model_card")
        if frozen is None:
            mism.append("SCANNER_MODEL_NOT_FROZEN")
        elif frozen[1] != scanner_model_sha256:
            mism.append("SCANNER_MODEL_DIFFERS_FROM_FROZEN")
    if engine_config is not None:
        engine_config_sha256 = sha256_file(engine_config)
    if engine_config_sha256 is not _UNCHECKED:
        frozen_cfg = next((e["items"]["engine_config"]["sha256"] for e in reversed(freezes)
                           if e["items"].get("engine_config")), None)
        if frozen_cfg != engine_config_sha256:
            mism.append("ENGINE_CONFIG_DIFFERS_FROM_FROZEN")

    waived: list[str] = []
    if code_fingerprints is not None:
        for name, fp in sorted(code_fingerprints.items()):
            frozen = authority.get(f"code:{name}")
            if frozen is None:
                mism.append(f"CODE_NOT_FROZEN:{name}")
                continue
            if frozen[1] == fp:
                continue
            code = f"{name.upper()}_CODE_CHANGED_SINCE_FREEZE"
            if any(w.get("scope") == code and w.get("fingerprint") == fp for w in waivers):
                waived.append(code)
            else:
                mism.append(code)

    return {"ok": not mism, "entry_id": first["entry_id"], "first_entry_sha256": first["entry_sha256"],
            "latest_freeze_entry_id": latest["entry_id"], "anchor_checked": anchor_checked,
            "mismatches": sorted(set(mism)), "waived": waived, "waivers": waivers, "qual_runs": qual_runs,
            "qual_seeds_reused": reused}
