"""Grouped, family-level splits for the Jev bend-candidate dataset (C10, v1).

Splits come only from an explicit manifest, written before any answer is seen::

    {"schema": "jev_split_manifest_v1",
     "families": {"<sku family>": {"split": "dev"|"calib"|"eval"|"excluded",
                                   "flags": ["holdout"|"sealed"|"one_shot"|
                                             "consumed"|"retrospective", ...]}},
     "sku_to_family": {"<SKU>": "<sku family>"}}

Rules:

- A family is the unit of assignment, so no family can span two splits, and
  every SKU maps to exactly one family.
- A family flagged ``holdout``, ``sealed`` or ``one_shot`` is **excluded**,
  whatever split the manifest names (the override is recorded).
- A family flagged ``consumed`` or ``retrospective`` stays in its split, but any
  result that touches it is labelled ``exploratory_not_unseen``.
- A record whose SKU is not in ``sku_to_family`` is ``excluded``
  (``UNASSIGNED``), never guessed.
- Split names and family names never appear in a state (the state deny-list
  rejects ``split``/``family`` tokens and every string with a digit run; the
  harness test checks the values as well).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SPLIT_MANIFEST_SCHEMA",
    "SPLITS",
    "SENDABLE_SPLITS",
    "EXCLUDING_FLAGS",
    "EXPLORATORY_FLAGS",
    "SplitError",
    "SplitManifest",
    "load_split_manifest",
    "canonical_group_sku",
]

SPLIT_MANIFEST_SCHEMA = "jev_split_manifest_v1"
SPLITS = ("dev", "calib", "eval", "excluded")
#: Splits the live client may send without ``--include-eval``.
SENDABLE_SPLITS = ("dev", "calib")
EXCLUDING_FLAGS = frozenset({"holdout", "sealed", "one_shot"})
EXPLORATORY_FLAGS = frozenset({"consumed", "retrospective"})
KNOWN_FLAGS = EXCLUDING_FLAGS | EXPLORATORY_FLAGS


class SplitError(ValueError):
    """The split manifest is malformed or inconsistent."""


def canonical_group_sku(sku: str | None) -> str | None:
    """Uppercase, stripped, internal whitespace as '-' (the C01 canonical_sku shape)."""
    if sku is None:
        return None
    text = "-".join(str(sku).strip().split())
    return text.upper() or None


@dataclass(frozen=True)
class SplitManifest:
    family_split: Mapping[str, str]
    family_flags: Mapping[str, frozenset[str]]
    sku_to_family: Mapping[str, str]
    sha256: str
    overrides: tuple[str, ...] = field(default_factory=tuple)

    def family_of(self, sku: str | None) -> str | None:
        key = canonical_group_sku(sku)
        return None if key is None else self.sku_to_family.get(key)

    def split_of(self, sku: str | None) -> str:
        family = self.family_of(sku)
        return "excluded" if family is None else self.family_split[family]

    def is_exploratory(self, sku: str | None) -> bool:
        family = self.family_of(sku)
        return family is not None and bool(self.family_flags[family] & EXPLORATORY_FLAGS)

    def assign(self, rows: Iterable[Any]) -> dict[str, str]:
        """``{record_id: split}`` for label rows (anything with ``record_id`` and ``group_sku``)."""
        out: dict[str, str] = {}
        for row in rows:
            rid = row.record_id if hasattr(row, "record_id") else row["record_id"]
            sku = row.group_sku if hasattr(row, "group_sku") else row.get("group_sku")
            split = self.split_of(sku)
            if rid in out and out[rid] != split:
                raise SplitError(f"record {rid} assigned to two splits")
            out[rid] = split
        return out

    def families_by_split(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {s: [] for s in SPLITS}
        for family, split in sorted(self.family_split.items()):
            out[split].append(family)
        return out


def _parse(doc: Any, sha: str) -> SplitManifest:
    if not isinstance(doc, Mapping) or doc.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise SplitError(f"split manifest schema must be {SPLIT_MANIFEST_SCHEMA}")
    families = doc.get("families")
    mapping = doc.get("sku_to_family")
    if not isinstance(families, Mapping) or not isinstance(mapping, Mapping):
        raise SplitError("families and sku_to_family must be objects")
    family_split: dict[str, str] = {}
    family_flags: dict[str, frozenset[str]] = {}
    overrides: list[str] = []
    for family, spec in families.items():
        if not isinstance(family, str) or not family.strip():
            raise SplitError("family names must be non-empty strings")
        if not isinstance(spec, Mapping):
            raise SplitError(f"family {family!r} must be an object")
        split = spec.get("split")
        if split not in SPLITS:
            raise SplitError(f"family {family!r}: split must be one of {SPLITS}")
        flags = spec.get("flags", [])
        if not isinstance(flags, list) or any(f not in KNOWN_FLAGS for f in flags):
            raise SplitError(f"family {family!r}: flags must be a subset of {sorted(KNOWN_FLAGS)}")
        flag_set = frozenset(flags)
        if flag_set & EXCLUDING_FLAGS and split != "excluded":
            overrides.append(f"{family}: {split} -> excluded ({','.join(sorted(flag_set & EXCLUDING_FLAGS))})")
            split = "excluded"
        family_split[family] = split
        family_flags[family] = flag_set
    sku_to_family: dict[str, str] = {}
    for sku, family in mapping.items():
        key = canonical_group_sku(sku)
        if key is None:
            raise SplitError("empty SKU in sku_to_family")
        if family not in family_split:
            raise SplitError(f"SKU {sku!r} maps to unknown family {family!r}")
        if key in sku_to_family and sku_to_family[key] != family:
            raise SplitError(f"SKU {sku!r} maps to two families (after canonicalisation)")
        sku_to_family[key] = family
    return SplitManifest(family_split, family_flags, sku_to_family, sha, tuple(overrides))


def load_split_manifest(source: Path | Mapping[str, Any]) -> SplitManifest:
    """Parse a manifest from a path or an already-loaded object."""
    if isinstance(source, Mapping):
        raw = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return _parse(source, hashlib.sha256(raw).hexdigest())
    data = Path(source).read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitError(f"{source}: invalid JSON") from exc
    return _parse(doc, hashlib.sha256(data).hexdigest())
