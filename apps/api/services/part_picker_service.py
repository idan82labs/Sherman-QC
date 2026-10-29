"""Station part picker (C07 extension): held-back scans, part suggestions,
released-part search and the operator's stage declaration.

When the hotfolder cannot resolve a scan to a released, compiled recipe the
scan is quarantined (UNMATCHED_SKU, AMBIGUOUS_SKU, UNPARSEABLE_FILENAME,
REVISION_MISMATCH, PART_NOT_READY). The station screen lists those scans and
lets an operator pick the part; :func:`production_line_service.assign_quarantine`
then validates the pick exactly like the automatic path and enqueues the
inspect job. This module only *reads*:

- :func:`held_scans`: every open quarantined scan of a station with the
  filename, the resolver outcome and reason, the parsed SKU/revision tokens,
  the scan stats and **suggested parts**. Suggestions come from the resolver
  (AMBIGUOUS candidates, the parsed SKU of a REVISION_MISMATCH /
  PART_NOT_READY scan) and from a fuzzy match of the filename tokens against
  released SKUs. They are operator hints only: nothing here is assigned
  automatically and nothing reaches the engine (plan §10 amendment 4). No
  shape-based suggestions.
- :func:`released_parts`: the SKUs whose active recipe is released, searchable
  by SKU prefix / substring or part name. Retired recipes never appear; a
  released recipe that is not compiled is listed with ``assignable: false``.
- :func:`declared_stage_for`: maps the operator's FINAL / IN_PROCESS choice to
  the contract's ``DeclaredStageV1`` (``stage_index`` = bends formed, source
  OPERATOR). The recipe's ``inspection_stage`` is part of the hashed recipe
  and is never changed here; a declaration is only ever *checked* by E3
  (disagreement is STAGE_MISMATCH, review), never trusted.

Import discipline (as :mod:`production_line_service`): no legacy server, no
``apps.production_line.recipes`` import at import time.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.production_line.ingest.ports import ActiveRecipeCatalog, ActiveRecipeRef, RecipeProvider, ResolvedRecipe
from apps.production_line.settings import LineSettings
from apps.production_line.store.line_store import LineStore

__all__ = [
    "RecipeFacts",
    "recipe_facts",
    "released_parts",
    "suggest_parts",
    "held_scans",
    "declared_stage_for",
    "DISMISS_REASONS",
    "reset_picker_caches",
]

logger = logging.getLogger("apps.production_line.api.picker")

#: Operator dismiss reasons (station picker). ``OTHER`` needs a note.
DISMISS_REASONS: tuple[str, ...] = ("RESCAN", "NOT_A_PART", "DUPLICATE", "WRONG_STATION", "OTHER")
_MAX_SUGGESTIONS = 5
_FUZZY_MIN = 0.75
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_STAGES = ("FINAL", "IN_PROCESS")


class PickerError(Exception):
    """A refused picker request (mapped to ``LineApiError`` by the caller)."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RecipeFacts:
    """Display facts of one recipe, read from its frozen bytes and compiled part."""

    recipe_id: str
    sku: str
    recipe_version: int
    part_name: str | None
    part_revision: str | None
    inspection_stage: str | None
    bends_total: int | None
    lifecycle: str | None
    compile_status: str | None


class _FactsCache:
    def __init__(self, capacity: int = 512) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[str, str, str | None], RecipeFacts] = OrderedDict()
        self.capacity = capacity

    def get(self, key: tuple[str, str, str | None]) -> RecipeFacts | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: tuple[str, str, str | None], facts: RecipeFacts) -> None:
        with self._lock:
            self._data[key] = facts
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_FACTS = _FactsCache()


def reset_picker_caches() -> None:
    _FACTS.clear()


def _bends_in(path: Path) -> int | None:
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    bends = doc.get("bends") if isinstance(doc, dict) else None
    return len(bends) if isinstance(bends, list) else None


def facts_of(resolved: ResolvedRecipe) -> RecipeFacts:
    """Facts of a loaded recipe (bends from the compiled part when installed, else the recipe)."""
    key = (resolved.recipe_id, resolved.recipe_sha256, resolved.compiled_part_sha256)
    hit = _FACTS.get(key)
    extra = resolved.extra or {}
    if hit is not None and hit.lifecycle == extra.get("status") and hit.compile_status == extra.get("compile_status"):
        return hit
    doc: dict[str, Any] = {}
    try:
        loaded = json.loads(resolved.recipe_json_bytes or b"{}")
        doc = loaded if isinstance(loaded, dict) else {}
    except ValueError:
        doc = {}
    bends: int | None = None
    if resolved.compiled_dir is not None:
        bends = _bends_in(Path(resolved.compiled_dir) / "compiled_part.json")
    if bends is None and isinstance(doc.get("bends"), list):
        bends = len(doc["bends"])
    stage = doc.get("inspection_stage")
    facts = RecipeFacts(
        recipe_id=resolved.recipe_id,
        sku=resolved.sku,
        recipe_version=int(resolved.recipe_version),
        part_name=(resolved.part_name or doc.get("part_name") or None),
        part_revision=(resolved.part_revision or doc.get("part_revision") or None),
        inspection_stage=stage if stage in _STAGES else None,
        bends_total=bends,
        lifecycle=extra.get("status"),
        compile_status=extra.get("compile_status"),
    )
    _FACTS.put(key, facts)
    return facts


def recipe_facts(provider: RecipeProvider, recipe_id: str) -> RecipeFacts | None:
    try:
        return facts_of(provider.load(recipe_id))
    except KeyError:
        return None
    except Exception as exc:  # a broken recipe row never breaks the picker list
        logger.error("recipe facts failed for %s: %s", recipe_id, type(exc).__name__)
        return None


def _norm(text: str) -> str:
    return "".join(_ALNUM.findall(text or "")).upper()


def _active_refs(provider: RecipeProvider) -> list[ActiveRecipeRef]:
    if not isinstance(provider, ActiveRecipeCatalog):
        raise PickerError(503, "RECIPE_CATALOG_UNAVAILABLE", "the recipe store cannot list active recipes")
    try:
        return list(provider.list_active_recipes())
    except Exception as exc:
        logger.error("recipe catalog failed: %s", type(exc).__name__)
        raise PickerError(503, "RECIPE_PROVIDER_ERROR", "the recipe store could not list active recipes") from None


def _versions(provider: RecipeProvider, sku: str, active: ActiveRecipeRef) -> list[dict[str, Any]]:
    lister = getattr(provider, "list_recipe_versions", None)
    rows: list[dict[str, Any]] = []
    if callable(lister):
        try:
            rows = [dict(r) for r in lister(sku)]
        except Exception as exc:
            logger.error("recipe versions failed for %s: %s", sku, type(exc).__name__)
            rows = []
    if not any(r.get("recipe_id") == active.recipe_id for r in rows):
        rows.append({"recipe_id": active.recipe_id, "recipe_version": active.recipe_version, "status": "released"})
    out = []
    for r in sorted(rows, key=lambda r: -int(r.get("recipe_version") or 0)):
        is_active = r.get("recipe_id") == active.recipe_id
        out.append(
            {
                "recipe_id": r.get("recipe_id"),
                "recipe_version": int(r.get("recipe_version") or 0),
                "status": "active" if is_active else ("retired" if r.get("status") == "retired" else "superseded"),
                "released_at": r.get("released_at"),
                "selectable": is_active,
            }
        )
    return out


def _part_entry(provider: RecipeProvider, ref: ActiveRecipeRef, *, with_versions: bool) -> dict[str, Any]:
    facts = recipe_facts(provider, ref.recipe_id)
    compiled = ref.compile_status == "COMPILED" and ref.compiled_part_sha256 is not None
    released = facts is None or facts.lifecycle in (None, "released")
    entry = {
        "sku": ref.sku,
        "recipe_id": ref.recipe_id,
        "recipe_version": int(ref.recipe_version),
        "revision": (facts.part_revision if facts else None) or ref.part_revision,
        "part_name": facts.part_name if facts else None,
        "inspection_stage": facts.inspection_stage if facts else None,
        "bends_total": facts.bends_total if facts else None,
        "compile_status": ref.compile_status,
        "assignable": bool(compiled and released and facts is not None),
        "not_ready_reason": None if compiled else f"RECIPE_COMPILE_{ref.compile_status}",
        "preview_url": None,  # no part thumbnails exist in the recipe store yet
    }
    if with_versions:
        entry["versions"] = _versions(provider, ref.sku, ref)
    return entry


def _rank(entry: Mapping[str, Any], q: str) -> tuple[int, str] | None:
    if not q:
        return (0, entry["sku"])
    nq = _norm(q)
    sku = _norm(entry["sku"])
    name = str(entry.get("part_name") or "").casefold()
    if nq and sku.startswith(nq):
        return (0, entry["sku"])
    if nq and nq in sku:
        return (1, entry["sku"])
    if q.strip().casefold() and q.strip().casefold() in name:
        return (2, entry["sku"])
    return None


def released_parts(provider: RecipeProvider, *, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Released parts (each SKU's active, non-retired recipe), filtered and ranked by ``query``:
    SKU prefix first, then SKU substring, then part-name substring."""
    q = (query or "").strip()[:64]
    refs = _active_refs(provider)
    ranked: list[tuple[tuple[int, str], dict[str, Any]]] = []
    for ref in refs:
        probe = {"sku": ref.sku, "part_name": None}
        rank = _rank(probe, q)
        if rank is None and q:
            facts = recipe_facts(provider, ref.recipe_id)
            rank = _rank({"sku": ref.sku, "part_name": facts.part_name if facts else None}, q)
        if rank is None:
            continue
        ranked.append((rank, _part_entry(provider, ref, with_versions=True)))
    ranked.sort(key=lambda r: (r[0][0], not r[1]["assignable"], r[0][1]))
    return [e for _, e in ranked[: max(1, min(int(limit), 200))]]


def _name_tokens(original_name: str) -> list[str]:
    stem = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    toks = [t.upper() for t in _ALNUM.findall(stem)]
    out = list(toks)
    out.extend(a + b for a, b in zip(toks, toks[1:]))  # "10839 000" -> "10839000"
    return out


def suggest_parts(scan: Mapping[str, Any], refs: Iterable[ActiveRecipeRef]) -> list[dict[str, Any]]:
    """Ordered SKU suggestions for a held scan: resolver candidates, then the parsed
    SKU, then fuzzy filename matches (score >= 0.75). Only SKUs with an active
    recipe are suggested; ``score`` is a display hint, never a decision."""
    by_sku = {r.sku: r for r in refs}
    parse = scan.get("parse") or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(sku: str | None, source: str, score: float) -> None:
        if not sku or sku in seen or sku not in by_sku:
            return
        seen.add(sku)
        out.append({"sku": sku, "recipe_id": by_sku[sku].recipe_id, "source": source, "score": round(score, 3)})

    for cand in parse.get("candidates") or []:
        add(str(cand), "RESOLVER_CANDIDATE", 1.0)
    add(parse.get("sku"), "FILENAME_SKU", 1.0)
    tokens = _name_tokens(str(scan.get("original_name") or ""))
    fuzzy: list[tuple[float, str]] = []
    for sku in by_sku:
        if sku in seen:
            continue
        target = _norm(sku)
        if not target:
            continue
        best = 0.0
        for tok in tokens:
            if tok == target:
                best = 1.0
                break
            if len(tok) >= 4 and (target.startswith(tok) or tok.startswith(target)):
                best = max(best, min(len(tok), len(target)) / max(len(tok), len(target)))
            best = max(best, difflib.SequenceMatcher(None, tok, target).ratio())
        if best >= _FUZZY_MIN:
            fuzzy.append((best, sku))
    for score, sku in sorted(fuzzy, key=lambda f: (-f[0], f[1])):
        add(sku, "FILENAME_FUZZY", score)
    return out[:_MAX_SUGGESTIONS]


def held_scans(
    store: LineStore,
    provider: RecipeProvider | None,
    settings: LineSettings,
    station_id: str,
    principal_kind: str,
) -> list[dict[str, Any]]:
    """Open quarantined scans of a station, newest first, with suggestions."""
    from apps.api.services.production_line_service import quarantine_view

    items = store.list_quarantine(station_id=station_id)
    refs: list[ActiveRecipeRef] = []
    catalog_error: str | None = None
    if items:
        try:
            refs = _active_refs(provider) if provider is not None else []
        except PickerError as exc:
            catalog_error = exc.code
    entries = []
    by_sku = {r.sku: r for r in refs}
    for item in items:
        view = quarantine_view(item, settings, principal_kind)
        scan = view["scan_file"]
        parse = scan.get("parse") or {}
        match = scan.get("match") or {}
        suggestions = suggest_parts(scan, refs) if view["resolvable"] else []
        for s in suggestions:
            ref = by_sku[s["sku"]]
            entry = _part_entry(provider, ref, with_versions=False) if provider is not None else {}
            s.update({k: entry.get(k) for k in ("recipe_version", "revision", "part_name", "inspection_stage", "assignable")})
        entries.append(
            {
                "scan_file_id": scan["scan_file_id"],
                "station_id": scan["station_id"],
                "file_name": scan["original_name"],
                "received_at": scan["received_at"],
                "quarantined_at": view.get("quarantined_at"),
                "reason_code": view["quarantine_reason"],
                "resolvable": view["resolvable"],
                "message": view["message"],
                "outcome": match.get("status"),
                "outcome_reason": match.get("reason"),
                "parsed": {
                    k: parse.get(k)
                    for k in ("status", "sku", "revision", "expected_revision", "revision_status", "serial", "scan_label", "candidates")
                },
                "stats": {
                    "size_bytes": scan.get("size_bytes"),
                    "format": scan.get("format"),
                    "vertex_count": scan.get("vertex_count"),
                    "face_count": scan.get("face_count"),
                    "bbox_mm": scan.get("bbox_mm"),
                },
                "declared_stage": scan.get("declared_stage"),
                "suggestions": suggestions,
                "catalog_error": catalog_error,
            }
        )
    return entries


def declared_stage_for(facts: RecipeFacts, stage: str | None, stage_index: int | None) -> dict[str, Any] | None:
    """The operator's stage choice as a ``DeclaredStageV1`` dict, or ``None`` (no declaration).

    - nothing chosen, or the recipe's own ``inspection_stage`` with no index: ``None``
      (the scan keeps any sidecar declaration; the recipe policy applies);
    - FINAL: every bend formed, ``stage_index = bends_total``;
    - IN_PROCESS with ``stage_index`` (0..bends_total): that many bends formed.
      IN_PROCESS without an index is accepted only when the recipe itself is
      IN_PROCESS (E3 infers the stage); on a FINAL recipe it needs the index,
      because the recipe policy cannot be changed at the station.
    """
    if stage is None and stage_index is None:
        return None
    if stage is not None and stage not in _STAGES:
        raise PickerError(422, "DECLARED_STAGE_INVALID", "declared_stage must be FINAL or IN_PROCESS")
    total = facts.bends_total
    if stage_index is not None and (isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0):
        raise PickerError(422, "STAGE_INDEX_INVALID", "stage_index must be a non-negative integer")
    if stage_index is not None and total is not None and stage_index > total:
        raise PickerError(422, "STAGE_INDEX_INVALID", f"stage_index {stage_index} exceeds the {total} bends")
    if stage == "FINAL":
        if stage_index is not None and total is not None and stage_index != total:
            raise PickerError(422, "STAGE_INDEX_INVALID", "FINAL means every bend is formed")
        if facts.inspection_stage == "FINAL" and stage_index is None:
            return None
        if total is None:
            raise PickerError(409, "STAGE_UNAVAILABLE", "the recipe's bend count is unknown; cannot declare FINAL")
        return {"stage_index": total, "source": "OPERATOR"}
    # IN_PROCESS (or a bare index)
    if stage_index is None:
        if facts.inspection_stage == "IN_PROCESS":
            return None
        raise PickerError(422, "STAGE_INDEX_REQUIRED", "IN_PROCESS on a FINAL recipe needs the number of bends formed")
    return {"stage_index": int(stage_index), "source": "OPERATOR"}
