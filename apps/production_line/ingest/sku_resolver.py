"""Filename -> recipe resolution through C01's ``parse_scan_filename``.

:class:`FilenameRecipeResolver` implements the
:class:`~apps.production_line.ingest.ports.ScanRecipeResolver` port over an
:class:`~apps.production_line.ingest.ports.ActiveRecipeCatalog` (C05 provides
the catalog from its recipe store). It does not parse names itself: every
rule lives in ``domains.production.recipe.sku.parse_scan_filename``, which
handles the real ZEISS shapes (``BEFORE BENDING  11979003_C.ply``,
``23553000_B_FULL_SCAN 1.ply``, ``38172000_01_SCAN_2 1.ply``, ``(1)`` and
``- Copy`` markers) and never guesses.

Outcomes:

- parse OK, active recipe COMPILED with a compiled-part sha -> ``MATCHED``;
- parse OK but the SKU has no active recipe, or its compile is not COMPILED
  -> ``PART_NOT_READY``;
- ``UNMATCHED``, ``AMBIGUOUS``, ``UNPARSEABLE`` and ``REVISION_MISMATCH`` pass
  through (the hotfolder quarantines them with UNMATCHED_SKU, AMBIGUOUS_SKU,
  UNPARSEABLE_FILENAME and REVISION_MISMATCH).

The known-SKU map (SKU -> expected revision) is cached for ``cache_ttl_s``;
call :meth:`invalidate` after a release. This module imports C01 code, so the
hotfolder does not import it; the service wiring (C11) passes an instance in.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from apps.production_line.ingest.ports import ActiveRecipeCatalog, ActiveRecipeRef, ResolveOutcome, minimal_parse
from domains.production.recipe.sku import SkuError, compile_filename_pattern, parse_scan_filename

__all__ = ["FilenameRecipeResolver"]

_LOG = logging.getLogger(__name__)


class FilenameRecipeResolver:
    """``ScanRecipeResolver`` backed by C01 filename parsing (module docstring)."""

    def __init__(
        self,
        catalog: ActiveRecipeCatalog,
        pattern: str | re.Pattern[str] | None = None,
        *,
        cache_ttl_s: float = 10.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._catalog = catalog
        # Validate the operator regex up front (SkuError FILENAME_REGEX_INVALID).
        self._pattern = compile_filename_pattern(pattern) if pattern is not None else None
        self._ttl = float(cache_ttl_s)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._cache: tuple[float, dict[str, str | None], dict[str, ActiveRecipeRef]] | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def _snapshot(self) -> tuple[dict[str, str | None], dict[str, ActiveRecipeRef]]:
        now = self._monotonic()
        with self._lock:
            if self._cache is not None and now - self._cache[0] < self._ttl:
                return self._cache[1], self._cache[2]
        active = {ref.sku: ref for ref in self._catalog.list_active_recipes()}
        known: dict[str, str | None] = {sku: None for sku in self._catalog.list_registered_skus()}
        for sku, ref in active.items():
            known[sku] = ref.part_revision
        with self._lock:
            self._cache = (now, known, active)
        return known, active

    def resolve(self, original_filename: str) -> ResolveOutcome:
        try:
            known, active = self._snapshot()
            parsed = parse_scan_filename(original_filename, known, self._pattern)
        except SkuError as exc:
            # A catalog entry that is not a canonical SKU: never guess a match.
            _LOG.error("filename resolution failed for %r: %s", original_filename, exc)
            return ResolveOutcome(
                status="UNPARSEABLE",
                reason=f"CATALOG_SKU_INVALID: {exc}"[:200],
                parse=minimal_parse("UNPARSEABLE", reason="CATALOG_SKU_INVALID"),
            )
        parse = _parse_contract_dict(parsed)
        common: dict[str, Any] = {
            "sku": parsed.sku,
            "serial": parsed.serial,
            "scan_timestamp": parsed.scan_timestamp,
            "parse": parse,
        }
        if parsed.status != "OK":
            reason = parsed.reason
            if parsed.status == "AMBIGUOUS" and parsed.candidates:
                reason = f"{parsed.reason}: {', '.join(parsed.candidates)}"
            elif parsed.status == "REVISION_MISMATCH":
                reason = f"name revision {parsed.revision!r} != recipe revision {parsed.expected_revision!r}"
            return ResolveOutcome(status=parsed.status, reason=(reason or parsed.status)[:200], **common)
        ref = active.get(parsed.sku or "")
        if ref is None:
            return ResolveOutcome(status="PART_NOT_READY", reason="NO_ACTIVE_RECIPE", **common)
        if ref.compile_status != "COMPILED" or ref.compiled_part_sha256 is None:
            return ResolveOutcome(
                status="PART_NOT_READY",
                reason=f"RECIPE_COMPILE_{ref.compile_status}"[:200],
                recipe_id=ref.recipe_id,
                recipe_version=ref.recipe_version,
                recipe_sha256=ref.recipe_sha256,
                **common,
            )
        return ResolveOutcome(
            status="MATCHED",
            recipe_id=ref.recipe_id,
            recipe_version=ref.recipe_version,
            recipe_sha256=ref.recipe_sha256,
            compiled_part_sha256=ref.compiled_part_sha256,
            reason=None if parsed.revision_status in ("MATCH", None) else f"REVISION_{parsed.revision_status}",
            **common,
        )


def _parse_contract_dict(parsed: Any) -> dict[str, Any]:
    """``ScanNameParseV1`` JSON for the stored scan row (validated by the contract)."""
    return parsed.to_contract().model_dump(mode="json")
