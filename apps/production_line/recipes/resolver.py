"""Scan filename -> active recipe, recipe provider and compiled-part sink (C05 side of the C02 ports).

:class:`LineRecipeResolver` implements, over a :class:`~.store.RecipeStore`:

- ``ActiveRecipeCatalog`` (``list_active_recipes``, ``list_registered_skus``):
  every SKU's latest non-retired recipe with its compile status and expected
  revision, and every registered SKU (plus, optionally, the canonical part
  numbers of the part catalog, so a scan of a catalogued-but-unreleased part
  is PART_NOT_READY rather than UNMATCHED);
- ``ScanRecipeResolver`` (``resolve``): delegates to C02's
  :class:`~apps.production_line.ingest.sku_resolver.FilenameRecipeResolver`,
  which calls C01's ``parse_scan_filename`` (the real ZEISS shapes, copy
  markers, REVISION_MISMATCH against the recipe's ``part_revision``). The
  known-SKU snapshot is cached for 10 s and dropped on every release or
  retirement (:func:`.release_service.add_release_listener`);
- ``RecipeProvider`` (``load``): the frozen canonical recipe bytes and the
  absolute CAD, flat and compiled-part paths under the data root. With
  ``compiled_part_sha256=`` it returns the directory of *that* compile, also
  when a recompute has superseded it (kept under ``recipes/compiled_history/``),
  so a job pinned to it can still run;
- ``CompiledPartSink`` (``store_compiled``), for ``recipe_compile`` jobs:

  - COMPILED, same ``compiled_part_sha256`` as the install: kept, record
    refreshed;
  - COMPILED, new sha: the previous COMPILED install is moved to
    ``recipes/compiled_history/<recipe_sha256>/<old sha>/`` (history row
    SUPERSEDED), the new output is installed at
    ``recipes/compiled/<recipe_sha256>/``, and the READY expectations and the
    benchmark nominals move to it in the same store transaction (the
    benchmark row always names an artifact that exists);
  - BLOCKED while a COMPILED install exists: the working install, its
    record, expectations and benchmark row are **kept** (a newer E1 that
    blocks never takes a released SKU off the line by itself). The blocked
    output is kept under ``recipes/compiled_history/`` (history row
    RECOMPUTE_BLOCKED) and logged as a compile attempt of the part, so the
    wizard shows it;
  - BLOCKED with no COMPILED install: installed and recorded BLOCKED (there
    was nothing working to keep).

Outcomes of :meth:`LineRecipeResolver.resolve`: MATCHED (active recipe
COMPILED), PART_NOT_READY (registered or catalogued SKU without an active
recipe, only retired versions, or a compile that is not COMPILED), UNMATCHED,
AMBIGUOUS, UNPARSEABLE, REVISION_MISMATCH. Parsed filename fields never reach
the engine (plan §10 amendment 4).

Factories :func:`build_recipe_provider` and :func:`build_scan_resolver` build
the resolver from :class:`~apps.production_line.settings.LineSettings`
(imported lazily by C07 / C11).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from apps.production_line.ingest.ports import (
    ActiveRecipeRef,
    ResolvedRecipe,
    ResolveOutcome,
)
from apps.production_line.recipes.store import CompiledRecord, RecipeStore

__all__ = ["LineRecipeResolver", "build_recipe_provider", "build_scan_resolver", "catalog_sku_source"]

_LOG = logging.getLogger(__name__)


def catalog_sku_source(catalog: Any) -> Callable[[], list[str]]:
    """Canonical SKUs of every catalog part (invalid part numbers skipped, never guessed)."""
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    def _source() -> list[str]:
        out: set[str] = set()
        for part in catalog.list_parts(limit=100000):
            try:
                out.add(canonical_sku(str(part.get("part_number") or "")))
            except SkuError:
                continue
        return sorted(out)

    return _source


class LineRecipeResolver:
    """``ActiveRecipeCatalog`` + ``ScanRecipeResolver`` + ``RecipeProvider`` +
    ``CompiledPartSink`` over the recipe store (module docstring)."""

    def __init__(
        self,
        recipe_store: RecipeStore,
        pattern: str | re.Pattern[str] | None = None,
        *,
        data_root: Path,
        extra_skus: Callable[[], Iterable[str]] | None = None,
        cache_ttl_s: float = 10.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        from apps.production_line.ingest.sku_resolver import FilenameRecipeResolver

        self.store = recipe_store
        self.data_root = Path(data_root)
        self._extra_skus = extra_skus
        self._filename = FilenameRecipeResolver(self, pattern, cache_ttl_s=cache_ttl_s, monotonic=monotonic)

    # -- ActiveRecipeCatalog ------------------------------------------------------
    def list_active_recipes(self) -> list[ActiveRecipeRef]:
        refs = []
        for stored in self.store.list_active():
            compiled = self.store.get_compiled(stored.recipe_sha256)
            status = compiled.status if compiled is not None else "PENDING"
            try:
                revision = stored.recipe().part_revision
            except ValueError:
                _LOG.error("stored recipe %s does not validate; treated as not ready", stored.recipe_id)
                refs.append(ActiveRecipeRef(stored.sku, stored.recipe_id, stored.version, stored.recipe_sha256,
                                            None, None, "FAILED"))
                continue
            refs.append(
                ActiveRecipeRef(
                    sku=stored.sku,
                    recipe_id=stored.recipe_id,
                    recipe_version=stored.version,
                    recipe_sha256=stored.recipe_sha256,
                    part_revision=revision,
                    compiled_part_sha256=compiled.compiled_part_sha256 if status == "COMPILED" and compiled else None,
                    compile_status=status,
                )
            )
        return refs

    def list_registered_skus(self) -> list[str]:
        from domains.production.recipe.canonical_sku import is_canonical_sku

        skus = set(self.store.list_registered_skus())
        skus.update(r.sku for r in self.store.list_all())
        if self._extra_skus is not None:
            try:
                skus.update(s for s in self._extra_skus() if is_canonical_sku(s))
            except Exception:
                _LOG.exception("catalog SKU source failed; using the registry only")
        return sorted(skus)

    def list_recipe_versions(self, sku: str) -> list[dict[str, Any]]:
        """Every stored version of ``sku`` (released or retired), oldest first (station part picker)."""
        return [
            {"recipe_id": r.recipe_id, "recipe_version": r.version, "status": r.status, "released_at": r.released_at}
            for r in self.store.list_versions(sku)
        ]

    # -- ScanRecipeResolver ----------------------------------------------------------
    def resolve(self, original_filename: str) -> ResolveOutcome:
        """Resolve a scan's original filename (never raises)."""
        try:
            return self._filename.resolve(original_filename)
        except Exception as exc:  # the port contract: never raise for any filename
            _LOG.exception("scan resolution failed for %r", original_filename)
            return ResolveOutcome(status="UNPARSEABLE", reason=f"RESOLVER_ERROR: {type(exc).__name__}"[:200])

    def invalidate(self, *_: Any) -> None:
        """Drop the cached SKU snapshot (called after every release / retirement)."""
        self._filename.invalidate()

    # -- RecipeProvider ------------------------------------------------------------------
    def load(self, recipe_id: str, *, compiled_part_sha256: str | None = None) -> ResolvedRecipe:
        """The recipe with absolute paths. ``compiled_part_sha256`` selects a pinned
        compile (installed or kept in history); when it is unknown the installed
        compile is returned and the runner's hash check refuses it."""
        stored = self.store.get(recipe_id)
        if stored is None:
            raise KeyError(recipe_id)
        recipe = stored.recipe()
        compiled = self.store.get_compiled(stored.recipe_sha256)
        compiled_dir: Path | None = None
        compiled_sha: str | None = None
        compile_status = None if compiled is None else compiled.status
        if compiled is not None and compiled.stored_rel_path:
            compiled_dir = (self.data_root / compiled.stored_rel_path).parent
            compiled_sha = compiled.compiled_part_sha256
        if compiled_part_sha256 is not None and compiled_part_sha256 != compiled_sha:
            found = self.store.find_compiled(stored.recipe_sha256, compiled_part_sha256)
            if found is not None and found[1] is not None:
                compile_status, rel = found[0], str(found[1])
                compiled_dir = (self.data_root / rel).parent
                compiled_sha = compiled_part_sha256
        flat_path = self.data_root / recipe.flat.stored_rel_path if recipe.flat is not None else None
        return ResolvedRecipe(
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.recipe_version,
            recipe_sha256=recipe.recipe_sha256,
            sku=recipe.sku,
            recipe_json_bytes=stored.recipe_json,
            cad_path=(self.data_root / recipe.cad.stored_rel_path).resolve(),
            cad_sha256=recipe.cad.sha256,
            part_name=recipe.part_name,
            part_revision=recipe.part_revision,
            compiled_dir=None if compiled_dir is None else compiled_dir.resolve(),
            compiled_part_sha256=compiled_sha,
            flat_path=None if flat_path is None else flat_path.resolve(),
            flat_sha256=recipe.flat.sha256 if recipe.flat is not None else None,
            extra={"compile_status": compile_status, "status": stored.status},
        )

    # -- CompiledPartSink --------------------------------------------------------------------
    def store_compiled(
        self,
        recipe_sha256: str,
        compile_out_dir: Path,
        *,
        status: str,
        compiled_part_sha256: str,
        blocker_codes: tuple[str, ...],
    ) -> None:
        """Install or keep a ``recipe_compile`` job's output (module docstring rules)."""
        import shutil
        import tempfile

        from apps.production_line.recipes.release_service import (
            _compile_report,
            archive_compiled,
            benchmark_nominals,
            expectations_document,
            install_compiled,
            notify_release_listeners,
        )
        from contracts.production.compiled_part import CompiledPartV1

        stored = self.store.get_by_sha(recipe_sha256)
        if stored is None:
            raise KeyError(recipe_sha256)
        compiled = CompiledPartV1.model_validate_json((Path(compile_out_dir) / "compiled_part.json").read_bytes())
        if compiled.compiled_part_sha256 != compiled_part_sha256 or compiled.status != status:
            raise ValueError("compile output does not match the reported status / hash")
        if compiled.recipe.recipe_sha256 != recipe_sha256:
            raise ValueError("compile output was made for another recipe")
        if status not in ("COMPILED", "BLOCKED"):
            raise ValueError(f"a recipe_compile job installs COMPILED or BLOCKED, not {status!r}")
        recipe = stored.recipe()
        record = CompiledRecord(
            recipe_sha256=recipe_sha256,
            status=status,
            compiled_part_sha256=compiled_part_sha256,
            compiler_version=compiled.compiler.version,
            blocker_codes=tuple(blocker_codes) if status == "BLOCKED" else (),
        )
        current = self.store.get_compiled(recipe_sha256)
        working = (
            current is not None
            and current.status == "COMPILED"
            and current.stored_rel_path is not None
            and (self.data_root / current.stored_rel_path).is_file()
        )
        compiled_root = self.data_root / "recipes" / "compiled"
        compiled_root.mkdir(parents=True, exist_ok=True)
        # Work on a copy: the runner's workspace stays intact for its own logs.
        scratch = Path(tempfile.mkdtemp(prefix=".sink-", dir=str(compiled_root)))
        try:
            copy = scratch / "out"
            shutil.copytree(compile_out_dir, copy, symlinks=False)
            if status == "BLOCKED" and working:
                assert current is not None
                rel = archive_compiled(copy, self.data_root, recipe_sha256, compiled_part_sha256)
                report = _compile_report(compiled, recipe)
                self.store.record_compiled_history(
                    dataclasses.replace(record, stored_rel_path=rel),
                    reason="RECOMPUTE_BLOCKED",
                    stored_rel_path=rel,
                    report=report,
                )
                self.store.record_compile_attempt(
                    stored.part_id,
                    status="BLOCKED",
                    codes=tuple(sorted({"COMPILE_BLOCKED", *blocker_codes})),
                    report={**report, "recipe_id": stored.recipe_id, "source": "RECOMPUTE",
                            "kept_compiled_part_sha256": current.compiled_part_sha256},
                    cad_sha256=recipe.cad.sha256,
                    flat_sha256=recipe.flat.sha256 if recipe.flat is not None else None,
                    recipe_sha256=recipe_sha256,
                    content_sha256=recipe.content_sha256,
                )
                _LOG.warning("recompute of %s is BLOCKED (%s); the COMPILED install %s is kept",
                             stored.recipe_id, ",".join(blocker_codes), current.compiled_part_sha256)
                return
            superseded = None
            superseded_rel = None
            if working and current is not None and current.compiled_part_sha256 != compiled_part_sha256:
                assert current.compiled_part_sha256 is not None and current.stored_rel_path is not None
                old_dir = self.data_root / Path(current.stored_rel_path).parent
                superseded_rel = archive_compiled(old_dir, self.data_root, recipe_sha256,
                                                  current.compiled_part_sha256)
                superseded = current
            try:
                rel = install_compiled(copy, self.data_root, recipe_sha256, compiled_part_sha256)
            except BaseException:
                if superseded_rel is not None and not (old_dir / "compiled_part.json").exists():
                    # Put the working compile back: the record still names it.
                    os.replace(self.data_root / Path(superseded_rel).parent, old_dir)
                raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        record = dataclasses.replace(record, stored_rel_path=rel)
        if status == "COMPILED":
            self.store.set_compiled(
                record,
                expectations=expectations_document(recipe, compiled),
                benchmark_nominals=benchmark_nominals(recipe, compiled),
                superseded=superseded,
                superseded_rel_path=superseded_rel,
            )
        else:
            self.store.set_compiled(record)
            self.store.set_expectations(recipe_sha256, compiled.compiler.version, "FAILED")
        self.invalidate()
        notify_release_listeners(recipe)

    # -- expectations (spec name) ---------------------------------------------------------
    def store_expectations(self, recipe_sha256: str, algo_version: str, expectations: dict[str, Any],
                           job_id: str | None = None) -> None:
        """Set a recipe's expectations READY (kept for the survey-spec port name)."""
        self.store.set_expectations(recipe_sha256, algo_version, "READY", expectations, job_id)


def _resolver_from_settings(settings: Any, catalog: Any | None = None) -> LineRecipeResolver:
    from apps.production_line.recipes.release_service import add_release_listener

    store = RecipeStore(settings.paths.db_recipes)
    resolver = LineRecipeResolver(
        store,
        settings.filename_regex,
        data_root=settings.data_root,
        extra_skus=catalog_sku_source(catalog) if catalog is not None else None,
    )
    add_release_listener(resolver.invalidate)
    return resolver


def build_recipe_provider(settings: Any) -> LineRecipeResolver:
    """``RecipeProvider`` (and ``CompiledPartSink``) for the runner (C11 wiring)."""
    return _resolver_from_settings(settings)


def build_scan_resolver(settings: Any, catalog: Any | None = None) -> LineRecipeResolver:
    """``ScanRecipeResolver`` for the hotfolder (C11 wiring). Pass the catalog port
    so catalogued-but-unreleased SKUs are PART_NOT_READY instead of UNMATCHED."""
    return _resolver_from_settings(settings, catalog)
