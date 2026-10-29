"""Ports between the line backbone (C02) and the recipe side (C01/C05).

The hotfolder and the runner depend only on these small, frozen shapes, never
on a recipe store implementation:

- :class:`ScanRecipeResolver` maps an original scan filename to a recipe
  (:class:`ResolveOutcome`). The production implementation is
  :class:`apps.production_line.ingest.sku_resolver.FilenameRecipeResolver`,
  which calls C01's ``domains.production.recipe.sku.parse_scan_filename``
  over an :class:`ActiveRecipeCatalog` (C05 implements the catalog).
- :class:`RecipeProvider` loads the frozen recipe bytes, CAD path and
  compiled-part directory for a job (:class:`ResolvedRecipe`).
- :class:`CompiledPartSink` receives the output of a ``recipe_compile`` job.

Parsed filename fields are informational for the operator and the resolver.
None of them is ever passed to the engine (plan §10 amendment 4).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from apps.production_line.store.vocabulary import PARSE_STATUSES, RESOLVE_STATUSES

__all__ = [
    "ResolveOutcome",
    "ScanRecipeResolver",
    "ActiveRecipeRef",
    "ActiveRecipeCatalog",
    "ResolvedRecipe",
    "RecipeProvider",
    "CompiledPartSink",
    "parse_status_for",
    "minimal_parse",
]


def parse_status_for(resolve_status: str) -> str:
    """Filename-parse status implied by a resolve status (MATCHED/PART_NOT_READY parsed OK)."""
    if resolve_status in ("MATCHED", "PART_NOT_READY"):
        return "OK"
    if resolve_status in PARSE_STATUSES:
        return resolve_status
    raise ValueError(f"unknown resolve status {resolve_status!r}")


def minimal_parse(resolve_status: str, sku: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """A ``ScanNameParseV1``-shaped dict for resolvers that do not supply one."""
    return {
        "status": parse_status_for(resolve_status),
        "sku": sku,
        "revision": None,
        "expected_revision": None,
        "revision_status": None,
        "serial": None,
        "scan_timestamp": None,
        "scan_label": None,
        "export_index": None,
        "copy_index": None,
        "candidates": [],
        "reason": reason,
    }


@dataclass(frozen=True)
class ResolveOutcome:
    """Result of resolving one scan filename.

    ``status`` is one of MATCHED, UNMATCHED, AMBIGUOUS, UNPARSEABLE,
    PART_NOT_READY or REVISION_MISMATCH. A MATCHED outcome names the recipe
    (id, version, sha) and the compiled part the job must pin. ``parse`` is a
    ``ScanNameParseV1``-shaped dict (``None`` = derive a minimal one).
    """

    status: str
    sku: str | None = None
    serial: str | None = None
    scan_timestamp: str | None = None
    recipe_id: str | None = None
    recipe_version: int | None = None
    recipe_sha256: str | None = None
    compiled_part_sha256: str | None = None
    reason: str | None = None
    parse: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in RESOLVE_STATUSES:
            raise ValueError(f"unknown resolve status {self.status!r}")
        if self.status == "MATCHED" and (
            self.sku is None
            or self.recipe_id is None
            or self.recipe_version is None
            or self.recipe_sha256 is None
            or self.compiled_part_sha256 is None
        ):
            raise ValueError("a MATCHED outcome names sku, recipe id/version/sha and compiled_part_sha256")

    def match_dict(self) -> dict[str, Any]:
        """``ScanMatchV1``-shaped dict."""
        return {
            "status": self.status,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "recipe_sha256": self.recipe_sha256,
            "reason": self.reason,
        }

    def parse_dict(self) -> dict[str, Any]:
        if self.parse is not None:
            return dict(self.parse)
        out = minimal_parse(self.status, self.sku, self.reason)
        out["serial"] = self.serial
        out["scan_timestamp"] = self.scan_timestamp
        return out


@runtime_checkable
class ScanRecipeResolver(Protocol):
    def resolve(self, original_filename: str) -> ResolveOutcome:
        """Resolve a scan's original filename. Must not raise for any filename."""
        ...


@dataclass(frozen=True)
class ActiveRecipeRef:
    """One SKU's active (latest non-retired) recipe as the catalog sees it.

    ``compile_status`` is COMPILED, BLOCKED, PENDING or FAILED; only a COMPILED
    recipe with a ``compiled_part_sha256`` can be inspected, otherwise the
    scan is PART_NOT_READY. ``part_revision`` is the recipe's expected
    revision, compared against the revision read from the filename.
    """

    sku: str
    recipe_id: str
    recipe_version: int
    recipe_sha256: str
    part_revision: str | None = None
    compiled_part_sha256: str | None = None
    compile_status: str = "PENDING"


@runtime_checkable
class ActiveRecipeCatalog(Protocol):
    def list_active_recipes(self) -> Iterable[ActiveRecipeRef]:
        """Every SKU's active recipe (canonical SKUs)."""
        ...

    def list_registered_skus(self) -> Iterable[str]:
        """Every registered SKU, including those without an active recipe yet."""
        ...


@dataclass(frozen=True)
class ResolvedRecipe:
    """Everything the runner needs to run a recipe's job.

    ``recipe_json_bytes`` are the frozen canonical ``PartRecipeV1`` bytes.
    ``cad_path``, ``compiled_dir`` and ``flat_path`` are absolute paths under
    the line data root (``compiled_dir`` = ``recipes/compiled/<recipe_sha256>``,
    holding ``compiled_part.json`` and ``stages/``).
    """

    recipe_id: str
    recipe_version: int
    recipe_sha256: str
    sku: str
    recipe_json_bytes: bytes
    cad_path: Path
    cad_sha256: str
    part_name: str | None = None
    part_revision: str | None = None
    compiled_dir: Path | None = None
    compiled_part_sha256: str | None = None
    flat_path: Path | None = None
    flat_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RecipeProvider(Protocol):
    def load(self, recipe_id: str) -> ResolvedRecipe:
        """Return the recipe; raise ``KeyError`` when it does not exist."""
        ...


@runtime_checkable
class CompiledPartSink(Protocol):
    def store_compiled(
        self,
        recipe_sha256: str,
        compile_out_dir: Path,
        *,
        status: str,
        compiled_part_sha256: str,
        blocker_codes: tuple[str, ...],
    ) -> None:
        """Install a finished compile (COMPILED or BLOCKED) for the recipe.

        The implementation copies ``compile_out_dir`` to
        ``recipes/compiled/<recipe_sha256>/`` and records the status.
        """
        ...
