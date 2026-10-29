"""Ingest of the E3 output directory (ENGINE_IO §4.1 / §4.3 / §5 step 2).

``result.json`` is loaded as ``ShopfloorInspectionResultV1`` (its validators
re-derive the stage decision and the summary, so an internally dishonest
result cannot pass), and ``check_result_against_compiled`` compares it with
the job's CompiledPart and recipe (every recipe characteristic present, the
compiled nominals and tolerances unchanged, requiredness as the recipe says,
stage model consistent). Any problem raises
``PostprocessError("ENGINE_OUTPUT_INVALID", ...)``: C06 never "repairs" an
engine verdict.

The recipe is re-validated as ``PartRecipeV1`` (hashes recomputed) and the
CompiledPart as ``CompiledPartV1``; failures are ``RECIPE_INVALID`` /
``COMPILED_PART_INVALID``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from domains.production.inspection.errors import PostprocessError

if TYPE_CHECKING:  # pragma: no cover
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.inspection_result import ShopfloorInspectionResultV1
    from contracts.production.part_recipe import PartRecipeV1

__all__ = [
    "EngineOutput",
    "load_result",
    "load_recipe",
    "load_compiled",
    "check_consistency",
    "check_recipe_ref",
    "load_engine_output",
    "load_job_context",
    "same_verdict",
    "VERDICT_FREE_FIELDS",
]

#: The only top-level result fields C06 may change (runner ``_same_verdict``).
VERDICT_FREE_FIELDS: frozenset[str] = frozenset({"viewer_assets", "artifacts", "warnings"})


def _first_error(exc: Exception) -> str:
    try:
        errors = exc.errors()  # type: ignore[attr-defined]
    except AttributeError:
        return str(exc)[:500]
    if not errors:
        return str(exc)[:500]
    err = errors[0]
    loc = ".".join(str(p) for p in err.get("loc", ()))
    return f"{loc}: {err.get('msg', '')}"[:500] if loc else str(err.get("msg", ""))[:500]


def _read(path: Path, code: str, what: str) -> bytes:
    if not path.is_file():
        raise PostprocessError("INPUT_MISSING", f"{what} not found: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PostprocessError(code, f"{what} unreadable: {exc}") from None


def load_result(path: str | Path) -> "ShopfloorInspectionResultV1":
    """Load and validate a ``ShopfloorInspectionResultV1`` (``ENGINE_OUTPUT_INVALID``)."""
    from pydantic import ValidationError

    from contracts.production.inspection_result import ShopfloorInspectionResultV1

    raw = _read(Path(path), "ENGINE_OUTPUT_INVALID", "result.json")
    try:
        return ShopfloorInspectionResultV1.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise PostprocessError("ENGINE_OUTPUT_INVALID", f"result.json does not validate: {_first_error(exc)}") from None


def load_recipe(path: str | Path) -> "PartRecipeV1":
    from pydantic import ValidationError

    from contracts.production.part_recipe import PartRecipeV1

    raw = _read(Path(path), "RECIPE_INVALID", "recipe.json")
    try:
        return PartRecipeV1.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise PostprocessError("RECIPE_INVALID", f"recipe.json does not validate: {_first_error(exc)}") from None


def load_compiled(path: str | Path) -> "CompiledPartV1":
    from pydantic import ValidationError

    from contracts.production.compiled_part import CompiledPartV1

    raw = _read(Path(path), "COMPILED_PART_INVALID", "compiled_part.json")
    try:
        return CompiledPartV1.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise PostprocessError(
            "COMPILED_PART_INVALID", f"compiled_part.json does not validate: {_first_error(exc)}"
        ) from None


def load_job_context(path: str | Path | None) -> "ResultContextV1 | None":
    """The job identity in a workspace's ``input/context.json`` (ENGINE_IO §5), or
    ``None`` when there is no such file (stand-alone use outside a job workspace).

    The runner writes it before the engine step and never passes it to E3; C06
    copies it into ``result.context``. A file that is present but invalid is
    ``RESULT_INVALID`` (exit 2): the published result would carry it.
    """
    if path is None or not Path(path).is_file():
        return None
    from pydantic import ValidationError

    from contracts.production.inspection_result import ResultContextV1

    raw = _read(Path(path), "RESULT_INVALID", "input/context.json")
    try:
        return ResultContextV1.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise PostprocessError("RESULT_INVALID", f"input/context.json does not validate: {_first_error(exc)}") from None


def check_recipe_ref(result: "ShopfloorInspectionResultV1", recipe: "PartRecipeV1") -> None:
    """Raise ``RECIPE_INVALID`` unless ``recipe`` is the exact recipe the result names.

    Used wherever a recipe is paired with a result without a CompiledPart
    (PDF-only re-render): the recipe supplies header fields (drawing
    revision, material, thickness, tolerance standard), so a wrong recipe
    file would otherwise print another part's revision with no error.
    The full :func:`check_consistency` remains the check when the
    CompiledPart is also available.
    """
    ref = result.recipe
    mismatched = [
        name
        for name, got, want in (
            ("recipe_id", recipe.recipe_id, ref.recipe_id),
            ("recipe_version", recipe.recipe_version, ref.version),
            ("recipe_sha256", recipe.recipe_sha256, ref.recipe_sha256),
            ("content_sha256", recipe.content_sha256, ref.content_sha256),
            ("sku", recipe.sku, result.sku),
        )
        if got != want
    ]
    if mismatched:
        raise PostprocessError(
            "RECIPE_INVALID",
            f"recipe {recipe.recipe_id} is not the recipe the result names ({ref.recipe_id}): "
            "RECIPE_REF_MISMATCH on " + ", ".join(mismatched),
        )


def check_consistency(
    result: "ShopfloorInspectionResultV1", compiled: "CompiledPartV1", recipe: "PartRecipeV1"
) -> None:
    """Raise ``ENGINE_OUTPUT_INVALID`` listing every cross-document problem."""
    from contracts.production.consistency import check_result_against_compiled

    problems = check_result_against_compiled(result, compiled, recipe)
    if problems:
        raise PostprocessError(
            "ENGINE_OUTPUT_INVALID", "result disagrees with compiled part/recipe: " + ", ".join(problems)
        )


def same_verdict(engine_result: "ShopfloorInspectionResultV1", rendered: "ShopfloorInspectionResultV1") -> bool:
    """True when ``rendered`` differs from ``engine_result`` only in viewer assets,
    artifacts and *appended* warnings (the runner's RESULT_ALTERED rule)."""
    a = engine_result.model_dump(mode="json")
    b = rendered.model_dump(mode="json")
    if {k: v for k, v in a.items() if k not in VERDICT_FREE_FIELDS} != {
        k: v for k, v in b.items() if k not in VERDICT_FREE_FIELDS
    }:
        return False
    wa, wb = list(a.get("warnings", ())), list(b.get("warnings", ()))
    return wb[: len(wa)] == wa


@dataclass(frozen=True)
class EngineOutput:
    """A validated, cross-checked E3 output directory."""

    engine_dir: Path
    result_path: Path
    result: "ShopfloorInspectionResultV1"
    result_sha256: str
    recipe: "PartRecipeV1"
    compiled: "CompiledPartV1"


def load_engine_output(result_path: str | Path, recipe_path: str | Path, compiled_path: str | Path) -> EngineOutput:
    """Load ``result.json`` (its directory is the engine dir), recipe and
    CompiledPart; validate all three and their consistency."""
    rp = Path(result_path)
    recipe = load_recipe(recipe_path)
    compiled = load_compiled(compiled_path)
    result = load_result(rp)
    check_consistency(result, compiled, recipe)
    return EngineOutput(
        engine_dir=rp.parent,
        result_path=rp,
        result=result,
        result_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
        recipe=recipe,
        compiled=compiled,
    )
