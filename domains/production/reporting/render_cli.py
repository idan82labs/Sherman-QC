"""Render step of the line runner (C06) and stand-alone PDF re-render.

Runner step 4 (ENGINE_IO §5)::

    python -m domains.production.reporting.render_cli --job-dir WS \\
        --result WS/engine/result.json --out WS/result [--engine-bundle-id ID]

``--out`` is a directory: the full C06 postprocess runs
(:mod:`domains.production.inspection.postprocess_cli`): validation and
``check_result_against_compiled`` against ``WS/recipe.json`` and
``WS/compiled/compiled_part.json``, viewer assets, snapshots, the PDF and
``result/shopfloor_result.json``; the job identity in ``WS/input/context.json``
fills the engine's empty ``result.context`` (ENGINE_IO §5). Exit 0 / 2 / 3 with
``result/postprocess_error.json`` on failure.

PDF only::

    python -m domains.production.reporting.render_cli --result RESULT.json --out report.pdf \\
        [--job-dir WS | --recipe RECIPE.json [--compiled COMPILED.json]] [--engine-bundle-id ID]

``--out`` ending in ``.pdf`` re-renders the report from any validated
``ShopfloorInspectionResultV1`` (an engine ``result.json`` or a
``result/shopfloor_result.json``); its viewer assets are read relative to the
result's directory for the 3D views. When both the recipe and the
CompiledPart are available the result is cross-checked first (exit 2 on any
problem); a recipe given alone must still be the exact recipe the result
names (id, version, both hashes, SKU), else ``RECIPE_INVALID`` (exit 2),
because it supplies header fields (revision, material). The
output is deterministic: the same input gives the same bytes.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from domains.production.inspection.errors import PostprocessError

__all__ = ["main", "render_pdf_only"]


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise PostprocessError("USAGE_ERROR", message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m domains.production.reporting.render_cli", description="C06 render step and stand-alone PDF re-render")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="result directory, or a .pdf file for a PDF-only render")
    p.add_argument("--job-dir", type=Path)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--compiled", type=Path)
    p.add_argument("--engine-bundle-id")
    p.add_argument("--cap-points", type=int)
    return p


def render_pdf_only(
    *, result: Path, out_pdf: Path, recipe: Path | None, compiled: Path | None, engine_bundle_id: str | None
) -> int:
    """PDF-only mode with the stdout protocol; returns the exit code."""
    from domains.production.inspection.adapter import emit
    from domains.production.inspection.engine_output import (
        check_consistency,
        check_recipe_ref,
        load_compiled,
        load_recipe,
        load_result,
    )
    from domains.production.reporting.shopfloor_pdf import write_pdf

    try:
        emit("PROGRESS", {"stage": "ingest", "progress": 5, "message": "validating result"})
        res = load_result(result)
        rec = load_recipe(recipe) if recipe is not None else None
        if rec is not None:
            check_recipe_ref(res, rec)
            if compiled is not None:
                check_consistency(res, load_compiled(compiled), rec)
        emit("PROGRESS", {"stage": "report", "progress": 50, "message": "rendering the bilingual PDF report"})
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        size = write_pdf(out_pdf, res, recipe=rec, assets_dir=result.parent, engine_bundle_id=engine_bundle_id)
    except PostprocessError as exc:
        emit("ERROR", {"code": exc.code, "message": exc.message[:2000]})
        return exc.exit_code
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("ERROR", {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:2000]})
        return 3
    emit("RESULT", {"pdf": str(out_pdf), "bytes": size, "summary_state": res.summary.state})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from domains.production.inspection.adapter import emit
    from domains.production.inspection.postprocess_cli import (
        job_context_path,
        job_overlay_path,
        resolve_inputs,
        run_directory_mode,
    )

    try:
        args = _parser().parse_args(argv)
        if args.cap_points is not None and args.cap_points < 1:
            raise PostprocessError("USAGE_ERROR", "--cap-points must be >= 1")
    except PostprocessError as exc:
        emit("ERROR", {"code": exc.code, "message": exc.message})
        return 2
    if args.out.suffix.lower() == ".pdf":
        recipe = args.recipe or (args.job_dir / "recipe.json" if args.job_dir is not None else None)
        compiled = args.compiled or (args.job_dir / "compiled" / "compiled_part.json" if args.job_dir is not None else None)
        return render_pdf_only(
            result=args.result, out_pdf=args.out, recipe=recipe, compiled=compiled,
            engine_bundle_id=args.engine_bundle_id,
        )
    try:
        result, recipe, compiled = resolve_inputs(
            job_dir=args.job_dir, result=args.result, recipe=args.recipe, compiled=args.compiled
        )
    except PostprocessError as exc:
        emit("ERROR", {"code": exc.code, "message": exc.message})
        return 2
    return run_directory_mode(
        result=result, recipe=recipe, compiled=compiled, out=args.out, engine_bundle_id=args.engine_bundle_id,
        cap_points=args.cap_points, context=job_context_path(args.job_dir), overlay=job_overlay_path(args.job_dir),
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
