"""Command line of the C06 postprocess step (E3 output directory -> ``result/``).

::

    python -m domains.production.inspection.postprocess_cli \\
        (--job-dir WS | --result ENGINE/result.json --recipe RECIPE.json --compiled COMPILED.json) \\
        [--engine-dir ENGINE] --out WS/result [--engine-bundle-id ID] [--cap-points N] [--no-pdf]

With ``--job-dir WS`` the inputs follow the workspace layout (ENGINE_IO §5):
``WS/engine/result.json``, ``WS/recipe.json``, ``WS/compiled/compiled_part.json``;
explicit flags override them. ``WS/input/context.json`` (the job identity the
runner wrote; E3 never sees it), when present, fills the engine's empty
``result.context``. The runner calls the equivalent
``python -m domains.production.reporting.render_cli --job-dir WS --result
WS/engine/result.json --out WS/result``.

Stdout protocol (one record per line, tab separated, as every supervised
step): ``PROGRESS\\t{"stage", "progress", "message"}`` for ingest / viewer /
snapshots / report / write, then ``RESULT\\t{"shopfloor_result",
"summary_state", "inspection_validity", "pdf", "overview_png", "warnings"}``
or ``ERROR\\t{"code", "message"}``.

Exit codes: 0 = ``result/shopfloor_result.json`` written (a PDF rendering
failure is a warning, not an error); 2 = invalid input; 3 = internal error.
On 2 and 3 ``result/postprocess_error.json {code, message}`` is written.
A log is appended to ``result/postprocess.log``.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from domains.production.inspection import ADAPTER_VERSION
from domains.production.inspection.errors import PostprocessError

__all__ = ["main", "run_directory_mode", "resolve_inputs", "job_context_path"]


def resolve_inputs(
    *,
    job_dir: Path | None,
    result: Path | None,
    recipe: Path | None,
    compiled: Path | None,
    engine_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """``(result, recipe, compiled)`` from a job workspace and/or explicit paths."""
    if result is None:
        if engine_dir is not None:
            result = engine_dir / "result.json"
        elif job_dir is not None:
            result = job_dir / "engine" / "result.json"
    if recipe is None and job_dir is not None:
        recipe = job_dir / "recipe.json"
    if compiled is None and job_dir is not None:
        compiled = job_dir / "compiled" / "compiled_part.json"
    missing = [name for name, value in (("--result", result), ("--recipe", recipe), ("--compiled", compiled)) if value is None]
    if missing:
        raise PostprocessError("USAGE_ERROR", f"missing {', '.join(missing)} (or give --job-dir)")
    return result, recipe, compiled  # type: ignore[return-value]


class _Log:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, line: str) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {line}\n")
        except OSError:
            pass


def run_directory_mode(
    *,
    result: Path,
    recipe: Path,
    compiled: Path,
    out: Path,
    engine_bundle_id: str | None,
    cap_points: int | None,
    render_pdf: bool = True,
    context: Path | None = None,
    overlay: Path | None = None,
) -> int:
    """Run the postprocess step with the stdout protocol; returns the exit code.

    ``context`` is the job workspace's ``input/context.json`` (see :func:`job_context_path`)."""
    from contracts.production.layout import RESULT_FILES
    from domains.production.inspection.adapter import emit, run_postprocess, write_error_file

    log = _Log(out / RESULT_FILES["log"] if not (out.exists() and not out.is_dir()) else None)
    log(f"{ADAPTER_VERSION} start result={result} recipe={recipe} compiled={compiled} out={out}")
    try:
        outcome = run_postprocess(
            result_path=result,
            recipe_path=recipe,
            compiled_path=compiled,
            out_dir=out,
            engine_bundle_id=engine_bundle_id,
            cap_points=cap_points,
            render_pdf=render_pdf,
            progress=lambda stage, pct, msg: emit("PROGRESS", {"stage": stage, "progress": pct, "message": msg}),
            log=log,
            context_path=context,
            overlay_path=overlay,
        )
    except PostprocessError as exc:
        log(f"ERROR {exc.code}: {exc.message}")
        if out.is_dir() or not out.exists():
            write_error_file(out, exc.code, exc.message)
        emit("ERROR", {"code": exc.code, "message": exc.message[:2000]})
        return exc.exit_code
    except Exception as exc:  # internal error: exit 3 with a traceback on stderr
        traceback.print_exc(file=sys.stderr)
        message = f"{type(exc).__name__}: {exc}"[:2000]
        log(f"ERROR INTERNAL_ERROR: {message}")
        if out.is_dir() or not out.exists():
            write_error_file(out, "INTERNAL_ERROR", message)
        emit("ERROR", {"code": "INTERNAL_ERROR", "message": message})
        return 3
    log(f"done {outcome.summary_state} pdf={outcome.pdf} durations={outcome.durations_s}")
    emit(
        "RESULT",
        {
            "shopfloor_result": RESULT_FILES["shopfloor_result"],
            "summary_state": outcome.summary_state,
            "inspection_validity": outcome.inspection_validity,
            "pdf": outcome.pdf,
            "overview_png": outcome.overview_png,
            "warnings": list(outcome.warnings),
        },
    )
    return 0


def job_overlay_path(job_dir: Path | None) -> Path | None:
    """``<job_dir>/zeroshot/zeroshot_overlay_v1.json`` (runner step 3b), or None without a job workspace."""
    return None if job_dir is None else job_dir / "zeroshot" / "zeroshot_overlay_v1.json"


def job_context_path(job_dir: Path | None) -> Path | None:
    """``<job_dir>/input/context.json`` (ENGINE_IO §5), or None without a job workspace."""
    from contracts.production.layout import WORKSPACE_NAMES

    return None if job_dir is None else job_dir / WORKSPACE_NAMES["context"]


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise PostprocessError("USAGE_ERROR", message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m domains.production.inspection.postprocess_cli", description="C06 postprocess: E3 output directory -> result/")
    p.add_argument("--job-dir", type=Path)
    p.add_argument("--engine-dir", type=Path)
    p.add_argument("--result", type=Path)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--compiled", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--engine-bundle-id")
    p.add_argument("--cap-points", type=int)
    p.add_argument("--no-pdf", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from domains.production.inspection.adapter import emit

    try:
        args = _parser().parse_args(argv)
        if args.cap_points is not None and args.cap_points < 1:
            raise PostprocessError("USAGE_ERROR", "--cap-points must be >= 1")
        result, recipe, compiled = resolve_inputs(
            job_dir=args.job_dir, result=args.result, recipe=args.recipe, compiled=args.compiled,
            engine_dir=args.engine_dir,
        )
    except PostprocessError as exc:
        emit("ERROR", {"code": exc.code, "message": exc.message})
        return 2
    return run_directory_mode(
        result=result, recipe=recipe, compiled=compiled, out=args.out, engine_bundle_id=args.engine_bundle_id,
        cap_points=args.cap_points, render_pdf=not args.no_pdf, context=job_context_path(args.job_dir),
        overlay=job_overlay_path(args.job_dir),
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
