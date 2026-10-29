"""C06 postprocess: E3 output directory -> ``result/`` (ENGINE_IO §5 step 2).

Steps (each announced as a ``PROGRESS`` record when a callback is given):

1. ``ingest``: load ``engine/result.json``, the job recipe and CompiledPart;
   validate all three and ``check_result_against_compiled``
   (:mod:`.engine_output`). Any problem is a :class:`PostprocessError`
   (exit 2) and nothing is published.
2. ``viewer``: decimated index-aligned viewer copies, display meshes and
   ``bend_geometry.json`` (:mod:`.viewer_assets`).
3. ``snapshots``: offscreen iso/front/top/right views
   (:mod:`domains.production.reporting.snapshots`); the iso view is also
   written as ``overview.png``. Failures only add a warning.
4. ``report``: the bilingual PDF (:mod:`domains.production.reporting.shopfloor_pdf`).
   A rendering failure leaves ``artifacts.pdf`` null and adds the warning
   ``PDF_RENDER_FAILED`` (exit code stays 0, ENGINE_IO §5 / CONTRACTS §6b).
5. ``write``: the final result differs from the engine's only in
   ``viewer_assets``, ``artifacts`` and appended ``warnings``, plus ``context``:
   in a job workspace the job identity from ``input/context.json`` is filled
   into the engine's empty ``result.context`` (ENGINE_IO §5; E3 never sees it),
   so the PDF and overview carry it too. It is re-validated, re-checked against
   the CompiledPart/recipe and compared with the engine verdict
   (``same_verdict``) before ``shopfloor_result.json`` is written last
   (tmp + rename). Stale outputs of an earlier attempt are
   removed first, so a failed run can never leave an old success behind.

Warnings appended by this step are prefixed ``C06:`` so their origin is clear.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domains.production.inspection.errors import PostprocessError

__all__ = ["PostprocessOutcome", "run_postprocess", "STAGES"]

#: PROGRESS stage names and their progress percentage within the step.
STAGES: dict[str, int] = {"ingest": 5, "viewer": 25, "snapshots": 50, "report": 75, "write": 95}


@dataclass(frozen=True)
class PostprocessOutcome:
    result_path: Path
    summary_state: str
    inspection_validity: str
    pdf: str | None
    overview_png: str | None
    warnings: tuple[str, ...]
    kept_points: int | None
    source_points: int | None
    voxel_mm: float | None
    durations_s: dict[str, float]


def _remove_stale(out: Path) -> None:
    from contracts.production.layout import RESULT_FILES

    # Every servable output of an earlier attempt goes before ingest, so a run
    # that fails validation never leaves stale viewer files under allowlisted
    # names. Only the append-only log is kept.
    for key in sorted(RESULT_FILES):
        if key == "log":
            continue
        path = out / RESULT_FILES[key]
        if path.exists() or path.is_symlink():
            path.unlink()


def run_postprocess(
    *,
    result_path: str | Path,
    recipe_path: str | Path,
    compiled_path: str | Path,
    out_dir: str | Path,
    engine_bundle_id: str | None = None,
    cap_points: int | None = None,
    render_pdf: bool = True,
    progress: Callable[[str, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    context_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> PostprocessOutcome:
    """Run the whole step (module docstring) and return what was written.

    ``context_path`` is the job workspace's ``input/context.json``; when it exists
    and the engine left ``result.context`` empty, the job identity is filled in.
    ``overlay_path`` is the job's ``zeroshot/zeroshot_overlay_v1.json``: when it
    validates and belongs to this engine result (scan, summary state, result.json
    sha256) the PDF lists it in its "detections (unverified)" section. It never
    changes the result document, and a missing or invalid overlay is ignored.
    """
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.inspection_result import ArtifactsV1, ResultContextV1, ShopfloorInspectionResultV1
    from contracts.production.layout import RESULT_FILES
    from domains.production.inspection.engine_output import (
        check_consistency,
        load_engine_output,
        load_job_context,
        same_verdict,
    )
    from domains.production.inspection.ply_io import atomic_write_bytes
    from domains.production.inspection.viewer_assets import DEFAULT_CAP_POINTS, build_viewer_assets

    def step(name: str, message: str) -> None:
        if progress is not None:
            progress(name, STAGES[name], message)
        if log is not None:
            log(f"[{name}] {message}")

    durations: dict[str, float] = {}
    t0 = time.monotonic()
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise PostprocessError("USAGE_ERROR", f"--out is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _remove_stale(out)

    step("ingest", "validating engine result against compiled part and recipe")
    eo = load_engine_output(result_path, recipe_path, compiled_path)
    engine_result = eo.result
    job_context = load_job_context(context_path)
    # Only an empty engine context is filled; a non-empty one is kept as is (the runner
    # refuses an engine that wrote job identity: ENGINE_CONTEXT_NOT_EMPTY).
    fill_context = job_context is not None and engine_result.context == ResultContextV1()
    warnings: list[str] = []
    if engine_bundle_id and engine_result.engine.bundle_id and engine_bundle_id != engine_result.engine.bundle_id:
        warnings.append(
            f"C06: ENGINE_BUNDLE_ID_MISMATCH: runner bundle {engine_bundle_id} != result bundle {engine_result.engine.bundle_id}"
        )
    durations["ingest"] = time.monotonic() - t0

    t = time.monotonic()
    step("viewer", "writing decimated index-aligned viewer assets")
    viewer = build_viewer_assets(
        engine_result, eo.engine_dir, out, cap_points=cap_points or DEFAULT_CAP_POINTS
    )
    warnings.extend(f"C06: {w}" for w in viewer.warnings)
    if log is not None:
        log(f"viewer: {viewer.source_points} -> {viewer.kept_points} points (voxel {viewer.voxel_mm} mm)")
    durations["viewer"] = time.monotonic() - t

    def assemble(pdf: str | None, overview: str | None, extra: list[str]) -> ShopfloorInspectionResultV1:
        data = engine_result.model_dump(mode="json")
        if fill_context:
            data["context"] = job_context.model_dump(mode="json")  # type: ignore[union-attr]
        data["viewer_assets"] = viewer.assets.model_dump(mode="json")
        data["artifacts"] = ArtifactsV1(pdf=pdf, overview_png=overview).model_dump(mode="json")
        data["warnings"] = list(engine_result.warnings) + [w for w in extra if w not in engine_result.warnings]
        try:
            return ShopfloorInspectionResultV1.model_validate(data)
        except ValueError as exc:
            raise PostprocessError("RESULT_INVALID", f"rendered result does not validate: {exc}"[:600]) from None

    # -- snapshots
    t = time.monotonic()
    step("snapshots", "rendering iso/front/top/right views")
    views: dict[str, bytes] | None = None
    overview_rel: str | None = None
    views_result = assemble(None, None, warnings)
    try:
        from domains.production.reporting.snapshots import load_scene, render_view, render_views

        scene = load_scene(views_result, out)
        warnings.extend(f"C06: {w}" for w in scene.warnings)
        views = render_views(scene)
        # the PDF's page-1 picture (clean style, bends numbered in shop order)
        from domains.production.reporting.shopfloor_pdf import OVERVIEW_PX

        views["overview"] = render_view(scene, "iso", width_px=OVERVIEW_PX[0], height_px=OVERVIEW_PX[1], style="overview")
        atomic_write_bytes(out / RESULT_FILES["overview_png"], views["iso"])
        overview_rel = RESULT_FILES["overview_png"]
    except Exception as exc:  # display only: never fail the step on a snapshot
        warnings.append(f"C06: SNAPSHOT_FAILED: {type(exc).__name__}: {exc}"[:300])
        views = None
    durations["snapshots"] = time.monotonic() - t

    # -- zero-shot overlay (display only; never part of the result document)
    overlay = None
    if overlay_path is not None:
        import hashlib

        from domains.production.zeroshot.overlay import load_overlay

        try:
            engine_sha = hashlib.sha256(Path(result_path).read_bytes()).hexdigest()
        except OSError:
            engine_sha = None
        overlay = load_overlay(overlay_path, scan_sha256=engine_result.scan.sha256,
                               e3_summary_state=engine_result.summary.state, engine_result_sha256=engine_sha)
        if log is not None:
            log(f"overlay: {'listed in the PDF' if overlay is not None else 'none'} ({overlay_path})")

    # -- PDF (rendered from the final result, with artifacts pointing at it)
    t = time.monotonic()
    pdf_rel: str | None = None
    if render_pdf:
        step("report", "rendering the bilingual PDF report")
        candidate = assemble(RESULT_FILES["report_pdf"], overview_rel, warnings)
        try:
            from domains.production.reporting.shopfloor_pdf import write_pdf

            write_pdf(out / RESULT_FILES["report_pdf"], candidate, recipe=eo.recipe, views=views,
                      engine_bundle_id=engine_bundle_id, overlay=overlay)
            pdf_rel = RESULT_FILES["report_pdf"]
        except Exception as exc:
            warnings.append(f"C06: PDF_RENDER_FAILED: {type(exc).__name__}: {exc}"[:300])
            stale = out / RESULT_FILES["report_pdf"]
            if stale.exists():
                stale.unlink()
    durations["report"] = time.monotonic() - t

    # -- final result: self-checks, then written last
    step("write", "writing shopfloor_result.json")
    final = assemble(pdf_rel, overview_rel, warnings)
    reference = engine_result.model_copy(update={"context": job_context}) if fill_context else engine_result
    if not same_verdict(reference, final):
        raise PostprocessError("RESULT_ALTERED", "the rendered result differs from the engine verdict", exit_code=3)
    try:
        check_consistency(final, eo.compiled, eo.recipe)
    except PostprocessError as exc:
        raise PostprocessError("RESULT_INVALID", exc.message, exit_code=3) from None
    final_path = out / RESULT_FILES["shopfloor_result"]
    atomic_write_bytes(final_path, canonical_json_bytes(final))
    durations["total"] = time.monotonic() - t0
    return PostprocessOutcome(
        result_path=final_path,
        summary_state=final.summary.state,
        inspection_validity=final.inspection_validity.state,
        pdf=pdf_rel,
        overview_png=overview_rel,
        warnings=final.warnings,
        kept_points=viewer.kept_points,
        source_points=viewer.source_points,
        voxel_mm=viewer.voxel_mm,
        durations_s={k: round(v, 3) for k, v in durations.items()},
    )


def emit(tag: str, payload: dict[str, Any]) -> None:
    """Print one stdout protocol record (``TAG\\t{json}``) and flush."""
    import json

    print(f"{tag}\t{json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def write_error_file(out_dir: Path, code: str, message: str) -> None:
    """``postprocess_error.json {code, message}`` (best effort)."""
    import json

    from contracts.production.layout import RESULT_FILES
    from domains.production.inspection.ply_io import atomic_write_bytes

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            out_dir / RESULT_FILES["postprocess_error"],
            json.dumps({"code": code, "message": message[:2000]}, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    except OSError:
        pass
