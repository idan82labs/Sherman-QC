"""C06 postprocess / render step end to end (subprocess and in-process).

The runner (C02) calls ``python -m domains.production.reporting.render_cli
--job-dir WS --result WS/engine/result.json --out WS/result
[--engine-bundle-id ID]`` and then requires ``result/shopfloor_result.json``
to validate, stay consistent with the CompiledPart/recipe, and keep the
engine verdict (only viewer assets, artifacts and appended warnings may
change). These tests drive exactly that command line against fake E3 output
directories (synthetic SYN-HAT-1 data in tmp_path).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from contracts.production.consistency import check_result_against_compiled
from contracts.production.inspection_result import ShopfloorInspectionResultV1
from contracts.production.layout import ARTIFACT_ALLOWLIST, RESULT_FILES
from domains.production.inspection.engine_output import load_compiled, load_recipe, load_result, same_verdict
from domains.production.inspection.errors import PostprocessError
from domains.production.inspection.fake_engine_output import build_fake_engine_output
from domains.production.inspection.ply_io import read_ply

REPO = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")}
    env.update(PYTHONPATH=f"{REPO}{os.pathsep}{REPO / 'backend'}", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    return env


def _run(module: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)], cwd=REPO, env=_env(), capture_output=True, text=True,
        timeout=180,
    )


def _records(stdout: str) -> list[tuple[str, dict]]:
    out = []
    for line in stdout.splitlines():
        tag, _, raw = line.partition("\t")
        if tag in ("PROGRESS", "RESULT", "ERROR"):
            out.append((tag, json.loads(raw)))
    return out


def _render(ws: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run("domains.production.reporting.render_cli", "--job-dir", ws, "--result", ws / "engine" / "result.json",
                "--out", ws / "result", *extra)


def _check_published(ws: Path) -> ShopfloorInspectionResultV1:
    """Everything the runner checks after the render step, plus the asset files."""
    final = ShopfloorInspectionResultV1.model_validate_json((ws / "result" / "shopfloor_result.json").read_bytes())
    engine = load_result(ws / "engine" / "result.json")
    compiled = load_compiled(ws / "compiled" / "compiled_part.json")
    recipe = load_recipe(ws / "recipe.json")
    assert check_result_against_compiled(final, compiled, recipe) == []
    assert same_verdict(engine, final)
    from apps.production_line.runner.runner import _same_verdict as runner_same_verdict

    assert runner_same_verdict(engine, final)
    va = final.viewer_assets
    served = {Path(p).name for p in ARTIFACT_ALLOWLIST.values()}
    for rel in (va.registered_scan, va.deviations, va.stage_mesh, va.final_mesh, va.bend_geometry,
                final.artifacts.pdf, final.artifacts.overview_png):
        if rel is not None:
            assert (ws / "result" / rel).is_file(), rel
            assert rel in served, f"{rel} is referenced but not servable"
    if va.registered_scan is not None:
        n = len(read_ply(ws / "result" / va.registered_scan).vertices)
        assert n == va.point_count
        if va.deviations is not None:
            assert (ws / "result" / va.deviations).stat().st_size == 4 * n
    return final


def test_render_cli_final_scan(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws", scenario="final")
    proc = _render(fake.workspace, "--engine-bundle-id", "bundle-example")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = _records(proc.stdout)
    assert [r[1]["stage"] for r in recs if r[0] == "PROGRESS"] == ["ingest", "viewer", "snapshots", "report", "write"]
    result_line = [r[1] for r in recs if r[0] == "RESULT"]
    assert len(result_line) == 1 and result_line[0]["summary_state"] == "OUT_OF_TOLERANCE_FOUND"
    final = _check_published(fake.workspace)
    assert final.summary.state == "OUT_OF_TOLERANCE_FOUND"
    assert final.artifacts.pdf == "report.pdf" and final.artifacts.overview_png == "overview.png"
    assert (fake.workspace / "result" / "report.pdf").read_bytes().startswith(b"%PDF-")
    assert (fake.workspace / "result" / "overview.png").read_bytes().startswith(b"\x89PNG")
    assert final.viewer_assets.deviations == "viewer_deviations.f32"
    assert not (fake.workspace / "result" / "postprocess_error.json").exists()
    assert (fake.workspace / "result" / "postprocess.log").is_file()
    # the engine's own warning is kept first; nothing the engine said is dropped
    assert final.warnings[0] == "SYNTHETIC FAKE ENGINE OUTPUT"


@pytest.mark.parametrize("scenario, state", [("in_process", "STAGE_IN_TOLERANCE"), ("incomplete", "INCOMPLETE"),
                                             ("invalid", "INVALID_INSPECTION")])
def test_postprocess_cli_keeps_every_verdict(tmp_path: Path, scenario: str, state: str) -> None:
    fake = build_fake_engine_output(tmp_path / "ws", scenario=scenario)
    proc = _run("domains.production.inspection.postprocess_cli", "--job-dir", fake.workspace,
                "--out", fake.workspace / "result")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    final = _check_published(fake.workspace)
    assert final.summary.state == state
    if scenario == "invalid":
        assert final.viewer_assets.registered_scan is None and final.viewer_assets.deviations is None
        assert final.viewer_assets.heatmap_unavailable_reason == "REGISTRATION_UNTRUSTED"
        assert final.artifacts.pdf == "report.pdf"


def test_decimation_cap_keeps_alignment_through_the_cli(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws", scenario="final")
    proc = _render(fake.workspace, "--cap-points", "3000")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    final = _check_published(fake.workspace)
    assert final.viewer_assets.point_count <= 3000 < fake.point_count


def test_inconsistent_engine_output_fails_and_removes_stale_success(tmp_path: Path) -> None:
    def other(data):
        data["engine"]["compiled_part_sha256"] = "0" * 64

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=other)
    result_dir = fake.workspace / "result"
    result_dir.mkdir()
    (result_dir / "shopfloor_result.json").write_text("{}")  # a stale success from an earlier attempt
    (result_dir / "report.pdf").write_bytes(b"%PDF-stale")
    stale_viewer = [RESULT_FILES[k] for k in ("viewer_scan", "viewer_deviations", "stage_mesh", "final_mesh",
                                              "bend_geometry", "overview_png")]
    for name in stale_viewer:  # servable files of an earlier attempt must not survive a failed ingest
        (result_dir / name).write_bytes(b"stale")
    (result_dir / RESULT_FILES["log"]).write_text("earlier attempt\n")
    proc = _render(fake.workspace)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    err = json.loads((result_dir / "postprocess_error.json").read_text())
    assert err["code"] == "ENGINE_OUTPUT_INVALID" and "COMPILED_PART_REF_MISMATCH" in err["message"]
    assert not (result_dir / "shopfloor_result.json").exists()
    assert not (result_dir / "report.pdf").exists()
    for name in stale_viewer:
        assert not (result_dir / name).exists(), name
    assert (result_dir / RESULT_FILES["log"]).read_text().startswith("earlier attempt\n")  # log is append-only
    errors = [r[1] for r in _records(proc.stdout) if r[0] == "ERROR"]
    assert errors and errors[0]["code"] == "ENGINE_OUTPUT_INVALID"


def test_missing_compiled_part_is_exit_2(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    fake.compiled_path.unlink()
    proc = _render(fake.workspace)
    assert proc.returncode == 2
    assert json.loads((fake.workspace / "result" / "postprocess_error.json").read_text())["code"] == "INPUT_MISSING"


def test_usage_errors_are_exit_2(tmp_path: Path) -> None:
    proc = _run("domains.production.reporting.render_cli", "--out", tmp_path / "x")
    assert proc.returncode == 2
    assert _records(proc.stdout)[0] == ("ERROR", {"code": "USAGE_ERROR", "message": "the following arguments are required: --result"})
    proc = _run("domains.production.inspection.postprocess_cli", "--out", tmp_path / "x")
    assert proc.returncode == 2 and _records(proc.stdout)[0][1]["code"] == "USAGE_ERROR"


def test_outputs_are_deterministic(tmp_path: Path) -> None:
    a = build_fake_engine_output(tmp_path / "a", scenario="final")
    b = build_fake_engine_output(tmp_path / "b", scenario="final")
    for ws in (a.workspace, b.workspace):
        assert _render(ws, "--cap-points", "8000").returncode == 0
    for name in ("shopfloor_result.json", "viewer_scan.ply", "viewer_deviations.f32", "stage_mesh.ply",
                 "final_mesh.ply", "bend_geometry.json", "overview.png", "report.pdf"):
        assert (a.workspace / "result" / name).read_bytes() == (b.workspace / "result" / name).read_bytes(), name


def test_pdf_only_rerender(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws", scenario="in_process")
    assert _render(fake.workspace).returncode == 0
    out_pdf = tmp_path / "again" / "report.pdf"
    proc = _run("domains.production.reporting.render_cli", "--result", fake.workspace / "result" / "shopfloor_result.json",
                "--job-dir", fake.workspace, "--out", out_pdf)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_pdf.read_bytes().startswith(b"%PDF-")
    # PDF-only mode still refuses a result that disagrees with its recipe/compiled part
    other = tmp_path / "other"
    other_fake = build_fake_engine_output(other, scenario="final",
                                          mutate_result=lambda d: d["engine"].update(compiled_part_sha256="1" * 64))
    proc = _run("domains.production.reporting.render_cli", "--result", other_fake.result_path,
                "--job-dir", fake.workspace, "--out", tmp_path / "bad.pdf")
    assert proc.returncode == 2 and not (tmp_path / "bad.pdf").exists()


def _other_recipe(fake, out: Path) -> Path:
    """A valid recipe (hashes recomputed) that is not the one the result names."""
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.part_recipe import PartRecipeV1

    rec = load_recipe(fake.recipe_path)
    data = rec.model_dump(mode="json")
    for key in ("content_sha256", "recipe_sha256"):
        data.pop(key)
    data["part_revision"] = "Z9"
    other = PartRecipeV1.finalize(**data)
    assert other.recipe_id == rec.recipe_id and other.recipe_sha256 != rec.recipe_sha256
    out.write_bytes(canonical_json_bytes(other))
    return out


def test_pdf_only_refuses_a_recipe_the_result_does_not_name(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws", scenario="final")
    good = tmp_path / "good.pdf"
    proc = _run("domains.production.reporting.render_cli", "--result", fake.result_path,
                "--recipe", fake.recipe_path, "--out", good)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    other = _other_recipe(fake, tmp_path / "other_recipe.json")
    bad = tmp_path / "bad.pdf"
    proc = _run("domains.production.reporting.render_cli", "--result", fake.result_path,
                "--recipe", other, "--out", bad)  # recipe without --compiled
    assert proc.returncode == 2, proc.stdout + proc.stderr
    err = [r[1] for r in _records(proc.stdout) if r[0] == "ERROR"]
    assert err and err[0]["code"] == "RECIPE_INVALID" and "RECIPE_REF_MISMATCH" in err[0]["message"]
    assert not bad.exists()
    # the library entry point refuses it too (no wrong drawing revision in a header)
    from domains.production.reporting.shopfloor_pdf import render_pdf

    with pytest.raises(PostprocessError) as info:
        render_pdf(load_result(fake.result_path), recipe=load_recipe(other))
    assert info.value.code == "RECIPE_INVALID"


# ---------------------------------------------------------------- in-process failure paths
def _inproc(fake, **kwargs):
    from domains.production.inspection.adapter import run_postprocess

    return run_postprocess(result_path=fake.result_path, recipe_path=fake.recipe_path,
                           compiled_path=fake.compiled_path, out_dir=fake.workspace / "result", **kwargs)


def test_pdf_failure_is_a_warning_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import domains.production.reporting.shopfloor_pdf as pdf

    def boom(*a, **k):
        raise RuntimeError("font backend down")

    monkeypatch.setattr(pdf, "write_pdf", boom)
    fake = build_fake_engine_output(tmp_path / "ws")
    outcome = _inproc(fake)
    assert outcome.pdf is None
    final = _check_published(fake.workspace)
    assert final.artifacts.pdf is None
    assert any(w.startswith("C06: PDF_RENDER_FAILED") for w in final.warnings)
    assert not (fake.workspace / "result" / "report.pdf").exists()


def test_snapshot_failure_still_publishes_result_and_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import domains.production.reporting.snapshots as snaps

    monkeypatch.setattr(snaps, "render_views", lambda *a, **k: (_ for _ in ()).throw(MemoryError("agg")))
    fake = build_fake_engine_output(tmp_path / "ws")
    outcome = _inproc(fake)
    assert outcome.overview_png is None and outcome.pdf == "report.pdf"
    final = _check_published(fake.workspace)
    assert any(w.startswith("C06: SNAPSHOT_FAILED") for w in final.warnings)


def test_verdict_guard_blocks_an_altered_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import domains.production.inspection.engine_output as eo

    monkeypatch.setattr(eo, "same_verdict", lambda a, b: False)
    fake = build_fake_engine_output(tmp_path / "ws")
    with pytest.raises(PostprocessError) as info:
        _inproc(fake)
    assert info.value.code == "RESULT_ALTERED" and info.value.exit_code == 3
    assert not (fake.workspace / "result" / "shopfloor_result.json").exists()


def test_bundle_id_mismatch_is_warned(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    _inproc(fake, engine_bundle_id="bundle-other")
    final = _check_published(fake.workspace)
    assert any("ENGINE_BUNDLE_ID_MISMATCH" in w for w in final.warnings)


def test_job_identity_from_the_workspace_fills_the_empty_engine_context(tmp_path: Path) -> None:
    """ENGINE_IO §5: E3 writes ``result.context`` empty (no --context, amendment 4); the
    render step fills it from ``input/context.json`` so the PDF names the job. A
    workspace without the file keeps the empty context; an invalid file is exit 2."""
    import fitz

    from contracts.production.inspection_result import ResultContextV1

    ws = tmp_path / "ws"
    build_fake_engine_output(ws)
    # The fake is built from the post-C06 contract examples (context filled); E3 itself
    # writes the block empty, so empty it here as E3 does.
    raw = json.loads((ws / "engine" / "result.json").read_text(encoding="utf-8"))
    raw["context"] = {}
    (ws / "engine" / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    engine = load_result(ws / "engine" / "result.json")
    assert engine.context == ResultContextV1()
    ctx = ResultContextV1(job_id="line_" + "ab" * 16, station_id="cell-1", scan_file_name="SYN-HAT-1_A_FULL_SCAN 1.ply",
                          serial="SN0042", scan_timestamp=None, received_at="2026-10-17T12:00:00.000Z")
    (ws / "input").mkdir(parents=True, exist_ok=True)
    (ws / "input" / "context.json").write_text(ctx.model_dump_json(), encoding="utf-8")
    proc = _render(ws)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    final = ShopfloorInspectionResultV1.model_validate_json((ws / "result" / "shopfloor_result.json").read_bytes())
    assert final.context == ctx
    assert check_result_against_compiled(final, load_compiled(ws / "compiled" / "compiled_part.json"),
                                         load_recipe(ws / "recipe.json")) == []
    assert same_verdict(engine.model_copy(update={"context": ctx}), final)  # nothing else changed
    assert not same_verdict(engine, final)  # the context did change
    from apps.production_line.runner.runner import _same_verdict as runner_same_verdict

    assert runner_same_verdict(engine, final)
    with fitz.open(ws / "result" / final.artifacts.pdf) as doc:
        assert ctx.job_id in (doc.metadata or {}).get("keywords", "")
        text = "".join(page.get_text() for page in doc)
    assert "SN0042" in text

    # A non-empty engine context is never overwritten (the runner refuses such an engine).
    other = tmp_path / "other"
    build_fake_engine_output(other)
    leaked = load_result(other / "engine" / "result.json").context
    assert leaked != ResultContextV1() and leaked != ctx
    (other / "input").mkdir(parents=True, exist_ok=True)
    (other / "input" / "context.json").write_text(ctx.model_dump_json(), encoding="utf-8")
    assert _render(other).returncode == 0
    kept = ShopfloorInspectionResultV1.model_validate_json((other / "result" / "shopfloor_result.json").read_bytes())
    assert kept.context == leaked

    # An invalid context file fails closed (exit 2), nothing published.
    (ws / "input" / "context.json").write_text('{"serial": 42, "operator": "x"}', encoding="utf-8")
    bad = _render(ws)
    assert bad.returncode == 2
    assert ("ERROR", {"code": "RESULT_INVALID"}) in [(t, {"code": r.get("code")}) for t, r in _records(bad.stdout)]
    assert not (ws / "result" / "shopfloor_result.json").exists()


def test_heatmap_values_survive_into_result_dir(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    _inproc(fake)
    dev = np.fromfile(fake.workspace / "result" / RESULT_FILES["viewer_deviations"], dtype="<f4")
    engine = np.fromfile(fake.engine_dir / "deviations.f32", dtype="<f4")
    np.testing.assert_array_equal(np.isnan(dev), np.isnan(engine))  # no decimation below the cap
    np.testing.assert_array_equal(dev[~np.isnan(dev)], engine[~np.isnan(engine)])
