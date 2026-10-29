"""Synthetic line scorecard: a seeded synthetic grid run end to end through the line.

SYNTHETIC ENGINEERING CAPABILITY. NOT PHYSICAL ACCURACY. NOT PROMOTION EVIDENCE.

What it answers: on parts with exactly known truth, does the *installed line*
report correct values and verdicts? The installed line here means the pinned
engine bundle, the supervised JobRunner, the C06 render step and the result
validation. E4's tier runner (``python -m domains.production.engine.eval run``)
measures the engine alone and owns the G1 gates (plan §10 amendment 2). This
tool reuses E4's pieces instead of re-implementing them:

- E2 :func:`~domains.production.engine.synthetic.cases.generate_case` builds
  each case: real STEP BRep, flat DXF, PartRecipeV1, a virtual ZEISS scan and
  analytic truth.
- E4 :func:`~domains.production.engine.eval.runner.compile_case` runs the E1
  compile with E4's compile cache.
- The scan is archived and registered in a **private** LineStore, the job is
  enqueued and one :class:`~apps.production_line.runner.runner.JobRunner` runs it.
  The run uses the real engine step (``--engine-root BUNDLE``) or E4's truth
  oracle (``--fake-engine``), then the real C06 render step, under the line's
  hard limits. Cases run strictly one after another.
- E4 :func:`~domains.production.engine.eval.scorer.score_case` scores the
  rendered ``result/shopfloor_result.json`` against E2 truth.
  :func:`domains.production.benchmark.evaluator.from_e4_score` and
  :func:`~domains.production.benchmark.evaluator.aggregate` summarise it.

The grid (:data:`GRID`, version :data:`GRID_VERSION`) is frozen and seeded. It is
built from E2 templates on the L_BRACKET, Z_STEP and HAT families, all
asymmetric (no ``symmetric`` feature), with these variations:

- radius and angle through the ``large_radius`` / ``acute`` / ``short_flange``
  feature requests; thickness comes from E2's seeded thickness draw per case;
- exact nominal parts (as-built delta 0) and ``COMPLETE_ERRORS`` /
  ``COMPLETE_SPRINGBACK`` parts, whose injected deltas are E2's policy draws
  (angle ±0.2..3°, flange ±0.1..1.5 mm, bend line ±0.1..1.25 mm, springback
  0.5..2.5°). The scorecard reports the realised true deviations. It does not
  hand-pick deltas, because E2's generator does not take explicit magnitudes;
- scanner noise sigma 0.02 / 0.05 / 0.10 mm on E2's ASSUMED scanner model
  (not the amendment-4b model card);
- one occluded case (HAT, ``SPARSE`` plan), where hidden edges must give
  UNKNOWN, not an error. Occlusion is checked against E2 truth, not assumed:
  each case row lists ``truth_unobservable`` (applicable characteristics whose
  E2 ``observability.observable`` is false) and how the line reported them, and
  the scorecard's ``occlusion`` block says whether the occluded cells actually
  hid anything (``exercised``). Grid 1 used ``TOP_ONLY``, which on this part
  hid nothing that is scored, so the case did not test occlusion.

Case *i* uses ``GRID[i % len(GRID)]`` and the DEV seed
``case_seed("DEV", family, "c12-line-grid-s<seed>", i)``. QUAL seeds are never
used.

Outputs under ``--out``:

- ``scorecard.json`` / ``scorecard.md`` (headed :data:`HEADER`): per-quantity
  MAE / P95 / max, false OUT on conforming parts, missed OUT on nonconforming
  parts, the NOT_MEASURED rate, and per-case ``duration_s`` and
  ``peak_rss_bytes``. An informational E4 G1 gate view of the same rows is
  included; it is never qualifying, because this grid is not an E2 tier.
- ``cases/`` (E2 case directories, reused when identical), ``compile_cache/``
  (E4), ``line/`` (the private data root, rebuilt each run) and
  ``records/<case_id>.json`` (the E4 case score and the evaluation).

``--out`` may not be inside the repository or its research directories. A
non-empty ``--out`` must already be a scorecard directory (it holds
:data:`MARKER`).

With ``--fake-engine`` the numbers describe the line plumbing and the harness,
**not the engine**. The oracle reports truth plus a small seeded error, and the
scorecard says so in its header.

Exit codes: 0 scorecard written (engine failures are counted in it), 2 refused
or invalid input, 3 internal error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "HEADER",
    "GRID_VERSION",
    "GRID",
    "MARKER",
    "ORACLE_MODULE",
    "GridCell",
    "ScorecardConfig",
    "ScorecardError",
    "grid_specs",
    "OCCLUDED_PLANS",
    "truth_unobservable",
    "occlusion_summary",
    "run_scorecard",
    "scorecard_markdown",
    "main",
]

HEADER = "SYNTHETIC ENGINEERING CAPABILITY — NOT PHYSICAL ACCURACY — NOT PROMOTION EVIDENCE"
GRID_VERSION = "C12-LINE-GRID-2 (2026-10-17)"
MARKER = ".c12_synthetic_scorecard"
ORACLE_MODULE = "domains.production.engine.eval.oracle"
SCORECARD_SCHEMA = "sherman_c12_line_scorecard_v1"
#: Oracle error model under --fake-engine (mm and deg sigma; the oracle's U stays at its default).
FAKE_ENGINE_MODE = "noise=0.02"
STATION_ID = "scorecard"
_FAILURE_TO_RUN_STATUS = {
    "WORKER_TIMEOUT": "ENGINE_TIMEOUT",
    "WORKER_RSS_LIMIT": "ENGINE_RSS_LIMIT",
    "WORKER_CRASH": "ENGINE_CRASH",
    "WORKER_ERROR": "ENGINE_EXIT_3",
    "WORKER_NO_RESULT": "RESULT_MISSING",
    "ENGINE_OUTPUT_INVALID": "RESULT_INVALID",
}


@dataclass(frozen=True)
class GridCell:
    family: str
    scenario: str
    features: tuple[str, ...]
    plan: str
    output: str
    exact: bool
    noise_mm: float

    def label(self) -> str:
        feats = "+".join(self.features) or "plain"
        return f"{self.family}/{self.scenario}/{feats}/{self.plan}/{self.output}/noise={self.noise_mm:g}"


#: Frozen grid (module docstring). Order matters: --cases N takes the first N cells (cycling).
GRID: tuple[GridCell, ...] = (
    GridCell("L_BRACKET", "COMPLETE_NOMINAL", (), "FULL", "points", True, 0.02),
    GridCell("Z_STEP", "COMPLETE_ERRORS", (), "FULL", "points", False, 0.05),
    GridCell("HAT", "COMPLETE_ERRORS", ("acute",), "FULL", "mesh", False, 0.05),
    GridCell("L_BRACKET", "COMPLETE_ERRORS", ("large_radius",), "FULL", "points", False, 0.10),
    GridCell("Z_STEP", "COMPLETE_NOMINAL", ("short_flange",), "FULL", "mesh", True, 0.02),
    GridCell("HAT", "COMPLETE_ERRORS", (), "SPARSE", "points", False, 0.05),  # occluded-edge case (truth-checked)
    GridCell("L_BRACKET", "COMPLETE_ERRORS", ("acute",), "FULL", "mesh", False, 0.05),
    GridCell("HAT", "COMPLETE_NOMINAL", ("large_radius",), "FULL", "points", True, 0.10),
    GridCell("Z_STEP", "COMPLETE_ERRORS", ("acute",), "FULL", "points", False, 0.10),
    GridCell("HAT", "COMPLETE_SPRINGBACK", (), "FULL", "points", False, 0.02),
    GridCell("L_BRACKET", "COMPLETE_ERRORS", ("short_flange",), "FULL", "points", False, 0.02),
    GridCell("Z_STEP", "COMPLETE_ERRORS", ("large_radius",), "FULL", "mesh", False, 0.05),
)


class ScorecardError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ScorecardConfig:
    out: Path
    cases: int
    seed: int = 0
    engine_root: Optional[Path] = None
    fake_engine: bool = False
    #: None: E2 native scanner spacing; an int: explicit point-count override (recorded).
    point_target: Optional[int] = None
    python: Path = Path(sys.executable)
    engine_timeout_s: float = 180.0
    render_timeout_s: float = 120.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def grid_specs(n: int, seed: int, point_target: Optional[int]) -> list[tuple[Any, GridCell]]:
    """The first ``n`` (CaseSpec, GridCell) pairs of the grid for ``seed``."""
    from domains.production.engine.synthetic.cases import FAMILY_CODES, CaseSpec, case_seed, opaque_sku
    from domains.production.engine.synthetic.families import FeatureRequest

    if n < 1:
        raise ScorecardError("VALUE_INVALID", "--cases must be >= 1")
    if seed < 0:
        raise ScorecardError("VALUE_INVALID", "--seed must be >= 0")
    group = f"c12-line-grid-s{seed}"
    out = []
    for i in range(n):
        cell = GRID[i % len(GRID)]
        s = case_seed("DEV", cell.family, group, i)
        spec = CaseSpec(
            case_id=f"C12-DEV-{FAMILY_CODES[cell.family]}-s{seed}-{i:03d}", tier="c12_line_grid", seedset="DEV",
            family=cell.family, template_index=i, scenario=cell.scenario,
            features=FeatureRequest(**{f: True for f in cell.features}), plan=cell.plan, output=cell.output,
            exact=cell.exact, seed=s, point_target=point_target, sku=opaque_sku("DEV", s),
        )
        out.append((spec, cell))
    return out


def _scanner_for(cell: GridCell) -> Any:
    from domains.production.engine.synthetic.scanner import ScannerModel

    base = ScannerModel()
    return replace(base, noise_sigma_mm=cell.noise_mm,
                   notes=f"E2 ASSUMED defaults with noise_sigma_mm={cell.noise_mm:g} (C12 grid variant; not a model card).")


# ---------------------------------------------------------------------------
# line plumbing
# ---------------------------------------------------------------------------


class _Provider:
    """In-memory RecipeProvider for the private data root."""

    def __init__(self) -> None:
        self.recipes: dict[str, Any] = {}

    def load(self, recipe_id: str) -> Any:
        return self.recipes[recipe_id]


def _prepare_out(out: Path) -> None:
    from apps.production_line.settings import forbidden_location_reason

    if not out.is_absolute():
        raise ScorecardError("OUT_INVALID", "--out must be an absolute path")
    reason = forbidden_location_reason(out, _repo_root())
    if reason:
        raise ScorecardError("OUT_FORBIDDEN", f"--out {reason}")
    if out.exists() and any(out.iterdir()) and not (out / MARKER).is_file():
        raise ScorecardError("OUT_NOT_A_SCORECARD", f"{out} is not empty and is not a scorecard directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("C12 synthetic line scorecard output directory\n", encoding="utf-8")
    for name in ("line", "records"):
        if (out / name).exists():
            shutil.rmtree(out / name)
    for name in ("scorecard.json", "scorecard.md"):
        (out / name).unlink(missing_ok=True)


def _engine(cfg: ScorecardConfig) -> tuple[Path, Optional[str], str]:
    """(engine_root, bundle_id, mode)."""
    if cfg.fake_engine == (cfg.engine_root is not None):
        raise ScorecardError("VALUE_INVALID", "pass exactly one of --engine-root BUNDLE or --fake-engine")
    if cfg.fake_engine:
        return _repo_root(), None, "FAKE_ENGINE_E4_TRUTH_ORACLE"
    from apps.production_line import engine_bundle

    root = Path(cfg.engine_root).absolute()  # type: ignore[arg-type]
    manifest_path = root / engine_bundle.MANIFEST_NAME
    if not manifest_path.is_file():
        raise ScorecardError("BUNDLE_INVALID", f"{root} has no {engine_bundle.MANIFEST_NAME}")
    problems = engine_bundle.verify(root)
    if problems:
        raise ScorecardError("BUNDLE_INVALID", f"bundle verify failed: {problems[:5]}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return root, str(manifest.get("bundle_id")), f"ENGINE_BUNDLE_{str(manifest.get('profile', '')).upper()}"


def _install_case(tc: Any, co: Any, line_root: Path, provider: _Provider) -> Any:
    """Install the case's recipe, CAD and compiled part into the private data root."""
    from apps.production_line.ingest.ports import ResolvedRecipe
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.part_recipe import PartRecipeV1

    recipe = PartRecipeV1.model_validate(json.loads((tc.case_dir / "recipe.json").read_text(encoding="utf-8")))
    cad_dest = line_root / recipe.cad.stored_rel_path
    cad_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tc.case_dir / "part.step", cad_dest)
    compiled_dir = line_root / "recipes" / "compiled" / recipe.recipe_sha256
    if compiled_dir.exists():
        shutil.rmtree(compiled_dir)
    shutil.copytree(co.compiled_dir, compiled_dir)
    resolved = ResolvedRecipe(
        recipe_id=recipe.recipe_id, recipe_version=recipe.recipe_version, recipe_sha256=recipe.recipe_sha256,
        sku=recipe.sku, recipe_json_bytes=canonical_json_bytes(recipe), cad_path=cad_dest,
        cad_sha256=recipe.cad.sha256, part_name=recipe.part_name, part_revision=recipe.part_revision,
        compiled_dir=compiled_dir, compiled_part_sha256=co.compiled_part_sha256,
    )
    provider.recipes[recipe.recipe_id] = resolved
    return recipe


def _enqueue_scan(store: Any, line_root: Path, tc: Any, case_doc: Mapping[str, Any], recipe: Any,
                  compiled_sha: str) -> str:
    from apps.production_line.ingest.hotfolder import sha256_file
    from apps.production_line.ingest.ply_probe import probe
    from apps.production_line.ingest.ports import minimal_parse
    from contracts.production.layout import archive_scan_rel_path

    scan = tc.case_dir / "scan.ply"
    sha = sha256_file(scan)
    pr = probe(scan)
    if not pr.ok:
        raise ScorecardError("SCAN_INVALID", f"{tc.case_id}: generated scan fails the line probe ({pr.reason})")
    now = datetime.now(timezone.utc)
    rel = archive_scan_rel_path(sha, ".ply", now.year, now.month)
    dest = line_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scan, dest)
    match = {"status": "MATCHED", "recipe_id": recipe.recipe_id, "recipe_version": recipe.recipe_version,
             "recipe_sha256": recipe.recipe_sha256}
    row, _ = store.register_scan(
        sha256=sha, original_name=str(case_doc.get("zeiss_filename") or "scan.ply")[:255], size_bytes=pr.size_bytes,
        station_id=STATION_ID, stored_rel_path=rel, parse=minimal_parse("MATCHED", recipe.sku), match=match,
        format=pr.fmt, vertex_count=pr.vertex_count, face_count=pr.face_count,
    )
    job_id, _ = store.match_and_enqueue(row["scan_file_id"], match=match, sku=recipe.sku,
                                        compiled_part_sha256=compiled_sha)
    return job_id


def _run_status(job: Mapping[str, Any]) -> tuple[str, str]:
    """(E4 run status, detail) for a finished line job."""
    from apps.production_line.store.vocabulary import ENGINE_INPUT_ERROR_CODES

    if job["execution_status"] == "completed":
        return "OK", ""
    failure = job.get("failure") or {}
    code = str(failure.get("code") or job["execution_status"])
    detail = f"LINE:{code}: {str(failure.get('message') or '')[:300]}"
    if code in _FAILURE_TO_RUN_STATUS:
        return _FAILURE_TO_RUN_STATUS[code], detail
    if code in ENGINE_INPUT_ERROR_CODES:
        return "ENGINE_EXIT_2", detail
    if code.startswith("POSTPROCESS"):
        return "RESULT_MISSING", detail
    return "ENGINE_EXIT_OTHER", detail


def ratio_doc(numerator: int, denominator: int) -> dict[str, Any]:
    """Explicit-denominator rate (``value`` is None on an empty denominator), as in the evaluator."""
    return {"numerator": numerator, "denominator": denominator,
            "value": None if denominator == 0 else numerator / denominator}


#: Scan plans that hide part of the part on purpose (E2 SCAN_PLANS other than FULL).
OCCLUDED_PLANS: frozenset[str] = frozenset({"TOP_ONLY", "SPARSE", "SINGLE"})


def truth_unobservable(tc: Any) -> list[str]:
    """Applicable characteristics that E2 truth marks as not observable in this scan."""
    return sorted(c.characteristic_id for c in tc.characteristics.values()
                  if c.applicable and c.observable is False)


def _reported_as(score: Any, ids: Sequence[str]) -> dict[str, str]:
    """How the line reported each of ``ids``: MEASURED:<decision>, NOT_MEASURED:<reason>, ABSENT, RUN_FAILED."""
    rows = {r.characteristic_id: r for r in score.chars}
    out: dict[str, str] = {}
    for cid in ids:
        r = rows.get(cid)
        if score.case.run_status != "OK":
            out[cid] = f"RUN_FAILED:{score.case.run_status}"
        elif r is None or r.engine_state == "ABSENT":
            out[cid] = "ABSENT"
        elif r.engine_state == "MEASURED":
            out[cid] = f"MEASURED:{r.decision}"
        else:
            out[cid] = f"{r.engine_state}:{r.reason}"
    return out


def _within_u(score: Any, ids: Sequence[str]) -> dict[str, Optional[bool]]:
    """For each MEASURED id: is |actual - truth| within the reported U (E4's ``within_u``)?"""
    rows = {r.characteristic_id: r for r in score.chars}
    return {cid: rows[cid].within_u for cid in ids
            if score.case.run_status == "OK" and cid in rows and rows[cid].engine_state == "MEASURED"}


def occlusion_summary(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The scorecard's ``occlusion`` block (module docstring)."""
    occluded = [r for r in case_rows if r.get("plan") in OCCLUDED_PLANS]
    reported = [v for r in occluded for v in (r.get("unobservable_reported") or {}).values()]
    classes: dict[str, int] = {}
    for v in reported:
        key = v.split(":", 1)[0]
        classes[key] = classes.get(key, 0) + 1
    exercised = any(r.get("truth_unobservable") for r in occluded)
    return {
        "occluded_cases": len(occluded),
        "exercised": exercised,
        "truth_unobservable_total": len(reported),
        "reported_classes": dict(sorted(classes.items())),
        "decided_on_unobservable": sum(1 for v in reported if v in ("MEASURED:PASS", "MEASURED:FAIL")),
        "measured_on_unobservable_within_u": ratio_doc(
            sum(1 for r in occluded for v in (r.get("unobservable_within_u") or {}).values() if v is True),
            sum(1 for r in occluded for v in (r.get("unobservable_within_u") or {}).values() if v is not None)),
        "engine_failures": sum(1 for r in occluded if r.get("run_status") != "OK"),
        "note": ("no occluded case in this run" if not occluded else
                 "the occluded cases hid nothing that is scored: occlusion NOT exercised" if not exercised else
                 "E2's observable flag is a conservative heuristic (200 OML points, 60 % far-edge coverage). "
                 "UNKNOWN (NOT_MEASURED) is the expected report for a hidden edge; a value the engine still "
                 "measures there is acceptable only within its U (counted above); an engine failure is a defect"),
    }


def _true_deviation_ranges(tc: Any) -> dict[str, Any]:
    out: dict[str, list[float]] = {}
    for c in tc.characteristics.values():
        if c.true_deviation is not None and c.applicable:
            out.setdefault(c.kind, []).append(float(c.true_deviation))
    return {k: {"n": len(v), "min": round(min(v), 4), "max": round(max(v), 4)} for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run_scorecard(cfg: ScorecardConfig, *, progress: Optional[Callable[[int, int, Mapping[str, Any]], None]] = None
                  ) -> dict[str, Any]:
    """Run the grid (module docstring); returns the scorecard document."""
    from apps.production_line.runner.runner import JobRunner, RunnerConfig
    from apps.production_line.settings import LinePaths, ensure_layout
    from apps.production_line.store.line_store import LineStore
    from contracts.production.inspection_result import ShopfloorInspectionResultV1
    from domains.production.benchmark.evaluator import aggregate, from_e4_score
    from domains.production.engine.eval import runner as e4_runner
    from domains.production.engine.eval.engine_output import load_compiled
    from domains.production.engine.eval.scorer import EngineRun, build_scorecard, score_case
    from domains.production.engine.eval.truth import load_truth_case
    from domains.production.engine.synthetic.cases import generate_case, load_case

    out = Path(cfg.out)
    specs = grid_specs(cfg.cases, cfg.seed, cfg.point_target)  # validate before touching --out
    engine_root, bundle_id, mode = _engine(cfg)
    _prepare_out(out)
    cases_dir, line_root, records = out / "cases", out / "line", out / "records"
    for d in (cases_dir, records):
        d.mkdir(parents=True, exist_ok=True)
    paths = ensure_layout(LinePaths.for_root(line_root))
    store = LineStore(paths.db_line)
    provider = _Provider()
    extra_env = {"SHERMAN_E4_ORACLE_CASES": str(cases_dir), "SHERMAN_E4_ORACLE_MODE": FAKE_ENGINE_MODE} if cfg.fake_engine else {}
    rcfg = RunnerConfig(
        engine_root=engine_root, python_exe=Path(cfg.python), data_root=line_root,
        inspect_module=ORACLE_MODULE if cfg.fake_engine else "domains.production.engine.inspect",
        engine_timeout_s=min(float(cfg.engine_timeout_s), 180.0), render_timeout_s=float(cfg.render_timeout_s),
        engine_bundle_id=bundle_id, extra_env=extra_env, supervisor_poll_s=0.2, idle_poll_s=0.2,
        thread_env={k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    )
    runner = JobRunner(rcfg, store, provider, instance_id=f"scorecard@{socket.gethostname()}", recover_on_start=False)
    e4cfg = e4_runner.RunnerConfig(tier_dir=cases_dir, out_dir=out, repo_root=engine_root, python=str(cfg.python))
    compile_env = e4_runner.engine_env(e4cfg)
    compile_fp = e4_runner.code_fingerprint(engine_root, e4cfg.compile_code_roots)

    started = _now_iso()
    scores = []
    evaluations = []
    case_rows: list[dict[str, Any]] = []
    for i, (spec, cell) in enumerate(specs):
        t0 = time.monotonic()
        entry = generate_case(spec, cases_dir, scanner_model=_scanner_for(cell), ray_threads=1)
        gen_s = time.monotonic() - t0
        case_dir = cases_dir / spec.case_id
        tc = load_truth_case(case_dir)
        case_doc = load_case(case_dir, verify=False)["case"]
        t1 = time.monotonic()
        co = e4_runner.compile_case(tc, e4cfg, compile_fp=compile_fp, env=compile_env)
        compile_s = time.monotonic() - t1
        compiled = None
        if co.status in ("COMPILED", "COMPILE_BLOCKED") and (co.compiled_dir / "compiled_part.json").is_file():
            compiled = load_compiled(co.compiled_dir / "compiled_part.json")
        result = None
        job: dict[str, Any] = {}
        line_wall = None
        if co.status != "COMPILED":
            run = EngineRun(co.status, co.detail)
        else:
            recipe = _install_case(tc, co, line_root, provider)
            job_id = _enqueue_scan(store, line_root, tc, case_doc, recipe, str(co.compiled_part_sha256))
            t2 = time.monotonic()
            ran = runner.run_once()
            line_wall = time.monotonic() - t2
            if ran != job_id:
                raise ScorecardError("INTERNAL", f"runner leased {ran!r}, expected {job_id}")
            job = store.get_job(job_id) or {}
            status, detail = _run_status(job)
            if status == "OK":
                rpath = line_root / "jobs" / job_id / "result" / "shopfloor_result.json"
                try:
                    result = ShopfloorInspectionResultV1.model_validate(json.loads(rpath.read_text(encoding="utf-8")))
                except (OSError, ValueError) as exc:
                    status, detail = "RESULT_INVALID", f"LINE:shopfloor_result.json: {str(exc).splitlines()[0][:200]}"
            run = EngineRun(status, detail, wall_s=job.get("duration_s"), peak_rss_bytes=job.get("peak_rss_bytes"),
                            engine_version=result.engine.engine_version if result is not None else None)
        score = score_case(tc, run, result, compiled)
        line_status = "OK" if job.get("execution_status") == "completed" else (
            (job.get("failure") or {}).get("code") or co.status)
        ev = from_e4_score(score, sku_family=tc.family, duration_s=job.get("duration_s"),
                           peak_rss_bytes=job.get("peak_rss_bytes"),
                           extras={"grid_cell": cell.label(), "line_status": line_status})
        scores.append(score)
        evaluations.append(ev)
        hidden = truth_unobservable(tc)
        row = {
            "case_id": spec.case_id, "grid_cell": cell.label(), "family": cell.family, "scenario": tc.scenario,
            "plan": cell.plan, "noise_mm": cell.noise_mm, "points": tc.point_count,
            "density_policy": tc.density_policy, "inspection_stage": tc.inspection_stage,
            "compile_status": co.status, "line_status": line_status, "run_status": run.status,
            "run_detail": run.detail[:300], "job_id": job.get("job_id"),
            "summary_state": result.summary.state if result is not None else None,
            "validity_state": result.inspection_validity.state if result is not None else None,
            "part_truth": ev.part_truth, "duration_s": job.get("duration_s"),
            "peak_rss_bytes": job.get("peak_rss_bytes"), "line_wall_s": None if line_wall is None else round(line_wall, 3),
            "generate_s": round(gen_s, 3), "compile_s": round(compile_s, 3),
            "generated": not entry.get("skipped", False), "true_deviation_ranges": _true_deviation_ranges(tc),
            "truth_unobservable": hidden, "unobservable_reported": _reported_as(score, hidden),
            "unobservable_within_u": _within_u(score, hidden),
        }
        case_rows.append(row)
        (records / f"{spec.case_id}.json").write_text(
            json.dumps({"case": row, "e4_score": score.to_dict(), "evaluation": ev.to_dict()}, indent=1,
                       sort_keys=True, default=str) + "\n", encoding="utf-8")
        if progress is not None:
            progress(i + 1, len(specs), {"case_id": spec.case_id, "run_status": run.status,
                                         "summary_state": row["summary_state"], "duration_s": row["duration_s"]})

    agg = aggregate(evaluations)
    try:
        e4_card = build_scorecard(scores, tier="c12_line_grid", provenance={"source": "C12 line scorecard"},
                                  bootstrap=False)
        e4_view = {"label": "informational E4 G1 gate view of this grid (never qualifying: not an E2 tier)",
                   "verdict": e4_card.get("verdict"), "regression_verdict": e4_card.get("regression_verdict"),
                   "failed_gates": e4_card.get("failed_gates"), "not_evaluable_gates": e4_card.get("not_evaluable_gates"),
                   "gates": [{k: g.get(k) for k in ("id", "verdict", "value", "threshold", "n", "informational")}
                             for g in e4_card.get("gates", [])]}
    except Exception as exc:  # the line scorecard must not depend on the E4 gate view
        e4_view = {"label": "E4 G1 gate view unavailable", "error": f"{type(exc).__name__}: {exc}"[:300]}
    doc = {
        "schema": SCORECARD_SCHEMA,
        "header": HEADER,
        "label": ("FAKE ENGINE (E4 truth oracle): these numbers measure the line plumbing and harness, NOT the engine"
                  if cfg.fake_engine else "real engine bundle run through the line (synthetic truth)"),
        "engine_mode": mode,
        "engine_bundle_id": bundle_id,
        "grid_version": GRID_VERSION,
        "seedset": "DEV",
        "seed": cfg.seed,
        "cases_requested": cfg.cases,
        "point_target": cfg.point_target,
        "scanner_model": "E2 ASSUMED defaults with per-cell noise_sigma_mm (not the amendment-4b model card)",
        "fake_engine_mode": FAKE_ENGINE_MODE if cfg.fake_engine else None,
        "started_at": started,
        "finished_at": _now_iso(),
        "host": socket.gethostname(),
        "evaluation": agg,
        "cases": case_rows,
        "occlusion": occlusion_summary(case_rows),
        "e4_g1_view": e4_view,
    }
    (out / "scorecard.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "scorecard.md").write_text(scorecard_markdown(doc), encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _f(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _rate(r: Mapping[str, Any]) -> str:
    v = r.get("value")
    return f"{r['numerator']}/{r['denominator']} = {'n/a' if v is None else f'{v:.3f}'}"


def _occlusion_line(occ: Mapping[str, Any]) -> str:
    if not occ.get("occluded_cases"):
        return "No occluded case in this run."
    return (f"{occ['occluded_cases']} occluded case(s); truth marks {occ['truth_unobservable_total']} applicable "
            f"characteristic(s) unobservable; exercised: **{'yes' if occ['exercised'] else 'NO'}**. "
            f"Reported as {occ['reported_classes'] or '{}'}; decided on unobservable: "
            f"{occ['decided_on_unobservable']}; measured there within U: "
            f"{_rate(occ['measured_on_unobservable_within_u'])}; engine failures: {occ['engine_failures']}. "
            f"{occ['note']}.")


def scorecard_markdown(doc: Mapping[str, Any]) -> str:
    ev = doc["evaluation"]
    ov = ev["overall"]
    lines = [
        f"# {doc['header']}",
        "",
        f"**{doc['label']}**",
        "",
        f"- Engine: {doc['engine_mode']} (bundle {doc.get('engine_bundle_id') or 'none'})",
        f"- Grid: {doc['grid_version']}, seed set {doc['seedset']}, seed {doc['seed']}, {len(doc['cases'])} cases, "
        f"point target {doc['point_target'] if doc['point_target'] is not None else 'native spacing'}",
        f"- Scanner: {doc['scanner_model']}",
        f"- Run: {doc['started_at']} → {doc['finished_at']} on {doc['host']}; evaluator {ev['evaluator_version']}",
        "",
        "Synthetic parts with analytic truth, run through the installed line (LineStore, JobRunner, engine step, "
        "C06 render step). It is engineering evidence about the software, never physical accuracy, unseen-family "
        "accuracy or promotion evidence (AGENTS.md: separate ledgers).",
        "",
        "## Per-quantity error (reported actual − truth)",
        "",
        "Decided = PASS/FAIL characteristics (the headline). All measured adds REVIEW and undecided values.",
        "",
        "| Quantity | Unit | n decided | Bias | MAE | P95 abs | Max abs | n all measured | MAE all | P95 abs all | NOT_MEASURED |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, blk in ov["per_kind"].items():
        d, a = blk["errors_decided"], blk["errors_all_measured"]
        lines.append(f"| {kind} | {blk['unit']} | {d['n']} | {_f(d['bias'])} | {_f(d['mae'])} | {_f(d['p95_abs'])} | "
                     f"{_f(d['max_abs'])} | {a['n']} | {_f(a['mae'])} | {_f(a['p95_abs'])} | {_rate(blk['not_measured'])} |")
    cr, pr = ov["characteristic_rates"], ov["part_rates"]
    lines += [
        "",
        "## NOT_MEASURED rate",
        "",
        f"Overall: **{_rate(ov['not_measured'])}** (denominator: scorable truth characteristics that are not "
        f"NOT_APPLICABLE; {ov['not_applicable_count']} NOT_APPLICABLE). Engine/line failures count as NOT_MEASURED.",
        "",
        "## Occlusion",
        "",
        _occlusion_line(doc.get("occlusion") or {}),
        "",
        "## Conformance",
        "",
        "| Rate | Value |",
        "|---|---|",
        f"| False OUT on conforming parts (OUT_OF_TOLERANCE_FOUND) | {_rate(pr['false_out_on_conforming'])} |",
        f"| Missed OUT on nonconforming parts (summary not OUT_OF_TOLERANCE_FOUND) | {_rate(pr['missed_out_on_nonconforming'])} |",
        f"| False accept on nonconforming parts (ALL_MEASURED_IN_TOLERANCE) | {_rate(pr['false_accept_on_nonconforming'])} |",
        f"| Characteristic false OUT on truth IN | {_rate(cr['false_out_on_truth_in'])} |",
        f"| Characteristic missed OUT on truth OUT | {_rate(cr['missed_out_on_truth_out'])} |",
        f"| Characteristic false IN (PASS) on truth OUT | {_rate(cr['false_in_on_truth_out'])} |",
        f"| Decided (PASS/FAIL) on truth IN or OUT | {_rate(cr['decided_on_truth_in_or_out'])} |",
        f"| U calibration: abs(actual − truth) ≤ U on MEASURED | {_rate(ov['u_calibration'])} |",
        "",
        "Characteristic confusion (rows truth, columns reported):",
        "",
        "| Truth \\ reported | " + " | ".join(next(iter(ov["characteristic_confusion"].values())).keys()) + " |",
        "|---|" + "---|" * len(next(iter(ov["characteristic_confusion"].values()))),
    ]
    for t, row in ov["characteristic_confusion"].items():
        lines.append(f"| {t} | " + " | ".join(str(v) for v in row.values()) + " |")
    lines += ["", "Part confusion (truth → summary state): `" + json.dumps(ov["part_confusion"], sort_keys=True) + "`",
              "", "## Per family", "",
              "| Family | Cases | Flange MAE | Angle MAE | Line MAE | NOT_MEASURED | False accept |",
              "|---|---|---|---|---|---|---|"]
    for fam, blk in ev["by_family"].items():
        pk = blk["per_kind"]

        def mae(kind: str) -> str:
            return _f(pk.get(kind, {}).get("errors_decided", {}).get("mae"))

        lines.append(f"| {fam} | {blk['cases']} | {mae('FLANGE_LENGTH')} | {mae('INCLUDED_ANGLE')} | "
                     f"{mae('BEND_LINE_POSITION')} | {_rate(blk['not_measured'])} | "
                     f"{_rate(blk['part_rates']['false_accept_on_nonconforming'])} |")
    lines += ["", "Worst family per metric (higher is worse; small n is noisy):", ""]
    for _path, w in ev["worst_family"].items():
        lines.append(f"- {w['label']}: {w['family'] or 'n/a'} ({_f(w['value'])})")
    rt = ov["runtime"]
    lines += [
        "", "## Per case (duration and peak RSS from the line job: supervised process group, max over steps)", "",
        f"Duration p50 {_f(rt['duration_s']['p50'])} s, p95 {_f(rt['duration_s']['p95'])} s, max "
        f"{_f(rt['duration_s']['max'])} s; peak RSS max {_f(rt['peak_rss_bytes_max'])} bytes.",
        "",
        "| Case | Grid cell | Points | Line status | Run status | Summary | Part truth | duration_s | peak_rss_bytes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in doc["cases"]:
        lines.append(f"| {r['case_id']} | {r['grid_cell']} | {r['points']} | {r['line_status']} | {r['run_status']} | "
                     f"{r['summary_state'] or '—'} | {r['part_truth']} | {_f(r['duration_s'])} | {_f(r['peak_rss_bytes'])} |")
    e4 = doc.get("e4_g1_view") or {}
    lines += ["", "## E4 G1 gate view (informational, never qualifying)", "",
              f"{e4.get('label')}: verdict {e4.get('verdict')}; failed gates "
              f"{', '.join(e4.get('failed_gates') or []) or 'none'}; not evaluable "
              f"{', '.join(e4.get('not_evaluable_gates') or []) or 'none'}.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.ops.synthetic_scorecard",
                                     description="Synthetic line scorecard (synthetic engineering capability only).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine-root", help="engine bundle built by apps.production_line.engine_bundle")
    group.add_argument("--fake-engine", action="store_true", help="E4 truth oracle in place of E3 (harness self-test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-target", type=int, default=None,
                        help="explicit scan point-count override (default: native scanner spacing)")
    args = parser.parse_args(argv)
    cfg = ScorecardConfig(
        out=Path(os.path.abspath(os.path.expanduser(args.out))), cases=args.cases, seed=args.seed,
        engine_root=Path(os.path.abspath(os.path.expanduser(args.engine_root))) if args.engine_root else None,
        fake_engine=args.fake_engine, point_target=args.point_target,
    )

    def progress(i: int, n: int, d: Mapping[str, Any]) -> None:
        sys.stdout.write("PROGRESS\t" + json.dumps({"case": i, "of": n, **d}, sort_keys=True, default=str) + "\n")
        sys.stdout.flush()

    try:
        doc = run_scorecard(cfg, progress=progress)
    except ScorecardError as exc:
        sys.stdout.write("ERROR\t" + json.dumps({"code": exc.code, "message": exc.message}) + "\n")
        return 2
    except Exception as exc:
        sys.stdout.write("ERROR\t" + json.dumps({"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        return 3
    ov = doc["evaluation"]["overall"]
    sys.stdout.write("RESULT\t" + json.dumps({"scorecard": str(cfg.out / "scorecard.json"), "cases": ov["cases"],
                                              "run_status": ov["run_status"], "not_measured": ov["not_measured"]},
                                             sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
