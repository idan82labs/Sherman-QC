"""Tiered synthetic runner: E2 tier -> E1 compile -> E3 inspect -> score (serial).

For every case of an E2 tier directory (``regression`` ~50 cases, ``full`` >= 200
cases, ``runtime`` ~10 cases at ~1M points; amendment 3), one at a time:

1. **Truth** is loaded and every case file hash is verified (:mod:`.truth`).
2. **Compile** with the E1 command line, cached by
   ``sha256(recipe | cad | flat | E1 code fingerprint | compile config)`` under
   ``<out>/compile_cache/<key>/``. The recipe, STEP and flat are staged there
   under fixed names, so no path E1 sees names the case. A transient compile
   failure (timeout, signal, RSS kill) is recorded but never reused from the
   cache: the next run compiles again.
3. **Inspect** with the E3 command line (ENGINE_IO §4) under the runner's hard
   limits (default 180 s, 4 GiB, :mod:`.procmon`), passing ``--scan-sha256``,
   ``--compiled-sha256`` and ``--recipe-sha256`` like the C02 runner. Blindness
   (amendment 4): the scan is staged as ``input/scan.ply`` in a workspace named
   by an opaque hash, the recipe as ``input/recipe.json``; E3 never receives a
   case id, ZEISS-like file name, truth or metadata count. Inputs are read-only
   **copies** (never hard links into the E2 tier), re-hashed after the run: an
   engine that modified its input is RESULT_INCONSISTENT (ENGINE_MODIFIED_INPUT).
4. **Check** the output: model validation plus
   ``check_result_against_compiled`` (ENGINE_IO §4.3); any problem is the run
   status RESULT_INVALID / RESULT_INCONSISTENT -- an engine failure that stays
   in every denominator.
5. **Score** (:mod:`.scorer`) and persist ``cases/<case_id>.json``.

Wall time and peak RSS come from the monitored subprocess, so the runtime
tier measures exactly what the 45 s / 4 GiB gate is about. Two identities make
reruns cheap and honest: the **resume key** (input hashes, E3 code fingerprint,
engine command, config, limits) decides whether the engine must run again;
the **score key** (eval version, eval code fingerprint, thresholds hash)
decides whether a stored engine output must be re-scored. ``force`` reruns
everything. Outputs: ``run_manifest.json``, ``cases/*.json``,
``rows/{cases,chars,bends}.jsonl``, ``scorecard.json`` and ``scorecard.md``.

Run conditions (:func:`run_conditions`: engine command, limits, extra engine
environment) are recorded in the scorecard provenance; anything but the
standard ones makes the scorecard non-qualifying. QUAL seeds (amendment 4c)
run only with ``final_qual_run`` and a ledger, and every such run is logged as
a RUN ledger entry before the first engine call.

:func:`blindness_check` reruns cases with the scan staged under a different
opaque name and directory and compares the results with ``context`` and the
timing fields removed (amendment 4: renamed copies must give identical output).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.production.canonical import canonical_json_bytes, sha256_hex
from contracts.production.line_job import ENGINE_HARD_TIMEOUT_S, ENGINE_RSS_LIMIT_BYTES

from .engine_output import EngineOutput, load_compiled, load_engine_output, load_recipe
from .plyio import sha256_file
from .procmon import ProcessOutcome, run_monitored
from .scorer import EVAL_VERSION, CaseScore, EngineRun, build_scorecard, score_case
from .thresholds import G1_THRESHOLDS, thresholds_sha256
from .truth import TruthCase, TruthError, load_manifest, load_truth_case

__all__ = [
    "REPO_ROOT",
    "RUNNER_SCHEMA",
    "RunnerConfig",
    "RunnerPolicyError",
    "code_fingerprint",
    "engine_env",
    "run_conditions",
    "compile_case",
    "inspect_case",
    "run_tier",
    "rescore_run",
    "write_scorecard",
    "blindness_check",
    "scorecard_markdown",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_SCHEMA = "sherman_e4_run_case_v2"
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")
_TRANSIENT_COMPILE = ("timeout", "rss_limit", "signal")


class RunnerPolicyError(ValueError):
    """A run the evaluation protocol forbids (e.g. QUAL seeds without the final-run opt-in)."""


@dataclass(frozen=True)
class RunnerConfig:
    tier_dir: Path
    out_dir: Path
    repo_root: Path = REPO_ROOT
    python: str = sys.executable
    #: Engine command prefix; the E3 arguments are appended. Default: python -m <engine_module>.
    engine_cmd: tuple[str, ...] | None = None
    engine_module: str = "domains.production.engine.inspect"
    compile_module: str = "domains.production.engine.compile"
    engine_timeout_s: float = float(ENGINE_HARD_TIMEOUT_S)
    engine_rss_limit_bytes: int = int(ENGINE_RSS_LIMIT_BYTES)
    engine_time_budget_s: float = 170.0
    compile_timeout_s: float = 300.0
    compile_rss_limit_bytes: int = int(ENGINE_RSS_LIMIT_BYTES)
    engine_config: Path | None = None
    compile_config: Path | None = None
    engine_bundle_id: str | None = None
    case_ids: tuple[str, ...] | None = None
    families: tuple[str, ...] | None = None
    limit: int | None = None
    #: Rerun the engine even when a case record with the same resume identity exists.
    force: bool = False
    #: Recompile even when the compile cache holds this exact input/code key.
    force_compile: bool = False
    keep_engine_outputs: bool = True
    extra_env: Mapping[str, str] = field(default_factory=dict)
    bootstrap: bool = True
    #: Freeze ledger to check the run against (a scorecard is never qualifying without a match).
    ledger: Path | None = None
    #: sha256 of the ledger's first FREEZE entry, recorded outside the ledger (detects a rewritten ledger).
    ledger_anchor: str | None = None
    #: Explicit opt-in for the one final run on QUAL seeds (amendment 4c).
    final_qual_run: bool = False
    #: Code roots whose fingerprint is part of the engine identity.
    engine_code_roots: tuple[str, ...] = ("domains/production/engine/inspect", "contracts/production")
    compile_code_roots: tuple[str, ...] = ("domains/production/engine/compile", "contracts/production")
    #: Code roots whose fingerprint is part of the score identity (scorer, thresholds, truth loaders).
    eval_code_roots: tuple[str, ...] = ("domains/production/engine/eval",)

    def engine_prefix(self) -> list[str]:
        return list(self.engine_cmd) if self.engine_cmd else [self.python, "-m", self.engine_module]


def code_fingerprint(repo_root: Path, roots: Sequence[str]) -> str:
    """sha256 over (relative path, file sha256) of every ``*.py`` under ``roots``
    (``__pycache__`` excluded); ``"absent"`` entries for missing roots."""
    items: list[str] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            items.append(f"{root}:absent")
            continue
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
        for f in files:
            items.append(f"{f.relative_to(repo_root).as_posix()}:{sha256_file(f)}")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def engine_env(cfg: RunnerConfig) -> dict[str, str]:
    """Environment of every E1/E3 subprocess (ENGINE_IO §3/§4): thread caps,
    PYTHONPATH=<repo>:<repo>/backend, no bytecode writes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHERMAN_E4_")}
    for var in _THREAD_VARS:
        env[var] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{cfg.repo_root}{os.pathsep}{cfg.repo_root / 'backend'}"
    env.update(cfg.extra_env)
    return env


def run_conditions(cfg: RunnerConfig) -> dict[str, Any]:
    """The run conditions a scorecard's qualification depends on
    (:func:`.scorer.run_condition_reasons` compares them with the standard ones)."""
    prefix = cfg.engine_prefix()
    return {"engine_argv_tail": prefix[1:], "engine_python": prefix[0],
            "engine_timeout_s": float(cfg.engine_timeout_s), "engine_rss_limit_bytes": int(cfg.engine_rss_limit_bytes),
            "engine_time_budget_s": float(cfg.engine_time_budget_s),
            "engine_extra_env_keys": sorted(cfg.extra_env), "engine_bundle_id": cfg.engine_bundle_id,
            "compile_timeout_s": float(cfg.compile_timeout_s), "compile_module": cfg.compile_module}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stage(src: Path, dst: Path) -> None:
    """A read-only copy of ``src`` at ``dst`` (never a hard link: the engine must not
    be able to touch the E2 tier through its inputs)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _rmtree(path: Path) -> None:
    """Remove a workspace including read-only staged inputs."""
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
    shutil.rmtree(path)


def _file_sha(cfg_path: Path | None) -> str | None:
    return None if cfg_path is None else sha256_file(cfg_path)


@dataclass
class CompileOutcome:
    status: str  # COMPILED | COMPILE_BLOCKED | COMPILE_FAILED
    key: str
    compiled_dir: Path
    compiled_part_sha256: str | None = None
    detail: str = ""
    process: dict[str, Any] | None = None
    reused: bool = False


def _compile_key(tc: TruthCase, compile_fp: str, cfg: RunnerConfig) -> str:
    recipe = json.loads((tc.case_dir / "recipe.json").read_text(encoding="utf-8"))
    flat = (recipe.get("flat") or {}).get("sha256")
    doc = {"recipe": tc.files.get("recipe.json"), "cad": tc.files.get("part.step"), "flat": flat,
           "compile_fp": compile_fp, "config": _file_sha(cfg.compile_config)}
    return sha256_hex(doc)


def compile_case(tc: TruthCase, cfg: RunnerConfig, *, compile_fp: str, env: Mapping[str, str]) -> CompileOutcome:
    """E1 compile of the case recipe, cached by input and code hashes (transient failures are not reused)."""
    key = _compile_key(tc, compile_fp, cfg)
    cache = cfg.out_dir / "compile_cache" / key[:24]
    out = cache / "out"
    record = cache / "compile_outcome.json"
    if record.is_file() and not cfg.force_compile:
        doc = json.loads(record.read_text(encoding="utf-8"))
        if doc.get("key") == key and not doc.get("transient"):
            return CompileOutcome(doc["status"], key, out, doc.get("compiled_part_sha256"), doc.get("detail", ""),
                                  doc.get("process"), reused=True)
    _rmtree(cache)
    inputs = cache / "inputs"
    _stage(tc.case_dir / "recipe.json", inputs / "recipe.json")
    _stage(tc.case_dir / "part.step", inputs / "part.step")
    recipe = json.loads((inputs / "recipe.json").read_text(encoding="utf-8"))
    argv = [cfg.python, "-m", cfg.compile_module, "--recipe", str(inputs / "recipe.json"),
            "--cad", str(inputs / "part.step"), "--out", str(out)]
    if recipe.get("flat"):
        ext = Path(recipe["flat"].get("stored_rel_path") or "flat.dxf").suffix or ".dxf"
        src = tc.case_dir / ("flat.dxf" if ext.lower() == ".dxf" else f"flat{ext}")
        _stage(src, inputs / f"flat{ext}")
        argv += ["--flat", str(inputs / f"flat{ext}")]
    if cfg.compile_config is not None:
        argv += ["--config", str(cfg.compile_config)]
    proc = run_monitored(argv, cwd=cfg.repo_root, env=env, timeout_s=cfg.compile_timeout_s,
                         rss_limit_bytes=cfg.compile_rss_limit_bytes, stderr_path=cache / "compile.stderr.log")
    status, sha, detail = "COMPILE_FAILED", None, ""
    transient = False
    cp = out / "compiled_part.json"
    if proc.timed_out or proc.rss_exceeded or proc.signal is not None:
        detail = "timeout" if proc.timed_out else ("rss_limit" if proc.rss_exceeded else f"signal {proc.signal}")
        transient = True
    elif cp.is_file() and proc.returncode in (0, 2):
        try:
            compiled = load_compiled(cp)
        except (OSError, ValueError) as exc:
            detail = f"compiled_part.json invalid: {str(exc).splitlines()[0][:200]}"
        else:
            sha = compiled.compiled_part_sha256
            if compiled.status == "COMPILED" and proc.returncode == 0:
                status = "COMPILED"
            else:
                status, detail = "COMPILE_BLOCKED", ",".join(compiled.blocker_codes)
    else:
        err = proc.last("ERROR") or {}
        detail = f"exit {proc.returncode}: {err.get('code', '')} {err.get('message', '')}"[:300]
    out_doc = {"key": key, "status": status, "compiled_part_sha256": sha, "detail": detail, "transient": transient,
               "process": proc.to_dict(), "recorded_at": _now()}
    cache.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(out_doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return CompileOutcome(status, key, out, sha, detail, proc.to_dict())


def _engine_args(cfg: RunnerConfig, compiled_dir: Path, recipe_path: Path, scan_path: Path, out: Path,
                 scan_sha: str, compiled_sha: str, recipe_sha: str) -> list[str]:
    argv = cfg.engine_prefix() + [
        "--compiled", str(compiled_dir / "compiled_part.json"), "--recipe", str(recipe_path), "--scan", str(scan_path),
        "--out", str(out), "--scan-sha256", scan_sha, "--compiled-sha256", compiled_sha,
        "--recipe-sha256", recipe_sha, "--time-budget-s", f"{cfg.engine_time_budget_s:g}",
    ]
    if cfg.engine_bundle_id:
        argv += ["--engine-bundle-id", cfg.engine_bundle_id]
    if cfg.engine_config is not None:
        argv += ["--config", str(cfg.engine_config)]
    return argv


def _classify(proc: ProcessOutcome, out: Path) -> tuple[str, str]:
    if proc.timed_out:
        return "ENGINE_TIMEOUT", f"hard stop after {proc.wall_s:.1f} s"
    if proc.rss_exceeded:
        return "ENGINE_RSS_LIMIT", f"peak RSS {proc.peak_rss_bytes}"
    if proc.signal is not None:
        return "ENGINE_CRASH", f"signal {proc.signal}"
    if proc.returncode == 0:
        return "OK", ""
    code = ""
    err_file = out / "engine_error.json"
    if err_file.is_file():
        try:
            code = json.loads(err_file.read_text(encoding="utf-8")).get("code", "")
        except (OSError, ValueError):
            code = "UNREADABLE_ENGINE_ERROR"
    if not code:
        code = (proc.last("ERROR") or {}).get("code", "")
    if proc.returncode == 2:
        return "ENGINE_EXIT_2", code
    if proc.returncode == 3:
        return "ENGINE_EXIT_3", code
    return "ENGINE_EXIT_OTHER", f"exit {proc.returncode} {code}".strip()


def inspect_case(
    tc: TruthCase,
    co: CompileOutcome,
    cfg: RunnerConfig,
    *,
    env: Mapping[str, str],
    workspace: Path,
    scan_name: str = "scan.ply",
) -> tuple[EngineRun, EngineOutput | None, dict[str, Any]]:
    """Run E3 on one compiled case in ``workspace`` (blind, read-only staged inputs)."""
    _rmtree(workspace)
    inp = workspace / "input"
    scan = inp / scan_name
    _stage(tc.case_dir / "scan.ply", scan)
    recipe_path = inp / "recipe.json"
    _stage(tc.case_dir / "recipe.json", recipe_path)
    out = workspace / "engine"
    recipe = load_recipe(recipe_path)
    compiled = load_compiled(co.compiled_dir / "compiled_part.json")
    scan_sha = tc.files["scan.ply"]
    recipe_sha_file = tc.files.get("recipe.json")
    argv = _engine_args(cfg, co.compiled_dir, recipe_path, scan, out, scan_sha, compiled.compiled_part_sha256,
                        recipe.recipe_sha256)
    proc = run_monitored(argv, cwd=cfg.repo_root, env=env, timeout_s=cfg.engine_timeout_s,
                         rss_limit_bytes=cfg.engine_rss_limit_bytes, stderr_path=workspace / "engine.stderr.log",
                         stdout_path=workspace / "engine.stdout.log")
    status, detail = _classify(proc, out)
    modified = [name for path, digest, name in ((scan, scan_sha, "scan"), (recipe_path, recipe_sha_file, "recipe"))
                if digest is not None and (not path.is_file() or sha256_file(path) != digest)]
    eo: EngineOutput | None = None
    if status == "OK":
        eo = load_engine_output(out, compiled=compiled, recipe=recipe)
        if eo.status != "OK":
            status, detail = eo.status, (eo.error or ",".join(eo.problems))[:500]
        elif eo.result is not None and eo.result.scan.sha256 != scan_sha:
            status, detail = "RESULT_INCONSISTENT", "SCAN_SHA_MISMATCH"
    if modified and status in ("OK", "RESULT_INCONSISTENT", "RESULT_INVALID", "RESULT_MISSING"):
        status, detail = "RESULT_INCONSISTENT", f"ENGINE_MODIFIED_INPUT:{','.join(modified)}"
    version = eo.result.engine.engine_version if eo is not None and eo.result is not None else None
    run = EngineRun(status=status, detail=detail, wall_s=round(proc.wall_s, 3), peak_rss_bytes=proc.peak_rss_bytes,
                    engine_version=version)
    return run, eo, proc.to_dict()


def _resume_key(tc: TruthCase, co: CompileOutcome, engine_fp: str, cfg: RunnerConfig) -> str:
    """Engine identity of a case run: the engine must run again when it changes (including
    the compile outcome, so a recompile after a transient failure reruns the case)."""
    return sha256_hex({
        "scan": tc.files.get("scan.ply"), "truth": tc.files.get("truth.json"), "compile_key": co.key,
        "compile_status": co.status, "compiled_part_sha256": co.compiled_part_sha256,
        "engine_fp": engine_fp, "engine_cmd": cfg.engine_prefix()[1:], "engine_config": _file_sha(cfg.engine_config),
        "engine_env": sorted(cfg.extra_env.items()), "engine_bundle_id": cfg.engine_bundle_id,
        "limits": [cfg.engine_timeout_s, cfg.engine_rss_limit_bytes, cfg.engine_time_budget_s],
    })


def _score_key(eval_fp: str) -> str:
    """Score identity: a stored engine output is re-scored when it changes."""
    return sha256_hex({"eval": EVAL_VERSION, "eval_fp": eval_fp, "thresholds": thresholds_sha256(G1_THRESHOLDS)})


def _compiled_for(co_status: str, compiled_dir: Path) -> Any:
    if co_status not in ("COMPILED", "COMPILE_BLOCKED"):
        return None
    try:
        return load_compiled(compiled_dir / "compiled_part.json")
    except (OSError, ValueError):
        return None


def _rescore(doc: Mapping[str, Any], tc: TruthCase, compiled_dir: Path) -> CaseScore | None:
    """Score a stored case record again (current eval code) from its stored engine output.
    None when the stored output needed for that is gone."""
    run = EngineRun(**doc["run"])
    compiled = _compiled_for(doc["compile"]["status"], compiled_dir)
    result = None
    if run.status == "OK":
        eo_doc = doc.get("engine_output") or {}
        out_dir = Path(eo_doc.get("dir") or "")
        if not (out_dir / "result.json").is_file() or compiled is None:
            return None
        eo = load_engine_output(out_dir, compiled=compiled, recipe=load_recipe(tc.case_dir / "recipe.json"))
        if eo.status != "OK" or eo.result_sha256 != eo_doc.get("result_sha256"):
            return None
        result = eo.result
    return score_case(tc, run, result, compiled)


def _write_record(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _ledger_check(cfg_ledger: Path, *, case_ids: Sequence[str], manifest: Mapping[str, Any],
                  engine_config_sha256: str | None, repo_root: Path, engine_identity: Mapping[str, Any],
                  anchor: str | None) -> dict[str, Any]:
    from .freeze import FROZEN_CODE, check_against_ledger, current_code_fingerprints

    return check_against_ledger(
        cfg_ledger, case_ids=list(case_ids), tier=manifest.get("tier"), seedset=manifest.get("seedset"),
        scanner_model_sha256=manifest.get("scanner_model_sha256"), engine_config_sha256=engine_config_sha256,
        code_fingerprints=current_code_fingerprints(repo_root, FROZEN_CODE), engine_identity=engine_identity,
        anchor_sha256=anchor)


def run_tier(cfg: RunnerConfig, *, progress: Callable[[int, int, dict[str, Any]], None] | None = None
             ) -> dict[str, Any]:
    """Run, score and summarise one E2 tier directory (module docstring)."""
    manifest = load_manifest(cfg.tier_dir)
    out = cfg.out_dir
    entries = list(manifest["cases"])
    if cfg.case_ids:
        wanted = set(cfg.case_ids)
        entries = [e for e in entries if e["case_id"] in wanted]
    if cfg.families:
        fams = set(cfg.families)
        entries = [e for e in entries if e["family"] in fams]
    if cfg.limit is not None:
        entries = entries[: cfg.limit]
    engine_fp = code_fingerprint(cfg.repo_root, cfg.engine_code_roots)
    compile_fp = code_fingerprint(cfg.repo_root, cfg.compile_code_roots)
    eval_fp = code_fingerprint(cfg.repo_root, cfg.eval_code_roots)
    engine_cfg_sha = _file_sha(cfg.engine_config)
    engine_identity = {"engine_fingerprint": engine_fp, "engine_config_sha256": engine_cfg_sha,
                       "engine_cmd": cfg.engine_prefix()[1:]}
    case_ids = [e["case_id"] for e in entries]
    started = _now()
    # The QUAL policy applies when the manifest or any selected case's own spec says QUAL
    # (an edited manifest cannot hide QUAL cases).
    seedsets = {manifest.get("seedset")}
    for e in entries:
        try:
            spec = json.loads((cfg.tier_dir / e["case_id"] / "case.json").read_text(encoding="utf-8")).get("spec") or {}
        except (OSError, ValueError):
            spec = {}
        seedsets.add(spec.get("seedset"))
    if "QUAL" in seedsets:
        if not cfg.final_qual_run:
            raise RunnerPolicyError("QUAL seeds are used only on the final run (amendment 4c): pass --final-qual-run")
        if cfg.ledger is None:
            raise RunnerPolicyError("a QUAL run must be logged: pass --ledger")
        from .freeze import append_entry

        append_entry(cfg.ledger, kind="RUN", label=f"QUAL {manifest.get('tier')} run", items={
            "tier": manifest.get("tier"), "seedset": "QUAL", "case_ids_sha256": sha256_hex(case_ids),
            "cases": len(case_ids), **engine_identity, "eval_fingerprint": eval_fp, "run_started_at": started})
    (out / "cases").mkdir(parents=True, exist_ok=True)
    env = engine_env(cfg)
    skey = _score_key(eval_fp)
    t_start = time.monotonic()
    conditions = run_conditions(cfg)
    run_manifest = {
        "schema": "sherman_e4_run_manifest_v2",
        "eval_version": EVAL_VERSION,
        "started_at": started,
        "tier": manifest.get("tier"),
        "seedset": manifest.get("seedset"),
        "tier_dir": str(cfg.tier_dir),
        "generator_version": manifest.get("generator_version"),
        "scanner_model_sha256": manifest.get("scanner_model_sha256"),
        "engine_cmd": cfg.engine_prefix()[1:],
        "fingerprints": {"engine": engine_fp, "compile": compile_fp, "eval": eval_fp},
        "code_roots": {"engine": list(cfg.engine_code_roots), "compile": list(cfg.compile_code_roots),
                       "eval": list(cfg.eval_code_roots)},
        "limits": {"engine_timeout_s": cfg.engine_timeout_s, "engine_rss_limit_bytes": cfg.engine_rss_limit_bytes,
                   "engine_time_budget_s": cfg.engine_time_budget_s, "compile_timeout_s": cfg.compile_timeout_s},
        "run_conditions": conditions,
        "configs": {"engine": engine_cfg_sha, "compile": _file_sha(cfg.compile_config),
                    "engine_path": None if cfg.engine_config is None else str(cfg.engine_config),
                    "compile_path": None if cfg.compile_config is None else str(cfg.compile_config)},
        "thresholds_sha256": thresholds_sha256(G1_THRESHOLDS),
        "score_key": skey,
        "case_ids": case_ids,
        "final_qual_run": cfg.final_qual_run,
        "ledger": None if cfg.ledger is None else str(cfg.ledger),
        "serial": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    scores: list[CaseScore] = []
    for i, entry in enumerate(entries, start=1):
        case_dir = cfg.tier_dir / entry["case_id"]
        record_path = out / "cases" / f"{entry['case_id']}.json"
        try:
            tc = load_truth_case(case_dir, verify_hashes=True)
        except TruthError as exc:
            raise TruthError(f"{entry['case_id']}: {exc}") from exc
        co = compile_case(tc, cfg, compile_fp=compile_fp, env=env)
        key = _resume_key(tc, co, engine_fp, cfg)
        if record_path.is_file() and not cfg.force:
            doc = json.loads(record_path.read_text(encoding="utf-8"))
            if doc.get("resume_key") == key and doc.get("schema") == RUNNER_SCHEMA:
                if doc.get("score_key") == skey:
                    scores.append(CaseScore.from_dict(doc["score"]))
                    if progress:
                        progress(i, len(entries), {"case_id": tc.case_id, "skipped": True,
                                                   "status": doc["run"]["status"]})
                    continue
                again = _rescore(doc, tc, co.compiled_dir)
                if again is not None:
                    doc = dict(doc, score=again.to_dict(), score_key=skey, rescored_at=_now())
                    _write_record(record_path, doc)
                    scores.append(again)
                    if progress:
                        progress(i, len(entries), {"case_id": tc.case_id, "skipped": True, "rescored": True,
                                                   "status": doc["run"]["status"]})
                    continue
        compiled = None
        proc_doc: dict[str, Any] | None = None
        eo = None
        workspace = out / "work" / key[:16]
        if co.status == "COMPILED":
            compiled = load_compiled(co.compiled_dir / "compiled_part.json")
            run, eo, proc_doc = inspect_case(tc, co, cfg, env=env, workspace=workspace)
            if not cfg.keep_engine_outputs:
                _rmtree(workspace / "input")
                if eo is not None:
                    for p in sorted((workspace / "engine").rglob("*"), reverse=True):
                        if p.is_file() and p.name != "result.json":
                            p.unlink()
        else:
            run = EngineRun(status=co.status, detail=co.detail)
            compiled = _compiled_for(co.status, co.compiled_dir)
        result = eo.result if eo is not None else None
        score = score_case(tc, run, result, compiled)
        scores.append(score)
        doc = {
            "schema": RUNNER_SCHEMA,
            "case_id": tc.case_id,
            "resume_key": key,
            "score_key": skey,
            "recorded_at": _now(),
            "compile": {"status": co.status, "key": co.key, "compiled_part_sha256": co.compiled_part_sha256,
                        "detail": co.detail, "reused": co.reused, "process": co.process,
                        "compiled_dir": str(co.compiled_dir)},
            "run": asdict(run),
            "process": proc_doc,
            "engine_output": None if eo is None else {"status": eo.status, "problems": eo.problems,
                                                       "result_sha256": eo.result_sha256, "dir": str(eo.out_dir)},
            "score": score.to_dict(),
        }
        _write_record(record_path, doc)
        if progress:
            progress(i, len(entries), {"case_id": tc.case_id, "status": run.status, "wall_s": run.wall_s,
                                       "peak_rss_bytes": run.peak_rss_bytes})
    ledger_check = None
    if cfg.ledger is not None:
        ledger_check = _ledger_check(cfg.ledger, case_ids=case_ids, manifest=manifest,
                                     engine_config_sha256=engine_cfg_sha, repo_root=cfg.repo_root,
                                     engine_identity=engine_identity, anchor=cfg.ledger_anchor)
    return write_scorecard(out, scores, manifest=manifest, provenance={
        "run_started_at": started, "run_finished_at": _now(), "fingerprints": run_manifest["fingerprints"],
        "engine_cmd": run_manifest["engine_cmd"], "engine_config_sha256": engine_cfg_sha,
        "limits": run_manifest["limits"], "run_conditions": conditions,
        "run_elapsed_s": round(time.monotonic() - t_start, 1)}, ledger_check=ledger_check, bootstrap=cfg.bootstrap)


def rescore_run(run_dir: Path, *, tier_dir: Path | None = None, ledger: Path | None = None,
                ledger_anchor: str | None = None, repo_root: Path = REPO_ROOT, bootstrap: bool = True
                ) -> dict[str, Any]:
    """Rebuild the scorecard of a run directory from its case records (CLI ``score``).

    Records scored by other eval code are re-scored from their stored engine
    outputs; the provenance (fingerprints, engine command, configs, limits, run
    conditions) and the ledger check (including the engine config) are taken
    from ``run_manifest.json``, so a rescored card qualifies exactly like the
    original would have."""
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("schema") != "sherman_e4_run_manifest_v2":
        raise ValueError(f"{run_dir}: run manifest {run_manifest.get('schema')!r} predates this eval version; rerun")
    tdir = tier_dir or Path(run_manifest["tier_dir"])
    manifest = load_manifest(tdir)
    eval_roots = (run_manifest.get("code_roots") or {}).get("eval") or ["domains/production/engine/eval"]
    eval_fp = code_fingerprint(repo_root, tuple(eval_roots))
    skey = _score_key(eval_fp)
    scores: list[CaseScore] = []
    rescored = 0
    for cid in run_manifest["case_ids"]:
        path = run_dir / "cases" / f"{cid}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("score_key") == skey:
            scores.append(CaseScore.from_dict(doc["score"]))
            continue
        tc = load_truth_case(tdir / cid, verify_hashes=True)
        again = _rescore(doc, tc, Path(doc["compile"]["compiled_dir"]))
        if again is None:
            raise ValueError(f"{cid}: scored by other eval code and its engine output is gone; rerun the case")
        _write_record(path, dict(doc, score=again.to_dict(), score_key=skey, rescored_at=_now()))
        scores.append(again)
        rescored += 1
    ledger_check = None
    if ledger is not None:
        engine_identity = {"engine_fingerprint": run_manifest["fingerprints"]["engine"],
                           "engine_config_sha256": run_manifest["configs"]["engine"],
                           "engine_cmd": run_manifest["engine_cmd"]}
        ledger_check = _ledger_check(ledger, case_ids=run_manifest["case_ids"], manifest=manifest,
                                     engine_config_sha256=run_manifest["configs"]["engine"], repo_root=repo_root,
                                     engine_identity=engine_identity, anchor=ledger_anchor)
    provenance = {"rescored": True, "records_rescored": rescored, "run_started_at": run_manifest.get("started_at"),
                  "fingerprints": run_manifest.get("fingerprints"), "engine_cmd": run_manifest.get("engine_cmd"),
                  "engine_config_sha256": run_manifest["configs"]["engine"], "limits": run_manifest.get("limits"),
                  "run_conditions": run_manifest.get("run_conditions")}
    return write_scorecard(run_dir, scores, manifest=manifest, provenance=provenance, ledger_check=ledger_check,
                           bootstrap=bootstrap)


def write_scorecard(out: Path, scores: Sequence[CaseScore], *, manifest: Mapping[str, Any],
                    provenance: Mapping[str, Any] | None = None, ledger_check: Mapping[str, Any] | None = None,
                    bootstrap: bool = True) -> dict[str, Any]:
    rows = out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    with (rows / "cases.jsonl").open("w", encoding="utf-8") as fc, \
            (rows / "chars.jsonl").open("w", encoding="utf-8") as fch, \
            (rows / "bends.jsonl").open("w", encoding="utf-8") as fb:
        for s in scores:
            fc.write(json.dumps(asdict(s.case), sort_keys=True) + "\n")
            for r in s.chars:
                fch.write(json.dumps(asdict(r), sort_keys=True) + "\n")
            for r in s.bends:
                fb.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    card = build_scorecard(scores, tier=str(manifest.get("tier")), manifest=manifest, provenance=provenance,
                           ledger_check=ledger_check, bootstrap=bootstrap)
    (out / "scorecard.json").write_text(json.dumps(card, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "scorecard.md").write_text(scorecard_markdown(card), encoding="utf-8")
    return card


def scorecard_markdown(card: Mapping[str, Any]) -> str:
    """Human-readable table of the gates (the JSON is authoritative)."""
    lines = [f"# E4 G1 scorecard: tier {card['tier']} ({card.get('seedset')})", "",
             f"**{card['label']}**", "",
             f"Verdict: **{card['verdict']}**; regression verdict (A1's per-change signal): "
             f"**{card.get('regression_verdict')}**; qualifying: {card['qualifying']}"
             + (f" (not: {', '.join(card['not_qualifying_reasons'])})" if card["not_qualifying_reasons"] else ""),
             f"Thresholds {card['thresholds']['version']} sha256 {card['thresholds']['sha256'][:16]}",
             f"Binding gates failed: {', '.join(card.get('failed_gates') or []) or 'none'}; "
             f"not evaluable: {', '.join(card.get('not_evaluable_gates') or []) or 'none'} "
             f"(of which not applicable to this tier: {', '.join(card.get('not_applicable_to_tier_gates') or []) or 'none'})",
             "",
             "| Gate | Verdict | Value | Threshold | n | 95% CI (family bootstrap) |",
             "|---|---|---|---|---|---|"]
    for g in card["gates"]:
        ci = g.get("ci") or {}
        cis = "" if ci.get("lo") is None else f"[{ci['lo']:.4g}, {ci['hi']:.4g}]"
        tag = " (info)" if g["informational"] else (" (n/a in tier)" if g.get("not_applicable_to_tier") else "")
        lines.append(f"| {g['id']}{tag} | {g['verdict']} | {json.dumps(g['value'])} | {json.dumps(g['threshold'])} "
                     f"| {g['n']} | {cis} |")
    lines += ["", "Counts: " + json.dumps(card["counts"]["run_status"], sort_keys=True), ""]
    return "\n".join(lines) + "\n"


def _strip_volatile(doc: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k not in ("context", "created_at")}
    eng = dict(out.get("engine") or {})
    for k in ("duration_s", "peak_rss_bytes"):
        eng.pop(k, None)
    out["engine"] = eng
    return out


def blindness_check(cfg: RunnerConfig, case_ids: Sequence[str]) -> dict[str, Any]:
    """Run each case twice under different opaque staging names and compare
    results (``context``, ``created_at`` and timing removed)."""
    env = engine_env(cfg)
    compile_fp = code_fingerprint(cfg.repo_root, cfg.compile_code_roots)
    report = []
    for cid in case_ids:
        tc = load_truth_case(cfg.tier_dir / cid)
        co = compile_case(tc, cfg, compile_fp=compile_fp, env=env)
        if co.status != "COMPILED":
            report.append({"case_id": cid, "status": co.status, "identical": None})
            continue
        docs = []
        for tag, name in (("a", "scan.ply"), ("b", "q7f3c1d9.ply")):
            ws = cfg.out_dir / "blindness" / hashlib.sha256(f"{cid}|{tag}|{time.time_ns()}".encode()).hexdigest()[:16]
            run, eo, _ = inspect_case(tc, co, cfg, env=env, workspace=ws, scan_name=name)
            docs.append((run, eo))
        (ra, ea), (rb, eb) = docs
        if ea is None or eb is None or ea.result is None or eb.result is None:
            report.append({"case_id": cid, "status": f"{ra.status}/{rb.status}", "identical": None})
            continue
        a = _strip_volatile(ea.result.model_dump(mode="json"))
        b = _strip_volatile(eb.result.model_dump(mode="json"))
        same = canonical_json_bytes(a) == canonical_json_bytes(b)
        diff = [] if same else sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        report.append({"case_id": cid, "status": "OK", "identical": same, "differing_keys": diff})
    ok = all(r["identical"] for r in report) and bool(report)
    doc = {"schema": "sherman_e4_blindness_check_v1", "cases": report, "verdict": "PASS" if ok else "FAIL",
           "rule": "amendment 4: renamed copies with opaque names must give identical output"}
    (cfg.out_dir / "blindness_check.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    return doc
