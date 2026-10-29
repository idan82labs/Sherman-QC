"""DEV_AUTO_CONFIRMED recipes for every authorized STEP, compiled with E1 (plan §10 amendment 4d).

Amendment 4d freezes "the DEV_AUTO_CONFIRMED recipes: every V10/native bend,
both flange sides REQUIRED, exclusions only through a pre-written per-feature
UNSUPPORTED rule. Compile failures stay in the denominators, and at least 80%
of authorized STEPs must compile."

Per unique authorized STEP (sha256-deduplicated, :mod:`.inventory`), one child
process (:func:`build_recipes_for_step`, CLI ``_build-recipes``) runs E1's own
policy code and writes three recipes:

- ``dev`` -- the frozen evaluation recipe: ``dev_recipe.dev_candidates`` (policy
  ``E1-DEV-POLICY-1``: union of the V10 and native extractor bends, geometric
  duplicates dropped, pieces of one flange-graph bend merged, candidates on
  rule-declared features excluded by E1's frozen UNSUPPORTED rule), then
  ``dev_recipe.dev_recipe`` (every candidate an active confirmed mark, both
  flange sides REQUIRED, shop-standard tolerances, ``inspection_stage`` FINAL,
  ``confirmation`` DEV_AUTO_CONFIRMED);
- ``v10_strict`` and ``native_strict`` -- every bend of one extractor exactly
  as reported, no merge and no exclusion: the strict single-source figures E1
  asked to report next to the policy figure (ENGINE_COMPILE.md §10: the strict
  V10 rate is "the amendment-4d figure"; the policy rate is "with E1-graph
  dedupe"). The 4d floor gates the frozen ``dev`` recipes, the ones the
  evaluation runs on, on both denominators; the strict figures are reported as
  diagnostics (:data:`.physical.COMPILE_GATE_RULE`, decided before A1).

**Blindness** (amendment 4): the STEP is staged under an opaque name
(``cad_<sha12>.step``) and the SKU is opaque (``DEV-<10 hex of the CAD sha>``),
so no file name, part number or stage token reaches E1 or, through the recipe,
E3. The recipe carries no thickness (E1 measures it).

Every recipe is compiled with the E1 command line (``python -m
domains.production.engine.compile``) in its own monitored subprocess (300 s,
4 GiB, one at a time) and the outcome recorded: COMPILED, BLOCKED (blocker
codes), FAILED (exit 3 / no output / limits) or NO_RECIPE (no candidate bend
or a recipe the builder refused). Every outcome other than COMPILED stays in
every denominator (:func:`compile_rates`). Recipes are written as canonical JSON
bytes; compiled outputs stay in the work directory (they are reproducible: the
compiled-part hash excludes ``compiled_at`` and diagnostics).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from contracts.production.canonical import canonical_json_bytes

from ..procmon import run_monitored
from ..runner import engine_env
from .common import (
    REPO_ROOT,
    SOURCE_ASSETS,
    custody_check,
    dev_sku_for,
    now_utc,
    read_json,
    rel_to_repo,
    sha256_file,
    write_json,
)

__all__ = [
    "COMPILE_REPORT_SCHEMA",
    "VARIANTS",
    "FROZEN_VARIANT",
    "build_recipes_for_step",
    "compile_recipe",
    "summarize_compiled",
    "compile_rates",
    "build_and_compile_all",
]

COMPILE_REPORT_SCHEMA = "sherman_e4b_compile_report_v2"
VARIANTS = ("dev", "v10_strict", "native_strict")
FROZEN_VARIANT = "dev"
COMPILE_TIMEOUT_S = 300.0
CHILD_TIMEOUT_S = 600.0
RSS_LIMIT_BYTES = 4 * 1024**3


def _stage_readonly(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_recipe(path: Path, recipe: Any) -> str:
    data = canonical_json_bytes(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_file(path)


def _recipe_facts(recipe: Any) -> dict[str, Any]:
    sides = [(b.flange_inspection.side_a, b.flange_inspection.side_b) for b in recipe.bends]
    return {"bends": len(recipe.bends), "recipe_sha256": recipe.recipe_sha256, "content_sha256": recipe.content_sha256,
            "sku": recipe.sku, "confirmation": getattr(recipe, "confirmation", None),
            "inspection_stage": recipe.inspection_stage,
            "both_sides_required": all(a == "REQUIRED" and b == "REQUIRED" for a, b in sides),
            "cad_original_filename": recipe.cad.original_filename}


def build_recipes_for_step(step_path: str | os.PathLike, work_dir: str | os.PathLike) -> dict[str, Any]:
    """Child-process body: stage the STEP opaquely and write the three recipe variants.

    Returns a JSON-able summary (also written to ``<work>/recipes/<sku>/build.json``)."""
    from domains.production.engine.compile import dev_recipe as dr

    src = custody_check(step_path, "DEV recipe STEP")
    sha = sha256_file(src)
    sku = dev_sku_for(sha)
    work = Path(work_dir)
    staged = work / "inputs" / f"cad_{sha[:12]}.step"
    _stage_readonly(src, staged)
    out_dir = work / "recipes" / sku
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"sku": sku, "cad_sha256": sha, "staged_cad": str(staged), "variants": {}}
    sources: dict[str, Any] = {}
    for name in ("v10", "native"):
        try:
            sources[name] = dr.native_candidates(staged, name)
        except Exception as exc:  # an extractor refusing a file is evidence, not a crash
            sources[name] = []
            summary[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        dev_cands, report = dr.dev_candidates(staged)
    except Exception as exc:  # pragma: no cover - dev_candidates already guards its extractors
        dev_cands, report = [], {"error": f"{type(exc).__name__}: {exc}"[:300]}
    summary["dev_policy_report"] = report
    plans = {"dev": dev_cands, "v10_strict": sources["v10"], "native_strict": sources["native"]}
    for variant, cands in plans.items():
        entry: dict[str, Any] = {"candidates": len(cands)}
        if not cands:
            entry.update(status="NO_RECIPE", reason="NO_CANDIDATE_BENDS")
        else:
            try:
                recipe = dr.dev_recipe(staged, cands, sku=sku, inspection_stage="FINAL")
            except Exception as exc:  # the builder refused the draft: recorded, stays in the denominators
                entry.update(status="NO_RECIPE", reason=f"RECIPE_BUILD_REFUSED: {type(exc).__name__}: {exc}"[:400])
            else:
                path = out_dir / f"{variant}.recipe.json"
                entry.update(status="RECIPE_BUILT", recipe_path=str(path), recipe_file_sha256=_write_recipe(path, recipe),
                             **_recipe_facts(recipe))
        summary["variants"][variant] = entry
    write_json(out_dir / "build.json", summary)
    return summary


def summarize_compiled(compiled_json: Path, recipe_json: Path | None = None) -> dict[str, Any]:
    """Status, blockers and nominal facts of one ``compiled_part.json``."""
    from contracts.production.compiled_part import check_compiled_against_recipe

    from ..engine_output import load_compiled, load_recipe

    cp = load_compiled(compiled_json)
    unavailable = [c for c in cp.characteristics if c.nominal is None]
    out = {
        "status": cp.status, "compiled_part_sha256": cp.compiled_part_sha256, "compiler_version": cp.compiler.version,
        "blocker_codes": list(cp.blocker_codes),
        "warning_codes": sorted({i.code for i in cp.issues if i.severity == "WARNING"}),
        "bends": len(cp.bends), "panels": len(cp.panels), "stages": len(cp.stages),
        "characteristics": len(cp.characteristics),
        "unavailable_nominals": len(unavailable),
        "unavailable_by_reason": {r: sum(1 for c in unavailable if c.unavailable_reason == r)
                                  for r in sorted({c.unavailable_reason or "NONE" for c in unavailable})},
        "unsupported_features": [f.feature_kind for f in cp.unsupported_features],
        "thickness_mm": None if cp.thickness is None else cp.thickness.t_mm,
        "k_factor": None if cp.k_factor is None else {"value": cp.k_factor.value, "source": cp.k_factor.source},
        "ambiguous_stages": [s.stage_model_id for s in cp.stages if s.symmetry.ambiguous_bend_groups],
    }
    if recipe_json is not None:
        out["recipe_consistency_problems"] = list(check_compiled_against_recipe(cp, load_recipe(recipe_json)))
    return out


def compile_recipe(recipe_json: Path, cad_path: Path, out_dir: Path, *, env: Mapping[str, str],
                   repo_root: Path = REPO_ROOT, timeout_s: float = COMPILE_TIMEOUT_S,
                   rss_limit_bytes: int = RSS_LIMIT_BYTES) -> dict[str, Any]:
    """E1 compile CLI in a monitored subprocess; the outcome record (module docstring)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "domains.production.engine.compile", "--recipe", str(recipe_json),
            "--cad", str(cad_path), "--out", str(out_dir)]
    proc = run_monitored(argv, cwd=repo_root, env=env, timeout_s=timeout_s, rss_limit_bytes=rss_limit_bytes,
                         stderr_path=out_dir.parent / f"{out_dir.name}.stderr.log")
    rec: dict[str, Any] = {"exit_code": proc.returncode, "signal": proc.signal, "wall_s": round(proc.wall_s, 3),
                           "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out,
                           "rss_exceeded": proc.rss_exceeded, "compiled_dir": str(out_dir)}
    cp = out_dir / "compiled_part.json"
    if proc.timed_out or proc.rss_exceeded or proc.signal is not None:
        rec.update(status="FAILED", reason="TIMEOUT" if proc.timed_out else ("RSS_LIMIT" if proc.rss_exceeded
                                                                            else f"SIGNAL_{proc.signal}"))
        return rec
    if cp.is_file() and proc.returncode in (0, 2):
        try:
            facts = summarize_compiled(cp, recipe_json)
        except (OSError, ValueError) as exc:
            rec.update(status="FAILED", reason=f"COMPILED_PART_INVALID: {str(exc).splitlines()[0][:200]}")
            return rec
        status = "COMPILED" if facts["status"] == "COMPILED" and proc.returncode == 0 else "BLOCKED"
        size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
        rec.update({k: v for k, v in facts.items() if k != "status"})
        rec.update(status=status, compiled_status=facts["status"], reason=",".join(facts["blocker_codes"]) or None,
                   output_bytes=size)
        return rec
    err = proc.last("ERROR") or {}
    err_file = out_dir / "compile_error.json"
    if err_file.is_file():
        try:
            err = read_json(err_file)
        except (OSError, ValueError):
            pass
    rec.update(status="FAILED", reason=f"EXIT_{proc.returncode}:{err.get('code', '')}".rstrip(":"),
               detail=str(err.get("message", ""))[:300])
    return rec


def compile_rates(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    """COMPILED share of every authorized unique STEP for one recipe variant (failures in the denominator)."""
    total = len(rows)
    statuses = [((r.get("variants") or {}).get(variant) or {}).get("status", "NO_RECIPE") for r in rows]
    compiled = sum(1 for s in statuses if s == "COMPILED")
    return {"variant": variant, "authorized_unique_steps": total, "compiled": compiled,
            "blocked": sum(1 for s in statuses if s == "BLOCKED"), "failed": sum(1 for s in statuses if s == "FAILED"),
            "no_recipe": sum(1 for s in statuses if s == "NO_RECIPE"),
            "rate": (compiled / total) if total else None}


def build_and_compile_all(inventory: Mapping[str, Any], work_dir: str | os.PathLike, *,
                          source_root: str | os.PathLike = SOURCE_ASSETS, repo_root: Path = REPO_ROOT,
                          only_sha: Sequence[str] | None = None,
                          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Every unique authorized STEP, serially: child recipe build, then the E1 compiles."""
    from ..runner import RunnerConfig

    work = Path(work_dir)
    env = engine_env(RunnerConfig(tier_dir=work, out_dir=work, repo_root=repo_root))
    from .inventory import all_entries, entry_path

    steps = [e for e in all_entries(inventory, "STEP") if not only_sha or e["sha256"] in only_sha]
    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(steps, start=1):
        src = entry_path(entry, Path(source_root))
        sku = dev_sku_for(entry["sha256"])
        if progress:
            progress(f"[{i}/{len(steps)}] {sku} recipes")
        argv = [sys.executable, "-m", "domains.production.engine.eval.freeze.prefreeze", "_build-recipes",
                "--cad", str(src), "--work", str(work)]
        log_dir = work / "recipes" / sku
        log_dir.mkdir(parents=True, exist_ok=True)
        proc = run_monitored(argv, cwd=repo_root, env=env, timeout_s=CHILD_TIMEOUT_S, rss_limit_bytes=RSS_LIMIT_BYTES,
                             stderr_path=log_dir / "build.stderr.log")
        origin = entry.get("origin", "SOURCE_ASSETS")
        row: dict[str, Any] = {"sku": sku, "cad_sha256": entry["sha256"],
                               "cad_rel": ("../source-assets/02_CAD_QC_3D/" + entry["rel_path"]
                                           if origin == "SOURCE_ASSETS" else entry["rel_path"]),
                               "cad_origin": origin, "part_id": entry.get("part_id", entry.get("family")),
                               "cad_aliases": entry.get("aliases", []), "family": entry.get("family"),
                               "build": {"exit_code": proc.returncode, "wall_s": round(proc.wall_s, 3),
                                         "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out}}
        build_json = log_dir / "build.json"
        if proc.returncode != 0 or not build_json.is_file():
            row["variants"] = {v: {"status": "NO_RECIPE", "reason": f"RECIPE_BUILD_FAILED exit={proc.returncode}"}
                               for v in VARIANTS}
            rows.append(row)
            continue
        built = read_json(build_json)
        row["dev_policy_report"] = built.get("dev_policy_report")
        row["extractor_errors"] = {k: v for k, v in built.items() if k.endswith("_error")}
        staged = Path(built["staged_cad"])
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            v = dict(built["variants"].get(variant) or {"status": "NO_RECIPE"})
            if v.get("status") == "RECIPE_BUILT":
                if progress:
                    progress(f"[{i}/{len(steps)}] {sku} compile {variant}")
                outcome = compile_recipe(Path(v["recipe_path"]), staged, work / "compiled" / variant / sku, env=env,
                                         repo_root=repo_root)
                v.update(outcome)
            variants[variant] = v
        row["variants"] = variants
        rows.append(row)
    rates = {v: compile_rates(rows, v) for v in VARIANTS}
    src_rows = [r for r in rows if r.get("cad_origin", "SOURCE_ASSETS") == "SOURCE_ASSETS"]
    rates_src = {v: compile_rates(src_rows, v) for v in VARIANTS}
    from ..thresholds import G2_PHYSICAL_THRESHOLDS
    from .physical import compile_gate_fields

    floor = float(G2_PHYSICAL_THRESHOLDS["compile_rate_min"])
    return {
        "schema": COMPILE_REPORT_SCHEMA,
        "created_at": now_utc(),
        "rule": "plan §10 amendment 4d: every V10/native bend, both sides REQUIRED, exclusions only via E1's "
                "frozen per-feature UNSUPPORTED rule; failures stay in every denominator; >= 80 % must compile",
        "frozen_variant": FROZEN_VARIANT,
        "compile_rate_min": floor,
        "denominators": {"rates": "every authorized unique STEP: the source tree (plan §9) plus the STEP CAD of "
                                  "every CONSUMED part of BENCHMARK_V2 (owner directions 2026-10-17)",
                         "rates_source_assets": "the source-tree STEPs only (the plan §9 denominator of the "
                                                "first E4b pass)"},
        "rates": rates,
        "rates_source_assets": rates_src,
        **compile_gate_fields(rates, rates_src),  # the gate on the frozen DEV recipes; strict figures diagnostic
        "work_dir": str(work),
        "rows": rows,
    }


def main_build(argv: Sequence[str]) -> int:
    """``_build-recipes`` child entry point: prints one RESULT/ERROR record."""
    import argparse

    ap = argparse.ArgumentParser(prog="prefreeze _build-recipes")
    ap.add_argument("--cad", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    a = ap.parse_args(list(argv))
    try:
        summary = build_recipes_for_step(a.cad, a.work)
    except Exception as exc:
        traceback.print_exc()
        sys.stdout.write("ERROR\t" + json.dumps({"code": "RECIPE_BUILD_FAILED", "message": str(exc)[:300]}) + "\n")
        return 3
    sys.stdout.write("RESULT\t" + json.dumps({"sku": summary["sku"], "variants": {
        k: v.get("status") for k, v in summary["variants"].items()}}) + "\n")
    return 0


def recipe_paths_for_freeze(report: Mapping[str, Any]) -> list[tuple[str, Path]]:
    """``(sku, recipe path)`` of every frozen-variant recipe that was built."""
    out = []
    for row in report["rows"]:
        v = (row.get("variants") or {}).get(FROZEN_VARIANT) or {}
        if v.get("recipe_path"):
            out.append((row["sku"], Path(v["recipe_path"])))
    return out


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A compile-report row without work-directory paths (for the docs run log)."""
    def clean(v: Mapping[str, Any]) -> dict[str, Any]:
        return {k: x for k, x in v.items() if k not in ("recipe_path", "compiled_dir")}

    out = {k: v for k, v in row.items() if k != "variants"}
    rel = str(row.get("cad_rel", ""))
    out["cad_rel"] = rel if rel.startswith(("..", "~", "/", "data/")) else rel_to_repo(SOURCE_ASSETS / rel)
    out["variants"] = {k: clean(v) for k, v in (row.get("variants") or {}).items()}
    return out
