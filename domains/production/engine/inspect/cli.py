"""Command line of the per-scan engine (ENGINE_IO §4).

::

    python -m domains.production.engine.inspect \\
        --compiled DIR/compiled_part.json | DIR   # its directory holds stages/
        --recipe RECIPE.json                      # canonical PartRecipeV1
        --scan SCAN.ply|SCAN.stl                  # staged under an opaque name
        --out OUT                                 # created; must be empty or absent
        --scan-sha256 HEX                         # sha256 pinned for the scan bytes
        [--compiled-sha256 HEX] [--recipe-sha256 HEX]
        [--declared-stage K --declared-stage-source SIDECAR|OPERATOR|API]
        [--unit-factor 1|25.4|1000]               # default 1; never an auto-scale
        [--time-budget-s 170] [--engine-bundle-id ID] [--config CONFIG.json]

There is no ``--context`` option (plan §10 amendment 4, ENGINE_IO §4): the
engine never receives the scan's file name, serial or any other job identity,
and ``result.context`` is always written empty; C06 or the runner fills it
from the job workspace after the engine step.

Stdout carries only ``PROGRESS\\t{"stage", "progress"}``, ``RESULT\\t{...}`` and
``ERROR\\t{"code", "message"}`` records; logs go to stderr and ``engine.log``.

Exit codes: 0 ``result.json`` written and valid (any inspection outcome,
INVALID_INSPECTION included); 2 invalid input (``engine_error.json`` with one
of the ENGINE_IO §4 codes, or only an ERROR record for usage problems such as
OUT_DIR_NOT_EMPTY, CONFIG_INVALID, DECLARED_STAGE_INVALID);
3 internal error (including a result that fails its own consistency check,
and any unexpected failure outside the engine: an unreadable input file, an
output directory that cannot be created or written).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

__all__ = ["main", "build_parser"]


def _emit(tag: str, payload: dict) -> None:
    sys.stdout.write(f"{tag}\t{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n")
    sys.stdout.flush()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors are protocol errors, exit 2
        _emit("ERROR", {"code": "USAGE", "message": message})
        sys.stderr.write(self.format_usage())
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m domains.production.engine.inspect", description="Sherman per-scan inspection (E3)")
    p.add_argument("--compiled", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scan-sha256", required=True)
    p.add_argument("--compiled-sha256")
    p.add_argument("--recipe-sha256")
    p.add_argument("--declared-stage", type=int)
    p.add_argument("--declared-stage-source", choices=("SIDECAR", "OPERATOR", "API"))
    p.add_argument("--unit-factor", type=float, default=1.0, choices=(1.0, 25.4, 1000.0))
    p.add_argument("--time-budget-s", type=float)
    p.add_argument("--engine-bundle-id")
    p.add_argument("--config")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; every failure maps onto the ENGINE_IO §4 exit codes (0, 2 or 3), never a bare traceback."""
    try:
        return _main(argv)
    except SystemExit:
        raise  # usage errors (exit 2) and --help (exit 0) from argparse
    except Exception as exc:  # noqa: BLE001 - anything outside the engine is still an internal error (exit 3)
        sys.stderr.write(traceback.format_exc())
        _emit("ERROR", {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:1000]})
        return 3


def _main(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    if (args.declared_stage is None) != (args.declared_stage_source is None):
        _emit("ERROR", {"code": "DECLARED_STAGE_INVALID",
                        "message": "--declared-stage and --declared-stage-source go together"})
        return 2
    if args.declared_stage is not None and args.declared_stage < 0:
        _emit("ERROR", {"code": "DECLARED_STAGE_INVALID", "message": "--declared-stage must be >= 0"})
        return 2
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _emit("ERROR", {"code": "OUT_DIR_NOT_EMPTY", "message": f"output directory is not empty: {out}"})
        return 2
    # heavy imports after the cheap argument checks
    from contracts.production.inspection_result import ResultContextV1
    from domains.production.engine.inspect.config import ConfigError, load_config
    from domains.production.engine.inspect.engine import InspectOptions, inspect_scan
    from domains.production.engine.inspect.outputs import write_json
    from domains.production.engine.inspect.part import InspectInputError, load_inputs

    try:
        cfg = load_config(args.config, time_budget_s=args.time_budget_s)
    except ConfigError as exc:
        _emit("ERROR", {"code": "CONFIG_INVALID", "message": str(exc)})
        return 2
    context = ResultContextV1()  # always empty: E3 never receives job identity
    out.mkdir(parents=True, exist_ok=True)
    try:
        inputs = load_inputs(args.compiled, args.recipe, args.scan, args.scan_sha256,
                             compiled_sha256=args.compiled_sha256, recipe_sha256=args.recipe_sha256)
    except InspectInputError as exc:
        write_json(out / "engine_error.json", {"code": exc.code, "message": exc.message})
        _emit("ERROR", {"code": exc.code, "message": exc.message})
        return 2
    if args.declared_stage is not None and args.declared_stage > len(inputs.compiled.bends):
        _emit("ERROR", {"code": "DECLARED_STAGE_INVALID",
                        "message": f"--declared-stage {args.declared_stage} exceeds the {len(inputs.compiled.bends)} bends"})
        return 2
    opts = InspectOptions(context=context, declared_stage=args.declared_stage,
                          declared_source=args.declared_stage_source, unit_factor=float(args.unit_factor),
                          engine_bundle_id=args.engine_bundle_id, time_budget_s=cfg.time_budget_s)
    try:
        outcome = inspect_scan(inputs, out, cfg, opts, progress=lambda st, pc: _emit("PROGRESS", {"stage": st, "progress": pc}))
    except Exception as exc:  # noqa: BLE001 - every unexpected failure is an internal error (exit 3)
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        try:
            (out / "engine.log").write_text(tb, encoding="utf-8")
        except OSError:
            pass
        _emit("ERROR", {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:1000]})
        return 3
    res = outcome.result
    _emit("PROGRESS", {"stage": "written", "progress": 100})
    _emit("RESULT", {"summary_state": res.summary.state, "inspection_validity": res.inspection_validity.state,
                     "stage_status": res.stage_decision.status, "result": "result.json"})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
