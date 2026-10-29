"""CLI: ``python -m domains.production.engine.compile`` (ENGINE_IO §3).

::

    python -m domains.production.engine.compile --recipe RECIPE.json --cad PART.step
        [--flat FLAT.dxf|FLAT.step] --out DIR [--shop-k 0.33] [--config CONFIG.json]

Stdout carries only the tab-separated protocol records (``PROGRESS``, ``RESULT``,
``ERROR``); every other byte written to stdout by native code (OCC) is
redirected to stderr. Exit codes: 0 COMPILED; 2 BLOCKED (``compiled_part.json``
with blockers) or invalid input (``compile_error.json {code, message}``);
3 internal error (nothing guaranteed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

__all__ = ["main"]


class _Protocol:
    """Writes protocol records to the original stdout; native stdout goes to stderr."""

    def __init__(self) -> None:
        sys.stdout.flush()
        self._fd = os.dup(1)
        self._saved_stdout = sys.stdout
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self._out = os.fdopen(os.dup(self._fd), "w", encoding="utf-8", buffering=1)

    def emit(self, kind: str, payload: dict) -> None:
        self._out.write(f"{kind}\t{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n")
        self._out.flush()

    def close(self) -> None:
        try:
            self._out.flush()
            self._out.close()
        finally:
            sys.stderr.flush()
            os.dup2(self._fd, 1)
            os.close(self._fd)
            sys.stdout = self._saved_stdout


def _write_error(out: Path, code: str, message: str) -> None:
    try:
        if out.exists() and any(p.name != "compile.log" for p in out.iterdir()):
            return
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "compile_error.json.tmp"
        tmp.write_text(json.dumps({"code": code, "message": message[:2000]}, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, out / "compile_error.json")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m domains.production.engine.compile",
                                     description="Sherman QC onboarding compile (E1)")
    parser.add_argument("--recipe", required=True, type=Path, help="canonical PartRecipeV1 JSON")
    parser.add_argument("--cad", required=True, type=Path, help="folded STEP (hash must match the recipe)")
    parser.add_argument("--flat", type=Path, default=None, help="flat DXF or flat STEP (when the recipe has one)")
    parser.add_argument("--out", required=True, type=Path, help="output directory (absent or empty)")
    parser.add_argument("--shop-k", type=float, default=None, help="K-factor when no flat is bound (default 0.33)")
    parser.add_argument("--config", type=Path, default=None, help="pinned compile config JSON")
    args = parser.parse_args(argv)

    proto = _Protocol()
    try:
        return _run(args, proto)
    finally:
        proto.close()


def _run(args, proto: _Protocol) -> int:
    from pydantic import ValidationError

    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile.compiler import CompileInputError, CompileTimeout, compile_part
    from domains.production.engine.compile.config import DEFAULT_SHOP_K, load_config

    out: Path = args.out
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        proto.emit("ERROR", {"code": "OUT_DIR_NOT_EMPTY", "message": f"{out} exists and is not an empty directory"})
        return 2
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        _write_error(out, "CONFIG_INVALID", str(exc))
        proto.emit("ERROR", {"code": "CONFIG_INVALID", "message": str(exc)[:500]})
        return 2
    try:
        raw = json.loads(args.recipe.read_text(encoding="utf-8"))
        recipe = PartRecipeV1.model_validate(raw)
    except (OSError, ValueError, ValidationError) as exc:
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        _write_error(out, "RECIPE_INVALID", message)
        proto.emit("ERROR", {"code": "RECIPE_INVALID", "message": message[:500]})
        return 2
    shop_k = DEFAULT_SHOP_K if args.shop_k is None else float(args.shop_k)

    def progress(stage: str, pct: int) -> None:
        proto.emit("PROGRESS", {"stage": stage, "progress": int(max(0, min(100, pct)))})

    try:
        result = compile_part(recipe, args.cad, out, flat_path=args.flat, shop_k=shop_k, config=cfg, progress=progress)
    except CompileInputError as exc:
        _write_error(out, exc.code, exc.message)
        proto.emit("ERROR", {"code": exc.code, "message": exc.message[:500]})
        return 2
    except CompileTimeout as exc:
        proto.emit("ERROR", {"code": "COMPILE_TIMEOUT", "message": str(exc)[:500]})
        return 3
    except Exception as exc:  # internal error: nothing is guaranteed
        proto.emit("ERROR", {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:500]})
        return 3
    compiled = result.compiled
    proto.emit("RESULT", {"status": compiled.status, "compiled_part_sha256": compiled.compiled_part_sha256,
                          "blocker_codes": list(compiled.blocker_codes), "out": str(out)})
    return 0 if compiled.status == "COMPILED" else 2


if __name__ == "__main__":
    sys.exit(main())
