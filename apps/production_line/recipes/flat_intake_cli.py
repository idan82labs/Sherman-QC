"""Flat intake subprocess (C05): parse a flat and propose its bend binding with E1.

The line API never runs OCP or ezdxf in its own process. It launches this
module under the C02 supervisor (wall-clock timeout, RSS ceiling, process
group kill) and reads the JSON it writes::

    python -m apps.production_line.recipes.flat_intake_cli parse \\
        --flat FLAT.dxf|FLAT.step [--format DXF|STEP] --out OUT.json
    python -m apps.production_line.recipes.flat_intake_cli propose \\
        --cad PART.step --flat FLAT.dxf|FLAT.step [--format DXF|STEP] \\
        --marks MARKS.json [--shop-k 0.33] --out OUT.json

``MARKS.json`` is ``[{"bend_uid", "line_start": [x,y,z], "line_end": [x,y,z]}, ...]``
in bending (sequence) order: the operator's draft marks with their recipe
``bend_uid`` identities (``domains.production.recipe.bend_identity``).

Output (``OUT.json``, written tmp + rename):

- parse: ``{"status": "PARSED", "summary": ParsedFlat.summary(), "geometry":
  {"units": "mm", "outer": [[x, y], ...], "inner": [[[x, y], ...], ...],
  "bend_lines": [...], "decimated": bool}}`` or ``{"status": "PARSE_FAILED",
  "reason": FlatParseError.reason, "detail": str}``. Geometry is in the flat's
  own 2D frame, in mm, decimated to at most :data:`MAX_LOOP_POINTS` points per
  loop for display (the unfolded view of the onboarding wizard).
- propose: E1 ``propose_flat_binding`` output (``status`` PROPOSED with
  ``flat`` and ``proposal``, or RECONCILIATION_BLOCKED with the reconciliation
  summary), or ``{"status": "PARSE_FAILED", ...}``, or ``{"status":
  "CAD_READ_FAILED", "reason", "detail"}``.

Exit codes: 0 output written (any status above); 2 invalid arguments or
unreadable marks file; 3 internal error (traceback on stderr, nothing
guaranteed). Nothing here is ever an engine measurement; the operator confirms
or edits the proposal before release.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

__all__ = ["MAX_LOOP_POINTS", "main", "loop_points"]

#: Display cap per outline loop (the wizard's unfolded view needs shape, not every vertex).
MAX_LOOP_POINTS = 2000


def _json_safe(obj: Any) -> Any:
    """JSON-ready copy: non-finite floats become None, tuples become lists, numpy scalars plain."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    item = getattr(obj, "item", None)
    if callable(item):
        return _json_safe(item())
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(obj)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(doc), sort_keys=True, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _finite(v: float) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite coordinate")
    out = round(f, 6)
    return 0.0 if out == 0.0 else out


def loop_points(loop: Any, cap: int = MAX_LOOP_POINTS) -> tuple[list[list[float]], bool]:
    """``(points, decimated)``: a loop as ``[[x, y], ...]``, keeping every k-th vertex above ``cap``."""
    pts = [[_finite(p[0]), _finite(p[1])] for p in loop]
    if len(pts) <= cap:
        return pts, False
    step = math.ceil(len(pts) / cap)
    return pts[::step], True


def _parse(args: argparse.Namespace) -> dict[str, Any]:
    from domains.production.engine.compile.flat import FlatParseError, parse_flat

    try:
        parsed = parse_flat(args.flat, args.format)
    except FlatParseError as exc:
        return {"status": "PARSE_FAILED", "reason": exc.reason, "detail": str(exc.detail)[:500]}
    outer, dec_outer = loop_points(parsed.outer)
    inner: list[list[list[float]]] = []
    decimated = dec_outer
    for lp in parsed.inner:
        pts, dec = loop_points(lp)
        inner.append(pts)
        decimated = decimated or dec
    summary = parsed.summary()
    return {
        "status": "PARSED",
        "summary": summary,
        "geometry": {
            "units": "mm",
            "frame": "flat_2d",
            "outer": outer,
            "inner": inner,
            "bend_lines": summary.get("bend_lines", []),
            "decimated": decimated,
        },
    }


def _load_marks(path: Path) -> list[Any]:
    from domains.production.engine.compile.reconcile import Mark

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("marks must be a non-empty list")
    marks = []
    for item in raw:
        start = tuple(float(v) for v in item["line_start"])
        end = tuple(float(v) for v in item["line_end"])
        if len(start) != 3 or len(end) != 3:
            raise ValueError("mark lines are 3D")
        marks.append(Mark(bend_uid=str(item["bend_uid"]), line_start=start, line_end=end))  # type: ignore[arg-type]
    return marks


def _propose(args: argparse.Namespace, marks: list[Any]) -> dict[str, Any]:
    from domains.production.engine.compile.api import propose_flat_binding
    from domains.production.engine.compile.brep import CadReadError
    from domains.production.engine.compile.config import DEFAULT_SHOP_K
    from domains.production.engine.compile.flat import FlatParseError

    shop_k = DEFAULT_SHOP_K if args.shop_k is None else float(args.shop_k)
    try:
        return dict(propose_flat_binding(args.cad, marks, args.flat, args.format, shop_k=shop_k))
    except FlatParseError as exc:
        return {"status": "PARSE_FAILED", "reason": exc.reason, "detail": str(exc.detail)[:500]}
    except CadReadError as exc:
        return {"status": "CAD_READ_FAILED", "reason": getattr(exc, "reason", "CAD_READ_FAILED"),
                "detail": str(getattr(exc, "detail", exc))[:500]}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.recipes.flat_intake_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse", help="parse a flat DXF / flat STEP")
    p.add_argument("--flat", type=Path, required=True)
    p.add_argument("--format", choices=("DXF", "STEP"), default=None)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("propose", help="propose the flat bend-line binding for draft marks")
    q.add_argument("--cad", type=Path, required=True)
    q.add_argument("--flat", type=Path, required=True)
    q.add_argument("--format", choices=("DXF", "STEP"), default=None)
    q.add_argument("--marks", type=Path, required=True)
    q.add_argument("--shop-k", type=float, default=None)
    q.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    if not args.flat.is_file():
        print(f"flat not found: {args.flat}", file=sys.stderr)
        return 2
    try:
        if args.command == "parse":
            doc = _parse(args)
        else:
            if not args.cad.is_file():
                print(f"CAD not found: {args.cad}", file=sys.stderr)
                return 2
            if args.shop_k is not None and not (0.0 < args.shop_k < 1.0):
                print("--shop-k must be in (0, 1)", file=sys.stderr)
                return 2
            try:
                marks = _load_marks(args.marks)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"marks file invalid: {exc}", file=sys.stderr)
                return 2
            doc = _propose(args, marks)
        _write_json(args.out, doc)
    except Exception:
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
