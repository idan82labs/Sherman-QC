"""Runner overlay step: merge E3's result + the CompiledPart + the zs lines into ``zeroshot_overlay_v1.json``.

Usage (runner, cwd = engine bundle, after E3 validated and before the render step)::

    python -m domains.production.zeroshot.overlay_cli --result <ws>/engine/result.json
        --compiled <ws>/compiled/compiled_part.json --zs <ws>/zeroshot/zs_lines.json
        --out <ws>/zeroshot/zeroshot_overlay_v1.json

It reads no job identity (no context, no file names). Exit 0: the overlay was written. Exit 1: no overlay (the
zs lines are absent, not OK, or for another scan). Exit 3: internal error. Only exit 0 leaves a file; the runner
treats everything else as "no overlay", never as a job failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m domains.production.zeroshot.overlay_cli")
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--compiled", required=True, type=Path)
    ap.add_argument("--zs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        a.out.unlink()
    try:
        if not a.zs.is_file():
            print("overlay: no zs lines", flush=True)
            return 1
        zs = json.loads(a.zs.read_text(encoding="utf-8"))
        if zs.get("status") != "OK":
            print(f"overlay: zs status {zs.get('status')}", flush=True)
            return 1
        from contracts.production.compiled_part import CompiledPartV1
        from contracts.production.inspection_result import ShopfloorInspectionResultV1
        from domains.production.zeroshot.overlay import build_overlay, overlay_json_bytes

        raw = a.result.read_bytes()
        result = ShopfloorInspectionResultV1.model_validate_json(raw)
        compiled = CompiledPartV1.model_validate_json(a.compiled.read_bytes())
        if zs.get("scan_sha256") != result.scan.sha256:
            print("overlay: zs lines belong to another scan", flush=True)
            return 1
        ov = build_overlay(result, compiled, zs, engine_result_sha256=hashlib.sha256(raw).hexdigest())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_name(a.out.name + ".tmp")
        tmp.write_bytes(overlay_json_bytes(ov))
        os.replace(tmp, a.out)
    except Exception as exc:  # noqa: BLE001 - contained: no overlay
        traceback.print_exc()
        print(f"overlay: ERROR {type(exc).__name__}: {exc}"[:500], flush=True)
        return 3
    print(f"overlay: {ov.e3_summary_state} -> {ov.effective_summary_state} lines={len(ov.lines)} "
          f"flags={len(ov.flags)} withheld={ov.withheld_bends_located}/{ov.withheld_bends}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
