"""Subprocess entry of :mod:`apps.production_line.recipes.chirality` (adapter over the research map).

usage: python -m apps.production_line.recipes.chirality_worker COMPILED_DIR --out DIR [--stages S12]

Writes ``chirality_map_v0.json`` / ``.npz`` (``research.chirality.chirality_map.write``) into DIR, then
``chirality_summary_v0.json`` = {summary: chirality_map.summary(doc), stages_computed, computed_at,
map_seconds} last and atomically, so its presence means the map is complete. Exit 3 when the research
module cannot be imported; any other failure exits non-zero with a message on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("compiled_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages")
    a = ap.parse_args(argv)
    try:
        from research.chirality import chirality_map as cm
    except ImportError as exc:
        print(f"chirality module unavailable: {exc}", file=sys.stderr)
        return 3
    from apps.production_line.recipes.chirality import SUMMARY_FILE

    out = Path(a.out)
    stages = [s for s in (a.stages or "").split(",") if s] or None
    path = cm.write(Path(a.compiled_dir), out, stages)
    doc, _ = cm.load(path)
    body = {
        "summary": cm.summary(doc),
        "stages_computed": [s["stage_model_id"] for s in doc.get("stages", [])],
        "computed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "map_seconds": doc.get("seconds"),
    }
    tmp = out / f".{SUMMARY_FILE}.tmp-{os.getpid()}"
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out / SUMMARY_FILE)
    print(json.dumps({"part_class": body["summary"]["part_class"], "stages": body["stages_computed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
