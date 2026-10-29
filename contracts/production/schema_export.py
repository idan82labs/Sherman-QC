"""Export JSON Schemas of the production document contracts.

Usage::

    python -m contracts.production.schema_export --out DIR

Writes one ``<name>.schema.json`` per document model in :data:`SCHEMA_MODELS`
(sorted keys, UTF-8). The only file IO in contracts/production. It refuses an
output directory that resolves inside the repo's data/, output/, outputs/,
artifacts/, uploads/ or live_scans/ (custody rule).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from contracts.production.compiled_part import CompiledPartV1
from contracts.production.events import LineEventV1
from contracts.production.inspection_result import BendGeometryFileV1, ShopfloorInspectionResultV1, StationFeedV1
from contracts.production.layout import FORBIDDEN_DATA_ROOT_PARENTS
from contracts.production.line_job import LineJobV1, ScanFileV1
from contracts.production.part_recipe import PartRecipeV1, ReleaseRequestV1

__all__ = ["SCHEMA_MODELS", "REPO_ROOT", "export_schemas", "main"]

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "part_recipe_v1": PartRecipeV1,
    "release_request_v1": ReleaseRequestV1,
    "scan_file_v1": ScanFileV1,
    "line_job_v1": LineJobV1,
    "line_event_v1": LineEventV1,
    "shopfloor_inspection_result_v1": ShopfloorInspectionResultV1,
    "station_feed_v1": StationFeedV1,
    "compiled_part_v1": CompiledPartV1,
    "bend_geometry_v1": BendGeometryFileV1,
}


def _refuse_forbidden(out: Path) -> None:
    resolved = out.resolve()
    for name in FORBIDDEN_DATA_ROOT_PARENTS:
        forbidden = (REPO_ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write schemas under the repo's {name}/ directory")


def export_schemas(out: Path) -> list[Path]:
    """Write every schema into ``out`` (created if needed); return the written paths."""
    _refuse_forbidden(out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        path = out / f"{name}.schema.json"
        schema = model.model_json_schema(mode="validation")
        path.write_text(json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)
    for path in export_schemas(args.out):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
