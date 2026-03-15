#!/opt/homebrew/bin/python3.11
"""
Upsert a single part into the existing bend corpus without deleting evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/Users/idant/82Labs/Sherman QC/Full Project")
SCRIPTS_DIR = ROOT / "scripts"
BACKEND_DIR = ROOT / "backend"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bend_improvement_loop import discover_parts  # noqa: E402
from build_bend_corpus import (  # noqa: E402
    _apply_seed_scan_overrides,
    _build_part_record,
    _load_normalized_expected_overrides,
    _load_seed_labels,
    _materialize_part_folder,
    _write_operational_views,
    _write_summary_md,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _refresh_inventory(corpus_root: Path) -> Dict[str, Any]:
    parts_dir = corpus_root / "parts"
    parts: List[Dict[str, Any]] = []
    for part_dir in sorted(parts_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        meta_path = part_dir / "metadata.json"
        if not meta_path.exists():
            continue
        parts.append(_load_json(meta_path))
    inventory = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "source_dir": str(Path.home() / "Downloads"),
        "corpus_root": str(corpus_root),
        "counts": {
            "parts": len(parts),
            "cad_files": sum(p["cad_count"] for p in parts),
            "scan_files": sum(p["scan_count"] for p in parts),
            "drawing_files": sum(p["drawing_count"] for p in parts),
            "spec_files": sum(p["spec_count"] for p in parts),
            "ready_regression_parts": sum(1 for p in parts if p["readiness"]["ready_regression"]),
            "ready_partial_parts": sum(1 for p in parts if p["readiness"]["ready_partial_regression"]),
            "ready_full_parts": sum(1 for p in parts if p["readiness"]["ready_full_regression"]),
            "needs_confirmation_parts": sum(1 for p in parts if p["readiness"]["needs_confirmation"]),
        },
        "parts": parts,
    }
    _dump_json(corpus_root / "dataset_inventory.json", inventory)
    _write_summary_md(corpus_root, inventory)
    _write_operational_views(corpus_root, inventory)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert one part into the existing bend corpus")
    parser.add_argument("--part-key", required=True)
    parser.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "bend_corpus"))
    args = parser.parse_args()

    downloads_dir = Path(args.downloads)
    corpus_root = Path(args.corpus)
    datasets = {ds.part_key: ds for ds in discover_parts(downloads_dir)}
    if args.part_key not in datasets:
        raise SystemExit(f"Part {args.part_key} not found under {downloads_dir}")

    overrides = _load_normalized_expected_overrides()
    seed_labels = _load_seed_labels()
    dataset = _apply_seed_scan_overrides(datasets[args.part_key], seed_labels)
    expected_override = overrides.get(dataset.part_key)
    if dataset.part_key in seed_labels and isinstance(seed_labels[dataset.part_key], dict):
        seed_expected = seed_labels[dataset.part_key].get("expected_total_bends")
        if isinstance(seed_expected, int) and seed_expected > 0:
            expected_override = seed_expected

    part_record = _build_part_record(dataset, expected_override, seed_labels)
    _materialize_part_folder(corpus_root / "parts" / dataset.part_key, part_record)
    inventory = _refresh_inventory(corpus_root)
    print(json.dumps({"part": dataset.part_key, "inventory_counts": inventory["counts"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
