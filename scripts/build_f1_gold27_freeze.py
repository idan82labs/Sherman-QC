#!/usr/bin/env python3
"""Freeze opaque inference, sealed gold, and grouped folds for the 27-part corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/freeze"
SPLIT_SALT = "sherman-f1-gold27-grouped-v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(catalog_path: Path, output_dir: Path) -> dict[str, str]:
    catalog = _read(catalog_path)
    cases = sorted(catalog["cases"], key=lambda row: (str(row["part_id"]), str(row["case_id"])))
    freeze_id = f"sherman-f1-gold27-{catalog['source_snapshot_sha256'][:16]}"
    inference_cases: list[dict[str, Any]] = []
    gold_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        scan_path = Path(str(case["studio"]["path"])).resolve()
        label_path = Path(str(case["source_labels_path"])).resolve()
        if not scan_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"{case['case_id']}: frozen source or label missing")
        token = f"C{index:03d}"
        scan_hash = _sha256(scan_path)
        inference_cases.append(
            {
                "case_token": token,
                "opaque_input_name": "input.ply",
                "source_scan_path": str(scan_path),
                "source_scan_sha256": scan_hash,
                "source_scan_bytes": scan_path.stat().st_size,
            }
        )
        gold_cases.append(
            {
                "case_token": token,
                "record_id": case["case_id"],
                "part_key": case["part_id"],
                "scan_name": case["scan_name"],
                "scan_state": "full",
                "truth_tier": case["truth_tier"],
                "promotion_eligible": bool(case["promotion_eligible"]),
                "source_scan_sha256": scan_hash,
                "label_path": str(label_path),
                "label_sha256": _sha256(label_path),
                "gold_bend_count": int(case["bend_count"]),
                "manual_bends": case["bends"],
            }
        )

    families = sorted({str(case["part_key"]) for case in gold_cases})
    ranked = sorted(
        families,
        key=lambda family: hashlib.sha256(f"{SPLIT_SALT}:{family}".encode()).hexdigest(),
    )
    family_fold = {family: index % 5 for index, family in enumerate(ranked)}
    folds = [
        {
            "fold": fold,
            "locked_part_families": sorted(
                family for family, assigned in family_fold.items() if assigned == fold
            ),
            "development_part_families": sorted(
                family for family, assigned in family_fold.items() if assigned != fold
            ),
        }
        for fold in range(5)
    ]
    case_folds = [
        {
            "case_token": case["case_token"],
            "record_id": case["record_id"],
            "part_key": case["part_key"],
            "fold": family_fold[str(case["part_key"])],
        }
        for case in gold_cases
    ]
    protocol = {
        "inference_receives_truth": False,
        "inference_receives_expected_count": False,
        "inference_receives_part_identifier": False,
        "inference_receives_cad": False,
        "source_points_serialized": False,
        "duplicate_scans_grouped_by_part_family": True,
        "deterministic_cpu_path": True,
    }
    common = {
        "schema_version": 1,
        "corpus_freeze_id": freeze_id,
        "case_count": len(cases),
        "unique_part_family_count": len(families),
        "gold_bend_count": sum(int(case["bend_count"]) for case in cases),
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "protocol": protocol,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_path = output_dir / "inference_manifest.json"
    gold_path = output_dir / "sealed_gold_manifest.json"
    split_path = output_dir / "grouped_five_fold_split.json"
    inference_path.write_text(json.dumps({**common, "cases": inference_cases}, indent=2) + "\n")
    gold_path.write_text(json.dumps({**common, "cases": gold_cases}, indent=2) + "\n")
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_id": SPLIT_SALT,
                "corpus_freeze_id": freeze_id,
                "rule": "SHA256-ranked part families assigned round-robin to five locked folds",
                "folds": folds,
                "cases": case_folds,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "inference_manifest": str(inference_path),
        "sealed_gold_manifest": str(gold_path),
        "grouped_split": str(split_path),
        "corpus_freeze_id": freeze_id,
        "case_count": len(cases),
        "unique_part_count": len(families),
        "gold_bend_count": common["gold_bend_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.catalog.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
