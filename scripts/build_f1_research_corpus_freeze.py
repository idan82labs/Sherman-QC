#!/usr/bin/env python3
"""Freeze separate scan-only and sealed-gold manifests for F1 research evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data/bend_corpus/annotations/ground_truth_registry.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-corpus-evidence-20260729-v1/freeze"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_freeze(registry_path: Path, output_dir: Path) -> tuple[Path, Path]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in registry.get("records") or []
        if record.get("release_metric_eligible") is True
        and record.get("truth_status") in {"manual_verified", "dual_review_verified"}
    ]
    records.sort(key=lambda row: (str(row.get("part_key")), str(row.get("scan_name"))))
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        scan_path = Path(str(record["scan_path"])).resolve()
        instance = dict(record.get("instance_geometry") or {})
        label_path = Path(str(instance["source_file"])).resolve()
        if not scan_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing frozen source for {record.get('record_id')}")
        label_payload = json.loads(label_path.read_text(encoding="utf-8"))
        bends = list(label_payload.get("manual_bends") or [])
        expected = int((record.get("counts") or {})["observed_conventional_bends"])
        if len(bends) != expected:
            raise ValueError(
                f"manual-axis count mismatch for {record.get('record_id')}: {len(bends)} != {expected}"
            )
        case_token = f"C{index:02d}"
        inputs.append(
            {
                "case_token": case_token,
                "opaque_input_name": f"input{scan_path.suffix.lower()}",
                "source_scan_path": str(scan_path),
                "source_scan_sha256": _sha256(scan_path),
                "source_scan_bytes": scan_path.stat().st_size,
            }
        )
        gold.append(
            {
                "case_token": case_token,
                "record_id": record["record_id"],
                "part_key": record["part_key"],
                "scan_name": record["scan_name"],
                "scan_state": record["scan_state"],
                "source_scan_sha256": inputs[-1]["source_scan_sha256"],
                "label_path": str(label_path),
                "label_sha256": _sha256(label_path),
                "gold_bend_count": expected,
                "manual_bends": bends,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": 1,
        "corpus_freeze_id": "sherman-f1-manual13-20260729-v1",
        "case_count": len(inputs),
        "unique_part_family_count": len({item["part_key"] for item in gold}),
        "protocol": {
            "inference_receives_truth": False,
            "inference_receives_expected_count": False,
            "inference_receives_part_identifier": False,
            "duplicate_scans_grouped_by_part_family": True,
        },
    }
    inference_path = output_dir / "inference_manifest.json"
    gold_path = output_dir / "sealed_gold_manifest.json"
    inference_path.write_text(
        json.dumps({**common, "cases": inputs}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps({**common, "cases": gold}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return inference_path, gold_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inference, gold = build_freeze(args.registry.resolve(), args.output_dir.resolve())
    print(json.dumps({"inference_manifest": str(inference), "sealed_gold_manifest": str(gold)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
