#!/usr/bin/env python3
"""Materialize a validated delivered cohort into opaque evaluator custody."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_pending_delivery_intake_v1 import inspect_delivery, materialize_copy_plan
from scripts.prepare_f1_physical_intake_v2 import _atomic, _consumed_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--lineage-registry", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    parser.add_argument("--evaluator-receipt", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.lineage_registry.read_text(encoding="utf-8"))
    result = inspect_delivery(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        download_root=args.download_root,
        target_root=args.target_root,
        consumed_scan_hashes=_consumed_hashes(registry, args.workspace_root.resolve()),
    )
    if not args.validate_only:
        materialize_copy_plan(result["copy_plan"])
        result["public_receipt"]["status"] = "MATERIALIZED_PENDING_CUSTODIAN_METADATA_AND_LINE_TRUTH"
        result["evaluator_receipt"]["status"] = "MATERIALIZED_PENDING_CUSTODIAN_METADATA_AND_LINE_TRUTH"
    _atomic(args.public_receipt, result["public_receipt"])
    _atomic(args.evaluator_receipt, result["evaluator_receipt"])
    print(json.dumps({
        "status": result["public_receipt"]["status"],
        "scan_count": result["public_receipt"]["scan_count"],
        "qualification_capacity_credit": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

