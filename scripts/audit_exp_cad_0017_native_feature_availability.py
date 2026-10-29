#!/usr/bin/env python3
"""Audit authenticated SLDPRTs for the unchanged EXP17 native selector."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
EXP17_PATH = HERE / "run_exp_cad_0017_native_roundbend_patch_axes.py"
SPEC = importlib.util.spec_from_file_location("exp17_native_roundbend", EXP17_PATH)
EXP17 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP17)
V1 = EXP17.V1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--source-sldprt", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    records = []
    seen = set()
    for source in sorted((path.resolve() for path in args.source_sldprt), key=lambda path: path.name):
        source_hash = V1.sha256_path(source)
        if source_hash in seen:
            continue
        seen.add(source_hash)
        record = {"source_basename": source.name, "source_sha256": source_hash}
        try:
            cadir = V1.decode_cadir(source, args.cadmpeg.resolve())
            native_features = cadir.get("native", {}).get("sldprt", {}).get("arenas", {}).get("features", [])
            record["active_roundbend_feature_count"] = sum(
                1 for item in native_features if item.get("kind") == "RoundBend" and item.get("suppressed") is False
            )
            for source_name, output_name in (("Bends", "source_bends_property"), ("Sheet Metal Thickness", "source_sheet_thickness_property")):
                try:
                    record[output_name] = V1._custom_property(cadir, source_name)
                except Exception as error:
                    record[output_name] = None
                    record[f"{output_name}_error"] = str(error)
            extraction = EXP17.extract_native_roundbend_patch_axes(cadir)
            record.update(
                {
                    "status": "PASS_NATIVE_SELECTOR_AVAILABLE",
                    "qualifying_event_count": extraction["qualifying_event_count"],
                    "qualifying_axis_cluster_count": extraction["qualifying_axis_cluster_count"],
                    "sheet_thickness_mm": extraction["sheet_thickness_mm"],
                }
            )
        except Exception as error:
            record.update({"status": "FAIL_CLOSED_NATIVE_SELECTOR_UNAVAILABLE", "reason": str(error)})
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    payload = {
        "schema": "EXP_CAD_0017_NATIVE_FEATURE_AVAILABILITY_AUDIT_V1",
        "selector": "ACTIVE_ROUNDBEND_COUNT_SOURCE_THICKNESS_COAXIAL_INTERVAL_PATCH_PAIR_V1",
        "source_count": len(records),
        "pass_count": sum(row["status"].startswith("PASS_") for row in records),
        "fail_count": sum(row["status"].startswith("FAIL_") for row in records),
        "records": records,
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
    }
    V1.atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
