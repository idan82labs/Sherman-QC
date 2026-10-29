#!/usr/bin/env python3
"""Run deterministic registration and persist only fail-closed canonical semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.f1_registration_semantic_canonical_v1 import canonical_json_bytes, semantic_payload  # noqa: E402
from scripts.run_exp_cad_0012_deterministic_registration import execute_case  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("gold_or_expected_counts_present") is not False:
        raise ValueError("truth-denying manifest required")
    started = perf_counter()
    records = []
    for index, row in enumerate(manifest["records"], 1):
        print(f"[{index}/{len(manifest['records'])}] {row['case_token']}", flush=True)
        raw = execute_case(row, args.seed)
        semantic = semantic_payload(raw)
        case_root = args.output_root / "cases" / row["case_token"]
        semantic_path = case_root / "semantic_result.json"
        diagnostic_path = case_root / "diagnostic_result.json"
        atomic_bytes(semantic_path, canonical_json_bytes(semantic))
        atomic_json(diagnostic_path, raw)
        records.append({
            "case_token": row["case_token"],
            "semantic_result_sha256": sha(semantic_path),
            "diagnostic_result_sha256": sha(diagnostic_path),
            "datum_selected": semantic["datum_selected"],
            "rejected_datum_candidate_fail_closed": semantic["rejected_datum_candidate_fail_closed"],
        })
    summary = {
        "schema": "EXP_CAD_0013_CANONICAL_REGISTRATION_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0013",
        "manifest_sha256": sha(args.manifest),
        "case_count": len(records),
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_gib": peak_rss_gib(),
        "records": records,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("case_count", "elapsed_seconds", "peak_rss_gib")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
