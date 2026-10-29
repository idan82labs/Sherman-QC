#!/usr/bin/env python3
"""Run all 37 frozen LPF-M1 cases in isolated worker processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_f1_local_plane_fragment_probe import (  # noqa: E402
    build_local_plane_fragment_probe_artifact,
    sha256_file,
)


PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
BATCH_KIND = "f1_local_plane_fragment_probe_batch_v2"
BATCH_SCHEMA_VERSION = 2
EXPECTED_CASE_TOKENS = frozenset(f"C{index:03d}" for index in range(1, 38))
PREREGISTERED_INFERENCE_MANIFEST_SHA256 = (
    "bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66"
)
PREREGISTERED_V2_POOL_MANIFEST_SHA256 = (
    "c3ed8f6356c49c28ca56670cc9d061fe25dc4f1dea71625886e989330f8df3c7"
)
MAX_WORKERS = 18
MAX_PEAK_RSS_BYTES = 16 * 1024**3


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _rows_by_token(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    raw_rows = payload.get("cases")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(raw)
    if set(result) != EXPECTED_CASE_TOKENS:
        missing = sorted(EXPECTED_CASE_TOKENS - set(result))
        extra = sorted(set(result) - EXPECTED_CASE_TOKENS)
        raise ValueError(f"{label} must contain exact C001-C037 set; missing={missing}, extra={extra}")
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux reports KiB.
    return value if sys.platform == "darwin" else value * 1024


def _run_case(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    token, scan_raw, pool_raw, prediction_raw, output_raw = task
    scan = Path(scan_raw)
    pool = Path(pool_raw)
    prediction = Path(prediction_raw)
    output = Path(output_raw)
    started = perf_counter()
    try:
        payload, encoded = build_local_plane_fragment_probe_artifact(
            scan_path=scan,
            candidate_pool_path=pool,
            a0_prediction_path=prediction,
        )
        peak_rss = _peak_rss_bytes()
        if peak_rss >= MAX_PEAK_RSS_BYTES:
            raise MemoryError(
                f"peak RSS cap breached: {peak_rss} >= {MAX_PEAK_RSS_BYTES}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    except Exception as exc:
        raise RuntimeError(f"{token}: LPF-M1 inference job failed: {exc}") from exc
    return {
        "case_token": token,
        "status": "SUCCESS",
        "probe_path": str(output.resolve()),
        "probe_sha256": sha256_file(output),
        "scan_path": str(scan.resolve()),
        "scan_sha256": str(payload["scan_sha256"]),
        "candidate_pool_path": str(pool.resolve()),
        "candidate_pool_sha256": str(payload["candidate_pool_sha256"]),
        "a0_prediction_path": str(prediction.resolve()),
        "a0_prediction_sha256": str(payload["a0_prediction_sha256"]),
        "input_table_hashes": dict(payload["input_table_hashes"]),
        "output_table_hashes": dict(payload["output_table_hashes"]),
        "input_contract_attestation_sha256": str(
            payload["input_contract_attestation"]["input_contract_attestation_sha256"]
        ),
        "inference_semantic_sha256": str(payload["inference_semantic_sha256"]),
        "engineering_profile_sha256": str(payload["engineering_profile_sha256"]),
        "case_counters": dict(payload["case_counters"]),
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak_rss,
        "input_json_bytes": pool.stat().st_size,
        "serialized_output_bytes": len(encoded),
    }


def run_batch(
    *,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve()
    a0_prediction_root = a0_prediction_root.resolve()
    output_dir = output_dir.resolve()
    inference_sha = sha256_file(inference_manifest_path)
    pool_manifest_sha = sha256_file(candidate_pool_manifest_path)
    if inference_sha != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("inference manifest differs from the preregistered Gold37 freeze")
    if pool_manifest_sha != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("V2 pool manifest differs from the preregistered Gold37 freeze")
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference manifest")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), label="V2 pool manifest")
    tasks: list[tuple[str, str, str, str, str]] = []
    for token in sorted(EXPECTED_CASE_TOKENS):
        scan = Path(str(inference_rows[token].get("source_scan_path") or "")).resolve()
        pool = Path(str(pool_rows[token].get("path") or "")).resolve()
        prediction = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        for label, path in (("scan", scan), ("candidate pool", pool), ("A0 prediction", prediction)):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        tasks.append(
            (token, str(scan), str(pool), str(prediction), str(output_dir / f"{token}.json"))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # max_tasks_per_child=1 makes every case a fresh process even when workers
    # are reused as a bounded scheduling pool.
    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
    ) as executor:
        rows = list(executor.map(_run_case, tasks))
    rows.sort(key=lambda item: str(item["case_token"]))
    if len(rows) != 37 or any(row.get("status") != "SUCCESS" for row in rows):
        raise RuntimeError("LPF-M1 batch requires 37/37 successful isolated jobs")
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "kind": BATCH_KIND,
        "profile": PROFILE,
        "research_only": True,
        "inference_safe": True,
        "prediction_changed": False,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "case_count": 37,
        "successful_case_count": 37,
        "workers": workers,
        "isolated_process_per_case": True,
        "peak_rss_cap_bytes": MAX_PEAK_RSS_BYTES,
        "inference_manifest_sha256": inference_sha,
        "candidate_pool_manifest_sha256": pool_manifest_sha,
        "cases": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_batch(
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    manifest_path = args.output_dir.resolve() / "manifest.json"
    print(
        json.dumps(
            {
                "case_count": manifest["case_count"],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
