#!/usr/bin/env python3
"""Run the 37-case cross-parent opposite-fragment rung probe in isolation."""

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

from scripts.diagnose_f1_v2_cross_parent_opposite_fragment_rung_probe import (  # noqa: E402
    PRIMITIVE_PROFILE,
    build_cross_parent_rung_probe_artifact,
    sha256_file,
)


BATCH_SCHEMA_VERSION = 1
BATCH_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_batch_v1"
EXPECTED_CASE_TOKENS = frozenset(f"C{index:03d}" for index in range(1, 38))
PREREGISTERED_INFERENCE_MANIFEST_SHA256 = (
    "bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66"
)
PREREGISTERED_V2_POOL_MANIFEST_SHA256 = (
    "c3ed8f6356c49c28ca56670cc9d061fe25dc4f1dea71625886e989330f8df3c7"
)
FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256 = (
    "3b9dfa156746a01cb38195f7d75c92e23bff839228de96e425b8902edf262510"
)
FROZEN_COMPLETE_EDGE_BASELINE_MANIFEST_SHA256 = (
    "4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5"
)
MAX_BATCH_WORKERS = 8
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
CASE_ROW_FIELDS = frozenset(
    {
        "case_token",
        "status",
        "replacement_probe_sha256",
        "frozen_control_probe_sha256",
        "complete_edge_baseline_probe_sha256",
        "scan_sha256",
        "candidate_pool_sha256",
        "a0_prediction_sha256",
        "input_table_hashes",
        "baseline_table_hashes",
        "output_table_hashes",
        "added_cross_parent_rung_table_sha256",
        "effective_mode_edge_semantic_sha256",
        "local_plane_pair_records_sha256",
        "certificate_novelty_table_sha256",
        "input_contract_attestation_sha256",
        "baseline_rebuild_attestation_sha256",
        "cross_parent_rung_attestation_sha256",
        "global_deletion_counterfactual_sha256",
        "engineering_profile_sha256",
        "inference_semantic_sha256",
        "case_counters",
        "wall_seconds",
        "peak_rss_bytes",
        "input_json_bytes",
        "serialized_output_bytes",
    }
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _rows_by_token(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} tokens must be nonempty and unique")
        result[token] = dict(raw)
    if set(result) != EXPECTED_CASE_TOKENS:
        raise ValueError(f"{label} must contain exact C001-C037")
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _run_case(task: tuple[str, str, str, str, str, str, str]) -> dict[str, Any]:
    token, scan_raw, pool_raw, a0_raw, control_sha, baseline_sha, output_raw = task
    scan = Path(scan_raw)
    pool = Path(pool_raw)
    a0 = Path(a0_raw)
    output = Path(output_raw)
    started = perf_counter()
    try:
        payload, encoded = build_cross_parent_rung_probe_artifact(
            scan_path=scan,
            candidate_pool_path=pool,
            a0_prediction_path=a0,
            expected_frozen_control_probe_sha256=control_sha,
            expected_complete_edge_baseline_probe_sha256=baseline_sha,
        )
        peak_rss = _peak_rss_bytes()
        if peak_rss >= PEAK_RSS_LIMIT_BYTES:
            raise MemoryError(
                f"peak RSS cap breached: {peak_rss} >= {PEAK_RSS_LIMIT_BYTES}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    except Exception as exc:
        raise RuntimeError(f"{token}: cross-parent rung job failed: {exc}") from exc
    row = {
        "case_token": token,
        "status": "SUCCESS",
        "replacement_probe_sha256": sha256_file(output),
        "frozen_control_probe_sha256": str(payload["frozen_control_probe_sha256"]),
        "complete_edge_baseline_probe_sha256": str(
            payload["complete_edge_baseline_probe_sha256"]
        ),
        "scan_sha256": str(payload["scan_sha256"]),
        "candidate_pool_sha256": str(payload["candidate_pool_sha256"]),
        "a0_prediction_sha256": str(payload["a0_prediction_sha256"]),
        "input_table_hashes": dict(payload["input_table_hashes"]),
        "baseline_table_hashes": dict(payload["baseline_table_hashes"]),
        "output_table_hashes": dict(payload["output_table_hashes"]),
        "added_cross_parent_rung_table_sha256": str(
            payload["output_table_hashes"]["added_cross_parent_rung_table_sha256"]
        ),
        "effective_mode_edge_semantic_sha256": str(
            payload["cross_parent_rung_attestation"]["effective_mode_edge_semantic_sha256"]
        ),
        "local_plane_pair_records_sha256": str(
            payload["output_table_hashes"]["local_plane_pair_records_sha256"]
        ),
        "certificate_novelty_table_sha256": str(
            payload["output_table_hashes"]["certificate_novelty_table_sha256"]
        ),
        "input_contract_attestation_sha256": str(
            payload["input_contract_attestation"]["input_contract_attestation_sha256"]
        ),
        "baseline_rebuild_attestation_sha256": str(
            payload["baseline_rebuild_attestation"]["baseline_rebuild_attestation_sha256"]
        ),
        "cross_parent_rung_attestation_sha256": str(
            payload["cross_parent_rung_attestation"]["cross_parent_rung_attestation_sha256"]
        ),
        "global_deletion_counterfactual_sha256": str(
            payload["global_deletion_counterfactual"]["global_deletion_counterfactual_sha256"]
        ),
        "engineering_profile_sha256": str(payload["engineering_profile_sha256"]),
        "inference_semantic_sha256": str(payload["inference_semantic_sha256"]),
        "case_counters": dict(payload["case_counters"]),
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak_rss,
        "input_json_bytes": pool.stat().st_size,
        "serialized_output_bytes": len(encoded),
    }
    if set(row) != CASE_ROW_FIELDS:
        raise AssertionError("cross-parent rung ordinary row schema drift")
    return row


def _validate_control_manifest(manifest: Mapping[str, Any], actual_sha: str) -> dict[str, dict[str, Any]]:
    if actual_sha != FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256:
        raise ValueError("frozen LPF V2 control manifest hash changed")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "f1_local_plane_fragment_probe_batch_v2"
        or manifest.get("profile") != "LOCAL_PLANE_FRAGMENT_PROBE_V2"
        or int(manifest.get("case_count") or -1) != 37
        or int(manifest.get("successful_case_count") or -1) != 37
    ):
        raise ValueError("frozen LPF control manifest identity/count mismatch")
    return _rows_by_token(manifest, label="frozen LPF control manifest")


def _validate_baseline_manifest(manifest: Mapping[str, Any], actual_sha: str) -> dict[str, dict[str, Any]]:
    if actual_sha != FROZEN_COMPLETE_EDGE_BASELINE_MANIFEST_SHA256:
        raise ValueError("complete-edge baseline manifest hash changed")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "f1_v2_complete_same_surface_edge_probe_batch_v1"
        or manifest.get("profile") != "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
        or int(manifest.get("case_count") or -1) != 37
        or int(manifest.get("successful_case_count") or -1) != 37
    ):
        raise ValueError("complete-edge baseline manifest identity/count mismatch")
    return _rows_by_token(manifest, label="complete-edge baseline manifest")


def run_batch(
    *,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_BATCH_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_BATCH_WORKERS}]")
    paths = [
        inference_manifest_path.resolve(),
        candidate_pool_manifest_path.resolve(),
        a0_prediction_root.resolve(),
        frozen_control_manifest_path.resolve(),
        complete_edge_baseline_manifest_path.resolve(),
        output_dir.resolve(),
    ]
    inference_manifest_path, candidate_pool_manifest_path, a0_prediction_root, frozen_control_manifest_path, complete_edge_baseline_manifest_path, output_dir = paths
    inference_sha = sha256_file(inference_manifest_path)
    pool_manifest_sha = sha256_file(candidate_pool_manifest_path)
    control_manifest_sha = sha256_file(frozen_control_manifest_path)
    baseline_manifest_sha = sha256_file(complete_edge_baseline_manifest_path)
    if inference_sha != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("inference manifest differs from frozen Gold37")
    if pool_manifest_sha != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("candidate-pool manifest differs from frozen V2")
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference manifest")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), label="candidate-pool manifest")
    control_rows = _validate_control_manifest(_read(frozen_control_manifest_path), control_manifest_sha)
    baseline_rows = _validate_baseline_manifest(_read(complete_edge_baseline_manifest_path), baseline_manifest_sha)
    tasks: list[tuple[str, str, str, str, str, str, str]] = []
    for token in sorted(EXPECTED_CASE_TOKENS):
        scan = Path(str(inference_rows[token].get("source_scan_path") or "")).resolve()
        pool = Path(str(pool_rows[token].get("path") or "")).resolve()
        a0 = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        for label, path in (("scan", scan), ("candidate pool", pool), ("A0 prediction", a0)):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        tasks.append(
            (
                token,
                str(scan),
                str(pool),
                str(a0),
                str(control_rows[token]["probe_sha256"]),
                str(baseline_rows[token]["replacement_probe_sha256"]),
                str(output_dir / f"{token}.json"),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        rows = list(executor.map(_run_case, tasks))
    rows.sort(key=lambda row: str(row["case_token"]))
    if len(rows) != 37 or any(row.get("status") != "SUCCESS" for row in rows):
        raise RuntimeError("cross-parent rung batch requires 37/37 successful isolated jobs")
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "kind": BATCH_KIND,
        "profile": PRIMITIVE_PROFILE,
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "gold_open_authorized": False,
        "case_count": 37,
        "successful_case_count": 37,
        "workers": workers,
        "isolated_process_per_case": True,
        "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
        "inference_manifest_sha256": inference_sha,
        "candidate_pool_manifest_sha256": pool_manifest_sha,
        "frozen_control_manifest_sha256": control_manifest_sha,
        "complete_edge_baseline_manifest_sha256": baseline_manifest_sha,
        "cases": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_batch(
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    path = args.output_dir.resolve() / "manifest.json"
    print(json.dumps({"manifest": str(path), "manifest_sha256": sha256_file(path), "case_count": manifest["case_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
