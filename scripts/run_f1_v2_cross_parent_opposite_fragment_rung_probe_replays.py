#!/usr/bin/env python3
"""Run and attest the exact twelve designated cross-parent-rung replays."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_f1_v2_cross_parent_opposite_fragment_rung_probe import (  # noqa: E402
    PRIMITIVE_PROFILE,
    sha256_file,
)
from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_batch import (  # noqa: E402
    BATCH_KIND,
    BATCH_SCHEMA_VERSION,
    CASE_ROW_FIELDS,
    EXPECTED_CASE_TOKENS,
    FROZEN_COMPLETE_EDGE_BASELINE_MANIFEST_SHA256,
    FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256,
    PEAK_RSS_LIMIT_BYTES,
    PREREGISTERED_INFERENCE_MANIFEST_SHA256,
    PREREGISTERED_V2_POOL_MANIFEST_SHA256,
    _read,
    _rows_by_token,
    _run_case,
    _validate_baseline_manifest,
    _validate_control_manifest,
)


REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_replay_attestation_v1"
DESIGNATED_REPLAY_CASES = frozenset({"C001", "C003", "C014", "C030"})
SELECTION_REASONS = {
    "C001": "INHERITED_200000_POINT_REFERENCE",
    "C003": "FROZEN_MAXIMUM_TRANSITION_MODE_EDGE_COUNT",
    "C014": "FROZEN_MAXIMUM_COMPLETE_SAME_SURFACE_EDGE_COUNT_REFERENCE",
    "C030": "FROZEN_MAXIMUM_TRANSITION_MODE_COUNT",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "profile",
        "research_only",
        "inference_safe",
        "gold_open_authorized",
        "case_count",
        "total_isolated_run_count",
        "probe_manifest_sha256",
        "inference_manifest_sha256",
        "candidate_pool_manifest_sha256",
        "frozen_control_manifest_sha256",
        "complete_edge_baseline_manifest_sha256",
        "cases",
    }
)
CASE_FIELDS = frozenset(
    {
        "case_token",
        "selection_reason",
        "batch_replacement_probe_sha256",
        "batch_frozen_control_probe_sha256",
        "batch_complete_edge_baseline_probe_sha256",
        "batch_added_cross_parent_rung_table_sha256",
        "batch_effective_mode_edge_semantic_sha256",
        "batch_local_plane_pair_records_sha256",
        "batch_certificate_novelty_table_sha256",
        "batch_inference_semantic_sha256",
        "replay_count",
        "replay_replacement_probe_sha256s",
        "maximum_peak_rss_bytes",
        "runs",
    }
)
RUN_FIELDS = frozenset(
    {
        "run_index",
        "replacement_probe_sha256",
        "frozen_control_probe_sha256",
        "complete_edge_baseline_probe_sha256",
        "added_cross_parent_rung_table_sha256",
        "effective_mode_edge_semantic_sha256",
        "local_plane_pair_records_sha256",
        "certificate_novelty_table_sha256",
        "inference_semantic_sha256",
        "resource_profile_sha256",
        "peak_rss_bytes",
        "wall_seconds",
        "input_json_bytes",
        "serialized_output_bytes",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_replay_attestation_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _row_table_hash(row: Mapping[str, Any], key: str) -> str:
    if key in row:
        return _require_sha(row[key], label=key)
    matches = []
    for map_field in ("output_table_hashes", "baseline_table_hashes"):
        table = row.get(map_field)
        if isinstance(table, Mapping) and key in table:
            matches.append(_require_sha(table[key], label=f"{map_field}.{key}"))
    if len(matches) != 1:
        raise ValueError(f"ordinary row must expose exactly one {key}")
    return matches[0]


def _resource_record(row: Mapping[str, Any], *, run_index: int) -> dict[str, Any]:
    record = {
        "run_index": run_index,
        "replacement_probe_sha256": str(row["replacement_probe_sha256"]),
        "frozen_control_probe_sha256": str(row["frozen_control_probe_sha256"]),
        "complete_edge_baseline_probe_sha256": str(row["complete_edge_baseline_probe_sha256"]),
        "added_cross_parent_rung_table_sha256": _row_table_hash(row, "added_cross_parent_rung_table_sha256"),
        "effective_mode_edge_semantic_sha256": _row_table_hash(row, "effective_mode_edge_semantic_sha256"),
        "local_plane_pair_records_sha256": _row_table_hash(row, "local_plane_pair_records_sha256"),
        "certificate_novelty_table_sha256": _row_table_hash(row, "certificate_novelty_table_sha256"),
        "inference_semantic_sha256": str(row["inference_semantic_sha256"]),
        "peak_rss_bytes": int(row["peak_rss_bytes"]),
        "wall_seconds": float(row["wall_seconds"]),
        "input_json_bytes": int(row["input_json_bytes"]),
        "serialized_output_bytes": int(row["serialized_output_bytes"]),
    }
    return {**record, "resource_profile_sha256": _sha_json(record)}


def validate_replay_attestation(
    attestation: Mapping[str, Any], *, batch_manifest: Mapping[str, Any]
) -> None:
    if set(attestation) != TOP_FIELDS:
        raise ValueError("cross-parent replay top-level schema mismatch")
    if (
        attestation.get("schema_version") != REPLAY_SCHEMA_VERSION
        or attestation.get("kind") != REPLAY_KIND
        or attestation.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("cross-parent replay identity mismatch")
    if (
        attestation.get("research_only") is not True
        or attestation.get("inference_safe") is not True
        or attestation.get("gold_open_authorized") is not False
    ):
        raise ValueError("cross-parent replay safety flags mismatch")
    if (
        batch_manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or batch_manifest.get("kind") != BATCH_KIND
        or batch_manifest.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("ordinary cross-parent batch identity mismatch")
    batch_rows = _rows_by_token(batch_manifest, label="ordinary cross-parent batch")
    batch_workers = int(batch_manifest.get("workers") or 0)
    if (
        int(batch_manifest.get("case_count") or -1) != 37
        or int(batch_manifest.get("successful_case_count") or -1) != 37
        or batch_workers < 1
        or batch_workers > 8
        or batch_manifest.get("isolated_process_per_case") is not True
        or batch_manifest.get("gold_open_authorized") is not False
    ):
        raise ValueError("ordinary cross-parent batch count/worker/safety mismatch")
    for token, row in batch_rows.items():
        if set(row) != CASE_ROW_FIELDS or row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary row schema/status mismatch")
    raw_cases = attestation.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("replay cases must be a sequence")
    cases = {str(case.get("case_token") or ""): dict(case) for case in raw_cases if isinstance(case, Mapping)}
    if len(cases) != 4 or set(cases) != DESIGNATED_REPLAY_CASES or int(attestation.get("case_count") or -1) != 4:
        raise ValueError("replay requires exact C001/C003/C014/C030 set")
    if int(attestation.get("total_isolated_run_count") or -1) != 12:
        raise ValueError("replay requires exactly twelve isolated runs")
    for field in TOP_FIELDS:
        if field.endswith("_sha256"):
            _require_sha(attestation.get(field), label=f"replay {field}")
    for token in sorted(DESIGNATED_REPLAY_CASES):
        case = cases[token]
        ordinary = batch_rows[token]
        if set(case) != CASE_FIELDS:
            raise ValueError(f"{token}: replay case schema mismatch")
        if case.get("selection_reason") != SELECTION_REASONS[token]:
            raise ValueError(f"{token}: replay selection reason mismatch")
        expected = {
            "batch_replacement_probe_sha256": ordinary["replacement_probe_sha256"],
            "batch_frozen_control_probe_sha256": ordinary["frozen_control_probe_sha256"],
            "batch_complete_edge_baseline_probe_sha256": ordinary["complete_edge_baseline_probe_sha256"],
            "batch_added_cross_parent_rung_table_sha256": _row_table_hash(ordinary, "added_cross_parent_rung_table_sha256"),
            "batch_effective_mode_edge_semantic_sha256": _row_table_hash(ordinary, "effective_mode_edge_semantic_sha256"),
            "batch_local_plane_pair_records_sha256": _row_table_hash(ordinary, "local_plane_pair_records_sha256"),
            "batch_certificate_novelty_table_sha256": _row_table_hash(ordinary, "certificate_novelty_table_sha256"),
            "batch_inference_semantic_sha256": ordinary["inference_semantic_sha256"],
        }
        for field, value in expected.items():
            if _require_sha(case.get(field), label=f"{token} {field}") != _require_sha(value, label=f"{token} ordinary {field}"):
                raise ValueError(f"{token}: {field} differs from ordinary batch")
        runs_raw = case.get("runs")
        if not isinstance(runs_raw, Sequence) or isinstance(runs_raw, (str, bytes)):
            raise ValueError(f"{token}: runs must be a sequence")
        runs = [dict(run) for run in runs_raw if isinstance(run, Mapping)]
        if len(runs) != 3 or int(case.get("replay_count") or -1) != 3:
            raise ValueError(f"{token}: requires exactly three fresh runs")
        if [int(run.get("run_index") or -1) for run in runs] != [1, 2, 3]:
            raise ValueError(f"{token}: run indices must equal 1,2,3")
        listed = [_require_sha(value, label=f"{token} listed replay") for value in case.get("replay_replacement_probe_sha256s") or ()]
        if len(listed) != 3 or len(set(listed)) != 1 or listed[0] != expected["batch_replacement_probe_sha256"]:
            raise ValueError(f"{token}: replacement replay hashes differ")
        comparisons = {
            "replacement_probe_sha256": "batch_replacement_probe_sha256",
            "frozen_control_probe_sha256": "batch_frozen_control_probe_sha256",
            "complete_edge_baseline_probe_sha256": "batch_complete_edge_baseline_probe_sha256",
            "added_cross_parent_rung_table_sha256": "batch_added_cross_parent_rung_table_sha256",
            "effective_mode_edge_semantic_sha256": "batch_effective_mode_edge_semantic_sha256",
            "local_plane_pair_records_sha256": "batch_local_plane_pair_records_sha256",
            "certificate_novelty_table_sha256": "batch_certificate_novelty_table_sha256",
            "inference_semantic_sha256": "batch_inference_semantic_sha256",
        }
        for run, listed_sha in zip(runs, listed):
            if set(run) != RUN_FIELDS:
                raise ValueError(f"{token}: replay run schema mismatch")
            if run["replacement_probe_sha256"] != listed_sha:
                raise ValueError(f"{token}: run replacement hash mismatch")
            for run_field, case_field in comparisons.items():
                if _require_sha(run.get(run_field), label=f"{token} run {run_field}") != _require_sha(case.get(case_field), label=f"{token} batch {run_field}"):
                    raise ValueError(f"{token}: replay {run_field} differs from batch")
            resource = {field: run[field] for field in RUN_FIELDS if field != "resource_profile_sha256"}
            if _require_sha(run.get("resource_profile_sha256"), label=f"{token} resource") != _sha_json(resource):
                raise ValueError(f"{token}: resource profile hash mismatch")
            if int(run["peak_rss_bytes"]) < 0 or int(run["peak_rss_bytes"]) >= PEAK_RSS_LIMIT_BYTES:
                raise ValueError(f"{token}: replay RSS reaches/exceeds 16 GiB")
            if float(run["wall_seconds"]) < 0:
                raise ValueError(f"{token}: replay wall time invalid")
        if int(case.get("maximum_peak_rss_bytes") or -1) != max(int(run["peak_rss_bytes"]) for run in runs):
            raise ValueError(f"{token}: maximum replay RSS mismatch")
        for field in ("input_json_bytes", "serialized_output_bytes"):
            if len({int(run[field]) for run in runs}) != 1 or int(runs[0][field]) != int(ordinary[field]):
                raise ValueError(f"{token}: replay {field} mismatch")


def run_replays(
    *, probe_manifest_path: Path, inference_manifest_path: Path,
    candidate_pool_manifest_path: Path, a0_prediction_root: Path,
    frozen_control_manifest_path: Path, complete_edge_baseline_manifest_path: Path,
    output_dir: Path, workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise ValueError("replay workers must be in [1,8]")
    probe_manifest_path = probe_manifest_path.resolve(); inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve(); a0_prediction_root = a0_prediction_root.resolve()
    frozen_control_manifest_path = frozen_control_manifest_path.resolve(); complete_edge_baseline_manifest_path = complete_edge_baseline_manifest_path.resolve(); output_dir = output_dir.resolve()
    if sha256_file(inference_manifest_path) != PREREGISTERED_INFERENCE_MANIFEST_SHA256 or sha256_file(candidate_pool_manifest_path) != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("replay frozen source manifest changed")
    control_sha = sha256_file(frozen_control_manifest_path); baseline_sha = sha256_file(complete_edge_baseline_manifest_path)
    control_rows = _validate_control_manifest(_read(frozen_control_manifest_path), control_sha)
    baseline_rows = _validate_baseline_manifest(_read(complete_edge_baseline_manifest_path), baseline_sha)
    batch = _read(probe_manifest_path)
    batch_rows = _rows_by_token(batch, label="ordinary cross-parent batch")
    for token, row in batch_rows.items():
        if set(row) != CASE_ROW_FIELDS or row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary row schema/status mismatch")
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference manifest")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), label="candidate-pool manifest")
    tasks = []; task_meta = []
    for token in sorted(DESIGNATED_REPLAY_CASES):
        scan = Path(str(inference_rows[token]["source_scan_path"])).resolve(); pool = Path(str(pool_rows[token]["path"])).resolve()
        a0 = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        for run_index in (1, 2, 3):
            tasks.append((token, str(scan), str(pool), str(a0), str(control_rows[token]["probe_sha256"]), str(baseline_rows[token]["replacement_probe_sha256"]), str(output_dir / token / f"run_{run_index}.json")))
            task_meta.append((token, run_index))
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        raw_rows = list(executor.map(_run_case, tasks))
    by_token = {token: [] for token in DESIGNATED_REPLAY_CASES}
    for raw, (token, run_index) in zip(raw_rows, task_meta):
        by_token[token].append(_resource_record(raw, run_index=run_index))
    cases = []
    for token in sorted(DESIGNATED_REPLAY_CASES):
        ordinary = batch_rows[token]; runs = sorted(by_token[token], key=lambda row: row["run_index"])
        cases.append({
            "case_token": token, "selection_reason": SELECTION_REASONS[token],
            "batch_replacement_probe_sha256": ordinary["replacement_probe_sha256"],
            "batch_frozen_control_probe_sha256": ordinary["frozen_control_probe_sha256"],
            "batch_complete_edge_baseline_probe_sha256": ordinary["complete_edge_baseline_probe_sha256"],
            "batch_added_cross_parent_rung_table_sha256": _row_table_hash(ordinary, "added_cross_parent_rung_table_sha256"),
            "batch_effective_mode_edge_semantic_sha256": _row_table_hash(ordinary, "effective_mode_edge_semantic_sha256"),
            "batch_local_plane_pair_records_sha256": _row_table_hash(ordinary, "local_plane_pair_records_sha256"),
            "batch_certificate_novelty_table_sha256": _row_table_hash(ordinary, "certificate_novelty_table_sha256"),
            "batch_inference_semantic_sha256": ordinary["inference_semantic_sha256"],
            "replay_count": 3, "replay_replacement_probe_sha256s": [run["replacement_probe_sha256"] for run in runs],
            "maximum_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs), "runs": runs,
        })
    attestation = {
        "schema_version": REPLAY_SCHEMA_VERSION, "kind": REPLAY_KIND, "profile": PRIMITIVE_PROFILE,
        "research_only": True, "inference_safe": True, "gold_open_authorized": False,
        "case_count": 4, "total_isolated_run_count": 12,
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "candidate_pool_manifest_sha256": sha256_file(candidate_pool_manifest_path),
        "frozen_control_manifest_sha256": control_sha,
        "complete_edge_baseline_manifest_sha256": baseline_sha, "cases": cases,
    }
    validate_replay_attestation(attestation, batch_manifest=batch)
    (output_dir / "replay_attestation.json").write_bytes(canonical_replay_attestation_bytes(attestation))
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path); parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path); parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path); parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_replays(probe_manifest_path=args.probe_manifest, inference_manifest_path=args.inference_manifest, candidate_pool_manifest_path=args.candidate_pool_manifest, a0_prediction_root=args.a0_prediction_root, frozen_control_manifest_path=args.frozen_control_manifest, complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest, output_dir=args.output_dir, workers=args.workers)
    path = args.output_dir.resolve() / "replay_attestation.json"
    print(json.dumps({"output": str(path), "replay_attestation_sha256": sha256_file(path), "isolated_run_count": result["total_isolated_run_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
