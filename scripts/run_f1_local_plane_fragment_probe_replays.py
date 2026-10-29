#!/usr/bin/env python3
"""Create the exact nine-run LPF-M1-V2 designated replay attestation."""

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

from scripts.diagnose_f1_local_plane_fragment_probe import sha256_file  # noqa: E402
from scripts.run_f1_local_plane_fragment_probe_batch import (  # noqa: E402
    BATCH_KIND,
    BATCH_SCHEMA_VERSION,
    EXPECTED_CASE_TOKENS,
    MAX_PEAK_RSS_BYTES,
    PREREGISTERED_INFERENCE_MANIFEST_SHA256,
    PREREGISTERED_V2_POOL_MANIFEST_SHA256,
    _read,
    _rows_by_token,
    _run_case,
)


SCHEMA_VERSION = 2
KIND = "f1_local_plane_fragment_probe_replay_attestation_v2"
PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
EXPECTED_CASES = frozenset({"C001", "C003", "C030"})
SELECTION_REASONS = {
    "C001": "INHERITED_200000_POINT_REFERENCE",
    "C030": "MAXIMUM_TRANSITION_MODE_COUNT",
    "C003": "MAXIMUM_TRANSITION_MODE_EDGE_COUNT",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTESTATION_FIELDS = frozenset({
    "schema_version", "kind", "profile", "research_only", "inference_safe",
    "case_count", "total_isolated_run_count", "probe_manifest_sha256",
    "inference_manifest_sha256", "candidate_pool_manifest_sha256", "cases",
})
_CASE_FIELDS = frozenset({
    "case_token", "selection_reason", "working_point_count",
    "transition_mode_count", "transition_mode_edge_count", "batch_probe_sha256",
    "replay_count", "replay_probe_sha256s", "maximum_peak_rss_bytes", "runs",
})
_RUN_FIELDS = frozenset({
    "run_index", "probe_sha256", "resource_profile_sha256", "peak_rss_bytes",
    "wall_seconds", "input_json_bytes", "serialized_output_bytes",
})


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def canonical_replay_attestation_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(payload) + b"\n"


def _sha_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _case_rows(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be unique")
        result[token] = dict(raw)
    return result


def select_designated_cases(pool_manifest: Mapping[str, Any]) -> dict[str, str]:
    rows = _case_rows(pool_manifest, "V2 pool manifest")
    if "C001" not in rows or int(rows["C001"].get("working_point_count") or -1) != 200_000:
        raise ValueError("C001 is not the inherited 200,000-point reference")

    def maximum(field: str) -> str:
        values = {
            token: int(row.get(field) or -1) for token, row in rows.items()
        }
        if not values or min(values.values()) < 0:
            raise ValueError(f"V2 pool manifest lacks valid {field}")
        maximum_value = max(values.values())
        return min(token for token, value in values.items() if value == maximum_value)

    selected = {
        "C001": SELECTION_REASONS["C001"],
        maximum("transition_mode_count"): "MAXIMUM_TRANSITION_MODE_COUNT",
        maximum("transition_mode_edge_count"): "MAXIMUM_TRANSITION_MODE_EDGE_COUNT",
    }
    if selected != SELECTION_REASONS:
        raise ValueError(f"designated replay selection differs from frozen ruling: {selected}")
    return selected


def _resource_record(row: Mapping[str, Any], run_index: int) -> dict[str, Any]:
    record = {
        "run_index": int(run_index),
        "probe_sha256": str(row["probe_sha256"]),
        "peak_rss_bytes": int(row["peak_rss_bytes"]),
        "wall_seconds": float(row["wall_seconds"]),
        "input_json_bytes": int(row["input_json_bytes"]),
        "serialized_output_bytes": int(row["serialized_output_bytes"]),
    }
    return {
        **record,
        "resource_profile_sha256": _sha_json(record),
    }


def validate_replay_attestation(
    attestation: Mapping[str, Any],
    *,
    batch_rows: Mapping[str, Mapping[str, Any]] | None = None,
    pool_manifest: Mapping[str, Any] | None = None,
) -> None:
    if set(attestation) != _ATTESTATION_FIELDS:
        raise ValueError("replay attestation top-level schema mismatch")
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != KIND
        or attestation.get("profile") != PROFILE
    ):
        raise ValueError("replay attestation V2 identity mismatch")
    if attestation.get("research_only") is not True or attestation.get("inference_safe") is not True:
        raise ValueError("replay attestation safety flags are invalid")
    cases = _case_rows(attestation, "replay attestation")
    if set(cases) != EXPECTED_CASES or int(attestation.get("case_count") or -1) != 3:
        raise ValueError("replay attestation requires exact C001/C003/C030 set")
    if int(attestation.get("total_isolated_run_count") or -1) != 9:
        raise ValueError("replay attestation requires exactly nine isolated runs")
    for field in (
        "probe_manifest_sha256", "inference_manifest_sha256",
        "candidate_pool_manifest_sha256",
    ):
        _require_sha(attestation.get(field), f"replay {field}")
    if pool_manifest is not None and select_designated_cases(pool_manifest) != SELECTION_REASONS:
        raise ValueError("replay designated-case recomputation failed")
    pool_rows = _case_rows(pool_manifest, "V2 pool manifest") if pool_manifest is not None else None
    for token in sorted(EXPECTED_CASES):
        case = cases[token]
        if set(case) != _CASE_FIELDS:
            raise ValueError(f"{token}: replay case schema mismatch")
        if case.get("selection_reason") != SELECTION_REASONS[token]:
            raise ValueError(f"{token}: replay selection reason mismatch")
        if pool_rows is not None:
            pool_row = pool_rows[token]
            for attested_field, pool_field in (
                ("working_point_count", "working_point_count"),
                ("transition_mode_count", "transition_mode_count"),
                ("transition_mode_edge_count", "transition_mode_edge_count"),
            ):
                if int(case.get(attested_field) or -1) != int(pool_row.get(pool_field) or -1):
                    raise ValueError(f"{token}: replay {attested_field} differs from frozen pool")
        runs = case.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or len(runs) != 3:
            raise ValueError(f"{token}: replay requires exactly three runs")
        if int(case.get("replay_count") or -1) != 3:
            raise ValueError(f"{token}: replay_count must equal three")
        hashes = [_require_sha(value, f"{token} replay hash") for value in case.get("replay_probe_sha256s") or ()]
        if len(hashes) != 3 or len(set(hashes)) != 1:
            raise ValueError(f"{token}: three replay hashes must be byte-identical")
        batch_hash = _require_sha(case.get("batch_probe_sha256"), f"{token} batch probe")
        if hashes[0] != batch_hash:
            raise ValueError(f"{token}: replay hash differs from batch probe")
        if batch_rows is not None and batch_hash != _require_sha(
            batch_rows[token].get("probe_sha256"), f"{token} batch manifest probe"
        ):
            raise ValueError(f"{token}: attested batch probe hash mismatch")
        run_records = [dict(run) for run in runs if isinstance(run, Mapping)]
        if len(run_records) != 3 or [int(run.get("run_index") or -1) for run in run_records] != [1, 2, 3]:
            raise ValueError(f"{token}: replay run indices must be exactly 1,2,3")
        for run, digest in zip(run_records, hashes):
            if set(run) != _RUN_FIELDS:
                raise ValueError(f"{token}: replay run schema mismatch")
            if _require_sha(run.get("probe_sha256"), f"{token} run probe") != digest:
                raise ValueError(f"{token}: run/replay probe hash mismatch")
            resource = {key: run[key] for key in (
                "run_index", "probe_sha256", "peak_rss_bytes", "wall_seconds",
                "input_json_bytes", "serialized_output_bytes",
            )}
            if _require_sha(run.get("resource_profile_sha256"), f"{token} resource profile") != _sha_json(resource):
                raise ValueError(f"{token}: resource profile hash mismatch")
            if int(run["peak_rss_bytes"]) < 0 or int(run["peak_rss_bytes"]) >= MAX_PEAK_RSS_BYTES:
                raise ValueError(f"{token}: replay RSS reaches or exceeds 16 GiB")
            if float(run["wall_seconds"]) < 0.0:
                raise ValueError(f"{token}: replay wall time must be nonnegative")
        if int(case.get("maximum_peak_rss_bytes") or -1) != max(
            int(run["peak_rss_bytes"]) for run in run_records
        ):
            raise ValueError(f"{token}: maximum replay RSS mismatch")
        for field in ("input_json_bytes", "serialized_output_bytes"):
            if len({int(run[field]) for run in run_records}) != 1:
                raise ValueError(f"{token}: replay {field} values differ")


def run_replays(
    *,
    probe_manifest_path: Path,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 9:
        raise ValueError("replay workers must be in [1,9]")
    probe_manifest_path = probe_manifest_path.resolve()
    inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if sha256_file(inference_manifest_path) != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("replay inference manifest differs from frozen Gold37")
    if sha256_file(candidate_pool_manifest_path) != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("replay pool manifest differs from frozen V2 manifest")
    probe_manifest = _read(probe_manifest_path)
    if (
        probe_manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or probe_manifest.get("kind") != BATCH_KIND
        or probe_manifest.get("profile") != PROFILE
    ):
        raise ValueError("replays require the V2 ordinary batch manifest")
    batch_rows = _case_rows(probe_manifest, "V2 batch manifest")
    if set(batch_rows) != EXPECTED_CASE_TOKENS:
        raise ValueError("replays require the exact C001-C037 ordinary batch")
    if _require_sha(
        probe_manifest.get("inference_manifest_sha256"), "batch inference manifest"
    ) != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("ordinary batch references a different inference freeze")
    if _require_sha(
        probe_manifest.get("candidate_pool_manifest_sha256"), "batch pool manifest"
    ) != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("ordinary batch references a different V2 pool freeze")
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference manifest")
    pool_manifest = _read(candidate_pool_manifest_path)
    pool_rows = _rows_by_token(pool_manifest, label="V2 pool manifest")
    selection = select_designated_cases(pool_manifest)
    tasks: list[tuple[str, str, str, str, str]] = []
    task_meta: list[tuple[str, int]] = []
    for token in sorted(selection):
        scan = Path(str(inference_rows[token]["source_scan_path"])).resolve()
        pool = Path(str(pool_rows[token]["path"])).resolve()
        prediction = a0_prediction_root.resolve() / "cases" / token / "prediction_owned_regions.json"
        for run_index in (1, 2, 3):
            output = output_dir / token / f"run_{run_index}.json"
            tasks.append((token, str(scan), str(pool), str(prediction), str(output)))
            task_meta.append((token, run_index))
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        raw_rows = list(executor.map(_run_case, tasks))
    by_token: dict[str, list[dict[str, Any]]] = {token: [] for token in selection}
    for raw, (token, run_index) in zip(raw_rows, task_meta):
        by_token[token].append(_resource_record(raw, run_index))
    cases = []
    for token in sorted(selection):
        runs = sorted(by_token[token], key=lambda row: int(row["run_index"]))
        pool_row = pool_rows[token]
        case = {
            "case_token": token,
            "selection_reason": selection[token],
            "working_point_count": int(pool_row["working_point_count"]),
            "transition_mode_count": int(pool_row["transition_mode_count"]),
            "transition_mode_edge_count": int(pool_row["transition_mode_edge_count"]),
            "batch_probe_sha256": str(batch_rows[token]["probe_sha256"]),
            "replay_count": 3,
            "replay_probe_sha256s": [str(run["probe_sha256"]) for run in runs],
            "maximum_peak_rss_bytes": max(int(run["peak_rss_bytes"]) for run in runs),
            "runs": runs,
        }
        cases.append(case)
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "research_only": True,
        "inference_safe": True,
        "case_count": 3,
        "total_isolated_run_count": 9,
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "candidate_pool_manifest_sha256": sha256_file(candidate_pool_manifest_path),
        "cases": cases,
    }
    validate_replay_attestation(
        attestation, batch_rows=batch_rows, pool_manifest=pool_manifest
    )
    path = output_dir / "replay_attestation.json"
    path.write_bytes(canonical_replay_attestation_bytes(attestation))
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attestation = run_replays(
        probe_manifest_path=args.probe_manifest,
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    path = args.output_dir.resolve() / "replay_attestation.json"
    print(json.dumps({
        "output": str(path),
        "replay_attestation_sha256": sha256_file(path),
        "isolated_run_count": attestation["total_isolated_run_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
