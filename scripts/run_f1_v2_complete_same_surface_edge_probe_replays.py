#!/usr/bin/env python3
"""Create the exact twelve-run complete-SAME_SURFACE replay attestation."""

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


from scripts.diagnose_f1_v2_complete_same_surface_edge_probe import (
    PRIMITIVE_PROFILE,
    sha256_file,
)
from scripts.run_f1_v2_complete_same_surface_edge_probe_batch import (
    BATCH_KIND,
    BATCH_ROW_FIELDS,
    BATCH_SCHEMA_VERSION,
    EXPECTED_CASE_TOKENS,
    FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256,
    PEAK_RSS_LIMIT_BYTES,
    PREREGISTERED_INFERENCE_MANIFEST_SHA256,
    PREREGISTERED_V2_POOL_MANIFEST_SHA256,
    _read,
    _rows_by_token,
    _run_case,
    _validate_frozen_control_manifest,
)


REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = "f1_v2_complete_same_surface_edge_probe_replay_attestation_v1"
FROZEN_REPLAY_CASES = frozenset({"C001", "C003", "C030"})
SELECTION_REASONS = {
    "C001": "INHERITED_200000_POINT_REFERENCE",
    "C003": "FROZEN_MAXIMUM_TRANSITION_MODE_EDGE_COUNT",
    "C030": "FROZEN_MAXIMUM_TRANSITION_MODE_COUNT",
}
OUTSIDE_MAX_REASON = "MAXIMUM_COMPLETE_SAME_SURFACE_EDGE_COUNT_OUTSIDE_FROZEN_SET"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REPLAY_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "profile",
        "research_only",
        "inference_safe",
        "case_count",
        "total_isolated_run_count",
        "probe_manifest_sha256",
        "inference_manifest_sha256",
        "candidate_pool_manifest_sha256",
        "frozen_control_manifest_sha256",
        "cases",
    }
)
REPLAY_CASE_FIELDS = frozenset(
    {
        "case_token",
        "selection_reason",
        "working_point_count",
        "transition_mode_count",
        "transition_mode_edge_count",
        "complete_same_surface_edge_count",
        "batch_replacement_probe_sha256",
        "batch_frozen_control_probe_sha256",
        "batch_complete_same_surface_edge_table_sha256",
        "batch_effective_mode_edge_table_sha256",
        "batch_inference_semantic_sha256",
        "replay_count",
        "replay_replacement_probe_sha256s",
        "maximum_peak_rss_bytes",
        "runs",
    }
)
REPLAY_RUN_FIELDS = frozenset(
    {
        "run_index",
        "replacement_probe_sha256",
        "frozen_control_probe_sha256",
        "complete_same_surface_edge_table_sha256",
        "effective_mode_edge_table_sha256",
        "inference_semantic_sha256",
        "resource_profile_sha256",
        "peak_rss_bytes",
        "wall_seconds",
        "input_json_bytes",
        "serialized_output_bytes",
    }
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def canonical_replay_attestation_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def _sha_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _batch_complete_edge_count(row: Mapping[str, Any]) -> int:
    counters = row.get("case_counters")
    if not isinstance(counters, Mapping):
        raise ValueError("ordinary batch row lacks case_counters")
    raw = counters.get("complete_same_surface_edge_count")
    value = -1 if raw is None else int(raw)
    if value < 0:
        raise ValueError("ordinary batch row lacks complete_same_surface_edge_count")
    return value


def select_designated_cases(
    batch_manifest: Mapping[str, Any],
) -> dict[str, str]:
    rows = _rows_by_token(batch_manifest, label="ordinary edge-probe batch")
    working = {token: _row_counts(row)[0] for token, row in rows.items()}
    modes = {token: _row_counts(row)[1] for token, row in rows.items()}
    inherited_edges = {token: _row_counts(row)[2] for token, row in rows.items()}
    if working["C001"] != 200_000:
        raise ValueError("C001 is not the inherited 200,000-point reference")

    def lexical_maximum(values: Mapping[str, int]) -> str:
        maximum = max(values.values())
        return min(token for token, value in values.items() if value == maximum)

    if lexical_maximum(modes) != "C030":
        raise ValueError("C030 is not the frozen maximum transition-mode case")
    if lexical_maximum(inherited_edges) != "C003":
        raise ValueError("C003 is not the frozen maximum transition-mode-edge case")
    outside = {
        token: _batch_complete_edge_count(row)
        for token, row in rows.items()
        if token not in FROZEN_REPLAY_CASES
    }
    if not outside:
        raise ValueError("ordinary batch has no case outside the frozen replay set")
    maximum = max(outside.values())
    fourth = min(token for token, count in outside.items() if count == maximum)
    return {**SELECTION_REASONS, fourth: OUTSIDE_MAX_REASON}


def _resource_record(row: Mapping[str, Any], *, run_index: int) -> dict[str, Any]:
    edge_hashes = row.get("replacement_edge_table_hashes")
    if not isinstance(edge_hashes, Mapping):
        raise ValueError("replay row lacks replacement edge-table hashes")
    record = {
        "run_index": int(run_index),
        "replacement_probe_sha256": str(row["replacement_probe_sha256"]),
        "frozen_control_probe_sha256": str(row["frozen_control_probe_sha256"]),
        "complete_same_surface_edge_table_sha256": str(
            edge_hashes["complete_same_surface_edge_table_sha256"]
        ),
        "effective_mode_edge_table_sha256": str(
            edge_hashes["effective_mode_edge_table_sha256"]
        ),
        "inference_semantic_sha256": str(row["inference_semantic_sha256"]),
        "peak_rss_bytes": int(row["peak_rss_bytes"]),
        "wall_seconds": float(row["wall_seconds"]),
        "input_json_bytes": int(row["input_json_bytes"]),
        "serialized_output_bytes": int(row["serialized_output_bytes"]),
    }
    return {**record, "resource_profile_sha256": _sha_json(record)}


def _row_counts(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    counters = row.get("case_counters")
    if not isinstance(counters, Mapping):
        raise ValueError("ordinary batch row lacks structural counters")
    working_raw = counters.get("inherited_working_point_count")
    modes_raw = counters.get("input_mode_count")
    same_raw = counters.get("input_same_surface_edge_count")
    cross_raw = counters.get("input_cross_mode_edge_count")
    working = -1 if working_raw is None else int(working_raw)
    modes = -1 if modes_raw is None else int(modes_raw)
    edges = (
        -1
        if same_raw is None or cross_raw is None
        else int(same_raw) + int(cross_raw)
    )
    complete = _batch_complete_edge_count(row)
    if min(working, modes, edges, complete) < 0:
        raise ValueError("ordinary batch structural counters are invalid")
    return working, modes, edges, complete


def validate_replay_attestation(
    attestation: Mapping[str, Any], *, batch_manifest: Mapping[str, Any]
) -> None:
    if set(attestation) != REPLAY_TOP_FIELDS:
        raise ValueError("replay attestation top-level schema mismatch")
    if (
        attestation.get("schema_version") != REPLAY_SCHEMA_VERSION
        or attestation.get("kind") != REPLAY_KIND
        or attestation.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("replay attestation identity mismatch")
    if attestation.get("research_only") is not True or attestation.get("inference_safe") is not True:
        raise ValueError("replay attestation safety flags are invalid")
    for field in (
        "probe_manifest_sha256",
        "inference_manifest_sha256",
        "candidate_pool_manifest_sha256",
        "frozen_control_manifest_sha256",
    ):
        _require_sha(attestation.get(field), label=f"replay {field}")
    if (
        batch_manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or batch_manifest.get("kind") != BATCH_KIND
        or batch_manifest.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("ordinary batch identity mismatch")
    rows = _rows_by_token(batch_manifest, label="ordinary edge-probe batch")
    selected = select_designated_cases(batch_manifest)
    cases = _rows_by_token(
        attestation, label="edge-probe replay attestation", exact_gold37=False
    )
    if set(cases) != set(selected) or int(attestation.get("case_count") or -1) != 4:
        raise ValueError("replay attestation requires the exact four designated cases")
    if int(attestation.get("total_isolated_run_count") or -1) != 12:
        raise ValueError("replay attestation requires exactly twelve isolated runs")
    for token in sorted(selected):
        case = cases[token]
        if set(case) != REPLAY_CASE_FIELDS:
            raise ValueError(f"{token}: replay case schema mismatch")
        if case.get("selection_reason") != selected[token]:
            raise ValueError(f"{token}: replay selection reason mismatch")
        working, modes, edges, complete = _row_counts(rows[token])
        for field, expected in (
            ("working_point_count", working),
            ("transition_mode_count", modes),
            ("transition_mode_edge_count", edges),
            ("complete_same_surface_edge_count", complete),
        ):
            raw_value = case.get(field)
            if raw_value is None or int(raw_value) != expected:
                raise ValueError(f"{token}: replay {field} differs from ordinary batch")
        batch_edge_hashes = rows[token].get("replacement_edge_table_hashes")
        if not isinstance(batch_edge_hashes, Mapping):
            raise ValueError(f"{token}: ordinary batch lacks edge-table hashes")
        expected_hashes = {
            "batch_replacement_probe_sha256": rows[token]["replacement_probe_sha256"],
            "batch_frozen_control_probe_sha256": rows[token]["frozen_control_probe_sha256"],
            "batch_complete_same_surface_edge_table_sha256": batch_edge_hashes[
                "complete_same_surface_edge_table_sha256"
            ],
            "batch_effective_mode_edge_table_sha256": batch_edge_hashes[
                "effective_mode_edge_table_sha256"
            ],
            "batch_inference_semantic_sha256": rows[token]["inference_semantic_sha256"],
        }
        for field, value in expected_hashes.items():
            if _require_sha(case.get(field), label=f"{token} {field}") != _require_sha(
                value, label=f"{token} ordinary {field}"
            ):
                raise ValueError(f"{token}: replay {field} differs from ordinary batch")
        runs_raw = case.get("runs")
        if not isinstance(runs_raw, Sequence) or isinstance(runs_raw, (str, bytes)):
            raise ValueError(f"{token}: replay runs must be a sequence")
        runs = [dict(run) for run in runs_raw if isinstance(run, Mapping)]
        if len(runs) != 3 or int(case.get("replay_count") or -1) != 3:
            raise ValueError(f"{token}: replay requires exactly three runs")
        if [int(run.get("run_index") or -1) for run in runs] != [1, 2, 3]:
            raise ValueError(f"{token}: replay run indices must be exactly 1,2,3")
        replacement_hashes = [
            _require_sha(value, label=f"{token} replay replacement hash")
            for value in case.get("replay_replacement_probe_sha256s") or ()
        ]
        if len(replacement_hashes) != 3 or len(set(replacement_hashes)) != 1:
            raise ValueError(f"{token}: replay replacement hashes are not byte-identical")
        if replacement_hashes[0] != expected_hashes["batch_replacement_probe_sha256"]:
            raise ValueError(f"{token}: replay replacement hash differs from ordinary batch")
        comparison_fields = {
            "replacement_probe_sha256": "batch_replacement_probe_sha256",
            "frozen_control_probe_sha256": "batch_frozen_control_probe_sha256",
            "complete_same_surface_edge_table_sha256": "batch_complete_same_surface_edge_table_sha256",
            "effective_mode_edge_table_sha256": "batch_effective_mode_edge_table_sha256",
            "inference_semantic_sha256": "batch_inference_semantic_sha256",
        }
        for run, replacement_hash in zip(runs, replacement_hashes):
            if set(run) != REPLAY_RUN_FIELDS:
                raise ValueError(f"{token}: replay run schema mismatch")
            if run["replacement_probe_sha256"] != replacement_hash:
                raise ValueError(f"{token}: run replacement hash mismatch")
            for run_field, case_field in comparison_fields.items():
                if _require_sha(run.get(run_field), label=f"{token} run {run_field}") != _require_sha(
                    case.get(case_field), label=f"{token} batch {run_field}"
                ):
                    raise ValueError(f"{token}: replay {run_field} differs from ordinary batch")
            resource = {
                field: run[field]
                for field in REPLAY_RUN_FIELDS
                if field != "resource_profile_sha256"
            }
            if _require_sha(
                run.get("resource_profile_sha256"), label=f"{token} resource profile"
            ) != _sha_json(resource):
                raise ValueError(f"{token}: replay resource profile hash mismatch")
            if int(run["peak_rss_bytes"]) < 0 or int(run["peak_rss_bytes"]) >= PEAK_RSS_LIMIT_BYTES:
                raise ValueError(f"{token}: replay RSS reaches or exceeds 16 GiB")
            if float(run["wall_seconds"]) < 0.0:
                raise ValueError(f"{token}: replay wall time is invalid")
        if int(case.get("maximum_peak_rss_bytes") or -1) != max(
            int(run["peak_rss_bytes"]) for run in runs
        ):
            raise ValueError(f"{token}: maximum replay RSS mismatch")
        for field in ("input_json_bytes", "serialized_output_bytes"):
            if len({int(run[field]) for run in runs}) != 1:
                raise ValueError(f"{token}: replay {field} values differ")
            if int(runs[0][field]) != int(rows[token][field]):
                raise ValueError(f"{token}: replay {field} differs from ordinary batch")


def run_replays(
    *,
    probe_manifest_path: Path,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    frozen_control_manifest_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 12:
        raise ValueError("replay workers must be in [1,12]")
    probe_manifest_path = probe_manifest_path.resolve()
    inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve()
    a0_prediction_root = a0_prediction_root.resolve()
    frozen_control_manifest_path = frozen_control_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if sha256_file(inference_manifest_path) != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("replay inference manifest differs from frozen Gold37")
    if sha256_file(candidate_pool_manifest_path) != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("replay candidate pool differs from frozen V2")
    control_manifest_sha = sha256_file(frozen_control_manifest_path)
    control_rows = _validate_frozen_control_manifest(
        _read(frozen_control_manifest_path), actual_sha256=control_manifest_sha
    )
    probe_manifest = _read(probe_manifest_path)
    if (
        probe_manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or probe_manifest.get("kind") != BATCH_KIND
        or probe_manifest.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("replays require the complete-edge ordinary batch")
    if probe_manifest.get("inference_manifest_sha256") != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("ordinary batch references a different inference manifest")
    if probe_manifest.get("candidate_pool_manifest_sha256") != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("ordinary batch references a different candidate pool")
    if probe_manifest.get("frozen_control_manifest_sha256") != FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256:
        raise ValueError("ordinary batch references a different frozen control")
    batch_rows = _rows_by_token(probe_manifest, label="ordinary edge-probe batch")
    for token, row in batch_rows.items():
        if set(row) != BATCH_ROW_FIELDS or row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary batch row schema/status mismatch")
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference manifest")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), label="candidate-pool manifest")
    selection = select_designated_cases(probe_manifest)
    tasks: list[tuple[str, str, str, str, str, str]] = []
    task_meta: list[tuple[str, int]] = []
    for token in sorted(selection):
        scan = Path(str(inference_rows[token]["source_scan_path"])).resolve()
        pool = Path(str(pool_rows[token]["path"])).resolve()
        prediction = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        for label, path in (("scan", scan), ("candidate pool", pool), ("A0 prediction", prediction)):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        for run_index in (1, 2, 3):
            tasks.append(
                (
                    token,
                    str(scan),
                    str(pool),
                    str(prediction),
                    str(control_rows[token]["probe_sha256"]),
                    str(output_dir / token / f"run_{run_index}.json"),
                )
            )
            task_meta.append((token, run_index))
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        raw_rows = list(executor.map(_run_case, tasks))
    by_token: dict[str, list[dict[str, Any]]] = {token: [] for token in selection}
    for raw, (token, run_index) in zip(raw_rows, task_meta):
        by_token[token].append(_resource_record(raw, run_index=run_index))
    cases = []
    for token in sorted(selection):
        ordinary = batch_rows[token]
        runs = sorted(by_token[token], key=lambda row: int(row["run_index"]))
        working, modes, edges, complete = _row_counts(ordinary)
        edge_hashes = ordinary["replacement_edge_table_hashes"]
        cases.append(
            {
                "case_token": token,
                "selection_reason": selection[token],
                "working_point_count": working,
                "transition_mode_count": modes,
                "transition_mode_edge_count": edges,
                "complete_same_surface_edge_count": complete,
                "batch_replacement_probe_sha256": ordinary["replacement_probe_sha256"],
                "batch_frozen_control_probe_sha256": ordinary["frozen_control_probe_sha256"],
                "batch_complete_same_surface_edge_table_sha256": edge_hashes[
                    "complete_same_surface_edge_table_sha256"
                ],
                "batch_effective_mode_edge_table_sha256": edge_hashes[
                    "effective_mode_edge_table_sha256"
                ],
                "batch_inference_semantic_sha256": ordinary["inference_semantic_sha256"],
                "replay_count": 3,
                "replay_replacement_probe_sha256s": [
                    run["replacement_probe_sha256"] for run in runs
                ],
                "maximum_peak_rss_bytes": max(
                    int(run["peak_rss_bytes"]) for run in runs
                ),
                "runs": runs,
            }
        )
    attestation = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "kind": REPLAY_KIND,
        "profile": PRIMITIVE_PROFILE,
        "research_only": True,
        "inference_safe": True,
        "case_count": 4,
        "total_isolated_run_count": 12,
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "candidate_pool_manifest_sha256": sha256_file(candidate_pool_manifest_path),
        "frozen_control_manifest_sha256": control_manifest_sha,
        "cases": cases,
    }
    validate_replay_attestation(attestation, batch_manifest=probe_manifest)
    path = output_dir / "replay_attestation.json"
    path.write_bytes(canonical_replay_attestation_bytes(attestation))
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attestation = run_replays(
        probe_manifest_path=args.probe_manifest,
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    path = args.output_dir.resolve() / "replay_attestation.json"
    print(
        json.dumps(
            {
                "output": str(path),
                "replay_attestation_sha256": sha256_file(path),
                "isolated_run_count": attestation["total_isolated_run_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
