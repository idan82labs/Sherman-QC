#!/usr/bin/env python3
"""Run the three isolated 1,688,000-edge synthetic resource envelopes."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_synthetic_worst_cap_resource_v1"
PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1"
RESOURCE_PROFILE = "V2_CROSS_PARENT_RUNG_ABSTRACT_INDEX_CAP_ENVELOPE_V1"
PARENT_COUNT = 4_000
MODE_COUNT = 16_000
MODES_PER_PARENT = 4
ADJACENT_PARENT_PAIR_COUNT = 52_000
CANDIDATE_EVALUATION_COUNT = 832_000
BASELINE_COMPLETE_EDGE_COUNT = 832_000
INHERITED_LOCAL_RUNG_COUNT = 24_000
ADDED_CROSS_PARENT_RUNG_COUNT = 832_000
EFFECTIVE_EDGE_COUNT = 1_688_000
COMPONENT_INDEX_BUCKET_COUNT = 65_536
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
TOP_FIELDS = frozenset(
    {
        "schema_version", "kind", "profile", "resource_profile",
        "resource_envelope_only", "physical_voxel_realizable", "resource_limits",
        "runs", "semantic_hashes_identical", "all_runs_below_16_gib",
        "attestation_sha256",
    }
)
RUN_FIELDS = frozenset(
    {
        "run_index", "status", "parent_count", "mode_count", "modes_per_parent",
        "adjacent_parent_pair_count", "cross_parent_rung_candidate_evaluation_count",
        "baseline_complete_same_surface_edge_count", "inherited_cross_mode_local_edge_count",
        "added_cross_parent_rung_count", "effective_mode_edge_count",
        "candidate_index_entry_count", "component_edge_index_bucket_count",
        "component_edge_index_entry_count", "baseline_complete_edge_table_sha256",
        "inherited_local_rung_table_sha256", "added_cross_parent_rung_table_sha256",
        "effective_mode_edge_semantic_sha256", "candidate_index_sha256",
        "component_edge_index_sha256", "canonical_resource_semantic_sha256",
        "wall_seconds", "peak_rss_bytes",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_resource_attestation_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _record_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return hashlib.sha256(_canonical_bytes({key: item for key, item in value.items() if key != hash_field})).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _run_synthetic(run_index: int) -> dict[str, Any]:
    started = perf_counter()
    # Full candidate index: every one of the 832k evaluations receives one
    # canonical packed synthetic mode pair.
    candidates = array("Q")
    candidate_hash = hashlib.sha256()
    for index in range(CANDIDATE_EVALUATION_COUNT):
        packed = ((index % MODE_COUNT) << 32) | ((index * 17 + 1) % MODE_COUNT)
        candidates.append(packed)
        candidate_hash.update(struct.pack(">Q", packed))

    # Full component-edge index: a compact linked-list index covers every
    # effective edge without allocating 1.688m Python dicts.
    heads = array("i", [-1]) * COMPONENT_INDEX_BUCKET_COUNT
    next_edges = array("i", [-1]) * EFFECTIVE_EDGE_COUNT
    edge_components = array("I", [0]) * EFFECTIVE_EDGE_COUNT
    component_hash = hashlib.sha256()
    baseline_hash = hashlib.sha256()
    inherited_hash = hashlib.sha256()
    added_hash = hashlib.sha256()
    effective_hash = hashlib.sha256()
    for edge_index in range(EFFECTIVE_EDGE_COUNT):
        component = edge_index % COMPONENT_INDEX_BUCKET_COUNT
        edge_components[edge_index] = component
        next_edges[edge_index] = heads[component]
        heads[component] = edge_index
        component_hash.update(struct.pack(">II", edge_index, component))
        if edge_index < BASELINE_COMPLETE_EDGE_COUNT:
            kind = 0
            local_index = edge_index
            baseline_hash.update(struct.pack(">Q", local_index))
        elif edge_index < BASELINE_COMPLETE_EDGE_COUNT + INHERITED_LOCAL_RUNG_COUNT:
            kind = 1
            local_index = edge_index - BASELINE_COMPLETE_EDGE_COUNT
            inherited_hash.update(struct.pack(">Q", local_index))
        else:
            kind = 2
            local_index = edge_index - BASELINE_COMPLETE_EDGE_COUNT - INHERITED_LOCAL_RUNG_COUNT
            added_hash.update(struct.pack(">Q", local_index))
        effective_hash.update(struct.pack(">BQI", kind, local_index, component))

    semantic_record = {
        "resource_profile": RESOURCE_PROFILE,
        "parent_count": PARENT_COUNT,
        "mode_count": MODE_COUNT,
        "candidate_evaluation_count": len(candidates),
        "effective_edge_count": len(edge_components),
        "component_index_bucket_count": len(heads),
        "baseline_complete_edge_table_sha256": baseline_hash.hexdigest(),
        "inherited_local_rung_table_sha256": inherited_hash.hexdigest(),
        "added_cross_parent_rung_table_sha256": added_hash.hexdigest(),
        "effective_mode_edge_semantic_sha256": effective_hash.hexdigest(),
        "candidate_index_sha256": candidate_hash.hexdigest(),
        "component_edge_index_sha256": component_hash.hexdigest(),
    }
    peak = _peak_rss_bytes()
    if peak >= PEAK_RSS_LIMIT_BYTES:
        raise MemoryError(f"synthetic RSS cap breached: {peak}")
    row = {
        "run_index": run_index,
        "status": "SUCCESS",
        "parent_count": PARENT_COUNT,
        "mode_count": MODE_COUNT,
        "modes_per_parent": MODES_PER_PARENT,
        "adjacent_parent_pair_count": ADJACENT_PARENT_PAIR_COUNT,
        "cross_parent_rung_candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
        "baseline_complete_same_surface_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
        "inherited_cross_mode_local_edge_count": INHERITED_LOCAL_RUNG_COUNT,
        "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
        "effective_mode_edge_count": EFFECTIVE_EDGE_COUNT,
        "candidate_index_entry_count": len(candidates),
        "component_edge_index_bucket_count": len(heads),
        "component_edge_index_entry_count": len(edge_components),
        **{key: semantic_record[key] for key in (
            "baseline_complete_edge_table_sha256", "inherited_local_rung_table_sha256",
            "added_cross_parent_rung_table_sha256", "effective_mode_edge_semantic_sha256",
            "candidate_index_sha256", "component_edge_index_sha256",
        )},
        "canonical_resource_semantic_sha256": hashlib.sha256(_canonical_bytes(semantic_record)).hexdigest(),
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak,
    }
    if set(row) != RUN_FIELDS:
        raise AssertionError("synthetic resource run schema drift")
    return row


def validate_synthetic_resource_attestation(attestation: Mapping[str, Any]) -> int:
    if set(attestation) != TOP_FIELDS:
        raise ValueError("synthetic resource top-level schema mismatch")
    if attestation.get("schema_version") != SCHEMA_VERSION or attestation.get("kind") != KIND or attestation.get("profile") != PROFILE or attestation.get("resource_profile") != RESOURCE_PROFILE:
        raise ValueError("synthetic resource identity mismatch")
    if attestation.get("resource_envelope_only") is not True or attestation.get("physical_voxel_realizable") is not False:
        raise ValueError("synthetic resource must identify the non-geometric abstract envelope")
    limits = attestation.get("resource_limits")
    expected_limits = {
        "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
        "parent_count": PARENT_COUNT, "mode_count": MODE_COUNT,
        "modes_per_parent": MODES_PER_PARENT,
        "adjacent_parent_pair_count": ADJACENT_PARENT_PAIR_COUNT,
        "candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
        "baseline_complete_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
        "inherited_local_rung_count": INHERITED_LOCAL_RUNG_COUNT,
        "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
        "effective_edge_count": EFFECTIVE_EDGE_COUNT,
        "component_index_bucket_count": COMPONENT_INDEX_BUCKET_COUNT,
    }
    if limits != expected_limits:
        raise ValueError("synthetic resource limits mismatch")
    runs_raw = attestation.get("runs")
    if not isinstance(runs_raw, Sequence) or isinstance(runs_raw, (str, bytes)):
        raise ValueError("synthetic runs must be a sequence")
    runs = [dict(run) for run in runs_raw if isinstance(run, Mapping)]
    if len(runs) != 3 or [run.get("run_index") for run in runs] != [1, 2, 3]:
        raise ValueError("synthetic resource requires exact isolated runs 1,2,3")
    semantic_hashes = set()
    maximum_rss = 0
    expected_counts = {
        "parent_count": PARENT_COUNT, "mode_count": MODE_COUNT,
        "modes_per_parent": MODES_PER_PARENT,
        "adjacent_parent_pair_count": ADJACENT_PARENT_PAIR_COUNT,
        "cross_parent_rung_candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
        "baseline_complete_same_surface_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
        "inherited_cross_mode_local_edge_count": INHERITED_LOCAL_RUNG_COUNT,
        "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
        "effective_mode_edge_count": EFFECTIVE_EDGE_COUNT,
        "candidate_index_entry_count": CANDIDATE_EVALUATION_COUNT,
        "component_edge_index_bucket_count": COMPONENT_INDEX_BUCKET_COUNT,
        "component_edge_index_entry_count": EFFECTIVE_EDGE_COUNT,
    }
    for run in runs:
        if set(run) != RUN_FIELDS or run.get("status") != "SUCCESS":
            raise ValueError("synthetic run schema/status mismatch")
        for field, expected in expected_counts.items():
            if int(run.get(field) or -1) != expected:
                raise ValueError(f"synthetic run {field} mismatch")
        for field in RUN_FIELDS:
            if field.endswith("_sha256") and len(str(run.get(field) or "")) != 64:
                raise ValueError(f"synthetic run {field} hash invalid")
        if float(run["wall_seconds"]) < 0:
            raise ValueError("synthetic wall time invalid")
        rss = int(run["peak_rss_bytes"])
        if rss < 0 or rss >= PEAK_RSS_LIMIT_BYTES:
            raise ValueError("synthetic RSS reaches/exceeds 16 GiB")
        maximum_rss = max(maximum_rss, rss)
        semantic_hashes.add(run["canonical_resource_semantic_sha256"])
    if len(semantic_hashes) != 1 or attestation.get("semantic_hashes_identical") is not True:
        raise ValueError("synthetic semantic hashes differ")
    if attestation.get("all_runs_below_16_gib") is not True:
        raise ValueError("synthetic RSS attestation false")
    if attestation.get("attestation_sha256") != _record_hash(attestation, "attestation_sha256"):
        raise ValueError("synthetic attestation hash mismatch")
    return maximum_rss


def run_synthetic_resource(*, output_path: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 3:
        raise ValueError("synthetic workers must be in [1,3]")
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        runs = list(executor.map(_run_synthetic, (1, 2, 3)))
    runs.sort(key=lambda run: int(run["run_index"]))
    attestation = {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "profile": PROFILE,
        "resource_profile": RESOURCE_PROFILE, "resource_envelope_only": True,
        "physical_voxel_realizable": False,
        "resource_limits": {
            "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
            "parent_count": PARENT_COUNT, "mode_count": MODE_COUNT,
            "modes_per_parent": MODES_PER_PARENT,
            "adjacent_parent_pair_count": ADJACENT_PARENT_PAIR_COUNT,
            "candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
            "baseline_complete_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
            "inherited_local_rung_count": INHERITED_LOCAL_RUNG_COUNT,
            "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
            "effective_edge_count": EFFECTIVE_EDGE_COUNT,
            "component_index_bucket_count": COMPONENT_INDEX_BUCKET_COUNT,
        },
        "runs": runs,
        "semantic_hashes_identical": len({run["canonical_resource_semantic_sha256"] for run in runs}) == 1,
        "all_runs_below_16_gib": all(run["peak_rss_bytes"] < PEAK_RSS_LIMIT_BYTES for run in runs),
    }
    attestation["attestation_sha256"] = _record_hash(attestation, "attestation_sha256")
    validate_synthetic_resource_attestation(attestation)
    output_path = output_path.resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_resource_attestation_bytes(attestation))
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_synthetic_resource(output_path=args.output, workers=args.workers)
    print(json.dumps({"output": str(args.output.resolve()), "attestation_sha256": result["attestation_sha256"], "maximum_peak_rss_bytes": max(run["peak_rss_bytes"] for run in result["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
