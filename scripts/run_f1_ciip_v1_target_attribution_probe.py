#!/usr/bin/env python3
"""Execute the one-shot CIIP-V1 frozen-target attribution tranche."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import f1_v2_complete_same_surface_edge_probe as complete_core  # noqa: E402
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1_core  # noqa: E402
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as v2_core  # noqa: E402
from research.f1_ciip_v1 import container, primary_parser  # noqa: E402
from research.f1_ciip_v1 import target_attribution as core  # noqa: E402
from research.f1_ciip_v1.independent_target_attribution import validate_result_tables  # noqa: E402


WALL_LIMIT_SECONDS = 30 * 60
RSS_LIMIT_BYTES = 2 * 1024**3
OUTPUT_LIMIT_BYTES = 1024**3


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _verify_lock(lock_path: Path, claim_path: Path) -> tuple[dict[str, Any], str]:
    lock = _read(lock_path)
    expected = str(lock.get("execution_lock_sha256") or "")
    actual = core.sha_json({key: value for key, value in lock.items() if key != "execution_lock_sha256"})
    if not expected or expected != actual:
        raise ValueError("execution lock self-commitment mismatch")
    claim = _read(claim_path)
    if claim != {
        "claim_state": "READY_FOR_FIRST_AND_ONLY_EXECUTION",
        "execution_lock_sha256": expected,
        "one_shot": True,
        "rerun_authorized": False,
    }:
        raise ValueError("one-shot claim is absent, changed, or already consumed")
    consumed = dict(claim); consumed["claim_state"] = "CONSUMED_BEFORE_SOURCE_OPEN"
    _atomic_write(claim_path, core.canonical_json_bytes(consumed) + b"\n")
    return lock, actual


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw); grouped.setdefault(str(row["case_token"]), []).append(row)
    return grouped


def execute(*, lock_path: Path, claim_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("authoritative output already exists; rerun is forbidden")
    started = perf_counter()
    lock, lock_sha = _verify_lock(lock_path.resolve(), claim_path.resolve())
    paths = {name: (ROOT / relative).resolve() for name, relative in lock["inputs"]["paths"].items()}
    observed_hashes = {name: _sha_file(path) for name, path in paths.items() if name != "ordinary_case_directory"}
    if observed_hashes != lock["inputs"]["sha256"]:
        raise ValueError("immutable input byte hash mismatch")
    v4_result = _read(paths["v4_result"])
    if str(v4_result.get("execution_lock_sha256")) != lock["bindings"]["v4_execution_lock_sha256"]:
        raise ValueError("V4 result execution-lock binding mismatch")
    decoded = container.decode_container(paths["v4_container"].read_bytes(), materialize_rows=False)
    if decoded.metadata["raw_source_sha256"] != lock["bindings"]["source_sha256"]:
        raise ValueError("V4 container source binding mismatch")
    source_fd = os.open(paths["raw_source"], os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(source_fd)
    finally:
        os.close(source_fd)
    if parsed.diagnostics.raw_ply_sha256 != lock["bindings"]["source_sha256"]:
        raise ValueError("raw source binding mismatch")
    raw_pool = _read(paths["transition_pool_c006"])
    projected = v2_core.project_mapping_preflight_input(raw_pool)
    frozen_control = _read(paths["frozen_control_c006"])
    frozen_baseline = _read(paths["complete_baseline_c006"])
    mapping = _read(paths["mapping_preflight"])
    mapping_cases = {str(row["case_token"]): row for row in mapping["cases"]}
    c006_summary = mapping_cases["C006"]
    if core.sha_json(frozen_control) != str(c006_summary["frozen_control_probe_sha256"]):
        raise ValueError("C006 frozen control hash mismatch")
    if core.sha_json(frozen_baseline) != str(c006_summary["complete_edge_baseline_probe_sha256"]):
        raise ValueError("C006 complete baseline hash mismatch")
    regenerated_control = complete_core.regenerate_frozen_control_probe(
        deepcopy(projected), scan_sha256=str(c006_summary["scan_sha256"]),
        candidate_pool_sha256=str(c006_summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(c006_summary["a0_prediction_sha256"]),
    )
    regenerated_baseline = complete_core.build_v2_complete_same_surface_edge_probe(
        deepcopy(projected), scan_sha256=str(c006_summary["scan_sha256"]),
        candidate_pool_sha256=str(c006_summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(c006_summary["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(c006_summary["frozen_control_probe_sha256"]),
    )
    baseline_reproduced = (
        core.canonical_json_bytes(regenerated_control) == core.canonical_json_bytes(frozen_control)
        and core.canonical_json_bytes(regenerated_baseline) == core.canonical_json_bytes(frozen_baseline)
    )
    if not baseline_reproduced:
        raise ValueError("frozen C006 baseline/control failed reproduction")
    c006_mapping = v2_core._build_case_from_payloads(
        deepcopy(projected), case_token="C006",
        scan_sha256=str(c006_summary["scan_sha256"]),
        candidate_pool_sha256=str(c006_summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(c006_summary["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(c006_summary["frozen_control_probe_sha256"]),
        complete_edge_baseline_probe_sha256=str(c006_summary["complete_edge_baseline_probe_sha256"]),
        frozen_control=regenerated_control, complete_edge_baseline=regenerated_baseline,
    )
    all_relations = list(mapping["frozen_target_baseline_relation_table"])
    if len(all_relations) != core.EXPECTED_TARGET_COUNT:
        raise ValueError("frozen target population is not exactly 4,788")
    grouped_relations = _group(all_relations)
    if c006_mapping["frozen_target_baseline_relation_table"] != grouped_relations["C006"]:
        raise ValueError("C006 frozen target relation table changed")
    baseline_case_payloads: dict[str, dict[str, Any]] = {}
    case_directory = paths["ordinary_case_directory"]
    for token in sorted(grouped_relations):
        payload = _read(case_directory / f"{token}.json")
        if payload["frozen_target_baseline_relation_table"] != grouped_relations[token]:
            raise ValueError(f"{token}: frozen relation table changed")
        baseline_case_payloads[token] = payload
    context = v1_core._baseline_context(projected, regenerated_baseline)
    pair_contexts = {
        pair_id: v1_core._pair_context(pair, context)
        for pair_id, pair in sorted(context["pair_by_id"].items())
    }
    witnesses, exceptions = core.build_merge_witnesses(
        field_rows=decoded.rows["cell_scale_field"],
        adjacency_rows=decoded.rows["space_scale_adjacency"],
        persistence_rows=decoded.rows["persistence_disposition"],
        source_xyz=parsed.xyz, source_sha256=lock["bindings"]["source_sha256"],
    )
    witness_relations, rungs = core.map_witnesses_to_targets(
        witnesses=witnesses, context=context, pair_contexts=pair_contexts,
        relations=c006_mapping["frozen_target_baseline_relation_table"],
        source_sha256=lock["bindings"]["source_sha256"],
    )
    target_rows, certificate_rows, replacement_pairs = core.evaluate_target_population(
        relations=all_relations, baseline_case_payloads=baseline_case_payloads,
        c006_mapping_payload=c006_mapping, c006_context=context,
        c006_pair_contexts=pair_contexts, c006_control=regenerated_control,
        c006_rungs=rungs, witness_relations=witness_relations,
    )
    tables = {
        "ciip_merge_witness_table": witnesses,
        "ciip_nonlocalizable_merge_table": exceptions,
        "ciip_witness_target_relation_table": witness_relations,
        "added_ciip_rung_table": rungs,
        "per_target_gate_comparison_table": target_rows,
        "certificate_attribution_table": certificate_rows,
    }
    specs = {
        "ciip_merge_witness_table": "witness_id",
        "ciip_nonlocalizable_merge_table": "ciip_interval_or_merge_id",
        "ciip_witness_target_relation_table": "relation_id",
        "added_ciip_rung_table": "mode_edge_id",
        "per_target_gate_comparison_table": "target_id",
        "certificate_attribution_table": "target_id",
    }
    table_hashes = {
        f"{name}_sha256": core.table_hash(tables[name], id_field)
        for name, id_field in specs.items()
    }
    metrics = core.aggregate_metrics(
        persistence_rows=decoded.rows["persistence_disposition"], witnesses=witnesses,
        exceptions=exceptions, witness_relations=witness_relations, rungs=rungs,
        target_rows=target_rows, baseline_hash_reproduction_ok=baseline_reproduced,
        producer_validator_consensus_ok=False,
    )
    result = {
        "schema_version": core.SCHEMA_VERSION,
        "kind": core.KIND,
        "construction_version": core.CONSTRUCTION_VERSION,
        "research_only": True,
        "prediction_mutated": False,
        "gold_consumed": False,
        "labels_consumed": False,
        "cad_consumed": False,
        "expected_counts_consumed": False,
        "benchmark_feedback_consumed": False,
        "production_runtime_main_consumed": False,
        "synthetic_rescue_consumed": False,
        "replacement_input_consumed": False,
        "persistence_cutoff_used": False,
        "ranking_or_top_k_used": False,
        "source_points_consumed_only_for_locked_grid_transform": True,
        "execution_lock_sha256": lock_sha,
        "source_sha256": lock["bindings"]["source_sha256"],
        "v4_result_sha256": lock["bindings"]["v4_result_sha256"],
        "v4_container_sha256": lock["bindings"]["v4_container_sha256"],
        "target_population_sha256": core.table_hash(all_relations, "frozen_target_component_id"),
        "target_count": len(all_relations),
        "baseline_pair_records_c006_sha256": core.table_hash(replacement_pairs, "local_plane_pair_id"),
        "tables": tables,
        "table_hashes": table_hashes,
        "aggregate_metrics": metrics,
        "independent_validator": None,
        "decision": "PENDING_INDEPENDENT_VALIDATION",
        "resource": {},
    }
    validator_hashes = validate_result_tables(result)
    if validator_hashes != table_hashes:
        raise ValueError("producer/validator table consensus failed")
    result["aggregate_metrics"]["producer_validator_consensus_ok"] = True
    result["independent_validator"] = {
        "implementation": "research.f1_ciip_v1.independent_target_attribution",
        "complete_table_hashes": validator_hashes,
        "byte_exact_consensus": True,
    }
    result["decision"] = core.decision_from_metrics(result["aggregate_metrics"])
    result["resource"] = {
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": _peak_rss_bytes(),
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "peak_rss_limit_bytes": RSS_LIMIT_BYTES,
        "output_limit_bytes": OUTPUT_LIMIT_BYTES,
    }
    validate_result_tables(result)
    encoded = core.canonical_json_bytes(result) + b"\n"
    result["resource"]["serialized_output_bytes"] = len(encoded)
    encoded = core.canonical_json_bytes(result) + b"\n"
    if result["resource"]["wall_seconds"] > WALL_LIMIT_SECONDS or result["resource"]["peak_rss_bytes"] > RSS_LIMIT_BYTES or len(encoded) > OUTPUT_LIMIT_BYTES:
        result["decision"] = "INVALID_CIIP_V1_ATTRIBUTION_TRANCHE_STOP"
        encoded = core.canonical_json_bytes(result) + b"\n"
    _atomic_write(output_path.resolve(), encoded)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute(lock_path=args.lock, claim_path=args.claim, output_path=args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": _sha_file(args.output.resolve()),
        "decision": result["decision"],
        "aggregate_metrics": result["aggregate_metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
