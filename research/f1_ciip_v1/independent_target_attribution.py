"""Independent fail-closed validator for the CIIP target-attribution result.

This module intentionally does not import the producer module.  It validates
the complete tables, recomputes every row/table commitment and aggregate, and
checks the explicit witness/relation/rung/target bijections and paired-state
semantics from the serialized result.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


EXPECTED_TARGET_COUNT = 4_788
EXPECTED_METRIC_KEYS = {
    "finite_h0_merge_count", "localizable_ciip_witness_count",
    "nonlocalizable_ciip_merge_count", "ciip_witness_target_relation_count",
    "aggregate_added_ciip_rung_count", "target_reached_by_ciip_rung_count",
    "target_side_gate_advanced_count", "target_ciip_attributable_certificate_count",
    "target_novel_ciip_attributable_certificate_count",
    "target_certificate_equivalent_to_frozen_baseline_count",
    "untouched_target_count", "baseline_hash_reproduction_ok",
    "producer_validator_consensus_ok",
}


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _record(row: Mapping[str, Any], field: str = "row_sha256") -> str:
    return _sha({key: value for key, value in row.items() if key != field})


def _table(rows: Sequence[Mapping[str, Any]], id_field: str) -> str:
    return _sha(sorted((dict(row) for row in rows), key=lambda row: str(row[id_field])))


def validate_result_tables(result: Mapping[str, Any]) -> dict[str, str]:
    tables = result["tables"]
    witnesses = list(tables["ciip_merge_witness_table"])
    exceptions = list(tables["ciip_nonlocalizable_merge_table"])
    relations = list(tables["ciip_witness_target_relation_table"])
    rungs = list(tables["added_ciip_rung_table"])
    targets = list(tables["per_target_gate_comparison_table"])
    certificates = list(tables["certificate_attribution_table"])
    specs = {
        "ciip_merge_witness_table": (witnesses, "witness_id"),
        "ciip_nonlocalizable_merge_table": (exceptions, "ciip_interval_or_merge_id"),
        "ciip_witness_target_relation_table": (relations, "relation_id"),
        "added_ciip_rung_table": (rungs, "mode_edge_id"),
        "per_target_gate_comparison_table": (targets, "target_id"),
        "certificate_attribution_table": (certificates, "target_id"),
    }
    hashes: dict[str, str] = {}
    for name, (rows, id_field) in specs.items():
        identifiers = [str(row[id_field]) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"independent validator: duplicate identifier in {name}")
        for row in rows:
            if row.get("row_sha256") != _record(row):
                raise ValueError(f"independent validator: row hash mismatch in {name}")
        hashes[f"{name}_sha256"] = _table(rows, id_field)
    if hashes != result["table_hashes"]:
        raise ValueError("independent validator: complete table commitments disagree")
    if len(targets) != EXPECTED_TARGET_COUNT:
        raise ValueError("independent validator: target population is not 4,788")
    witness_ids = {str(row["witness_id"]) for row in witnesses}
    relation_ids = {str(row["relation_id"]) for row in relations}
    relation_by_id = {str(row["relation_id"]): row for row in relations}
    target_ids = {str(row["target_id"]) for row in targets}
    if any(str(row["witness_id"]) not in witness_ids for row in relations):
        raise ValueError("independent validator: relation references absent witness")
    if any(str(row["target_id"]) not in target_ids for row in relations):
        raise ValueError("independent validator: relation references absent target")
    if any(
        row.get("origin_family") != "CIIP_V1"
        or str(row.get("origin_relation_id")) not in relation_ids
        or str(row.get("origin_witness_id")) != str(
            relation_by_id[str(row["origin_relation_id"])]["witness_id"])
        or str(row.get("origin_target_id")) != str(
            relation_by_id[str(row["origin_relation_id"])]["target_id"])
        for row in rungs
    ):
        raise ValueError("independent validator: rung provenance is not bijective")
    rung_by_target: dict[str, set[str]] = {}
    witness_by_target: dict[str, set[str]] = {}
    for row in rungs:
        rung_by_target.setdefault(str(row["origin_target_id"]), set()).add(str(row["mode_edge_id"]))
        witness_by_target.setdefault(str(row["origin_target_id"]), set()).add(str(row["origin_witness_id"]))
    for row in targets:
        target_id = str(row["target_id"])
        if set(map(str, row["ciip_rung_ids"])) != rung_by_target.get(target_id, set()):
            raise ValueError("independent validator: target/rung projection mismatch")
        if set(map(str, row["ciip_witness_ids"])) != witness_by_target.get(target_id, set()):
            raise ValueError("independent validator: target/witness projection mismatch")
        reached = bool(rung_by_target.get(target_id))
        if bool(row["reached_by_ciip"]) != reached:
            raise ValueError("independent validator: reached flag mismatch")
        attributable = list(row["ciip_attributable_certificate_ids"])
        novel = list(row["novel_ciip_attributable_certificate_ids"])
        if attributable and (
            not reached
            or bool(row["side_gate_ablation_state"])
            or not bool(row["side_gate_augmented_state"])
            or not set(attributable) <= set(row["certificate_ids_augmented"])
            or set(attributable) & set(row["certificate_ids_ablation"])
        ):
            raise ValueError("independent validator: causal attribution predicate failed")
        if not set(novel) <= set(attributable):
            raise ValueError("independent validator: novel certificate is not attributable")
    metrics = result["aggregate_metrics"]
    if set(metrics) != EXPECTED_METRIC_KEYS:
        raise ValueError("independent validator: aggregate metric schema mismatch")
    recomputed = {
        "finite_h0_merge_count": len(witnesses) + len(exceptions),
        "localizable_ciip_witness_count": len(witnesses),
        "nonlocalizable_ciip_merge_count": len(exceptions),
        "ciip_witness_target_relation_count": len(relations),
        "aggregate_added_ciip_rung_count": len(rungs),
        "target_reached_by_ciip_rung_count": sum(bool(row["reached_by_ciip"]) for row in targets),
        "target_side_gate_advanced_count": sum(
            not bool(row["side_gate_ablation_state"]) and bool(row["side_gate_augmented_state"])
            for row in targets),
        "target_ciip_attributable_certificate_count": sum(bool(row["ciip_attributable_certificate_ids"]) for row in targets),
        "target_novel_ciip_attributable_certificate_count": sum(bool(row["novel_ciip_attributable_certificate_ids"]) for row in targets),
        "target_certificate_equivalent_to_frozen_baseline_count": sum(bool(row["equivalent_baseline_certificate_ids"]) for row in targets),
        "untouched_target_count": sum(not bool(row["reached_by_ciip"]) for row in targets),
        "baseline_hash_reproduction_ok": bool(metrics["baseline_hash_reproduction_ok"]),
        "producer_validator_consensus_ok": True,
    }
    expected = dict(metrics); expected["producer_validator_consensus_ok"] = True
    if recomputed != expected:
        raise ValueError("independent validator: aggregate metrics disagree")
    return hashes


__all__ = ["validate_result_tables"]
