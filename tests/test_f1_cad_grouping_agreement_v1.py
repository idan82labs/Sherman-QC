from __future__ import annotations

import copy

import pytest

from research.f1_cad_grouping_agreement_v1 import CadGroupingInputError, score


def mapping(groups: list[list[str]], *, abstain: bool = False) -> dict:
    base = {
        "schema": "CAD_EVENT_CORRESPONDENCE_V1",
        "primary_truth_sha256": "a" * 64,
        "step_sha256": "b" * 64,
        "overlay_transform_sha256": "c" * 64,
        "overlay_status": "ABSTAIN" if abstain else "TRUSTED",
        "events": [],
        "cad_only_unobserved": [],
    }
    if abstain:
        base["reason"] = "OVERLAY_UNAVAILABLE_OR_UNTRUSTED"
        return base
    face = 1
    for index, group in enumerate(groups, 1):
        base["events"].append({
            "event_id": f"E{index:04d}",
            "step_face_indices": [face],
            "v9_primitive_ids": group,
            "relation": "MANY_TO_ONE" if len(group) > 1 else "ONE_TO_ONE",
            "mapping_status": "RESOLVED",
        })
        face += 1
    return base


def case(case_id: str, first: dict, second: dict) -> dict:
    native = sorted({
        primitive
        for mapping_value in (first, second)
        for event in mapping_value["events"]
        for primitive in event["v9_primitive_ids"]
    })
    return {
        "case_id": case_id,
        "mapper_a": first,
        "mapper_b": second,
        "native_v9_primitive_ids": native,
        "mapper_a_replay_identical": True,
        "mapper_b_replay_identical": True,
        "load_export_latency_seconds": [1.0, 1.1, 1.0, 1.1],
        "peak_rss_bytes": 100_000_000,
        "jobs_attempted": 4,
        "jobs_succeeded": 4,
    }


def qualification(cases: list[dict]) -> dict:
    return {
        "schema": "CIBEC_V1_CAD_GROUPING_QUALIFICATION_INPUT",
        "hash_manifest_verified": True,
        "leakage_audit_prohibited_field_or_asset_count": 0,
        "cases": cases,
    }


def test_identical_many_to_one_grouping_passes_with_explicit_abstention() -> None:
    first = mapping([["P1", "P2"], ["P3"]])
    abstain = mapping([], abstain=True)
    excluded = case("C13", abstain, copy.deepcopy(abstain))
    excluded["registration_abstention_excluded"] = True
    trusted = [case(f"C{index:02d}", first, copy.deepcopy(first)) for index in range(1, 13)]
    result = score(qualification([*trusted, excluded]))
    assert result["status"] == "PASS"
    assert result["metrics"]["grouping_precision"] == 1.0
    assert result["metrics"]["registration_abstention_count"] == 1


def test_split_group_fails_recall_and_merge_fails_precision() -> None:
    grouped = mapping([["P1", "P2", "P3"]])
    split = mapping([["P1", "P2"], ["P3"]])
    result = score(qualification([case("C01", grouped, split)]))
    assert result["metrics"]["grouping_precision"] == 1.0
    assert result["metrics"]["grouping_recall"] < 0.90
    assert result["status"] == "FAIL"
    result = score(qualification([case("C01", split, grouped)]))
    assert result["metrics"]["grouping_precision"] < 0.90
    assert result["status"] == "FAIL"


def test_operational_failures_remain_separate() -> None:
    first = mapping([["P1", "P2"]])
    row = case("C01", first, copy.deepcopy(first))
    row["mapper_b_replay_identical"] = False
    row["load_export_latency_seconds"] = [1.0, 1.0, 1.0, 30.01]
    row["jobs_succeeded"] = 3
    result = score(qualification([row]))
    assert result["gates"]["deterministic_replay"] is False
    assert result["gates"]["latency"] is False
    assert result["gates"]["job_success"] is False


def test_duplicate_primitive_ownership_is_rejected() -> None:
    bad = mapping([["P1", "P2"], ["P2", "P3"]])
    with pytest.raises(CadGroupingInputError, match="duplicate V9 primitive"):
        score(qualification([case("C01", bad, copy.deepcopy(bad))]))


def test_incomplete_denominator_cannot_pass() -> None:
    first = mapping([["P1", "P2"]])
    result = score(qualification([case("C01", first, copy.deepcopy(first))]))
    assert result["gates"]["complete_13_case_denominator"] is False
    assert result["status"] == "FAIL"


def test_native_inventory_prevents_silent_singleton_omission() -> None:
    first = mapping([["P1"], ["P2"]])
    row = case("C01", first, copy.deepcopy(first))
    row["native_v9_primitive_ids"] = ["P1", "P2", "P3"]
    result = score(qualification([row]))
    assert result["metrics"]["pair_universe_count"] == 3


def test_unknown_primitive_is_rejected_against_native_inventory() -> None:
    first = mapping([["P1", "P2"]])
    row = case("C01", first, copy.deepcopy(first))
    row["native_v9_primitive_ids"] = ["P1"]
    with pytest.raises(CadGroupingInputError, match="outside the native inventory"):
        score(qualification([row]))


def test_abstention_operations_and_global_integrity_are_still_gated() -> None:
    first = mapping([["P1", "P2"]])
    abstain = mapping([], abstain=True)
    trusted = [case(f"C{index:02d}", first, copy.deepcopy(first)) for index in range(1, 13)]
    excluded = case("C13", abstain, copy.deepcopy(abstain))
    excluded["registration_abstention_excluded"] = True
    excluded["mapper_b_replay_identical"] = False
    excluded["peak_rss_bytes"] = 2 * 1024**3
    payload = qualification([*trusted, excluded])
    payload["hash_manifest_verified"] = False
    payload["leakage_audit_prohibited_field_or_asset_count"] = 1
    result = score(payload)
    assert result["gates"]["deterministic_replay"] is False
    assert result["gates"]["memory"] is False
    assert result["gates"]["hash_manifest"] is False
    assert result["gates"]["leakage_audit"] is False
