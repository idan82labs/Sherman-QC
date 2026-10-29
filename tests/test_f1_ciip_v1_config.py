from __future__ import annotations

import copy
import dataclasses

import pytest

from research.f1_ciip_v1.config import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PACKAGE_ID,
    EXPECTED_SCHEMA_SHA256,
    load_frozen_v7,
    select_smoke_source_by_content,
    validate_frozen_allocation_plan,
)


def test_loads_exact_frozen_v7_bindings() -> None:
    frozen = load_frozen_v7()
    assert frozen.configuration["package_id"] == EXPECTED_PACKAGE_ID
    assert frozen.configuration["configuration_sha256"] == EXPECTED_CONFIG_SHA256
    assert frozen.schema["schema_sha256"] == EXPECTED_SCHEMA_SHA256
    assert frozen.sample_cap == 200_000
    assert frozen.rss_watchdog_bytes == 12_884_901_888
    assert frozen.wall_seconds == 300


def test_smoke_source_selection_uses_only_minimum_content_hash() -> None:
    frozen = load_frozen_v7()
    selected = select_smoke_source_by_content(frozen)
    observed = selected["raw_ply_sha256"]
    expected = min(row["raw_ply_sha256"] for row in frozen.source_manifest["cases"])
    assert observed == expected
    assert observed == "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"


def test_allocation_plan_is_recomputed_and_tamper_stops() -> None:
    frozen = load_frozen_v7()
    observed = validate_frozen_allocation_plan(frozen)
    assert observed == {
        "semantic_allocation_bytes": 596_856_088,
        "fixed_reserve_bytes": 1_140_850_688,
        "static_process_tree_bytes": 1_737_706_776,
        "scratch_required_bytes": 135_266_304,
        "serialized_payload_bytes": 64_487_336,
    }

    proof = copy.deepcopy(dict(frozen.resource_proof))
    proof["semantic_allocation_ledger_bytes"]["adjacency_work"] += 1
    with pytest.raises(ValueError, match="semantic allocation ledger"):
        validate_frozen_allocation_plan(
            dataclasses.replace(frozen, resource_proof=proof)
        )

    configuration = copy.deepcopy(dict(frozen.configuration))
    configuration["allocation_model"]["adjacency_work_bytes_per_edge"]["value"] += 1
    with pytest.raises(ValueError, match="semantic allocation ledger"):
        validate_frozen_allocation_plan(
            dataclasses.replace(frozen, configuration=configuration)
        )
