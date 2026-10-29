from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as edge

from scripts.build_f1_v2_complete_same_surface_edge_probe_bundle import (
    CERTIFICATE_NOVELTY_FIELDS,
    LOCK_CRITERION_KEYS,
    build_certificate_novelty_table,
    build_lock_attestation,
    canonical_bundle_bytes,
    SYNTHETIC_RESOURCE_KIND,
    SYNTHETIC_RESOURCE_PROFILE,
    SYNTHETIC_RESOURCE_RUN_FIELDS,
    SYNTHETIC_RESOURCE_TOP_FIELDS,
    _control_census,
    validate_lock_attestation,
    validate_synthetic_resource_attestation,
    _canonical_json_bytes,
    _execute_case_rebuilds,
)


SHA = "1" * 64


def _write_canonical(path: Path, payload: dict) -> str:
    encoded = edge.canonical_json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _parallel_rebuild_tasks(tmp_path: Path) -> list[dict]:
    parents = [
        {
            "parent_atom_id": "P_a", "centroid": [0.0, 0.0, 0.0],
            "point_count": 10, "normalized_support_mass": 1,
            "mode_child_ids": ["M_a"], "mode_overflow": False,
            "transition_certificate_allowed": True, "voxel_key": [0, 0, 0],
            "voxel_level": 0, "voxel_size_mm": 1.0,
        },
        {
            "parent_atom_id": "P_b", "centroid": [1.0, 0.0, 0.0],
            "point_count": 10, "normalized_support_mass": 1,
            "mode_child_ids": ["M_b"], "mode_overflow": False,
            "transition_certificate_allowed": True, "voxel_key": [1, 0, 0],
            "voxel_level": 0, "voxel_size_mm": 1.0,
        },
    ]
    modes = [
        {"mode_atom_id": "M_a", "parent_atom_id": "P_a",
         "centroid": [0.0, 0.0, 0.0], "normal_projective": [0.0, 0.0, 1.0],
         "point_count": 10, "surface_fit_p90_mm": 0.0},
        {"mode_atom_id": "M_b", "parent_atom_id": "P_b",
         "centroid": [1.0, 0.0, 0.0], "normal_projective": [0.0, 0.0, 1.0],
         "point_count": 10, "surface_fit_p90_mm": 0.0},
    ]
    projected = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": parents, "support_mode_table": modes,
        "support_mode_edges": [], "plane_table": [],
        "transition_parent_table_sha256": lpf._table_sha256(parents, "parent_atom_id"),
        "transition_mode_table_sha256": lpf._table_sha256(modes, "mode_atom_id"),
        "transition_mode_edge_table_sha256": lpf._table_sha256([], "mode_edge_id"),
        "transition_local_spacing_mm": 1.0,
        "transition_atom_caps": dict(lpf.TRANSITION_ATOM_CAPS),
        "working_point_budget": 200_000, "transient_sampled_point_count": 20,
    }
    pool_path = tmp_path / "pool.json"
    pool_sha = _write_canonical(pool_path, projected)
    scan_sha, a0_sha = "a" * 64, "b" * 64
    control = lpf.build_local_plane_fragment_probe(
        projected, scan_sha256=scan_sha, candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
    )
    control_path = tmp_path / "control.json"
    control_sha = _write_canonical(control_path, control)
    replacement = edge.build_v2_complete_same_surface_edge_probe(
        projected, scan_sha256=scan_sha, candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha, frozen_control_probe_sha256=control_sha,
    )
    replacement_path = tmp_path / "replacement.json"
    replacement_sha = _write_canonical(replacement_path, replacement)
    base = {
        "replacement_path": str(replacement_path), "control_path": str(control_path),
        "pool_path": str(pool_path), "replacement_probe_sha256": replacement_sha,
        "frozen_control_probe_sha256": control_sha, "candidate_pool_sha256": pool_sha,
        "scan_sha256": scan_sha, "a0_prediction_sha256": a0_sha,
    }
    return [{"case_token": token, **base} for token in ("C002", "C001")]


def _synthetic_attestation() -> dict:
    runs = []
    for index, peak in enumerate((100_000_000, 110_000_000, 105_000_000), 1):
        run = {
            "run_index": index,
            "status": "SUCCESS",
            "parent_count": 4_000,
            "mode_count": 16_000,
            "modes_per_parent": 4,
            "adjacent_parent_pair_count": 52_000,
            "eligible_mode_pair_evaluation_count": 832_000,
            "complete_same_surface_edge_count": 832_000,
            "inherited_cross_mode_edge_count": 24_000,
            "effective_mode_edge_count": 856_000,
            "complete_same_surface_edge_table_sha256": "2" * 64,
            "inherited_cross_mode_edge_table_sha256": "3" * 64,
            "effective_mode_edge_semantic_sha256": "4" * 64,
            "canonical_resource_semantic_sha256": "5" * 64,
            "wall_seconds": float(index),
            "peak_rss_bytes": peak,
        }
        assert set(run) == SYNTHETIC_RESOURCE_RUN_FIELDS
        runs.append(run)
    record = {
        "schema_version": 1,
        "kind": SYNTHETIC_RESOURCE_KIND,
        "profile": "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1",
        "resource_profile": SYNTHETIC_RESOURCE_PROFILE,
        "resource_envelope_only": True,
        "physical_voxel_realizable": False,
        "resource_limits": {
            "peak_rss_limit_bytes": 16 * 1024**3,
            "parent_cap": 4_000,
            "mode_cap": 16_000,
            "modes_per_parent_cap": 4,
            "adjacent_parent_pair_cap": 52_000,
            "complete_edge_evaluation_cap": 832_000,
            "complete_same_surface_edge_cap": 832_000,
            "inherited_cross_mode_edge_cap": 24_000,
            "effective_mode_edge_cap": 856_000,
        },
        "runs": runs,
        "semantic_hashes_identical": True,
        "all_runs_below_16_gib": True,
    }
    record["attestation_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    assert set(record) == SYNTHETIC_RESOURCE_TOP_FIELDS
    return record


def _component(identifier: str, y: float, *, valid: bool = True) -> dict:
    return {
        "local_transition_component_id": identifier,
        "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
        "geometry_certificate_valid": valid,
        "geometry_gate_depth": 13 if valid else 2,
        "first_rejection_code": None if valid else "INSUFFICIENT_AXIAL_SPAN",
    }


def _probe(component: dict, *, replacement: bool) -> dict:
    payload = {
        "local_plane_pair_records": [{"components": [component]}],
        "component_edge_attribution_table": [],
    }
    if replacement:
        payload["component_edge_attribution_table"] = [{
            "local_transition_component_id": component["local_transition_component_id"],
            "edge_semantics_dependent": True,
            "causal_added_same_surface_edge_ids_sha256": SHA,
            "geometry_certificate_valid": component["geometry_certificate_valid"],
        }]
    return payload


def test_lock_requires_exactly_all_39_criteria() -> None:
    assert len(LOCK_CRITERION_KEYS) == 39
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    lock = build_lock_attestation(criteria)
    validate_lock_attestation(lock)
    assert lock["ready_for_gold"] is True

    missing = deepcopy(criteria)
    missing.pop("intervention_nonempty")
    refused = build_lock_attestation(missing)
    validate_lock_attestation(refused)
    assert refused["ready_for_gold"] is False
    assert refused["criteria"]["intervention_nonempty"] is False

    extra = {**criteria, "not_preregistered": True}
    refused_extra = build_lock_attestation(extra)
    assert refused_extra["ready_for_gold"] is False


def test_certificate_novelty_is_causal_and_control_relative() -> None:
    replacement = _probe(_component("VEC_NEW", 0.0), replacement=True)
    equivalent_control = _probe(_component("LTC_CONTROL", 1.0), replacement=False)
    rows = build_certificate_novelty_table(
        case_token="C001",
        replacement_probe=replacement,
        frozen_control_probe=equivalent_control,
    )
    assert len(rows) == 1
    assert set(rows[0]) == CERTIFICATE_NOVELTY_FIELDS
    assert rows[0]["control_equivalent_certificate_ids"] == ["LTC_CONTROL"]
    assert rows[0]["novel_edge_attributable"] is False

    distant_control = _probe(_component("LTC_CONTROL", 20.0), replacement=False)
    novel = build_certificate_novelty_table(
        case_token="C001",
        replacement_probe=replacement,
        frozen_control_probe=distant_control,
    )[0]
    assert novel["control_equivalent_certificate_ids"] == []
    assert novel["novel_edge_attributable"] is True
    expected = hashlib.sha256(
        json.dumps([], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    assert novel["control_equivalent_certificate_ids_sha256"] == expected


def test_compact_bundle_serialization_is_deterministic_and_non_embedding() -> None:
    compact = {
        "schema_version": 1,
        "embedded_probe_payloads": False,
        "cases": [{
            "case_token": "C001",
            "replacement_probe_sha256": SHA,
            "certificate_novelty_table": [],
        }],
    }
    encoded = canonical_bundle_bytes(compact)
    assert encoded == canonical_bundle_bytes(json.loads(encoded))
    assert b'"complete_same_surface_edge_table"' not in encoded
    assert b'"local_plane_pair_records"' not in encoded


def test_frozen_control_census_counts_intersection_zero_spans_exactly() -> None:
    zero_span_component = {
        "intersection_span_endpoints": [
            [-116.141062, -105.973598, 219.680567],
            [-116.141062, -105.973598, 219.680567],
        ],
        "geometry_certificate_valid": False,
    }
    finite_component = {
        "intersection_span_endpoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "geometry_certificate_valid": False,
    }
    # Reproduce the exact locked aggregate.  The repeated reference keeps the
    # fixture compact while ensuring a regression to the generic `_endpoints`
    # helper would return zero instead of 15,609.
    components = [zero_span_component] * 15_609 + [finite_component]
    census, distribution = _control_census(
        {
            "C001": {
                "local_plane_fragment_table": [],
                "local_plane_orphan_table": [],
                "local_plane_pair_records": [{"components": components}],
            }
        }
    )
    assert census["local_transition_component_count"] == 15_610
    assert census["zero_span_component_count"] == 15_609
    assert census["geometry_certificate_count"] == 0
    assert distribution == {}


def test_lock_validator_rejects_extra_criterion_even_with_forged_ready_flag() -> None:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    lock = build_lock_attestation(criteria)
    lock["criteria"]["extra"] = True
    with pytest.raises(ValueError, match="exact 39"):
        validate_lock_attestation(lock)


def test_synthetic_worst_cap_attestation_requires_exact_three_run_envelope() -> None:
    attestation = _synthetic_attestation()
    assert validate_synthetic_resource_attestation(attestation) == 110_000_000


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"physical_voxel_realizable": True}), "abstract envelope"),
        (lambda value: value["runs"][1].update({"parent_count": 3_999}), "parent_count"),
        (
            lambda value: value["runs"][2].update(
                {"canonical_resource_semantic_sha256": "6" * 64}
            ),
            "semantic hashes",
        ),
        (
            lambda value: value["runs"][0].update(
                {"peak_rss_bytes": 16 * 1024**3}
            ),
            "RSS",
        ),
    ],
)
def test_synthetic_worst_cap_attestation_fails_closed(mutation, message: str) -> None:
    attestation = _synthetic_attestation()
    mutation(attestation)
    attestation["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in attestation.items() if key != "attestation_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match=message):
        validate_synthetic_resource_attestation(attestation)


def test_serial_and_parallel_rebuilds_are_canonical_and_read_only(tmp_path: Path) -> None:
    tasks = _parallel_rebuild_tasks(tmp_path)
    protected_paths = {
        Path(task[field])
        for task in tasks
        for field in ("replacement_path", "control_path", "pool_path")
    }
    before = {path: path.read_bytes() for path in protected_paths}

    serial = _execute_case_rebuilds(tasks, workers=1)
    parallel = _execute_case_rebuilds(tasks, workers=2)
    parallel_at_cap = _execute_case_rebuilds(tasks, workers=18)

    serial_bytes = _canonical_json_bytes(serial)
    parallel_bytes = _canonical_json_bytes(parallel)
    assert serial_bytes == parallel_bytes
    assert serial_bytes == _canonical_json_bytes(parallel_at_cap)
    assert hashlib.sha256(serial_bytes).hexdigest() == hashlib.sha256(parallel_bytes).hexdigest()
    assert [row["case_token"] for row in parallel] == ["C001", "C002"]
    assert before == {path: path.read_bytes() for path in protected_paths}


@pytest.mark.parametrize("workers", [0, 19])
def test_bundle_rebuild_worker_bound_fails_closed(tmp_path: Path, workers: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 18"):
        _execute_case_rebuilds([], workers=workers)
