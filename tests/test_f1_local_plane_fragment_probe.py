from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import pytest

from backend.f1_local_plane_fragment_probe import (
    FORBIDDEN_KEYS,
    LOCAL_PLANE_FRAGMENT_PROFILE,
    ORPHAN_KIND,
    ORPHAN_ROLE_CONTRACT,
    PROBE_KIND,
    PROBE_SCHEMA_VERSION,
    REJECTION_ORDER,
    _table_sha256,
    build_local_plane_fragment_probe,
    canonical_json_bytes,
    engineering_profile_descriptor_v2,
    engineering_profile_sha256,
    project_lpf_input,
    reject_forbidden_fields,
    validate_local_plane_fragment_payload,
    verify_orphan_deletion_counterfactual,
)


SCAN_HASH = "0" * 64
POOL_HASH = "1" * 64
PREDICTION_HASH = "2" * 64


def _mode(mode_id: str, parent_id: str, centroid, normal, *, fit=0.0) -> dict:
    return {
        "mode_atom_id": mode_id,
        "parent_atom_id": parent_id,
        "centroid": list(centroid),
        "normal_projective": list(normal),
        "point_count": 4,
        "surface_fit_p90_mm": fit,
    }


def _edge(edge_id: str, left: str, right: str, kind: str) -> dict:
    return {
        "mode_edge_id": edge_id,
        "mode_ids": [left, right],
        "edge_kind": kind,
        "normal_residual": 0.0,
        "plane_residual": 0.0,
        "spatial_residual": 0.0,
    }


def _wedge_pool(
    *,
    parent_count: int = 3,
    spacing: float = 1.0,
    global_planes: list[dict] | None = None,
    overflow_parent: int | None = None,
) -> dict:
    bin_width = max(3.0, 2.0 * spacing)
    parents = []
    modes = []
    edges = []
    for index in range(parent_count):
        parent_id = f"P{index}"
        left_id, right_id = f"LX{index}", f"RY{index}"
        z_value = bin_width * index
        parents.append(
            {
                "parent_atom_id": parent_id,
                "centroid": [0.0, 0.0, z_value],
                "point_count": 8,
                "normalized_support_mass": 1,
                "mode_child_ids": [left_id, right_id],
                "mode_overflow": index == overflow_parent,
                "transition_certificate_allowed": index != overflow_parent,
            }
        )
        modes.extend(
            (
                _mode(left_id, parent_id, [0.0, 0.0, z_value], [1.0, 0.0, 0.0]),
                _mode(right_id, parent_id, [0.0, 0.0, z_value], [0.0, 1.0, 0.0]),
            )
        )
        edges.append(_edge(f"C{index}", left_id, right_id, "CROSS_MODE_LOCAL"))
        if index:
            edges.append(_edge(f"SL{index}", f"LX{index - 1}", left_id, "SAME_SURFACE"))
            edges.append(_edge(f"SR{index}", f"RY{index - 1}", right_id, "SAME_SURFACE"))
    pool = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": parents,
        "support_mode_table": modes,
        "support_mode_edges": edges,
        "plane_table": list(global_planes or ()),
        "transition_local_spacing_mm": spacing,
        "transition_atom_caps": {
            "input_point_cap": 200_000,
            "parent_cap": 4_000,
            "mode_cap": 16_000,
            "mode_children_per_parent_cap": 4,
            "parent_plus_mode_cap": 20_000,
            "mode_edge_cap": 200_000,
        },
        "working_point_budget": 200_000,
        "transient_sampled_point_count": parent_count * 8,
    }
    _refresh_hashes(pool)
    return pool


def _refresh_hashes(pool: dict) -> None:
    pool["transition_parent_table_sha256"] = _table_sha256(
        pool["support_parent_table"], "parent_atom_id"
    )
    pool["transition_mode_table_sha256"] = _table_sha256(
        pool["support_mode_table"], "mode_atom_id"
    )
    pool["transition_mode_edge_table_sha256"] = _table_sha256(
        pool["support_mode_edges"], "mode_edge_id"
    )


def _build(pool: dict) -> dict:
    return build_local_plane_fragment_probe(
        project_lpf_input(pool),
        scan_sha256=SCAN_HASH,
        candidate_pool_sha256=POOL_HASH,
        a0_prediction_sha256=PREDICTION_HASH,
    )


def _components(payload: dict) -> list[dict]:
    return [
        component
        for pair in payload["local_plane_pair_records"]
        for component in pair["components"]
    ]


def test_three_bin_wedge_without_global_planes_is_one_valid_local_certificate() -> None:
    payload = _build(_wedge_pool())
    assert payload["profile"] == LOCAL_PLANE_FRAGMENT_PROFILE
    assert payload["schema_version"] == PROBE_SCHEMA_VERSION
    assert payload["kind"] == PROBE_KIND
    assert payload["local_plane_orphan_table"] == []
    assert len(payload["local_plane_fragment_table"]) == 2
    assert len(payload["local_plane_pair_records"]) == 1
    components = _components(payload)
    assert len(components) == 1
    assert components[0]["geometry_certificate_valid"] is True
    assert components[0]["first_rejection_code"] is None
    assert components[0]["global_pair_line_state"] == "UNAVAILABLE"
    assert payload["case_counters"]["geometry_certificate_count"] == 1
    assert payload["prediction_mutated"] is False
    assert payload["birth_columns_enabled"] is False


def test_flat_panel_and_same_normal_children_do_not_manufacture_pair() -> None:
    pool = _wedge_pool()
    pool["support_mode_table"] = [
        mode for mode in pool["support_mode_table"] if mode["mode_atom_id"].startswith("LX")
    ]
    pool["support_parent_table"] = [
        {**parent, "point_count": 4, "mode_child_ids": [f"LX{index}"]}
        for index, parent in enumerate(pool["support_parent_table"])
    ]
    pool["support_mode_edges"] = [
        edge for edge in pool["support_mode_edges"] if edge["edge_kind"] == "SAME_SURFACE"
        and edge["mode_ids"][0].startswith("LX")
    ]
    _refresh_hashes(pool)
    payload = _build(pool)
    assert len(payload["local_plane_fragment_table"]) == 1
    assert payload["local_plane_pair_records"] == []
    assert payload["case_counters"]["geometry_certificate_count"] == 0


def test_same_parent_same_normal_children_and_fragmented_panel_remain_separate() -> None:
    pool = _wedge_pool(parent_count=1)
    pool["support_mode_table"] = [
        _mode("A", "P0", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _mode("B", "P0", [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ]
    pool["support_parent_table"] = [
        {
            "parent_atom_id": "P0",
            "centroid": [0.0, 0.5, 0.0],
            "point_count": 8,
            "normalized_support_mass": 1,
            "mode_child_ids": ["A", "B"],
            "mode_overflow": False,
            "transition_certificate_allowed": True,
        }
    ]
    pool["support_mode_edges"] = [_edge("C", "A", "B", "CROSS_MODE_LOCAL")]
    _refresh_hashes(pool)
    payload = _build(pool)
    assert len(payload["local_plane_fragment_table"]) == 2
    assert len(payload["local_plane_pair_records"]) == 1
    assert payload["local_plane_pair_records"][0]["pair_rejection_counts"] == {
        "PLANE_PAIR_ANGLE_INELIGIBLE": 1
    }
    assert payload["case_counters"]["geometry_certificate_count"] == 0


def test_missing_one_side_has_exact_pair_rejection_without_invented_edges() -> None:
    parents = [
        {
            "parent_atom_id": "P0", "centroid": [0.0, 0.0, 0.0], "point_count": 4,
            "normalized_support_mass": 1, "mode_child_ids": ["L0"],
            "mode_overflow": False, "transition_certificate_allowed": True,
        },
        {
            "parent_atom_id": "P1", "centroid": [0.0, 0.0, 3.0], "point_count": 4,
            "normalized_support_mass": 1, "mode_child_ids": ["L1"],
            "mode_overflow": False, "transition_certificate_allowed": True,
        },
        {
            "parent_atom_id": "PF", "centroid": [10.0, 10.0, 0.0], "point_count": 8,
            "normalized_support_mass": 1, "mode_child_ids": ["LF", "R"],
            "mode_overflow": False, "transition_certificate_allowed": True,
        },
    ]
    modes = [
        _mode("L0", "P0", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _mode("L1", "P1", [0.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
        _mode("LF", "PF", [0.0, 10.0, 0.0], [1.0, 0.0, 0.0]),
        _mode("R", "PF", [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ]
    edges = [
        _edge("S0", "L0", "L1", "SAME_SURFACE"),
        _edge("S1", "L1", "LF", "SAME_SURFACE"),
        _edge("C", "LF", "R", "CROSS_MODE_LOCAL"),
    ]
    pool = _wedge_pool()
    pool.update(
        support_parent_table=parents,
        support_mode_table=modes,
        support_mode_edges=edges,
        transient_sampled_point_count=16,
    )
    _refresh_hashes(pool)
    payload = _build(pool)
    pair = payload["local_plane_pair_records"][0]
    assert pair["right_mode_count"] == 0
    assert pair["pair_rejection_counts"]["MISSING_RIGHT_MODE"] == 1
    assert pair["pair_rejection_counts"]["NO_CROSS_MODE_RUNG"] == 1
    assert pair["cross_mode_rung_count"] == 0


def test_normal_drift_chain_cannot_exceed_complete_fragment_diameter() -> None:
    parents, modes, edges = [], [], []
    for index, angle in enumerate((0.0, 10.0, 20.0)):
        mode_id, parent_id = f"M{index}", f"P{index}"
        radians = math.radians(angle)
        parents.append(
            {
                "parent_atom_id": parent_id,
                "centroid": [0.0, index * 3.0, 0.0],
                "point_count": 4,
                "normalized_support_mass": 1,
                "mode_child_ids": [mode_id],
                "mode_overflow": False,
                "transition_certificate_allowed": True,
            }
        )
        modes.append(_mode(mode_id, parent_id, [0.0, index * 3.0, 0.0], [math.cos(radians), 0.0, math.sin(radians)]))
        if index:
            edges.append(_edge(f"S{index}", f"M{index - 1}", mode_id, "SAME_SURFACE"))
    pool = _wedge_pool()
    pool.update(
        support_parent_table=parents,
        support_mode_table=modes,
        support_mode_edges=edges,
        plane_table=[],
        transient_sampled_point_count=12,
    )
    _refresh_hashes(pool)
    payload = _build(pool)
    assert len(payload["local_plane_fragment_table"]) == 2
    assert payload["case_counters"]["plane_union_reject_normal_diameter_count"] == 1
    assert all(fragment["normal_diameter_deg"] <= 15.0 for fragment in payload["local_plane_fragment_table"])


def test_short_return_and_overflow_fail_in_frozen_rejection_order() -> None:
    short = _build(_wedge_pool(parent_count=2))
    component = _components(short)[0]
    assert component["first_rejection_code"] == "COMPONENT_TOO_SMALL"
    assert component["geometry_gate_depth"] == 0

    overflow = _build(_wedge_pool(overflow_parent=1))
    component = _components(overflow)[0]
    assert component["first_rejection_code"] == "MODE_OVERFLOW_CERTIFICATE_DISABLED"
    assert component["geometry_gates"]["transition_certificate_complete"] is False
    assert component["geometry_certificate_valid"] is False


def test_global_pair_equivalent_and_wrong_offset_are_diagnostic_only() -> None:
    equivalent_planes = [
        {"plane_id": "GX", "normal": [1.0, 0.0, 0.0], "offset": 0.0},
        {"plane_id": "GY", "normal": [0.0, 1.0, 0.0], "offset": 0.0},
    ]
    equivalent = _build(_wedge_pool(global_planes=equivalent_planes))
    assert _components(equivalent)[0]["global_pair_line_state"] == "EQUIVALENT"
    assert _components(equivalent)[0]["geometry_certificate_valid"] is True

    wrong_planes = [
        {"plane_id": "GX", "normal": [1.0, 0.0, 0.0], "offset": -3.0},
        {"plane_id": "GY", "normal": [0.0, 1.0, 0.0], "offset": -3.0},
    ]
    wrong = _build(_wedge_pool(spacing=2.0, global_planes=wrong_planes))
    assert _components(wrong)[0]["global_pair_line_state"] == "NON_EQUIVALENT"
    assert _components(wrong)[0]["geometry_certificate_valid"] is True
    assert _components(wrong)[0]["geometry_gates"] == _components(equivalent)[0]["geometry_gates"]


def test_mode_edge_permutation_and_projective_sign_leave_local_outputs_stable() -> None:
    pool = _wedge_pool()
    baseline = _build(pool)
    permuted = deepcopy(pool)
    permuted["support_parent_table"].reverse()
    permuted["support_mode_table"].reverse()
    permuted["support_mode_edges"].reverse()
    # The frozen table hashes are canonical by record ID, so record order is irrelevant.
    replay = _build(permuted)
    assert canonical_json_bytes(replay) == canonical_json_bytes(baseline)

    flipped = deepcopy(pool)
    for mode in flipped["support_mode_table"]:
        mode["normal_projective"] = [-value for value in mode["normal_projective"]]
    _refresh_hashes(flipped)
    flipped_payload = _build(flipped)
    # Input hashes correctly attest different bytes; all derived LPF tables remain invariant.
    assert flipped_payload["output_table_hashes"] == baseline["output_table_hashes"]
    assert flipped_payload["local_plane_fragment_table"] == baseline["local_plane_fragment_table"]
    assert flipped_payload["local_plane_pair_records"] == baseline["local_plane_pair_records"]


def test_high_fit_singleton_is_orphan_not_fragment_and_cannot_certify() -> None:
    pool = _wedge_pool()
    # Disconnect one high-fit mode so it remains an inherited singleton.
    pool["support_mode_table"][0]["surface_fit_p90_mm"] = 3.01
    pool["support_mode_edges"] = [
        edge for edge in pool["support_mode_edges"] if "LX0" not in edge["mode_ids"]
    ]
    _refresh_hashes(pool)
    payload = _build(pool)
    orphan = next(
        item for item in payload["local_plane_orphan_table"]
        if item["mode_atom_id"] == "LX0"
    )
    assert orphan["orphan_kind"] == ORPHAN_KIND
    assert orphan["computed_singleton_surface_fit_p90_mm"] == 3.01
    assert orphan["role_contract"] == ORPHAN_ROLE_CONTRACT
    assert all("LX0" not in fragment["witness_mode_ids"] for fragment in payload["local_plane_fragment_table"])
    assert payload["case_counters"]["local_plane_orphan_count"] == 1
    assert payload["case_counters"]["orphan_mode_count"] == 1
    assert all(
        "LX0" not in component["witness_mode_ids"]
        for pair in payload["local_plane_pair_records"]
        for component in pair["components"]
    )


def test_orphan_incident_edges_are_counted_excluded_and_match_deletion_reference() -> None:
    pool = _wedge_pool()
    pool["support_mode_table"][0]["surface_fit_p90_mm"] = 3.01
    _refresh_hashes(pool)
    payload = _build(pool)
    orphan = next(item for item in payload["local_plane_orphan_table"] if item["mode_atom_id"] == "LX0")
    assert orphan["incident_same_surface_edge_count"] == 1
    assert orphan["incident_cross_mode_edge_count"] == 1
    assert payload["case_counters"]["excluded_cross_mode_seed_edge_count_due_to_orphan"] == 1
    assert all("LX0" not in component["witness_mode_ids"] for component in _components(payload))
    projected = project_lpf_input(pool)
    counterfactual = verify_orphan_deletion_counterfactual(
        projected,
        scan_sha256=SCAN_HASH,
        candidate_pool_sha256=POOL_HASH,
        a0_prediction_sha256=PREDICTION_HASH,
    )
    assert counterfactual["orphan_mode_count"] == 1
    assert counterfactual["fragment_table_identical"] is True
    assert counterfactual["pair_component_table_identical"] is True
    assert counterfactual["certificate_set_identical"] is True
    assert counterfactual["non_orphan_downstream_counters_identical"] is True
    assert (
        counterfactual["normal_inference_semantic_sha256"]
        != counterfactual["deletion_reference_inference_semantic_sha256"]
    )


def test_orphan_witness_cap_preserves_full_incident_edge_hash() -> None:
    parents = [
        {
            "parent_atom_id": "PO", "centroid": [0.0, 0.0, 0.0], "point_count": 4,
            "normalized_support_mass": 1, "mode_child_ids": ["X"],
            "mode_overflow": False, "transition_certificate_allowed": True,
        }
    ]
    modes = [_mode("X", "PO", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fit=3.01)]
    edges = []
    for index in range(9):
        parent_id, mode_id = f"P{index}", f"M{index}"
        parents.append(
            {
                "parent_atom_id": parent_id, "centroid": [0.0, 0.0, index + 1.0],
                "point_count": 4, "normalized_support_mass": 1, "mode_child_ids": [mode_id],
                "mode_overflow": False, "transition_certificate_allowed": True,
            }
        )
        modes.append(_mode(mode_id, parent_id, [0.0, 0.0, index + 1.0], [0.0, 1.0, 0.0]))
        edges.append(_edge(f"E{index}", "X", mode_id, "CROSS_MODE_LOCAL"))
    pool = _wedge_pool()
    pool.update(
        support_parent_table=parents,
        support_mode_table=modes,
        support_mode_edges=edges,
        transient_sampled_point_count=40,
    )
    _refresh_hashes(pool)
    payload = _build(pool)
    orphan = payload["local_plane_orphan_table"][0]
    full_edge_ids = [f"E{index}" for index in range(9)]
    assert orphan["witness_mode_edge_ids"] == full_edge_ids[:8]
    assert orphan["witness_ids_truncated"] is True
    assert orphan["incident_cross_mode_edge_count"] == 9
    assert orphan["incident_mode_edge_ids_sha256"] == hashlib.sha256(
        canonical_json_bytes(full_edge_ids)
    ).hexdigest()
    assert payload["case_counters"]["excluded_cross_mode_seed_edge_count_due_to_orphan"] == 9


def test_singleton_classifier_uses_strict_binary64_greater_than() -> None:
    tolerance = 1.5
    equal = _wedge_pool(parent_count=1)
    equal["support_mode_table"] = [
        _mode("M", "P0", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fit=tolerance)
    ]
    equal["support_parent_table"][0].update(
        point_count=4, mode_child_ids=["M"]
    )
    equal["support_mode_edges"] = []
    _refresh_hashes(equal)
    equal_payload = _build(equal)
    assert len(equal_payload["local_plane_fragment_table"]) == 1
    assert equal_payload["local_plane_orphan_table"] == []

    above = deepcopy(equal)
    strict_above = math.nextafter(tolerance, math.inf)
    above["support_mode_table"][0]["surface_fit_p90_mm"] = strict_above
    _refresh_hashes(above)
    above_payload = _build(above)
    assert above_payload["local_plane_fragment_table"] == []
    assert above_payload["local_plane_orphan_table"][0][
        "computed_singleton_surface_fit_p90_mm"
    ] == strict_above


def test_nonfinite_singleton_and_overfit_multimode_fail_closed() -> None:
    nonfinite = _wedge_pool(parent_count=1)
    nonfinite["support_mode_table"][0]["surface_fit_p90_mm"] = float("nan")
    with pytest.raises(ValueError, match="Out of range|finite"):
        _build(nonfinite)

    tolerance = 1.5
    overfit = _wedge_pool(parent_count=1)
    overfit["support_mode_table"] = [
        _mode("A", "P0", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fit=math.nextafter(tolerance, math.inf)),
        _mode("B", "P0", [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], fit=math.nextafter(tolerance, math.inf)),
    ]
    overfit["support_parent_table"][0].update(
        point_count=8, mode_child_ids=["A", "B"]
    )
    overfit["support_mode_edges"] = [_edge("S", "A", "B", "SAME_SURFACE")]
    _refresh_hashes(overfit)
    with pytest.raises(ValueError, match="multi-mode final component"):
        _build(overfit)


def test_engineering_hash_recomputes_complete_v2_descriptor_and_is_sensitive() -> None:
    payload = _build(_wedge_pool())
    descriptor = engineering_profile_descriptor_v2()
    assert payload["engineering_profile_sha256"] == engineering_profile_sha256(
        payload["thresholds"]
    )
    for path, replacement in (
        (("final_component_classification", "comparison"), "GREATER_OR_EQUAL"),
        (("orphan_role_contract", "local_pair_seed_allowed"), True),
        (("hard_caps", "local_plane_orphan_cap"), 15_999),
    ):
        mutated = deepcopy(descriptor)
        mutated[path[0]][path[1]] = replacement
        assert engineering_profile_sha256(
            payload["thresholds"], descriptor=mutated
        ) != payload["engineering_profile_sha256"]


def test_exact_projection_and_input_contract_attestation_exclude_extra_fields() -> None:
    raw = _wedge_pool()
    raw.update(
        part_id="leak",
        solver={"objective": 1},
        manual_truth={"bend_count": 9},
        typed_graph={"atom_owners": []},
    )
    projection = project_lpf_input(raw)
    assert "part_id" not in projection and "solver" not in projection
    payload = build_local_plane_fragment_probe(
        projection,
        scan_sha256=SCAN_HASH,
        candidate_pool_sha256=POOL_HASH,
        a0_prediction_sha256=PREDICTION_HASH,
    )
    attestation = payload["input_contract_attestation"]
    assert attestation["transition_parent_count"] == 3
    assert attestation["transition_mode_count"] == 6
    assert attestation["transition_mode_edge_count"] == 7
    assert attestation["maximum_observed_modes_per_parent"] == 2
    assert attestation["all_inherited_caps_satisfied"] is True
    with pytest.raises(ValueError, match="exact allowlisted"):
        build_local_plane_fragment_probe(
            raw,
            scan_sha256=SCAN_HASH,
            candidate_pool_sha256=POOL_HASH,
            a0_prediction_sha256=PREDICTION_HASH,
        )


def test_v2_rejects_v1_identity_nested_schema_and_gate_depth_mutation() -> None:
    payload = _build(_wedge_pool())
    v1 = deepcopy(payload)
    v1.update(schema_version=1, kind="f1_local_plane_fragment_probe_v1", profile="LOCAL_PLANE_FRAGMENT_PROBE_V1")
    with pytest.raises(ValueError, match="V2 payload identity"):
        validate_local_plane_fragment_payload(v1)
    nested = deepcopy(payload)
    nested["local_plane_fragment_table"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="fragment nested schema"):
        validate_local_plane_fragment_payload(nested)
    depth = deepcopy(payload)
    component = depth["local_plane_pair_records"][0]["components"][0]
    component["geometry_gate_depth"] += 1
    component["component_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in component.items() if key != "component_sha256"})
    ).hexdigest()
    pair = depth["local_plane_pair_records"][0]
    pair["pair_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in pair.items() if key != "pair_sha256"})
    ).hexdigest()
    depth["output_table_hashes"]["local_plane_pair_records_sha256"] = _table_sha256(
        depth["local_plane_pair_records"], "local_plane_pair_id"
    )
    with pytest.raises(ValueError, match="gate depth"):
        validate_local_plane_fragment_payload(depth)
def test_schema_hashes_forbidden_keys_and_malformed_values_fail_closed() -> None:
    payload = _build(_wedge_pool())
    validate_local_plane_fragment_payload(json.loads(canonical_json_bytes(payload)))
    assert tuple(payload["local_plane_pair_records"][0]["components"][0]["geometry_gates"]) == tuple(
        gate for gate, _code in REJECTION_ORDER
    )
    for key in FORBIDDEN_KEYS:
        with pytest.raises(ValueError, match="forbids"):
            reject_forbidden_fields({"nested": [{key.upper(): "forbidden"}]})

    corrupted = deepcopy(payload)
    corrupted["local_plane_fragment_table"][0]["centroid"][0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        validate_local_plane_fragment_payload(corrupted)
    bad_pool = _wedge_pool()
    bad_pool["support_parent_table"][0]["point_count"] = -1
    _refresh_hashes(bad_pool)
    with pytest.raises(ValueError, match="nonnegative"):
        _build(bad_pool)
    hash_mismatch = _wedge_pool()
    hash_mismatch["support_mode_table"][0]["point_count"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        _build(hash_mismatch)


def test_metadata_is_ignored_and_never_enters_inference_payload() -> None:
    pool = _wedge_pool()
    baseline = _build(pool)
    mutated = deepcopy(pool)
    mutated.update(
        {
            "case_token": "C999",
            "part_id": "forbidden-if-consumed",
            "manual_truth": {"count": 99},
            "family_id": "leak",
            "expected_count": 99,
        }
    )
    assert canonical_json_bytes(_build(mutated)) == canonical_json_bytes(baseline)
    encoded = canonical_json_bytes(baseline).lower()
    for key in (b"case_token", b"part_id", b"manual_truth", b"family_id", b"expected_count"):
        assert key not in encoded
