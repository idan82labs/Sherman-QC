from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as v2


ROOT = Path("output/research/f1-gold27-20260803")


def _v1_fixture():
    path = Path(__file__).with_name(
        "test_f1_v2_cross_parent_opposite_fragment_rung_probe.py")
    spec = importlib.util.spec_from_file_location("v1_rung_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._fixture()


def _tiny_v2_payload():
    projected, control, baseline, kwargs = _v1_fixture()
    payload = v2._build_case_from_payloads(
        projected, frozen_control=control, complete_edge_baseline=baseline, **kwargs)
    return projected, payload


def _locked_case_payload(case_token: str) -> dict:
    control_manifest = json.loads(
        (ROOT / "local_plane_fragment_probe_v2/manifest.json").read_text())
    baseline_manifest = json.loads(
        (ROOT / "v2_complete_same_surface_edge_probe_v1/manifest.json").read_text())
    control_row = next(row for row in control_manifest["cases"] if row["case_token"] == case_token)
    baseline_row = next(row for row in baseline_manifest["cases"] if row["case_token"] == case_token)
    projected = v2.project_mapping_preflight_input(
        json.loads(Path(control_row["candidate_pool_path"]).read_text()))
    control = json.loads((ROOT / f"local_plane_fragment_probe_v2/{case_token}.json").read_text())
    baseline = json.loads((ROOT / f"v2_complete_same_surface_edge_probe_v1/{case_token}.json").read_text())
    return v2._build_case_from_payloads(
        projected, case_token=case_token,
        scan_sha256=control_row["scan_sha256"],
        candidate_pool_sha256=control_row["candidate_pool_sha256"],
        a0_prediction_sha256=control_row["a0_prediction_sha256"],
        frozen_control_probe_sha256=control_row["probe_sha256"],
        complete_edge_baseline_probe_sha256=baseline_row["replacement_probe_sha256"],
        frozen_control=control, complete_edge_baseline=baseline)


def _locked_case_summary(case_token: str) -> dict:
    payload = _locked_case_payload(case_token)
    relations = payload["frozen_target_baseline_relation_table"]
    outcomes = payload["frozen_target_outcome_table"]
    return {
        "case_token": case_token,
        "relation_count": len(relations), "outcome_count": len(outcomes),
        "target_ids": [row["frozen_target_component_id"] for row in relations],
        "mode_hashes": [row["frozen_target_mode_ids_sha256"] for row in relations],
        "target_hashes": [row["frozen_target_sha256"] for row in relations],
        "present": payload["case_counters"]["target_component_image_present_count"],
        "absent": payload["case_counters"]["target_component_image_absent_outside_tube_count"],
        "all_pair_count": payload["case_counters"]["all_pair_membership_row_count"],
    }


def test_mapped_targets_have_unique_pair_and_component_images() -> None:
    projected, payload = _tiny_v2_payload()
    v2.validate_v2_cross_parent_mapping_case_payload(payload, projected_input=projected)
    assert payload["schema_version"] == 2
    assert payload["kind"] == v2.PROBE_KIND
    assert len(payload["frozen_target_baseline_relation_table"]) == 2
    assert all(row["target_baseline_relation_status"] == v2.RELATION_COMPONENT_UNIQUE
               for row in payload["frozen_target_baseline_relation_table"])
    assert all(row["baseline_pair_mapping_count"] == 1
               and row["baseline_component_mapping_count"] == 1
               for row in payload["frozen_target_baseline_relation_table"])
    assert payload["case_counters"]["added_cross_parent_rung_count"] == 0


def test_zero_or_multiple_pair_lineages_fail_closed() -> None:
    target = {"frozen_target_mode_ids": ["M_a", "M_b"]}
    fragments = {"M_a": "F_a", "M_b": "F_b"}
    pair = {"local_plane_pair_id": "P", "fragment_ids": ["F_a", "F_b"]}
    with pytest.raises(ValueError, match="PAIR_LINEAGE_NOT_UNIQUE"):
        v2._exact_pair_lineage(target, fragments, [])
    with pytest.raises(ValueError, match="PAIR_LINEAGE_NOT_UNIQUE"):
        v2._exact_pair_lineage(target, fragments, [pair, deepcopy(pair)])


def test_missing_fragment_or_non_two_side_mapping_fails_closed() -> None:
    target = {"frozen_target_mode_ids": ["M_a", "M_b"]}
    with pytest.raises(ValueError, match="NO_BASELINE_FRAGMENT"):
        v2._exact_pair_lineage(target, {"M_a": "F_a"}, [])
    with pytest.raises(ValueError, match="EXACTLY_TWO_FRAGMENTS"):
        v2._exact_pair_lineage(target, {"M_a": "F_a", "M_b": "F_a"}, [])


def test_in_tube_and_out_of_tube_component_relation_failures() -> None:
    projected, control, baseline, _kwargs = _v1_fixture()
    context = v1._baseline_context(projected, baseline)
    target = v1._frozen_targets(control, "C999")[0]
    pair, rows, fragments = v2._exact_pair_lineage(
        target, context["fragment_of_mode"], list(context["pair_by_id"].values()))
    pair_context = v1._pair_context(pair, context)
    # In tube but with no component image is inadmissible.
    no_components = {**pair_context, "baseline_components": []}
    with pytest.raises(ValueError, match="IN_TUBE_TARGET_COMPONENT_IMAGE_NOT_UNIQUE"):
        v2._relation_row(target=target, pair=pair, mode_fragment_rows=rows,
                         fragment_ids=fragments, pair_context=no_components, context=context)
    # Outside tube but retaining a component image is also inadmissible.
    outside_with_component = {
        **pair_context,
        "transverse_parent_ids": set(pair_context["transverse_parent_ids"])
        - {target["frozen_target_parent_id"]},
    }
    with pytest.raises(ValueError, match="OUT_OF_TUBE_TARGET_HAS_COMPONENT_IMAGE"):
        v2._relation_row(target=target, pair=pair, mode_fragment_rows=rows,
                         fragment_ids=fragments, pair_context=outside_with_component,
                         context=context)


def test_c001_maps_to_pair_but_not_component_and_tube_stays_3mm() -> None:
    payload = _locked_case_payload("C001")
    row = next(row for row in payload["frozen_target_baseline_relation_table"]
               if row["frozen_target_component_id"] == "LTC_32d3609a79a9c314")
    assert row["frozen_target_pair_id"] == "LPP_daf852ef51c06429"
    assert row["frozen_target_parent_id"] == "P_47b0d4bd56e0f8ce"
    assert row["frozen_target_mode_ids"] == [
        "M_73e7f7dd53e69d0f", "M_d5d9b9a524d05ccc"]
    assert row["baseline_fragment_ids"] == [
        "VEF_6fa481eb4e1f5bc8", "VEF_ca3d2d1302651a00"]
    assert row["baseline_pair_mapping_count"] == 1
    assert row["baseline_local_plane_pair_id"] == "VEP_7610cbf5af1e3558"
    assert row["baseline_pair_sha256"] == (
        "e48e7f2d53e2885a6bbc5e5af9256c81e1761b3b0e0bfe9efd104f9b65f805b2")
    assert row["target_parent_line_distance_mm"] == 7.421545400596292
    assert row["line_tube_tolerance_mm"] == 3.0
    assert row["target_parent_in_frozen_transverse_set"] is False
    assert row["baseline_component_mapping_count"] == 0
    assert row["complete_edge_baseline_component_id"] is None
    assert row["target_baseline_relation_status"] == v2.RELATION_COMPONENT_NONE_OUTSIDE
    assert row["target_baseline_relation_sha256"] == (
        "74672bccf152041d26bcf31e21a6ceacb31d439339a3f705a371fd3cc3a2814f")


def test_out_of_tube_target_remains_structurally_negative_outcome() -> None:
    payload = _locked_case_payload("C001")
    outcome = next(row for row in payload["frozen_target_outcome_table"]
                   if row["frozen_target_component_id"] == "LTC_32d3609a79a9c314")
    assert outcome["target_outcome_code"] == v2.OUTCOME_STRUCTURALLY_UNREACHABLE
    assert outcome["complete_edge_baseline_component_id"] is None
    assert outcome["replacement_component_id"] is None
    for field in (
        "support_parent_count_before", "support_parent_count_after",
        "occupied_bin_count_before", "occupied_bin_count_after",
        "left_side_bin_count_before", "left_side_bin_count_after",
        "right_side_bin_count_before", "right_side_bin_count_after",
        "transition_rung_bin_count_before", "transition_rung_bin_count_after",
        "geometry_gate_depth_before", "geometry_gate_depth_after",
        "first_rejection_code_before", "first_rejection_code_after",
    ):
        assert outcome[field] is None
    assert outcome["target_reached_by_added_rung"] is False
    assert outcome["target_side_gate_advanced"] is False
    assert outcome["target_novel_rung_attributable"] is False
    assert outcome["target_outcome_sha256"] == (
        "61f781a8396e223617492509083d8780525fb360c49ece4b5f4136038a5a1dd2")


def test_all_pair_lineage_rows_feed_target_eligible_pair_accounting() -> None:
    payload = _locked_case_payload("C001")
    relation_pair_ids = {
        row["baseline_local_plane_pair_id"]
        for row in payload["frozen_target_baseline_relation_table"]
    }
    assert "VEP_7610cbf5af1e3558" in relation_pair_ids
    assert payload["case_counters"]["target_eligible_pair_count"] == len(relation_pair_ids)
    assert payload["case_counters"]["pair_lineage_mapped_target_count"] == 260


def test_no_nearest_pair_distance_fallback_or_target_exception_exists() -> None:
    source = inspect.getsource(v2._exact_pair_lineage)
    assert "distance" not in source.lower()
    assert "nearest" not in source.lower()
    assert "C001" not in source
    assert "target-specific" not in source.lower()


def test_validator_recomputes_attestation_predicates() -> None:
    _projected, payload = _tiny_v2_payload()
    payload["case_mapping_attestation"]["every_target_has_exactly_one_pair_lineage"] = False
    payload["case_mapping_attestation"]["case_mapping_attestation_sha256"] = v2._record_hash(
        payload["case_mapping_attestation"], "case_mapping_attestation_sha256")
    with pytest.raises(ValueError, match="predicate mismatch"):
        v2.validate_v2_cross_parent_mapping_case_payload(payload)


def test_target_id_exchange_with_same_count_is_detected() -> None:
    _projected, payload = _tiny_v2_payload()
    frozen = payload["frozen_target_projection"]
    relations = deepcopy(payload["frozen_target_baseline_relation_table"])
    relations[0]["frozen_target_component_id"] = "LTC_exchanged"
    derived = v2.derive_target_preservation_predicates(relations, frozen)
    assert derived["frozen_target_rows_byte_identical"] is False
    assert derived["frozen_target_id_sets_exact"] is False


def test_target_mode_hash_exchange_with_same_count_is_detected() -> None:
    _projected, payload = _tiny_v2_payload()
    frozen = payload["frozen_target_projection"]
    relations = deepcopy(payload["frozen_target_baseline_relation_table"])
    relations[0]["frozen_target_mode_ids_sha256"] = "f" * 64
    derived = v2.derive_target_preservation_predicates(relations, frozen)
    assert derived["frozen_target_rows_byte_identical"] is False
    assert derived["frozen_target_mode_hash_sets_exact"] is False


def test_valid_looking_wrong_threshold_hash_is_rejected() -> None:
    _projected, payload = _tiny_v2_payload()
    payload["thresholds_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="threshold hash mismatch"):
        v2.validate_v2_cross_parent_mapping_case_payload(payload)


def test_non_target_pair_membership_tamper_is_independently_rejected() -> None:
    control_manifest = json.loads(
        (ROOT / "local_plane_fragment_probe_v2/manifest.json").read_text())
    control_row = control_manifest["cases"][0]
    projected = v2.project_mapping_preflight_input(
        json.loads(Path(control_row["candidate_pool_path"]).read_text()))
    payload = _locked_case_payload("C001")
    target_pair_ids = {
        row["baseline_local_plane_pair_id"]
        for row in payload["frozen_target_baseline_relation_table"]
    }
    tampered = next(
        row for row in payload["all_pair_membership_table"]
        if row["baseline_local_plane_pair_id"] not in target_pair_ids)
    tampered["transverse_parent_ids"] = [*tampered["transverse_parent_ids"], "P_tampered"]
    tampered["transverse_parent_ids"].sort()
    tampered["transverse_parent_ids_sha256"] = v2._sha(
        tampered["transverse_parent_ids"])
    payload["all_pair_membership_table_sha256"] = v2._table_sha(
        payload["all_pair_membership_table"], "baseline_local_plane_pair_id")
    # Preserve the falsely asserted Boolean and make its record hash internally valid.
    assert payload["case_mapping_attestation"]["pair_line_tube_membership_unchanged"] is True
    payload["case_mapping_attestation"]["case_mapping_attestation_sha256"] = v2._record_hash(
        payload["case_mapping_attestation"], "case_mapping_attestation_sha256")
    with pytest.raises(ValueError, match="independent all-pair membership"):
        v2.validate_v2_cross_parent_mapping_case_payload(
            payload, projected_input=projected)


def test_v1_payload_is_rejected_by_v2() -> None:
    projected, _control, _baseline, kwargs = _v1_fixture()
    v1_payload = v1.build_v2_cross_parent_opposite_fragment_rung_probe(
        projected, **kwargs)
    with pytest.raises(ValueError, match="schema mismatch|rejects V1"):
        v2.validate_v2_cross_parent_mapping_case_payload(v1_payload)


def test_ordinary_v2_builds_only_complete_adjacent_opposite_rungs() -> None:
    projected, _control, _baseline, kwargs = _v1_fixture()
    mapping_before = v2._build_case_from_payloads(
        projected, frozen_control=_control, complete_edge_baseline=_baseline, **kwargs)
    mapping_sha_before = hashlib.sha256(v2.canonical_json_bytes(mapping_before)).hexdigest()
    payload = v2.build_v2_cross_parent_ordinary_case(projected, **kwargs)
    v2.validate_v2_cross_parent_ordinary_case_payload(
        payload, projected_input=projected)
    assert payload["packet_status"] == "ORDINARY_V2_CASE_COMPLETE"
    assert payload["all_ordinary_v2_criteria_valid"] is True
    assert all(payload["ordinary_v2_criteria"].values())
    assert payload["case_counters"]["added_cross_parent_rung_count"] == 2
    assert payload["case_counters"]["target_reached_by_added_rung_count"] == 2
    assert [row["frozen_target_component_id"] for row in payload[
        "frozen_target_baseline_relation_table"]] == [
            row["frozen_target_component_id"]
            for row in payload["frozen_target_outcome_table"]]
    assert len(payload["target_eligible_pair_table"]) == 1
    for row in payload["added_cross_parent_rung_table"]:
        assert row["edge_kind"] == "CROSS_MODE_ADJACENT_PARENT"
        assert row["parent_voxel_chebyshev_distance"] == 1
        assert row["axial_parent_separation_mm"] <= row["axial_bin_width_mm"]
    # The mapping-only API and canonical payload remain stable after ordinary build.
    mapping_after = v2._build_case_from_payloads(
        projected, frozen_control=_control, complete_edge_baseline=_baseline, **kwargs)
    assert hashlib.sha256(v2.canonical_json_bytes(mapping_after)).hexdigest() == mapping_sha_before


def test_ordinary_criteria_initialize_false() -> None:
    criteria = v2.new_ordinary_v2_criteria()
    assert set(criteria) == set(v2.ORDINARY_V2_CRITERION_KEYS)
    assert criteria and not any(criteria.values())


def test_structural_negative_outcome_is_copied_byte_exact_without_graph_entry() -> None:
    mapping = _locked_case_payload("C001")
    relation = next(
        row for row in mapping["frozen_target_baseline_relation_table"]
        if row["target_baseline_relation_status"] == v2.RELATION_COMPONENT_NONE_OUTSIDE)
    outcome = next(
        row for row in mapping["frozen_target_outcome_table"]
        if row["frozen_target_component_id"] == relation["frozen_target_component_id"])
    minimal_mapping = {
        "frozen_target_baseline_relation_table": [relation],
        "frozen_target_outcome_table": [outcome],
    }
    rebuilt = v2._ordinary_target_outcomes(
        mapping_payload=minimal_mapping, replacement_meta={},
        attribution_by_component={}, pair_contexts={})
    assert rebuilt == [outcome]


def test_executed_deletion_evidence_contains_raw_equal_canonical_tables() -> None:
    projected, _control, _baseline, kwargs = _v1_fixture()
    payload = v2.build_v2_cross_parent_ordinary_case(projected, **kwargs)
    deletion = payload["global_deletion_counterfactual"]
    assert deletion["deleted_added_cross_parent_rung_table"] == payload[
        "added_cross_parent_rung_table"]
    for rebuilt, frozen in (
        ("rebuilt_effective_edge_table", "frozen_effective_edge_table"),
        ("rebuilt_fragment_table", "frozen_fragment_table"),
        ("rebuilt_orphan_table", "frozen_orphan_table"),
        ("rebuilt_pair_records", "frozen_pair_records"),
        ("rebuilt_all_pair_membership_table", "frozen_all_pair_membership_table"),
        ("rebuilt_component_table", "frozen_component_table"),
        ("rebuilt_geometry_gate_table", "frozen_geometry_gate_table"),
    ):
        assert deletion[rebuilt] == deletion[frozen]
    assert deletion["all_required_equalities"] is True


def test_deletion_raw_table_tamper_rejected_even_with_rehashed_true_attestation() -> None:
    projected, _control, _baseline, kwargs = _v1_fixture()
    payload = v2.build_v2_cross_parent_ordinary_case(projected, **kwargs)
    deletion = payload["global_deletion_counterfactual"]
    deletion["rebuilt_geometry_gate_table"][0]["geometry_gate_depth"] += 1
    deletion["all_required_equalities"] = True
    deletion["deletion_evidence_sha256"] = v2._record_hash(
        deletion, "deletion_evidence_sha256")
    with pytest.raises(ValueError, match="deletion evidence is not valid"):
        v2.validate_v2_cross_parent_ordinary_case_payload(payload)


def test_deletion_input_must_bind_exact_added_rung_table() -> None:
    projected, _control, _baseline, kwargs = _v1_fixture()
    payload = v2.build_v2_cross_parent_ordinary_case(projected, **kwargs)
    payload["global_deletion_counterfactual"][
        "deleted_added_cross_parent_rung_table"] = []
    payload["global_deletion_counterfactual"]["deletion_evidence_sha256"] = v2._record_hash(
        payload["global_deletion_counterfactual"], "deletion_evidence_sha256")
    with pytest.raises(ValueError, match="does not match added-rung"):
        v2.validate_v2_cross_parent_ordinary_case_payload(payload)


@pytest.mark.slow
def test_exact_4788_target_rows_and_hash_sets_are_preserved() -> None:
    tokens = [f"C{index:03d}" for index in range(1, 38)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        summaries = list(executor.map(_locked_case_summary, tokens))
    assert sum(row["relation_count"] for row in summaries) == 4_788
    assert sum(row["outcome_count"] for row in summaries) == 4_788
    target_ids = [target_id for row in summaries for target_id in row["target_ids"]]
    mode_hashes = [value for row in summaries for value in row["mode_hashes"]]
    target_hashes = [value for row in summaries for value in row["target_hashes"]]
    assert len(target_ids) == len(set(target_ids)) == 4_788
    assert len(mode_hashes) == 4_788
    assert len(target_hashes) == 4_788
    assert sum(row["present"] + row["absent"] for row in summaries) == 4_788
    assert sum(row["all_pair_count"] for row in summaries) == 16_988
