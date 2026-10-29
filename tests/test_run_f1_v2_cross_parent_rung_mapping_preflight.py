from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as rung_core
from scripts import run_f1_v2_cross_parent_rung_mapping_preflight as preflight


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "output/research/f1-gold27-20260803"


def test_mapping_preflight_cli_is_mapping_only_and_worker_bounded() -> None:
    options = {
        option
        for action in preflight.build_parser()._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {
        "--inference-manifest",
        "--candidate-pool-manifest",
        "--a0-prediction-root",
        "--frozen-control-manifest",
        "--complete-edge-baseline-manifest",
        "--output",
        "--workers",
    }
    assert preflight.MAX_WORKERS == 8
    source = inspect.getsource(preflight._run_case) + inspect.getsource(
        preflight.build_relation_and_outcome_tables
    )
    assert "_build_rungs" not in source
    assert "build_v2_cross_parent_opposite_fragment_rung_probe(" not in source


def test_decision_requires_exactly_all_twenty_criteria() -> None:
    assert len(preflight.MAPPING_PREFLIGHT_CRITERIA) == 20
    criteria = {key: True for key in preflight.MAPPING_PREFLIGHT_CRITERIA}
    assert preflight.mapping_preflight_decision(criteria) == (
        preflight.DECISION_GO,
        True,
    )
    for key in preflight.MAPPING_PREFLIGHT_CRITERIA:
        mutated = dict(criteria)
        mutated[key] = False
        assert preflight.mapping_preflight_decision(mutated) == (
            preflight.DECISION_INVALID,
            False,
        )
    assert preflight.mapping_preflight_decision({**criteria, "extra": True}) == (
        preflight.DECISION_INVALID,
        False,
    )


def test_c001_generic_rule_maps_pair_but_no_component_without_rungs(
    monkeypatch,
) -> None:
    raw_pool = json.loads(
        (EVIDENCE / "candidate_pool_transition_v2_probe/C001.json").read_text()
    )
    projected = rung_core.project_cross_parent_rung_input(raw_pool)
    control = json.loads(
        (EVIDENCE / "local_plane_fragment_probe_v2/C001.json").read_text()
    )
    baseline = json.loads(
        (EVIDENCE / "v2_complete_same_surface_edge_probe_v1/C001.json").read_text()
    )

    def forbidden_rung_enumeration(*_args, **_kwargs):
        raise AssertionError("mapping preflight attempted added-rung enumeration")

    monkeypatch.setattr(rung_core, "_build_rungs", forbidden_rung_enumeration)
    relations, outcomes, audit = preflight.build_relation_and_outcome_tables(
        case_token="C001",
        projected=projected,
        control=control,
        baseline=baseline,
    )
    assert len(relations) == len(outcomes) == audit["frozen_target_count"] == 260
    row = next(
        value
        for value in relations
        if value["frozen_target_component_id"] == "LTC_32d3609a79a9c314"
    )
    assert set(row) == preflight.RELATION_FIELDS
    assert row["baseline_fragment_ids"] == [
        "VEF_6fa481eb4e1f5bc8",
        "VEF_ca3d2d1302651a00",
    ]
    assert row["baseline_pair_mapping_count"] == 1
    assert row["baseline_local_plane_pair_id"] == "VEP_7610cbf5af1e3558"
    assert row["target_parent_line_distance_mm"] == 7.421545400596292
    assert row["line_tube_tolerance_mm"] == 3.0
    assert row["target_parent_in_frozen_transverse_set"] is False
    assert row["baseline_component_mapping_count"] == 0
    assert row["complete_edge_baseline_component_id"] is None
    assert row["target_baseline_relation_status"] == preflight.RELATION_STATUS_ABSENT
    outcome = next(
        value
        for value in outcomes
        if value["frozen_target_component_id"] == row["frozen_target_component_id"]
    )
    assert set(outcome) == preflight.OUTCOME_FIELDS
    assert outcome["target_outcome_code"] == preflight.OUTCOME_ABSENT
    assert outcome["target_reached_by_added_rung"] is False
    assert outcome["target_side_gate_advanced"] is False
    assert outcome["target_novel_rung_attributable"] is False
    assert preflight._c001_locked_regression(relations) is True


def test_relation_and_outcome_tables_are_sorted_and_untruncated_for_c001() -> None:
    raw_pool = json.loads(
        (EVIDENCE / "candidate_pool_transition_v2_probe/C001.json").read_text()
    )
    relations, outcomes, _audit = preflight.build_relation_and_outcome_tables(
        case_token="C001",
        projected=rung_core.project_cross_parent_rung_input(raw_pool),
        control=json.loads(
            (EVIDENCE / "local_plane_fragment_probe_v2/C001.json").read_text()
        ),
        baseline=json.loads(
            (EVIDENCE / "v2_complete_same_surface_edge_probe_v1/C001.json").read_text()
        ),
    )
    assert relations == sorted(
        relations,
        key=lambda row: (row["case_token"], row["frozen_target_component_id"]),
    )
    assert outcomes == sorted(
        outcomes,
        key=lambda row: (row["case_token"], row["frozen_target_component_id"]),
    )
    assert len({row["frozen_target_component_id"] for row in relations}) == len(
        relations
    )


def test_complete_all_pair_membership_table_covers_non_target_pairs() -> None:
    raw_pool = json.loads(
        (EVIDENCE / "candidate_pool_transition_v2_probe/C001.json").read_text()
    )
    baseline = json.loads(
        (EVIDENCE / "v2_complete_same_surface_edge_probe_v1/C001.json").read_text()
    )
    memberships = preflight.build_all_pair_membership_table(
        rung_core.project_cross_parent_rung_input(raw_pool), baseline
    )
    assert len(memberships) == baseline["case_counters"]["local_plane_pair_count"] == 842
    assert len({row["baseline_local_plane_pair_id"] for row in memberships}) == 842
    assert all(set(row) == preflight.ALL_PAIR_MEMBERSHIP_FIELDS for row in memberships)
    assert all(
        row["transverse_parent_ids_sha256"]
        == preflight._sha_json(row["transverse_parent_ids"])
        for row in memberships
    )

    control = json.loads(
        (EVIDENCE / "local_plane_fragment_probe_v2/C001.json").read_text()
    )
    relations, _outcomes, _audit = preflight.build_relation_and_outcome_tables(
        case_token="C001",
        projected=rung_core.project_cross_parent_rung_input(raw_pool),
        control=control,
        baseline=baseline,
    )
    target_pair_ids = {row["baseline_local_plane_pair_id"] for row in relations}
    non_target = next(
        row for row in memberships
        if row["baseline_local_plane_pair_id"] not in target_pair_ids
    )
    tampered = deepcopy(memberships)
    tampered[memberships.index(non_target)]["transverse_parent_ids"].append(
        "P_TAMPERED_NON_TARGET_PAIR"
    )
    assert preflight.derive_pair_membership_predicate(memberships, tampered) is False


def test_exact_target_threshold_and_safety_predicates_reject_tampering() -> None:
    raw_pool = json.loads(
        (EVIDENCE / "candidate_pool_transition_v2_probe/C001.json").read_text()
    )
    control = json.loads(
        (EVIDENCE / "local_plane_fragment_probe_v2/C001.json").read_text()
    )
    baseline = json.loads(
        (EVIDENCE / "v2_complete_same_surface_edge_probe_v1/C001.json").read_text()
    )
    relations, outcomes, _audit = preflight.build_relation_and_outcome_tables(
        case_token="C001",
        projected=rung_core.project_cross_parent_rung_input(raw_pool),
        control=control,
        baseline=baseline,
    )
    frozen = preflight._frozen_target_projection(control, "C001")
    projected = preflight._relation_target_projection(relations)
    exact = preflight.derive_target_preservation_predicates(
        frozen_targets=frozen,
        regenerated_targets=deepcopy(frozen),
        relation_targets=projected,
    )
    assert exact == {
        "target_rows_byte_identical": True,
        "target_ids_equal": True,
        "target_mode_hashes_equal": True,
    }

    id_tamper = deepcopy(projected)
    id_tamper[0]["frozen_target_component_id"] = "LTC_VALID_COUNT_WRONG_ID"
    assert preflight.derive_target_preservation_predicates(
        frozen_targets=frozen,
        regenerated_targets=deepcopy(frozen),
        relation_targets=id_tamper,
    )["target_ids_equal"] is False

    mode_tamper = deepcopy(projected)
    mode_tamper[0]["frozen_target_mode_ids_sha256"] = "f" * 64
    assert preflight.derive_target_preservation_predicates(
        frozen_targets=frozen,
        regenerated_targets=deepcopy(frozen),
        relation_targets=mode_tamper,
    )["target_mode_hashes_equal"] is False

    wrong_thresholds = deepcopy(baseline["thresholds"])
    first_key = sorted(wrong_thresholds)[0]
    wrong_thresholds[first_key] = "0" * 64
    assert preflight.derive_threshold_predicate(
        baseline["thresholds"], wrong_thresholds
    ) is False

    safety_artifact = {
        "gold_open_authorized": False,
        "truth_inputs_consumed": False,
        "ordinary_batch_authorized": False,
        "frozen_target_outcome_table": outcomes,
    }
    assert preflight.derive_mapping_only_safety_predicate(safety_artifact) is True
    safety_artifact["manual_truth"] = []
    assert preflight.derive_mapping_only_safety_predicate(safety_artifact) is False


def test_runner_does_not_trust_core_literal_attestations() -> None:
    source = inspect.getsource(preflight)
    assert "pair_line_tube_membership_unchanged\"] is True" not in source
    assert '"gold_open_authorized_false": True' not in source
    assert "len(relations) == int(counters[\"frozen_target_count\"])" not in source
