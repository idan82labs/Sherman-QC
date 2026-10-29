from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend import f1_train18_candidate2_conservative_deletion_a1_v1 as runtime
from backend.f1_train18_frozen_source_only_candidate_v1 import (
    canonical_bytes,
    feature_names,
)
from scripts import optimize_f1_train18_candidate2_conservative_deletion_a1_v1 as optimizer


ROOT = Path(__file__).resolve().parents[1]


def _row(
    bend_id: str,
    *,
    sidecar: bool = False,
    score: float = 0.5,
    y: float = 0.0,
) -> dict:
    return {
        "bend_id": bend_id,
        "line_start": [0.0, y, 0.0],
        "line_end": [10.0, y, 0.0],
        "selector_score": score,
        "source_kind": (
            runtime.SIDECAR_SOURCE_KIND if sidecar else "plane_lattice"
        ),
        "selection_route": (
            "v41_fragment_tolerant_parent_bridge"
            if sidecar
            else "v16_full_resolution"
        ),
    }


def _source(rows: list[dict]) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "predictions": rows,
    }


def _pool() -> dict:
    return {"working_point_budget": 200000, "candidates": [], "line_objects": []}


def _hashes() -> dict[str, str]:
    return {
        name: f"{index:064x}"
        for index, name in enumerate(runtime.HASH_BINDINGS, 1)
    }


def _explosion_model(rule: str = runtime.EXPLOSION_RULES[0]) -> dict:
    return {
        "schema_version": 1,
        "kind": runtime.MODEL_KIND,
        "attempt_ordinal": 2,
        "engineering_amendment": 1,
        "training_gate_passed": True,
        "train_gate_result_kind": optimizer.GATE_PASS_KIND,
        "train_gate_status": "PASS",
        "holdout_score_opened": False,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "source_only": True,
        "v51_fallback_default": True,
        "deletion_only": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "observed_source_multiplicity_used": True,
        "classifier_count_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "profile_name": "SYNTHETIC_C2",
        "profile": {"policy_mode": "EXPLOSION_ONLY", "explosion_rule": rule},
        "feature_names": None,
        "classifier": None,
        **_hashes(),
    }


def _veto_model(probability: float, threshold: float) -> dict:
    names = feature_names("absolute_quality")
    model = _explosion_model()
    model.update(
        {
            "observed_source_multiplicity_used": False,
            "profile": {
                "policy_mode": "VETO_ONLY",
                "learner": optimizer.LEARNER_SPECS[0].name,
                "representation": "absolute_quality",
                "veto_keep_threshold": threshold,
            },
            "feature_names": list(names),
            "classifier": {
                "imputer_medians": [0.0] * len(names),
                "kind": "constant",
                "positive_probability": probability,
            },
        }
    )
    return model


def _metrics(*, tp: int, fp: int, fn: int, exact: int, matched: int) -> dict:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_case_count": exact,
        "label_support_tiers": {
            f"REVIEWER_SUPPORT_{support}": {"gold": matched, "matched": matched}
            for support in (1, 2, 3)
        },
    }


def test_explosion_drop_all_sidecars_preserves_every_other_row() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    retained, removed = runtime.apply_explosion_guard(
        rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2"
    )
    assert removed == ["s1", "s2"]
    assert canonical_bytes(retained) == canonical_bytes([rows[0]])
    assert canonical_bytes(rows[0]) == canonical_bytes(retained[0])


def test_explosion_does_not_trigger_below_locked_multiplicity() -> None:
    rows = [_row("base"), _row("s1", sidecar=True)]
    retained, removed = runtime.apply_explosion_guard(
        rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2"
    )
    assert removed == []
    assert canonical_bytes(retained) == canonical_bytes(rows)


def test_keep_top1_uses_score_then_stable_input_order() -> None:
    rows = [
        _row("s-first", sidecar=True, score=0.8),
        _row("s-second", sidecar=True, score=0.8),
        _row("base"),
    ]
    retained, removed = runtime.apply_explosion_guard(
        rows, "KEEP_TOP1_SIDECAR_IF_AT_LEAST_2"
    )
    assert [row["bend_id"] for row in retained] == ["s-first", "base"]
    assert removed == ["s-second"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[1].update(bend_id=rows[0]["bend_id"]),
        lambda rows: rows[0].pop("bend_id"),
        lambda rows: rows[0].update(line_end=[0.0, 0.0, 0.0]),
        lambda rows: rows[0].update(selector_score=float("nan")),
        lambda rows: rows[0].update(source_kind=""),
        lambda rows: rows[0].update(selection_route=""),
    ],
)
def test_explosion_rejects_malformed_incumbent_stream(mutator) -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    mutator(rows)
    with pytest.raises(ValueError):
        runtime.apply_explosion_guard(rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2")


def test_non_sidecar_nan_score_is_rejected_before_explosion() -> None:
    rows = [_row("base", score=float("nan")), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    with pytest.raises(ValueError, match="selector_score"):
        runtime.apply_explosion_guard(rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2")


@pytest.mark.parametrize("invalid", [None, "0.5", True, float("inf")])
def test_any_present_invalid_selector_score_is_rejected(invalid) -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    rows[0]["selector_score"] = invalid
    with pytest.raises(ValueError, match="selector_score"):
        runtime.apply_explosion_guard(rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2")


def test_absent_scores_are_valid_for_drop_all_exact_v51_schema() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    for row in rows:
        row.pop("selector_score")
    retained, removed = runtime.apply_explosion_guard(
        rows, "DROP_ALL_SIDECARS_IF_AT_LEAST_2"
    )
    assert [row["bend_id"] for row in retained] == ["base"]
    assert removed == ["s1", "s2"]


def test_absent_non_sidecar_score_is_valid_for_keep_top1() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    rows[0].pop("selector_score")
    retained, _removed = runtime.apply_explosion_guard(
        rows, "KEEP_TOP1_SIDECAR_IF_AT_LEAST_2"
    )
    assert {row["bend_id"] for row in retained} == {"base", "s1"}


def test_keep_top1_rejects_absent_sidecar_score() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    rows[1].pop("selector_score")
    with pytest.raises(ValueError, match="required by KEEP_TOP1"):
        runtime.apply_explosion_guard(rows, "KEEP_TOP1_SIDECAR_IF_AT_LEAST_2")


def test_model_requires_distinct_explicit_gate_pass_and_base_lineage() -> None:
    model = _explosion_model()
    assert model["fit_audit_core_sha256"] != model["train_gate_result_sha256"]
    assert runtime.validate_model(model)["policy_mode"] == "EXPLOSION_ONLY"
    for field in (
        "base_runtime_code_sha256",
        "base_optimizer_code_sha256",
        "label_projection_sha256",
        "train_cache_manifest_sha256",
        "train_source_cache_binding_sha256",
    ):
        broken = deepcopy(model)
        broken.pop(field)
        with pytest.raises(ValueError):
            runtime.validate_model(broken)


def test_failed_or_unbound_train_gate_model_is_not_executable() -> None:
    for field, value in (
        ("training_gate_passed", False),
        ("engineering_amendment", 0),
        ("train_gate_status", "FAIL"),
        ("train_gate_result_kind", "wrong"),
    ):
        model = _explosion_model()
        model[field] = value
        with pytest.raises(ValueError):
            runtime.validate_model(model)


def test_multiplicity_and_forbidden_count_contract_are_separate() -> None:
    model = _explosion_model()
    assert "count_feature_used" not in model
    assert model["observed_source_multiplicity_used"] is True
    for field in (
        "classifier_count_feature_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
    ):
        assert model[field] is False
    broken = deepcopy(model)
    broken["observed_source_multiplicity_used"] = False
    with pytest.raises(ValueError, match="multiplicity"):
        runtime.validate_model(broken)


def test_compose_explosion_output_schema_is_exact_and_not_duplicated() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    result = runtime.compose_candidate2(_source(rows), _pool(), _explosion_model())
    assert set(result) == {
        "schema_version",
        "kind",
        "profile",
        "predictions",
        "candidate_count",
        "source_only",
        "research_only",
        "labels_used",
        "cad_used",
        "drawing_used",
        "part_identity_used",
        "case_id_used",
        "runtime_id_used",
        "filename_used",
        "observed_source_multiplicity_used",
        "classifier_count_feature_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
        "deletion_only",
        "v51_fallback_default",
        "training_gate_passed",
        "holdout_score_opened",
        "frozen_model_sha256",
        "deletion_audit",
    }
    assert [row["bend_id"] for row in result["predictions"]] == ["base"]
    assert result["observed_source_multiplicity_used"] is True
    assert "count_feature_used" not in result


def test_veto_is_incumbent_only_and_deletion_only() -> None:
    rows = [_row("a"), _row("b", y=2.0)]
    dropped = runtime.compose_candidate2(
        _source(rows), _pool(), _veto_model(probability=0.1, threshold=0.2)
    )
    kept = runtime.compose_candidate2(
        _source(rows), _pool(), _veto_model(probability=0.2, threshold=0.2)
    )
    assert dropped["predictions"] == []
    assert canonical_bytes(kept["predictions"]) == canonical_bytes(rows)


def test_fit_and_runtime_share_exact_twelve_decimal_probability_rule() -> None:
    rows = [_row("a")]
    probability = 0.1234567890124
    mapped = runtime.incumbent_probability_map(rows, [probability])
    assert mapped == {"a": 0.123456789012}
    result = runtime.compose_candidate2(
        _source(rows),
        _pool(),
        _veto_model(probability=probability, threshold=0.123456789012),
    )
    assert canonical_bytes(result["predictions"]) == canonical_bytes(rows)


def test_inner_activation_requires_f1_gain_and_all_no_harm_checks() -> None:
    baseline = _metrics(tp=4, fp=6, fn=2, exact=1, matched=1)
    candidate = _metrics(tp=4, fp=3, fn=2, exact=1, matched=1)
    assert optimizer._inner_activation_gate(candidate, baseline)["all_checks_passed"]
    harmed = deepcopy(candidate)
    harmed["label_support_tiers"]["REVIEWER_SUPPORT_2"]["matched"] = 0
    assert not optimizer._inner_activation_gate(harmed, baseline)["all_checks_passed"]


def test_fallback_returns_exact_v51_prediction_science_bytes() -> None:
    rows = [_row("a"), _row("b", y=2.0)]
    case = SimpleNamespace(primary_v51=_source(rows))
    returned = optimizer._apply_profile(case, {"policy_mode": "V51_FALLBACK"})
    assert canonical_bytes(returned) == canonical_bytes(rows)


def test_outer_held_cases_never_enter_arm_selection_or_classifier_fit(monkeypatch) -> None:
    cases = [SimpleNamespace(case_id=f"c{i}", scan_sha256=f"{i:064x}") for i in range(6)]
    monkeypatch.setattr(
        optimizer,
        "case_folds",
        lambda values, count, seed: [[value] for value in values],
    )
    selected_on: list[set[str]] = []
    fitted_on: list[set[str]] = []

    def fake_select(values, *, n_splits, seed):
        selected_on.append({case.case_id for case in values})
        profile = {
            "policy_mode": "VETO_ONLY",
            "learner": optimizer.LEARNER_SPECS[0].name,
            "representation": "absolute_quality",
            "veto_keep_threshold": 0.1,
        }
        return {
            "profile": profile,
            "profile_sha256": optimizer._profile_fingerprint(profile),
        }, [{}]

    def fake_fit(spec, values):
        fit_ids = {case.case_id for case in values}
        fitted_on.append(fit_ids)
        return {"fit_ids": fit_ids}

    def fake_apply(case, profile, *, classifier=None, probabilities=None):
        assert classifier is not None
        assert case.case_id not in classifier["fit_ids"]
        return []

    def fake_score(values, predictions):
        rows = [
            {
                "case_id": case.case_id,
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "exact": True,
            }
            for case in values
        ]
        metrics = {
            **_metrics(tp=len(values), fp=0, fn=0, exact=len(values), matched=1),
            "case_count": len(values),
        }
        return metrics, rows

    monkeypatch.setattr(optimizer, "select_configuration", fake_select)
    monkeypatch.setattr(optimizer, "fit_portable_classifier", fake_fit)
    monkeypatch.setattr(optimizer, "_apply_profile", fake_apply)
    monkeypatch.setattr(optimizer, "score_predictions", fake_score)
    monkeypatch.setattr(
        optimizer,
        "baseline_predictions",
        lambda values: {case.case_id: [] for case in values},
    )
    monkeypatch.setattr(
        optimizer,
        "training_gate",
        lambda candidate, baseline: {
            "all_conjunctive_checks_passed": True,
            "checks": {},
        },
    )
    report = optimizer.nested_outer_evaluation(cases)
    assert report["all_18_cases_scored_exactly_once_outer_held_out"]
    assert len(selected_on) == len(fitted_on) == 6
    assert all(len(ids) == 5 for ids in selected_on + fitted_on)


def test_deployed_replay_failure_returns_terminalizable_evidence(monkeypatch) -> None:
    case = SimpleNamespace(
        case_id="c1",
        primary_v51={},
        primary_pool={},
        replay_v51={},
        replay_pool={},
    )
    monkeypatch.setattr(
        optimizer,
        "compose_candidate2",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic")),
    )
    result = optimizer.evaluate_deployed_replay([case], {}, {"c1": []})
    assert result == {
        "evaluated": True,
        "execution_succeeded": False,
        "primary_replay_bytes_exact": False,
        "deployed_model_matches_evaluated_predictions": False,
    }


def test_failed_gate_publication_cannot_emit_holdout_eligible_artifacts(tmp_path) -> None:
    names = optimizer.publication_artifact_names(False)
    assert not {
        "train_gate_pass.json",
        "frozen_model.json",
        "freeze_lock.json",
        "pre_holdout_seal.json",
    }.intersection(names)
    payloads = {name: {"name": name} for name in names}
    optimizer.publish_artifacts(
        tmp_path / "failed", training_gate_passed=False, payloads=payloads
    )
    assert {path.name for path in (tmp_path / "failed").iterdir()} == set(names)
    bad = dict(payloads)
    bad["frozen_model.json"] = {"training_gate_passed": False}
    with pytest.raises(ValueError):
        optimizer.publish_artifacts(
            tmp_path / "bad", training_gate_passed=False, payloads=bad
        )
    assert not (tmp_path / "bad").exists()


def test_passing_publication_names_bind_gate_model_lock_and_preseal(tmp_path) -> None:
    names = optimizer.publication_artifact_names(True)
    assert {
        "train_gate_pass.json",
        "frozen_model.json",
        "freeze_lock.json",
        "pre_holdout_seal.json",
    }.issubset(names)
    payloads = {name: {"name": name, "training_gate_passed": True} for name in names}
    optimizer.publish_artifacts(
        tmp_path / "pass", training_gate_passed=True, payloads=payloads
    )
    assert {path.name for path in (tmp_path / "pass").iterdir()} == set(names)


def test_feature_contract_has_no_identity_cad_or_classifier_count_features() -> None:
    for representation in ("absolute_quality", "case_relative_quality", "intrinsic_quality"):
        names = feature_names(representation)
        assert not any(
            token in feature.lower()
            for feature in names
            for token in ("case_id", "runtime_id", "filename", "cad", "count")
        )


def test_design_lock_discloses_multiplicity_and_exact_nested_contract() -> None:
    lock = optimizer.design_lock()
    assert lock["attempt_ordinal"] == 2
    assert lock["engineering_amendment"] == 1
    assert lock["a0_engineering_abort_sha256"] == optimizer.CANDIDATE2_A0_ABORT_SHA256
    assert lock["adaptive_train_development"] is True
    assert lock["explosion_source_contract"]["observed_source_multiplicity_used"] is True
    assert lock["explosion_source_contract"]["expected_count_used"] is False
    assert lock["folds"] == {
        "outer": 6,
        "inner": 3,
        "final_selection": 6,
        "seed": optimizer.FOLD_SEED,
        "group": "COMPLETE_CASE",
    }
    assert lock["publication"]["train_gate_false"].find("NO_MODEL") >= 0
    assert lock["holdout_score_opened"] is False
    assert lock["holdout_paths_available"] is False


def test_candidate1_terminal_and_imported_sources_remain_immutable() -> None:
    expected = {
        "research/f1_aug30_train18_candidate1_terminal_negative_v1.json": "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8",
        "backend/f1_train18_frozen_source_only_candidate_v1.py": "85aba64cb7c088ee281344c675e3dd1578ac643a744cdc2b84da1967b6a11288",
        "scripts/optimize_f1_train18_source_only_v1.py": "51f8e8baa3b55535dc05a7d87d62ca553d0128ac2f1258fc092ee8ae986e98ab",
    }
    for relative, sha in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha
    terminal_bytes = optimizer.CANDIDATE1_TERMINAL.read_bytes()
    assert canonical_bytes(json.loads(terminal_bytes)) == terminal_bytes


def test_a0_abort_is_canonical_non_evidentiary_and_truthful() -> None:
    abort_bytes = optimizer.CANDIDATE2_A0_ABORT.read_bytes()
    abort = json.loads(abort_bytes)
    assert canonical_bytes(abort) == abort_bytes
    assert hashlib.sha256(abort_bytes).hexdigest() == optimizer.CANDIDATE2_A0_ABORT_SHA256
    assert abort["status"] == "NONREPRODUCIBLE_ENGINEERING_ABORT"
    assert abort["source_bytes_status"] == "A0_SOURCE_BYTES_NOT_MATERIALIZED"
    assert abort["train_label_projection_opened"] is True
    assert abort["train_cache_inputs_opened"] is True
    assert abort["train_v51_reference_baseline_computed_in_memory"] is True
    assert abort["candidate_metric_computed"] is False
    assert abort["candidate_comparison_performed"] is False
    assert abort["output_created"] is False
    assert abort["holdout_access_count"] == 0


def test_actual_candidate2_preregistration_is_canonical_and_matches_code() -> None:
    prereg_bytes = optimizer.CANDIDATE2_PREREGISTRATION.read_bytes()
    prereg = json.loads(prereg_bytes)
    assert canonical_bytes(prereg) == prereg_bytes
    optimizer.validate_candidate2_preregistration(
        prereg,
        lock=optimizer.load_lock_context(),
        label_attestation_sha256=optimizer.LABEL_ATTESTATION_SHA256,
        cache_attestation_sha256=optimizer.CACHE_ATTESTATION_SHA256,
    )
    for key in (
        "candidate2_a0_engineering_abort",
        "candidate2_a0_preregistration",
        "candidate1_base_runtime",
        "candidate1_base_optimizer",
        "candidate2_runtime",
        "candidate2_optimizer",
        "candidate2_tests",
    ):
        binding = prereg["artifact_bindings"][key]
        data = (ROOT / binding["path"]).read_bytes()
        assert len(data) == binding["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == binding["sha256"]
