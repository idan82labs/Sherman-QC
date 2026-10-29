#!/usr/bin/env python3
"""Attempt-2 TRAIN18 optimizer for a conservative deletion-only V51 successor.

This truth-aware process inherits the audited TRAIN-only custody loader from
attempt 1, but has a new preregistration, runtime, optimizer, test, output, and
read-ledger lineage.  It never has a path or capability for a holdout label,
prediction, status, or score artifact.

Publication is fail closed.  A failed nested TRAIN gate produces a terminal
negative and no model, freeze lock, gate-PASS artifact, or pre-holdout seal.
Only a passing nested gate plus exact primary/replay inference can produce one
locally sealed candidate, still awaiting an independent external anchor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_train18_candidate2_conservative_deletion_v1 import (  # noqa: E402
    EXPLOSION_RULES,
    MODEL_KIND,
    POLICY_MODES,
    PREREG_KIND,
    apply_explosion_guard,
    compose_candidate2,
    incumbent_probability_map,
    validate_incumbent_predictions,
)
from backend.f1_train18_frozen_source_only_candidate_v1 import (  # noqa: E402
    canonical_bytes,
    classifier_probabilities,
    feature_names,
    sha256_bytes,
)
from scripts.optimize_f1_train18_source_only_v1 import (  # noqa: E402
    CaseData,
    GuardedReader,
    LearnerSpec,
    LockContext,
    baseline_predictions,
    build_guarded_reader,
    case_folds,
    fit_portable_classifier,
    load_lock_context,
    load_train_cases,
    prepare_view,
    score_predictions,
    training_gate,
)


CANDIDATE1_TERMINAL = ROOT / "research/f1_aug30_train18_candidate1_terminal_negative_v1.json"
CANDIDATE2_PREREGISTRATION = ROOT / "research/f1_aug30_train18_candidate2_preregistration_v1.json"
BASE_RUNTIME_SOURCE = ROOT / "backend/f1_train18_frozen_source_only_candidate_v1.py"
BASE_OPTIMIZER_SOURCE = ROOT / "scripts/optimize_f1_train18_source_only_v1.py"
RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate2_conservative_deletion_v1.py"
OPTIMIZER_SOURCE = Path(__file__).resolve()
TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate2_conservative_deletion_v1.py"
DEFAULT_OUTPUT_DIR = ROOT / "output/research/f1-aug30-pseudo-unseen-train18-candidate2-v1"

CANDIDATE1_TERMINAL_SHA256 = "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
LABEL_ATTESTATION_SHA256 = "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
CACHE_ATTESTATION_SHA256 = "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"

TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate_terminal_v1"
GATE_PASS_KIND = "sherman_qc_f1_aug30_train18_candidate2_train_gate_pass_v1"
FIT_AUDIT_CORE_KIND = "sherman_qc_f1_aug30_train18_candidate2_nested_fit_audit_core_v1"
FIT_AUDIT_KIND = "sherman_qc_f1_aug30_train18_candidate2_nested_fit_audit_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate2_exact_read_ledger_v1"

OUTER_FOLDS = 6
INNER_FOLDS = 3
FINAL_SELECTION_FOLDS = 6
FOLD_SEED = "f1-aug30-train18-candidate2-nested-case-groups-v1"
CLASSIFIER_RANDOM_SEED = 1701
VETO_KEEP_THRESHOLDS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65)

LEARNER_SPECS = (
    LearnerSpec(
        "C2_INCUMBENT_ABSOLUTE_LOGISTIC_C025",
        "incumbent_only",
        "absolute_quality",
        "logistic",
        {"c": 0.25},
    ),
    LearnerSpec(
        "C2_INCUMBENT_ABSOLUTE_TREE_D3_L2",
        "incumbent_only",
        "absolute_quality",
        "tree",
        {"max_depth": 3, "min_samples_leaf": 2},
    ),
    LearnerSpec(
        "C2_INCUMBENT_RELATIVE_LOGISTIC_C05",
        "incumbent_only",
        "case_relative_quality",
        "logistic",
        {"c": 0.5},
    ),
    LearnerSpec(
        "C2_INCUMBENT_RELATIVE_FOREST_D5_L2",
        "incumbent_only",
        "case_relative_quality",
        "forest",
        {"max_depth": 5, "min_samples_leaf": 2},
    ),
    LearnerSpec(
        "C2_INCUMBENT_INTRINSIC_LOGISTIC_C05",
        "incumbent_only",
        "intrinsic_quality",
        "logistic",
        {"c": 0.5},
    ),
    LearnerSpec(
        "C2_INCUMBENT_INTRINSIC_TREE_D3_L2",
        "incumbent_only",
        "intrinsic_quality",
        "tree",
        {"max_depth": 3, "min_samples_leaf": 2},
    ),
)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _lower_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _learner_lock_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "universe": spec.universe,
            "representation": spec.representation,
            "classifier_kind": spec.classifier_kind,
            "classifier_parameters": dict(spec.classifier_parameters),
        }
        for spec in LEARNER_SPECS
    ]


def design_lock() -> dict[str, Any]:
    """Return the exact preregistered search and publication contract."""

    return {
        "attempt_ordinal": 2,
        "adaptive_train_development": True,
        "base_policy": "FROZEN_V51_EXACT_FALLBACK",
        "candidate_family": "DELETION_ONLY_CONSERVATIVE_SUCCESSOR",
        "policy_modes": list(POLICY_MODES),
        "explosion_rules": list(EXPLOSION_RULES),
        "explosion_source_contract": {
            "source_kind_exact": "v41_fragment_tolerant_parent_bridge_sidecar",
            "selection_route_contains": "fragment_tolerant_parent_bridge",
            "drop_all_thresholds": [2, 3],
            "keep_top1_threshold": 2,
            "top1_order": ["selector_score_descending", "input_order_ascending"],
            "selector_score_schema": {
                "absent_permitted": "ALL_ROWS_EXCEPT_KEEP_TOP1_SIDECARS",
                "present_value_must_be_finite_numeric": True,
                "keep_top1_sidecar_value_required": True,
            },
            "observed_source_multiplicity_used": True,
            "expected_count_used": False,
            "sender_count_used": False,
            "gold_count_used": False,
            "label_count_used_at_inference": False,
        },
        "veto_keep_thresholds": list(VETO_KEEP_THRESHOLDS),
        "learner_specs": _learner_lock_rows(),
        "classifier_random_seed": CLASSIFIER_RANDOM_SEED,
        "folds": {
            "outer": OUTER_FOLDS,
            "inner": INNER_FOLDS,
            "final_selection": FINAL_SELECTION_FOLDS,
            "seed": FOLD_SEED,
            "group": "COMPLETE_CASE",
        },
        "inner_activation_gate": {
            "micro_f1_strict_improvement": True,
            "true_positive_nondecrease": True,
            "false_positive_nonincrease": True,
            "exact_case_count_nondecrease": True,
            "every_label_support_tier_tp_nondecrease": True,
            "otherwise": "EXACT_V51_FALLBACK",
        },
        "final_conjunctive_gate": {
            "nested_outer_micro_f1_delta_at_least": 0.05,
            "full_train_oof_micro_f1_delta_at_least": 0.05,
            "true_positive_nondecrease": True,
            "false_positive_increase_bound": "MAX_2_OR_10_PERCENT_OF_V51_FP",
            "exact_case_count_nondecrease": True,
            "every_label_support_tier_tp_nondecrease": True,
            "primary_replay_bytes_exact": True,
            "replay_execution_succeeded": True,
            "deployed_model_matches_evaluated_predictions": True,
        },
        "feature_contract": {
            "source_only": True,
            "case_id_feature_used": False,
            "runtime_id_feature_used": False,
            "filename_feature_used": False,
            "cad_feature_used": False,
            "classifier_count_feature_used": False,
            "observed_source_multiplicity_policy_control_possible": True,
            "expected_count_used": False,
            "sender_count_used": False,
            "gold_count_used": False,
            "label_count_used_at_inference": False,
        },
        "outer_held_case_rule": (
            "ARM_AND_THRESHOLD_SELECTED_ONLY_FROM_OUTER_TRAIN_INNER_OOF;"
            "HELD_CASE_TRUTH_NEVER_AVAILABLE_TO_CHOOSER"
        ),
        "retained_line_contract": "CANONICAL_PREDICTION_OBJECT_BYTE_IDENTICAL_TO_V51",
        "publication": {
            "train_gate_false": (
                "FIT_AUDIT_LEDGER_AND_TERMINAL_NEGATIVE_ONLY;NO_MODEL;NO_GATE_PASS;"
                "NO_FREEZE_LOCK;NO_PRE_HOLDOUT_SEAL;REPLAY_INTEGRITY_FAILURE_IS_TERMINAL"
            ),
            "train_gate_true": (
                "ONE_MODEL_PLUS_DISTINCT_GATE_PASS_PLUS_FREEZE_LOCK_PLUS_LOCAL_PRESEAL;"
                "INDEPENDENT_EXTERNAL_ANCHOR_REQUIRED"
            ),
        },
        "holdout_score_opened": False,
        "holdout_paths_available": False,
    }


def _required_binding(
    prereg: Mapping[str, Any], key: str, *, path: Path | None = None
) -> dict[str, Any]:
    bindings = prereg.get("artifact_bindings")
    if not isinstance(bindings, dict) or not isinstance(bindings.get(key), dict):
        raise ValueError(f"candidate2 preregistration missing binding: {key}")
    row = dict(bindings[key])
    _lower_sha(row.get("sha256"), name=f"{key} SHA")
    size = row.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"candidate2 preregistration has invalid {key} size")
    declared = row.get("path")
    if not isinstance(declared, str) or not declared or Path(declared).is_absolute():
        raise ValueError(f"candidate2 preregistration has invalid {key} path")
    if path is not None and (ROOT / declared).resolve() != path.resolve():
        raise ValueError(f"candidate2 preregistration {key} path mismatch")
    return row


def validate_candidate2_preregistration(
    prereg: Mapping[str, Any],
    *,
    lock: LockContext,
    label_attestation_sha256: str,
    cache_attestation_sha256: str,
) -> None:
    if prereg.get("schema_version") != 1 or prereg.get("kind") != PREREG_KIND:
        raise ValueError("candidate2 preregistration schema/kind mismatch")
    if prereg.get("status") != "SEALED_BEFORE_CANDIDATE2_TRAIN_FIT":
        raise ValueError("candidate2 preregistration is not sealed pre-fit")
    if prereg.get("attempt_ordinal") != 2 or prereg.get("adaptive_train_development") is not True:
        raise ValueError("candidate2 attempt/development disclosure mismatch")
    if prereg.get("holdout_score_opened") is not False or prereg.get("holdout_paths_available") is not False:
        raise ValueError("candidate2 preregistration does not keep holdout closed")
    if prereg.get("design_lock") != design_lock():
        raise ValueError("candidate2 preregistered search design differs from code")

    exact = {
        "candidate1_terminal": (CANDIDATE1_TERMINAL, CANDIDATE1_TERMINAL_SHA256),
        "candidate1_base_runtime": (BASE_RUNTIME_SOURCE, None),
        "candidate1_base_optimizer": (BASE_OPTIMIZER_SOURCE, None),
        "candidate2_runtime": (RUNTIME_SOURCE, None),
        "candidate2_optimizer": (OPTIMIZER_SOURCE, None),
        "candidate2_tests": (TESTS_SOURCE, None),
        "sealed_split": (ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json", lock.split_sha256),
        "train_allowlist": (ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json", lock.allowlist_sha256),
        "original_preregistration": (ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json", lock.preregistration_sha256),
        "custody_label_attestation": (
            ROOT / "output/research/f1-aug30-train-only-custody-v2/labels/train_label_projection_attestation.json",
            label_attestation_sha256,
        ),
        "custody_cache_attestation": (
            ROOT / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json",
            cache_attestation_sha256,
        ),
        "custody_staged_validator_pass": (
            ROOT / "output/research/f1-aug30-train-only-custody-v2/independent_validation_report.json",
            CUSTODY_PASS_SHA256,
        ),
    }
    for key, (path, expected_sha) in exact.items():
        row = _required_binding(prereg, key, path=path)
        if expected_sha is not None and row["sha256"] != expected_sha:
            raise ValueError(f"candidate2 preregistration {key} SHA mismatch")
    if set(prereg.get("artifact_bindings") or {}) != set(exact):
        raise ValueError("candidate2 preregistration has an unexpected binding set")


def prepare_guarded_inputs(
    *,
    lock: LockContext,
    preregistration: Path,
    preregistration_sha256: str,
    label_attestation: Path,
    label_attestation_sha256: str,
    cache_attestation: Path,
    cache_attestation_sha256: str,
) -> tuple[GuardedReader, dict[str, Any], dict[str, str]]:
    """Read candidate2's label-free lock and source bindings before labels."""

    reader = build_guarded_reader(
        lock,
        label_attestation=label_attestation,
        label_attestation_sha256=label_attestation_sha256,
        cache_attestation=cache_attestation,
        cache_attestation_sha256=cache_attestation_sha256,
    )
    preregistration_sha256 = _lower_sha(
        preregistration_sha256, name="external candidate2 preregistration SHA"
    )
    reader.authorize(
        preregistration,
        purpose="CANDIDATE2_PREREGISTRATION",
        expected_sha256=preregistration_sha256,
    )
    prereg_bytes = reader.read_bytes(
        preregistration, purpose="CANDIDATE2_PREREGISTRATION"
    )
    prereg = json.loads(prereg_bytes)
    if not isinstance(prereg, dict) or canonical_bytes(prereg) != prereg_bytes:
        raise ValueError("candidate2 preregistration must be canonical JSON")
    validate_candidate2_preregistration(
        prereg,
        lock=lock,
        label_attestation_sha256=label_attestation_sha256,
        cache_attestation_sha256=cache_attestation_sha256,
    )

    # The inherited loader already grants these two exact paths for its own
    # phase-3 source reads.  Add the candidate2 preregistered hash and size
    # expectations before that phase, without changing their read purpose.
    for key, path in (
        ("candidate1_base_runtime", BASE_RUNTIME_SOURCE),
        ("candidate1_base_optimizer", BASE_OPTIMIZER_SOURCE),
    ):
        row = _required_binding(prereg, key, path=path)
        resolved = path.resolve()
        if resolved not in reader.allowed:
            raise RuntimeError(f"inherited loader did not authorize {key}")
        reader.expected_sha256[resolved] = str(row["sha256"])
        reader.expected_size_bytes[resolved] = int(row["size_bytes"])

    source_keys = {
        "candidate1_terminal": "CANDIDATE1_TERMINAL_NEGATIVE",
        "candidate2_runtime": "CANDIDATE2_RUNTIME_SOURCE",
        "candidate2_optimizer": "CANDIDATE2_OPTIMIZER_SOURCE",
        "candidate2_tests": "CANDIDATE2_TESTS_SOURCE",
    }
    hashes: dict[str, str] = {"candidate2_preregistration": preregistration_sha256}
    source_payloads: dict[str, bytes] = {}
    for key, purpose in source_keys.items():
        row = _required_binding(prereg, key)
        path = (ROOT / row["path"]).resolve()
        reader.authorize(
            path,
            purpose=purpose,
            expected_sha256=str(row["sha256"]),
            expected_size_bytes=int(row["size_bytes"]),
        )
        data = reader.read_bytes(path, purpose=purpose)
        source_payloads[key] = data
        hashes[key] = sha256_bytes(data)
    terminal_bytes = source_payloads["candidate1_terminal"]
    terminal = json.loads(terminal_bytes)
    if (
        not isinstance(terminal, dict)
        or canonical_bytes(terminal) != terminal_bytes
        or terminal.get("kind") != TERMINAL_KIND
        or terminal.get("candidate_id") != 1
        or terminal.get("status") != "REJECTED_TRAIN_GATE"
        or terminal.get("nested_training_retain_gate_passed") is not False
        or terminal.get("holdout_score_opened") is not False
        or terminal.get("holdout_execution_authorized") is not False
        or terminal.get("pre_holdout_execution_anchor_created") is not False
        or terminal.get("local_pre_holdout_seal_disposition") != "INVALID_NOT_AUTHORIZING"
    ):
        raise ValueError("candidate1 terminal-negative lineage is invalid")
    return reader, dict(prereg), hashes


def _spec(name: str) -> LearnerSpec:
    matches = [spec for spec in LEARNER_SPECS if spec.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate2 learner: {name}")
    return matches[0]


def _profile_fingerprint(profile: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(profile))


def _exact_f1(metrics: Mapping[str, Any]) -> float:
    tp = int(metrics["true_positive"])
    fp = int(metrics["false_positive"])
    fn = int(metrics["false_negative"])
    return 2.0 * tp / (2.0 * tp + fp + fn) if tp else 0.0


def _inner_activation_gate(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    tier_checks = {
        tier: int(candidate["label_support_tiers"][tier]["matched"])
        >= int(baseline["label_support_tiers"][tier]["matched"])
        for tier in baseline["label_support_tiers"]
    }
    checks = {
        "micro_f1_strict_improvement": _exact_f1(candidate) > _exact_f1(baseline) + 1e-12,
        "true_positive_nondecrease": int(candidate["true_positive"]) >= int(baseline["true_positive"]),
        "false_positive_nonincrease": int(candidate["false_positive"]) <= int(baseline["false_positive"]),
        "exact_case_count_nondecrease": int(candidate["exact_case_count"]) >= int(baseline["exact_case_count"]),
        "every_label_support_tier_tp_nondecrease": all(tier_checks.values()),
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "label_support_tier_checks": tier_checks,
    }


def _oof_probabilities(
    cases: Sequence[CaseData], spec: LearnerSpec, *, n_splits: int, seed: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    probabilities: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(case_folds(cases, n_splits, seed=seed), 1):
        held_ids = {case.case_id for case in held_out}
        fit_cases = [case for case in cases if case.case_id not in held_ids]
        classifier = fit_portable_classifier(spec, fit_cases)
        for case in held_out:
            view = prepare_view(case, spec)
            probabilities[case.case_id] = classifier_probabilities(view.matrix, classifier)
        audit.append(
            {
                "fold": fold_index,
                "fit_case_count": len(fit_cases),
                "held_out_case_count": len(held_out),
                "fit_scan_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in fit_cases))
                ),
                "held_out_scan_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in held_out))
                ),
            }
        )
    if set(probabilities) != {case.case_id for case in cases}:
        raise RuntimeError("candidate2 OOF probabilities do not cover complete cases")
    return probabilities, audit


def _apply_profile(
    case: CaseData,
    profile: Mapping[str, Any],
    *,
    classifier: Mapping[str, Any] | None = None,
    probabilities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mode = str(profile["policy_mode"])
    original = validate_incumbent_predictions(
        case.primary_v51.get("predictions"),
        require_sidecar_selector_score=(
            mode in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO")
            and profile.get("explosion_rule") == "KEEP_TOP1_SIDECAR_IF_AT_LEAST_2"
        ),
    )
    if mode == "V51_FALLBACK":
        return original
    rows = original
    if mode in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO"):
        rows, _removed = apply_explosion_guard(rows, str(profile["explosion_rule"]))
    if mode in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
        spec = _spec(str(profile["learner"]))
        view = prepare_view(case, spec)
        if probabilities is None:
            if classifier is None:
                raise ValueError("candidate2 veto requires classifier or OOF probabilities")
            probabilities = classifier_probabilities(view.matrix, classifier)
        probability_by_id = incumbent_probability_map(original, probabilities)
        threshold = float(profile["veto_keep_threshold"])
        rows = [row for row in rows if probability_by_id[str(row["bend_id"])] >= threshold]
    original_bytes = {
        str(row["bend_id"]): canonical_bytes(row) for row in original
    }
    if any(original_bytes.get(str(row["bend_id"])) != canonical_bytes(row) for row in rows):
        raise RuntimeError("candidate2 changed a retained V51 prediction")
    return rows


def _trial(
    cases: Sequence[CaseData],
    baseline: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    probabilities: Mapping[str, np.ndarray] | None = None,
    fold_audit: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    predictions = {
        case.case_id: _apply_profile(
            case,
            profile,
            probabilities=None if probabilities is None else probabilities[case.case_id],
        )
        for case in cases
    }
    metrics, rows = score_predictions(cases, predictions)
    return {
        "profile": dict(profile),
        "profile_sha256": _profile_fingerprint(profile),
        "metrics": metrics,
        "activation_gate": _inner_activation_gate(metrics, baseline),
        "case_group_fold_audit": [dict(row) for row in fold_audit],
        "scored_complete_case_count": len(rows),
    }


def select_configuration(
    cases: Sequence[CaseData], *, n_splits: int, seed: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline, _ = score_predictions(cases, baseline_predictions(cases))
    trials: list[dict[str, Any]] = []
    for rule in EXPLOSION_RULES:
        trials.append(
            _trial(
                cases,
                baseline,
                {"policy_mode": "EXPLOSION_ONLY", "explosion_rule": rule},
            )
        )
    for spec in LEARNER_SPECS:
        probabilities, fold_audit = _oof_probabilities(
            cases, spec, n_splits=n_splits, seed=f"{seed}|{spec.name}"
        )
        for threshold in VETO_KEEP_THRESHOLDS:
            base = {
                "learner": spec.name,
                "representation": spec.representation,
                "veto_keep_threshold": threshold,
            }
            trials.append(
                _trial(
                    cases,
                    baseline,
                    {"policy_mode": "VETO_ONLY", **base},
                    probabilities=probabilities,
                    fold_audit=fold_audit,
                )
            )
            for rule in EXPLOSION_RULES:
                trials.append(
                    _trial(
                        cases,
                        baseline,
                        {
                            "policy_mode": "EXPLOSION_THEN_VETO",
                            "explosion_rule": rule,
                            **base,
                        },
                        probabilities=probabilities,
                        fold_audit=fold_audit,
                    )
                )
    eligible = [row for row in trials if row["activation_gate"]["all_checks_passed"]]
    if not eligible:
        selected = {
            "profile": {"policy_mode": "V51_FALLBACK"},
            "profile_sha256": _profile_fingerprint({"policy_mode": "V51_FALLBACK"}),
            "metrics": baseline,
            "activation_gate": {
                "all_checks_passed": False,
                "fallback_reason": "NO_PREREGISTERED_PROFILE_PASSED_INNER_ACTIVATION_GATE",
            },
            "case_group_fold_audit": [],
            "scored_complete_case_count": len(cases),
        }
    else:
        selected = max(
            eligible,
            key=lambda row: (
                _exact_f1(row["metrics"]),
                int(row["metrics"]["true_positive"]),
                int(row["metrics"]["exact_case_count"]),
                -int(row["metrics"]["false_positive"]),
                row["profile_sha256"],
            ),
        )
    trials.sort(
        key=lambda row: (
            -int(row["activation_gate"]["all_checks_passed"]),
            -_exact_f1(row["metrics"]),
            -int(row["metrics"]["true_positive"]),
            int(row["metrics"]["false_positive"]),
            row["profile_sha256"],
        )
    )
    return deepcopy(selected), trials


def _case_change_counts(
    candidate_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    base = {str(row["case_id"]): row for row in baseline_rows}
    gain = harm = equal = 0
    for row in candidate_rows:
        delta = _exact_f1(row) - _exact_f1(base[str(row["case_id"])])
        if delta > 1e-12:
            gain += 1
        elif delta < -1e-12:
            harm += 1
        else:
            equal += 1
    return {"gain_case_count": gain, "harm_case_count": harm, "equal_case_count": equal}


def nested_outer_evaluation(cases: Sequence[CaseData]) -> dict[str, Any]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    fold_reports: list[dict[str, Any]] = []
    selected_profiles: list[str] = []
    folds = case_folds(cases, OUTER_FOLDS, seed=f"{FOLD_SEED}|outer")
    for fold_index, held_out in enumerate(folds, 1):
        held_ids = {case.case_id for case in held_out}
        train = [case for case in cases if case.case_id not in held_ids]
        selected, trials = select_configuration(
            train,
            n_splits=INNER_FOLDS,
            seed=f"{FOLD_SEED}|outer-{fold_index}|inner",
        )
        profile = selected["profile"]
        classifier = None
        if profile["policy_mode"] in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
            classifier = fit_portable_classifier(_spec(str(profile["learner"])), train)
        held_predictions = {
            case.case_id: _apply_profile(case, profile, classifier=classifier)
            for case in held_out
        }
        predictions.update(held_predictions)
        held_metrics, held_rows = score_predictions(held_out, held_predictions)
        held_baseline, held_baseline_rows = score_predictions(
            held_out, baseline_predictions(held_out)
        )
        selected_profiles.append(str(selected["profile_sha256"]))
        fold_reports.append(
            {
                "outer_fold": fold_index,
                "train_case_count": len(train),
                "held_out_case_count": len(held_out),
                "train_scan_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in train))
                ),
                "held_out_scan_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in held_out))
                ),
                "inner_selected_configuration": selected,
                "inner_trial_count": len(trials),
                "outer_held_candidate_metrics": held_metrics,
                "outer_held_v51_metrics": held_baseline,
                "outer_held_case_change_counts": _case_change_counts(
                    held_rows, held_baseline_rows
                ),
            }
        )
    if set(predictions) != {case.case_id for case in cases}:
        raise RuntimeError("candidate2 outer predictions do not cover every case exactly once")
    candidate, candidate_rows = score_predictions(cases, predictions)
    baseline, baseline_rows = score_predictions(cases, baseline_predictions(cases))
    stability = Counter(selected_profiles)
    return {
        "method": "NESTED_COMPLETE_CASE_GROUP_VALIDATION",
        "outer_fold_count": OUTER_FOLDS,
        "inner_fold_count": INNER_FOLDS,
        "all_18_cases_scored_exactly_once_outer_held_out": len(candidate_rows) == len(cases),
        "candidate_metrics": candidate,
        "v51_reference_metrics": baseline,
        "training_retain_gate": training_gate(candidate, baseline),
        "case_change_counts": _case_change_counts(candidate_rows, baseline_rows),
        "selection_stability": {
            "distinct_profile_count": len(stability),
            "profile_selection_counts": dict(sorted(stability.items())),
            "fallback_fold_count": sum(
                row["inner_selected_configuration"]["profile"]["policy_mode"] == "V51_FALLBACK"
                for row in fold_reports
            ),
        },
        "folds": fold_reports,
    }


def _final_profile_predictions(
    cases: Sequence[CaseData], profile: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any] | None]:
    classifier = None
    if profile["policy_mode"] in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
        classifier = fit_portable_classifier(_spec(str(profile["learner"])), cases)
    return (
        {
            case.case_id: _apply_profile(case, profile, classifier=classifier)
            for case in cases
        },
        classifier,
    )


def publication_artifact_names(training_gate_passed: bool) -> tuple[str, ...]:
    common = ("fit_audit_core.json", "fit_audit.json", "train_read_ledger.json")
    if not training_gate_passed:
        return (*common, "terminal_negative.json")
    return (
        *common,
        "train_gate_pass.json",
        "frozen_model.json",
        "freeze_lock.json",
        "pre_holdout_seal.json",
    )


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))


def publish_artifacts(
    output: Path, *, training_gate_passed: bool, payloads: Mapping[str, Mapping[str, Any]]
) -> None:
    expected = set(publication_artifact_names(training_gate_passed))
    if set(payloads) != expected:
        raise ValueError("candidate2 publication payload set violates gate branch")
    forbidden = {"train_gate_pass.json", "frozen_model.json", "freeze_lock.json", "pre_holdout_seal.json"}
    if not training_gate_passed and forbidden.intersection(payloads):
        raise RuntimeError("failed TRAIN gate cannot publish holdout-eligible artifacts")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("candidate2 output path must be new")
    staging = output.with_name(f"{output.name}.staging-candidate2")
    if staging.exists():
        raise FileExistsError("candidate2 staging path already exists")
    for name in publication_artifact_names(training_gate_passed):
        _write_new(staging / name, payloads[name])
    staging.replace(output)


def _build_model(
    *,
    profile: Mapping[str, Any],
    classifier: Mapping[str, Any] | None,
    prereg: Mapping[str, Any],
    preregistration_sha256: str,
    fit_audit_core_sha256: str,
    train_gate_result_sha256: str,
    label_projection_sha256: str,
    label_attestation_sha256: str,
    cache_manifest_sha256: str,
    cache_attestation_sha256: str,
    train_source_cache_binding_sha256: str,
) -> dict[str, Any]:
    mode = str(profile["policy_mode"])
    if mode not in POLICY_MODES:
        raise ValueError("a V51 fallback cannot become an executable candidate2 model")
    bindings = prereg["artifact_bindings"]
    representation = profile.get("representation")
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "model_name": "F1_AUG30_TRAIN18_CANDIDATE2_CONSERVATIVE_DELETION_V1",
        "attempt_ordinal": 2,
        "training_gate_passed": True,
        "train_gate_result_kind": GATE_PASS_KIND,
        "train_gate_status": "PASS",
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "source_only": True,
        "v51_fallback_default": True,
        "deletion_only": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "observed_source_multiplicity_used": mode in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO"),
        "classifier_count_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "profile_name": "F1_AUG30_TRAIN18_CANDIDATE2_V1",
        "profile": dict(profile),
        "feature_names": None if representation is None else list(feature_names(str(representation))),
        "classifier": None if classifier is None else dict(classifier),
        "candidate1_terminal_sha256": bindings["candidate1_terminal"]["sha256"],
        "candidate2_preregistration_sha256": preregistration_sha256,
        "split_sha256": bindings["sealed_split"]["sha256"],
        "train_allowlist_sha256": bindings["train_allowlist"]["sha256"],
        "original_preregistration_sha256": bindings["original_preregistration"]["sha256"],
        "label_attestation_sha256": label_attestation_sha256,
        "cache_attestation_sha256": cache_attestation_sha256,
        "custody_pass_sha256": bindings["custody_staged_validator_pass"]["sha256"],
        "base_runtime_code_sha256": bindings["candidate1_base_runtime"]["sha256"],
        "base_optimizer_code_sha256": bindings["candidate1_base_optimizer"]["sha256"],
        "runtime_code_sha256": bindings["candidate2_runtime"]["sha256"],
        "optimizer_code_sha256": bindings["candidate2_optimizer"]["sha256"],
        "tests_code_sha256": bindings["candidate2_tests"]["sha256"],
        "fit_audit_core_sha256": fit_audit_core_sha256,
        "train_gate_result_sha256": train_gate_result_sha256,
        "label_projection_sha256": label_projection_sha256,
        "train_cache_manifest_sha256": cache_manifest_sha256,
        "train_source_cache_binding_sha256": train_source_cache_binding_sha256,
        "post_holdout_mutation_policy": "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT",
    }


def _read_ledger(lock: LockContext, reader: GuardedReader) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": LEDGER_KIND,
        "attempt_ordinal": 2,
        "holdout_score_opened": False,
        "exact_path_capability": True,
        "train_case_count": 18,
        "train_case_binding_sha256": sha256_bytes(
            canonical_bytes(sorted(lock.scan_sha_by_case.values()))
        ),
        "read_count": len(reader.entries),
        "denied_read_count": len(reader.denied_attempts),
        "denied_attempts": list(reader.denied_attempts),
        "entries": list(reader.entries),
        "assertions": {
            "full_28_case_catalog_read": False,
            "full_28_case_execution_manifest_read": False,
            "complete_label_manifest_read": False,
            "holdout_label_read_count": 0,
            "holdout_prediction_read_count": 0,
            "holdout_score_read_count": 0,
            "all_case_scoped_reads_train_allowlisted": True,
        },
    }


def evaluate_deployed_replay(
    cases: Sequence[CaseData],
    model: Mapping[str, Any],
    evaluated_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, bool]:
    """Verify replay and training/runtime identity without leaking an error path."""

    replay_identities: list[bool] = []
    deployed_identities: list[bool] = []
    try:
        for case in cases:
            primary = compose_candidate2(case.primary_v51, case.primary_pool, model)
            replay = compose_candidate2(case.replay_v51, case.replay_pool, model)
            replay_identities.append(canonical_bytes(primary) == canonical_bytes(replay))
            deployed_identities.append(
                canonical_bytes(primary["predictions"])
                == canonical_bytes(evaluated_predictions[case.case_id])
            )
    except Exception:
        return {
            "evaluated": True,
            "execution_succeeded": False,
            "primary_replay_bytes_exact": False,
            "deployed_model_matches_evaluated_predictions": False,
        }
    return {
        "evaluated": True,
        "execution_succeeded": True,
        "primary_replay_bytes_exact": (
            len(replay_identities) == len(cases) and all(replay_identities)
        ),
        "deployed_model_matches_evaluated_predictions": (
            len(deployed_identities) == len(cases) and all(deployed_identities)
        ),
    }


def run_optimizer(
    *,
    preregistration: Path,
    preregistration_sha256: str,
    label_attestation: Path,
    label_attestation_sha256: str,
    cache_attestation: Path,
    cache_attestation_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    lock = load_lock_context()
    reader, prereg, source_hashes = prepare_guarded_inputs(
        lock=lock,
        preregistration=preregistration.resolve(),
        preregistration_sha256=preregistration_sha256,
        label_attestation=label_attestation.resolve(),
        label_attestation_sha256=label_attestation_sha256,
        cache_attestation=cache_attestation.resolve(),
        cache_attestation_sha256=cache_attestation_sha256,
    )
    (
        cases,
        projection_sha,
        actual_label_attestation_sha,
        base_runtime_sha,
        base_optimizer_sha,
        cache_manifest_sha,
        actual_cache_attestation_sha,
        cache_binding_sha,
    ) = load_train_cases(
        lock,
        reader,
        label_attestation=label_attestation.resolve(),
        cache_attestation=cache_attestation.resolve(),
    )
    if base_runtime_sha != prereg["artifact_bindings"]["candidate1_base_runtime"]["sha256"]:
        raise ValueError("candidate2 base runtime differs from preregistration")
    if base_optimizer_sha != prereg["artifact_bindings"]["candidate1_base_optimizer"]["sha256"]:
        raise ValueError("candidate2 base optimizer differs from preregistration")

    nested = nested_outer_evaluation(cases)
    final_selected, final_trials = select_configuration(
        cases,
        n_splits=FINAL_SELECTION_FOLDS,
        seed=f"{FOLD_SEED}|final-selection",
    )
    final_profile = final_selected["profile"]
    final_predictions, final_classifier = _final_profile_predictions(cases, final_profile)
    final_fit_metrics, final_fit_rows = score_predictions(cases, final_predictions)
    baseline_metrics, baseline_rows = score_predictions(cases, baseline_predictions(cases))
    final_oof_gate = training_gate(final_selected["metrics"], baseline_metrics)
    nested_gate = nested["training_retain_gate"]
    preliminary_pass = bool(
        nested_gate["all_conjunctive_checks_passed"]
        and final_oof_gate["all_conjunctive_checks_passed"]
        and final_profile["policy_mode"] != "V51_FALLBACK"
    )
    fit_audit_core = {
        "schema_version": 1,
        "kind": FIT_AUDIT_CORE_KIND,
        "attempt_ordinal": 2,
        "adaptive_train_development": True,
        "holdout_score_opened": False,
        "candidate1_terminal_sha256": source_hashes["candidate1_terminal"],
        "candidate2_preregistration_sha256": preregistration_sha256,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "label_projection_sha256": projection_sha,
        "label_attestation_sha256": actual_label_attestation_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_attestation_sha256": actual_cache_attestation_sha,
        "train_source_cache_binding_sha256": cache_binding_sha,
        "base_runtime_code_sha256": base_runtime_sha,
        "base_optimizer_code_sha256": base_optimizer_sha,
        "runtime_code_sha256": source_hashes["candidate2_runtime"],
        "optimizer_code_sha256": source_hashes["candidate2_optimizer"],
        "tests_code_sha256": source_hashes["candidate2_tests"],
        "case_grouping": "COMPLETE_CASE_ONLY",
        "nested_outer_evaluation": nested,
        "full_train_oof_selection": final_selected,
        "full_train_oof_gate": final_oof_gate,
        "full_train_trial_count": len(final_trials),
        "full_train_trials": final_trials,
        "final_fit_train_metrics_not_used_for_gate": final_fit_metrics,
        "final_fit_case_change_counts_not_used_for_gate": _case_change_counts(
            final_fit_rows, baseline_rows
        ),
        "pre_replay_gate_passed": preliminary_pass,
        "design_lock": design_lock(),
    }
    fit_core_sha = sha256_bytes(canonical_bytes(fit_audit_core))
    ledger = _read_ledger(lock, reader)

    gate_pass = None
    model = None
    replay_exact = False
    if preliminary_pass:
        gate_pass = {
            "schema_version": 1,
            "kind": GATE_PASS_KIND,
            "candidate_id": 2,
            "status": "PASS",
            "training_gate_passed": True,
            "nested_outer_gate_passed": True,
            "full_train_oof_gate_passed": True,
            "primary_replay_bytes_exact": True,
            "deployed_model_matches_evaluated_predictions": True,
            "holdout_score_opened": False,
            "holdout_execution_authorized": False,
            "fit_audit_core_sha256": fit_core_sha,
            "candidate1_terminal_sha256": source_hashes["candidate1_terminal"],
            "candidate2_preregistration_sha256": preregistration_sha256,
            "base_runtime_code_sha256": base_runtime_sha,
            "base_optimizer_code_sha256": base_optimizer_sha,
            "runtime_code_sha256": source_hashes["candidate2_runtime"],
            "optimizer_code_sha256": source_hashes["candidate2_optimizer"],
            "tests_code_sha256": source_hashes["candidate2_tests"],
            "label_projection_sha256": projection_sha,
            "train_cache_manifest_sha256": cache_manifest_sha,
            "train_source_cache_binding_sha256": cache_binding_sha,
        }
        gate_pass_sha = sha256_bytes(canonical_bytes(gate_pass))
        model = _build_model(
            profile=final_profile,
            classifier=final_classifier,
            prereg=prereg,
            preregistration_sha256=preregistration_sha256,
            fit_audit_core_sha256=fit_core_sha,
            train_gate_result_sha256=gate_pass_sha,
            label_projection_sha256=projection_sha,
            label_attestation_sha256=actual_label_attestation_sha,
            cache_manifest_sha256=cache_manifest_sha,
            cache_attestation_sha256=actual_cache_attestation_sha,
            train_source_cache_binding_sha256=cache_binding_sha,
        )
    replay_evaluated = False
    replay_execution_succeeded = False
    deployed_matches_evaluated = False
    if preliminary_pass:
        assert model is not None
        replay_result = evaluate_deployed_replay(cases, model, final_predictions)
        replay_evaluated = replay_result["evaluated"]
        replay_execution_succeeded = replay_result["execution_succeeded"]
        replay_exact = replay_result["primary_replay_bytes_exact"]
        deployed_matches_evaluated = replay_result[
            "deployed_model_matches_evaluated_predictions"
        ]

    training_gate_passed = bool(
        preliminary_pass and replay_exact and deployed_matches_evaluated
    )
    fit_audit = {
        "schema_version": 1,
        "kind": FIT_AUDIT_KIND,
        "fit_audit_core_sha256": fit_core_sha,
        "training_gate_passed": training_gate_passed,
        "primary_replay_evaluated": replay_evaluated,
        "replay_execution_succeeded": replay_execution_succeeded,
        "primary_replay_bytes_exact": replay_exact if replay_evaluated else None,
        "deployed_model_matches_evaluated_predictions": (
            deployed_matches_evaluated if replay_evaluated else None
        ),
        "holdout_score_opened": False,
        "holdout_retention_gates_not_evaluated": True,
    }
    if training_gate_passed:
        assert gate_pass is not None
        fit_audit["train_gate_result_sha256"] = sha256_bytes(
            canonical_bytes(gate_pass)
        )
    payloads: dict[str, Mapping[str, Any]] = {
        "fit_audit_core.json": fit_audit_core,
        "fit_audit.json": fit_audit,
        "train_read_ledger.json": ledger,
    }
    output = output_dir.resolve()
    try:
        output_relative = str(output.relative_to(ROOT))
    except ValueError as error:
        raise ValueError("candidate2 output must be repository-relative") from error
    if not training_gate_passed:
        terminal_checks = {
            **{key: bool(value) for key, value in nested_gate["checks"].items()},
            "nested_outer_gate_passed": bool(
                nested_gate["all_conjunctive_checks_passed"]
            ),
            "full_train_oof_gate_passed": bool(
                final_oof_gate["all_conjunctive_checks_passed"]
            ),
        }
        if replay_evaluated:
            terminal_checks.update(
                {
                    "replay_execution_succeeded": replay_execution_succeeded,
                    "primary_replay_bytes_exact": replay_exact,
                    "deployed_model_matches_evaluated_predictions": deployed_matches_evaluated,
                }
            )
        artifact_bindings = deepcopy(prereg["artifact_bindings"])
        for key, name, payload in (
            ("candidate2_fit_audit_core", "fit_audit_core.json", fit_audit_core),
            ("candidate2_fit_audit", "fit_audit.json", fit_audit),
            ("candidate2_train_read_ledger", "train_read_ledger.json", ledger),
        ):
            data = canonical_bytes(payload)
            artifact_bindings[key] = {
                "path": f"{output_relative}/{name}",
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
        terminal = {
            "schema_version": 1,
            "kind": TERMINAL_KIND,
            "candidate_id": 2,
            "attempt_ordinal": 2,
            "status": (
                "REJECTED_TRAIN_REPLAY_INTEGRITY_GATE"
                if preliminary_pass
                else "REJECTED_TRAIN_GATE"
            ),
            "adaptive_train_development": True,
            "nested_training_retain_gate_passed": bool(
                nested_gate["all_conjunctive_checks_passed"]
            ),
            "full_train_oof_gate_passed": bool(
                final_oof_gate["all_conjunctive_checks_passed"]
            ),
            "primary_replay_evaluated": replay_evaluated,
            "boolean_gate_checks": terminal_checks,
            "artifact_bindings": artifact_bindings,
            "candidate2_output_root": output_relative,
            "candidate1_terminal_sha256": source_hashes["candidate1_terminal"],
            "candidate2_preregistration_sha256": preregistration_sha256,
            "fit_audit_core_sha256": fit_core_sha,
            "holdout_score_opened": False,
            "holdout_execution_authorized": False,
            "pre_holdout_execution_anchor_created": False,
            "holdout_eligible_model_created": False,
            "fit_audit_core_created": True,
            "fit_audit_created": True,
            "train_read_ledger_created": True,
            "train_gate_pass_artifact_created": False,
            "freeze_lock_created": False,
            "pre_holdout_seal_created": False,
            "post_terminal_model_or_threshold_mutation": "FORBIDDEN",
        }
        payloads["terminal_negative.json"] = terminal
    else:
        assert gate_pass is not None and model is not None
        gate_pass_sha = sha256_bytes(canonical_bytes(gate_pass))
        model_sha = sha256_bytes(canonical_bytes(model))
        audit_sha = sha256_bytes(canonical_bytes(fit_audit))
        ledger_sha = sha256_bytes(canonical_bytes(ledger))
        freeze_lock = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_aug30_train18_candidate2_pre_holdout_freeze_lock_v1",
            "status": "FROZEN_AFTER_TRAIN_GATE_PASS_BEFORE_HOLDOUT",
            "candidate_id": 2,
            "training_gate_passed": True,
            "holdout_score_opened": False,
            "holdout_execution_authorized": False,
            "model_sha256": model_sha,
            "train_gate_result_sha256": gate_pass_sha,
            "fit_audit_core_sha256": fit_core_sha,
            "fit_audit_sha256": audit_sha,
            "train_read_ledger_sha256": ledger_sha,
            "candidate1_terminal_sha256": source_hashes["candidate1_terminal"],
            "candidate2_preregistration_sha256": preregistration_sha256,
            "split_sha256": lock.split_sha256,
            "train_allowlist_sha256": lock.allowlist_sha256,
            "label_projection_sha256": projection_sha,
            "label_attestation_sha256": actual_label_attestation_sha,
            "train_cache_manifest_sha256": cache_manifest_sha,
            "cache_attestation_sha256": actual_cache_attestation_sha,
            "train_source_cache_binding_sha256": cache_binding_sha,
            "base_runtime_code_sha256": base_runtime_sha,
            "base_optimizer_code_sha256": base_optimizer_sha,
            "runtime_code_sha256": source_hashes["candidate2_runtime"],
            "optimizer_code_sha256": source_hashes["candidate2_optimizer"],
            "tests_code_sha256": source_hashes["candidate2_tests"],
            "holdout_stage_order_frozen": [
                {"tier": "SIMILAR_FAMILY", "case_count": 4},
                {"tier": "ADJACENT_FAMILY", "case_count": 3},
                {"tier": "DISTANT_FAMILY", "case_count": 3},
            ],
            "adaptation_after_similar_or_adjacent_results": "FORBIDDEN",
        }
        lock_sha = sha256_bytes(canonical_bytes(freeze_lock))
        preseal = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_aug30_train18_candidate2_pre_holdout_seal_v1",
            "status": "SEALED_LOCALLY_AWAITING_INDEPENDENT_EXTERNAL_ANCHOR",
            "candidate_id": 2,
            "training_gate_passed": True,
            "holdout_score_opened": False,
            "holdout_execution_authorized": False,
            "freeze_lock_sha256": lock_sha,
            "frozen_model_sha256": model_sha,
            "train_gate_result_sha256": gate_pass_sha,
            "fit_audit_core_sha256": fit_core_sha,
            "candidate1_terminal_sha256": source_hashes["candidate1_terminal"],
            "candidate2_preregistration_sha256": preregistration_sha256,
            "split_sha256": lock.split_sha256,
            "train_allowlist_sha256": lock.allowlist_sha256,
            "label_projection_sha256": projection_sha,
            "train_cache_manifest_sha256": cache_manifest_sha,
            "train_source_cache_binding_sha256": cache_binding_sha,
            "runtime_code_sha256": source_hashes["candidate2_runtime"],
            "optimizer_code_sha256": source_hashes["candidate2_optimizer"],
            "tests_code_sha256": source_hashes["candidate2_tests"],
            "external_anchor_required_before_holdout_score": True,
            "external_anchor_must_be_created_by_independent_protocol_auditor": True,
        }
        payloads.update(
            {
                "train_gate_pass.json": gate_pass,
                "frozen_model.json": model,
                "freeze_lock.json": freeze_lock,
                "pre_holdout_seal.json": preseal,
            }
        )
    publish_artifacts(output, training_gate_passed=training_gate_passed, payloads=payloads)
    return {
        "status": "TRAIN_GATE_PASS" if training_gate_passed else str(terminal["status"]),
        "training_gate_passed": training_gate_passed,
        "holdout_score_opened": False,
        "output_dir": str(output),
        "nested_train_candidate_metrics": nested["candidate_metrics"],
        "nested_train_v51_metrics": nested["v51_reference_metrics"],
        "artifact_sha256": {
            name: _sha(output / name) for name in publication_artifact_names(training_gate_passed)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate2-preregistration", type=Path, required=True)
    parser.add_argument("--candidate2-preregistration-sha256", required=True)
    parser.add_argument("--label-attestation", type=Path, required=True)
    parser.add_argument("--label-attestation-sha256", required=True)
    parser.add_argument("--cache-attestation", type=Path, required=True)
    parser.add_argument("--cache-attestation-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_optimizer(
        preregistration=args.candidate2_preregistration,
        preregistration_sha256=args.candidate2_preregistration_sha256,
        label_attestation=args.label_attestation,
        label_attestation_sha256=args.label_attestation_sha256,
        cache_attestation=args.cache_attestation,
        cache_attestation_sha256=args.cache_attestation_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
