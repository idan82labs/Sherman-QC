#!/usr/bin/env python3
"""TRAIN18-only nested-CV optimizer for Candidate3 conservative deletion.

This executable is fail-closed until it is given an immutable Candidate2
terminal-negative and a sealed Candidate3 preregistration that binds it. It
never resolves a case outside the attested TRAIN custody manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_train18_candidate3_conservative_deletion import (  # noqa: E402
    ALLOWED_NUMERIC_FEATURES,
    ATTEMPT_ORDINAL,
    MODEL_KIND,
    PASS_KIND,
    PROFILE,
    apply_candidate3,
    canonical_sha256,
    extract_prediction_features,
    runtime_source_sha256,
    validate_rule,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402


SPLIT_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
SPLIT_SHA256 = "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
ORIGINAL_PREREG_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ORIGINAL_PREREG_SHA256 = "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a"
ALLOWLIST_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
ALLOWLIST_SHA256 = "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
CANDIDATE1_TERMINAL_PATH = ROOT / "research/f1_aug30_train18_candidate1_terminal_negative_v1.json"
CANDIDATE1_TERMINAL_SHA256 = "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
LABEL_ATTESTATION_PATH = (
    ROOT / "output/research/f1-aug30-train-only-custody-v2/labels/train_label_projection_attestation.json"
)
LABEL_ATTESTATION_SHA256 = "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
CACHE_ATTESTATION_PATH = (
    ROOT / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json"
)
CACHE_ATTESTATION_SHA256 = "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output/research/f1-aug30-train18-candidate3-conservative-deletion-v1"
)
TESTS_PATH = ROOT / "tests/test_f1_train18_candidate3_conservative_deletion.py"

MINIMUM_RULE_FIT_FP_REMOVAL = 2
MINIMUM_RULE_FIT_DELETED_CASES = 2
MINIMUM_INNER_OOF_FP_REMOVAL = 2
MINIMUM_INNER_OOF_DELETED_CASES = 2
SUPPORT_TIERS = ("1", "2", "3")
METRIC_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    gold: tuple[dict[str, Any], ...]
    v51: dict[str, Any]
    pool: dict[str, Any]
    v51_artifact_sha256: str
    pool_artifact_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def optimizer_source_sha256() -> str:
    return _file_sha256(Path(__file__))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _write_canonical(path: Path, value: Any) -> str:
    data = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


def _read_bound_json(
    path: Path,
    expected_sha256: str,
    ledger: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    data = resolved.read_bytes()
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(f"{role} hash mismatch: {actual} != {expected_sha256}")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object")
    ledger.append(
        {
            "role": role,
            "path": str(resolved.relative_to(ROOT)),
            "sha256": actual,
            "size_bytes": len(data),
        }
    )
    return payload


def _prediction_features(case: PreparedCase) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidate_by_id = {
        str(row["candidate_id"]): row for row in case.pool.get("candidates") or []
    }
    result = []
    for prediction in case.v51.get("predictions") or []:
        hypothesis_id = str(prediction.get("source_hypothesis_id") or "")
        result.append(
            (prediction, extract_prediction_features(prediction, candidate_by_id.get(hypothesis_id)))
        )
    return result


def _rule_matches(rule: dict[str, Any], features: dict[str, Any]) -> bool:
    feature = rule["feature"]
    value = features[feature]
    if feature == "source_kind":
        return value == rule["value"]
    if value is None:
        return False
    return (
        float(value) <= float(rule["threshold"])
        if rule["operator"] == "le"
        else float(value) >= float(rule["threshold"])
    )


def predictions_after_rule(case: PreparedCase, rule: dict[str, Any] | None) -> list[dict[str, Any]]:
    if rule is None:
        return deepcopy(list(case.v51.get("predictions") or []))
    validate_rule(rule)
    return [
        deepcopy(prediction)
        for prediction, features in _prediction_features(case)
        if not _rule_matches(rule, features)
    ]


def _case_result(case: PreparedCase, rule: dict[str, Any] | None) -> dict[str, Any]:
    result = match_lines(list(case.gold), predictions_after_rule(case, rule), METRIC_THRESHOLDS)
    support_by_gold = {
        str(row["bend_id"]): str(int(row["reviewer_support_count"])) for row in case.gold
    }
    support_tp = {tier: 0 for tier in SUPPORT_TIERS}
    for match in result["matches"]:
        tier = support_by_gold[str(match["gold_id"])]
        if tier not in support_tp:
            raise ValueError(f"unsupported reviewer support tier: {tier}")
        support_tp[tier] += 1
    return {
        "case_id": case.case_id,
        "true_positive": int(result["true_positive"]),
        "false_positive": int(result["false_positive"]),
        "false_negative": int(result["false_negative"]),
        "prediction_count": int(result["prediction_count"]),
        "exact": bool(result["exact"]),
        "support_tier_tp": support_tp,
    }


def aggregate_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    tp = sum(row["true_positive"] for row in values)
    fp = sum(row["false_positive"] for row in values)
    fn = sum(row["false_negative"] for row in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "case_count": len(values),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "prediction_count": sum(row["prediction_count"] for row in values),
        "precision": round(precision, 12),
        "recall": round(recall, 12),
        "f1": round(f1, 12),
        "exact_case_count": sum(bool(row["exact"]) for row in values),
        "support_tier_tp": {
            tier: sum(row["support_tier_tp"][tier] for row in values)
            for tier in SUPPORT_TIERS
        },
    }


def score_cases(cases: Iterable[PreparedCase], rule: dict[str, Any] | None) -> dict[str, Any]:
    return aggregate_results(_case_result(case, rule) for case in cases)


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "true_positive": candidate["true_positive"] - reference["true_positive"],
        "false_positive": candidate["false_positive"] - reference["false_positive"],
        "false_negative": candidate["false_negative"] - reference["false_negative"],
        "prediction_count": candidate["prediction_count"] - reference["prediction_count"],
        "f1": round(candidate["f1"] - reference["f1"], 12),
        "exact_case_count": candidate["exact_case_count"] - reference["exact_case_count"],
        "support_tier_tp": {
            tier: candidate["support_tier_tp"][tier] - reference["support_tier_tp"][tier]
            for tier in SUPPORT_TIERS
        },
    }


def _deleted_case_count(cases: Iterable[PreparedCase], rule: dict[str, Any] | None) -> int:
    return sum(
        len(predictions_after_rule(case, rule)) < len(case.v51.get("predictions") or [])
        for case in cases
    )


def _no_harm(delta: dict[str, Any]) -> bool:
    return bool(
        delta["true_positive"] == 0
        and delta["false_negative"] == 0
        and delta["exact_case_count"] >= 0
        and min(delta["support_tier_tp"].values()) >= 0
    )


def strategy_space() -> list[dict[str, str]]:
    strategies = [
        {"feature": feature, "operator": operator}
        for feature in ALLOWED_NUMERIC_FEATURES
        for operator in ("le", "ge")
    ]
    strategies.append({"feature": "source_kind", "operator": "eq"})
    return strategies


def _rules_for_strategy(
    cases: Iterable[PreparedCase], strategy: dict[str, str]
) -> list[dict[str, Any]]:
    feature = strategy["feature"]
    operator = strategy["operator"]
    values = []
    for case in cases:
        for _prediction, features in _prediction_features(case):
            value = features[feature]
            if value is not None:
                values.append(value)
    if feature == "source_kind":
        return [
            {"feature": feature, "operator": "eq", "value": value}
            for value in sorted({str(value) for value in values if str(value)})
        ]
    return [
        {"feature": feature, "operator": operator, "threshold": float(value)}
        for value in sorted({float(value) for value in values})
    ]


def fit_rule_for_strategy(
    cases: list[PreparedCase], strategy: dict[str, str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    reference = score_cases(cases, None)
    candidates = []
    for rule in _rules_for_strategy(cases, strategy):
        candidate = score_cases(cases, rule)
        delta = metric_delta(candidate, reference)
        deleted_cases = _deleted_case_count(cases, rule)
        if (
            _no_harm(delta)
            and -delta["false_positive"] >= MINIMUM_RULE_FIT_FP_REMOVAL
            and deleted_cases >= MINIMUM_RULE_FIT_DELETED_CASES
        ):
            rank = (
                -delta["false_positive"],
                delta["exact_case_count"],
                delta["f1"],
                deleted_cases,
                canonical_sha256(rule),
            )
            candidates.append((rank, rule, candidate, delta, deleted_cases))
    candidates.sort(key=lambda row: row[0], reverse=True)
    if not candidates:
        return None, {
            "strategy": strategy,
            "candidate_rule_count": len(_rules_for_strategy(cases, strategy)),
            "eligible_rule_count": 0,
        }
    selected = candidates[0]
    return deepcopy(selected[1]), {
        "strategy": strategy,
        "candidate_rule_count": len(_rules_for_strategy(cases, strategy)),
        "eligible_rule_count": len(candidates),
        "selected_rule": selected[1],
        "fit_metrics": selected[2],
        "fit_delta": selected[3],
        "deleted_case_count": selected[4],
    }


def inner_oof_for_strategy(
    cases: list[PreparedCase], strategy: dict[str, str]
) -> dict[str, Any]:
    reference_rows = []
    candidate_rows = []
    fold_audit = []
    for validation_index, validation_case in enumerate(cases):
        training = [case for index, case in enumerate(cases) if index != validation_index]
        rule, fit_audit = fit_rule_for_strategy(training, strategy)
        reference_rows.append(_case_result(validation_case, None))
        candidate_rows.append(_case_result(validation_case, rule))
        fold_audit.append(
            {
                "validation_case_id": validation_case.case_id,
                "training_case_count": len(training),
                "selected_rule": rule,
                "fit_audit": fit_audit,
            }
        )
    reference = aggregate_results(reference_rows)
    candidate = aggregate_results(candidate_rows)
    delta = metric_delta(candidate, reference)
    deleted_cases = sum(
        candidate_row["prediction_count"] < reference_row["prediction_count"]
        for candidate_row, reference_row in zip(candidate_rows, reference_rows)
    )
    eligible = bool(
        _no_harm(delta)
        and -delta["false_positive"] >= MINIMUM_INNER_OOF_FP_REMOVAL
        and deleted_cases >= MINIMUM_INNER_OOF_DELETED_CASES
    )
    return {
        "strategy": strategy,
        "eligible": eligible,
        "reference_metrics": reference,
        "candidate_metrics": candidate,
        "delta": delta,
        "deleted_validation_case_count": deleted_cases,
        "active_fold_count": sum(fold["selected_rule"] is not None for fold in fold_audit),
        "folds": fold_audit,
    }


def select_strategy_by_inner_oof(cases: list[PreparedCase]) -> tuple[dict[str, str] | None, dict[str, Any]]:
    audits = [inner_oof_for_strategy(cases, strategy) for strategy in strategy_space()]
    eligible = [audit for audit in audits if audit["eligible"]]
    eligible.sort(
        key=lambda audit: (
            -audit["delta"]["false_positive"],
            audit["delta"]["exact_case_count"],
            audit["delta"]["f1"],
            audit["deleted_validation_case_count"],
            canonical_sha256(audit["strategy"]),
        ),
        reverse=True,
    )
    selected = deepcopy(eligible[0]["strategy"]) if eligible else None
    return selected, {
        "strategy_count": len(audits),
        "eligible_strategy_count": len(eligible),
        "selected_strategy": selected,
        "strategies": audits,
    }


def complete_nested_oof(cases: list[PreparedCase]) -> dict[str, Any]:
    reference_rows = []
    candidate_rows = []
    folds = []
    for outer_index, validation_case in enumerate(cases):
        outer_training = [case for index, case in enumerate(cases) if index != outer_index]
        strategy, inner_audit = select_strategy_by_inner_oof(outer_training)
        rule = None
        fit_audit: dict[str, Any] = {"reason": "no_inner_oof_eligible_strategy"}
        if strategy is not None:
            rule, fit_audit = fit_rule_for_strategy(outer_training, strategy)
        reference_rows.append(_case_result(validation_case, None))
        candidate_rows.append(_case_result(validation_case, rule))
        folds.append(
            {
                "outer_validation_case_id": validation_case.case_id,
                "outer_training_case_count": len(outer_training),
                "selected_strategy": strategy,
                "selected_rule": rule,
                "fit_audit": fit_audit,
                "inner_selection": inner_audit,
            }
        )
    reference = aggregate_results(reference_rows)
    candidate = aggregate_results(candidate_rows)
    delta = metric_delta(candidate, reference)
    deleted_cases = sum(
        candidate_row["prediction_count"] < reference_row["prediction_count"]
        for candidate_row, reference_row in zip(candidate_rows, reference_rows)
    )
    checks = {
        "nested_oof_f1_increase": delta["f1"] > 0.0,
        "nested_oof_true_positive_nondecrease": delta["true_positive"] == 0,
        "nested_oof_false_positive_decrease": delta["false_positive"] < 0,
        "nested_oof_exact_case_nondecrease": delta["exact_case_count"] >= 0,
        "nested_oof_support_tier_tp_nondecrease": min(delta["support_tier_tp"].values()) >= 0,
    }
    checks["complete_case_outer_coverage"] = len(folds) == len(cases) == len(
        {fold["outer_validation_case_id"] for fold in folds}
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "delta": delta,
        "deleted_validation_case_count": deleted_cases,
        "checks": checks,
        "folds": folds,
    }


def _validate_candidate_terminal(payload: dict[str, Any], ordinal: int, role: str) -> None:
    if payload.get("candidate_id") != ordinal and payload.get("attempt_ordinal") != ordinal:
        raise ValueError(f"{role} does not bind attempt {ordinal}")
    if payload.get("status") != "REJECTED_TRAIN_GATE":
        raise ValueError(f"{role} is not an immutable TRAIN terminal-negative")
    if payload.get("holdout_score_opened") is not False:
        raise ValueError(f"{role} observed holdout score state")
    if payload.get("post_terminal_model_or_threshold_mutation") != "FORBIDDEN":
        raise ValueError(f"{role} does not prohibit post-terminal mutation")


def _validate_candidate3_prereg(
    payload: dict[str, Any],
    prereg_sha256: str,
    candidate2_path: Path,
    candidate2_sha256: str,
) -> None:
    if payload.get("kind") != "sherman_qc_f1_aug30_train18_candidate3_prereg_v1":
        raise ValueError("Candidate3 prereg kind mismatch")
    if payload.get("status") != "SEALED_TRAIN_ONLY" or payload.get("attempt_ordinal") != 3:
        raise ValueError("Candidate3 prereg is not sealed for attempt 3")
    if payload.get("holdout_score_opened") is not False:
        raise ValueError("Candidate3 prereg observed holdout score state")
    bindings = payload.get("bindings") or {}
    expected = {
        "candidate1_terminal_sha256": CANDIDATE1_TERMINAL_SHA256,
        "candidate2_terminal_sha256": candidate2_sha256,
        "split_sha256": SPLIT_SHA256,
        "train_allowlist_sha256": ALLOWLIST_SHA256,
        "original_prereg_sha256": ORIGINAL_PREREG_SHA256,
        "custody_label_attestation_sha256": LABEL_ATTESTATION_SHA256,
        "custody_cache_attestation_sha256": CACHE_ATTESTATION_SHA256,
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate3 prereg binding mismatch")
    terminal_binding = payload.get("candidate2_terminal_binding") or {}
    if terminal_binding.get("path") != str(candidate2_path.resolve().relative_to(ROOT)):
        raise ValueError("Candidate3 prereg Candidate2 terminal path mismatch")
    if terminal_binding.get("sha256") != candidate2_sha256:
        raise ValueError("Candidate3 prereg Candidate2 terminal hash mismatch")
    if payload.get("adaptive_search_disclosure") != {
        "rule_form": "one categorical equality or one numeric tail threshold",
        "numeric_features": list(ALLOWED_NUMERIC_FEATURES),
        "numeric_operators": ["le", "ge"],
        "missing_feature_policy": "KEEP_V51",
        "minimum_rule_fit_fp_removal": MINIMUM_RULE_FIT_FP_REMOVAL,
        "minimum_rule_fit_deleted_cases": MINIMUM_RULE_FIT_DELETED_CASES,
        "complete_nested_cv": "outer_leave_one_case_out_with_inner_leave_one_case_out",
    }:
        raise ValueError("Candidate3 prereg adaptive-search disclosure mismatch")
    code_bindings = payload.get("code_bindings")
    expected_code_bindings = {
        "runtime": {
            "path": "backend/f1_train18_candidate3_conservative_deletion.py",
            "sha256": runtime_source_sha256(),
        },
        "optimizer": {
            "path": "scripts/optimize_f1_aug30_train18_candidate3_conservative_deletion.py",
            "sha256": optimizer_source_sha256(),
        },
        "tests": {
            "path": "tests/test_f1_train18_candidate3_conservative_deletion.py",
            "sha256": _file_sha256(TESTS_PATH),
        },
    }
    if code_bindings != expected_code_bindings:
        raise ValueError("Candidate3 prereg code binding mismatch")
    if not prereg_sha256:
        raise ValueError("Candidate3 prereg hash is required")


def _load_train_cases(
    ledger: list[dict[str, Any]],
) -> tuple[list[PreparedCase], list[PreparedCase], dict[str, str]]:
    label_attestation = _read_bound_json(
        LABEL_ATTESTATION_PATH, LABEL_ATTESTATION_SHA256, ledger, "custody_label_attestation"
    )
    cache_attestation = _read_bound_json(
        CACHE_ATTESTATION_PATH, CACHE_ATTESTATION_SHA256, ledger, "custody_cache_attestation"
    )
    split = _read_bound_json(SPLIT_PATH, SPLIT_SHA256, ledger, "sealed_split")
    _read_bound_json(ORIGINAL_PREREG_PATH, ORIGINAL_PREREG_SHA256, ledger, "original_prereg")
    allowlist = _read_bound_json(ALLOWLIST_PATH, ALLOWLIST_SHA256, ledger, "train_allowlist")
    train_ids = tuple(split.get("train_case_ids") or [])
    allow_ids = tuple(row["case_id"] for row in allowlist.get("cases") or [])
    if len(train_ids) != 18 or set(train_ids) != set(allow_ids):
        raise ValueError("sealed split and TRAIN allowlist differ")
    if label_attestation.get("case_count") != 18 or cache_attestation.get("case_count") != 18:
        raise ValueError("custody attestation case counts are not TRAIN18")
    if label_attestation.get("optimizer_holdout_access_count") != 0:
        raise ValueError("label custody attestation reports forbidden optimizer access")
    if cache_attestation.get("optimizer_holdout_access_count") != 0:
        raise ValueError("cache custody attestation reports forbidden optimizer access")

    projection_binding = label_attestation["projection_binding"]
    manifest_binding = cache_attestation["manifest_binding"]
    projection = _read_bound_json(
        ROOT / projection_binding["path"],
        projection_binding["sha256"],
        ledger,
        "attested_train_label_projection",
    )
    manifest = _read_bound_json(
        ROOT / manifest_binding["path"],
        manifest_binding["sha256"],
        ledger,
        "attested_train_cache_manifest",
    )
    if projection.get("case_count") != 18 or manifest.get("case_count") != 18:
        raise ValueError("TRAIN custody payload case count mismatch")
    if manifest.get("holdout_entry_count") != 0 or manifest.get("record_count") != 36:
        raise ValueError("TRAIN cache manifest is not the attested 18x2 projection")
    gold_by_id = {row["case_id"]: row for row in projection["cases"]}
    record_by_key = {(row["case_id"], row["pass"]): row for row in manifest["records"]}
    if set(gold_by_id) != set(train_ids):
        raise ValueError("TRAIN label projection differs from sealed TRAIN IDs")
    if set(record_by_key) != {(case_id, pass_name) for case_id in train_ids for pass_name in ("primary", "replay")}:
        raise ValueError("TRAIN cache record set differs from sealed TRAIN IDs")

    passes: dict[str, list[PreparedCase]] = {"primary": [], "replay": []}
    for case_id in train_ids:
        gold_row = gold_by_id[case_id]
        allow_sha = next(row["scan_sha256"] for row in allowlist["cases"] if row["case_id"] == case_id)
        if gold_row["scan_sha256"] != allow_sha:
            raise ValueError(f"{case_id}: label projection scan hash differs from allowlist")
        for pass_name in ("primary", "replay"):
            record = record_by_key[(case_id, pass_name)]
            if record["scan_sha256"] != allow_sha:
                raise ValueError(f"{case_id}: cache scan hash differs from allowlist")
            semantic = _read_bound_json(
                ROOT / record["semantic_path"],
                record["semantic_sha256"],
                ledger,
                f"train_{case_id}_{pass_name}_v51",
            )
            pool = _read_bound_json(
                ROOT / record["pool_path"],
                record["pool_sha256"],
                ledger,
                f"train_{case_id}_{pass_name}_pool",
            )
            passes[pass_name].append(
                PreparedCase(
                    case_id=case_id,
                    gold=tuple(deepcopy(gold_row["bends"])),
                    v51=semantic,
                    pool=pool,
                    v51_artifact_sha256=record["semantic_sha256"],
                    pool_artifact_sha256=record["pool_sha256"],
                )
            )
    for primary, replay in zip(passes["primary"], passes["replay"]):
        if primary.case_id != replay.case_id:
            raise ValueError("TRAIN primary/replay order mismatch")
        if primary.v51_artifact_sha256 != replay.v51_artifact_sha256:
            raise ValueError(f"{primary.case_id}: V51 primary/replay bytes differ")
        if primary.pool_artifact_sha256 != replay.pool_artifact_sha256:
            raise ValueError(f"{primary.case_id}: pool primary/replay bytes differ")
    bindings = {
        "split_sha256": SPLIT_SHA256,
        "train_allowlist_sha256": ALLOWLIST_SHA256,
        "original_prereg_sha256": ORIGINAL_PREREG_SHA256,
        "custody_label_attestation_sha256": LABEL_ATTESTATION_SHA256,
        "custody_cache_attestation_sha256": CACHE_ATTESTATION_SHA256,
    }
    return passes["primary"], passes["replay"], bindings


def run(
    *,
    candidate2_terminal_path: Path,
    candidate2_terminal_sha256: str,
    candidate3_prereg_path: Path,
    candidate3_prereg_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Candidate3 output: {output_root}")
    ledger: list[dict[str, Any]] = []
    candidate1 = _read_bound_json(
        CANDIDATE1_TERMINAL_PATH,
        CANDIDATE1_TERMINAL_SHA256,
        ledger,
        "candidate1_terminal_negative",
    )
    _validate_candidate_terminal(candidate1, 1, "Candidate1 terminal")
    candidate2 = _read_bound_json(
        candidate2_terminal_path,
        candidate2_terminal_sha256,
        ledger,
        "candidate2_terminal_negative",
    )
    _validate_candidate_terminal(candidate2, 2, "Candidate2 terminal")
    candidate3_prereg = _read_bound_json(
        candidate3_prereg_path,
        candidate3_prereg_sha256,
        ledger,
        "candidate3_prereg",
    )
    _validate_candidate3_prereg(
        candidate3_prereg,
        candidate3_prereg_sha256,
        candidate2_terminal_path,
        candidate2_terminal_sha256,
    )
    primary_cases, replay_cases, base_bindings = _load_train_cases(ledger)

    nested = complete_nested_oof(primary_cases)
    selected_strategy, final_inner = select_strategy_by_inner_oof(primary_cases)
    final_rule = None
    final_fit: dict[str, Any] = {"reason": "no_inner_oof_eligible_strategy"}
    if selected_strategy is not None:
        final_rule, final_fit = fit_rule_for_strategy(primary_cases, selected_strategy)
    resub_reference = score_cases(primary_cases, None)
    resub_candidate = score_cases(primary_cases, final_rule)
    resub_delta = metric_delta(resub_candidate, resub_reference)
    primary_replay_exact = all(
        predictions_after_rule(primary, final_rule) == predictions_after_rule(replay, final_rule)
        for primary, replay in zip(primary_cases, replay_cases)
    )
    nested_checks = nested["checks"]
    train_gate_passed = bool(
        final_rule is not None
        and all(nested_checks.values())
        and _no_harm(nested["delta"])
        and _no_harm(resub_delta)
        and resub_delta["false_positive"] < 0
        and primary_replay_exact
    )

    bindings = {
        "candidate1_terminal_sha256": CANDIDATE1_TERMINAL_SHA256,
        "candidate2_terminal_sha256": candidate2_terminal_sha256,
        **base_bindings,
        "candidate3_prereg_sha256": candidate3_prereg_sha256,
        "runtime_source_sha256": runtime_source_sha256(),
        "optimizer_source_sha256": optimizer_source_sha256(),
        "tests_source_sha256": _file_sha256(TESTS_PATH),
    }
    fit_audit = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate3_fit_audit_v1",
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "status": "TRAIN_GATE_PASS" if train_gate_passed else "TRAIN_GATE_FAIL",
        "holdout_score_opened": False,
        "source_only": True,
        "uses_prediction_or_candidate_multiplicity": False,
        "adaptive_search_disclosure": candidate3_prereg["adaptive_search_disclosure"],
        "bindings": bindings,
        "complete_nested_oof": nested,
        "final_inner_selection": final_inner,
        "selected_strategy": selected_strategy,
        "selected_deletion_rule": final_rule,
        "final_fit": final_fit,
        "train_resubstitution": {
            "reference": resub_reference,
            "candidate": resub_candidate,
            "delta": resub_delta,
        },
        "primary_replay_exact": primary_replay_exact,
        "train_gate_passed": train_gate_passed,
    }
    fit_audit_sha = _write_canonical(output_root / "fit_audit.json", fit_audit)
    ledger_sha = _write_canonical(
        output_root / "train_read_ledger.json",
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_aug30_train18_candidate3_read_ledger_v1",
            "attempt_ordinal": 3,
            "holdout_score_opened": False,
            "read_count": len(ledger),
            "reads": ledger,
        },
    )

    if not train_gate_passed:
        terminal = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_aug30_train18_candidate3_terminal_v1",
            "candidate_id": 3,
            "attempt_ordinal": 3,
            "status": "REJECTED_TRAIN_GATE",
            "holdout_execution_authorized": False,
            "holdout_score_opened": False,
            "post_terminal_model_or_threshold_mutation": "FORBIDDEN",
            "fit_audit_sha256": fit_audit_sha,
            "train_read_ledger_sha256": ledger_sha,
            "bindings": bindings,
        }
        terminal_sha = _write_canonical(output_root / "terminal_negative.json", terminal)
        return {"status": terminal["status"], "terminal_sha256": terminal_sha}

    checks = {
        "nested_oof_f1_increase": nested["delta"]["f1"] > 0.0,
        "nested_oof_true_positive_nondecrease": nested["delta"]["true_positive"] == 0,
        "nested_oof_false_positive_decrease": nested["delta"]["false_positive"] < 0,
        "nested_oof_exact_case_nondecrease": nested["delta"]["exact_case_count"] >= 0,
        "nested_oof_support_tier_tp_nondecrease": min(
            nested["delta"]["support_tier_tp"].values()
        )
        >= 0,
        "primary_replay_exact": primary_replay_exact,
    }
    pass_artifact = {
        "schema_version": 1,
        "kind": PASS_KIND,
        "status": "PASS",
        "attempt_ordinal": 3,
        "holdout_score_opened": False,
        "selected_deletion_rule": final_rule,
        "bindings": bindings,
        "checks": checks,
        "fit_audit_binding": {
            "kind": "sherman_qc_f1_aug30_train18_candidate3_fit_audit_v1",
            "sha256": fit_audit_sha,
        },
        "train_read_ledger_binding": {
            "kind": "sherman_qc_f1_aug30_train18_candidate3_read_ledger_v1",
            "sha256": ledger_sha,
        },
        "nested_oof_metrics": {
            "reference": {key: nested["reference"][key] for key in ("true_positive", "false_positive", "false_negative", "f1", "exact_case_count", "support_tier_tp")},
            "candidate": {key: nested["candidate"][key] for key in ("true_positive", "false_positive", "false_negative", "f1", "exact_case_count", "support_tier_tp")},
            "delta": {key: nested["delta"][key] for key in ("true_positive", "false_positive", "false_negative", "f1", "exact_case_count", "support_tier_tp")},
        },
    }
    pass_sha = _write_canonical(output_root / "train_gate_pass.json", pass_artifact)
    model = {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "profile": PROFILE,
        "attempt_ordinal": 3,
        "holdout_score_opened": False,
        "deletion_rule": final_rule,
        "source_only": True,
        "uses_prediction_or_candidate_multiplicity": False,
        "forbidden_inference_inputs": [],
        "bindings": bindings,
        "train_gate_pass_binding": {"kind": PASS_KIND, "sha256": pass_sha},
    }
    model_sha = _write_canonical(output_root / "frozen_model.json", model)
    runtime_outputs = []
    for primary, replay in zip(primary_cases, replay_cases):
        primary_result = apply_candidate3(
            primary.v51,
            primary.pool,
            model,
            pass_artifact,
            train_gate_pass_artifact_sha256=pass_sha,
            v51_artifact_sha256=primary.v51_artifact_sha256,
            pool_artifact_sha256=primary.pool_artifact_sha256,
        )
        replay_result = apply_candidate3(
            replay.v51,
            replay.pool,
            model,
            pass_artifact,
            train_gate_pass_artifact_sha256=pass_sha,
            v51_artifact_sha256=replay.v51_artifact_sha256,
            pool_artifact_sha256=replay.pool_artifact_sha256,
        )
        if canonical_sha256(primary_result) != canonical_sha256(replay_result):
            raise ValueError(f"{primary.case_id}: Candidate3 primary/replay output mismatch")
        runtime_outputs.append(
            {
                "case_id": primary.case_id,
                "candidate3_result_sha256": canonical_sha256(primary_result),
            }
        )
    freeze_lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate3_freeze_lock_v1",
        "attempt_ordinal": 3,
        "status": "TRAIN_GATE_PASS_NOT_HOLDOUT_AUTHORIZATION",
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "model_sha256": model_sha,
        "train_gate_pass_sha256": pass_sha,
        "fit_audit_sha256": fit_audit_sha,
        "train_read_ledger_sha256": ledger_sha,
        "bindings": bindings,
        "train_runtime_outputs": runtime_outputs,
    }
    freeze_sha = _write_canonical(output_root / "freeze_lock.json", freeze_lock)
    return {
        "status": freeze_lock["status"],
        "model_sha256": model_sha,
        "train_gate_pass_sha256": pass_sha,
        "freeze_lock_sha256": freeze_sha,
        "nested_oof_metrics": pass_artifact["nested_oof_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate2-terminal", type=Path, required=True)
    parser.add_argument("--candidate2-terminal-sha256", required=True)
    parser.add_argument("--candidate3-prereg", type=Path, required=True)
    parser.add_argument("--candidate3-prereg-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(
        candidate2_terminal_path=args.candidate2_terminal.resolve(),
        candidate2_terminal_sha256=args.candidate2_terminal_sha256,
        candidate3_prereg_path=args.candidate3_prereg.resolve(),
        candidate3_prereg_sha256=args.candidate3_prereg_sha256,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
