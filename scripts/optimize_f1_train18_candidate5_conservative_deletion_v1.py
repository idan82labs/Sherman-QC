#!/usr/bin/env python3
"""Preregistered TRAIN18 optimizer for Candidate5 conservative deletion.

This file is an engineering harness, not fit authorization.  It is stdlib-only
at import time.  A future execution must first validate a canonical Candidate5
preregistration and a distinct one-run authorization artifact, then verify and
execute exact retained source bytes.  The TRAIN label projection is derived
only from its attestation and is opened exactly once, after every non-label
input, as the final artifact read.

Candidate5 searches a finite preregistered family containing either unchanged
V51 or one source-only deletion rule.  Complete cases are the only fold groups.
No case, filename, count, audit, CAD, drawing, part, sender, gold, label, scan
identity, bend ID, or source-hypothesis ID is a model feature.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import types
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 5
ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_SOURCE = Path(__file__).resolve()
PREREGISTRATION = ROOT / "research/f1_aug30_train18_candidate5_preregistration_v1.json"
FIT_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate5_fit_authorization_v1.json"
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate5-conservative-deletion-v1"

RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate5_conservative_deletion_v1.py"
PREFLIGHT_SOURCE = ROOT / "scripts/preflight_f1_train18_candidate5_semantic_cache_v1.py"
RUNTIME_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate5_conservative_deletion_v1.py"
PREFLIGHT_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate5_semantic_cache_preflight_v1.py"
OPTIMIZER_TESTS_SOURCE = ROOT / "tests/test_optimize_f1_train18_candidate5_conservative_deletion_v1.py"
EVALUATOR_SOURCE = ROOT / "scripts/evaluate_f1_against_manual_gold.py"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
CANDIDATE1_TERMINAL = ROOT / "research/f1_aug30_train18_candidate1_terminal_negative_v1.json"
CANDIDATE2_TERMINAL = ROOT / "research/f1_aug30_train18_candidate2_terminal_negative_v1.json"
CANDIDATE3_INCIDENT = ROOT / "research/f1_aug30_train18_candidate3_prefit_custody_incident_v1.json"
CANDIDATE3_TERMINAL = ROOT / "research/f1_aug30_train18_candidate3_terminal_negative_v1.json"
CANDIDATE4_PREFLIGHT_FAILURE = ROOT / "research/f1_aug30_train18_candidate4_cache_preflight_failure_v1.json"
CANDIDATE4_TERMINAL = ROOT / "research/f1_aug30_train18_candidate4_terminal_negative_v1.json"

CUSTODY_ROOT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
LABEL_ATTESTATION = CUSTODY_ROOT / "labels/train_label_projection_attestation.json"
CACHE_ATTESTATION = CUSTODY_ROOT / "cache/train_v51_cache_manifest_attestation.json"
CACHE_MANIFEST = CUSTODY_ROOT / "cache/train_v51_cache_manifest.json"
CUSTODY_PASS = CUSTODY_ROOT / "independent_validation_report.json"
ALLOWED_SEMANTIC_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"

C5_PREFLIGHT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate5-semantic-cache-preflight-v1"
C5_PREFLIGHT_REPORT = C5_PREFLIGHT_ROOT / "preflight_report.json"
C5_PREFLIGHT_LOCK = C5_PREFLIGHT_ROOT / "preflight_lock.json"
C5_PREFLIGHT_LEDGER = C5_PREFLIGHT_ROOT / "preflight_read_ledger.json"

FIXED_BINDINGS = {
    "sealed_split": (SPLIT, "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185", 10706),
    "original_preregistration": (ORIGINAL_PREREG, "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a", 2651),
    "train_allowlist": (ALLOWLIST, "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0", 2428),
    "candidate1_terminal": (CANDIDATE1_TERMINAL, "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8", 3989),
    "candidate2_terminal": (CANDIDATE2_TERMINAL, "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515", 7001),
    "candidate3_custody_incident": (CANDIDATE3_INCIDENT, "05c044224918a9400f7c7da23f508ec3ca69c44c1b9f5c3700dcee1c8f4021f1", 2277),
    "candidate3_terminal": (CANDIDATE3_TERMINAL, "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116", 5518),
    "candidate4_preflight_failure": (CANDIDATE4_PREFLIGHT_FAILURE, "f2357864b5f9a545ccb9333e6cd7827bd9958eb64c670e38cbc2ec290ebbd96d", 4650),
    "candidate4_terminal": (CANDIDATE4_TERMINAL, "10d6bd14ad10dfd1ace464e551cb79f65abeb3c12bc0094fb2695f61a1aa2a50", 5384),
    "custody_label_attestation": (LABEL_ATTESTATION, "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad", 2000),
    "custody_cache_attestation": (CACHE_ATTESTATION, "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71", 1834),
    "custody_cache_manifest": (CACHE_MANIFEST, "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022", 19601),
    "custody_staged_validator_pass": (CUSTODY_PASS, "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433", 1854),
    "candidate5_preflight_report": (C5_PREFLIGHT_REPORT, "467132ab13047af60d7b9066ea74e54d08626657c96c8a35075ef4f41a0a1c65", 3838),
    "candidate5_preflight_lock": (C5_PREFLIGHT_LOCK, "21013eff7c1828f44a706ef137392166b4ff627d9b909ea3645f328ecce03564", 2668),
    "candidate5_preflight_read_ledger": (C5_PREFLIGHT_LEDGER, "bd0779ba4cfb228dd09abc35f3843a6ce030031fcc6e9e77b2cb48bc21b767ed", 15706),
}

FROZEN_SOURCE_BINDINGS = {
    "runtime": (RUNTIME_SOURCE, "d3358f7f194f2b5e16bb91a15d5baa0bf4667f5c5d661d0a1a9717a2ec94cae6", 35608),
    "semantic_cache_preflight": (PREFLIGHT_SOURCE, "1f696f5b17116da720d6f4b07cbae202a5209f4ba16a7d5f7cc4b6c0bc5c56cb", 68564),
    "runtime_tests": (RUNTIME_TESTS_SOURCE, "ea49cd87f55af2d9b6971bb60e2b1ae30e2d526580c6c34a0c176d29a3933280", 18608),
    "semantic_cache_preflight_tests": (PREFLIGHT_TESTS_SOURCE, "822619385dfa4c106eee0c9282f0d824169d8e8c8bddbb71fb6ba419168d6fb6", 44970),
    "evaluator": (EVALUATOR_SOURCE, "1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62", 13134),
}

V51_SCAN_ONLY_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"
PROJECTION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_v1"
LABEL_ATTESTATION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_attestation_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
PREREG_KIND = "sherman_qc_f1_aug30_train18_candidate5_preregistration_v1"
FIT_AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate5_fit_authorization_v1"
FIT_AUDIT_KIND = "sherman_qc_f1_aug30_train18_candidate5_fit_audit_core_v1"
READ_LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate5_fit_read_ledger_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate_terminal_v1"
FREEZE_KIND = "sherman_qc_f1_aug30_train18_candidate5_freeze_lock_v1"
PRESEAL_KIND = "sherman_qc_f1_aug30_train18_candidate5_pre_holdout_local_seal_v1"

LINE_LENGTH_THRESHOLDS_MM = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0)
SELECTOR_SCORE_THRESHOLDS = (-1.0, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0)
SOURCE_KIND_VALUES = (
    "cylindrical_standard",
    "paired_return",
    "plane_lattice",
    "scan_mesh_adjacency",
    "source_consensus_midpoint",
    "strict_owned",
    "v41_fragment_tolerant_parent_bridge_sidecar",
)
SELECTION_ROUTE_VALUES = ("bounded_complex", "bounded_small", "full_resolution", "v16", "v41")
SUPPORT_TIERS = ("1", "2", "3")
HOLDOUT_SEQUENCE = (
    {"stage": 1, "tier": "SIMILAR_FAMILY", "case_count": 4},
    {"stage": 2, "tier": "ADJACENT_FAMILY", "case_count": 3},
    {"stage": 3, "tier": "DISTANT_FAMILY", "case_count": 3},
)
MATCH_THRESHOLDS = {
    "angle_deg": 5.0,
    "transverse_offset_mm": 4.0,
    "gold_span_overlap": 0.60,
    "prediction_span_overlap": 0.60,
}

FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    "backend.f1_train18_candidate5_conservative_deletion_v1",
    "scripts.preflight_f1_train18_candidate5_semantic_cache_v1",
    "scripts.evaluate_f1_against_manual_gold",
    "numpy",
    "scipy",
)
FORBIDDEN_INPUT_BASENAMES = frozenset(
    {
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
        "development_execution_manifest.json",
    }
)
PASS_FILENAMES = frozenset(
    {
        "fit_audit_core.json",
        "freeze_lock.json",
        "frozen_model.json",
        "pre_holdout_local_seal.json",
        "train_gate_pass.json",
        "train_read_ledger.json",
    }
)
TERMINAL_FILENAMES = frozenset({"terminal_negative.json"})

_RUNTIME: types.ModuleType | None = None
_EVALUATOR: types.ModuleType | None = None


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if text == "0" * 64:
        raise ValueError(f"{name} must not be an unresolved zero commitment")
    return text


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _lexical_absolute(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} path must not contain parent traversal")
    return Path(os.path.abspath(os.fspath(candidate)))


def _lexical_repo_path(path: Path, *, role: str) -> Path:
    candidate = _lexical_absolute(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    if os.path.commonpath((os.fspath(root), os.fspath(candidate))) != os.fspath(root):
        raise ValueError(f"{role} path leaves repository root")
    return candidate


def _reject_symlink_chain(path: Path, *, role: str, require_exists: bool = True) -> Path:
    candidate = _lexical_repo_path(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    current = candidate
    chain: list[Path] = []
    while current != root:
        chain.append(current)
        current = current.parent
    chain.append(root)
    missing = False
    for item in reversed(chain):
        try:
            metadata = os.lstat(item)
        except FileNotFoundError:
            missing = True
            continue
        if missing:
            raise ValueError(f"{role} has an existing child below a missing ancestor")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path or ancestor must not be a symlink")
        if item != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor must be a directory")
    if require_exists and missing:
        raise FileNotFoundError(f"{role} must exist")
    return candidate


def _read_regular_file_once(
    path: Path,
    *,
    role: str,
    on_read_capability_consumed: Callable[[], None] | None = None,
) -> tuple[bytes, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    metadata = os.lstat(candidate)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{role} must be one non-hard-linked regular file")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate5 optimizer requires O_NOFOLLOW")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        _assert_visible_directory_identity(
            candidate.parent, parent_identity, role=f"{role} parent before leaf open"
        )
        descriptor = os.open(candidate.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        identity = (int(opened.st_dev), int(opened.st_ino))
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError(f"{role} opened object must be one regular link")
        if identity != (int(metadata.st_dev), int(metadata.st_ino)):
            raise ValueError(f"{role} identity changed during open")
        if on_read_capability_consumed is not None:
            on_read_capability_consumed()
        _assert_visible_directory_identity(
            candidate.parent, parent_identity, role=f"{role} parent after leaf open"
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        data = b"".join(chunks)
        if (int(after.st_dev), int(after.st_ino)) != identity or len(data) != int(after.st_size):
            raise ValueError(f"{role} changed during read")
        leaf = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (int(leaf.st_dev), int(leaf.st_ino)) != identity:
            raise ValueError(f"{role} path changed after read")
        _assert_visible_directory_identity(
            candidate.parent, parent_identity, role=f"{role} parent after leaf read"
        )
        return data, identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact_once(path: Path, *, sha256: str, size_bytes: int, role: str) -> tuple[bytes, dict[str, Any]]:
    data, identity = _read_regular_file_once(path, role=role)
    if sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA") or len(data) != size_bytes:
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, {
        "case_id": None,
        "device_id": identity[0],
        "inode": identity[1],
        "path": str(_lexical_repo_path(path, role=role).relative_to(ROOT)),
        "purpose": role,
        "sha256": sha256,
        "size_bytes": len(data),
    }


def _decode_object(data: bytes, *, role: str, canonical_required: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    if canonical_required and canonical_bytes(value) != data:
        raise ValueError(f"{role} must be LF-canonical JSON")
    return value


def fixed_rule_space() -> tuple[dict[str, Any], ...]:
    rules: list[dict[str, Any]] = []
    for threshold in LINE_LENGTH_THRESHOLDS_MM:
        for operator in ("le", "ge"):
            rules.append({"feature": "line_length_mm", "operator": operator, "threshold": threshold})
    for threshold in SELECTOR_SCORE_THRESHOLDS:
        for operator in ("le", "ge"):
            rules.append({"feature": "selector_score", "operator": operator, "threshold": threshold})
    for value in SOURCE_KIND_VALUES:
        rules.append({"feature": "source_kind", "operator": "eq", "value": value})
    for value in SELECTION_ROUTE_VALUES:
        rules.append({"feature": "selection_route", "operator": "eq", "value": value})
    return tuple(rules)


def search_contract() -> dict[str, Any]:
    rules = fixed_rule_space()
    return {
        "allowed_model_family": "UNCHANGED_V51_OR_EXACTLY_ONE_DELETION_RULE",
        "categorical_values": {
            "selection_route": list(SELECTION_ROUTE_VALUES),
            "source_kind": list(SOURCE_KIND_VALUES),
        },
        "line_length_thresholds_mm": list(LINE_LENGTH_THRESHOLDS_MM),
        "missing_feature_policy": "KEEP_V51",
        "numeric_operators": ["le", "ge"],
        "random_seed": None,
        "randomness_used": False,
        "rule_count": len(rules),
        "rule_space_sha256": sha256_bytes(canonical_bytes(list(rules))),
        "selector_score_thresholds": list(SELECTOR_SCORE_THRESHOLDS),
    }


def nested_cv_contract() -> dict[str, Any]:
    return {
        "case_grouping": "COMPLETE_CASE_ONLY",
        "final_selection": "ALL_18_CASES_AS_FIXED_RULE_INNER_VALIDATION",
        "inner_fold_count_per_outer_fold": 17,
        "inner_fold_order": "SEALED_TRAIN_ALLOWLIST_ORDER_WITH_OUTER_CASE_REMOVED",
        "inner_train_case_count_per_fold": 16,
        "inner_validation_complete_case_groups_per_fold": 1,
        "outer_fold_indices": list(range(18)),
        "outer_fold_order": "SEALED_TRAIN_ALLOWLIST_ORDER",
        "outer_validation_case_count": 18,
        "outer_rule_stability": "DISCLOSED_DIAGNOSTIC_NOT_A_PASS_GATE",
        "oof_predictions_are_not_final_model_predictions": True,
        "seed": None,
        "selection_objective_and_tie_break": [
            "MAX_FALSE_POSITIVE_REMOVAL",
            "MAX_EXACT_CASE_GAIN",
            "MAX_MICRO_F1_GAIN",
            "LEXICOGRAPHIC_CANONICAL_RULE_BYTES_ASCENDING",
        ],
    }


def train_gate_contract() -> dict[str, Any]:
    return {
        "all_conjunctive": True,
        "exact_case_count_delta_min": 0,
        "false_positive_delta_max": -1,
        "false_negative_delta_max": 0,
        "micro_f1_delta_strictly_positive": True,
        "per_case_exact_nondecrease": True,
        "per_case_true_positive_nondecrease": True,
        "primary_replay_exact": True,
        "support_tier_tp_delta_min": {"1": 0, "2": 0, "3": 0},
        "true_positive_delta_min": 0,
    }


def holdout_contract() -> dict[str, Any]:
    return {
        "adaptation_between_stages": "FORBIDDEN",
        "external_anchor_required_before_any_holdout_score": True,
        "holdout_execution_authorized_by_fit": False,
        "post_first_holdout_model_or_threshold_mutation": "FORBIDDEN",
        "stage_order": [dict(row) for row in HOLDOUT_SEQUENCE],
    }


def _binding_record(role: str, *, status: str | None = None) -> dict[str, Any]:
    path, sha, size = FIXED_BINDINGS[role]
    result: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha,
        "size_bytes": size,
    }
    if status is not None:
        result["status"] = status
    return result


def expected_lineage_bindings() -> dict[str, Any]:
    return {
        "candidate1_terminal": _binding_record("candidate1_terminal", status="REJECTED_TRAIN_GATE"),
        "candidate2_terminal": _binding_record("candidate2_terminal", status="REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY"),
        "candidate3_custody_incident": _binding_record("candidate3_custody_incident", status="CONFIRMED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"),
        "candidate3_terminal": _binding_record("candidate3_terminal", status="REJECTED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"),
        "candidate4_preflight_failure": _binding_record("candidate4_preflight_failure", status="FAILED_SOURCE_SCHEMA_COMPATIBILITY"),
        "candidate4_terminal": _binding_record("candidate4_terminal", status="REJECTED_PREFIT_SOURCE_SCHEMA_INCOMPATIBILITY"),
        "original_preregistration": _binding_record("original_preregistration"),
        "sealed_split": _binding_record("sealed_split"),
        "train_allowlist": _binding_record("train_allowlist"),
    }


def expected_preflight_bindings() -> dict[str, Any]:
    return {
        "lock": _binding_record("candidate5_preflight_lock", status="PASS_ENGINEERING_ONLY_NONAUTHORIZING"),
        "read_ledger": _binding_record("candidate5_preflight_read_ledger", status="PASS"),
        "report": _binding_record("candidate5_preflight_report", status="PASS_ENGINEERING_ONLY_NONAUTHORIZING"),
    }


def expected_custody_bindings() -> dict[str, Any]:
    return {
        "cache_attestation": _binding_record("custody_cache_attestation"),
        "cache_manifest": _binding_record("custody_cache_manifest"),
        "label_attestation": _binding_record("custody_label_attestation"),
        "label_projection_inherited": {
            "kind": PROJECTION_KIND,
            "path": "output/research/f1-aug30-train-only-custody-v2/labels/train_label_projection.json",
            "sha256": "519ef074b1815058e3807a2b4e5a451786e479587a7afbbd47c31a5ee824e111",
            "size_bytes": 7741,
        },
        "label_projection_inherited_from_attestation_sha256": FIXED_BINDINGS["custody_label_attestation"][1],
        "staged_validator_pass": _binding_record("custody_staged_validator_pass", status="PASS"),
        "v51_scan_only_manifest_sha256": V51_SCAN_ONLY_MANIFEST_SHA256,
    }


def _validate_preregistration(value: Mapping[str, Any], *, prereg_sha256: str) -> None:
    required = {
        "adaptive_search_disclosed",
        "attempt_ordinal",
        "authorization_contract",
        "candidate_id",
        "custody_bindings",
        "deployment_identity_contract",
        "forbidden_access_contract",
        "holdout_contract",
        "holdout_score_opened",
        "kind",
        "lineage_bindings",
        "nested_cv_contract",
        "preflight_bindings",
        "prospective_or_unseen_claim_authorized",
        "publication_contract",
        "schema_version",
        "sealed_preflight_test_temporal_note",
        "search_contract",
        "source_binding_commitment_sha256",
        "source_bindings",
        "status",
        "train_gate_contract",
    }
    if set(value) != required:
        raise ValueError("Candidate5 preregistration exact-key schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != PREREG_KIND
        or value["status"] != "SEALED_ENGINEERING_ONLY_FIT_UNAUTHORIZED"
        or value["candidate_id"] != 5
        or value["attempt_ordinal"] != 5
        or value["holdout_score_opened"] is not False
        or value["adaptive_search_disclosed"] is not True
        or value["prospective_or_unseen_claim_authorized"] is not False
    ):
        raise ValueError("Candidate5 preregistration status contract mismatch")
    if value["search_contract"] != search_contract():
        raise ValueError("Candidate5 fixed search contract mismatch")
    if value["nested_cv_contract"] != nested_cv_contract():
        raise ValueError("Candidate5 nested CV contract mismatch")
    if value["train_gate_contract"] != train_gate_contract():
        raise ValueError("Candidate5 TRAIN gate contract mismatch")
    if value["holdout_contract"] != holdout_contract():
        raise ValueError("Candidate5 holdout contract mismatch")
    if value["lineage_bindings"] != expected_lineage_bindings():
        raise ValueError("Candidate5 prereg lineage binding mismatch")
    if value["preflight_bindings"] != expected_preflight_bindings():
        raise ValueError("Candidate5 prereg preflight binding mismatch")
    if value["custody_bindings"] != expected_custody_bindings():
        raise ValueError("Candidate5 prereg custody binding mismatch")
    if value["sealed_preflight_test_temporal_note"] != {
        "after_one_shot_publication_expected_result": "110_PASS_1_EXPECTED_TEMPORAL_ABSENCE_ASSERTION_FAILURE",
        "bound_test_bytes_must_not_be_mutated": True,
        "prepublication_result": "111_PASS",
    }:
        raise ValueError("Candidate5 sealed preflight test temporal note mismatch")
    authorization = value["authorization_contract"]
    if authorization != {
        "fit_authorization_artifact_required": True,
        "fit_authorization_path": str(FIT_AUTHORIZATION.relative_to(ROOT)),
        "fit_execution_authorized_by_preregistration": False,
        "fresh_process_required": True,
        "single_authorized_fit_run_required": True,
    }:
        raise ValueError("Candidate5 authorization contract mismatch")
    if value["deployment_identity_contract"] != {
        "deployed_model_exact_bytes_match_evaluated_predictions_scope": "FINAL_RULE_SEPARATE_TRAIN_PRIMARY_REPLAY_EVALUATION_ONLY",
        "oof_predictions_are_not_final_model_predictions": True,
        "replay_must_complete_before_any_pass_artifact_publication": True,
    }:
        raise ValueError("Candidate5 deployment identity contract mismatch")
    if value["publication_contract"] != {
        "failed_gate_inventory": ["terminal_negative.json"],
        "no_clobber_atomic_directory_publication": True,
        "pass_inventory": sorted(PASS_FILENAMES),
        "replay_failure_terminalizes": True,
    }:
        raise ValueError("Candidate5 publication contract mismatch")
    if value["forbidden_access_contract"] != {
        "full_28_catalog_or_manifest": "FORBIDDEN",
        "holdout_paths_scores_status_hashes": "FORBIDDEN",
        "label_projection": "ONE_FD_GUARDED_READ_LAST_AFTER_SEPARATE_AUTHORIZATION",
        "pool_artifacts": "FORBIDDEN",
    }:
        raise ValueError("Candidate5 forbidden access contract mismatch")
    _require_sha(prereg_sha256, name="Candidate5 preregistration SHA")
    source_bindings = value["source_bindings"]
    if not isinstance(source_bindings, dict):
        raise ValueError("Candidate5 prereg source bindings must be an object")
    if value["source_binding_commitment_sha256"] != sha256_bytes(
        canonical_bytes(source_bindings)
    ):
        raise ValueError("Candidate5 prereg source binding commitment mismatch")


@dataclass(frozen=True)
class BootstrapContext:
    preregistration: Mapping[str, Any]
    preregistration_sha256: str
    source_bytes: Mapping[str, bytes]
    source_bindings: Mapping[str, Mapping[str, Any]]
    bootstrap_entries: tuple[dict[str, Any], ...]


def _reject_preloaded_sources(paths: Sequence[Path]) -> None:
    current = sys.modules.get(__name__)
    controlled: dict[tuple[int, int], Path] = {}
    resolved: set[Path] = set()
    for path in paths:
        canonical = _reject_symlink_chain(path, role="Candidate5 controlled source")
        metadata = os.stat(canonical, follow_symlinks=True)
        controlled[(int(metadata.st_dev), int(metadata.st_ino))] = canonical
        resolved.add(canonical.resolve(strict=True))
    for name in FORBIDDEN_PRELOADED_MODULES:
        if name in sys.modules:
            raise RuntimeError(f"Candidate5 optimizer rejects preloaded module: {name}")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        loaded_file = getattr(module, "__file__", None)
        if not isinstance(loaded_file, str) or not loaded_file:
            continue
        path = _lexical_absolute(Path(loaded_file), role=f"preloaded module {name}")
        try:
            target = path.resolve(strict=True)
            metadata = os.stat(path, follow_symlinks=True)
        except OSError:
            continue
        if target in resolved or (int(metadata.st_dev), int(metadata.st_ino)) in controlled:
            raise RuntimeError(f"Candidate5 optimizer rejects preloaded source alias: {name}")


def bootstrap_verified_sources(*, preregistration_sha256: str) -> BootstrapContext:
    expected_paths_before_prereg = (
        RUNTIME_SOURCE,
        PREFLIGHT_SOURCE,
        RUNTIME_TESTS_SOURCE,
        PREFLIGHT_TESTS_SOURCE,
        OPTIMIZER_SOURCE,
        OPTIMIZER_TESTS_SOURCE,
        EVALUATOR_SOURCE,
    )
    _reject_preloaded_sources(expected_paths_before_prereg)
    prereg_data, prereg_entry = _read_exact_once(
        PREREGISTRATION,
        sha256=_require_sha(preregistration_sha256, name="Candidate5 preregistration SHA"),
        size_bytes=os.lstat(PREREGISTRATION).st_size,
        role="CANDIDATE5_PREREGISTRATION",
    )
    prereg = _decode_object(prereg_data, role="Candidate5 preregistration")
    _validate_preregistration(prereg, prereg_sha256=preregistration_sha256)
    source_bindings = prereg["source_bindings"]
    expected_names = {
        "evaluator",
        "optimizer",
        "optimizer_tests",
        "publication_helper",
        "runtime",
        "runtime_tests",
        "semantic_cache_preflight_tests",
    }
    if not isinstance(source_bindings, dict) or set(source_bindings) != expected_names:
        raise ValueError("Candidate5 prereg source binding inventory mismatch")
    expected_paths = {
        "evaluator": EVALUATOR_SOURCE,
        "optimizer": OPTIMIZER_SOURCE,
        "optimizer_tests": OPTIMIZER_TESTS_SOURCE,
        "publication_helper": PREFLIGHT_SOURCE,
        "runtime": RUNTIME_SOURCE,
        "runtime_tests": RUNTIME_TESTS_SOURCE,
        "semantic_cache_preflight_tests": PREFLIGHT_TESTS_SOURCE,
    }
    source_bytes: dict[str, bytes] = {}
    entries = [prereg_entry]
    identities: set[tuple[int, int]] = {(prereg_entry["device_id"], prereg_entry["inode"])}
    normalized_bindings: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        binding = source_bindings[name]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"Candidate5 source binding schema mismatch: {name}")
        expected_path = expected_paths[name]
        if binding["path"] != str(expected_path.relative_to(ROOT)):
            raise ValueError(f"Candidate5 source binding path mismatch: {name}")
        data, entry = _read_exact_once(
            expected_path,
            sha256=binding["sha256"],
            size_bytes=_strict_int(binding["size_bytes"], name=f"{name} size"),
            role=f"CANDIDATE5_SOURCE_{name.upper()}",
        )
        identity = (entry["device_id"], entry["inode"])
        if identity in identities:
            raise ValueError("Candidate5 prereg/source inputs contain a hard-link alias")
        identities.add(identity)
        entries.append(entry)
        source_bytes[name] = data
        normalized_bindings[name] = dict(binding)
    for name, (path, sha, size) in FROZEN_SOURCE_BINDINGS.items():
        prereg_name = "publication_helper" if name == "semantic_cache_preflight" else name
        if prereg_name in normalized_bindings and normalized_bindings[prereg_name] != {
            "path": str(path.relative_to(ROOT)), "sha256": sha, "size_bytes": size
        }:
            raise ValueError(f"Candidate5 frozen source lineage mismatch: {name}")
    return BootstrapContext(
        preregistration=prereg,
        preregistration_sha256=preregistration_sha256,
        source_bytes=source_bytes,
        source_bindings=normalized_bindings,
        bootstrap_entries=tuple(entries),
    )


def _load_module(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError(f"Candidate5 refuses to replace preloaded verified module: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    module.__loader__ = None
    sys.modules[name] = module
    try:
        code = compile(data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def configure_source_free_dependencies(runtime: types.ModuleType, evaluator: types.ModuleType) -> None:
    global _RUNTIME, _EVALUATOR
    _RUNTIME = runtime
    _EVALUATOR = evaluator


@dataclass
class GuardedReader:
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)
    allowed: dict[Path, dict[str, Any]] = field(default_factory=dict)
    identities: dict[tuple[int, int], Path] = field(default_factory=dict)
    projection_path: Path | None = None
    projection_read_count: int = 0
    projection_read_complete: bool = False

    def __post_init__(self) -> None:
        for row in self.entries:
            identity = (int(row["device_id"]), int(row["inode"]))
            path = _lexical_repo_path(ROOT / str(row["path"]), role=str(row["purpose"]))
            prior = self.identities.get(identity)
            if prior is not None and prior != path:
                raise ValueError("Candidate5 bootstrap ledger contains a hard-link alias")
            self.identities[identity] = path

    def _deny(self, path: Path, purpose: str, reason: str) -> None:
        self.denied_attempts.append(
            {"path": str(path), "purpose": purpose, "reason": reason}
        )
        raise PermissionError(f"Candidate5 read denied ({reason}): {path}")

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        sha256: str,
        size_bytes: int,
        case_id: str | None = None,
    ) -> None:
        candidate = _lexical_repo_path(path, role=purpose)
        if self.projection_read_complete:
            self._deny(candidate, purpose, "NONPROJECTION_AUTHORIZATION_AFTER_PROJECTION")
        lowered_parts = {part.lower() for part in candidate.parts}
        if candidate.name in FORBIDDEN_INPUT_BASENAMES:
            self._deny(candidate, purpose, "FORBIDDEN_FULL_COHORT_BASENAME")
        if "POOL" in purpose or any("holdout" in part for part in lowered_parts):
            self._deny(candidate, purpose, "POOL_OR_HOLDOUT_CAPABILITY_FORBIDDEN")
        if purpose == "TRAIN_LABEL_PROJECTION":
            self._deny(candidate, purpose, "USE_AUTHORIZE_PROJECTION_LAST")
        canonical = _reject_symlink_chain(candidate, role=purpose)
        metadata = os.lstat(canonical)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{purpose} must be one non-hard-linked regular file")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        prior_path = self.identities.get(identity)
        if prior_path is not None and prior_path != canonical:
            raise ValueError(f"Candidate5 input hard-link alias: {canonical} aliases {prior_path}")
        binding = {
            "case_id": case_id,
            "identity": identity,
            "purpose": purpose,
            "sha256": _require_sha(sha256, name=f"{purpose} SHA"),
            "size_bytes": _strict_int(size_bytes, name=f"{purpose} size"),
        }
        if binding["size_bytes"] <= 0:
            raise ValueError(f"{purpose} size must be positive")
        prior = self.allowed.get(canonical)
        if prior is not None and prior != binding:
            raise ValueError(f"Candidate5 conflicting authorization: {canonical}")
        self.allowed[canonical] = binding
        self.identities[identity] = canonical

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        candidate = _lexical_repo_path(path, role=purpose)
        if self.projection_read_complete:
            self._deny(candidate, purpose, "ARTIFACT_READ_AFTER_PROJECTION")
        binding = self.allowed.get(candidate)
        if binding is None or binding["purpose"] != purpose:
            self._deny(candidate, purpose, "NOT_EXACTLY_AUTHORIZED")
        data, identity = _read_regular_file_once(candidate, role=purpose)
        if identity != binding["identity"]:
            raise ValueError(f"Candidate5 {purpose} identity changed")
        if sha256_bytes(data) != binding["sha256"] or len(data) != binding["size_bytes"]:
            raise ValueError(f"Candidate5 {purpose} byte binding mismatch")
        self.entries.append(
            {
                "case_id": binding["case_id"],
                "device_id": identity[0],
                "inode": identity[1],
                "path": str(candidate.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": binding["sha256"],
                "size_bytes": len(data),
            }
        )
        return data

    def authorize_projection_last(
        self,
        path: Path,
        *,
        sha256: str,
        size_bytes: int,
        all_nonprojection_reads_complete: bool,
    ) -> None:
        if not all_nonprojection_reads_complete:
            raise PermissionError("Candidate5 projection cannot be authorized before all other reads")
        if self.projection_path is not None or self.projection_read_count != 0:
            raise PermissionError("Candidate5 projection capability is single-use")
        candidate = _lexical_repo_path(path, role="TRAIN_LABEL_PROJECTION")
        canonical = _reject_symlink_chain(candidate, role="TRAIN_LABEL_PROJECTION")
        metadata = os.lstat(canonical)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Candidate5 projection must be one non-hard-linked regular file")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        prior = self.identities.get(identity)
        if prior is not None and prior != canonical:
            raise ValueError("Candidate5 projection hard-link aliases an earlier input")
        self.allowed[canonical] = {
            "case_id": None,
            "identity": identity,
            "purpose": "TRAIN_LABEL_PROJECTION",
            "sha256": _require_sha(sha256, name="TRAIN projection SHA"),
            "size_bytes": _strict_int(size_bytes, name="TRAIN projection size"),
        }
        self.identities[identity] = canonical
        self.projection_path = canonical

    def read_projection_once_last(self) -> bytes:
        if self.projection_path is None or self.projection_read_count != 0:
            raise PermissionError("Candidate5 projection is absent or already consumed")
        binding = self.allowed[self.projection_path]
        def consume_capability() -> None:
            # The capability is consumed once the exact leaf fd is opened and
            # fstat-verified for reading, before EOF/data, parent post-open
            # checks, byte binding, JSON, or schema validation.  A truncated
            # file and every later failure therefore remain single-use.
            self.projection_read_count = 1
            self.projection_read_complete = True

        data, identity = _read_regular_file_once(
            self.projection_path,
            role="TRAIN_LABEL_PROJECTION",
            on_read_capability_consumed=consume_capability,
        )
        if (
            identity != binding["identity"]
            or sha256_bytes(data) != binding["sha256"]
            or len(data) != binding["size_bytes"]
        ):
            raise ValueError("Candidate5 projection exact byte binding mismatch")
        self.entries.append(
            {
                "case_id": None,
                "device_id": identity[0],
                "inode": identity[1],
                "path": str(self.projection_path.relative_to(ROOT)),
                "purpose": "TRAIN_LABEL_PROJECTION_LAST_READ",
                "sha256": binding["sha256"],
                "size_bytes": len(data),
            }
        )
        return data


def _validate_fit_authorization(
    value: Mapping[str, Any], *, authorization_sha256: str, preregistration_sha256: str
) -> None:
    required = {
        "attempt_ordinal",
        "auditor_authorized",
        "authorized_run_limit",
        "authorized_scope",
        "cache_semantic_read_authorized",
        "candidate_id",
        "candidate5_preregistration_sha256",
        "fit_authorized",
        "fresh_process_verified",
        "full_28_access_authorized",
        "holdout_access_authorized",
        "kind",
        "label_projection_read_authorized_once_last",
        "parent_authorized",
        "pool_access_authorized",
        "schema_version",
        "source_binding_commitment_sha256",
        "status",
    }
    if set(value) != required:
        raise ValueError("Candidate5 fit authorization schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != FIT_AUTHORIZATION_KIND
        or value["status"] != "AUTHORIZED_ONCE"
        or value["candidate_id"] != 5
        or value["attempt_ordinal"] != 5
        or value["fit_authorized"] is not True
        or value["fresh_process_verified"] is not True
        or value["parent_authorized"] is not True
        or value["auditor_authorized"] is not True
        or value["authorized_run_limit"] != 1
        or value["authorized_scope"] != "CANDIDATE5_TRAIN18_FIT_ONLY"
        or value["cache_semantic_read_authorized"] is not True
        or value["label_projection_read_authorized_once_last"] is not True
        or value["pool_access_authorized"] is not False
        or value["full_28_access_authorized"] is not False
        or value["holdout_access_authorized"] is not False
        or value["candidate5_preregistration_sha256"] != preregistration_sha256
    ):
        raise ValueError("Candidate5 fit authorization status contract mismatch")
    _require_sha(authorization_sha256, name="Candidate5 fit authorization SHA")
    _require_sha(
        value["source_binding_commitment_sha256"],
        name="Candidate5 authorized source commitment",
    )


def _validate_fixed_control(role: str, value: Mapping[str, Any]) -> None:
    if role == "candidate1_terminal":
        if value.get("candidate_id") != 1 or value.get("status") != "REJECTED_TRAIN_GATE":
            raise ValueError("Candidate1 terminal mismatch")
    elif role == "candidate2_terminal":
        if (
            value.get("candidate_id") != 2
            or value.get("status") != "REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY"
            or value.get("candidate2_continuation_authorized") is not False
        ):
            raise ValueError("Candidate2 terminal mismatch")
    elif role == "candidate3_custody_incident":
        if (
            value.get("status") != "CONFIRMED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"
            or value.get("terminalization_required") is not True
        ):
            raise ValueError("Candidate3 incident mismatch")
    elif role == "candidate3_terminal":
        if (
            value.get("candidate_id") != 3
            or value.get("status") != "REJECTED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"
            or value.get("candidate3_continuation_authorized") is not False
        ):
            raise ValueError("Candidate3 terminal mismatch")
    elif role == "candidate4_preflight_failure":
        if (
            value.get("candidate_id") != 4
            or value.get("status") != "FAILED_SOURCE_SCHEMA_COMPATIBILITY"
            or value.get("continuation_authorized") is not False
        ):
            raise ValueError("Candidate4 preflight failure mismatch")
    elif role == "candidate4_terminal":
        if (
            value.get("candidate_id") != 4
            or value.get("status") != "REJECTED_PREFIT_SOURCE_SCHEMA_INCOMPATIBILITY"
            or value.get("candidate4_continuation_authorized") is not False
        ):
            raise ValueError("Candidate4 terminal mismatch")
    elif role == "candidate5_preflight_report":
        if (
            value.get("attempt_ordinal") != 5
            or value.get("status") != "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
            or value.get("all_primary_replay_bytes_exact") is not True
            or value.get("pool_artifacts_opened") is not False
            or value.get("label_projection_path_resolved_or_opened") is not False
        ):
            raise ValueError("Candidate5 preflight report mismatch")
    elif role == "candidate5_preflight_lock":
        if (
            value.get("attempt_ordinal") != 5
            or value.get("status") != "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
            or value.get("candidate5_fit_authorized") is not False
        ):
            raise ValueError("Candidate5 preflight lock mismatch")
    elif role == "candidate5_preflight_read_ledger":
        assertions = value.get("assertions") or {}
        if (
            value.get("attempt_ordinal") != 5
            or value.get("status") != "PASS"
            or assertions.get("pool_artifacts_opened") is not False
            or assertions.get("label_projection_path_resolved_or_opened") is not False
        ):
            raise ValueError("Candidate5 preflight ledger mismatch")


def _validate_label_attestation(
    value: Mapping[str, Any], *, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != LABEL_ATTESTATION_KIND
        or value.get("canonical_json") is not True
        or value.get("case_count") != 18
        or value.get("file_inventory_count") != 18
        or value.get("train_exact_match") is not True
        or value.get("train_holdout_intersection_count") != 0
        or value.get("holdout_label_row_emit_count") != 0
        or value.get("holdout_case_file_emit_count") != 0
        or value.get("optimizer_holdout_access_count") != 0
        or value.get("projection_contains_only_train") is not True
        or value.get("independent_validation_passed") is not True
        or value.get("split_sha256") != FIXED_BINDINGS["sealed_split"][1]
        or value.get("train_allowlist_sha256") != FIXED_BINDINGS["train_allowlist"][1]
    ):
        raise ValueError("Candidate5 label attestation custody contract mismatch")
    projection = value.get("projection_binding")
    if not isinstance(projection, dict) or set(projection) != {
        "kind", "path", "sha256", "size_bytes"
    }:
        raise ValueError("Candidate5 projection attestation binding schema mismatch")
    inherited = preregistration["custody_bindings"]["label_projection_inherited"]
    if projection != inherited or projection["kind"] != PROJECTION_KIND:
        raise ValueError("Candidate5 projection binding differs from preregistered attestation inheritance")
    if Path(str(projection["path"])).is_absolute() or Path(str(projection["path"])).name != "train_label_projection.json":
        raise ValueError("Candidate5 projection path is not the exact TRAIN projection")
    _require_sha(projection["sha256"], name="Candidate5 projection SHA")
    if _strict_int(projection["size_bytes"], name="Candidate5 projection size") <= 0:
        raise ValueError("Candidate5 projection size must be positive")
    staged = value.get("staged_validator_pass_binding") or {}
    if staged.get("sha256") != FIXED_BINDINGS["custody_staged_validator_pass"][1]:
        raise ValueError("Candidate5 label attestation custody PASS mismatch")
    return dict(projection)


def _validate_cache_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != CACHE_ATTESTATION_KIND
        or value.get("canonical_json") is not True
        or value.get("case_count") != 18
        or value.get("record_count") != 36
        or value.get("train_exact_match") is not True
        or value.get("train_holdout_intersection_count") != 0
        or value.get("holdout_entry_count") != 0
        or value.get("holdout_hash_count") != 0
        or value.get("holdout_status_count") != 0
        or value.get("optimizer_holdout_access_count") != 0
        or value.get("independent_validation_passed") is not True
        or value.get("split_sha256") != FIXED_BINDINGS["sealed_split"][1]
        or value.get("train_allowlist_sha256") != FIXED_BINDINGS["train_allowlist"][1]
    ):
        raise ValueError("Candidate5 cache attestation custody contract mismatch")
    binding = value.get("manifest_binding")
    if not isinstance(binding, dict) or binding != {
        "kind": CACHE_MANIFEST_KIND,
        "path": str(CACHE_MANIFEST.relative_to(ROOT)),
        "sha256": FIXED_BINDINGS["custody_cache_manifest"][1],
        "size_bytes": FIXED_BINDINGS["custody_cache_manifest"][2],
    }:
        raise ValueError("Candidate5 cache manifest attestation binding mismatch")
    staged = value.get("staged_validator_pass_binding") or {}
    if staged.get("sha256") != FIXED_BINDINGS["custody_staged_validator_pass"][1]:
        raise ValueError("Candidate5 cache attestation custody PASS mismatch")
    return dict(binding)


def _validate_split_allowlist(
    split: Mapping[str, Any], allowlist: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, str], str]:
    if (
        split.get("kind") != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
        or split.get("label_or_outcome_fields_present") is not False
    ):
        raise ValueError("Candidate5 sealed split contract mismatch")
    cases = allowlist.get("cases")
    if (
        allowlist.get("kind") != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
        or allowlist.get("reject_any_case_not_listed") is not True
        or not isinstance(cases, list)
        or len(cases) != 18
    ):
        raise ValueError("Candidate5 TRAIN allowlist contract mismatch")
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    if (
        len(set(train_ids)) != 18
        or any(not case_id for case_id in train_ids)
        or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids
    ):
        raise ValueError("Candidate5 split/allowlist order or membership mismatch")
    scan_by_case = {
        str(row["case_id"]): _require_sha(row.get("scan_sha256"), name="TRAIN scan SHA")
        for row in cases
    }
    commitment = sha256_bytes(
        canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    return train_ids, scan_by_case, commitment


def _validate_cache_manifest(
    value: Mapping[str, Any],
    *,
    train_ids: tuple[str, ...],
    scan_by_case: Mapping[str, str],
    train_commitment: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    expected_keys = {
        "case_count",
        "holdout_entry_count",
        "kind",
        "record_count",
        "records",
        "schema_version",
        "source_only",
        "split_sha256",
        "train_allowlist_sha256",
        "train_set_commitment_sha256",
        "v51_scan_only_manifest_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("kind") != CACHE_MANIFEST_KIND
        or value.get("source_only") is not True
        or value.get("case_count") != 18
        or value.get("record_count") != 36
        or value.get("holdout_entry_count") != 0
        or value.get("split_sha256") != FIXED_BINDINGS["sealed_split"][1]
        or value.get("train_allowlist_sha256") != FIXED_BINDINGS["train_allowlist"][1]
        or value.get("train_set_commitment_sha256") != train_commitment
        or value.get("v51_scan_only_manifest_sha256") != V51_SCAN_ONLY_MANIFEST_SHA256
    ):
        raise ValueError("Candidate5 cache manifest header mismatch")
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("Candidate5 cache manifest must contain exactly 36 records")
    expected_pairs = [
        (case_id, pass_name)
        for case_id in train_ids
        for pass_name in ("primary", "replay")
    ]
    record_keys = {
        "case_id",
        "pass",
        "pool_path",
        "pool_sha256",
        "runtime_case_id",
        "scan_sha256",
        "semantic_path",
        "semantic_sha256",
    }
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    semantic_paths: set[Path] = set()
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    binding_rows: list[dict[str, Any]] = []
    for raw, pair in zip(rows, expected_pairs):
        if not isinstance(raw, dict) or set(raw) != record_keys:
            raise ValueError("Candidate5 cache record schema mismatch")
        if (str(raw["case_id"]), str(raw["pass"])) != pair:
            raise PermissionError("Candidate5 cache record order/membership mismatch")
        case_id, pass_name = pair
        runtime_id = str(raw["runtime_case_id"])
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id
            or case_by_runtime.setdefault(runtime_id, case_id) != case_id
        ):
            raise ValueError("Candidate5 cache runtime identity mapping mismatch")
        if _require_sha(raw["scan_sha256"], name="manifest scan SHA") != scan_by_case[case_id]:
            raise ValueError("Candidate5 manifest scan identity differs from allowlist")
        relative = Path(str(raw["semantic_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Candidate5 semantic path must be repository-relative")
        semantic_path = _lexical_repo_path(ROOT / relative, role="TRAIN_V51_SEMANTIC")
        expected_path = _lexical_repo_path(
            ALLOWED_SEMANTIC_ROOT / pass_name / runtime_id / "semantic_result.json",
            role="expected TRAIN semantic path",
        )
        if semantic_path != expected_path or semantic_path in semantic_paths:
            raise ValueError("Candidate5 semantic path is aliased or outside exact cache layout")
        semantic_paths.add(semantic_path)
        semantic_sha = _require_sha(raw["semantic_sha256"], name="semantic SHA")
        # Pool strings/hashes are inert manifest fields.  Never form a pool Path.
        _require_sha(raw["pool_sha256"], name="inert pool manifest SHA")
        if not isinstance(raw["pool_path"], str) or not raw["pool_path"]:
            raise ValueError("Candidate5 inert pool manifest field is malformed")
        mapping[pair] = {
            "case_id": case_id,
            "pass": pass_name,
            "runtime_case_id": runtime_id,
            "scan_sha256": scan_by_case[case_id],
            "semantic_path": semantic_path,
            "semantic_sha256": semantic_sha,
        }
        binding_rows.append(
            {
                "case_id": case_id,
                "pass": pass_name,
                "runtime_case_id": runtime_id,
                "scan_sha256": scan_by_case[case_id],
                "semantic_path": str(semantic_path.relative_to(ROOT)),
                "semantic_sha256": semantic_sha,
            }
        )
    for case_id in train_ids:
        primary = mapping[(case_id, "primary")]
        replay = mapping[(case_id, "replay")]
        if primary["semantic_sha256"] != replay["semantic_sha256"]:
            raise ValueError("Candidate5 primary/replay semantic hash mismatch")
    return mapping, sha256_bytes(canonical_bytes(binding_rows))


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    runtime_case_id: str
    scan_sha256: str
    primary_semantic_bytes: bytes
    primary_semantic_sha256: str
    primary_predictions: tuple[dict[str, Any], ...]
    replay_semantic_bytes: bytes
    replay_semantic_sha256: str
    replay_predictions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    runtime_case_id: str
    scan_sha256: str
    gold: tuple[dict[str, Any], ...]
    primary_semantic_bytes: bytes
    primary_semantic_sha256: str
    primary_predictions: tuple[dict[str, Any], ...]
    replay_semantic_bytes: bytes
    replay_semantic_sha256: str
    replay_predictions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PostProjectionState:
    """Value-free state sufficient to terminalize after label bytes are read."""

    reader: GuardedReader
    input_bindings: Mapping[str, str]
    candidate3_terminal_bytes: bytes
    candidate4_terminal_bytes: bytes


class PostProjectionFailure(RuntimeError):
    """Carries no label values; it only proves the one read was consumed."""

    def __init__(self, state: PostProjectionState, original: BaseException) -> None:
        super().__init__(f"Candidate5 post-projection failure: {type(original).__name__}")
        self.state = state
        self.reason_class = type(original).__name__


def _point(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-coordinate list")
    result = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{name}[{index}] must be numeric")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


def _validate_projection(
    value: Mapping[str, Any],
    *,
    train_ids: tuple[str, ...],
    scan_by_case: Mapping[str, str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != PROJECTION_KIND
        or value.get("case_count") != 18
        or value.get("split_sha256") != FIXED_BINDINGS["sealed_split"][1]
        or value.get("train_allowlist_sha256") != FIXED_BINDINGS["train_allowlist"][1]
    ):
        raise ValueError("Candidate5 TRAIN projection header mismatch")
    rows = value.get("cases")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("Candidate5 TRAIN projection must contain exactly 18 cases")
    forbidden_case_keys = {
        "cad",
        "drawing",
        "expected_count",
        "filename",
        "part_id",
        "sender_count",
    }
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    observed_order: list[str] = []
    for raw in rows:
        if not isinstance(raw, dict) or forbidden_case_keys.intersection(raw):
            raise ValueError("Candidate5 projection case row exposes a forbidden field")
        case_id = str(raw.get("case_id") or "")
        if case_id not in scan_by_case or case_id in result:
            raise PermissionError("Candidate5 projection includes a non-TRAIN or duplicate case")
        if raw.get("scan_sha256") != scan_by_case[case_id]:
            raise ValueError("Candidate5 projection scan binding mismatch")
        bends = raw.get("bends")
        if not isinstance(bends, list):
            raise ValueError("Candidate5 projection bends must be a list")
        bend_ids: set[str] = set()
        validated: list[dict[str, Any]] = []
        for bend in bends:
            if not isinstance(bend, dict):
                raise ValueError("Candidate5 gold bend must be an object")
            bend_id = str(bend.get("bend_id") or "")
            if not bend_id or bend_id in bend_ids:
                raise ValueError("Candidate5 gold bend IDs must be unique and nonempty")
            bend_ids.add(bend_id)
            start = _point(bend.get("line_start"), name="gold.line_start")
            end = _point(bend.get("line_end"), name="gold.line_end")
            if math.dist(start, end) <= 1e-9:
                raise ValueError("Candidate5 gold bend line must be nondegenerate")
            support = _strict_int(
                bend.get("reviewer_support_count"), name="reviewer_support_count"
            )
            if support not in (1, 2, 3):
                raise ValueError("Candidate5 gold support tier must be 1, 2, or 3")
            # Only the validated geometry and support tier cross into scoring.
            # In particular, evaluator-preferred aliases such as
            # ``span_endpoints`` and arbitrary label metadata are deliberately
            # ignored rather than propagated.
            validated.append(
                {
                    "bend_id": bend_id,
                    "line_end": list(end),
                    "line_start": list(start),
                    "reviewer_support_count": support,
                }
            )
        result[case_id] = tuple(validated)
        observed_order.append(case_id)
    if tuple(observed_order) != train_ids:
        raise ValueError("Candidate5 projection order differs from sealed TRAIN order")
    return result


def _load_verified_runtime(context: BootstrapContext) -> types.ModuleType:
    module = _load_module(
        "_candidate5_verified_runtime",
        RUNTIME_SOURCE,
        context.source_bytes["runtime"],
        package="backend",
    )
    if module.ATTEMPT_ORDINAL != 5:
        raise ValueError("Candidate5 runtime attempt mismatch")
    return module


def _load_verified_evaluator(context: BootstrapContext) -> types.ModuleType:
    return _load_module(
        "_candidate5_verified_evaluator",
        EVALUATOR_SOURCE,
        context.source_bytes["evaluator"],
        package="scripts",
    )


def _load_verified_publication_helper(context: BootstrapContext) -> types.ModuleType:
    return _load_module(
        "_candidate5_verified_publication_helper",
        PREFLIGHT_SOURCE,
        context.source_bytes["publication_helper"],
        package="scripts",
    )


def _read_json_control(
    reader: GuardedReader,
    *,
    role: str,
    path: Path,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    reader.authorize(path, purpose=role.upper(), sha256=sha256, size_bytes=size_bytes)
    data = reader.read_bytes(path, purpose=role.upper())
    return _decode_object(data, role=role)


def _read_projection_and_prepare_cases(
    reader: GuardedReader,
    state: PostProjectionState,
    *,
    train_ids: tuple[str, ...],
    scan_by_case: Mapping[str, str],
    source_cases: Sequence[SourceCase],
) -> list[PreparedCase]:
    """Consume the one label read and terminalize every subsequent failure."""

    try:
        projection_bytes = reader.read_projection_once_last()
        projection = _decode_object(
            projection_bytes, role="TRAIN label projection", canonical_required=True
        )
        gold_by_case = _validate_projection(
            projection, train_ids=train_ids, scan_by_case=scan_by_case
        )
        return [
            PreparedCase(
                case_id=case.case_id,
                runtime_case_id=case.runtime_case_id,
                scan_sha256=case.scan_sha256,
                gold=gold_by_case[case.case_id],
                primary_semantic_bytes=case.primary_semantic_bytes,
                primary_semantic_sha256=case.primary_semantic_sha256,
                primary_predictions=case.primary_predictions,
                replay_semantic_bytes=case.replay_semantic_bytes,
                replay_semantic_sha256=case.replay_semantic_sha256,
                replay_predictions=case.replay_predictions,
            )
            for case in source_cases
        ]
    except BaseException as error:
        if reader.projection_read_count == 1:
            raise PostProjectionFailure(state, error) from error
        raise


def load_train_inputs(
    context: BootstrapContext,
    *,
    fit_authorization_sha256: str,
) -> tuple[list[PreparedCase], GuardedReader, dict[str, str], bytes, bytes]:
    reader = GuardedReader(entries=[dict(row) for row in context.bootstrap_entries])

    auth_size = os.lstat(FIT_AUTHORIZATION).st_size
    authorization = _read_json_control(
        reader,
        role="fit_authorization",
        path=FIT_AUTHORIZATION,
        sha256=fit_authorization_sha256,
        size_bytes=auth_size,
    )
    _validate_fit_authorization(
        authorization,
        authorization_sha256=fit_authorization_sha256,
        preregistration_sha256=context.preregistration_sha256,
    )
    source_commitment = sha256_bytes(canonical_bytes(context.source_bindings))
    if authorization["source_binding_commitment_sha256"] != source_commitment:
        raise ValueError("Candidate5 authorization source commitment mismatch")

    controls: dict[str, dict[str, Any]] = {}
    for role, (path, sha, size) in FIXED_BINDINGS.items():
        controls[role] = _read_json_control(
            reader, role=role, path=path, sha256=sha, size_bytes=size
        )
        _validate_fixed_control(role, controls[role])

    split = controls["sealed_split"]
    original = controls["original_preregistration"]
    if original.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN":
        raise ValueError("Candidate5 original preregistration mismatch")
    train_ids, scan_by_case, train_commitment = _validate_split_allowlist(
        split, controls["train_allowlist"]
    )
    label_attestation = controls["custody_label_attestation"]
    projection_binding = _validate_label_attestation(
        label_attestation, preregistration=context.preregistration
    )
    _validate_cache_attestation(controls["custody_cache_attestation"])
    custody_pass = controls["custody_staged_validator_pass"]
    if custody_pass.get("status") != "PASS":
        raise ValueError("Candidate5 custody independent validation is not PASS")
    manifest = controls["custody_cache_manifest"]
    records, cache_binding_sha = _validate_cache_manifest(
        manifest,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=train_commitment,
    )

    runtime = _load_verified_runtime(context)
    global _RUNTIME, _EVALUATOR
    _RUNTIME = runtime
    source_cases: list[SourceCase] = []
    for case_id in train_ids:
        loaded: dict[str, tuple[bytes, tuple[dict[str, Any], ...], str, str]] = {}
        for pass_name in ("primary", "replay"):
            record = records[(case_id, pass_name)]
            path = record["semantic_path"]
            size = os.lstat(path).st_size
            purpose = f"TRAIN_{pass_name.upper()}_V51_SEMANTIC"
            reader.authorize(
                path,
                purpose=purpose,
                sha256=record["semantic_sha256"],
                size_bytes=size,
                case_id=case_id,
            )
            data = reader.read_bytes(path, purpose=purpose)
            _semantic, predictions = runtime._validate_semantic_bytes(
                data, expected_sha256=record["semantic_sha256"]
            )
            loaded[pass_name] = (
                data,
                tuple(dict(row) for row in predictions),
                record["semantic_sha256"],
                record["runtime_case_id"],
            )
        if loaded["primary"][0] != loaded["replay"][0]:
            raise ValueError("Candidate5 primary/replay semantic bytes differ")
        source_cases.append(
            SourceCase(
                case_id=case_id,
                runtime_case_id=loaded["primary"][3],
                scan_sha256=scan_by_case[case_id],
                primary_semantic_bytes=loaded["primary"][0],
                primary_semantic_sha256=loaded["primary"][2],
                primary_predictions=loaded["primary"][1],
                replay_semantic_bytes=loaded["replay"][0],
                replay_semantic_sha256=loaded["replay"][2],
                replay_predictions=loaded["replay"][1],
            )
        )

    # Import verified evaluator/helper before the projection's final read.  No
    # artifact path is read after projection bytes enter memory.
    _EVALUATOR = _load_verified_evaluator(context)
    _load_verified_publication_helper(context)

    projection_relative = Path(str(projection_binding["path"]))
    if projection_relative.is_absolute() or ".." in projection_relative.parts:
        raise ValueError("Candidate5 attested projection path is not repository-relative")
    projection_path = _lexical_repo_path(
        ROOT / projection_relative, role="TRAIN_LABEL_PROJECTION"
    )
    reader.authorize_projection_last(
        projection_path,
        sha256=projection_binding["sha256"],
        size_bytes=projection_binding["size_bytes"],
        all_nonprojection_reads_complete=True,
    )
    bindings = {
        "label_projection_sha256": projection_binding["sha256"],
        "train_cache_manifest_sha256": FIXED_BINDINGS["custody_cache_manifest"][1],
        "train_source_cache_binding_sha256": cache_binding_sha,
    }
    candidate3_terminal_bytes = canonical_bytes(controls["candidate3_terminal"])
    candidate4_terminal_bytes = canonical_bytes(controls["candidate4_terminal"])
    post_projection_state = PostProjectionState(
        reader=reader,
        input_bindings=bindings,
        candidate3_terminal_bytes=candidate3_terminal_bytes,
        candidate4_terminal_bytes=candidate4_terminal_bytes,
    )
    cases = _read_projection_and_prepare_cases(
        reader,
        post_projection_state,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        source_cases=source_cases,
    )
    return (
        cases,
        reader,
        bindings,
        candidate3_terminal_bytes,
        candidate4_terminal_bytes,
    )


def _dependencies() -> tuple[types.ModuleType, types.ModuleType]:
    if _RUNTIME is None or _EVALUATOR is None:
        raise RuntimeError("Candidate5 verified runtime/evaluator dependencies are not configured")
    return _RUNTIME, _EVALUATOR


def predictions_after_rule(
    case: PreparedCase,
    rule: Mapping[str, Any] | None,
    *,
    replay: bool = False,
) -> list[dict[str, Any]]:
    runtime, _evaluator = _dependencies()
    rows = case.replay_predictions if replay else case.primary_predictions
    if rule is None:
        return [dict(row) for row in rows]
    runtime.validate_rule(rule)
    result: list[dict[str, Any]] = []
    for row in rows:
        features = runtime.extract_prediction_features(row)
        if not runtime._matches(rule, features):
            result.append(dict(row))
    return result


def _score_case(
    case: PreparedCase,
    rule: Mapping[str, Any] | None,
    *,
    replay: bool = False,
) -> dict[str, Any]:
    _runtime, evaluator = _dependencies()
    thresholds = evaluator.Thresholds(
        MATCH_THRESHOLDS["angle_deg"],
        MATCH_THRESHOLDS["transverse_offset_mm"],
        MATCH_THRESHOLDS["gold_span_overlap"],
        MATCH_THRESHOLDS["prediction_span_overlap"],
    )
    result = evaluator.match_lines(
        [dict(row) for row in case.gold],
        predictions_after_rule(case, rule, replay=replay),
        thresholds,
    )
    support_by_gold = {
        str(row["bend_id"]): str(int(row["reviewer_support_count"]))
        for row in case.gold
    }
    support_tier_tp = {tier: 0 for tier in SUPPORT_TIERS}
    for match in result["matches"]:
        tier = support_by_gold[str(match["gold_id"])]
        if tier not in support_tier_tp:
            raise ValueError("Candidate5 evaluator returned an unsupported label tier")
        support_tier_tp[tier] += 1
    return {
        "case_id": case.case_id,
        "exact": bool(result["exact"]),
        "false_negative": int(result["false_negative"]),
        "false_positive": int(result["false_positive"]),
        "prediction_count": int(result["prediction_count"]),
        "support_tier_tp": support_tier_tp,
        "true_positive": int(result["true_positive"]),
    }


def aggregate_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    tp = sum(int(row["true_positive"]) for row in values)
    fp = sum(int(row["false_positive"]) for row in values)
    fn = sum(int(row["false_negative"]) for row in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "case_count": len(values),
        "exact_case_count": sum(bool(row["exact"]) for row in values),
        "f1": round(f1, 12),
        "false_negative": fn,
        "false_positive": fp,
        "precision": round(precision, 12),
        "prediction_count": sum(int(row["prediction_count"]) for row in values),
        "recall": round(recall, 12),
        "support_tier_tp": {
            tier: sum(int(row["support_tier_tp"][tier]) for row in values)
            for tier in SUPPORT_TIERS
        },
        "true_positive": tp,
    }


def metric_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exact_case_count": int(candidate["exact_case_count"]) - int(reference["exact_case_count"]),
        "f1": round(float(candidate["f1"]) - float(reference["f1"]), 12),
        "false_negative": int(candidate["false_negative"]) - int(reference["false_negative"]),
        "false_positive": int(candidate["false_positive"]) - int(reference["false_positive"]),
        "prediction_count": int(candidate["prediction_count"]) - int(reference["prediction_count"]),
        "support_tier_tp": {
            tier: int(candidate["support_tier_tp"][tier]) - int(reference["support_tier_tp"][tier])
            for tier in SUPPORT_TIERS
        },
        "true_positive": int(candidate["true_positive"]) - int(reference["true_positive"]),
    }


def _per_case_no_harm(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if len(reference_rows) != len(candidate_rows):
        return False
    for reference, candidate in zip(reference_rows, candidate_rows):
        if reference["case_id"] != candidate["case_id"]:
            return False
        if int(candidate["true_positive"]) < int(reference["true_positive"]):
            return False
        if int(candidate["false_negative"]) > int(reference["false_negative"]):
            return False
        if bool(candidate["exact"]) < bool(reference["exact"]):
            return False
        if any(
            int(candidate["support_tier_tp"][tier])
            < int(reference["support_tier_tp"][tier])
            for tier in SUPPORT_TIERS
        ):
            return False
    return True


def _aggregate_gate(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> bool:
    return bool(
        int(delta["true_positive"]) >= 0
        and int(delta["false_negative"]) <= 0
        and int(delta["false_positive"]) <= -1
        and float(delta["f1"]) > 0.0
        and int(delta["exact_case_count"]) >= 0
        and min(int(value) for value in delta["support_tier_tp"].values()) >= 0
        and int(candidate["case_count"]) == int(reference["case_count"])
    )


def evaluate_fixed_rule(
    cases: Sequence[PreparedCase], rule: Mapping[str, Any]
) -> dict[str, Any]:
    reference_rows = [_score_case(case, None) for case in cases]
    candidate_rows = [_score_case(case, rule) for case in cases]
    reference = aggregate_results(reference_rows)
    candidate = aggregate_results(candidate_rows)
    delta = metric_delta(candidate, reference)
    return {
        "candidate": candidate,
        "candidate_rows": candidate_rows,
        "delta": delta,
        "eligible": bool(
            _per_case_no_harm(reference_rows, candidate_rows)
            and _aggregate_gate(reference, candidate, delta)
        ),
        "reference": reference,
        "reference_rows": reference_rows,
        "rule": dict(rule),
    }


def select_rule_by_fixed_inner_folds(
    cases: Sequence[PreparedCase],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not cases:
        raise ValueError("Candidate5 rule selection requires complete cases")
    trials = [evaluate_fixed_rule(cases, rule) for rule in fixed_rule_space()]
    eligible = [trial for trial in trials if trial["eligible"]]
    eligible.sort(
        key=lambda trial: (
            int(trial["delta"]["false_positive"]),
            -int(trial["delta"]["exact_case_count"]),
            -float(trial["delta"]["f1"]),
            canonical_bytes(trial["rule"]),
        )
    )
    selected = dict(eligible[0]["rule"]) if eligible else None
    selected_hash = sha256_bytes(canonical_bytes(selected)) if selected is not None else None
    fold_rows = [
        {
            "inner_fold_index": index,
            "inner_train_case_count": len(cases) - 1,
            "inner_train_validation_disjoint": True,
            "inner_validation_case_id": case.case_id,
            "inner_validation_complete_case_count": 1,
        }
        for index, case in enumerate(cases)
    ]
    return selected, {
        "eligible_rule_count": len(eligible),
        "fixed_rule_count": len(trials),
        "inner_fold_count": len(fold_rows),
        "inner_folds": fold_rows,
        "selected_rule": selected,
        "selected_rule_sha256": selected_hash,
        "selection_objective_and_tie_break": nested_cv_contract()[
            "selection_objective_and_tie_break"
        ],
    }


def complete_nested_oof(cases: Sequence[PreparedCase]) -> dict[str, Any]:
    if len(cases) != 18 or len({case.case_id for case in cases}) != 18:
        raise ValueError("Candidate5 nested OOF requires exactly 18 unique complete cases")
    reference_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    outer_rules: list[dict[str, Any] | None] = []
    for outer_index, validation_case in enumerate(cases):
        outer_train = [case for index, case in enumerate(cases) if index != outer_index]
        if validation_case in outer_train or len(outer_train) != 17:
            raise RuntimeError("Candidate5 outer fold overlap")
        rule, inner_audit = select_rule_by_fixed_inner_folds(outer_train)
        reference_row = _score_case(validation_case, None)
        candidate_row = _score_case(validation_case, rule)
        reference_rows.append(reference_row)
        candidate_rows.append(candidate_row)
        outer_rules.append(rule)
        folds.append(
            {
                "inner_selection": inner_audit,
                "outer_fold_index": outer_index,
                "outer_rule": rule,
                "outer_rule_sha256": (
                    sha256_bytes(canonical_bytes(rule)) if rule is not None else None
                ),
                "outer_train_case_count": 17,
                "outer_train_validation_disjoint": True,
                "outer_validation_case_count": 1,
                "outer_validation_case_id": validation_case.case_id,
            }
        )
    final_rule, final_selection = select_rule_by_fixed_inner_folds(cases)
    reference = aggregate_results(reference_rows)
    candidate = aggregate_results(candidate_rows)
    delta = metric_delta(candidate, reference)
    stable_rule = bool(
        final_rule is not None
        and all(rule is not None and canonical_bytes(rule) == canonical_bytes(final_rule) for rule in outer_rules)
    )
    outer_rule_hashes = [
        sha256_bytes(canonical_bytes(rule)) if rule is not None else None
        for rule in outer_rules
    ]
    frequencies: dict[str, int] = {}
    for rule_hash in outer_rule_hashes:
        key = rule_hash or "V51_FALLBACK"
        frequencies[key] = frequencies.get(key, 0) + 1
    checks = {
        "complete_case_outer_coverage": (
            len(folds) == 18
            and len({fold["outer_validation_case_id"] for fold in folds}) == 18
            and all(fold["outer_validation_case_count"] == 1 for fold in folds)
        ),
        "every_label_support_tier_tp_nondecrease": min(delta["support_tier_tp"].values()) >= 0,
        "exact_case_count_nondecrease": delta["exact_case_count"] >= 0,
        "false_positive_decrease": delta["false_positive"] <= -1,
        "final_rule_selected": final_rule is not None,
        "micro_f1_increase": delta["f1"] > 0.0,
        "outer_all_18_cases_scored_exactly_once": len(reference_rows) == len(candidate_rows) == 18,
        "outer_train_validation_fold_disjoint": all(fold["outer_train_validation_disjoint"] for fold in folds),
        "per_case_no_harm": _per_case_no_harm(reference_rows, candidate_rows),
        "true_positive_nondecrease": delta["true_positive"] >= 0 and delta["false_negative"] <= 0,
    }
    checks["all_conjunctive_checks_passed"] = bool(all(checks.values()))
    return {
        "candidate": candidate,
        "candidate_rows": candidate_rows,
        "checks": checks,
        "delta": delta,
        "final_rule": final_rule,
        "final_selection": final_selection,
        "folds": folds,
        "oof_predictions_are_not_final_model_predictions": True,
        "outer_rule_diagnostics": {
            "all_outer_rules_equal_final_rule": stable_rule,
            "most_common_outer_rule_fraction": round(max(frequencies.values()) / 18.0, 12),
            "outer_rule_hash_sequence_sha256": sha256_bytes(canonical_bytes(outer_rule_hashes)),
            "unique_outer_rule_or_fallback_count": len(frequencies),
        },
        "reference": reference,
        "reference_rows": reference_rows,
    }


def _runtime_model_bindings(
    context: BootstrapContext,
    *,
    input_bindings: Mapping[str, str],
    fit_audit_core_sha256: str,
    train_read_ledger_sha256: str,
) -> dict[str, str]:
    source = context.source_bindings
    return {
        "candidate1_terminal_sha256": FIXED_BINDINGS["candidate1_terminal"][1],
        "candidate2_terminal_sha256": FIXED_BINDINGS["candidate2_terminal"][1],
        "candidate3_terminal_sha256": FIXED_BINDINGS["candidate3_terminal"][1],
        "candidate4_terminal_sha256": FIXED_BINDINGS["candidate4_terminal"][1],
        "candidate5_optimizer_source_sha256": source["optimizer"]["sha256"],
        "candidate5_preflight_lock_sha256": FIXED_BINDINGS["candidate5_preflight_lock"][1],
        "candidate5_preflight_read_ledger_sha256": FIXED_BINDINGS["candidate5_preflight_read_ledger"][1],
        "candidate5_preflight_report_sha256": FIXED_BINDINGS["candidate5_preflight_report"][1],
        "candidate5_preflight_source_sha256": source["publication_helper"]["sha256"],
        "candidate5_preflight_tests_sha256": source["semantic_cache_preflight_tests"]["sha256"],
        "candidate5_preregistration_sha256": context.preregistration_sha256,
        "candidate5_runtime_source_sha256": source["runtime"]["sha256"],
        "candidate5_runtime_tests_sha256": source["runtime_tests"]["sha256"],
        "custody_cache_attestation_sha256": FIXED_BINDINGS["custody_cache_attestation"][1],
        "custody_label_attestation_sha256": FIXED_BINDINGS["custody_label_attestation"][1],
        "custody_staged_validator_pass_sha256": FIXED_BINDINGS["custody_staged_validator_pass"][1],
        "fit_audit_core_sha256": _require_sha(fit_audit_core_sha256, name="fit audit SHA"),
        "original_preregistration_sha256": FIXED_BINDINGS["original_preregistration"][1],
        "split_sha256": FIXED_BINDINGS["sealed_split"][1],
        "train_allowlist_sha256": FIXED_BINDINGS["train_allowlist"][1],
        "train_cache_manifest_sha256": input_bindings["train_cache_manifest_sha256"],
        "train_label_projection_sha256": input_bindings["label_projection_sha256"],
        "train_read_ledger_sha256": _require_sha(train_read_ledger_sha256, name="read ledger SHA"),
        "train_source_cache_binding_sha256": input_bindings["train_source_cache_binding_sha256"],
    }


def _read_ledger_payload(
    reader: GuardedReader,
    *,
    fit_authorization_sha256: str,
) -> dict[str, Any]:
    purposes = [str(row["purpose"]) for row in reader.entries]
    return {
        "assertions": {
            "all_artifact_reads_precede_projection_except_projection_itself": (
                bool(purposes) and purposes[-1] == "TRAIN_LABEL_PROJECTION_LAST_READ"
            ),
            "all_input_file_identities_unique": len(reader.identities) == len(set(reader.identities)),
            "full_28_catalog_or_manifest_opened": False,
            "holdout_path_or_score_opened": False,
            "label_projection_read_count": reader.projection_read_count,
            "pool_artifacts_opened": False,
            "projection_derived_from_exact_attestation": True,
            "projection_read_once_and_last": reader.projection_read_count == 1,
            "semantic_internal_scan_identity_used": False,
        },
        "attempt_ordinal": 5,
        "denied_attempts": list(reader.denied_attempts),
        "denied_read_count": len(reader.denied_attempts),
        "fit_authorization_sha256": fit_authorization_sha256,
        "holdout_score_opened": False,
        "kind": READ_LEDGER_KIND,
        "read_count": len(reader.entries),
        "reads": list(reader.entries),
        "schema_version": 1,
        "status": "COMPLETE_PROJECTION_LAST",
    }


def _final_rule_resubstitution(
    cases: Sequence[PreparedCase], rule: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = evaluate_fixed_rule(cases, rule)
    return {
        "candidate": evaluation["candidate"],
        "delta": evaluation["delta"],
        "eligible": evaluation["eligible"],
        "reference": evaluation["reference"],
    }


def _model_payload(
    runtime: types.ModuleType,
    *,
    rule: Mapping[str, Any],
    bindings: Mapping[str, str],
    pass_sha256: str,
) -> dict[str, Any]:
    feature = str(rule["feature"])
    false_fields = {
        "audit_metadata_feature_used": False,
        "bend_id_feature_used": False,
        "cad_feature_used": False,
        "candidate_count_feature_used": False,
        "case_identity_feature_used": False,
        "drawing_feature_used": False,
        "expected_count_feature_used": False,
        "filename_feature_used": False,
        "gold_or_truth_feature_used": False,
        "label_feature_used": False,
        "observed_source_or_prediction_multiplicity_used": False,
        "part_identity_feature_used": False,
        "runtime_identity_feature_used": False,
        "semantic_internal_scan_identity_feature_used": False,
        "sender_count_feature_used": False,
        "source_hypothesis_id_feature_used": False,
    }
    model = {
        "attempt_ordinal": 5,
        **false_fields,
        "bindings": dict(bindings),
        "deletion_only": True,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "inference_features": [feature],
        "kind": runtime.MODEL_KIND,
        "line_length_feature_used": feature == "line_length_mm",
        "post_holdout_model_or_threshold_mutation": "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT",
        "pre_holdout_execution_anchor_created": False,
        "profile": runtime.PROFILE,
        "rule": dict(rule),
        "schema_version": 1,
        "selection_route_feature_used": feature == "selection_route",
        "selector_score_feature_used": feature == "selector_score",
        "source_kind_feature_used": feature == "source_kind",
        "source_only": True,
        "train_gate_pass_binding": {"kind": runtime.PASS_KIND, "sha256": pass_sha256},
        "training_gate_passed": True,
        "v51_fallback_default": True,
    }
    if set(model) != runtime.MODEL_KEYS:
        raise RuntimeError("Candidate5 optimizer model schema differs from frozen runtime")
    return model


def _terminal_payload(
    context: BootstrapContext,
    *,
    status: str,
    reason_class: str,
    fit_authorization_sha256: str,
    input_bindings: Mapping[str, str],
    gate_checks: Mapping[str, bool] | None,
    metric_computation_started: bool,
    metric_computed: bool,
    model_materialized_in_memory: bool,
) -> dict[str, Any]:
    source_commitment = sha256_bytes(canonical_bytes(context.source_bindings))
    return {
        "artifact_bindings": {
            "candidate1_terminal_sha256": FIXED_BINDINGS["candidate1_terminal"][1],
            "candidate2_terminal_sha256": FIXED_BINDINGS["candidate2_terminal"][1],
            "candidate3_custody_incident_sha256": FIXED_BINDINGS["candidate3_custody_incident"][1],
            "candidate3_terminal_sha256": FIXED_BINDINGS["candidate3_terminal"][1],
            "candidate4_preflight_failure_sha256": FIXED_BINDINGS["candidate4_preflight_failure"][1],
            "candidate4_terminal_sha256": FIXED_BINDINGS["candidate4_terminal"][1],
            "candidate5_preflight_lock_sha256": FIXED_BINDINGS["candidate5_preflight_lock"][1],
            "candidate5_preflight_read_ledger_sha256": FIXED_BINDINGS["candidate5_preflight_read_ledger"][1],
            "candidate5_preflight_report_sha256": FIXED_BINDINGS["candidate5_preflight_report"][1],
            "candidate5_preregistration_sha256": context.preregistration_sha256,
            "custody_cache_attestation_sha256": FIXED_BINDINGS["custody_cache_attestation"][1],
            "custody_label_attestation_sha256": FIXED_BINDINGS["custody_label_attestation"][1],
            "custody_staged_validator_pass_sha256": FIXED_BINDINGS["custody_staged_validator_pass"][1],
            "fit_authorization_sha256": fit_authorization_sha256,
            "label_projection_sha256": input_bindings["label_projection_sha256"],
            "sealed_split_sha256": FIXED_BINDINGS["sealed_split"][1],
            "source_binding_commitment_sha256": source_commitment,
            "train_allowlist_sha256": FIXED_BINDINGS["train_allowlist"][1],
        },
        "attempt_ordinal": 5,
        "candidate5_continuation_authorized": False,
        "candidate5_fit_started": True,
        "candidate5_label_projection_opened": True,
        "candidate5_label_projection_read_count": 1,
        "candidate5_metric_computation_started": metric_computation_started,
        "candidate5_metric_computed": metric_computed,
        "candidate5_model_materialized_in_memory": model_materialized_in_memory,
        "candidate5_model_published": False,
        "candidate5_pass_artifact_published": False,
        "candidate5_rule_published": False,
        "candidate_id": 5,
        "gate_checks_value_free": dict(gate_checks or {}),
        "holdout_access_count": 0,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "kind": TERMINAL_KIND,
        "output_inventory": ["terminal_negative.json"],
        "post_terminal_candidate5_model_or_threshold_mutation": "FORBIDDEN",
        "promotion_eligible": False,
        "prospective_or_unseen_claim_authorized": False,
        "schema_version": 1,
        "status": status,
        "terminal_reason_class": reason_class,
    }


class ReplayDeploymentFailure(RuntimeError):
    pass


def build_fit_publication_payloads(
    cases: Sequence[PreparedCase],
    reader: GuardedReader,
    context: BootstrapContext,
    *,
    input_bindings: Mapping[str, str],
    fit_authorization_sha256: str,
    candidate3_terminal_bytes: bytes,
    candidate4_terminal_bytes: bytes,
    replay_result_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    runtime, _evaluator = _dependencies()
    nested = complete_nested_oof(cases)
    final_rule = nested["final_rule"]
    final_resub = (
        _final_rule_resubstitution(cases, final_rule) if final_rule is not None else None
    )
    initial_gate = bool(
        nested["checks"]["all_conjunctive_checks_passed"]
        and final_rule is not None
        and final_resub is not None
        and final_resub["eligible"] is True
    )
    value_free_gate = {
        "complete_nested_oof_passed": bool(nested["checks"]["all_conjunctive_checks_passed"]),
        "final_rule_resubstitution_passed": bool(final_resub and final_resub["eligible"]),
        "final_rule_selected": final_rule is not None,
        "oof_predictions_are_not_final_model_predictions": True,
    }
    if not initial_gate:
        terminal = _terminal_payload(
            context,
            status="REJECTED_TRAIN_GATE",
            reason_class="NESTED_OOF_OR_FINAL_RULE_CONSERVATIVE_GATE_FAILED",
            fit_authorization_sha256=fit_authorization_sha256,
            input_bindings=input_bindings,
            gate_checks=value_free_gate,
            metric_computation_started=True,
            metric_computed=True,
            model_materialized_in_memory=False,
        )
        return {"terminal_negative.json": canonical_bytes(terminal)}, {
            "status": terminal["status"],
            "terminal_only": True,
        }

    ledger = _read_ledger_payload(
        reader, fit_authorization_sha256=fit_authorization_sha256
    )
    ledger_bytes = canonical_bytes(ledger)
    ledger_sha = sha256_bytes(ledger_bytes)
    fit_audit_core = {
        "adaptive_search_disclosure": search_contract(),
        "attempt_ordinal": 5,
        "case_grouping": "COMPLETE_CASE_ONLY",
        "final_rule_resubstitution": final_resub,
        "holdout_score_opened": False,
        "kind": FIT_AUDIT_KIND,
        "match_thresholds": dict(MATCH_THRESHOLDS),
        "nested_oof_adaptive_procedure": nested,
        "oof_predictions_are_not_final_model_predictions": True,
        "schema_version": 1,
        "selected_final_rule": dict(final_rule),
        "source_only": True,
        "status": "TRAIN_GATE_PASS_CORE_EVALUATION_BOUND_TO_DEPLOYED_REPLAY",
    }
    fit_audit_bytes = canonical_bytes(fit_audit_core)
    fit_audit_sha = sha256_bytes(fit_audit_bytes)
    runtime_bindings = _runtime_model_bindings(
        context,
        input_bindings=input_bindings,
        fit_audit_core_sha256=fit_audit_sha,
        train_read_ledger_sha256=ledger_sha,
    )
    pass_checks = {
        "cache_preflight_passed": True,
        "deployed_model_exact_bytes_match_evaluated_predictions": True,
        "every_label_support_tier_tp_nondecrease": True,
        "exact_case_count_nondecrease": True,
        "false_positive_decrease": True,
        "fixed_holdout_stage_order_similar4_adjacent3_distant3": True,
        "fixed_preregistered_search_space": True,
        "holdout_access_zero": True,
        "nested_complete_case_group_validation": True,
        "outer_all_18_cases_scored_exactly_once": True,
        "outer_train_validation_fold_disjoint": True,
        "overall_micro_f1_increase": True,
        "primary_replay_bytes_exact": True,
        "replay_execution_succeeded_before_publication": True,
        "train18_allowlist_exact": True,
        "train_read_ledger_exact": True,
        "true_positive_nondecrease": True,
    }
    if set(pass_checks) != runtime.GATE_CHECK_KEYS:
        raise RuntimeError("Candidate5 optimizer PASS checks differ from frozen runtime")
    pass_artifact = {
        "attempt_ordinal": 5,
        "bindings": runtime_bindings,
        "checks": pass_checks,
        "gate_result_value_free": True,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "kind": runtime.PASS_KIND,
        "label_derived_metric_values_present": False,
        "pre_holdout_execution_anchor_created": False,
        "schema_version": 1,
        "selected_rule": dict(final_rule),
        "status": "PASS",
        "training_gate_passed": True,
    }
    pass_bytes = canonical_bytes(pass_artifact)
    pass_sha = sha256_bytes(pass_bytes)
    model = _model_payload(
        runtime,
        rule=final_rule,
        bindings=runtime_bindings,
        pass_sha256=pass_sha,
    )
    model_bytes = canonical_bytes(model)
    model_sha = sha256_bytes(model_bytes)

    deployed_case_rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            expected_primary = predictions_after_rule(case, final_rule, replay=False)
            expected_replay = predictions_after_rule(case, final_rule, replay=True)
            primary_result = runtime.apply_candidate5(
                case.primary_semantic_bytes,
                model,
                pass_bytes,
                candidate3_terminal_bytes,
                candidate4_terminal_bytes,
                v51_semantic_sha256=case.primary_semantic_sha256,
                train_gate_pass_sha256=pass_sha,
                candidate3_terminal_sha256=FIXED_BINDINGS["candidate3_terminal"][1],
                candidate4_terminal_sha256=FIXED_BINDINGS["candidate4_terminal"][1],
                verified_runtime_source_sha256=context.source_bindings["runtime"]["sha256"],
            )
            replay_result = runtime.apply_candidate5(
                case.replay_semantic_bytes,
                model,
                pass_bytes,
                candidate3_terminal_bytes,
                candidate4_terminal_bytes,
                v51_semantic_sha256=case.replay_semantic_sha256,
                train_gate_pass_sha256=pass_sha,
                candidate3_terminal_sha256=FIXED_BINDINGS["candidate3_terminal"][1],
                candidate4_terminal_sha256=FIXED_BINDINGS["candidate4_terminal"][1],
                verified_runtime_source_sha256=context.source_bindings["runtime"]["sha256"],
            )
            if replay_result_mutator is not None:
                replay_result = replay_result_mutator(dict(replay_result))
            if canonical_bytes(primary_result["predictions"]) != canonical_bytes(expected_primary):
                raise ReplayDeploymentFailure("deployed primary differs from final-rule evaluation")
            if canonical_bytes(replay_result["predictions"]) != canonical_bytes(expected_replay):
                raise ReplayDeploymentFailure("deployed replay differs from final-rule evaluation")
            if canonical_bytes(primary_result) != canonical_bytes(replay_result):
                raise ReplayDeploymentFailure("deployed primary/replay runtime results differ")
            deployed_case_rows.append(
                {
                    "case_id": case.case_id,
                    "runtime_result_sha256": sha256_bytes(canonical_bytes(primary_result)),
                }
            )
    except Exception as error:
        terminal = _terminal_payload(
            context,
            status="REJECTED_TRAIN_REPLAY_DEPLOYMENT_MISMATCH",
            reason_class=type(error).__name__,
            fit_authorization_sha256=fit_authorization_sha256,
            input_bindings=input_bindings,
            gate_checks=value_free_gate,
            metric_computation_started=True,
            metric_computed=True,
            model_materialized_in_memory=True,
        )
        return {"terminal_negative.json": canonical_bytes(terminal)}, {
            "status": terminal["status"],
            "terminal_only": True,
        }

    deployed_digest = sha256_bytes(canonical_bytes(deployed_case_rows))
    freeze_lock = {
        "attempt_ordinal": 5,
        "bindings": {
            "evaluator_source_sha256": context.source_bindings["evaluator"]["sha256"],
            "fit_audit_core_sha256": fit_audit_sha,
            "frozen_model_sha256": model_sha,
            "optimizer_source_sha256": context.source_bindings["optimizer"]["sha256"],
            "optimizer_tests_sha256": context.source_bindings["optimizer_tests"]["sha256"],
            "train_gate_pass_sha256": pass_sha,
            "train_read_ledger_sha256": ledger_sha,
        },
        "deployed_final_rule_train_replay": {
            "all_18_primary_replay_exact": True,
            "deployed_model_exact_bytes_match_final_rule_train_evaluation": True,
            "result_digest_sha256": deployed_digest,
        },
        "external_anchor_present": False,
        "external_anchor_required_before_any_holdout_score": True,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "holdout_stage_order_frozen": [dict(row) for row in HOLDOUT_SEQUENCE],
        "kind": FREEZE_KIND,
        "no_retune_after_any_holdout_score": True,
        "oof_predictions_are_not_final_model_predictions": True,
        "post_first_holdout_model_or_threshold_mutation": "FORBIDDEN",
        "schema_version": 1,
        "status": "FROZEN_AWAITING_INDEPENDENT_EXTERNAL_ANCHOR",
    }
    freeze_bytes = canonical_bytes(freeze_lock)
    freeze_sha = sha256_bytes(freeze_bytes)
    preseal = {
        "attempt_ordinal": 5,
        "external_anchor_must_be_created_by_independent_auditor": True,
        "external_anchor_present": False,
        "external_anchor_required_before_any_holdout_score": True,
        "fit_audit_core_sha256": fit_audit_sha,
        "freeze_lock_sha256": freeze_sha,
        "frozen_model_sha256": model_sha,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "kind": PRESEAL_KIND,
        "preregistration_sha256": context.preregistration_sha256,
        "schema_version": 1,
        "status": "LOCAL_SEAL_ONLY_NOT_EXTERNAL_ANCHOR",
        "train_gate_pass_sha256": pass_sha,
        "train_read_ledger_sha256": ledger_sha,
    }
    payloads = {
        "fit_audit_core.json": fit_audit_bytes,
        "freeze_lock.json": freeze_bytes,
        "frozen_model.json": model_bytes,
        "pre_holdout_local_seal.json": canonical_bytes(preseal),
        "train_gate_pass.json": pass_bytes,
        "train_read_ledger.json": ledger_bytes,
    }
    if set(payloads) != PASS_FILENAMES:
        raise RuntimeError("Candidate5 pass publication inventory mismatch")
    return payloads, {
        "fit_audit_core_sha256": fit_audit_sha,
        "freeze_lock_sha256": freeze_sha,
        "frozen_model_sha256": model_sha,
        "pre_holdout_local_seal_sha256": sha256_bytes(payloads["pre_holdout_local_seal.json"]),
        "status": "TRAIN_GATE_PASS_FROZEN_AWAITING_EXTERNAL_ANCHOR",
        "terminal_only": False,
        "train_gate_pass_sha256": pass_sha,
        "train_read_ledger_sha256": ledger_sha,
    }


def _renameat_noreplace(parent_fd: int, source: str, destination: str) -> None:
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 1)
    else:
        raise RuntimeError("Candidate5 optimizer requires a no-replace rename primitive")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error_number,
                "Candidate5 publication destination exists",
                destination,
            )
        raise OSError(error_number, os.strerror(error_number), destination)


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino))


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{role} must be a directory")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise RuntimeError("Candidate5 optimizer requires O_NOFOLLOW and O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory_flag)
    opened = os.fstat(descriptor)
    expected_identity = _identity(before)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected_identity:
        os.close(descriptor)
        raise RuntimeError(f"{role} identity changed during open")
    visible = _reject_symlink_chain(candidate, role=role, require_exists=True)
    after = os.lstat(visible)
    if not stat.S_ISDIR(after.st_mode) or _identity(after) != expected_identity:
        os.close(descriptor)
        raise RuntimeError(f"{role} identity changed after open")
    return descriptor, expected_identity


def _staging_prefix(output: Path) -> str:
    return f".{output.name}.staging-"


def _quarantine_prefix(output: Path) -> str:
    return f".{output.name}.quarantine-"


def _sibling_names(parent_fd: int, *, prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix)))


def _assert_no_publication_residue(parent_fd: int, output: Path) -> None:
    staging = _sibling_names(parent_fd, prefix=_staging_prefix(output))
    quarantine = _sibling_names(parent_fd, prefix=_quarantine_prefix(output))
    if staging or quarantine:
        raise FileExistsError(
            f"Candidate5 optimizer rejects staging/quarantine residue: {staging + quarantine}"
        )


def _assert_visible_directory_identity(
    path: Path, expected_identity: tuple[int, int], *, role: str
) -> None:
    visible = _reject_symlink_chain(
        path, role=role, require_exists=True
    )
    metadata = os.lstat(visible)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected_identity:
        raise RuntimeError(f"{role} identity changed")


def _assert_visible_parent_identity(
    output: Path, expected_identity: tuple[int, int]
) -> None:
    _assert_visible_directory_identity(
        output.parent,
        expected_identity,
        role="Candidate5 visible output parent",
    )


def _validate_output_target(output: Path) -> Path:
    candidate = _lexical_repo_path(output, role="Candidate5 optimizer output")
    expected = _lexical_repo_path(OUTPUT_ROOT, role="frozen Candidate5 optimizer output")
    if candidate != expected:
        raise ValueError("Candidate5 optimizer output namespace is frozen")
    parent_fd, _parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate5 output parent"
    )
    try:
        _assert_no_publication_residue(parent_fd, candidate)
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate5 optimizer output must be fresh and absent")
    finally:
        os.close(parent_fd)
    return candidate


def _create_owned_staging(
    parent_fd: int, output: Path
) -> tuple[str, tuple[int, int]]:
    _assert_no_publication_residue(parent_fd, output)
    name = f"{_staging_prefix(output)}{secrets.token_hex(16)}"
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("Candidate5 staging object is not a directory")
    return name, _identity(metadata)


def _open_owned_directory(
    parent_fd: int, name: str, expected_identity: tuple[int, int]
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise RuntimeError("Candidate5 optimizer requires O_NOFOLLOW and O_DIRECTORY")
    descriptor = os.open(
        name, os.O_RDONLY | nofollow | directory_flag, dir_fd=parent_fd
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected_identity:
        os.close(descriptor)
        raise RuntimeError("Candidate5 owned publication directory identity changed")
    return descriptor


def _write_staging_file(staging_fd: int, name: str, data: bytes) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate5 optimizer requires O_NOFOLLOW")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
        0o600,
        dir_fd=staging_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate5 staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != len(data)
        ):
            raise RuntimeError("Candidate5 staged payload identity/size mismatch")
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _verify_owned_payloads(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    payloads: Mapping[str, bytes],
) -> None:
    directory_fd = _open_owned_directory(parent_fd, name, expected_identity)
    try:
        if set(os.listdir(directory_fd)) != set(payloads):
            raise RuntimeError("Candidate5 publication inventory changed")
        identities: set[tuple[int, int]] = set()
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate5 optimizer requires O_NOFOLLOW")
        for filename, expected in sorted(payloads.items()):
            descriptor = os.open(filename, os.O_RDONLY | nofollow, dir_fd=directory_fd)
            try:
                before = os.fstat(descriptor)
                file_identity = _identity(before)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or file_identity in identities
                ):
                    raise RuntimeError("Candidate5 payload is non-regular or hard-linked")
                identities.add(file_identity)
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                observed = b"".join(chunks)
                after = os.fstat(descriptor)
                if (
                    _identity(after) != file_identity
                    or int(after.st_size) != len(observed)
                    or observed != expected
                ):
                    raise RuntimeError("Candidate5 publication payload bytes changed")
            finally:
                os.close(descriptor)
    finally:
        os.close(directory_fd)


def _quarantine_owned_directory(
    parent_fd: int,
    source_name: str,
    output: Path,
    *,
    expected_identity: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    before = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or _identity(before) != expected_identity:
        raise RuntimeError("Candidate5 refuses to quarantine an unexpected inode")
    quarantine_name = f"{_quarantine_prefix(output)}{secrets.token_hex(16)}"
    _renameat_noreplace(parent_fd, source_name, quarantine_name)
    quarantined = os.stat(quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(quarantined.st_mode)
        or _identity(quarantined) != expected_identity
    ):
        raise RuntimeError("Candidate5 quarantine captured an unexpected inode")
    try:
        os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("Candidate5 quarantine left the source name populated")
    os.fsync(parent_fd)
    return quarantine_name, _identity(quarantined)


def publish_atomically(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    after_stage_write: Callable[[Path], None] | None = None,
    after_rename: Callable[[Path], None] | None = None,
    after_parent_open: Callable[[Path], None] | None = None,
) -> None:
    names = set(payloads)
    if names not in (PASS_FILENAMES, TERMINAL_FILENAMES):
        raise ValueError("Candidate5 publication inventory is neither exact PASS nor terminal-only")
    if any(
        not name.endswith(".json")
        or Path(name).name != name
        or "/" in name
        for name in names
    ):
        raise ValueError("Candidate5 publication filenames must be flat JSON basenames")
    if any(not isinstance(data, bytes) for data in payloads.values()):
        raise ValueError("Candidate5 publication payload values must be bytes")
    output = _validate_output_target(output)
    parent_fd, parent_identity = _open_stable_directory(
        output.parent, role="Candidate5 output parent"
    )
    staging_name = ""
    staging_identity: tuple[int, int] | None = None
    rename_returned = False
    try:
        if after_parent_open is not None:
            after_parent_open(output.parent)
        if _identity(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError("Candidate5 pinned output parent identity changed")
        _assert_visible_parent_identity(output, parent_identity)
        _assert_no_publication_residue(parent_fd, output)
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate5 output raced into existence before staging")
        staging_name, staging_identity = _create_owned_staging(parent_fd, output)
        staging_fd = _open_owned_directory(parent_fd, staging_name, staging_identity)
        try:
            for name, data in sorted(payloads.items()):
                _write_staging_file(staging_fd, name, data)
            os.fsync(staging_fd)
            os.fchmod(staging_fd, 0o500)
        finally:
            os.close(staging_fd)
        if after_stage_write is not None:
            after_stage_write(output.parent / staging_name)
        _verify_owned_payloads(parent_fd, staging_name, staging_identity, payloads)
        if _identity(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError("Candidate5 pinned output parent identity changed")
        _assert_visible_parent_identity(output, parent_identity)
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate5 output raced into existence")
        _renameat_noreplace(parent_fd, staging_name, output.name)
        rename_returned = True
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate5 published directory identity mismatch")
        _assert_visible_parent_identity(output, parent_identity)
        visible_output = _reject_symlink_chain(
            output, role="Candidate5 published output", require_exists=True
        )
        if _identity(os.lstat(visible_output)) != staging_identity:
            raise RuntimeError("Candidate5 visible publication identity changed")
        _verify_owned_payloads(parent_fd, output.name, staging_identity, payloads)
        if after_rename is not None:
            after_rename(output)
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate5 published directory changed after rename")
        _assert_visible_parent_identity(output, parent_identity)
        _verify_owned_payloads(parent_fd, output.name, staging_identity, payloads)
        os.fsync(parent_fd)
        if _sibling_names(parent_fd, prefix=_staging_prefix(output)):
            raise RuntimeError("Candidate5 staging residue remains after publication")
        if _sibling_names(parent_fd, prefix=_quarantine_prefix(output)):
            raise RuntimeError("Candidate5 quarantine unexpectedly exists after publication")
    except BaseException:
        if staging_identity is not None:
            owned_name = output.name if rename_returned else staging_name
            try:
                metadata = os.stat(owned_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                metadata = None
            if metadata is not None and _identity(metadata) == staging_identity:
                _quarantine_owned_directory(
                    parent_fd,
                    owned_name,
                    output,
                    expected_identity=staging_identity,
                )
        raise
    finally:
        os.close(parent_fd)


def run(
    *,
    preregistration_sha256: str,
    fit_authorization_sha256: str,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output = _validate_output_target(output_root)
    context = bootstrap_verified_sources(
        preregistration_sha256=preregistration_sha256
    )
    try:
        (
            cases,
            reader,
            input_bindings,
            candidate3_terminal_bytes,
            candidate4_terminal_bytes,
        ) = load_train_inputs(
            context,
            fit_authorization_sha256=fit_authorization_sha256,
        )
    except PostProjectionFailure as failure:
        terminal = _terminal_payload(
            context,
            status="REJECTED_POST_PROJECTION_INPUT_FAILURE",
            reason_class=failure.reason_class,
            fit_authorization_sha256=fit_authorization_sha256,
            input_bindings=failure.state.input_bindings,
            gate_checks=None,
            metric_computation_started=False,
            metric_computed=False,
            model_materialized_in_memory=False,
        )
        payloads = {"terminal_negative.json": canonical_bytes(terminal)}
        result = {"status": terminal["status"], "terminal_only": True}
    else:
        try:
            payloads, result = build_fit_publication_payloads(
                cases,
                reader,
                context,
                input_bindings=input_bindings,
                fit_authorization_sha256=fit_authorization_sha256,
                candidate3_terminal_bytes=candidate3_terminal_bytes,
                candidate4_terminal_bytes=candidate4_terminal_bytes,
            )
        except BaseException as error:
            terminal = _terminal_payload(
                context,
                status="REJECTED_POST_LABEL_EXECUTION_FAILURE",
                reason_class=type(error).__name__,
                fit_authorization_sha256=fit_authorization_sha256,
                input_bindings=input_bindings,
                gate_checks=None,
                metric_computation_started=True,
                metric_computed=False,
                model_materialized_in_memory=False,
            )
            payloads = {"terminal_negative.json": canonical_bytes(terminal)}
            result = {"status": terminal["status"], "terminal_only": True}
    publish_atomically(output, payloads)
    result = dict(result)
    result["output_root"] = str(output.relative_to(ROOT))
    result["published_inventory"] = sorted(payloads)
    result["published_artifact_sha256"] = {
        name: sha256_bytes(data) for name, data in sorted(payloads.items())
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--fit-authorization-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-check-only", action="store_true")
    args = parser.parse_args()
    if args.source_check_only:
        context = bootstrap_verified_sources(
            preregistration_sha256=args.preregistration_sha256
        )
        print(
            json.dumps(
                {
                    "artifact_or_label_input_opened": False,
                    "fit_authorization_opened": False,
                    "preregistration_sha256": context.preregistration_sha256,
                    "source_binding_count": len(context.source_bindings),
                    "status": "PREREG_AND_SOURCE_BYTES_VERIFIED_ONLY",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run(
        preregistration_sha256=args.preregistration_sha256,
        fit_authorization_sha256=args.fit_authorization_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
