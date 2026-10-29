#!/usr/bin/env python3
"""Nested, train-only optimizer for the sealed Aug-30 18/10 split.

The process is fail closed:

* it verifies the sealed split/hash manifest first;
* it constructs the 18-case allowlist before resolving or opening labels;
* it accepts only a separately attested TRAIN-only label projection;
* its read capability contains exact files for those 18 cases only;
* complete cases, never candidate rows, are the validation groups;
* it serializes one portable source-only model and seals it before any holdout
  score is permitted to exist.

This module is the only truth-aware component.  Frozen inference is in
``backend.f1_train18_frozen_source_only_candidate_v1`` and has no filesystem or
label API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_train18_frozen_source_only_candidate_v1 import (  # noqa: E402
    MODEL_KIND,
    Observation,
    build_observations,
    canonical_bytes,
    classifier_probabilities,
    compose_frozen_candidate,
    endpoints_from_record,
    feature_matrix,
    feature_names,
    predictions_from_selection,
    select_observations,
    sha256_bytes,
    validate_source_input,
)
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    is_match,
    match_lines,
    pair_metrics,
)


HASH_MANIFEST_SHA256 = "df8d821ad3816f8fa72f7f9c34e3e0a9f8b07c88a503c9cd5b239a2a195928c8"
HASH_MANIFEST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_hashes_v1.json"
SPLIT_LOCK = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
TRAIN_ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
PREREGISTRATION = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
RUNTIME_SOURCE = ROOT / "backend/f1_train18_frozen_source_only_candidate_v1.py"
OPTIMIZER_SOURCE = Path(__file__).resolve()

DEFAULT_CACHE_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
DEFAULT_OUTPUT_DIR = ROOT / "output/research/f1-aug30-pseudo-unseen-train18-optimizer-v1"

HASH_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_hash_manifest_v1"
SPLIT_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
ALLOWLIST_KIND = "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
PREREG_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_prereg_v1"
PROJECTION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_v1"
ATTESTATION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_attestation_v1"
TRAIN_CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
TRAIN_CACHE_ATTESTATION_KIND = (
    "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
)

MATCH_THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)
SPAN_SCALES = (0.50, 0.75, 1.00, 1.50, 2.00)
FIXED_SPANS_MM = (16.0, 24.0, 40.0, 80.0, 160.0)
PROBABILITY_THRESHOLDS = (0.35, 0.55, 0.75, 0.90, 0.97)
OUTPUT_CAPS = (2, 4, 6)
OUTER_FOLDS = 6
INNER_FOLDS = 3
FINAL_SELECTION_FOLDS = 6
FOLD_SEED = "f1-aug30-train18-nested-case-groups-v1"

DUPLICATE_PROFILE = {
    "duplicate_angle_deg": 5.0,
    "duplicate_offset_mm": 4.0,
    "duplicate_span_overlap": 0.60,
    "span_scales": list(SPAN_SCALES),
    "fixed_spans_mm": list(FIXED_SPANS_MM),
}

FORBIDDEN_LABEL_PROJECTION_KEYS = {
    "filename",
    "private_filename",
    "scan_name",
    "scan_path",
    "part_id",
    "cad",
    "drawing",
    "expected_count",
    "sender_count",
}


@dataclass(frozen=True)
class LockContext:
    split_sha256: str
    allowlist_sha256: str
    preregistration_sha256: str
    train_case_ids: tuple[str, ...]
    scan_sha_by_case: Mapping[str, str]
    preregistration: Mapping[str, Any]
    sealed_input_manifest_sha256: str
    train_commitment_sha256: str
    bootstrap_entries: tuple[dict[str, Any], ...]


@dataclass
class GuardedReader:
    """Exact-path read capability with an append-only in-memory audit ledger."""

    allowed: Mapping[Path, str]
    expected_sha256: dict[Path, str] = field(default_factory=dict)
    expected_size_bytes: dict[Path, int] = field(default_factory=dict)
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)

    def read_bytes(self, path: Path, *, purpose: str, case_id: str | None = None) -> bytes:
        resolved = path.resolve()
        declared = self.allowed.get(resolved)
        if declared is None:
            self.denied_attempts.append(
                {"path": str(resolved), "purpose": purpose, "case_id": case_id}
            )
            raise PermissionError(f"read denied by TRAIN18 exact allowlist: {resolved}")
        if purpose != declared:
            raise PermissionError(
                f"read purpose mismatch for {resolved}: expected {declared}, got {purpose}"
            )
        data = resolved.read_bytes()
        actual_sha = sha256_bytes(data)
        expected_sha = self.expected_sha256.get(resolved)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"authorized read hash mismatch: {resolved}")
        expected_size = self.expected_size_bytes.get(resolved)
        if expected_size is not None and len(data) != expected_size:
            raise ValueError(f"authorized read size mismatch: {resolved}")
        self.entries.append(
            {
                "ordinal": len(self.entries) + 1,
                "path": str(resolved),
                "purpose": purpose,
                "case_id": case_id,
                "size_bytes": len(data),
                "sha256": actual_sha,
                "read_succeeded": True,
            }
        )
        return data

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        expected_sha256: str,
        expected_size_bytes: int | None = None,
    ) -> bool:
        resolved = path.resolve()
        if resolved in self.allowed:
            same_purpose = self.allowed[resolved] == purpose
            same_sha = self.expected_sha256.get(resolved) == expected_sha256
            same_size = self.expected_size_bytes.get(resolved) == expected_size_bytes
            if not (same_purpose and same_sha and same_size):
                raise ValueError(f"conflicting authorization for shared path: {resolved}")
            return False
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("authorized read requires a lowercase SHA-256")
        # ``allowed`` is constructed as a dict by this module even though the
        # dataclass annotation exposes only Mapping to callers.
        assert isinstance(self.allowed, dict)
        self.allowed[resolved] = purpose
        self.expected_sha256[resolved] = expected_sha256
        if expected_size_bytes is not None:
            if (
                isinstance(expected_size_bytes, bool)
                or not isinstance(expected_size_bytes, int)
                or expected_size_bytes <= 0
            ):
                raise ValueError("authorized read size must be a positive integer")
            self.expected_size_bytes[resolved] = expected_size_bytes
        return True

    def read_json(self, path: Path, *, purpose: str, case_id: str | None = None) -> dict[str, Any]:
        payload = json.loads(self.read_bytes(path, purpose=purpose, case_id=case_id))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: JSON payload must be an object")
        return payload


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    universe: str
    representation: str
    classifier_kind: str
    classifier_parameters: Mapping[str, Any]


@dataclass
class PreparedView:
    observations: list[Observation]
    raw_reference: list[Observation]
    matrix: np.ndarray
    target: np.ndarray


@dataclass
class CaseData:
    case_id: str
    runtime_case_id: str
    scan_sha256: str
    bends: list[dict[str, Any]]
    primary_v51: dict[str, Any]
    primary_pool: dict[str, Any]
    replay_v51: dict[str, Any]
    replay_pool: dict[str, Any]
    views: dict[tuple[str, str], PreparedView] = field(default_factory=dict)


LEARNER_SPECS = (
    LearnerSpec("INCUMBENT_ABSOLUTE_LOGISTIC", "incumbent_only", "absolute_quality", "logistic", {"c": 0.5}),
    LearnerSpec("INCUMBENT_ABSOLUTE_TREE", "incumbent_only", "absolute_quality", "tree", {"max_depth": 5, "min_samples_leaf": 2}),
    LearnerSpec("INCUMBENT_ABSOLUTE_FOREST", "incumbent_only", "absolute_quality", "forest", {"max_depth": 7, "min_samples_leaf": 2}),
    LearnerSpec("RAW_ABSOLUTE_LOGISTIC", "raw_reselect", "absolute_quality", "logistic", {"c": 0.5}),
    LearnerSpec("RAW_ABSOLUTE_TREE", "raw_reselect", "absolute_quality", "tree", {"max_depth": 6, "min_samples_leaf": 3}),
    LearnerSpec("RAW_RELATIVE_LOGISTIC", "raw_reselect", "case_relative_quality", "logistic", {"c": 0.5}),
    LearnerSpec("RAW_RELATIVE_FOREST", "raw_reselect", "case_relative_quality", "forest", {"max_depth": 8, "min_samples_leaf": 2}),
    LearnerSpec("HYBRID_RAW_RELATIVE_TREE", "hybrid_raw", "case_relative_quality", "tree", {"max_depth": 7, "min_samples_leaf": 3}),
    LearnerSpec("HYBRID_RAW_RELATIVE_FOREST", "hybrid_raw", "case_relative_quality", "forest", {"max_depth": 8, "min_samples_leaf": 2}),
    LearnerSpec("AXIS_SPAN_RELATIVE_LOGISTIC", "axis_span_reselect", "case_relative_quality", "logistic", {"c": 0.25}),
    LearnerSpec("HYBRID_AXIS_RELATIVE_TREE", "hybrid_axis_span", "case_relative_quality", "tree", {"max_depth": 8, "min_samples_leaf": 3}),
    LearnerSpec("HYBRID_ALL_INTRINSIC_TREE", "hybrid_all", "intrinsic_quality", "tree", {"max_depth": 8, "min_samples_leaf": 3}),
)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _binding_map(hash_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = hash_manifest.get("artifact_bindings")
    if not isinstance(rows, list):
        raise ValueError("sealed hash manifest has no artifact bindings")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("path"):
            raise ValueError("sealed hash manifest contains an invalid binding")
        result[str(row["path"])] = dict(row)
    return result


def _read_bound_bootstrap(
    path: Path,
    *,
    expected_sha256: str,
    purpose: str,
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    resolved = path.resolve()
    data = resolved.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(f"sealed bootstrap hash mismatch: {resolved}")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"sealed bootstrap artifact is not an object: {resolved}")
    entries.append(
        {
            "ordinal": len(entries) + 1,
            "path": str(resolved),
            "purpose": purpose,
            "case_id": None,
            "size_bytes": len(data),
            "sha256": actual,
            "read_succeeded": True,
        }
    )
    return payload, data


def load_lock_context() -> LockContext:
    """Verify sealed artifacts and derive TRAIN membership before label access."""

    bootstrap: list[dict[str, Any]] = []
    hashes, hashes_bytes = _read_bound_bootstrap(
        HASH_MANIFEST,
        expected_sha256=HASH_MANIFEST_SHA256,
        purpose="SEALED_HASH_MANIFEST",
        entries=bootstrap,
    )
    if hashes.get("kind") != HASH_KIND or hashes.get("model_score_performed") is not False:
        raise ValueError("hash manifest status/kind mismatch")
    bindings = _binding_map(hashes)

    def bound(relative: str, purpose: str) -> tuple[dict[str, Any], str]:
        row = bindings.get(relative)
        if row is None:
            raise ValueError(f"sealed binding is missing: {relative}")
        payload, _data = _read_bound_bootstrap(
            ROOT / relative,
            expected_sha256=str(row["sha256"]),
            purpose=purpose,
            entries=bootstrap,
        )
        return payload, str(row["sha256"])

    split, split_sha = bound(
        "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json",
        "SEALED_SPLIT_LOCK",
    )
    prereg, prereg_sha = bound(
        "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json",
        "SEALED_PREREGISTRATION",
    )
    allowlist, allowlist_sha = bound(
        "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json",
        "SEALED_TRAIN_ALLOWLIST",
    )
    if split.get("kind") != SPLIT_KIND or split.get("status") != "LOCKED_RETROSPECTIVE_PSEUDO_UNSEEN":
        raise ValueError("split lock status/kind mismatch")
    if split.get("label_or_outcome_fields_present") is not False:
        raise ValueError("split lock reports label/outcome fields")
    if prereg.get("kind") != PREREG_KIND or prereg.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN":
        raise ValueError("preregistration status/kind mismatch")
    if prereg.get("evaluation_deferred") is not True:
        raise ValueError("preregistration does not defer evaluation")
    if allowlist.get("kind") != ALLOWLIST_KIND or allowlist.get("status") != "LOCKED_TRAIN_ONLY":
        raise ValueError("TRAIN allowlist status/kind mismatch")
    if allowlist.get("reject_any_case_not_listed") is not True:
        raise ValueError("TRAIN allowlist is not fail closed")
    rows = allowlist.get("cases")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("TRAIN allowlist must contain exactly 18 rows")
    train_ids = tuple(str(row.get("case_id") or "") for row in rows)
    split_train = tuple(str(value) for value in split.get("train_case_ids") or [])
    if train_ids != split_train or len(set(train_ids)) != 18:
        raise ValueError("TRAIN allowlist and split membership disagree")
    scan_sha = {str(row["case_id"]): str(row["scan_sha256"]) for row in rows}
    if any(len(value) != 64 for value in scan_sha.values()):
        raise ValueError("TRAIN allowlist contains an invalid scan hash")
    if allowlist.get("split_binding", {}).get("sha256") != split_sha:
        raise ValueError("TRAIN allowlist split binding mismatch")
    if allowlist.get("prereg_binding", {}).get("sha256") != prereg_sha:
        raise ValueError("TRAIN allowlist preregistration binding mismatch")
    # Retain the actual first read rather than re-reading the manifest later.
    if sha256_bytes(hashes_bytes) != HASH_MANIFEST_SHA256:
        raise RuntimeError("hash manifest changed during bootstrap")
    source_bindings = hashes.get("source_bindings")
    if not isinstance(source_bindings, dict):
        raise ValueError("sealed hash manifest has no source bindings")
    input_manifest = source_bindings.get("input_manifest")
    if not isinstance(input_manifest, dict):
        raise ValueError("sealed hash manifest has no input-manifest binding")
    sealed_input_sha = str(input_manifest.get("sha256") or "")
    if len(sealed_input_sha) != 64:
        raise ValueError("sealed input-manifest hash is invalid")
    train_commitment = sha256_bytes(
        canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_sha[case_id]}
                for case_id in train_ids
            ]
        )
    )
    return LockContext(
        split_sha256=split_sha,
        allowlist_sha256=allowlist_sha,
        preregistration_sha256=prereg_sha,
        train_case_ids=train_ids,
        scan_sha_by_case=scan_sha,
        preregistration=prereg,
        sealed_input_manifest_sha256=sealed_input_sha,
        train_commitment_sha256=train_commitment,
        bootstrap_entries=tuple(bootstrap),
    )


def build_guarded_reader(
    lock: LockContext,
    *,
    label_attestation: Path,
    label_attestation_sha256: str,
    cache_attestation: Path,
    cache_attestation_sha256: str,
) -> GuardedReader:
    """Create only the label-free preauthorization capability.

    The projection and case source files are deliberately absent.  They are
    authorized later from a validated attestation and execution manifest.
    """

    attestation = label_attestation.resolve()
    cache_attestation = cache_attestation.resolve()
    forbidden_names = {
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
    }
    if attestation.name in forbidden_names or cache_attestation.name in forbidden_names:
        raise PermissionError("full-cohort label/catalog paths are forbidden")
    if len(label_attestation_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in label_attestation_sha256
    ):
        raise ValueError("an independently supplied attestation SHA-256 is required")
    _lower_hex_sha(
        cache_attestation_sha256,
        name="independently supplied TRAIN cache attestation hash",
    )
    allowed: dict[Path, str] = {
        attestation: "TRAIN_LABEL_PROJECTION_ATTESTATION",
        cache_attestation: "TRAIN_CACHE_MANIFEST_ATTESTATION",
        RUNTIME_SOURCE.resolve(): "RUNTIME_SOURCE_HASH_BINDING",
        OPTIMIZER_SOURCE.resolve(): "OPTIMIZER_SOURCE_HASH_BINDING",
    }
    return GuardedReader(
        allowed=allowed,
        expected_sha256={
            attestation: label_attestation_sha256,
            cache_attestation: cache_attestation_sha256,
        },
        entries=[dict(row) for row in lock.bootstrap_entries],
    )


def _lower_hex_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _validate_attestation_before_projection(
    attestation: Mapping[str, Any],
    *,
    lock: LockContext,
    root: Path,
) -> tuple[Path, str, int, list[tuple[Path, str, int, str]]]:
    """Validate only label-value-free custody evidence before projection access."""

    if attestation.get("kind") != ATTESTATION_KIND or attestation.get("schema_version") != 1:
        raise ValueError("TRAIN label projection attestation schema/kind mismatch")
    if attestation.get("split_sha256") != lock.split_sha256:
        raise ValueError("label projection attestation split binding mismatch")
    if attestation.get("train_allowlist_sha256") != lock.allowlist_sha256:
        raise ValueError("label projection attestation allowlist binding mismatch")
    if attestation.get("train_set_commitment_sha256") != lock.train_commitment_sha256:
        raise ValueError("label projection attestation TRAIN commitment mismatch")
    complement = _lower_hex_sha(
        attestation.get("blinded_complement_commitment_sha256"),
        name="blinded complement commitment",
    )
    if complement == lock.train_commitment_sha256:
        raise ValueError("blinded complement commitment aliases TRAIN")
    exact_checks = {
        "case_count": attestation.get("case_count") == 18,
        "file_inventory_count": attestation.get("file_inventory_count") == 18,
        "train_exact_match": attestation.get("train_exact_match") is True,
        "train_holdout_intersection_count": attestation.get("train_holdout_intersection_count") == 0,
        "aggregate_catalog_read_count": attestation.get("aggregate_catalog_read_count") == 1,
        "holdout_label_row_emit_count": attestation.get("holdout_label_row_emit_count") == 0,
        "holdout_case_file_emit_count": attestation.get("holdout_case_file_emit_count") == 0,
        "optimizer_holdout_access_count": attestation.get("optimizer_holdout_access_count") == 0,
        "projection_contains_only_train": attestation.get("projection_contains_only_train") is True,
        "independent_validation_passed": attestation.get("independent_validation_passed") is True,
    }
    if not all(exact_checks.values()):
        failed = sorted(name for name, passed in exact_checks.items() if not passed)
        raise ValueError(f"label projection custody attestation failed: {failed}")
    evidence_bindings: list[tuple[Path, str, int, str]] = []
    for prefix in (
        "builder",
        "validator",
        "staged_validator_pass",
        "access_evidence",
    ):
        binding = attestation.get(f"{prefix}_binding")
        if not isinstance(binding, dict) or not binding.get("path"):
            raise ValueError(f"label attestation is missing {prefix} binding")
        binding_sha = _lower_hex_sha(
            binding.get("sha256"), name=f"{prefix} binding hash"
        )
        binding_size = binding.get("size_bytes")
        if (
            isinstance(binding_size, bool)
            or not isinstance(binding_size, int)
            or binding_size <= 0
        ):
            raise ValueError(f"{prefix} binding size must be positive")
        relative_binding = str(binding["path"])
        if Path(relative_binding).is_absolute():
            raise ValueError(f"{prefix} binding path must be repository-relative")
        binding_path = (root / relative_binding).resolve()
        if root.resolve() not in binding_path.parents:
            raise ValueError(f"{prefix} binding path escapes repository root")
        evidence_bindings.append(
            (
                binding_path,
                binding_sha,
                binding_size,
                f"CUSTODY_{prefix.upper()}_BINDING",
            )
        )
    projection = attestation.get("projection_binding")
    if not isinstance(projection, dict):
        raise ValueError("label attestation is missing projection binding")
    if projection.get("kind") != PROJECTION_KIND:
        raise ValueError("attested projection kind mismatch")
    relative = str(projection.get("path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("attested projection path must be repository-relative")
    projection_path = (root / relative).resolve()
    if root.resolve() not in projection_path.parents:
        raise ValueError("attested projection path escapes repository root")
    if projection_path.name in {
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
    }:
        raise PermissionError("attested path points to a forbidden full-cohort artifact")
    projection_sha = _lower_hex_sha(
        projection.get("sha256"), name="attested projection hash"
    )
    size = projection.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("attested projection size must be a positive integer")
    return projection_path, projection_sha, size, evidence_bindings


def _validate_projection(
    projection: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    projection_sha256: str,
    attestation_sha256: str,
    lock: LockContext,
) -> dict[str, dict[str, Any]]:
    if projection.get("kind") != PROJECTION_KIND or projection.get("schema_version") != 1:
        raise ValueError("TRAIN label projection schema/kind mismatch")
    if attestation.get("kind") != ATTESTATION_KIND or attestation.get("schema_version") != 1:
        raise ValueError("TRAIN label projection attestation schema/kind mismatch")
    if attestation.get("projection_binding", {}).get("sha256") != projection_sha256:
        raise ValueError("label projection attestation hash mismatch")
    if attestation.get("split_sha256") != lock.split_sha256:
        raise ValueError("label projection split binding mismatch")
    if attestation.get("train_allowlist_sha256") != lock.allowlist_sha256:
        raise ValueError("label projection allowlist binding mismatch")
    if (
        attestation.get("case_count") != 18
        or attestation.get("optimizer_holdout_access_count") != 0
        or attestation.get("holdout_label_row_emit_count") != 0
    ):
        raise ValueError("label projection attestation does not prove optimizer isolation")
    if projection.get("split_sha256") != lock.split_sha256:
        raise ValueError("label projection split binding mismatch")
    if projection.get("train_allowlist_sha256") != lock.allowlist_sha256:
        raise ValueError("label projection allowlist binding mismatch")
    if projection.get("case_count") != 18:
        raise ValueError("TRAIN label projection must contain exactly 18 cases")
    rows = projection.get("cases")
    if not isinstance(rows, list):
        raise ValueError("TRAIN label projection cases must be an array")
    mapping: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("TRAIN label projection rows must be objects")
        forbidden = FORBIDDEN_LABEL_PROJECTION_KEYS.intersection(raw)
        if forbidden:
            raise ValueError(f"TRAIN label projection contains forbidden keys: {sorted(forbidden)}")
        case_id = str(raw.get("case_id") or "")
        if case_id not in lock.scan_sha_by_case:
            raise PermissionError(f"label projection contains non-TRAIN case: {case_id}")
        if case_id in mapping:
            raise ValueError(f"label projection contains duplicate case: {case_id}")
        if str(raw.get("scan_sha256") or "") != lock.scan_sha_by_case[case_id]:
            raise ValueError(f"{case_id}: label projection scan hash mismatch")
        bends = raw.get("bends")
        if not isinstance(bends, list):
            raise ValueError(f"{case_id}: label projection bends must be an array")
        ids: set[str] = set()
        clean_bends: list[dict[str, Any]] = []
        for bend in bends:
            if not isinstance(bend, dict):
                raise ValueError(f"{case_id}: bend rows must be objects")
            bend_id = str(bend.get("bend_id") or "")
            if not bend_id or bend_id in ids:
                raise ValueError(f"{case_id}: bend IDs must be unique and non-empty")
            ids.add(bend_id)
            endpoints_from_record(bend)
            support = bend.get("reviewer_support_count")
            if isinstance(support, bool) or not isinstance(support, int) or support not in (1, 2, 3):
                raise ValueError(f"{case_id}/{bend_id}: reviewer support tier must be 1, 2, or 3")
            clean_bends.append(dict(bend))
        mapping[case_id] = {"case_id": case_id, "scan_sha256": raw["scan_sha256"], "bends": clean_bends}
    if tuple(mapping) != lock.train_case_ids:
        raise ValueError("TRAIN label projection order/membership differs from sealed allowlist")
    if attestation_sha256 != sha256_bytes(canonical_bytes(attestation)):
        # Attestations are required to use repository canonical JSON bytes.
        raise ValueError("label projection attestation is not canonical JSON")
    return mapping


def _validate_cache_attestation_before_manifest(
    attestation: Mapping[str, Any],
    *,
    lock: LockContext,
    root: Path,
) -> tuple[Path, str, int, list[tuple[Path, str, int, str]]]:
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind") != TRAIN_CACHE_ATTESTATION_KIND
    ):
        raise ValueError("TRAIN cache attestation schema/kind mismatch")
    if attestation.get("split_sha256") != lock.split_sha256:
        raise ValueError("TRAIN cache attestation split binding mismatch")
    if attestation.get("train_allowlist_sha256") != lock.allowlist_sha256:
        raise ValueError("TRAIN cache attestation allowlist binding mismatch")
    if attestation.get("train_set_commitment_sha256") != lock.train_commitment_sha256:
        raise ValueError("TRAIN cache attestation commitment mismatch")
    _lower_hex_sha(
        attestation.get("blinded_complement_commitment_sha256"),
        name="TRAIN cache blinded complement commitment",
    )
    checks = {
        "case_count": attestation.get("case_count") == 18,
        "record_count": attestation.get("record_count") == 36,
        "train_exact_match": attestation.get("train_exact_match") is True,
        "train_holdout_intersection_count": attestation.get("train_holdout_intersection_count") == 0,
        "holdout_entry_count": attestation.get("holdout_entry_count") == 0,
        "holdout_status_count": attestation.get("holdout_status_count") == 0,
        "holdout_hash_count": attestation.get("holdout_hash_count") == 0,
        "optimizer_holdout_access_count": attestation.get("optimizer_holdout_access_count") == 0,
        "independent_validation_passed": attestation.get("independent_validation_passed") is True,
    }
    if not all(checks.values()):
        raise ValueError(
            "TRAIN cache custody attestation failed: "
            + str(sorted(name for name, value in checks.items() if not value))
        )
    evidence: list[tuple[Path, str, int, str]] = []
    for prefix in (
        "builder",
        "validator",
        "staged_validator_pass",
        "access_evidence",
    ):
        binding = attestation.get(f"{prefix}_binding")
        if not isinstance(binding, dict) or not binding.get("path"):
            raise ValueError(f"TRAIN cache attestation is missing {prefix} binding")
        relative = str(binding["path"])
        if Path(relative).is_absolute():
            raise ValueError(f"TRAIN cache {prefix} path must be repository-relative")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"TRAIN cache {prefix} path escapes repository root")
        binding_size = binding.get("size_bytes")
        if (
            isinstance(binding_size, bool)
            or not isinstance(binding_size, int)
            or binding_size <= 0
        ):
            raise ValueError(f"TRAIN cache {prefix} binding size must be positive")
        evidence.append(
            (
                path,
                _lower_hex_sha(binding.get("sha256"), name=f"TRAIN cache {prefix} hash"),
                binding_size,
                f"CUSTODY_{prefix.upper()}_BINDING",
            )
        )
    binding = attestation.get("manifest_binding")
    if not isinstance(binding, dict) or binding.get("kind") != TRAIN_CACHE_MANIFEST_KIND:
        raise ValueError("TRAIN cache manifest binding is missing or invalid")
    relative = str(binding.get("path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("TRAIN cache manifest path must be repository-relative")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("TRAIN cache manifest path escapes repository root")
    if path.name == "development_execution_manifest.json":
        raise PermissionError("full 28-case execution manifest is forbidden")
    size = binding.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("TRAIN cache manifest size must be positive")
    return (
        path,
        _lower_hex_sha(binding.get("sha256"), name="TRAIN cache manifest hash"),
        size,
        evidence,
    )


def _validate_train_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    lock: LockContext,
    root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    if manifest.get("schema_version") != 1 or manifest.get("kind") != TRAIN_CACHE_MANIFEST_KIND:
        raise ValueError("TRAIN cache manifest schema/kind mismatch")
    if manifest.get("source_only") is not True or manifest.get("holdout_entry_count") != 0:
        raise ValueError("TRAIN cache manifest is not source-only/holdout-empty")
    if manifest.get("split_sha256") != lock.split_sha256:
        raise ValueError("TRAIN cache manifest split binding mismatch")
    if manifest.get("train_allowlist_sha256") != lock.allowlist_sha256:
        raise ValueError("TRAIN cache manifest allowlist binding mismatch")
    if manifest.get("train_set_commitment_sha256") != lock.train_commitment_sha256:
        raise ValueError("TRAIN cache manifest commitment mismatch")
    if manifest.get("case_count") != 18 or manifest.get("record_count") != 36:
        raise ValueError("TRAIN cache manifest must contain exactly 18 cases / 36 records")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError("TRAIN cache manifest records must contain exactly 36 rows")
    expected_order = [
        (case_id, pass_name)
        for case_id in lock.train_case_ids
        for pass_name in ("primary", "replay")
    ]
    allowed_keys = {
        "case_id",
        "runtime_case_id",
        "pass",
        "scan_sha256",
        "semantic_path",
        "semantic_sha256",
        "pool_path",
        "pool_sha256",
    }
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    runtime_by_case: dict[str, str] = {}
    cases_by_runtime: dict[str, str] = {}
    binding_rows: list[dict[str, Any]] = []
    for raw, expected in zip(records, expected_order):
        if not isinstance(raw, dict) or set(raw) != allowed_keys:
            raise ValueError("TRAIN cache record fields differ from the sealed contract")
        opaque_id = str(raw["case_id"])
        pass_name = str(raw["pass"])
        if (opaque_id, pass_name) != expected:
            raise PermissionError("TRAIN cache manifest order/membership is not exact")
        runtime_id = str(raw["runtime_case_id"])
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
        ):
            raise ValueError("TRAIN cache runtime identity is invalid")
        prior_runtime = runtime_by_case.setdefault(opaque_id, runtime_id)
        if prior_runtime != runtime_id:
            raise ValueError("TRAIN cache case maps to different primary/replay runtime IDs")
        prior_case = cases_by_runtime.setdefault(runtime_id, opaque_id)
        if prior_case != opaque_id:
            raise ValueError("TRAIN cache runtime identity is not a case bijection")
        scan_sha = _lower_hex_sha(raw["scan_sha256"], name="TRAIN cache scan hash")
        if scan_sha != lock.scan_sha_by_case[opaque_id]:
            raise ValueError("TRAIN cache scan hash differs from sealed allowlist")
        paths: dict[str, Path] = {}
        for prefix, expected_name in (
            ("semantic", "semantic_result.json"),
            ("pool", f"{runtime_id}.json"),
        ):
            relative = str(raw[f"{prefix}_path"])
            if Path(relative).is_absolute():
                raise ValueError("TRAIN cache data paths must be repository-relative")
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or path.name != expected_name:
                raise ValueError("TRAIN cache data path is invalid")
            paths[prefix] = path
        row = {
            "case_id": opaque_id,
            "runtime_case_id": runtime_id,
            "pass": pass_name,
            "scan_sha256": scan_sha,
            "semantic_path": paths["semantic"],
            "semantic_sha256": _lower_hex_sha(raw["semantic_sha256"], name="semantic hash"),
            "pool_path": paths["pool"],
            "pool_sha256": _lower_hex_sha(raw["pool_sha256"], name="pool hash"),
        }
        mapping[(opaque_id, pass_name)] = row
        binding_rows.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in row.items()
            }
        )
    for opaque_id in lock.train_case_ids:
        primary = mapping[(opaque_id, "primary")]
        replay = mapping[(opaque_id, "replay")]
        if primary["semantic_sha256"] != replay["semantic_sha256"]:
            raise ValueError(f"{opaque_id}: semantic replay hashes differ")
        if primary["pool_sha256"] != replay["pool_sha256"]:
            raise ValueError(f"{opaque_id}: pool replay hashes differ")
    return mapping, sha256_bytes(canonical_bytes(binding_rows))


def load_train_cases(
    lock: LockContext,
    reader: GuardedReader,
    *,
    label_attestation: Path,
    cache_attestation: Path,
) -> tuple[list[CaseData], str, str, str, str, str, str, str]:
    # Phase 1: open and validate only the independently hashed, label-value-free
    # custody attestation.  The projection is not yet an authorized path.
    attestation_bytes = reader.read_bytes(
        label_attestation, purpose="TRAIN_LABEL_PROJECTION_ATTESTATION"
    )
    attestation = json.loads(attestation_bytes)
    if not isinstance(attestation, dict):
        raise ValueError("TRAIN label projection attestation must be an object")
    attestation_sha = sha256_bytes(attestation_bytes)
    if attestation_sha != sha256_bytes(canonical_bytes(attestation)):
        raise ValueError("label projection attestation is not canonical JSON")
    (
        projection_path,
        projection_sha,
        projection_size,
        evidence_bindings,
    ) = _validate_attestation_before_projection(
        attestation,
        lock=lock,
        root=ROOT,
    )
    for evidence_path, evidence_sha, evidence_size, evidence_purpose in evidence_bindings:
        newly_authorized = reader.authorize(
            evidence_path,
            purpose=evidence_purpose,
            expected_sha256=evidence_sha,
            expected_size_bytes=evidence_size,
        )
        if newly_authorized:
            reader.read_bytes(evidence_path, purpose=evidence_purpose)

    # Phase 2: independently hash-check the label-free TRAIN cache attestation,
    # then authorize its 18-only cache manifest.  The full 28-case execution
    # manifest is never a readable path in this process.
    cache_attestation_bytes = reader.read_bytes(
        cache_attestation, purpose="TRAIN_CACHE_MANIFEST_ATTESTATION"
    )
    cache_attestation_payload = json.loads(cache_attestation_bytes)
    if not isinstance(cache_attestation_payload, dict):
        raise ValueError("TRAIN cache attestation must be an object")
    cache_attestation_sha = sha256_bytes(cache_attestation_bytes)
    if cache_attestation_sha != sha256_bytes(canonical_bytes(cache_attestation_payload)):
        raise ValueError("TRAIN cache attestation is not canonical JSON")
    (
        cache_manifest_path,
        cache_manifest_sha,
        cache_manifest_size,
        cache_evidence_bindings,
    ) = _validate_cache_attestation_before_manifest(
        cache_attestation_payload,
        lock=lock,
        root=ROOT,
    )
    for evidence_path, evidence_sha, evidence_size, evidence_purpose in cache_evidence_bindings:
        newly_authorized = reader.authorize(
            evidence_path,
            purpose=evidence_purpose,
            expected_sha256=evidence_sha,
            expected_size_bytes=evidence_size,
        )
        if newly_authorized:
            reader.read_bytes(evidence_path, purpose=evidence_purpose)
    reader.authorize(
        cache_manifest_path,
        purpose="TRAIN_CACHE_MANIFEST",
        expected_sha256=cache_manifest_sha,
    )
    cache_manifest_bytes = reader.read_bytes(
        cache_manifest_path, purpose="TRAIN_CACHE_MANIFEST"
    )
    if len(cache_manifest_bytes) != cache_manifest_size:
        raise ValueError("TRAIN cache manifest size differs from attestation")
    cache_manifest = json.loads(cache_manifest_bytes)
    if not isinstance(cache_manifest, dict):
        raise ValueError("TRAIN cache manifest must be an object")
    cache_records, cache_binding_sha = _validate_train_cache_manifest(
        cache_manifest,
        lock=lock,
        root=ROOT,
    )
    reader.authorize(
        projection_path,
        purpose="TRAIN_LABEL_PROJECTION",
        expected_sha256=projection_sha,
    )
    for opaque_id in lock.train_case_ids:
        for pass_name in ("primary", "replay"):
            record = cache_records[(opaque_id, pass_name)]
            reader.authorize(
                record["semantic_path"],
                purpose=f"{pass_name.upper()}_V51_SOURCE",
                expected_sha256=record["semantic_sha256"],
            )
            reader.authorize(
                record["pool_path"],
                purpose=f"{pass_name.upper()}_SCAN_ONLY_POOL",
                expected_sha256=record["pool_sha256"],
            )

    # Phase 3: source/code inputs are fully bound before any label value opens.
    runtime_code = reader.read_bytes(RUNTIME_SOURCE, purpose="RUNTIME_SOURCE_HASH_BINDING")
    optimizer_code = reader.read_bytes(OPTIMIZER_SOURCE, purpose="OPTIMIZER_SOURCE_HASH_BINDING")
    source_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for opaque_id in lock.train_case_ids:
        runtime_id = str(cache_records[(opaque_id, "primary")]["runtime_case_id"])
        values: dict[str, dict[str, Any]] = {}
        raw_bytes: dict[str, bytes] = {}
        for pass_name in ("primary", "replay"):
            record = cache_records[(opaque_id, pass_name)]
            semantic_path = record["semantic_path"]
            pool_path = record["pool_path"]
            semantic_purpose = f"{pass_name.upper()}_V51_SOURCE"
            pool_purpose = f"{pass_name.upper()}_SCAN_ONLY_POOL"
            raw_bytes[f"{pass_name}_semantic"] = reader.read_bytes(
                semantic_path, purpose=semantic_purpose, case_id=opaque_id
            )
            raw_bytes[f"{pass_name}_pool"] = reader.read_bytes(
                pool_path, purpose=pool_purpose, case_id=opaque_id
            )
            values[f"{pass_name}_semantic"] = json.loads(raw_bytes[f"{pass_name}_semantic"])
            values[f"{pass_name}_pool"] = json.loads(raw_bytes[f"{pass_name}_pool"])
        if raw_bytes["primary_semantic"] != raw_bytes["replay_semantic"]:
            raise ValueError(f"{opaque_id}: frozen V51 primary/replay semantic bytes differ")
        if raw_bytes["primary_pool"] != raw_bytes["replay_pool"]:
            raise ValueError(f"{opaque_id}: scan-only pool primary/replay bytes differ")
        for pass_name in ("primary", "replay"):
            semantic = values[f"{pass_name}_semantic"]
            validate_source_input(semantic)
            record = cache_records[(opaque_id, pass_name)]
            if record["scan_sha256"] != lock.scan_sha_by_case[opaque_id]:
                raise ValueError(f"{opaque_id}: bound semantic scan hash is missing or wrong")
            pool = values[f"{pass_name}_pool"]
            if str(pool.get("case_id") or "") != runtime_id:
                raise ValueError(f"{opaque_id}: candidate pool runtime identity mismatch")
            inference = pool.get("inference_contract")
            if not isinstance(inference, dict) or any(
                inference.get(field) is not False
                for field in (
                    "part_identifier_exposed",
                    "expected_count_exposed",
                    "manual_truth_exposed",
                    "cad_exposed",
                )
            ):
                raise ValueError(f"{opaque_id}: candidate pool violates source-only contract")
        source_by_case[opaque_id] = values

    # Phase 4: only now may the exact attested projection bytes be opened.
    projection_bytes = reader.read_bytes(
        projection_path, purpose="TRAIN_LABEL_PROJECTION"
    )
    if len(projection_bytes) != projection_size:
        raise ValueError("TRAIN label projection size differs from attestation")
    projection = json.loads(projection_bytes)
    if not isinstance(projection, dict):
        raise ValueError("TRAIN label projection must be an object")
    label_rows = _validate_projection(
        projection,
        attestation,
        projection_sha256=projection_sha,
        attestation_sha256=attestation_sha,
        lock=lock,
    )
    cases: list[CaseData] = []
    for opaque_id in lock.train_case_ids:
        runtime_id = str(cache_records[(opaque_id, "primary")]["runtime_case_id"])
        values = source_by_case[opaque_id]
        cases.append(
            CaseData(
                case_id=opaque_id,
                runtime_case_id=runtime_id,
                scan_sha256=lock.scan_sha_by_case[opaque_id],
                bends=[dict(row) for row in label_rows[opaque_id]["bends"]],
                primary_v51=values["primary_semantic"],
                primary_pool=values["primary_pool"],
                replay_v51=values["replay_semantic"],
                replay_pool=values["replay_pool"],
            )
        )
    return (
        cases,
        projection_sha,
        attestation_sha,
        sha256_bytes(runtime_code),
        sha256_bytes(optimizer_code),
        cache_manifest_sha,
        cache_attestation_sha,
        cache_binding_sha,
    )


def _observation_target(observation: Observation, bends: Sequence[dict[str, Any]]) -> bool:
    prediction = np.asarray(observation.endpoints, dtype=np.float64)
    return any(
        is_match(pair_metrics(endpoints_from_record(bend), prediction), MATCH_THRESHOLDS)
        for bend in bends
    )


def prepare_view(case: CaseData, spec: LearnerSpec) -> PreparedView:
    validate_source_input(case.primary_v51)
    key = (spec.universe, spec.representation)
    if key not in case.views:
        observations, raw = build_observations(
            case.primary_v51,
            case.primary_pool,
            universe=spec.universe,
            span_scales=SPAN_SCALES,
            fixed_spans_mm=FIXED_SPANS_MM,
        )
        matrix = feature_matrix(observations, raw, representation=spec.representation)
        target = np.asarray(
            [_observation_target(row, case.bends) for row in observations], dtype=np.bool_
        )
        case.views[key] = PreparedView(observations, raw, matrix, target)
    return case.views[key]


def _imputer_medians(matrix: np.ndarray) -> np.ndarray:
    values = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        values[column] = float(np.median(finite)) if len(finite) else 0.0
    return values


def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    rows, columns = np.where(~np.isfinite(result))
    if len(rows):
        result[rows, columns] = medians[columns]
    return result


def _portable_tree(classifier: Any) -> dict[str, Any]:
    tree = classifier.tree_
    classes = list(classifier.classes_)
    positive_index = classes.index(True)
    probabilities = []
    for node in range(tree.node_count):
        values = np.asarray(tree.value[node], dtype=np.float64).reshape(-1)
        total = float(values.sum())
        probabilities.append(float(values[positive_index] / total) if total else 0.0)
    return {
        "children_left": [int(value) for value in tree.children_left],
        "children_right": [int(value) for value in tree.children_right],
        "feature_index": [int(value) for value in tree.feature],
        "threshold": [float(value) for value in tree.threshold],
        "positive_probability": probabilities,
    }


def fit_portable_classifier(
    spec: LearnerSpec, cases: Sequence[CaseData]
) -> dict[str, Any]:
    views = [prepare_view(case, spec) for case in cases]
    matrix = np.concatenate([view.matrix for view in views])
    target = np.concatenate([view.target for view in views])
    if matrix.shape[0] == 0:
        return {
            "imputer_medians": [0.0] * len(feature_names(spec.representation)),
            "kind": "constant",
            "positive_probability": 0.0,
        }
    medians = _imputer_medians(matrix)
    fitted = _impute(matrix, medians)
    base: dict[str, Any] = {"imputer_medians": medians.tolist()}
    unique = np.unique(target)
    if len(unique) == 1:
        return {**base, "kind": "constant", "positive_probability": float(unique[0])}
    if spec.classifier_kind == "logistic":
        mean = np.mean(fitted, axis=0)
        scale = np.std(fitted, axis=0)
        scale[scale <= 1e-12] = 1.0
        normalized = (fitted - mean) / scale
        classifier = LogisticRegression(
            C=float(spec.classifier_parameters["c"]),
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=1701,
        )
        classifier.fit(normalized, target)
        return {
            **base,
            "kind": "logistic",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficients": classifier.coef_[0].astype(float).tolist(),
            "intercept": float(classifier.intercept_[0]),
        }
    if spec.classifier_kind == "tree":
        classifier = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=int(spec.classifier_parameters["max_depth"]),
            min_samples_leaf=int(spec.classifier_parameters["min_samples_leaf"]),
            class_weight="balanced",
            random_state=1701,
        )
        classifier.fit(fitted, target)
        return {**base, "kind": "tree", "tree": _portable_tree(classifier)}
    if spec.classifier_kind == "forest":
        classifier = ExtraTreesClassifier(
            n_estimators=32,
            criterion="entropy",
            max_depth=int(spec.classifier_parameters["max_depth"]),
            min_samples_leaf=int(spec.classifier_parameters["min_samples_leaf"]),
            max_features="sqrt",
            class_weight="balanced",
            random_state=1701,
            n_jobs=1,
        )
        classifier.fit(fitted, target)
        return {
            **base,
            "kind": "forest",
            "trees": [_portable_tree(estimator) for estimator in classifier.estimators_],
        }
    raise ValueError(f"unsupported learner classifier: {spec.classifier_kind}")


def _profile(threshold: float, cap: int) -> dict[str, Any]:
    return {
        "probability_threshold": float(threshold),
        "output_cap": int(cap),
        **DUPLICATE_PROFILE,
    }


def predict_with_classifier(
    case: CaseData,
    spec: LearnerSpec,
    classifier: Mapping[str, Any],
    *,
    threshold: float,
    cap: int,
) -> list[dict[str, Any]]:
    view = prepare_view(case, spec)
    probabilities = classifier_probabilities(view.matrix, classifier)
    selected = select_observations(view.observations, probabilities, _profile(threshold, cap))
    return predictions_from_selection(selected, model_sha256="0" * 64)


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if tp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tiers: dict[str, dict[str, int | float]] = {}
    for support in (1, 2, 3):
        name = f"REVIEWER_SUPPORT_{support}"
        gold = sum(int(row["support_tiers"][name]["gold"]) for row in rows)
        matched = sum(int(row["support_tiers"][name]["matched"]) for row in rows)
        tiers[name] = {
            "gold": gold,
            "matched": matched,
            "recall": round(matched / gold, 6) if gold else 1.0,
        }
    return {
        "case_count": len(rows),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row["exact"]) for row in rows),
        "label_support_tiers": tiers,
    }


def score_predictions(
    cases: Sequence[CaseData], predictions: Mapping[str, Sequence[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = match_lines(case.bends, list(predictions[case.case_id]), MATCH_THRESHOLDS)
        support_by_id = {
            str(row["bend_id"]): int(row["reviewer_support_count"])
            for row in case.bends
        }
        matched = {str(row["gold_id"]) for row in result["matches"]}
        result["support_tiers"] = {
            f"REVIEWER_SUPPORT_{support}": {
                "gold": sum(value == support for value in support_by_id.values()),
                "matched": sum(
                    support_by_id[gold_id] == support for gold_id in matched
                ),
            }
            for support in (1, 2, 3)
        }
        result["case_id"] = case.case_id
        rows.append(result)
    rows.sort(key=lambda row: row["case_id"])
    return _metric_summary(rows), rows


def baseline_predictions(cases: Sequence[CaseData]) -> dict[str, list[dict[str, Any]]]:
    return {
        case.case_id: [dict(row) for row in case.primary_v51.get("predictions") or []]
        for case in cases
    }


def training_gate(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    fp_bound = max(2.0, 0.10 * float(baseline["false_positive"]))
    def exact_f1(row: Mapping[str, Any]) -> float:
        tp = int(row["true_positive"])
        fp = int(row["false_positive"])
        fn = int(row.get("false_negative", 0))
        return 2.0 * tp / (2.0 * tp + fp + fn) if tp else 0.0

    candidate_f1 = exact_f1(candidate)
    baseline_f1 = exact_f1(baseline)
    support_checks = {
        tier: int(candidate["label_support_tiers"][tier]["matched"])
        >= int(baseline["label_support_tiers"][tier]["matched"])
        for tier in baseline["label_support_tiers"]
    }
    checks = {
        "overall_micro_f1_delta_at_least_0_05": candidate_f1 - baseline_f1 >= 0.05 - 1e-12,
        "true_positive_nondecrease": int(candidate["true_positive"]) >= int(baseline["true_positive"]),
        "false_positive_increase_bounded": int(candidate["false_positive"]) - int(baseline["false_positive"]) <= fp_bound,
        "exact_case_count_nondecrease": int(candidate["exact_case_count"]) >= int(baseline["exact_case_count"]),
        "every_label_support_tier_tp_nondecrease": all(support_checks.values()),
    }
    return {
        "all_conjunctive_checks_passed": all(checks.values()),
        "checks": checks,
        "label_support_tier_checks": support_checks,
        "candidate_minus_reference": {
            "f1": round(candidate_f1 - baseline_f1, 6),
            "true_positive": int(candidate["true_positive"]) - int(baseline["true_positive"]),
            "false_positive": int(candidate["false_positive"]) - int(baseline["false_positive"]),
            "exact_case_count": int(candidate["exact_case_count"]) - int(baseline["exact_case_count"]),
        },
        "false_positive_max_increase": fp_bound,
    }


def case_folds(cases: Sequence[CaseData], n_splits: int, *, seed: str) -> list[list[CaseData]]:
    if n_splits < 2 or n_splits > len(cases):
        raise ValueError("invalid case-group fold count")
    ordered = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f"{seed}|{case.scan_sha256}".encode("ascii")
        ).hexdigest(),
    )
    folds = [[] for _ in range(n_splits)]
    for index, case in enumerate(ordered):
        folds[index % n_splits].append(case)
    if any(not fold for fold in folds):
        raise RuntimeError("case-group fold assignment produced an empty fold")
    return folds


def _oof_probabilities(
    cases: Sequence[CaseData], spec: LearnerSpec, *, n_splits: int, seed: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    probabilities: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    folds = case_folds(cases, n_splits, seed=seed)
    for fold_index, held_out in enumerate(folds, 1):
        held_ids = {case.case_id for case in held_out}
        fit_cases = [case for case in cases if case.case_id not in held_ids]
        classifier = fit_portable_classifier(spec, fit_cases)
        for case in held_out:
            view = prepare_view(case, spec)
            probabilities[case.case_id] = classifier_probabilities(view.matrix, classifier)
        audit.append(
            {
                "fold": fold_index,
                "fit_case_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in fit_cases))
                ),
                "held_out_case_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in held_out))
                ),
                "fit_case_count": len(fit_cases),
                "held_out_case_count": len(held_out),
            }
        )
    if set(probabilities) != {case.case_id for case in cases}:
        raise RuntimeError("OOF probabilities do not cover every complete case")
    return probabilities, audit


def _predictions_from_probabilities(
    cases: Sequence[CaseData],
    spec: LearnerSpec,
    probabilities: Mapping[str, np.ndarray],
    *,
    threshold: float,
    cap: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    profile = _profile(threshold, cap)
    for case in cases:
        view = prepare_view(case, spec)
        selected = select_observations(
            view.observations, probabilities[case.case_id], profile
        )
        result[case.case_id] = predictions_from_selection(
            selected, model_sha256="0" * 64
        )
    return result


def select_configuration(
    cases: Sequence[CaseData], *, n_splits: int, seed: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline, _baseline_rows = score_predictions(cases, baseline_predictions(cases))
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for spec in LEARNER_SPECS:
        probabilities, fold_audit = _oof_probabilities(
            cases, spec, n_splits=n_splits, seed=f"{seed}|{spec.name}"
        )
        for threshold in PROBABILITY_THRESHOLDS:
            for cap in OUTPUT_CAPS:
                predictions = _predictions_from_probabilities(
                    cases,
                    spec,
                    probabilities,
                    threshold=threshold,
                    cap=cap,
                )
                metrics, _rows = score_predictions(cases, predictions)
                gate = training_gate(metrics, baseline)
                trial = {
                    "learner": spec.name,
                    "universe": spec.universe,
                    "representation": spec.representation,
                    "classifier_kind": spec.classifier_kind,
                    "threshold": threshold,
                    "cap": cap,
                    "metrics": metrics,
                    "training_gate": gate,
                    "case_group_fold_audit": fold_audit,
                }
                trials.append(trial)
                rank = (
                    int(gate["all_conjunctive_checks_passed"]),
                    float(metrics["f1"]),
                    int(metrics["true_positive"]),
                    -int(metrics["false_positive"]),
                    int(metrics["exact_case_count"]),
                    -cap,
                    threshold,
                    spec.name,
                )
                if best is None or rank > best[0]:
                    best = (rank, trial)
    if best is None:
        raise RuntimeError("configuration search produced no trials")
    trials.sort(
        key=lambda row: (
            -int(row["training_gate"]["all_conjunctive_checks_passed"]),
            -float(row["metrics"]["f1"]),
            -int(row["metrics"]["true_positive"]),
            int(row["metrics"]["false_positive"]),
            row["learner"],
            row["threshold"],
            row["cap"],
        )
    )
    return dict(best[1]), trials


def nested_outer_evaluation(cases: Sequence[CaseData]) -> dict[str, Any]:
    folds = case_folds(cases, OUTER_FOLDS, seed=f"{FOLD_SEED}|outer")
    predictions: dict[str, list[dict[str, Any]]] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(folds, 1):
        held_ids = {case.case_id for case in held_out}
        train = [case for case in cases if case.case_id not in held_ids]
        selected, trials = select_configuration(
            train,
            n_splits=INNER_FOLDS,
            seed=f"{FOLD_SEED}|outer-{fold_index}|inner",
        )
        spec = next(row for row in LEARNER_SPECS if row.name == selected["learner"])
        classifier = fit_portable_classifier(spec, train)
        held_predictions = {
            case.case_id: predict_with_classifier(
                case,
                spec,
                classifier,
                threshold=float(selected["threshold"]),
                cap=int(selected["cap"]),
            )
            for case in held_out
        }
        predictions.update(held_predictions)
        metrics, _rows = score_predictions(held_out, held_predictions)
        fold_reports.append(
            {
                "outer_fold": fold_index,
                "train_case_count": len(train),
                "held_out_case_count": len(held_out),
                "train_case_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in train))
                ),
                "held_out_case_binding_sha256": sha256_bytes(
                    canonical_bytes(sorted(case.scan_sha256 for case in held_out))
                ),
                "inner_selected_configuration": selected,
                "inner_trial_count": len(trials),
                "outer_held_out_metrics": metrics,
            }
        )
    candidate, rows = score_predictions(cases, predictions)
    baseline, _baseline_rows = score_predictions(cases, baseline_predictions(cases))
    return {
        "method": "nested_complete_case_group_validation",
        "outer_fold_count": OUTER_FOLDS,
        "inner_fold_count": INNER_FOLDS,
        "all_18_cases_scored_exactly_once_outer_held_out": len(rows) == len(cases),
        "candidate_metrics": candidate,
        "v51_reference_metrics": baseline,
        "training_retain_gate": training_gate(candidate, baseline),
        "folds": fold_reports,
    }


def _spec(name: str) -> LearnerSpec:
    return next(row for row in LEARNER_SPECS if row.name == name)


def build_model(
    *,
    lock: LockContext,
    cases: Sequence[CaseData],
    selected: Mapping[str, Any],
    classifier: Mapping[str, Any],
    projection_sha256: str,
    attestation_sha256: str,
    train_cache_manifest_sha256: str,
    train_cache_manifest_attestation_sha256: str,
    train_source_cache_binding_sha256: str,
    runtime_code_sha256: str,
    optimizer_code_sha256: str,
    fit_audit_sha256: str,
) -> dict[str, Any]:
    spec = _spec(str(selected["learner"]))
    prereg_gates = dict(lock.preregistration["retention_gates"])
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "model_name": "F1_AUG30_TRAIN18_FROZEN_SOURCE_ONLY_CANDIDATE_V1",
        "freeze_status": "FROZEN_BEFORE_HOLDOUT_SCORE",
        "holdout_score_opened": False,
        "historical_cohort_consumed": True,
        "retrospective_pseudo_unseen_only": True,
        "prospective_or_unseen_claim_authorized": False,
        "promotion_eligible": False,
        "source_only": True,
        "case_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "count_feature_used": False,
        "representation": spec.representation,
        "universe": spec.universe,
        "feature_names": list(feature_names(spec.representation)),
        "classifier": dict(classifier),
        "profile_name": "F1_AUG30_TRAIN18_FROZEN_V1",
        "profile": _profile(float(selected["threshold"]), int(selected["cap"])),
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "preregistration_sha256": lock.preregistration_sha256,
        "label_projection_sha256": projection_sha256,
        "label_projection_attestation_sha256": attestation_sha256,
        "train_cache_manifest_sha256": train_cache_manifest_sha256,
        "train_cache_manifest_attestation_sha256": train_cache_manifest_attestation_sha256,
        "train_source_cache_binding_sha256": train_source_cache_binding_sha256,
        "optimizer_code_sha256": optimizer_code_sha256,
        "runtime_code_sha256": runtime_code_sha256,
        "fit_audit_sha256": fit_audit_sha256,
        "fit_audit_binding_scope": "fit_audit_core.json",
        "training_case_count": len(cases),
        "training_case_binding_sha256": sha256_bytes(
            canonical_bytes(sorted(case.scan_sha256 for case in cases))
        ),
        "selected_learner": spec.name,
        "selected_classifier_parameters": dict(spec.classifier_parameters),
        "holdout_retention_gates_frozen_verbatim": prereg_gates,
        "post_holdout_mutation_policy": "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT",
        "inference_contract": {
            "runtime_inputs": ["frozen_v51_semantic_result", "scan_only_candidate_pool", "this_model"],
            "labels_opened": False,
            "expected_count_opened": False,
            "cad_opened": False,
            "drawing_opened": False,
            "filename_opened": False,
            "case_identity_used_as_feature": False,
        },
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def run_optimizer(
    *,
    label_attestation: Path,
    label_attestation_sha256: str,
    cache_attestation: Path,
    cache_attestation_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    lock = load_lock_context()
    # The label paths are not resolved until load_lock_context has produced the
    # exact sealed TRAIN membership.
    reader = build_guarded_reader(
        lock,
        label_attestation=label_attestation,
        label_attestation_sha256=label_attestation_sha256,
        cache_attestation=cache_attestation,
        cache_attestation_sha256=cache_attestation_sha256,
    )
    (
        cases,
        projection_sha,
        attestation_sha,
        runtime_code_sha,
        optimizer_code_sha,
        cache_manifest_sha,
        cache_manifest_attestation_sha,
        train_cache_binding_sha,
    ) = load_train_cases(
        lock,
        reader,
        label_attestation=label_attestation.resolve(),
        cache_attestation=cache_attestation.resolve(),
    )
    nested = nested_outer_evaluation(cases)
    selected, trials = select_configuration(
        cases,
        n_splits=FINAL_SELECTION_FOLDS,
        seed=f"{FOLD_SEED}|final-selection",
    )
    spec = _spec(str(selected["learner"]))
    classifier = fit_portable_classifier(spec, cases)
    fit_audit_core = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_nested_fit_audit_v1",
        "holdout_score_opened": False,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "label_projection_sha256": projection_sha,
        "label_projection_attestation_sha256": attestation_sha,
        "train_cache_manifest_sha256": cache_manifest_sha,
        "train_cache_manifest_attestation_sha256": cache_manifest_attestation_sha,
        "train_source_cache_binding_sha256": train_cache_binding_sha,
        "case_grouping": "complete case; no candidate-row split crosses a fold",
        "outer_nested_evaluation": nested,
        "full_train_oof_selection": selected,
        "full_train_trial_count": len(trials),
        "full_train_trials": trials,
        "explored_representation_policy_families": [
            {
                "learner": row.name,
                "universe": row.universe,
                "representation": row.representation,
                "classifier_kind": row.classifier_kind,
            }
            for row in LEARNER_SPECS
        ],
        "forbidden_features": {
            "case_id": False,
            "filename": False,
            "cad": False,
            "count": False,
        },
    }
    fit_audit_sha = sha256_bytes(canonical_bytes(fit_audit_core))
    model = build_model(
        lock=lock,
        cases=cases,
        selected=selected,
        classifier=classifier,
        projection_sha256=projection_sha,
        attestation_sha256=attestation_sha,
        train_cache_manifest_sha256=cache_manifest_sha,
        train_cache_manifest_attestation_sha256=cache_manifest_attestation_sha,
        train_source_cache_binding_sha256=train_cache_binding_sha,
        runtime_code_sha256=runtime_code_sha,
        optimizer_code_sha256=optimizer_code_sha,
        fit_audit_sha256=fit_audit_sha,
    )

    replay_identity: dict[str, bool] = {}
    for case in cases:
        primary = compose_frozen_candidate(case.primary_v51, case.primary_pool, model)
        replay = compose_frozen_candidate(case.replay_v51, case.replay_pool, model)
        replay_identity[case.case_id] = canonical_bytes(primary) == canonical_bytes(replay)
    if not all(replay_identity.values()):
        raise RuntimeError("frozen candidate primary/replay science bytes differ")

    fit_audit = {
        **fit_audit_core,
        "fit_audit_core_sha256": fit_audit_sha,
        "frozen_model_replay_identity": {
            "all_18_exact": True,
            "case_count": len(replay_identity),
            "case_identity_binding_sha256": sha256_bytes(
                canonical_bytes(sorted(lock.scan_sha_by_case.values()))
            ),
        },
        "nested_training_retain_gate_with_replay": {
            **nested["training_retain_gate"],
            "primary_replay_bytes_exact": True,
            "all_conjunctive_checks_passed": bool(
                nested["training_retain_gate"]["all_conjunctive_checks_passed"]
            ),
        },
        "holdout_retention_gates_not_evaluated": True,
    }
    ledger = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_exact_read_ledger_v1",
        "holdout_score_opened": False,
        "exact_path_capability": True,
        "train_allowlist_constructed_before_label_path_resolution": True,
        "train_case_count": 18,
        "train_case_binding_sha256": sha256_bytes(
            canonical_bytes(sorted(lock.scan_sha_by_case.values()))
        ),
        "read_count": len(reader.entries),
        "denied_read_count": len(reader.denied_attempts),
        "denied_attempts": reader.denied_attempts,
        "entries": reader.entries,
        "assertions": {
            "full_28_case_catalog_read": False,
            "complete_label_manifest_read": False,
            "holdout_label_read_count": 0,
            "holdout_prediction_read_count": 0,
            "holdout_score_read_count": 0,
            "all_case_scoped_reads_train_allowlisted": True,
        },
    }

    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("optimizer output directory is not empty")
    model_path = output / "frozen_model.json"
    audit_core_path = output / "fit_audit_core.json"
    audit_path = output / "fit_audit.json"
    ledger_path = output / "train_read_ledger.json"
    _write_new(model_path, canonical_bytes(model))
    _write_new(audit_core_path, canonical_bytes(fit_audit_core))
    _write_new(audit_path, canonical_bytes(fit_audit))
    _write_new(ledger_path, canonical_bytes(ledger))
    model_sha = _sha(model_path)
    audit_core_sha = _sha(audit_core_path)
    audit_sha = _sha(audit_path)
    ledger_sha = _sha(ledger_path)
    if audit_core_sha != fit_audit_sha:
        raise RuntimeError("serialized fit-audit core hash differs from model binding")
    freeze_lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_pre_holdout_freeze_lock_v1",
        "status": "FROZEN_BEFORE_ANY_HOLDOUT_SCORE",
        "holdout_score_opened": False,
        "model_sha256": model_sha,
        "fit_audit_core_sha256": audit_core_sha,
        "fit_audit_sha256": audit_sha,
        "train_read_ledger_sha256": ledger_sha,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "preregistration_sha256": lock.preregistration_sha256,
        "label_projection_sha256": projection_sha,
        "label_projection_attestation_sha256": attestation_sha,
        "train_cache_manifest_sha256": cache_manifest_sha,
        "train_cache_manifest_attestation_sha256": cache_manifest_attestation_sha,
        "train_source_cache_binding_sha256": train_cache_binding_sha,
        "optimizer_code_sha256": optimizer_code_sha,
        "runtime_code_sha256": runtime_code_sha,
        "candidate_count_frozen": 1,
        "holdout_stage_order_frozen": [
            {"tier": "SIMILAR_FAMILY", "case_count": 4},
            {"tier": "ADJACENT_FAMILY", "case_count": 3},
            {"tier": "DISTANT_FAMILY", "case_count": 3},
        ],
        "first_holdout_score_event_is_terminal_for": [
            "model_parameters",
            "feature_representation",
            "policy",
            "threshold",
            "output_cap",
            "tier_assignment",
            "case_membership",
        ],
        "adaptation_after_similar_or_adjacent_results": "FORBIDDEN",
        "historical_cohort_consumed": True,
        "prospective_or_unseen_claim_authorized": False,
        "promotion_eligible": False,
    }
    lock_path = output / "freeze_lock.json"
    _write_new(lock_path, canonical_bytes(freeze_lock))
    lock_sha = _sha(lock_path)
    pre_holdout_seal = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_pre_holdout_seal_v1",
        "status": "SEALED_LOCALLY_AWAITING_INDEPENDENT_EXTERNAL_ANCHOR",
        "holdout_score_opened": False,
        "freeze_lock_sha256": lock_sha,
        "frozen_model_sha256": model_sha,
        "fit_audit_core_sha256": audit_core_sha,
        "fit_audit_sha256": audit_sha,
        "train_read_ledger_sha256": ledger_sha,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "preregistration_sha256": lock.preregistration_sha256,
        "external_anchor_required_before_holdout_score": True,
        "external_anchor_must_be_created_by_independent_protocol_auditor": True,
    }
    seal_path = output / "pre_holdout_seal.json"
    _write_new(seal_path, canonical_bytes(pre_holdout_seal))
    seal_sha = _sha(seal_path)
    _write_new(output / "frozen_model.sha256", f"{model_sha}  frozen_model.json\n".encode())
    _write_new(output / "pre_holdout_seal.sha256", f"{seal_sha}  pre_holdout_seal.json\n".encode())
    return {
        "output_dir": str(output),
        "frozen_model": str(model_path),
        "frozen_model_sha256": model_sha,
        "freeze_lock": str(lock_path),
        "pre_holdout_seal": str(seal_path),
        "pre_holdout_seal_sha256": seal_sha,
        "external_anchor_required_before_holdout_score": True,
        "holdout_score_opened": False,
        "nested_training_candidate_metrics": nested["candidate_metrics"],
        "nested_training_v51_reference_metrics": nested["v51_reference_metrics"],
        "nested_training_retain_gate_passed": nested["training_retain_gate"]["all_conjunctive_checks_passed"],
        "selected_learner": spec.name,
        "selected_threshold": selected["threshold"],
        "selected_cap": selected["cap"],
        "primary_replay_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-label-attestation", type=Path, required=True)
    parser.add_argument("--train-label-attestation-sha256", required=True)
    parser.add_argument("--train-cache-attestation", type=Path, required=True)
    parser.add_argument("--train-cache-attestation-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_optimizer(
        label_attestation=args.train_label_attestation,
        label_attestation_sha256=args.train_label_attestation_sha256,
        cache_attestation=args.train_cache_attestation,
        cache_attestation_sha256=args.train_cache_attestation_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
