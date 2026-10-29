#!/usr/bin/env python3
"""Independent fail-closed audit of the Aug-30 TRAIN-only custody products.

The audit never opens the aggregate development catalog or the full V51
execution manifest.  It opens only the sealed split/allowlist, the scan-only
identity manifest, the isolated TRAIN label bundle, and TRAIN cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
SOURCE_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
V51_SCAN_MANIFEST = ROOT / "data/bend_corpus/incoming/2026-08-30_shaked_daily_scans/pipeline_manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
BUILDER = ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py"
VALIDATOR = Path(__file__).resolve()

SPLIT_SHA = "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
ALLOWLIST_SHA = "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
CATALOG_SHA = "7cbfd0a0da62ab6181d89a796c5ff217c82660dc718696606d2eed4546a4dee0"
SCAN_MANIFEST_SHA = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

LABEL_KIND = "sherman_qc_f1_aug30_train_only_label_projection_v1"
LABEL_ATTEST_KIND = "sherman_qc_f1_aug30_train_only_label_projection_attestation_v1"
CASE_KIND = "sherman_qc_f1_aug30_train_only_case_label_v1"
CACHE_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CACHE_ATTEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
ACCESS_KIND = "sherman_qc_f1_aug30_train_only_access_evidence_v1"
VALIDATION_KIND = "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict) or data != canonical_bytes(payload):
        raise ValueError("custody artifact is not a canonical JSON object")
    return payload, data


def read_source(path: Path, expected: str) -> dict[str, Any]:
    data = path.read_bytes()
    if sha_bytes(data) != expected:
        raise ValueError("sealed source hash mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("sealed source is not an object")
    return payload


def _lock() -> tuple[list[dict[str, str]], set[str], set[str], dict[str, str], str, str]:
    split = read_source(SPLIT, SPLIT_SHA)
    allowlist = read_source(ALLOWLIST, ALLOWLIST_SHA)
    raw_train = allowlist.get("cases")
    assignments = split.get("assignments")
    if not isinstance(raw_train, list) or len(raw_train) != 18 or not isinstance(assignments, list):
        raise ValueError("sealed TRAIN partition schema drift")
    train = [{"case_id": str(row["case_id"]), "scan_sha256": str(row["scan_sha256"])} for row in raw_train]
    train_ids = {row["case_id"] for row in train}
    if len(train_ids) != 18 or [row["case_id"] for row in train] != list(split.get("train_case_ids") or []):
        raise ValueError("sealed TRAIN membership drift")
    assignment_sha = {
        str(row["case_id"]): str(row["scan_sha256"])
        for row in assignments
        if isinstance(row, dict)
    }
    complement_ids = set(assignment_sha) - train_ids
    if len(assignment_sha) != 28 or len(complement_ids) != 10:
        raise ValueError("sealed complement drift")
    complement_hashes = {assignment_sha[value] for value in complement_ids}
    train_commitment = sha_bytes(canonical_bytes(train))
    complement_commitment = sha_bytes(
        canonical_bytes(
            {
                "domain": "SHERMANQC_AUG30_BLINDED_COMPLEMENT_V1",
                "split_sha256": SPLIT_SHA,
                "opaque_case_ids": sorted(complement_ids),
            }
        )
    )
    return train, complement_ids, complement_hashes, assignment_sha, train_commitment, complement_commitment


def _runtime_map() -> dict[str, str]:
    manifest = read_source(V51_SCAN_MANIFEST, SCAN_MANIFEST_SHA)
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError("V51 scan identity manifest drift")
    result = {str(row["scan_sha256"]): f"T{index:03d}" for index, row in enumerate(rows, 1)}
    if len(result) != 28:
        raise ValueError("V51 scan identity manifest duplicate hash")
    return result


def _binding_ok(
    binding: Any,
    path: Path,
    *,
    kind: str | None = None,
    published_path: Path | None = None,
) -> bool:
    if not isinstance(binding, dict):
        return False
    expected_path = (published_path or path).resolve().relative_to(ROOT).as_posix()
    return (
        binding.get("path") == expected_path
        and binding.get("sha256") == sha_file(path)
        and binding.get("size_bytes") == path.stat().st_size
        and (kind is None or binding.get("kind") == kind)
    )


def _attestation_is_value_free(payload: dict[str, Any]) -> bool:
    forbidden = {"cases", "bends", "records", "predictions", "labels", "manual_bends"}
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if forbidden.intersection(current):
                return False
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return True


def _expected_source_binding(path: Path, sha256: str, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _valid_bends(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    ids: set[str] = set()
    for bend in value:
        if not isinstance(bend, dict) or set(bend) != {"bend_id", "line_start", "line_end", "reviewer_support_count"}:
            return False
        bend_id = str(bend["bend_id"])
        start, end = bend["line_start"], bend["line_end"]
        support = bend["reviewer_support_count"]
        if (
            not re.fullmatch(r"B\d{3}", bend_id)
            or bend_id in ids
            or not isinstance(start, list)
            or not isinstance(end, list)
            or len(start) != 3
            or len(end) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in start + end)
            or start == end
            or isinstance(support, bool)
            or support not in (1, 2, 3)
        ):
            return False
        ids.add(bend_id)
    return True


def audit_custody(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    published_root: Path | None = None,
    provisional: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    published_root = (published_root or output_root).resolve()

    def published(path: Path) -> Path:
        return published_root / path.resolve().relative_to(output_root)

    train, complement_ids, complement_hashes, _assignment_sha, train_commitment, complement_commitment = _lock()
    train_ids = [row["case_id"] for row in train]
    scan_by_train = {row["case_id"]: row["scan_sha256"] for row in train}
    runtime_by_scan = _runtime_map()
    runtime_complement = {runtime_by_scan[value] for value in complement_hashes}

    label_root = output_root / "labels"
    cache_root = output_root / "cache"
    projection_path = label_root / "train_label_projection.json"
    label_attest_path = label_root / "train_label_projection_attestation.json"
    label_access_path = label_root / "access_evidence.json"
    cache_manifest_path = cache_root / "train_v51_cache_manifest.json"
    cache_attest_path = cache_root / "train_v51_cache_manifest_attestation.json"
    cache_access_path = cache_root / "access_evidence.json"
    validation_path = output_root / "independent_validation_report.json"
    case_paths = [label_root / "cases" / f"{case_id}.json" for case_id in train_ids]
    allowed_files = {
        projection_path,
        label_attest_path,
        label_access_path,
        cache_manifest_path,
        cache_attest_path,
        cache_access_path,
        *case_paths,
    }
    if not provisional:
        allowed_files.add(validation_path)
    actual_files = {path for path in output_root.rglob("*") if path.is_file()}
    if actual_files != allowed_files or any(path.is_symlink() for path in output_root.rglob("*")):
        raise ValueError("custody bundle file inventory is not exact")

    projection, projection_bytes = read_canonical(projection_path)
    label_attest, label_attest_bytes = read_canonical(label_attest_path)
    label_access, _ = read_canonical(label_access_path)
    cache_manifest, cache_manifest_bytes = read_canonical(cache_manifest_path)
    cache_attest, cache_attest_bytes = read_canonical(cache_attest_path)
    cache_access, _ = read_canonical(cache_access_path)
    validation_report = None
    validation_bytes = None
    if not provisional:
        validation_report, validation_bytes = read_canonical(validation_path)

    if not _attestation_is_value_free(label_attest) or not _attestation_is_value_free(cache_attest):
        raise ValueError("custody attestation contains label/cache values")
    label_attest_keys = {
        "schema_version", "kind", "canonical_json", "split_sha256",
        "train_allowlist_sha256", "source_catalog_sha256",
        "train_set_commitment_sha256", "blinded_complement_commitment_sha256",
        "case_count", "file_inventory_count", "train_exact_match",
        "train_holdout_intersection_count", "aggregate_catalog_read_count",
        "holdout_case_file_emit_count", "holdout_label_row_emit_count",
        "optimizer_holdout_access_count", "projection_contains_only_train",
        "independent_validation_passed", "projection_binding", "builder_binding",
        "validator_binding", "access_evidence_binding",
    }
    cache_attest_keys = {
        "schema_version", "kind", "canonical_json", "split_sha256",
        "train_allowlist_sha256", "train_set_commitment_sha256",
        "blinded_complement_commitment_sha256", "case_count", "record_count",
        "train_exact_match", "train_holdout_intersection_count",
        "holdout_entry_count", "holdout_status_count", "holdout_hash_count",
        "optimizer_holdout_access_count", "independent_validation_passed",
        "manifest_binding", "builder_binding", "validator_binding",
        "access_evidence_binding",
    }
    if not provisional:
        label_attest_keys.add("staged_validator_pass_binding")
        cache_attest_keys.add("staged_validator_pass_binding")
    if set(label_attest) != label_attest_keys or set(cache_attest) != cache_attest_keys:
        raise ValueError("custody attestation exact-key schema drift")
    if projection.get("kind") != LABEL_KIND or projection.get("schema_version") != 1:
        raise ValueError("TRAIN label projection kind drift")
    if projection.get("split_sha256") != SPLIT_SHA or projection.get("train_allowlist_sha256") != ALLOWLIST_SHA:
        raise ValueError("TRAIN label projection lock binding drift")
    if projection.get("source_catalog_sha256") != CATALOG_SHA or projection.get("train_set_commitment_sha256") != train_commitment:
        raise ValueError("TRAIN label projection source/commitment drift")
    label_rows = projection.get("cases")
    if not isinstance(label_rows, list) or len(label_rows) != 18:
        raise ValueError("TRAIN label projection cardinality drift")
    if [str(row.get("case_id") or "") for row in label_rows if isinstance(row, dict)] != train_ids:
        raise ValueError("TRAIN label projection membership/order drift")
    for row, path in zip(label_rows, case_paths):
        if not isinstance(row, dict) or set(row) != {"case_id", "scan_sha256", "bends"}:
            raise ValueError("TRAIN label projection row schema drift")
        case_id = str(row["case_id"])
        if row["scan_sha256"] != scan_by_train[case_id] or not _valid_bends(row["bends"]):
            raise ValueError("TRAIN label projection case validation failed")
        case_payload, _ = read_canonical(path)
        if case_payload != {"schema_version": 1, "kind": CASE_KIND, **row}:
            raise ValueError("canonical per-case TRAIN label differs from projection")

    label_checks = {
        "schema": label_attest.get("schema_version") == 1,
        "kind": label_attest.get("kind") == LABEL_ATTEST_KIND,
        "canonical_json": label_attest.get("canonical_json") is True,
        "split": label_attest.get("split_sha256") == SPLIT_SHA,
        "allowlist": label_attest.get("train_allowlist_sha256") == ALLOWLIST_SHA,
        "catalog": label_attest.get("source_catalog_sha256") == CATALOG_SHA,
        "train_commitment": label_attest.get("train_set_commitment_sha256") == train_commitment,
        "complement_commitment": label_attest.get("blinded_complement_commitment_sha256") == complement_commitment,
        "case_count": label_attest.get("case_count") == 18,
        "file_count": label_attest.get("file_inventory_count") == 18,
        "exact": label_attest.get("train_exact_match") is True,
        "intersection": label_attest.get("train_holdout_intersection_count") == 0,
        "aggregate_catalog_read": label_attest.get("aggregate_catalog_read_count") == 1,
        "holdout_case_file_emit": label_attest.get("holdout_case_file_emit_count") == 0,
        "holdout_emit": label_attest.get("holdout_label_row_emit_count") == 0,
        "optimizer_holdout_access": label_attest.get("optimizer_holdout_access_count") == 0,
        "only_train": label_attest.get("projection_contains_only_train") is True,
        "validated": label_attest.get("independent_validation_passed") is (not provisional),
        "projection_binding": _binding_ok(label_attest.get("projection_binding"), projection_path, kind=LABEL_KIND, published_path=published(projection_path)),
        "builder_binding": _binding_ok(label_attest.get("builder_binding"), BUILDER),
        "validator_binding": _binding_ok(label_attest.get("validator_binding"), VALIDATOR),
        "access_binding": _binding_ok(label_attest.get("access_evidence_binding"), label_access_path, published_path=published(label_access_path)),
    }
    if not all(label_checks.values()):
        raise ValueError("TRAIN label attestation validation failed")
    if (
        set(label_access) != {
            "schema_version", "kind", "scope", "canonical_json", "source_reads",
            "case_file_inventory", "aggregate_catalog_read_count",
            "holdout_case_file_emit_count", "holdout_label_row_emit_count",
            "denied_open_attempt_count", "label_values_written_to_log",
        }
        or label_access.get("schema_version") != 1
        or label_access.get("kind") != ACCESS_KIND
        or label_access.get("scope") != "TRAIN_LABEL_PROJECTION_ONLY"
        or label_access.get("canonical_json") is not True
        or label_access.get("aggregate_catalog_read_count") != 1
        or label_access.get("holdout_case_file_emit_count") != 0
        or label_access.get("holdout_label_row_emit_count") != 0
        or label_access.get("denied_open_attempt_count") != 0
        or label_access.get("label_values_written_to_log") is not False
    ):
        raise ValueError("TRAIN label access evidence drift")
    expected_label_sources = [
        _expected_source_binding(SPLIT, SPLIT_SHA, "SEALED_SPLIT_MEMBERSHIP"),
        _expected_source_binding(ALLOWLIST, ALLOWLIST_SHA, "SEALED_TRAIN_ALLOWLIST"),
        _expected_source_binding(SOURCE_CATALOG, CATALOG_SHA, "AGGREGATE_CATALOG_FILTERED_BY_TRAIN_SCAN_HASH"),
    ]
    if label_access.get("source_reads") != expected_label_sources:
        raise ValueError("TRAIN label access source-read binding drift")
    inventory = label_access.get("case_file_inventory")
    if not isinstance(inventory, list) or len(inventory) != 18:
        raise ValueError("TRAIN label file inventory drift")
    for row, path, identity in zip(inventory, case_paths, train):
        if not isinstance(row, dict) or row != {
            "case_id": identity["case_id"],
            "scan_sha256": identity["scan_sha256"],
            "label_path": path.relative_to(output_root).as_posix(),
            "label_sha256": sha_file(path),
            "size_bytes": path.stat().st_size,
        }:
            raise ValueError("TRAIN label file binding drift")

    if cache_manifest.get("kind") != CACHE_KIND or cache_manifest.get("schema_version") != 1:
        raise ValueError("TRAIN cache manifest kind drift")
    if cache_manifest.get("source_only") is not True or cache_manifest.get("holdout_entry_count") != 0:
        raise ValueError("TRAIN cache source-only/empty-holdout contract drift")
    if cache_manifest.get("split_sha256") != SPLIT_SHA or cache_manifest.get("train_allowlist_sha256") != ALLOWLIST_SHA:
        raise ValueError("TRAIN cache lock binding drift")
    if cache_manifest.get("train_set_commitment_sha256") != train_commitment or cache_manifest.get("v51_scan_only_manifest_sha256") != SCAN_MANIFEST_SHA:
        raise ValueError("TRAIN cache commitment/source-manifest drift")
    records = cache_manifest.get("records")
    if cache_manifest.get("case_count") != 18 or cache_manifest.get("record_count") != 36 or not isinstance(records, list) or len(records) != 36:
        raise ValueError("TRAIN cache cardinality drift")
    record_keys = {"case_id", "runtime_case_id", "pass", "scan_sha256", "semantic_path", "semantic_sha256", "pool_path", "pool_sha256"}
    expected_order = [(row["case_id"], value) for row in train for value in ("primary", "replay")]
    if [(str(row.get("case_id") or ""), str(row.get("pass") or "")) for row in records if isinstance(row, dict)] != expected_order:
        raise ValueError("TRAIN cache record membership/order drift")
    cache_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != record_keys:
            raise ValueError("TRAIN cache record field drift")
        case_id = str(row["case_id"])
        pass_name = str(row["pass"])
        runtime = runtime_by_scan[scan_by_train[case_id]]
        if row["runtime_case_id"] != runtime or row["scan_sha256"] != scan_by_train[case_id]:
            raise ValueError("TRAIN cache scan/runtime cross-map drift")
        semantic = ROOT / str(row["semantic_path"])
        pool = ROOT / str(row["pool_path"])
        expected_semantic = ROOT / f"output/research/f1-v51-today-scan-only-development-20260830-r2/{pass_name}/{runtime}/semantic_result.json"
        expected_pool = ROOT / f"output/research/f1-v51-today-scan-only-development-20260830-r2/{pass_name}/{runtime}/pool/{runtime}.json"
        if semantic.resolve() != expected_semantic.resolve() or pool.resolve() != expected_pool.resolve():
            raise ValueError("TRAIN cache artifact path drift")
        if not semantic.is_file() or not pool.is_file() or row["semantic_sha256"] != sha_file(semantic) or row["pool_sha256"] != sha_file(pool):
            raise ValueError("TRAIN cache artifact hash drift")
        cache_map[(case_id, pass_name)] = row
    for identity in train:
        primary = cache_map[(identity["case_id"], "primary")]
        replay = cache_map[(identity["case_id"], "replay")]
        if primary["semantic_sha256"] != replay["semantic_sha256"] or primary["pool_sha256"] != replay["pool_sha256"]:
            raise ValueError("TRAIN cache replay identity drift")

    cache_checks = {
        "schema": cache_attest.get("schema_version") == 1,
        "kind": cache_attest.get("kind") == CACHE_ATTEST_KIND,
        "canonical_json": cache_attest.get("canonical_json") is True,
        "split": cache_attest.get("split_sha256") == SPLIT_SHA,
        "allowlist": cache_attest.get("train_allowlist_sha256") == ALLOWLIST_SHA,
        "train_commitment": cache_attest.get("train_set_commitment_sha256") == train_commitment,
        "complement_commitment": cache_attest.get("blinded_complement_commitment_sha256") == complement_commitment,
        "case_count": cache_attest.get("case_count") == 18,
        "record_count": cache_attest.get("record_count") == 36,
        "exact": cache_attest.get("train_exact_match") is True,
        "intersection": cache_attest.get("train_holdout_intersection_count") == 0,
        "holdout_entries": cache_attest.get("holdout_entry_count") == 0,
        "holdout_status": cache_attest.get("holdout_status_count") == 0,
        "holdout_hash": cache_attest.get("holdout_hash_count") == 0,
        "optimizer_holdout_access": cache_attest.get("optimizer_holdout_access_count") == 0,
        "validated": cache_attest.get("independent_validation_passed") is (not provisional),
        "manifest_binding": _binding_ok(cache_attest.get("manifest_binding"), cache_manifest_path, kind=CACHE_KIND, published_path=published(cache_manifest_path)),
        "builder_binding": _binding_ok(cache_attest.get("builder_binding"), BUILDER),
        "validator_binding": _binding_ok(cache_attest.get("validator_binding"), VALIDATOR),
        "access_binding": _binding_ok(cache_attest.get("access_evidence_binding"), cache_access_path, published_path=published(cache_access_path)),
    }
    if not all(cache_checks.values()):
        raise ValueError("TRAIN cache attestation validation failed")
    if (
        set(cache_access) != {
            "schema_version", "kind", "scope", "canonical_json", "source_reads",
            "authorized_train_artifacts", "full_execution_manifest_opened",
            "holdout_entry_emit_count", "holdout_status_emit_count",
            "holdout_hash_emit_count", "denied_open_attempt_count",
        }
        or cache_access.get("schema_version") != 1
        or cache_access.get("kind") != ACCESS_KIND
        or cache_access.get("scope") != "TRAIN_V51_CACHE_PROJECTION_ONLY"
        or cache_access.get("canonical_json") is not True
        or cache_access.get("full_execution_manifest_opened") is not False
    ):
        raise ValueError("TRAIN cache access evidence drift")
    if any(cache_access.get(name) != 0 for name in ("holdout_entry_emit_count", "holdout_status_emit_count", "holdout_hash_emit_count", "denied_open_attempt_count")):
        raise ValueError("TRAIN cache access evidence reports forbidden activity")
    if cache_access.get("authorized_train_artifacts") != records:
        raise ValueError("TRAIN cache access inventory differs from manifest")
    expected_cache_sources = [
        _expected_source_binding(SPLIT, SPLIT_SHA, "SEALED_SPLIT_MEMBERSHIP"),
        _expected_source_binding(ALLOWLIST, ALLOWLIST_SHA, "SEALED_TRAIN_ALLOWLIST"),
        _expected_source_binding(V51_SCAN_MANIFEST, SCAN_MANIFEST_SHA, "V51_SCAN_HASH_TO_RUNTIME_IDENTITY"),
    ]
    if cache_access.get("source_reads") != expected_cache_sources:
        raise ValueError("TRAIN cache access source-read binding drift")

    if provisional:
        if "staged_validator_pass_binding" in label_attest or "staged_validator_pass_binding" in cache_attest:
            raise ValueError("provisional attestations must not pre-bind validator PASS")
    else:
        if validation_report is None or validation_bytes is None:
            raise ValueError("independent validation report is missing")
        if (
            set(validation_report) != {
                "schema_version", "kind", "status", "canonical_json", "checks",
                "projection_binding", "cache_manifest_binding",
                "label_access_evidence_binding", "cache_access_evidence_binding",
                "builder_binding", "validator_binding",
            }
            or validation_report.get("schema_version") != 1
            or validation_report.get("kind") != VALIDATION_KIND
            or validation_report.get("status") != "PASS"
        ):
            raise ValueError("independent validation report did not PASS")
        if validation_report.get("canonical_json") is not True:
            raise ValueError("independent validation report canonical flag drift")
        report_binding = {
            "path": published(validation_path).relative_to(ROOT).as_posix(),
            "sha256": sha_bytes(validation_bytes),
            "size_bytes": len(validation_bytes),
            "kind": VALIDATION_KIND,
        }
        if label_attest.get("staged_validator_pass_binding") != report_binding or cache_attest.get("staged_validator_pass_binding") != report_binding:
            raise ValueError("final attestations do not bind staged validator PASS")
        expected_report_checks = {
            "train_exact_match": True,
            "train_holdout_intersection_count": 0,
            "label_case_count": 18,
            "label_file_count": 18,
            "cache_record_count": 36,
            "holdout_label_row_emit_count": 0,
            "holdout_case_file_emit_count": 0,
            "holdout_cache_entry_emit_count": 0,
            "holdout_cache_status_emit_count": 0,
            "holdout_cache_hash_emit_count": 0,
            "aggregate_catalog_read_count": 1,
            "optimizer_holdout_access_count": 0,
            "full_execution_manifest_opened": False,
        }
        if validation_report.get("checks") != expected_report_checks:
            raise ValueError("independent validation report checks drift")
        for name, path, kind in (
            ("projection_binding", projection_path, LABEL_KIND),
            ("cache_manifest_binding", cache_manifest_path, CACHE_KIND),
            ("label_access_evidence_binding", label_access_path, None),
            ("cache_access_evidence_binding", cache_access_path, None),
        ):
            if not _binding_ok(validation_report.get(name), path, kind=kind, published_path=published(path)):
                raise ValueError("independent validation report artifact binding drift")
        if not _binding_ok(validation_report.get("builder_binding"), BUILDER) or not _binding_ok(validation_report.get("validator_binding"), VALIDATOR):
            raise ValueError("independent validation report code binding drift")

    published = b"\n".join(path.read_bytes() for path in sorted(actual_files))
    forbidden_tokens = {
        *complement_ids,
        *complement_hashes,
        *runtime_complement,
    }
    if any(token.encode() in published for token in forbidden_tokens):
        raise ValueError("non-TRAIN identifier/hash leaked into custody bundle")

    return {
        "status": "PASS",
        "projection_path": projection_path.relative_to(ROOT).as_posix(),
        "projection_sha256": sha_bytes(projection_bytes),
        "projection_size_bytes": len(projection_bytes),
        "label_attestation_path": label_attest_path.relative_to(ROOT).as_posix(),
        "label_attestation_sha256": sha_bytes(label_attest_bytes),
        "cache_manifest_path": cache_manifest_path.relative_to(ROOT).as_posix(),
        "cache_manifest_sha256": sha_bytes(cache_manifest_bytes),
        "cache_manifest_size_bytes": len(cache_manifest_bytes),
        "cache_attestation_path": cache_attest_path.relative_to(ROOT).as_posix(),
        "cache_attestation_sha256": sha_bytes(cache_attest_bytes),
        "case_count": 18,
        "label_file_count": 18,
        "cache_record_count": 36,
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "holdout_rows_emitted": 0,
    }


def _published_binding(actual: Path, staging_root: Path, published_root: Path, *, kind: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": (published_root / actual.relative_to(staging_root)).relative_to(ROOT).as_posix(),
        "sha256": sha_file(actual),
        "size_bytes": actual.stat().st_size,
    }
    if kind is not None:
        result["kind"] = kind
    return result


def seal_staged_custody(staging_root: Path, published_root: Path) -> dict[str, Any]:
    """Validate hidden staging bytes, author PASS, then finalize attestations."""

    staging_root = staging_root.resolve()
    published_root = published_root.resolve()
    audit_custody(staging_root, published_root=published_root, provisional=True)
    projection = staging_root / "labels/train_label_projection.json"
    label_access = staging_root / "labels/access_evidence.json"
    cache_manifest = staging_root / "cache/train_v51_cache_manifest.json"
    cache_access = staging_root / "cache/access_evidence.json"
    report_path = staging_root / "independent_validation_report.json"
    report = {
        "schema_version": 1,
        "kind": VALIDATION_KIND,
        "status": "PASS",
        "canonical_json": True,
        "checks": {
            "train_exact_match": True,
            "train_holdout_intersection_count": 0,
            "label_case_count": 18,
            "label_file_count": 18,
            "cache_record_count": 36,
            "holdout_label_row_emit_count": 0,
            "holdout_case_file_emit_count": 0,
            "holdout_cache_entry_emit_count": 0,
            "holdout_cache_status_emit_count": 0,
            "holdout_cache_hash_emit_count": 0,
            "aggregate_catalog_read_count": 1,
            "optimizer_holdout_access_count": 0,
            "full_execution_manifest_opened": False,
        },
        "projection_binding": _published_binding(projection, staging_root, published_root, kind=LABEL_KIND),
        "cache_manifest_binding": _published_binding(cache_manifest, staging_root, published_root, kind=CACHE_KIND),
        "label_access_evidence_binding": _published_binding(label_access, staging_root, published_root),
        "cache_access_evidence_binding": _published_binding(cache_access, staging_root, published_root),
        "builder_binding": {
            "path": BUILDER.relative_to(ROOT).as_posix(),
            "sha256": sha_file(BUILDER),
            "size_bytes": BUILDER.stat().st_size,
        },
        "validator_binding": {
            "path": VALIDATOR.relative_to(ROOT).as_posix(),
            "sha256": sha_file(VALIDATOR),
            "size_bytes": VALIDATOR.stat().st_size,
        },
    }
    report_path.write_bytes(canonical_bytes(report))
    report_binding = _published_binding(report_path, staging_root, published_root, kind=VALIDATION_KIND)
    for relative in (
        "labels/train_label_projection_attestation.json",
        "cache/train_v51_cache_manifest_attestation.json",
    ):
        path = staging_root / relative
        payload, _ = read_canonical(path)
        payload["independent_validation_passed"] = True
        payload["staged_validator_pass_binding"] = report_binding
        path.write_bytes(canonical_bytes(payload))
    return audit_custody(staging_root, published_root=published_root, provisional=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit_custody(args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
