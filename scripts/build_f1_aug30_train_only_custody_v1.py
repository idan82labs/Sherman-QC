#!/usr/bin/env python3
"""Materialize the sealed Aug-30 TRAIN-only label and V51 cache projections.

This custodian is the only component allowed to read the aggregate development
catalog.  It emits no non-TRAIN identity, label, prediction, status, or hash
row.  The optimizer is bootstrapped from the two value-free attestations only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
SOURCE_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
V51_SCAN_MANIFEST = ROOT / "data/bend_corpus/incoming/2026-08-30_shaked_daily_scans/pipeline_manifest.json"
V51_CACHE_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
DEFAULT_OUTPUT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
BUILDER = Path(__file__).resolve()
VALIDATOR = ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py"

EXPECTED_SPLIT_SHA256 = "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
EXPECTED_ALLOWLIST_SHA256 = "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
EXPECTED_SOURCE_CATALOG_SHA256 = "7cbfd0a0da62ab6181d89a796c5ff217c82660dc718696606d2eed4546a4dee0"
EXPECTED_V51_SCAN_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

LABEL_PROJECTION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_v1"
LABEL_ATTESTATION_KIND = "sherman_qc_f1_aug30_train_only_label_projection_attestation_v1"
LABEL_CASE_KIND = "sherman_qc_f1_aug30_train_only_case_label_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
ACCESS_KIND = "sherman_qc_f1_aug30_train_only_access_evidence_v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if kind is not None:
        row["kind"] = kind
    return row


def read_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise ValueError("sealed source hash mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("sealed source must be a JSON object")
    return payload


def write_canonical(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))


def _train_lock() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], set[str], str, str]:
    split = read_json(SPLIT, EXPECTED_SPLIT_SHA256)
    allowlist = read_json(ALLOWLIST, EXPECTED_ALLOWLIST_SHA256)
    rows = allowlist.get("cases")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("TRAIN allowlist cardinality drift")
    train_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != {"case_id", "scan_sha256"}:
            raise ValueError("TRAIN allowlist row schema drift")
        case_id = str(raw["case_id"])
        scan_sha = str(raw["scan_sha256"])
        if not re.fullmatch(r"A30-\d{3}", case_id) or len(scan_sha) != 64:
            raise ValueError("TRAIN allowlist identity format drift")
        if case_id in seen_ids or scan_sha in seen_hashes:
            raise ValueError("TRAIN allowlist contains duplicates")
        seen_ids.add(case_id)
        seen_hashes.add(scan_sha)
        train_rows.append({"case_id": case_id, "scan_sha256": scan_sha})
    if [str(value) for value in split.get("train_case_ids") or []] != [row["case_id"] for row in train_rows]:
        raise ValueError("split and TRAIN allowlist disagree")
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("split assignment schema drift")
    all_ids = {str(row.get("case_id") or "") for row in assignments if isinstance(row, dict)}
    if len(all_ids) != 28 or not seen_ids < all_ids:
        raise ValueError("split partition drift")
    complement = all_ids - seen_ids
    if len(complement) != 10:
        raise ValueError("split complement cardinality drift")
    train_commitment = sha256_bytes(canonical_bytes(train_rows))
    complement_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "SHERMANQC_AUG30_BLINDED_COMPLEMENT_V1",
                "split_sha256": EXPECTED_SPLIT_SHA256,
                "opaque_case_ids": sorted(complement),
            }
        )
    )
    return split, allowlist, train_rows, complement, train_commitment, complement_commitment


def _clean_bends(source_case: dict[str, Any]) -> list[dict[str, Any]]:
    bends = source_case.get("bends")
    if not isinstance(bends, list):
        raise ValueError("selected TRAIN source label schema drift")
    clean: list[dict[str, Any]] = []
    for index, bend in enumerate(bends, 1):
        if not isinstance(bend, dict):
            raise ValueError("selected TRAIN bend schema drift")
        start = bend.get("line_start")
        end = bend.get("line_end")
        support = bend.get("reviewer_support_count")
        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 3
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in start + end)
            and start != end
            and isinstance(support, int)
            and not isinstance(support, bool)
            and support in (1, 2, 3)
        ):
            raise ValueError("selected TRAIN bend value schema drift")
        clean.append(
            {
                "bend_id": f"B{index:03d}",
                "line_start": [float(value) for value in start],
                "line_end": [float(value) for value in end],
                "reviewer_support_count": support,
            }
        )
    return clean


def _label_rows(train_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = read_json(SOURCE_CATALOG, EXPECTED_SOURCE_CATALOG_SHA256)
    if catalog.get("kind") != "sherman_qc_bend_line_development_catalog":
        raise ValueError("development catalog kind drift")
    wanted = {row["scan_sha256"]: row["case_id"] for row in train_rows}
    selected: dict[str, dict[str, Any]] = {}
    cases = catalog.get("cases")
    if not isinstance(cases, list):
        raise ValueError("development catalog case schema drift")
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("development catalog case schema drift")
        scan_sha = str(raw.get("scan_sha256") or "")
        opaque_id = wanted.get(scan_sha)
        if opaque_id is not None:
            if opaque_id in selected:
                raise ValueError("duplicate TRAIN scan in development catalog")
            selected[opaque_id] = raw
    if set(selected) != {row["case_id"] for row in train_rows}:
        raise ValueError("development catalog does not exactly cover TRAIN")
    output: list[dict[str, Any]] = []
    for identity in train_rows:
        source_case = selected[identity["case_id"]]
        output.append(
            {
                "case_id": identity["case_id"],
                "scan_sha256": identity["scan_sha256"],
                "bends": _clean_bends(source_case),
            }
        )
    return output, catalog


def _runtime_by_scan() -> dict[str, str]:
    manifest = read_json(V51_SCAN_MANIFEST, EXPECTED_V51_SCAN_MANIFEST_SHA256)
    if manifest.get("kind") != "sherman_qc_f1_v51_opaque_scan_only_development_input":
        raise ValueError("V51 scan-only manifest kind drift")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError("V51 scan-only manifest cardinality drift")
    result: dict[str, str] = {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict) or set(raw) != {"scan_path", "scan_sha256"}:
            raise ValueError("V51 scan-only manifest row schema drift")
        scan_sha = str(raw["scan_sha256"])
        if scan_sha in result:
            raise ValueError("V51 scan-only manifest duplicate scan hash")
        result[scan_sha] = f"T{index:03d}"
    return result


def _cache_records(train_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    runtime_by_scan = _runtime_by_scan()
    records: list[dict[str, str]] = []
    for identity in train_rows:
        runtime = runtime_by_scan.get(identity["scan_sha256"])
        if runtime is None:
            raise ValueError("TRAIN scan is absent from the correct V51 scan-only manifest")
        for pass_name in ("primary", "replay"):
            semantic = V51_CACHE_ROOT / pass_name / runtime / "semantic_result.json"
            pool = V51_CACHE_ROOT / pass_name / runtime / "pool" / f"{runtime}.json"
            if not semantic.is_file() or not pool.is_file():
                raise FileNotFoundError("selected TRAIN cache artifact is missing")
            records.append(
                {
                    "case_id": identity["case_id"],
                    "runtime_case_id": runtime,
                    "pass": pass_name,
                    "scan_sha256": identity["scan_sha256"],
                    "semantic_path": semantic.resolve().relative_to(ROOT).as_posix(),
                    "semantic_sha256": sha256_file(semantic),
                    "pool_path": pool.resolve().relative_to(ROOT).as_posix(),
                    "pool_sha256": sha256_file(pool),
                }
            )
    by_key = {(row["case_id"], row["pass"]): row for row in records}
    for identity in train_rows:
        primary = by_key[(identity["case_id"], "primary")]
        replay = by_key[(identity["case_id"], "replay")]
        if primary["semantic_sha256"] != replay["semantic_sha256"] or primary["pool_sha256"] != replay["pool_sha256"]:
            raise ValueError("selected TRAIN V51 primary/replay bytes differ")
    return records


def _source_binding(path: Path, sha256: str, role: str) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256, "size_bytes": path.stat().st_size, "role": role}


def build(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("custody output already exists")
    if not VALIDATOR.is_file():
        raise FileNotFoundError("independent validator is missing")
    split, _allowlist, train_rows, _complement, train_commitment, complement_commitment = _train_lock()
    labels, _catalog = _label_rows(train_rows)
    cache_records = _cache_records(train_rows)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        label_dir = stage / "labels"
        case_dir = label_dir / "cases"
        cache_dir = stage / "cache"
        case_inventory: list[dict[str, Any]] = []
        for row in labels:
            path = case_dir / f"{row['case_id']}.json"
            payload = {"schema_version": 1, "kind": LABEL_CASE_KIND, **row}
            write_canonical(path, payload)
            case_inventory.append(
                {
                    "case_id": row["case_id"],
                    "scan_sha256": row["scan_sha256"],
                    "label_path": path.relative_to(stage).as_posix(),
                    "label_sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        projection = {
            "schema_version": 1,
            "kind": LABEL_PROJECTION_KIND,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "train_allowlist_sha256": EXPECTED_ALLOWLIST_SHA256,
            "source_catalog_sha256": EXPECTED_SOURCE_CATALOG_SHA256,
            "train_set_commitment_sha256": train_commitment,
            "case_count": 18,
            "cases": labels,
        }
        projection_path = label_dir / "train_label_projection.json"
        write_canonical(projection_path, projection)

        label_access = {
            "schema_version": 1,
            "kind": ACCESS_KIND,
            "scope": "TRAIN_LABEL_PROJECTION_ONLY",
            "canonical_json": True,
            "source_reads": [
                _source_binding(SPLIT, EXPECTED_SPLIT_SHA256, "SEALED_SPLIT_MEMBERSHIP"),
                _source_binding(ALLOWLIST, EXPECTED_ALLOWLIST_SHA256, "SEALED_TRAIN_ALLOWLIST"),
                _source_binding(SOURCE_CATALOG, EXPECTED_SOURCE_CATALOG_SHA256, "AGGREGATE_CATALOG_FILTERED_BY_TRAIN_SCAN_HASH"),
            ],
            "case_file_inventory": case_inventory,
            "aggregate_catalog_read_count": 1,
            "holdout_case_file_emit_count": 0,
            "holdout_label_row_emit_count": 0,
            "denied_open_attempt_count": 0,
            "label_values_written_to_log": False,
        }
        label_access_path = label_dir / "access_evidence.json"
        write_canonical(label_access_path, label_access)

        cache_manifest = {
            "schema_version": 1,
            "kind": CACHE_MANIFEST_KIND,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "train_allowlist_sha256": EXPECTED_ALLOWLIST_SHA256,
            "train_set_commitment_sha256": train_commitment,
            "v51_scan_only_manifest_sha256": EXPECTED_V51_SCAN_MANIFEST_SHA256,
            "source_only": True,
            "case_count": 18,
            "record_count": 36,
            "holdout_entry_count": 0,
            "records": cache_records,
        }
        cache_manifest_path = cache_dir / "train_v51_cache_manifest.json"
        write_canonical(cache_manifest_path, cache_manifest)
        cache_access = {
            "schema_version": 1,
            "kind": ACCESS_KIND,
            "scope": "TRAIN_V51_CACHE_PROJECTION_ONLY",
            "canonical_json": True,
            "source_reads": [
                _source_binding(SPLIT, EXPECTED_SPLIT_SHA256, "SEALED_SPLIT_MEMBERSHIP"),
                _source_binding(ALLOWLIST, EXPECTED_ALLOWLIST_SHA256, "SEALED_TRAIN_ALLOWLIST"),
                _source_binding(V51_SCAN_MANIFEST, EXPECTED_V51_SCAN_MANIFEST_SHA256, "V51_SCAN_HASH_TO_RUNTIME_IDENTITY"),
            ],
            "authorized_train_artifacts": cache_records,
            "full_execution_manifest_opened": False,
            "holdout_entry_emit_count": 0,
            "holdout_status_emit_count": 0,
            "holdout_hash_emit_count": 0,
            "denied_open_attempt_count": 0,
        }
        cache_access_path = cache_dir / "access_evidence.json"
        write_canonical(cache_access_path, cache_access)

        builder_binding = binding(BUILDER)
        validator_binding = binding(VALIDATOR)
        label_attestation = {
            "schema_version": 1,
            "kind": LABEL_ATTESTATION_KIND,
            "canonical_json": True,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "train_allowlist_sha256": EXPECTED_ALLOWLIST_SHA256,
            "source_catalog_sha256": EXPECTED_SOURCE_CATALOG_SHA256,
            "train_set_commitment_sha256": train_commitment,
            "blinded_complement_commitment_sha256": complement_commitment,
            "case_count": 18,
            "file_inventory_count": 18,
            "train_exact_match": True,
            "train_holdout_intersection_count": 0,
            "aggregate_catalog_read_count": 1,
            "holdout_case_file_emit_count": 0,
            "holdout_label_row_emit_count": 0,
            "optimizer_holdout_access_count": 0,
            "projection_contains_only_train": True,
            "independent_validation_passed": False,
            "projection_binding": {
                **binding(projection_path, kind=LABEL_PROJECTION_KIND),
                "path": (output_root / projection_path.relative_to(stage)).relative_to(ROOT).as_posix(),
            },
            "builder_binding": builder_binding,
            "validator_binding": validator_binding,
            "access_evidence_binding": {
                **binding(label_access_path),
                "path": (output_root / label_access_path.relative_to(stage)).relative_to(ROOT).as_posix(),
            },
        }
        label_attestation_path = label_dir / "train_label_projection_attestation.json"
        write_canonical(label_attestation_path, label_attestation)

        cache_attestation = {
            "schema_version": 1,
            "kind": CACHE_ATTESTATION_KIND,
            "canonical_json": True,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "train_allowlist_sha256": EXPECTED_ALLOWLIST_SHA256,
            "train_set_commitment_sha256": train_commitment,
            "blinded_complement_commitment_sha256": complement_commitment,
            "case_count": 18,
            "record_count": 36,
            "train_exact_match": True,
            "train_holdout_intersection_count": 0,
            "holdout_entry_count": 0,
            "holdout_status_count": 0,
            "holdout_hash_count": 0,
            "optimizer_holdout_access_count": 0,
            "independent_validation_passed": False,
            "manifest_binding": {
                **binding(cache_manifest_path, kind=CACHE_MANIFEST_KIND),
                "path": (output_root / cache_manifest_path.relative_to(stage)).relative_to(ROOT).as_posix(),
            },
            "builder_binding": builder_binding,
            "validator_binding": validator_binding,
            "access_evidence_binding": {
                **binding(cache_access_path),
                "path": (output_root / cache_access_path.relative_to(stage)).relative_to(ROOT).as_posix(),
            },
        }
        cache_attestation_path = cache_dir / "train_v51_cache_manifest_attestation.json"
        write_canonical(cache_attestation_path, cache_attestation)

        # The independent validator validates the still-hidden staging tree,
        # authors a sealed PASS report, and only then rewrites final
        # attestations that bind that report.
        from scripts.audit_f1_aug30_train_only_custody_v1 import seal_staged_custody

        seal_staged_custody(stage, output_root)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    from scripts.audit_f1_aug30_train_only_custody_v1 import audit_custody

    try:
        return audit_custody(output_root)
    except Exception:
        # A post-rename verification failure must not leave a published tree.
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
