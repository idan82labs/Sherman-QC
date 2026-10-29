#!/usr/bin/env python3
"""Label-incapable TRAIN-cache source compatibility preflight for Candidate3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from typing import Any, Callable


# Bootstrap stays stdlib-only. The canonical runtime is rejected if preloaded,
# then compiled/executed from the exact bytes authorized on the command line.
CANONICAL_RUNTIME_MODULE = "backend.f1_train18_candidate3_conservative_deletion"

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "backend/f1_train18_candidate3_conservative_deletion.py"
TESTS_PATH = ROOT / "tests/test_f1_train18_candidate3_conservative_deletion.py"
PREFLIGHT_PATH = ROOT / "scripts/preflight_f1_aug30_train18_candidate3_source_compatibility.py"

SPLIT_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
SPLIT_SHA256 = "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
ORIGINAL_PREREG_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ORIGINAL_PREREG_SHA256 = "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a"
ALLOWLIST_PATH = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
ALLOWLIST_SHA256 = "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
CANDIDATE1_TERMINAL_PATH = ROOT / "research/f1_aug30_train18_candidate1_terminal_negative_v1.json"
CANDIDATE1_TERMINAL_SHA256 = "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
CANDIDATE2_TERMINAL_PATH = ROOT / "research/f1_aug30_train18_candidate2_terminal_negative_v1.json"
CANDIDATE2_TERMINAL_SHA256 = "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515"
CANDIDATE2_TERMINAL_SIZE_BYTES = 7001
CACHE_ATTESTATION_PATH = (
    ROOT / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json"
)
CACHE_ATTESTATION_SHA256 = "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
LABEL_ATTESTATION_SHA256_NOT_OPENED = (
    "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output/research/f1-aug30-train18-candidate3-source-compatibility-preflight-v1"
).resolve()
ALLOWED_CACHE_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
).resolve()
FORBIDDEN_SOURCE_VOLUME_KEYS = frozenset(
    {
        "prediction_row_count",
        "feature_row_count",
        "rule_path_check_count",
        "missing_numeric_feature_counts",
        "prediction_count",
        "candidate_count",
    }
)
PERSISTED_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile",
        "attempt_ordinal",
        "status",
        "case_count",
        "record_count",
        "attested_artifact_file_count",
        "primary_replay_pair_count",
        "all_primary_replay_exact",
        "all_records_source_compatible",
        "all_allowlisted_feature_and_rule_paths_exercised",
        "scan_identity_authority",
        "semantic_internal_scan_sha256_trusted",
        "label_attestation_opened",
        "label_projection_opened",
        "holdout_score_opened",
        "candidate_metric_computed",
        "model_fit_performed",
        "deletion_rule_selected",
        "attestation_checks",
        "read_ledger_sha256",
        "bindings",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _read_exact_bytes_once(path: Path, expected_sha256: str, role: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{role} path must not be a symlink")
    data = path.read_bytes()
    actual = _sha256(data)
    if actual != expected_sha256:
        raise ValueError(f"{role} SHA mismatch: {actual} != {expected_sha256}")
    return data


def _bootstrap_verified_sources(
    *, runtime_sha256: str, preflight_sha256: str, tests_sha256: str
) -> tuple[types.ModuleType, dict[str, str]]:
    if CANONICAL_RUNTIME_MODULE in sys.modules:
        raise RuntimeError("Candidate3 preflight rejects a preloaded canonical runtime module")
    expected = {
        "runtime_source_sha256": _require_sha256(runtime_sha256, "runtime_sha256"),
        "preflight_source_sha256": _require_sha256(preflight_sha256, "preflight_sha256"),
        "tests_source_sha256": _require_sha256(tests_sha256, "tests_sha256"),
    }
    # All source bytes are read once, verified, and retained before any cache read.
    preflight_bytes = _read_exact_bytes_once(
        PREFLIGHT_PATH, expected["preflight_source_sha256"], "preflight source"
    )
    runtime_bytes = _read_exact_bytes_once(
        RUNTIME_PATH, expected["runtime_source_sha256"], "runtime source"
    )
    _read_exact_bytes_once(TESTS_PATH, expected["tests_source_sha256"], "tests source")
    # Keep the self bytes alive to make the no-reread property explicit.
    if _sha256(preflight_bytes) != expected["preflight_source_sha256"]:
        raise AssertionError("verified preflight bytes changed in memory")
    module = types.ModuleType("_candidate3_verified_runtime")
    module.__file__ = str(RUNTIME_PATH)
    module.__package__ = ""
    code = compile(runtime_bytes, str(RUNTIME_PATH), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module, expected


def _decode_json(data: bytes, role: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _read_bound_json(
    path: Path,
    expected_sha256: str,
    ledger: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{role} path must not be a symlink")
    resolved = path.resolve()
    data = _read_exact_bytes_once(resolved, expected_sha256, role)
    value = _decode_json(data, role)
    ledger.append(
        {
            "role": role,
            "path": str(resolved.relative_to(ROOT)),
            "sha256": expected_sha256,
            "size_bytes": len(data),
        }
    )
    return value


def _validate_candidate1(value: dict[str, Any]) -> None:
    if value.get("candidate_id") != 1 or value.get("status") != "REJECTED_TRAIN_GATE":
        raise ValueError("Candidate1 terminal-negative mismatch")
    if value.get("holdout_score_opened") is not False:
        raise ValueError("Candidate1 terminal reports holdout score access")
    if value.get("post_terminal_model_or_threshold_mutation") != "FORBIDDEN":
        raise ValueError("Candidate1 terminal does not prohibit mutation")


def _validate_candidate2(value: dict[str, Any]) -> None:
    if value.get("attempt_ordinal") != 2:
        raise ValueError("Candidate2 terminal attempt mismatch")
    if value.get("status") != "REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY":
        raise ValueError("Candidate2 is not the confirmed prefit terminal-negative")
    required_false = (
        "candidate2_continuation_authorized",
        "candidate_metric_computed",
        "candidate_model_created",
        "candidate_output_created",
        "holdout_execution_authorized",
        "holdout_score_opened",
    )
    if any(value.get(key) is not False for key in required_false):
        raise ValueError("Candidate2 terminal contains an enabled continuation/output flag")
    if value.get("holdout_access_count") != 0:
        raise ValueError("Candidate2 terminal reports forbidden holdout access")
    if value.get("post_terminal_candidate2_model_or_threshold_mutation") != "FORBIDDEN":
        raise ValueError("Candidate2 terminal does not prohibit mutation")


def _inside_allowed_cache(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ALLOWED_CACHE_ROOT or ALLOWED_CACHE_ROOT in resolved.parents


def _validate_output_target(output_root: Path) -> Path:
    if output_root != DEFAULT_OUTPUT_ROOT:
        raise ValueError("Candidate3 preflight output path is frozen to its exact namespace")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Candidate3 preflight requires a fresh, absent output directory")
    current = output_root.parent
    while current != ROOT.parent:
        if current.is_symlink():
            raise ValueError("Candidate3 preflight rejects symlinked output ancestry")
        if current == ROOT:
            break
        current = current.parent
    if current != ROOT:
        raise ValueError("Candidate3 preflight output must remain inside the repository")
    return output_root


def _publish_atomically(
    output_root: Path,
    payloads: dict[str, bytes],
    *,
    after_stage_write: Callable[[Path], None] | None = None,
) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("atomic publication target already exists")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=parent))
    try:
        for name, data in sorted(payloads.items()):
            target = staging / name
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if after_stage_write is not None:
            after_stage_write(staging)
        os.replace(staging, output_root)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if output_root.exists():
            raise RuntimeError("atomic publication failed after target became visible")
        raise


def _assert_no_forbidden_source_volume_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_SOURCE_VOLUME_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"persisted preflight contains forbidden source-volume keys: {sorted(forbidden)}")
        for child in value.values():
            _assert_no_forbidden_source_volume_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_source_volume_keys(child)


def _build_preflight_payloads(
    runtime: types.ModuleType, source_bindings: dict[str, str]
) -> dict[str, bytes]:
    ledger: list[dict[str, Any]] = []
    split = _read_bound_json(SPLIT_PATH, SPLIT_SHA256, ledger, "sealed_split")
    original_prereg = _read_bound_json(
        ORIGINAL_PREREG_PATH, ORIGINAL_PREREG_SHA256, ledger, "original_prereg"
    )
    allowlist = _read_bound_json(ALLOWLIST_PATH, ALLOWLIST_SHA256, ledger, "train_allowlist")
    candidate1 = _read_bound_json(
        CANDIDATE1_TERMINAL_PATH,
        CANDIDATE1_TERMINAL_SHA256,
        ledger,
        "candidate1_terminal_negative",
    )
    candidate2 = _read_bound_json(
        CANDIDATE2_TERMINAL_PATH,
        CANDIDATE2_TERMINAL_SHA256,
        ledger,
        "candidate2_terminal_negative",
    )
    if ledger[-1]["size_bytes"] != CANDIDATE2_TERMINAL_SIZE_BYTES:
        raise ValueError("Candidate2 terminal size binding mismatch")
    cache_attestation = _read_bound_json(
        CACHE_ATTESTATION_PATH,
        CACHE_ATTESTATION_SHA256,
        ledger,
        "custody_cache_attestation",
    )
    _validate_candidate1(candidate1)
    _validate_candidate2(candidate2)
    if original_prereg.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN":
        raise ValueError("original preregistration status mismatch")
    train_ids = tuple(split.get("train_case_ids") or [])
    allow_rows = allowlist.get("cases") or []
    allow_by_id = {row["case_id"]: row["scan_sha256"] for row in allow_rows}
    if len(train_ids) != 18 or set(train_ids) != set(allow_by_id):
        raise ValueError("split/allowlist TRAIN18 mismatch")

    attestation_checks = {
        "case_count": cache_attestation.get("case_count") == 18,
        "record_count": cache_attestation.get("record_count") == 36,
        "holdout_entry_count": cache_attestation.get("holdout_entry_count") == 0,
        "holdout_hash_count": cache_attestation.get("holdout_hash_count") == 0,
        "holdout_status_count": cache_attestation.get("holdout_status_count") == 0,
        "optimizer_holdout_access_count": cache_attestation.get("optimizer_holdout_access_count")
        == 0,
        "independent_validation_passed": cache_attestation.get("independent_validation_passed")
        is True,
        "split_binding": cache_attestation.get("split_sha256") == SPLIT_SHA256,
        "allowlist_binding": cache_attestation.get("train_allowlist_sha256")
        == ALLOWLIST_SHA256,
    }
    if not all(attestation_checks.values()):
        raise ValueError("TRAIN cache attestation failed a source-compatibility precondition")
    manifest_binding = cache_attestation.get("manifest_binding") or {}
    manifest = _read_bound_json(
        ROOT / manifest_binding["path"],
        manifest_binding["sha256"],
        ledger,
        "attested_train_cache_manifest",
    )
    if manifest.get("case_count") != 18 or manifest.get("record_count") != 36:
        raise ValueError("attested TRAIN cache manifest size mismatch")
    if manifest.get("holdout_entry_count") != 0 or manifest.get("source_only") is not True:
        raise ValueError("attested TRAIN cache manifest boundary mismatch")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("attested TRAIN cache records must be a list")
    by_key = {(row["case_id"], row["pass"]): row for row in records}
    expected_keys = {
        (case_id, pass_name)
        for case_id in train_ids
        for pass_name in ("primary", "replay")
    }
    if len(records) != len(by_key) or set(by_key) != expected_keys:
        raise ValueError("attested TRAIN cache record identity mismatch")

    results_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    all_rule_paths_exercised = True
    for case_id in train_ids:
        for pass_name in ("primary", "replay"):
            record = by_key[(case_id, pass_name)]
            if record.get("scan_sha256") != allow_by_id[case_id]:
                raise ValueError("manifest-derived scan identity differs from TRAIN allowlist")
            semantic_path = ROOT / record["semantic_path"]
            pool_path = ROOT / record["pool_path"]
            if not _inside_allowed_cache(semantic_path) or not _inside_allowed_cache(pool_path):
                raise ValueError("attested artifact path leaves allowed TRAIN cache root")
            semantic = _read_bound_json(
                semantic_path, record["semantic_sha256"], ledger, "attested_train_v51"
            )
            pool = _read_bound_json(
                pool_path, record["pool_sha256"], ledger, "attested_train_pool"
            )
            # Deliberately never read semantic["scan_sha256"].
            result = runtime.preflight_candidate3_source_compatibility(
                semantic,
                pool,
                v51_artifact_sha256=record["semantic_sha256"],
                pool_artifact_sha256=record["pool_sha256"],
            )
            results_by_key[(case_id, pass_name)] = result
            all_rule_paths_exercised = all_rule_paths_exercised and bool(
                result["feature_extraction_complete"]
                and result["all_numeric_le_ge_and_categorical_paths_exercised"]
            )
    exact_pair_count = 0
    for case_id in train_ids:
        primary = results_by_key[(case_id, "primary")]
        replay = results_by_key[(case_id, "replay")]
        if _canonical_bytes(primary) != _canonical_bytes(replay):
            raise ValueError("Candidate3 source compatibility differs across primary/replay")
        exact_pair_count += 1

    ledger_payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate3_preflight_read_ledger_v1",
        "attempt_ordinal": 3,
        "label_attestation_opened": False,
        "label_projection_opened": False,
        "holdout_score_opened": False,
        "read_count": len(ledger),
        "reads": ledger,
        "denied_capabilities": {
            "label_paths_resolved": False,
            "full_cohort_manifest_resolved": False,
            "holdout_paths_resolved": False,
            "semantic_internal_scan_identity_read": False,
        },
    }
    ledger_bytes = _canonical_bytes(ledger_payload)
    ledger_sha = _sha256(ledger_bytes)
    bindings = {
        "candidate1_terminal": {
            "path": str(CANDIDATE1_TERMINAL_PATH.relative_to(ROOT)),
            "sha256": CANDIDATE1_TERMINAL_SHA256,
        },
        "candidate2_terminal": {
            "path": str(CANDIDATE2_TERMINAL_PATH.relative_to(ROOT)),
            "sha256": CANDIDATE2_TERMINAL_SHA256,
            "size_bytes": CANDIDATE2_TERMINAL_SIZE_BYTES,
            "status": "REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY",
        },
        "split_sha256": SPLIT_SHA256,
        "train_allowlist_sha256": ALLOWLIST_SHA256,
        "original_prereg_sha256": ORIGINAL_PREREG_SHA256,
        "custody_cache_attestation_sha256": CACHE_ATTESTATION_SHA256,
        "custody_cache_manifest_sha256": manifest_binding["sha256"],
        "custody_label_attestation_sha256_not_opened": LABEL_ATTESTATION_SHA256_NOT_OPENED,
        **source_bindings,
    }
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate3_source_compatibility_report_v1",
        "profile": runtime.PROFILE,
        "attempt_ordinal": runtime.ATTEMPT_ORDINAL,
        "status": "PASS",
        "case_count": 18,
        "record_count": 36,
        "attested_artifact_file_count": 72,
        "primary_replay_pair_count": exact_pair_count,
        "all_primary_replay_exact": exact_pair_count == 18,
        "all_records_source_compatible": True,
        "all_allowlisted_feature_and_rule_paths_exercised": all_rule_paths_exercised,
        "scan_identity_authority": "ATTESTED_CACHE_MANIFEST_ONLY",
        "semantic_internal_scan_sha256_trusted": False,
        "label_attestation_opened": False,
        "label_projection_opened": False,
        "holdout_score_opened": False,
        "candidate_metric_computed": False,
        "model_fit_performed": False,
        "deletion_rule_selected": False,
        "attestation_checks": attestation_checks,
        "read_ledger_sha256": ledger_sha,
        "bindings": bindings,
    }
    if set(report) != PERSISTED_REPORT_KEYS:
        raise ValueError("Candidate3 persisted preflight report schema mismatch")
    _assert_no_forbidden_source_volume_keys(report)
    report_bytes = _canonical_bytes(report)
    report_sha = _sha256(report_bytes)
    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate3_source_compatibility_lock_v1",
        "attempt_ordinal": 3,
        "status": "PASS_ENGINEERING_ONLY",
        "candidate3_preregistration_authorized_by_this_lock": False,
        "candidate3_fit_authorized_by_this_lock": False,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "preflight_report_sha256": report_sha,
        "read_ledger_sha256": ledger_sha,
        "bindings": bindings,
    }
    lock_bytes = _canonical_bytes(lock)
    return {
        "read_ledger.json": ledger_bytes,
        "preflight_report.json": report_bytes,
        "preflight_lock.json": lock_bytes,
    }


def run(
    *, runtime_sha256: str, preflight_sha256: str, tests_sha256: str
) -> dict[str, Any]:
    output_root = _validate_output_target(DEFAULT_OUTPUT_ROOT)
    runtime, source_bindings = _bootstrap_verified_sources(
        runtime_sha256=runtime_sha256,
        preflight_sha256=preflight_sha256,
        tests_sha256=tests_sha256,
    )
    payloads = _build_preflight_payloads(runtime, source_bindings)
    _publish_atomically(output_root, payloads)
    return {
        "status": "PASS_ENGINEERING_ONLY",
        "case_count": 18,
        "record_count": 36,
        "all_primary_replay_exact": True,
        "label_projection_opened": False,
        "preflight_report_sha256": _sha256(payloads["preflight_report.json"]),
        "read_ledger_sha256": _sha256(payloads["read_ledger.json"]),
        "preflight_lock_sha256": _sha256(payloads["preflight_lock.json"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--tests-sha256", required=True)
    parser.add_argument("--source-check-only", action="store_true")
    args = parser.parse_args()
    runtime, bindings = _bootstrap_verified_sources(
        runtime_sha256=args.runtime_sha256,
        preflight_sha256=args.preflight_sha256,
        tests_sha256=args.tests_sha256,
    )
    if args.source_check_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_BYTES_VERIFIED_ONLY",
                    "profile": runtime.PROFILE,
                    "bindings": bindings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    # Reuse the already verified module/bytes; never bootstrap a second time.
    output_root = _validate_output_target(DEFAULT_OUTPUT_ROOT)
    payloads = _build_preflight_payloads(runtime, bindings)
    _publish_atomically(output_root, payloads)
    print(
        json.dumps(
            {
                "status": "PASS_ENGINEERING_ONLY",
                "case_count": 18,
                "record_count": 36,
                "all_primary_replay_exact": True,
                "label_projection_opened": False,
                "preflight_report_sha256": _sha256(payloads["preflight_report.json"]),
                "read_ledger_sha256": _sha256(payloads["read_ledger.json"]),
                "preflight_lock_sha256": _sha256(payloads["preflight_lock.json"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
