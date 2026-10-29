#!/usr/bin/env python3
"""Assemble the immutable legacy37 + today28 V51 prediction map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LEGACY_COUNT = 37
TODAY_COUNT = 28
TOTAL_COUNT = LEGACY_COUNT + TODAY_COUNT
KNOWN_LEGACY_V1_SHA256 = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _inside(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes declared execution root: {candidate}") from exc
    return candidate


def _prediction_rows(payload: dict[str, Any]) -> list[Any]:
    for key in ("owned_bend_regions", "manual_bends", "bends", "predictions"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("semantic result has no supported prediction array")


def _validate_abstention(token: str, payload: dict[str, Any]) -> None:
    if payload.get("fail_closed_abstention") is not True:
        raise ValueError(f"{token}: abstention must set fail_closed_abstention=true")
    reason = payload.get("abstention_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{token}: abstention requires a non-empty reason")
    if _prediction_rows(payload):
        raise ValueError(f"{token}: fail-closed abstention must have zero predictions")


def _manifest_artifacts(
    manifest_path: Path,
    root: Path,
    *,
    expected_count: int,
) -> dict[str, str]:
    manifest = _read(manifest_path)
    if int(manifest.get("case_count") or -1) != expected_count:
        raise ValueError(f"{manifest_path}: case_count must be {expected_count}")
    expected_tokens = [f"T{index:03d}" for index in range(1, expected_count + 1)]
    if manifest.get("case_ids") != expected_tokens:
        raise ValueError(f"{manifest_path}: case_ids must be complete and ordered {expected_tokens[0]}..{expected_tokens[-1]}")
    identity = manifest.get("semantic_replay_byte_identity")
    if not isinstance(identity, dict) or set(identity) != set(expected_tokens) or not all(identity[token] is True for token in expected_tokens):
        raise ValueError(f"{manifest_path}: semantic_replay_byte_identity must be true for every case")
    records = manifest.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{manifest_path}: records must be an object array")
    record_index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("case_id") or ""), str(record.get("pass") or ""))
        if key in record_index:
            raise ValueError(f"{manifest_path}: duplicate execution record {key}")
        record_index[key] = record
    expected_record_keys = {(token, branch) for token in expected_tokens for branch in ("primary", "replay")}
    if set(record_index) != expected_record_keys:
        raise ValueError(f"{manifest_path}: record set must equal primary+replay for every case")

    artifacts: dict[str, str] = {}
    success_counts = {"primary": 0, "replay": 0}
    failure_counts = {"primary": 0, "replay": 0}
    for token in expected_tokens:
        branches = []
        for branch_name in ("primary", "replay"):
            record = record_index[(token, branch_name)]
            expected_sha = record.get("semantic_result_sha256")
            if not isinstance(expected_sha, str):
                raise ValueError(f"{token}: {branch_name} requires semantic_result_sha256")
            path = _inside(root, f"{branch_name}/{token}/semantic_result.json")
            if not path.is_file() or path.name != "semantic_result.json":
                raise ValueError(f"{token}: {branch_name} semantic_result.json missing")
            actual_sha = _sha(path)
            if actual_sha != expected_sha:
                raise ValueError(f"{token}: {branch_name} artifact hash mismatch")
            payload = _read(path)
            failed = record.get("pipeline_success") is False
            if failed:
                failure_type = record.get("failure_type")
                if not isinstance(failure_type, str) or not failure_type.strip():
                    raise ValueError(f"{token}: fail-closed record requires failure_type")
                if payload.get("kind") != "sherman_qc_v51_fail_closed_case_abstention" or payload.get("pipeline_status") != "FAILED_ABSTAIN":
                    raise ValueError(f"{token}: invalid fail-closed semantic abstention kind/status")
                if payload.get("predictions") != []:
                    raise ValueError(f"{token}: fail-closed abstention predictions must be empty")
                failure_path = _inside(root, f"{branch_name}/{token}/case_failure.json")
                expected_failure_sha = record.get("case_failure_sha256")
                if not failure_path.is_file():
                    raise ValueError(f"{token}: fail-closed case_failure.json missing")
                if expected_failure_sha is not None and (
                    not isinstance(expected_failure_sha, str) or _sha(failure_path) != expected_failure_sha
                ):
                    raise ValueError(f"{token}: fail-closed case_failure.json hash mismatch")
                failure_payload = _read(failure_path)
                if (
                    failure_payload.get("case_id") != token
                    or failure_payload.get("pass") != branch_name
                    or failure_payload.get("failure_type") != failure_type
                ):
                    raise ValueError(f"{token}: fail-closed case_failure.json identity mismatch")
                failure_counts[branch_name] += 1
            else:
                if record.get("pipeline_success", True) is not True:
                    raise ValueError(f"{token}: pipeline_success must be absent/true or false")
                _prediction_rows(payload)
                success_counts[branch_name] += 1
            branches.append((path, actual_sha, payload, failed, record.get("scan_sha256")))
        primary, replay = branches
        if primary[0] == replay[0]:
            raise ValueError(f"{token}: primary and replay must be distinct artifacts")
        if primary[1] != replay[1] or primary[2] != replay[2] or primary[0].read_bytes() != replay[0].read_bytes():
            raise ValueError(f"{token}: primary/replay full semantic identity mismatch")
        if primary[3] != replay[3] or not isinstance(primary[4], str) or primary[4] != replay[4]:
            raise ValueError(f"{token}: primary/replay status or scan identity mismatch")
        artifacts[token] = str(primary[0])
    count_fields = {
        "primary_success_count": success_counts["primary"],
        "replay_success_count": success_counts["replay"],
        "primary_failure_abstention_count": failure_counts["primary"],
        "replay_failure_abstention_count": failure_counts["replay"],
    }
    for field, expected in count_fields.items():
        actual = manifest.get(field, 0 if "failure_abstention" in field else None)
        if actual != expected:
            raise ValueError(f"{manifest_path}: {field} mismatch")
    return artifacts


def assemble_prediction_map(
    legacy_catalog_path: Path,
    legacy_manifest_path: Path,
    legacy_root: Path,
    today_manifest_path: Path,
    today_root: Path,
    *,
    expected_legacy_sha256: str,
) -> dict[str, str]:
    legacy_catalog_path = legacy_catalog_path.resolve()
    legacy_before = legacy_catalog_path.read_bytes()
    if hashlib.sha256(legacy_before).hexdigest() != expected_legacy_sha256:
        raise ValueError("immutable legacy catalog SHA-256 mismatch")
    catalog = json.loads(legacy_before)
    cases = catalog.get("cases")
    if not isinstance(cases, list) or len(cases) != LEGACY_COUNT:
        raise ValueError(f"legacy catalog must contain exactly {LEGACY_COUNT} ordered cases")
    legacy_ids = [str(case.get("case_id") or "") for case in cases]
    if any(not case_id for case_id in legacy_ids) or len(set(legacy_ids)) != LEGACY_COUNT:
        raise ValueError("legacy catalog case IDs must be non-empty and unique")

    legacy_artifacts = _manifest_artifacts(
        legacy_manifest_path.resolve(), legacy_root.resolve(), expected_count=LEGACY_COUNT
    )
    today_artifacts = _manifest_artifacts(
        today_manifest_path.resolve(), today_root.resolve(), expected_count=TODAY_COUNT
    )
    mapping = {
        **{
            case_id: legacy_artifacts[f"T{ordinal:03d}"]
            for ordinal, case_id in enumerate(legacy_ids, 1)
        },
        **{f"T{ordinal:03d}": today_artifacts[f"T{ordinal:03d}"] for ordinal in range(1, TODAY_COUNT + 1)},
    }
    if len(mapping) != TOTAL_COUNT or set(legacy_ids) & set(today_artifacts):
        raise ValueError("expanded prediction map must contain exactly 65 collision-free keys")
    if legacy_catalog_path.read_bytes() != legacy_before:
        raise RuntimeError("immutable legacy catalog changed during assembly")
    return dict(sorted(mapping.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-catalog", type=Path, required=True)
    parser.add_argument("--legacy-catalog-sha256", default=KNOWN_LEGACY_V1_SHA256)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--today-manifest", type=Path, required=True)
    parser.add_argument("--today-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = assemble_prediction_map(
        args.legacy_catalog,
        args.legacy_manifest,
        args.legacy_root,
        args.today_manifest,
        args.today_root,
        expected_legacy_sha256=args.legacy_catalog_sha256,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"key_count": len(mapping), "sha256": _sha(output)}, indent=2))


if __name__ == "__main__":
    main()
