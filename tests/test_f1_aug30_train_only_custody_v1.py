from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

import pytest

from scripts.audit_f1_aug30_train_only_custody_v1 import (
    CACHE_ATTEST_KIND,
    DEFAULT_OUTPUT,
    LABEL_ATTEST_KIND,
    audit_custody,
    canonical_bytes,
)


def test_published_bundle_passes_independent_exact_train_audit() -> None:
    result = audit_custody(DEFAULT_OUTPUT)
    assert result == {
        **result,
        "status": "PASS",
        "case_count": 18,
        "label_file_count": 18,
        "cache_record_count": 36,
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "holdout_rows_emitted": 0,
    }


@pytest.mark.parametrize(
    ("relative", "kind"),
    (
        ("labels/train_label_projection_attestation.json", LABEL_ATTEST_KIND),
        ("cache/train_v51_cache_manifest_attestation.json", CACHE_ATTEST_KIND),
    ),
)
def test_optimizer_bootstrap_attestations_are_canonical_and_value_free(
    relative: str, kind: str
) -> None:
    path = DEFAULT_OUTPUT / relative
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_bytes(payload)
    assert payload["kind"] == kind
    assert payload["independent_validation_passed"] is True
    assert payload["staged_validator_pass_binding"]["sha256"]
    encoded = raw.lower()
    for forbidden in (b'"bends"', b'"predictions"', b'"records"', b'"manual_bends"'):
        assert forbidden not in encoded


def test_independent_validator_rejects_any_extra_projection_file(tmp_path: Path) -> None:
    copied = tmp_path / "custody"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    (copied / "labels/cases/A30-999.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory is not exact"):
        audit_custody(copied, published_root=DEFAULT_OUTPUT)


def _write(path: Path, payload: dict) -> None:
    path.write_bytes(canonical_bytes(payload))


def _binding(path: Path, published: Path, *, kind: str | None = None) -> dict:
    raw = path.read_bytes()
    row = {
        "path": published.relative_to(DEFAULT_OUTPUT.parents[2]).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    if kind is not None:
        row["kind"] = kind
    return row


def _rebind_validation_report(copied: Path) -> None:
    report_path = copied / "independent_validation_report.json"
    report = json.loads(report_path.read_bytes())
    label_access = copied / "labels/access_evidence.json"
    cache_access = copied / "cache/access_evidence.json"
    report["label_access_evidence_binding"] = _binding(
        label_access, DEFAULT_OUTPUT / "labels/access_evidence.json"
    )
    report["cache_access_evidence_binding"] = _binding(
        cache_access, DEFAULT_OUTPUT / "cache/access_evidence.json"
    )
    _write(report_path, report)
    report_binding = _binding(
        report_path,
        DEFAULT_OUTPUT / "independent_validation_report.json",
        kind=report["kind"],
    )
    for relative in (
        "labels/train_label_projection_attestation.json",
        "cache/train_v51_cache_manifest_attestation.json",
    ):
        path = copied / relative
        payload = json.loads(path.read_bytes())
        payload["staged_validator_pass_binding"] = report_binding
        _write(path, payload)


def test_validator_rejects_false_zero_denied_read_claim(tmp_path: Path) -> None:
    copied = tmp_path / "custody"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    access_path = copied / "labels/access_evidence.json"
    access = json.loads(access_path.read_bytes())
    access["denied_open_attempt_count"] = 1
    _write(access_path, access)
    attestation_path = copied / "labels/train_label_projection_attestation.json"
    attestation = json.loads(attestation_path.read_bytes())
    attestation["access_evidence_binding"] = _binding(
        access_path, DEFAULT_OUTPUT / "labels/access_evidence.json"
    )
    _write(attestation_path, attestation)
    _rebind_validation_report(copied)
    with pytest.raises(ValueError, match="label access evidence drift"):
        audit_custody(copied, published_root=DEFAULT_OUTPUT)


def test_validator_rejects_attestation_schema_extension(tmp_path: Path) -> None:
    copied = tmp_path / "custody"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    path = copied / "labels/train_label_projection_attestation.json"
    payload = json.loads(path.read_bytes())
    payload["unreviewed_extension"] = True
    _write(path, payload)
    with pytest.raises(ValueError, match="exact-key schema drift"):
        audit_custody(copied, published_root=DEFAULT_OUTPUT)


def test_validator_rejects_false_full_manifest_nonaccess_claim(tmp_path: Path) -> None:
    copied = tmp_path / "custody"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    access_path = copied / "cache/access_evidence.json"
    access = json.loads(access_path.read_bytes())
    access["full_execution_manifest_opened"] = True
    _write(access_path, access)
    attestation_path = copied / "cache/train_v51_cache_manifest_attestation.json"
    attestation = json.loads(attestation_path.read_bytes())
    attestation["access_evidence_binding"] = _binding(
        access_path, DEFAULT_OUTPUT / "cache/access_evidence.json"
    )
    _write(attestation_path, attestation)
    _rebind_validation_report(copied)
    with pytest.raises(ValueError, match="cache access evidence drift"):
        audit_custody(copied, published_root=DEFAULT_OUTPUT)
