import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_f1_v51_20260831_prospective_three_arm_freeze_v1 import (
    DEFAULT_FREEZE,
    DEFAULT_SEAL,
    audit_freeze,
    validate_payload,
)


def test_sealed_freeze_audits_against_current_bound_bytes() -> None:
    result = audit_freeze()
    assert result["audit_pass"] is True
    assert result["arms"] == ["A", "B", "C"]
    assert result["component_count"] == 10
    assert result["artifact_binding_count"] >= 100


def test_seal_detects_any_freeze_byte_change(tmp_path: Path) -> None:
    freeze = tmp_path / DEFAULT_FREEZE.name
    seal = tmp_path / DEFAULT_SEAL.name
    freeze.write_bytes(DEFAULT_FREEZE.read_bytes() + b" ")
    seal.write_text(
        f"{hashlib.sha256(DEFAULT_FREEZE.read_bytes()).hexdigest()}  {freeze.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="freeze seal mismatch"):
        audit_freeze(freeze, seal)


def test_component_path_substitution_fails_closed() -> None:
    payload = json.loads(DEFAULT_FREEZE.read_text(encoding="utf-8"))
    payload["component_lock"]["v43_guard"]["path"] = payload["component_lock"][
        "precision_runtime"
    ]["path"]
    with pytest.raises(ValueError, match="component path mismatch"):
        validate_payload(payload)


def test_forbidden_attestation_open_fails_closed() -> None:
    payload = json.loads(DEFAULT_FREEZE.read_text(encoding="utf-8"))
    payload["pre_discovery_attestation"]["gmail_inspected_for_cohort"] = True
    with pytest.raises(ValueError, match="pre-discovery attestation opened"):
        validate_payload(payload)
