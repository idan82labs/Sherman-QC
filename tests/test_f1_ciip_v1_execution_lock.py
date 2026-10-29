from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.f1_ciip_v1.config import load_frozen_v7, select_smoke_source_by_content
from research.f1_ciip_v1.execution_lock import (
    AUTHORIZATION_RECEIPT,
    AUTHORIZATION_SCOPE,
    DEPENDENCY_LOCK_TAG,
    EXECUTION_LOCK_TAG,
    ExecutionLockError,
    PERMUTATION_RULE,
    ROLE_ORDER,
    code_record,
    hash_open_regular_source,
    tagged_json_digest,
    validate_execution_lock,
)


def test_content_selected_source_matches_frozen_digest_without_path_semantics() -> None:
    selected = select_smoke_source_by_content(load_frozen_v7())
    observed, size = hash_open_regular_source(Path(selected["raw_path"]))
    assert observed == "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
    assert size == 15_015_773


def test_code_record_is_path_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.cpp"
    first.write_bytes(b"alpha\n")
    second.write_bytes(b"beta\n")
    record_a, digest_a = code_record(tmp_path, [second, first])
    record_b, digest_b = code_record(tmp_path, [first, second])
    assert record_a == record_b
    assert digest_a == digest_b
    second.write_bytes(b"changed\n")
    assert code_record(tmp_path, [first, second])[1] != digest_a


def _minimal_lock() -> dict:
    dependency = {"probe": "exact"}
    body = {
        "schema_version": 1,
        "authorization": {"receipt": AUTHORIZATION_RECEIPT, "scope": AUTHORIZATION_SCOPE},
        "execution": {
            "input_order_permutation": PERMUTATION_RULE,
            "role_order": list(ROLE_ORDER),
        },
        "dependency": dependency,
        "dependency_lock_sha256": tagged_json_digest(DEPENDENCY_LOCK_TAG, dependency),
    }
    body["execution_lock_sha256"] = tagged_json_digest(EXECUTION_LOCK_TAG, body)
    return body


def test_execution_lock_self_hash_and_frozen_intervention() -> None:
    lock = _minimal_lock()
    with pytest.raises(ExecutionLockError, match="top-level shape"):
        validate_execution_lock(lock)
    tampered = {**lock, "execution": {**lock["execution"], "role_order": ["BASE_A"]}}
    with pytest.raises(ExecutionLockError, match="top-level shape"):
        validate_execution_lock(tampered)


def test_source_hash_helper_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "source.ply"
    target.write_bytes(b"ply\n")
    link = tmp_path / "link.ply"
    link.symlink_to(target)
    with pytest.raises(ExecutionLockError):
        hash_open_regular_source(link)
