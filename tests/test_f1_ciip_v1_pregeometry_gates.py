from __future__ import annotations

from typing import Any

import pytest

from research.f1_ciip_v1 import independent_validator, semantic_core


class _ParserReached(RuntimeError):
    pass


def test_semantic_role_runs_all_source_free_gates_before_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    frozen = object()

    monkeypatch.setattr(
        semantic_core,
        "validate_v7_known_answer_vectors",
        lambda: events.append("primary_kat"),
    )
    monkeypatch.setattr(
        semantic_core,
        "load_frozen_v7",
        lambda: events.append("load_frozen") or frozen,
    )
    monkeypatch.setattr(
        semantic_core,
        "validate_frozen_allocation_plan",
        lambda value: events.append("allocation") if value is frozen else None,
    )

    class Worker:
        def validate_known_answers(self) -> None:
            events.append("native_kat")

    def parser(_fd: int) -> Any:
        events.append("parser")
        raise _ParserReached

    monkeypatch.setattr(semantic_core, "parse_ply_fd", parser)
    with pytest.raises(_ParserReached):
        semantic_core.execute_semantic_role(
            9,
            expected_source={},
            native_worker=Worker(),
            code_sha256="11" * 32,
            dependency_lock_sha256="22" * 32,
            role="BASE_A",
        )
    assert events == ["primary_kat", "native_kat", "load_frozen", "allocation", "parser"]


def test_independent_role_runs_own_kat_and_allocation_before_fd_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    frozen = object()
    monkeypatch.setattr(
        independent_validator,
        "validate_independent_known_answer_vectors",
        lambda: events.append("independent_kat"),
    )
    monkeypatch.setattr(
        independent_validator,
        "load_frozen_v7",
        lambda: events.append("load_frozen") or frozen,
    )
    monkeypatch.setattr(
        independent_validator,
        "validate_frozen_allocation_plan",
        lambda value: events.append("allocation") if value is frozen else None,
    )

    def fstat(_fd: int) -> Any:
        events.append("fstat")
        raise _ParserReached

    monkeypatch.setattr(independent_validator.os, "fstat", fstat)
    with pytest.raises(_ParserReached):
        independent_validator.parse_and_verify_ply_fd(9, {})
    assert events == ["independent_kat", "load_frozen", "allocation", "fstat"]

