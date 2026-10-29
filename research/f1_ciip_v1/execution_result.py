"""Closed verifier for the single frozen CIIP-V1 smoke result artifact."""

from __future__ import annotations

import re
from typing import Any, Mapping


PASS_TOKEN = "CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
STOP_TOKEN = "INVALID_CII_PERSISTENCE_PREFLIGHT_STOP"
ROLE_ORDER = ("BASE_A", "BASE_B", "INPUT_ORDER_PERMUTED", "INDEPENDENT_VALIDATOR")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RESULT_KEYS = {
    "outcome", "failure", "execution_lock_sha256", "raw_ply_sha256", "roles",
    "rerun_authorized", "replacement_input_authorized", "gold_or_labels_accessed",
}
_ROLE_KEYS = {
    "role", "return_code", "cpu_nanoseconds", "wall_nanoseconds",
    "peak_process_tree_rss_bytes", "scratch_bytes", "stdout_sha256",
    "stderr_sha256", "failure", "container_sha256",
}


class ExecutionResultError(ValueError):
    """A terminal result artifact failed the frozen closed-schema contract."""


def _u64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ExecutionResultError(f"{label} must be a u64 integer")
    return value


def _digest(value: Any, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ExecutionResultError(f"{label} must be lowercase SHA-256")


def validate_execution_result(result: Mapping[str, Any]) -> None:
    """Reject altered tokens, forged success, open booleans, or malformed role evidence."""

    if not isinstance(result, Mapping) or set(result) != _RESULT_KEYS:
        raise ExecutionResultError("result keys are not the frozen exact set")
    outcome = result["outcome"]
    if outcome not in {PASS_TOKEN, STOP_TOKEN}:
        raise ExecutionResultError("terminal outcome token is not frozen")
    _digest(result["execution_lock_sha256"], "execution_lock_sha256")
    _digest(result["raw_ply_sha256"], "raw_ply_sha256")
    for key in ("rerun_authorized", "replacement_input_authorized", "gold_or_labels_accessed"):
        if result[key] is not False:
            raise ExecutionResultError(f"{key} must remain false")

    roles = result["roles"]
    if not isinstance(roles, list) or len(roles) > len(ROLE_ORDER):
        raise ExecutionResultError("role evidence must be a bounded ordered list")
    observed_roles: list[str] = []
    for index, row in enumerate(roles):
        if not isinstance(row, Mapping) or set(row) != _ROLE_KEYS:
            raise ExecutionResultError("role evidence keys are not the frozen exact set")
        if row["role"] != ROLE_ORDER[index]:
            raise ExecutionResultError("role evidence is reordered, duplicated, or skipped")
        observed_roles.append(row["role"])
        if isinstance(row["return_code"], bool) or not isinstance(row["return_code"], int):
            raise ExecutionResultError("role return code must be an integer")
        for key in ("cpu_nanoseconds", "wall_nanoseconds", "peak_process_tree_rss_bytes", "scratch_bytes"):
            _u64(row[key], key)
        _digest(row["stdout_sha256"], "stdout_sha256")
        _digest(row["stderr_sha256"], "stderr_sha256")
        _digest(row["container_sha256"], "container_sha256", optional=True)
        if row["failure"] is not None and not isinstance(row["failure"], str):
            raise ExecutionResultError("role failure must be null or a string")

    failure = result["failure"]
    if outcome == PASS_TOKEN:
        if failure is not None or tuple(observed_roles) != ROLE_ORDER:
            raise ExecutionResultError("pass requires four complete roles and no failure")
        if any(row["failure"] is not None or row["return_code"] != 0 or row["container_sha256"] is None for row in roles):
            raise ExecutionResultError("pass contains failed or incomplete role evidence")
    else:
        if not isinstance(failure, str) or not failure:
            raise ExecutionResultError("STOP requires a nonempty terminal failure")


__all__ = [
    "ExecutionResultError",
    "PASS_TOKEN",
    "ROLE_ORDER",
    "STOP_TOKEN",
    "validate_execution_result",
]
