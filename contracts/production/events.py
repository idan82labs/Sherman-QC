"""LineEventV1: the append-only line event log (SQLite autoincrement ids).

``JOB_PROGRESS`` is rate limited by the producer to at most one per second
per job. Payloads are small JSON objects; they never carry secrets or tokens.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from contracts.production.common import ContractModel, Principal, UtcIsoTimestamp

__all__ = ["LINE_EVENT_SCHEMA", "EventKind", "EVENT_KINDS", "LineEventV1"]

LINE_EVENT_SCHEMA = "sherman_line_event_v1"

EventKind = Literal[
    "SCAN_RECEIVED",
    "SCAN_DUPLICATE",
    "SCAN_QUARANTINED",
    "JOB_QUEUED",
    "JOB_STARTED",
    "JOB_PROGRESS",
    "JOB_COMPLETED",
    "JOB_FAILED",
    "JOB_CANCELLED",
    "JOB_RECOVERED",
    "JOB_RETRY_REQUESTED",
    "JOB_CANCEL_REQUESTED",
    "QUARANTINE_ASSIGNED",
    "QUARANTINE_DISMISSED",
    "RECIPE_RELEASED",
    "RECIPE_RETIRED",
    "RECIPE_COMPILED",
    "RECIPE_COMPILE_BLOCKED",
    "RECIPE_COMPILE_FAILED",
    "DEVICE_TOKEN_ISSUED",
    "DEVICE_TOKEN_REVOKED",
]
EVENT_KINDS: frozenset[str] = frozenset(EventKind.__args__)  # type: ignore[attr-defined]


class LineEventV1(ContractModel):
    schema_: Literal["sherman_line_event_v1"] = Field(default=LINE_EVENT_SCHEMA, alias="schema")
    id: int = Field(ge=1)
    ts: UtcIsoTimestamp
    kind: EventKind
    station_id: str | None = Field(default=None, max_length=64)
    job_id: str | None = Field(default=None, pattern=r"^line_[0-9a-f]{32}$")
    scan_file_id: str | None = Field(default=None, pattern=r"^scan_[0-9a-f]{24}$")
    principal: Principal | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
