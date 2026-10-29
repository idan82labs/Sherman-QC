"""Line ingest and job contracts: ScanFileV1, LineJobV1, LineJobSummaryV1.

Execution status never encodes conformance: a failed or timed-out engine run
is an execution failure (``failure.code``), never a part FAIL.

The literal sets below are the single source of truth; wave-1 chunks that
duplicate them (C02 store/runner, C04 foundation) are checked by C11's parity
test (tests/test_prod_contract_parity.py), which imports these frozensets.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from contracts.production.common import ContractModel, NonNegFloat, RelPath, Sha256Hex, UtcIsoTimestamp, Vec3
from contracts.production.part_recipe import RECIPE_ID_PATTERN, SKU_PATTERN
from contracts.production.states import StageStatus, SummaryState, ValidityState

__all__ = [
    "SCAN_FILE_SCHEMA",
    "LINE_JOB_SCHEMA",
    "JOB_ID_PATTERN",
    "SCAN_FILE_ID_PATTERN",
    "ExecutionStatus",
    "EXECUTION_STATUSES",
    "JobKind",
    "JOB_KINDS",
    "Stage",
    "STAGES",
    "FailureCode",
    "FAILURE_CODES",
    "ParseStatus",
    "PARSE_STATUSES",
    "RevisionStatus",
    "REVISION_STATUSES",
    "ResolveStatus",
    "RESOLVE_STATUSES",
    "IngestState",
    "INGEST_STATES",
    "QuarantineReason",
    "QUARANTINE_REASONS",
    "ScanFormat",
    "SCAN_FORMATS",
    "PRIORITY_BY_KIND",
    "MAX_ATTEMPTS_BY_KIND",
    "ENGINE_HARD_TIMEOUT_S",
    "ENGINE_RSS_LIMIT_BYTES",
    "DeclaredStageV1",
    "ScanNameParseV1",
    "ScanMatchV1",
    "ScanFileV1",
    "JobFailureV1",
    "JobResultSummaryV1",
    "LineJobV1",
    "LineJobSummaryV1",
]

SCAN_FILE_SCHEMA = "sherman_scan_file_v1"
LINE_JOB_SCHEMA = "sherman_line_job_v1"
JOB_ID_PATTERN = r"^line_[0-9a-f]{32}$"
SCAN_FILE_ID_PATTERN = r"^scan_[0-9a-f]{24}$"

ExecutionStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
EXECUTION_STATUSES: frozenset[str] = frozenset(ExecutionStatus.__args__)  # type: ignore[attr-defined]

#: ``inspect``: one scan through E3 + C06. ``recipe_compile``: E1 onboarding compile
#: (its CompiledPart nominals are the recipe "expectations").
JobKind = Literal["inspect", "recipe_compile"]
JOB_KINDS: frozenset[str] = frozenset(JobKind.__args__)  # type: ignore[attr-defined]

Stage = Literal["queued", "preparing", "engine", "postprocess", "done"]
STAGES: frozenset[str] = frozenset(Stage.__args__)  # type: ignore[attr-defined]

FailureCode = Literal[
    "RECIPE_MISSING",
    "RECIPE_SHA_MISMATCH",
    "RECIPE_CAD_MISSING",
    "RECIPE_CAD_HASH_MISMATCH",
    "COMPILED_PART_MISSING",
    "COMPILED_PART_HASH_MISMATCH",
    "COMPILE_BLOCKED",
    "SCAN_MISSING",
    "SCAN_HASH_MISMATCH",
    "WORKER_TIMEOUT",
    "WORKER_RSS_LIMIT",
    "WORKER_ERROR",
    "WORKER_CRASH",
    "WORKER_NO_RESULT",
    "ENGINE_OUTPUT_INVALID",
    "POSTPROCESS_TIMEOUT",
    "POSTPROCESS_ERROR",
    "WORKER_INTERRUPTED",
    "CANCELLED",
    "INTERNAL_ERROR",
]
FAILURE_CODES: frozenset[str] = frozenset(FailureCode.__args__)  # type: ignore[attr-defined]

ParseStatus = Literal["OK", "UNPARSEABLE", "UNMATCHED", "AMBIGUOUS", "REVISION_MISMATCH"]
PARSE_STATUSES: frozenset[str] = frozenset(ParseStatus.__args__)  # type: ignore[attr-defined]

#: MATCH: name revision equals the recipe's; MISMATCH: a revision-shaped token
#: differs (-> REVISION_MISMATCH); NOT_IN_NAME: no revision token; UNVERIFIED:
#: a revision token was read but no expected revision was known.
RevisionStatus = Literal["MATCH", "MISMATCH", "NOT_IN_NAME", "UNVERIFIED"]
REVISION_STATUSES: frozenset[str] = frozenset(RevisionStatus.__args__)  # type: ignore[attr-defined]

ResolveStatus = Literal["MATCHED", "UNMATCHED", "AMBIGUOUS", "UNPARSEABLE", "PART_NOT_READY", "REVISION_MISMATCH"]
RESOLVE_STATUSES: frozenset[str] = frozenset(ResolveStatus.__args__)  # type: ignore[attr-defined]

IngestState = Literal["received", "enqueued", "quarantined", "duplicate", "dismissed"]
INGEST_STATES: frozenset[str] = frozenset(IngestState.__args__)  # type: ignore[attr-defined]

QuarantineReason = Literal[
    "UNPARSEABLE_FILENAME",
    "UNMATCHED_SKU",
    "AMBIGUOUS_SKU",
    "REVISION_MISMATCH",
    "PART_NOT_READY",
    "UNSUPPORTED_FORMAT",
    "PLY_HEADER_INVALID",
    "PLY_TRUNCATED",
    "STL_INVALID",
    "EMPTY_SCAN",
    "SIDECAR_INVALID",
    "READ_ERROR",
]
QUARANTINE_REASONS: frozenset[str] = frozenset(QuarantineReason.__args__)  # type: ignore[attr-defined]

ScanFormat = Literal["PLY_ASCII", "PLY_BINARY_LE", "PLY_BINARY_BE", "STL_BINARY", "STL_ASCII"]
SCAN_FORMATS: frozenset[str] = frozenset(ScanFormat.__args__)  # type: ignore[attr-defined]

#: Higher runs first.
PRIORITY_BY_KIND: dict[str, int] = {"inspect": 100, "recipe_compile": 10}
#: Retries cover interruption only; timeouts and RSS breaches are never auto-retried.
MAX_ATTEMPTS_BY_KIND: dict[str, int] = {"inspect": 2, "recipe_compile": 1}
#: Engine step limits (plan §10 amendment 1; RSS 32 GiB by ENGINE_IO-A5, owner decision 2026-10-26).
ENGINE_HARD_TIMEOUT_S = 180
ENGINE_RSS_LIMIT_BYTES = 32 * 1024**3

JobId = Annotated[str, Field(pattern=JOB_ID_PATTERN)]
ScanFileId = Annotated[str, Field(pattern=SCAN_FILE_ID_PATTERN)]
Sku = Annotated[str, Field(pattern=SKU_PATTERN)]
RecipeId = Annotated[str, Field(pattern=RECIPE_ID_PATTERN)]


class DeclaredStageV1(ContractModel):
    """Optional per-scan stage declaration: ``stage_index`` bends formed.

    Never trusted or overridden silently: disagreement with the engine's
    inference is STAGE_MISMATCH (review). Never derived from filename tokens.
    """

    stage_index: int = Field(ge=0)
    source: Literal["SIDECAR", "OPERATOR", "API"]


class ScanNameParseV1(ContractModel):
    """Filename parse (domains.production.recipe.sku). Informational for the
    operator and the resolver only; none of it reaches the engine."""

    status: ParseStatus
    sku: Sku | None = None
    revision: str | None = Field(default=None, max_length=32)
    expected_revision: str | None = Field(default=None, max_length=32)
    revision_status: RevisionStatus | None = None
    serial: str | None = Field(default=None, max_length=64)
    scan_timestamp: str | None = Field(default=None, max_length=40)
    #: Descriptor words such as FULL_SCAN, SCAN_2, BEFORE_BENDING (display only).
    scan_label: str | None = Field(default=None, max_length=64)
    #: Trailing " N" export counter (e.g. "FULL_SCAN 1").
    export_index: int | None = Field(default=None, ge=0)
    #: "(N)" / "copy N" duplicate marker.
    copy_index: int | None = Field(default=None, ge=0)
    candidates: tuple[str, ...] = ()
    reason: str | None = Field(default=None, max_length=200)


class ScanMatchV1(ContractModel):
    status: ResolveStatus
    recipe_id: RecipeId | None = None
    recipe_version: int | None = Field(default=None, ge=1)
    recipe_sha256: Sha256Hex | None = None
    reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _validate(self) -> "ScanMatchV1":
        bound = (self.recipe_id, self.recipe_version, self.recipe_sha256)
        if self.status == "MATCHED" and any(v is None for v in bound):
            raise ValueError("a MATCHED scan names its recipe id, version and sha")
        return self


class ScanFileV1(ContractModel):
    schema_: Literal["sherman_scan_file_v1"] = Field(default=SCAN_FILE_SCHEMA, alias="schema")
    scan_file_id: ScanFileId
    sha256: Sha256Hex
    original_name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    station_id: str = Field(min_length=1, max_length=64)
    received_at: UtcIsoTimestamp
    stored_rel_path: RelPath
    format: ScanFormat | None = None
    vertex_count: int | None = Field(default=None, ge=0)
    face_count: int | None = Field(default=None, ge=0)
    bbox_mm: tuple[Vec3, Vec3] | None = None
    parse: ScanNameParseV1
    match: ScanMatchV1
    declared_stage: DeclaredStageV1 | None = None
    ingest_state: IngestState
    quarantine_reason: QuarantineReason | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ScanFileV1":
        if self.scan_file_id != f"scan_{self.sha256[:24]}":
            raise ValueError("scan_file_id must be 'scan_' + sha256[:24]")
        if (self.ingest_state == "quarantined") != (self.quarantine_reason is not None):
            raise ValueError("quarantine_reason is required exactly when quarantined")
        if self.ingest_state == "enqueued" and self.match.status != "MATCHED":
            raise ValueError("only MATCHED scans are enqueued")
        return self


class JobFailureV1(ContractModel):
    code: FailureCode
    message: str = Field(default="", max_length=2000)


class JobResultSummaryV1(ContractModel):
    inspection_validity: ValidityState
    summary_state: SummaryState
    bends_total: int = Field(ge=0)
    required_fail: int = Field(ge=0)
    required_review: int = Field(ge=0)
    required_unknown: int = Field(ge=0)
    stage_status: StageStatus | None = None


class LineJobV1(ContractModel):
    schema_: Literal["sherman_line_job_v1"] = Field(default=LINE_JOB_SCHEMA, alias="schema")
    job_id: JobId
    kind: JobKind
    station_id: str | None = Field(default=None, max_length=64)
    scan_file_id: ScanFileId | None = None
    scan_sha256: Sha256Hex | None = None
    sku: Sku
    recipe_id: RecipeId
    recipe_version: int = Field(ge=1)
    recipe_sha256: Sha256Hex
    compiled_part_sha256: Sha256Hex | None = None
    declared_stage: DeclaredStageV1 | None = None
    priority: int
    dedupe_key: str | None = Field(default=None, max_length=200)
    retry_of: JobId | None = None
    execution_status: ExecutionStatus
    stage: Stage
    progress: int = Field(ge=0, le=100)
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    lease_owner: str | None = Field(default=None, max_length=128)
    lease_expires_at: UtcIsoTimestamp | None = None
    cancel_requested: bool = False
    created_at: UtcIsoTimestamp
    started_at: UtcIsoTimestamp | None = None
    finished_at: UtcIsoTimestamp | None = None
    failure: JobFailureV1 | None = None
    engine_bundle_id: str | None = Field(default=None, max_length=128)
    workspace_rel: RelPath
    result_summary: JobResultSummaryV1 | None = None
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    duration_s: NonNegFloat | None = None

    @model_validator(mode="after")
    def _validate(self) -> "LineJobV1":
        if self.workspace_rel != f"jobs/{self.job_id}":
            raise ValueError("workspace_rel must be 'jobs/<job_id>'")
        if self.recipe_id != f"{self.sku}@v{self.recipe_version}":
            raise ValueError("recipe_id must be '<sku>@v<recipe_version>'")
        if self.kind == "inspect" and (self.scan_file_id is None or self.scan_sha256 is None):
            raise ValueError("inspect jobs name their scan")
        if self.kind == "inspect" and self.compiled_part_sha256 is None:
            # E3 cannot run without a compile; the runner checks this hash
            # (COMPILED_PART_HASH_MISMATCH) and passes it to --compiled-sha256.
            raise ValueError("inspect jobs pin their compiled_part_sha256")
        if self.scan_file_id is not None and self.scan_sha256 is not None and (
            self.scan_file_id != f"scan_{self.scan_sha256[:24]}"
        ):
            raise ValueError("scan_file_id must match scan_sha256")
        if (self.execution_status == "failed") != (self.failure is not None):
            raise ValueError("failure is present exactly when execution_status is failed")
        if self.execution_status == "completed" and self.stage != "done":
            raise ValueError("a completed job is at stage done")
        if self.attempts > self.max_attempts:
            raise ValueError("attempts cannot exceed max_attempts")
        return self


class LineJobSummaryV1(ContractModel):
    job_id: JobId
    kind: JobKind
    station_id: str | None = None
    sku: Sku
    part_name: str | None = None
    recipe_id: RecipeId
    recipe_version: int = Field(ge=1)
    scan_file_name: str | None = None
    serial: str | None = None
    execution_status: ExecutionStatus
    stage: Stage
    progress: int = Field(ge=0, le=100)
    created_at: UtcIsoTimestamp
    started_at: UtcIsoTimestamp | None = None
    finished_at: UtcIsoTimestamp | None = None
    failure_code: FailureCode | None = None
    summary_state: SummaryState | None = None
    inspection_validity: ValidityState | None = None
    stage_status: StageStatus | None = None
