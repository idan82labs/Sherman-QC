"""Line vocabularies duplicated from ``contracts/production`` (C01).

The line backbone (store, hotfolder, runner) must import cheaply and without
pydantic at module import time, so these literal sets are copied here instead
of imported. ``tests/test_prod_line_store.py`` and C11's
``tests/test_prod_contract_parity.py`` assert that every copy equals the
contract frozenset it mirrors. If a contract set changes, update the copy.
"""
from __future__ import annotations

__all__ = [
    "SCAN_FILE_SCHEMA",
    "LINE_JOB_SCHEMA",
    "LINE_EVENT_SCHEMA",
    "EXECUTION_STATUSES",
    "TERMINAL_STATUSES",
    "JOB_KINDS",
    "STAGES",
    "FAILURE_CODES",
    "PARSE_STATUSES",
    "RESOLVE_STATUSES",
    "INGEST_STATES",
    "QUARANTINE_REASONS",
    "CONTENT_QUARANTINE_REASONS",
    "RESOLVABLE_QUARANTINE_REASONS",
    "RESOLVE_STATUS_TO_QUARANTINE_REASON",
    "SCAN_FORMATS",
    "EVENT_KINDS",
    "SUMMARY_STATES",
    "VALIDITY_STATES",
    "STAGE_STATUSES",
    "PRIORITY_BY_KIND",
    "MAX_ATTEMPTS_BY_KIND",
    "ENGINE_HARD_TIMEOUT_S",
    "ENGINE_RSS_LIMIT_BYTES",
    "JOB_RESULT_SUMMARY_KEYS",
    "ENGINE_INPUT_ERROR_CODES",
]

# mirrors contracts/production/line_job.py (C11 parity test)
SCAN_FILE_SCHEMA = "sherman_scan_file_v1"
# mirrors contracts/production/line_job.py (C11 parity test)
LINE_JOB_SCHEMA = "sherman_line_job_v1"
# mirrors contracts/production/events.py (C11 parity test)
LINE_EVENT_SCHEMA = "sherman_line_event_v1"

# mirrors contracts/production/line_job.py EXECUTION_STATUSES (C11 parity test)
EXECUTION_STATUSES: frozenset[str] = frozenset({"queued", "running", "completed", "failed", "cancelled"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# mirrors contracts/production/line_job.py JOB_KINDS (C11 parity test)
JOB_KINDS: frozenset[str] = frozenset({"inspect", "recipe_compile"})

# mirrors contracts/production/line_job.py STAGES (C11 parity test)
STAGES: frozenset[str] = frozenset({"queued", "preparing", "engine", "postprocess", "done"})

# mirrors contracts/production/line_job.py FAILURE_CODES (C11 parity test)
FAILURE_CODES: frozenset[str] = frozenset(
    {
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
    }
)

# mirrors contracts/production/line_job.py PARSE_STATUSES (C11 parity test)
PARSE_STATUSES: frozenset[str] = frozenset({"OK", "UNPARSEABLE", "UNMATCHED", "AMBIGUOUS", "REVISION_MISMATCH"})

# mirrors contracts/production/line_job.py RESOLVE_STATUSES (C11 parity test)
RESOLVE_STATUSES: frozenset[str] = frozenset(
    {"MATCHED", "UNMATCHED", "AMBIGUOUS", "UNPARSEABLE", "PART_NOT_READY", "REVISION_MISMATCH"}
)

# mirrors contracts/production/line_job.py INGEST_STATES (C11 parity test)
INGEST_STATES: frozenset[str] = frozenset({"received", "enqueued", "quarantined", "duplicate", "dismissed"})

# mirrors contracts/production/line_job.py QUARANTINE_REASONS (C11 parity test)
QUARANTINE_REASONS: frozenset[str] = frozenset(
    {
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
    }
)

#: Reasons about the file bytes themselves. Such a scan can never be assigned
#: to a recipe: the operator must re-export it from the cell.
CONTENT_QUARANTINE_REASONS: frozenset[str] = frozenset(
    {
        "UNSUPPORTED_FORMAT",
        "PLY_HEADER_INVALID",
        "PLY_TRUNCATED",
        "STL_INVALID",
        "EMPTY_SCAN",
        "SIDECAR_INVALID",
        "READ_ERROR",
    }
)
#: Reasons about the filename-to-recipe match; an admin may assign a recipe.
RESOLVABLE_QUARANTINE_REASONS: frozenset[str] = QUARANTINE_REASONS - CONTENT_QUARANTINE_REASONS

#: Resolver outcome -> quarantine reason (MATCHED is never quarantined).
RESOLVE_STATUS_TO_QUARANTINE_REASON: dict[str, str] = {
    "UNPARSEABLE": "UNPARSEABLE_FILENAME",
    "UNMATCHED": "UNMATCHED_SKU",
    "AMBIGUOUS": "AMBIGUOUS_SKU",
    "REVISION_MISMATCH": "REVISION_MISMATCH",
    "PART_NOT_READY": "PART_NOT_READY",
}

# mirrors contracts/production/line_job.py SCAN_FORMATS (C11 parity test)
SCAN_FORMATS: frozenset[str] = frozenset({"PLY_ASCII", "PLY_BINARY_LE", "PLY_BINARY_BE", "STL_BINARY", "STL_ASCII"})

# mirrors contracts/production/events.py EVENT_KINDS (C11 parity test)
EVENT_KINDS: frozenset[str] = frozenset(
    {
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
    }
)

# mirrors contracts/production/states.py SUMMARY_STATES (C11 parity test)
SUMMARY_STATES: frozenset[str] = frozenset(
    {
        "ALL_MEASURED_IN_TOLERANCE",
        "STAGE_IN_TOLERANCE",
        "OUT_OF_TOLERANCE_FOUND",
        "REVIEW_REQUIRED",
        "INCOMPLETE",
        "INVALID_INSPECTION",
    }
)

# mirrors contracts/production/states.py VALIDITY_STATES (C11 parity test)
VALIDITY_STATES: frozenset[str] = frozenset(
    {
        "VALID",
        "INSUFFICIENT_CAPTURE",
        "UNTRUSTED_REGISTRATION",
        "UNIT_MISMATCH_SUSPECTED",
        "UNSUPPORTED_INPUT",
        "WRONG_PART_SUSPECTED",
        "MIRRORED_PART_SUSPECTED",
        "TIMEOUT",
    }
)

# mirrors contracts/production/states.py STAGE_STATUSES (C11 parity test)
STAGE_STATUSES: frozenset[str] = frozenset(
    {
        "CONSISTENT",
        "STAGE_MISMATCH",
        "OUT_OF_ORDER_SUSPECTED",
        "PARTIAL_BEND_FOUND",
        "FINAL_INCOMPLETE",
        "UNDETERMINED",
        "AMBIGUOUS_POSE",
        "NOT_EVALUATED",
    }
)

# mirrors contracts/production/line_job.py PRIORITY_BY_KIND (C11 parity test)
PRIORITY_BY_KIND: dict[str, int] = {"inspect": 100, "recipe_compile": 10}
# mirrors contracts/production/line_job.py MAX_ATTEMPTS_BY_KIND (C11 parity test)
MAX_ATTEMPTS_BY_KIND: dict[str, int] = {"inspect": 2, "recipe_compile": 1}
# mirrors contracts/production/line_job.py ENGINE_HARD_TIMEOUT_S (C11 parity test)
ENGINE_HARD_TIMEOUT_S = 180
# mirrors contracts/production/line_job.py ENGINE_RSS_LIMIT_BYTES (C11 parity test; ENGINE_IO-A5: 32 GiB)
ENGINE_RSS_LIMIT_BYTES = 32 * 1024**3

# mirrors contracts/production/line_job.py JobResultSummaryV1 fields (C11 parity test)
JOB_RESULT_SUMMARY_KEYS: tuple[str, ...] = (
    "inspection_validity",
    "summary_state",
    "bends_total",
    "required_fail",
    "required_review",
    "required_unknown",
    "stage_status",
)

#: E3 exit-2 codes (contracts/production/ENGINE_IO.md §4); each is also a FAILURE_CODE.
ENGINE_INPUT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "COMPILED_PART_MISSING",
        "COMPILED_PART_HASH_MISMATCH",
        "COMPILE_BLOCKED",
        "RECIPE_SHA_MISMATCH",
        "SCAN_MISSING",
        "SCAN_HASH_MISMATCH",
    }
)
