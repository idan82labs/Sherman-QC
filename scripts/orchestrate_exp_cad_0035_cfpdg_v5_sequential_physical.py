#!/usr/bin/env python3
"""One-shot physical primary/replay controller for EXP_CAD_0035.

The controller has no preparation or retry mode.  It verifies the immutable
execution authority, consumes the sole launch claim before starting a child,
and follows one of the two preregistered schedules:

* source-interface abstention: PRIMARY source, REPLAY source, no renderer;
* scientific completion: PRIMARY source/renderer, REPLAY source/renderer.

All wall/RSS telemetry and scan-access journals remain outside the compared
source and renderer roots.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Any, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPERIMENT_ID = "EXP_CAD_0035"
ARCHITECTURE = (
    "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_PHYSICAL_SEQUENTIAL_MEMBER_INTAKE"
)
ORCHESTRATION_SCHEMA = "EXP_CAD_0035_PHYSICAL_PRIMARY_REPLAY_RESULT_V1"
SOURCE_SCHEMA = "EXP_CAD_0035_PHYSICAL_SOURCE_ENVELOPE_V1"
INTERFACE_STATUS = "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
INTERFACE_TERMINAL_STATUS = (
    "COMPLETE_SOURCE_INTERFACE_ABSTENTION_REPLAY_EXACT_NO_RETRY"
)
SCIENCE_TERMINAL_STATUS = (
    "COMPLETE_SCIENTIFIC_PRIMARY_REPLAY_EXACT_AWAITING_NATIVE_REVIEW"
)
FAILURE_STATUS = "FAIL_TERMINAL_NO_RETRY"
CLAIM_SCHEMA = "EXP_CAD_0035_PHYSICAL_LAUNCH_CLAIM_V1"
NATIVE_MANIFEST_SCHEMA = "EXP_CAD_0030_CFPDG_NATIVE_VISUAL_MANIFEST_V1"
NATIVE_WIDTH = 3000
NATIVE_HEIGHT = 2400
SOURCE_WALL_LIMIT_SECONDS = 180.0
RENDERER_WALL_LIMIT_SECONDS = 60.0
RSS_LIMIT_BYTES = 4 * 1024**3
LOCK_RELATIVE_PATH = (
    "docs/research/2026-08-30_exp_cad_0035_physical_execution_lock.json"
)
LOCK_SCHEMA = "EXP_CAD_0035_PHYSICAL_EXECUTION_LOCK_V1"
LOCK_STATUS = "IMMUTABLE_LOCKED_AWAITING_DETACHED_APPROVAL"
SOURCE_CAPABILITY_SCHEMA = "EXP_CAD_0035_SOURCE_LAUNCH_CAPABILITY_V1"
SOURCE_CHILD_CLAIM_SCHEMA = "EXP_CAD_0035_SOURCE_LAUNCH_CHILD_CLAIM_V1"
SCAN_JOURNAL_SCHEMA = "EXP_CAD_0035_SCAN_ACCOUNTING_JOURNAL_V1"
LIFECYCLE_MEASUREMENT_SCHEMA = "EXP_CAD_0035_SOURCE_LIFECYCLE_MEASUREMENT_V1"
REQUIRED_RELEASE_EVENTS = [
    "N009_RELEASE_VERIFIED_BEFORE_N010_LOAD",
    "N010_RELEASE_VERIFIED_BEFORE_PAIR_ACTIONABILITY",
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _null_canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _null_canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _null_canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_write(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def _uninitialized_dependency(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("EXP35 bound runtime dependency was not bootstrap-verified")


verify_execution_authority = _uninitialized_dependency
run_measured_command = _uninitialized_dependency
_BOUND_RUNTIME_INITIALIZED = False
_SOURCE_FREE_TEST_INITIALIZED = False


def initialize_source_free_test_dependencies(*, authority_verifier: Any,
                                             measured_runner: Any) -> None:
    """Explicit source-free-only dependency injection; never used physically."""

    global verify_execution_authority, run_measured_command
    global _BOUND_RUNTIME_INITIALIZED, _SOURCE_FREE_TEST_INITIALIZED
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("EXP35 source-free dependency injection is pytest-only")
    if not callable(authority_verifier) or not callable(measured_runner):
        raise TypeError("source-free test dependencies must be callable")
    verify_execution_authority = authority_verifier
    run_measured_command = measured_runner
    _BOUND_RUNTIME_INITIALIZED = True
    _SOURCE_FREE_TEST_INITIALIZED = True


def _verify_bootstrap_row(lock: Mapping[str, Any], key: str,
                          expected: Path) -> None:
    row = lock.get("bound_files", {}).get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"EXP35 bootstrap lock lacks {key}")
    relative = row.get("path")
    if not isinstance(relative, str):
        raise ValueError(f"EXP35 bootstrap row {key} lacks path")
    canonical = (ROOT / relative).resolve(strict=True)
    expected = expected.resolve(strict=True)
    if canonical != expected or str(ROOT / relative) != str(expected):
        raise ValueError(f"EXP35 bootstrap path mismatch: {key}")
    if expected.is_symlink() or not expected.is_file():
        raise ValueError(f"EXP35 bootstrap dependency is not a regular file: {key}")
    if row.get("size_bytes") != expected.stat().st_size:
        raise ValueError(f"EXP35 bootstrap size mismatch: {key}")
    if row.get("sha256") != sha256_path(expected):
        raise ValueError(f"EXP35 bootstrap hash mismatch: {key}")


def _initialize_bound_runtime(args: argparse.Namespace) -> Mapping[str, Any] | None:
    """Stdlib-only lock bootstrap before importing either bound dependency."""

    global verify_execution_authority, run_measured_command
    global _BOUND_RUNTIME_INITIALIZED
    if _SOURCE_FREE_TEST_INITIALIZED:
        return None
    if _BOUND_RUNTIME_INITIALIZED:
        raise RuntimeError("EXP35 physical runtime may be initialized only once")
    expected_lock = (ROOT / LOCK_RELATIVE_PATH).resolve(strict=True)
    supplied_lock = Path(args.lock)
    if not supplied_lock.is_absolute() or str(supplied_lock) != str(expected_lock):
        raise ValueError("EXP35 bootstrap lock path is not exact")
    if not isinstance(args.lock_sha256, str) or len(args.lock_sha256) != 64:
        raise ValueError("EXP35 bootstrap lock digest is malformed")
    if sha256_path(expected_lock) != args.lock_sha256:
        raise ValueError("EXP35 bootstrap lock digest mismatch")
    lock = _load_json(expected_lock, label="EXP35 bootstrap lock")
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("status") != LOCK_STATUS
    ):
        raise ValueError("EXP35 bootstrap lock identity/status mismatch")
    orchestrator = Path(__file__).resolve(strict=True)
    contract = (ROOT / "scripts/exp_cad_0035_execution_contract.py").resolve(strict=True)
    monitor = (ROOT / "scripts/exp_cad_0035_process_monitor.py").resolve(strict=True)
    _verify_bootstrap_row(lock, "exp35_orchestrator", orchestrator)
    _verify_bootstrap_row(lock, "exp35_contract", contract)
    _verify_bootstrap_row(lock, "exp35_monitor", monitor)
    if str(args.orchestrator) != str(orchestrator) or str(args.process_monitor) != str(monitor):
        raise ValueError("EXP35 bootstrap CLI dependency path mismatch")
    contract_module = importlib.import_module("scripts.exp_cad_0035_execution_contract")
    monitor_module = importlib.import_module("scripts.exp_cad_0035_process_monitor")
    if Path(contract_module.__file__).resolve(strict=True) != contract:
        raise ValueError("EXP35 imported contract module path mismatch")
    if Path(monitor_module.__file__).resolve(strict=True) != monitor:
        raise ValueError("EXP35 imported monitor module path mismatch")
    if (
        monitor_module.SOURCE_WALL_LIMIT_SECONDS != SOURCE_WALL_LIMIT_SECONDS
        or monitor_module.RENDERER_WALL_LIMIT_SECONDS != RENDERER_WALL_LIMIT_SECONDS
        or monitor_module.RSS_LIMIT_BYTES != RSS_LIMIT_BYTES
    ):
        raise ValueError("EXP35 monitor resource constants differ from preregistration")
    verify_execution_authority = contract_module.verify_execution_authority
    run_measured_command = monitor_module.run_measured_command
    _BOUND_RUNTIME_INITIALIZED = True
    return lock


# The source runner consumes these exact static path flags.  Keys are also the
# lock-binding names supplied to the shared authority verifier.
SOURCE_STATIC_ARGUMENTS: tuple[tuple[str, str], ...] = (
    ("parent_preregistration", "parent-preregistration"),
    ("exp31_preregistration", "exp31-preregistration"),
    ("v5_preregistration", "v5-preregistration"),
    ("cad_preflight", "cad-preflight"),
    ("nodes", "nodes"),
    ("edges", "edges"),
    ("edge_geodesic_mm", "edge-geodesic-mm"),
    ("mesh", "mesh"),
    ("scan_first", "scan-first"),
    ("scan_second", "scan-second"),
    ("transform_first", "transform-first"),
    ("transform_second", "transform-second"),
)

PATH_ARGUMENTS: tuple[str, ...] = (
    "preregistration",
    "lock",
    "detached_approval",
    "source_runner",
    "process_monitor",
    "orchestrator",
    "renderer",
    *(name for name, _ in SOURCE_STATIC_ARGUMENTS),
    "attempt_root",
    "primary_source_root",
    "replay_source_root",
    "primary_measurement_root",
    "replay_measurement_root",
    "primary_renderer_root",
    "replay_renderer_root",
    "launch_claim",
    "result",
)

ACCOUNTING_KEYS = (
    "physical_scan_hash_call_count",
    "physical_scan_point_cloud_load_call_count",
    "physical_scan_byte_open_operation_count",
)
SUCCESSFUL_CHILD_ACCOUNTING = {
    "physical_scan_hash_call_count": 2,
    "physical_scan_point_cloud_load_call_count": 2,
    "physical_scan_byte_open_operation_count": 4,
}


class TerminalExecutionError(RuntimeError):
    """Raised only after a consumed attempt publishes its terminal failure."""


def _canonical_absolute(path: Path, *, must_exist: bool, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = raw.resolve(strict=must_exist)
    if str(raw) != str(resolved):
        raise ValueError(f"{label} must use its canonical absolute spelling")
    return resolved


def _strictly_beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return path != parent
    except ValueError:
        return False


def _refuse_nonempty_root(path: Path, *, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise ValueError(f"{label} must be absent or an empty directory")


def validate_output_topology(paths: Mapping[str, Path]) -> dict[str, Path]:
    """Validate disjoint compared/non-compared roots before claim creation."""

    required = {
        "attempt_root",
        "primary_source_root",
        "replay_source_root",
        "primary_measurement_root",
        "replay_measurement_root",
        "primary_renderer_root",
        "replay_renderer_root",
        "launch_claim",
        "result",
        "manual_native_review",
        "global_result",
    }
    if set(paths) != required:
        raise ValueError("EXP35 output topology keys differ from the frozen controller")
    resolved = {
        key: _canonical_absolute(value, must_exist=False, label=key)
        for key, value in paths.items()
    }
    attempt = resolved["attempt_root"]
    roots = [
        resolved["primary_source_root"],
        resolved["replay_source_root"],
        resolved["primary_measurement_root"],
        resolved["replay_measurement_root"],
        resolved["primary_renderer_root"],
        resolved["replay_renderer_root"],
    ]
    if len(set(roots)) != len(roots):
        raise ValueError("EXP35 output roots must be mutually distinct")
    if any(not _strictly_beneath(root, attempt) for root in roots):
        raise ValueError("every EXP35 child root must be strictly beneath attempt root")
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _strictly_beneath(first, second) or _strictly_beneath(second, first):
                raise ValueError("EXP35 output roots may not have ancestor overlap")
    for name in ("launch_claim", "result", "manual_native_review"):
        value = resolved[name]
        if value.parent != attempt:
            raise ValueError(f"{name} must be a direct child of attempt root")
        if any(value == root or _strictly_beneath(value, root) for root in roots):
            raise ValueError(f"{name} must remain outside child roots")
    if resolved["launch_claim"].name != "EXP_CAD_0035_LAUNCH.claim.json":
        raise ValueError("EXP35 launch-claim basename mismatch")
    if resolved["result"].name != "EXP_CAD_0035_ORCHESTRATION_RESULT.json":
        raise ValueError("EXP35 orchestration-result basename mismatch")
    if resolved["manual_native_review"].name != (
        "EXP_CAD_0035_MANUAL_NATIVE_REVIEW.json"
    ):
        raise ValueError("EXP35 manual-native-review basename mismatch")
    expected_global = (
        ROOT / "output/research/autoresearch-loop/EXP_CAD_0035_RESULT.json"
    ).resolve(strict=False)
    if resolved["global_result"] != expected_global:
        raise ValueError("EXP35 global-result path mismatch")
    if resolved["global_result"] == attempt or _strictly_beneath(
        resolved["global_result"], attempt
    ):
        raise ValueError("EXP35 global result must remain outside attempt namespace")
    for key in (
        "primary_source_root",
        "replay_source_root",
        "primary_measurement_root",
        "replay_measurement_root",
        "primary_renderer_root",
        "replay_renderer_root",
    ):
        _refuse_nonempty_root(resolved[key], label=key)
    if resolved["result"].exists():
        raise ValueError("EXP35 orchestration result already exists")
    if resolved["manual_native_review"].exists():
        raise ValueError("EXP35 manual native review already exists")
    if resolved["global_result"].exists():
        raise ValueError("EXP35 global result already exists")
    if resolved["launch_claim"].exists():
        raise ValueError("EXP35 launch claim already exists; attempt is permanently consumed")
    return resolved


def validate_attempt_namespace_preclaim(paths: Mapping[str, Path]) -> None:
    """Allow only the absent-or-empty authorized leaf topology pre-claim."""

    attempt = paths["attempt_root"]
    if not attempt.exists():
        return
    if not attempt.is_dir() or attempt.is_symlink():
        raise ValueError("EXP35 attempt namespace is not a regular directory")
    leaves = {
        paths["primary_source_root"],
        paths["replay_source_root"],
        paths["primary_measurement_root"],
        paths["replay_measurement_root"],
        paths["primary_renderer_root"],
        paths["replay_renderer_root"],
    }
    for entry in attempt.rglob("*"):
        resolved = entry.resolve(strict=True)
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("EXP35 attempt namespace contains a preexisting file or symlink")
        if not any(resolved == leaf or resolved in leaf.parents for leaf in leaves):
            raise ValueError("EXP35 attempt namespace contains an unauthorized directory")


def _strict_regular_tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    """Inventory a tree without following a symlink or accepting specials."""

    root = Path(root)
    if not root.is_absolute():
        raise ValueError(f"deterministic tree path is not absolute: {root}")
    cursor = root
    while True:
        if cursor.is_symlink():
            raise ValueError(f"deterministic tree path contains a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if not root.is_dir():
        raise ValueError(f"deterministic tree is missing or invalid: {root}")
    files: set[str] = set()
    directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"deterministic tree contains a symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            files.add(relative)
        elif stat.S_ISDIR(mode):
            directories.add(relative)
        else:
            raise ValueError(f"deterministic tree contains a special file: {path}")
    return files, directories


def tree_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    files, _directories = _strict_regular_tree_inventory(root)
    if not files:
        raise ValueError(f"deterministic tree is empty: {root}")
    return {relative: sha256_path(root / relative) for relative in sorted(files)}


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"{label} is missing or invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _source_result(root: Path, measurement_root: Path) -> Mapping[str, Any]:
    root = Path(root)
    actual_files, actual_directories = _strict_regular_tree_inventory(root)
    if actual_directories:
        raise ValueError("EXP35 deterministic source tree contains a directory")
    result = _load_json(root / "SOURCE_RESULT.json", label="EXP35 source result")
    if (
        result.get("schema") != SOURCE_SCHEMA
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("architecture") != ARCHITECTURE
        or not isinstance(result.get("scientific_execute_source_reached"), bool)
    ):
        raise ValueError("EXP35 source result identity is invalid")
    visual = root / "SOURCE_VISUAL_PAYLOAD.json"
    base_files = {
        "SOURCE_RESULT.json",
        "FIRST_SOURCE_INTERFACE_REPORT.json",
        "SECOND_SOURCE_INTERFACE_REPORT.json",
        "PAIR_ACTIONABILITY.json",
        "RETAINED_IDENTITY_HASHES.json",
        "ENGINE_INVOCATION_ACCOUNTING.json",
    }
    science_files = {
        "SOURCE_SEMANTIC_PAYLOAD.json",
        "SOURCE_VISUAL_PAYLOAD.json",
        "EXP32_ENGINE_COUNTER_PROJECTION.json",
    }
    expected_files = base_files | (
        science_files if result["scientific_execute_source_reached"] else set()
    )
    if actual_files != expected_files:
        raise ValueError("EXP35 deterministic source-tree inventory mismatch")
    accounting = _load_json(
        root / "ENGINE_INVOCATION_ACCOUNTING.json",
        label="EXP35 engine-invocation accounting",
    )
    if accounting.get("schema") != "EXP_CAD_0034_ENGINE_INVOCATION_ACCOUNTING_V1":
        raise ValueError("EXP35 engine-invocation accounting schema mismatch")
    if result["scientific_execute_source_reached"]:
        if result.get("status") == INTERFACE_STATUS or not visual.is_file():
            raise ValueError("scientific source completion lacks an honest visual payload")
        metrics = _load_json(
            measurement_root / "EXP32_ENGINE_METRICS.json",
            label="complete EXP32 engine metrics",
        )
        if (
            metrics.get("schema") != "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1"
            or metrics.get("experiment_id") != "EXP_CAD_0032"
            or accounting.get("v5_engine_invocation_count") != 1
            or accounting.get("unique_member_fit_count") != 38
            or accounting.get("kd_node_query_count") != 306926
            or accounting.get("robustness_row_count") != 32
            or not isinstance(accounting.get("local_fit_call_count"), int)
            or accounting.get("local_fit_call_count") <= 0
            or accounting.get("exp32_engine_metrics_payload_present") is not True
        ):
            raise ValueError("EXP35 scientific engine identity/counter gate failed")
    else:
        if result.get("status") != INTERFACE_STATUS:
            raise ValueError("pre-science result has an unexpected nested status")
        if visual.exists():
            raise ValueError("interface abstention must not publish a visual payload")
        for key in (
            "v5_engine_invocation_count",
            "unique_member_fit_count",
            "kd_node_query_count",
            "local_fit_call_count",
            "robustness_row_count",
        ):
            if accounting.get(key) != 0:
                raise ValueError("interface abstention has nonzero engine accounting")
        if accounting.get("exp32_engine_metrics_payload_present") is not False:
            raise ValueError("interface abstention falsely advertises EXP32 metrics")
        if (measurement_root / "EXP32_ENGINE_METRICS.json").exists():
            raise ValueError("interface abstention published forbidden EXP32 metrics")
    nested = result.get("nested_payload_sha256")
    artifact_map = {
        "first_source_interface_report_sha256": "FIRST_SOURCE_INTERFACE_REPORT.json",
        "second_source_interface_report_sha256": "SECOND_SOURCE_INTERFACE_REPORT.json",
        "pair_actionability_sha256": "PAIR_ACTIONABILITY.json",
        "retained_identity_hashes_sha256": "RETAINED_IDENTITY_HASHES.json",
        "engine_invocation_accounting_sha256": "ENGINE_INVOCATION_ACCOUNTING.json",
    }
    if result["scientific_execute_source_reached"]:
        artifact_map.update(
            {
                "scientific_semantic_payload_sha256": "SOURCE_SEMANTIC_PAYLOAD.json",
                "scientific_visual_payload_sha256": "SOURCE_VISUAL_PAYLOAD.json",
                "exp32_engine_counter_projection_sha256": "EXP32_ENGINE_COUNTER_PROJECTION.json",
            }
        )
    if not isinstance(nested, Mapping) or set(nested) != set(artifact_map):
        raise ValueError("EXP35 outer envelope nested-hash inventory mismatch")
    for field, filename in artifact_map.items():
        if nested.get(field) != sha256_path(root / filename):
            raise ValueError(f"EXP35 outer envelope nested hash mismatch: {field}")
    return result


def _validate_lifecycle_measurement(
    measurement_root: Path, *, mode: str
) -> Mapping[str, Any]:
    """Require the exact successful sequential-lifetime proof."""

    row = _load_json(
        measurement_root / "SOURCE_LIFECYCLE_MEASUREMENT.json",
        label=f"{mode} EXP35 source lifecycle measurement",
    )
    expected_keys = {
        "schema",
        "experiment_id",
        "mode",
        "status",
        "explicit_gc_collect_call_count",
        "maximum_live_source_cloud_count",
        "maximum_live_unfiltered_member_bundle_count",
        "N009_release_verified",
        "N010_release_verified",
        "ordered_events",
    }
    if set(row) != expected_keys:
        raise ValueError("EXP35 lifecycle measurement field inventory mismatch")
    if (
        row.get("schema") != LIFECYCLE_MEASUREMENT_SCHEMA
        or row.get("experiment_id") != EXPERIMENT_ID
        or row.get("mode") != mode
        or row.get("status") != "COMPLETE_RELEASE_VERIFIED"
        or row.get("explicit_gc_collect_call_count") != 2
        or row.get("maximum_live_source_cloud_count") != 1
        or row.get("maximum_live_unfiltered_member_bundle_count") != 1
        or row.get("N009_release_verified") is not True
        or row.get("N010_release_verified") is not True
        or row.get("ordered_events") != REQUIRED_RELEASE_EVENTS
    ):
        raise ValueError("EXP35 successful source lifecycle proof is not exact")
    return row


def _accounting_from_journal(
    measurement_root: Path, *, mode: str
) -> dict[str, int]:
    journal = _load_json(
        measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json",
        label="EXP35 durable scan-accounting journal",
    )
    if (
        journal.get("schema") != SCAN_JOURNAL_SCHEMA
        or journal.get("experiment_id") != EXPERIMENT_ID
        or journal.get("mode") != mode
        or journal.get(
            "evaluator_role_label_gold_quality_or_expected_count_access_count"
        )
        != 0
    ):
        raise ValueError("EXP35 scan-accounting journal identity/access gate failed")
    result: dict[str, int] = {}
    for key in ACCOUNTING_KEYS:
        value = journal.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"EXP35 scan-accounting journal has invalid {key}")
        result[key] = value
    expected_operations = [
        ("N009", "SHA256"),
        ("N010", "SHA256"),
        ("N009", "POINT_CLOUD_LOAD"),
        ("N010", "POINT_CLOUD_LOAD"),
    ]
    attempted = journal.get("attempted_operations")
    if not isinstance(attempted, list) or len(attempted) > len(expected_operations):
        raise ValueError("EXP35 scan-accounting attempted-operation list is invalid")
    for ordinal, (row, expected) in enumerate(
        zip(attempted, expected_operations), start=1
    ):
        if not isinstance(row, Mapping) or row != {
            "attempt_ordinal": ordinal,
            "opaque_token": expected[0],
            "operation": expected[1],
        }:
            raise ValueError("EXP35 scan operation order/payload is not exact")
    derived = {
        "physical_scan_hash_call_count": sum(
            row["operation"] == "SHA256" for row in attempted
        ),
        "physical_scan_point_cloud_load_call_count": sum(
            row["operation"] == "POINT_CLOUD_LOAD" for row in attempted
        ),
        "physical_scan_byte_open_operation_count": len(attempted),
    }
    if result != derived:
        raise ValueError("EXP35 scan counters disagree with durable operations")
    return result


def _safe_accounting(
    measurement_root: Path, *, mode: str
) -> tuple[dict[str, int], str | None]:
    journal = measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    if not journal.is_file():
        return ({key: 0 for key in ACCOUNTING_KEYS}, None)
    try:
        return (_accounting_from_journal(measurement_root, mode=mode), None)
    except Exception as error:
        return ({key: 0 for key in ACCOUNTING_KEYS}, str(error))


def _require_successful_source_accounting(
    measurement_root: Path, *, mode: str
) -> None:
    observed = _accounting_from_journal(measurement_root, mode=mode)
    if observed != SUCCESSFUL_CHILD_ACCOUNTING:
        raise ValueError(
            "successful source child scan accounting differs from exact 2/2/4"
        )
    journal = _load_json(
        measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json",
        label="successful EXP35 durable scan-accounting journal",
    )
    expected_keys = {
        "schema",
        "experiment_id",
        "mode",
        "status",
        "stage",
        *ACCOUNTING_KEYS,
        "attempted_operations",
        "evaluator_role_label_gold_quality_or_expected_count_access_count",
    }
    if (
        set(journal) != expected_keys
        or journal.get("status") != "COMPLETE"
        or journal.get("stage") != "COMPLETE"
    ):
        raise ValueError("successful EXP35 scan journal is not terminally exact")


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"native asset is not a PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def validate_native_renderer_tree(root: Path) -> Mapping[str, Any]:
    """Validate the renderer-declared native inventory without relabeling it."""

    root = Path(root)
    actual_files, actual_directories = _strict_regular_tree_inventory(root)
    manifest_path = root / "native_visual_manifest.json"
    manifest = _load_json(manifest_path, label="EXP30 native visual manifest")
    if (
        manifest.get("schema") != NATIVE_MANIFEST_SCHEMA
        or manifest.get("experiment_id") != "EXP_CAD_0030"
        or manifest.get("native_resolution")
        != {"width_px": NATIVE_WIDTH, "height_px": NATIVE_HEIGHT}
    ):
        raise ValueError("EXP30 native visual manifest identity/resolution mismatch")
    rows: list[Mapping[str, Any]] = []
    global_asset = manifest.get("global_asset")
    if not isinstance(global_asset, Mapping):
        raise ValueError("native manifest lacks global asset")
    rows.append(global_asset)
    for field in ("component_assets", "node_assets"):
        value = manifest.get(field)
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise ValueError(f"native manifest has invalid {field}")
        rows.extend(value)
    declared: set[str] = set()
    for row in rows:
        relative_text = row.get("path")
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("native manifest asset path is invalid")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError("native manifest asset path is not normalized relative")
        if relative.suffix.lower() != ".png" or relative.as_posix() in declared:
            raise ValueError("native manifest has duplicate or non-PNG asset")
        declared.add(relative.as_posix())
        # Keep this path lexical.  Resolving first would erase the evidence that
        # a declared leaf is a symlink and could incorrectly authorize replay.
        asset = root / relative
        if asset.is_symlink() or not asset.is_file():
            raise ValueError("native asset is missing, non-regular, or a symlink")
        if row.get("sha256") != sha256_path(asset):
            raise ValueError("native asset hash differs from renderer manifest")
        if row.get("width_px") != NATIVE_WIDTH or row.get("height_px") != NATIVE_HEIGHT:
            raise ValueError("native asset declared dimensions are not exact")
        if _png_dimensions(asset) != (NATIVE_WIDTH, NATIVE_HEIGHT):
            raise ValueError("native asset intrinsic dimensions are not exact")
    expected_files = {"native_visual_manifest.json", *declared}
    expected_directories = {
        parent.as_posix()
        for relative_text in declared
        for parent in Path(relative_text).parents
        if parent != Path(".")
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("native renderer tree inventory is not exact")
    return manifest


def _source_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output_root: Path,
    measurement_root: Path,
    launch_claim_sha256: str,
    capability_path: Path,
    child_claim_path: Path,
) -> list[str]:
    journal = measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    lifecycle = measurement_root / "SOURCE_LIFECYCLE_MEASUREMENT.json"
    command = [
        str(Path(sys.executable)),
        str(args.source_runner),
        "--mode",
        mode,
        "--preregistration",
        str(args.preregistration),
        "--lock",
        str(args.lock),
        "--lock-sha256",
        args.lock_sha256,
        "--detached-approval",
        str(args.detached_approval),
        "--detached-approval-sha256",
        args.detached_approval_sha256,
        "--launch-claim",
        str(args.launch_claim),
        "--launch-claim-sha256",
        launch_claim_sha256,
        "--source-launch-capability",
        str(capability_path),
        "--source-launch-child-claim",
        str(child_claim_path),
    ]
    for attribute, flag in SOURCE_STATIC_ARGUMENTS:
        command.extend((f"--{flag}", str(getattr(args, attribute))))
    command.extend(
        (
            "--output-root",
            str(output_root),
            "--measurement-root",
            str(measurement_root),
            "--scan-accounting-journal",
            str(journal),
            "--lifecycle-measurement",
            str(lifecycle),
        )
    )
    return command


def _source_capability_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    output_root: Path,
    measurement_root: Path,
    command: Sequence[str],
    launch_claim_sha256: str,
    capability_path: Path,
    child_claim_path: Path,
) -> dict[str, Any]:
    return {
        "schema": SOURCE_CAPABILITY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUTHORIZED_NOT_CLAIMED",
        "source_child_ordinal": 1 if mode == "PRIMARY" else 2,
        "mode": mode,
        "command_argv": list(command),
        "command_sha256": hashlib.sha256(canonical_bytes(list(command))).hexdigest(),
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": sha256_path(args.preregistration),
        "lock_path": str(args.lock),
        "lock_sha256": args.lock_sha256,
        "detached_approval_path": str(args.detached_approval),
        "detached_approval_sha256": args.detached_approval_sha256,
        "launch_claim_path": str(args.launch_claim),
        "launch_claim_sha256": launch_claim_sha256,
        "output_root": str(output_root),
        "measurement_root": str(measurement_root),
        "scan_accounting_journal": str(
            measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
        ),
        "lifecycle_measurement": str(
            measurement_root / "SOURCE_LIFECYCLE_MEASUREMENT.json"
        ),
        "source_launch_capability_path": str(capability_path),
        "source_launch_child_claim_path": str(child_claim_path),
        "evaluator_input_authorized": False,
    }


def _verify_source_child_claim(
    *,
    child_claim_path: Path,
    capability_path: Path,
    capability_sha256: str,
    launch_claim_path: Path,
    launch_claim_sha256: str,
    command: Sequence[str],
    mode: str,
    observed_process_tree_pids: Any,
) -> Mapping[str, Any]:
    row = _load_json(child_claim_path, label=f"{mode} source launch child claim")
    expected_command_sha = hashlib.sha256(canonical_bytes(list(command))).hexdigest()
    expected_fields = {
        "schema",
        "experiment_id",
        "status",
        "source_child_ordinal",
        "source_process_pid",
        "mode",
        "source_launch_capability_path",
        "source_launch_capability_sha256",
        "command_sha256",
        "launch_claim_path",
        "launch_claim_sha256",
    }
    source_process_pid = row.get("source_process_pid")
    if not isinstance(observed_process_tree_pids, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in observed_process_tree_pids
    ):
        raise ValueError(f"{mode} measured process-tree PID inventory is invalid")
    if observed_process_tree_pids != sorted(set(observed_process_tree_pids)):
        raise ValueError(f"{mode} measured process-tree PID inventory is invalid")
    if (
        set(row) != expected_fields
        or row.get("schema") != SOURCE_CHILD_CLAIM_SCHEMA
        or row.get("experiment_id") != EXPERIMENT_ID
        or row.get("status") != "CLAIMED_BEFORE_STATIC_OR_SCAN_ACCESS"
        or row.get("source_child_ordinal") != (1 if mode == "PRIMARY" else 2)
        or isinstance(source_process_pid, bool)
        or not isinstance(source_process_pid, int)
        or source_process_pid <= 0
        or source_process_pid not in observed_process_tree_pids
        or row.get("mode") != mode
        or row.get("source_launch_capability_path") != str(capability_path)
        or row.get("source_launch_capability_sha256") != capability_sha256
        or row.get("command_sha256") != expected_command_sha
        or row.get("launch_claim_path") != str(launch_claim_path)
        or row.get("launch_claim_sha256") != launch_claim_sha256
    ):
        raise ValueError(f"{mode} source launch child claim binding mismatch")
    return row


def _renderer_command(args: argparse.Namespace, *, source_root: Path,
                      renderer_root: Path) -> list[str]:
    bootstrap = (
        "import importlib.util,json,sys;"
        "from pathlib import Path;"
        "p=Path(sys.argv[1]).resolve(strict=True);"
        "s=importlib.util.spec_from_file_location('exp35_bound_exp30_renderer',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "x=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));"
        "m.render_native_visuals(x,Path(sys.argv[3]))"
    )
    return [
        sys.executable,
        "-c",
        bootstrap,
        str(args.renderer),
        str(source_root / "SOURCE_VISUAL_PAYLOAD.json"),
        str(renderer_root),
    ]


def _measurement_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "status",
            "returncode",
            "timed_out",
            "rss_limit_exceeded",
            "sampler_failed",
            "resource_kill_reason",
            "wall_seconds",
            "wall_limit_seconds",
            "peak_process_tree_rss_bytes",
            "rss_limit_bytes",
            "process_tree_sample_count",
            "observed_process_tree_pids",
        )
    }


def _reverify_bound_file(
    lock: Mapping[str, Any], *, key: str, supplied: Path | None = None
) -> None:
    bound = lock.get("bound_files")
    if not isinstance(bound, Mapping) or not bound:
        raise ValueError("EXP35 pre-spawn lock has no bound-file inventory")
    row = bound.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"EXP35 pre-spawn lock lacks bound file {key}")
    if set(row) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"EXP35 pre-spawn lock row is malformed: {key}")
    raw_value = row.get("path")
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"EXP35 pre-spawn lock row lacks a path: {key}")
    raw_path = Path(raw_value)
    lexical = raw_path if raw_path.is_absolute() else ROOT / raw_path
    if lexical.is_symlink():
        raise ValueError(f"EXP35 pre-spawn bound path became a symlink: {key}")
    expected = lexical.resolve(strict=True)
    if str(lexical) != str(expected) or not expected.is_file():
        raise ValueError(f"EXP35 pre-spawn bound path is not a regular file: {key}")
    if supplied is not None:
        supplied = Path(supplied)
        if str(supplied) != str(expected) or supplied.resolve(strict=True) != expected:
            raise ValueError(f"EXP35 pre-spawn path drift: {key}")
    if row.get("size_bytes") != expected.stat().st_size:
        raise ValueError(f"EXP35 pre-spawn size drift: {key}")
    if row.get("sha256") != sha256_path(expected):
        raise ValueError(f"EXP35 pre-spawn hash drift: {key}")


def _require_absent_authority_file(path: Path, *, label: str) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"EXP35 stale {label} exists before authorization")


def _reverify_global_authority_state(
    args: argparse.Namespace,
    *,
    lock: Mapping[str, Any],
    launch_claim_sha256: str,
    verified_lock_payload_sha256: str,
) -> None:
    """Revalidate every non-scan lock row and global authority byte."""

    for path, label in (
        (args.lock, "immutable lock"),
        (args.detached_approval, "detached approval"),
        (args.launch_claim, "global launch claim"),
    ):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"EXP35 {label} is not a regular file before spawn")
    if sha256_path(args.lock) != args.lock_sha256:
        raise ValueError("EXP35 immutable lock changed after authority verification")
    if hashlib.sha256(canonical_bytes(lock)).hexdigest() != (
        verified_lock_payload_sha256
    ):
        raise ValueError("EXP35 verified original-lock payload changed in memory")
    if sha256_path(args.detached_approval) != args.detached_approval_sha256:
        raise ValueError("EXP35 detached approval changed after authority verification")
    if sha256_path(args.launch_claim) != launch_claim_sha256:
        raise ValueError("EXP35 launch claim changed after attempt consumption")
    bound = lock.get("bound_files")
    if not isinstance(bound, Mapping) or not bound:
        raise ValueError("EXP35 original lock has no non-scan bound files")
    for key in sorted(bound):
        _reverify_bound_file(lock, key=key)
    for key, supplied in (
        ("exp35_orchestrator", args.orchestrator),
        ("exp35_contract", ROOT / "scripts/exp_cad_0035_execution_contract.py"),
        ("exp35_monitor", args.process_monitor),
    ):
        _reverify_bound_file(lock, key=key, supplied=supplied)
    environment = lock.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("EXP35 original lock lacks the runtime environment")
    invocation = Path(sys.executable)
    realpath = invocation.resolve(strict=True)
    if (
        str(invocation) != environment.get("python_executable_invocation_path")
        or str(realpath) != environment.get("python_executable_realpath")
        or sha256_path(realpath) != environment.get("python_executable_sha256")
    ):
        raise ValueError("EXP35 Python invocation/realpath changed before spawn")


def _reverify_before_spawn(
    args: argparse.Namespace,
    *,
    lock: Mapping[str, Any],
    kind: str,
    launch_claim_sha256: str,
    verified_lock_payload_sha256: str,
    capability_path: Path | None = None,
    capability_payload: Mapping[str, Any] | None = None,
    capability_sha256: str | None = None,
    child_claim_path: Path | None = None,
) -> None:
    """Close every static/capability/claim mutation window before a child."""

    _reverify_global_authority_state(
        args,
        lock=lock,
        launch_claim_sha256=launch_claim_sha256,
        verified_lock_payload_sha256=verified_lock_payload_sha256,
    )
    if kind == "SOURCE":
        _reverify_bound_file(
            lock, key="exp35_source_runner", supplied=args.source_runner
        )
        if (
            capability_path is None
            or capability_payload is None
            or capability_sha256 is None
            or child_claim_path is None
        ):
            raise ValueError("EXP35 source spawn lacks exact capability/claim binding")
        capability_path = Path(capability_path)
        if capability_path.is_symlink() or not capability_path.is_file():
            raise ValueError("EXP35 source capability is absent or non-regular")
        expected_bytes = canonical_bytes(capability_payload)
        if (
            capability_path.read_bytes() != expected_bytes
            or hashlib.sha256(expected_bytes).hexdigest() != capability_sha256
            or sha256_path(capability_path) != capability_sha256
        ):
            raise ValueError("EXP35 source capability bytes are not exact before spawn")
        _require_absent_authority_file(
            child_claim_path, label="source child claim"
        )
    elif kind == "RENDERER":
        if any(
            value is not None
            for value in (
                capability_path,
                capability_payload,
                capability_sha256,
                child_claim_path,
            )
        ):
            raise ValueError("EXP35 renderer spawn may not carry source capability state")
        _reverify_bound_file(lock, key="renderer", supplied=args.renderer)
    else:
        raise ValueError(f"unexpected EXP35 child kind: {kind}")


def _require_measured_pass(result: Mapping[str, Any], *, label: str,
                           wall_limit: float) -> None:
    if (
        result.get("status") != "PASS"
        or result.get("returncode") != 0
        or result.get("timed_out") is not False
        or result.get("rss_limit_exceeded") is not False
        or result.get("sampler_failed") is not False
        or not isinstance(result.get("wall_seconds"), (int, float))
        or float(result["wall_seconds"]) > wall_limit
        or not isinstance(result.get("peak_process_tree_rss_bytes"), int)
        or int(result["peak_process_tree_rss_bytes"]) > RSS_LIMIT_BYTES
    ):
        raise RuntimeError(f"{label} failed wall/RSS/process gate")


def _run_source(args: argparse.Namespace, *, label: str, output_root: Path,
                measurement_root: Path, verified_lock: Mapping[str, Any],
                launch_claim_sha256: str,
                verified_lock_payload_sha256: str,
                launches: list[dict[str, Any]]) -> Mapping[str, Any]:
    measurement_root.mkdir(parents=True, exist_ok=True)
    capability_path = measurement_root / "SOURCE_LAUNCH_CAPABILITY.json"
    child_claim_path = measurement_root / "SOURCE_CHILD.claim.json"
    command = _source_command(
        args,
        mode=label,
        output_root=output_root,
        measurement_root=measurement_root,
        launch_claim_sha256=launch_claim_sha256,
        capability_path=capability_path,
        child_claim_path=child_claim_path,
    )
    capability = _source_capability_payload(
        args,
        mode=label,
        output_root=output_root,
        measurement_root=measurement_root,
        command=command,
        launch_claim_sha256=launch_claim_sha256,
        capability_path=capability_path,
        child_claim_path=child_claim_path,
    )
    _reverify_global_authority_state(
        args,
        lock=verified_lock,
        launch_claim_sha256=launch_claim_sha256,
        verified_lock_payload_sha256=verified_lock_payload_sha256,
    )
    _require_absent_authority_file(
        capability_path, label="source launch capability"
    )
    _require_absent_authority_file(child_claim_path, label="source child claim")
    capability_sha256 = create_launch_claim(capability_path, capability)
    launch: dict[str, Any] = {
        "ordinal": len(launches) + 1,
        "label": label,
        "kind": "SOURCE",
        "status": "STARTED",
        "command": command,
        "measurement_root": str(measurement_root / "source_process"),
        "source_launch_capability_path": str(capability_path),
        "source_launch_capability_sha256": capability_sha256,
        "source_launch_child_claim_path": str(child_claim_path),
    }
    _reverify_before_spawn(
        args,
        lock=verified_lock,
        kind="SOURCE",
        launch_claim_sha256=launch_claim_sha256,
        verified_lock_payload_sha256=verified_lock_payload_sha256,
        capability_path=capability_path,
        capability_payload=capability,
        capability_sha256=capability_sha256,
        child_claim_path=child_claim_path,
    )
    launches.append(launch)
    try:
        measured = run_measured_command(
            command,
            measurement_root=measurement_root / "source_process",
            wall_limit_seconds=SOURCE_WALL_LIMIT_SECONDS,
            rss_limit_bytes=RSS_LIMIT_BYTES,
        )
    except BaseException as error:
        launch["status"] = "MONITOR_EXCEPTION"
        launch["error_type"] = type(error).__name__
        launch["error_message"] = str(error)
        raise
    launch["status"] = "RETURNED"
    launch["measurement"] = _measurement_projection(measured)
    child_claim = _verify_source_child_claim(
        child_claim_path=child_claim_path,
        capability_path=capability_path,
        capability_sha256=capability_sha256,
        launch_claim_path=args.launch_claim,
        launch_claim_sha256=launch_claim_sha256,
        command=command,
        mode=label,
        observed_process_tree_pids=measured.get("observed_process_tree_pids"),
    )
    launch["source_launch_child_claim_sha256"] = sha256_path(child_claim_path)
    launch["source_launch_child_claim_status"] = child_claim["status"]
    launch["source_process_pid"] = child_claim["source_process_pid"]
    _require_measured_pass(measured, label=f"{label} source child", wall_limit=SOURCE_WALL_LIMIT_SECONDS)
    source = _source_result(output_root, measurement_root)
    _require_successful_source_accounting(measurement_root, mode=label)
    _validate_lifecycle_measurement(measurement_root, mode=label)
    launch["source_lifecycle_measurement_sha256"] = sha256_path(
        measurement_root / "SOURCE_LIFECYCLE_MEASUREMENT.json"
    )
    return source


def _run_renderer(args: argparse.Namespace, *, label: str, source_root: Path,
                  renderer_root: Path, measurement_root: Path,
                  verified_lock: Mapping[str, Any],
                  launch_claim_sha256: str,
                  verified_lock_payload_sha256: str,
                  launches: list[dict[str, Any]]) -> Mapping[str, Any]:
    command = _renderer_command(
        args, source_root=source_root, renderer_root=renderer_root
    )
    launch: dict[str, Any] = {
        "ordinal": len(launches) + 1,
        "label": label,
        "kind": "RENDERER",
        "status": "STARTED",
        "command": command,
        "measurement_root": str(measurement_root / "renderer_process"),
    }
    _reverify_before_spawn(
        args,
        lock=verified_lock,
        kind="RENDERER",
        launch_claim_sha256=launch_claim_sha256,
        verified_lock_payload_sha256=verified_lock_payload_sha256,
    )
    launches.append(launch)
    try:
        measured = run_measured_command(
            command,
            measurement_root=measurement_root / "renderer_process",
            wall_limit_seconds=RENDERER_WALL_LIMIT_SECONDS,
            rss_limit_bytes=RSS_LIMIT_BYTES,
        )
    except BaseException as error:
        launch["status"] = "MONITOR_EXCEPTION"
        launch["error_type"] = type(error).__name__
        launch["error_message"] = str(error)
        raise
    launch["status"] = "RETURNED"
    launch["measurement"] = _measurement_projection(measured)
    _require_measured_pass(
        measured, label=f"{label} renderer child", wall_limit=RENDERER_WALL_LIMIT_SECONDS
    )
    return validate_native_renderer_tree(renderer_root)


def create_launch_claim(path: Path, payload: Mapping[str, Any]) -> str:
    """Consume the sole orchestration using durable O_EXCL publication."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short EXP35 O_EXCL write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(data).hexdigest()


def _launch_counts(launches: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "fresh_primary_source_child_count": sum(
            row.get("label") == "PRIMARY" and row.get("kind") == "SOURCE"
            for row in launches
        ),
        "fresh_replay_source_child_count": sum(
            row.get("label") == "REPLAY" and row.get("kind") == "SOURCE"
            for row in launches
        ),
        "fresh_primary_renderer_child_count": sum(
            row.get("label") == "PRIMARY" and row.get("kind") == "RENDERER"
            for row in launches
        ),
        "fresh_replay_renderer_child_count": sum(
            row.get("label") == "REPLAY" and row.get("kind") == "RENDERER"
            for row in launches
        ),
    }


def _terminal_payload(
    *,
    status: str,
    paths: Mapping[str, Path],
    launches: Sequence[Mapping[str, Any]],
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    source_tree_hashes: Mapping[str, str] | None = None,
    renderer_tree_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    primary_accounting, primary_error = _safe_accounting(
        paths["primary_measurement_root"], mode="PRIMARY"
    )
    replay_accounting, replay_error = _safe_accounting(
        paths["replay_measurement_root"], mode="REPLAY"
    )
    totals = {
        key: primary_accounting[key] + replay_accounting[key]
        for key in ACCOUNTING_KEYS
    }
    return {
        "schema": ORCHESTRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "status": status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        **_launch_counts(launches),
        **totals,
        "primary_scan_accounting": primary_accounting,
        "replay_scan_accounting": replay_accounting,
        "scan_accounting_read_errors": {
            "primary": primary_error,
            "replay": replay_error,
        },
        "source_tree_byte_exact": source_tree_hashes is not None,
        "renderer_tree_byte_exact": (
            renderer_tree_hashes is not None
            if status != INTERFACE_TERMINAL_STATUS
            else None
        ),
        "source_tree_hashes": dict(source_tree_hashes or {}),
        "renderer_tree_hashes": dict(renderer_tree_hashes or {}),
        "launches": list(launches),
        "launch_claim_path": str(paths["launch_claim"]),
        "primary_source_root": str(paths["primary_source_root"]),
        "replay_source_root": str(paths["replay_source_root"]),
        "primary_renderer_root": str(paths["primary_renderer_root"]),
        "replay_renderer_root": str(paths["replay_renderer_root"]),
        "native_visual_review_status": (
            "NOT_APPLICABLE_EXACT_SOURCE_INTERFACE_ABSTENTION"
            if status == INTERFACE_TERMINAL_STATUS
            else "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW"
            if status == SCIENCE_TERMINAL_STATUS
            else "NOT_COMPLETED"
        ),
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
    }


def _static_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "exp35_source_runner": args.source_runner,
        "exp35_monitor": args.process_monitor,
        "exp35_orchestrator": args.orchestrator,
        "renderer": args.renderer,
        "parent_preregistration": args.parent_preregistration,
        "exp31_preregistration": args.exp31_preregistration,
        "exp32_preregistration": args.v5_preregistration,
        "cad_preflight": args.cad_preflight,
        "cad_nodes": args.nodes,
        "cad_edges": args.edges,
        "cad_edge_geodesic_mm": args.edge_geodesic_mm,
        "accepted_mesh": args.mesh,
        "scan_N009": args.scan_first,
        "scan_N010": args.scan_second,
        "transform_N009": args.transform_first,
        "transform_N010": args.transform_second,
    }


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "attempt_root": args.attempt_root,
        "primary_source_root": args.primary_source_root,
        "replay_source_root": args.replay_source_root,
        "primary_measurement_root": args.primary_measurement_root,
        "replay_measurement_root": args.replay_measurement_root,
        "primary_renderer_root": args.primary_renderer_root,
        "replay_renderer_root": args.replay_renderer_root,
        "launch_claim": args.launch_claim,
        "result": args.result,
        "manual_native_review": (
            args.attempt_root / "EXP_CAD_0035_MANUAL_NATIVE_REVIEW.json"
        ),
        "global_result": (
            ROOT / "output/research/autoresearch-loop/EXP_CAD_0035_RESULT.json"
        ),
    }


def _authority_output_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    return {
        "result_namespace": paths["attempt_root"],
        "primary_source": paths["primary_source_root"],
        "replay_source": paths["replay_source_root"],
        "primary_measurement": paths["primary_measurement_root"],
        "replay_measurement": paths["replay_measurement_root"],
        "primary_renderer": paths["primary_renderer_root"],
        "replay_renderer": paths["replay_renderer_root"],
        "primary_scan_accounting_journal": (
            paths["primary_measurement_root"] / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
        ),
        "replay_scan_accounting_journal": (
            paths["replay_measurement_root"] / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
        ),
        "primary_lifecycle_measurement": (
            paths["primary_measurement_root"] / "SOURCE_LIFECYCLE_MEASUREMENT.json"
        ),
        "replay_lifecycle_measurement": (
            paths["replay_measurement_root"] / "SOURCE_LIFECYCLE_MEASUREMENT.json"
        ),
        "primary_source_launch_capability": (
            paths["primary_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
        ),
        "replay_source_launch_capability": (
            paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
        ),
        "primary_source_child_claim": (
            paths["primary_measurement_root"] / "SOURCE_CHILD.claim.json"
        ),
        "replay_source_child_claim": (
            paths["replay_measurement_root"] / "SOURCE_CHILD.claim.json"
        ),
        "launch_claim": paths["launch_claim"],
        "orchestration_result": paths["result"],
        "manual_native_review": paths["manual_native_review"],
        "global_result": paths["global_result"],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--detached-approval-sha256", required=True)
    return parser.parse_args(argv)


def execute(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    _initialize_bound_runtime(args)
    if args.orchestrator.resolve(strict=True) != Path(__file__).resolve(strict=True):
        raise ValueError("EXP35 orchestrator self-binding mismatch")
    if args.process_monitor.resolve(strict=True) != (
        ROOT / "scripts/exp_cad_0035_process_monitor.py"
    ).resolve(strict=True):
        raise ValueError("EXP35 process-monitor path mismatch")
    raw_outputs = _output_paths(args)
    paths = validate_output_topology(raw_outputs)
    validate_attempt_namespace_preclaim(paths)
    static_paths = {
        key: _canonical_absolute(value, must_exist=True, label=key)
        for key, value in _static_paths(args).items()
    }
    # Shared authority verification must be the final gate before claim
    # consumption.  In particular, it path-binds scans without opening or
    # hashing their bytes.
    verified_lock = verify_execution_authority(
        preregistration_path=_canonical_absolute(
            args.preregistration, must_exist=True, label="preregistration"
        ),
        lock_path=_canonical_absolute(args.lock, must_exist=True, label="lock"),
        lock_sha256=args.lock_sha256,
        detached_approval_path=_canonical_absolute(
            args.detached_approval, must_exist=True, label="detached approval"
        ),
        detached_approval_sha256=args.detached_approval_sha256,
        consumed_paths=static_paths,
        output_paths=_authority_output_paths(paths),
    )
    verified_lock_payload_sha256 = hashlib.sha256(
        canonical_bytes(verified_lock)
    ).hexdigest()

    attempt = paths["attempt_root"]
    attempt.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": CLAIM_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "status": "IN_PROGRESS_CONSUMED_BEFORE_FIRST_CHILD",
        "invocation_ordinal": 1,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": sha256_path(args.preregistration),
        "lock_path": str(args.lock),
        "lock_sha256": args.lock_sha256,
        "detached_approval_path": str(args.detached_approval),
        "detached_approval_sha256": args.detached_approval_sha256,
        "resolved_inputs": {
            key: str(value) for key, value in sorted(static_paths.items())
        },
        "resolved_output_paths": {
            key: str(ROOT / relative)
            for key, relative in sorted(verified_lock["authorized_outputs"].items())
        },
        "maximum_primary_source_children": 1,
        "maximum_replay_source_children": 1,
        "physical_source_execution_authorized": True,
        "evaluator_input_authorized": False,
    }
    launches: list[dict[str, Any]] = []
    try:
        launch_claim_sha256 = create_launch_claim(paths["launch_claim"], claim)
    except FileExistsError:
        # Another invocation owns the one-shot.  Never overwrite its terminal
        # namespace or misreport a second launch as this controller's failure.
        raise
    except BaseException as error:
        if paths["launch_claim"].exists():
            terminal = _terminal_payload(
                status=FAILURE_STATUS,
                paths=paths,
                launches=launches,
                failure_stage="LAUNCH_CLAIM_PUBLICATION",
                failure_reason=f"{type(error).__name__}: {error}",
            )
            canonical_json_write(paths["result"], terminal)
            raise TerminalExecutionError(
                "EXP35 launch claim was consumed but did not publish durably"
            ) from error
        raise

    failure_stage = "PRIMARY_SOURCE"
    try:
        primary = _run_source(
            args,
            label="PRIMARY",
            output_root=paths["primary_source_root"],
            measurement_root=paths["primary_measurement_root"],
            verified_lock=verified_lock,
            launch_claim_sha256=launch_claim_sha256,
            verified_lock_payload_sha256=verified_lock_payload_sha256,
            launches=launches,
        )
        primary_science = bool(primary["scientific_execute_source_reached"])
        if not primary_science:
            failure_stage = "REPLAY_SOURCE"
            replay = _run_source(
                args,
                label="REPLAY",
                output_root=paths["replay_source_root"],
                measurement_root=paths["replay_measurement_root"],
                verified_lock=verified_lock,
                launch_claim_sha256=launch_claim_sha256,
                verified_lock_payload_sha256=verified_lock_payload_sha256,
                launches=launches,
            )
            if replay["scientific_execute_source_reached"]:
                raise RuntimeError("replay diverged from interface abstention to science")
            failure_stage = "SOURCE_TREE_COMPARE"
            primary_tree = tree_hashes(paths["primary_source_root"])
            replay_tree = tree_hashes(paths["replay_source_root"])
            if primary_tree != replay_tree:
                raise RuntimeError("primary/replay source trees are not byte-exact")
            terminal = _terminal_payload(
                status=INTERFACE_TERMINAL_STATUS,
                paths=paths,
                launches=launches,
                source_tree_hashes=primary_tree,
            )
            canonical_json_write(paths["result"], terminal)
            return terminal

        failure_stage = "PRIMARY_RENDERER"
        _run_renderer(
            args,
            label="PRIMARY",
            source_root=paths["primary_source_root"],
            renderer_root=paths["primary_renderer_root"],
            measurement_root=paths["primary_measurement_root"],
            verified_lock=verified_lock,
            launch_claim_sha256=launch_claim_sha256,
            verified_lock_payload_sha256=verified_lock_payload_sha256,
            launches=launches,
        )
        failure_stage = "REPLAY_SOURCE"
        replay = _run_source(
            args,
            label="REPLAY",
            output_root=paths["replay_source_root"],
            measurement_root=paths["replay_measurement_root"],
            verified_lock=verified_lock,
            launch_claim_sha256=launch_claim_sha256,
            verified_lock_payload_sha256=verified_lock_payload_sha256,
            launches=launches,
        )
        if not replay["scientific_execute_source_reached"]:
            raise RuntimeError("replay diverged from science to interface abstention")
        failure_stage = "REPLAY_RENDERER"
        _run_renderer(
            args,
            label="REPLAY",
            source_root=paths["replay_source_root"],
            renderer_root=paths["replay_renderer_root"],
            measurement_root=paths["replay_measurement_root"],
            verified_lock=verified_lock,
            launch_claim_sha256=launch_claim_sha256,
            verified_lock_payload_sha256=verified_lock_payload_sha256,
            launches=launches,
        )
        failure_stage = "SOURCE_TREE_COMPARE"
        primary_source_tree = tree_hashes(paths["primary_source_root"])
        replay_source_tree = tree_hashes(paths["replay_source_root"])
        if primary_source_tree != replay_source_tree:
            raise RuntimeError("primary/replay source trees are not byte-exact")
        failure_stage = "RENDERER_TREE_COMPARE"
        primary_renderer_tree = tree_hashes(paths["primary_renderer_root"])
        replay_renderer_tree = tree_hashes(paths["replay_renderer_root"])
        if primary_renderer_tree != replay_renderer_tree:
            raise RuntimeError("primary/replay renderer trees are not byte-exact")
        terminal = _terminal_payload(
            status=SCIENCE_TERMINAL_STATUS,
            paths=paths,
            launches=launches,
            source_tree_hashes=primary_source_tree,
            renderer_tree_hashes=primary_renderer_tree,
        )
        canonical_json_write(paths["result"], terminal)
        return terminal
    except BaseException as error:
        terminal = _terminal_payload(
            status=FAILURE_STATUS,
            paths=paths,
            launches=launches,
            failure_stage=failure_stage,
            failure_reason=f"{type(error).__name__}: {error}",
        )
        try:
            canonical_json_write(paths["result"], terminal)
        except BaseException as publication_error:
            raise TerminalExecutionError(
                "EXP35 terminal result publication failed after attempt consumption"
            ) from publication_error
        raise TerminalExecutionError(
            f"EXP35 consumed terminal failure at {failure_stage}: {error}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    result = execute(argv)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
