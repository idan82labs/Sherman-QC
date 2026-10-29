#!/usr/bin/env python3
"""EXP33 lock-verifying wrapper around the hash-bound candidate worker.

The candidate worker deliberately retains its EXP32 schema and experiment ID.
The EXP33 outer harness nests that candidate payload instead of relabelling it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.exp_cad_0033_qualification_contract import (
    ROOT,
    canonical_bytes,
    canonical_json_write,
    sha256_path,
    verify_lock,
    verify_lock_bootstrap,
    verify_current_environment,
    verify_detached_postlock_approval,
    verify_self_binding,
    verify_worker_cli,
)


JOURNAL_SCHEMA = "EXP_CAD_0033_DURABLE_QUALIFICATION_ATTEMPT_JOURNAL_V1"
EXPECTED_LAUNCH_SCHEDULE = (
    ("source", "full/run_1/source_measurement", 1_000_000),
    ("render", "full/run_1/renderer_measurement", None),
    ("source", "full/run_2/source_measurement", 1_000_000),
    ("render", "full/run_2/renderer_measurement", None),
    ("source", "full/run_3/source_measurement", 1_000_000),
    ("render", "full/run_3/renderer_measurement", None),
    ("source", "half/run_1/source_measurement", 500_000),
    ("reference", "reference_differential/frozen_parent/run_1/measurement", 100_000),
    ("source", "reference_differential/v5/run_1/measurement", 100_000),
    ("reference", "reference_differential/frozen_parent/run_2/measurement", 100_000),
    ("source", "reference_differential/v5/run_2/measurement", 100_000),
    ("reference", "reference_differential/frozen_parent/run_3/measurement", 100_000),
    ("source", "reference_differential/v5/run_3/measurement", 100_000),
)


def _claim_launch_capability(
    args: argparse.Namespace,
    lock: Any,
    *,
    raw_argv: Sequence[str],
) -> Any:
    """Atomically consume one exact journaled child capability."""

    verify_detached_postlock_approval(
        args.post_lock_approval,
        args.post_lock_approval_sha256,
        qualification_lock=args.qualification_lock,
        qualification_lock_sha256=args.qualification_lock_sha256,
    )
    result_root = (ROOT / lock["authorized_paths"]["result_root"]).resolve(strict=True)
    expected_journal = result_root / "EXP_CAD_0033_ATTEMPT_JOURNAL.json"
    supplied_journal = Path(args.attempt_journal)
    if (
        not supplied_journal.is_absolute()
        or str(supplied_journal) != str(expected_journal)
        or not supplied_journal.is_file()
        or supplied_journal.resolve(strict=True) != expected_journal
    ):
        raise ValueError("EXP33 worker journal path differs from locked attempt journal")
    journal = json.loads(expected_journal.read_text(encoding="utf-8"))
    launch_id = int(args.launch_id)
    if (
        journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("experiment_id") != "EXP_CAD_0033"
        or journal.get("authorized_attempts") != 1
        or journal.get("attempts_started") != 1
        or journal.get("attempt_status") != "IN_PROGRESS_CONSUMED"
        or journal.get("qualification_lock_path") != str(args.qualification_lock)
        or journal.get("qualification_lock_sha256") != args.qualification_lock_sha256
        or journal.get("detached_approval_path") != str(args.post_lock_approval)
        or journal.get("detached_approval_sha256") != args.post_lock_approval_sha256
        or launch_id < 1
        or launch_id > len(EXPECTED_LAUNCH_SCHEDULE)
    ):
        raise ValueError("EXP33 worker launch journal identity/authority mismatch")
    rows = journal.get("child_launches")
    if not isinstance(rows, list) or len(rows) != launch_id:
        raise ValueError("EXP33 worker launch ID is not the exact current journal row")
    if any(row.get("status") != "RETURNED" for row in rows[:-1]):
        raise ValueError("EXP33 worker launch has incomplete or replayed predecessors")
    row = rows[-1]
    if row.get("launch_id") != launch_id or row.get("status") != "STARTED":
        raise ValueError("EXP33 worker launch row is not uniquely STARTED")
    expected_mode, measurement_relative, expected_rows = EXPECTED_LAUNCH_SCHEDULE[
        launch_id - 1
    ]
    if args.mode != expected_mode:
        raise ValueError("EXP33 worker launch mode is out of order")
    if expected_rows is not None and int(args.rows_per_member) != expected_rows:
        raise ValueError("EXP33 worker launch row count is out of order")
    if row.get("measurement_root") != str(result_root / measurement_relative):
        raise ValueError("EXP33 worker measurement root is out of order")
    environment = journal.get("verified_environment")
    if not isinstance(environment, dict):
        raise ValueError("EXP33 worker journal lacks verified environment")
    full_command = [
        environment["python_executable_invocation_path"],
        str(Path(__file__).resolve(strict=True)),
        *list(raw_argv),
    ]
    command_hash = hashlib.sha256(canonical_bytes(full_command)).hexdigest()
    if row.get("command") != full_command or row.get("command_sha256") != command_hash:
        raise ValueError("EXP33 worker canonical child command differs from journal")

    claim_path = result_root / f"EXP_CAD_0033_LAUNCH_{launch_id:02d}.claim.json"
    claim = {
        "schema": "EXP_CAD_0033_WORKER_LAUNCH_CLAIM_V1",
        "experiment_id": "EXP_CAD_0033",
        "launch_id": launch_id,
        "command_sha256": command_hash,
        "qualification_lock_sha256": args.qualification_lock_sha256,
        "detached_approval_sha256": args.post_lock_approval_sha256,
        "worker_pid": os.getpid(),
    }
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        block = canonical_bytes(claim)
        remaining = memoryview(block)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short EXP33 launch-claim write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(result_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    row["status"] = "WORKER_CLAIMED"
    row["claim_path"] = str(claim_path)
    row["claim_sha256"] = hashlib.sha256(block).hexdigest()
    row["worker_pid"] = os.getpid()
    canonical_json_write(expected_journal, journal)
    return journal


def _load_candidate_worker(lock: Any) -> Any:
    path = Path(lock["bound_files"]["candidate_worker"]["path"])
    absolute = ROOT / path
    if sha256_path(absolute) != lock["bound_files"]["candidate_worker"]["sha256"]:
        raise ValueError("candidate worker changed immediately before import")
    spec = importlib.util.spec_from_file_location("exp33_hash_bound_candidate_worker", absolute)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load hash-bound candidate worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _verified_candidate_dispatch(candidate: Any, lock: Any) -> Iterator[None]:
    """Suppress only the obsolete EXP32 lock parser after EXP33 verification."""

    original = candidate.verify_qualification_lock

    def already_verified(_path: Path, _sha256: str) -> Any:
        return lock

    candidate.verify_qualification_lock = already_verified
    try:
        yield
    finally:
        candidate.verify_qualification_lock = original


@contextmanager
def _scoped_reference_dijkstra(candidate: Any) -> Iterator[None]:
    """Wrap dynamic reference execution with the exact missing primitive."""

    from scipy.sparse.csgraph import dijkstra

    original_loader = candidate._load_module
    loaded: list[tuple[Any, Any]] = []

    def bound_loader(path: Path, name: str) -> Any:
        module = original_loader(path, name)
        if name == "exp32_frozen_parent_bound":
            original_execute = module.execute_source

            def execute_with_scoped_dijkstra(
                *args: Any,
                _module: Any = module,
                _execute: Any = original_execute,
                **kwargs: Any,
            ) -> Any:
                existed = hasattr(_module, "dijkstra")
                prior = getattr(_module, "dijkstra", None)
                _module.dijkstra = dijkstra
                try:
                    return _execute(*args, **kwargs)
                finally:
                    if existed:
                        _module.dijkstra = prior
                    else:
                        delattr(_module, "dijkstra")

            module.execute_source = execute_with_scoped_dijkstra
            loaded.append((module, original_execute))
        return module

    candidate._load_module = bound_loader
    try:
        yield
    finally:
        candidate._load_module = original_loader
        for module, original_execute in reversed(loaded):
            module.execute_source = original_execute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--benchmark-manifest", required=True, type=Path)
    source.add_argument("--rows-per-member", required=True, type=int)
    source.add_argument("--v5-preregistration", required=True, type=Path)
    source.add_argument("--parent-preregistration", required=True, type=Path)
    source.add_argument("--cad-preflight", required=True, type=Path)
    source.add_argument("--source-free-runner", required=True, type=Path)
    source.add_argument("--monitoring-output", required=True, type=Path)
    source.add_argument("--qualification-lock", required=True, type=Path)
    source.add_argument("--qualification-lock-sha256", required=True)
    source.add_argument("--output-root", required=True, type=Path)
    render = subparsers.add_parser("render")
    render.add_argument("--visual-payload", required=True, type=Path)
    render.add_argument("--renderer", required=True, type=Path)
    render.add_argument("--output-root", required=True, type=Path)
    render.add_argument("--qualification-lock", required=True, type=Path)
    render.add_argument("--qualification-lock-sha256", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--benchmark-manifest", required=True, type=Path)
    reference.add_argument("--rows-per-member", required=True, type=int)
    reference.add_argument("--parent-preregistration", required=True, type=Path)
    reference.add_argument("--cad-preflight", required=True, type=Path)
    reference.add_argument("--frozen-parent-runner", required=True, type=Path)
    reference.add_argument("--output-root", required=True, type=Path)
    reference.add_argument("--qualification-lock", required=True, type=Path)
    reference.add_argument("--qualification-lock-sha256", required=True)
    for child in (source, render, reference):
        child.add_argument("--post-lock-approval", required=True, type=Path)
        child.add_argument("--post-lock-approval-sha256", required=True)
        child.add_argument("--attempt-journal", required=True, type=Path)
        child.add_argument("--launch-id", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    lock = verify_lock_bootstrap(
        args.qualification_lock,
        args.qualification_lock_sha256,
        require_result_root_exists=True,
    )
    verify_self_binding(Path(__file__), "exp33_worker", lock)
    _claim_launch_capability(args, lock, raw_argv=raw_argv)
    lock = verify_lock(args.qualification_lock, args.qualification_lock_sha256)
    verify_worker_cli(args.mode, args, lock)
    verify_current_environment(lock)
    candidate = _load_candidate_worker(lock)
    with _verified_candidate_dispatch(candidate, lock):
        if args.mode == "source":
            return candidate._source(args)
        if args.mode == "reference":
            with _scoped_reference_dijkstra(candidate):
                return candidate._reference(args)
        return candidate._render(args)


if __name__ == "__main__":
    raise SystemExit(main())
