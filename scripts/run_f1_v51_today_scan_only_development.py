#!/usr/bin/env python3
"""Run frozen V51 scan-only inference on an opaque, manifest-defined cohort.

This is a reusable development wrapper, not a prospective one-shot executor.  It
imports the already frozen V51 case pipeline and redirects its invocation events
to the selected output root.  The input manifest has no place for truth, sender
counts, CAD, drawings, part identity, or other side information.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_f1_v51_new_untouched_corpus_blind_once as frozen_v51  # noqa: E402


SCHEMA_VERSION = 1
MANIFEST_KIND = "sherman_qc_f1_v51_opaque_scan_only_development_input"
SCAN_GEOMETRY_SUFFIXES = frozenset({".ply", ".stl"})
RESULT_KIND = "sherman_qc_f1_v51_opaque_scan_only_development_execution"
NAMESPACE = "f1-v51-today-scan-only-development-20260830"
CASE_ROW_KEYS = frozenset({"scan_path", "scan_sha256"})
TOP_LEVEL_KEYS = frozenset({"schema_version", "kind", "cases"})
HISTORICAL_ONE_SHOT_ROOT = (
    ROOT / "output/research/f1-v51-new-untouched-corpus-20260814"
).resolve()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if compact
        else json.dumps(payload, sort_keys=True, indent=2)
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def validate_scan_manifest(path: Path) -> list[dict[str, Any]]:
    """Validate the scan-only manifest and assign opaque sequential case IDs."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_KEYS:
        raise ValueError("manifest must contain only schema_version, kind, and cases")
    if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != MANIFEST_KIND:
        raise ValueError("scan-only manifest schema/kind mismatch")
    rows = payload["cases"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan-only manifest cases must be a non-empty list")
    if len(rows) > 999:
        raise ValueError("scan-only manifest supports at most 999 cases")

    cases: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, row in enumerate(rows, start=1):
        case_id = f"T{index:03d}"
        if not isinstance(row, dict) or set(row) != CASE_ROW_KEYS:
            raise ValueError(f"{case_id}: case row must contain only scan_path and scan_sha256")
        scan = Path(str(row["scan_path"])).expanduser().resolve()
        if scan in seen_paths:
            raise ValueError(f"{case_id}: duplicate scan path")
        seen_paths.add(scan)
        if not scan.is_file() or scan.suffix.lower() not in SCAN_GEOMETRY_SUFFIXES:
            raise ValueError(f"{case_id}: scan path must be existing .ply or .stl scan geometry")
        expected_sha = str(row["scan_sha256"])
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise ValueError(f"{case_id}: scan_sha256 must be lowercase hexadecimal")
        if _sha(scan) != expected_sha:
            raise ValueError(f"{case_id}: scan hash mismatch")
        cases.append({"case_id": case_id, "scan_path": str(scan), "scan_sha256": expected_sha})
    return cases


def _append_local_event(events_path: Path, payload: dict[str, Any]) -> None:
    record = {
        "namespace": NAMESPACE,
        "event_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "research_only": True,
        **payload,
    }
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


@contextmanager
def _local_event_redirect(events_path: Path) -> Iterator[None]:
    """Keep the reused frozen helper out of its historical one-shot event log."""

    original = frozen_v51._append_event
    frozen_v51._append_event = lambda payload: _append_local_event(events_path, payload)
    try:
        yield
    finally:
        frozen_v51._append_event = original


CaseRunner = Callable[[dict[str, Any], Path, str, Path, Path], dict[str, Any]]


def run_cohort(
    manifest_path: Path,
    v9_model: Path,
    v4_model: Path,
    output_root: Path,
    *,
    case_runner: CaseRunner | None = None,
    record_case_failures_as_abstentions: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    cases = validate_scan_manifest(manifest_path.resolve())
    v9_model = v9_model.resolve()
    v4_model = v4_model.resolve()
    output_root = output_root.resolve()
    if output_root == HISTORICAL_ONE_SHOT_ROOT:
        raise ValueError("development execution may not use the historical one-shot output root")
    if not v9_model.is_file() or not v4_model.is_file():
        raise FileNotFoundError("both frozen selector model files must exist")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("development output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    runner = case_runner or frozen_v51._run_case
    records: list[dict[str, Any]] = []
    events_path = output_root / "invocation_events.jsonl"

    def execute(case: dict[str, Any], pass_name: str) -> dict[str, Any]:
        try:
            record = runner(case, output_root, pass_name, v9_model, v4_model)
        except Exception as exc:
            if not record_case_failures_as_abstentions:
                raise
            case_root = output_root / pass_name / case["case_id"]
            semantic_path = case_root / "semantic_result.json"
            if not semantic_path.is_file():
                _write_json(
                    semantic_path,
                    {
                        "schema_version": 1,
                        "kind": "sherman_qc_v51_fail_closed_case_abstention",
                        "case_id": case["case_id"],
                        "scan_sha256": case["scan_sha256"],
                        "source_only": True,
                        "research_only": True,
                        "predictions": [],
                        "pipeline_status": "FAILED_ABSTAIN",
                        "failure_type": type(exc).__name__,
                    },
                    compact=True,
                )
            _write_json(
                case_root / "case_failure.json",
                {
                    "case_id": case["case_id"],
                    "pass": pass_name,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "semantic_artifact_preserved_if_present": True,
                },
                compact=True,
            )
            record = {
                "case_id": case["case_id"],
                "pass": pass_name,
                "scan_sha256": case["scan_sha256"],
                "semantic_result_sha256": _sha(semantic_path),
                "pipeline_success": False,
                "failure_type": type(exc).__name__,
            }
        if record.get("case_id") != case["case_id"] or record.get("pass") != pass_name:
            raise RuntimeError("case runner returned mismatched case/pass identity")
        return record

    with _local_event_redirect(events_path):
        for pass_name in ("primary", "replay"):
            if max_workers == 1:
                records.extend(execute(case, pass_name) for case in cases)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(execute, case, pass_name) for case in cases]
                    records.extend(future.result() for future in futures)

    case_ids = [case["case_id"] for case in cases]
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    semantic_identity = {
        case_id: primary[case_id]["semantic_result_sha256"]
        == replay[case_id]["semantic_result_sha256"]
        for case_id in case_ids
    }
    if not all(semantic_identity.values()):
        raise RuntimeError("primary/replay semantic result bytes differ")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "namespace": NAMESPACE,
        "research_only": True,
        "source_only": True,
        "truth_opened": False,
        "scoring_opened": False,
        "labels_used": False,
        "cad_used": False,
        "drawings_used": False,
        "sender_counts_used": False,
        "part_identity_used": False,
        "case_count": len(cases),
        "case_ids": case_ids,
        "primary_success_count": sum(row.get("pipeline_success", True) is not False for row in primary.values()),
        "replay_success_count": sum(row.get("pipeline_success", True) is not False for row in replay.values()),
        "primary_failure_abstention_count": sum(row.get("pipeline_success") is False for row in primary.values()),
        "replay_failure_abstention_count": sum(row.get("pipeline_success") is False for row in replay.values()),
        "semantic_replay_byte_identity": semantic_identity,
        "input_manifest_sha256": _sha(manifest_path.resolve()),
        "v9_model_sha256": _sha(v9_model),
        "v4_model_sha256": _sha(v4_model),
        "records": records,
    }
    result_path = output_root / "development_execution_manifest.json"
    _write_json(result_path, manifest, compact=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-case-failures-as-abstentions", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    manifest = run_cohort(
        args.scan_manifest,
        args.v9_model,
        args.v4_model,
        args.output_root,
        record_case_failures_as_abstentions=args.record_case_failures_as_abstentions,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "case_count": manifest["case_count"],
                "replay_identity": True,
                "manifest": str(args.output_root.resolve() / "development_execution_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
