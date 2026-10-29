from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from research.f1_shadow_batch_v1 import ShadowBatchError, run_batch


HASH = "a" * 64


def job(ordinal: int, source: str) -> dict:
    return {
        "job_id": f"{ordinal:032x}",
        "command": [sys.executable, "-c", source],
        "input_hashes": {"scan": HASH},
        "detector_sha256": HASH,
        "config_sha256": HASH,
        "ontology_sha256": HASH,
        "artifacts": {},
    }


def spec(jobs: list[dict]) -> dict:
    return {
        "schema": "F1_SHADOW_BATCH_SPEC_V1",
        "batch_id": "b" * 32,
        "lane": "SCAN_ONLY",
        "limits": {"timeout_seconds": 2.0, "rss_limit_bytes": 512 * 1024**2, "poll_interval_seconds": 0.02, "termination_grace_seconds": 0.05},
        "jobs": jobs,
    }


def test_batch_attempts_every_job_and_reports_failures_without_paths(tmp_path: Path) -> None:
    code, summary = run_batch(spec([job(1, "print('ok')"), job(2, "raise SystemExit(7)"), job(3, "print('still ran')")]), tmp_path)
    assert code == 2
    assert summary["jobs_attempted"] == 3
    assert summary["jobs_complete"] == 2
    assert summary["all_jobs_accounted"] is True
    assert summary["status_counts"]["FAILED"] == 1
    serialized = json.dumps(summary)
    assert str(tmp_path) not in serialized
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_batch_is_one_job_at_a_time_and_enforces_denominator(tmp_path: Path) -> None:
    code, summary = run_batch(spec([job(1, "import time; time.sleep(.08)"), job(2, "import time; time.sleep(.08)")]), tmp_path)
    assert code == 0
    assert summary["jobs_attempted"] == 2
    assert summary["job_success_rate"] == 1.0
    assert summary["p95_wall_seconds"] >= 0.08


def test_controlled_abstention_is_job_success_but_reduces_coverage(tmp_path: Path) -> None:
    code, summary = run_batch(spec([job(1, "raise SystemExit(3)"), job(2, "pass")]), tmp_path)
    assert code == 0
    assert summary["jobs_succeeded"] == 2
    assert summary["jobs_complete"] == 1
    assert summary["jobs_abstained"] == 1
    assert summary["job_success_rate"] == 1.0
    assert summary["decision_coverage_rate"] == 0.5
    assert summary["abstention_rate"] == 0.5


def test_batch_rejects_identity_and_duplicate_job_ids(tmp_path: Path) -> None:
    value = spec([job(1, "pass")])
    value["jobs"][0]["expected_count"] = 3
    with pytest.raises(ShadowBatchError, match="prohibited"):
        run_batch(value, tmp_path)
    value = spec([job(1, "pass"), job(1, "pass")])
    with pytest.raises(ShadowBatchError, match="duplicate"):
        run_batch(value, tmp_path)
