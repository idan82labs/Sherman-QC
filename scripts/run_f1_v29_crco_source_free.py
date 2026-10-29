#!/usr/bin/env python3
"""Execute one frozen V29 CRCO source-free fixture.

The runner can only open fixture IDs beneath the fixed source-free fixture root.
It has no CLI path for historical or real scan inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_calibrated_range_continuity_ownership import (  # noqa: E402
    ACTION_RESCUE,
    ACTION_VETO,
    CRCOContractError,
    PROFILE,
    canonical_result_bytes,
    select_unified_v29,
)


LOCK_PATH = ROOT / "research" / "f1_v29_crco_v1_lock.json"
SOURCE_FREE_ROOT = (
    ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_fixtures"
)
RESULT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_ids() -> tuple[str, ...]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return tuple(str(value) for value in lock["source_free_fixture_ids"])


def _load_fixture(fixture_id: str) -> tuple[Path, dict]:
    if fixture_id not in _fixture_ids():
        raise CRCOContractError("UNKNOWN_SOURCE_FREE_FIXTURE", fixture_id)
    path = (SOURCE_FREE_ROOT / f"{fixture_id}.json").resolve()
    if SOURCE_FREE_ROOT.resolve() not in path.parents:
        raise CRCOContractError("SOURCE_FREE_PATH_ESCAPE")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "v29_crco_source_free_fixture":
        raise CRCOContractError("NOT_SOURCE_FREE_FIXTURE")
    if payload.get("source_free") is not True or payload.get("fixture_id") != fixture_id:
        raise CRCOContractError("SOURCE_FREE_FIXTURE_BINDING")
    prohibited = {
        "cad",
        "drawing",
        "gold",
        "expected_count",
        "filename",
        "family_name",
        "part_id",
        "good_bad_status",
        "hidden_semantic_label",
        "benchmark_result",
    }
    if prohibited.intersection(payload):
        raise CRCOContractError("PROHIBITED_FIXTURE_INPUT")
    return path, payload


def run_fixture(fixture_id: str) -> dict:
    fixture_path, payload = _load_fixture(fixture_id)
    started = time.perf_counter()
    result = select_unified_v29(
        payload["candidate_records"], payload["range_frames"], float(payload["s_mm"])
    )
    elapsed = time.perf_counter() - started
    result["fixture_id"] = fixture_id
    result["fixture_sha256"] = _sha256(fixture_path)
    result["lock_sha256"] = _sha256(LOCK_PATH)
    result["elapsed_seconds"] = round(elapsed, 9)
    result["process_maxrss_native_units"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    result["rescue_count"] = sum(
        decision["action"] == ACTION_RESCUE for decision in result["decisions"]
    )
    result["veto_count"] = sum(
        decision["action"] == ACTION_VETO for decision in result["decisions"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-id", required=True, choices=_fixture_ids())
    parser.add_argument("--run-tag", required=True, choices=("run_a", "run_b"))
    args = parser.parse_args()
    result = run_fixture(args.fixture_id)
    output_dir = RESULT_ROOT / args.run_tag / args.fixture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    cache_path = output_dir / "cache.json"
    result_path.write_bytes(canonical_result_bytes(result))
    cache_path.write_text(
        json.dumps(result["cache"], sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "fixture_id": args.fixture_id,
                "rescue_count": result["rescue_count"],
                "veto_count": result["veto_count"],
                "final_line_count": len(result["final_lines"]),
                "result_sha256": _sha256(result_path),
                "cache_sha256": _sha256(cache_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
