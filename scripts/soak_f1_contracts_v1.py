#!/usr/bin/env python3
"""Sustained deterministic source-free soak for F1 research contracts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import tempfile
import time
from typing import Any

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cibec_v1.agreement import is_agreement, match_lines, pair_metrics
from research.f1_candidate_ir_v1 import CandidateIRValidationError, canonical_json_bytes, canonicalize_candidate, canonicalize_candidate_set
from research.f1_isolated_job_v1 import JobLimits, run_isolated_job
from research.f1_unified_gate_v1 import evaluate


DEFAULT_OUTPUT = ROOT / "output/research/f1-contract-soak-v1-20260825"
MODULES = (
    ROOT / "scripts/soak_f1_contracts_v1.py",
    ROOT / "research/f1_candidate_ir_v1.py",
    ROOT / "research/f1_unified_gate_v1.py",
    ROOT / "research/f1_isolated_job_v1.py",
    ROOT / "research/cibec_v1/agreement.py",
)
ENVIRONMENT = {
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "system": platform.system(),
    "machine": platform.machine(),
    "single_threaded": True,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def _gate_payload(version: str, *, f1_delta_sign: int) -> dict[str, Any]:
    hashes = {key: char * 64 for key, char in zip(("detector", "baseline", "scorer", "ontology", "dataset"), "abcde")}
    baseline_f1 = 0.5
    candidate_f1 = baseline_f1 + f1_delta_sign * 0.05
    return {
        "schema": "F1_GENERALIZATION_EVIDENCE_V1", "version": version,
        "lane": "SCAN_ONLY", "evidence_class": "CONSUMED_PHYSICAL",
        "protocol_valid": True, "leakage_free": True, "hashes": hashes,
        "counts": {"physical_scans": 10, "morphology_families": 5, "independent_instances": 10, "jobs_attempted": 20, "jobs_succeeded": 20},
        "truth": {"independent_line_truth": False, "dual_review_verified": False, "morphology_held_out": False},
        "comparison": {"matched_frozen_baseline": True, "baseline_lane": "SCAN_ONLY", "candidate_lane": "SCAN_ONLY"},
        "baseline_metrics": {"precision": .5, "recall": .5, "f1": baseline_f1, "exact_count_accuracy": .2, "exact_localized_target_set_accuracy": .1, "minimum_family_recall": .1, "marker_admissibility": .8},
        "candidate_metrics": {"precision": candidate_f1, "recall": candidate_f1, "f1": candidate_f1, "exact_count_accuracy": .2, "exact_localized_target_set_accuracy": .1, "minimum_family_recall": .1, "marker_admissibility": .8},
        "visual": {"complete": True, "passed": True},
        "operations": {"deterministic_replay_rate": 1.0, "p95_latency_sec": 1.0, "peak_process_tree_rss_bytes": 1024, "job_success_rate": 1.0},
    }


def _one_iteration(rng: random.Random, ordinal: int, digest: hashlib._Hash) -> None:
    axis = rng.randrange(3); sign = -1 if rng.randrange(2) else 1
    start = [rng.randint(-100000, 100000) / 1000 for _ in range(3)]
    span = rng.randint(8000, 100000) / 1000
    end = list(start); end[axis] += sign * span
    route = f"route_{rng.randrange(8)}"; stage = f"stage_{rng.randrange(8)}"; source = f"source_{ordinal}"
    base = {"line_start": start, "line_end": end, "source_route": route, "source_stage": stage, "source_ids": [source]}
    canonical = canonicalize_candidate(base)
    reversed_row = canonicalize_candidate({"span_endpoints": [end, start], "route": route, "stage": stage, "bend_id": source})
    midpoint = [(start[index] + end[index]) / 2 for index in range(3)]
    direction = [0.0, 0.0, 0.0]; direction[axis] = sign * 3.0
    anchored = canonicalize_candidate({"anchor": midpoint, "axis_direction": direction, "span_mm": span, "source_route": route, "source_stage": stage, "source_hypothesis_id": source})
    if not (canonical["candidate_id"] == reversed_row["candidate_id"] == anchored["candidate_id"]):
        raise AssertionError("equivalent candidate representations changed identity")
    if not (canonical_json_bytes(canonical) == canonical_json_bytes(reversed_row) == canonical_json_bytes(anchored)):
        raise AssertionError("equivalent candidate representations changed canonical bytes")
    other = {**base, "line_start": [value + (10.0 if index == (axis + 1) % 3 else 0.0) for index, value in enumerate(start)], "line_end": [value + (10.0 if index == (axis + 1) % 3 else 0.0) for index, value in enumerate(end)], "source_ids": [source + "_other"]}
    first_set = canonicalize_candidate_set([base, other]); second_set = canonicalize_candidate_set([other, base])
    if canonical_json_bytes(first_set) != canonical_json_bytes(second_set):
        raise AssertionError("candidate set ordering changed canonical bytes")
    try:
        canonicalize_candidate({**base, "metadata": {"expected_count": 3}})
    except CandidateIRValidationError:
        pass
    else:
        raise AssertionError("leakage key was accepted")
    line = [canonical["line_start_mm"], canonical["line_end_mm"]]
    if not is_agreement(pair_metrics(line, list(reversed(line)))):
        raise AssertionError("agreement changed under endpoint reversal")
    agreement = match_lines(
        [{"event_id": "A", "line_start_mm": line[0], "line_end_mm": line[1]}],
        [{"event_id": "B", "line_start_mm": line[1], "line_end_mm": line[0]}],
    )
    if not agreement["exact_localized_target_set"]:
        raise AssertionError("self-equivalent agreement did not match")
    gate = evaluate(_gate_payload(f"soak_{ordinal}", f1_delta_sign=1 if ordinal % 2 else -1))
    if gate["grade"] != "C_NO_GENERALIZATION_EVIDENCE":
        raise AssertionError("consumed evidence escaped GPI-C")
    digest.update(canonical_json_bytes(canonical)); digest.update(canonical_json_bytes(gate))


def run(*, output: Path, duration_seconds: float, seed: int, max_iterations: int | None,
        child_interval: int, checkpoint_seconds: float = 60.0) -> dict[str, Any]:
    if duration_seconds <= 0 and (max_iterations is None or max_iterations <= 0):
        raise ValueError("positive duration or iteration limit required")
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc); started = time.monotonic(); next_checkpoint = started
    rng = random.Random(seed); digest = hashlib.sha256(); iterations = 0; child_jobs = 0; abstention_jobs = 0
    report: dict[str, Any] = {}
    try:
        while True:
            elapsed = time.monotonic() - started
            if duration_seconds > 0 and elapsed >= duration_seconds:
                break
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            _one_iteration(rng, iterations, digest)
            if child_interval > 0 and iterations % child_interval == 0:
                child_jobs += 1
                declared_abstention = child_jobs % 10 == 0
                if declared_abstention:
                    abstention_jobs += 1
                with tempfile.TemporaryDirectory(prefix="f1-soak-child-") as raw:
                    target = Path(raw)
                    code, receipt = run_isolated_job(
                        job_id=hashlib.sha256(f"{seed}:{child_jobs}".encode()).hexdigest()[:32],
                        lane="SCAN_ONLY", command=[
                            sys.executable, "-c",
                            "raise SystemExit(3)" if declared_abstention else "print('ok')",
                        ],
                        receipt_path=target / "receipt.json", log_directory=target / "logs",
                        input_hashes={"synthetic": "a" * 64}, detector_sha256="b" * 64,
                        config_sha256="c" * 64, ontology_sha256="d" * 64,
                        limits=JobLimits(timeout_seconds=5.0, rss_limit_bytes=512 * 1024**2, poll_interval_seconds=0.02),
                    )
                    expected_status = "ABSTAIN" if declared_abstention else "COMPLETE"
                    if code != 0 or receipt["status"] != expected_status:
                        raise AssertionError("isolated child soak trial failed")
                    digest.update(receipt["stdout"]["sha256"].encode())
            now = time.monotonic()
            if now >= next_checkpoint:
                report = {
                    "schema": "F1_CONTRACT_SOAK_V1", "status": "RUNNING",
                    "seed": seed, "iterations": iterations, "isolated_child_jobs": child_jobs,
                    "isolated_abstention_jobs": abstention_jobs,
                    "elapsed_seconds": round(now - started, 3), "digest_chain_sha256": digest.hexdigest(),
                    "module_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in MODULES},
                    "environment": ENVIRONMENT,
                    "physical_assets_opened": 0, "prediction_assets_opened": 0,
                }
                _atomic(output / "checkpoint.json", report)
                print(json.dumps({key: report[key] for key in ("status", "iterations", "isolated_child_jobs", "elapsed_seconds")}), flush=True)
                next_checkpoint = now + checkpoint_seconds
    except BaseException as exc:
        report = {"schema": "F1_CONTRACT_SOAK_V1", "status": "FAIL", "seed": seed, "iterations": iterations, "isolated_child_jobs": child_jobs, "isolated_abstention_jobs": abstention_jobs, "elapsed_seconds": round(time.monotonic() - started, 3), "failure_type": type(exc).__name__, "failure": str(exc), "digest_chain_sha256": digest.hexdigest(), "module_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in MODULES}, "environment": ENVIRONMENT, "physical_assets_opened": 0, "prediction_assets_opened": 0}
        _atomic(output / "final_report.json", report)
        raise
    report = {
        "schema": "F1_CONTRACT_SOAK_V1", "status": "PASS", "seed": seed,
        "started_at_utc": started_at.isoformat(), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations, "isolated_child_jobs": child_jobs,
        "isolated_abstention_jobs": abstention_jobs,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "digest_chain_sha256": digest.hexdigest(),
        "module_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in MODULES},
        "environment": ENVIRONMENT,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 3),
        "physical_assets_opened": 0, "prediction_assets_opened": 0,
    }
    _atomic(output / "final_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-seconds", type=float, default=10800.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--child-interval", type=int, default=3000)
    parser.add_argument("--checkpoint-seconds", type=float, default=60.0)
    args = parser.parse_args()
    result = run(output=args.output.resolve(), duration_seconds=args.duration_seconds, seed=args.seed, max_iterations=args.max_iterations, child_interval=args.child_interval, checkpoint_seconds=args.checkpoint_seconds)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
