#!/usr/bin/env python3
"""Strict source-free V35 runner with native postvisual approval gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research/f1-v35-observed-phase-veto-v1"
LOCK = ROOT / "research/f1_v35_observed_phase_veto_v1_lock.json"
V34_LOCK = ROOT / "research/f1_v34_opgt_v1_lock.json"
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"
COUNTER = OUT / "invocations.jsonl"
RESULTS = OUT / "source_free_results"
POST = OUT / "postinvoke_visuals"
REVIEWS = OUT / "postinvoke_visual_reviews.json"
ENGINE = ROOT / "backend/f1_observed_projective_phase_reversal_v35.py"
BASE_ENGINE = ROOT / "backend/f1_ordered_projective_gauss_transport.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_observed_projective_phase_reversal_v35 import select_unified_v35  # noqa: E402
from backend.f1_ordered_projective_gauss_transport import canonical_result_bytes  # noqa: E402
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars  # noqa: E402
from scripts.render_f1_v34_opgt_source_free import render  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def invocation_rows() -> list[dict[str, Any]]:
    if not COUNTER.exists():
        return []
    return [json.loads(line) for line in COUNTER.read_text(encoding="utf-8").splitlines() if line]


def review_rows() -> list[dict[str, Any]]:
    if not REVIEWS.exists():
        return []
    return load(REVIEWS)["rows"]


def append_invocation(value: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with COUNTER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load(LOCK)
    manifest = load(MANIFEST)
    checks = {
        "engine": sha(ENGINE) == lock["v35_engine_sha256"],
        "base_engine": sha(BASE_ENGINE) == lock["base_v34_engine_sha256"],
        "manifest": sha(MANIFEST) == lock["fixture_manifest_sha256"],
        "tests": sha(ROOT / lock["unit_test_path"]) == lock["unit_test_sha256"],
        "schedule": len(manifest["ordered_fixture_ids"]) == lock["ordered_fixture_count"],
    }
    if not all(checks.values()):
        raise SystemExit(f"V35 lock drift: {checks}")
    return lock, manifest


def fixture(manifest: dict[str, Any], fixture_id: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    ids = manifest["ordered_fixture_ids"]
    if fixture_id not in ids:
        raise SystemExit(f"unknown fixture: {fixture_id}")
    index = ids.index(fixture_id)
    row = manifest["rows"][index]
    descriptor_path = ROOT / row["descriptor_path"]
    if sha(descriptor_path) != row["descriptor_sha256"]:
        raise SystemExit(f"descriptor drift: {fixture_id}")
    descriptor = load(descriptor_path)
    for field in ("source_payload", "triangle_binding"):
        path = ROOT / descriptor[f"{field}_path"]
        if sha(path) != descriptor[f"{field}_sha256"]:
            raise SystemExit(f"{field} drift: {fixture_id}")
    return index, row, descriptor


def invoke(fixture_id: str) -> int:
    lock, manifest = require_lock()
    index, row, descriptor = fixture(manifest, fixture_id)
    prior = invocation_rows()
    if index != len(prior):
        expected = manifest["ordered_fixture_ids"][len(prior)]
        raise SystemExit(f"strict order violation: expected {expected}")
    if index:
        previous = manifest["ordered_fixture_ids"][index - 1]
        approved = {value["fixture_id"] for value in review_rows() if value["approved"]}
        if previous not in approved:
            raise SystemExit(f"previous V35 visual not approved: {previous}")
    append_invocation(
        {
            "schema_version": 1,
            "event": "V35_OBSERVED_PHASE_VETO_INVOKE",
            "physical_invocation": len(prior) + 1,
            "logical_ordinal": index + 1,
            "fixture_id": fixture_id,
            "descriptor_sha256": row["descriptor_sha256"],
            "lock_sha256": sha(LOCK),
            "engine_sha256": sha(ENGINE),
            "base_engine_sha256": sha(BASE_ENGINE),
            "started_ns": time.time_ns(),
        }
    )
    raw = load(ROOT / descriptor["triangle_binding_path"])
    started = time.perf_counter()
    result = select_unified_v35(
        raw,
        descriptor["candidate_records"],
        config_sha256=sha(LOCK),
        worker_count=1,
    )
    elapsed = time.perf_counter() - started
    public = {key: value for key, value in result.items() if key != "_ephemeral"}
    public_bytes = canonical_result_bytes(normalize_numpy_scalars(public))
    cache = result["_ephemeral"]["fixed_cache"]
    result_path = RESULTS / f"{index + 1:03d}_{fixture_id}.json"
    cache_path = RESULTS / f"{index + 1:03d}_{fixture_id}.cache"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(public_bytes)
    cache_path.write_bytes(cache)

    observed = [int(result["pre_nms_rescue_count"]), int(result["rescue_count"])]
    expected = descriptor["expected_pre_final"]
    expected_pass = expected is None or observed == expected
    base_causal = load(V34_LOCK)["causal_reason_contract"].get(fixture_id)
    causal_expected = lock["causal_reason_override"].get(fixture_id, base_causal)
    causal_observed = result["decisions"][0]["reason"] if result["decisions"] else None
    causal_pass = causal_expected is None or causal_observed == causal_expected
    chosen = next(
        (
            candidate_index
            for candidate_index, visual in enumerate(result["_ephemeral"]["candidate_visuals"])
            if visual.get("kind") == "opgt"
        ),
        0,
    )
    post_row = render(
        fixture_id,
        POST / f"{index + 1:03d}_{fixture_id}_post_4096.png",
        result["decisions"][chosen],
        result["_ephemeral"]["candidate_visuals"][chosen],
    )
    certificate = result["_ephemeral"]["certificates"][chosen]
    record = {
        "schema_version": 1,
        "kind": "v35_observed_phase_veto_one_shot_result_record",
        "ordinal": index + 1,
        "fixture_id": fixture_id,
        "category": descriptor["category"],
        "expected_pre_final": expected,
        "observed_pre_final": observed,
        "expected_pass": expected_pass,
        "causal_expected": causal_expected,
        "causal_observed": causal_observed,
        "causal_pass": causal_pass,
        "direct_phase_certificate": None if certificate is None else {
            key: value for key, value in certificate.items() if key != "steps"
        },
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha(result_path),
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_sha256": sha(cache_path),
        "post_visual": post_row,
        "lock_sha256": sha(LOCK),
        "passed_pending_visual": bool(expected_pass and causal_pass),
    }
    write_json(RESULTS / f"{index + 1:03d}_{fixture_id}.record.json", record)
    print(json.dumps(record, indent=2))
    return 0 if record["passed_pending_visual"] else 2


def approve(fixture_id: str, disposition: str) -> int:
    _lock, manifest = require_lock()
    index, _row, _descriptor = fixture(manifest, fixture_id)
    prior = invocation_rows()
    if index >= len(prior) or prior[index]["fixture_id"] != fixture_id:
        raise SystemExit(f"fixture has not been invoked: {fixture_id}")
    record_path = RESULTS / f"{index + 1:03d}_{fixture_id}.record.json"
    if not record_path.exists():
        raise SystemExit(f"result record absent: {fixture_id}")
    record = load(record_path)
    if not record["passed_pending_visual"]:
        raise SystemExit(f"numerical/causal gate failed: {fixture_id}")
    visual_path = ROOT / record["post_visual"]["path"]
    if sha(visual_path) != record["post_visual"]["sha256"]:
        raise SystemExit(f"postvisual drift: {fixture_id}")
    rows = review_rows()
    if any(value["fixture_id"] == fixture_id for value in rows):
        raise SystemExit(f"visual already reviewed: {fixture_id}")
    rows.append(
        {
            "ordinal": index + 1,
            "fixture_id": fixture_id,
            "approved": True,
            "disposition": disposition,
            "post_visual_path": record["post_visual"]["path"],
            "post_visual_sha256": record["post_visual"]["sha256"],
        }
    )
    write_json(
        REVIEWS,
        {"schema_version": 1, "kind": "v35_observed_phase_veto_visual_reviews", "rows": rows},
    )
    print(json.dumps(rows[-1], indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    invoke_parser = subparsers.add_parser("invoke")
    invoke_parser.add_argument("--fixture", required=True)
    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("--fixture", required=True)
    approve_parser.add_argument("--disposition", required=True)
    args = parser.parse_args()
    if args.command == "invoke":
        return invoke(args.fixture)
    return approve(args.fixture, args.disposition)


if __name__ == "__main__":
    raise SystemExit(main())
