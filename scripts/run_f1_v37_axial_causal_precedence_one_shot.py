#!/usr/bin/env python3
"""Incremental V37 one-shot runner inheriting V36 ordinals 1-10."""

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
OUT = ROOT / "output/research/f1-v37-axial-causal-precedence-v1"
LOCK = ROOT / "research/f1_v37_axial_causal_precedence_v1_lock.json"
V34_LOCK = ROOT / "research/f1_v34_opgt_v1_lock.json"
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"
INHERITANCE = OUT / "inheritance_manifest.json"
COUNTER = OUT / "invocations.jsonl"
RESULTS = OUT / "source_free_results"
POST = OUT / "postinvoke_visuals"
REVIEWS = OUT / "postinvoke_visual_reviews.json"
V37 = ROOT / "backend/f1_axial_causal_precedence_v37.py"
V36 = ROOT / "backend/f1_observed_projective_phase_reversal_v36.py"
V35 = ROOT / "backend/f1_observed_projective_phase_reversal_v35.py"
V34 = ROOT / "backend/f1_ordered_projective_gauss_transport.py"
TEST = ROOT / "tests/test_f1_axial_causal_precedence_v37.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_axial_causal_precedence_v37 import select_unified_v37  # noqa: E402
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


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return load(path)["rows"]


def append_event(value: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with COUNTER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock, manifest, inheritance = load(LOCK), load(MANIFEST), load(INHERITANCE)
    checks = [
        sha(V37) == lock["v37_engine_sha256"],
        sha(V36) == lock["v36_engine_sha256"],
        sha(V35) == lock["v35_rule_module_sha256"],
        sha(V34) == lock["base_v34_engine_sha256"],
        sha(MANIFEST) == lock["fixture_manifest_sha256"],
        sha(INHERITANCE) == lock["inheritance_manifest_sha256"],
        sha(TEST) == lock["equivalence_test_sha256"],
        len(inheritance["inherited_fixture_ids"]) == lock["inherited_fixture_count"],
        len(manifest["ordered_fixture_ids"]) == lock["ordered_fixture_count"],
    ]
    if not all(checks):
        raise SystemExit(f"V37 lock drift: {checks}")
    return lock, manifest, inheritance


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
        if sha(ROOT / descriptor[f"{field}_path"]) != descriptor[f"{field}_sha256"]:
            raise SystemExit(f"{field} drift: {fixture_id}")
    return index, row, descriptor


def invoke(fixture_id: str) -> int:
    lock, manifest, inheritance = require_lock()
    index, row, descriptor = fixture(manifest, fixture_id)
    prior = rows(COUNTER)
    expected_index = lock["inherited_fixture_count"] + len(prior)
    if index != expected_index:
        raise SystemExit(f"strict order violation: expected {manifest['ordered_fixture_ids'][expected_index]}")
    if not prior:
        if fixture_id != inheritance["next_fixture_id"]:
            raise SystemExit("V37 inheritance boundary mismatch")
    else:
        previous = manifest["ordered_fixture_ids"][index - 1]
        if previous not in {value["fixture_id"] for value in rows(REVIEWS) if value["approved"]}:
            raise SystemExit(f"previous V37 visual not approved: {previous}")
    append_event(
        {
            "schema_version": 1,
            "event": "V37_AXIAL_CAUSAL_PRECEDENCE_INVOKE",
            "physical_invocation": len(prior) + 1,
            "logical_ordinal": index + 1,
            "fixture_id": fixture_id,
            "descriptor_sha256": row["descriptor_sha256"],
            "lock_sha256": sha(LOCK),
            "engine_sha256": sha(V37),
            "started_ns": time.time_ns(),
        }
    )
    started = time.perf_counter()
    result = select_unified_v37(
        load(ROOT / descriptor["triangle_binding_path"]),
        descriptor["candidate_records"],
        config_sha256=sha(LOCK),
        worker_count=1,
    )
    public = normalize_numpy_scalars({key: value for key, value in result.items() if key != "_ephemeral"})
    result_bytes = canonical_result_bytes(public)
    cache = result["_ephemeral"]["fixed_cache"]
    result_path = RESULTS / f"{index + 1:03d}_{fixture_id}.json"
    cache_path = RESULTS / f"{index + 1:03d}_{fixture_id}.cache"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(result_bytes)
    cache_path.write_bytes(cache)
    observed = [int(result["pre_nms_rescue_count"]), int(result["rescue_count"])]
    expected = descriptor["expected_pre_final"]
    expected_pass = expected is None or observed == expected
    causal_expected = load(V34_LOCK)["causal_reason_contract"].get(fixture_id)
    causal_observed = result["decisions"][0]["reason"] if result["decisions"] else None
    causal_pass = causal_expected is None or causal_observed == causal_expected
    chosen = next((i for i, value in enumerate(result["_ephemeral"]["candidate_visuals"]) if value.get("kind") == "opgt"), 0)
    post_row = render(
        fixture_id,
        POST / f"{index + 1:03d}_{fixture_id}_post_4096.png",
        result["decisions"][chosen],
        result["_ephemeral"]["candidate_visuals"][chosen],
    )
    record = {
        "schema_version": 1,
        "kind": "v37_axial_causal_precedence_one_shot_result_record",
        "ordinal": index + 1,
        "fixture_id": fixture_id,
        "category": descriptor["category"],
        "expected_pre_final": expected,
        "observed_pre_final": observed,
        "expected_pass": expected_pass,
        "causal_expected": causal_expected,
        "causal_observed": causal_observed,
        "causal_pass": causal_pass,
        "causal_precedence_count": int(result["causal_precedence_count"]),
        "elapsed_seconds": time.perf_counter() - started,
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
    lock, manifest, _inheritance = require_lock()
    index, _row, _descriptor = fixture(manifest, fixture_id)
    local_index = index - lock["inherited_fixture_count"]
    prior = rows(COUNTER)
    if local_index < 0 or local_index >= len(prior) or prior[local_index]["fixture_id"] != fixture_id:
        raise SystemExit(f"fixture has not been invoked by V37: {fixture_id}")
    record = load(RESULTS / f"{index + 1:03d}_{fixture_id}.record.json")
    if not record["passed_pending_visual"]:
        raise SystemExit(f"numerical/causal gate failed: {fixture_id}")
    visual_path = ROOT / record["post_visual"]["path"]
    if sha(visual_path) != record["post_visual"]["sha256"]:
        raise SystemExit(f"postvisual drift: {fixture_id}")
    review_rows = rows(REVIEWS)
    if any(value["fixture_id"] == fixture_id for value in review_rows):
        raise SystemExit(f"visual already reviewed: {fixture_id}")
    review_rows.append(
        {
            "ordinal": index + 1,
            "fixture_id": fixture_id,
            "approved": True,
            "disposition": disposition,
            "post_visual_path": record["post_visual"]["path"],
            "post_visual_sha256": record["post_visual"]["sha256"],
        }
    )
    write_json(REVIEWS, {"schema_version": 1, "kind": "v37_axial_causal_precedence_visual_reviews", "rows": review_rows})
    print(json.dumps(review_rows[-1], indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("invoke")
    run.add_argument("--fixture", required=True)
    review = commands.add_parser("approve")
    review.add_argument("--fixture", required=True)
    review.add_argument("--disposition", required=True)
    args = parser.parse_args()
    return invoke(args.fixture) if args.command == "invoke" else approve(args.fixture, args.disposition)


if __name__ == "__main__":
    raise SystemExit(main())
