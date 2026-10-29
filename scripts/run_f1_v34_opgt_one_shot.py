#!/usr/bin/env python3
"""Strict one-fixture V34 runner with mandatory post-visual approval."""

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
OUT = ROOT / "output/research/f1-v34-opgt-v1"
LOCK = ROOT / "research/f1_v34_opgt_v1_lock.json"
PRECOMMIT = OUT / "precommit_manifest.json"
WAIVER_PRECOMMIT = OUT / "waiver_precommit_manifest.json"
FIXTURE_MANIFEST = OUT / "fixture_manifest.json"
VISUAL_MANIFEST = OUT / "preinvoke_visual_manifest.json"
COUNTER = OUT / "invocations.jsonl"
RESULTS = OUT / "source_free_results"
POST = OUT / "postinvoke_visuals"
REVIEWS = OUT / "postinvoke_visual_reviews.json"
ENGINE = ROOT / "backend/f1_ordered_projective_gauss_transport.py"
RUNNER = Path(__file__).resolve()
WAIVER_ID = "V34_OPGT_EXACT_RERUN_WAIVER_2026_08_14"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_ordered_projective_gauss_transport import (  # noqa: E402
    OPGTContractError,
    canonical_result_bytes,
    select_unified_v34,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars  # noqa: E402
from scripts.render_f1_v34_opgt_source_free import render  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload_sha256(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_payload_sha256"}
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def invocation_rows() -> list[dict[str, Any]]:
    if not COUNTER.exists():
        return []
    return [json.loads(line) for line in COUNTER.read_text(encoding="utf-8").splitlines() if line]


def reviews() -> dict[str, Any]:
    return load(REVIEWS) if REVIEWS.exists() else {"schema_version": 1, "kind": "v34_opgt_postinvoke_visual_reviews", "rows": []}


def require_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not PRECOMMIT.exists():
        raise SystemExit("V34 precommit manifest is absent")
    precommit = load(PRECOMMIT)
    if not precommit.get("passed") or precommit.get("token") != "V34_OPGT_MECHANISM_LOCKED":
        raise SystemExit("V34 mechanism is not locked")
    if not WAIVER_PRECOMMIT.exists():
        raise SystemExit("V34 exact-rerun waiver precommit is absent")
    waiver = load(WAIVER_PRECOMMIT)
    if not waiver.get("passed") or waiver.get("token") != "V34_OPGT_EXACT_RERUN_WAIVER_LOCKED":
        raise SystemExit("V34 exact-rerun waiver is not locked")
    manifest = load(FIXTURE_MANIFEST)
    visual = load(VISUAL_MANIFEST)
    for relative, expected in precommit["frozen_hashes"].items():
        path = ROOT / relative
        actual = sha(path)
        runner_waived = (
            path == RUNNER
            and expected == waiver["original_runner_sha256"]
            and actual == waiver["waiver_harness_hashes"][relative]
        )
        if actual != expected and not runner_waived:
            raise SystemExit(f"locked hash drift: {path}")
    if sha(ENGINE) != waiver["unchanged_engine_sha256"]:
        raise SystemExit("frozen V34 detector engine drift")
    for relative, expected in waiver["waiver_harness_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise SystemExit(f"waiver harness hash drift: {relative}")
    if manifest_payload_sha256(waiver) != waiver["manifest_payload_sha256"]:
        raise SystemExit("V34 waiver precommit payload hash mismatch")
    return precommit, manifest, visual, waiver


def fixture_row(manifest: dict[str, Any], fixture_id: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
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


def _append_invocation(value: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with COUNTER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def invoke(fixture_id: str, *, replacement: bool = False) -> int:
    precommit, manifest, _visual, waiver = require_lock()
    index, row, descriptor = fixture_row(manifest, fixture_id)
    prior = invocation_rows()
    if replacement:
        if fixture_id != "SF01_FOLD60_MESH_C2" or index != 0:
            raise SystemExit("waiver permits replacement of SF01_FOLD60_MESH_C2 only")
        if len(prior) != 1 or sha(COUNTER) != waiver["original_invocation_counter_sha256"]:
            raise SystemExit("authorized replacement checkpoint is not intact")
        first = prior[0]
        if first.get("fixture_id") != fixture_id or first.get("ordinal") != 1:
            raise SystemExit("aborted physical invocation 1 is not intact")
        forbidden = [
            RESULTS / f"001_{fixture_id}.json",
            RESULTS / f"001_{fixture_id}.cache",
            RESULTS / f"001_{fixture_id}.record.json",
            POST / f"001_{fixture_id}_post_4096.png",
        ]
        if any(path.exists() for path in forbidden):
            raise SystemExit("aborted invocation unexpectedly has a persisted empirical artifact")
    else:
        logical_ordinals = {
            int(value.get("logical_ordinal", value.get("ordinal", 0)))
            for value in prior
        }
        if index + 1 in logical_ordinals:
            raise SystemExit(f"one-shot fixture already consumed: {fixture_id}")
        if index != len(logical_ordinals):
            expected = manifest["ordered_fixture_ids"][len(logical_ordinals)]
            raise SystemExit(f"strict order violation: expected {expected}")
    if index and not replacement:
        previous = manifest["ordered_fixture_ids"][index - 1]
        approved = {value["fixture_id"] for value in reviews()["rows"] if value["approved"]}
        if previous not in approved:
            raise SystemExit(f"previous native visual not approved: {previous}")

    event: dict[str, Any] = {
        "schema_version": 1,
        "event": "V34_OPGT_REPLACEMENT_INVOKE" if replacement else "V34_OPGT_INVOKE",
        "physical_invocation": len(prior) + 1,
        "logical_ordinal": index + 1,
        "ordinal": index + 1,
        "fixture_id": fixture_id,
        "descriptor_sha256": row["descriptor_sha256"],
        "locked_precommit_sha256": sha(PRECOMMIT),
        "waiver_precommit_sha256": sha(WAIVER_PRECOMMIT),
        "started_ns": time.time_ns(),
    }
    if replacement:
        event.update(
            {
                "replacement_of_physical_invocation": 1,
                "waiver_id": WAIVER_ID,
                "engine_sha256": sha(ENGINE),
                "fixture_source_payload_sha256": descriptor["source_payload_sha256"],
                "fixture_triangle_binding_sha256": descriptor["triangle_binding_sha256"],
            }
        )
    _append_invocation(event)
    raw = load(ROOT / descriptor["triangle_binding_path"])
    started = time.perf_counter()
    try:
        result = select_unified_v34(
            raw,
            descriptor["candidate_records"],
            config_sha256=sha(LOCK),
            worker_count=1,
        )
    except OPGTContractError as exc:
        failure = {
            "schema_version": 1,
            "fixture_id": fixture_id,
            "passed": False,
            "terminal_token": exc.token,
            "detail": exc.detail,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(RESULTS / f"{index + 1:03d}_{fixture_id}.json", failure)
        print(json.dumps(failure, indent=2))
        return 2

    elapsed = time.perf_counter() - started
    cache = result["_ephemeral"]["fixed_cache"]
    public_result = {key: value for key, value in result.items() if key != "_ephemeral"}
    normalized_public_result = normalize_numpy_scalars(public_result)
    public_bytes = canonical_result_bytes(normalized_public_result)
    result_path = RESULTS / f"{index + 1:03d}_{fixture_id}.json"
    cache_path = RESULTS / f"{index + 1:03d}_{fixture_id}.cache"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(public_bytes)
    cache_path.write_bytes(cache)
    observed = [int(result["pre_nms_rescue_count"]), int(result["rescue_count"])]
    expected = descriptor["expected_pre_final"]
    expected_pass = expected is None or observed == expected
    causal = load(LOCK)["causal_reason_contract"].get(fixture_id)
    causal_observed = result["decisions"][0]["reason"] if result["decisions"] else None
    causal_pass = causal is None or causal_observed == causal

    chosen = 0
    for candidate_index, visual in enumerate(result["_ephemeral"]["candidate_visuals"]):
        if visual.get("kind") == "opgt":
            chosen = candidate_index
            break
    post_row = render(
        fixture_id,
        POST / f"{index + 1:03d}_{fixture_id}_post_4096.png",
        result["decisions"][chosen],
        result["_ephemeral"]["candidate_visuals"][chosen],
    )
    record = {
        "schema_version": 1,
        "kind": "v34_opgt_one_shot_result_record",
        "ordinal": index + 1,
        "fixture_id": fixture_id,
        "category": descriptor["category"],
        "expected_pre_final": expected,
        "observed_pre_final": observed,
        "expected_pass": expected_pass,
        "causal_expected": causal,
        "causal_observed": causal_observed,
        "causal_pass": causal_pass,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha(result_path),
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_sha256": sha(cache_path),
        "post_visual": post_row,
        "locked_precommit_sha256": sha(PRECOMMIT),
        "waiver_precommit_sha256": sha(WAIVER_PRECOMMIT),
        "physical_invocation": len(prior) + 1,
        "passed_pending_visual": bool(expected_pass and causal_pass),
    }
    write_json(RESULTS / f"{index + 1:03d}_{fixture_id}.record.json", record)
    print(json.dumps(record, indent=2))
    return 0 if record["passed_pending_visual"] else 2


def approve(fixture_id: str, disposition: str) -> int:
    _precommit, manifest, _visual, _waiver = require_lock()
    index, _row, _descriptor = fixture_row(manifest, fixture_id)
    invocations = invocation_rows()
    if not any(
        value["fixture_id"] == fixture_id
        and int(value.get("logical_ordinal", value.get("ordinal", 0))) == index + 1
        for value in invocations
    ):
        raise SystemExit(f"fixture has not been invoked: {fixture_id}")
    record_path = RESULTS / f"{index + 1:03d}_{fixture_id}.record.json"
    if not record_path.exists():
        raise SystemExit(f"completed result record absent: {fixture_id}")
    record = load(record_path)
    if not record["passed_pending_visual"]:
        raise SystemExit(f"numerical/causal gate failed; visual cannot advance: {fixture_id}")
    visual_path = ROOT / record["post_visual"]["path"]
    if sha(visual_path) != record["post_visual"]["sha256"]:
        raise SystemExit(f"post visual drift: {fixture_id}")
    value = reviews()
    if any(row["fixture_id"] == fixture_id for row in value["rows"]):
        raise SystemExit(f"visual already reviewed: {fixture_id}")
    value["rows"].append(
        {
            "ordinal": index + 1,
            "fixture_id": fixture_id,
            "approved": True,
            "disposition": disposition,
            "post_visual_path": record["post_visual"]["path"],
            "post_visual_sha256": record["post_visual"]["sha256"],
        }
    )
    write_json(REVIEWS, value)
    print(json.dumps(value["rows"][-1], indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    invoke_parser = subparsers.add_parser("invoke")
    invoke_parser.add_argument("--fixture", required=True)
    replacement_parser = subparsers.add_parser("replace-aborted")
    replacement_parser.add_argument("--fixture", required=True)
    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("--fixture", required=True)
    approve_parser.add_argument("--disposition", required=True)
    args = parser.parse_args()
    if args.command == "invoke":
        return invoke(args.fixture)
    if args.command == "replace-aborted":
        return invoke(args.fixture, replacement=True)
    return approve(args.fixture, args.disposition)


if __name__ == "__main__":
    raise SystemExit(main())
