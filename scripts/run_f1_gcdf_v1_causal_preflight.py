#!/usr/bin/env python3
"""Consume and execute the single authorized GCDF-V1 source-only preflight."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_gcdf_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402
from research.f1_gcdf_v1.core import build_surface_graph, canonicalize_sample, load_bound_sample, sha_file  # noqa: E402
from research.f1_gcdf_v1.tamper import assert_tamper_suite  # noqa: E402
from research.f1_gcdf_v1.visual import (  # noqa: E402
    CAMERA_COMMITMENT, RENDERER_COMMITMENT, build_visual_package, visual_manifest_sha256,
)

OUTPUT_RELATIVE = Path("output/research/gcdf-v1-c006-frozen-source-causal-preflight-v1")
DEFAULT_LOCK = ROOT / "docs/research/GCDF_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1_LOCK_20260809.json"
DEFAULT_CLAIM = ROOT / "docs/research/GCDF_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1_CLAIM_20260809.json"
THREAD_ENV = {
    "BLIS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _verify_and_consume(lock_path: Path, claim_path: Path) -> tuple[dict[str, Any], str]:
    lock = _read(lock_path)
    asserted = str(lock.get("execution_lock_sha256", ""))
    observed = sha_json({key: value for key, value in lock.items() if key != "execution_lock_sha256"})
    if asserted != observed or lock.get("configuration_sha256") != CONFIG_SHA256:
        raise ValueError("GCDF lock self/configuration hash mismatch")
    for relative, expected in lock["code_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise ValueError(f"GCDF code changed after lock: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise ValueError(f"GCDF input changed after lock: {relative}")
    claim = _read(claim_path)
    expected_claim = {
        "claim_state": "READY_FOR_FIRST_AND_ONLY_EXECUTION",
        "execution_lock_sha256": asserted,
        "one_shot": True,
        "rerun_authorized": False,
    }
    if claim != expected_claim:
        raise ValueError("GCDF one-shot claim missing, changed, or already consumed")
    consumed = dict(claim)
    consumed["claim_state"] = "CONSUMED_BEFORE_FIRST_ROLE_OPENS_RAW_SOURCE"
    _atomic_write(claim_path, canonical_json_bytes(consumed) + b"\n")
    return lock, asserted


def _role_command(role: str, source: Path, output: Path, scratch: Path | None) -> list[str]:
    command = [str(Path(sys.executable)), "scripts/run_f1_gcdf_v1_role.py", "--role", role, "--raw-source", str(source), "--output", str(output)]
    if scratch is not None:
        command.extend(["--visual-scratch", str(scratch)])
    return command


def _run_role(role: str, source: Path, output: Path, scratch: Path | None, log: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    command = _role_command(role, source, output, scratch)
    start = perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=CONFIG.role_wall_ceiling_seconds, check=False,
    )
    record = {
        "role": role, "command": command, "return_code": completed.returncode,
        "wall_seconds": perf_counter() - start, "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_bytes": len(completed.stdout), "stderr_bytes": len(completed.stderr),
        "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-4000:],
    }
    _atomic_write(log, canonical_json_bytes(record) + b"\n")
    if completed.returncode != 0:
        raise RuntimeError(f"{role} failed: {record['stderr_tail']}")
    value = _read(output)
    if float(value["wall_seconds"]) > CONFIG.role_wall_ceiling_seconds:
        raise RuntimeError(f"{role} wall ceiling exceeded")
    if int(value["peak_rss_bytes"]) > CONFIG.peak_rss_ceiling_bytes:
        raise RuntimeError(f"{role} RSS ceiling exceeded")
    return value


def _normalized_role(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    for key in ("role", "semantic_sha256", "wall_seconds", "peak_rss_bytes"):
        result.pop(key, None)
    return result


def _manual_template(output: Path, manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1, "attempt_id": CONFIG.attempt_id,
        "review_state": "PENDING_NATIVE_RESOLUTION_REVIEW",
        "reviewed_image_sha256": {},
        "required_image_sha256": {row["path"]: row["sha256"] for row in manifest},
        "observations": {
            "source_complete_and_unclipped": None,
            "surface_graph_visually_follows_sheet": None,
            "surface_graph_has_shortcut_bridges": None,
            "paths_visibly_leave_or_jump_the_surface": None,
            "deficit_peaks_follow_bend_like_transitions": None,
            "deficit_peaks_are_boundary_or_end_artifacts": None,
            "obvious_bend_like_structure_is_uncovered": None,
            "candidate_spines_are_source_space_coherent": None,
            "held_out_unfolding_is_mechanically_coherent": None,
            "canonical_permuted_visual_agreement": None,
            "producer_validator_visual_agreement": None,
            "visual_audit_ok": None,
            "aggregate_misleading": None,
            "counter_image_disagreement": None,
            "review_unresolved": None,
            "manual_veto": None,
        },
        "direct_observations": [], "interpretations": [],
    }


def execute(lock_path: Path, claim_path: Path) -> int:
    output = ROOT / OUTPUT_RELATIVE
    if output.exists():
        raise ValueError(f"immutable GCDF output already exists: {output}")
    lock, lock_sha = _verify_and_consume(lock_path, claim_path)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / ".scratch-producer-visual.json"
    source = ROOT / lock["raw_source_relative"]
    total_start = perf_counter()
    roles: list[dict[str, Any]] = []
    try:
        roles.append(_run_role("PRODUCER_A", source, output / "producer-a.json", scratch, output / "producer-a-process.json"))
        roles.append(_run_role("PRODUCER_A_PERMUTED", source, output / "producer-a-permuted.json", None, output / "producer-a-permuted-process.json"))
        roles.append(_run_role("VALIDATOR_B", source, output / "validator-b.json", None, output / "validator-b-process.json"))
        normalized = [_normalized_role(row) for row in roles]
        consensus = all(row == normalized[0] for row in normalized[1:])
        if not consensus:
            raise RuntimeError("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP")
        binding = lock["source_binding"]
        for row in roles:
            for key in (
                "ordered_source_ordinal_sha256", "sampled_coordinate_sha256",
                "sampler_executable_sha256", "sampler_configuration_sha256",
            ):
                if row[key] != binding[key]:
                    raise RuntimeError(f"source binding mismatch: {key}")
        tamper = assert_tamper_suite()
        render_start = perf_counter()
        sample = load_bound_sample(source)
        nodes = canonicalize_sample(sample)
        graph = build_surface_graph(nodes)
        visual_scratch = _read(scratch)
        manifest = build_visual_package(output, nodes=nodes, graph=graph, scratch=visual_scratch, roles=roles)
        renderer_wall = perf_counter() - render_start
        if renderer_wall > CONFIG.renderer_wall_ceiling_seconds:
            raise RuntimeError("INVALID_GCDF_V1_RESOURCE_CEILING_STOP: renderer wall")
        _atomic_write(output / "visual-manifest.json", canonical_json_bytes(manifest) + b"\n")
        _atomic_write(output / "manual-visual-review-template.json", canonical_json_bytes(_manual_template(output, manifest)) + b"\n")
        scratch.unlink()
        machine = {
            "schema_version": 1, "attempt_id": CONFIG.attempt_id,
            "execution_lock_sha256": lock_sha, "roles": [
                {key: row[key] for key in ("role", "semantic_sha256", "decision", "wall_seconds", "peak_rss_bytes")}
                for row in roles
            ],
            "role_consensus": consensus, "tamper_results": tamper,
            "visual_manifest_sha256": visual_manifest_sha256(manifest),
            "visual_image_count": len(manifest), "camera_commitment_sha256": CAMERA_COMMITMENT,
            "renderer_commitment_sha256": RENDERER_COMMITMENT, "renderer_wall_seconds": renderer_wall,
            "machine_decision": roles[0]["decision"],
            "final_decision": "PENDING_MANDATORY_MANUAL_VISUAL_REVIEW",
            "target_capability_granted": False, "target_capability_opened": False,
            "target_process_launched": False, "total_wall_seconds": perf_counter() - total_start,
        }
        machine["machine_result_sha256"] = sha_json(machine)
        _atomic_write(output / "machine-result-pending-manual-review.json", canonical_json_bytes(machine) + b"\n")
        print(canonical_json_bytes(machine).decode("ascii"))
        return 0
    except BaseException as error:
        failure = {
            "schema_version": 1, "attempt_id": CONFIG.attempt_id,
            "execution_lock_sha256": lock_sha, "decision": "INVALID_GCDF_V1_BINDING_OR_EXECUTION_STOP",
            "error_type": type(error).__name__, "error": str(error),
            "roles_completed": [row.get("role") for row in roles], "claim_consumed": True,
            "rerun_authorized": False, "target_capability_opened": False,
            "total_wall_seconds": perf_counter() - total_start,
        }
        failure["result_sha256"] = sha_json(failure)
        _atomic_write(output / "result.json", canonical_json_bytes(failure) + b"\n")
        if scratch.exists():
            scratch.unlink()
        raise


def finalize(manual_path: Path) -> int:
    output = ROOT / OUTPUT_RELATIVE
    machine = _read(output / "machine-result-pending-manual-review.json")
    manual = _read(manual_path)
    if manual.get("review_state") != "COMPLETE_NATIVE_RESOLUTION_REVIEW":
        raise ValueError("manual review is not complete")
    required = manual.get("required_image_sha256")
    reviewed = manual.get("reviewed_image_sha256")
    if not isinstance(required, Mapping) or required != reviewed:
        raise ValueError("not every exact visual artifact was reviewed")
    for relative, expected in required.items():
        if sha_file(output / relative) != expected:
            raise ValueError(f"visual changed after review: {relative}")
    observations = manual.get("observations")
    if not isinstance(observations, Mapping) or any(value is None for value in observations.values()):
        raise ValueError("manual critique fields incomplete")
    machine_decision = str(machine["machine_decision"])
    final_decision = (
        "GCDF_V1_MANUAL_VISUAL_VETO_ARCHITECTURE_STOP"
        if machine_decision == "GCDF_V1_CAUSAL_PREFLIGHT_SURVIVES" and bool(observations["manual_veto"])
        else machine_decision
    )
    result = {
        **machine, "manual_review_sha256": sha_file(manual_path),
        "manual_review": manual, "final_decision": final_decision,
        "source_candidate_bundle_frozen": final_decision == "GCDF_V1_CAUSAL_PREFLIGHT_SURVIVES",
        "target_capability_granted": False, "target_capability_opened": False,
        "target_process_launched": False,
    }
    result.pop("machine_result_sha256", None)
    result["result_sha256"] = sha_json(result)
    _atomic_write(output / "result.json", canonical_json_bytes(result) + b"\n")
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("execute")
    run_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    run_parser.add_argument("--claim", type=Path, default=DEFAULT_CLAIM)
    final_parser = sub.add_parser("finalize")
    final_parser.add_argument("--manual-review", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "execute":
        return execute(args.lock, args.claim)
    return finalize(args.manual_review)


if __name__ == "__main__":
    raise SystemExit(main())
