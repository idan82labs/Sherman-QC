#!/usr/bin/env python3
"""Execute exactly one authorized PGRR-V1 C006 causal preflight."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
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
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_pgrr_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402
from research.f1_pgrr_v1 import geometry  # noqa: E402
from research.f1_pgrr_v1.frozen_handoff import evaluate_handoff, load_frozen_context  # noqa: E402
from research.f1_pgrr_v1.independent_validator import validate_against_producer  # noqa: E402
from research.f1_pgrr_v1.tamper_suite import assert_tamper_suite  # noqa: E402
from research.f1_pgrr_v1.visual import build_visual_package  # noqa: E402


OUTPUT_RELATIVE = Path("output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1")
DEFAULT_OUTPUT = ROOT / OUTPUT_RELATIVE
THREAD_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _verify_lock(lock_path: Path, claim_path: Path) -> tuple[dict[str, Any], str]:
    lock = _read(lock_path)
    asserted = str(lock.get("execution_lock_sha256") or "")
    observed = sha_json({key: value for key, value in lock.items() if key != "execution_lock_sha256"})
    if asserted != observed:
        raise ValueError("PGRR execution-lock self hash mismatch")
    if lock.get("configuration_sha256") != CONFIG_SHA256:
        raise ValueError("PGRR configuration lock mismatch")
    for relative, expected in lock["code_sha256"].items():
        if _sha_file(ROOT / relative) != expected:
            raise ValueError(f"PGRR code changed after lock: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if _sha_file(ROOT / relative) != expected:
            raise ValueError(f"PGRR input changed after lock: {relative}")
    claim = _read(claim_path)
    expected_claim = {
        "claim_state": "READY_FOR_FIRST_AND_ONLY_EXECUTION",
        "execution_lock_sha256": asserted,
        "one_shot": True,
        "rerun_authorized": False,
    }
    if claim != expected_claim:
        raise ValueError("PGRR one-shot claim missing, altered, or already consumed")
    consumed = dict(claim)
    consumed["claim_state"] = "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH"
    _atomic_write(claim_path, canonical_json_bytes(consumed) + b"\n")
    return lock, observed


def _normalize_commitment(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("role", None)
    result.pop("commitment_sha256", None)
    return result


def _run_role(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log = {
        "command": [str(value) for value in command],
        "return_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
    }
    _atomic_write(log_path, canonical_json_bytes(log) + b"\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"PGRR worker failed ({completed.returncode}): "
            + completed.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    if completed.stdout:
        return json.loads(completed.stdout.decode("utf-8"))
    return log


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def _write_tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_format_cell(row.get(column)) for column in columns])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _baseline_effect_rows(handoff: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if handoff is None:
        return []
    rows = []
    for baseline in handoff["baseline_outcomes"]:
        target_id = str(baseline["frozen_target_component_id"])
        baseline_hash = str(baseline["target_outcome_sha256"])
        rows.append({
            "target_id": target_id,
            "baseline_outcome_sha256": baseline_hash,
            "augmented_outcome_sha256": baseline_hash,
            "all_pgrr_ablation_outcome_sha256": baseline_hash,
            "mid_arc_cut_outcome_sha256": baseline_hash,
            "reached": False,
            "advanced": False,
            "attributable": False,
            "novel": False,
            "baseline_equivalent": True,
        })
    return rows


def _write_evidence(
    output: Path,
    *,
    lock_sha: str,
    producer: Mapping[str, Any],
    permuted: Mapping[str, Any],
    validator: Mapping[str, Any],
    consensus: Mapping[str, Any],
    handoff: Mapping[str, Any] | None,
    visual_manifest: Sequence[Mapping[str, Any]],
    tamper: Mapping[str, bool],
    resource: Mapping[str, Any],
) -> None:
    _write_tsv(output / "00_binding_census.tsv", (
        "source_sha256", "v4_container_sha256", "configuration_sha256", "execution_lock_sha256",
        "canonical_point_order_sha256", "point_count", "target_count", "c006_target_count",
        "density_scale", "target_access_before_candidate_freeze", "source_points_serialized",
        "peak_rss_ceiling_bytes", "wall_ceiling_seconds", "tamper_suite_sha256",
    ), [{
        "source_sha256": producer["source_sha256"],
        "v4_container_sha256": producer["v4_container_sha256"],
        "configuration_sha256": producer["configuration_sha256"],
        "execution_lock_sha256": lock_sha,
        "canonical_point_order_sha256": producer["canonical_point_order_sha256"],
        "point_count": producer["point_count"],
        "target_count": 0 if handoff is None else CONFIG.frozen_target_count,
        "c006_target_count": 0 if handoff is None else CONFIG.c006_target_count,
        "density_scale": producer["density_scale"],
        "target_access_before_candidate_freeze": producer["target_access_before_candidate_freeze"],
        "source_points_serialized": producer["source_points_serialized"],
        "peak_rss_ceiling_bytes": CONFIG.peak_rss_ceiling_bytes,
        "wall_ceiling_seconds": CONFIG.total_machine_wall_ceiling_seconds,
        "tamper_suite_sha256": sha_json(tamper),
    }])
    _write_tsv(output / "01_normal_field_census.tsv", (
        "scale", "conditioned_count", "stable_count", "knn_commitment_sha256",
        "normal_field_commitment_sha256", "stable_point_commitment_sha256",
    ), [
        {"scale": 24, "conditioned_count": producer["conditioned24_count"], "stable_count": producer["stable_point_count"],
         "knn_commitment_sha256": producer["knn_commitment_sha256"], "normal_field_commitment_sha256": producer["normal_field_commitment_sha256"],
         "stable_point_commitment_sha256": producer["stable_point_commitment_sha256"]},
        {"scale": 48, "conditioned_count": producer["conditioned48_count"], "stable_count": producer["stable_point_count"],
         "knn_commitment_sha256": producer["knn_commitment_sha256"], "normal_field_commitment_sha256": producer["normal_field_commitment_sha256"],
         "stable_point_commitment_sha256": producer["stable_point_commitment_sha256"]},
    ])
    _write_tsv(output / "02_pole_census.tsv", (
        "pole_id", "axis", "eligible_count", "angular_spread", "potentially_viable",
        "induced_edge_count", "component_count", "raw_pass_count", "eligible_set_sha256",
    ), producer["pole_census"])
    _write_tsv(output / "03_component_fail_census.tsv", (
        "pole_id", "fail_code", "count", "component_ids_sha256",
    ), producer["component_fail_census"])
    ribbon_columns = (
        "ribbon_id", "pole_id", "component_min_point_id", "membership_sha256", "member_count",
        "axis", "theta_start", "theta_span", "bin_count", "cut_successor_point_id", "maximum_gap",
        "q05", "q95", "occupancy_mask_hex", "column_coverage", "ruling_coherent_bin_count",
        "ruling_axis_maximum_angle", "ruling_minimum_elongation", "bounding_box_min", "bounding_box_max",
        "duplicate_cluster_size", "duplicate_cluster_sha256",
    )
    _write_tsv(output / "04_raw_ribbons.tsv", ribbon_columns, producer["raw_ribbons"])
    _write_tsv(output / "05_deduplicated_ribbons.tsv", ribbon_columns, producer["deduplicated_ribbons"])
    _write_tsv(output / "06_target_handoff.tsv", (
        "target_id", "ribbon_id", "forward", "swapped", "selected_orientation",
        "complete_target_handoff", "formal_near_miss_code", "failure_codes", "tube_tolerance",
        "density_scale", "handoff_row_sha256",
    ), [] if handoff is None else handoff["handoff_rows"])
    _write_tsv(output / "07_support_chains_and_midcut.tsv", (
        "relation_id", "target_id", "ribbon_id", "relation_type", "node_count", "edge_count",
        "first_parent_id", "last_parent_id", "central_cut_index", "removed_edge_count",
        "full_connected", "cut_connected", "chain_sha256",
    ), [] if handoff is None else handoff["chain_rows"])
    _write_tsv(output / "08_target_effects.tsv", (
        "target_id", "baseline_outcome_sha256", "augmented_outcome_sha256",
        "all_pgrr_ablation_outcome_sha256", "mid_arc_cut_outcome_sha256", "reached",
        "advanced", "attributable", "novel", "baseline_equivalent",
    ), _baseline_effect_rows(handoff))
    _write_tsv(output / "09_role_consensus.tsv", (
        "role_id", "canonical_point_order_sha256", "stable_point_commitment_sha256",
        "raw_ribbon_count", "deduplicated_ribbon_count", "decision", "semantic_sha256",
        "consensus_ok", "peak_rss_bytes", "wall_seconds",
    ), [
        {"role_id": "PRODUCER_A", "canonical_point_order_sha256": producer["canonical_point_order_sha256"],
         "stable_point_commitment_sha256": producer["stable_point_commitment_sha256"], "raw_ribbon_count": producer["raw_ribbon_count"],
         "deduplicated_ribbon_count": producer["deduplicated_ribbon_count"], "decision": producer["decision"],
         "semantic_sha256": consensus["producer_semantic_sha256"], "consensus_ok": consensus["semantic_consensus"],
         "peak_rss_bytes": resource["peak_rss_bytes"], "wall_seconds": resource["wall_seconds"]},
        {"role_id": "PRODUCER_A_PERM", "canonical_point_order_sha256": permuted["canonical_point_order_sha256"],
         "stable_point_commitment_sha256": permuted["stable_point_commitment_sha256"], "raw_ribbon_count": permuted["raw_ribbon_count"],
         "deduplicated_ribbon_count": permuted["deduplicated_ribbon_count"], "decision": permuted["decision"],
         "semantic_sha256": sha_json(_normalize_commitment(permuted)), "consensus_ok": _normalize_commitment(producer) == _normalize_commitment(permuted),
         "peak_rss_bytes": resource["peak_rss_bytes"], "wall_seconds": resource["wall_seconds"]},
        {"role_id": "VALIDATOR_B", "canonical_point_order_sha256": validator["canonical_point_order_sha256"],
         "stable_point_commitment_sha256": validator["stable_point_commitment_sha256"], "raw_ribbon_count": validator["raw_ribbon_count"],
         "deduplicated_ribbon_count": validator["deduplicated_ribbon_count"], "decision": validator["decision"],
         "semantic_sha256": consensus["validator_semantic_sha256"], "consensus_ok": consensus["semantic_consensus"],
         "peak_rss_bytes": resource["peak_rss_bytes"], "wall_seconds": resource["wall_seconds"]},
    ])
    _write_tsv(output / "10_visual_audit_manifest.tsv", (
        "visual_subject_id", "subject_type", "ribbon_id", "target_id", "image_path",
        "image_sha256", "camera_commitment_sha256", "counter_claim", "review_reconciliation",
    ), visual_manifest)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _bundle_root(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted((p for p in path.rglob("*") if p.is_file() and p.name not in {"result.json", "result.pending_visual.json"}), key=lambda p: p.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        raw = item.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def execute(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("PGRR output root already exists; rerun is forbidden")
    started = perf_counter()
    lock, lock_sha = _verify_lock(args.lock.resolve(), args.claim.resolve())
    tamper = assert_tamper_suite()
    output.mkdir(parents=True)
    internal = output / ".internal"
    internal.mkdir()
    raw_source = ROOT / lock["raw_source_relative"]
    python = Path(sys.executable)
    worker_script = ROOT / "scripts/run_f1_pgrr_v1_candidate_worker.py"
    producer_path = output / "producer-a-commitment.json"
    permuted_path = output / "producer-a-perm-commitment.json"
    validator_path = output / "validator-b-semantic.json"
    _run_role((str(python), str(worker_script), "--kind", "producer", "--role", "PRODUCER_A",
               "--raw-source", str(raw_source), "--commitment", str(producer_path),
               "--scratch", str(internal / "producer-a.npz")), output / "producer-a-process.json")
    _run_role((str(python), str(worker_script), "--kind", "producer", "--role", "PRODUCER_A_PERM",
               "--reverse-input", "--raw-source", str(raw_source), "--commitment", str(permuted_path),
               "--scratch", str(internal / "producer-a-perm.npz")), output / "producer-a-perm-process.json")
    _run_role((str(python), str(worker_script), "--kind", "validator", "--role", "VALIDATOR_B",
               "--raw-source", str(raw_source), "--commitment", str(validator_path),
               "--scratch", str(internal / "validator-b-unused.npz")), output / "validator-b-process.json")
    producer = _read(producer_path)
    permuted = _read(permuted_path)
    validator = _read(validator_path)
    permutation_consensus = _normalize_commitment(producer) == _normalize_commitment(permuted)
    consensus = validate_against_producer(producer, validator)
    if not permutation_consensus:
        scientific_decision = "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP"
    elif not consensus["semantic_consensus"]:
        scientific_decision = "PGRR_V1_NUMERIC_BOUNDARY_INVALID_STOP"
    else:
        scientific_decision = str(producer["decision"])
    # Rebuild target-blind state for the renderer and, only if candidates survived,
    # retain it across the later target-open boundary. This is not a new execution.
    sample = geometry.load_v4_sample(raw_source)
    field = geometry.build_normal_field(sample)
    rebuilt = geometry.build_candidates(sample, field)
    expected = _normalize_commitment(producer)
    observed = geometry.public_commitment(rebuilt)
    if canonical_json_bytes(expected) != canonical_json_bytes(observed):
        scientific_decision = "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP"
    handoff = None
    if scientific_decision == "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF":
        frozen = load_frozen_context()  # first target-ledger read in the execution
        handoff = evaluate_handoff(
            raw_source=raw_source,
            producer_commitment=producer,
            rebuilt_result=rebuilt,
            frozen=frozen,
        )
        scientific_decision = str(handoff["decision"])
    visual_manifest = build_visual_package(
        output,
        sample=sample,
        field=field,
        ribbons=rebuilt["_internal"]["deduplicated"],
        controls=rebuilt["_internal"]["control_subjects"],
        handoff=handoff,
    )
    child_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    self_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        child_rss *= 1024
        self_rss *= 1024
    resource_row = {
        "wall_seconds": perf_counter() - started,
        "peak_rss_bytes": max(child_rss, self_rss),
        "peak_rss_ceiling_bytes": CONFIG.peak_rss_ceiling_bytes,
        "wall_ceiling_seconds": CONFIG.total_machine_wall_ceiling_seconds,
    }
    _write_evidence(
        output,
        lock_sha=lock_sha,
        producer=producer,
        permuted=permuted,
        validator=validator,
        consensus=consensus,
        handoff=handoff,
        visual_manifest=visual_manifest,
        tamper=tamper,
        resource=resource_row,
    )
    shutil.rmtree(internal)
    total_bytes = _directory_size(output)
    visual_bytes = _directory_size(output / "visual")
    scientific_bytes = sum(item.stat().st_size for item in output.glob("*.tsv"))
    logs_bytes = sum(item.stat().st_size for item in output.glob("*-process.json"))
    all_resource_ceilings_respected = bool(
        resource_row["wall_seconds"] <= CONFIG.total_machine_wall_ceiling_seconds
        and resource_row["peak_rss_bytes"] <= CONFIG.peak_rss_ceiling_bytes
        and scientific_bytes <= CONFIG.scientific_serialization_ceiling_bytes
        and visual_bytes <= CONFIG.visual_package_ceiling_bytes
        and logs_bytes <= CONFIG.log_ceiling_bytes
        and total_bytes <= CONFIG.total_output_ceiling_bytes
    )
    if not all_resource_ceilings_respected:
        scientific_decision = "PGRR_V1_RESOURCE_CONTRACT_STOP"
    metrics = {
        "coarse_possible_pole_count": producer["coarse_possible_pole_count"],
        "raw_ribbon_count": producer["raw_ribbon_count"],
        "deduplicated_ribbon_count": producer["deduplicated_ribbon_count"],
        "complete_target_handoff_count": 0 if handoff is None else handoff["aggregate_metrics"]["complete_target_handoff_count"],
        "aggregate_added_pgrr_support_chain_count": 0 if handoff is None else handoff["aggregate_metrics"]["aggregate_added_pgrr_support_chain_count"],
        "target_reached_by_pgrr_support_chain_count": 0,
        "target_side_gate_advanced_count": 0,
        "target_pgrr_attributable_certificate_count": 0,
        "target_novel_pgrr_attributable_certificate_count": 0,
        "mid_arc_cut_connected_chain_count": 0,
        "producer_permutation_consensus_ok": permutation_consensus,
        "producer_validator_semantic_consensus_ok": consensus["semantic_consensus"],
        "target_access_before_candidate_freeze": False,
        "all_resource_ceilings_respected": all_resource_ceilings_respected,
    }
    pending = {
        "schema_version": 1,
        "kind": "pgrr_v1_c006_frozen_target_causal_preflight_v1",
        "research_only": True,
        "execution_lock_sha256": lock_sha,
        "source_sha256": geometry.SOURCE_SHA256,
        "v4_container_sha256": geometry.V4_CONTAINER_SHA256,
        "configuration_sha256": CONFIG_SHA256,
        "claim_consumed_before_first_worker": True,
        "execution_count": 1,
        "rerun_authorized": False,
        "gold_access": False,
        "truth_opening": False,
        "cad_access": False,
        "expected_count_access": False,
        "benchmark_feedback": False,
        "production_or_runtime_main_write": False,
        "source_points_serialized": False,
        "scratch_membership_bitsets_deleted": True,
        "scientific_decision_before_visual_review": scientific_decision,
        "decision": "PENDING_MANUAL_VISUAL_REVIEW",
        "aggregate_metrics": metrics,
        "consensus": consensus,
        "tamper_regressions": tamper,
        "resource": {
            **resource_row,
            "scientific_serialization_bytes": scientific_bytes,
            "visual_package_bytes": visual_bytes,
            "log_bytes": logs_bytes,
            "total_output_bytes_before_pending_result": total_bytes,
        },
        "visual_manifest_sha256": sha_json(visual_manifest),
        "evidence_bundle_root_before_review": _bundle_root(output),
    }
    _atomic_write(output / "result.pending_visual.json", canonical_json_bytes(pending) + b"\n")
    print(json.dumps({
        "output": str(output),
        "pending_result": str(output / "result.pending_visual.json"),
        "scientific_decision_before_visual_review": scientific_decision,
        "aggregate_metrics": metrics,
        "visual_subject_count": len(visual_manifest),
        "resource": pending["resource"],
    }, indent=2, sort_keys=True))
    return 0


_VISUAL_FLAGS = (
    "clipping_present",
    "voxel_or_grid_phase_artifact",
    "angular_partition_boundary_artifact",
    "misregistration",
    "duplicate_spatial_witness",
    "boundary_condition_accident",
    "aggregate_misleading",
    "counter_image_disagreement",
    "review_unresolved",
)


def finalize_review(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if (output / "result.json").exists():
        raise FileExistsError("final PGRR result already exists")
    pending = _read(output / "result.pending_visual.json")
    review = _read(args.review.resolve())
    if set(review.get("flags", {})) != set(_VISUAL_FLAGS):
        raise ValueError("visual review flag domain mismatch")
    if any(not isinstance(review["flags"][key], bool) for key in _VISUAL_FLAGS):
        raise ValueError("visual review flags must be Boolean")
    if not isinstance(review.get("subjects"), list) or not review["subjects"]:
        raise ValueError("visual review requires per-subject reconciliation")
    manifest_path = output / "10_visual_audit_manifest.tsv"
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    review_by_id = {str(row["visual_subject_id"]): row for row in review["subjects"]}
    for row in rows:
        if row["image_path"] == "NOT_PRESENT":
            row["review_reconciliation"] = "NOT_PRESENT"
            continue
        item = review_by_id.get(row["visual_subject_id"])
        if item is None or str(item.get("image_sha256")) != row["image_sha256"]:
            raise ValueError(f"missing or mismatched visual review: {row['visual_subject_id']}")
        row["review_reconciliation"] = str(item.get("reconciliation") or "")
        if not row["review_reconciliation"]:
            raise ValueError("visual reconciliation text is empty")
    _write_tsv(manifest_path, (
        "visual_subject_id", "subject_type", "ribbon_id", "target_id", "image_path",
        "image_sha256", "camera_commitment_sha256", "counter_claim", "review_reconciliation",
    ), rows)
    critique_path = output / "visual" / "visual_critique.md"
    critique_lines = ["# PGRR-V1 visual critique", "", str(review.get("summary") or ""), ""]
    for subject in review["subjects"]:
        critique_lines.extend([
            f"## {subject['visual_subject_id']}", "",
            f"- Scalar claim: {subject['scalar_claim']}",
            f"- Visible geometry: {subject['visible_geometry']}",
            f"- Agreement: {subject['agreement']}",
            f"- Threshold boundary: {subject['threshold_boundary']}",
            f"- Full-versus-cut causal reading: {subject['causal_reading']}", "",
        ])
    for key in _VISUAL_FLAGS:
        critique_lines.append(f"- {key}: {str(review['flags'][key]).lower()}")
    _atomic_write(critique_path, ("\n".join(critique_lines) + "\n").encode("utf-8"))
    visual_ok = not any(review["flags"].values())
    decision = str(pending["scientific_decision_before_visual_review"])
    if decision == "PGRR_V1_GOLD_FREE_CAUSAL_MECHANISM_GO" and not visual_ok:
        decision = "PGRR_V1_VISUAL_RECONCILIATION_STOP"
    final = dict(pending)
    final["decision"] = decision
    final["visual_audit_ok"] = visual_ok
    final["aggregate_metrics"] = dict(final["aggregate_metrics"])
    final["aggregate_metrics"]["visual_audit_ok"] = visual_ok
    final["visual_review_sha256"] = _sha_file(args.review.resolve())
    final["evidence_bundle_root_sha256"] = _bundle_root(output)
    final["resource"] = dict(final["resource"])
    final["resource"]["total_output_bytes"] = _directory_size(output)
    _atomic_write(output / "result.json", canonical_json_bytes(final) + b"\n")
    print(json.dumps({
        "result": str(output / "result.json"),
        "result_sha256": _sha_file(output / "result.json"),
        "decision": decision,
        "visual_audit_ok": visual_ok,
        "evidence_bundle_root_sha256": final["evidence_bundle_root_sha256"],
    }, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-review", action="store_true")
    parser.add_argument("--review", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.finalize_review:
        if args.review is None:
            raise ValueError("finalize-review requires --review")
        return finalize_review(args)
    if args.lock is None or args.claim is None:
        raise ValueError("execution requires --lock and --claim")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
