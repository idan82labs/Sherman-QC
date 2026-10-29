#!/usr/bin/env python3
"""Source runner semantics for EXP_CAD_0031 CF-PDG V4.

EXP31 changes only the physical source-row interface.  Invalid point/normal
rows are filtered and audited before the frozen EXP30 scientific execute_source
boundary.  This module contains no evaluator, role, label, gold, expected-count,
quality, filename, replacement, interpolation, or normal-estimation input.

The physical CLI remains audit-locked until fresh EXP31 CAD-reuse and source-
runner approvals exist.  The source-free ``execute_source_v4`` entry point is
fully testable without opening either physical scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import exp_cad_0031_source_interface as INTAKE
from scripts import run_exp_cad_0029_cfpdg_source as PARENT


EXPECTED_SCHEMA = "EXP_CAD_0031_CFPDG_V4_PREREGISTRATION_V1"
EXPECTED_EXPERIMENT = "EXP_CAD_0031"
EXPECTED_ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V4_SOURCE_INTERFACE_FILTER"
EXPECTED_PARENT_SCHEMA = "EXP_CAD_0030_CFPDG_V3_PREREGISTRATION_V1"
EXPECTED_PARENT_SHA256 = "c9129ccc103d45c89ccdc67eef938b242ccdbb081d3d0f261145cc5a5b9b8368"
PARENT_VISUAL_SCHEMA = "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1"
CAD_REUSE_APPROVAL_SCHEMA = "EXP_CAD_0031_V4_INDEPENDENT_CAD_REUSE_APPROVAL_V1"
SOURCE_APPROVAL_SCHEMA = "EXP_CAD_0031_V4_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1"
CAD_REUSE_BOUND_KEYS = frozenset(
    {
        "v4_preregistration", "parent_preregistration", "terminal_parent_result",
        "terminal_parent_report", "terminal_parent_orchestration",
        "cad_reuse_audit_packet", "cad_preflight", "nodes", "edges",
        "edge_geodesic_mm", "mesh", "cad_core", "cad_only_runner",
        "cad_only_wrapper", "cad_only_tests", "cad_identity_tests",
    }
)
SOURCE_BOUND_KEYS = frozenset(
    set(CAD_REUSE_BOUND_KEYS)
    | {
        "cad_reuse_approval", "source_runner_audit_packet", "interface_module", "parent_science_module",
        "parent_execute_source", "source_runner", "source_tests",
        "parent_renderer", "parent_renderer_tests", "orchestrator",
        "orchestrator_tests",
    }
)
CAD_ARTIFACT_KEYS = frozenset(
    {
        "CAD_ONLY_PREFLIGHT.json", "CAD_GRAPH_NODES.npy",
        "CAD_GRAPH_EDGES.npy", "CAD_GRAPH_EDGE_GEODESIC_MM.npy",
    }
)


def _validate_preregistration_contracts(
    v4_prereg: Mapping[str, Any], parent_prereg: Mapping[str, Any]
) -> None:
    if (
        v4_prereg.get("schema") != EXPECTED_SCHEMA
        or v4_prereg.get("experiment_id") != EXPECTED_EXPERIMENT
        or v4_prereg.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise ValueError("unexpected EXP31 preregistration identity")
    if parent_prereg.get("schema") != EXPECTED_PARENT_SCHEMA:
        raise ValueError("unexpected frozen scientific parent schema")
    lineage = v4_prereg.get("lineage")
    if not isinstance(lineage, Mapping) or lineage.get(
        "scientific_parent_preregistration_sha256"
    ) != EXPECTED_PARENT_SHA256:
        raise ValueError("EXP31 scientific parent hash contract mismatch")
    interface = v4_prereg.get("source_interface_actionability")
    if not isinstance(interface, Mapping):
        raise ValueError("EXP31 source-interface actionability contract missing")
    if interface.get("maximum_invalid_fraction_per_member") != 0.01:
        raise ValueError("EXP31 maximum invalid fraction must remain exactly 0.01")
    if interface.get("minimum_valid_row_count_per_member") != 100_000:
        raise ValueError("EXP31 minimum valid row count must remain exactly 100000")
    if interface.get("maximum_invalid_fraction_inclusive") is not True:
        raise ValueError("EXP31 invalid-fraction boundary must remain inclusive")


def _interface_semantic(
    *,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "EXP_CAD_0031_SOURCE_INTERFACE_RESULT_V1",
        "experiment_id": EXPECTED_EXPERIMENT,
        "architecture": EXPECTED_ARCHITECTURE,
        "status": pair["status"],
        "reason": (
            None if pair["actionable"] else "SOURCE_INTERFACE_ACTIONABILITY_GATE"
        ),
        "source_interface": {
            "first": first["report"],
            "second": second["report"],
            "pair": dict(pair),
        },
        "scientific_execute_source_reached": False,
        "numeric_pair_direction_produced": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_OPAQUE_PAIR_DIAGNOSTIC",
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
    }


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        PARENT._jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execute_source_v4(
    *,
    v4_prereg: Mapping[str, Any],
    parent_prereg: Mapping[str, Any],
    cad_preflight: Mapping[str, Any],
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    mesh: Any,
    raw_first: np.ndarray,
    normals_first: np.ndarray,
    raw_second: np.ndarray,
    normals_second: np.ndarray,
    transform_first: np.ndarray,
    transform_second: np.ndarray,
    first_token: str,
    second_token: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Apply the V4 interface gate, then call unchanged parent science once."""

    _validate_preregistration_contracts(v4_prereg, parent_prereg)
    frozen_tokens = list(
        v4_prereg.get("immutable_inputs", {}).get(
            "opaque_case_tokens_in_canonical_order", []
        )
    )
    parent_tokens = list(
        parent_prereg.get("immutable_inputs", {}).get(
            "opaque_case_tokens_in_canonical_order", []
        )
    )
    if (
        frozen_tokens != ["N009", "N010"]
        or parent_tokens != frozen_tokens
        or [first_token, second_token] != frozen_tokens
    ):
        raise ValueError("EXP31 opaque member token/order binding mismatch")
    first_points = np.asarray(raw_first)
    second_points = np.asarray(raw_second)
    # Physical acquisition identities are created exactly once, before any
    # filtering or scientific transform, and cannot be supplied by a caller.
    first_ids = np.arange(len(first_points), dtype=np.uint64)
    second_ids = np.arange(len(second_points), dtype=np.uint64)
    first = INTAKE.filter_source_rows(
        first_points,
        normals_first,
        member_token=first_token,
        original_row_ids=first_ids,
    )
    second = INTAKE.filter_source_rows(
        second_points,
        normals_second,
        member_token=second_token,
        original_row_ids=second_ids,
    )
    pair = INTAKE.pair_actionability(first, second)
    interface_payload = {
        "first": first["report"],
        "second": second["report"],
        "pair": pair,
    }
    filtered = {
        "first_original_row_ids": first["original_row_ids"],
        "second_original_row_ids": second["original_row_ids"],
    }
    if not pair["actionable"]:
        return (
            _interface_semantic(first=first, second=second, pair=pair),
            None,
            filtered,
        )

    parent_semantic, parent_visual = PARENT.execute_source(
        prereg=parent_prereg,
        cad_preflight=cad_preflight,
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        mesh=mesh,
        raw_first=first["points"],
        normals_first=first["normals"],
        raw_second=second["points"],
        normals_second=second["normals"],
        transform_first=transform_first,
        transform_second=transform_second,
        first_token=first_token,
        second_token=second_token,
    )
    if parent_semantic.get("schema") != "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1":
        raise RuntimeError("unchanged parent semantic payload schema drift")
    if parent_visual.get("schema") != PARENT_VISUAL_SCHEMA:
        raise RuntimeError("unchanged parent visual payload schema drift")
    semantic = {
        "schema": "EXP_CAD_0031_SOURCE_INTERFACE_EXECUTION_ENVELOPE_V1",
        "experiment_id": EXPECTED_EXPERIMENT,
        "architecture": EXPECTED_ARCHITECTURE,
        "status": parent_semantic.get("status"),
        "reason": parent_semantic.get("reason"),
        "source_interface": interface_payload,
        "scientific_execute_source_reached": True,
        "scientific_core": {
            "experiment_id": "EXP_CAD_0030",
            "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
            "preregistration_sha256": EXPECTED_PARENT_SHA256,
            "semantic_payload_sha256": _canonical_payload_sha256(parent_semantic),
            "visual_payload_sha256": _canonical_payload_sha256(parent_visual),
            "semantic_payload": parent_semantic,
        },
        "visual_payload_identity": {
            "schema": PARENT_VISUAL_SCHEMA,
            "renderer_context": "EXP31_INTAKE_SUCCESSOR_WITH_UNCHANGED_EXP30_SCIENTIFIC_CORE",
            "silent_relabeling_performed": False,
        },
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_OPAQUE_PAIR_DIAGNOSTIC",
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
    }
    # Return the parent's exact object without adding EXP31 fields.  The outer
    # semantic envelope and orchestration result carry successor identity.
    return semantic, parent_visual, filtered


def source_free_contract_summary() -> str:
    """Canonical summary used by pre-source audit tooling."""

    return json.dumps(
        {
            "experiment_id": EXPECTED_EXPERIMENT,
            "architecture": EXPECTED_ARCHITECTURE,
            "filter_before_parent_execute_source": True,
            "initial_identity": "arange(N)_uint64",
            "thresholds_caller_tunable": False,
            "parent_scientific_semantics_unchanged": True,
            "physical_cli_execution_authorized": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bound_file_sha(payload: Mapping[str, Any], key: str) -> str:
    files = payload.get("bound_files")
    if not isinstance(files, Mapping) or not isinstance(files.get(key), Mapping):
        raise ValueError(f"approval lacks exact bound_files.{key}")
    digest = files[key].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"approval has invalid bound_files.{key}.sha256")
    return digest


def _validate_approval_header(payload: Mapping[str, Any], *, source: bool) -> None:
    expected_schema = SOURCE_APPROVAL_SCHEMA if source else CAD_REUSE_APPROVAL_SCHEMA
    expected_decision = (
        "APPROVE_ONE_EXP31_PHYSICAL_PRIMARY_REPLAY_EXECUTION"
        if source
        else "PASS_REUSED_EXP30_CAD_FREEZE_FOR_EXP31_SOURCE_INTERFACE"
    )
    if payload.get("schema") != expected_schema:
        raise ValueError("unexpected EXP31 approval schema")
    if payload.get("experiment_id") != EXPECTED_EXPERIMENT or payload.get(
        "architecture"
    ) != EXPECTED_ARCHITECTURE:
        raise ValueError("EXP31 approval identity mismatch")
    if payload.get("decision") != expected_decision:
        raise ValueError("EXP31 approval decision unavailable")
    if payload.get("physical_source_execution_authorized") is not source:
        raise ValueError("EXP31 approval physical authority mismatch")
    files = payload.get("bound_files")
    expected_keys = SOURCE_BOUND_KEYS if source else CAD_REUSE_BOUND_KEYS
    if not isinstance(files, Mapping) or set(files) != set(expected_keys):
        raise ValueError("EXP31 approval bound_files keys are not exact")
    artifacts = payload.get("bound_cad_artifacts")
    if source:
        if artifacts is not None:
            raise ValueError("EXP31 source approval has unexpected artifact namespace")
    elif not isinstance(artifacts, Mapping) or set(artifacts) != set(CAD_ARTIFACT_KEYS):
        raise ValueError("EXP31 CAD-reuse artifact keys are not exact")


def _require_file_bindings(
    approval: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    for key, path in paths.items():
        if _bound_file_sha(approval, key) != PARENT.CAD.sha256_path(path):
            raise ValueError(f"EXP31 approval file mismatch: {key}")


def verify_pre_scan_approvals(
    *,
    v4_preregistration: Path,
    parent_preregistration: Path,
    terminal_parent_result: Path,
    terminal_parent_report: Path,
    terminal_parent_orchestration: Path,
    cad_reuse_audit_packet: Path,
    cad_preflight: Path,
    cad_reuse_approval: Path,
    source_runner_audit_packet: Path,
    source_approval: Path,
    nodes: Path,
    edges: Path,
    edge_geodesic_mm: Path,
    mesh: Path,
    source_tests: Path,
    parent_renderer: Path,
    parent_renderer_tests: Path,
    orchestrator: Path,
    orchestrator_tests: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Bind fresh EXP31 authority and immutable EXP30 science before scan bytes."""

    v4 = _load_json(v4_preregistration)
    parent = _load_json(parent_preregistration)
    _validate_preregistration_contracts(v4, parent)
    lineage = v4["lineage"]
    lineage_paths = {
        "terminal_parent_result": terminal_parent_result,
        "terminal_parent_report": terminal_parent_report,
        "terminal_parent_orchestration": terminal_parent_orchestration,
    }
    lineage_hash_fields = {
        "terminal_parent_result": "terminal_parent_result_sha256",
        "terminal_parent_report": "terminal_parent_report_sha256",
        "terminal_parent_orchestration": "terminal_parent_orchestration_sha256",
    }
    for key, path in lineage_paths.items():
        if PARENT.CAD.sha256_path(path) != lineage[lineage_hash_fields[key]]:
            raise ValueError(f"EXP31 terminal lineage mismatch: {key}")
    if PARENT.CAD.sha256_path(parent_preregistration) != EXPECTED_PARENT_SHA256:
        raise ValueError("EXP31 parent preregistration bytes changed")
    if PARENT.CAD.sha256_path(Path(PARENT.SRC.__file__)) != lineage[
        "scientific_semantics_module_sha256"
    ]:
        raise ValueError("EXP31 parent scientific semantics bytes changed")
    if PARENT.CAD.sha256_path(Path(PARENT.__file__)) != lineage[
        "scientific_execute_source_parent_sha256_at_preregistration"
    ]:
        raise ValueError("EXP31 parent execute_source bytes changed")

    cad_payload = _load_json(cad_preflight)
    PARENT.validate_cad_preflight_identity(cad_payload, EXPECTED_PARENT_SHA256)
    immutable = v4["immutable_inputs"]
    for path, expected, label in (
        (cad_preflight, immutable["cad_only_preflight_sha256"], "cad_preflight"),
        (nodes, immutable["cad_graph_nodes_sha256"], "nodes"),
        (edges, immutable["cad_graph_edges_sha256"], "edges"),
        (
            edge_geodesic_mm,
            immutable["cad_graph_edge_geodesic_mm_sha256"],
            "edge_geodesic_mm",
        ),
        (mesh, immutable["accepted_mesh_sha256"], "mesh"),
    ):
        if PARENT.CAD.sha256_path(path) != expected:
            raise ValueError(f"EXP31 immutable CAD mismatch: {label}")

    fixed_paths = {
        "v4_preregistration": v4_preregistration,
        "parent_preregistration": parent_preregistration,
        **lineage_paths,
        "cad_reuse_audit_packet": cad_reuse_audit_packet,
        "cad_preflight": cad_preflight,
        "nodes": nodes,
        "edges": edges,
        "edge_geodesic_mm": edge_geodesic_mm,
        "mesh": mesh,
        "cad_core": Path(PARENT.CAD.__file__).resolve(),
        "cad_only_runner": (ROOT / "scripts/run_exp_cad_0029_cad_only_preflight.py").resolve(),
        "cad_only_wrapper": (ROOT / "scripts/run_exp_cad_0030_cad_only_preflight.py").resolve(),
        "cad_only_tests": (ROOT / "tests/test_exp_cad_0029_cfpdg_preflight.py").resolve(),
        "cad_identity_tests": (ROOT / "tests/test_exp_cad_0030_cfpdg_cad_identity.py").resolve(),
    }
    audit_packet = _load_json(cad_reuse_audit_packet)
    if (
        audit_packet.get("schema") != "EXP_CAD_0031_V4_CAD_REUSE_AUDIT_PACKET_V1"
        or audit_packet.get("experiment_id") != EXPECTED_EXPERIMENT
        or audit_packet.get("architecture") != EXPECTED_ARCHITECTURE
        or audit_packet.get("status")
        != "PASS_CAD_REUSE_ACTIONABLE_SOURCE_NOT_AUTHORIZED"
        or audit_packet.get("source_execution_authorized") is not False
    ):
        raise ValueError("EXP31 CAD-reuse audit packet identity/status mismatch")
    packet_files = audit_packet.get("bound_files")
    if not isinstance(packet_files, Mapping):
        raise ValueError("EXP31 CAD-reuse audit packet lacks bound_files")
    for key, path in fixed_paths.items():
        if key == "cad_reuse_audit_packet":
            continue
        item = packet_files.get(key)
        if not isinstance(item, Mapping) or item.get("sha256") != PARENT.CAD.sha256_path(path):
            raise ValueError(f"EXP31 CAD-reuse audit packet mismatch: {key}")
    packet_artifacts = audit_packet.get("cad_artifacts")
    if not isinstance(packet_artifacts, Mapping) or set(packet_artifacts) != set(
        CAD_ARTIFACT_KEYS
    ):
        raise ValueError("EXP31 CAD-reuse audit packet artifact keys mismatch")
    for key, path in (
        ("CAD_ONLY_PREFLIGHT.json", cad_preflight),
        ("CAD_GRAPH_NODES.npy", nodes),
        ("CAD_GRAPH_EDGES.npy", edges),
        ("CAD_GRAPH_EDGE_GEODESIC_MM.npy", edge_geodesic_mm),
    ):
        if packet_artifacts[key] != PARENT.CAD.sha256_path(path):
            raise ValueError(f"EXP31 CAD-reuse packet artifact mismatch: {key}")
    cad_approval = _load_json(cad_reuse_approval)
    _validate_approval_header(cad_approval, source=False)
    _require_file_bindings(cad_approval, fixed_paths)
    artifacts = cad_approval["bound_cad_artifacts"]
    for key, path in (
        ("CAD_ONLY_PREFLIGHT.json", cad_preflight),
        ("CAD_GRAPH_NODES.npy", nodes),
        ("CAD_GRAPH_EDGES.npy", edges),
        ("CAD_GRAPH_EDGE_GEODESIC_MM.npy", edge_geodesic_mm),
    ):
        if artifacts[key] != PARENT.CAD.sha256_path(path):
            raise ValueError(f"EXP31 CAD-reuse artifact mismatch: {key}")

    source_paths = {
        **fixed_paths,
        "cad_reuse_approval": cad_reuse_approval,
        "source_runner_audit_packet": source_runner_audit_packet,
        "interface_module": Path(INTAKE.__file__).resolve(),
        "parent_science_module": Path(PARENT.SRC.__file__).resolve(),
        "parent_execute_source": Path(PARENT.__file__).resolve(),
        "source_runner": Path(__file__).resolve(),
        "source_tests": source_tests,
        "parent_renderer": parent_renderer,
        "parent_renderer_tests": parent_renderer_tests,
        "orchestrator": orchestrator,
        "orchestrator_tests": orchestrator_tests,
    }
    source_payload = _load_json(source_approval)
    source_audit = _load_json(source_runner_audit_packet)
    if (
        source_audit.get("schema") != "EXP_CAD_0031_V4_SOURCE_RUNNER_AUDIT_PACKET_V1"
        or source_audit.get("experiment_id") != EXPECTED_EXPERIMENT
        or source_audit.get("architecture") != EXPECTED_ARCHITECTURE
        or source_audit.get("status") != "PASS_SOURCE_FREE_AUDIT_EXECUTION_NOT_AUTHORIZED"
        or source_audit.get("physical_source_execution_authorized") is not False
    ):
        raise ValueError("EXP31 source-runner audit packet identity/status mismatch")
    audit_files = source_audit.get("bound_files")
    if not isinstance(audit_files, Mapping):
        raise ValueError("EXP31 source-runner audit packet lacks bound_files")
    for key, path in source_paths.items():
        if key == "source_runner_audit_packet":
            continue
        item = audit_files.get(key)
        if not isinstance(item, Mapping) or item.get("sha256") != PARENT.CAD.sha256_path(path):
            raise ValueError(f"EXP31 source-runner audit packet mismatch: {key}")
    _validate_approval_header(source_payload, source=True)
    _require_file_bindings(source_payload, source_paths)
    return v4, parent, cad_payload


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.rglob("*"))):
        raise ValueError(f"output root must be absent or an empty directory: {path}")


def _write_scan_accounting_journal(
    root: Path,
    accounting: Mapping[str, int],
    *,
    stage: str,
    status: str,
    error: Exception | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "EXP_CAD_0031_SCAN_ACCOUNTING_JOURNAL_V1",
        "experiment_id": EXPECTED_EXPERIMENT,
        "status": status,
        "stage": stage,
        **{key: int(value) for key, value in accounting.items()},
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    PARENT.CAD.atomic_json(root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json", payload)


def _publish_post_byte_failure(
    root: Path,
    accounting: Mapping[str, int],
    *,
    stage: str,
    error: Exception,
) -> None:
    # Best effort: the already-durable prior journal checkpoint remains valid
    # even if the filesystem failure also prevents these terminal writes.
    try:
        _write_scan_accounting_journal(
            root, accounting, stage=stage, status="FAIL_TERMINAL_NO_RERUN",
            error=error,
        )
    except Exception:
        pass
    try:
        PARENT.CAD.atomic_json(
            root / "SOURCE_CHILD_FAILURE_ACCOUNTING.json",
            {
                "schema": "EXP_CAD_0031_SOURCE_CHILD_FAILURE_ACCOUNTING_V1",
                "experiment_id": EXPECTED_EXPERIMENT,
                "status": "FAIL_AFTER_PHYSICAL_SCAN_BYTE_ACCESS_TERMINAL_NO_RERUN",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "failure_stage": stage,
                **{key: int(value) for key, value in accounting.items()},
            },
        )
    except Exception:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--scientific-preregistration", required=True, type=Path)
    parser.add_argument("--terminal-parent-result", required=True, type=Path)
    parser.add_argument("--terminal-parent-report", required=True, type=Path)
    parser.add_argument("--terminal-parent-orchestration", required=True, type=Path)
    parser.add_argument("--cad-reuse-audit-packet", required=True, type=Path)
    parser.add_argument("--cad-preflight", required=True, type=Path)
    parser.add_argument("--cad-reuse-approval", required=True, type=Path)
    parser.add_argument("--source-runner-audit-packet", required=True, type=Path)
    parser.add_argument("--source-runner-approval", required=True, type=Path)
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--edge-geodesic-mm", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--scan-first", required=True, type=Path)
    parser.add_argument("--scan-second", required=True, type=Path)
    parser.add_argument("--transform-first", required=True, type=Path)
    parser.add_argument("--transform-second", required=True, type=Path)
    parser.add_argument("--source-tests", required=True, type=Path)
    parser.add_argument("--parent-renderer", required=True, type=Path)
    parser.add_argument("--parent-renderer-tests", required=True, type=Path)
    parser.add_argument("--orchestrator", required=True, type=Path)
    parser.add_argument("--orchestrator-tests", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    _refuse_nonempty(args.output_root)

    # No physical scan hash or point-cloud open may precede this complete gate.
    v4, parent, cad_preflight = verify_pre_scan_approvals(
        v4_preregistration=args.preregistration.resolve(),
        parent_preregistration=args.scientific_preregistration.resolve(),
        terminal_parent_result=args.terminal_parent_result.resolve(),
        terminal_parent_report=args.terminal_parent_report.resolve(),
        terminal_parent_orchestration=args.terminal_parent_orchestration.resolve(),
        cad_reuse_audit_packet=args.cad_reuse_audit_packet.resolve(),
        cad_preflight=args.cad_preflight.resolve(),
        cad_reuse_approval=args.cad_reuse_approval.resolve(),
        source_runner_audit_packet=args.source_runner_audit_packet.resolve(),
        source_approval=args.source_runner_approval.resolve(),
        nodes=args.nodes.resolve(),
        edges=args.edges.resolve(),
        edge_geodesic_mm=args.edge_geodesic_mm.resolve(),
        mesh=args.mesh.resolve(),
        source_tests=args.source_tests.resolve(),
        parent_renderer=args.parent_renderer.resolve(),
        parent_renderer_tests=args.parent_renderer_tests.resolve(),
        orchestrator=args.orchestrator.resolve(),
        orchestrator_tests=args.orchestrator_tests.resolve(),
    )
    immutable = v4["immutable_inputs"]
    tokens = list(immutable["opaque_case_tokens_in_canonical_order"])
    scan_hashes = immutable["scan_sha256_without_loading_bytes"]
    transform_hashes = immutable["frozen_transform_semantic_sha256"]
    transform_first = PARENT.load_transform(
        args.transform_first, tokens[0], scan_hashes[tokens[0]], transform_hashes[tokens[0]]
    )
    transform_second = PARENT.load_transform(
        args.transform_second, tokens[1], scan_hashes[tokens[1]], transform_hashes[tokens[1]]
    )
    nodes, edges, weights, mesh = PARENT.load_cad_source_artifacts_before_scan(
        nodes_path=args.nodes,
        edges_path=args.edges,
        edge_weights_path=args.edge_geodesic_mm,
        mesh_path=args.mesh,
    )
    scan_accounting = {
        "physical_scan_hash_call_count": 0,
        "physical_scan_point_cloud_load_call_count": 0,
        "physical_scan_byte_open_operation_count": 0,
    }
    post_byte_stage = "FIRST_SCAN_HASH"
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_scan_accounting_journal(
        args.output_root, scan_accounting,
        stage="BEFORE_FIRST_SCAN_HASH", status="IN_PROGRESS",
    )
    try:
        # The following hash is the first physical source-byte access.  Every
        # subsequent path, including load/shape/interface failures, publishes
        # its exact attempted hash/load accounting before propagating failure.
        scan_accounting["physical_scan_hash_call_count"] += 1
        scan_accounting["physical_scan_byte_open_operation_count"] += 1
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        PARENT._require_hash(
            args.scan_first, scan_hashes[tokens[0]], f"{tokens[0]} scan"
        )
        post_byte_stage = "SECOND_SCAN_HASH"
        scan_accounting["physical_scan_hash_call_count"] += 1
        scan_accounting["physical_scan_byte_open_operation_count"] += 1
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        PARENT._require_hash(
            args.scan_second, scan_hashes[tokens[1]], f"{tokens[1]} scan"
        )
        post_byte_stage = "FIRST_POINT_CLOUD_LOAD"
        scan_accounting["physical_scan_point_cloud_load_call_count"] += 1
        scan_accounting["physical_scan_byte_open_operation_count"] += 1
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        first_cloud = o3d.io.read_point_cloud(str(args.scan_first.resolve()))
        post_byte_stage = "SECOND_POINT_CLOUD_LOAD"
        scan_accounting["physical_scan_point_cloud_load_call_count"] += 1
        scan_accounting["physical_scan_byte_open_operation_count"] += 1
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        second_cloud = o3d.io.read_point_cloud(str(args.scan_second.resolve()))
        post_byte_stage = "POINT_CLOUD_PRESENCE_GATE"
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        if not first_cloud.has_points() or not second_cloud.has_points():
            raise ValueError("physical scan is empty")
        post_byte_stage = "SOURCE_NORMAL_PRESENCE_GATE"
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        if not first_cloud.has_normals() or not second_cloud.has_normals():
            raise ValueError("EXP31 requires source normals; no estimation fallback")
        raw_first = np.asarray(first_cloud.points, dtype=np.float64)
        normals_first = np.asarray(first_cloud.normals, dtype=np.float64)
        raw_second = np.asarray(second_cloud.points, dtype=np.float64)
        normals_second = np.asarray(second_cloud.normals, dtype=np.float64)
        post_byte_stage = "SOURCE_ARRAY_SHAPE_GATE"
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        for token, points, normals in (
            (tokens[0], raw_first, normals_first),
            (tokens[1], raw_second, normals_second),
        ):
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or normals.shape != points.shape
            ):
                raise ValueError(f"{token} scan point/normal shape gate")
        post_byte_stage = "SOURCE_INTERFACE_AND_PARENT_SCIENCE"
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        semantic, visual, filtered = execute_source_v4(
            v4_prereg=v4,
            parent_prereg=parent,
            cad_preflight=cad_preflight,
            nodes=nodes,
            edges=edges,
            edge_weights=weights,
            mesh=mesh,
            raw_first=raw_first,
            normals_first=normals_first,
            raw_second=raw_second,
            normals_second=normals_second,
            transform_first=transform_first,
            transform_second=transform_second,
            first_token=tokens[0],
            second_token=tokens[1],
        )
    except Exception as error:
        _publish_post_byte_failure(
            args.output_root, scan_accounting,
            stage=post_byte_stage, error=error,
        )
        raise

    bindings = {
        "v4_preregistration_sha256": PARENT.CAD.sha256_path(args.preregistration),
        "parent_preregistration_sha256": PARENT.CAD.sha256_path(args.scientific_preregistration),
        "cad_reuse_approval_sha256": PARENT.CAD.sha256_path(args.cad_reuse_approval),
        "source_runner_approval_sha256": PARENT.CAD.sha256_path(args.source_runner_approval),
        "source_runner_sha256": PARENT.CAD.sha256_path(Path(__file__)),
        "interface_module_sha256": PARENT.CAD.sha256_path(Path(INTAKE.__file__)),
        "parent_execute_source_sha256": PARENT.CAD.sha256_path(Path(PARENT.__file__)),
    }
    semantic = dict(semantic) | {
        "bindings": bindings,
        "physical_scan_load_performed": True,
        **scan_accounting,
    }
    context = {
        "schema": "EXP_CAD_0031_EXECUTION_CONTEXT_V1",
        "experiment_id": EXPECTED_EXPERIMENT,
        "architecture": EXPECTED_ARCHITECTURE,
        "status": semantic["status"],
        "source_interface": semantic["source_interface"],
        "scientific_core": "EXP_CAD_0030_CFPDG_V3_UNCHANGED",
        "parent_visual_payload_emitted": visual is not None,
        "parent_renderer_required": visual is not None,
        "native_visual_status": "PENDING_PARENT_RENDERER" if visual is not None else "NOT_REACHED_SOURCE_INTERFACE_NONACTIONABLE",
        "bindings": bindings,
    }
    post_byte_stage = "PERSIST_SOURCE_ARTIFACTS"
    try:
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage=post_byte_stage, status="IN_PROGRESS",
        )
        _atomic_npy(
            args.output_root / "FIRST_RETAINED_ORIGINAL_ROW_IDS.npy",
            filtered["first_original_row_ids"],
        )
        _atomic_npy(
            args.output_root / "SECOND_RETAINED_ORIGINAL_ROW_IDS.npy",
            filtered["second_original_row_ids"],
        )
        PARENT.CAD.atomic_json(
            args.output_root / "SOURCE_RESULT.json", PARENT._jsonable(semantic)
        )
        if visual is not None:
            PARENT.CAD.atomic_json(
                args.output_root / "SOURCE_VISUAL_PAYLOAD.json",
                PARENT._jsonable(visual),
            )
            context["parent_visual_payload_file_sha256"] = PARENT.CAD.sha256_path(
                args.output_root / "SOURCE_VISUAL_PAYLOAD.json"
            )
        PARENT.CAD.atomic_json(
            args.output_root / "EXP31_EXECUTION_CONTEXT.json", context
        )
        _write_scan_accounting_journal(
            args.output_root, scan_accounting,
            stage="COMPLETE", status="COMPLETE",
        )
    except Exception as error:
        _publish_post_byte_failure(
            args.output_root, scan_accounting,
            stage=post_byte_stage, error=error,
        )
        raise
    print(json.dumps({"status": semantic["status"], "scientific_core_reached": visual is not None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
