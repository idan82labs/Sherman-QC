#!/usr/bin/env python3
"""Lock-bound physical source child for EXP_CAD_0034 CF-PDG V5.

The only physical byte access in this module is the four-operation sequence
frozen by the EXP34 preregistration: hash N009, hash N010, load N009, load
N010.  Every attempt is durably journalled before it occurs.  The exact EXP31
source filter runs once per member; actionable retained arrays then enter
``execute_parent_science_v5`` directly, without the EXP32 source-free intake.

The deterministic source tree contains no wall time, RSS, scan journal, or
full engine metrics.  Those values belong beneath the non-compared measurement
root.  No evaluator, role, label, gold, expected-count, or quality input is
accepted by this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Candidate code and numeric/native dependencies are deliberately absent from
# module import.  A physical child loads them only after the stdlib bootstrap,
# launch capability consumption, and full immutable authority verification.
GOVERNANCE: Any = None
INTAKE: Any = None
V5: Any = None
np: Any = None
o3d: Any = None


EXPERIMENT_ID = "EXP_CAD_0034"
ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_PHYSICAL_SOURCE_WITH_EXACT_V4_INTERFACE"
PREREGISTRATION_SHA256 = "323c7cf72884f33e6358471f80b100c5d0e00ec0ca4b795cff5aa6e9afea5058"
LOCK_SCHEMA = "EXP_CAD_0034_PHYSICAL_EXECUTION_LOCK_V1"
LOCK_STATUS = "IMMUTABLE_LOCKED_AWAITING_DETACHED_APPROVAL"
LOCK_RELATIVE_PATH = "docs/research/2026-08-30_exp_cad_0034_physical_execution_lock.json"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json"
)
CONTRACT_RELATIVE_PATH = "scripts/exp_cad_0034_execution_contract.py"
RUNNER_RELATIVE_PATH = "scripts/run_exp_cad_0034_cfpdg_source.py"
LAUNCH_CLAIM_SCHEMA = "EXP_CAD_0034_PHYSICAL_LAUNCH_CLAIM_V1"
SOURCE_CAPABILITY_SCHEMA = "EXP_CAD_0034_SOURCE_LAUNCH_CAPABILITY_V1"
SOURCE_CHILD_CLAIM_SCHEMA = "EXP_CAD_0034_SOURCE_LAUNCH_CHILD_CLAIM_V1"
OUTER_SOURCE_SCHEMA = "EXP_CAD_0034_PHYSICAL_SOURCE_ENVELOPE_V1"
ENGINE_ACCOUNTING_SCHEMA = "EXP_CAD_0034_ENGINE_INVOCATION_ACCOUNTING_V1"
SCAN_JOURNAL_SCHEMA = "EXP_CAD_0034_SCAN_ACCOUNTING_JOURNAL_V1"
TRANSFORM_SCHEMA = "EXP_CAD_0015_NEW_PHYSICAL_REGISTRATION_CASE_V2"
PARENT_PREREGISTRATION_SCHEMA = "EXP_CAD_0030_CFPDG_V3_PREREGISTRATION_V1"
EXP31_PREREGISTRATION_SCHEMA = "EXP_CAD_0031_CFPDG_V4_PREREGISTRATION_V1"
EXP32_PREREGISTRATION_SCHEMA = (
    "EXP_CAD_0032_CFPDG_V5_EXECUTION_EQUIVALENT_PREREGISTRATION_V1"
)
PARENT_SEMANTIC_SCHEMA = "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1"
PARENT_VISUAL_SCHEMA = "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1"
EXP32_METRICS_SCHEMA = "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1"
TOKENS = ("N009", "N010")
EXPECTED_ROW_COUNTS = {"N009": 1_156_302, "N010": 1_323_486}
EXPECTED_SCAN_SHA256 = {
    "N009": "3567fd96870a93f39fcc1e203cef847bb44b72f5a5868e355693c1bfadd87833",
    "N010": "823e6a7dc7c42091c6565393c82122eea9fbed2bda3e079247189d463a28f1b3",
}
EXPECTED_TRANSFORM_SHA256 = {
    "N009": "02e85cc8bc4e58c0ef0f9ddfb5eca097d26d3d3492378781babf42b65da67aca",
    "N010": "766aef6c8d286798b793f32c05405ca00b2bd67ca9656fab2f1daf360b372b14",
}
EXPECTED_PARENT_PREREGISTRATION_SHA256 = (
    "c9129ccc103d45c89ccdc67eef938b242ccdbb081d3d0f261145cc5a5b9b8368"
)
EXPECTED_FILTER_SHA256 = (
    "029570cbb94007b33da39eb0b4d201a653200e8f198be1a88f1a37fcc5c830ac"
)
EXPECTED_ENGINE_SHA256 = (
    "ecbcdd434a94dff5606cbbac960f24e2472f64deb2934faf8f638a1bb2cd2e85"
)
EXPECTED_NODE_COUNT = 8_077
EXPECTED_EDGE_COUNT = 19_267
EXPECTED_UNIQUE_MEMBER_FIT_COUNT = 38
EXPECTED_KD_NODE_QUERY_COUNT = 306_926
EXPECTED_ROBUSTNESS_ROW_COUNT = 32


def _strict_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_write(path: Path, value: Any) -> str:
    """Use the verified shared streaming writer after runtime activation."""

    if GOVERNANCE is None:
        raise RuntimeError("EXP34 governance runtime is not authority-activated")
    return GOVERNANCE.canonical_json_write(path, value)


def verify_execution_authority(**kwargs: Any) -> Mapping[str, Any]:
    if GOVERNANCE is None:
        raise RuntimeError("EXP34 governance runtime is not authority-activated")
    return GOVERNANCE.verify_execution_authority(**kwargs)


def _sha256_path_stdlib(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_absolute_path(
    path: Path, expected_relative: str, *, must_exist: bool, label: str
) -> Path:
    expected = ROOT / expected_relative
    raw = Path(path)
    if not raw.is_absolute() or str(raw) != expected.as_posix():
        raise ValueError(f"{label} is not the exact canonical repository path")
    _reject_symlink_components(raw, label=label)
    resolved = raw.resolve(strict=must_exist)
    if resolved != expected.resolve(strict=must_exist):
        raise ValueError(f"{label} uses a path alias or symlink")
    return resolved


def _reject_symlink_components(path: Path, *, label: str) -> None:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    probe = Path(raw.anchor)
    for part in raw.parts[1:]:
        probe = probe / part
        if probe.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {probe}")


def _validate_minimal_binding(
    lock: Mapping[str, Any], *, key: str, expected_relative: str
) -> Path:
    rows = lock.get("bound_files")
    row = rows.get(key) if isinstance(rows, Mapping) else None
    if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"minimal lock binding is malformed: {key}")
    if row.get("path") != expected_relative:
        raise ValueError(f"minimal lock path binding drift: {key}")
    path = _exact_absolute_path(
        ROOT / expected_relative, expected_relative, must_exist=True, label=key
    )
    if (
        isinstance(row.get("size_bytes"), bool)
        or not isinstance(row.get("size_bytes"), int)
        or int(row["size_bytes"]) != int(path.stat().st_size)
        or not isinstance(row.get("sha256"), str)
        or _sha256_path_stdlib(path) != row["sha256"]
    ):
        raise ValueError(f"minimal lock byte binding drift: {key}")
    return path


def stdlib_bootstrap_lock(args: argparse.Namespace) -> Mapping[str, Any]:
    """Verify lock bytes plus runner/contract rows before importing either."""

    lock_path = _exact_absolute_path(
        args.lock, LOCK_RELATIVE_PATH, must_exist=True, label="EXP34 lock"
    )
    if _sha256_path_stdlib(lock_path) != args.lock_sha256:
        raise ValueError("EXP34 lock SHA-256 mismatch in stdlib bootstrap")
    lock = _load_json(lock_path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("status") != LOCK_STATUS
        or lock.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or lock.get("physical_source_execution_authorized") is not False
        or lock.get("evaluator_input_authorized") is not False
    ):
        raise ValueError("EXP34 minimal lock identity/status mismatch")
    _validate_minimal_binding(
        lock, key="exp34_contract", expected_relative=CONTRACT_RELATIVE_PATH
    )
    _validate_minimal_binding(
        lock, key="exp34_source_runner", expected_relative=RUNNER_RELATIVE_PATH
    )
    scans = lock.get("physical_scans")
    if not isinstance(scans, Mapping) or set(scans) != {"scan_N009", "scan_N010"}:
        raise ValueError("EXP34 minimal physical scan rows are malformed")
    for token in TOKENS:
        row = scans[f"scan_{token}"]
        if (
            not isinstance(row, Mapping)
            or row.get("opaque_token") != token
            or row.get("expected_sha256") != EXPECTED_SCAN_SHA256[token]
            or row.get("expected_raw_point_row_count") != EXPECTED_ROW_COUNTS[token]
            or row.get("bytes_opened_or_hashed_during_lock_construction") is not False
        ):
            raise ValueError(f"EXP34 minimal scan metadata drift: {token}")
        # Deliberately do not resolve, stat, hash, or open row["path"].
    return lock


def _canonical_full_command(raw_argv: Sequence[str]) -> list[str]:
    return [
        # Preserve the attested invocation spelling (.venv/bin/python); the
        # fresh environment row separately binds its resolved realpath.
        str(Path(sys.executable)),
        str(Path(__file__).resolve(strict=True)),
        *[str(value) for value in raw_argv],
    ]


def _validate_launch_claim(
    *, args: argparse.Namespace, lock: Mapping[str, Any]
) -> Mapping[str, Any]:
    authorized = lock.get("authorized_outputs")
    if not isinstance(authorized, Mapping) or "launch_claim" not in authorized:
        raise ValueError("EXP34 lock lacks authorized launch claim")
    launch_path = _exact_absolute_path(
        args.launch_claim,
        str(authorized["launch_claim"]),
        must_exist=True,
        label="EXP34 launch claim",
    )
    if _sha256_path_stdlib(launch_path) != args.launch_claim_sha256:
        raise ValueError("EXP34 launch-claim SHA-256 mismatch")
    claim = _load_json(launch_path)
    if (
        claim.get("schema") != LAUNCH_CLAIM_SCHEMA
        or claim.get("experiment_id") != EXPERIMENT_ID
        or claim.get("architecture") != ARCHITECTURE
        or claim.get("status") != "IN_PROGRESS_CONSUMED_BEFORE_FIRST_CHILD"
        or claim.get("invocation_ordinal") != 1
        or claim.get("preregistration_path") != str(args.preregistration)
        or claim.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or claim.get("lock_path") != str(args.lock)
        or claim.get("lock_sha256") != args.lock_sha256
        or claim.get("detached_approval_path") != str(args.detached_approval)
        or claim.get("detached_approval_sha256") != args.detached_approval_sha256
        or claim.get("maximum_primary_source_children") != 1
        or claim.get("maximum_replay_source_children") != 1
        or claim.get("physical_source_execution_authorized") is not True
        or claim.get("evaluator_input_authorized") is not False
    ):
        raise ValueError("EXP34 orchestration launch claim identity/authority mismatch")
    expected_outputs = {
        str(key): str(ROOT / str(relative))
        for key, relative in sorted(authorized.items())
    }
    if claim.get("resolved_output_paths") != expected_outputs:
        raise ValueError("EXP34 launch claim output topology mismatch")
    inputs = claim.get("resolved_inputs")
    required_inputs = {
        "exp34_source_runner",
        "exp34_monitor",
        "exp34_orchestrator",
        "renderer",
        "parent_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "accepted_mesh",
        "scan_N009",
        "scan_N010",
        "transform_N009",
        "transform_N010",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != required_inputs:
        raise ValueError("EXP34 launch claim resolved-input inventory mismatch")
    bound = lock.get("bound_files", {})
    scans = lock.get("physical_scans", {})
    for key, path_text in inputs.items():
        row = scans.get(key) if key in scans else bound.get(key)
        if not isinstance(row, Mapping) or path_text != str(ROOT / str(row.get("path"))):
            raise ValueError(f"EXP34 launch claim input binding mismatch: {key}")
    expected_arg_paths = {
        "exp34_source_runner": Path(__file__).resolve(),
        "parent_preregistration": args.parent_preregistration,
        "exp31_preregistration": args.exp31_preregistration,
        "exp32_preregistration": args.v5_preregistration,
        "cad_preflight": args.cad_preflight,
        "cad_nodes": args.nodes,
        "cad_edges": args.edges,
        "cad_edge_geodesic_mm": args.edge_geodesic_mm,
        "accepted_mesh": args.mesh,
        "scan_N009": args.scan_first,
        "scan_N010": args.scan_second,
        "transform_N009": args.transform_first,
        "transform_N010": args.transform_second,
    }
    for key, path in expected_arg_paths.items():
        if inputs.get(key) != str(path):
            raise ValueError(f"EXP34 launch claim differs from source CLI: {key}")
    return claim


def validate_and_claim_source_capability(
    *,
    args: argparse.Namespace,
    lock: Mapping[str, Any],
    raw_argv: Sequence[str],
) -> Mapping[str, Any]:
    """Consume one orchestrator-issued source capability before other imports."""

    _validate_launch_claim(args=args, lock=lock)
    expected_ordinal = 1 if args.mode == "PRIMARY" else 2
    mode_outputs = {
        "PRIMARY": {
            "output_root": "primary_source",
            "measurement_root": "primary_measurement",
            "scan_accounting_journal": "primary_scan_accounting_journal",
            "source_launch_capability": "primary_source_launch_capability",
            "source_launch_child_claim": "primary_source_child_claim",
        },
        "REPLAY": {
            "output_root": "replay_source",
            "measurement_root": "replay_measurement",
            "scan_accounting_journal": "replay_scan_accounting_journal",
            "source_launch_capability": "replay_source_launch_capability",
            "source_launch_child_claim": "replay_source_child_claim",
        },
    }[args.mode]
    authorized = lock["authorized_outputs"]
    for argument, locked_key in mode_outputs.items():
        if str(getattr(args, argument)) != str(ROOT / authorized[locked_key]):
            raise ValueError(f"{argument} differs from the locked {args.mode} schedule")
    capability_path = Path(args.source_launch_capability)
    child_claim_path = Path(args.source_launch_child_claim)
    expected_capability = args.measurement_root / "SOURCE_LAUNCH_CAPABILITY.json"
    expected_child_claim = args.measurement_root / "SOURCE_CHILD.claim.json"
    for supplied, expected, label, must_exist in (
        (capability_path, expected_capability, "source launch capability", True),
        (child_claim_path, expected_child_claim, "source launch child claim", False),
    ):
        _reject_symlink_components(supplied, label=label)
        if (
            not supplied.is_absolute()
            or str(supplied) != str(expected)
            or supplied.resolve(strict=must_exist) != expected.resolve(strict=must_exist)
        ):
            raise ValueError(f"{label} path is not the exact mode-specific path")
    command = _canonical_full_command(raw_argv)
    command_sha256 = hashlib.sha256(_strict_canonical_bytes(command)).hexdigest()
    capability = _load_json(capability_path)
    exact_keys = {
        "schema",
        "experiment_id",
        "status",
        "source_child_ordinal",
        "mode",
        "command_argv",
        "command_sha256",
        "preregistration_path",
        "preregistration_sha256",
        "lock_path",
        "lock_sha256",
        "detached_approval_path",
        "detached_approval_sha256",
        "launch_claim_path",
        "launch_claim_sha256",
        "output_root",
        "measurement_root",
        "scan_accounting_journal",
        "source_launch_capability_path",
        "source_launch_child_claim_path",
        "evaluator_input_authorized",
    }
    if set(capability) != exact_keys:
        raise ValueError("EXP34 source capability field inventory mismatch")
    expected_values = {
        "schema": SOURCE_CAPABILITY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUTHORIZED_NOT_CLAIMED",
        "source_child_ordinal": expected_ordinal,
        "mode": args.mode,
        "command_argv": command,
        "command_sha256": command_sha256,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "lock_path": str(args.lock),
        "lock_sha256": args.lock_sha256,
        "detached_approval_path": str(args.detached_approval),
        "detached_approval_sha256": args.detached_approval_sha256,
        "launch_claim_path": str(args.launch_claim),
        "launch_claim_sha256": args.launch_claim_sha256,
        "output_root": str(args.output_root),
        "measurement_root": str(args.measurement_root),
        "scan_accounting_journal": str(args.scan_accounting_journal),
        "source_launch_capability_path": str(capability_path),
        "source_launch_child_claim_path": str(child_claim_path),
        "evaluator_input_authorized": False,
    }
    if capability != expected_values:
        raise ValueError("EXP34 source capability command/authority binding mismatch")
    capability_sha256 = _sha256_path_stdlib(capability_path)
    child_claim = {
        "schema": SOURCE_CHILD_CLAIM_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "CLAIMED_BEFORE_STATIC_OR_SCAN_ACCESS",
        "source_child_ordinal": expected_ordinal,
        "mode": args.mode,
        "source_launch_capability_path": str(capability_path),
        "source_launch_capability_sha256": capability_sha256,
        "command_sha256": command_sha256,
        "launch_claim_path": str(args.launch_claim),
        "launch_claim_sha256": args.launch_claim_sha256,
    }
    _o_excl_canonical_write(child_claim_path, child_claim)
    return child_claim


def _o_excl_canonical_write(path: Path, payload: Mapping[str, Any]) -> str:
    data = _strict_canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest


def _load_exact_module(path: Path, canonical_name: str) -> Any:
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(canonical_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build exact module spec: {canonical_name}")
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.pop(canonical_name, None)
    sys.modules[canonical_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(canonical_name, None)
        if prior is not None:
            sys.modules[canonical_name] = prior
        raise
    origin = Path(module.__file__).resolve(strict=True)
    if origin != path:
        raise ImportError(f"exact module origin drift: {canonical_name}")
    return module


def activate_governance(lock: Mapping[str, Any]) -> Any:
    global GOVERNANCE
    row = lock["bound_files"]["exp34_contract"]
    path = ROOT / row["path"]
    GOVERNANCE = _load_exact_module(
        path, "scripts.exp_cad_0034_execution_contract"
    )
    if (
        GOVERNANCE.LOCK_SCHEMA != LOCK_SCHEMA
        or GOVERNANCE.LOCK_STATUS != LOCK_STATUS
        or GOVERNANCE.PREREGISTRATION_SHA256 != PREREGISTRATION_SHA256
        or Path(GOVERNANCE.__file__).resolve(strict=True) != path.resolve(strict=True)
    ):
        raise ImportError("verified governance module identity drift")
    return GOVERNANCE


def activate_science_runtime(lock: Mapping[str, Any]) -> None:
    """Import numeric/native and science modules only after full authority."""

    global np, o3d, INTAKE, V5
    np = importlib.import_module("numpy")
    o3d = importlib.import_module("open3d")
    importlib.import_module("scripts")
    bound = lock["bound_files"]

    def locked(key: str, name: str) -> Any:
        row = bound[key]
        path = ROOT / row["path"]
        if (
            int(path.stat().st_size) != int(row["size_bytes"])
            or _sha256_path_stdlib(path) != row["sha256"]
        ):
            raise ImportError(f"locked module changed after authority verification: {key}")
        return _load_exact_module(path, name)

    locked("frozen_parent_core", "scripts.exp_cad_0029_cfpdg_core")
    locked("frozen_parent_science", "scripts.exp_cad_0029_cfpdg_source")
    locked("frozen_parent_runner", "scripts.run_exp_cad_0029_cfpdg_source")
    INTAKE = locked("exp31_filter_module", "scripts.exp_cad_0031_source_interface")
    V5 = locked("exp32_v5_engine", "scripts.exp_cad_0032_cfpdg_v5_engine")


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(GOVERNANCE.canonical_bytes(value)).hexdigest()


def _validate_preregistrations(
    exp34: Mapping[str, Any],
    parent: Mapping[str, Any],
    exp31: Mapping[str, Any],
    exp32: Mapping[str, Any],
) -> None:
    if (
        exp34.get("schema")
        != "EXP_CAD_0034_CFPDG_V5_PHYSICAL_SUCCESSOR_PREREGISTRATION_V1"
        or exp34.get("experiment_id") != EXPERIMENT_ID
        or exp34.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("unexpected EXP34 preregistration identity")
    if (
        parent.get("schema") != PARENT_PREREGISTRATION_SCHEMA
        or parent.get("experiment_id") != "EXP_CAD_0030"
        or parent.get("architecture")
        != "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
    ):
        raise ValueError("unexpected frozen EXP30 preregistration identity")
    if (
        exp31.get("schema") != EXP31_PREREGISTRATION_SCHEMA
        or exp31.get("experiment_id") != "EXP_CAD_0031"
        or exp31.get("architecture")
        != "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V4_SOURCE_INTERFACE_FILTER"
    ):
        raise ValueError("unexpected frozen EXP31 preregistration identity")
    if (
        exp32.get("schema") != EXP32_PREREGISTRATION_SCHEMA
        or exp32.get("experiment_id") != "EXP_CAD_0032"
        or exp32.get("architecture") != V5.ARCHITECTURE
    ):
        raise ValueError("unexpected qualified EXP32 preregistration identity")

    exp34_tokens = exp34.get("immutable_source_lineage", {}).get(
        "opaque_case_tokens_in_canonical_order"
    )
    parent_tokens = parent.get("immutable_inputs", {}).get(
        "opaque_case_tokens_in_canonical_order"
    )
    exp31_tokens = exp31.get("immutable_inputs", {}).get(
        "opaque_case_tokens_in_canonical_order"
    )
    if list(exp34_tokens or []) != list(TOKENS):
        raise ValueError("EXP34 opaque token order drift")
    if list(parent_tokens or []) != list(TOKENS) or list(exp31_tokens or []) != list(TOKENS):
        raise ValueError("parent source token/order binding drift")

    physical = exp34.get("immutable_source_lineage", {}).get("physical_scans")
    transforms = exp34.get("immutable_source_lineage", {}).get("frozen_transforms")
    if not isinstance(physical, Mapping) or not isinstance(transforms, Mapping):
        raise ValueError("EXP34 source lineage is missing")
    for token in TOKENS:
        scan = physical.get(token)
        transform = transforms.get(token)
        if (
            not isinstance(scan, Mapping)
            or scan.get("expected_sha256") != EXPECTED_SCAN_SHA256[token]
            or scan.get("expected_raw_point_row_count") != EXPECTED_ROW_COUNTS[token]
            or not isinstance(transform, Mapping)
            or transform.get("expected_semantic_file_sha256")
            != EXPECTED_TRANSFORM_SHA256[token]
        ):
            raise ValueError(f"EXP34 immutable source row drift: {token}")

    interface = exp34.get("exact_source_interface_contract")
    science = exp34.get("scientific_execution_contract")
    if not isinstance(interface, Mapping) or not isinstance(science, Mapping):
        raise ValueError("EXP34 execution contracts are missing")
    if (
        interface.get("normal_norm_epsilon_inclusive") != 1e-12
        or interface.get("maximum_invalid_fraction_per_member") != 0.01
        or interface.get("minimum_valid_row_count_per_member") != 100_000
        or science.get("cad_node_count") != EXPECTED_NODE_COUNT
        or science.get("cad_edge_count") != EXPECTED_EDGE_COUNT
        or science.get("unique_member_fit_count_on_science_path")
        != EXPECTED_UNIQUE_MEMBER_FIT_COUNT
        or science.get("required_full_fit_kd_node_query_count_on_science_path")
        != EXPECTED_KD_NODE_QUERY_COUNT
    ):
        raise ValueError("EXP34 frozen numeric contract drift")
    dependencies = exp34.get("hash_bound_execution_dependencies", {})
    if (
        dependencies.get("exact_exp31_filter_module", {}).get("sha256")
        != EXPECTED_FILTER_SHA256
        or dependencies.get("qualified_v5_engine", {}).get("sha256")
        != EXPECTED_ENGINE_SHA256
        or dependencies.get("frozen_parent_execute_source", {}).get("sha256")
        != V5.FROZEN_RUNNER.CAD.sha256_path(Path(V5.FROZEN_RUNNER.__file__))
    ):
        raise ValueError("EXP34 execution dependency contract drift")


def validate_transform_payload(
    payload: Mapping[str, Any],
    *,
    token: str,
    scan_sha256: str,
    expected_row_count: int,
) -> np.ndarray:
    """Accept only the exact EXP15 raw-to-reference proper-rigid transform."""

    required = {
        "schema": TRANSFORM_SCHEMA,
        "experiment_id": "EXP_CAD_0015",
        "case_token": token,
        "scan_sha256": scan_sha256,
        "raw_scan_point_count": int(expected_row_count),
        "alignment_abstained": False,
        "evaluator_data_opened": False,
        "gold_or_expected_counts_used": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{token} transform linkage mismatch: {key}")
    if payload.get("generalization_evidence") is not False:
        raise ValueError(f"{token} transform generalization flag is not sealed")
    if "selected_absolute_transform" in payload or "transform" in payload:
        raise ValueError(f"{token} transform contains a forbidden alternate field")
    if "raw_scan_to_reference_transform" not in payload:
        raise ValueError(f"{token} raw_scan_to_reference_transform is unavailable")
    transform = np.asarray(payload["raw_scan_to_reference_transform"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{token} transform must be finite float64 4-by-4")
    if not np.array_equal(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{token} transform bottom row is not exact homogeneous form")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError(f"{token} transform rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{token} transform determinant is not +1")
    return np.ascontiguousarray(transform, dtype=np.float64)


def load_validated_transform(
    path: Path,
    *,
    token: str,
    scan_sha256: str,
    expected_row_count: int,
    expected_semantic_sha256: str,
) -> np.ndarray:
    if GOVERNANCE.sha256_path(path) != expected_semantic_sha256:
        raise ValueError(f"{token} transform semantic SHA-256 mismatch")
    return validate_transform_payload(
        _load_json(path),
        token=token,
        scan_sha256=scan_sha256,
        expected_row_count=expected_row_count,
    )


def materialize_loaded_cloud(
    cloud: Any, *, token: str, expected_row_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize parser output without removal, repair, sorting, or estimation."""

    points = np.ascontiguousarray(np.asarray(cloud.points), dtype=np.float64)
    normals = np.ascontiguousarray(np.asarray(cloud.normals), dtype=np.float64)
    expected_shape = (int(expected_row_count), 3)
    if points.shape != expected_shape:
        raise ValueError(f"{token} raw point row/shape gate failed")
    if normals.shape != points.shape:
        raise ValueError(f"{token} supplied normal row/shape gate failed")
    if points.dtype != np.float64 or normals.dtype != np.float64:
        raise RuntimeError(f"{token} numeric materialization dtype drift")
    if not points.flags.c_contiguous or not normals.flags.c_contiguous:
        raise RuntimeError(f"{token} numeric materialization is not C-contiguous")
    return points, normals


def _new_scan_accounting() -> dict[str, Any]:
    return {
        "physical_scan_hash_call_count": 0,
        "physical_scan_point_cloud_load_call_count": 0,
        "physical_scan_byte_open_operation_count": 0,
        "attempted_operations": [],
    }


def _journal_payload(
    accounting: Mapping[str, Any],
    *,
    mode: str,
    stage: str,
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCAN_JOURNAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "status": status,
        "stage": stage,
        "physical_scan_hash_call_count": int(
            accounting["physical_scan_hash_call_count"]
        ),
        "physical_scan_point_cloud_load_call_count": int(
            accounting["physical_scan_point_cloud_load_call_count"]
        ),
        "physical_scan_byte_open_operation_count": int(
            accounting["physical_scan_byte_open_operation_count"]
        ),
        "attempted_operations": [dict(row) for row in accounting["attempted_operations"]],
        "evaluator_role_label_gold_quality_or_expected_count_access_count": 0,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    return payload


def write_scan_journal(
    journal_path: Path,
    accounting: Mapping[str, Any],
    *,
    mode: str,
    stage: str,
    status: str,
    error: BaseException | None = None,
) -> None:
    canonical_json_write(
        journal_path,
        _journal_payload(
            accounting, mode=mode, stage=stage, status=status, error=error
        ),
    )


def _checkpoint_operation(
    journal_path: Path,
    accounting: MutableMapping[str, Any],
    *,
    mode: str,
    operation: str,
    token: str,
) -> None:
    if operation == "SHA256":
        accounting["physical_scan_hash_call_count"] += 1
    elif operation == "POINT_CLOUD_LOAD":
        accounting["physical_scan_point_cloud_load_call_count"] += 1
    else:
        raise ValueError(f"unknown physical scan operation: {operation}")
    accounting["physical_scan_byte_open_operation_count"] += 1
    accounting["attempted_operations"].append(
        {
            "attempt_ordinal": len(accounting["attempted_operations"]) + 1,
            "opaque_token": token,
            "operation": operation,
        }
    )
    write_scan_journal(
        journal_path,
        accounting,
        mode=mode,
        stage=f"BEFORE_{token}_{operation}",
        status="IN_PROGRESS",
    )


def _scan_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def perform_ordered_scan_access(
    *,
    mode: str,
    scan_first: Path,
    scan_second: Path,
    journal_path: Path,
    accounting: MutableMapping[str, Any],
    expected_hashes: Mapping[str, str] = EXPECTED_SCAN_SHA256,
    sha256_func: Callable[[Path], str] | None = None,
    point_cloud_reader: Callable[..., Any] | None = None,
) -> tuple[Any, Any]:
    """Perform the exact journal-before-access N009/N010 sequence."""

    if mode not in {"PRIMARY", "REPLAY"}:
        raise ValueError("EXP34 source mode must be PRIMARY or REPLAY")
    hasher = _scan_sha256 if sha256_func is None else sha256_func
    reader = o3d.io.read_point_cloud if point_cloud_reader is None else point_cloud_reader
    scan_paths = {TOKENS[0]: Path(scan_first), TOKENS[1]: Path(scan_second)}
    for token in TOKENS:
        _checkpoint_operation(
            journal_path,
            accounting,
            mode=mode,
            operation="SHA256",
            token=token,
        )
        observed = hasher(scan_paths[token])
        if observed != expected_hashes[token]:
            raise ValueError(f"{token} physical scan SHA-256 mismatch")
    clouds: dict[str, Any] = {}
    for token in TOKENS:
        _checkpoint_operation(
            journal_path,
            accounting,
            mode=mode,
            operation="POINT_CLOUD_LOAD",
            token=token,
        )
        clouds[token] = reader(
            str(scan_paths[token]),
            format="auto",
            remove_nan_points=False,
            remove_infinite_points=False,
            print_progress=False,
        )
    return clouds[TOKENS[0]], clouds[TOKENS[1]]


def _retained_identity_projection(filtered: Mapping[str, Any]) -> dict[str, Any]:
    identities = np.asarray(filtered["original_row_ids"], dtype=np.uint64)
    report = filtered["report"]
    return {
        "member_token": report["member_token"],
        "retained_row_count": int(len(identities)),
        "retained_identity_sequence_sha256": INTAKE.identity_sha256(
            identities, sort_as_set=False
        ),
        "retained_identity_set_sha256": INTAKE.identity_sha256(identities),
    }


def _zero_engine_accounting() -> dict[str, Any]:
    return {
        "schema": ENGINE_ACCOUNTING_SCHEMA,
        "v5_engine_invocation_count": 0,
        "unique_member_fit_count": 0,
        "kd_node_query_count": 0,
        "local_fit_call_count": 0,
        "robustness_row_count": 0,
        "exp32_engine_metrics_payload_present": False,
    }


def _science_engine_accounting(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if (
        metrics.get("schema") != EXP32_METRICS_SCHEMA
        or metrics.get("experiment_id") != "EXP_CAD_0032"
        or metrics.get("architecture") != V5.ARCHITECTURE
    ):
        raise ValueError("qualified V5 engine metrics identity drift")
    counters = metrics.get("operation_counters")
    if not isinstance(counters, Mapping):
        raise ValueError("qualified V5 engine metrics lack operation counters")
    projected: dict[str, int] = {}
    for key, value in counters.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"non-integer V5 operation counter: {key}")
        if int(value) < 0:
            raise ValueError(f"negative V5 operation counter: {key}")
        projected[str(key)] = int(value)
    exact = {
        "unique_member_fit_count": EXPECTED_UNIQUE_MEMBER_FIT_COUNT,
        "kd_node_query_count": EXPECTED_KD_NODE_QUERY_COUNT,
        "robustness_row_count": EXPECTED_ROBUSTNESS_ROW_COUNT,
    }
    for key, expected in exact.items():
        if projected.get(key) != expected or metrics.get(key) != expected:
            raise ValueError(f"qualified V5 exact science counter drift: {key}")
    if "local_fit_call_count" not in projected:
        raise ValueError("qualified V5 local-fit counter is missing")
    if (
        metrics.get("full_fit_inventory_gate_passed") is not True
        or metrics.get("full_fit_batched_kd_gate_passed") is not True
        or metrics.get("full_fit_kd_query_gate_passed") is not True
        or metrics.get("fit_or_robustness_row_skipped") is not False
        or metrics.get("scientific_threshold_or_gate_changed") is not False
    ):
        raise ValueError("qualified V5 execution gate failed")
    return {
        "schema": ENGINE_ACCOUNTING_SCHEMA,
        "v5_engine_invocation_count": 1,
        "unique_member_fit_count": projected["unique_member_fit_count"],
        "kd_node_query_count": projected["kd_node_query_count"],
        "local_fit_call_count": projected["local_fit_call_count"],
        "robustness_row_count": projected["robustness_row_count"],
        "exp32_engine_metrics_payload_present": True,
        "deterministic_exp32_operation_counter_projection": dict(
            sorted(projected.items())
        ),
    }


def execute_source_exp34(
    *,
    exp34_prereg: Mapping[str, Any],
    parent_prereg: Mapping[str, Any],
    exp31_prereg: Mapping[str, Any],
    v5_prereg: Mapping[str, Any],
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
) -> dict[str, Any]:
    """Source-free numeric call surface used by tests and the physical child."""

    _validate_preregistrations(exp34_prereg, parent_prereg, exp31_prereg, v5_prereg)
    if (first_token, second_token) != TOKENS:
        raise ValueError("EXP34 source token/order mismatch")
    first_points = np.asarray(raw_first)
    second_points = np.asarray(raw_second)
    first_ids = np.arange(len(first_points), dtype=np.uint64)
    second_ids = np.arange(len(second_points), dtype=np.uint64)

    # These are the only two intake calls.  The V5 filter and source-free V5
    # execute_source wrapper are intentionally neither imported nor invoked.
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
    first_report = dict(first["report"])
    second_report = dict(second["report"])
    if (
        first_report.get("schema") != INTAKE.SCHEMA
        or second_report.get("schema") != INTAKE.SCHEMA
        or first_report.get("member_token") != first_token
        or second_report.get("member_token") != second_token
    ):
        raise RuntimeError("exact EXP31 interface report identity drift")
    retained = {
        "schema": "EXP_CAD_0034_RETAINED_IDENTITY_HASHES_V1",
        "first": _retained_identity_projection(first),
        "second": _retained_identity_projection(second),
    }
    common = {
        "status": pair["status"],
        "reason": None if pair["actionable"] else "SOURCE_INTERFACE_ACTIONABILITY_GATE",
        "source_interface": {
            "first": first_report,
            "second": second_report,
            "pair": dict(pair),
        },
        "retained_identity_hashes": retained,
    }
    if not pair["actionable"]:
        return {
            **common,
            "scientific_execute_source_reached": False,
            "semantic_payload": None,
            "visual_payload": None,
            "engine_metrics": None,
            "engine_accounting": _zero_engine_accounting(),
        }

    semantic, visual, metrics = V5.execute_parent_science_v5(
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
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("schema") != PARENT_SEMANTIC_SCHEMA
        or semantic.get("experiment_id") != "EXP_CAD_0030"
        or semantic.get("architecture")
        != "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
    ):
        raise RuntimeError("exact nested EXP30 semantic identity drift")
    if (
        not isinstance(visual, Mapping)
        or visual.get("schema") != PARENT_VISUAL_SCHEMA
    ):
        raise RuntimeError("exact nested EXP30 visual identity drift")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("qualified V5 metrics payload is missing")
    accounting = _science_engine_accounting(metrics)
    return {
        **common,
        "status": semantic.get("status"),
        "reason": semantic.get("reason"),
        "scientific_execute_source_reached": True,
        "semantic_payload": semantic,
        "visual_payload": visual,
        "engine_metrics": metrics,
        "engine_accounting": accounting,
    }


def _base_outer_envelope(computation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": OUTER_SOURCE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "status": computation["status"],
        "reason": computation["reason"],
        "scientific_execute_source_reached": bool(
            computation["scientific_execute_source_reached"]
        ),
        "source_interface": computation["source_interface"],
        "retained_identity_hashes": computation["retained_identity_hashes"],
        "engine_invocation_accounting": computation["engine_accounting"],
        "physical_scan_load_performed": True,
        "physical_scan_hash_call_count": 2,
        "physical_scan_point_cloud_load_call_count": 2,
        "physical_scan_byte_open_operation_count": 4,
        "bindings": {
            "exp34_preregistration_sha256": PREREGISTRATION_SHA256,
            "exp30_preregistration_sha256": EXPECTED_PARENT_PREREGISTRATION_SHA256,
            "exp31_filter_module_sha256": EXPECTED_FILTER_SHA256,
            "qualified_v5_engine_sha256": EXPECTED_ENGINE_SHA256,
        },
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "opaque_member_attribution_only": True,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_OPAQUE_PAIR_DIAGNOSTIC",
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
        "runtime_values_inside_scientific_payload": False,
        "silent_nested_identity_relabeling_performed": False,
    }


def _refuse_nonempty_source_root(path: Path) -> None:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("source output root must be absent or empty")


def publish_source_artifacts(
    *,
    output_root: Path,
    measurement_root: Path,
    computation: MutableMapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically publish deterministic source files, writing terminal result last."""

    output_root = Path(output_root)
    measurement_root = Path(measurement_root)
    output_root.mkdir(parents=True, exist_ok=True)
    measurement_root.mkdir(parents=True, exist_ok=True)
    interface = computation["source_interface"]
    artifact_sha256 = {
        "first_source_interface_report_sha256": canonical_json_write(
            output_root / "FIRST_SOURCE_INTERFACE_REPORT.json", interface["first"]
        ),
        "second_source_interface_report_sha256": canonical_json_write(
            output_root / "SECOND_SOURCE_INTERFACE_REPORT.json", interface["second"]
        ),
        "pair_actionability_sha256": canonical_json_write(
            output_root / "PAIR_ACTIONABILITY.json", interface["pair"]
        ),
        "retained_identity_hashes_sha256": canonical_json_write(
            output_root / "RETAINED_IDENTITY_HASHES.json",
            computation["retained_identity_hashes"],
        ),
        "engine_invocation_accounting_sha256": canonical_json_write(
            output_root / "ENGINE_INVOCATION_ACCOUNTING.json",
            computation["engine_accounting"],
        ),
    }
    outer = _base_outer_envelope(computation)
    outer["nested_payload_sha256"] = dict(artifact_sha256)
    if computation["scientific_execute_source_reached"]:
        semantic = computation.pop("semantic_payload")
        visual = computation.pop("visual_payload")
        metrics = computation.pop("engine_metrics")
        semantic_sha = canonical_json_write(
            output_root / "SOURCE_SEMANTIC_PAYLOAD.json", semantic
        )
        semantic_schema = semantic.get("schema")
        del semantic
        visual_sha = canonical_json_write(
            output_root / "SOURCE_VISUAL_PAYLOAD.json", visual
        )
        visual_schema = visual.get("schema")
        del visual
        canonical_json_write(
            measurement_root / "EXP32_ENGINE_METRICS.json", metrics
        )
        del metrics
        counter_sha = canonical_json_write(
            output_root / "EXP32_ENGINE_COUNTER_PROJECTION.json",
            computation["engine_accounting"][
                "deterministic_exp32_operation_counter_projection"
            ],
        )
        outer["nested_payload_sha256"].update(
            {
                "scientific_semantic_payload_sha256": semantic_sha,
                "scientific_visual_payload_sha256": visual_sha,
                "exp32_engine_counter_projection_sha256": counter_sha,
            }
        )
        outer["scientific_payload_identity"] = {
            "semantic_schema": semantic_schema,
            "semantic_experiment_id": "EXP_CAD_0030",
            "semantic_architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
            "visual_schema": visual_schema,
            "exp32_engine_metrics_persisted_outside_compared_tree": True,
        }
    else:
        if (
            computation.get("semantic_payload") is not None
            or computation.get("visual_payload") is not None
            or computation.get("engine_metrics") is not None
        ):
            raise RuntimeError("interface abstention carries forbidden science payload")
        outer["scientific_payload_identity"] = None
    canonical_json_write(output_root / "SOURCE_RESULT.json", outer)
    return outer


def _load_and_validate_cad(
    *,
    parent_prereg: Mapping[str, Any],
    cad_preflight_path: Path,
    nodes_path: Path,
    edges_path: Path,
    edge_weights_path: Path,
    mesh_path: Path,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, Any]:
    cad_preflight = _load_json(cad_preflight_path)
    V5.FROZEN_RUNNER.validate_cad_preflight_identity(
        cad_preflight, EXPECTED_PARENT_PREREGISTRATION_SHA256
    )
    nodes, edges, weights, mesh = V5.FROZEN_RUNNER.load_cad_source_artifacts_before_scan(
        nodes_path=nodes_path,
        edges_path=edges_path,
        edge_weights_path=edge_weights_path,
        mesh_path=mesh_path,
    )
    if len(nodes) != EXPECTED_NODE_COUNT or len(edges) != EXPECTED_EDGE_COUNT:
        raise ValueError("EXP34 CAD graph cardinality drift")
    return cad_preflight, nodes, edges, weights, mesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("PRIMARY", "REPLAY"))
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--detached-approval", required=True, type=Path)
    parser.add_argument("--detached-approval-sha256", required=True)
    parser.add_argument("--launch-claim", required=True, type=Path)
    parser.add_argument("--launch-claim-sha256", required=True)
    parser.add_argument("--source-launch-capability", required=True, type=Path)
    parser.add_argument("--source-launch-child-claim", required=True, type=Path)
    parser.add_argument("--parent-preregistration", required=True, type=Path)
    parser.add_argument("--exp31-preregistration", required=True, type=Path)
    parser.add_argument("--v5-preregistration", required=True, type=Path)
    parser.add_argument("--cad-preflight", required=True, type=Path)
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--edge-geodesic-mm", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--scan-first", required=True, type=Path)
    parser.add_argument("--scan-second", required=True, type=Path)
    parser.add_argument("--transform-first", required=True, type=Path)
    parser.add_argument("--transform-second", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--measurement-root", required=True, type=Path)
    parser.add_argument("--scan-accounting-journal", required=True, type=Path)
    return parser


def _consumed_paths(
    args: argparse.Namespace, lock: Mapping[str, Any]
) -> dict[str, Path]:
    bound = lock["bound_files"]

    def locked(key: str) -> Path:
        return ROOT / bound[key]["path"]

    return {
        "exp34_source_runner": Path(__file__).resolve(),
        "parent_preregistration": args.parent_preregistration,
        "exp31_preregistration": args.exp31_preregistration,
        "exp32_preregistration": args.v5_preregistration,
        "cad_preflight": args.cad_preflight,
        "cad_nodes": args.nodes,
        "cad_edges": args.edges,
        "cad_edge_geodesic_mm": args.edge_geodesic_mm,
        "accepted_mesh": args.mesh,
        "scan_N009": args.scan_first,
        "scan_N010": args.scan_second,
        "transform_N009": args.transform_first,
        "transform_N010": args.transform_second,
        "exp31_filter_module": locked("exp31_filter_module"),
        "exp32_v5_engine": locked("exp32_v5_engine"),
        "frozen_parent_core": locked("frozen_parent_core"),
        "frozen_parent_science": locked("frozen_parent_science"),
        "frozen_parent_runner": locked("frozen_parent_runner"),
    }


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "source_root": args.output_root,
        "measurement_root": args.measurement_root,
        "scan_accounting_journal": args.scan_accounting_journal,
        "launch_claim": args.launch_claim,
        "source_launch_capability": args.source_launch_capability,
        "source_child_claim": args.source_launch_child_claim,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    # Stdlib-only bootstrap: verify the immutable lock plus the exact shared
    # contract and runner bytes, then consume the orchestrator-issued child
    # capability.  No NumPy, Open3D, intake, engine, or frozen parent module has
    # been imported at this point.
    bootstrap_lock = stdlib_bootstrap_lock(args)
    validate_and_claim_source_capability(
        args=args, lock=bootstrap_lock, raw_argv=raw_argv
    )
    activate_governance(bootstrap_lock)
    output_paths = _output_paths(args)
    # The full verifier hashes every non-scan static input and attests the live
    # runtime.  Its two physical-scan rows remain path-only.
    lock = verify_execution_authority(
        preregistration_path=args.preregistration,
        lock_path=args.lock,
        lock_sha256=args.lock_sha256,
        detached_approval_path=args.detached_approval,
        detached_approval_sha256=args.detached_approval_sha256,
        consumed_paths=_consumed_paths(args, bootstrap_lock),
        output_paths=output_paths,
    )
    inferred_mode = GOVERNANCE.infer_source_mode(output_paths, lock)
    if inferred_mode != args.mode:
        raise ValueError("source mode differs from immutable output topology")
    activate_science_runtime(lock)
    if args.scan_accounting_journal != args.measurement_root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json":
        raise ValueError("scan journal must be the direct locked measurement child")
    _refuse_nonempty_source_root(args.output_root)

    exp34 = _load_json(args.preregistration)
    parent = _load_json(args.parent_preregistration)
    exp31 = _load_json(args.exp31_preregistration)
    exp32 = _load_json(args.v5_preregistration)
    _validate_preregistrations(exp34, parent, exp31, exp32)

    # Transform and every CAD artifact are validated before the first scan hash.
    transform_first = load_validated_transform(
        args.transform_first,
        token=TOKENS[0],
        scan_sha256=EXPECTED_SCAN_SHA256[TOKENS[0]],
        expected_row_count=EXPECTED_ROW_COUNTS[TOKENS[0]],
        expected_semantic_sha256=EXPECTED_TRANSFORM_SHA256[TOKENS[0]],
    )
    transform_second = load_validated_transform(
        args.transform_second,
        token=TOKENS[1],
        scan_sha256=EXPECTED_SCAN_SHA256[TOKENS[1]],
        expected_row_count=EXPECTED_ROW_COUNTS[TOKENS[1]],
        expected_semantic_sha256=EXPECTED_TRANSFORM_SHA256[TOKENS[1]],
    )
    cad_preflight, nodes, edges, weights, mesh = _load_and_validate_cad(
        parent_prereg=parent,
        cad_preflight_path=args.cad_preflight,
        nodes_path=args.nodes,
        edges_path=args.edges,
        edge_weights_path=args.edge_geodesic_mm,
        mesh_path=args.mesh,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.measurement_root.mkdir(parents=True, exist_ok=True)
    accounting = _new_scan_accounting()
    write_scan_journal(
        args.scan_accounting_journal,
        accounting,
        mode=args.mode,
        stage="READY_BEFORE_FIRST_SCAN_HASH",
        status="IN_PROGRESS",
    )
    try:
        first_cloud, second_cloud = perform_ordered_scan_access(
            mode=args.mode,
            scan_first=args.scan_first,
            scan_second=args.scan_second,
            journal_path=args.scan_accounting_journal,
            accounting=accounting,
        )
        raw_first, normals_first = materialize_loaded_cloud(
            first_cloud,
            token=TOKENS[0],
            expected_row_count=EXPECTED_ROW_COUNTS[TOKENS[0]],
        )
        raw_second, normals_second = materialize_loaded_cloud(
            second_cloud,
            token=TOKENS[1],
            expected_row_count=EXPECTED_ROW_COUNTS[TOKENS[1]],
        )
        del first_cloud, second_cloud
        computation = execute_source_exp34(
            exp34_prereg=exp34,
            parent_prereg=parent,
            exp31_prereg=exp31,
            v5_prereg=exp32,
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
            first_token=TOKENS[0],
            second_token=TOKENS[1],
        )
        del raw_first, normals_first, raw_second, normals_second
        result = publish_source_artifacts(
            output_root=args.output_root,
            measurement_root=args.measurement_root,
            computation=computation,
        )
        write_scan_journal(
            args.scan_accounting_journal,
            accounting,
            mode=args.mode,
            stage="COMPLETE",
            status="COMPLETE",
        )
    except BaseException as error:
        try:
            write_scan_journal(
                args.scan_accounting_journal,
                accounting,
                mode=args.mode,
                stage="FAIL_TERMINAL_NO_RETRY",
                status="FAIL_TERMINAL_NO_RETRY",
                error=error,
            )
        except BaseException:
            pass
        raise
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "scientific_execute_source_reached": result[
                    "scientific_execute_source_reached"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
