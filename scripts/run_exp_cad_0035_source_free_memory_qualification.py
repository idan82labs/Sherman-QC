#!/usr/bin/env python3
"""EXP_CAD_0035 source-free exact-scale memory qualification controller/worker.

``qualify`` is the single six-trial controller.  ``worker`` is its fresh child
surface for either the frozen EXP34 legacy batch lifetime or EXP35 sequential
lifetime.  The module accepts only generated source-free fixtures; it has no
physical-scan or evaluator path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys
import tempfile
import types
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or Path.cwd()).resolve(strict=False) != ROOT
]
sys.path.insert(0, str(ROOT))

_PREIMPORT_CANDIDATE_MODULE_NAMES = frozenset(
    {
        "numpy",
        "scipy",
        "open3d",
        "scripts.exp_cad_0029_cfpdg_core",
        "scripts.exp_cad_0029_cfpdg_source",
        "scripts.run_exp_cad_0029_cfpdg_source",
        "scripts.exp_cad_0031_source_interface",
        "scripts.exp_cad_0032_cfpdg_v5_engine",
        "scripts.exp_cad_0034_execution_contract",
        "scripts.exp_cad_0035_execution_contract",
        "scripts.exp_cad_0035_process_monitor",
        "scripts.run_exp_cad_0035_cfpdg_source",
    }
)
_PREEXISTING_CANDIDATE_MODULE_NAMES = frozenset(
    name for name in _PREIMPORT_CANDIDATE_MODULE_NAMES if name in sys.modules
)

import numpy as np
import open3d as o3d
import scipy

from scripts import exp_cad_0031_source_interface as INTAKE
from scripts import exp_cad_0032_cfpdg_v5_engine as V5
from scripts import exp_cad_0034_execution_contract as G34
from scripts import exp_cad_0035_execution_contract as G35
from scripts import exp_cad_0035_process_monitor as MONITOR
from scripts import run_exp_cad_0035_cfpdg_source as SEQUENTIAL


EXPERIMENT_ID = "EXP_CAD_0035"
PREREGISTRATION_SHA256 = (
    "462274a567cd0ed84b041b8e0576f93ceb1ce6b4df9abe75e63c15ec1f268cb2"
)
RESULT_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_RESULT_V1"
AGGREGATE_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_AGGREGATE_V1"
BOUNDARY_EVIDENCE_SCHEMA = "EXP_CAD_0035_BOUNDARY_EQUIVALENCE_EVIDENCE_V1"
MEASUREMENT_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_EXTERNAL_PROCESS_MEASUREMENT_V1"
FIXTURE_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST_V1"
LEGACY_SHA256 = "098e907278dc9398ddfff8fafa468d6b675ab221c8729d8449c8d2807d1324b1"
PREREGISTRATION_PATHS = {
    "exp30_preregistration": ROOT
    / "docs/research/2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json",
    "exp31_preregistration": ROOT
    / "docs/research/2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json",
    "exp32_preregistration": ROOT
    / "docs/research/2026-08-30_exp_cad_0032_cfpdg_v5_execution_equivalent_preregistration.json",
    "exp34_preregistration": ROOT
    / "docs/research/2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json",
    "exp35_preregistration": ROOT
    / "docs/research/2026-08-30_exp_cad_0035_sequential_member_intake_preregistration.json",
}
QUALIFICATION_PREREGISTRATION_SHA256 = {
    "exp30_preregistration": "c9129ccc103d45c89ccdc67eef938b242ccdbb081d3d0f261145cc5a5b9b8368",
    "exp31_preregistration": "20f4a370eac176dc4a91df19e68c2e2618dc7f383d0d45eb526267292b718eb6",
    "exp32_preregistration": "f0d9fbb363624f9e9d4500e08a2ebb6ac98589c21691fa618a168348bfa19390",
    "exp34_preregistration": "323c7cf72884f33e6358471f80b100c5d0e00ec0ca4b795cff5aa6e9afea5058",
    "exp35_preregistration": PREREGISTRATION_SHA256,
}
FIXTURE_ROOT = ROOT / "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures"
FIXTURE_MANIFEST = FIXTURE_ROOT / "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST.json"
FIXTURE_GENERATOR = ROOT / "scripts/generate_exp_cad_0035_source_free_fixtures.py"
FIXTURE_GENERATOR_RELATIVE_PATH = "scripts/generate_exp_cad_0035_source_free_fixtures.py"
FIXTURE_LAYOUT = (
    ("boundary_N009", "BOUNDARY_EQUIVALENCE", "N009", Path("boundary-equivalence/N009.ply")),
    ("boundary_N010", "BOUNDARY_EQUIVALENCE", "N010", Path("boundary-equivalence/N010.ply")),
    ("maximal_N009", "MAXIMAL_RETENTION", "N009", Path("maximal-retention/N009.ply")),
    ("maximal_N010", "MAXIMAL_RETENTION", "N010", Path("maximal-retention/N010.ply")),
)
QUALIFICATION_ROOT = ROOT / "output/research/autoresearch-loop/exp-cad-0035-source-free-qualification"
RESULT_NAME = "SOURCE_FREE_MEMORY_QUALIFICATION_RESULT.json"
MEASUREMENT_NAME = "EXTERNAL_PROCESS_MEASUREMENT.json"
AGGREGATE_NAME = "EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_AGGREGATE.json"
ROW_COUNTS = {"N009": 1_156_302, "N010": 1_323_486}
SCHEDULE = (
    (1, "LEGACY_BATCH", "trial-01-legacy-batch"),
    (2, "SEQUENTIAL", "trial-02-sequential"),
    (3, "LEGACY_BATCH", "trial-03-legacy-batch"),
    (4, "SEQUENTIAL", "trial-04-sequential"),
    (5, "LEGACY_BATCH", "trial-05-legacy-batch"),
    (6, "SEQUENTIAL", "trial-06-sequential"),
)
WALL_CAP_SECONDS = 180.0
SEQUENTIAL_RSS_CAP_BYTES = 4_294_967_296
LEGACY_RSS_CAP_BYTES = 8_589_934_592
MINIMUM_MEDIAN_DELTA_BYTES = 33_554_432
SAMPLE_INTERVAL_SECONDS = 0.05
EXPECTED_PYTHON_INVOCATION = ROOT / ".venv/bin/python"
EXP34_ENVIRONMENT_ATTESTATION = (
    ROOT
    / "docs/research/2026-08-30_exp_cad_0034_physical_environment_attestation.json"
)
EXP34_ENVIRONMENT_ATTESTATION_SHA256 = (
    "a333cbaf9a5352899cb64e8b96f6fb52ca355260474aaa858b6b96d65b346b28"
)
EXP34_ENVIRONMENT_FINGERPRINT_KEYS = frozenset(
    {
        "python_executable_invocation_path",
        "python_executable_realpath",
        "python_executable_sha256",
        "python_version",
        "numpy_version",
        "scipy_version",
        "open3d_version",
        "numpy_module_origin",
        "scipy_module_origin",
        "open3d_module_origin",
    }
)
RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "runtime_environment_fingerprint",
        "runtime_environment_fingerprint_sha256",
        "loaded_module_origin_path_sha256_map",
        "loaded_module_origin_path_sha256_map_sha256",
    }
)
MODULE_ORIGIN_ROW_KEYS = frozenset(
    {
        "module_name",
        "module_package",
        "module_spec_name",
        "origin_path",
        "realpath",
        "sha256",
        "distribution_version",
        "loader_type",
        "loader_name",
        "loader_path",
        "loader_identity_verified",
        "sys_modules_object_identity_verified",
        "function_globals_identity_verified",
    }
)
PROJECT_MODULE_PATHS = {
    "exp29_cad_core": ROOT / "scripts/exp_cad_0029_cfpdg_core.py",
    "exp29_frozen_source": ROOT / "scripts/exp_cad_0029_cfpdg_source.py",
    "exp29_frozen_runner": ROOT / "scripts/run_exp_cad_0029_cfpdg_source.py",
    "exp31_intake": ROOT / "scripts/exp_cad_0031_source_interface.py",
    "exp32_v5": ROOT / "scripts/exp_cad_0032_cfpdg_v5_engine.py",
    "exp34_contract": ROOT / "scripts/exp_cad_0034_execution_contract.py",
    "exp35_contract": ROOT / "scripts/exp_cad_0035_execution_contract.py",
    "exp35_monitor": ROOT / "scripts/exp_cad_0035_process_monitor.py",
    "exp35_sequential_runner": ROOT / "scripts/run_exp_cad_0035_cfpdg_source.py",
}
PROJECT_MODULE_NAMES = {
    "exp29_cad_core": "scripts.exp_cad_0029_cfpdg_core",
    "exp29_frozen_source": "scripts.exp_cad_0029_cfpdg_source",
    "exp29_frozen_runner": "scripts.run_exp_cad_0029_cfpdg_source",
    "exp31_intake": "scripts.exp_cad_0031_source_interface",
    "exp32_v5": "scripts.exp_cad_0032_cfpdg_v5_engine",
    "exp34_contract": "scripts.exp_cad_0034_execution_contract",
    "exp35_contract": "scripts.exp_cad_0035_execution_contract",
    "exp35_monitor": "scripts.exp_cad_0035_process_monitor",
    "exp35_sequential_runner": "scripts.run_exp_cad_0035_cfpdg_source",
}
MODULE_TO_CONSUMED_KEY = {
    "exp29_cad_core": "exp29_cad_core",
    "exp29_frozen_source": "exp29_frozen_source",
    "exp29_frozen_runner": "exp29_frozen_runner",
    "exp31_intake": "exp31_filter_module",
    "exp32_v5": "exp32_v5_engine",
    "exp34_contract": "exp34_execution_contract",
    "exp35_contract": "exp35_execution_contract",
    "exp35_monitor": "exp35_process_monitor",
    "exp35_sequential_runner": "exp35_source_runner",
}
IMMUTABLE_INHERITED_MODULE_SHA256 = {
    "exp29_cad_core": "592787c80beeeafb79d04e645e34681daf470f580267e6df26c4801a2f385d3d",
    "exp29_frozen_source": "a620567c3d2ff5aab48f30a7fac8b2bbf95e75d74660cf22b98ba5063a6dcb68",
    "exp29_frozen_runner": "149b1a256ca7a3c78d61727c463741b52194e1b099668d7afbacb97999c48adf",
    "exp31_intake": "029570cbb94007b33da39eb0b4d201a653200e8f198be1a88f1a37fcc5c830ac",
    "exp32_v5": "ecbcdd434a94dff5606cbbac960f24e2472f64deb2934faf8f638a1bb2cd2e85",
    "exp34_contract": "4d64ac063c665393125e0437d38bce299fb1e853991c6cadcdca6eb28ff6d01a",
}

# Capture the exact objects imported by this fresh process before any
# qualification entry point can run.  Re-reading a live module attribute in a
# verifier is insufficient: an in-place replacement can otherwise bring its
# own canonical-looking globals and co_filename.  These references are the
# post-import identity anchor rechecked at every runtime snapshot.
_IMPORTED_CRITICAL_FUNCTIONS: dict[
    str, tuple[tuple[str, Callable[..., Any]], ...]
] = {
    "exp29_cad_core": (("sha256_path", V5.CAD.sha256_path),),
    "exp29_frozen_source": (
        ("canonical_unoriented_normals", V5.FROZEN_SRC.canonical_unoriented_normals),
    ),
    "exp29_frozen_runner": (("load_transform", V5.FROZEN_RUNNER.load_transform),),
    "exp31_intake": (
        ("filter_source_rows", INTAKE.filter_source_rows),
        ("pair_actionability", INTAKE.pair_actionability),
    ),
    "exp32_v5": (("execute_parent_science_v5", V5.execute_parent_science_v5),),
    "exp34_contract": (
        ("sha256_path", G34.sha256_path),
        ("runtime_environment_fingerprint", G34.runtime_environment_fingerprint),
    ),
    "exp35_contract": (
        ("sha256_path", G35.sha256_path),
        ("runtime_environment_fingerprint", G35.runtime_environment_fingerprint),
    ),
    "exp35_monitor": (("run_measured_command", MONITOR.run_measured_command),),
    "exp35_sequential_runner": (
        ("perform_sequential_member_intake", SEQUENTIAL.perform_sequential_member_intake),
        ("execute_filtered_source_exp35", SEQUENTIAL.execute_filtered_source_exp35),
    ),
}

_IMPORTED_RUNTIME_MODULES = {
    "hashlib": hashlib,
    "importlib": importlib,
    "importlib.metadata": G35.importlib.metadata,
    "importlib.util": importlib.util,
    "platform": G35.platform,
    "struct": G35.struct,
    "sys": sys,
    "sysconfig": G35.sysconfig,
}
_IMPORTED_RUNTIME_CALLABLES = {
    "hashlib.sha256": (hashlib, "sha256", hashlib.sha256),
    "importlib.metadata.version": (
        G35.importlib.metadata,
        "version",
        G35.importlib.metadata.version,
    ),
    "importlib.util.find_spec": (importlib.util, "find_spec", importlib.util.find_spec),
    "platform.python_version": (G35.platform, "python_version", G35.platform.python_version),
    "platform.python_implementation": (
        G35.platform,
        "python_implementation",
        G35.platform.python_implementation,
    ),
    "platform.python_compiler": (
        G35.platform,
        "python_compiler",
        G35.platform.python_compiler,
    ),
    "platform.system": (G35.platform, "system", G35.platform.system),
    "platform.release": (G35.platform, "release", G35.platform.release),
    "platform.machine": (G35.platform, "machine", G35.platform.machine),
    "platform.platform": (G35.platform, "platform", G35.platform.platform),
    "struct.calcsize": (G35.struct, "calcsize", G35.struct.calcsize),
    "sysconfig.get_config_var": (
        G35.sysconfig,
        "get_config_var",
        G35.sysconfig.get_config_var,
    ),
    "Path.open": (Path, "open", Path.open),
    "Path.resolve": (Path, "resolve", Path.resolve),
}
def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_write(path: Path, value: Any) -> str:
    payload = _strict_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_canonical_json(path: Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    if raw != _strict_bytes(value):
        raise ValueError(f"JSON is not strict canonical bytes: {path}")
    return value


def _reject_symlink_components(path: Path, *, label: str) -> None:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    probe = Path(path.anchor)
    for part in path.parts[1:]:
        probe = probe / part
        if probe.is_symlink():
            raise ValueError(f"{label} path contains a symlink component")


def _require_regular_unsymlinked_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    _reject_symlink_components(path, label=label)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} file is absent") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} is not a regular file")
    return path


def _load_frozen_exp34_environment() -> Mapping[str, str]:
    path = _require_regular_unsymlinked_file(
        EXP34_ENVIRONMENT_ATTESTATION,
        label="frozen EXP34 environment attestation",
    )
    if _sha256_path(path) != EXP34_ENVIRONMENT_ATTESTATION_SHA256:
        raise RuntimeError("frozen EXP34 environment attestation SHA-256 drift")
    row = _load_canonical_json(path)
    if set(row) != {
        "schema",
        "experiment_id",
        "status",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "physical_scan_or_evaluator_input_used",
        "generalization_evidence",
        "environment",
    }:
        raise RuntimeError("frozen EXP34 environment attestation field drift")
    environment = row.get("environment")
    if (
        row.get("schema")
        != "EXP_CAD_0034_PHYSICAL_ENVIRONMENT_ATTESTATION_V1"
        or row.get("experiment_id") != "EXP_CAD_0034"
        or row.get("status") != "FROZEN"
        or row.get("physical_source_execution_authorized") is not False
        or row.get("evaluator_input_authorized") is not False
        or row.get("physical_scan_or_evaluator_input_used") is not False
        or row.get("generalization_evidence") is not False
        or not isinstance(environment, Mapping)
        or set(environment) != set(EXP34_ENVIRONMENT_FINGERPRINT_KEYS)
        or any(not isinstance(value, str) for value in environment.values())
    ):
        raise RuntimeError("frozen EXP34 environment attestation identity drift")
    return dict(environment)


def _module_objects() -> dict[str, Any]:
    return {
        "numpy": np,
        "scipy": scipy,
        "open3d": o3d,
        "exp29_cad_core": V5.CAD,
        "exp29_frozen_source": V5.FROZEN_SRC,
        "exp29_frozen_runner": V5.FROZEN_RUNNER,
        "exp31_intake": INTAKE,
        "exp32_v5": V5,
        "exp34_contract": G34,
        "exp35_contract": G35,
        "exp35_monitor": MONITOR,
        "exp35_sequential_runner": SEQUENTIAL,
    }


def _validate_scripts_namespace() -> None:
    namespace = sys.modules.get("scripts")
    spec = getattr(namespace, "__spec__", None)
    expected = (str(ROOT / "scripts"),)
    locations = getattr(spec, "submodule_search_locations", None)
    if (
        namespace is None
        or getattr(namespace, "__name__", None) != "scripts"
        or getattr(namespace, "__package__", None) != "scripts"
        or spec is None
        or getattr(spec, "name", None) != "scripts"
        or getattr(spec, "origin", None) is not None
        or locations is None
        or tuple(str(Path(value).resolve(strict=True)) for value in locations)
        != expected
        or tuple(str(Path(value).resolve(strict=True)) for value in namespace.__path__)
        != expected
    ):
        raise RuntimeError("EXP35 scripts namespace package path/identity drift")
    found = importlib.util.find_spec("scripts")
    found_locations = getattr(found, "submodule_search_locations", None)
    if (
        found is None
        or found.origin is not None
        or found_locations is None
        or tuple(str(Path(value).resolve(strict=True)) for value in found_locations)
        != expected
    ):
        raise RuntimeError("EXP35 scripts namespace find_spec drift")


def _validate_runtime_dependencies() -> None:
    if (
        V5.FROZEN_INTAKE is not INTAKE
        or V5.CAD is not sys.modules.get("scripts.exp_cad_0029_cfpdg_core")
        or V5.FROZEN_SRC
        is not sys.modules.get("scripts.exp_cad_0029_cfpdg_source")
        or V5.FROZEN_RUNNER
        is not sys.modules.get("scripts.run_exp_cad_0029_cfpdg_source")
        or V5.FROZEN_RUNNER.CAD is not V5.CAD
        or V5.FROZEN_RUNNER.SRC is not V5.FROZEN_SRC
        or V5.FROZEN_SRC.CAD is not V5.CAD
        or INTAKE.np is not np
        or V5.np is not np
        or V5.o3d is not o3d
        or V5.CAD.np is not np
        or V5.CAD.o3d is not o3d
        or V5.FROZEN_SRC.np is not np
        or V5.FROZEN_RUNNER.np is not np
        or V5.FROZEN_RUNNER.o3d is not o3d
    ):
        raise RuntimeError("EXP35 scientific/project dependency object identity drift")
    if any(
        SEQUENTIAL.__dict__.get(name) is not None
        for name in ("np", "o3d", "INTAKE", "V5", "GOVERNANCE")
    ):
        raise RuntimeError("EXP35 sequential runner retained stale injected globals")


def _function_inventory() -> dict[
    str, tuple[tuple[str, Callable[..., Any]], ...]
]:
    return dict(_IMPORTED_CRITICAL_FUNCTIONS)


def _prevalidate_runtime_fingerprint_dependencies() -> None:
    """Authenticate the modules/callables used to build the fingerprint."""

    expected_attributes = {
        "hashlib": (G34, G35),
        "importlib": (G34, G35),
        "platform": (G34, G35),
        "sys": (G34, G35),
        "sysconfig": (G35,),
        "struct": (G35,),
    }
    for name, module in _IMPORTED_RUNTIME_MODULES.items():
        if not isinstance(module, types.ModuleType) or sys.modules.get(name) is not module:
            raise RuntimeError(f"EXP35 runtime dependency module drift: {name}")
    for attribute, owners in expected_attributes.items():
        expected = _IMPORTED_RUNTIME_MODULES[attribute]
        if any(owner.__dict__.get(attribute) is not expected for owner in owners):
            raise RuntimeError(f"EXP35 contract runtime dependency drift: {attribute}")
    if G35.Path is not Path or G34.Path is not Path:
        raise RuntimeError("EXP35 contract Path dependency drift")
    for label, (owner, attribute, expected) in _IMPORTED_RUNTIME_CALLABLES.items():
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(f"EXP35 runtime dependency callable drift: {label}")


def _validate_safe_sys_path() -> None:
    if not sys.path or sys.path[0] != str(ROOT):
        raise RuntimeError("EXP35 sys.path does not begin with the exact repository root")
    trusted_prefixes = (
        Path(sys.base_prefix).resolve(strict=True),
        Path(sys.prefix).resolve(strict=True),
    )
    trusted_repo_entries = {ROOT, ROOT / "scripts"}
    observed: set[Path] = set()
    for raw in sys.path:
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise RuntimeError("EXP35 sys.path contains an empty/relative entry")
        resolved = Path(raw).resolve(strict=False)
        if resolved in observed:
            raise RuntimeError("EXP35 sys.path contains a duplicate resolved entry")
        observed.add(resolved)
        if resolved in trusted_repo_entries:
            continue
        if any(resolved == prefix or resolved.is_relative_to(prefix) for prefix in trusted_prefixes):
            continue
        raise RuntimeError(f"EXP35 sys.path contains an untrusted entry: {raw}")


def _prevalidate_environment_contract_modules() -> None:
    """Authenticate G34/G35 before executing either fingerprint function."""

    contracts = (("exp34_contract", G34), ("exp35_contract", G35))
    # First reject a swapped contract object using metadata only.  Only after
    # both objects pass do we call any hashing/runtime dependency.
    for key, module in contracts:
        module_name = PROJECT_MODULE_NAMES[key]
        path = PROJECT_MODULE_PATHS[key]
        spec = getattr(module, "__spec__", None)
        loader = getattr(module, "__loader__", None)
        if (
            not isinstance(module, types.ModuleType)
            or sys.modules.get(module_name) is not module
            or getattr(module, "__name__", None) != module_name
            or getattr(module, "__package__", None) != "scripts"
            or getattr(module, "__file__", None) != str(path)
            or spec is None
            or getattr(spec, "name", None) != module_name
            or getattr(spec, "origin", None) != str(path)
            or getattr(spec, "loader", None) is not loader
            or type(loader) is not importlib.machinery.SourceFileLoader
            or getattr(loader, "name", None) != module_name
            or getattr(loader, "path", None) != str(path)
            or loader.get_filename(module_name) != str(path)
        ):
            raise RuntimeError(f"EXP35 pre-fingerprint contract module drift: {key}")
    _prevalidate_runtime_fingerprint_dependencies()
    for key, module in contracts:
        path = PROJECT_MODULE_PATHS[key]
        _require_regular_unsymlinked_file(
            path, label=f"EXP35 pre-fingerprint contract {key}"
        )
        observed_sha256 = _sha256_path(path)
        if (
            key in IMMUTABLE_INHERITED_MODULE_SHA256
            and observed_sha256 != IMMUTABLE_INHERITED_MODULE_SHA256[key]
        ):
            raise RuntimeError(f"EXP35 immutable inherited module SHA drift: {key}")
        for function_name, function in _IMPORTED_CRITICAL_FUNCTIONS[key]:
            if (
                getattr(module, function_name, None) is not function
                or not isinstance(function, types.FunctionType)
                or function.__globals__ is not module.__dict__
                or function.__code__.co_filename != str(path)
            ):
                raise RuntimeError(
                    f"EXP35 pre-fingerprint contract function drift: {key}/{function_name}"
                )
        if (
            module.__dict__.get("sys") is not sys
            or module.__dict__.get("Path") is not Path
            or module.__dict__.get("hashlib") is not hashlib
            or module.__dict__.get("importlib") is not importlib
        ):
            raise RuntimeError(
                f"EXP35 pre-fingerprint contract dependency drift: {key}"
            )


def _runtime_identity_snapshot(*, strict_sys_path: bool) -> dict[str, Any]:
    """Fail-closed live interpreter/module proof without touching a fixture."""

    if strict_sys_path and _PREEXISTING_CANDIDATE_MODULE_NAMES:
        raise RuntimeError(
            "EXP35 exact qualification had preloaded candidate modules: "
            + ",".join(sorted(_PREEXISTING_CANDIDATE_MODULE_NAMES))
        )
    for variable in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        if variable in os.environ:
            raise RuntimeError(f"EXP35 unsafe Python environment variable is present: {variable}")
    if sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("EXP35 qualification requires Python -B")
    if strict_sys_path:
        if (
            sys.flags.isolated != 1
            or sys.flags.safe_path is not True
            or sys.flags.ignore_environment != 1
            or sys.flags.no_user_site != 1
        ):
            raise RuntimeError("EXP35 exact qualification requires isolated Python -I")
        _validate_safe_sys_path()

    expected_worker = ROOT / "scripts/run_exp_cad_0035_source_free_memory_qualification.py"
    worker_file = Path(__file__)
    if str(worker_file) != str(expected_worker):
        raise RuntimeError("EXP35 memory-worker invocation/import path drift")
    _require_regular_unsymlinked_file(worker_file, label="EXP35 memory worker")

    invocation = Path(sys.executable)
    if str(invocation) != str(EXPECTED_PYTHON_INVOCATION):
        raise RuntimeError("EXP35 Python invocation path drift")
    _reject_symlink_components(invocation.parent, label="EXP35 Python invocation parent")
    try:
        invocation_mode = invocation.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("EXP35 Python invocation path is absent") from error
    if not (stat.S_ISREG(invocation_mode) or stat.S_ISLNK(invocation_mode)):
        raise RuntimeError("EXP35 Python invocation path is not a file/symlink")
    realpath = invocation.resolve(strict=True)
    _require_regular_unsymlinked_file(realpath, label="EXP35 Python realpath")

    _prevalidate_environment_contract_modules()
    expected_parent_environment = _load_frozen_exp34_environment()
    environment = G35.runtime_environment_fingerprint()
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(G35.ENVIRONMENT_FINGERPRINT_KEYS)
        or any(not isinstance(value, str) for value in environment.values())
        or environment.get("python_executable_invocation_path") != str(invocation)
        or environment.get("python_executable_realpath") != str(realpath)
        or environment.get("python_executable_sha256") != _sha256_path(realpath)
        or any(
            environment.get(key) != value
            for key, value in expected_parent_environment.items()
        )
        or G34.runtime_environment_fingerprint() != expected_parent_environment
    ):
        raise RuntimeError("EXP35 live runtime/frozen EXP34 environment drift")

    _validate_scripts_namespace()
    _validate_runtime_dependencies()
    modules = _module_objects()
    expected_keys = {"numpy", "scipy", "open3d", *PROJECT_MODULE_PATHS}
    if set(modules) != expected_keys:
        raise RuntimeError("EXP35 loaded module inventory drift")
    function_inventory = _function_inventory()
    rows: dict[str, dict[str, Any]] = {}
    for key, module in sorted(modules.items()):
        module_name = (
            key if key in {"numpy", "scipy", "open3d"} else PROJECT_MODULE_NAMES[key]
        )
        if sys.modules.get(module_name) is not module:
            raise RuntimeError(f"EXP35 sys.modules object identity drift: {key}")
        spec = getattr(module, "__spec__", None)
        loader = getattr(module, "__loader__", None)
        origin = getattr(spec, "origin", None)
        file_value = getattr(module, "__file__", None)
        expected_path = (
            Path(environment[f"{key}_module_origin"])
            if key in {"numpy", "scipy", "open3d"}
            else PROJECT_MODULE_PATHS[key]
        )
        if (
            getattr(module, "__name__", None) != module_name
            or getattr(module, "__package__", None)
            != (module_name if key in {"numpy", "scipy", "open3d"} else "scripts")
            or spec is None
            or getattr(spec, "name", None) != module_name
            or getattr(spec, "loader", None) is not loader
            or type(loader) is not importlib.machinery.SourceFileLoader
            or getattr(loader, "name", None) != module_name
            or getattr(loader, "path", None) != str(expected_path)
            or loader.get_filename(module_name) != str(expected_path)
            or not isinstance(origin, str)
            or not isinstance(file_value, str)
            or origin != str(expected_path)
            or file_value != str(expected_path)
        ):
            raise RuntimeError(f"EXP35 loaded module metadata/origin drift: {key}")
        _require_regular_unsymlinked_file(expected_path, label=f"EXP35 module {key}")
        found = importlib.util.find_spec(module_name)
        if found is None or found.origin != str(expected_path):
            raise RuntimeError(f"EXP35 loaded module find_spec origin drift: {key}")
        functions = function_inventory.get(key, ())
        for function_name, function in functions:
            if (
                getattr(module, function_name, None) is not function
                or not isinstance(function, types.FunctionType)
                or function.__globals__ is not module.__dict__
                or function.__code__.co_filename != str(expected_path)
            ):
                raise RuntimeError(f"EXP35 module critical function identity drift: {key}")
        version = environment.get(f"{key}_version") if key in {"numpy", "scipy", "open3d"} else None
        if key in {"numpy", "scipy", "open3d"} and getattr(module, "__version__", None) != version:
            raise RuntimeError(f"EXP35 scientific module version drift: {key}")
        observed_sha256 = _sha256_path(expected_path)
        if (
            key in IMMUTABLE_INHERITED_MODULE_SHA256
            and observed_sha256 != IMMUTABLE_INHERITED_MODULE_SHA256[key]
        ):
            raise RuntimeError(f"EXP35 immutable inherited module SHA drift: {key}")
        rows[key] = {
            "module_name": module_name,
            "module_package": getattr(module, "__package__"),
            "module_spec_name": getattr(spec, "name"),
            "origin_path": str(expected_path),
            "realpath": str(expected_path.resolve(strict=True)),
            "sha256": observed_sha256,
            "distribution_version": version,
            "loader_type": "SourceFileLoader",
            "loader_name": loader.name,
            "loader_path": loader.path,
            "loader_identity_verified": True,
            "sys_modules_object_identity_verified": True,
            "function_globals_identity_verified": True,
        }

    environment = dict(environment)
    identity = {
        "runtime_environment_fingerprint": environment,
        "runtime_environment_fingerprint_sha256": hashlib.sha256(
            _strict_bytes(environment)
        ).hexdigest(),
        "loaded_module_origin_path_sha256_map": rows,
        "loaded_module_origin_path_sha256_map_sha256": hashlib.sha256(
            _strict_bytes(rows)
        ).hexdigest(),
    }
    _validate_runtime_identity(identity)
    return identity


def _validate_runtime_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping) or set(identity) != set(RUNTIME_IDENTITY_KEYS):
        raise ValueError("EXP35 runtime identity field inventory drift")
    environment = identity.get("runtime_environment_fingerprint")
    modules = identity.get("loaded_module_origin_path_sha256_map")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(G35.ENVIRONMENT_FINGERPRINT_KEYS)
        or any(not isinstance(value, str) for value in environment.values())
        or environment.get("python_executable_invocation_path")
        != str(EXPECTED_PYTHON_INVOCATION)
        or not Path(environment.get("python_executable_realpath", "")).is_absolute()
        or not _is_sha256(environment.get("python_executable_sha256"))
        or identity.get("runtime_environment_fingerprint_sha256")
        != hashlib.sha256(_strict_bytes(environment)).hexdigest()
        or not isinstance(modules, Mapping)
        or set(modules) != {"numpy", "scipy", "open3d", *PROJECT_MODULE_PATHS}
        or identity.get("loaded_module_origin_path_sha256_map_sha256")
        != hashlib.sha256(_strict_bytes(modules)).hexdigest()
    ):
        raise ValueError("EXP35 runtime identity digest/inventory drift")
    for key, row in modules.items():
        module_name = (
            key if key in {"numpy", "scipy", "open3d"} else PROJECT_MODULE_NAMES[key]
        )
        origin = (
            environment[f"{key}_module_origin"]
            if key in {"numpy", "scipy", "open3d"}
            else str(PROJECT_MODULE_PATHS[key])
        )
        version = (
            environment[f"{key}_version"]
            if key in {"numpy", "scipy", "open3d"}
            else None
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != set(MODULE_ORIGIN_ROW_KEYS)
            or row.get("module_name") != module_name
            or row.get("module_package")
            != (module_name if key in {"numpy", "scipy", "open3d"} else "scripts")
            or row.get("module_spec_name") != module_name
            or row.get("origin_path") != origin
            or row.get("realpath") != str(Path(origin).resolve(strict=True))
            or row.get("distribution_version") != version
            or row.get("loader_type") != "SourceFileLoader"
            or row.get("loader_name") != module_name
            or row.get("loader_path") != origin
            or row.get("loader_identity_verified") is not True
            or row.get("sys_modules_object_identity_verified") is not True
            or row.get("function_globals_identity_verified") is not True
            or not _is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXP35 runtime module-origin row drift: {key}")


def _validate_runtime_against_consumed(
    identity: Mapping[str, Any], consumed: Mapping[str, str]
) -> None:
    _validate_runtime_identity(identity)
    modules = identity["loaded_module_origin_path_sha256_map"]
    for module_key, consumed_key in MODULE_TO_CONSUMED_KEY.items():
        if modules[module_key]["sha256"] != consumed.get(consumed_key):
            raise RuntimeError(f"EXP35 runtime/consumed module SHA binding drift: {module_key}")
    if consumed.get("exp34_environment_attestation") != EXP34_ENVIRONMENT_ATTESTATION_SHA256:
        raise RuntimeError("EXP35 runtime/consumed parent environment binding drift")


def _assert_runtime_identity_unchanged(
    frozen: Mapping[str, Any], *, context: str, strict_sys_path: bool
) -> None:
    if _runtime_identity_snapshot(strict_sys_path=strict_sys_path) != dict(frozen):
        raise RuntimeError(f"EXP35 runtime/module identity TOCTOU {context}")


def _require_exact_manifest_path(path: Path) -> Path:
    raw = Path(path)
    expected = Path(FIXTURE_MANIFEST)
    if not raw.is_absolute() or str(raw) != str(expected) or ".." in raw.parts:
        raise ValueError("EXP35 fixture manifest is not the exact canonical path")
    _reject_symlink_components(raw, label="EXP35 fixture manifest")
    return raw


def _manifest_path_string(path: Path) -> str:
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        # Used only by unit tests after monkeypatching FIXTURE_ROOT to a fresh
        # temporary root.  The production constants never take this branch.
        return Path(path).as_posix()


def _expected_fixture_paths() -> dict[str, Path]:
    return {
        key: Path(FIXTURE_ROOT) / relative
        for key, _fixture, _token, relative in FIXTURE_LAYOUT
    }


def consumed_sha256_map(fixture_manifest: Path) -> dict[str, str]:
    """Hash every code/fixture artifact consumed by a qualification worker."""

    manifest_path = _require_exact_manifest_path(fixture_manifest)
    paths = {
        "memory_worker": Path(__file__).resolve(strict=True),
        "exp29_cad_core": ROOT / "scripts/exp_cad_0029_cfpdg_core.py",
        "exp29_frozen_source": ROOT / "scripts/exp_cad_0029_cfpdg_source.py",
        "exp29_frozen_runner": ROOT / "scripts/run_exp_cad_0029_cfpdg_source.py",
        "exp35_source_runner": ROOT / "scripts/run_exp_cad_0035_cfpdg_source.py",
        "exp35_process_monitor": ROOT / "scripts/exp_cad_0035_process_monitor.py",
        "exp35_execution_contract": ROOT / "scripts/exp_cad_0035_execution_contract.py",
        "exp34_legacy_runner": ROOT / "scripts/run_exp_cad_0034_cfpdg_source.py",
        "exp34_execution_contract": ROOT / "scripts/exp_cad_0034_execution_contract.py",
        "exp34_environment_attestation": EXP34_ENVIRONMENT_ATTESTATION,
        "exp31_filter_module": ROOT / "scripts/exp_cad_0031_source_interface.py",
        "exp32_v5_engine": ROOT / "scripts/exp_cad_0032_cfpdg_v5_engine.py",
        "fixture_generator": FIXTURE_GENERATOR,
        "fixture_manifest": manifest_path,
    }
    paths.update(PREREGISTRATION_PATHS)
    paths.update(
        {
            f"fixture_{key}": path
            for key, path in _expected_fixture_paths().items()
        }
    )
    return {key: _sha256_path(path) for key, path in sorted(paths.items())}


def _fixture_inputs(
    manifest_path: Path,
    *,
    require_exact_counts: bool,
    fixture: str = "MAXIMAL_RETENTION",
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    manifest_path = _require_exact_manifest_path(manifest_path)
    manifest = _load_canonical_json(manifest_path)
    if set(manifest) != {
        "schema",
        "experiment_id",
        "status",
        "preregistration_sha256",
        "generator_path",
        "generator_sha256",
        "common_formula",
        "fixtures",
        "physical_scan_hash_call_count",
        "physical_scan_point_cloud_load_call_count",
        "physical_scan_byte_open_operation_count",
        "evaluator_role_label_gold_quality_or_expected_count_access_count",
    }:
        raise ValueError("EXP35 source-free fixture manifest field inventory drift")
    if (
        manifest.get("schema") != FIXTURE_SCHEMA
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("status") != "COMPLETE_SOURCE_FREE_FIXTURES_GENERATED_ONCE"
        or manifest.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or manifest.get("generator_path") != FIXTURE_GENERATOR_RELATIVE_PATH
        or manifest.get("physical_scan_hash_call_count") != 0
        or manifest.get("physical_scan_point_cloud_load_call_count") != 0
        or manifest.get("physical_scan_byte_open_operation_count") != 0
        or manifest.get("evaluator_role_label_gold_quality_or_expected_count_access_count") != 0
    ):
        raise ValueError("EXP35 source-free fixture manifest identity/access drift")
    if manifest.get("common_formula") != {
        "point_x": "32+(i%4096)*2^-10",
        "point_y": "-16+((i//4096)%4096)*2^-10",
        "point_z": "4+(i//16777216)*2^-10",
        "normal": [0.0, 0.0, 1.0],
        "binary_ply_scalar_dtype": "LITTLE_ENDIAN_FLOAT64",
    }:
        raise ValueError("EXP35 source-free fixture formula identity drift")
    rows = manifest.get("fixtures")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("EXP35 fixture manifest must bind exactly four PLYs")
    if fixture not in {row[1] for row in FIXTURE_LAYOUT}:
        raise ValueError("unknown EXP35 source-free fixture identity")

    # First validate the complete lexical manifest inventory.  In particular,
    # no fixture path is statted, hashed, or opened until all four row paths
    # equal the fixed canonical source-free paths.  This prevents a crafted
    # manifest from redirecting a unit/qualification worker to physical input.
    expected_paths = _expected_fixture_paths()
    expected_keys = [row[0] for row in FIXTURE_LAYOUT]
    if [row.get("key") if isinstance(row, Mapping) else None for row in rows] != expected_keys:
        raise ValueError("EXP35 fixture manifest row order/inventory drift")
    validated_rows: list[tuple[Mapping[str, Any], Path]] = []
    for row, (key, fixture_name, token, _relative) in zip(rows, FIXTURE_LAYOUT):
        if not isinstance(row, Mapping):
            raise ValueError("EXP35 fixture row is malformed")
        if set(row) != {
            "key",
            "fixture",
            "opaque_token",
            "path",
            "row_count",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"EXP35 fixture row field inventory drift: {key}")
        path = expected_paths[key]
        if (
            row.get("key") != key
            or row.get("fixture") != fixture_name
            or row.get("opaque_token") != token
            or row.get("path") != _manifest_path_string(path)
            or isinstance(row.get("row_count"), bool)
            or not isinstance(row.get("row_count"), int)
            or int(row["row_count"]) <= 0
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or int(row["size_bytes"]) <= 0
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(row["sha256"]))
        ):
            raise ValueError(f"EXP35 fixture row identity/path/metadata drift: {key}")
        validated_rows.append((row, path))

    if require_exact_counts and any(
        int(row["row_count"]) != ROW_COUNTS[token]
        for row, (_key, _fixture_name, token, _relative) in zip(
            rows, FIXTURE_LAYOUT
        )
    ):
        raise ValueError("EXP35 exact-scale fixture row count drift")

    if manifest.get("generator_sha256") != _sha256_path(FIXTURE_GENERATOR):
        raise ValueError("EXP35 fixture generator SHA-256 identity drift")

    selected: dict[str, Mapping[str, Any]] = {}
    for (row, path), (key, fixture_name, token, _relative) in zip(
        validated_rows, FIXTURE_LAYOUT
    ):
        _reject_symlink_components(path, label=f"EXP35 fixture {key}")
        if path.is_symlink() or path.suffix != ".ply" or not path.is_file():
            raise ValueError(f"EXP35 fixture path is absent/symlinked/non-PLY: {key}")
        if int(row.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"EXP35 fixture size drift: {key}")
        if row.get("sha256") != _sha256_path(path):
            raise ValueError(f"EXP35 fixture SHA-256 drift: {key}")
        if fixture_name == fixture:
            selected[token] = {**row, "resolved_path": path}
    if set(selected) != {"N009", "N010"}:
        raise ValueError("EXP35 selected source-free fixture pair is missing")
    if require_exact_counts and any(
        int(selected[token]["row_count"]) != ROW_COUNTS[token] for token in ROW_COUNTS
    ):
        raise ValueError("EXP35 exact-scale fixture row count drift")
    return manifest, selected


def _validate_consumed_fixture_bindings(
    manifest: Mapping[str, Any], consumed: Mapping[str, str]
) -> None:
    if consumed.get("fixture_manifest") != hashlib.sha256(
        _strict_bytes(manifest)
    ).hexdigest():
        raise RuntimeError("EXP35 parsed fixture manifest changed before consumed freeze")
    if consumed.get("fixture_generator") != manifest.get("generator_sha256"):
        raise RuntimeError("EXP35 consumed fixture-generator binding drift")
    rows = manifest.get("fixtures")
    if not isinstance(rows, list):
        raise RuntimeError("EXP35 fixture rows disappeared before consumed freeze")
    for row in rows:
        key = str(row["key"])
        if consumed.get(f"fixture_{key}") != row.get("sha256"):
            raise RuntimeError(f"EXP35 consumed fixture binding drift: {key}")


def _load_legacy_runner() -> Any:
    path = ROOT / "scripts/run_exp_cad_0034_cfpdg_source.py"
    _require_regular_unsymlinked_file(path, label="frozen EXP34 legacy runner")
    if _sha256_path(path) != LEGACY_SHA256:
        raise ImportError("frozen EXP34 legacy batch oracle SHA-256 drift")
    spec = importlib.util.spec_from_file_location("_exp35_exact_exp34_legacy_oracle", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct exact EXP34 legacy oracle spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = np
    module.o3d = o3d
    module.INTAKE = INTAKE
    module.V5 = V5
    module.GOVERNANCE = G34
    for entry_point in (
        "perform_ordered_scan_access",
        "materialize_loaded_cloud",
        "execute_source_exp34",
    ):
        if not callable(getattr(module, entry_point, None)):
            raise ImportError(f"exact EXP34 legacy oracle lacks {entry_point}")
        if getattr(module, entry_point).__globals__ is not module.__dict__:
            raise ImportError(f"exact EXP34 legacy oracle globals drift: {entry_point}")
    if (
        module.__file__ != str(path)
        or module.__spec__ is None
        or module.__spec__.origin != str(path)
        or module.np is not np
        or module.o3d is not o3d
        or module.INTAKE is not INTAKE
        or module.V5 is not V5
        or module.GOVERNANCE is not G34
    ):
        raise ImportError("exact EXP34 legacy oracle module/dependency identity drift")
    return module


def _load_qualification_preregistrations() -> dict[str, Mapping[str, Any]]:
    """Load only the exact hash-bound preregistrations needed by both wrappers."""

    result: dict[str, Mapping[str, Any]] = {}
    for key, path in PREREGISTRATION_PATHS.items():
        observed = _sha256_path(path)
        if observed != QUALIFICATION_PREREGISTRATION_SHA256[key]:
            raise RuntimeError(f"EXP35 qualification preregistration SHA drift: {key}")
        result[key] = _load_json(path)
    return result


class _CapturingIntakeProxy:
    """Transparent exact-EXP31 proxy with fail-closed call/capture accounting."""

    def __init__(self, exact_module: Any) -> None:
        self._exact_module = exact_module
        self.filtered_by_token: dict[str, Mapping[str, Any]] = {}
        self.filter_call_count_by_token = {token: 0 for token in ("N009", "N010")}
        self.pair_actionability_call_count = 0
        self.pair: Mapping[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._exact_module, name)

    def filter_source_rows(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        token = kwargs.get("member_token")
        if token not in self.filter_call_count_by_token:
            raise RuntimeError("qualification EXP31 proxy received an unknown token")
        if self.filter_call_count_by_token[token] != 0:
            raise RuntimeError(f"qualification EXP31 {token} filter call is not unique")
        self.filter_call_count_by_token[token] += 1
        filtered = self._exact_module.filter_source_rows(*args, **kwargs)
        self.filtered_by_token[token] = filtered
        return filtered

    def pair_actionability(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        if self.pair_actionability_call_count != 0:
            raise RuntimeError("qualification EXP31 pair_actionability call is not unique")
        self.pair_actionability_call_count += 1
        pair = self._exact_module.pair_actionability(*args, **kwargs)
        self.pair = pair
        return pair


def _array_buffer_sha256(value: Any) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise RuntimeError("qualification stub received a non-contiguous array")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


class _QualificationParentV5Stub:
    """Qualification-only deterministic substitute for expensive parent V5 science."""

    ARCHITECTURE = V5.ARCHITECTURE
    FROZEN_RUNNER = V5.FROZEN_RUNNER

    def __init__(self) -> None:
        self.call_count = 0

    def execute_parent_science_v5(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.call_count != 0:
            raise RuntimeError("qualification parent-V5 stub would be called more than once")
        self.call_count += 1
        if (kwargs.get("first_token"), kwargs.get("second_token")) != ("N009", "N010"):
            raise RuntimeError("qualification parent-V5 stub token order drift")
        input_projection: dict[str, Any] = {}
        for token, points_key, normals_key in (
            ("N009", "raw_first", "normals_first"),
            ("N010", "raw_second", "normals_second"),
        ):
            points = np.asarray(kwargs[points_key])
            normals = np.asarray(kwargs[normals_key])
            if points.dtype != np.float64 or normals.dtype != np.float64 or points.shape != normals.shape:
                raise RuntimeError(f"qualification parent-V5 stub array contract drift: {token}")
            input_projection[token] = {
                "points_dtype": points.dtype.str,
                "points_shape": [int(value) for value in points.shape],
                "points_tobytes_sha256": _array_buffer_sha256(points),
                "normals_dtype": normals.dtype.str,
                "normals_shape": [int(value) for value in normals.shape],
                "normals_tobytes_sha256": _array_buffer_sha256(normals),
            }
        semantic = {
            "schema": "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1",
            "experiment_id": "EXP_CAD_0030",
            "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
            "status": "PASS_SOURCE_FREE_QUALIFICATION_PARENT_V5_STUB",
            "reason": None,
            "qualification_only_parent_v5_stub": True,
            "input_projection": input_projection,
        }
        visual = {
            "schema": "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1",
            "qualification_only_parent_v5_stub": True,
            "input_projection": input_projection,
        }
        counters = {
            "kd_node_query_count": 306_926,
            "local_fit_call_count": 38,
            "robustness_row_count": 32,
            "unique_member_fit_count": 38,
        }
        metrics = {
            "schema": "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1",
            "experiment_id": "EXP_CAD_0032",
            "architecture": self.ARCHITECTURE,
            "operation_counters": counters,
            **counters,
            "full_fit_inventory_gate_passed": True,
            "full_fit_batched_kd_gate_passed": True,
            "full_fit_kd_query_gate_passed": True,
            "fit_or_robustness_row_skipped": False,
            "scientific_threshold_or_gate_changed": False,
            "qualification_only_parent_v5_stub": True,
            "input_projection": input_projection,
        }
        return semantic, visual, metrics


def _qualification_wrapper_arguments(
    preregistrations: Mapping[str, Mapping[str, Any]], *, experiment: str
) -> dict[str, Any]:
    if experiment not in {"exp34", "exp35"}:
        raise ValueError("unknown qualification wrapper experiment")
    return {
        f"{experiment}_prereg": preregistrations[f"{experiment}_preregistration"],
        "parent_prereg": preregistrations["exp30_preregistration"],
        "exp31_prereg": preregistrations["exp31_preregistration"],
        "v5_prereg": preregistrations["exp32_preregistration"],
        "cad_preflight": {"qualification_only_parent_v5_stub": True},
        "nodes": np.empty((0, 3), dtype=np.float64),
        "edges": np.empty((0, 2), dtype=np.int64),
        "edge_weights": np.empty((0,), dtype=np.float64),
        "mesh": None,
        "transform_first": np.eye(4, dtype=np.float64),
        "transform_second": np.eye(4, dtype=np.float64),
        "first_token": "N009",
        "second_token": "N010",
    }


def _nested_payload_projection(computation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": computation["status"],
        "reason": computation["reason"],
        "scientific_execute_source_reached": computation[
            "scientific_execute_source_reached"
        ],
        "semantic_payload": computation["semantic_payload"],
        "visual_payload": computation["visual_payload"],
        "engine_metrics": computation["engine_metrics"],
        "engine_accounting": computation["engine_accounting"],
    }


def _scientific_projection(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "retained": {"N009": _projection(first), "N010": _projection(second)},
        "reports": {"N009": first["report"], "N010": second["report"]},
        "pair_actionability": pair,
    }


def _qualification_call_accounting(
    *,
    mode: str,
    proxy: _CapturingIntakeProxy,
    stub: _QualificationParentV5Stub,
    exact_exp34_wrapper_call_count: int,
    exact_exp35_filtered_wrapper_call_count: int,
) -> dict[str, Any]:
    if proxy.pair is None:
        raise RuntimeError(f"EXP35 {mode} qualification pair capture is absent")
    expected_stub_calls = int(bool(proxy.pair.get("actionable")))
    accounting = {
        "exact_exp34_execute_source_exp34_call_count": exact_exp34_wrapper_call_count,
        "exact_exp35_execute_filtered_source_exp35_call_count": (
            exact_exp35_filtered_wrapper_call_count
        ),
        "exp31_filter_source_rows_call_count_by_token": dict(
            proxy.filter_call_count_by_token
        ),
        "exp31_pair_actionability_call_count": proxy.pair_actionability_call_count,
        "qualification_parent_v5_stub_call_count": stub.call_count,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }
    expected = {
        "exact_exp34_execute_source_exp34_call_count": int(mode == "LEGACY_BATCH"),
        "exact_exp35_execute_filtered_source_exp35_call_count": int(mode == "SEQUENTIAL"),
        "exp31_filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "exp31_pair_actionability_call_count": 1,
        "qualification_parent_v5_stub_call_count": expected_stub_calls,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }
    if accounting != expected or set(proxy.filtered_by_token) != {"N009", "N010"}:
        raise RuntimeError(f"EXP35 {mode} qualification call accounting drift")
    return accounting


def _projection(filtered: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("points", "normals", "original_row_ids"):
        array = np.asarray(filtered[field])
        result[field] = {
            "dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
            "c_contiguous": bool(array.flags.c_contiguous),
            "owns_data": bool(array.flags.owndata),
            "tobytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def scientific_equivalence_projection(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result["scientific_equivalence_projection"]


def _assert_consumed_unchanged(
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
    *,
    context: str,
) -> None:
    if consumed_sha256_map(fixture_manifest) != dict(frozen_consumed_sha256):
        raise RuntimeError(f"EXP35 consumed code/fixture TOCTOU {context}")


def _guarded_fixture_hasher(
    *, fixture_manifest: Path, frozen_consumed_sha256: Mapping[str, str]
) -> Callable[[Path], str]:
    def guarded(path: Path) -> str:
        _assert_consumed_unchanged(
            fixture_manifest,
            frozen_consumed_sha256,
            context=f"immediately before SHA256 read {Path(path).name}",
        )
        try:
            observed = _sha256_path(Path(path))
        except BaseException:
            _assert_consumed_unchanged(
                fixture_manifest,
                frozen_consumed_sha256,
                context=f"immediately after failed SHA256 read {Path(path).name}",
            )
            raise
        _assert_consumed_unchanged(
            fixture_manifest,
            frozen_consumed_sha256,
            context=f"immediately after SHA256 read {Path(path).name}",
        )
        return observed

    return guarded


def _guarded_point_cloud_reader(
    *, fixture_manifest: Path, frozen_consumed_sha256: Mapping[str, str]
) -> Callable[..., Any]:
    def guarded(path: str, **kwargs: Any) -> Any:
        _assert_consumed_unchanged(
            fixture_manifest,
            frozen_consumed_sha256,
            context=f"immediately before Open3D read {Path(path).name}",
        )
        try:
            cloud = o3d.io.read_point_cloud(path, **kwargs)
        except BaseException:
            _assert_consumed_unchanged(
                fixture_manifest,
                frozen_consumed_sha256,
                context=f"immediately after failed Open3D read {Path(path).name}",
            )
            raise
        _assert_consumed_unchanged(
            fixture_manifest,
            frozen_consumed_sha256,
            context=f"immediately after Open3D read {Path(path).name}",
        )
        return cloud

    return guarded


def _legacy_execution(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
) -> Mapping[str, Any]:
    legacy = _load_legacy_runner()
    proxy = _CapturingIntakeProxy(INTAKE)
    stub = _QualificationParentV5Stub()
    legacy.INTAKE = proxy
    legacy.V5 = stub
    if (
        legacy.np is not np
        or legacy.o3d is not o3d
        or legacy.INTAKE is not proxy
        or legacy.V5 is not stub
        or legacy.GOVERNANCE is not G34
    ):
        raise RuntimeError("qualification EXP34 legacy injected dependency drift")
    preregistrations = _load_qualification_preregistrations()
    with tempfile.TemporaryDirectory(prefix="exp35-legacy-source-free-") as temporary:
        journal = Path(temporary) / "synthetic-journal.json"
        accounting = legacy._new_scan_accounting()
        expected = {token: str(inputs[token]["sha256"]) for token in ("N009", "N010")}
        first_cloud, second_cloud = legacy.perform_ordered_scan_access(
            mode="PRIMARY",
            scan_first=Path(inputs["N009"]["resolved_path"]),
            scan_second=Path(inputs["N010"]["resolved_path"]),
            journal_path=journal,
            accounting=accounting,
            expected_hashes=expected,
            sha256_func=_guarded_fixture_hasher(
                fixture_manifest=fixture_manifest,
                frozen_consumed_sha256=frozen_consumed_sha256,
            ),
            point_cloud_reader=_guarded_point_cloud_reader(
                fixture_manifest=fixture_manifest,
                frozen_consumed_sha256=frozen_consumed_sha256,
            ),
        )
        raw_first, normals_first = legacy.materialize_loaded_cloud(
            first_cloud,
            token="N009",
            expected_row_count=int(inputs["N009"]["row_count"]),
        )
        raw_second, normals_second = legacy.materialize_loaded_cloud(
            second_cloud,
            token="N010",
            expected_row_count=int(inputs["N010"]["row_count"]),
        )
        # Match the exact EXP34 production lifetime: release both Open3D
        # objects, retain all four raw arrays, then call the hash-bound wrapper.
        # That wrapper allocates both uint64 raw-ID arrays before its first
        # filter call.  The transparent proxy captures its exact filtered
        # outputs and pair result, so no second filter/pair call is needed.
        del first_cloud, second_cloud
        wrapper_call_count = 0
        wrapper_call_count += 1
        computation = legacy.execute_source_exp34(
            **_qualification_wrapper_arguments(preregistrations, experiment="exp34"),
            raw_first=raw_first,
            normals_first=normals_first,
            raw_second=raw_second,
            normals_second=normals_second,
        )
        del raw_first, normals_first, raw_second, normals_second
        if wrapper_call_count != 1:
            raise RuntimeError("exact EXP34 execute_source_exp34 call-count drift")
        first = proxy.filtered_by_token["N009"]
        second = proxy.filtered_by_token["N010"]
        pair = proxy.pair
        if pair is None:
            raise RuntimeError("exact EXP34 wrapper did not call pair_actionability")
        call_accounting = _qualification_call_accounting(
            mode="LEGACY_BATCH",
            proxy=proxy,
            stub=stub,
            exact_exp34_wrapper_call_count=wrapper_call_count,
            exact_exp35_filtered_wrapper_call_count=0,
        )
        lifecycle = {
            "explicit_gc_collect_call_count": 0,
            "maximum_live_source_cloud_count": 2,
            "maximum_live_unfiltered_member_bundle_count": 2,
            "N009_release_verified": False,
            "N010_release_verified": False,
            "filter_source_rows_call_count_by_token": dict(
                proxy.filter_call_count_by_token
            ),
            "pair_actionability_call_count": 1,
        }
        if (
            accounting["physical_scan_hash_call_count"],
            accounting["physical_scan_point_cloud_load_call_count"],
            accounting["physical_scan_byte_open_operation_count"],
        ) != (2, 2, 4):
            raise RuntimeError("legacy source-free synthetic operation count drift")
        return {
            "first": first,
            "second": second,
            "pair": pair,
            "lifecycle": lifecycle,
            "computation": computation,
            "call_accounting": call_accounting,
            "synthetic_operation_counts": {
                "hash": 2,
                "point_cloud_load": 2,
                "byte_open": 4,
            },
        }


def _legacy_intake(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    outcome = _legacy_execution(
        inputs,
        fixture_manifest=fixture_manifest,
        frozen_consumed_sha256=frozen_consumed_sha256,
    )
    return (
        outcome["first"],
        outcome["second"],
        outcome["pair"],
        outcome["lifecycle"],
    )


def _sequential_execution(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
) -> Mapping[str, Any]:
    proxy = _CapturingIntakeProxy(INTAKE)
    stub = _QualificationParentV5Stub()
    preregistrations = _load_qualification_preregistrations()
    injected_names = ("np", "o3d", "INTAKE", "V5", "GOVERNANCE")
    prior_injected = {name: SEQUENTIAL.__dict__.get(name) for name in injected_names}
    if any(value is not None for value in prior_injected.values()):
        raise RuntimeError("EXP35 sequential runner has stale pre-injection globals")
    SEQUENTIAL.np = np
    SEQUENTIAL.o3d = o3d
    SEQUENTIAL.INTAKE = proxy
    SEQUENTIAL.V5 = stub
    SEQUENTIAL.GOVERNANCE = G35
    if (
        SEQUENTIAL.np is not np
        or SEQUENTIAL.o3d is not o3d
        or SEQUENTIAL.INTAKE is not proxy
        or SEQUENTIAL.V5 is not stub
        or SEQUENTIAL.GOVERNANCE is not G35
    ):
        raise RuntimeError("EXP35 sequential qualification injected dependency drift")
    try:
        with tempfile.TemporaryDirectory(prefix="exp35-sequential-source-free-") as temporary:
            root = Path(temporary)
            accounting = SEQUENTIAL._new_scan_accounting()
            lifecycle = SEQUENTIAL._new_lifecycle_accounting(mode="PRIMARY")
            first, second = SEQUENTIAL.perform_sequential_member_intake(
                mode="PRIMARY",
                scan_first=Path(inputs["N009"]["resolved_path"]),
                scan_second=Path(inputs["N010"]["resolved_path"]),
                journal_path=root / "synthetic-journal.json",
                lifecycle_path=root / "synthetic-lifecycle.json",
                accounting=accounting,
                lifecycle=lifecycle,
                expected_hashes={token: str(inputs[token]["sha256"]) for token in ("N009", "N010")},
                expected_row_counts={token: int(inputs[token]["row_count"]) for token in ("N009", "N010")},
                sha256_func=_guarded_fixture_hasher(
                    fixture_manifest=fixture_manifest,
                    frozen_consumed_sha256=frozen_consumed_sha256,
                ),
                point_cloud_reader=_guarded_point_cloud_reader(
                    fixture_manifest=fixture_manifest,
                    frozen_consumed_sha256=frozen_consumed_sha256,
                ),
            )
            wrapper_call_count = 0
            wrapper_call_count += 1
            computation = SEQUENTIAL.execute_filtered_source_exp35(
                **_qualification_wrapper_arguments(
                    preregistrations, experiment="exp35"
                ),
                first=first,
                second=second,
            )
            if wrapper_call_count != 1:
                raise RuntimeError(
                    "exact EXP35 execute_filtered_source_exp35 call-count drift"
                )
            pair = proxy.pair
            if pair is None:
                raise RuntimeError(
                    "exact EXP35 filtered wrapper did not call pair_actionability"
                )
            lifecycle["status"] = "COMPLETE_RELEASE_VERIFIED"
            lifecycle_payload = SEQUENTIAL._lifecycle_payload(lifecycle)
            if (
                lifecycle_payload["explicit_gc_collect_call_count"] != 2
                or lifecycle_payload["maximum_live_source_cloud_count"] != 1
                or lifecycle_payload["maximum_live_unfiltered_member_bundle_count"] != 1
                or lifecycle_payload["N009_release_verified"] is not True
                or lifecycle_payload["N010_release_verified"] is not True
                or lifecycle_payload["ordered_events"]
                != [
                    "N009_RELEASE_VERIFIED_BEFORE_N010_LOAD",
                    "N010_RELEASE_VERIFIED_BEFORE_PAIR_ACTIONABILITY",
                ]
            ):
                raise RuntimeError("EXP35 source-free sequential lifecycle proof failed")
            if lifecycle.get("_filter_source_rows_call_count_by_token") != {
                "N009": 1,
                "N010": 1,
            }:
                raise RuntimeError("EXP35 source-free sequential filter-call proof failed")
            lifecycle_payload["filter_source_rows_call_count_by_token"] = dict(
                lifecycle["_filter_source_rows_call_count_by_token"]
            )
            lifecycle_payload["pair_actionability_call_count"] = 1
            call_accounting = _qualification_call_accounting(
                mode="SEQUENTIAL",
                proxy=proxy,
                stub=stub,
                exact_exp34_wrapper_call_count=0,
                exact_exp35_filtered_wrapper_call_count=wrapper_call_count,
            )
            return {
                "first": first,
                "second": second,
                "pair": pair,
                "lifecycle": lifecycle_payload,
                "computation": computation,
                "call_accounting": call_accounting,
                "synthetic_operation_counts": {
                    "hash": 2,
                    "point_cloud_load": 2,
                    "byte_open": 4,
                },
            }
    finally:
        # The global production module must not retain capturing proxies (and
        # therefore retained arrays) across boundary or measured executions.
        for name, value in prior_injected.items():
            setattr(SEQUENTIAL, name, value)


def _sequential_intake(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    outcome = _sequential_execution(
        inputs,
        fixture_manifest=fixture_manifest,
        frozen_consumed_sha256=frozen_consumed_sha256,
    )
    return (
        outcome["first"],
        outcome["second"],
        outcome["pair"],
        outcome["lifecycle"],
    )


def worker_compute(
    *, mode: str, fixture_manifest: Path, require_exact_counts: bool = True
) -> Mapping[str, Any]:
    runtime_identity = _runtime_identity_snapshot(
        strict_sys_path=require_exact_counts
    )
    manifest, inputs = _fixture_inputs(
        fixture_manifest, require_exact_counts=require_exact_counts
    )
    consumed_at_start = consumed_sha256_map(fixture_manifest)
    _validate_consumed_fixture_bindings(manifest, consumed_at_start)
    _validate_runtime_against_consumed(runtime_identity, consumed_at_start)
    for token, key in (("N009", "maximal_N009"), ("N010", "maximal_N010")):
        if consumed_at_start[f"fixture_{key}"] != str(inputs[token]["sha256"]):
            raise RuntimeError(f"EXP35 consumed-map fixture binding drift: {token}")
    _assert_runtime_identity_unchanged(
        runtime_identity,
        context="before worker science",
        strict_sys_path=require_exact_counts,
    )
    if mode == "LEGACY_BATCH":
        outcome = _legacy_execution(
            inputs,
            fixture_manifest=fixture_manifest,
            frozen_consumed_sha256=consumed_at_start,
        )
    elif mode == "SEQUENTIAL":
        outcome = _sequential_execution(
            inputs,
            fixture_manifest=fixture_manifest,
            frozen_consumed_sha256=consumed_at_start,
        )
    else:
        raise ValueError("unknown EXP35 memory worker mode")
    first = outcome["first"]
    second = outcome["second"]
    pair = outcome["pair"]
    reports = {"N009": first["report"], "N010": second["report"]}
    projection = _scientific_projection(first, second, pair)
    nested_projection = _nested_payload_projection(outcome["computation"])
    _assert_consumed_unchanged(
        fixture_manifest,
        consumed_at_start,
        context="at worker publication boundary",
    )
    _assert_runtime_identity_unchanged(
        runtime_identity,
        context="at worker publication boundary",
        strict_sys_path=require_exact_counts,
    )
    return {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_SOURCE_FREE_MEMORY_WORKER",
        "mode": mode,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fixture_manifest_sha256": consumed_at_start["fixture_manifest"],
        "consumed_sha256_map": consumed_at_start,
        "runtime_identity": runtime_identity,
        "maximal_fixture_sha256": {
            token: str(inputs[token]["sha256"]) for token in ("N009", "N010")
        },
        "scientific_equivalence_projection": projection,
        "report_sha256": {
            token: hashlib.sha256(_strict_bytes(reports[token])).hexdigest()
            for token in ("N009", "N010")
        },
        "pair_actionability_sha256": hashlib.sha256(_strict_bytes(pair)).hexdigest(),
        "nested_payload_projection": nested_projection,
        "nested_payload_projection_sha256": hashlib.sha256(
            _strict_bytes(nested_projection)
        ).hexdigest(),
        "qualification_call_accounting": outcome["call_accounting"],
        "lifecycle": outcome["lifecycle"],
        "synthetic_fixture_hash_call_count": 2,
        "synthetic_fixture_point_cloud_load_call_count": 2,
        "synthetic_fixture_byte_open_operation_count": 4,
        "physical_scan_hash_call_count": 0,
        "physical_scan_point_cloud_load_call_count": 0,
        "physical_scan_byte_open_operation_count": 0,
        "evaluator_role_label_gold_quality_or_expected_count_access_count": 0,
        "source_free_v5_wrapper_import_or_call_count": 0,
        "v5_filter_entry_point_call_count": 0,
    }


def _validate_boundary_projection(projection: Mapping[str, Any]) -> None:
    if set(projection) != {"retained", "reports", "pair_actionability"}:
        raise RuntimeError("EXP35 boundary projection inventory drift")
    for token in ("N009", "N010"):
        report = projection["reports"][token]
        if (
            report.get("original_row_count") != ROW_COUNTS[token]
            or report.get("retained_row_count") != ROW_COUNTS[token] - 8
            or report.get("rejected_row_count") != 8
            or report.get("raw_predicate_counts")
            != {
                "NONFINITE_POINT": 2,
                "NONFINITE_NORMAL": 4,
                "NORMAL_NORM_LE_1E_12": 3,
            }
            or report.get("exclusive_rejection_counts")
            != {
                "NONFINITE_POINT": 2,
                "NONFINITE_NORMAL": 3,
                "NORMAL_NORM_LE_1E_12": 3,
            }
            or report.get("rejected_original_row_ids_sorted") != list(range(8))
            or report.get("actionable") is not True
        ):
            raise RuntimeError(f"EXP35 exact-size boundary report drift: {token}")
        for field, expected_shape in (
            ("points", [ROW_COUNTS[token] - 8, 3]),
            ("normals", [ROW_COUNTS[token] - 8, 3]),
            ("original_row_ids", [ROW_COUNTS[token] - 8]),
        ):
            retained = projection["retained"][token][field]
            if retained.get("shape") != expected_shape:
                raise RuntimeError(
                    f"EXP35 exact-size boundary retained shape drift: {token}/{field}"
                )
    if projection.get("pair_actionability") != {
        "actionable": True,
        "failed_member_tokens": [],
        "status": "PASS_SOURCE_INTERFACE_ACTIONABLE",
    }:
        raise RuntimeError("EXP35 exact-size boundary pair actionability drift")


def _run_boundary_equivalence_stage(
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str],
    frozen_runtime_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the exact-size Open3D boundary oracle before any measured trial."""

    if any(int(inputs[token]["row_count"]) != ROW_COUNTS[token] for token in ROW_COUNTS):
        raise RuntimeError("EXP35 boundary stage is not exact-size")
    _validate_runtime_against_consumed(
        frozen_runtime_identity, frozen_consumed_sha256
    )
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context="before exact-size boundary-equivalence legacy execution",
        strict_sys_path=True,
    )
    _assert_consumed_unchanged(
        fixture_manifest,
        frozen_consumed_sha256,
        context="before exact-size boundary-equivalence legacy execution",
    )
    legacy = _legacy_execution(
        inputs,
        fixture_manifest=fixture_manifest,
        frozen_consumed_sha256=frozen_consumed_sha256,
    )
    legacy_projection = _scientific_projection(
        legacy["first"], legacy["second"], legacy["pair"]
    )
    legacy_nested = _nested_payload_projection(legacy["computation"])
    legacy_calls = dict(legacy["call_accounting"])
    legacy_lifecycle = dict(legacy["lifecycle"])
    _validate_boundary_projection(legacy_projection)
    _validate_nested_payload_projection(legacy_nested, legacy_projection)
    del legacy
    gc.collect()

    _assert_consumed_unchanged(
        fixture_manifest,
        frozen_consumed_sha256,
        context="after legacy release and before exact-size boundary-equivalence sequential execution",
    )
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context="after legacy release and before exact-size boundary-equivalence sequential execution",
        strict_sys_path=True,
    )
    sequential = _sequential_execution(
        inputs,
        fixture_manifest=fixture_manifest,
        frozen_consumed_sha256=frozen_consumed_sha256,
    )
    sequential_projection = _scientific_projection(
        sequential["first"], sequential["second"], sequential["pair"]
    )
    sequential_nested = _nested_payload_projection(sequential["computation"])
    sequential_calls = dict(sequential["call_accounting"])
    sequential_lifecycle = dict(sequential["lifecycle"])
    _validate_boundary_projection(sequential_projection)
    _validate_nested_payload_projection(sequential_nested, sequential_projection)
    scientific_equal = _strict_bytes(legacy_projection) == _strict_bytes(
        sequential_projection
    )
    reports_equal = _strict_bytes(legacy_projection["reports"]) == _strict_bytes(
        sequential_projection["reports"]
    )
    nested_equal = _strict_bytes(legacy_nested) == _strict_bytes(sequential_nested)
    if not scientific_equal or not reports_equal or not nested_equal:
        raise RuntimeError("EXP35 exact-size boundary legacy/sequential equivalence failed")
    del sequential
    gc.collect()
    _assert_consumed_unchanged(
        fixture_manifest,
        frozen_consumed_sha256,
        context="after exact-size boundary-equivalence release before measured trials",
    )
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context="after exact-size boundary-equivalence release before measured trials",
        strict_sys_path=True,
    )
    evidence = {
        "schema": BOUNDARY_EVIDENCE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE",
        "formal_stage": "AFTER_QUALIFICATION_ROOT_CLAIM_BEFORE_SIX_MEASURED_MAXIMAL_TRIALS",
        "fixture": "BOUNDARY_EQUIVALENCE",
        "fixture_sha256": {
            token: str(inputs[token]["sha256"]) for token in ("N009", "N010")
        },
        "row_counts": dict(ROW_COUNTS),
        "runtime_identity": dict(frozen_runtime_identity),
        "actual_open3d_reader_and_production_materializer_used_by_both_modes": True,
        "canonical_scientific_equivalence_projection": legacy_projection,
        "canonical_scientific_equivalence_projection_sha256": hashlib.sha256(
            _strict_bytes(legacy_projection)
        ).hexdigest(),
        "canonical_reports_sha256": hashlib.sha256(
            _strict_bytes(legacy_projection["reports"])
        ).hexdigest(),
        "canonical_nested_payload_projection": legacy_nested,
        "canonical_nested_payload_projection_sha256": hashlib.sha256(
            _strict_bytes(legacy_nested)
        ).hexdigest(),
        "scientific_projection_canonical_bytes_identical": scientific_equal,
        "reports_canonical_bytes_identical": reports_equal,
        "nested_payload_canonical_bytes_identical": nested_equal,
        "legacy_call_accounting": legacy_calls,
        "sequential_call_accounting": sequential_calls,
        "legacy_lifecycle": legacy_lifecycle,
        "sequential_lifecycle": sequential_lifecycle,
        "controller_delete_then_gc_collect_call_count": 2,
        "legacy_arrays_released_before_sequential_execution": True,
        "sequential_arrays_released_before_measured_trial_1": True,
        "extra_file_count": 0,
        "physical_scan_access_count": 0,
        "evaluator_access_count": 0,
    }
    # Prove the embedded evidence itself is strict-canonical serializable now,
    # before any measured trial can be launched.
    _strict_bytes(evidence)
    return evidence


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_scientific_projection(projection: Any) -> None:
    if not isinstance(projection, Mapping) or set(projection) != {
        "retained",
        "reports",
        "pair_actionability",
    }:
        raise ValueError("EXP35 scientific projection field inventory drift")
    retained = projection["retained"]
    reports = projection["reports"]
    pair = projection["pair_actionability"]
    if (
        not isinstance(retained, Mapping)
        or set(retained) != {"N009", "N010"}
        or not isinstance(reports, Mapping)
        or set(reports) != {"N009", "N010"}
        or not isinstance(pair, Mapping)
    ):
        raise ValueError("EXP35 scientific projection token inventory drift")
    for token in ("N009", "N010"):
        report = reports[token]
        if (
            not isinstance(report, Mapping)
            or report.get("schema") != INTAKE.SCHEMA
            or report.get("member_token") != token
            or isinstance(report.get("retained_row_count"), bool)
            or not isinstance(report.get("retained_row_count"), int)
            or report.get("original_row_count") != ROW_COUNTS[token]
            or report.get("retained_row_count") != ROW_COUNTS[token]
            or report.get("rejected_row_count") != 0
            or report.get("actionable") is not True
        ):
            raise ValueError(f"EXP35 projected source report identity drift: {token}")
        token_retained = retained[token]
        if not isinstance(token_retained, Mapping) or set(token_retained) != {
            "points",
            "normals",
            "original_row_ids",
        }:
            raise ValueError(f"EXP35 retained projection inventory drift: {token}")
        retained_count = int(report["retained_row_count"])
        expected = {
            "points": (np.dtype(np.float64).str, [retained_count, 3]),
            "normals": (np.dtype(np.float64).str, [retained_count, 3]),
            "original_row_ids": (np.dtype(np.uint64).str, [retained_count]),
        }
        for field, (dtype, shape) in expected.items():
            row = token_retained[field]
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"dtype", "shape", "c_contiguous", "owns_data", "tobytes_sha256"}
                or row.get("dtype") != dtype
                or row.get("shape") != shape
                or row.get("c_contiguous") is not True
                or row.get("owns_data") is not True
                or not _is_sha256(row.get("tobytes_sha256"))
            ):
                raise ValueError(f"EXP35 retained projection drift: {token}/{field}")
        if (
            token_retained["points"]["tobytes_sha256"]
            != report.get("retained_points_sha256")
            or token_retained["normals"]["tobytes_sha256"]
            != report.get("retained_normals_sha256")
            or token_retained["original_row_ids"]["tobytes_sha256"]
            != report.get("retained_identity_sequence_sha256")
        ):
            raise ValueError(f"EXP35 retained/report byte-hash linkage drift: {token}")
    expected_pair = INTAKE.pair_actionability(
        {"report": reports["N009"]}, {"report": reports["N010"]}
    )
    if pair != expected_pair:
        raise ValueError("EXP35 projected pair-actionability drift")


def _validate_nested_payload_projection(
    nested: Any, scientific: Mapping[str, Any]
) -> None:
    if not isinstance(nested, Mapping) or set(nested) != {
        "status",
        "reason",
        "scientific_execute_source_reached",
        "semantic_payload",
        "visual_payload",
        "engine_metrics",
        "engine_accounting",
    }:
        raise ValueError("EXP35 nested qualification projection inventory drift")
    semantic = nested["semantic_payload"]
    visual = nested["visual_payload"]
    metrics = nested["engine_metrics"]
    accounting = nested["engine_accounting"]
    if (
        nested.get("status") != "PASS_SOURCE_FREE_QUALIFICATION_PARENT_V5_STUB"
        or nested.get("reason") is not None
        or nested.get("scientific_execute_source_reached") is not True
        or not isinstance(semantic, Mapping)
        or semantic.get("schema") != "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1"
        or semantic.get("qualification_only_parent_v5_stub") is not True
        or not isinstance(visual, Mapping)
        or visual.get("schema") != "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1"
        or visual.get("qualification_only_parent_v5_stub") is not True
        or not isinstance(metrics, Mapping)
        or metrics.get("schema") != "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1"
        or metrics.get("qualification_only_parent_v5_stub") is not True
        or not isinstance(accounting, Mapping)
        or accounting.get("v5_engine_invocation_count") != 1
        or accounting.get("unique_member_fit_count") != 38
        or accounting.get("kd_node_query_count") != 306_926
        or accounting.get("robustness_row_count") != 32
    ):
        raise ValueError("EXP35 nested qualification payload identity drift")
    input_projection = semantic.get("input_projection")
    if (
        input_projection != visual.get("input_projection")
        or input_projection != metrics.get("input_projection")
        or not isinstance(input_projection, Mapping)
        or set(input_projection) != {"N009", "N010"}
    ):
        raise ValueError("EXP35 nested qualification input projection drift")
    retained = scientific["retained"]
    for token in ("N009", "N010"):
        row = input_projection[token]
        if not isinstance(row, Mapping) or row != {
            "points_dtype": retained[token]["points"]["dtype"],
            "points_shape": retained[token]["points"]["shape"],
            "points_tobytes_sha256": retained[token]["points"]["tobytes_sha256"],
            "normals_dtype": retained[token]["normals"]["dtype"],
            "normals_shape": retained[token]["normals"]["shape"],
            "normals_tobytes_sha256": retained[token]["normals"]["tobytes_sha256"],
        }:
            raise ValueError(f"EXP35 nested/retained projection linkage drift: {token}")


def _validate_worker_result(
    result: Mapping[str, Any],
    *,
    expected_mode: str,
    fixture_manifest_sha256: str,
    frozen_consumed_sha256: Mapping[str, str],
    frozen_runtime_identity: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "experiment_id",
        "status",
        "mode",
        "preregistration_sha256",
        "fixture_manifest_sha256",
        "consumed_sha256_map",
        "runtime_identity",
        "maximal_fixture_sha256",
        "scientific_equivalence_projection",
        "report_sha256",
        "pair_actionability_sha256",
        "nested_payload_projection",
        "nested_payload_projection_sha256",
        "qualification_call_accounting",
        "lifecycle",
        "synthetic_fixture_hash_call_count",
        "synthetic_fixture_point_cloud_load_call_count",
        "synthetic_fixture_byte_open_operation_count",
        "physical_scan_hash_call_count",
        "physical_scan_point_cloud_load_call_count",
        "physical_scan_byte_open_operation_count",
        "evaluator_role_label_gold_quality_or_expected_count_access_count",
        "source_free_v5_wrapper_import_or_call_count",
        "v5_filter_entry_point_call_count",
    }
    if set(result) != expected_keys:
        raise ValueError("EXP35 worker result field inventory drift")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "COMPLETE_SOURCE_FREE_MEMORY_WORKER"
        or result.get("mode") != expected_mode
        or result.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or result.get("fixture_manifest_sha256") != fixture_manifest_sha256
        or result.get("consumed_sha256_map") != dict(frozen_consumed_sha256)
        or result.get("runtime_identity") != dict(frozen_runtime_identity)
    ):
        raise ValueError("EXP35 worker result identity/binding drift")
    _validate_runtime_identity(result["runtime_identity"])
    _validate_runtime_against_consumed(
        result["runtime_identity"], frozen_consumed_sha256
    )
    expected_fixture_hashes = {
        "N009": frozen_consumed_sha256["fixture_maximal_N009"],
        "N010": frozen_consumed_sha256["fixture_maximal_N010"],
    }
    if result.get("maximal_fixture_sha256") != expected_fixture_hashes:
        raise ValueError("EXP35 worker maximal-fixture binding drift")
    if (
        result.get("synthetic_fixture_hash_call_count") != 2
        or result.get("synthetic_fixture_point_cloud_load_call_count") != 2
        or result.get("synthetic_fixture_byte_open_operation_count") != 4
    ):
        raise ValueError("EXP35 worker synthetic operation accounting drift")
    forbidden_zero_fields = (
        "physical_scan_hash_call_count",
        "physical_scan_point_cloud_load_call_count",
        "physical_scan_byte_open_operation_count",
        "evaluator_role_label_gold_quality_or_expected_count_access_count",
        "source_free_v5_wrapper_import_or_call_count",
        "v5_filter_entry_point_call_count",
    )
    if any(result.get(field) != 0 for field in forbidden_zero_fields):
        raise ValueError("EXP35 worker forbidden-access counter is nonzero")

    projection = result.get("scientific_equivalence_projection")
    _validate_scientific_projection(projection)
    reports = projection["reports"]
    pair = projection["pair_actionability"]
    expected_report_hashes = {
        token: hashlib.sha256(_strict_bytes(reports[token])).hexdigest()
        for token in ("N009", "N010")
    }
    if result.get("report_sha256") != expected_report_hashes:
        raise ValueError("EXP35 worker report hash projection drift")
    if result.get("pair_actionability_sha256") != hashlib.sha256(
        _strict_bytes(pair)
    ).hexdigest():
        raise ValueError("EXP35 worker pair hash projection drift")
    nested = result.get("nested_payload_projection")
    _validate_nested_payload_projection(nested, projection)
    if result.get("nested_payload_projection_sha256") != hashlib.sha256(
        _strict_bytes(nested)
    ).hexdigest():
        raise ValueError("EXP35 worker nested payload hash drift")

    expected_call_accounting = {
        "exact_exp34_execute_source_exp34_call_count": int(
            expected_mode == "LEGACY_BATCH"
        ),
        "exact_exp35_execute_filtered_source_exp35_call_count": int(
            expected_mode == "SEQUENTIAL"
        ),
        "exp31_filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "exp31_pair_actionability_call_count": 1,
        "qualification_parent_v5_stub_call_count": 1,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }
    if result.get("qualification_call_accounting") != expected_call_accounting:
        raise ValueError("EXP35 worker qualification call-accounting drift")

    lifecycle = result.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("EXP35 worker lifecycle is absent")
    common_counts = {
        "filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "pair_actionability_call_count": 1,
    }
    if expected_mode == "SEQUENTIAL":
        expected_lifecycle = {
            "schema": SEQUENTIAL.LIFECYCLE_MEASUREMENT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "mode": "PRIMARY",
            "status": "COMPLETE_RELEASE_VERIFIED",
            "explicit_gc_collect_call_count": 2,
            "maximum_live_source_cloud_count": 1,
            "maximum_live_unfiltered_member_bundle_count": 1,
            "N009_release_verified": True,
            "N010_release_verified": True,
            "ordered_events": [
                "N009_RELEASE_VERIFIED_BEFORE_N010_LOAD",
                "N010_RELEASE_VERIFIED_BEFORE_PAIR_ACTIONABILITY",
            ],
            **common_counts,
        }
    elif expected_mode == "LEGACY_BATCH":
        expected_lifecycle = {
            "explicit_gc_collect_call_count": 0,
            "maximum_live_source_cloud_count": 2,
            "maximum_live_unfiltered_member_bundle_count": 2,
            "N009_release_verified": False,
            "N010_release_verified": False,
            **common_counts,
        }
    else:
        raise ValueError("unknown EXP35 qualification mode")
    if lifecycle != expected_lifecycle:
        raise ValueError(f"EXP35 {expected_mode} lifecycle proof drift")


def _validate_external_measurement(
    measurement: Mapping[str, Any],
    *,
    ordinal: int,
    mode: str,
    frozen_runtime_identity: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "experiment_id",
        "trial_ordinal",
        "mode",
        "returncode",
        "timed_out",
        "rss_limit_exceeded",
        "sampler_failed",
        "resource_kill_reason",
        "elapsed_wall_seconds",
        "sampling_interval_seconds",
        "sampled_peak_process_tree_rss_bytes",
        "time_l_maximum_resident_set_size_bytes",
        "peak_process_tree_rss_bytes",
        "applicable_rss_cap_bytes",
        "runtime_identity",
        "worker_runtime_identity_validated",
        "worker_python_invocation_path",
        "worker_python_realpath",
        "worker_python_sha256",
        "worker_python_isolated_flag",
        "worker_python_no_bytecode_flag",
        "stdout",
        "stderr",
    }
    if set(measurement) != expected_keys:
        raise ValueError("EXP35 external measurement field inventory drift")
    cap = SEQUENTIAL_RSS_CAP_BYTES if mode == "SEQUENTIAL" else LEGACY_RSS_CAP_BYTES
    wall = measurement.get("elapsed_wall_seconds")
    sampled = measurement.get("sampled_peak_process_tree_rss_bytes")
    exact = measurement.get("time_l_maximum_resident_set_size_bytes")
    peak = measurement.get("peak_process_tree_rss_bytes")
    if (
        measurement.get("schema") != MEASUREMENT_SCHEMA
        or measurement.get("experiment_id") != EXPERIMENT_ID
        or isinstance(measurement.get("trial_ordinal"), bool)
        or not isinstance(measurement.get("trial_ordinal"), int)
        or measurement.get("trial_ordinal") != ordinal
        or measurement.get("mode") != mode
        or isinstance(measurement.get("returncode"), bool)
        or not isinstance(measurement.get("returncode"), int)
        or measurement.get("returncode") != 0
        or measurement.get("timed_out") is not False
        or measurement.get("rss_limit_exceeded") is not False
        or measurement.get("sampler_failed") is not False
        or measurement.get("resource_kill_reason") is not None
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
        or not float(wall) < WALL_CAP_SECONDS
        or measurement.get("sampling_interval_seconds") != SAMPLE_INTERVAL_SECONDS
        or isinstance(sampled, bool)
        or not isinstance(sampled, int)
        or sampled < 0
        or isinstance(exact, bool)
        or not isinstance(exact, int)
        or exact <= 0
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak != max(sampled, exact)
        or peak > cap
        or measurement.get("applicable_rss_cap_bytes") != cap
        or measurement.get("runtime_identity") != dict(frozen_runtime_identity)
        or measurement.get("worker_runtime_identity_validated") is not True
        or measurement.get("worker_python_invocation_path")
        != frozen_runtime_identity["runtime_environment_fingerprint"][
            "python_executable_invocation_path"
        ]
        or measurement.get("worker_python_realpath")
        != frozen_runtime_identity["runtime_environment_fingerprint"][
            "python_executable_realpath"
        ]
        or measurement.get("worker_python_sha256")
        != frozen_runtime_identity["runtime_environment_fingerprint"][
            "python_executable_sha256"
        ]
        or measurement.get("worker_python_isolated_flag") is not True
        or measurement.get("worker_python_no_bytecode_flag") is not True
        or not isinstance(measurement.get("stdout"), str)
        or not isinstance(measurement.get("stderr"), str)
    ):
        raise ValueError("EXP35 external measurement strict gate failed")
    _validate_runtime_identity(measurement["runtime_identity"])


def _validate_published_trial(
    *,
    ordinal: int,
    mode: str,
    trial_root: Path,
    fixture_manifest: Path,
    fixture_manifest_sha256: str,
    frozen_consumed_sha256: Mapping[str, str],
    frozen_runtime_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    result_path = Path(trial_root) / RESULT_NAME
    measurement_path = Path(trial_root) / MEASUREMENT_NAME
    if (
        result_path.is_symlink()
        or measurement_path.is_symlink()
        or not result_path.is_file()
        or not measurement_path.is_file()
    ):
        raise RuntimeError(f"EXP35 source-free trial {ordinal} publication is absent")
    result = _load_canonical_json(result_path)
    measurement = _load_canonical_json(measurement_path)
    _validate_worker_result(
        result,
        expected_mode=mode,
        fixture_manifest_sha256=fixture_manifest_sha256,
        frozen_consumed_sha256=frozen_consumed_sha256,
        frozen_runtime_identity=frozen_runtime_identity,
    )
    _validate_external_measurement(
        measurement,
        ordinal=ordinal,
        mode=mode,
        frozen_runtime_identity=frozen_runtime_identity,
    )
    _assert_consumed_unchanged(
        fixture_manifest,
        frozen_consumed_sha256,
        context=f"after controller validation of trial {ordinal}",
    )
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context=f"after controller validation of trial {ordinal}",
        strict_sys_path=True,
    )
    return {"measurement": measurement, "result": result}


def _run_trial(
    *,
    ordinal: int,
    mode: str,
    trial_root: Path,
    fixture_manifest: Path,
    frozen_consumed_sha256: Mapping[str, str] | None = None,
    frozen_runtime_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if frozen_consumed_sha256 is None or frozen_runtime_identity is None:
        raise RuntimeError("EXP35 trial lacks frozen consumed/runtime identity")
    _validate_runtime_against_consumed(
        frozen_runtime_identity, frozen_consumed_sha256
    )
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context=f"immediately before worker spawn for trial {ordinal}",
        strict_sys_path=True,
    )
    if consumed_sha256_map(fixture_manifest) != dict(frozen_consumed_sha256):
        raise RuntimeError("EXP35 consumed code/fixture TOCTOU before worker spawn")
    environment = frozen_runtime_identity["runtime_environment_fingerprint"]
    worker_command = [
        environment["python_executable_invocation_path"],
        "-I",
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "worker",
        "--mode",
        mode,
        "--fixture-manifest",
        str(fixture_manifest),
        "--output",
        RESULT_NAME,
    ]
    _assert_runtime_identity_unchanged(
        frozen_runtime_identity,
        context=f"at final worker-spawn boundary for trial {ordinal}",
        strict_sys_path=True,
    )
    cap = SEQUENTIAL_RSS_CAP_BYTES if mode == "SEQUENTIAL" else LEGACY_RSS_CAP_BYTES
    with tempfile.TemporaryDirectory(prefix=f"exp35-memory-monitor-{ordinal:02d}-") as temporary:
        prior_cwd = Path.cwd()
        try:
            os.chdir(trial_root)
            monitored = MONITOR.run_measured_command(
                worker_command,
                measurement_root=Path(temporary) / "measurement",
                wall_limit_seconds=WALL_CAP_SECONDS,
                rss_limit_bytes=cap,
            )
        finally:
            os.chdir(prior_cwd)
        monitor_root = Path(temporary) / "measurement"
        stdout = (monitor_root / "stdout.txt").read_text(encoding="utf-8")
        stderr = (monitor_root / "stderr.txt").read_text(encoding="utf-8")
    worker_runtime_identity_validated = False
    if monitored["status"] == "PASS":
        try:
            child_result = _load_canonical_json(trial_root / RESULT_NAME)
            child_identity = child_result.get("runtime_identity")
            _validate_runtime_identity(child_identity)
            worker_runtime_identity_validated = (
                child_identity == dict(frozen_runtime_identity)
            )
        except BaseException:
            worker_runtime_identity_validated = False
    measurement = {
        "schema": MEASUREMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "trial_ordinal": ordinal,
        "mode": mode,
        "returncode": int(monitored["returncode"]),
        "timed_out": bool(monitored["timed_out"]),
        "rss_limit_exceeded": bool(monitored["rss_limit_exceeded"]),
        "sampler_failed": bool(monitored["sampler_failed"]),
        "resource_kill_reason": monitored["resource_kill_reason"],
        "elapsed_wall_seconds": float(monitored["wall_seconds"]),
        "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "sampled_peak_process_tree_rss_bytes": int(
            monitored["sampled_process_tree_peak_rss_bytes"]
        ),
        "time_l_maximum_resident_set_size_bytes": monitored[
            "exact_worker_process_max_rss_bytes"
        ],
        "peak_process_tree_rss_bytes": int(monitored["peak_process_tree_rss_bytes"]),
        "applicable_rss_cap_bytes": cap,
        "runtime_identity": dict(frozen_runtime_identity),
        "worker_runtime_identity_validated": worker_runtime_identity_validated,
        "worker_python_invocation_path": environment[
            "python_executable_invocation_path"
        ],
        "worker_python_realpath": environment["python_executable_realpath"],
        "worker_python_sha256": environment["python_executable_sha256"],
        "worker_python_isolated_flag": True,
        "worker_python_no_bytecode_flag": True,
        "stdout": stdout,
        "stderr": stderr,
    }
    _strict_write(trial_root / MEASUREMENT_NAME, measurement)
    if monitored["status"] != "PASS":
        raise RuntimeError(f"EXP35 source-free qualification trial {ordinal} failed")
    if not worker_runtime_identity_validated:
        raise RuntimeError(
            f"EXP35 source-free qualification trial {ordinal} runtime identity failed"
        )
    return _validate_published_trial(
        ordinal=ordinal,
        mode=mode,
        trial_root=trial_root,
        fixture_manifest=fixture_manifest,
        fixture_manifest_sha256=frozen_consumed_sha256["fixture_manifest"],
        frozen_consumed_sha256=frozen_consumed_sha256,
        frozen_runtime_identity=frozen_runtime_identity,
    )


def qualify(*, fixture_manifest: Path, qualification_root: Path) -> Mapping[str, Any]:
    fixture_manifest = Path(fixture_manifest)
    qualification_root = Path(qualification_root)
    if (
        not fixture_manifest.is_absolute()
        or not qualification_root.is_absolute()
        or str(fixture_manifest) != str(FIXTURE_MANIFEST)
        or str(qualification_root) != str(QUALIFICATION_ROOT)
        or ".." in fixture_manifest.parts
        or ".." in qualification_root.parts
    ):
        raise ValueError("EXP35 qualification paths are not exact canonical paths")
    _reject_symlink_components(qualification_root, label="EXP35 qualification root")
    if qualification_root.exists():
        raise FileExistsError("EXP35 qualification root already exists; rerun is forbidden")
    # The absent canonical directory is the one-shot O_EXCL-equivalent claim.
    # Claim it before reading, statting, or hashing the fixture manifest/PLYs so
    # every post-claim preflight failure can publish a durable terminal record.
    qualification_root.mkdir(parents=True, exist_ok=False)
    trials: list[dict[str, Any]] = []
    frozen_consumed_sha256: Mapping[str, str] | None = None
    frozen_runtime_identity: Mapping[str, Any] | None = None
    failed_trial_ordinal: int | None = None
    failed_trial_mode: str | None = None
    failed_qualification_stage = "RUNTIME_ENVIRONMENT_PREFLIGHT_AFTER_ROOT_CLAIM"
    boundary_evidence: Mapping[str, Any] | None = None
    aggregate_path = qualification_root / AGGREGATE_NAME
    try:
        frozen_runtime_identity = _runtime_identity_snapshot(strict_sys_path=True)
        failed_qualification_stage = "FIXTURE_PREFLIGHT_AFTER_ROOT_CLAIM"
        manifest, _maximal_inputs = _fixture_inputs(
            fixture_manifest, require_exact_counts=True
        )
        _boundary_manifest, boundary_inputs = _fixture_inputs(
            fixture_manifest,
            require_exact_counts=True,
            fixture="BOUNDARY_EQUIVALENCE",
        )
        if _strict_bytes(_boundary_manifest) != _strict_bytes(manifest):
            raise RuntimeError("EXP35 boundary/maximal manifest parse drift")
        frozen_consumed_sha256 = consumed_sha256_map(fixture_manifest)
        _validate_consumed_fixture_bindings(manifest, frozen_consumed_sha256)
        _validate_runtime_against_consumed(
            frozen_runtime_identity, frozen_consumed_sha256
        )
        failed_qualification_stage = "EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE"
        boundary_evidence = _run_boundary_equivalence_stage(
            inputs=boundary_inputs,
            fixture_manifest=fixture_manifest,
            frozen_consumed_sha256=frozen_consumed_sha256,
            frozen_runtime_identity=frozen_runtime_identity,
        )
        if boundary_evidence.get("runtime_identity") != dict(
            frozen_runtime_identity
        ):
            raise RuntimeError("EXP35 boundary runtime identity drift")
        _validate_runtime_identity(boundary_evidence["runtime_identity"])
        failed_qualification_stage = "SIX_MEASURED_MAXIMAL_RETENTION_TRIALS"
        for ordinal, mode, directory in SCHEDULE:
            failed_trial_ordinal = ordinal
            failed_trial_mode = mode
            trial_root = qualification_root / directory
            trial_root.mkdir(exist_ok=False)
            outcome = _run_trial(
                ordinal=ordinal,
                mode=mode,
                trial_root=trial_root,
                fixture_manifest=fixture_manifest,
                frozen_consumed_sha256=frozen_consumed_sha256,
                frozen_runtime_identity=frozen_runtime_identity,
            )
            # A monkeypatched controller test cannot bypass the real validator;
            # always reload and validate the exact canonical child bytes here.
            outcome = _validate_published_trial(
                ordinal=ordinal,
                mode=mode,
                trial_root=trial_root,
                fixture_manifest=fixture_manifest,
                fixture_manifest_sha256=frozen_consumed_sha256["fixture_manifest"],
                frozen_consumed_sha256=frozen_consumed_sha256,
                frozen_runtime_identity=frozen_runtime_identity,
            )
            result_path = trial_root / RESULT_NAME
            measurement_path = trial_root / MEASUREMENT_NAME
            trials.append(
                {
                    "trial_ordinal": ordinal,
                    "mode": mode,
                    "result_path": _manifest_path_string(result_path),
                    "result_sha256": _sha256_path(result_path),
                    "measurement_path": _manifest_path_string(measurement_path),
                    "measurement_sha256": _sha256_path(measurement_path),
                    "peak_process_tree_rss_bytes": outcome["measurement"][
                        "peak_process_tree_rss_bytes"
                    ],
                    "scientific_projection_sha256": hashlib.sha256(
                        _strict_bytes(
                            scientific_equivalence_projection(outcome["result"])
                        )
                    ).hexdigest(),
                    "nested_payload_projection_sha256": outcome["result"][
                        "nested_payload_projection_sha256"
                    ],
                    "consumed_sha256_map_sha256": hashlib.sha256(
                        _strict_bytes(outcome["result"]["consumed_sha256_map"])
                    ).hexdigest(),
                    "runtime_environment_fingerprint_sha256": outcome["result"][
                        "runtime_identity"
                    ]["runtime_environment_fingerprint_sha256"],
                    "loaded_module_origin_path_sha256_map_sha256": outcome[
                        "result"
                    ]["runtime_identity"][
                        "loaded_module_origin_path_sha256_map_sha256"
                    ],
                }
            )

        # The sixth child is not sufficient by itself: rehash the complete
        # consumed map once more immediately after trial 6 and before deciding.
        _assert_consumed_unchanged(
            fixture_manifest,
            frozen_consumed_sha256,
            context="immediately after qualification trial 6",
        )
        _assert_runtime_identity_unchanged(
            frozen_runtime_identity,
            context="immediately after qualification trial 6",
            strict_sys_path=True,
        )
        failed_trial_ordinal = None
        failed_trial_mode = None
        projection_hashes = {row["scientific_projection_sha256"] for row in trials}
        nested_payload_hashes = {
            row["nested_payload_projection_sha256"] for row in trials
        }
        consumed_map_hashes = {row["consumed_sha256_map_sha256"] for row in trials}
        runtime_fingerprint_hashes = {
            row["runtime_environment_fingerprint_sha256"] for row in trials
        }
        module_origin_map_hashes = {
            row["loaded_module_origin_path_sha256_map_sha256"] for row in trials
        }
        per_mode_result_identical = {
            mode: len(
                {
                    row["result_sha256"]
                    for row in trials
                    if row["mode"] == mode
                }
            )
            == 1
            for mode in ("LEGACY_BATCH", "SEQUENTIAL")
        }
        legacy_peaks = [
            row["peak_process_tree_rss_bytes"]
            for row in trials
            if row["mode"] == "LEGACY_BATCH"
        ]
        sequential_peaks = [
            row["peak_process_tree_rss_bytes"]
            for row in trials
            if row["mode"] == "SEQUENTIAL"
        ]
        median_legacy = int(statistics.median(legacy_peaks))
        median_sequential = int(statistics.median(sequential_peaks))
        delta = median_legacy - median_sequential
        passed = (
            len(trials) == 6
            and boundary_evidence.get("status")
            == "PASS_EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE"
            and len(projection_hashes) == 1
            and len(nested_payload_hashes) == 1
            and len(consumed_map_hashes) == 1
            and runtime_fingerprint_hashes
            == {
                frozen_runtime_identity[
                    "runtime_environment_fingerprint_sha256"
                ]
            }
            and module_origin_map_hashes
            == {
                frozen_runtime_identity[
                    "loaded_module_origin_path_sha256_map_sha256"
                ]
            }
            and all(per_mode_result_identical.values())
            and delta >= MINIMUM_MEDIAN_DELTA_BYTES
        )
        aggregate = {
            "schema": AGGREGATE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "PASS_SOURCE_FREE_QUALIFICATION"
                if passed
                else "TERMINAL_PRE_PHYSICAL_NONQUALIFICATION"
            ),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "fixture_manifest_sha256": frozen_consumed_sha256["fixture_manifest"],
            "consumed_sha256_map": frozen_consumed_sha256,
            "runtime_identity": frozen_runtime_identity,
            "qualification_root_claimed_before_fixture_preflight": True,
            "boundary_equivalence_evidence": boundary_evidence,
            "trials": trials,
            "median_legacy_batch_peak_rss_bytes": median_legacy,
            "median_sequential_peak_rss_bytes": median_sequential,
            "median_peak_delta_bytes": delta,
            "minimum_required_delta_bytes": MINIMUM_MEDIAN_DELTA_BYTES,
            "scientific_equivalence_projection_identical": len(projection_hashes)
            == 1,
            "nested_payload_projection_identical": len(nested_payload_hashes) == 1,
            "consumed_sha256_maps_identical": len(consumed_map_hashes) == 1,
            "runtime_identity_identical_across_boundary_parent_and_six_worker_measurement_artifacts": (
                runtime_fingerprint_hashes
                == {
                    frozen_runtime_identity[
                        "runtime_environment_fingerprint_sha256"
                    ]
                }
                and module_origin_map_hashes
                == {
                    frozen_runtime_identity[
                        "loaded_module_origin_path_sha256_map_sha256"
                    ]
                }
            ),
            "per_mode_result_bytes_identical": per_mode_result_identical,
            "final_consumed_sha256_recheck_after_trial_6_passed": True,
            "final_runtime_identity_recheck_after_trial_6_passed": True,
            "boundary_equivalence_completed_before_measured_trial_1": True,
            "physical_scan_access_count": 0,
            "evaluator_access_count": 0,
        }
        _strict_write(aggregate_path, aggregate)
        if not passed:
            raise RuntimeError("EXP35 source-free comparative qualification gate failed")
        return aggregate
    except BaseException as error:
        if not aggregate_path.exists():
            terminal = {
                "schema": AGGREGATE_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "status": "TERMINAL_PRE_PHYSICAL_NONQUALIFICATION",
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "fixture_manifest_sha256": (
                    frozen_consumed_sha256.get("fixture_manifest")
                    if frozen_consumed_sha256 is not None
                    else None
                ),
                "consumed_sha256_map": frozen_consumed_sha256,
                "runtime_identity": frozen_runtime_identity,
                "qualification_root_claimed_before_fixture_preflight": True,
                "failed_qualification_stage": failed_qualification_stage,
                "boundary_equivalence_evidence": boundary_evidence,
                "completed_trials": trials,
                "failed_trial_ordinal": failed_trial_ordinal,
                "failed_trial_mode": failed_trial_mode,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "later_trials_launched": False,
                "physical_scan_access_count": 0,
                "evaluator_access_count": 0,
            }
            _strict_write(aggregate_path, terminal)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--mode", required=True, choices=("LEGACY_BATCH", "SEQUENTIAL"))
    worker.add_argument("--fixture-manifest", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    controller = commands.add_parser("qualify")
    controller.add_argument("--fixture-manifest", required=True, type=Path)
    controller.add_argument("--qualification-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        if args.fixture_manifest != FIXTURE_MANIFEST:
            raise ValueError("memory worker fixture manifest is not the canonical path")
        if args.output != Path(RESULT_NAME):
            raise ValueError("memory worker output must be the exact relative result name")
        result = worker_compute(
            mode=args.mode,
            fixture_manifest=args.fixture_manifest,
            require_exact_counts=True,
        )
        _strict_write(args.output, result)
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "mode": args.mode}, sort_keys=True))
        return 0
    aggregate = qualify(
        fixture_manifest=args.fixture_manifest,
        qualification_root=args.qualification_root,
    )
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": aggregate["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
