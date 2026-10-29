#!/usr/bin/env python3
"""Run three preregistered, source-only F1 arms on an opaque scan cohort.

The runner verifies an embedded component lock before opening a scan, executes
the frozen V51 case pipeline in primary and replay passes, and publishes three
prediction arms:

* A: frozen V51.
* B: frozen V51 with the source-only V43 explosion guard.
* C: frozen V51, source-recovery R3, and axis/span R2 (whose composition also
  applies the development precision-ownership layer).

No scorer is called.  The input rows can contain only a scan path and SHA-256,
and scan basenames must already be opaque sequential case IDs.  Truth, labels,
external expected counts, CAD, drawings, part identity, and evaluator inputs
have no runtime argument or manifest field.

Opaque filenames and parents do not sanitize metadata embedded inside PLY/STL
bytes.  The runner therefore relies on the sealed custodian's raw-byte custody
assumption: eligible delivery bytes are copied unchanged, and any embedded
header metadata is neither inspected nor used by this orchestrator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = 1
INPUT_KIND = "sherman_qc_f1_v51_opaque_scan_only_development_input"
COMPONENT_LOCK_KIND = "sherman_qc_f1_three_arm_component_lock_v1"
FREEZE_PACKET_KIND = "sherman_qc_f1_v51_20260831_prospective_three_arm_freeze_v1"
EXECUTION_KIND = "sherman_qc_f1_v51_three_arm_prospective_execution_v1"
CASE_MANIFEST_KIND = "sherman_qc_f1_v51_three_arm_case_manifest_v1"
ARM_MANIFEST_KIND = "sherman_qc_f1_v51_three_arm_arm_manifest_v1"
ORCHESTRATOR_BINDING_KIND = "sherman_qc_f1_three_arm_orchestrator_binding_v1"
FAIL_CLOSED_KIND = "sherman_qc_f1_three_arm_fail_closed_abstention_v1"
V51_FAIL_CLOSED_KIND = "sherman_qc_v51_fail_closed_case_abstention"

TOP_LEVEL_INPUT_KEYS = frozenset({"schema_version", "kind", "cases"})
CASE_INPUT_KEYS = frozenset({"scan_path", "scan_sha256"})
SCAN_SUFFIXES = frozenset({".ply", ".stl"})
SEALED_ARTIFACT_BINDING_COUNT = 117
SEALED_FREEZE_PACKET_SHA256 = (
    "8dfe45647ad251d8887fc04ffce1e5bb5079f655f85fd4da0ce65a8b49f20a59"
)

COMPONENT_RELATIVE_PATHS: dict[str, str] = {
    "v51_case_runner": "scripts/run_f1_v51_today_scan_only_development.py",
    "v43_guard": "backend/f1_post_v51_v43_explosion_guard_v1.py",
    "recall_runtime": "scripts/apply_f1_v51_source_only_recovery_v1.py",
    "precision_runtime": "backend/f1_post_v51_precision_ownership_v52_development.py",
    "axis_span_runtime": "scripts/apply_f1_v51_today_axis_span_recovery_v1.py",
    "scorer": "scripts/evaluate_f1_against_manual_gold.py",
    "v9_model": "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib",
    "v4_model": "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib",
    "recall_model": "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json",
    "axis_span_model": "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json",
}
ORCHESTRATOR_RELATIVE_PATH = "scripts/run_f1_v51_three_arm_prospective_v1.py"
FROZEN_V51_HELPER_RELATIVE_PATH = (
    "scripts/run_f1_v51_new_untouched_corpus_blind_once.py"
)
OPAQUE_SCAN_ROOT = (
    ROOT
    / "output/research/f1-v51-20260831-prospective-intake/opaque_scans"
).resolve()
LOCKED_SCIENTIFIC_MODULE_NAMES = frozenset(
    {
        "scripts.run_f1_v51_new_untouched_corpus_blind_once",
        "backend.f1_post_v51_v43_explosion_guard_v1",
        "scripts.apply_f1_v51_source_only_recovery_v1",
        "scripts.apply_f1_v51_today_axis_span_recovery_v1",
        "backend.f1_post_v51_precision_ownership_v52_development",
        "scripts.evaluate_f1_against_manual_gold",
    }
)

ARM_FILENAMES: dict[str, str] = {
    "A_V51": "arm_a_v51.json",
    "B_V51_V43_GUARD": "arm_b_v51_v43_guard.json",
    "C_V51_RECALL_R3_AXIS_SPAN_R2": "arm_c_v51_recall_r3_axis_span_r2.json",
}

BaseCaseRunner = Callable[[dict[str, Any], Path, str, Path, Path], dict[str, Any]]


@dataclass(frozen=True)
class RuntimeBundle:
    frozen_v51_helper: ModuleType
    compose_v43_explosion_guard: Callable[..., dict[str, Any]]
    apply_recovery: Callable[..., dict[str, Any]]
    validate_recall_model: Callable[..., dict[str, Any]]
    recall_profile: str
    compose_axis_span_recovery: Callable[..., dict[str, Any]]
    validate_axis_span_model: Callable[..., dict[str, Any]]


class CustodyIntegrityError(RuntimeError):
    """A sealed input/runtime identity changed during authoritative execution."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_scan_manifest(path: Path) -> list[dict[str, Any]]:
    """Validate the exact scan-only boundary and assign opaque case IDs."""

    payload = _read_json(path.resolve())
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_INPUT_KEYS:
        raise ValueError("scan manifest may contain only schema_version, kind, and cases")
    if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != INPUT_KIND:
        raise ValueError("scan manifest schema/kind mismatch")
    rows = payload["cases"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan manifest cases must be a non-empty ordered list")
    if len(rows) > 999:
        raise ValueError("scan manifest supports at most 999 cases")

    result: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    order_keys: list[tuple[str, int]] = []
    for ordinal, row in enumerate(rows, start=1):
        case_id = f"T{ordinal:03d}"
        if not isinstance(row, dict) or set(row) != CASE_INPUT_KEYS:
            raise ValueError(
                f"{case_id}: each case may contain only scan_path and scan_sha256"
            )
        if not isinstance(row["scan_path"], str) or not row["scan_path"]:
            raise ValueError(f"{case_id}: scan_path must be a non-empty string")
        scan = Path(row["scan_path"]).expanduser().resolve()
        if scan in seen_paths:
            raise ValueError(f"{case_id}: duplicate scan path")
        seen_paths.add(scan)
        suffix = scan.suffix.lower()
        if not scan.is_file() or suffix not in SCAN_SUFFIXES:
            raise ValueError(f"{case_id}: scan must be existing PLY or STL geometry")
        if scan.name != f"{case_id}{suffix}":
            raise ValueError(
                f"{case_id}: scan basename must be the opaque case ID plus lowercase extension"
            )
        if scan.parent != OPAQUE_SCAN_ROOT:
            raise ValueError(
                f"{case_id}: scan must be directly inside the canonical opaque scan root"
            )
        expected_sha = row["scan_sha256"]
        if not _valid_sha256(expected_sha):
            raise ValueError(f"{case_id}: scan_sha256 must be lowercase hexadecimal")
        if _sha(scan) != expected_sha:
            raise ValueError(f"{case_id}: scan hash mismatch")
        order_keys.append((expected_sha, scan.stat().st_size))
        result.append(
            {
                "case_id": case_id,
                "scan_path": str(scan),
                "scan_sha256": expected_sha,
            }
        )
    if order_keys != sorted(order_keys):
        raise ValueError("scan manifest violates the sealed SHA-256/byte-size case order")
    return result


def _resolve_locked_component(relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or relative_path != pure.as_posix()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"component lock path is not canonical repo-relative: {relative_path}")
    root = ROOT.resolve()
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"component lock path escapes repository: {relative_path}") from exc
    return resolved


def _validate_artifact_bindings(packet: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = packet.get("artifact_bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("freeze packet artifact_bindings must be a non-empty list")
    if len(rows) != SEALED_ARTIFACT_BINDING_COUNT:
        raise ValueError(
            f"freeze packet must contain exactly {SEALED_ARTIFACT_BINDING_COUNT} artifact bindings"
        )
    if packet.get("artifact_binding_count") != len(rows):
        raise ValueError("freeze packet artifact binding count mismatch")
    expected_digest = packet.get("artifact_binding_digest_sha256")
    if not _valid_sha256(expected_digest):
        raise ValueError("freeze packet artifact binding digest is invalid")
    if _sha_bytes(_canonical_bytes(rows)) != expected_digest:
        raise ValueError("freeze packet artifact binding digest mismatch")

    by_path: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for ordinal, entry in enumerate(rows):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "roles",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"artifact binding {ordinal} has unexpected fields")
        relative_path = entry["path"]
        roles = entry["roles"]
        if not isinstance(relative_path, str):
            raise ValueError(f"artifact binding {ordinal} path must be a string")
        if relative_path in by_path:
            raise ValueError(f"duplicate artifact binding path: {relative_path}")
        if not isinstance(roles, list) or not roles or not all(
            isinstance(role, str) and role for role in roles
        ):
            raise ValueError(f"artifact binding {relative_path} roles are invalid")
        if not _valid_sha256(entry["sha256"]):
            raise ValueError(f"artifact binding {relative_path} sha256 is invalid")
        if (
            isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 0
        ):
            raise ValueError(f"artifact binding {relative_path} size is invalid")
        artifact_path = _resolve_locked_component(relative_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"locked artifact is missing: {relative_path}")
        if artifact_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"artifact binding {relative_path} size mismatch")
        if _sha(artifact_path) != entry["sha256"]:
            raise ValueError(f"artifact binding {relative_path} hash mismatch")
        ordered_paths.append(relative_path)
        by_path[relative_path] = deepcopy(entry)
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("freeze packet artifact bindings are not lexically ordered")
    return by_path


def _validate_protocol_fields(packet: Mapping[str, Any]) -> None:
    if packet.get("status") != "SEALED_BEFORE_2026_08_31_DELIVERY_DISCOVERY":
        raise ValueError("freeze packet status is not the sealed pre-discovery state")
    if packet.get("local_cohort_date") != "2026-08-31" or packet.get(
        "local_timezone"
    ) != "Asia/Jerusalem":
        raise ValueError("freeze packet cohort date/timezone mismatch")
    attestation = packet.get("pre_discovery_attestation")
    if not isinstance(attestation, dict) or set(attestation) != {
        "external_delivery_discovery_started",
        "gmail_inspected_for_cohort",
        "sharepoint_inspected_for_cohort",
        "today_cad_or_drawings_opened",
        "today_part_identity_opened",
        "today_scan_bytes_inspected",
        "today_scan_count_known",
        "today_truth_or_labels_opened",
    } or any(value is not False for value in attestation.values()):
        raise ValueError("freeze packet pre-discovery attestation is invalid")

    arms = packet.get("arms")
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != [
        "A",
        "B",
        "C",
    ]:
        raise ValueError("freeze packet must define arms A, B, and C in order")
    if arms[0].get("name") != "FROZEN_V51_SCAN_ONLY" or arms[0].get(
        "composition"
    ) != ["V16", "V43", "V44", "V50", "V47", "V48", "V49", "V51"]:
        raise ValueError("freeze packet arm A contract mismatch")
    if (
        arms[1].get("name") != "V51_WITH_V43_EXPLOSION_GUARD_V1"
        or arms[1].get("maximum_safe_v43_post_merge_additions") != 1
    ):
        raise ValueError("freeze packet arm B contract mismatch")
    if (
        arms[2].get("name") != "V51_AGGRESSIVE_PRECISION_RECALL_AXIS_SPAN_V1"
        or arms[2].get("recall_profile") != "balanced"
        or arms[2].get("axis_span_profile")
        != "F1_V51_TODAY_AXIS_SPAN_RECOVERY_DEVELOPMENT_V1"
    ):
        raise ValueError("freeze packet arm C contract mismatch")

    custody = packet.get("inference_custody")
    if not isinstance(custody, dict) or (
        custody.get("all_three_arms_complete_before_labeling") is not True
        or custody.get("base_execution_passes") != ["primary", "replay"]
        or custody.get("derived_execution_passes") != ["primary", "replay"]
        or custody.get("allowed_inputs")
        != [
            "opaque raw scan geometry bytes",
            "frozen code bytes",
            "frozen model bytes",
        ]
        or custody.get("forbidden_inputs")
        != [
            "truth",
            "manual labels",
            "sender or expected bend counts",
            "CAD",
            "drawings",
            "original filenames",
            "part IDs",
            "family IDs",
            "delivery prose",
            "predictions from a different arm",
        ]
    ):
        raise ValueError("freeze packet inference custody mismatch")
    decision_rules = packet.get("decision_rules")
    if not isinstance(decision_rules, dict) or any(
        decision_rules.get(key) != "forbidden"
        for key in (
            "case_removal_after_freeze",
            "retuning_after_freeze",
            "threshold_changes_after_freeze",
        )
    ):
        raise ValueError("freeze packet no-retune/no-removal rules mismatch")
    labeling = packet.get("labeling_custody")
    if not isinstance(labeling, dict) or (
        labeling.get("prediction_hidden") is not True
        or labeling.get("opens_only_after")
        != "all arms and both passes are sealed and byte-identical"
    ):
        raise ValueError("freeze packet prediction-hidden labeling custody mismatch")


def _verify_freeze_sidecar_if_present(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        return
    if not sidecar.is_file():
        raise ValueError("freeze SHA-256 sidecar is not a regular file")
    expected = f"{SEALED_FREEZE_PACKET_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise ValueError("freeze SHA-256 sidecar mismatch")


def validate_freeze_packet(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate the freeze packet's exact embedded component path/hash lock."""

    path = path.resolve()
    raw_packet = path.read_bytes()
    if _sha_bytes(raw_packet) != SEALED_FREEZE_PACKET_SHA256:
        raise ValueError("freeze packet SHA-256 does not match the hard-pinned seal")
    _verify_freeze_sidecar_if_present(path)
    packet = json.loads(
        raw_packet.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(packet, dict):
        raise ValueError("freeze packet must be a JSON object")
    if packet.get("schema_version") != SCHEMA_VERSION or packet.get("kind") != FREEZE_PACKET_KIND:
        raise ValueError("freeze packet schema/kind mismatch")
    if (
        packet.get("research_only") is not True
        or packet.get("development_only") is not True
        or packet.get("promotion_eligible") is not False
        or packet.get("generalization_claim_authorized") is not False
    ):
        raise ValueError("freeze packet research authority flags are invalid")
    _validate_protocol_fields(packet)
    contract = (
        (packet.get("cohort_intake") or {})
        .get("scan_only_manifest_contract")
    )
    if contract != {
        "schema_version": SCHEMA_VERSION,
        "kind": INPUT_KIND,
        "top_level_keys_only": ["schema_version", "kind", "cases"],
        "case_keys_only": ["scan_path", "scan_sha256"],
    }:
        raise ValueError("freeze packet scan-only manifest contract mismatch")
    artifact_bindings = _validate_artifact_bindings(packet)

    lock = packet.get("component_lock")
    expected_lock_keys = {
        "schema_version",
        "kind",
        "locked_at_utc",
        "source_only",
        "fail_closed",
        "prospective_orchestrator",
        *COMPONENT_RELATIVE_PATHS,
    }
    if not isinstance(lock, dict) or set(lock) != expected_lock_keys:
        raise ValueError("freeze packet component_lock has unexpected fields")
    if lock["schema_version"] != SCHEMA_VERSION or lock["kind"] != COMPONENT_LOCK_KIND:
        raise ValueError("freeze packet component_lock schema/kind mismatch")
    if lock["source_only"] is not True or lock["fail_closed"] is not True:
        raise ValueError("freeze packet component lock must be source-only and fail-closed")
    orchestrator = lock["prospective_orchestrator"]
    if (
        not isinstance(orchestrator, dict)
        or set(orchestrator) != {"binding_state", "constraint"}
        or orchestrator.get("binding_state") != "AUTHORIZED_FUTURE_BINDING"
        or not isinstance(orchestrator.get("constraint"), str)
        or not orchestrator["constraint"]
    ):
        raise ValueError("freeze packet does not authorize a prospective orchestrator binding")

    resolved: dict[str, Path] = {}
    for key, expected_relative_path in COMPONENT_RELATIVE_PATHS.items():
        entry = lock[key]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"component {key} must contain only path and sha256")
        if entry["path"] != expected_relative_path:
            raise ValueError(f"component {key} path does not match the frozen lexical path")
        if not _valid_sha256(entry["sha256"]):
            raise ValueError(f"component {key} sha256 must be lowercase hexadecimal")
        component_path = _resolve_locked_component(entry["path"])
        if not component_path.is_file():
            raise FileNotFoundError(f"locked component is missing: {entry['path']}")
        actual_sha = _sha(component_path)
        if actual_sha != entry["sha256"]:
            raise ValueError(f"component {key} hash mismatch")
        artifact_entry = artifact_bindings.get(expected_relative_path)
        if artifact_entry is None or artifact_entry["sha256"] != entry["sha256"]:
            raise ValueError(f"component {key} does not match the sealed artifact inventory")
        resolved[key] = component_path
    helper_binding = artifact_bindings.get(FROZEN_V51_HELPER_RELATIVE_PATH)
    if helper_binding is None:
        raise ValueError("sealed artifact inventory omits the frozen V51 helper")
    helper_path = _resolve_locked_component(FROZEN_V51_HELPER_RELATIVE_PATH)
    if _sha(helper_path) != helper_binding["sha256"]:
        raise ValueError("frozen V51 helper runtime hash mismatch")
    resolved["frozen_v51_helper"] = helper_path
    return deepcopy(lock), resolved


def _require_module_path(module: ModuleType, expected_path: Path, *, name: str) -> None:
    module_path_value = getattr(module, "__file__", None)
    if not isinstance(module_path_value, str):
        raise RuntimeError(f"{name} has no filesystem module path")
    if Path(module_path_value).resolve() != expected_path.resolve():
        raise RuntimeError(f"{name} was not imported from its locked runtime path")


def _load_runtime_bundle(component_paths: Mapping[str, Path]) -> RuntimeBundle:
    """Import scientific modules only after the complete freeze has validated."""

    frozen_helper = importlib.import_module(
        "scripts.run_f1_v51_new_untouched_corpus_blind_once"
    )
    guard_module = importlib.import_module(
        "backend.f1_post_v51_v43_explosion_guard_v1"
    )
    recall_module = importlib.import_module(
        "scripts.apply_f1_v51_source_only_recovery_v1"
    )
    axis_module = importlib.import_module(
        "scripts.apply_f1_v51_today_axis_span_recovery_v1"
    )
    precision_module = sys.modules.get(
        "backend.f1_post_v51_precision_ownership_v52_development"
    )
    scorer_module = sys.modules.get("scripts.evaluate_f1_against_manual_gold")
    if not isinstance(precision_module, ModuleType) or not isinstance(
        scorer_module, ModuleType
    ):
        raise RuntimeError("axis/span runtime did not load its locked scientific dependencies")

    for module, key, name in (
        (frozen_helper, "frozen_v51_helper", "frozen V51 helper"),
        (guard_module, "v43_guard", "V43 guard"),
        (recall_module, "recall_runtime", "recall runtime"),
        (axis_module, "axis_span_runtime", "axis/span runtime"),
        (precision_module, "precision_runtime", "precision runtime"),
        (scorer_module, "scorer", "scorer geometry runtime"),
    ):
        _require_module_path(module, component_paths[key], name=name)

    return RuntimeBundle(
        frozen_v51_helper=frozen_helper,
        compose_v43_explosion_guard=guard_module.compose_v43_explosion_guard,
        apply_recovery=recall_module.apply_recovery,
        validate_recall_model=recall_module.validate_model,
        recall_profile=str(recall_module.DEFAULT_PROFILE),
        compose_axis_span_recovery=axis_module.compose_axis_span_recovery,
        validate_axis_span_model=axis_module.validate_model,
    )


def _revalidate_freeze_unchanged(
    freeze_packet_path: Path,
    expected_lock: Mapping[str, Any],
    expected_paths: Mapping[str, Path],
) -> None:
    current_lock, current_paths = validate_freeze_packet(freeze_packet_path)
    if _canonical_bytes(current_lock) != _canonical_bytes(expected_lock):
        raise RuntimeError("component lock changed after initial freeze validation")
    if set(current_paths) != set(expected_paths) or any(
        current_paths[key].resolve() != expected_paths[key].resolve()
        for key in expected_paths
    ):
        raise RuntimeError("locked runtime paths changed after initial validation")


def _assert_no_preloaded_scientific_modules() -> None:
    preloaded = sorted(LOCKED_SCIENTIFIC_MODULE_NAMES.intersection(sys.modules))
    if preloaded:
        raise RuntimeError(
            "authoritative prospective execution requires a fresh process; "
            f"locked scientific modules are already loaded: {preloaded}"
        )


def _validate_predictions(payload: Mapping[str, Any], *, name: str) -> int:
    if payload.get("source_only") is not True or payload.get("research_only") is not True:
        raise ValueError(f"{name} is not source-only research output")
    for key in (
        "expected_count_used",
        "labels_used",
        "cad_used",
        "part_identity_used",
        "drawing_used",
        "drawings_used",
        "sender_counts_used",
        "scoring_opened",
        "truth_opened",
    ):
        if payload.get(key) not in (None, False):
            raise ValueError(f"{name} reports forbidden inference input: {key}")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"{name} predictions must be a list")
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name} prediction {ordinal} is not an object")
        start = row.get("line_start")
        end = row.get("line_end")
        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 3
        ):
            raise ValueError(f"{name} prediction {ordinal} has invalid endpoints")
        coordinates = [float(value) for value in (*start, *end)]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{name} prediction {ordinal} has non-finite endpoints")
        if math.dist(coordinates[:3], coordinates[3:]) <= 1e-9:
            raise ValueError(f"{name} prediction {ordinal} is degenerate")
    if payload.get("candidate_count") not in (None, len(rows)):
        raise ValueError(f"{name} candidate_count does not match predictions")
    return len(rows)


def _build_orchestrator_binding(
    freeze_packet_path: Path,
    component_lock: Mapping[str, Any],
) -> dict[str, Any]:
    orchestrator_path = _resolve_locked_component(ORCHESTRATOR_RELATIVE_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ORCHESTRATOR_BINDING_KIND,
        "orchestrator_path": ORCHESTRATOR_RELATIVE_PATH,
        "orchestrator_sha256": _sha(orchestrator_path),
        "freeze_packet_sha256": _sha(freeze_packet_path),
        "component_lock_sha256": _sha_bytes(_canonical_bytes(component_lock)),
        "bound_before_scan_manifest_open": True,
        "source_only": True,
        "truth_opened": False,
        "labels_used": False,
        "external_counts_used": False,
        "cad_used": False,
        "drawings_used": False,
        "part_identity_used": False,
        "evaluator_inputs_used": False,
    }


def _fail_closed_result(
    case: Mapping[str, Any],
    *,
    arm_id: str,
    failure_stage: str,
    failure_type: str,
    kind: str = FAIL_CLOSED_KIND,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "case_id": case["case_id"],
        "scan_sha256": case["scan_sha256"],
        "arm_id": arm_id,
        "pipeline_status": "FAILED_ABSTAIN",
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "predictions": [],
        "candidate_count": 0,
        "source_only": True,
        "research_only": True,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "promotion_eligible": False,
        "generalization_evidence": False,
    }


def _load_case_payload(path: Path, *, name: str) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} payload must be an object")
    return payload


@contextmanager
def _suppress_nondeterministic_frozen_events(
    frozen_helper: ModuleType,
) -> Iterator[None]:
    """Prevent timestamps from entering the prospective evidence directory."""

    original = frozen_helper._append_event
    frozen_helper._append_event = lambda _payload: None
    try:
        yield
    finally:
        frozen_helper._append_event = original


def _write_arm_result(
    case_root: Path,
    arm_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    candidate_count = _validate_predictions(payload, name=arm_id)
    relative = Path("arms") / ARM_FILENAMES[arm_id]
    path = case_root / relative
    _write_json(path, payload)
    return {
        "artifact": relative.as_posix(),
        "sha256": _sha(path),
        "candidate_count": candidate_count,
        "pipeline_status": str(payload.get("pipeline_status") or "SUCCESS"),
        "kind": str(payload.get("kind") or ""),
    }


def _verify_case_scan_hash(case: Mapping[str, Any], *, phase: str) -> None:
    scan = Path(str(case["scan_path"]))
    try:
        actual = _sha(scan)
    except Exception as exc:
        raise CustodyIntegrityError(
            f"{case['case_id']}: scan unreadable during {phase} custody rehash"
        ) from exc
    if actual != case["scan_sha256"]:
        raise CustodyIntegrityError(
            f"{case['case_id']}: scan bytes changed during {phase} custody rehash"
        )


def _run_one_case(
    case: dict[str, Any],
    output_root: Path,
    pass_name: str,
    component_paths: Mapping[str, Path],
    recall_model: dict[str, Any],
    axis_span_model: dict[str, Any],
    runtime: RuntimeBundle,
    base_case_runner: BaseCaseRunner,
) -> dict[str, Any]:
    # These checks bracket the complete base-plus-derived case execution, not
    # merely the V51 subprocess, so an in-flight scan mutation is terminal.
    _verify_case_scan_hash(case, phase=f"immediately_before_{pass_name}")
    try:
        return _run_one_case_science(
            case,
            output_root,
            pass_name,
            component_paths,
            recall_model,
            axis_span_model,
            runtime,
            base_case_runner,
        )
    finally:
        _verify_case_scan_hash(case, phase=f"immediately_after_{pass_name}")


def _run_one_case_science(
    case: dict[str, Any],
    output_root: Path,
    pass_name: str,
    component_paths: Mapping[str, Path],
    recall_model: dict[str, Any],
    axis_span_model: dict[str, Any],
    runtime: RuntimeBundle,
    base_case_runner: BaseCaseRunner,
) -> dict[str, Any]:
    case_root = output_root / pass_name / case["case_id"]
    base_failure: tuple[str, str] | None = None
    try:
        record = base_case_runner(
            case,
            output_root,
            pass_name,
            component_paths["v9_model"],
            component_paths["v4_model"],
        )
        if record.get("case_id") != case["case_id"] or record.get("pass") != pass_name:
            raise CustodyIntegrityError(
                "frozen V51 case runner returned mismatched case/pass identity"
            )
        if record.get("scan_sha256") != case["scan_sha256"]:
            raise CustodyIntegrityError(
                "frozen V51 helper record scan SHA-256 does not match the sealed case"
            )
        semantic = _load_case_payload(case_root / "semantic_result.json", name="V51")
        _validate_predictions(semantic, name="V51")
    except CustodyIntegrityError:
        raise
    except Exception as exc:
        base_failure = ("frozen_v51_case_pipeline", type(exc).__name__)
        semantic = _fail_closed_result(
            case,
            arm_id="A_V51",
            failure_stage=base_failure[0],
            failure_type=base_failure[1],
            kind=V51_FAIL_CLOSED_KIND,
        )
        # Overwrite any partial semantic artifact: a failed base case may only
        # contribute an empty abstention to prospective scoring.
        _write_json(case_root / "semantic_result.json", semantic)

    arm_results: dict[str, dict[str, Any]] = {}
    arm_results["A_V51"] = _write_arm_result(case_root, "A_V51", semantic)

    if base_failure is None:
        try:
            v16 = _load_case_payload(case_root / "v16.json", name="V16")
            v43 = _load_case_payload(case_root / "v43.json", name="V43")
            guarded = runtime.compose_v43_explosion_guard(v16, v43, semantic)
        except Exception as exc:
            guarded = _fail_closed_result(
                case,
                arm_id="B_V51_V43_GUARD",
                failure_stage="v43_explosion_guard",
                failure_type=type(exc).__name__,
            )
    else:
        guarded = runtime.compose_v43_explosion_guard(None, None, semantic)
        guarded["pipeline_status"] = "FAILED_ABSTAIN"
        guarded["upstream_failure_stage"] = base_failure[0]
        guarded["upstream_failure_type"] = base_failure[1]
    arm_results["B_V51_V43_GUARD"] = _write_arm_result(
        case_root, "B_V51_V43_GUARD", guarded
    )

    recall_intermediate_path = case_root / "arms" / "arm_c_recall_r3_intermediate.json"
    if base_failure is None:
        try:
            pool = _load_case_payload(
                case_root / "pool" / f"{case['case_id']}.json",
                name="scan-only candidate pool",
            )
            recall_result = runtime.apply_recovery(
                semantic,
                pool,
                recall_model,
                profile_name=runtime.recall_profile,
            )
            _validate_predictions(recall_result, name="C recall intermediate")
            _write_json(recall_intermediate_path, recall_result)
            axis_result = runtime.compose_axis_span_recovery(
                semantic,
                recall_result,
                pool,
                axis_span_model,
            )
        except Exception as exc:
            recall_result = _fail_closed_result(
                case,
                arm_id="C_V51_RECALL_R3_INTERMEDIATE",
                failure_stage="recall_r3_or_axis_span_r2",
                failure_type=type(exc).__name__,
            )
            _write_json(recall_intermediate_path, recall_result)
            axis_result = _fail_closed_result(
                case,
                arm_id="C_V51_RECALL_R3_AXIS_SPAN_R2",
                failure_stage="recall_r3_or_axis_span_r2",
                failure_type=type(exc).__name__,
            )
    else:
        recall_result = _fail_closed_result(
            case,
            arm_id="C_V51_RECALL_R3_INTERMEDIATE",
            failure_stage=base_failure[0],
            failure_type=base_failure[1],
        )
        _write_json(recall_intermediate_path, recall_result)
        axis_result = _fail_closed_result(
            case,
            arm_id="C_V51_RECALL_R3_AXIS_SPAN_R2",
            failure_stage=base_failure[0],
            failure_type=base_failure[1],
        )
    arm_results["C_V51_RECALL_R3_AXIS_SPAN_R2"] = _write_arm_result(
        case_root,
        "C_V51_RECALL_R3_AXIS_SPAN_R2",
        axis_result,
    )
    arm_results["C_V51_RECALL_R3_AXIS_SPAN_R2"][
        "recall_intermediate_sha256"
    ] = _sha(recall_intermediate_path)

    return {
        "case_id": case["case_id"],
        "pass": pass_name,
        "scan_sha256": case["scan_sha256"],
        "base_pipeline_status": "FAILED_ABSTAIN" if base_failure else "SUCCESS",
        "base_failure_stage": None if base_failure is None else base_failure[0],
        "base_failure_type": None if base_failure is None else base_failure[1],
        "arms": arm_results,
    }


def _publish_manifests(
    *,
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    output_root: Path,
    scan_manifest_path: Path,
    freeze_packet_path: Path,
    component_lock: dict[str, Any],
    orchestrator_binding: dict[str, Any],
) -> dict[str, Any]:
    pass_records = {
        pass_name: {
            row["case_id"]: row
            for row in records
            if row["pass"] == pass_name
        }
        for pass_name in ("primary", "replay")
    }
    case_manifest_hashes: dict[str, str] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {arm_id: [] for arm_id in ARM_FILENAMES}
    all_identical = True

    for case in cases:
        case_id = case["case_id"]
        primary = pass_records["primary"][case_id]
        replay = pass_records["replay"][case_id]
        arms: dict[str, Any] = {}
        for arm_id in ARM_FILENAMES:
            left = primary["arms"][arm_id]
            right = replay["arms"][arm_id]
            identical = left["sha256"] == right["sha256"]
            if arm_id == "C_V51_RECALL_R3_AXIS_SPAN_R2":
                identical = identical and (
                    left["recall_intermediate_sha256"]
                    == right["recall_intermediate_sha256"]
                )
            all_identical = all_identical and identical
            arm_case = {
                "case_id": case_id,
                "scan_sha256": case["scan_sha256"],
                "primary_artifact": f"primary/{case_id}/{left['artifact']}",
                "replay_artifact": f"replay/{case_id}/{right['artifact']}",
                "result_sha256": left["sha256"],
                "candidate_count": left["candidate_count"],
                "pipeline_status": left["pipeline_status"],
                "primary_replay_byte_identical": identical,
            }
            if arm_id == "C_V51_RECALL_R3_AXIS_SPAN_R2":
                arm_case["recall_intermediate_sha256"] = left[
                    "recall_intermediate_sha256"
                ]
            arms[arm_id] = arm_case
            arm_rows[arm_id].append(deepcopy(arm_case))

        case_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": CASE_MANIFEST_KIND,
            "case_id": case_id,
            "scan_sha256": case["scan_sha256"],
            "source_only": True,
            "truth_opened": False,
            "labels_used": False,
            "external_counts_used": False,
            "cad_used": False,
            "drawings_used": False,
            "part_identity_used": False,
            "evaluator_inputs_used": False,
            "base_primary_status": primary["base_pipeline_status"],
            "base_replay_status": replay["base_pipeline_status"],
            "arms": arms,
        }
        case_path = output_root / "case_manifests" / f"{case_id}.json"
        _write_json(case_path, case_manifest)
        case_manifest_hashes[case_id] = _sha(case_path)

    arm_manifest_hashes: dict[str, str] = {}
    for arm_id, rows in arm_rows.items():
        arm_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": ARM_MANIFEST_KIND,
            "arm_id": arm_id,
            "source_only": True,
            "truth_opened": False,
            "labels_used": False,
            "external_counts_used": False,
            "cad_used": False,
            "drawings_used": False,
            "part_identity_used": False,
            "evaluator_inputs_used": False,
            "case_count": len(rows),
            "all_primary_replay_byte_identical": all(
                row["primary_replay_byte_identical"] for row in rows
            ),
            "cases": rows,
        }
        arm_path = output_root / "arm_manifests" / f"{arm_id.lower()}.json"
        _write_json(arm_path, arm_manifest)
        arm_manifest_hashes[arm_id] = _sha(arm_path)

    if not all_identical:
        raise RuntimeError("primary/replay bytes differ for at least one case/arm")
    if _sha(_resolve_locked_component(ORCHESTRATOR_RELATIVE_PATH)) != orchestrator_binding[
        "orchestrator_sha256"
    ]:
        raise RuntimeError("prospective orchestrator bytes changed during execution")
    if _sha(freeze_packet_path) != orchestrator_binding["freeze_packet_sha256"]:
        raise RuntimeError("freeze packet bytes changed during execution")

    execution = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXECUTION_KIND,
        "status": "PROSPECTIVE_PREDICTION_ONLY_COMPLETE",
        "research_only": True,
        "source_only": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "truth_opened": False,
        "scoring_opened": False,
        "labels_used": False,
        "external_counts_used": False,
        "cad_used": False,
        "drawings_used": False,
        "part_identity_used": False,
        "evaluator_inputs_used": False,
        "embedded_geometry_metadata_sanitized": False,
        "raw_byte_custody_assumption": (
            "the custodian copied eligible delivery geometry bytes unchanged; "
            "the orchestrator does not inspect or use embedded PLY/STL metadata"
        ),
        "retuning_permitted_after_prediction": False,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "all_arms_primary_replay_byte_identical": True,
        "input_manifest_sha256": _sha(scan_manifest_path),
        "freeze_packet_sha256": _sha(freeze_packet_path),
        "component_lock_sha256": _sha_bytes(_canonical_bytes(component_lock)),
        "component_lock": component_lock,
        "sealed_artifact_binding_count": SEALED_ARTIFACT_BINDING_COUNT,
        "freeze_inventory_validation_phases": [
            "before_scientific_import",
            "immediately_before_primary",
            "immediately_before_replay",
            "after_primary_and_replay",
        ],
        "orchestrator_binding": orchestrator_binding,
        "orchestrator_binding_sha256": _sha(
            output_root / "pre_inference_orchestrator_binding.json"
        ),
        "case_manifest_sha256": case_manifest_hashes,
        "arm_manifest_sha256": arm_manifest_hashes,
        "runtime_provenance": {
            "executed_case_runtime": FROZEN_V51_HELPER_RELATIVE_PATH,
            "lineage_bound_scan_only_wrapper": COMPONENT_RELATIVE_PATHS[
                "v51_case_runner"
            ],
            "case_runtime_mode": "direct_frozen_helper_execution",
            "lineage_note": (
                "the locked scan-only wrapper remains the lineage contract; "
                "this orchestrator calls its exact frozen helper directly"
            ),
        },
    }
    execution_path = output_root / "prospective_execution_manifest.json"
    _write_json(execution_path, execution)
    return execution


def _run_prospective_core(
    scan_manifest_path: Path,
    freeze_packet_path: Path,
    output_root: Path,
    *,
    max_workers: int = 1,
    _base_case_runner: BaseCaseRunner | None = None,
    _test_allow_preloaded_scientific_modules: bool = False,
) -> dict[str, Any]:
    """Execute all three arms twice and publish truth-free manifests."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not _test_allow_preloaded_scientific_modules:
        _assert_no_preloaded_scientific_modules()
    scan_manifest_path = scan_manifest_path.resolve()
    freeze_packet_path = freeze_packet_path.resolve()
    output_root = output_root.resolve()

    # Component integrity and model contracts are checked before scans are
    # validated/opened.  A stale or mutated freeze therefore fails closed.
    component_lock, component_paths = validate_freeze_packet(freeze_packet_path)
    runtime = _load_runtime_bundle(component_paths)
    recall_model = _load_case_payload(component_paths["recall_model"], name="recall model")
    axis_span_model = _load_case_payload(
        component_paths["axis_span_model"], name="axis/span model"
    )
    runtime.validate_recall_model(recall_model, runtime.recall_profile)
    runtime.validate_axis_span_model(axis_span_model)

    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise FileExistsError("prospective output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    orchestrator_binding = _build_orchestrator_binding(
        freeze_packet_path,
        component_lock,
    )
    _write_json(
        output_root / "pre_inference_orchestrator_binding.json",
        orchestrator_binding,
    )
    cases = validate_scan_manifest(scan_manifest_path)
    runner = _base_case_runner or runtime.frozen_v51_helper._run_case
    records: list[dict[str, Any]] = []

    with _suppress_nondeterministic_frozen_events(runtime.frozen_v51_helper):
        for pass_name in ("primary", "replay"):
            _revalidate_freeze_unchanged(
                freeze_packet_path,
                component_lock,
                component_paths,
            )
            if max_workers == 1:
                records.extend(
                    _run_one_case(
                        case,
                        output_root,
                        pass_name,
                        component_paths,
                        recall_model,
                        axis_span_model,
                        runtime,
                        runner,
                    )
                    for case in cases
                )
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            _run_one_case,
                            case,
                            output_root,
                            pass_name,
                            component_paths,
                            recall_model,
                            axis_span_model,
                            runtime,
                            runner,
                        )
                        for case in cases
                    ]
                    records.extend(future.result() for future in futures)

    _revalidate_freeze_unchanged(
        freeze_packet_path,
        component_lock,
        component_paths,
    )

    return _publish_manifests(
        cases=cases,
        records=records,
        output_root=output_root,
        scan_manifest_path=scan_manifest_path,
        freeze_packet_path=freeze_packet_path,
        component_lock=component_lock,
        orchestrator_binding=orchestrator_binding,
    )


def run_prospective(
    scan_manifest_path: Path,
    freeze_packet_path: Path,
    output_root: Path,
    *,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Authoritative fresh-process prospective entrypoint (no runner override)."""

    return _run_prospective_core(
        scan_manifest_path,
        freeze_packet_path,
        output_root,
        max_workers=max_workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--freeze-packet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    result = run_prospective(
        args.scan_manifest,
        args.freeze_packet,
        args.output_root,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "all_arms_primary_replay_byte_identical": result[
                    "all_arms_primary_replay_byte_identical"
                ],
                "manifest": str(
                    args.output_root.resolve() / "prospective_execution_manifest.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
