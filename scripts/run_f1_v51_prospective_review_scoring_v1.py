#!/usr/bin/env python3
"""Seal prediction-hidden review inputs, then score the frozen three-arm run.

This module is deliberately split into two commands:

``prepare-review``
    Revalidates the sealed prospective execution and every primary/replay byte
    identity.  It publishes a custodian-only execution seal plus a separate
    raw-scan-only render manifest.  The render manifest has no arm, artifact,
    candidate, or predicted-line field.

``score``
    Requires the execution seal, render manifest, and a byte-sealed catalog
    containing at least three independent high-performance reviews and two
    prediction-hidden adjudicators per case.  Only after the complete catalog
    and SHA-256 sidecar validate are primary prediction JSON files parsed.
    Every case is scored in all three arms with the frozen canonical scorer at
    exactly 5 degrees, 4 mm, 0.60 gold overlap, and 0.60 prediction overlap.

There are no threshold, case-selection, exclusion, or retuning arguments.
Failed inference cases remain in the denominator through the runner's empty
fail-closed prediction artifact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = 1
SCAN_INPUT_KIND = "sherman_qc_f1_v51_opaque_scan_only_development_input"
FREEZE_KIND = "sherman_qc_f1_v51_20260831_prospective_three_arm_freeze_v1"
EXECUTION_KIND = "sherman_qc_f1_v51_three_arm_prospective_execution_v1"
CASE_MANIFEST_KIND = "sherman_qc_f1_v51_three_arm_case_manifest_v1"
ARM_MANIFEST_KIND = "sherman_qc_f1_v51_three_arm_arm_manifest_v1"
EXECUTION_SEAL_KIND = "sherman_qc_f1_v51_three_arm_execution_identity_seal_v1"
RENDER_MANIFEST_KIND = "sherman_qc_f1_opaque_raw_scan_render_manifest_v1"
LABEL_CATALOG_KIND = "sherman_qc_f1_prediction_hidden_multi_review_catalog_v1"
SCORE_KIND = "sherman_qc_f1_v51_prospective_three_arm_score_v1"

SEALED_FREEZE_PACKET_SHA256 = (
    "8dfe45647ad251d8887fc04ffce1e5bb5079f655f85fd4da0ce65a8b49f20a59"
)
SCORER_RELATIVE_PATH = "scripts/evaluate_f1_against_manual_gold.py"
ORCHESTRATOR_RELATIVE_PATH = "scripts/run_f1_v51_three_arm_prospective_v1.py"
OPAQUE_SCAN_ROOT = (
    ROOT / "output/research/f1-v51-20260831-prospective-intake/opaque_scans"
).resolve()

ARM_FILENAMES: dict[str, str] = {
    "A_V51": "arm_a_v51.json",
    "B_V51_V43_GUARD": "arm_b_v51_v43_guard.json",
    "C_V51_RECALL_R3_AXIS_SPAN_R2": "arm_c_v51_recall_r3_axis_span_r2.json",
}
ARM_IDS = tuple(ARM_FILENAMES)

CANONICAL_THRESHOLDS = {
    "axis_angle_deg": 5.0,
    "transverse_offset_mm": 4.0,
    "gold_span_overlap": 0.60,
    "prediction_span_overlap": 0.60,
    "matching": "one_to_one_minimum_cost",
}

SCAN_TOP_LEVEL_KEYS = frozenset({"schema_version", "kind", "cases"})
SCAN_CASE_KEYS = frozenset({"scan_path", "scan_sha256"})
CASE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "case_id",
        "scan_sha256",
        "source_only",
        "truth_opened",
        "labels_used",
        "external_counts_used",
        "cad_used",
        "drawings_used",
        "part_identity_used",
        "evaluator_inputs_used",
        "base_primary_status",
        "base_replay_status",
        "arms",
    }
)
ARM_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "arm_id",
        "source_only",
        "truth_opened",
        "labels_used",
        "external_counts_used",
        "cad_used",
        "drawings_used",
        "part_identity_used",
        "evaluator_inputs_used",
        "case_count",
        "all_primary_replay_byte_identical",
        "cases",
    }
)
ARM_CASE_KEYS = frozenset(
    {
        "case_id",
        "scan_sha256",
        "primary_artifact",
        "replay_artifact",
        "result_sha256",
        "candidate_count",
        "pipeline_status",
        "primary_replay_byte_identical",
    }
)
C_ARM_CASE_KEYS = ARM_CASE_KEYS | {"recall_intermediate_sha256"}

RENDER_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "cohort_date",
        "source_scan_manifest_sha256",
        "execution_seal_sha256",
        "custody",
        "cases",
    }
)
RENDER_CASE_KEYS = frozenset({"case_id", "scan_path", "scan_sha256"})
RENDER_CUSTODY = {
    "raw_scan_geometry_only": True,
    "visual_review_blind": True,
    "labels_included": False,
    "expected_bend_total_included": False,
    "cad_included": False,
    "drawings_included": False,
    "part_identity_included": False,
    "source_filenames_included": False,
}

LABEL_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "cohort_date",
        "render_manifest_sha256",
        "execution_seal_sha256",
        "freeze_packet_sha256",
        "review_protocol",
        "cases",
    }
)
LABEL_CASE_KEYS = frozenset({"case_id", "scan_sha256", "reviews", "reconciliation"})
REVIEW_KEYS = frozenset(
    {"reviewer_id", "reviewer_class", "independent", "manual_bends"}
)
RECONCILIATION_KEYS = frozenset(
    {
        "status",
        "blind_to_algorithm_outputs",
        "source_reviewer_ids",
        "adjudicator_ids",
        "manual_bends",
    }
)
BEND_KEYS = frozenset({"bend_id", "line_start", "line_end"})
REVIEW_PROTOCOL = {
    "blind_to_algorithm_outputs": True,
    "independent_review_minimum": 3,
    "adjudicator_minimum": 2,
    "reconciliation_complete": True,
    "sealed_before_scoring": True,
    "allowed_inputs": ["opaque raw scan geometry", "truth-free multi-view renders"],
    "forbidden_inputs": [
        "algorithm outputs",
        "candidate pools",
        "expected bend totals",
        "CAD",
        "drawings",
        "part identity",
    ],
}

_CASE_ID_RE = re.compile(r"T[0-9]{3}")
_REVIEWER_ID_RE = re.compile(r"R[0-9]{3}")
_ADJUDICATOR_ID_RE = re.compile(r"A[0-9]{3}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class CustodyError(RuntimeError):
    """The prospective evidence chain does not satisfy the frozen custody contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.fullmatch(value))


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _write_sidecar(path: Path) -> str:
    digest = _sha(path)
    _sidecar(path).write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _validate_sidecar(path: Path) -> str:
    sidecar = _sidecar(path)
    if not sidecar.is_file():
        raise CustodyError(f"sealed SHA-256 sidecar is missing: {sidecar.name}")
    expected = f"{_sha(path)}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise CustodyError(f"sealed SHA-256 sidecar mismatch: {sidecar.name}")
    return expected[:64]


def _repo_path(relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or relative_path != pure.as_posix()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise CustodyError(f"noncanonical repository path: {relative_path}")
    resolved = (ROOT / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CustodyError(f"repository path escapes root: {relative_path}") from exc
    return resolved


def _under(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or relative_path != pure.as_posix()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise CustodyError(f"noncanonical execution artifact path: {relative_path}")
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CustodyError(f"execution artifact escapes root: {relative_path}") from exc
    return resolved


def _finite_line(row: Any, *, name: str, expected_id: str | None = None) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != BEND_KEYS:
        raise ValueError(f"{name} bend must contain only bend_id, line_start, and line_end")
    bend_id = row.get("bend_id")
    if not isinstance(bend_id, str) or not bend_id:
        raise ValueError(f"{name} bend_id must be non-empty")
    if expected_id is not None and bend_id != expected_id:
        raise ValueError(f"{name} bend IDs must be canonical and sequential")
    start, end = row.get("line_start"), row.get("line_end")
    if not (
        isinstance(start, list)
        and isinstance(end, list)
        and len(start) == len(end) == 3
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (*start, *end))
    ):
        raise ValueError(f"{name} bend endpoints must be numeric 3-vectors")
    coordinates = [float(value) for value in (*start, *end)]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"{name} bend endpoints must be finite")
    if math.dist(coordinates[:3], coordinates[3:]) <= 1e-9:
        raise ValueError(f"{name} bend line is degenerate")
    return deepcopy(row)


def validate_scan_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != SCAN_TOP_LEVEL_KEYS:
        raise ValueError("scan manifest may contain only schema_version, kind, and cases")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != SCAN_INPUT_KIND:
        raise ValueError("scan manifest schema/kind mismatch")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan manifest cases must be a non-empty list")
    result: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        case_id = f"T{ordinal:03d}"
        if not isinstance(row, dict) or set(row) != SCAN_CASE_KEYS:
            raise ValueError(f"{case_id}: scan row has forbidden fields")
        if not _valid_sha(row.get("scan_sha256")):
            raise ValueError(f"{case_id}: scan_sha256 is invalid")
        if not isinstance(row.get("scan_path"), str):
            raise ValueError(f"{case_id}: scan_path is invalid")
        scan = Path(row["scan_path"]).expanduser().resolve()
        if scan.parent != OPAQUE_SCAN_ROOT or scan.name not in {f"{case_id}.ply", f"{case_id}.stl"}:
            raise CustodyError(f"{case_id}: scan is outside the sealed opaque namespace")
        if not scan.is_file() or _sha(scan) != row["scan_sha256"]:
            raise CustodyError(f"{case_id}: scan bytes do not match the sealed manifest")
        result.append(
            {"case_id": case_id, "scan_path": str(scan), "scan_sha256": row["scan_sha256"]}
        )
    return result


def validate_freeze_packet(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if _sha(path) != SEALED_FREEZE_PACKET_SHA256:
        raise CustodyError("prospective freeze packet does not match the hard-pinned seal")
    packet = _read_json(path)
    if not isinstance(packet, dict) or packet.get("kind") != FREEZE_KIND:
        raise CustodyError("prospective freeze packet kind mismatch")
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise CustodyError("prospective freeze packet schema mismatch")
    if packet.get("status") != "SEALED_BEFORE_2026_08_31_DELIVERY_DISCOVERY":
        raise CustodyError("prospective freeze was not sealed before discovery")
    if packet.get("scoring", {}).get("thresholds") != {
        "axis_angle_deg_max": 5.0,
        "gold_span_overlap_min": 0.6,
        "prediction_span_overlap_min": 0.6,
        "transverse_offset_mm_max": 4.0,
    }:
        raise CustodyError("frozen scoring thresholds changed")
    if packet.get("scoring", {}).get("matching") != "one_to_one_minimum_cost":
        raise CustodyError("frozen matching rule changed")
    if packet.get("decision_rules", {}).get("retuning_after_freeze") != "forbidden":
        raise CustodyError("freeze does not forbid retuning")
    if packet.get("decision_rules", {}).get("case_removal_after_freeze") != "forbidden":
        raise CustodyError("freeze does not forbid case removal")
    scorer_lock = packet.get("component_lock", {}).get("scorer")
    if not isinstance(scorer_lock, dict) or scorer_lock.get("path") != SCORER_RELATIVE_PATH:
        raise CustodyError("canonical scorer path is not frozen")
    scorer = _repo_path(SCORER_RELATIVE_PATH)
    if not _valid_sha(scorer_lock.get("sha256")) or _sha(scorer) != scorer_lock["sha256"]:
        raise CustodyError("canonical scorer bytes do not match the freeze")
    return packet


def _require_truth_free_flags(payload: Mapping[str, Any], *, name: str) -> None:
    if payload.get("source_only") is not True:
        raise CustodyError(f"{name} is not source-only")
    for key in (
        "truth_opened",
        "labels_used",
        "external_counts_used",
        "cad_used",
        "drawings_used",
        "part_identity_used",
        "evaluator_inputs_used",
    ):
        if payload.get(key) is not False:
            raise CustodyError(f"{name} violates truth-free flag {key}")


def _validate_arm_case_row(
    row: Any,
    *,
    arm_id: str,
    case: Mapping[str, Any],
    execution_root: Path,
) -> dict[str, Any]:
    expected_keys = C_ARM_CASE_KEYS if arm_id == ARM_IDS[2] else ARM_CASE_KEYS
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise CustodyError(f"{case['case_id']} {arm_id}: arm case manifest fields changed")
    if row.get("case_id") != case["case_id"] or row.get("scan_sha256") != case["scan_sha256"]:
        raise CustodyError(f"{case['case_id']} {arm_id}: case/scan binding mismatch")
    if row.get("primary_replay_byte_identical") is not True:
        raise CustodyError(f"{case['case_id']} {arm_id}: replay identity is false")
    if not _valid_sha(row.get("result_sha256")):
        raise CustodyError(f"{case['case_id']} {arm_id}: result SHA-256 is invalid")
    if not isinstance(row.get("candidate_count"), int) or isinstance(row["candidate_count"], bool) or row["candidate_count"] < 0:
        raise CustodyError(f"{case['case_id']} {arm_id}: candidate count is invalid")
    expected_primary = f"primary/{case['case_id']}/arms/{ARM_FILENAMES[arm_id]}"
    expected_replay = f"replay/{case['case_id']}/arms/{ARM_FILENAMES[arm_id]}"
    if row.get("primary_artifact") != expected_primary or row.get("replay_artifact") != expected_replay:
        raise CustodyError(f"{case['case_id']} {arm_id}: artifact path is not canonical")
    primary = _under(execution_root, expected_primary)
    replay = _under(execution_root, expected_replay)
    if not primary.is_file() or not replay.is_file():
        raise CustodyError(f"{case['case_id']} {arm_id}: primary or replay artifact is missing")
    primary_sha, replay_sha = _sha(primary), _sha(replay)
    if primary_sha != replay_sha or primary_sha != row["result_sha256"]:
        raise CustodyError(f"{case['case_id']} {arm_id}: primary/replay byte identity failed")
    if arm_id == ARM_IDS[2]:
        intermediate_sha = row.get("recall_intermediate_sha256")
        if not _valid_sha(intermediate_sha):
            raise CustodyError(f"{case['case_id']} {arm_id}: recall intermediate SHA is invalid")
        primary_intermediate = _under(
            execution_root,
            f"primary/{case['case_id']}/arms/arm_c_recall_r3_intermediate.json",
        )
        replay_intermediate = _under(
            execution_root,
            f"replay/{case['case_id']}/arms/arm_c_recall_r3_intermediate.json",
        )
        if not primary_intermediate.is_file() or not replay_intermediate.is_file():
            raise CustodyError(f"{case['case_id']} {arm_id}: recall intermediate is missing")
        if _sha(primary_intermediate) != intermediate_sha or _sha(replay_intermediate) != intermediate_sha:
            raise CustodyError(f"{case['case_id']} {arm_id}: recall replay identity failed")
    return deepcopy(row)


def validate_execution_identity(
    scan_manifest_path: Path,
    execution_root: Path,
    freeze_packet_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Hash-validate all evidence without parsing any prediction result JSON."""

    scan_manifest_path = scan_manifest_path.resolve()
    execution_root = execution_root.resolve()
    freeze_packet_path = freeze_packet_path.resolve()
    freeze = validate_freeze_packet(freeze_packet_path)
    cases = validate_scan_manifest(scan_manifest_path)
    execution_path = execution_root / "prospective_execution_manifest.json"
    execution = _read_json(execution_path)
    if not isinstance(execution, dict) or execution.get("kind") != EXECUTION_KIND:
        raise CustodyError("prospective execution manifest kind mismatch")
    if execution.get("schema_version") != SCHEMA_VERSION:
        raise CustodyError("prospective execution manifest schema mismatch")
    if execution.get("status") != "PROSPECTIVE_PREDICTION_ONLY_COMPLETE":
        raise CustodyError("prospective execution is not complete")
    _require_truth_free_flags(execution, name="prospective execution")
    if execution.get("scoring_opened") is not False or execution.get("research_only") is not True:
        raise CustodyError("prospective execution scoring/research state is invalid")
    if execution.get("all_arms_primary_replay_byte_identical") is not True:
        raise CustodyError("prospective execution did not seal all replay identities")
    case_ids = [case["case_id"] for case in cases]
    if execution.get("case_ids") != case_ids or execution.get("case_count") != len(cases):
        raise CustodyError("prospective execution has a case omission or reordering")
    if execution.get("input_manifest_sha256") != _sha(scan_manifest_path):
        raise CustodyError("prospective execution is bound to a different scan manifest")
    if execution.get("freeze_packet_sha256") != _sha(freeze_packet_path):
        raise CustodyError("prospective execution is bound to a different freeze")
    frozen_lock = freeze.get("component_lock")
    if execution.get("component_lock") != frozen_lock:
        raise CustodyError("prospective execution component lock differs from the freeze")
    if execution.get("component_lock_sha256") != _sha_bytes(_canonical_bytes(frozen_lock)):
        raise CustodyError("prospective execution component lock digest mismatch")
    binding = execution.get("orchestrator_binding")
    if not isinstance(binding, dict):
        raise CustodyError("prospective orchestrator binding is missing")
    orchestrator = _repo_path(ORCHESTRATOR_RELATIVE_PATH)
    if binding.get("orchestrator_path") != ORCHESTRATOR_RELATIVE_PATH or binding.get("orchestrator_sha256") != _sha(orchestrator):
        raise CustodyError("prospective orchestrator bytes do not match its pre-inference binding")
    if binding.get("bound_before_scan_manifest_open") is not True:
        raise CustodyError("orchestrator was not bound before scan discovery")

    case_hashes = execution.get("case_manifest_sha256")
    arm_hashes = execution.get("arm_manifest_sha256")
    if not isinstance(case_hashes, dict) or set(case_hashes) != set(case_ids):
        raise CustodyError("execution case-manifest inventory is incomplete")
    if not isinstance(arm_hashes, dict) or set(arm_hashes) != set(ARM_IDS):
        raise CustodyError("execution arm-manifest inventory is incomplete")

    case_arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        case_id = case["case_id"]
        path = execution_root / "case_manifests" / f"{case_id}.json"
        if not path.is_file() or not _valid_sha(case_hashes[case_id]) or _sha(path) != case_hashes[case_id]:
            raise CustodyError(f"{case_id}: case manifest hash mismatch")
        payload = _read_json(path)
        if not isinstance(payload, dict) or set(payload) != CASE_MANIFEST_KEYS:
            raise CustodyError(f"{case_id}: case manifest fields changed")
        if payload.get("kind") != CASE_MANIFEST_KIND or payload.get("schema_version") != SCHEMA_VERSION:
            raise CustodyError(f"{case_id}: case manifest schema/kind mismatch")
        _require_truth_free_flags(payload, name=f"{case_id} case manifest")
        if payload.get("case_id") != case_id or payload.get("scan_sha256") != case["scan_sha256"]:
            raise CustodyError(f"{case_id}: case manifest identity mismatch")
        arm_rows = payload.get("arms")
        if not isinstance(arm_rows, dict) or set(arm_rows) != set(ARM_IDS):
            raise CustodyError(f"{case_id}: case manifest arm inventory is incomplete")
        case_arm_rows[case_id] = {
            arm_id: _validate_arm_case_row(
                arm_rows[arm_id], arm_id=arm_id, case=case, execution_root=execution_root
            )
            for arm_id in ARM_IDS
        }

    for arm_id in ARM_IDS:
        path = execution_root / "arm_manifests" / f"{arm_id.lower()}.json"
        if not path.is_file() or not _valid_sha(arm_hashes[arm_id]) or _sha(path) != arm_hashes[arm_id]:
            raise CustodyError(f"{arm_id}: arm manifest hash mismatch")
        payload = _read_json(path)
        if not isinstance(payload, dict) or set(payload) != ARM_MANIFEST_KEYS:
            raise CustodyError(f"{arm_id}: arm manifest fields changed")
        if payload.get("kind") != ARM_MANIFEST_KIND or payload.get("schema_version") != SCHEMA_VERSION:
            raise CustodyError(f"{arm_id}: arm manifest schema/kind mismatch")
        _require_truth_free_flags(payload, name=f"{arm_id} arm manifest")
        if payload.get("arm_id") != arm_id or payload.get("case_count") != len(cases):
            raise CustodyError(f"{arm_id}: arm manifest case inventory mismatch")
        if payload.get("all_primary_replay_byte_identical") is not True:
            raise CustodyError(f"{arm_id}: arm manifest replay identity is false")
        expected_rows = [case_arm_rows[case["case_id"]][arm_id] for case in cases]
        if payload.get("cases") != expected_rows:
            raise CustodyError(f"{arm_id}: arm and case manifests disagree")

    seal_cases = []
    for case in cases:
        arms = {}
        for arm_id in ARM_IDS:
            row = case_arm_rows[case["case_id"]][arm_id]
            arms[arm_id] = {
                "primary_artifact": row["primary_artifact"],
                "replay_artifact": row["replay_artifact"],
                "result_sha256": row["result_sha256"],
                "primary_replay_byte_identical": True,
            }
            if arm_id == ARM_IDS[2]:
                arms[arm_id]["recall_intermediate_sha256"] = row[
                    "recall_intermediate_sha256"
                ]
        seal_cases.append(
            {"case_id": case["case_id"], "scan_sha256": case["scan_sha256"], "arms": arms}
        )
    identity = {
        "execution_manifest_sha256": _sha(execution_path),
        "cases": seal_cases,
    }
    return cases, execution, identity


def prepare_prediction_hidden_review(
    scan_manifest_path: Path,
    execution_root: Path,
    freeze_packet_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite review namespace: {output_root}")
    cases, execution, identity = validate_execution_identity(
        scan_manifest_path, execution_root, freeze_packet_path
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        seal = {
            "schema_version": SCHEMA_VERSION,
            "kind": EXECUTION_SEAL_KIND,
            "status": "ALL_THREE_ARMS_PRIMARY_REPLAY_IDENTICAL",
            "freeze_packet_sha256": _sha(freeze_packet_path.resolve()),
            "source_scan_manifest_sha256": _sha(scan_manifest_path.resolve()),
            "prospective_execution_manifest_sha256": identity[
                "execution_manifest_sha256"
            ],
            "all_cases_included": True,
            "all_arms_primary_replay_byte_identical": True,
            "scoring_opened": False,
            "labels_opened": False,
            "cases": identity["cases"],
        }
        seal_path = staging / "execution_identity_seal.json"
        _write_json(seal_path, seal)
        seal_sha = _write_sidecar(seal_path)

        render_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": RENDER_MANIFEST_KIND,
            "status": "SEALED_RAW_SCAN_ONLY_REVIEW_READY",
            "cohort_date": "2026-08-31",
            "source_scan_manifest_sha256": _sha(scan_manifest_path.resolve()),
            "execution_seal_sha256": seal_sha,
            "custody": deepcopy(RENDER_CUSTODY),
            "cases": [
                {
                    "case_id": case["case_id"],
                    "scan_path": case["scan_path"],
                    "scan_sha256": case["scan_sha256"],
                }
                for case in cases
            ],
        }
        render_path = staging / "raw_scan_only_render_manifest.json"
        _write_json(render_path, render_manifest)
        render_sha = _write_sidecar(render_path)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "execution_identity_seal": str(output_root / seal_path.name),
        "execution_identity_seal_sha256": seal_sha,
        "raw_scan_only_render_manifest": str(output_root / render_path.name),
        "raw_scan_only_render_manifest_sha256": render_sha,
        "all_arms_primary_replay_byte_identical": execution[
            "all_arms_primary_replay_byte_identical"
        ],
    }


def _validate_execution_seal(
    path: Path,
    *,
    cases: list[dict[str, Any]],
    identity: Mapping[str, Any],
    freeze_packet_path: Path,
    scan_manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    digest = _validate_sidecar(path)
    payload = _read_json(path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXECUTION_SEAL_KIND,
        "status": "ALL_THREE_ARMS_PRIMARY_REPLAY_IDENTICAL",
        "freeze_packet_sha256": _sha(freeze_packet_path),
        "source_scan_manifest_sha256": _sha(scan_manifest_path),
        "prospective_execution_manifest_sha256": identity["execution_manifest_sha256"],
        "all_cases_included": True,
        "all_arms_primary_replay_byte_identical": True,
        "scoring_opened": False,
        "labels_opened": False,
        "cases": identity["cases"],
    }
    if payload != expected or len(payload["cases"]) != len(cases):
        raise CustodyError("execution identity seal differs from current sealed evidence")
    return payload, digest


def _validate_render_manifest(
    path: Path,
    *,
    cases: list[dict[str, Any]],
    execution_seal_sha256: str,
    scan_manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    digest = _validate_sidecar(path)
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != RENDER_TOP_LEVEL_KEYS:
        raise CustodyError("raw-scan render manifest fields changed")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != RENDER_MANIFEST_KIND:
        raise CustodyError("raw-scan render manifest schema/kind mismatch")
    if payload.get("status") != "SEALED_RAW_SCAN_ONLY_REVIEW_READY":
        raise CustodyError("raw-scan render manifest is not sealed")
    if payload.get("cohort_date") != "2026-08-31":
        raise CustodyError("raw-scan render manifest cohort date mismatch")
    if payload.get("source_scan_manifest_sha256") != _sha(scan_manifest_path):
        raise CustodyError("raw-scan render manifest source binding mismatch")
    if payload.get("execution_seal_sha256") != execution_seal_sha256:
        raise CustodyError("raw-scan render manifest execution-seal binding mismatch")
    if payload.get("custody") != RENDER_CUSTODY:
        raise CustodyError("raw-scan render custody declaration changed")
    expected_rows = [
        {"case_id": case["case_id"], "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"]}
        for case in cases
    ]
    rows = payload.get("cases")
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != RENDER_CASE_KEYS for row in rows):
        raise CustodyError("raw-scan render cases contain a forbidden field")
    if rows != expected_rows:
        raise CustodyError("raw-scan render manifest omits, adds, or reorders a case")
    return payload, digest


def validate_sealed_label_catalog(
    path: Path,
    *,
    cases: list[dict[str, Any]],
    render_manifest_sha256: str,
    execution_seal_sha256: str,
    freeze_packet_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Validate the entire sealed catalog before returning any gold geometry."""

    digest = _validate_sidecar(path)
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != LABEL_TOP_LEVEL_KEYS:
        raise CustodyError("label catalog has forbidden or missing top-level fields")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != LABEL_CATALOG_KIND:
        raise CustodyError("label catalog schema/kind mismatch")
    if payload.get("status") != "SEALED_BEFORE_SCORING" or payload.get("cohort_date") != "2026-08-31":
        raise CustodyError("label catalog is not sealed for the prospective cohort")
    if payload.get("render_manifest_sha256") != render_manifest_sha256:
        raise CustodyError("label catalog is bound to a different render manifest")
    if payload.get("execution_seal_sha256") != execution_seal_sha256:
        raise CustodyError("label catalog is bound to a different execution seal")
    if payload.get("freeze_packet_sha256") != freeze_packet_sha256:
        raise CustodyError("label catalog is bound to a different freeze")
    if payload.get("review_protocol") != REVIEW_PROTOCOL:
        raise CustodyError("label catalog review protocol changed")
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise CustodyError("label catalog must include every prospective case exactly once")

    gold: dict[str, list[dict[str, Any]]] = {}
    for case, row in zip(cases, rows):
        case_id = case["case_id"]
        if not isinstance(row, dict) or set(row) != LABEL_CASE_KEYS:
            raise CustodyError(f"{case_id}: label case has forbidden or missing fields")
        if row.get("case_id") != case_id or row.get("scan_sha256") != case["scan_sha256"]:
            raise CustodyError(f"{case_id}: label case/scan binding mismatch")
        reviews = row.get("reviews")
        if not isinstance(reviews, list) or len(reviews) < REVIEW_PROTOCOL["independent_review_minimum"]:
            raise CustodyError(f"{case_id}: fewer than three independent reviews")
        reviewer_ids: list[str] = []
        for review_index, review in enumerate(reviews, start=1):
            if not isinstance(review, dict) or set(review) != REVIEW_KEYS:
                raise CustodyError(f"{case_id}: review has forbidden or missing fields")
            reviewer_id = review.get("reviewer_id")
            if not isinstance(reviewer_id, str) or not _REVIEWER_ID_RE.fullmatch(reviewer_id):
                raise CustodyError(f"{case_id}: reviewer ID is not opaque")
            if reviewer_id in reviewer_ids:
                raise CustodyError(f"{case_id}: duplicate reviewer")
            reviewer_ids.append(reviewer_id)
            if review.get("reviewer_class") != "high_performance_agent" or review.get("independent") is not True:
                raise CustodyError(f"{case_id}: reviewer is not an independent high-performance agent")
            bends = review.get("manual_bends")
            if not isinstance(bends, list):
                raise CustodyError(f"{case_id}: review bends must be a list")
            for bend_index, bend in enumerate(bends, start=1):
                _finite_line(
                    bend,
                    name=f"{case_id} {reviewer_id}",
                    expected_id=f"{reviewer_id}-B{bend_index:03d}",
                )

        reconciliation = row.get("reconciliation")
        if not isinstance(reconciliation, dict) or set(reconciliation) != RECONCILIATION_KEYS:
            raise CustodyError(f"{case_id}: reconciliation has forbidden or missing fields")
        if reconciliation.get("status") != "SEALED" or reconciliation.get("blind_to_algorithm_outputs") is not True:
            raise CustodyError(f"{case_id}: reconciliation is not prediction-hidden and sealed")
        if reconciliation.get("source_reviewer_ids") != reviewer_ids:
            raise CustodyError(f"{case_id}: reconciliation does not bind every review")
        adjudicators = reconciliation.get("adjudicator_ids")
        if (
            not isinstance(adjudicators, list)
            or len(adjudicators) < REVIEW_PROTOCOL["adjudicator_minimum"]
            or len(set(adjudicators)) != len(adjudicators)
            or any(not isinstance(value, str) or not _ADJUDICATOR_ID_RE.fullmatch(value) for value in adjudicators)
        ):
            raise CustodyError(f"{case_id}: at least two opaque adjudicators are required")
        final_bends = reconciliation.get("manual_bends")
        if not isinstance(final_bends, list):
            raise CustodyError(f"{case_id}: reconciled bends must be a list")
        gold[case_id] = [
            _finite_line(
                bend,
                name=f"{case_id} reconciliation",
                expected_id=f"{case_id}-B{bend_index:03d}",
            )
            for bend_index, bend in enumerate(final_bends, start=1)
        ]
    return gold, digest


def _load_canonical_scorer(freeze: Mapping[str, Any]) -> ModuleType:
    scorer_path = _repo_path(SCORER_RELATIVE_PATH)
    expected_sha = freeze["component_lock"]["scorer"]["sha256"]
    if _sha(scorer_path) != expected_sha:
        raise CustodyError("canonical scorer changed before scoring")
    module = importlib.import_module("scripts.evaluate_f1_against_manual_gold")
    if Path(str(getattr(module, "__file__", ""))).resolve() != scorer_path:
        raise CustodyError("canonical scorer imported from an unexpected path")
    return module


def _load_prediction_records(path: Path, *, case_id: str, arm_id: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise CustodyError(f"{case_id} {arm_id}: prediction result is not an object")
    if payload.get("source_only") is not True or payload.get("research_only") is not True:
        raise CustodyError(f"{case_id} {arm_id}: prediction result is not source-only research output")
    for key in ("labels_used", "cad_used", "drawing_used", "drawings_used", "part_identity_used", "truth_opened", "scoring_opened"):
        if payload.get(key) not in (None, False):
            raise CustodyError(f"{case_id} {arm_id}: forbidden inference input reported: {key}")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise CustodyError(f"{case_id} {arm_id}: predictions must be a list")
    canonical = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise CustodyError(f"{case_id} {arm_id}: prediction row is not an object")
        geometry = {
            "bend_id": f"{arm_id}-P{index:03d}",
            "line_start": row.get("line_start"),
            "line_end": row.get("line_end"),
        }
        canonical.append(_finite_line(geometry, name=f"{case_id} {arm_id}"))
    if payload.get("candidate_count") not in (None, len(canonical)):
        raise CustodyError(f"{case_id} {arm_id}: candidate count does not match predictions")
    return canonical


def _aggregate_arm(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["true_positive"] for row in case_results)
    fp = sum(row["false_positive"] for row in case_results)
    fn = sum(row["false_negative"] for row in case_results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metrics": {
            "case_count": len(case_results),
            "gold_bend_count": tp + fn,
            "prediction_count": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "exact_case_count": sum(1 for row in case_results if row["exact"]),
        },
        "cases": case_results,
    }


def score_sealed_three_arm(
    scan_manifest_path: Path,
    execution_root: Path,
    freeze_packet_path: Path,
    review_root: Path,
    label_catalog_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Score without opening prediction JSON until the full label seal validates."""

    scan_manifest_path = scan_manifest_path.resolve()
    execution_root = execution_root.resolve()
    freeze_packet_path = freeze_packet_path.resolve()
    review_root = review_root.resolve()
    label_catalog_path = label_catalog_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite score namespace: {output_root}")

    freeze = validate_freeze_packet(freeze_packet_path)
    cases, _execution, identity = validate_execution_identity(
        scan_manifest_path, execution_root, freeze_packet_path
    )
    seal_path = review_root / "execution_identity_seal.json"
    _seal, seal_sha = _validate_execution_seal(
        seal_path,
        cases=cases,
        identity=identity,
        freeze_packet_path=freeze_packet_path,
        scan_manifest_path=scan_manifest_path,
    )
    render_path = review_root / "raw_scan_only_render_manifest.json"
    _render, render_sha = _validate_render_manifest(
        render_path,
        cases=cases,
        execution_seal_sha256=seal_sha,
        scan_manifest_path=scan_manifest_path,
    )

    # This is the critical custody boundary.  No prediction JSON is parsed by
    # any code above this point; all three-arm identity checks are byte hashes.
    gold_by_case, label_sha = validate_sealed_label_catalog(
        label_catalog_path,
        cases=cases,
        render_manifest_sha256=render_sha,
        execution_seal_sha256=seal_sha,
        freeze_packet_sha256=_sha(freeze_packet_path),
    )
    scorer = _load_canonical_scorer(freeze)
    thresholds = scorer.Thresholds(5.0, 4.0, 0.60, 0.60)

    scored_arms: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        case_results = []
        for case, sealed_case in zip(cases, identity["cases"]):
            case_id = case["case_id"]
            relative = sealed_case["arms"][arm_id]["primary_artifact"]
            predictions = _load_prediction_records(
                _under(execution_root, relative), case_id=case_id, arm_id=arm_id
            )
            result = scorer.match_lines(gold_by_case[case_id], predictions, thresholds)
            result["case_id"] = case_id
            case_results.append(result)
        scored_arms[arm_id] = _aggregate_arm(case_results)

    a = scored_arms[ARM_IDS[0]]["metrics"]
    b = scored_arms[ARM_IDS[1]]["metrics"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": SCORE_KIND,
        "status": "PROSPECTIVE_THREE_ARM_SCORED_NO_RETUNING",
        "research_only": True,
        "promotion_eligible": False,
        "generalization_claim_authorized": False,
        "all_cases_included": True,
        "case_exclusions": [],
        "retuning_after_freeze": False,
        "threshold_overrides": False,
        "replay_used_for_identity_only": True,
        "freeze_packet_sha256": _sha(freeze_packet_path),
        "prospective_execution_manifest_sha256": identity["execution_manifest_sha256"],
        "execution_identity_seal_sha256": seal_sha,
        "raw_scan_only_render_manifest_sha256": render_sha,
        "sealed_label_catalog_sha256": label_sha,
        "canonical_scorer": {
            "path": SCORER_RELATIVE_PATH,
            "sha256": freeze["component_lock"]["scorer"]["sha256"],
        },
        "thresholds": deepcopy(CANONICAL_THRESHOLDS),
        "arms": scored_arms,
        "decision_readout": {
            "arm_b_zero_tp_loss_vs_a": b["true_positive"] >= a["true_positive"],
            "arm_b_strictly_fewer_fp_vs_a": b["false_positive"] < a["false_positive"],
            "arm_b_transfer_success": (
                b["true_positive"] >= a["true_positive"]
                and b["false_positive"] < a["false_positive"]
            ),
            "arm_c_reporting": "DESCRIPTIVE_ONLY_NO_AUTOMATIC_PROMOTION",
        },
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        report_path = staging / "prospective_three_arm_score.json"
        _write_json(report_path, report)
        report_sha = _write_sidecar(report_path)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "report": str(output_root / report_path.name),
        "report_sha256": report_sha,
        "metrics": {arm_id: scored_arms[arm_id]["metrics"] for arm_id in ARM_IDS},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-review")
    prepare.add_argument("--scan-manifest", type=Path, required=True)
    prepare.add_argument("--execution-root", type=Path, required=True)
    prepare.add_argument("--freeze-packet", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--scan-manifest", type=Path, required=True)
    score.add_argument("--execution-root", type=Path, required=True)
    score.add_argument("--freeze-packet", type=Path, required=True)
    score.add_argument("--review-root", type=Path, required=True)
    score.add_argument("--label-catalog", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-review":
        result = prepare_prediction_hidden_review(
            args.scan_manifest,
            args.execution_root,
            args.freeze_packet,
            args.output_root,
        )
    else:
        result = score_sealed_three_arm(
            args.scan_manifest,
            args.execution_root,
            args.freeze_packet,
            args.review_root,
            args.label_catalog,
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
