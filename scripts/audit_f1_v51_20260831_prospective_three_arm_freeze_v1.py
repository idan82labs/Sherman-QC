#!/usr/bin/env python3
"""Fail-closed audit for the sealed 2026-08-31 three-arm F1 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.json"
DEFAULT_SEAL = ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.sha256"

EXPECTED_KIND = "sherman_qc_f1_v51_20260831_prospective_three_arm_freeze_v1"
EXPECTED_LOCK_KIND = "sherman_qc_f1_three_arm_component_lock_v1"
COMPONENT_PATHS = {
    "v51_case_runner": "scripts/run_f1_v51_today_scan_only_development.py",
    "v43_guard": "backend/f1_post_v51_v43_explosion_guard_v1.py",
    "recall_runtime": "scripts/apply_f1_v51_source_only_recovery_v1.py",
    "precision_runtime": "backend/f1_post_v51_precision_ownership_v52_development.py",
    "axis_span_runtime": "scripts/apply_f1_v51_today_axis_span_recovery_v1.py",
    "v9_model": "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib",
    "v4_model": "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib",
    "recall_model": "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json",
    "axis_span_model": "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json",
    "scorer": "scripts/evaluate_f1_against_manual_gold.py",
}
THRESHOLDS = {
    "axis_angle_deg_max": 5.0,
    "transverse_offset_mm_max": 4.0,
    "gold_span_overlap_min": 0.60,
    "prediction_span_overlap_min": 0.60,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _lexical_repo_path(value: Any) -> str:
    path = str(value)
    pure = PurePosixPath(path)
    _require(path == pure.as_posix(), f"non-canonical component path: {path}")
    _require(not pure.is_absolute(), f"absolute component path: {path}")
    _require(".." not in pure.parts and "." not in pure.parts, f"unsafe component path: {path}")
    return path


def validate_payload(payload: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, "freeze schema mismatch")
    _require(payload.get("kind") == EXPECTED_KIND, "freeze kind mismatch")
    _require(
        payload.get("status") == "SEALED_BEFORE_2026_08_31_DELIVERY_DISCOVERY",
        "freeze status mismatch",
    )
    for flag in ("research_only", "development_only"):
        _require(payload.get(flag) is True, f"{flag} must be true")
    for flag in ("promotion_eligible", "generalization_claim_authorized"):
        _require(payload.get(flag) is False, f"{flag} must be false")
    attestation = payload.get("pre_discovery_attestation")
    _require(isinstance(attestation, dict) and attestation, "pre-discovery attestation missing")
    _require(all(value is False for value in attestation.values()), "pre-discovery attestation opened")

    bindings = payload.get("artifact_bindings")
    _require(isinstance(bindings, list) and bindings, "artifact bindings missing")
    _require(payload.get("artifact_binding_count") == len(bindings), "binding count mismatch")
    _require(
        payload.get("artifact_binding_digest_sha256")
        == hashlib.sha256(_canonical_bytes(bindings)).hexdigest(),
        "binding digest mismatch",
    )
    paths = [str(row.get("path")) for row in bindings]
    _require(paths == sorted(paths) and len(paths) == len(set(paths)), "bindings not unique/sorted")
    by_path: dict[str, dict[str, Any]] = {}
    for row in bindings:
        _require(set(row) == {"path", "sha256", "size_bytes", "roles"}, "binding schema mismatch")
        relative = _lexical_repo_path(row["path"])
        _require(isinstance(row["roles"], list) and row["roles"] == sorted(set(row["roles"])), "binding roles invalid")
        artifact = root / relative
        _require(artifact.is_file(), f"bound artifact missing: {relative}")
        _require(artifact.stat().st_size == row["size_bytes"], f"bound artifact size drift: {relative}")
        _require(_sha(artifact) == row["sha256"], f"bound artifact hash drift: {relative}")
        by_path[relative] = row

    lock = payload.get("component_lock")
    _require(isinstance(lock, dict), "component lock missing")
    _require(lock.get("schema_version") == 1 and lock.get("kind") == EXPECTED_LOCK_KIND, "component lock kind mismatch")
    _require(lock.get("source_only") is True and lock.get("fail_closed") is True, "component lock is not source-only/fail-closed")
    for key, expected_path in COMPONENT_PATHS.items():
        component = lock.get(key)
        _require(isinstance(component, dict) and set(component) == {"path", "sha256"}, f"invalid component binding: {key}")
        actual_path = _lexical_repo_path(component["path"])
        _require(actual_path == expected_path, f"component path mismatch: {key}")
        _require(actual_path in by_path, f"component absent from artifact bindings: {key}")
        _require(component["sha256"] == by_path[actual_path]["sha256"], f"component hash mismatch: {key}")
    orchestrator = lock.get("prospective_orchestrator")
    _require(
        isinstance(orchestrator, dict)
        and orchestrator.get("binding_state") == "AUTHORIZED_FUTURE_BINDING",
        "prospective orchestrator authorization missing",
    )

    arms = payload.get("arms")
    _require(isinstance(arms, list) and [row.get("arm_id") for row in arms] == ["A", "B", "C"], "arm order mismatch")
    _require(arms[1].get("maximum_safe_v43_post_merge_additions") == 1, "guard cutoff drift")
    _require(arms[2].get("recall_profile") == "balanced", "recall profile drift")
    _require(payload.get("scoring", {}).get("thresholds") == THRESHOLDS, "scorer threshold drift")

    custody = payload.get("inference_custody") or {}
    _require(custody.get("all_three_arms_complete_before_labeling") is True, "arm-before-label gate missing")
    _require(custody.get("base_execution_passes") == ["primary", "replay"], "base replay contract drift")
    _require(custody.get("derived_execution_passes") == ["primary", "replay"], "derived replay contract drift")
    labeling = payload.get("labeling_custody") or {}
    _require(labeling.get("prediction_hidden") is True, "labeling is not prediction-hidden")
    rules = payload.get("decision_rules") or {}
    _require(rules.get("retuning_after_freeze") == "forbidden", "retuning prohibition missing")
    _require(rules.get("case_removal_after_freeze") == "forbidden", "case-removal prohibition missing")

    inventory = payload.get("v51_inventory_source") or {}
    inventory_path = _lexical_repo_path(inventory.get("path"))
    _require(_sha(root / inventory_path) == inventory.get("sha256"), "V51 inventory provenance drift")
    return {
        "artifact_binding_count": len(bindings),
        "component_count": len(COMPONENT_PATHS),
        "locked_at_utc": payload.get("locked_at_utc"),
        "arms": ["A", "B", "C"],
    }


def audit_freeze(
    freeze_path: Path = DEFAULT_FREEZE,
    seal_path: Path = DEFAULT_SEAL,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    freeze_path = freeze_path.resolve()
    seal_path = seal_path.resolve()
    _require(freeze_path.is_file(), "freeze file missing")
    _require(seal_path.is_file(), "freeze seal missing")
    expected_seal = f"{_sha(freeze_path)}  {freeze_path.name}\n"
    _require(seal_path.read_text(encoding="utf-8") == expected_seal, "freeze seal mismatch")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    result = validate_payload(payload, root=root)
    result.update(
        {
            "freeze_path": str(freeze_path),
            "freeze_sha256": _sha(freeze_path),
            "seal_path": str(seal_path),
            "audit_pass": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    print(json.dumps(audit_freeze(args.freeze, args.seal), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
