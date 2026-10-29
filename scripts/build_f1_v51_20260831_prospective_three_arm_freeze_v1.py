#!/usr/bin/env python3
"""Seal the three-arm 2026-08-31 V51 prospective-development protocol.

This builder is intentionally source-only.  It inventories code and model
bytes already present in the repository and contains no delivery-discovery,
scan-loading, labeling, truth, or scoring operation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.json"
SEAL_PATH = ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.sha256"
V51_INVENTORY_PATH = ROOT / "research/f1_v51_new_untouched_corpus_lock_v2.json"
LOCKED_AT_UTC = "2026-08-31T08:02:43Z"

EXPECTED_MODEL_HASHES = {
    "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib":
        "ac48c8b9ed48d352ca5a125137a6128a73f793cef012c967eb3a998e460a39ae",
    "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib":
        "412946c78a63d8550f9abf1d476c94dcf01b7ed1215cfef28053a6dfd7c966f1",
    "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json":
        "5f8b5e43cefb9114dafccaf65159270bec7875657a9d3df1acc699b0156a1897",
    "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json":
        "8689db67f8a7f1df04c4cc43b104ef116e0cecb9004d5aca580764091318ec7b",
}

ROLE_PATHS = {
    "V51_SCAN_ONLY_RUNNER": {
        "scripts/run_f1_v51_today_scan_only_development.py",
    },
    "V51_FROZEN_HELPER": {
        "scripts/run_f1_v51_new_untouched_corpus_blind_once.py",
    },
    "V51_SELECTOR_MODEL": {
        "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib",
        "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib",
    },
    "ARM_B_GUARD_RUNTIME": {
        "backend/f1_post_v51_v43_explosion_guard_v1.py",
    },
    "ARM_C_PRECISION_RUNTIME": {
        "backend/f1_post_v51_precision_ownership_v52_development.py",
    },
    "ARM_C_RECALL_RUNTIME": {
        "scripts/apply_f1_v51_source_only_recovery_v1.py",
    },
    "ARM_C_RECALL_MODEL": {
        "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json",
    },
    "ARM_C_AXIS_SPAN_RUNTIME": {
        "scripts/apply_f1_v51_today_axis_span_recovery_v1.py",
    },
    "ARM_C_AXIS_SPAN_MODEL": {
        "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json",
    },
    "WITHHELD_CANONICAL_SCORER": {
        "scripts/evaluate_f1_against_manual_gold.py",
    },
    "V51_INVENTORY_PROVENANCE": {
        "research/f1_v51_new_untouched_corpus_lock_v2.json",
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _artifact_bindings() -> list[dict[str, Any]]:
    source_inventory = json.loads(V51_INVENTORY_PATH.read_text(encoding="utf-8"))
    if source_inventory.get("kind") != "sherman_qc_f1_v51_new_untouched_corpus_blind_once_lock":
        raise ValueError("unexpected V51 inventory kind")
    paths = {
        str(row["path"])
        for row in source_inventory.get("source_bindings") or []
    }
    role_by_path: dict[str, set[str]] = {
        path: {"V51_TRANSITIVE_ORIGINAL_LOCK_INVENTORY"} for path in paths
    }
    for role, role_paths in ROLE_PATHS.items():
        paths.update(role_paths)
        for path in role_paths:
            role_by_path.setdefault(path, set()).add(role)

    bindings = []
    for relative in sorted(paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"cannot freeze missing artifact: {relative}")
        bindings.append(
            {
                "path": relative,
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
                "roles": sorted(role_by_path[relative]),
            }
        )
    for relative, expected in EXPECTED_MODEL_HASHES.items():
        actual = next(row["sha256"] for row in bindings if row["path"] == relative)
        if actual != expected:
            raise ValueError(f"model byte identity changed before freeze: {relative}")
    return bindings


def build_payload() -> dict[str, Any]:
    bindings = _artifact_bindings()
    binding_digest = hashlib.sha256(_canonical_bytes(bindings)).hexdigest()
    by_path = {row["path"]: row for row in bindings}

    def component(path: str) -> dict[str, str]:
        row = by_path[path]
        return {"path": path, "sha256": str(row["sha256"])}

    component_lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_three_arm_component_lock_v1",
        "source_only": True,
        "fail_closed": True,
        "locked_at_utc": LOCKED_AT_UTC,
        "v51_case_runner": component("scripts/run_f1_v51_today_scan_only_development.py"),
        "v43_guard": component("backend/f1_post_v51_v43_explosion_guard_v1.py"),
        "recall_runtime": component("scripts/apply_f1_v51_source_only_recovery_v1.py"),
        "precision_runtime": component("backend/f1_post_v51_precision_ownership_v52_development.py"),
        "axis_span_runtime": component("scripts/apply_f1_v51_today_axis_span_recovery_v1.py"),
        "v9_model": component("output/research/f1-new-parts-20260812/v9/model/router_v9.joblib"),
        "v4_model": component("output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib"),
        "recall_model": component("data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json"),
        "axis_span_model": component("data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json"),
        "scorer": component("scripts/evaluate_f1_against_manual_gold.py"),
        "prospective_orchestrator": {
            "binding_state": "AUTHORIZED_FUTURE_BINDING",
            "constraint": "may only orchestrate these locked scientific components, enforce primary/replay byte identity, and fail closed; its bytes must be separately hashed before first scan inference",
        },
    }
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_20260831_prospective_three_arm_freeze_v1",
        "status": "SEALED_BEFORE_2026_08_31_DELIVERY_DISCOVERY",
        "locked_at_utc": LOCKED_AT_UTC,
        "local_cohort_date": "2026-08-31",
        "local_timezone": "Asia/Jerusalem",
        "research_only": True,
        "development_only": True,
        "promotion_eligible": False,
        "generalization_claim_authorized": False,
        "pre_discovery_attestation": {
            "external_delivery_discovery_started": False,
            "gmail_inspected_for_cohort": False,
            "sharepoint_inspected_for_cohort": False,
            "today_scan_bytes_inspected": False,
            "today_scan_count_known": False,
            "today_truth_or_labels_opened": False,
            "today_cad_or_drawings_opened": False,
            "today_part_identity_opened": False,
        },
        "component_lock": component_lock,
        "cohort_intake": {
            "eligibility": "every newly delivered raw PLY or STL scan belonging to the 2026-08-31 daily delivery, discovered only after this freeze",
            "case_exclusions_after_discovery": "none; unreadable or failed cases remain in the denominator as empty-prediction abstentions",
            "opaque_materialization": "copy each eligible byte stream to an opaque case path before inference; preserve a sealed custodian-only provenance map outside all inference and labeling inputs",
            "opaque_case_order": "ascending scan SHA-256, then byte size, then sealed delivery-member ordinal solely to break exact-byte ties",
            "scan_only_manifest_contract": {
                "schema_version": 1,
                "kind": "sherman_qc_f1_v51_opaque_scan_only_development_input",
                "top_level_keys_only": ["schema_version", "kind", "cases"],
                "case_keys_only": ["scan_path", "scan_sha256"],
            },
        },
        "inference_custody": {
            "allowed_inputs": ["opaque raw scan geometry bytes", "frozen code bytes", "frozen model bytes"],
            "forbidden_inputs": [
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
            ],
            "base_execution_passes": ["primary", "replay"],
            "derived_execution_passes": ["primary", "replay"],
            "required_identity": "canonical semantic result bytes must match primary versus replay for every case and every arm before labels may open",
            "failure_policy": "fail closed to an empty prediction list; do not exclude, repair, rerun with changed bytes, or reveal side information",
            "all_three_arms_complete_before_labeling": True,
        },
        "arms": [
            {
                "arm_id": "A",
                "name": "FROZEN_V51_SCAN_ONLY",
                "input": "opaque scan-only manifest",
                "composition": ["V16", "V43", "V44", "V50", "V47", "V48", "V49", "V51"],
                "output": "V51 semantic_result.json predictions",
                "runtime_roles": ["V51_SCAN_ONLY_RUNNER", "V51_FROZEN_HELPER", "V51_SELECTOR_MODEL", "V51_TRANSITIVE_ORIGINAL_LOCK_INVENTORY"],
            },
            {
                "arm_id": "B",
                "name": "V51_WITH_V43_EXPLOSION_GUARD_V1",
                "input": "the same-pass frozen V16, V43, and V51 source-only artifacts",
                "rule": "if v43_post_merge_addition_count > 1, emit V16 predictions; otherwise emit V51 predictions",
                "maximum_safe_v43_post_merge_additions": 1,
                "failed_case_rule": "preserve the empty fail-closed V51 abstention",
                "runtime_roles": ["ARM_B_GUARD_RUNTIME"],
            },
            {
                "arm_id": "C",
                "name": "V51_AGGRESSIVE_PRECISION_RECALL_AXIS_SPAN_V1",
                "input": "the same-pass V51 semantic result and scan-only candidate pool",
                "composition": [
                    "apply frozen precision-ownership gates to V51",
                    "apply frozen R3 source-only recovery with profile balanced",
                    "append only recovery bend IDs absent from original V51",
                    "apply frozen R2 source-axis/span recovery to the precision-plus-recall union",
                ],
                "recall_profile": "balanced",
                "axis_span_profile": "F1_V51_TODAY_AXIS_SPAN_RECOVERY_DEVELOPMENT_V1",
                "runtime_roles": [
                    "ARM_C_PRECISION_RUNTIME",
                    "ARM_C_RECALL_RUNTIME",
                    "ARM_C_RECALL_MODEL",
                    "ARM_C_AXIS_SPAN_RUNTIME",
                    "ARM_C_AXIS_SPAN_MODEL",
                ],
                "consumed_label_warning": "all cutoffs/models were fitted on the consumed 2026-08-30 daily development cohort",
            },
        ],
        "labeling_custody": {
            "opens_only_after": "all arms and both passes are sealed and byte-identical",
            "prediction_hidden": True,
            "labeler_inputs": ["opaque scan geometry", "truth-free multi-view visual overpass"],
            "labeler_forbidden_inputs": ["any arm prediction", "algorithmic candidate pool", "expected count", "CAD", "drawing", "part identity"],
            "adjudication": "independent high-performance-agent bend marking followed by prediction-hidden reconciliation; seal one canonical manual bend-line catalog before scoring",
            "catalog_mutation_after_scoring": "forbidden",
        },
        "scoring": {
            "opens_only_after": "canonical prediction-hidden label catalog is sealed",
            "scorer_role": "WITHHELD_CANONICAL_SCORER",
            "matching": "one_to_one_minimum_cost",
            "thresholds": {
                "axis_angle_deg_max": 5.0,
                "transverse_offset_mm_max": 4.0,
                "gold_span_overlap_min": 0.60,
                "prediction_span_overlap_min": 0.60,
            },
            "primary_artifacts_scored": True,
            "replay_artifacts_scored_only_for_identity_audit": True,
            "report_for_each_arm": [
                "case_count",
                "gold bend count",
                "prediction count",
                "TP",
                "FP",
                "FN",
                "micro precision",
                "micro recall",
                "micro F1",
                "exact-case count",
                "per-case results",
            ],
        },
        "decision_rules": {
            "retuning_after_freeze": "forbidden",
            "case_removal_after_freeze": "forbidden",
            "threshold_changes_after_freeze": "forbidden",
            "arm_b_transfer_success": "zero TP loss versus arm A and strictly fewer FP on this cohort",
            "arm_c_reporting": "report descriptively against arm A and arm B; no automatic promotion",
            "baseline_policy": "arm A remains the frozen reference; no result from this protocol alone is authorized as a production baseline",
            "null_delivery_policy": "if no eligible scans exist, report zero-case prospective cohort and do not substitute historical scans",
        },
        "v51_inventory_source": {
            "path": str(V51_INVENTORY_PATH.relative_to(ROOT)),
            "sha256": _sha(V51_INVENTORY_PATH),
            "note": "paths were rehashed at this freeze; prior lock hashes are provenance, not assumed current identity",
        },
        "artifact_binding_count": len(bindings),
        "artifact_binding_digest_sha256": binding_digest,
        "artifact_bindings": bindings,
    }


def main() -> int:
    if FREEZE_PATH.exists() or SEAL_PATH.exists():
        raise FileExistsError("prospective freeze or seal already exists")
    payload = build_payload()
    encoded = _canonical_bytes(payload)
    freeze_sha = hashlib.sha256(encoded).hexdigest()
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FREEZE_PATH.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    seal = f"{freeze_sha}  {FREEZE_PATH.name}\n".encode()
    with SEAL_PATH.open("xb") as handle:
        handle.write(seal)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "freeze": str(FREEZE_PATH),
                "freeze_sha256": freeze_sha,
                "seal": str(SEAL_PATH),
                "artifact_binding_count": payload["artifact_binding_count"],
                "locked_at_utc": payload["locked_at_utc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
