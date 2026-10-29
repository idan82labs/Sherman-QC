"""Build the current lane-separated F1 program status without opening truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _count_at_least(row: Mapping[str, Any], key: str, minimum: int) -> bool:
    value = row.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def build(
    *,
    unified_gate: Mapping[str, Any],
    soak_report: Mapping[str, Any],
    authority_validation: Mapping[str, Any],
    pending_intake: Mapping[str, Any],
    pending_intake_root: Path,
    lineage_registry: Mapping[str, Any],
    additional_pending_intakes: Sequence[tuple[Mapping[str, Any], Path]] = (),
) -> dict[str, Any]:
    intake_sources = [(pending_intake, pending_intake_root), *additional_pending_intakes]
    opaque_cases: list[Mapping[str, Any]] = []
    present: list[Mapping[str, Any]] = []
    pair_count = 0
    download_statuses: list[str] = []
    opaque_ids: set[str] = set()
    for intake, intake_root in intake_sources:
        pair_count += int(intake.get("pair_count") or 0)
        download_statuses.append(str(intake.get("download_status") or "UNKNOWN"))
        for row in intake.get("opaque_cases") or []:
            opaque_id = str(row.get("opaque_id") or "")
            if not opaque_id or opaque_id in opaque_ids:
                raise ValueError("pending intake opaque IDs must be globally unique")
            opaque_ids.add(opaque_id)
            opaque_cases.append(row)
            if (intake_root / str(row.get("target_path") or "")).is_file():
                present.append(row)
    sealed = [
        row for row in lineage_registry.get("intakes") or []
        if row.get("status") == "UNTOUCHED_CUSTODIAN_SEALED"
    ]
    sealed_capacity_verified = any(
        _count_at_least(row, "scan_count", 30)
        and _count_at_least(row, "morphology_family_count", 15)
        and _count_at_least(row, "independent_instance_count", 30)
        for row in sealed
    )
    gates = unified_gate.get("gates") or {}
    lane_status = {
        lane: {
            "grade": payload.get("grade"),
            "direction": payload.get("direction"),
            "operational_badge": payload.get("operational_badge"),
            "evidence_class": payload.get("evidence_class"),
            "baseline_metrics": payload.get("baseline_metrics"),
            "candidate_metrics": payload.get("candidate_metrics"),
            "retained_metrics": (
                payload.get("baseline_metrics")
                if payload.get("direction") == "NEGATIVE_DEVELOPMENT"
                else payload.get("candidate_metrics")
            ),
            "candidate_metric_role": "EXPERIMENTAL_CHALLENGER",
            "retained_metric_role": "CURRENT_RETAINED_INCUMBENT",
            "visual_passed": (payload.get("visual") or {}).get("passed"),
        }
        for lane, payload in gates.items()
    }
    authority_ready = authority_validation.get("status") == "READY_TO_AMEND_CIBEC_CHARTER"
    soak_pass = soak_report.get("status") == "PASS" and float(soak_report.get("elapsed_seconds") or 0) >= 10800
    pending_count = len(opaque_cases)
    present_count = len(present)
    blockers = []
    if not authority_ready:
        blockers.append("TWO_INDEPENDENT_REAL_HUMAN_63_EVENT_CAD_ONTOLOGY_REVIEWS_REQUIRED")
    if present_count < pending_count:
        blockers.append("NEW_PHYSICAL_SCAN_BYTES_NOT_DOWNLOADED_TO_OPAQUE_INTAKE")
    if not sealed:
        blockers.append("NO_UNTOUCHED_CUSTODIAN_SEALED_QUALIFICATION_COHORT")
    if not sealed_capacity_verified:
        blockers.append("UNTOUCHED_QUALIFICATION_CAPACITY_BELOW_30_SCANS_15_MORPHOLOGIES")
    return {
        "schema_version": 2,
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "objective": "PRODUCTION_READY_ZERO_SHOT_SCAN_ONLY_AND_CAD_ASSISTED_BEND_INSPECTION",
        "physical_generalization_progress": False,
        "infrastructure_progress": True,
        "production_promotion_authorized": False,
        "promotion_evidence": {
            "status": "NOT_AVAILABLE",
            "candidate_lock_sha256": None,
            "qualification_report_sha256": None,
            "dual_review_truth_sha256": None,
        },
        "evaluation_integrity": {
            "zero_shot_default": True,
            "part_truth_hints_disabled_by_default": True,
            "filename_scan_state_inference_disabled": True,
            "expected_count_scan_state_inference_disabled": True,
            "operator_assisted_truth_requires_explicit_opt_in": True,
        },
        "lane_status": lane_status,
        "contract_soak": {
            "status": soak_report.get("status"),
            "qualified_three_hours": soak_pass,
            "elapsed_seconds": soak_report.get("elapsed_seconds"),
            "isolated_child_jobs": soak_report.get("isolated_child_jobs"),
            "isolated_abstention_jobs": soak_report.get("isolated_abstention_jobs"),
        },
        "cad_ontology_authority": {
            "status": authority_validation.get("status"),
            "review_pack_sha256": authority_validation.get("review_pack_sha256"),
            "event_denominator": authority_validation.get("event_denominator"),
            "ready": authority_ready,
            "disagreement_event_count": len(authority_validation.get("disagreement_event_refs") or []),
            "unresolved_event_count": len(authority_validation.get("unresolved_event_refs") or []),
        },
        "pending_new_physical_intake": {
            "intake_count": len(intake_sources),
            "download_status": (
                "MATERIALIZED"
                if present_count == pending_count
                else "PENDING_AUTHENTICATED_SHAREPOINT_CONNECTION"
            ),
            "source_download_statuses": download_statuses,
            "declared_scan_count": pending_count,
            "declared_pair_count": pair_count,
            "locally_present_scan_count": present_count,
            "download_pending_scan_count": pending_count - present_count,
            "qualification_capacity_credit": 0,
            "capacity_credit_reason": "V/X pairs are not independent-instance or line-truth evidence until custodian review and sealing",
            "identity_exposed_to_model_lane": False,
        },
        "sealed_qualification_intake_count": len(sealed),
        "sealed_qualification_capacity_verified": sealed_capacity_verified,
        "active_blockers": blockers,
        "permitted_parallel_work": [
            "DOWNLOAD_PENDING_SCAN_BYTES_UNDER_EVALUATOR_CUSTODY",
            "COMPLETE_TWO_INDEPENDENT_HUMAN_CAD_ONTOLOGY_REVIEWS",
            "VALIDATE_AND_COMPILE_RESOLVED_CAD_ONTOLOGY_CONSENSUS",
            "PREPARE_CUSTODIAN_METADATA_AND_INDEPENDENT_LINE_ANNOTATION_WORKFLOW",
            "HARDEN_FAIL_CLOSED_PRODUCTION_CONTRACTS_WITHOUT_MODEL_TUNING",
        ],
        "next_empirical_actions_after_unlock": {
            "cad_assisted": "ONE_PREREGISTERED_SIGNED_CIBEC_PRIMITIVE_TO_MANUFACTURING_EVENT_CONTRACTION",
            "scan_only": "AUTHENTIC_ORGANIZED_FRAME_CAPABILITY_OR_FORMAL_ACQUISITION_OPENING",
            "promotion": "UNTOUCHED_30_SCAN_15_MORPHOLOGY_DUAL_REVIEW_COMPARISON",
        },
        "generalization_progress_indicator": {
            "scan_only": "GPI-C_NO_NEW_GENERALIZATION_EVIDENCE",
            "cad_assisted": "GPI-C_NO_NEW_GENERALIZATION_EVIDENCE",
            "infrastructure": "GPI-C_POSITIVE_INFRASTRUCTURE_NOT_ACCURACY",
        },
    }
