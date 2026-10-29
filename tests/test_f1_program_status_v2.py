from pathlib import Path

from research.f1_program_status_v2 import build


def test_status_counts_pending_bytes_and_fails_promotion_closed(tmp_path: Path):
    (tmp_path / "opaque").mkdir()
    (tmp_path / "opaque/N001.ply").write_bytes(b"ply")
    result = build(
        unified_gate={
            "gates": {
                "scan_only": {
                    "grade": "C",
                    "direction": "NEGATIVE_DEVELOPMENT",
                    "baseline_metrics": {"f1": .59},
                    "candidate_metrics": {"f1": .52},
                    "visual": {"passed": False},
                },
                "cad_assisted": {
                    "grade": "C",
                    "direction": "NEGATIVE_DEVELOPMENT",
                    "baseline_metrics": {"f1": .458},
                    "candidate_metrics": {"f1": .456},
                    "visual": {"passed": False},
                },
            }
        },
        soak_report={"status": "PASS", "elapsed_seconds": 10800},
        authority_validation={"status": "AWAITING_AUTHORITY_REVIEW", "event_denominator": 63},
        pending_intake={
                "pair_count": 1,
                "opaque_cases": [
                    {"opaque_id": "N001", "target_path": "opaque/N001.ply"},
                    {"opaque_id": "N002", "target_path": "opaque/N002.ply"},
                ],
        },
        pending_intake_root=tmp_path,
        lineage_registry={"intakes": []},
    )
    intake = result["pending_new_physical_intake"]
    assert intake["locally_present_scan_count"] == 1
    assert intake["download_pending_scan_count"] == 1
    assert intake["qualification_capacity_credit"] == 0
    assert result["contract_soak"]["qualified_three_hours"] is True
    assert result["production_promotion_authorized"] is False
    assert result["promotion_evidence"]["status"] == "NOT_AVAILABLE"
    assert result["evaluation_integrity"] == {
        "zero_shot_default": True,
        "part_truth_hints_disabled_by_default": True,
        "filename_scan_state_inference_disabled": True,
        "expected_count_scan_state_inference_disabled": True,
        "operator_assisted_truth_requires_explicit_opt_in": True,
    }
    assert result["lane_status"]["cad_assisted"]["candidate_metrics"] == {"f1": .456}
    assert result["lane_status"]["cad_assisted"]["retained_metrics"] == {"f1": .458}
    assert result["lane_status"]["cad_assisted"]["candidate_metric_role"] == "EXPERIMENTAL_CHALLENGER"


def test_capacity_blocker_can_unlock_from_sealed_custodian_aggregates(tmp_path: Path):
    result = build(
        unified_gate={"gates": {}},
        soak_report={"status": "PASS", "elapsed_seconds": 10800},
        authority_validation={"status": "READY_TO_AMEND_CIBEC_CHARTER", "event_denominator": 63},
        pending_intake={"pair_count": 0, "opaque_cases": []},
        pending_intake_root=tmp_path,
        lineage_registry={"intakes": [{
            "root": "sealed",
            "status": "UNTOUCHED_CUSTODIAN_SEALED",
            "scan_count": 30,
            "morphology_family_count": 15,
            "independent_instance_count": 30,
        }]},
    )
    assert result["sealed_qualification_capacity_verified"] is True
    assert "UNTOUCHED_QUALIFICATION_CAPACITY_BELOW_30_SCANS_15_MORPHOLOGIES" not in result["active_blockers"]


def test_status_aggregates_multiple_pending_deliveries(tmp_path: Path):
    result = build(
        unified_gate={"gates": {}},
        soak_report={"status": "PASS", "elapsed_seconds": 10800},
        authority_validation={"status": "AWAITING_AUTHORITY_REVIEW", "event_denominator": 63},
        pending_intake={"pair_count": 1, "opaque_cases": [{"opaque_id": "N001", "target_path": "opaque/N001.ply"}]},
        pending_intake_root=tmp_path / "first",
        lineage_registry={"intakes": []},
        additional_pending_intakes=[
            ({"pair_count": 1, "opaque_cases": [{"opaque_id": "N002", "target_path": "opaque/N002.ply"}]}, tmp_path / "second"),
        ],
    )

    intake = result["pending_new_physical_intake"]
    assert intake["intake_count"] == 2
    assert intake["declared_scan_count"] == 2
    assert intake["declared_pair_count"] == 2
    assert intake["download_pending_scan_count"] == 2
