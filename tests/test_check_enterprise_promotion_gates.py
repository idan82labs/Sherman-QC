import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_enterprise_promotion_gates.py"
SPEC = importlib.util.spec_from_file_location("check_enterprise_promotion_gates", MODULE_PATH)
gates = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gates)


def _truth_records(count=3):
    return {
        "records": [
            {
                "part_key": f"P{index}",
                "part_family": "conventional",
                "evaluation_partition": "locked_holdout",
                "release_metric_eligible": True,
                "instance_geometry": [{"bend_id": "B1"}],
            }
            for index in range(count)
        ]
    }


def test_promotion_gates_pass_with_blind_complete_evidence():
    thresholds = {
        "minimum_truth_scans": 3,
        "minimum_truth_parts": 3,
        "minimum_locked_holdout_parts": 3,
        "minimum_family_scans": 3,
    }
    report = gates.evaluate_promotion_gates(
        truth_registry=_truth_records(),
        ply_summary={
            "aggregate": {
                "exact_count_evaluated": 3,
                "candidate_exact_accuracy": 0.90,
                "candidate_range_contains_truth_rate": 1.0,
                "render_admissible_exact_count_evaluated": 3,
                "render_admissible_exact_accuracy": 1.0,
            }
        },
        cad_ply_summary={
            "inference_mode": "blind_cad_ply",
            "aggregate": {"full_scans": 3, "full_scan_exact_completion_rate": 0.96},
            "typed_release_metrics": {
                "scans_processed": 3,
                "exact_count_accuracy": 0.96,
                "blind_inference_contract_rate": 1.0,
            },
        },
        operational_summary={
            "deterministic_repeat_rate": 1.0,
            "p95_latency_sec": 200.0,
            "job_success_rate": 1.0,
        },
        thresholds=thresholds,
    )

    assert report["promotion_status"] == "eligible"
    assert report["blocked_gate_count"] == 0


def test_promotion_gates_reject_operator_assisted_cad_accuracy():
    report = gates.evaluate_promotion_gates(
        truth_registry=_truth_records(),
        ply_summary=None,
        cad_ply_summary={
            "inference_mode": "operator_assisted",
            "aggregate": {"full_scans": 500, "full_scan_exact_completion_rate": 1.0},
        },
        operational_summary=None,
        thresholds={
            "minimum_truth_scans": 3,
            "minimum_truth_parts": 3,
            "minimum_locked_holdout_parts": 3,
            "minimum_family_scans": 3,
        },
    )

    contract = next(item for item in report["checks"] if item["gate_id"] == "cad_ply.blind_inference_contract")
    assert contract["passed"] is False
    assert report["promotion_status"] == "blocked"


def test_promotion_gates_reject_blind_mode_without_explicit_truth_boundary_proof():
    report = gates.evaluate_promotion_gates(
        truth_registry=_truth_records(),
        ply_summary=None,
        cad_ply_summary={
            "inference_mode": "blind_cad_ply",
            "aggregate": {"full_scans": 500, "full_scan_exact_completion_rate": 1.0},
        },
        operational_summary=None,
        thresholds={
            "minimum_truth_scans": 3,
            "minimum_truth_parts": 3,
            "minimum_locked_holdout_parts": 3,
            "minimum_family_scans": 3,
        },
    )

    contract = next(item for item in report["checks"] if item["gate_id"] == "cad_ply.blind_inference_contract")
    assert contract["passed"] is False
    assert contract["observed"]["inference_mode"] == "blind_cad_ply"
    assert contract["observed"]["explicit_contract_rate"] is None


def test_promotion_gates_prefer_all_scan_typed_release_metrics():
    report = gates.evaluate_promotion_gates(
        truth_registry=_truth_records(),
        ply_summary={
            "aggregate": {
                "exact_count_evaluated": 1,
                "candidate_exact_accuracy": 1.0,
                "candidate_range_contains_truth_rate": 1.0,
                "render_admissible_exact_count_evaluated": 1,
                "render_admissible_exact_accuracy": 1.0,
            },
            "typed_release_metrics": {
                "scans_processed": 3,
                "exact_count_accuracy": 0.0,
                "range_containment_rate": 1 / 3,
                "render_admissible_exact_accuracy": 0.0,
            },
        },
        cad_ply_summary={
            "inference_mode": "blind_cad_ply",
            "aggregate": {"full_scans": 1, "full_scan_exact_completion_rate": 1.0},
            "typed_release_metrics": {"scans_processed": 3, "exact_count_accuracy": 1 / 3},
        },
        operational_summary=None,
        thresholds={
            "minimum_truth_scans": 3,
            "minimum_truth_parts": 3,
            "minimum_locked_holdout_parts": 3,
            "minimum_family_scans": 3,
        },
    )

    checks = {item["gate_id"]: item for item in report["checks"]}
    assert checks["ply_only.evaluated_scan_count"]["observed"] == 3
    assert checks["ply_only.exact_accuracy"]["observed"] == 0.0
    assert checks["cad_ply.evaluated_scan_count"]["observed"] == 3
    assert checks["cad_ply.exact_accuracy"]["observed"] == 1 / 3


def test_cad_uplift_gate_rejects_undersized_mismatched_evidence():
    report = gates.evaluate_promotion_gates(
        truth_registry=_truth_records(),
        ply_summary={
            "typed_release_metrics": {
                "scans_processed": 1,
                "exact_count_accuracy": 0.0,
                "range_containment_rate": 1.0,
                "render_admissible_exact_accuracy": 1.0,
            }
        },
        cad_ply_summary={
            "inference_mode": "blind_cad_ply",
            "typed_release_metrics": {
                "scans_processed": 2,
                "exact_count_accuracy": 1.0,
                "blind_inference_contract_rate": 1.0,
            },
        },
        operational_summary=None,
        thresholds={
            "minimum_truth_scans": 3,
            "minimum_truth_parts": 3,
            "minimum_locked_holdout_parts": 3,
            "minimum_family_scans": 3,
        },
    )

    uplift = next(item for item in report["checks"] if item["gate_id"] == "cad_ply.uplift_over_ply_only")
    assert uplift["observed"]["uplift"] == 1.0
    assert uplift["passed"] is False
