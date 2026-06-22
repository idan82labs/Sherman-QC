import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_f1_next_solver_recovery_plan.py"
    spec = importlib.util.spec_from_file_location("summarize_f1_next_solver_recovery_plan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


planner = _load_module()


def _row(part_id, recommendation, target=4, selected=0, safe=0, track="known", family=None):
    return {
        "part_id": part_id,
        "target_count": target,
        "track": track,
        "part_family": family,
        "final_offline_recommendation": recommendation,
        "quality_decision": "limited_observed_support",
        "sparse_status": "sparse_arrangement_underfit",
        "selected_family_count": selected,
        "safe_candidate_count": safe,
    }


def test_hard_known_underfit_routes_to_latent_support_creation():
    report = planner.summarize_next_solver_recovery_plan(
        {
            "cases": [
                _row("hard", "hold_exact_count_scan_quality_limited", target=12, selected=2, safe=2),
            ]
        }
    )

    case = report["cases"][0]
    assert case["category"] == "hard_known_latent_support_creation"
    assert case["next_experiment"] == "typed_latent_support_creation_or_junction_decomposition"
    assert report["primary_next_step"] == "T2_hard_known_latent_support_creation"


def test_partial_limited_support_routes_to_scan_quality_gate():
    report = planner.summarize_next_solver_recovery_plan(
        {
            "cases": [
                _row("partial", "hold_exact_count_scan_quality_limited", target=5, selected=1, safe=2, track="partial"),
            ]
        }
    )

    assert report["cases"][0]["category"] == "scan_quality_or_observed_support_limited"


def test_rolled_semantics_routes_to_transition_semantics():
    report = planner.summarize_next_solver_recovery_plan(
        {
            "cases": [
                _row("rolled", "hold_rolled_mixed_transition_semantics_needed", target=2, selected=1, safe=2, family="rolled"),
            ]
        }
    )

    assert report["cases"][0]["category"] == "rolled_mixed_transition_semantics"
    assert report["tranches"][3]["case_ids"] == ["rolled"]


def test_sparse_candidate_and_abstain_controls_are_separated():
    report = planner.summarize_next_solver_recovery_plan(
        {
            "cases": [
                _row("clean", "candidate_for_sparse_arrangement_review", target=4, selected=4, safe=6),
                _row("exact", "preserve_existing_f1_exact_raw_lane_abstains", target=3, selected=0, safe=0),
            ]
        }
    )

    categories = {row["part_id"]: row["category"] for row in report["cases"]}
    assert categories["clean"] == "sparse_arrangement_review_candidate"
    assert categories["exact"] == "preserve_existing_f1_raw_lane_abstains"


def test_markdown_contains_tranches_and_case_table():
    report = planner.summarize_next_solver_recovery_plan(
        {
            "cases": [
                _row("hard", "hold_exact_count_scan_quality_limited", target=12, selected=2, safe=2),
            ]
        }
    )

    markdown = planner.render_markdown(report)

    assert "T2_hard_known_latent_support_creation" in markdown
    assert "| hard | 12 | 2 | 2 | 10 |" in markdown
