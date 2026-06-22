import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_hard_case_latent_support_gap.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_hard_case_latent_support_gap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gap = _load_module()


def _decomposition(reason_counts=None, birth_reason_counts=None, skipped_birth_reason=None):
    reason_counts = reason_counts or {}
    birth_reason_counts = birth_reason_counts or {}
    skipped = [{"reason": skipped_birth_reason}] if skipped_birth_reason else []
    return {
        "part_id": "hard",
        "exact_bend_count": 2,
        "bend_count_range": [2, 3],
        "candidate_solution_counts": [2, 3],
        "atom_graph": {"atoms": [{"atom_id": "A1"}]},
        "flange_hypotheses": [{"flange_id": "F1"}],
        "bend_hypotheses": [{"bend_id": "B1"}],
        "owned_bend_regions": [{"bend_id": "OB1"}],
        "repair_diagnostics": {
            "scan_family_route": {"route_id": "fragmented_repeat90_interface_birth", "solver_profile": "interface_birth_enabled"},
            "iterations": [
                {
                    "repair_steps": [
                        {
                            "activated_bend_labels": [],
                            "prune_diagnostics": {
                                "fragment_pruned_atoms": 7,
                                "inadmissible_pruned_labels": ["B1", "B2"],
                                "duplicate_pruned_labels": [],
                                "inadmissible_reason_counts": reason_counts,
                            },
                            "selected_pair_prune_diagnostics": {
                                "selected_pair_duplicate_candidates": 3,
                                "selected_pair_duplicate_pruned": [],
                            },
                            "birth_rescue_diagnostics": {
                                "admissibility_reason_counts": birth_reason_counts,
                                "skipped_birth_details": skipped,
                            },
                        }
                    ]
                }
            ],
        },
    }


def _sparse(target=8, selected=4, safe=4):
    return {
        "part_id": "hard",
        "target_count": target,
        "selected_family_count": selected,
        "safe_candidate_count": safe,
        "status": "sparse_arrangement_underfit",
    }


def test_contact_pruned_hard_underfit_routes_to_latent_support_creation():
    report = gap.analyze_case(
        decomposition=_decomposition(
            {
                "contact_imbalance": 4,
                "missing_designated_flange_contact": 4,
            }
        ),
        sparse_arrangement=_sparse(target=8, selected=4, safe=4),
    )

    assert report["gap_classification"] == "latent_support_creation_with_contact_recovery_gap"
    assert "many_candidates_die_on_flange_contact_or_attachment" in report["reason_codes"]
    assert "raw_bridge_safe_support_under_covers_target" in report["reason_codes"]
    assert not report["promotion_safe"]


def test_support_generation_gap_when_raw_safe_support_under_covers_without_contact_pressure():
    report = gap.analyze_case(
        decomposition=_decomposition({}),
        sparse_arrangement=_sparse(target=10, selected=2, safe=2),
    )

    assert report["gap_classification"] == "candidate_support_generation_gap"
    assert report["recommended_next_experiment"] == "add support creation/birth proposals before sparse arrangement selection"


def test_fragmentation_gap_when_safe_support_exists_but_selection_underfits():
    report = gap.analyze_case(
        decomposition=_decomposition({}),
        sparse_arrangement=_sparse(target=6, selected=4, safe=6),
    )

    assert report["gap_classification"] == "same_family_or_fragmentation_gap"
    assert "selected_pair_duplicate_pressure_present" in report["reason_codes"]
    assert "fragment_pruning_present" in report["reason_codes"]


def test_birth_cap_is_reported_as_reason_code():
    report = gap.analyze_case(
        decomposition=_decomposition(
            {"contact_imbalance": 1},
            {"boundary_leakage": 3},
            skipped_birth_reason="active_count_cap",
        ),
        sparse_arrangement=_sparse(target=12, selected=2, safe=2),
    )

    assert report["gap_classification"] == "latent_support_creation_with_contact_recovery_gap"
    assert "birth_recovery_blocked_by_active_count_cap" in report["reason_codes"]


def test_markdown_summary_contains_case_rows():
    case = gap.analyze_case(
        decomposition=_decomposition({"contact_imbalance": 4, "missing_designated_flange_contact": 4}),
        sparse_arrangement=_sparse(target=8, selected=4, safe=4),
    )
    markdown = gap.render_markdown({"generated_at": "now", "cases": [case]})

    assert "Hard Known F1 Latent Support Gap Summary" in markdown
    assert "| hard | 8 | 4 | 4 |" in markdown
