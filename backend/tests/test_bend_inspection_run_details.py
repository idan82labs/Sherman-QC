from bend_inspection_pipeline import BendInspectionRunDetails


def test_run_details_to_dict_includes_dense_local_refinement_diagnostics():
    details = BendInspectionRunDetails(
        cad_source="cad_import",
        cad_strategy="detector_mesh",
        cad_vertex_count=100,
        cad_triangle_count=200,
        scan_point_count=300,
        cad_bend_count=7,
        scan_bend_count=6,
        expected_bend_count=7,
        expected_progress_pct=85.7142857,
        overdetected_count=0,
        warnings=["example"],
        scan_state="full",
        scan_quality_status="GOOD",
        scan_coverage_pct=91.234,
        scan_density_pts_per_cm2=18.765,
        dense_local_refinement_decision="merged_over_primary_and_local",
        dense_local_refinement_report_ranks={
            "primary": [6, 2, 0, -3, -1],
            "local": [6, 1, 0, -2, -2],
            "merged": [7, 2, 0, -2, -1],
        },
        dense_local_refinement_geometry_penalties={
            "primary": 44.2,
            "local": 12.1,
            "merged": 10.5,
        },
        profile_mode="cad_nominal",
        nominal_source_class="semantic_nominal_cad",
        profile_confidence=1.0,
        cad_structural_target={"expected_bend_count": 7},
        scan_part_profile={
            "profile_mode": "scan_zero_shot",
            "nominal_source_class": "scan_only",
            "scan_hypothesis": {"estimated_bend_count": 6},
        },
        scan_structural_hypothesis={"estimated_bend_count": 6, "bend_support_strips": [{"strip_id": "S1"}]},
        profile_selector_result={"mode": "cad_nominal"},
        local_alignment_seed=101,
        local_alignment_fitness=0.89573,
        local_alignment_rmse=0.89231,
    )

    payload = details.to_dict()

    assert payload["dense_local_refinement_decision"] == "merged_over_primary_and_local"
    assert payload["dense_local_refinement_report_ranks"]["merged"] == [7, 2, 0, -2, -1]
    assert payload["dense_local_refinement_geometry_penalties"]["local"] == 12.1
    assert payload["local_alignment_seed"] == 101
    assert payload["local_alignment_fitness"] == 0.8957
    assert payload["local_alignment_rmse"] == 0.8923
    assert payload["profile_mode"] == "cad_nominal"
    assert payload["nominal_source_class"] == "semantic_nominal_cad"
    assert payload["cad_structural_target"]["expected_bend_count"] == 7
    assert payload["scan_part_profile"]["profile_mode"] == "scan_zero_shot"
    assert payload["scan_structural_hypothesis"]["estimated_bend_count"] == 6
