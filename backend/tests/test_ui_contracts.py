from domains.bend.mappers.ui_contracts import map_bend_report_to_contract


def test_map_bend_report_to_contract_preserves_heatmap_consistency_fields():
    report = {
        'summary': {
            'total_bends': 2,
            'completed_bends': 1,
            'completed_in_spec': 1,
            'completed_out_of_spec': 0,
            'remaining_bends': 1,
            'heatmap_consistency_summary': {
                'status': 'SUPPORTED',
                'inconsistent_bend_count': 0,
                'insufficient_evidence_bend_count': 1,
                'evaluable_bend_count': 1,
                'bend_roi_energy_share': 0.12,
                'global_in_tolerance_rate': 0.98,
                'notes': ['ok'],
            },
        },
        'matches': [
            {
                'bend_id': 'CAD_B4',
                'status': 'PASS',
                'expected_angle': 22.45,
                'heatmap_consistency': {
                'status': 'SUPPORTED',
                'consistency_score': 0.82,
                'independence_class': 'PARTIAL',
                'roi_mode': 'bend_center_local_v2',
                'axial_half_window_mm': 10.0,
                'radial_radius_mm': 4.0,
                'roi_point_count': 120,
                'roi_coverage_ratio': 0.78,
                'local_abs_mean_mm': 0.11,
                    'local_abs_p95_mm': 0.24,
                    'local_out_of_tol_ratio': 0.0,
                    'side_asymmetry_score': 0.08,
                    'signal_extent_along_flange': 0.62,
                    'notes': ['quiet'],
                },
            }
        ],
    }

    contract = map_bend_report_to_contract('PART', report)

    assert contract.summary.heatmap_consistency_summary is not None
    assert contract.summary.heatmap_consistency_summary.status == 'SUPPORTED'
    assert contract.matches[0].heatmap_consistency is not None
    assert contract.matches[0].heatmap_consistency.status == 'SUPPORTED'
    assert contract.matches[0].heatmap_consistency.roi_mode == 'bend_center_local_v2'


def test_map_bend_report_to_contract_preserves_report_alignment_payload():
    report = {
        'alignment': {
            'alignment_source': 'PLANE_DATUM_REFINEMENT',
            'parent_island_id': 'PRIMARY_PLANE',
            'alignment_confidence': 0.91,
            'alignment_abstained': False,
            'alignment_conditioning': {'datum_scan_point_count': 2400},
        },
        'summary': {'total_bends': 1},
        'matches': [{'bend_id': 'CAD_B1', 'status': 'NOT_DETECTED'}],
    }

    contract = map_bend_report_to_contract('PART', report)

    assert contract.alignment is not None
    assert contract.alignment['alignment_source'] == 'PLANE_DATUM_REFINEMENT'
