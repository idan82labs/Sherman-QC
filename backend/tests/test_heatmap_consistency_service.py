from types import SimpleNamespace

import numpy as np

from domains.bend.services.heatmap_consistency_service import (
    ROI_MODE,
    evaluate_bend_heatmap_consistency,
    merge_heatmap_consistency_into_report,
)


class _CadBend(SimpleNamespace):
    pass


class _Match(SimpleNamespace):
    pass


class _Report(SimpleNamespace):
    pass


def _make_match(*, bend_id: str, status: str, measurement_method: str = 'profile_section', feature_type: str = 'DISCRETE_BEND', countable: bool = True):
    cad_bend = _CadBend(
        bend_id=bend_id,
        bend_line_start=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        bend_line_end=np.array([40.0, 0.0, 0.0], dtype=np.float64),
        flange1_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        flange2_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        tolerance_radius=0.5,
        target_radius=1.0,
        feature_type=feature_type,
        countable_in_regression=countable,
    )
    return _Match(cad_bend=cad_bend, status=status, measurement_method=measurement_method)


def _roi_points(count: int = 120) -> np.ndarray:
    xs = np.linspace(-2.0, 42.0, count)
    ys = np.sin(np.linspace(-1.2, 1.2, count)) * 1.5
    zs = np.zeros(count)
    return np.stack([xs, ys, zs], axis=1)


def _center_roi_points(count: int = 160, *, x_lo: float = 12.0, x_hi: float = 28.0, z_offset: float = 0.0) -> np.ndarray:
    xs = np.linspace(x_lo, x_hi, count)
    ys = np.sin(np.linspace(-0.7, 0.7, count)) * 0.9
    zs = np.full(count, z_offset, dtype=np.float64)
    return np.stack([xs, ys, zs], axis=1)


def test_heatmap_consistency_marks_process_features_not_applicable():
    report = _Report(matches=[_make_match(bend_id='CAD_B1', status='PASS', feature_type='ROLLED_SECTION', countable=False)], scan_quality={'density_pts_per_cm2': 20.0})
    points = _roi_points(120)
    deviations = np.full(120, 0.2, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B1']['status'] == 'NOT_APPLICABLE'
    assert bundle['summary']['status'] == 'WEAK_SUPPORT'


def test_heatmap_consistency_supports_quiet_pass_and_merges_into_report():
    report = _Report(matches=[_make_match(bend_id='CAD_B4', status='PASS')], scan_quality={'density_pts_per_cm2': 25.0})
    points = _center_roi_points(160)
    deviations = np.full(160, 0.08, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B4']['status'] == 'SUPPORTED'
    assert bundle['matches']['CAD_B4']['independence_class'] == 'PARTIAL'
    assert bundle['matches']['CAD_B4']['roi_mode'] == ROI_MODE
    merged = merge_heatmap_consistency_into_report({'summary': {}, 'matches': [{'bend_id': 'CAD_B4'}]}, bundle)
    assert merged['matches'][0]['heatmap_consistency']['status'] == 'SUPPORTED'
    assert merged['summary']['heatmap_consistency_summary']['status'] == 'SUPPORTED'


def test_heatmap_consistency_contradicts_quiet_fail_on_full_scan():
    report = _Report(matches=[_make_match(bend_id='CAD_B5', status='FAIL', measurement_method='probe_region')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _roi_points(180)
    deviations = np.full(180, 0.05, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B5']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert bundle['summary']['status'] in {'WEAK_SUPPORT', 'NOT_APPLICABLE', 'SUPPORTED'}


def test_heatmap_consistency_abstains_on_weak_roi_coverage():
    report = _Report(matches=[_make_match(bend_id='CAD_B6', status='PASS')], scan_quality={'density_pts_per_cm2': 5.0})
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    deviations = np.array([0.1, 0.2, 0.1], dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='partial', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B6']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert bundle['summary']['status'] == 'NOT_APPLICABLE'


def test_heatmap_consistency_does_not_infer_full_state_from_filename():
    report = _Report(matches=[_make_match(bend_id='CAD_NAME', status='PASS')], scan_quality={'density_pts_per_cm2': 25.0})
    points = _center_roi_points(160)
    deviations = np.full(160, 0.08, dtype=np.float64)
    bundle = evaluate_bend_heatmap_consistency(
        report,
        points,
        deviations,
        scan_state=None,
        scan_name='full_final_complete.ply',
        global_tolerance_mm=0.5,
    )
    assert bundle['summary']['status'] == 'NOT_APPLICABLE'


def test_heatmap_consistency_never_contradicts_not_detected():
    report = _Report(matches=[_make_match(bend_id='CAD_B7', status='NOT_DETECTED', measurement_method='probe_region')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _roi_points(220)
    deviations = np.full(220, 1.2, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B7']['status'] == 'INSUFFICIENT_EVIDENCE'


def test_heatmap_consistency_never_contradicts_non_independent_pass():
    report = _Report(matches=[_make_match(bend_id='CAD_B8', status='PASS', measurement_method='signed_distance_gradient')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _roi_points(220)
    deviations = np.full(220, 1.4, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B8']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert bundle['matches']['CAD_B8']['independence_class'] == 'NON_INDEPENDENT'


def test_heatmap_consistency_can_contradict_hot_independent_pass_on_full_scan():
    report = _Report(matches=[_make_match(bend_id='CAD_B9', status='PASS', measurement_method='probe_region')], scan_quality={'density_pts_per_cm2': 30.0})
    points = np.concatenate([
        _center_roi_points(160, z_offset=2.0),
        _center_roi_points(80, z_offset=-2.0),
    ], axis=0)
    deviations = np.concatenate([
        np.full(160, 1.4, dtype=np.float64),
        np.full(80, 0.0, dtype=np.float64),
    ])

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B9']['status'] == 'CONTRADICTED'


def test_heatmap_consistency_support_score_matches_relaxed_fail_threshold():
    report = _Report(matches=[_make_match(bend_id='CAD_B10', status='FAIL', measurement_method='profile_section')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _center_roi_points(160)
    deviations = np.full(160, 0.62, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B10']['status'] == 'SUPPORTED'
    assert bundle['matches']['CAD_B10']['consistency_score'] >= 1.0


def test_heatmap_consistency_supports_warning_with_relaxed_active_fail_gate():
    report = _Report(matches=[_make_match(bend_id='CAD_B11', status='WARNING', measurement_method='profile_section')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _center_roi_points(160)
    deviations = np.full(160, 0.62, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B11']['status'] == 'SUPPORTED'
    assert bundle['matches']['CAD_B11']['consistency_score'] >= 1.0


def test_heatmap_consistency_supports_quiet_full_scan_partial_pass():
    report = _Report(matches=[_make_match(bend_id='CAD_B12', status='PASS', measurement_method='surface_classification')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _center_roi_points(160)
    deviations = np.concatenate([
        np.full(144, 0.45, dtype=np.float64),
        np.full(16, 0.60, dtype=np.float64),
    ])

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B12']['status'] == 'SUPPORTED'
    assert bundle['matches']['CAD_B12']['independence_class'] == 'PARTIAL'


def test_heatmap_consistency_supports_full_scan_fail_with_moderate_local_signal():
    report = _Report(matches=[_make_match(bend_id='CAD_B13', status='FAIL', measurement_method='probe_region')], scan_quality={'density_pts_per_cm2': 30.0})
    points = _center_roi_points(180)
    deviations = np.full(180, 0.58, dtype=np.float64)

    bundle = evaluate_bend_heatmap_consistency(report, points, deviations, scan_state='full', global_tolerance_mm=0.5)

    assert bundle['matches']['CAD_B13']['status'] == 'SUPPORTED'
