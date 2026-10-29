import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0022_proper_rigid_orientation_recovery.py"
SPEC = importlib.util.spec_from_file_location("exp22", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _candidate(candidate_id, harmonic, p90, median, proper=True, event_pass=True):
    return {
        "candidate_id": candidate_id,
        "proper_rigid_transform": proper,
        "native_event_qualification_pass": event_pass,
        "overlap_metrics": {
            "harmonic_overlap_fraction": harmonic,
            "combined_p90_distance_mm": p90,
            "combined_median_distance_mm": median,
        },
    }


def test_signed_permutation_pool_is_exactly_so3_24():
    matrices = MODULE.proper_signed_permutation_matrices()
    assert len(matrices) == 24
    assert len({tuple(matrix.ravel()) for matrix in matrices}) == 24
    for matrix in matrices:
        assert np.allclose(matrix.T @ matrix, np.eye(3))
        assert np.linalg.det(matrix) == 1.0


def test_pca_initial_transform_is_proper_and_maps_centers():
    scan_center = np.array([1.0, 2.0, 3.0])
    reference_center = np.array([-4.0, 5.0, 6.0])
    transform = MODULE.pca_initial_transform(
        scan_center,
        np.eye(3),
        reference_center,
        np.eye(3),
        MODULE.proper_signed_permutation_matrices()[0],
    )
    assert np.linalg.det(transform[:3, :3]) > 0
    assert np.allclose(MODULE.EXP21.apply_transform(scan_center[None, :], transform)[0], reference_center)


def test_absolute_selector_never_selects_reflection_or_failed_event_candidate():
    rows = [
        _candidate("GOOD", 0.90, 1.0, 0.5),
        _candidate("REFLECTION", 0.99, 0.1, 0.1, proper=False),
        _candidate("NO_BENDS", 0.98, 0.2, 0.2, event_pass=False),
    ]
    assert MODULE.select_absolute_candidate(rows, legacy_id=None) == "GOOD"


def test_control_retains_legacy_without_material_gain():
    rows = [
        _candidate("LEGACY_PROPER", 0.95, 0.50, 0.30),
        _candidate("TRIAL", 0.951, 0.48, 0.29),
    ]
    assert MODULE.select_absolute_candidate(rows, legacy_id="LEGACY_PROPER") == "LEGACY_PROPER"


def test_control_accepts_material_pareto_gain():
    rows = [
        _candidate("LEGACY_PROPER", 0.95, 0.50, 0.30),
        _candidate("TRIAL", 0.956, 0.49, 0.29),
    ]
    assert MODULE.select_absolute_candidate(rows, legacy_id="LEGACY_PROPER") == "TRIAL"
