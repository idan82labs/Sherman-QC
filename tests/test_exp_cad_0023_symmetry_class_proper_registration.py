import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0023_symmetry_class_proper_registration.py"
SPEC = importlib.util.spec_from_file_location("exp23", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_strict_event_pass_requires_every_bin_and_residuals():
    good = [{
        "event_id": "E1", "supported_bin_count": 8, "bin_count": 8,
        "radial_tolerance_mm": 1.2, "median_carrier_radial_residual_mm": 0.4,
        "p90_carrier_radial_residual_mm": 1.0,
    }]
    assert MODULE.strict_event_pass(good) == (True, [])
    bad = [dict(good[0], supported_bin_count=7)]
    passed, reasons = MODULE.strict_event_pass(bad)
    assert not passed and "E1:NOT_ALL_BINS_SUPPORTED" in reasons


def test_rank_candidate_prefers_h2_then_l5():
    rows = [
        {"candidate_id": "A", "proper_rigid_transform": True, "strict_native_event_pass": True,
         "full_bidirectional_metrics": {"harmonic_overlap_h2": 0.6, "clipped_bidirectional_loss_l5": 0.3}},
        {"candidate_id": "B", "proper_rigid_transform": True, "strict_native_event_pass": True,
         "full_bidirectional_metrics": {"harmonic_overlap_h2": 0.61, "clipped_bidirectional_loss_l5": 0.4}},
    ]
    assert MODULE.rank_candidate(rows) == "B"
