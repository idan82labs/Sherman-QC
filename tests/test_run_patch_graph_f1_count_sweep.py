import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_patch_graph_f1_count_sweep.py"
    spec = importlib.util.spec_from_file_location("run_patch_graph_f1_count_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_module()


def test_parse_caps_deduplicates_and_preserves_order():
    assert runner._parse_caps("5,6,5,8") == [5, 6, 8]


def test_variant_overrides_marks_probe_as_non_default():
    overrides = runner._variant_overrides(6)

    assert overrides["upper_tail_probe"] is True
    assert overrides["limit_birth_rescue_to_control_upper"] is False
    assert overrides["birth_rescue_max_active_bends"] == 6


def test_variant_summary_reports_energy_delta():
    payload = {
        "raw_f1_candidate": {
            "map_bend_count": 5,
            "exact_bend_count": 5,
            "bend_count_range": [5, 5],
            "solution_counts": [5, 5, 5],
            "solution_energies": [12.0, 18.0],
            "owned_bend_region_count": 5,
        },
        "candidate": {
            "bend_count_range": [5, 16],
            "candidate_source": "control_constrained_f1",
            "guard_reason": "singleton_below_control_upper",
        },
        "raw_f1_promotion_diagnostic": {
            "candidate": False,
            "blockers": ["diagnostic_upper_tail_probe_run"],
        },
    }

    row = runner._variant_summary(variant_id="cap_5", cap=5, payload=payload, base_energy=10.0)

    assert row["raw_bend_count_range"] == [5, 5]
    assert row["energy_delta_vs_baseline"] == 2.0
    assert row["promotion_blockers"] == ["diagnostic_upper_tail_probe_run"]
