import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_f1_determinism.py"
SPEC = importlib.util.spec_from_file_location("audit_f1_determinism", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_fingerprint_ignores_timing_cache_and_path_metadata():
    base = {
        "scan_path": "/a/scan.ply",
        "exact_bend_count": 2,
        "bend_count_range": [2, 2],
        "assignment": {"atom_labels": {"A1": "B1"}},
        "owned_bend_regions": [{"bend_id": "OB1"}],
        "repair_diagnostics": {
            "cache_hit": False,
            "stage_timings_seconds": {"solve": 1.0},
            "solver_options": {"route_mode": "apply"},
        },
    }
    changed_metadata = {
        **base,
        "scan_path": "/different/scan.ply",
        "repair_diagnostics": {
            **base["repair_diagnostics"],
            "cache_hit": True,
            "stage_timings_seconds": {"solve": 99.0},
        },
    }

    assert audit._result_fingerprint(base) == audit._result_fingerprint(changed_metadata)


def test_fingerprint_changes_when_owned_result_changes():
    first = {
        "exact_bend_count": 1,
        "bend_count_range": [1, 1],
        "assignment": {"atom_labels": {"A1": "B1"}},
        "owned_bend_regions": [{"bend_id": "OB1"}],
    }
    second = {
        **first,
        "owned_bend_regions": [{"bend_id": "OB1"}, {"bend_id": "OB2"}],
    }

    assert audit._result_fingerprint(first) != audit._result_fingerprint(second)
