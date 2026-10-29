import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "solve_raw_crease_sparse_arrangement.py"
    spec = importlib.util.spec_from_file_location("solve_raw_crease_sparse_arrangement", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


solver = _load_module()


def _candidate(candidate_id, pair, score=3.0, safe=True):
    return {
        "raw_candidate_id": candidate_id,
        "bridge_safe": safe,
        "best_flange_pair": list(pair),
        "best_pair_score": score,
        "linearity": 0.9,
        "span_mm": 40.0,
        "thickness_mm": 2.0,
    }


def test_sparse_arrangement_suppresses_same_pair_duplicates_and_matches_target():
    bridge = {
        "part_id": "demo",
        "scan_path": "/tmp/demo.ply",
        "candidates": [
            _candidate("A1", ("FG1", "FG2"), 3.5),
            _candidate("A2", ("FG2", "FG1"), 3.2),
            _candidate("B1", ("FG1", "FG3"), 3.4),
        ],
    }

    report = solver.solve_raw_crease_sparse_arrangement(bridge_validation=bridge, target_count=2)

    assert report["status"] == "sparse_arrangement_exact_candidate"
    assert report["selected_family_count"] == 2
    assert report["suppressed_duplicate_count"] == 1
    assert "A1" in report["selected_candidate_ids"]
    assert "A2" in report["suppressed_duplicate_candidate_ids"]
    assert report["families"][0]["source_candidate_ids"] == ["A1", "A2"]
    assert report["families"][0]["merged_family_candidate"]["source_candidate_ids"] == ["A1", "A2"]


def test_sparse_arrangement_underfits_when_unique_pairs_below_target():
    bridge = {
        "candidates": [
            _candidate("A1", ("FG1", "FG2"), 3.5),
            _candidate("A2", ("FG1", "FG2"), 3.4),
        ],
    }

    report = solver.solve_raw_crease_sparse_arrangement(bridge_validation=bridge, target_count=3)

    assert report["status"] == "sparse_arrangement_underfit"
    assert "unique_flange_pair_count_below_target" in report["reason_codes"]


def test_sparse_arrangement_overcomplete_when_unique_pairs_above_target():
    bridge = {
        "candidates": [
            _candidate("A", ("FG1", "FG2")),
            _candidate("B", ("FG1", "FG3")),
            _candidate("C", ("FG1", "FG4")),
        ],
    }

    report = solver.solve_raw_crease_sparse_arrangement(bridge_validation=bridge, target_count=2)

    assert report["status"] == "sparse_arrangement_overcomplete"
    assert "unique_flange_pair_count_above_target" in report["reason_codes"]


def test_sparse_arrangement_ignores_unsafe_candidates():
    bridge = {
        "candidates": [
            _candidate("A", ("FG1", "FG2"), safe=True),
            _candidate("B", ("FG1", "FG3"), safe=False),
        ],
    }

    report = solver.solve_raw_crease_sparse_arrangement(bridge_validation=bridge, target_count=1)

    assert report["status"] == "sparse_arrangement_exact_candidate"
    assert report["selected_candidate_ids"] == ["A"]
