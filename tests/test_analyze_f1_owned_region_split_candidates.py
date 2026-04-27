import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_owned_region_split_candidates.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_owned_region_split_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x):
    return {"atom_id": atom_id, "centroid": [float(x), 0.0, 0.0]}


def _region(bend_id, atom_ids):
    return {
        "bend_id": bend_id,
        "owned_atom_ids": atom_ids,
        "axis_direction": [1.0, 0.0, 0.0],
        "anchor": [0.0, 0.0, 0.0],
        "support_mass": 100.0,
        "incident_flange_ids": ["F1", "F2"],
        "angle_deg": 90.0,
    }


def test_split_candidate_detects_balanced_large_gap():
    left = [_atom(f"L{i}", i * 3.0) for i in range(5)]
    right = [_atom(f"R{i}", 60.0 + i * 3.0) for i in range(5)]
    atoms = left + right
    report = analyzer.analyze_owned_region_split_candidates(
        decomposition={
            "scan_path": "/tmp/demo.ply",
            "part_id": "demo",
            "atom_graph": {"atoms": atoms},
            "owned_bend_regions": [_region("OB1", [atom["atom_id"] for atom in atoms])],
        },
        feature_label_summary={
            "label_capacity": {
                "owned_region_capacity_deficit": 1,
            }
        },
    )

    assert report["status"] == "split_candidate_available"
    assert report["top_split_candidate_bend_ids"] == ["OB1"]
    assert report["regions"][0]["balanced_split"] is True
    assert report["regions"][0]["split_candidate_rank"] == "strong"


def test_missing_region_status_when_no_split_candidate_for_deficit():
    atoms = [_atom(f"A{i}", i * 3.0) for i in range(10)]
    report = analyzer.analyze_owned_region_split_candidates(
        decomposition={
            "atom_graph": {"atoms": atoms},
            "owned_bend_regions": [_region("OB1", [atom["atom_id"] for atom in atoms])],
        },
        feature_label_summary={"label_capacity": {"owned_region_capacity_deficit": 1}},
    )

    assert report["status"] == "missing_region_recovery_needed"
    assert report["split_candidate_count"] == 0


def test_conventional_region_deficit_drives_split_recovery_even_when_raw_count_is_not_short():
    left = [_atom(f"L{i}", i * 3.0) for i in range(5)]
    right = [_atom(f"R{i}", 60.0 + i * 3.0) for i in range(5)]
    atoms = left + right
    report = analyzer.analyze_owned_region_split_candidates(
        decomposition={
            "scan_path": "/tmp/demo.ply",
            "part_id": "demo",
            "atom_graph": {"atoms": atoms},
            "owned_bend_regions": [_region("OB1", [atom["atom_id"] for atom in atoms])],
        },
        feature_label_summary={
            "label_capacity": {
                "owned_region_capacity_deficit": 0,
                "valid_conventional_region_deficit": 2,
                "valid_counted_feature_deficit": 2,
            }
        },
    )

    assert report["status"] == "split_candidate_available"
    assert report["capacity_deficit"] == 2
    assert report["owned_region_capacity_deficit"] == 0
    assert report["valid_conventional_region_deficit"] == 2
    assert "valid_conventional_region_deficit" in report["deficit_basis"]
    assert report["top_split_candidate_bend_ids"] == ["OB1"]


def test_no_capacity_deficit_status():
    atoms = [_atom(f"A{i}", i * 3.0) for i in range(10)]
    report = analyzer.analyze_owned_region_split_candidates(
        decomposition={
            "atom_graph": {"atoms": atoms},
            "owned_bend_regions": [_region("OB1", [atom["atom_id"] for atom in atoms])],
        },
        feature_label_summary={"label_capacity": {"owned_region_capacity_deficit": 0}},
    )

    assert report["status"] == "no_capacity_deficit"
