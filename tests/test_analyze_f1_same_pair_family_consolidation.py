import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_same_pair_family_consolidation.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_same_pair_family_consolidation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, y=0.0, z=0.0):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": [0.0, 1.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "curvature_proxy": 0.04,
        "weight": 10.0,
    }


def _payload(raw_atoms):
    atom_ids = sorted({atom_id for ids in raw_atoms.values() for atom_id in ids})
    atoms = [_atom(atom_id, index * 4.0) for index, atom_id in enumerate(atom_ids)]
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": {}},
        "owned_bend_regions": [
            {
                "bend_id": bend_id,
                "incident_flange_ids": ["F1", "F5"],
                "owned_atom_ids": atom_ids,
            }
            for bend_id, atom_ids in raw_atoms.items()
        ],
    }


def test_recovered_contact_aliases_same_pair_family_when_line_overlaps():
    decomposition = _payload({"OB2": [f"A{i}" for i in range(8)]})
    arrangement = {
        "top_recovered_contact_birth_candidates": [
            {
                "candidate_id": "RCB1",
                "flange_pair": ["F5", "F1"],
                "atom_ids": [f"A{i}" for i in range(2, 8)],
                "duplicate_diagnostics": {"status": "same_pair_line_alias"},
            }
        ]
    }

    report = analyzer.analyze_same_pair_family_consolidation(
        decomposition=decomposition,
        arrangement=arrangement,
        flange_pair=("F1", "F5"),
    )

    assert report["status"] == "recovered_contact_aliases_existing_family"
    assert report["component_count"] == 1
    assert report["components"][0] == ["OB2", "RCB1"]
    assert report["relation_counts"]["alias"] == 1


def test_recovered_contact_is_separate_when_far_from_same_pair_family():
    decomposition = _payload({"OB2": [f"A{i}" for i in range(8)]})
    distant_atoms = [_atom(f"B{i}", 200.0 + i * 4.0, y=30.0) for i in range(8)]
    decomposition["atom_graph"]["atoms"].extend(distant_atoms)
    arrangement = {
        "top_recovered_contact_birth_candidates": [
            {
                "candidate_id": "RCB1",
                "flange_pair": ["F1", "F5"],
                "atom_ids": [atom["atom_id"] for atom in distant_atoms],
                "duplicate_diagnostics": {"status": "unique_recovered_support"},
            }
        ]
    }

    report = analyzer.analyze_same_pair_family_consolidation(
        decomposition=decomposition,
        arrangement=arrangement,
        flange_pair=("F1", "F5"),
    )

    assert report["status"] == "recovered_contact_separate_family"
    assert report["component_count"] == 2
    assert report["relation_counts"]["separable"] == 1


def test_raw_only_same_pair_family_reports_raw_status():
    report = analyzer.analyze_same_pair_family_consolidation(
        decomposition=_payload({"OB2": [f"A{i}" for i in range(8)]}),
        arrangement={"top_recovered_contact_birth_candidates": []},
        flange_pair=("F1", "F5"),
    )

    assert report["status"] == "raw_same_pair_family_only"
    assert report["line_count"] == 1


def test_interior_gap_candidate_can_be_reported_as_separate_family():
    decomposition = _payload({"OB2": [f"A{i}" for i in range(8)]})
    distant_atoms = [_atom(f"C{i}", 160.0 + i * 4.0, y=30.0) for i in range(8)]
    decomposition["atom_graph"]["atoms"].extend(distant_atoms)
    interior_gaps = {
        "candidates": [
            {
                "candidate_id": "IBG1",
                "atom_ids": [atom["atom_id"] for atom in distant_atoms],
            }
        ],
        "merged_hypotheses": [],
    }
    contact_recovery = {
        "candidates": [
            {
                "source_id": "IBG1",
                "best_pair": {
                    "flange_pair": ["F5", "F1"],
                    "status": "recovered_contact_rejected",
                    "score": 1.9,
                    "reason_codes": ["weak_two_sided_balance"],
                },
            }
        ]
    }

    report = analyzer.analyze_same_pair_family_consolidation(
        decomposition=decomposition,
        arrangement={"top_recovered_contact_birth_candidates": []},
        interior_gaps=interior_gaps,
        contact_recovery=contact_recovery,
        flange_pair=("F1", "F5"),
    )

    assert report["status"] == "interior_candidate_separate_family"
    assert report["interior_separate_component_count"] == 1
    assert report["interior_separate_components"] == [["IBG1"]]
