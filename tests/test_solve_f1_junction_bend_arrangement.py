import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "solve_f1_junction_bend_arrangement.py"
    spec = importlib.util.spec_from_file_location("solve_f1_junction_bend_arrangement", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


solver = _load_module()


def _atom(atom_id, x, normal=(0.707, 0.707, 0.0), label="residual"):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), 0.0, 0.0],
        "normal": list(normal),
        "axis_direction": [0.0, 0.0, 1.0],
        "curvature_proxy": 0.05,
        "weight": 10.0,
        "_label": label,
    }


def _payload(atoms):
    labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {atom_id: [] for atom_id in atom_ids}
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": adjacency},
        "flange_hypotheses": [
            {"flange_id": "F11", "normal": [1.0, 0.0, 0.0], "d": 0.0, "atom_ids": []},
            {"flange_id": "F14", "normal": [0.0, 1.0, 0.0], "d": 0.0, "atom_ids": []},
            {"flange_id": "F17", "normal": [0.0, 1.0, 0.0], "d": 0.0, "atom_ids": []},
        ],
        "owned_bend_regions": [
            {
                "bend_id": "OB2",
                "incident_flange_ids": ["F11", "F14"],
                "owned_atom_ids": atom_ids[:3],
            }
        ],
        "assignment": {
            "atom_labels": labels,
            "label_types": {"residual": "residual", "F11": "flange", "F14": "flange", "F17": "flange"},
        },
    }


def test_junction_arrangement_keeps_locked_region_and_scores_candidates():
    atoms = [_atom(f"A{i}", i * 3.0) for i in range(8)]
    payload = _payload(atoms)
    ridge = {
        "ridges": [{"atom_ids": [atom["atom_id"] for atom in atoms]}],
        "atom_scores": [{"atom_id": atom["atom_id"], "ridge_score": 0.8} for atom in atoms],
    }

    report = solver.solve_junction_bend_arrangement(
        decomposition=payload,
        feature_labels={"region_labels": [{"bend_id": "OB2", "human_feature_label": "conventional_bend"}]},
        local_ridge_payload=ridge,
        flange_ids=["F11", "F14", "F17"],
    )

    assert report["locked_accepted_pair"] == ["F11", "F14"]
    assert report["candidate_count"] >= 1
    assert any(candidate["locked_accepted"] for candidate in report["candidates"])
    assert report["best_hypothesis"] is not None


def test_subset_score_penalizes_overlapping_candidates():
    lhs = {"candidate_id": "A", "candidate_score": 4.0, "atom_ids": ["1", "2", "3"], "locked_accepted": False}
    rhs = {"candidate_id": "B", "candidate_score": 4.0, "atom_ids": ["2", "3", "4"], "locked_accepted": False}

    row = solver._subset_score([lhs, rhs])

    assert row["overlap_penalty"] > 0
    assert row["conflicts"]


def test_chord_candidates_are_not_countable_without_unique_support():
    atoms = [_atom(f"A{i}", i * 3.0, normal=(0.707, 0.707, 0.0)) for i in range(12)]
    payload = _payload(atoms)
    payload["flange_hypotheses"].append({"flange_id": "F1", "normal": [0.0, 0.0, 1.0], "d": 0.0, "atom_ids": []})
    atom_lookup = solver._atom_lookup(payload)
    flange_lookup = solver._flange_lookup(payload)
    atom_labels, label_types = solver._assignment(payload)
    candidate = solver._line_candidate(
        candidate_id="JBLX",
        pair=("F1", "F11"),
        atom_ids=[atom["atom_id"] for atom in atoms],
        atoms=atom_lookup,
        flanges=flange_lookup,
        adjacency=payload["atom_graph"]["adjacency"],
        atom_labels=atom_labels,
        label_types=label_types,
        locked_pair=("F11", "F14"),
    )

    assert not candidate["admissible"]
    assert "possible_chord" in candidate["reason_codes"]


def test_best_hypothesis_exports_junction_transition_atoms():
    atoms = [_atom(f"A{i}", i * 3.0) for i in range(10)]
    payload = _payload(atoms)
    ridge = {
        "ridges": [{"atom_ids": [atom["atom_id"] for atom in atoms]}],
        "atom_scores": [{"atom_id": atom["atom_id"], "ridge_score": 0.8} for atom in atoms],
    }

    report = solver.solve_junction_bend_arrangement(
        decomposition=payload,
        feature_labels={"region_labels": [{"bend_id": "OB2", "human_feature_label": "conventional_bend"}]},
        local_ridge_payload=ridge,
        flange_ids=["F11", "F14", "F17"],
    )

    assert "junction_transition_atom_ids" in report
    assert report["junction_transition_atom_count"] == len(report["junction_transition_atom_ids"])


def test_target_new_bends_limits_subset_size():
    first = {"candidate_id": "A", "candidate_score": 5.0, "atom_ids": ["1", "2"], "locked_accepted": False}
    second = {"candidate_id": "B", "candidate_score": 4.9, "atom_ids": ["3", "4"], "locked_accepted": False}
    row = solver._subset_score([first, second])
    assert row["new_bend_count"] == 2

    atoms = [_atom(f"A{i}", i * 3.0) for i in range(12)]
    payload = _payload(atoms)
    ridge = {
        "ridges": [{"atom_ids": [atom["atom_id"] for atom in atoms]}],
        "atom_scores": [{"atom_id": atom["atom_id"], "ridge_score": 0.8} for atom in atoms],
    }
    report = solver.solve_junction_bend_arrangement(
        decomposition=payload,
        feature_labels={"region_labels": []},
        local_ridge_payload=ridge,
        flange_ids=["F11", "F14", "F17"],
        target_new_bends=1,
    )

    assert report["target_new_bends"] == 1
    assert report["best_hypothesis"]["new_bend_count"] <= 1
