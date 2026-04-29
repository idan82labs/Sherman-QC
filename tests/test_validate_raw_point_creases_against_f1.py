import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_raw_point_creases_against_f1.py"
    spec = importlib.util.spec_from_file_location("validate_raw_point_creases_against_f1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validator = _load_module()


def _atom(atom_id, x, y, z, label):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": [1.0, 0.0, 0.0],
        "curvature_proxy": 0.0,
        "axis_direction": [0.0, 0.0, 1.0],
        "weight": 1.0,
        "_label": label,
    }


def _payload():
    flange_atoms = [
        _atom(f"L{i}", 0.0, -4.0, i * 4.0, "F1") for i in range(8)
    ] + [
        _atom(f"R{i}", -4.0, 0.0, i * 4.0, "F2") for i in range(8)
    ]
    labels = {atom["atom_id"]: atom.pop("_label") for atom in flange_atoms}
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": flange_atoms, "adjacency": {}, "local_spacing_mm": 1.0},
        "flange_hypotheses": [
            {"flange_id": "F1", "normal": [1.0, 0.0, 0.0], "d": 0.0, "atom_ids": [f"L{i}" for i in range(8)]},
            {"flange_id": "F2", "normal": [0.0, 1.0, 0.0], "d": 0.0, "atom_ids": [f"R{i}" for i in range(8)]},
        ],
        "assignment": {"atom_labels": labels, "label_types": {"F1": "flange", "F2": "flange"}},
    }


def test_validate_raw_point_crease_selects_flange_pair_for_clean_line():
    raw = {
        "scan_path": "/tmp/demo.ply",
        "candidates": [
            {
                "candidate_id": "RPC1",
                "admissible": True,
                "point_count": 40,
                "span_mm": 28.0,
                "thickness_mm": 1.0,
                "linearity": 0.95,
                "mean_score": 1.0,
                "axis_direction": [0.0, 0.0, 1.0],
                "sampled_points": [[0.0, 0.0, i * 4.0] for i in range(8)],
            }
        ],
    }

    report = validator.validate_raw_point_creases_against_f1(decomposition=_payload(), raw_creases=raw)

    assert report["status"] == "bridge_candidates_found"
    assert report["bridge_safe_candidate_count"] == 1
    assert report["candidates"][0]["best_flange_pair"] == ["F1", "F2"]


def test_validate_raw_point_crease_rejects_wrong_axis_line():
    raw = {
        "scan_path": "/tmp/demo.ply",
        "candidates": [
            {
                "candidate_id": "RPC1",
                "admissible": True,
                "point_count": 40,
                "span_mm": 28.0,
                "thickness_mm": 1.0,
                "linearity": 0.95,
                "mean_score": 1.0,
                "axis_direction": [1.0, 0.0, 0.0],
                "sampled_points": [[0.0, 0.0, i * 4.0] for i in range(8)],
            }
        ],
    }

    report = validator.validate_raw_point_creases_against_f1(decomposition=_payload(), raw_creases=raw)

    assert report["status"] == "no_bridge_safe_candidate"
    assert "axis_mismatch" in report["candidates"][0]["reason_codes"]
