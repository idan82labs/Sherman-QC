import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_raw_point_creases_against_flange_families.py"
    spec = importlib.util.spec_from_file_location("validate_raw_point_creases_against_flange_families", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validator = _load_module()


def _flange(flange_id, normal, d, atoms):
    return {
        "flange_id": flange_id,
        "normal": list(normal),
        "d": float(d),
        "atom_ids": list(atoms),
        "centroid": [0.0, 0.0, 0.0],
        "confidence": 1.0,
    }


def test_flange_groups_merge_coplanar_fragments_but_not_offset_planes():
    flanges = {
        "F1": _flange("F1", (1.0, 0.0, 0.0), 0.0, ["A1", "A2"]),
        "F2": _flange("F2", (1.0, 0.01, 0.0), 1.5, ["A3"]),
        "F3": _flange("F3", (1.0, 0.0, 0.0), 12.0, ["A4"]),
        "F4": _flange("F4", (0.0, 1.0, 0.0), 0.0, ["A5"]),
    }

    groups = validator._flange_groups(flanges, normal_tol_deg=3.0, offset_tol_mm=3.0)

    assert ["F1", "F2"] in groups
    assert ["F3"] in groups
    assert ["F4"] in groups


def test_family_validator_reports_family_members_for_safe_candidate():
    decomposition = {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {
            "local_spacing_mm": 1.0,
            "atoms": [
                {"atom_id": "L1", "centroid": [0.0, -2.0, 0.0]},
                {"atom_id": "L2", "centroid": [0.0, -2.0, 10.0]},
                {"atom_id": "R1", "centroid": [-2.0, 0.0, 0.0]},
                {"atom_id": "R2", "centroid": [-2.0, 0.0, 10.0]},
            ],
        },
        "flange_hypotheses": [
            _flange("F1", (1.0, 0.0, 0.0), 0.0, ["L1"]),
            _flange("F1b", (1.0, 0.01, 0.0), 1.0, ["L2"]),
            _flange("F2", (0.0, 1.0, 0.0), 0.0, ["R1", "R2"]),
        ],
    }
    raw = {
        "scan_path": "/tmp/demo.ply",
        "candidates": [
            {
                "candidate_id": "RPC1",
                "admissible": True,
                "point_count": 20,
                "span_mm": 20.0,
                "thickness_mm": 1.0,
                "linearity": 0.95,
                "mean_score": 1.0,
                "axis_direction": [0.0, 0.0, 1.0],
                "sampled_points": [[0.0, 0.0, i * 2.0] for i in range(6)],
            }
        ],
    }

    report = validator.validate_raw_point_creases_against_flange_families(
        decomposition=decomposition,
        raw_creases=raw,
        normal_tol_deg=3.0,
        offset_tol_mm=3.0,
    )

    assert report["family_bridge_safe_candidate_count"] == 1
    assert any("F1b" in members for members in report["candidates"][0]["best_flange_family_pair_members"])
