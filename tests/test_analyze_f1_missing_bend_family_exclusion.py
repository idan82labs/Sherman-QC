import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_missing_bend_family_exclusion.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_missing_bend_family_exclusion", path)
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


def _decomposition(raw_ids, extra_atoms=()):
    atoms = [_atom(atom_id, index * 4.0) for index, atom_id in enumerate(raw_ids)]
    atoms.extend(extra_atoms)
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms},
        "owned_bend_regions": [
            {
                "bend_id": "OB1",
                "incident_flange_ids": ["F1", "F5"],
                "owned_atom_ids": list(raw_ids),
            }
        ],
    }


def _interior(source_id, atom_ids):
    return {
        "candidates": [
            {
                "candidate_id": source_id,
                "source_kind": "interior_fragment",
                "atom_ids": list(atom_ids),
            }
        ],
        "merged_hypotheses": [],
    }


def _contact(source_id, *, status="recovered_contact_validated"):
    return {
        "candidates": [
            {
                "source_id": source_id,
                "best_pair": {
                    "flange_pair": ["F5", "F1"],
                    "status": status,
                    "score": 3.0,
                    "reason_codes": ["recovered_two_sided_contact"] if status == "recovered_contact_validated" else ["weak_two_sided_balance"],
                },
            }
        ]
    }


def test_family_exclusion_blocks_same_pair_alias_candidate():
    raw_ids = [f"A{i}" for i in range(8)]
    report = analyzer.analyze_missing_bend_family_exclusion(
        decomposition=_decomposition(raw_ids),
        interior_gaps=_interior("IBG1", raw_ids[2:]),
        contact_recovery=_contact("IBG1"),
    )

    assert report["status"] == "no_separable_missing_bend_candidate"
    assert report["family_exclusion_pass_count"] == 0
    assert report["candidates"][0]["blockers"] == ["same_pair_alias_to_raw_family"]


def test_family_exclusion_allows_separable_valid_candidate():
    raw_ids = [f"A{i}" for i in range(8)]
    extra = [_atom(f"B{i}", 200.0 + i * 4.0, y=40.0) for i in range(8)]
    report = analyzer.analyze_missing_bend_family_exclusion(
        decomposition=_decomposition(raw_ids, extra_atoms=extra),
        interior_gaps=_interior("IBG1", [atom["atom_id"] for atom in extra]),
        contact_recovery=_contact("IBG1"),
    )

    assert report["status"] == "separable_missing_bend_candidates_found"
    assert report["family_exclusion_pass_count"] == 1
    assert report["candidates"][0]["family_exclusion_status"] == "separable_missing_bend_candidate"


def test_family_exclusion_blocks_contact_rejected_even_when_separable():
    raw_ids = [f"A{i}" for i in range(8)]
    extra = [_atom(f"B{i}", 200.0 + i * 4.0, y=40.0) for i in range(8)]
    report = analyzer.analyze_missing_bend_family_exclusion(
        decomposition=_decomposition(raw_ids, extra_atoms=extra),
        interior_gaps=_interior("IBG1", [atom["atom_id"] for atom in extra]),
        contact_recovery=_contact("IBG1", status="recovered_contact_rejected"),
    )

    assert report["status"] == "no_separable_missing_bend_candidate"
    assert report["candidates"][0]["blockers"] == ["contact_not_validated"]
