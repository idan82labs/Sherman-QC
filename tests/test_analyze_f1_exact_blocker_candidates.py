import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_exact_blocker_candidates.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_exact_blocker_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, curvature=0.04, centroid=None):
    return {
        "atom_id": atom_id,
        "centroid": centroid or [0.0, 0.0, 0.0],
        "curvature_proxy": curvature,
    }


def _flange(flange_id, normal, d=0.0):
    return {
        "flange_id": flange_id,
        "normal": list(normal),
        "d": d,
        "atom_ids": [],
        "centroid": [0.0, 0.0, 0.0],
        "confidence": 1.0,
        "source_kind": "test",
    }


def _region(bend_id, atoms, centroid, pair=("F1", "F2"), angle=90.0):
    return {
        "bend_id": bend_id,
        "incident_flange_ids": list(pair),
        "owned_atom_ids": list(atoms),
        "support_centroid": list(centroid),
        "axis_direction": [1.0, 0.0, 0.0],
        "span_endpoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "angle_deg": angle,
        "support_mass": float(len(atoms) * 100.0),
    }


def _with_assignment(payload, labels, label_types=None, adjacency=None):
    payload = dict(payload)
    payload["assignment"] = {
        "atom_labels": labels,
        "label_types": label_types or {"F1": "flange", "F2": "flange", "F3": "flange", "B1": "bend", "residual": "residual"},
    }
    payload.setdefault("atom_graph", {})
    payload["atom_graph"] = dict(payload["atom_graph"])
    payload["atom_graph"]["adjacency"] = adjacency or {}
    return payload


def test_candidate_report_ranks_non_duplicate_above_overlapping_duplicate():
    baseline = {
        "exact_bend_count": 1,
        "bend_count_range": [1, 1],
        "candidate_solution_counts": [1],
        "candidate_solution_energies": [10.0],
        "atom_graph": {
            "atoms": [
                _atom("A1"),
                _atom("A2"),
                _atom("A3"),
                _atom("B1"),
                _atom("B2"),
                _atom("B3"),
                _atom("B4"),
                _atom("B5"),
            ]
        },
        "owned_bend_regions": [
            _region("OB1", ["A1", "A2", "A3"], [0.0, 0.0, 0.0]),
        ],
    }
    overcomplete = {
        "exact_bend_count": 2,
        "bend_count_range": [2, 2],
        "candidate_solution_counts": [2],
        "candidate_solution_energies": [12.0],
        "atom_graph": baseline["atom_graph"],
        "owned_bend_regions": [
            _region("DUP", ["A1", "A2", "A3"], [0.5, 0.0, 0.0]),
            _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [40.0, 0.0, 0.0], pair=("F2", "F3")),
        ],
    }

    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )

    assert report["ranked_candidates"][0]["bend_id"] == "EXTRA"
    assert report["ranked_candidates"][0]["reason_codes"] == ["non_duplicate_candidate"]
    assert "likely_duplicate_overlap" in report["ranked_candidates"][1]["reason_codes"]


def test_local_selection_promotes_unique_non_duplicate_candidate():
    baseline = {
        "atom_graph": {
            "atoms": [_atom("A1"), _atom("A2"), _atom("A3"), _atom("B1"), _atom("B2"), _atom("B3"), _atom("B4"), _atom("B5")]
        },
        "owned_bend_regions": [_region("OB1", ["A1", "A2", "A3"], [0.0, 0.0, 0.0])],
    }
    overcomplete = {
        "atom_graph": baseline["atom_graph"],
        "owned_bend_regions": [
            _region("DUP", ["A1", "A2", "A3"], [0.5, 0.0, 0.0]),
            _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
        ],
    }

    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )

    assert selection["selection_status"] == "single_candidate_promotable"
    assert selection["recommended_candidate"]["bend_id"] == "EXTRA"
    assert selection["target_one_birth_count"] == 2

    conflict_set = analyzer.build_conflict_set_report(selection_report=selection)
    assert conflict_set["selection_status"] == "single_birth_structurally_dominant"
    assert conflict_set["best_by_added_count"]["+1"]["bend_ids"] == ["EXTRA"]

    sensitivity = analyzer.build_conflict_sensitivity_report(
        selection_report=selection,
        label_costs=(0.42, 0.70),
        conflict_weights=(0.85,),
    )
    assert sensitivity["dominant_candidate_counts"] == {"EXTRA": 2}
    assert sensitivity["interpretation_status"] == "stable_single_candidate_region_exists"

    coverage = analyzer.build_corridor_coverage_report(selection_report=selection)
    assert coverage["coverage_status"] == "single_corridor_gap"
    assert coverage["corridors"][0]["best_candidate"]["bend_id"] == "EXTRA"

    audited_baseline = _with_assignment(
        baseline,
        labels={"A1": "B1", "A2": "F1", "A3": "F2"},
        adjacency={"A1": ["A2", "A3"], "A2": ["A1"], "A3": ["A1"]},
    )
    quotient = analyzer.build_quotient_corridor_audit_report(
        baseline_payload=audited_baseline,
        coverage_report=coverage,
    )
    assert quotient["audit_status"] == "single_uncovered_corridor"
    assert quotient["uncovered_exact_pair_count"] == 1

    geometry_payload = dict(overcomplete)
    geometry_region = _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [0.0, 0.0, 2.0], pair=("F2", "F3"))
    geometry_region["axis_direction"] = [0.0, 0.0, 1.0]
    geometry_payload["owned_bend_regions"] = [geometry_region]
    geometry_payload["flange_hypotheses"] = [
        _flange("F2", [1.0, 0.0, 0.0]),
        _flange("F3", [0.0, 1.0, 0.0]),
    ]
    geometry_payload["atom_graph"] = {
        "local_spacing_mm": 1.0,
        "atoms": [
            _atom("B1", centroid=[0.0, 0.0, 0.0]),
            _atom("B2", centroid=[0.0, 0.0, 1.0]),
            _atom("B3", centroid=[0.0, 0.0, 2.0]),
            _atom("B4", centroid=[0.0, 0.0, 3.0]),
            _atom("B5", centroid=[0.0, 0.0, 4.0]),
        ],
    }
    comparison = analyzer.build_uncovered_corridor_comparison_report(
        overcomplete_payload=geometry_payload,
        quotient_audit_report=quotient,
    )
    assert comparison["comparison_status"] == "single_uncovered_corridor_dominant"
    assert comparison["recommended_uncovered_corridor"]["bend_id"] == "EXTRA"

    ownership_payload = _with_assignment(
        geometry_payload,
        labels={"B1": "EXTRA", "B2": "EXTRA", "B3": "EXTRA", "B4": "EXTRA", "B5": "EXTRA", "A2": "F2", "A3": "F3"},
        label_types={"F2": "flange", "F3": "flange", "EXTRA": "bend", "residual": "residual"},
        adjacency={
            "B1": ["B2", "A2"],
            "B2": ["B1", "B3", "A2"],
            "B3": ["B2", "B4", "A3"],
            "B4": ["B3", "B5", "A3"],
            "B5": ["B4", "A3"],
            "A2": ["B1", "B2"],
            "A3": ["B3", "B4", "B5"],
        },
    )
    baseline_residual_payload = _with_assignment(
        geometry_payload,
        labels={"B1": "residual", "B2": "residual", "B3": "residual", "B4": "residual", "B5": "residual"},
        label_types={"residual": "residual"},
        adjacency=ownership_payload["atom_graph"]["adjacency"],
    )
    ownership = analyzer.build_boundary_residual_ownership_report(
        baseline_payload=baseline_residual_payload,
        overcomplete_payload=ownership_payload,
        quotient_audit_report=quotient,
    )
    assert ownership["ownership_status"] == "single_boundary_residual_candidate"
    assert ownership["candidates"][0]["reason_codes"] == ["residual_support_with_clean_boundary"]


def test_quotient_audit_blocks_when_candidate_pair_is_already_covered():
    baseline = {
        "atom_graph": {
            "atoms": [_atom("A1"), _atom("A2"), _atom("A3"), _atom("B1"), _atom("B2"), _atom("B3"), _atom("B4"), _atom("B5")]
        },
        "owned_bend_regions": [
            _region("OB1", ["A1", "A2", "A3"], [0.0, 0.0, 0.0]),
            _region("OB2", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
        ],
    }
    overcomplete = {
        "atom_graph": baseline["atom_graph"],
        "owned_bend_regions": [
            _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
        ],
    }
    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )
    coverage = analyzer.build_corridor_coverage_report(selection_report=selection)
    quotient = analyzer.build_quotient_corridor_audit_report(
        baseline_payload=baseline,
        coverage_report=coverage,
    )

    assert quotient["audit_status"] == "hold_exact_promotion"
    assert quotient["uncovered_exact_pair_count"] == 0


def test_flange_family_incidence_marks_raw_alias_as_family_covered():
    baseline = {
        "atom_graph": {"atoms": [_atom("A1"), _atom("A2"), _atom("B1"), _atom("B2"), _atom("B3"), _atom("B4"), _atom("B5")]},
        "owned_bend_regions": [
            _region("BASE", ["A1", "A2"], [0.0, 0.0, 0.0], pair=("F1", "F3")),
        ],
    }
    overcomplete = {
        "atom_graph": baseline["atom_graph"],
        "flange_hypotheses": [
            _flange("F1", [1.0, 0.0, 0.0], d=0.0),
            _flange("F2", [1.0, 0.0, 0.0], d=1.0),
            _flange("F3", [0.0, 1.0, 0.0], d=0.0),
        ],
        "owned_bend_regions": [
            _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
        ],
    }
    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )
    coverage = analyzer.build_corridor_coverage_report(selection_report=selection)
    quotient = analyzer.build_quotient_corridor_audit_report(
        baseline_payload=baseline,
        coverage_report=coverage,
    )
    family = analyzer.build_flange_family_report(payload=overcomplete)
    family_incidence = analyzer.build_flange_family_incidence_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        quotient_audit_report=quotient,
        family_report=family,
    )

    assert family["flange_to_family"]["F1"] == family["flange_to_family"]["F2"]
    assert family_incidence["candidates"][0]["family_pair_uncovered"] is False
    assert "family_pair_already_covered_by_baseline" in family_incidence["candidates"][0]["reason_codes"]


def test_family_residual_repair_evidence_accepts_single_uncovered_residual_family_gap():
    baseline = {
        "part_id": "demo",
        "exact_bend_count": 1,
        "atom_graph": {"atoms": [_atom("A1"), _atom("A2"), _atom("B1"), _atom("B2"), _atom("B3"), _atom("B4"), _atom("B5")]},
        "owned_bend_regions": [
            _region("BASE", ["A1", "A2"], [0.0, 0.0, 0.0], pair=("F1", "F3")),
        ],
    }
    overcomplete = {
        "part_id": "demo",
        "atom_graph": baseline["atom_graph"],
        "flange_hypotheses": [
            _flange("F1", [1.0, 0.0, 0.0], d=0.0),
            _flange("F2", [1.0, 0.0, 0.0], d=1.0),
            _flange("F3", [0.0, 1.0, 0.0], d=0.0),
            _flange("F4", [0.0, 0.0, 1.0], d=0.0),
        ],
        "owned_bend_regions": [
            _region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F4")),
        ],
    }
    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )
    coverage = analyzer.build_corridor_coverage_report(selection_report=selection)
    quotient = analyzer.build_quotient_corridor_audit_report(
        baseline_payload=baseline,
        coverage_report=coverage,
    )
    family = analyzer.build_flange_family_report(payload=overcomplete)
    family_incidence = analyzer.build_flange_family_incidence_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        quotient_audit_report=quotient,
        family_report=family,
    )
    baseline_residual = _with_assignment(
        baseline,
        labels={"B1": "residual", "B2": "residual", "B3": "residual", "B4": "residual", "B5": "residual"},
        label_types={"residual": "residual"},
    )
    overcomplete_owned = _with_assignment(
        overcomplete,
        labels={"B1": "EXTRA", "B2": "EXTRA", "B3": "EXTRA", "B4": "EXTRA", "B5": "EXTRA"},
        label_types={"EXTRA": "bend", "residual": "residual"},
    )
    ownership = analyzer.build_boundary_residual_ownership_report(
        baseline_payload=baseline_residual,
        overcomplete_payload=overcomplete_owned,
        quotient_audit_report=quotient,
    )

    evidence = analyzer.build_family_residual_repair_promotion_evidence(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        family_incidence_report=family_incidence,
        ownership_report=ownership,
    )

    assert evidence["family_residual_repair_eligible"] is True
    assert evidence["promoted_exact_count"] == 2
    assert evidence["candidate_bend_id"] == "EXTRA"
    assert evidence["checks"]["baseline_residual_fraction"] == 1.0
    assert evidence["supplemental_owned_bend_regions"][0]["bend_id"] == "OB2"


def test_family_residual_repair_evidence_blocks_non_residual_baseline_support():
    baseline = {
        "part_id": "demo",
        "exact_bend_count": 1,
        "owned_bend_regions": [_region("BASE", ["A1", "A2"], [0.0, 0.0, 0.0], pair=("F1", "F3"))],
    }
    overcomplete = {
        "part_id": "demo",
        "owned_bend_regions": [_region("EXTRA", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F4"))],
    }
    family_incidence = {
        "family_incidence_status": "single_uncovered_flange_family_pair",
        "candidates": [
            {
                "bend_id": "EXTRA",
                "corridor_id": "C1",
                "family_pair_uncovered": True,
                "reason_codes": ["family_pair_uncovered"],
            }
        ],
    }
    ownership = {
        "candidates": [
            {
                "bend_id": "EXTRA",
                "baseline_label_distribution": {
                    "type_fractions": {"flange": 1.0}
                },
            }
        ]
    }

    evidence = analyzer.build_family_residual_repair_promotion_evidence(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        family_incidence_report=family_incidence,
        ownership_report=ownership,
    )

    assert evidence["family_residual_repair_eligible"] is False
    assert "candidate_baseline_support_not_residual_owned" in evidence["blockers"]


def test_local_selection_holds_when_top_candidates_are_not_unique():
    baseline = {
        "atom_graph": {
            "atoms": [
                _atom("A1"),
                _atom("A2"),
                _atom("A3"),
                _atom("B1"),
                _atom("B2"),
                _atom("B3"),
                _atom("B4"),
                _atom("B5"),
                _atom("C1"),
                _atom("C2"),
                _atom("C3"),
                _atom("C4"),
                _atom("C5"),
            ]
        },
        "owned_bend_regions": [_region("OB1", ["A1", "A2", "A3"], [0.0, 0.0, 0.0])],
    }
    overcomplete = {
        "atom_graph": baseline["atom_graph"],
        "owned_bend_regions": [
            _region("EXTRA_A", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
            _region("EXTRA_B", ["C1", "C2", "C3", "C4", "C5"], [82.0, 0.0, 0.0], pair=("F4", "F5")),
        ],
    }

    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )

    assert selection["selection_status"] == "hold_exact_promotion"
    assert "top_candidate_not_unique_enough" in selection["hold_reasons"]

    conflict_set = analyzer.build_conflict_set_report(selection_report=selection)
    assert conflict_set["selection_status"] == "hold_exact_promotion"
    assert "one_birth_subset_not_dominant" in conflict_set["hold_reasons"]

    sensitivity = analyzer.build_conflict_sensitivity_report(
        selection_report=selection,
        label_costs=(0.42,),
        conflict_weights=(0.85,),
    )
    assert sensitivity["interpretation_status"] == "no_reasonable_sweep_promotes_exact_12"


def test_corridor_coverage_holds_when_conflicting_fragments_share_gap():
    baseline = {
        "atom_graph": {
            "atoms": [
                _atom("A1"),
                _atom("A2"),
                _atom("A3"),
                _atom("B1"),
                _atom("B2"),
                _atom("B3"),
                _atom("B4"),
                _atom("B5"),
                _atom("C1"),
                _atom("C2"),
                _atom("C3"),
                _atom("C4"),
                _atom("C5"),
            ]
        },
        "owned_bend_regions": [_region("OB1", ["A1", "A2", "A3"], [0.0, 0.0, 0.0])],
    }
    overcomplete = {
        "atom_graph": baseline["atom_graph"],
        "owned_bend_regions": [
            _region("EXTRA_A", ["B1", "B2", "B3", "B4", "B5"], [80.0, 0.0, 0.0], pair=("F2", "F3")),
            _region("EXTRA_B", ["C1", "C2", "C3", "C4", "C5"], [81.0, 0.0, 0.0], pair=("F4", "F5")),
        ],
    }
    report = analyzer.build_candidate_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        baseline_label="baseline",
        overcomplete_label="overcomplete",
    )
    selection = analyzer.build_local_selection_report(
        baseline_payload=baseline,
        overcomplete_payload=overcomplete,
        candidate_report=report,
    )
    coverage = analyzer.build_corridor_coverage_report(selection_report=selection)

    assert coverage["coverage_status"] == "hold_exact_promotion"
    assert "internal_corridor_conflicts" in coverage["hold_reasons"]
