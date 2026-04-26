import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_feature_semantic_blockers.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_feature_semantic_blockers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _region(bend_id, atom_count, support_mass, pair=("F1", "F2"), span=20.0):
    return {
        "bend_id": bend_id,
        "owned_atom_ids": [f"A{i}" for i in range(atom_count)],
        "support_mass": support_mass,
        "incident_flange_ids": list(pair),
        "angle_deg": 90.0,
        "span_endpoints": [[0.0, 0.0, 0.0], [span, 0.0, 0.0]],
        "anchor": [0.0, 0.0, 0.0],
        "post_bend_class": "transition_only",
    }


def test_analyzer_reports_blocker_region_roles(tmp_path):
    decomp_dir = tmp_path / "known" / "case"
    decomp_dir.mkdir(parents=True)
    (decomp_dir / "decomposition_result.json").write_text(
        json.dumps(
            {
                "part_id": "10839000",
                "scan_path": "/tmp/10839000_01_full_scan.ply",
                "owned_bend_regions": [
                    _region("OB1", 20, 2000.0, ("F1", "F2"), span=100.0),
                    _region("OB2", 4, 100.0, ("F3", "F4"), span=12.0),
                    _region("OB3", 5, 120.0, ("F3", "F4"), span=10.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = analyzer.analyze_feature_semantic_blockers(
        feature_semantics_audit={
            "blockers": [
                {
                    "part_key": "10839000",
                    "scan_name": "10839000_01_full_scan.ply",
                    "feature_semantics_status": "total_feature_match_but_conventional_miss",
                    "truth_conventional_bends": 8,
                    "truth_raised_form_features": 2,
                    "truth_total_features": 10,
                    "algorithm_range": [9, 10],
                }
            ]
        },
        decomposition_root=tmp_path,
    )

    blocker = report["blockers"][0]
    assert blocker["owned_region_count"] == 3
    assert blocker["diagnostic_summary"]["likely_conventional_bend"] == 1
    assert blocker["diagnostic_summary"]["candidate_raised_form_or_fragment"] == 2
    raised = [row for row in blocker["owned_region_diagnostics"] if row["provisional_role"] == "candidate_raised_form_or_fragment"]
    assert any("duplicate_incident_pair" in row["diagnostic_tags"] for row in raised)


def test_analyzer_reports_missing_decomposition_path(tmp_path):
    report = analyzer.analyze_feature_semantic_blockers(
        feature_semantics_audit={
            "blockers": [
                {
                    "part_key": "missing",
                    "scan_name": "missing.ply",
                    "feature_semantics_status": "conventional_range_miss",
                }
            ]
        },
        decomposition_root=tmp_path,
    )

    blocker = report["blockers"][0]
    assert blocker["decomposition_result_path"] is None
    assert blocker["recommended_next_action"] == "locate decomposition_result.json before geometry diagnosis"
