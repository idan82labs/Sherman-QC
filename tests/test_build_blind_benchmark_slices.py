import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_blind_benchmark_slices.py"
SPEC = importlib.util.spec_from_file_location("build_blind_benchmark_slices", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_blind_slices_use_visible_truth_and_require_cad_for_cad_lane(tmp_path):
    scan = tmp_path / "scan.ply"
    cad = tmp_path / "part.stp"
    scan.write_bytes(b"scan")
    cad.write_bytes(b"cad")
    registry = {
        "schema_version": 2,
        "semantic_contract": {"primary_count": "visible_observed_conventional_bends"},
        "records": [
            {
                "record_id": "P1::scan.ply",
                "part_key": "P1",
                "scan_state": "partial",
                "truth_status": "manual_verified",
                "release_metric_eligible": True,
                "count_target": "visible_observed_conventional_bends",
                "counts": {"observed_conventional_bends": 3, "full_part_conventional_bends": 7, "raised_form_features": 1},
                "provenance": {"source_type": "manual_visual_review"},
                "source_geometry": {
                    "scan_available": True,
                    "resolved_scan_path": str(scan),
                    "resolved_cad_paths": [str(cad)],
                },
            }
        ],
    }

    ply, cad_ply = MODULE.build_blind_slices(registry)

    assert ply["entries"][0]["expectation"]["expected_bend_count"] == 3
    assert ply["entries"][0]["expectation"]["known_whole_part_total_context"] == 7
    assert ply["entries"][0]["expectation"]["scan_limited"] is True
    assert cad_ply["entries"][0]["inference_mode"] == "blind_cad_ply"


def test_blind_slices_reject_cad_feature_history_even_if_misflagged_release_eligible(tmp_path):
    scan = tmp_path / "not_a_measured_scan.stl"
    scan.write_bytes(b"cad tessellation")
    registry = {
        "schema_version": 2,
        "semantic_contract": {"primary_count": "visible_observed_conventional_bends"},
        "records": [
            {
                "record_id": "P2::not_a_measured_scan.stl",
                "part_key": "P2",
                "scan_state": "full",
                "truth_status": "cad_feature_history_candidate_only",
                "release_metric_eligible": True,
                "count_target": "native_cad_bend_features_all_families_not_conventional_bend_count",
                "counts": {"observed_conventional_bends": 4},
                "provenance": {"source_type": "native_sldprt_feature_history"},
                "source_geometry": {
                    "scan_available": True,
                    "resolved_scan_path": str(scan),
                    "resolved_cad_paths": [],
                },
            }
        ],
    }

    try:
        MODULE.build_blind_slices(registry)
    except ValueError as exc:
        assert "unapproved truth status" in str(exc)
    else:
        raise AssertionError("CAD feature-history evidence entered a blind release slice")


def test_blind_slices_reject_registry_semantic_contract_drift():
    registry = {
        "schema_version": 2,
        "semantic_contract": {"primary_count": "all_cad_bend_features"},
        "records": [],
    }

    try:
        MODULE.build_blind_slices(registry)
    except ValueError as exc:
        assert "primary-count semantic contract" in str(exc)
    else:
        raise AssertionError("invalid registry semantics were accepted")
