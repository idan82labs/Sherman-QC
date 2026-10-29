import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_bend_ground_truth_registry.py"
SPEC = importlib.util.spec_from_file_location("build_bend_ground_truth_registry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manual_truth_overrides_legacy_and_flags_conflict():
    inventory = {
        "parts": [
            {
                "part_key": "P1",
                "expected_bends_override": 4,
                "scan_files": [{"name": "scan.ply", "path": "/tmp/scan.ply", "state": "full"}],
            }
        ]
    }
    manual = {
        "records": [
            {
                "part_key": "P1",
                "scan_name": "scan.ply",
                "scan_path": "/tmp/scan.ply",
                "track": "known",
                "human_observed_bend_count": 3,
                "human_full_part_bend_count": 3,
            }
        ]
    }

    registry = MODULE.build_registry(inventory, manual)

    record = registry["records"][0]
    assert record["counts"]["observed_conventional_bends"] == 3
    assert record["truth_status"] == "manual_verified"
    assert record["release_metric_eligible"] is True
    assert "legacy_total_conflicts_with_manual" in record["validation_issues"]


def test_impossible_partial_count_is_quarantined():
    inventory = {
        "parts": [
            {
                "part_key": "P2",
                "expected_bends_override": 1,
                "scan_files": [{"name": "partial.ply", "path": "/tmp/partial.ply", "state": "partial"}],
            }
        ]
    }
    manual = {
        "records": [
            {
                "part_key": "P2",
                "scan_name": "partial.ply",
                "scan_path": "/tmp/partial.ply",
                "track": "partial",
                "human_observed_bend_count": 2,
                "human_full_part_bend_count": 1,
            }
        ]
    }

    record = MODULE.build_registry(inventory, manual)["records"][0]

    assert record["truth_status"] == "needs_adjudication"
    assert record["release_metric_eligible"] is False
    assert "observed_exceeds_full_part" in record["validation_issues"]


def test_unreviewed_legacy_record_is_not_release_eligible():
    inventory = {
        "parts": [
            {
                "part_key": "P3",
                "expected_bends_override": 5,
                "scan_files": [{"name": "scan.ply", "path": "/tmp/scan.ply", "state": "full"}],
            }
        ]
    }

    record = MODULE.build_registry(inventory, {"records": []})["records"][0]

    assert record["truth_status"] == "legacy_unadjudicated"
    assert record["release_metric_eligible"] is False


def test_source_resolution_relocates_scan_and_quarantines_missing_sources(tmp_path):
    available = tmp_path / "archive" / "scan.ply"
    available.parent.mkdir()
    available.write_bytes(b"ply")
    inventory = {
        "parts": [
            {
                "part_key": "P1",
                "expected_bends_override": 1,
                "scan_files": [{"name": "scan.ply", "path": "/missing/scan.ply", "state": "full"}],
                "cad_files": [],
            },
            {
                "part_key": "P2",
                "expected_bends_override": 1,
                "scan_files": [{"name": "other.ply", "path": "/missing/other.ply", "state": "full"}],
                "cad_files": [],
            },
        ]
    }
    manual = {
        "records": [
            {"part_key": "P1", "scan_name": "scan.ply", "scan_path": "/missing/scan.ply", "human_observed_bend_count": 1},
            {"part_key": "P2", "scan_name": "other.ply", "scan_path": "/missing/other.ply", "human_observed_bend_count": 1},
        ]
    }

    records = MODULE.build_registry(inventory, manual, source_roots=[tmp_path])["records"]

    assert records[0]["source_geometry"]["resolved_scan_path"] == str(available.resolve())
    assert records[0]["release_metric_eligible"] is True
    assert records[1]["release_metric_eligible"] is False
    assert "scan_source_unavailable" in records[1]["validation_issues"]


def test_instance_geometry_is_attached_and_count_checked():
    inventory = {
        "parts": [
            {
                "part_key": "P1",
                "expected_bends_override": 2,
                "scan_files": [{"name": "scan.ply", "path": "/tmp/scan.ply", "state": "full"}],
            }
        ]
    }
    manual = {
        "records": [
            {
                "part_key": "P1",
                "scan_name": "scan.ply",
                "scan_path": "/tmp/scan.ply",
                "human_observed_bend_count": 2,
                "human_full_part_bend_count": 2,
            }
        ]
    }
    instance = {
        "P1::scan.ply": {
            "coordinate_frame": "reference_mesh_raw",
            "review_status": "complete",
            "bend_count": 2,
            "bends": [{"bend_id": "M01"}, {"bend_id": "M02"}],
        }
    }

    record = MODULE.build_registry(
        inventory,
        manual,
        instance_geometry_by_record=instance,
    )["records"][0]

    assert record["instance_geometry"]["bend_count"] == 2
    assert "instance_count_conflicts_with_observed" not in record["validation_issues"]
