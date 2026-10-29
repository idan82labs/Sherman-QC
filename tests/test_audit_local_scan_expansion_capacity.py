from scripts.audit_local_scan_expansion_capacity import audit_local_capacity


def test_capacity_deduplicates_content_and_never_promotes_candidates(tmp_path):
    scan_root = tmp_path / "assets"
    scan_root.mkdir()
    release_scan = scan_root / "release.ply"
    duplicate_scan = scan_root / "release-copy.ply"
    pending_scan = scan_root / "pending.stl"
    unregistered_scan = scan_root / "new.ply"
    release_scan.write_bytes(b"same scan")
    duplicate_scan.write_bytes(b"same scan")
    pending_scan.write_bytes(b"pending")
    unregistered_scan.write_bytes(b"new scan")
    registry = {
        "schema_version": 2,
        "records": [
            {
                "record_id": "P1::release.ply",
                "part_key": "P1",
                "truth_status": "manual_verified",
                "release_metric_eligible": True,
                "source_geometry": {"resolved_scan_path": str(release_scan)},
            },
            {
                "record_id": "P2::pending.stl",
                "part_key": "P2",
                "truth_status": "legacy_unadjudicated",
                "release_metric_eligible": False,
                "source_geometry": {"resolved_scan_path": str(pending_scan)},
            },
        ],
    }

    result = audit_local_capacity(
        registry=registry,
        registry_sha256="registry",
        scan_root=scan_root,
        target_scan_count=50,
    )

    assert result["status"] == "capacity_audit_not_truth"
    assert result["counts"] == {
        "discovered_file_paths": 4,
        "unique_geometry_sha256": 3,
        "release_eligible_unique_geometry": 1,
        "registered_pending_unique_geometry": 1,
        "unregistered_ply_unique_geometry": 1,
        "duplicate_file_paths": 1,
        "optimistic_local_capacity_if_all_candidates_are_valid": 3,
        "minimum_new_external_scans_even_at_optimistic_local_capacity": 47,
        "additional_verified_scans_needed_from_current_state": 49,
        "missing_registry_source_records": 0,
    }
    duplicate_group = next(entry for entry in result["entries"] if entry["duplicate_path_count"] == 2)
    assert duplicate_group["release_metric_eligible"] is True
    pending_group = next(entry for entry in result["entries"] if "P2" in entry["part_keys"])
    assert pending_group["release_metric_eligible"] is False


def test_capacity_reports_missing_registry_sources_without_counting_them(tmp_path):
    registry = {
        "schema_version": 2,
        "records": [
            {
                "record_id": "P3::missing.ply",
                "source_geometry": {"resolved_scan_path": str(tmp_path / "missing.ply")},
            }
        ],
    }

    result = audit_local_capacity(
        registry=registry,
        registry_sha256="registry",
        scan_root=tmp_path,
    )

    assert result["counts"]["unique_geometry_sha256"] == 0
    assert result["missing_registry_source_record_ids"] == ["P3::missing.ply"]
