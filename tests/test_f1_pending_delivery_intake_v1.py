from __future__ import annotations

import hashlib

import pytest

from research.f1_pending_delivery_intake_v1 import inspect_delivery, materialize_copy_plan, validate_correspondence
from research.f1_physical_intake_v2 import IntakeError


def _ply(seed: int) -> bytes:
    return (
        "ply\nformat ascii 1.0\n"
        "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        f"comment seed {seed}\nend_header\n{seed} 0 0\n"
    ).encode()


def _manifest():
    return {
        "kind": "sherman_qc_new_physical_scan_evaluator_only_intake",
        "opaque_cases": [
            {"opaque_id": "N001", "part_key": "P1", "role": "V_good", "target_path": "opaque/N001.ply", "delivery_date": "2026-08-20", "sender_count_non_authoritative": 2},
            {"opaque_id": "N002", "part_key": "P1", "role": "X_defective", "target_path": "opaque/N002.ply", "delivery_date": "2026-08-20", "sender_count_non_authoritative": 2},
        ],
    }


def test_valid_delivery_is_opaque_publicly_and_materializes_atomically(tmp_path):
    downloads, target = tmp_path / "downloads", tmp_path / "target"
    downloads.mkdir()
    (downloads / "P1 (V).ply").write_bytes(_ply(1))
    (downloads / "P1 (X).ply").write_bytes(_ply(2))
    result = inspect_delivery(_manifest(), download_root=downloads, target_root=target, consumed_scan_hashes=set())
    public = result["public_receipt"]
    assert public["scan_count"] == 2
    assert public["qualification_capacity_credit"] == 0
    assert "P1" not in str(public)
    assert "V_good" not in str(public)
    materialize_copy_plan(result["copy_plan"])
    assert (target / "opaque/N001.ply").read_bytes() == _ply(1)
    assert (target / "opaque/N002.ply").read_bytes() == _ply(2)
    replay = inspect_delivery(_manifest(), download_root=downloads, target_root=target, consumed_scan_hashes=set())
    assert replay["copy_plan"] == []
    assert all(row["already_materialized"] for row in replay["public_receipt"]["cases"])


def test_consumed_or_duplicate_content_is_rejected(tmp_path):
    downloads = tmp_path / "downloads"; downloads.mkdir()
    (downloads / "P1 (V).ply").write_bytes(_ply(1))
    (downloads / "P1 (X).ply").write_bytes(_ply(2))
    digest = hashlib.sha256(_ply(1)).hexdigest()
    with pytest.raises(IntakeError, match="development-consumed"):
        inspect_delivery(_manifest(), download_root=downloads, target_root=tmp_path / "target", consumed_scan_hashes={digest})
    (downloads / "P1 (X).ply").write_bytes(_ply(1))
    with pytest.raises(IntakeError, match="duplicate"):
        inspect_delivery(_manifest(), download_root=downloads, target_root=tmp_path / "target", consumed_scan_hashes=set())


def test_missing_file_and_target_escape_fail_closed(tmp_path):
    downloads = tmp_path / "downloads"; downloads.mkdir()
    (downloads / "P1 (V).ply").write_bytes(_ply(1))
    with pytest.raises(IntakeError, match="missing delivered scan"):
        inspect_delivery(_manifest(), download_root=downloads, target_root=tmp_path / "target", consumed_scan_hashes=set())
    (downloads / "P1 (X).ply").write_bytes(_ply(2))
    manifest = _manifest(); manifest["opaque_cases"][0]["target_path"] = "../escape.ply"
    with pytest.raises(IntakeError, match="escapes"):
        inspect_delivery(manifest, download_root=downloads, target_root=tmp_path / "target", consumed_scan_hashes=set())


def test_existing_opaque_target_with_different_bytes_is_rejected(tmp_path):
    downloads, target = tmp_path / "downloads", tmp_path / "target"
    downloads.mkdir(); (target / "opaque").mkdir(parents=True)
    (downloads / "P1 (V).ply").write_bytes(_ply(1))
    (downloads / "P1 (X).ply").write_bytes(_ply(2))
    (target / "opaque/N001.ply").write_bytes(_ply(9))
    with pytest.raises(IntakeError, match="target conflict"):
        inspect_delivery(_manifest(), download_root=downloads, target_root=target, consumed_scan_hashes=set())


def test_correspondence_requires_exact_names_unique_hashes_and_no_raw_links():
    audit = {
        "schema": "SHERMAN_QC_SOURCE_DELIVERY_CORRESPONDENCE_AUDIT_V1",
        "cases": [
            {"opaque_id": "N001", "source_filename": "P1 (V).ply", "source_message_id": "m1", "sharepoint_link_sha256": "a" * 64},
            {"opaque_id": "N002", "source_filename": "P1 (X).ply", "source_message_id": "m1", "sharepoint_link_sha256": "b" * 64},
        ],
    }
    assert validate_correspondence(_manifest(), audit)["status"] == "PASS"
    audit["cases"][1]["source_filename"] = "wrong.ply"
    with pytest.raises(IntakeError, match="does not match"):
        validate_correspondence(_manifest(), audit)
    audit["cases"][1]["source_filename"] = "P1 (X).ply"
    audit["cases"][1]["raw"] = "https://example.test"
    with pytest.raises(IntakeError, match="raw links"):
        validate_correspondence(_manifest(), audit)
