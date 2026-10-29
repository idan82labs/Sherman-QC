import hashlib
from pathlib import Path

import pytest

from scripts.build_local_scan_adjudication_queue import build_review_queue


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_queue_is_prediction_hidden_deduplicated_and_drawing_hash_locked(tmp_path):
    geometry = tmp_path / "assets" / "P2" / "scan.ply"
    duplicate = tmp_path / "assets" / "P2" / "scan-copy.ply"
    geometry.parent.mkdir(parents=True)
    geometry.write_bytes(b"scan")
    duplicate.write_bytes(b"scan")
    drawing = tmp_path / "drawings" / "P2" / "drawing.pdf"
    drawing.parent.mkdir(parents=True)
    drawing.write_bytes(b"pdf")
    cad = tmp_path / "part.stp"
    cad.write_bytes(b"cad")
    digest = _sha(b"scan")
    capacity = {
        "status": "capacity_audit_not_truth",
        "entries": [
            {
                "geometry_sha256": digest,
                "release_metric_eligible": False,
                "candidate_role": "registered_scan_candidate_requires_adjudication",
                "part_keys": ["P2"],
                "paths": [{"path": str(geometry)}, {"path": str(duplicate)}],
                "records": [
                    {
                        "record_id": "P2::scan.ply",
                        "truth_status": "legacy_unadjudicated",
                    }
                ],
            },
            {
                "geometry_sha256": _sha(b"release"),
                "release_metric_eligible": True,
                "paths": [],
                "records": [],
            },
        ],
    }
    registry = {
        "schema_version": 2,
        "records": [
            {
                "record_id": "P2::scan.ply",
                "source_geometry": {"resolved_cad_paths": [str(cad)]},
            }
        ],
    }

    queue, template = build_review_queue(
        capacity=capacity,
        capacity_sha256="capacity",
        registry=registry,
        registry_sha256="registry",
        drawing_roots=[drawing.parent.parent],
    )

    assert queue["counts"]["pending_unique_geometries"] == 1
    entry = queue["entries"][0]
    assert len(entry["source_paths"]) == 2
    assert entry["drawing_status"] == "available"
    assert entry["drawing_evidence"][0]["sha256"] == _sha(b"pdf")
    assert entry["cad_evidence_withheld_until_reconciliation"][0]["sha256"] == _sha(b"cad")
    assert queue["review_boundary"]["cad_visible_on_first_pass"] is False
    review = template["entries"][0]
    assert review["visible_observed_conventional_bend_count"] is None
    assert "cad_evidence_withheld_until_reconciliation" not in review
    forbidden = ("f1", "prediction", "expected", "legacy", "holdout", "split", "cad_evidence")

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not any(fragment in key for key in keys(template["entries"]) for fragment in forbidden)


def test_queue_rejects_geometry_hash_drift(tmp_path):
    geometry = tmp_path / "scan.ply"
    geometry.write_bytes(b"changed")
    capacity = {
        "status": "capacity_audit_not_truth",
        "entries": [
            {
                "geometry_sha256": "0" * 64,
                "release_metric_eligible": False,
                "paths": [{"path": str(geometry)}],
                "records": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="geometry hash drift"):
        build_review_queue(
            capacity=capacity,
            capacity_sha256="capacity",
            registry={"schema_version": 2, "records": []},
            registry_sha256="registry",
        )
