import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_score_bend_line_development_catalog_v2 import build_catalog
from scripts.merge_ai_multiview_consensus_labels_v1 import (
    CASE_IDS,
    TRUTH_TIER,
    materialize_complete_manifest,
    merge_reviewer_manifests,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(offset: float, *, reverse: bool = False, suffix: str = "") -> dict:
    start, end = [offset, 0.0, 0.0], [10.0 + offset, 0.0, 0.0]
    if reverse:
        start, end = end, start
    return {"bend_id": f"R{suffix}", "line_start": start, "line_end": end}


def _review_fixtures(tmp_path: Path) -> list[Path]:
    scans = []
    for index in range(1, 29):
        scan = tmp_path / "scans" / f"scan_{index:03d}.ply"
        scan.parent.mkdir(parents=True, exist_ok=True)
        scan.write_bytes(f"scan {index}".encode())
        scans.append(scan)
    paths = []
    for reviewer in range(3):
        rows = []
        for index, (case_id, scan) in enumerate(zip(CASE_IDS, scans), 1):
            # Two bends are deliberately reordered and endpoint-reversed.
            bends = [
                _line(100.0 + reviewer * 0.1, reverse=reviewer == 1, suffix="B"),
                _line(reviewer * 0.1, reverse=reviewer == 2, suffix="A"),
            ]
            if reviewer == 1:
                bends.reverse()
            rows.append(
                {
                    "case_id": case_id,
                    "part_id": f"private-{index:03d}",
                    "private_filename": scan.name,
                    "scan_path": str(scan),
                    "scan_sha256": _sha(scan),
                    "coordinate_frame": "scan_raw",
                    "review_status": "COMPLETE",
                    "manual_bends": bends,
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": "sherman_qc_independent_ai_multiview_review",
            "label_date": "2026-08-30",
            "review_status": "COMPLETE",
            "reviewer_id": f"reviewer-{reviewer + 1}",
            "independent_review": True,
            "prediction_hidden": True,
            "cases": rows,
        }
        path = tmp_path / f"reviewer-{reviewer + 1}.json"
        _write(path, manifest)
        paths.append(path)
    return paths


def test_three_way_consensus_is_deterministic_and_builder_compatible(tmp_path):
    paths = _review_fixtures(tmp_path)
    first = merge_reviewer_manifests(paths, tolerance_mm=1.0)
    second = merge_reviewer_manifests(list(reversed(paths)), tolerance_mm=1.0)

    assert first["consensus_status"] == "COMPLETE"
    assert first["resolved_case_count"] == 28
    assert first["unresolved_case_count"] == 0
    assert all(case["truth_tier"] == TRUTH_TIER for case in first["cases"])
    assert all(case["promotion_eligible"] is False for case in first["cases"])
    assert all(case["authoritative_ground_truth"] is False for case in first["cases"])
    # Reviewer order does not change reconciled geometry or assigned IDs.
    assert [case["manual_bends"] for case in first["cases"]] == [case["manual_bends"] for case in second["cases"]]
    assert first["cases"][0]["manual_bends"][0]["line_start"] == pytest.approx([0.1, 0.0, 0.0])
    assert [bend["bend_id"] for bend in first["cases"][0]["manual_bends"]] == ["T001-B001", "T001-B002"]

    manifest_path = materialize_complete_manifest(first, tmp_path / "merged")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["consensus_status"] == "COMPLETE"
    assert manifest["case_count"] == 28
    assert manifest["truth_tier"] == TRUTH_TIER
    assert all(Path(row["source_labels_path"]).is_file() for row in manifest["cases"])

    legacy = {
        "schema_version": 1,
        "kind": "sherman_qc_bend_line_gold_catalog",
        "source_snapshot_sha256": "a" * 64,
        "semantic_contract": {},
        "cases": [],
    }
    legacy_path = tmp_path / "legacy.json"
    _write(legacy_path, legacy)
    catalog, _ = build_catalog(legacy_path, manifest_path, freeze_id="new-v2-freeze")
    assert catalog["summary"]["today_case_count"] == 28
    assert catalog["summary"]["today_bend_count"] == 56


def test_unmatched_review_line_emits_explicit_unresolved_case_and_no_manifest(tmp_path):
    paths = _review_fixtures(tmp_path)
    payload = json.loads(paths[2].read_text())
    payload["cases"][4]["manual_bends"].pop()
    _write(paths[2], payload)
    merge = merge_reviewer_manifests(paths, tolerance_mm=1.0)
    assert merge["consensus_status"] == "UNRESOLVED"
    assert merge["unresolved_case_count"] == 1
    assert merge["unresolved_cases"][0]["case_id"] == "T005"
    assert merge["unresolved_cases"][0]["reason"] == "NO_COMPLETE_ONE_TO_ONE_THREE_REVIEWER_CONSENSUS"
    with pytest.raises(ValueError, match="cannot materialize"):
        materialize_complete_manifest(merge, tmp_path / "must-not-exist")


@pytest.mark.parametrize("mutation, message", [
    ("duplicate_reviewer", "distinct"),
    ("prediction_visible", "prediction_hidden"),
    ("not_independent", "independent_review"),
    ("incomplete_case", "incomplete"),
    ("scan_hash_disagreement", "path/hash disagreement"),
])
def test_rejects_nonindependent_incomplete_or_unbound_reviews(tmp_path, mutation, message):
    paths = _review_fixtures(tmp_path)
    payload = json.loads(paths[1].read_text())
    if mutation == "duplicate_reviewer":
        payload["reviewer_id"] = "reviewer-1"
    elif mutation == "prediction_visible":
        payload["prediction_hidden"] = False
    elif mutation == "not_independent":
        payload["independent_review"] = False
    elif mutation == "incomplete_case":
        payload["cases"][0]["review_status"] = "PENDING"
    else:
        payload["cases"][0]["scan_sha256"] = "f" * 64
    _write(paths[1], payload)
    with pytest.raises(ValueError, match=message):
        merge_reviewer_manifests(paths)


def test_rejects_actual_scan_hash_mismatch(tmp_path):
    paths = _review_fixtures(tmp_path)
    scan = tmp_path / "scans" / "scan_001.ply"
    scan.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="scan file/hash mismatch"):
        merge_reviewer_manifests(paths)
