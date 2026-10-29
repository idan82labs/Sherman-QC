import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_score_bend_line_development_catalog_v2 import (
    TODAY_TRUTH_TIER,
    _prediction_map,
    build_catalog,
    score_catalog,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(bend_id: str, y: float = 0.0) -> dict:
    return {"bend_id": bend_id, "line_start": [0.0, y, 0.0], "line_end": [10.0, y, 0.0]}


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    legacy_label = tmp_path / "legacy-label.json"
    _write(legacy_label, {"bends": [_line("L1")]})
    legacy_scan = tmp_path / "legacy.ply"
    legacy_scan.write_bytes(b"legacy")
    legacy_prediction = tmp_path / "legacy-prediction.json"
    _write(legacy_prediction, {"predictions": [_line("P1")]})
    legacy = {
        "schema_version": 1,
        "kind": "sherman_qc_bend_line_gold_catalog",
        "source_snapshot_sha256": "a" * 64,
        "semantic_contract": {"target": "visible conventional bend lines"},
        "cases": [
            {
                "case_id": "legacy/C001",
                "part_id": "legacy-part",
                "scan_sha256": _sha(legacy_scan),
                "bend_count": 1,
                "bends": [_line("L1")],
                "truth_tier": "authoritative_user_verified",
                "promotion_eligible": True,
                "source_labels_path": str(legacy_label),
            }
        ],
    }
    legacy_path = tmp_path / "bend_line_gold_catalog_v1.json"
    _write(legacy_path, legacy)
    predictions = {"legacy/C001": str(legacy_prediction)}
    rows = []
    for index in range(1, 29):
        token = f"T{index:03d}"
        scan = tmp_path / "scans" / f"scan_{index:03d}.ply"
        scan.parent.mkdir(parents=True, exist_ok=True)
        scan.write_bytes(f"scan-{index}".encode())
        label = tmp_path / "labels" / f"label_{index:03d}.json"
        _write(label, {"coordinate_frame": "scan_raw", "manual_bends": [_line(f"M{index:03d}", float(index))]})
        prediction = tmp_path / "predictions" / f"prediction_{index:03d}.json"
        _write(prediction, {"owned_bend_regions": [_line(f"P{index:03d}", float(index))]})
        predictions[token] = str(prediction)
        rows.append(
            {
                "case_id": token,
                "part_id": f"private-{index:03d}",
                "private_filename": scan.name,
                "scan_path": str(scan),
                "scan_sha256": _sha(scan),
                "source_labels_path": str(label),
                "source_labels_sha256": _sha(label),
                "bend_count": 1,
                "consensus_status": "COMPLETE",
            }
        )
    manifest_path = tmp_path / "labels-manifest.json"
    _write(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "sherman_qc_ai_multiview_consensus_label_manifest",
            "label_date": "2026-08-30",
            "consensus_status": "COMPLETE",
            "cases": rows,
        },
    )
    return legacy_path, manifest_path, predictions


def test_builds_new_lineage_and_scores_legacy_today_and_pooled(tmp_path):
    legacy_path, manifest_path, predictions = _fixture(tmp_path)
    before = legacy_path.read_bytes()
    catalog, audit = build_catalog(
        legacy_path,
        manifest_path,
        freeze_id="BEND_LINE_DEVELOPMENT_CATALOG_V2_20260830",
    )
    scores = score_catalog(catalog, predictions, Thresholds())

    assert legacy_path.read_bytes() == before
    assert audit["legacy_sha256_before"] == hashlib.sha256(before).hexdigest()
    assert catalog["freeze_id"] == "BEND_LINE_DEVELOPMENT_CATALOG_V2_20260830"
    assert catalog["freeze_id"] != catalog["lineage"]["legacy_source_snapshot_sha256"]
    assert catalog["summary"]["legacy_case_count"] == 1
    assert catalog["summary"]["today_case_count"] == 28
    assert catalog["summary"]["promotion_eligible_today_case_count"] == 0
    assert catalog["summary"]["authoritative_today_case_count"] == 0
    today = catalog["cases"][1:]
    assert [case["case_id"] for case in today] == [f"T{i:03d}" for i in range(1, 29)]
    assert all(case["truth_tier"] == TODAY_TRUTH_TIER for case in today)
    assert all(case["review_status"] == "complete" for case in today)
    assert all(case["scoring_eligible"] is True for case in today)
    assert all(case["promotion_eligible"] is False for case in today)
    assert all(case["authoritative_ground_truth"] is False for case in today)
    assert scores["legacy_only"]["metrics"]["case_count"] == 1
    assert scores["today_only"]["metrics"]["case_count"] == 28
    assert scores["pooled"]["metrics"]["case_count"] == 29
    assert all(report["metrics"]["f1"] == 1.0 for report in scores.values())


def test_rejects_incomplete_or_reordered_today_manifest(tmp_path):
    legacy_path, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["consensus_status"] = "PENDING"
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="consensus_status must be COMPLETE"):
        build_catalog(legacy_path, manifest_path, freeze_id="new-freeze")

    _, manifest_path, _ = _fixture(tmp_path / "second")
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0], manifest["cases"][1] = manifest["cases"][1], manifest["cases"][0]
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="lexicographic private-filename order"):
        build_catalog(legacy_path=tmp_path / "second" / "bend_line_gold_catalog_v1.json", label_manifest_path=manifest_path, freeze_id="new-freeze")


def test_rejects_legacy_freeze_reuse_and_partial_prediction_map(tmp_path):
    legacy_path, manifest_path, predictions = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must not reuse"):
        build_catalog(legacy_path, manifest_path, freeze_id="a" * 64)
    catalog, _ = build_catalog(legacy_path, manifest_path, freeze_id="new-freeze")
    predictions.pop("T028")
    with pytest.raises(ValueError, match="must exactly cover"):
        score_catalog(catalog, predictions, Thresholds())


def test_prediction_map_accepts_rows_and_rejects_duplicates():
    assert _prediction_map({"rows": [{"case_id": "T001", "prediction_path": "/p.json"}]}) == {"T001": "/p.json"}
    with pytest.raises(ValueError, match="duplicate"):
        _prediction_map(
            {
                "rows": [
                    {"case_id": "T001", "prediction_path": "/p1.json"},
                    {"case_id": "T001", "prediction_path": "/p2.json"},
                ]
            }
        )
