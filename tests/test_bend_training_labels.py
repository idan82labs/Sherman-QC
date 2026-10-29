from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.bend_training_labels import (
    BendTrainingLabelError,
    asset_id_for,
    empty_labels,
    iter_ply_assets,
    load_labels,
    resolve_asset,
    save_labels,
)


def _write_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n",
        encoding="ascii",
    )


def _fixture(tmp_path: Path):
    project = tmp_path / "FullProject"
    source_assets = tmp_path / "source-assets"
    ply = source_assets / "batch" / "sample_scan.ply"
    _write_ply(ply)
    labels = project / "data" / "labels"
    roots = {
        "source-assets": source_assets,
        "data": project / "data-empty",
        "uploads": project / "uploads",
        "live_scans": project / "live_scans",
        "output": project / "output",
    }
    asset = list(iter_ply_assets(project, labels, roots=roots))[0]
    return project, labels, asset


def test_inventory_reads_bounded_header_and_stable_identity(tmp_path: Path) -> None:
    project, labels, asset = _fixture(tmp_path)
    assert asset["asset_id"] == asset_id_for("source-assets", "batch/sample_scan.ply")
    assert asset["vertex_count"] == 2
    assert asset["face_count"] == 0
    assert asset["asset_kind"] == "scan"
    assert asset["label_status"] == "not_started"


def test_save_reload_and_revision_conflict(tmp_path: Path) -> None:
    _project, labels_root, asset = _fixture(tmp_path)
    payload = empty_labels(asset)
    payload.update({
        "annotator": "Maya Cohen",
        "bends": [{
            "bend_id": "B1",
            "line_start": [0, 0, 0],
            "line_end": [1, 0, 0],
            "confidence": "certain",
            "notes": "",
        }],
    })
    saved = save_labels(labels_root, payload, asset)
    assert saved["revision"] == 1
    assert load_labels(labels_root, asset) == saved
    with pytest.raises(BendTrainingLabelError, match="revision conflict"):
        save_labels(labels_root, payload, asset)


def test_complete_zero_bend_requires_explicit_confirmation(tmp_path: Path) -> None:
    _project, labels_root, asset = _fixture(tmp_path)
    payload = empty_labels(asset)
    payload.update({"annotator": "Maya Cohen", "review_status": "complete"})
    with pytest.raises(BendTrainingLabelError, match="complete labels require"):
        save_labels(labels_root, payload, asset)
    payload["no_bends_present"] = True
    assert save_labels(labels_root, payload, asset)["review_status"] == "complete"


def test_nonfinite_or_unknown_payload_is_rejected(tmp_path: Path) -> None:
    _project, labels_root, asset = _fixture(tmp_path)
    payload = empty_labels(asset)
    payload.update({"annotator": "Maya Cohen", "unexpected": True})
    with pytest.raises(BendTrainingLabelError, match="unexpected keys"):
        save_labels(labels_root, payload, asset)
    payload.pop("unexpected")
    payload["bends"] = [{
        "bend_id": "B1",
        "line_start": [float("nan"), 0, 0],
        "line_end": [1, 0, 0],
    }]
    with pytest.raises(BendTrainingLabelError, match="finite number"):
        save_labels(labels_root, payload, asset)


def test_resolver_does_not_accept_forged_asset_id(tmp_path: Path) -> None:
    project, labels_root, _asset = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_asset(project, labels_root, "../../sample_scan.ply")


def test_invalid_saved_json_is_not_silently_overwritten(tmp_path: Path) -> None:
    _project, labels_root, asset = _fixture(tmp_path)
    labels_root.mkdir(parents=True)
    (labels_root / f"{asset['asset_id']}.json").write_text(json.dumps({"bad": True}))
    with pytest.raises(BendTrainingLabelError):
        load_labels(labels_root, asset)
