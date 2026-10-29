import json

import pytest

from scripts.render_f1_gold_overlay_review import _discover_label_cases


def test_discover_label_cases_uses_declared_scan_path(tmp_path):
    scan = tmp_path / "scan.ply"
    scan.write_text("ply\n")
    case = tmp_path / "labels" / "01_part"
    case.mkdir(parents=True)
    label = case / "bend_instance_labels.json"
    label.write_text(json.dumps({"case_id": "01_part", "scan_path": str(scan)}))

    assert _discover_label_cases(tmp_path / "labels") == [("01_part", label, scan)]


def test_discover_label_cases_fails_on_missing_scan(tmp_path):
    case = tmp_path / "01_part"
    case.mkdir()
    (case / "bend_instance_labels.json").write_text(
        json.dumps({"case_id": "01_part", "scan_path": str(tmp_path / "missing.ply")})
    )
    with pytest.raises(ValueError, match="scan_path is missing"):
        _discover_label_cases(tmp_path)
