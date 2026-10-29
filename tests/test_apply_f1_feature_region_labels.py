import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "apply_f1_feature_region_labels.py"
    spec = importlib.util.spec_from_file_location("apply_f1_feature_region_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


apply_labels = _load_module()


def _template():
    return {
        "part_key": "10839000",
        "scan_name": "full.ply",
        "truth_conventional_bends": 1,
        "truth_raised_form_features": 1,
        "truth_total_features": 2,
        "region_labels": [
            {"bend_id": "OB1", "human_feature_label": None, "notes": ""},
            {"bend_id": "OB2", "human_feature_label": None, "notes": ""},
            {"bend_id": "OB3", "human_feature_label": None, "notes": ""},
        ],
    }


def test_apply_inline_label_map_updates_template_and_summary():
    filled = apply_labels.apply_feature_region_labels(
        template_payload=_template(),
        label_map={
            "OB1": "conventional_bend",
            "OB2": "raised_form_feature",
            "OB3": "fragment_or_duplicate",
        },
    )

    labels = {row["bend_id"]: row["human_feature_label"] for row in filled["region_labels"]}
    assert labels == {
        "OB1": "conventional_bend",
        "OB2": "raised_form_feature",
        "OB3": "fragment_or_duplicate",
    }
    summary = apply_labels.summarize_feature_region_labels(filled)
    assert summary["blocker_status"] == "resolved_by_region_labels"


def test_apply_rejects_unknown_bend_id():
    try:
        apply_labels.apply_feature_region_labels(template_payload=_template(), label_map={"OB9": "unclear"})
    except ValueError as exc:
        assert "unknown bend ids" in str(exc)
    else:
        raise AssertionError("expected unknown bend id failure")


def test_parse_label_assignment_validates_label():
    assert apply_labels._parse_label_assignment("OB1=unclear") == ("OB1", "unclear")
    try:
        apply_labels._parse_label_assignment("OB1=nope")
    except ValueError as exc:
        assert "invalid label" in str(exc)
    else:
        raise AssertionError("expected invalid label failure")


def test_parse_label_assignments_accepts_comma_separated_reply():
    assert apply_labels._parse_label_assignments(
        ["OB1=conventional_bend, OB2=raised_form_feature\nOB3=unclear"]
    ) == {
        "OB1": "conventional_bend",
        "OB2": "raised_form_feature",
        "OB3": "unclear",
    }


def test_cli_applies_labels_and_writes_summary(tmp_path):
    template_path = tmp_path / "template.json"
    output_path = tmp_path / "filled.json"
    summary_path = tmp_path / "summary.json"
    template_path.write_text(json.dumps(_template()), encoding="utf-8")

    result = apply_labels.main(
        [
            "--template-json",
            str(template_path),
            "--label",
            "OB1=conventional_bend",
            "--label",
            "OB2=raised_form_feature",
            "--output-json",
            str(output_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert result == 0
    assert json.loads(output_path.read_text())["region_labels"][0]["human_feature_label"] == "conventional_bend"
    assert json.loads(summary_path.read_text())["pending_region_ids"] == ["OB3"]
