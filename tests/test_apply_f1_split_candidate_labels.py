import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "apply_f1_split_candidate_labels.py"
    spec = importlib.util.spec_from_file_location("apply_f1_split_candidate_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


apply_labels = _load_module()


def _template():
    return {
        "part_key": "10839000",
        "capacity_deficit": 1,
        "candidate_labels": [
            {"bend_id": "OB2", "human_split_label": None, "notes": ""},
            {"bend_id": "OB1", "human_split_label": None, "notes": ""},
        ],
    }


def test_apply_inline_split_map_updates_template_and_summary():
    filled = apply_labels.apply_split_candidate_labels(
        template_payload=_template(),
        label_map={
            "OB2": "split_into_two_features",
            "OB1": "keep_single",
        },
    )

    labels = {row["bend_id"]: row["human_split_label"] for row in filled["candidate_labels"]}
    assert labels == {"OB2": "split_into_two_features", "OB1": "keep_single"}
    summary = apply_labels.summarize_split_candidate_labels(filled)
    assert summary["blocker_status"] == "resolved_by_split_labels"


def test_apply_rejects_unknown_split_candidate_id():
    try:
        apply_labels.apply_split_candidate_labels(template_payload=_template(), label_map={"OB9": "unclear"})
    except ValueError as exc:
        assert "unknown split candidate ids" in str(exc)
    else:
        raise AssertionError("expected unknown split candidate id failure")


def test_parse_split_assignments_accepts_comma_separated_reply():
    assert apply_labels._parse_label_assignments(
        ["OB2=split_into_two_features, OB1=keep_single\nOB3=unclear"]
    ) == {
        "OB2": "split_into_two_features",
        "OB1": "keep_single",
        "OB3": "unclear",
    }


def test_cli_applies_split_labels_and_writes_summary(tmp_path):
    template_path = tmp_path / "template.json"
    output_path = tmp_path / "filled.json"
    summary_path = tmp_path / "summary.json"
    template_path.write_text(json.dumps(_template()), encoding="utf-8")

    result = apply_labels.main(
        [
            "--template-json",
            str(template_path),
            "--label",
            "OB2=split_into_two_features",
            "--label",
            "OB1=keep_single",
            "--output-json",
            str(output_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert result == 0
    assert json.loads(output_path.read_text())["candidate_labels"][0]["human_split_label"] == "split_into_two_features"
    assert json.loads(summary_path.read_text())["blocker_status"] == "resolved_by_split_labels"
