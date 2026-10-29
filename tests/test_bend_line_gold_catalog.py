import json
import math
from pathlib import Path

from scripts.evaluate_bend_line_gold_catalog import evaluate
from scripts.evaluate_f1_against_manual_gold import Thresholds


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
STUDIO_ROOT = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1"
ADJUDICATION = (
    ROOT
    / "data/bend_corpus/annotations/codex_manual_multiview_adjudication_resolved_20260803.json"
)


def test_consolidated_bend_line_gold_catalog_is_complete_and_tiered():
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert payload["kind"] == "sherman_qc_bend_line_gold_catalog"
    assert summary["case_count"] == 37
    assert summary["unique_part_count"] == 27
    assert summary["bend_count"] == 182
    assert summary["promotion_case_count"] == 13
    assert summary["promotion_bend_count"] == 71
    assert summary["development_case_count"] == 24
    assert summary["development_bend_count"] == 111
    assert summary["studio_canonical_case_count"] == 37
    assert summary["studio_canonical_bend_count"] == 182
    assert summary["non_ply_case_count"] == 0
    assert summary["tier_case_counts"] == {
        "authoritative_user_verified": 13,
        "codex_manual_multiview_count_backed": 12,
        "codex_manual_multiview_drawing_backed": 12,
    }

    cases = payload["cases"]
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert len({case["part_id"] for case in cases}) == 27
    for case in cases:
        assert case["bend_count"] == len(case["bends"])
        assert len(case["scan_sha256"]) == 64
        assert case["scoring_eligible"] is True
        assert case["review_status"] in {"complete", "codex_manual_multiview_complete"}
        bend_ids = [bend["bend_id"] for bend in case["bends"]]
        assert len(bend_ids) == len(set(bend_ids))
        for bend in case["bends"]:
            start = bend["line_start"]
            end = bend["line_end"]
            assert len(start) == len(end) == 3
            assert all(math.isfinite(value) for value in [*start, *end])
            assert start != end


def test_every_ply_backed_gold_case_has_studio_labels_with_matching_geometry():
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        studio = case["studio"]
        if not studio["eligible"]:
            continue
        label_path = STUDIO_ROOT / f"{studio['asset_id']}.json"
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        assert labels["source_relative_path"] == studio["relative_path"]
        assert labels["coordinate_frame"] == "source_ply_raw"
        assert labels["review_status"] == "complete"
        assert len(labels["bends"]) == case["bend_count"]
        assert [bend["line_start"] for bend in labels["bends"]] == [
            bend["line_start"] for bend in case["bends"]
        ]
        assert [bend["line_end"] for bend in labels["bends"]] == [
            bend["line_end"] for bend in case["bends"]
        ]


def test_catalog_evaluator_scores_perfect_one_case_prediction(tmp_path):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    case = next(case for case in payload["cases"] if case["promotion_eligible"])
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(
        json.dumps({"owned_bend_regions": case["bends"]}),
        encoding="utf-8",
    )
    one_case_catalog = {
        **payload,
        "cases": [case],
    }
    report = evaluate(
        one_case_catalog,
        {case["case_id"]: str(prediction_path)},
        tier="promotion",
        thresholds=Thresholds(),
    )
    assert report["metrics"] == {
        "case_count": 1,
        "unique_part_count": 1,
        "gold_bend_count": case["bend_count"],
        "prediction_count": case["bend_count"],
        "true_positive": case["bend_count"],
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "exact_case_count": 1,
        "exact_case_rate": 1.0,
    }


def test_manual_multiview_adjudication_covers_every_development_case():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    adjudication = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    development = {
        case["case_id"]: case
        for case in catalog["cases"]
        if not case["promotion_eligible"]
    }
    assert adjudication["case_count"] == len(development) == 24
    assert adjudication["bend_count"] == sum(case["bend_count"] for case in development.values()) == 111
    assert set(adjudication["cases"]) == set(development)
    for case_id, resolved in adjudication["cases"].items():
        labels = json.loads(Path(resolved["label_path"]).read_text(encoding="utf-8"))
        assert labels["review_status"] == "codex_manual_multiview_complete"
        assert labels["manual_visual_adjudication"]["decision"] == "accept_all_present_bends"
        assert len(labels["manual_visual_adjudication"]["visual_artifacts"]) == 2
        assert all(
            bend["review_decision"] == "accepted_codex_manual_multiview"
            for bend in labels["manual_bends"]
        )
