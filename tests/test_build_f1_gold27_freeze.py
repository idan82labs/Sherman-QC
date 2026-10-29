import json
from pathlib import Path

from scripts.build_f1_gold27_freeze import build


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"


def test_gold27_freeze_separates_opaque_inference_from_truth(tmp_path):
    result = build(CATALOG, tmp_path)
    inference = json.loads(Path(result["inference_manifest"]).read_text(encoding="utf-8"))
    gold = json.loads(Path(result["sealed_gold_manifest"]).read_text(encoding="utf-8"))
    split = json.loads(Path(result["grouped_split"]).read_text(encoding="utf-8"))

    assert inference["case_count"] == gold["case_count"] == 37
    assert inference["unique_part_family_count"] == gold["unique_part_family_count"] == 27
    assert gold["gold_bend_count"] == 182
    assert all(set(case) == {
        "case_token", "opaque_input_name", "source_scan_path", "source_scan_sha256",
        "source_scan_bytes",
    } for case in inference["cases"])
    assert all(case["opaque_input_name"] == "input.ply" for case in inference["cases"])
    assert all("manual_bends" in case and "part_key" in case for case in gold["cases"])

    assignments = {row["part_key"]: row["fold"] for row in split["cases"]}
    assert len(set(assignments.values())) == 5
    for row in split["cases"]:
        assert assignments[row["part_key"]] == row["fold"]
    assert sorted(len(fold["locked_part_families"]) for fold in split["folds"]) == [5, 5, 5, 6, 6]
