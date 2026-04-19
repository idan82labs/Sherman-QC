import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_bend_corpus_summaries.py"
    spec = importlib.util.spec_from_file_location("compare_bend_corpus_summaries", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare_bend_corpus_summaries = _load_module()


def test_compare_summaries_reports_changed_nested_metrics():
    baseline = {
        "aggregate": {
            "mae_vs_expected_total_completed": 1.0,
            "scoreboards": {
                "correspondence": {
                    "confident_bends": 10,
                    "ambiguous_bends": 5,
                }
            },
            "contradiction_case_count": 4,
        },
        "results": [{"ignored": True}],
        "manifest_path": "/tmp/manifest.json",
    }
    candidate = {
        "aggregate": {
            "mae_vs_expected_total_completed": 1.0,
            "scoreboards": {
                "correspondence": {
                    "confident_bends": 12,
                    "ambiguous_bends": 3,
                }
            },
            "contradiction_case_count": 3,
        },
        "warnings": [],
    }

    report = compare_bend_corpus_summaries.compare_summaries(
        baseline["aggregate"],
        candidate["aggregate"],
    )

    changed = {row["metric"]: row for row in report["changed_metrics"]}
    assert changed["scoreboards.correspondence.confident_bends"]["delta"] == 2.0
    assert changed["scoreboards.correspondence.ambiguous_bends"]["delta"] == -2.0
    assert changed["contradiction_case_count"]["delta"] == -1.0
    assert "results" not in changed


def test_cli_writes_json_and_markdown(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    out_json = tmp_path / "comparison.json"
    out_md = tmp_path / "comparison.md"
    baseline_path.write_text(json.dumps({"aggregate": {"scoreboards": {"completion": {"formed_bends": 2}}}}), encoding="utf-8")
    candidate_path.write_text(json.dumps({"aggregate": {"scoreboards": {"completion": {"formed_bends": 3}}}}), encoding="utf-8")

    rc = compare_bend_corpus_summaries.main(
        [str(baseline_path), str(candidate_path), "--output-json", str(out_json), "--output-markdown", str(out_md)]
    )
    assert rc == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["changed_metric_count"] == 1
    assert payload["changed_metrics"][0]["metric"] == "scoreboards.completion.formed_bends"
    md = out_md.read_text(encoding="utf-8")
    assert "Bend Corpus Summary Comparison" in md
    assert "scoreboards.completion.formed_bends" in md
