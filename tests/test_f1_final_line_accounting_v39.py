import hashlib
import json
from pathlib import Path

from backend.f1_final_line_accounting_v39 import select_unified_v39


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"


def _run(fixture_id: str):
    manifest = json.loads(MANIFEST.read_text())
    row = next(value for value in manifest["rows"] if value["fixture_id"] == fixture_id)
    descriptor = json.loads((ROOT / row["descriptor_path"]).read_text())
    source = json.loads((ROOT / descriptor["triangle_binding_path"]).read_text())
    result = select_unified_v39(
        source,
        descriptor["candidate_records"],
        config_sha256=hashlib.sha256(b"v39-test").hexdigest(),
        worker_count=1,
    )
    return descriptor, result, result["_ephemeral"]["base_result"]


def test_legacy_dominance_reports_one_total_final_line_without_action_change():
    descriptor, result, base = _run("SD03_LEGACY_PLUS_RESCUE_DUPLICATE")
    assert descriptor["expected_pre_final"] == [1, 1]
    assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [1, 1]
    assert result["post_nms_rescue_only_count"] == 0
    assert result["decisions"] == base["decisions"]
    assert result["final_lines"] == base["final_lines"]


def test_rescue_only_cases_keep_existing_counts():
    for fixture_id, expected in (
        ("SN12_FIVE_DUPLICATE_AXIS_ALIASES", [5, 1]),
        ("SD02_TWO_EXACT_RESCUE_CANDIDATES", [2, 1]),
    ):
        descriptor, result, base = _run(fixture_id)
        assert descriptor["expected_pre_final"] == expected
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == expected
        assert result["decisions"] == base["decisions"]
        assert result["final_lines"] == base["final_lines"]
