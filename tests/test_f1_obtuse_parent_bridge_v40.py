import hashlib
import json
from pathlib import Path

from backend.f1_obtuse_parent_bridge_v40 import select_unified_v40


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"


def _run(fixture_id):
    manifest = json.loads(MANIFEST.read_text())
    row = next(value for value in manifest["rows"] if value["fixture_id"] == fixture_id)
    descriptor = json.loads((ROOT / row["descriptor_path"]).read_text())
    source = json.loads((ROOT / descriptor["triangle_binding_path"]).read_text())
    result = select_unified_v40(source, descriptor["candidate_records"], config_sha256=hashlib.sha256(b"v40-test").hexdigest())
    return descriptor, result


def test_clean_obtuse_mesh_fold_is_restored():
    descriptor, result = _run("H31_SF03_FOLD120_MESH_C2_d722aa1877df")
    assert descriptor["expected_pre_final"] == [1, 1]
    assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [1, 1]
    cert = result["decisions"][0]["v40_obtuse_parent_bridge_certificate"]
    assert cert["source_component_count"] == 2
    assert cert["bridging_skin_count"] == 2
    assert cert["passed"]


def test_disconnected_and_extra_route_nuisances_are_not_restored():
    for fixture_id in ("SN09_NEARBY_NONTOUCHING_PANELS_1P25S", "SN10_UNSUPPORTED_PLANE_INTERSECTION", "SN08_CORNER_CYCLE_TWO_ROUTE"):
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [0, 0]
        assert not result["decisions"][0]["v40_obtuse_parent_bridge_certificate"]["passed"]


def test_prior_v38_and_v39_outcomes_remain_stable():
    expected = {
        "SN12_FIVE_DUPLICATE_AXIS_ALIASES": [5, 1],
        "SD03_LEGACY_PLUS_RESCUE_DUPLICATE": [1, 1],
        "SN06_NONMONOTONE_BEAD_0_70_35_90": [0, 0],
        "SF_PNFM_ADV01_AXIAL_TWISTED_FAN": [0, 0],
    }
    for fixture_id, counts in expected.items():
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == counts
