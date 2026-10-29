import hashlib
import json
from pathlib import Path

from backend.f1_fragment_tolerant_parent_bridge_v41 import select_unified_v41


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"


def _run(fixture_id):
    manifest = json.loads(MANIFEST.read_text())
    row = next(value for value in manifest["rows"] if value["fixture_id"] == fixture_id)
    descriptor = json.loads((ROOT / row["descriptor_path"]).read_text())
    source = json.loads((ROOT / descriptor["triangle_binding_path"]).read_text())
    result = select_unified_v41(source, descriptor["candidate_records"], config_sha256=hashlib.sha256(b"v41-test").hexdigest())
    return descriptor, result


def test_all_three_fragmented_folds_are_restored():
    for fixture_id in (
        "H31_SF07_FOLD60_FRAGMENTED_888608336727",
        "H31_SF08_FOLD90_FRAGMENTED_687025e89690",
        "H31_SF09_FOLD120_FRAGMENTED_ddc435528bce",
    ):
        descriptor, result = _run(fixture_id)
        assert descriptor["expected_pre_final"] == [1, 1]
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [1, 1]
        certificate = result["decisions"][0]["v41_fragment_tolerant_parent_bridge_certificate"]
        assert certificate["bridging_skin_count"] == 2
        assert certificate["source_component_count"] > 2
        assert certificate["residual_aligned_area_ratio"] <= 0.01
        assert certificate["passed"]


def test_route_and_disconnection_nuisances_are_not_restored():
    for fixture_id in (
        "SN08_CORNER_CYCLE_TWO_ROUTE",
        "SN09_NEARBY_NONTOUCHING_PANELS_1P25S",
        "SN10_UNSUPPORTED_PLANE_INTERSECTION",
        "SN11_SCAN_BOUNDARY_TRUNCATION",
    ):
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [0, 0]
        assert not result["decisions"][0]["v41_fragment_tolerant_parent_bridge_certificate"]["passed"]


def test_prior_retained_outcomes_and_accounting_remain_stable():
    expected = {
        "SN12_FIVE_DUPLICATE_AXIS_ALIASES": [5, 1],
        "SD03_LEGACY_PLUS_RESCUE_DUPLICATE": [1, 1],
        "H31_SF03_FOLD120_MESH_C2_d722aa1877df": [1, 1],
        "SN06_NONMONOTONE_BEAD_0_70_35_90": [0, 0],
        "SF_PNFM_ADV01_AXIAL_TWISTED_FAN": [0, 0],
    }
    for fixture_id, counts in expected.items():
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == counts
