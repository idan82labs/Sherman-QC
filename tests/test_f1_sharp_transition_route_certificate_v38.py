import hashlib
import json
from pathlib import Path

from backend.f1_sharp_transition_route_certificate_v38 import select_unified_v38


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"


def _run(fixture_id: str):
    manifest = json.loads(MANIFEST.read_text())
    row = next(value for value in manifest["rows"] if value["fixture_id"] == fixture_id)
    descriptor = json.loads((ROOT / row["descriptor_path"]).read_text())
    source = json.loads((ROOT / descriptor["triangle_binding_path"]).read_text())
    result = select_unified_v38(
        source,
        descriptor["candidate_records"],
        config_sha256=hashlib.sha256(b"v38-test").hexdigest(),
        worker_count=1,
    )
    return descriptor, result


def test_clean_duplicate_aliases_are_restored_then_collapsed():
    descriptor, result = _run("SN12_FIVE_DUPLICATE_AXIS_ALIASES")
    assert descriptor["expected_pre_final"] == [5, 1]
    assert [result["pre_nms_rescue_count"], result["rescue_count"]] == [5, 1]
    assert result["source_mesh_component_count"] == 2
    assert result["v38_restored_count"] == 5
    assert all(value["v38_sharp_transition_route_certificate"]["passed"] for value in result["decisions"])


def test_extra_surface_route_is_not_restored():
    _descriptor, result = _run("SN08_CORNER_CYCLE_TWO_ROUTE")
    assert [result["pre_nms_rescue_count"], result["rescue_count"]] == [0, 0]
    assert result["source_mesh_component_count"] == 3
    assert not result["decisions"][0]["v38_sharp_transition_route_certificate"]["passed"]


def test_gaps_are_not_restored_without_observed_transition():
    for fixture_id in (
        "SN09_NEARBY_NONTOUCHING_PANELS_1P25S",
        "SN10_UNSUPPORTED_PLANE_INTERSECTION",
    ):
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["rescue_count"]] == [0, 0]
        certificate = result["decisions"][0]["v38_sharp_transition_route_certificate"]
        assert certificate["transition_strip_count"] == 0
        assert not certificate["passed"]


def test_prior_positive_and_vetoes_remain_stable():
    expected = {
        "SF01_FOLD60_MESH_C2": [1, 1],
        "SN06_NONMONOTONE_BEAD_0_70_35_90": [0, 0],
        "SF_PNFM_ADV01_AXIAL_TWISTED_FAN": [0, 0],
    }
    for fixture_id, counts in expected.items():
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["rescue_count"]] == counts
    _descriptor, twisted = _run("SF_PNFM_ADV01_AXIAL_TWISTED_FAN")
    assert twisted["decisions"][0]["reason"] == "AXIAL_INCOHERENCE"
