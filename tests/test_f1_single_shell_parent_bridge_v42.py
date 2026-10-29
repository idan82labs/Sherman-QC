from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.f1_single_shell_parent_bridge_v42 import (
    select_unified_v42,
    single_shell_certificate,
)
from scripts.run_f1_v42_v16_real_scan_compatibility import (
    MEASUREMENT_TRIANGLE_BUDGET,
    measurement_source,
)
from backend.f1_two_sided_parent_support_guard_v43 import support_guard


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/research/f1-v34-opgt-v1/fixture_manifest.json"


def _run(fixture_id: str):
    manifest = json.loads(MANIFEST.read_text())
    row = next(value for value in manifest["rows"] if value["fixture_id"] == fixture_id)
    descriptor = json.loads((ROOT / row["descriptor_path"]).read_text())
    source = json.loads((ROOT / descriptor["triangle_binding_path"]).read_text())
    result = select_unified_v42(
        source, source, descriptor["candidate_records"],
        config_sha256=hashlib.sha256(b"v42-test").hexdigest(),
    )
    return descriptor, result


def test_one_shell_certificate_accepts_only_tail_transition_failure_shape() -> None:
    decision = {
        "predicate_passes": [True, True, True, True, False, True, True],
        "v38_sharp_transition_route_certificate": {"direct_phase_passed": True},
    }
    topology = {
        "signed_turn_degrees": 89.0,
        "source_component_count": 8,
        "bridging_skin_count": 1,
        "component_details": [
            {"area_a": 100.0, "area_b": 100.0, "bridges_parents": True},
            {"area_a": 0.2, "area_b": 0.2, "bridges_parents": False},
        ],
    }
    assert single_shell_certificate(decision, topology)["passed"]
    decision["predicate_passes"][2] = False
    assert not single_shell_certificate(decision, topology)["passed"]


def test_v41_fragmented_folds_remain_restored() -> None:
    for fixture_id in (
        "H31_SF07_FOLD60_FRAGMENTED_888608336727",
        "H31_SF08_FOLD90_FRAGMENTED_687025e89690",
        "H31_SF09_FOLD120_FRAGMENTED_ddc435528bce",
    ):
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [1, 1]


def test_adversarial_nuisances_remain_rejected() -> None:
    for fixture_id in (
        "SN08_CORNER_CYCLE_TWO_ROUTE",
        "SN09_NEARBY_NONTOUCHING_PANELS_1P25S",
        "SN10_UNSUPPORTED_PLANE_INTERSECTION",
        "SN11_SCAN_BOUNDARY_TRUNCATION",
        "SN06_NONMONOTONE_BEAD_0_70_35_90",
        "SF_PNFM_ADV01_AXIAL_TWISTED_FAN",
    ):
        _descriptor, result = _run(fixture_id)
        assert [result["pre_nms_rescue_count"], result["final_line_count"]] == [0, 0]


def test_measurement_mesh_is_bounded_below_local_support_cap() -> None:
    positions = []
    triangles = []
    for index in range(70_000):
        base = 3 * index
        positions.extend([
            [float(index), 0.0, 0.0],
            [float(index), 1.0, 0.0],
            [float(index), 0.0, 1.0],
        ])
        triangles.append([base, base + 1, base + 2])
    source, replay = measurement_source({"positions": positions, "triangles": triangles})
    assert replay["decimation_used"]
    assert len(source["triangles"]) <= MEASUREMENT_TRIANGLE_BUDGET < 65_536


def test_v43_two_sided_guard_vetoes_rim_but_preserves_non_v42_actions() -> None:
    restored = {"v42_restored_single_or_double_shell_fold": True}
    assert not support_guard(
        {"two_sided_min_support_fraction": 0.155}, restored
    )["passed"]
    assert support_guard(
        {"two_sided_min_support_fraction": 0.403}, restored
    )["passed"]
    assert support_guard({}, {})["passed"]
