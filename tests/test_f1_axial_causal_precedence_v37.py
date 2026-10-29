from __future__ import annotations

import json
from pathlib import Path

from backend.f1_axial_causal_precedence_v37 import select_unified_v37
from backend.f1_ordered_projective_gauss_transport import canonical_result_bytes


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "output/research/f1-v34-opgt-v1/source_free_fixtures"
INHERITED = [
    "SF01_FOLD60_MESH_C2",
    "SF01_RM_ALL_DIAGONALS_FLIPPED",
    "SF01_RM_CHECKERBOARD_DIAGONALS",
    "SF01_RM_SKEW_LATTICE",
    "SN06_NONMONOTONE_BEAD_0_70_35_90",
    "SN01_SILHOUETTE_FREE_EDGE",
    "SN02_OPENING_RIM",
    "SN03_THICKNESS_RETURN_1P5S",
    "SN04_DISCONNECTED_MITER90",
    "SN05_LOCAL_ROUNDED_BRIDGE_4P5S",
]


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run(fixture_id: str) -> dict[str, object]:
    descriptor = load(FIXTURES / f"{fixture_id}.json")
    assert isinstance(descriptor, dict)
    raw = load(ROOT / descriptor["triangle_binding_path"])
    assert isinstance(raw, dict)
    result = select_unified_v37(
        raw,
        descriptor["candidate_records"],
        config_sha256="11" * 32,
        worker_count=1,
    )
    canonical_result_bytes(result)
    json.dumps(result["decisions"], allow_nan=False)
    return result


def test_first_ten_inherited_actions_counts_and_lines_are_identical() -> None:
    for fixture_id in INHERITED:
        result = run(fixture_id)
        base = result["_ephemeral"]["base_result"]
        assert result["pre_nms_rescue_count"] == base["pre_nms_rescue_count"]
        assert result["rescue_count"] == base["rescue_count"]
        assert result["final_lines"] == base["final_lines"]
        assert [value["action"] for value in result["decisions"]] == [
            value["action"] for value in base["decisions"]
        ]


def test_twisted_fan_keeps_zero_count_and_exposes_axial_cause() -> None:
    result = run("SF_PNFM_ADV01_AXIAL_TWISTED_FAN")
    base = result["_ephemeral"]["base_result"]
    assert result["pre_nms_rescue_count"] == base["pre_nms_rescue_count"] == 0
    assert result["rescue_count"] == base["rescue_count"] == 0
    assert result["final_lines"] == base["final_lines"] == []
    assert result["decisions"][0]["reason"] == "AXIAL_INCOHERENCE"
    assert result["decisions"][0]["predicate_passes"][5] is False
    assert result["causal_precedence_count"] == 1
