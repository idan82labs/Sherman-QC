from __future__ import annotations

import json
from pathlib import Path

from backend.f1_observed_projective_phase_reversal_v36 import select_unified_v36
from backend.f1_ordered_projective_gauss_transport import canonical_result_bytes


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "output/research/f1-v34-opgt-v1/source_free_fixtures"


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run(fixture_id: str) -> dict[str, object]:
    descriptor = load(FIXTURES / f"{fixture_id}.json")
    assert isinstance(descriptor, dict)
    raw = load(ROOT / descriptor["triangle_binding_path"])
    assert isinstance(raw, dict)
    result = select_unified_v36(
        raw,
        descriptor["candidate_records"],
        config_sha256="00" * 32,
        worker_count=1,
    )
    canonical_result_bytes(result)
    json.dumps(result["decisions"], allow_nan=False)
    return result


def test_positive_fold_survives_and_all_serializers_complete() -> None:
    result = run("SF01_FOLD60_MESH_C2")
    assert result["pre_nms_rescue_count"] == 1
    assert result["rescue_count"] == 1
    assert result["veto_count"] == 0
    certificate = result["decisions"][0]["v36_direct_phase_certificate"]
    assert certificate["passed"] is True
    assert not (certificate["positive_step_count"] and certificate["negative_step_count"])


def test_nonmonotone_bead_is_vetoed_and_all_serializers_complete() -> None:
    result = run("SN06_NONMONOTONE_BEAD_0_70_35_90")
    assert result["pre_veto_rescue_count"] == 1
    assert result["pre_nms_rescue_count"] == 0
    assert result["rescue_count"] == 0
    assert result["veto_count"] == 1
    assert result["decisions"][0]["reason"] == "OBSERVED_PROJECTIVE_PHASE_REVERSAL"
    certificate = result["decisions"][0]["v36_direct_phase_certificate"]
    assert certificate["passed"] is False
    assert certificate["positive_step_count"] > 0
    assert certificate["negative_step_count"] > 0
