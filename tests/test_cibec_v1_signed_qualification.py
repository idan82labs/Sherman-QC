from __future__ import annotations

import json
from pathlib import Path

from research.cibec_v1.contract import CHARTER_FIELDS, validate_charter
from scripts.materialize_cibec_v1_signed_authority import materialize
from scripts.run_cibec_v1_source_free_qualification import run


def _accepted_pack(tmp_path: Path) -> Path:
    source = Path("output/research/cibec-v1-authority-candidate-v2")
    pack = tmp_path / "pack"; pack.mkdir()
    for name in ("CIBEC_V1_CHARTER_CANDIDATE.json", "S01_S14_EXPECTATIONS_CANDIDATE.json", "AUTHORITY_ATTESTATION_TEMPLATE.json"):
        (pack / name).write_bytes((source / name).read_bytes())
    attestation = json.loads((pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").read_text())
    payload_sha = attestation["review_payload_sha256"]
    attestation["manufacturing_authority"] = {"name": "Authority One", "decision": "ACCEPT", "signed_payload_sha256": payload_sha}
    attestation["countersigning_authority"] = {"name": "Authority Two", "decision": "ACCEPT", "signed_payload_sha256": payload_sha}
    (pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").write_text(json.dumps(attestation))
    return pack


def test_materializer_refuses_current_unsigned_pack(tmp_path: Path) -> None:
    output = tmp_path / "signed"
    code, report = materialize(Path("output/research/cibec-v1-authority-candidate-v2"), output)
    assert code == 2
    assert report["status"] == "REFUSED_UNSIGNED_OR_INVALID_REVIEW"
    assert report["outputs_written"] == []
    assert not output.exists()


def test_temporary_two_authority_path_materializes_and_qualifies(tmp_path: Path) -> None:
    pack = _accepted_pack(tmp_path)
    signed = tmp_path / "signed"
    code, report = materialize(pack, signed)
    assert code == 0
    assert report["status"] == "SIGNED_ARTIFACTS_MATERIALIZED"
    charter = json.loads((signed / "CIBEC_V1_CHARTER.yaml").read_text())
    validate_charter(charter)
    assert tuple(charter["decisions"]) == CHARTER_FIELDS
    qualification_code, qualification = run(signed, tmp_path / "qualification.json")
    assert qualification_code == 0
    assert qualification["status"] == "PASS"
    assert qualification["fixture_count"] == 14
    assert qualification["gates"]["semantic_exact"]["exact"] == 14
    assert qualification["gates"]["ten_run_replay"]["exact"] == 14
    assert qualification["gates"]["malformed_rejection"]["passed"] is True
    assert qualification["gates"]["cad_face_ownership"]["passed"] is True
    assert qualification["gates"]["s14_cad_abstention"]["passed"] is True
    assert qualification["physical_assets_opened"] == 0
    assert qualification["prediction_assets_opened"] == 0


def test_materializer_is_idempotent_but_refuses_output_collision(tmp_path: Path) -> None:
    pack = _accepted_pack(tmp_path)
    signed = tmp_path / "signed"
    assert materialize(pack, signed)[0] == 0
    assert materialize(pack, signed)[0] == 0
    (signed / "CIBEC_V1_CHARTER.yaml").write_text("{}")
    code, report = materialize(pack, signed)
    assert code == 2
    assert report["status"] == "REFUSED_OUTPUT_COLLISION"
